from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .joint import JointResidualRidge, fit_joint_with_validation
from .robust import RobustJointAdapter, fit_robust_joint_adapter
from .safe import (
    AblationSemantics,
    CandidateAudit,
    CandidateSpec,
    DatasetSafetyDecision,
    SafePolicy,
    SafeSelection,
    apply_bounded_combined_correction,
    apply_safe_selection,
    evaluate_dataset_safety_gate,
    select_safe_candidate,
)
from .statistics import ErrorCell, error_cells_from_arrays
from .variants import SequentialAdapter, fit_full_with_validation


FIT_SPLIT_KEYS = (
    "features",
    "base_prediction",
    "target",
    "mask",
    "sample_id",
    "group_id",
)
SAFE_RUNNER_VARIANTS = (
    "APN",
    "Joint",
    "Full",
    "Safe",
    "SafeNoBalance",
    "SafeNoRobust",
    "SafeNoBound",
    "SafeNoGate",
)


class SafeRunnerError(RuntimeError):
    """Raised when frozen Safe states cannot be fit or paired."""


@dataclass(frozen=True)
class SafeFitPolicy:
    latent_alphas: tuple[float, ...] = (1, 10, 100, 1000, 10000, 100000)
    cross_alphas: tuple[float, ...] = (1, 10, 100, 1000, 10000, 100000)
    huber_delta: float = 1.345
    max_iterations: int = 25
    tolerance: float = 1e-8
    scale_floor: float = 1e-6
    feature_clip: float = 5.0
    minimum_rows: int | None = None
    minimum_groups: int = 20
    gate: SafePolicy = SafePolicy(
        selection_min_groups=20,
        selection_min_observations=400,
        dataset_min_groups=20,
    )

    def __post_init__(self) -> None:
        for name, values in (
            ("latent_alphas", self.latent_alphas),
            ("cross_alphas", self.cross_alphas),
        ):
            if not values or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
                raise ValueError(f"{name} must contain finite positive values")
        if self.minimum_groups <= 0:
            raise ValueError("minimum_groups must be positive")


@dataclass(frozen=True)
class SafeSeedStates:
    joint: JointResidualRidge
    full: SequentialAdapter
    robust: RobustJointAdapter
    no_balance: RobustJointAdapter
    no_robust: RobustJointAdapter
    selection: SafeSelection
    raw_spec: CandidateSpec | None
    grid_audit: tuple[Mapping[str, Any], ...]

    @property
    def residual_scale(self) -> Tensor:
        values = [
            cell.residual_scale
            for row in self.robust.cells
            for cell in row
        ]
        return torch.stack(values).reshape(
            self.robust.horizon, self.robust.channels
        )

    def prediction_map(
        self,
        split: Mapping[str, Tensor],
        *,
        dataset_gate_enabled: bool,
    ) -> dict[str, Tensor]:
        checked = validate_fit_split(split, label="apply")
        base = checked["base_prediction"]
        latent = checked["features"]
        joint = self.joint.apply(base, latent, base)
        full = self.full.apply(base, latent, graph_source=base)
        robust_prediction = self.robust.apply(latent, base)
        robust_correction = robust_prediction - base
        no_balance_correction = self.no_balance.apply(latent, base) - base
        no_robust_correction = self.no_robust.apply(latent, base) - base
        if dataset_gate_enabled:
            safe = apply_safe_selection(
                base,
                robust_correction,
                self.selection,
                residual_scale=self.residual_scale,
            )
        else:
            safe = base.clone()
        if self.raw_spec is None:
            no_balance = base.clone()
            no_robust = base.clone()
            no_gate = base.clone()
        else:
            common = {
                "residual_scale": self.residual_scale,
                "cap_multiplier": self.raw_spec.cap_multiplier,
                "shrink": self.raw_spec.shrink,
            }
            no_balance = apply_bounded_combined_correction(
                base, no_balance_correction, **common
            )
            no_robust = apply_bounded_combined_correction(
                base, no_robust_correction, **common
            )
            no_gate = apply_bounded_combined_correction(
                base, robust_correction, **common
            )
        output = {
            "APN": base.clone(),
            "Joint": joint,
            "Full": full,
            "Safe": safe,
            "SafeNoBalance": no_balance,
            "SafeNoRobust": no_robust,
            "SafeNoBound": robust_prediction,
            "SafeNoGate": no_gate,
        }
        if tuple(output) != SAFE_RUNNER_VARIANTS:
            raise SafeRunnerError("Safe runner variant order changed")
        return output


def validate_fit_split(
    split: Mapping[str, Tensor], *, label: str
) -> dict[str, Tensor]:
    missing = [name for name in FIT_SPLIT_KEYS if name not in split]
    if missing:
        raise SafeRunnerError(f"{label} split is missing {missing}")
    values = {name: split[name] for name in FIT_SPLIT_KEYS}
    if any(not isinstance(value, Tensor) for value in values.values()):
        raise SafeRunnerError(f"{label} split values must be tensors")
    base = values["base_prediction"]
    target = values["target"]
    mask = values["mask"]
    features = values["features"]
    rows, horizon, channels = base.shape
    if base.ndim != 3 or target.shape != base.shape or mask.shape != base.shape:
        raise SafeRunnerError(f"{label} forecast tensors violate [N,H,C]")
    if features.ndim != 3 or features.shape[:2] != (rows, channels):
        raise SafeRunnerError(f"{label} feature tensor violates [N,C,D]")
    for name in ("sample_id", "group_id"):
        if values[name].shape != (rows,):
            raise SafeRunnerError(f"{label} {name} must be [N]")
    observed = mask > 0
    if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(features).all()):
        raise SafeRunnerError(f"{label} base/features contain non-finite values")
    if not bool(torch.isfinite(target[observed]).all()):
        raise SafeRunnerError(f"{label} observed targets contain non-finite values")
    return values


def _group_labels(group_ids: Tensor) -> list[str]:
    return [str(int(value)) for value in group_ids.detach().cpu().tolist()]


def _preferred_candidate_audit(
    selection: SafeSelection, *, tie_tolerance: float
) -> CandidateAudit | None:
    if not selection.candidates:
        return None
    if selection.selected is not None:
        selected = [
            audit for audit in selection.candidates
            if audit.spec == selection.selected
        ]
        if len(selected) != 1:
            raise SafeRunnerError("Selected candidate lacks one audit row")
        return selected[0]
    best_score = min(audit.score for audit in selection.candidates)
    tied = [
        audit for audit in selection.candidates
        if audit.score <= best_score + tie_tolerance
    ]
    return min(
        tied,
        key=lambda audit: (
            audit.spec.shrink,
            audit.spec.cap_multiplier,
        ),
    )


def _raw_spec(
    selection: SafeSelection, *, tie_tolerance: float = 1e-4
) -> CandidateSpec | None:
    audit = _preferred_candidate_audit(
        selection, tie_tolerance=tie_tolerance
    )
    return None if audit is None else audit.spec


def _alpha_candidate_better(
    candidate: tuple[bool, float, CandidateSpec, float, float],
    incumbent: tuple[bool, float, CandidateSpec, float, float] | None,
    *,
    tie_tolerance: float,
) -> bool:
    if incumbent is None:
        return True
    deployable, score, spec, alpha_l, alpha_c = candidate
    old_deployable, old_score, old_spec, old_l, old_c = incumbent
    if deployable != old_deployable:
        return deployable
    if score < old_score - tie_tolerance:
        return True
    if score > old_score + tie_tolerance:
        return False
    return (
        spec.shrink,
        spec.cap_multiplier,
        -(alpha_l * alpha_c),
        -alpha_l,
        -alpha_c,
    ) < (
        old_spec.shrink,
        old_spec.cap_multiplier,
        -(old_l * old_c),
        -old_l,
        -old_c,
    )


def _fit_robust(
    train: Mapping[str, Tensor],
    *,
    alpha_latent: float,
    alpha_cross: float,
    policy: SafeFitPolicy,
    observation_weighted: bool = False,
    squared_loss: bool = False,
) -> RobustJointAdapter:
    return fit_robust_joint_adapter(
        train["features"],
        train["base_prediction"],
        train["target"],
        train["mask"],
        train["group_id"],
        alpha_latent=alpha_latent,
        alpha_cross=alpha_cross,
        huber_delta=policy.huber_delta,
        max_iterations=policy.max_iterations,
        tolerance=policy.tolerance,
        scale_floor=policy.scale_floor,
        feature_clip=policy.feature_clip,
        minimum_rows=policy.minimum_rows,
        minimum_groups=policy.minimum_groups,
        observation_weighted=observation_weighted,
        squared_loss=squared_loss,
    )


def _residual_scale(state: RobustJointAdapter) -> Tensor:
    return torch.stack(
        [cell.residual_scale for row in state.cells for cell in row]
    ).reshape(state.horizon, state.channels)


def fit_safe_seed(
    adapter_train: Mapping[str, Tensor],
    val_select: Mapping[str, Tensor],
    *,
    policy: SafeFitPolicy = SafeFitPolicy(),
) -> SafeSeedStates:
    """Fit all post-hoc states for one frozen APN checkpoint."""

    train = validate_fit_split(adapter_train, label="adapter_train")
    val = validate_fit_split(val_select, label="val_select")
    if set(train["sample_id"].tolist()) & set(val["sample_id"].tolist()):
        raise SafeRunnerError("adapter_train and val_select sample IDs overlap")
    alphas_latent = tuple(float(value) for value in policy.latent_alphas)
    alphas_cross = tuple(float(value) for value in policy.cross_alphas)
    joint, joint_audit = fit_joint_with_validation(
        train["base_prediction"],
        train["features"],
        train["base_prediction"],
        train["target"],
        train["mask"],
        val["base_prediction"],
        val["features"],
        val["base_prediction"],
        val["target"],
        val["mask"],
        latent_alphas=alphas_latent,
        graph_alphas=alphas_cross,
    )
    full, full_audit = fit_full_with_validation(
        train["base_prediction"],
        train["features"],
        train["target"],
        train["mask"],
        val["base_prediction"],
        val["features"],
        val["target"],
        val["mask"],
        latent_alphas=alphas_latent,
        graph_alphas=alphas_cross,
    )
    val_joint = joint.apply(
        val["base_prediction"], val["features"], val["base_prediction"]
    )
    group_labels = _group_labels(val["group_id"])
    best_rank: tuple[bool, float, CandidateSpec, float, float] | None = None
    best_state: RobustJointAdapter | None = None
    best_selection: SafeSelection | None = None
    best_spec: CandidateSpec | None = None
    grid_audit: list[dict[str, Any]] = []
    for alpha_latent, alpha_cross in product(alphas_latent, alphas_cross):
        state = _fit_robust(
            train,
            alpha_latent=alpha_latent,
            alpha_cross=alpha_cross,
            policy=policy,
        )
        raw_prediction = state.apply(val["features"], val["base_prediction"])
        correction = raw_prediction - val["base_prediction"]
        selection = select_safe_candidate(
            val["base_prediction"],
            correction,
            val_joint,
            val["target"],
            val["mask"],
            group_labels,
            residual_scale=_residual_scale(state),
            policy=policy.gate,
            semantics=AblationSemantics.full(),
        )
        candidate_audit = _preferred_candidate_audit(
            selection, tie_tolerance=policy.gate.tie_tolerance
        )
        if candidate_audit is None:
            continue
        spec = candidate_audit.spec
        deployable = selection.selected is not None and not selection.fallback_to_apn
        rank = (
            deployable,
            float(candidate_audit.score),
            spec,
            float(alpha_latent),
            float(alpha_cross),
        )
        grid_audit.append({
            "alpha_latent": alpha_latent,
            "alpha_cross": alpha_cross,
            "minimax_relative_loss": float(candidate_audit.score),
            "candidate_shrink": spec.shrink,
            "candidate_cap_multiplier": spec.cap_multiplier,
            "candidate_relative_losses": dict(candidate_audit.relative_losses),
            "deployable": deployable,
            "selection_sha256": selection.audit_sha256,
            "fitted_cells": state.audit.fitted_cells,
            "zero_cells": state.audit.zero_cells,
        })
        if _alpha_candidate_better(
            rank, best_rank, tie_tolerance=policy.gate.tie_tolerance
        ):
            best_rank, best_state, best_selection, best_spec = (
                rank,
                state,
                selection,
                spec,
            )
    if best_state is None or best_selection is None:
        raise SafeRunnerError("No robust alpha candidate was fitted")
    no_balance = _fit_robust(
        train,
        alpha_latent=best_state.alpha_latent,
        alpha_cross=best_state.alpha_cross,
        policy=policy,
        observation_weighted=True,
    )
    no_robust = _fit_robust(
        train,
        alpha_latent=best_state.alpha_latent,
        alpha_cross=best_state.alpha_cross,
        policy=policy,
        squared_loss=True,
    )
    grid_audit.append(
        {
            "joint_surface": joint_audit,
            "full_surface": full_audit,
            "selected_alpha_latent": best_state.alpha_latent,
            "selected_alpha_cross": best_state.alpha_cross,
        }
    )
    return SafeSeedStates(
        joint,
        full,
        best_state,
        no_balance,
        no_robust,
        best_selection,
        best_spec,
        tuple(grid_audit),
    )


def _validation_group_hashes(dataset_id: str, salt: str, ids: Tensor) -> list[str]:
    return [
        hashlib.sha256(
            f"{salt}{dataset_id}:{int(value)}".encode("utf-8")
        ).hexdigest()
        for value in ids.detach().cpu().tolist()
    ]


def decide_validation_safety_gate(
    states_by_seed: Mapping[int, SafeSeedStates],
    val_safety_by_seed: Mapping[int, Mapping[str, Tensor]],
    *,
    dataset_id: str,
    group_salt: str,
    policy: SafePolicy,
) -> tuple[DatasetSafetyDecision, dict[str, Any]]:
    seeds = sorted(states_by_seed)
    if seeds != sorted(val_safety_by_seed) or len(seeds) != policy.required_checkpoints:
        raise SafeRunnerError("Safety gate requires the same five states and validation splits")
    cells: list[ErrorCell] = []
    groups_reference: list[str] | None = None
    validation_cells = 0
    for seed in seeds:
        split = validate_fit_split(val_safety_by_seed[seed], label="val_safety")
        predictions = states_by_seed[seed].prediction_map(
            split, dataset_gate_enabled=True
        )
        group_hashes = _validation_group_hashes(
            dataset_id, group_salt, split["group_id"]
        )
        if groups_reference is None:
            groups_reference = group_hashes
            validation_cells = int((split["mask"] > 0).sum().item())
        elif group_hashes != groups_reference:
            raise SafeRunnerError("Five checkpoints do not share val_safety identities")
        for variant, source in (
            ("APN", predictions["APN"]),
            ("Joint", predictions["Joint"]),
            ("Full", predictions["Full"]),
            ("SafeCandidate", predictions["Safe"]),
        ):
            cells.extend(
                error_cells_from_arrays(
                    group_hashes=group_hashes,
                    checkpoint=str(seed),
                    variant=variant,
                    prediction=source.detach().cpu().numpy(),
                    target=split["target"].detach().cpu().numpy(),
                    mask=(split["mask"] > 0).detach().cpu().numpy(),
                )
            )
    decision = evaluate_dataset_safety_gate(
        cells,
        dataset_id=dataset_id,
        apn_variant="APN",
        joint_variant="Joint",
        safe_variant="SafeCandidate",
        original_variant="Full",
        policy=policy,
    )
    group_count = len(set(groups_reference or ()))
    manifest = {
        **decision.ledger_fields(),
        "improved_checkpoints": int(decision.improved_checkpoints),
        "passed": bool(decision.passed),
        "fallback_to_apn": bool(decision.fallback_to_apn),
        "deploy_variant": decision.deploy_variant,
        "reasons": list(decision.reasons),
        "locally_recomputed_validation_groups": group_count,
        "policy_sha256": decision.policy_sha256,
        "decision": decision.as_dict(),
    }
    return decision, manifest
