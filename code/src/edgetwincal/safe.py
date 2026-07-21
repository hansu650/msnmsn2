from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .statistics import (
    ErrorCell,
    accumulate_error_cells,
    crossed_cluster_checkpoint_bootstrap,
)


class SafeGateError(ValueError):
    """Raised when a validation-only Safe decision cannot be audited."""


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SafePolicy:
    cap_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    shrink_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    selection_min_groups: int = 20
    selection_min_observations: int = 400
    dataset_min_groups: int = 20
    dataset_min_observations: int = 400
    required_checkpoints: int = 5
    required_improved_checkpoints: int = 4
    selection_point_loss_margin: float = 0.0
    selection_joint_loss_margin: float = 0.005
    validation_relative_loss_margin: float = 0.01
    validation_joint_loss_margin: float = 0.005
    final_relative_loss_margin: float = 0.01
    final_joint_noninferiority_margin: float = 0.001
    max_positive_gain_concentration: float = 0.25
    confidence_level: float = 0.95
    validation_bootstrap_resamples: int = 10_000
    dataset_bootstrap_resamples: int = 10_000
    random_seed: int = 20_260_721
    tie_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if not self.cap_grid or not self.shrink_grid:
            raise SafeGateError("candidate grids must be non-empty")
        if any((not np.isfinite(x) or x <= 0) for x in self.cap_grid):
            raise SafeGateError("cap multipliers must be finite and positive")
        if any((not np.isfinite(x) or x < 0 or x > 1) for x in self.shrink_grid):
            raise SafeGateError("shrink factors must lie in [0,1]")
        if min(
            self.selection_min_groups,
            self.selection_min_observations,
            self.dataset_min_groups,
            self.dataset_min_observations,
            self.required_checkpoints,
            self.required_improved_checkpoints,
            self.validation_bootstrap_resamples,
            self.dataset_bootstrap_resamples,
        ) <= 0:
            raise SafeGateError("counts and bootstrap sizes must be positive")
        if self.required_improved_checkpoints > self.required_checkpoints:
            raise SafeGateError("improvement requirement exceeds checkpoint count")
        if not 0 < self.confidence_level < 1:
            raise SafeGateError("confidence_level must lie in (0,1)")
        if not 0 <= self.max_positive_gain_concentration <= 1:
            raise SafeGateError("gain concentration threshold must lie in [0,1]")

    @property
    def sha256(self) -> str:
        return _sha256(self)


@dataclass(frozen=True)
class AblationSemantics:
    robust_fit: bool = True
    safety_envelope: bool = True
    group_equalization: bool = True
    leave_one_group_out: bool = True

    @classmethod
    def full(cls) -> "AblationSemantics":
        return cls()

    @classmethod
    def no_robust_fit(cls) -> "AblationSemantics":
        return cls(robust_fit=False, group_equalization=False)

    @classmethod
    def no_safety_envelope(cls) -> "AblationSemantics":
        return cls(safety_envelope=False, leave_one_group_out=False)

    @classmethod
    def raw_disabled(cls) -> "AblationSemantics":
        return cls(False, False, False, False)

    @property
    def variant_id(self) -> str:
        enabled = [
            name
            for name, value in asdict(self).items()
            if value
        ]
        return "safe_" + ("+".join(enabled) if enabled else "disabled")

    @property
    def sha256(self) -> str:
        return _sha256(self)


@dataclass(frozen=True)
class ValidationGroupSplit:
    selection_groups: tuple[str, ...]
    safety_groups: tuple[str, ...]
    salt: str
    sha256: str


@dataclass(frozen=True)
class GroupMetric:
    group_id: str
    sse: float
    sae: float
    n: int

    @property
    def mse(self) -> float:
        return self.sse / self.n

    @property
    def mae(self) -> float:
        return self.sae / self.n


@dataclass(frozen=True)
class CandidateSpec:
    cap_multiplier: float
    shrink: float


@dataclass(frozen=True)
class CandidateAudit:
    spec: CandidateSpec
    feasible: bool
    score: float
    relative_losses: Mapping[str, float]
    positive_gain_concentration: float
    min_leave_one_group_out_gain: float
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SafeSelection:
    fallback_to_apn: bool
    raw_retention: bool
    selected: CandidateSpec | None
    reason: str
    candidates: tuple[CandidateAudit, ...]
    bootstrap_upper_bounds: Mapping[str, float]
    policy_sha256: str
    semantics_sha256: str
    audit_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True)
class DatasetSafetyDecision:
    dataset_id: str
    enabled: bool
    passed: bool
    deploy_variant: str
    fallback_to_apn: bool
    reasons: tuple[str, ...]
    improved_checkpoints: int
    checkpoints: int
    validation_groups: int
    validation_cells: int
    checkpoint_audit: tuple[Mapping[str, Any], ...]
    bootstrap_audit: tuple[Mapping[str, Any], ...]
    original_variant: str | None
    policy_sha256: str
    audit_sha256: str

    @property
    def gate_sha256(self) -> str:
        return self.audit_sha256

    def ledger_fields(self) -> dict[str, bool | int | str]:
        return {
            "enabled": self.enabled,
            "checkpoints": self.checkpoints,
            "validation_groups": self.validation_groups,
            "validation_cells": self.validation_cells,
            "gate_sha256": self.gate_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        return _json_ready(self)


def deterministic_validation_group_split(
    group_ids: Sequence[str], *, salt: str = "edgetwincal-safe-v1"
) -> ValidationGroupSplit:
    groups = {str(group) for group in group_ids}
    if "" in groups:
        raise SafeGateError("group ids must be non-empty")
    if len(groups) < 2:
        raise SafeGateError("at least two validation groups are required")
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{salt}|{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    selection = tuple(sorted(ranked[::2]))
    safety = tuple(sorted(ranked[1::2]))
    payload = {"selection_groups": selection, "safety_groups": safety, "salt": salt}
    return ValidationGroupSplit(selection, safety, salt, _sha256(payload))


def _as_tensor(value: Tensor | np.ndarray, *, like: Tensor | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if like is not None:
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    return tensor


def _validate_arrays(
    prediction: Tensor, target: Tensor, mask: Tensor, group_ids: Sequence[str]
) -> tuple[Tensor, Tensor, Tensor]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise SafeGateError("prediction, target, and mask shapes must match")
    if prediction.ndim == 0 or prediction.shape[0] != len(group_ids):
        raise SafeGateError("one group id is required per leading tensor row")
    selected = mask.to(dtype=torch.bool)
    if torch.any(selected & (~torch.isfinite(prediction) | ~torch.isfinite(target))):
        raise SafeGateError("selected validation values must be finite")
    return prediction, target, selected


def group_error_metrics(
    prediction: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    group_ids: Sequence[str],
) -> tuple[GroupMetric, ...]:
    pred = _as_tensor(prediction)
    tgt = _as_tensor(target, like=pred)
    msk = _as_tensor(mask).to(device=pred.device)
    pred, tgt, selected = _validate_arrays(pred, tgt, msk, group_ids)
    error = (pred - tgt).reshape(pred.shape[0], -1)
    selected = selected.reshape(pred.shape[0], -1)
    totals: dict[str, list[float]] = {}
    for row, group in enumerate(map(str, group_ids)):
        slot = totals.setdefault(group, [0.0, 0.0, 0.0])
        values = error[row][selected[row]].detach().to("cpu", torch.float64).numpy()
        slot[0] += float(np.square(values).sum(dtype=np.float64))
        slot[1] += float(np.abs(values).sum(dtype=np.float64))
        slot[2] += int(values.size)
    if not totals or any(values[2] <= 0 for values in totals.values()):
        raise SafeGateError("every validation group must have observations")
    return tuple(
        GroupMetric(group, values[0], values[1], int(values[2]))
        for group, values in sorted(totals.items())
    )


def positive_gain_concentration(
    baseline: Sequence[GroupMetric], candidate: Sequence[GroupMetric]
) -> float:
    left = {metric.group_id: metric for metric in baseline}
    right = {metric.group_id: metric for metric in candidate}
    if left.keys() != right.keys():
        raise SafeGateError("group metrics must be paired")
    gains = np.asarray(
        [max(0.0, left[group].mse - right[group].mse) for group in sorted(left)],
        dtype=np.float64,
    )
    total = float(gains.sum())
    return 1.0 if total <= 0 else float(gains.max() / total)


def _corrections(
    values: Tensor | np.ndarray | Sequence[Tensor | np.ndarray], base: Tensor
) -> list[Tensor]:
    sequence = [values] if isinstance(values, (Tensor, np.ndarray)) else list(values)
    output = [_as_tensor(value, like=base) for value in sequence]
    if any(value.shape != base.shape for value in output):
        raise SafeGateError("each correction must match the base prediction")
    if any(not torch.isfinite(value).all() for value in output):
        raise SafeGateError("corrections must be finite")
    return output


def apply_bounded_combined_correction(
    base: Tensor,
    corrections: Tensor | np.ndarray | Sequence[Tensor | np.ndarray],
    *,
    residual_scale: Tensor | np.ndarray | float,
    cap_multiplier: float,
    shrink: float,
    correction_enabled: bool = True,
    envelope_enabled: bool = True,
    fallback_to_base: bool = False,
) -> Tensor:
    if fallback_to_base or not correction_enabled or shrink == 0:
        return base.clone()
    parts = _corrections(corrections, base)
    if not parts:
        return base.clone()
    combined = torch.stack(parts).sum(dim=0)
    if not envelope_enabled:
        return base.clone() + combined
    if not np.isfinite(cap_multiplier) or cap_multiplier <= 0 or not 0 <= shrink <= 1:
        raise SafeGateError("invalid cap or shrink")
    scale = _as_tensor(
        residual_scale if isinstance(residual_scale, (Tensor, np.ndarray)) else np.asarray(residual_scale),
        like=base,
    )
    if not torch.isfinite(scale).all() or torch.any(scale < 0):
        raise SafeGateError("residual scale must be finite and nonnegative")
    try:
        cap = torch.broadcast_to(scale.abs() * cap_multiplier, base.shape)
    except RuntimeError as exc:
        raise SafeGateError("residual scale is not broadcastable") from exc
    bounded = torch.maximum(torch.minimum(combined, cap), -cap)
    return base.clone() + shrink * bounded


def _relative_loss(candidate: float, reference: float) -> float:
    if reference > 0:
        return (candidate - reference) / reference
    return 0.0 if candidate == 0 else float("inf")


def _summaries(metrics: Sequence[GroupMetric]) -> dict[str, float]:
    n = sum(metric.n for metric in metrics)
    return {
        "mse": sum(metric.sse for metric in metrics) / n,
        "mae": sum(metric.sae for metric in metrics) / n,
        "macro_mse": float(np.mean([metric.mse for metric in metrics])),
    }


def _loo_gain(
    baseline: Sequence[GroupMetric], candidate: Sequence[GroupMetric]
) -> float:
    b = {metric.group_id: metric for metric in baseline}
    c = {metric.group_id: metric for metric in candidate}
    if b.keys() != c.keys() or len(b) < 2:
        raise SafeGateError("leave-one-group-out requires paired groups")
    gains: list[float] = []
    for omitted in sorted(b):
        bm = float(np.mean([v.mse for k, v in b.items() if k != omitted]))
        cm = float(np.mean([v.mse for k, v in c.items() if k != omitted]))
        gains.append(-_relative_loss(cm, bm))

    return min(gains)

def _validation_upper_bounds(
    baseline: Sequence[GroupMetric],
    candidate: Sequence[GroupMetric],
    joint: Sequence[GroupMetric],
    policy: SafePolicy,
) -> dict[str, float]:
    b = list(baseline)
    c = {x.group_id: x for x in candidate}
    j = {x.group_id: x for x in joint}
    if {x.group_id for x in b} != c.keys() or c.keys() != j.keys():
        raise SafeGateError("validation bootstrap requires paired groups")
    groups = len(b)
    rng = np.random.default_rng(policy.random_seed)
    mult = rng.multinomial(
        groups, np.full(groups, 1.0 / groups), size=policy.validation_bootstrap_resamples
    ).astype(np.float64)
    def columns(items: Sequence[GroupMetric], name: str) -> np.ndarray:
        return np.asarray([getattr(x, name) for x in items], dtype=np.float64)
    ordered_c = [c[x.group_id] for x in b]
    ordered_j = [j[x.group_id] for x in b]
    bn = mult @ columns(b, "n")
    cn = mult @ columns(ordered_c, "n")
    mse_b = (mult @ columns(b, "sse")) / bn
    mse_c = (mult @ columns(ordered_c, "sse")) / cn
    mae_b = (mult @ columns(b, "sae")) / bn
    mae_c = (mult @ columns(ordered_c, "sae")) / cn
    macro_b = (mult @ np.asarray([x.mse for x in b])) / groups
    macro_c = (mult @ np.asarray([x.mse for x in ordered_c])) / groups
    macro_j = (mult @ np.asarray([x.mse for x in ordered_j])) / groups
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = {
            "mse_vs_apn": (mse_c - mse_b) / mse_b,
            "mae_vs_apn": (mae_c - mae_b) / mae_b,
            "macro_mse_vs_apn": (macro_c - macro_b) / macro_b,
            "macro_mse_vs_joint": (macro_c - macro_j) / macro_j,
        }
    quantile = policy.confidence_level
    return {key: float(np.quantile(value, quantile)) for key, value in draws.items()}


def _selection(
    *, fallback: bool, raw: bool, selected: CandidateSpec | None, reason: str,
    candidates: Sequence[CandidateAudit], bounds: Mapping[str, float],
    policy: SafePolicy, semantics: AblationSemantics
) -> SafeSelection:
    payload = {
        "fallback_to_apn": fallback, "raw_retention": raw, "selected": selected,
        "reason": reason, "candidates": tuple(candidates),
        "bootstrap_upper_bounds": dict(bounds), "policy_sha256": policy.sha256,
        "semantics_sha256": semantics.sha256,
    }
    return SafeSelection(
        fallback, raw, selected, reason, tuple(candidates), dict(bounds),
        policy.sha256, semantics.sha256, _sha256(payload)
    )


def select_safe_candidate(
    validation_base: Tensor,
    validation_raw_corrections: Tensor | np.ndarray | Sequence[Tensor | np.ndarray],
    validation_joint_prediction: Tensor,
    validation_target: Tensor,
    validation_mask: Tensor,
    validation_group_ids: Sequence[str],
    *,
    residual_scale: Tensor | np.ndarray | float,
    policy: SafePolicy = SafePolicy(),
    semantics: AblationSemantics = AblationSemantics(),
) -> SafeSelection:
    if not semantics.robust_fit:
        return _selection(
            fallback=True, raw=False, selected=None, reason="robust_fit_disabled",
            candidates=(), bounds={}, policy=policy, semantics=semantics
        )
    if not semantics.safety_envelope:
        spec = CandidateSpec(max(policy.cap_grid), 1.0)
        return _selection(
            fallback=False, raw=True, selected=spec, reason="envelope_disabled_raw_retention",
            candidates=(), bounds={}, policy=policy, semantics=semantics
        )
    try:
        baseline = group_error_metrics(
            validation_base, validation_target, validation_mask, validation_group_ids
        )
        joint = group_error_metrics(
            validation_joint_prediction, validation_target, validation_mask,
            validation_group_ids
        )
    except SafeGateError as exc:
        return _selection(
            fallback=True, raw=False, selected=None, reason=str(exc), candidates=(),
            bounds={}, policy=policy, semantics=semantics
        )
    if (
        len(baseline) < policy.selection_min_groups
        or sum(metric.n for metric in baseline) < policy.selection_min_observations
    ):
        return _selection(
            fallback=True, raw=False, selected=None,
            reason="insufficient_validation_groups_or_observations", candidates=(),
            bounds={}, policy=policy, semantics=semantics
        )
    base_summary = _summaries(baseline)
    joint_summary = _summaries(joint)
    audits: list[CandidateAudit] = []
    predictions: dict[tuple[float, float], Tensor] = {}
    for shrink in sorted(set(policy.shrink_grid)):
        for cap in sorted(set(policy.cap_grid)):
            spec = CandidateSpec(cap, shrink)
            prediction = apply_bounded_combined_correction(
                validation_base, validation_raw_corrections,
                residual_scale=residual_scale, cap_multiplier=cap, shrink=shrink
            )
            candidate = group_error_metrics(
                prediction, validation_target, validation_mask, validation_group_ids
            )
            summary = _summaries(candidate)
            losses = {
                key: _relative_loss(summary[key], base_summary[key])
                for key in ("mse", "mae", "macro_mse")
            }
            losses["macro_mse_vs_joint"] = _relative_loss(
                summary["macro_mse"], joint_summary["macro_mse"]
            )
            concentration = positive_gain_concentration(baseline, candidate)
            loo = _loo_gain(baseline, candidate)
            reasons = []
            if any(
                losses[key] > policy.selection_point_loss_margin + policy.tie_tolerance
                for key in ("mse", "mae", "macro_mse")
            ):
                reasons.append("point_loss")
            if losses["macro_mse_vs_joint"] > policy.selection_joint_loss_margin:
                reasons.append("joint_loss")
            if concentration > policy.max_positive_gain_concentration + policy.tie_tolerance:
                reasons.append("gain_concentration")
            if semantics.leave_one_group_out and loo < -policy.tie_tolerance:
                reasons.append("leave_one_group_out")
            score = max(losses.values())
            audits.append(
                CandidateAudit(spec, not reasons, score, losses, concentration, loo, tuple(reasons))
            )
            predictions[(cap, shrink)] = prediction
    feasible = [audit for audit in audits if audit.feasible]
    if not feasible:
        return _selection(
            fallback=True, raw=False, selected=None, reason="no_feasible_candidate",
            candidates=audits, bounds={}, policy=policy, semantics=semantics
        )
    best_score = min(audit.score for audit in feasible)
    tied = [
        audit for audit in feasible
        if audit.score <= best_score + policy.tie_tolerance
    ]
    chosen = min(tied, key=lambda audit: (audit.spec.shrink, audit.spec.cap_multiplier))
    candidate_metrics = group_error_metrics(
        predictions[(chosen.spec.cap_multiplier, chosen.spec.shrink)],
        validation_target, validation_mask, validation_group_ids
    )
    bounds = _validation_upper_bounds(baseline, candidate_metrics, joint, policy)
    ci_ok = (
        bounds["mse_vs_apn"] <= policy.validation_relative_loss_margin
        and bounds["mae_vs_apn"] <= policy.validation_relative_loss_margin
        and bounds["macro_mse_vs_apn"] <= policy.validation_relative_loss_margin
        and bounds["macro_mse_vs_joint"] <= policy.validation_joint_loss_margin
    )
    return _selection(
        fallback=not ci_ok, raw=False, selected=chosen.spec if ci_ok else None,
        reason="selected" if ci_ok else "validation_confidence_failed",
        candidates=audits, bounds=bounds, policy=policy, semantics=semantics
    )


def apply_safe_selection(
    base: Tensor,
    raw_corrections: Tensor | np.ndarray | Sequence[Tensor | np.ndarray],
    selection: SafeSelection,
    *,
    residual_scale: Tensor | np.ndarray | float,
    semantics: AblationSemantics = AblationSemantics(),
) -> Tensor:
    if selection.fallback_to_apn or selection.selected is None:
        return base.clone()
    return apply_bounded_combined_correction(
        base, raw_corrections, residual_scale=residual_scale,
        cap_multiplier=selection.selected.cap_multiplier,
        shrink=selection.selected.shrink,
        correction_enabled=semantics.robust_fit,
        envelope_enabled=semantics.safety_envelope,
    )


def _bootstrap_lookup(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> Mapping[str, Any]:
    matches = [row for row in rows if row["metric"] == metric]
    if len(matches) != 1:
        raise SafeGateError(f"missing bootstrap row for {metric}")
    return matches[0]


def _dataset_decision(
    dataset_id: str, passed: bool, apn_variant: str, safe_variant: str,
    reasons: Sequence[str], improved: int, checkpoint_audit: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]], original_variant: str | None,
    policy: SafePolicy, context: tuple[int, int, int]
) -> DatasetSafetyDecision:
    checkpoint_count, validation_groups, validation_cells = context
    payload = {
        "dataset_id": dataset_id, "enabled": passed, "passed": passed,
        "deploy_variant": safe_variant if passed else apn_variant,
        "fallback_to_apn": not passed, "reasons": tuple(reasons),
        "improved_checkpoints": improved, "checkpoints": checkpoint_count,
        "validation_groups": validation_groups, "validation_cells": validation_cells,
        "checkpoint_audit": tuple(checkpoint_audit),
        "bootstrap_audit": tuple(bootstrap), "original_variant": original_variant,
        "policy_sha256": policy.sha256,
    }
    return DatasetSafetyDecision(
        dataset_id, passed, passed, safe_variant if passed else apn_variant, not passed,
        tuple(reasons), improved, checkpoint_count, validation_groups, validation_cells,
        tuple(checkpoint_audit), tuple(bootstrap), original_variant, policy.sha256,
        _sha256(payload)
    )


def _legacy_evaluate_dataset_safety_gate(
    cells: Sequence[ErrorCell] | Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    apn_variant: str = "APN",
    joint_variant: str = "Joint",
    safe_variant: str = "Safe",
    original_variant: str | None = None,
    policy: SafePolicy = SafePolicy(),
) -> DatasetSafetyDecision:
    if not cells:
        raise SafeGateError("dataset safety gate requires error cells")
    normalized = (
        list(cells) if isinstance(cells[0], ErrorCell)
        else accumulate_error_cells(cells)  # type: ignore[arg-type]
    )
    required = {apn_variant, joint_variant, safe_variant}
    if original_variant is not None:
        required.add(original_variant)
    variants = {cell.variant for cell in normalized}
    checkpoints = sorted({cell.checkpoint for cell in normalized})
    groups = {cell.group_hash for cell in normalized}
    validation_cells = sum(
        cell.n for cell in normalized if cell.variant == apn_variant
    )
    context = (len(checkpoints), len(groups), validation_cells)
    if not required.issubset(variants):
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant, ("missing_variant",), 0,
            (), (), original_variant, policy, context
        )
    if len(checkpoints) != policy.required_checkpoints:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant, ("checkpoint_count",), 0,
            (), (), original_variant, policy, context
        )
    if len(groups) < policy.dataset_min_groups:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant, ("insufficient_groups",), 0,
            (), (), original_variant, policy, context
        )
    if validation_cells < policy.dataset_min_observations:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant,
            ("insufficient_observations",), 0, (), (), original_variant, policy, context
        )
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        row: dict[str, Any] = {"checkpoint": checkpoint}
        for metric, field in (("mse", "sse"), ("mae", "sae")):
            values: dict[str, float] = {}
            for variant in (apn_variant, safe_variant):
                subset = [
                    cell for cell in normalized
                    if cell.checkpoint == checkpoint and cell.variant == variant
                ]
                n = sum(cell.n for cell in subset)
                if n <= 0:
                    raise SafeGateError("checkpoint has no observations")
                values[variant] = sum(getattr(cell, field) for cell in subset) / n
            row[f"apn_{metric}"] = values[apn_variant]
            row[f"safe_{metric}"] = values[safe_variant]
            row[f"{metric}_relative_loss"] = _relative_loss(
                values[safe_variant], values[apn_variant]
            )
        checkpoint_rows.append(row)
    improved = sum(row["mse_relative_loss"] < 0 for row in checkpoint_rows)
    try:
        apn_bootstrap = crossed_cluster_checkpoint_bootstrap(
            normalized, baseline=apn_variant, comparators=[safe_variant],
            resamples=policy.dataset_bootstrap_resamples,
            random_seed=policy.random_seed
        )
        joint_bootstrap = crossed_cluster_checkpoint_bootstrap(
            normalized, baseline=joint_variant, comparators=[safe_variant],
            resamples=policy.dataset_bootstrap_resamples,
            random_seed=policy.random_seed + 1
        )
    except (ValueError, RuntimeError) as exc:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant,
            (f"paired_bootstrap_failed:{exc}",), improved, checkpoint_rows, (),
            original_variant, policy, context
        )
    audit = tuple(apn_bootstrap + joint_bootstrap)
    apn_mse = _bootstrap_lookup(apn_bootstrap, "mse")
    reasons: list[str] = []
    if not (
        apn_mse["relative_gain"] > 0
        and apn_mse["relative_gain_ci_low"] > 0
    ):
        reasons.append("efficacy_ci")
    if improved < policy.required_improved_checkpoints:
        reasons.append("checkpoint_consistency")
    if any(
        row[f"{metric}_relative_loss"] > policy.final_relative_loss_margin
        for row in checkpoint_rows for metric in ("mse", "mae")
    ):
        reasons.append("checkpoint_harm")
    for row in apn_bootstrap:
        if -float(row["relative_gain_ci_low"]) > policy.final_relative_loss_margin:
            reasons.append(f"apn_{row['metric']}_harm_ci")
    for row in joint_bootstrap:
        if -float(row["relative_gain"]) > policy.final_joint_noninferiority_margin:
            reasons.append(f"joint_{row['metric']}_point")
        if -float(row["relative_gain_ci_low"]) > policy.final_joint_noninferiority_margin:
            reasons.append(f"joint_{row['metric']}_ci")
    return _dataset_decision(
        dataset_id, not reasons, apn_variant, safe_variant, reasons, improved,
        checkpoint_rows, audit, original_variant, policy, context
    )


def _paired_gate_cube(
    cells: Sequence[ErrorCell], variants: Sequence[str]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    groups = sorted({cell.group_hash for cell in cells if cell.variant in variants})
    checkpoints = sorted({cell.checkpoint for cell in cells if cell.variant in variants})
    vix = {value: index for index, value in enumerate(variants)}
    gix = {value: index for index, value in enumerate(groups)}
    kix = {value: index for index, value in enumerate(checkpoints)}
    shape = (len(variants), len(groups), len(checkpoints))
    sse = np.full(shape, np.nan, dtype=np.float64)
    sae = np.full(shape, np.nan, dtype=np.float64)
    n = np.full(shape, np.nan, dtype=np.float64)
    for cell in cells:
        if cell.variant not in vix:
            continue
        index = (vix[cell.variant], gix[cell.group_hash], kix[cell.checkpoint])
        if np.isfinite(n[index]):
            raise SafeGateError("duplicate validation gate cell")
        if (
            not np.isfinite(cell.sse)
            or not np.isfinite(cell.sae)
            or cell.sse < 0
            or cell.sae < 0
            or cell.n < 0
        ):
            raise SafeGateError("invalid validation gate cell")
        sse[index], sae[index], n[index] = cell.sse, cell.sae, cell.n
    if not groups or not checkpoints or not np.isfinite(n).all():
        raise SafeGateError("incomplete crossed validation gate cube")
    if not np.array_equal(n, np.broadcast_to(n[:1], n.shape)):
        raise SafeGateError("validation variants have different target counts")
    empty = np.all(n == 0, axis=(0, 2))
    evaluable = np.all(n > 0, axis=(0, 2))
    if np.any(~(empty | evaluable)):
        raise SafeGateError("validation group is not paired across checkpoints")
    if not np.any(evaluable):
        raise SafeGateError("validation gate has no evaluable groups")
    if np.any(empty):
        groups = [group for group, keep in zip(groups, evaluable) if keep]
        sse, sae, n = sse[:, evaluable, :], sae[:, evaluable, :], n[:, evaluable, :]
    return groups, checkpoints, sse, sae, n


def _gate_point_metrics(
    sse: np.ndarray, sae: np.ndarray, n: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "micro_mse": sse.sum(axis=(1, 2)) / n.sum(axis=(1, 2)),
        "group_macro_mse": (sse.sum(axis=2) / n.sum(axis=2)).mean(axis=1),
        "micro_mae": sae.sum(axis=(1, 2)) / n.sum(axis=(1, 2)),
    }


def _crossed_gate_bootstrap(
    cells: Sequence[ErrorCell],
    *,
    apn_variant: str,
    joint_variant: str,
    safe_variant: str,
    policy: SafePolicy,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    dict[str, np.ndarray],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    variants = (apn_variant, joint_variant, safe_variant)
    groups, checkpoints, sse, sae, n = _paired_gate_cube(cells, variants)
    point = _gate_point_metrics(sse, sae, n)
    vix = {value: index for index, value in enumerate(variants)}
    metrics = ("micro_mse", "group_macro_mse", "micro_mae")
    draw_losses = {
        (reference, metric): np.empty(
            policy.dataset_bootstrap_resamples, dtype=np.float64
        )
        for reference in (apn_variant, joint_variant)
        for metric in metrics
    }
    rng = np.random.default_rng(policy.random_seed)
    group_probability = np.full(len(groups), 1.0 / len(groups))
    checkpoint_probability = np.full(len(checkpoints), 1.0 / len(checkpoints))
    write_at = 0
    while write_at < policy.dataset_bootstrap_resamples:
        count = min(256, policy.dataset_bootstrap_resamples - write_at)
        gm = rng.multinomial(
            len(groups), group_probability, size=count
        ).astype(np.float64)
        km = rng.multinomial(
            len(checkpoints), checkpoint_probability, size=count
        ).astype(np.float64)
        weighted_n = np.einsum("bg,vgk,bk->bv", gm, n, km, optimize=True)
        weighted_sse = np.einsum("bg,vgk,bk->bv", gm, sse, km, optimize=True)
        weighted_sae = np.einsum("bg,vgk,bk->bv", gm, sae, km, optimize=True)
        group_sse = np.einsum("vgk,bk->bvg", sse, km, optimize=True)
        group_n = np.einsum("vgk,bk->bvg", n, km, optimize=True)
        sampled = {
            "micro_mse": weighted_sse / weighted_n,
            "group_macro_mse": (
                np.einsum("bg,bvg->bv", gm, group_sse / group_n, optimize=True)
                / len(groups)
            ),
            "micro_mae": weighted_sae / weighted_n,
        }
        for reference in (apn_variant, joint_variant):
            for metric, values in sampled.items():
                reference_values = values[:, vix[reference]]
                safe_values = values[:, vix[safe_variant]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    losses = (safe_values - reference_values) / reference_values
                if not np.isfinite(losses).all():
                    raise SafeGateError("non-finite validation bootstrap effect")
                draw_losses[(reference, metric)][write_at:write_at + count] = losses
        write_at += count
    audit: list[Mapping[str, Any]] = []
    gain_lower_quantile = (1.0 - policy.confidence_level) / 2.0
    for reference in (apn_variant, joint_variant):
        for metric in metrics:
            losses = draw_losses[(reference, metric)]
            point_loss = _relative_loss(
                float(point[metric][vix[safe_variant]]),
                float(point[metric][vix[reference]]),
            )
            audit.append({
                "reference": reference,
                "candidate": safe_variant,
                "metric": metric,
                "reference_point": float(point[metric][vix[reference]]),
                "candidate_point": float(point[metric][vix[safe_variant]]),
                "relative_loss": point_loss,
                "relative_loss_ucb": float(
                    np.quantile(losses, policy.confidence_level)
                ),
                "relative_gain_ci_low": float(
                    np.quantile(-losses, gain_lower_quantile)
                ),
                "resamples": policy.dataset_bootstrap_resamples,
                "random_seed": policy.random_seed,
                "groups": len(groups),
                "checkpoints": len(checkpoints),
                "shared_group_checkpoint_multiplicities": True,
            })
    return tuple(audit), point, groups, checkpoints, sse, sae, n


def _gate_row(
    audit: Sequence[Mapping[str, Any]], reference: str, metric: str
) -> Mapping[str, Any]:
    matches = [
        row for row in audit
        if row["reference"] == reference and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise SafeGateError(f"missing gate audit row: {reference}/{metric}")
    return matches[0]


def _collapsed_group_metrics(
    groups: Sequence[str],
    sse: np.ndarray,
    sae: np.ndarray,
    n: np.ndarray,
    variant_index: int,
) -> tuple[GroupMetric, ...]:
    return tuple(
        GroupMetric(
            group,
            float(sse[variant_index, index].sum()),
            float(sae[variant_index, index].sum()),
            int(n[variant_index, index].sum()),
        )
        for index, group in enumerate(groups)
    )


def evaluate_dataset_safety_gate(
    cells: Sequence[ErrorCell] | Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    apn_variant: str = "APN",
    joint_variant: str = "Joint",
    safe_variant: str = "Safe",
    original_variant: str | None = None,
    policy: SafePolicy = SafePolicy(),
) -> DatasetSafetyDecision:
    if not cells:
        raise SafeGateError("dataset safety gate requires error cells")
    normalized = (
        list(cells) if isinstance(cells[0], ErrorCell)
        else accumulate_error_cells(cells)
    )
    required = {apn_variant, joint_variant, safe_variant}
    if original_variant is not None:
        required.add(original_variant)
    variants = {cell.variant for cell in normalized}
    raw_checkpoints = sorted({cell.checkpoint for cell in normalized})
    raw_groups = {cell.group_hash for cell in normalized}
    validation_cells = sum(
        cell.n for cell in normalized if cell.variant == apn_variant
    )
    raw_context = (len(raw_checkpoints), len(raw_groups), validation_cells)
    if not required.issubset(variants):
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant, ("missing_variant",), 0,
            (), (), original_variant, policy, raw_context
        )
    if len(raw_checkpoints) != policy.required_checkpoints:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant, ("checkpoint_count",), 0,
            (), (), original_variant, policy, raw_context
        )
    try:
        audit, _, groups, checkpoints, sse, sae, n = _crossed_gate_bootstrap(
            normalized, apn_variant=apn_variant, joint_variant=joint_variant,
            safe_variant=safe_variant, policy=policy,
        )
    except (SafeGateError, ValueError, RuntimeError) as exc:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant,
            (f"paired_bootstrap_failed:{exc}",), 0, (), (),
            original_variant, policy, raw_context
        )
    context = (len(checkpoints), len(groups), validation_cells)
    if len(groups) < policy.dataset_min_groups:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant,
            ("insufficient_groups",), 0, (), (), original_variant, policy, context
        )
    if validation_cells < policy.dataset_min_observations:
        return _dataset_decision(
            dataset_id, False, apn_variant, safe_variant,
            ("insufficient_observations",), 0, (), (), original_variant, policy, context
        )
    vix = {apn_variant: 0, joint_variant: 1, safe_variant: 2}
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        row: dict[str, Any] = {"checkpoint": checkpoint}
        for variant in (apn_variant, joint_variant, safe_variant):
            index = vix[variant]
            denominator = float(n[index, :, checkpoint_index].sum())
            row[f"{variant}_micro_mse"] = float(
                sse[index, :, checkpoint_index].sum() / denominator
            )
            row[f"{variant}_group_macro_mse"] = float(np.mean(
                sse[index, :, checkpoint_index] / n[index, :, checkpoint_index]
            ))
            row[f"{variant}_micro_mae"] = float(
                sae[index, :, checkpoint_index].sum() / denominator
            )
        row["mse_relative_loss"] = _relative_loss(
            row[f"{safe_variant}_micro_mse"],
            row[f"{apn_variant}_micro_mse"],
        )
        row["mae_relative_loss"] = _relative_loss(
            row[f"{safe_variant}_micro_mae"],
            row[f"{apn_variant}_micro_mae"],
        )
        checkpoint_rows.append(row)
    improved = sum(row["mse_relative_loss"] < 0 for row in checkpoint_rows)
    baseline_groups = _collapsed_group_metrics(groups, sse, sae, n, vix[apn_variant])
    safe_groups = _collapsed_group_metrics(groups, sse, sae, n, vix[safe_variant])
    loo_gain = _loo_gain(baseline_groups, safe_groups)
    concentration = positive_gain_concentration(baseline_groups, safe_groups)
    reasons: list[str] = []
    for metric in ("micro_mse", "group_macro_mse", "micro_mae"):
        row = _gate_row(audit, apn_variant, metric)
        if float(row["relative_loss"]) > policy.tie_tolerance:
            reasons.append(f"apn_point_loss:{metric}")
        if float(row["relative_loss_ucb"]) > (
            policy.validation_relative_loss_margin + policy.tie_tolerance
        ):
            reasons.append(f"apn_harm_ucb:{metric}")
    apn_mse = _gate_row(audit, apn_variant, "micro_mse")
    if not (
        float(apn_mse["relative_loss"]) < 0
        and float(apn_mse["relative_gain_ci_low"]) > 0
    ):
        reasons.extend(("apn_mse_gain_unreliable", "efficacy_ci"))
    joint_macro = _gate_row(audit, joint_variant, "group_macro_mse")
    if float(joint_macro["relative_loss"]) > (
        policy.validation_joint_loss_margin + policy.tie_tolerance
    ):
        reasons.append("joint_macro_point")
    if float(joint_macro["relative_loss_ucb"]) > (
        policy.validation_joint_loss_margin + policy.tie_tolerance
    ):
        reasons.append("joint_macro_ucb")
    if improved < policy.required_improved_checkpoints:
        reasons.append("checkpoint_consistency")
    if any(
        row[f"{metric}_relative_loss"]
        > policy.validation_relative_loss_margin + policy.tie_tolerance
        for row in checkpoint_rows
        for metric in ("mse", "mae")
    ):
        reasons.append("checkpoint_harm")
    if loo_gain < -policy.tie_tolerance:
        reasons.append("macro_leave_one_group_out")
    if concentration > (
        policy.max_positive_gain_concentration + policy.tie_tolerance
    ):
        reasons.append("group_mse_gain_concentration")
    checkpoint_rows.append({
        "summary": "group_robustness",
        "macro_leave_one_group_out_gain": loo_gain,
        "group_mse_positive_gain_concentration": concentration,
    })
    return _dataset_decision(
        dataset_id, not reasons, apn_variant, safe_variant, reasons, improved,
        checkpoint_rows, audit, original_variant, policy, context
    )
