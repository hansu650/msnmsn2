"""Fail-closed confirmatory aggregation and claim gates for msn2026_v1.

The public entry point consumes an explicit list of expected run manifests.
It never discovers runs by globbing: a missing, failed, or corrupt expected run
aborts aggregation.  Group and checkpoint identifiers are used only inside the
paired calculations and are deliberately absent from the returned compact
payload.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import ResolvedConfig, load_resolved_config
from .schema import ManifestError, load_complete_manifest
from .statistics import (
    ErrorCell,
    accumulate_error_cells,
    classify_safety,
    crossed_cluster_checkpoint_bootstrap,
    holm_adjust,
    pooled_metrics,
)


AGGREGATE_SCHEMA = "edgetwincal.confirmatory-aggregate.v1"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_WINDOWS_PATH = re.compile(r"(?i)\b[a-z]:[\\/][^\s,;]*")
_UNIX_PRIVATE_PATH = re.compile(r"(?<!\w)/(?:home|users|mnt|tmp)/[^\s,;]*", re.I)
_BLOCKER_STATES = frozenset({"BLOCKED", "FAILED", "INVALIDATED"})
_MAE_HARM_MARGIN = 0.002


class ConfirmatoryAggregationError(RuntimeError):
    """Raised when an explicitly expected run cannot be safely aggregated."""


def _config_data(
    config: ResolvedConfig | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return load_resolved_config() if config is None else config


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ConfirmatoryAggregationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfirmatoryAggregationError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ConfirmatoryAggregationError(f"{label} must be a finite number")
    return result


def _redact_reason(value: Any) -> str:
    text = " ".join(str(value or "unspecified blocker").split())[:512]
    text = _WINDOWS_PATH.sub("<redacted-local-path>", text)
    return _UNIX_PRIVATE_PATH.sub("<redacted-local-path>", text)


def _compact_blockers(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ConfirmatoryAggregationError(f"blocker_records[{index}] must be an object")
        status = str(record.get("status", "BLOCKED")).upper()
        if status not in _BLOCKER_STATES:
            raise ConfirmatoryAggregationError(
                f"blocker_records[{index}].status must be BLOCKED/FAILED/INVALIDATED"
            )
        scope = str(record.get("scope", record.get("component", "campaign"))).strip()
        if not scope:
            raise ConfirmatoryAggregationError(f"blocker_records[{index}].scope is empty")
        reason_code = str(record.get("reason_code", record.get("code", "UNSPECIFIED")))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", reason_code):
            raise ConfirmatoryAggregationError(
                f"blocker_records[{index}].reason_code must be a compact token"
            )
        item = {
            "scope": scope,
            "status": status,
            "reason_code": reason_code,
            "reason": _redact_reason(record.get("reason")),
        }
        dataset = record.get("dataset")
        if dataset is not None:
            item["dataset"] = str(dataset)
        compact.append(item)
    return compact


def _load_expected(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    if isinstance(paths, (str, bytes, Path)) or not isinstance(paths, Sequence):
        raise ConfirmatoryAggregationError("expected_manifest_paths must be a sequence")
    if not paths:
        raise ConfirmatoryAggregationError("expected_manifest_paths may not be empty")
    path_keys = [str(Path(path)) for path in paths]
    if len(set(path_keys)) != len(path_keys):
        raise ConfirmatoryAggregationError("expected_manifest_paths contains duplicates")

    manifests: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        try:
            manifest = load_complete_manifest(path)
        except (ManifestError, OSError, ValueError) as exc:
            # The basename and ordinal identify the registry entry without
            # leaking a machine-local directory into logs or return artifacts.
            label = Path(path).name or f"entry-{index}"
            raise ConfirmatoryAggregationError(
                f"Expected manifest #{index} ({label}) is unavailable, incomplete, or corrupt: "
                f"{type(exc).__name__}"
            ) from exc
        if manifest.get("status") != "complete":
            raise ConfirmatoryAggregationError(
                f"Expected manifest #{index} is not complete"
            )
        manifests.append(manifest)
    return manifests


def _validate_manifest_identity(manifests: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str, str, int, str]] = set()
    for index, manifest in enumerate(manifests):
        try:
            identity = (
                str(manifest["dataset"]),
                str(manifest["protocol"]),
                str(manifest["fold"]),
                int(manifest["seed"]),
                str(manifest["variant_id"]),
            )
            cells = manifest["cells"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfirmatoryAggregationError(
                f"Expected manifest #{index} lacks a run identity or cells"
            ) from exc
        if identity in seen:
            raise ConfirmatoryAggregationError(
                "Duplicate dataset/protocol/fold/seed/variant run in expected registry"
            )
        seen.add(identity)
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or not cells:
            raise ConfirmatoryAggregationError(f"Expected manifest #{index} has no cells")
        asset_checkpoint = manifest.get("assets", {}).get("checkpoint_sha256")
        for cell in cells:
            if not isinstance(cell, Mapping):
                raise ConfirmatoryAggregationError(
                    f"Expected manifest #{index} contains a non-object cell"
                )
            if str(cell.get("variant")) != identity[-1]:
                raise ConfirmatoryAggregationError(
                    "Cell variant differs from its run manifest variant_id"
                )
            checkpoint = str(cell.get("checkpoint_sha256", ""))
            if not _SHA256.fullmatch(checkpoint):
                raise ConfirmatoryAggregationError("Cell checkpoint_sha256 is invalid")
            if asset_checkpoint is not None and checkpoint != str(asset_checkpoint):
                raise ConfirmatoryAggregationError(
                    "Cell checkpoint differs from the manifest asset checkpoint"
                )


def _is_strict(
    dataset: str,
    protocol: str,
    config: Mapping[str, Any],
) -> bool:
    dataset_definition = config.get("datasets", {}).get(dataset, {})
    registered = dataset_definition.get("strict_protocol") if isinstance(
        dataset_definition, Mapping
    ) else None
    return protocol == registered or protocol.lower().startswith("strict_")


def _has_reliable_groups(manifest: Mapping[str, Any]) -> bool:
    split = manifest.get("split_manifest")
    if not isinstance(split, Mapping):
        return False
    if split.get("group_ids_reliable") is True:
        return True
    return (
        split.get("schema_version") == "edgetwincal.protocol.split.v1"
        and bool(_SHA256.fullmatch(str(split.get("group_id_hash", ""))))
    )


def _block_cells(
    manifests: Sequence[Mapping[str, Any]],
    *,
    require_hashed_groups: bool,
) -> list[ErrorCell]:
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        for cell in manifest["cells"]:
            group = str(cell["group_hash"])
            if require_hashed_groups and not _SHA256.fullmatch(group):
                raise ConfirmatoryAggregationError(
                    "A reliable strict run contains a non-hashed group identifier"
                )
            records.append(
                {
                    "group_hash": group,
                    "checkpoint": str(cell["checkpoint_sha256"]),
                    "variant": str(cell["variant"]),
                    "sse": cell["sse"],
                    "sae": cell["sae"],
                    "n": cell["n"],
                }
            )
    try:
        return accumulate_error_cells(records)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfirmatoryAggregationError(f"Invalid SSE/SAE/N cells: {exc}") from exc


def _target_support_audit(cells: Sequence[ErrorCell]) -> dict[str, int]:
    group_n: dict[str, int] = defaultdict(int)
    zero_target_cells = 0
    for cell in cells:
        group_n[cell.group_hash] += cell.n
        zero_target_cells += cell.n == 0
    zero_target_groups = sum(total == 0 for total in group_n.values())
    return {
        "manifest_groups": len(group_n),
        "evaluable_groups": len(group_n) - zero_target_groups,
        "zero_target_groups_excluded": zero_target_groups,
        "zero_target_cells_excluded": int(zero_target_cells),
    }


def _seed_descriptive(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    totals: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for manifest in manifests:
        seed = int(manifest["seed"])
        variant = str(manifest["variant_id"])
        for cell in manifest["cells"]:
            totals[(variant, seed)][0] += float(cell["sse"])
            totals[(variant, seed)][1] += float(cell["sae"])
            totals[(variant, seed)][2] += int(cell["n"])

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in sorted({key[0] for key in totals}):
        seed_rows: dict[str, dict[str, float | int]] = {}
        mse_values: list[float] = []
        mae_values: list[float] = []
        for (name, seed), (sse, sae, n_float) in sorted(totals.items()):
            if name != variant:
                continue
            n = int(n_float)
            if n <= 0:
                raise ConfirmatoryAggregationError(
                    f"Variant {variant}, seed {seed} has no observed targets"
                )
            mse, mae = sse / n, sae / n
            mse_values.append(mse)
            mae_values.append(mae)
            seed_rows[str(seed)] = {"mse": mse, "mae": mae, "n": n}
        ddof = 1 if len(mse_values) > 1 else 0
        by_variant[variant] = {
            "seeds": seed_rows,
            "seed_count": len(seed_rows),
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values, ddof=ddof)),
            "mae_mean": float(np.mean(mae_values)),
            "mae_std": float(np.std(mae_values, ddof=ddof)),
        }
    return by_variant


def _paired_checkpoint_directions(
    cells: Sequence[ErrorCell],
    *,
    baseline: str,
    candidate: str,
) -> tuple[int, int]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for cell in cells:
        if cell.variant in {baseline, candidate}:
            totals[(cell.checkpoint, cell.variant)][0] += cell.sse
            totals[(cell.checkpoint, cell.variant)][1] += cell.n
    baseline_checkpoints = {checkpoint for checkpoint, variant in totals if variant == baseline}
    candidate_checkpoints = {checkpoint for checkpoint, variant in totals if variant == candidate}
    if baseline_checkpoints != candidate_checkpoints:
        raise ConfirmatoryAggregationError(
            f"Unpaired checkpoints for {baseline} vs {candidate}"
        )
    improved = 0
    for checkpoint in baseline_checkpoints:
        base_sse, base_n = totals[(checkpoint, baseline)]
        cand_sse, cand_n = totals[(checkpoint, candidate)]
        if base_n <= 0 or cand_n <= 0:
            raise ConfirmatoryAggregationError("A paired checkpoint has no observations")
        improved += cand_sse / cand_n < base_sse / base_n
    return int(improved), len(baseline_checkpoints)


def _comparison(
    cells: Sequence[ErrorCell],
    *,
    reference: str,
    resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    selected = [cell for cell in cells if cell.variant in {reference, "Full"}]
    try:
        rows = crossed_cluster_checkpoint_bootstrap(
            selected,
            baseline=reference,
            comparators=["Full"],
            resamples=resamples,
            random_seed=random_seed,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ConfirmatoryAggregationError(
            f"Cannot construct paired crossed inference for Full vs {reference}: {exc}"
        ) from exc
    metrics: dict[str, dict[str, float | int]] = {}
    for row in rows:
        # With reference as bootstrap baseline, effect already is Full-reference.
        gain_low = float(row["relative_gain_ci_low"])
        gain_high = float(row["relative_gain_ci_high"])
        metrics[str(row["metric"])] = {
            "reference_point": float(row["baseline_point"]),
            "full_point": float(row["candidate_point"]),
            "effect_full_minus_reference": float(
                row["effect_candidate_minus_baseline"]
            ),
            "effect_ci_low": float(row["effect_ci_low"]),
            "effect_ci_high": float(row["effect_ci_high"]),
            "relative_gain": float(row["relative_gain"]),
            "relative_gain_ci_low": gain_low,
            "relative_gain_ci_high": gain_high,
            "relative_loss_full_vs_reference": -float(row["relative_gain"]),
            "relative_loss_ci_low": -gain_high,
            "relative_loss_ci_high": -gain_low,
            "p_full_not_better": float(row["one_sided_p_candidate_not_better"]),
            "resamples": int(row["resamples"]),
        }
    return {"reference": reference, "metrics": metrics}


def _variant_diagnostic_mean(
    manifests: Sequence[Mapping[str, Any]],
    variant: str,
    key: str,
) -> float | None:
    values: list[float] = []
    for manifest in manifests:
        if str(manifest["variant_id"]) != variant:
            continue
        metrics = manifest.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get(key)
        mechanism = metrics.get("mechanism")
        if value is None and isinstance(mechanism, Mapping):
            value = mechanism.get(key)
        if value is not None:
            values.append(_finite(value, label=f"metrics.{key}"))
    return float(np.mean(values)) if values else None


def classify_dataset_evidence(
    *,
    relative_mse_gain: float,
    mse_effect_ci_low: float | None,
    mse_effect_ci_high: float | None,
    relative_mae_gain_ci_low: float | None,
    relative_mae_gain_ci_high: float | None,
    improved_checkpoints: int,
    checkpoint_count: int,
    inferential: bool,
    primary_holm_adjusted_p: float | None = None,
    minimum_relative_mse_improvement: float = 0.005,
    minimum_improved_checkpoints: int = 4,
    mae_harm_margin: float = _MAE_HARM_MARGIN,
) -> str:
    """Apply the frozen five-way G3 semantics without overstating harm.

    ``harmful`` is reserved for an inferential interval that confirms harm.
    A negative point estimate or an interval that merely permits harm is
    ``safety-inconclusive`` instead.
    """

    gain = _finite(relative_mse_gain, label="relative_mse_gain")
    if not inferential:
        if abs(gain) <= 1e-12:
            return "neutral"
        return "supportive" if gain > 0 else "safety-inconclusive"
    if None in {
        mse_effect_ci_low,
        mse_effect_ci_high,
        relative_mae_gain_ci_low,
        relative_mae_gain_ci_high,
    }:
        raise ValueError("Inferential classification requires MSE and MAE intervals")
    mse_low = _finite(mse_effect_ci_low, label="mse_effect_ci_low")
    mse_high = _finite(mse_effect_ci_high, label="mse_effect_ci_high")
    mae_gain_low = _finite(
        relative_mae_gain_ci_low, label="relative_mae_gain_ci_low"
    )
    mae_gain_high = _finite(
        relative_mae_gain_ci_high, label="relative_mae_gain_ci_high"
    )
    safety = classify_safety(
        gain_ci_low=mae_gain_low,
        gain_ci_high=mae_gain_high,
        harm_margin=mae_harm_margin,
    )
    if mse_low > 0 or safety == "harmful":
        return "harmful"
    if safety == "safety-inconclusive":
        return "safety-inconclusive"
    holm_p = (
        None
        if primary_holm_adjusted_p is None
        else _finite(primary_holm_adjusted_p, label="primary_holm_adjusted_p")
    )
    if (
        gain >= minimum_relative_mse_improvement
        and mse_high < 0
        and holm_p is not None
        and holm_p < 0.05
        and checkpoint_count >= 5
        and improved_checkpoints >= minimum_improved_checkpoints
    ):
        return "strong"
    if gain > 0:
        return "supportive"
    if mse_low <= 0 <= mse_high and abs(gain) < minimum_relative_mse_improvement:
        return "neutral"
    return "safety-inconclusive"


def _apply_holm_families(analyses: Sequence[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = {"primary": [], "secondary": []}
    for analysis in analyses:
        for reference, comparison in analysis.get("comparisons", {}).items():
            family = "primary" if reference == "APN" else "secondary"
            mse = comparison["metrics"]["mse"]
            mse["holm_family"] = family
            families[family].append(mse)
    summary: dict[str, Any] = {}
    for name, members in families.items():
        adjusted = holm_adjust([float(member["p_full_not_better"]) for member in members])
        for member, value in zip(members, adjusted):
            member["holm_adjusted_p"] = float(value)
        summary[name] = {
            "comparison_count": len(members),
            "alpha": 0.05,
            "rejected_not_better": sum(value < 0.05 for value in adjusted),
        }
    return summary


def _evaluate_g2(
    analysis: dict[str, Any],
    *,
    controls_required: Sequence[str],
    joint_margin: float,
) -> dict[str, Any]:
    if not analysis["strict"]:
        return {"status": "NOT_APPLICABLE", "reason": "official descriptive block"}
    if analysis["inference"] != "crossed_group_checkpoint":
        return {"status": "BLOCKED", "reason": "reliable strict group IDs unavailable"}
    comparisons = analysis["comparisons"]
    variants = set(analysis["variant_metrics"])
    missing = sorted(set(controls_required).difference(variants))
    if missing:
        return {"status": "FAIL", "missing_controls": missing}

    simple: dict[str, bool] = {}
    for variant in ("V01", "V02", "V03"):
        mse = comparisons[variant]["metrics"]["mse"]
        simple[variant] = (
            float(mse["effect_ci_high"]) < 0
            and float(mse.get("holm_adjusted_p", 1.0)) < 0.05
        )

    joint_ci_high = float(
        comparisons["V08"]["metrics"]["mse"]["relative_loss_ci_high"]
    )
    joint_ok = joint_ci_high <= joint_margin
    point = analysis["variant_metrics"]
    apn = float(point["APN"]["mse"])
    slrh = float(point["SLRH"]["mse"])
    full = float(point["Full"]["mse"])
    delta_cfg = slrh - full
    delta_slrh = apn - slrh
    cfg_ratio = None if delta_cfg <= 1e-12 else (slrh - float(point["V11"]["mse"])) / delta_cfg
    slrh_ratio = (
        None
        if delta_slrh <= 1e-12
        else (apn - float(point["V12"]["mse"])) / delta_slrh
    )
    cfg_ok = cfg_ratio is not None and cfg_ratio <= 0.5
    slrh_ok = slrh_ratio is not None and slrh_ratio <= 0.5

    diagonal_relative = (float(point["V10"]["mse"]) - full) / full
    diagonal_fraction = analysis.get("diagonal_correction_variance_fraction")
    if diagonal_relative <= -0.001 and diagonal_fraction is None:
        diagonal_decision = "INCONCLUSIVE_MISSING_VARIANCE_DIAGNOSTIC"
    elif (
        diagonal_relative <= -0.001
        and diagonal_fraction is not None
        and float(diagonal_fraction) >= 0.5
    ):
        diagonal_decision = "DELETE_PRIMARY_CROSS_SENSOR_CLAIM"
    else:
        diagonal_decision = "CROSS_SENSOR_CLAIM_NOT_REFUTED"

    reverse_relative = (float(point["V07"]["mse"]) - full) / full
    if abs(reverse_relative) < 0.001:
        reverse_decision = "NO_ORDER_ADVANTAGE_CLAIM"
    elif reverse_relative > 0:
        reverse_decision = "FULL_ORDER_SUPPORTED"
    else:
        reverse_decision = "REVERSE_ORDER_BETTER"

    core = all(simple.values()) and joint_ok and cfg_ok and slrh_ok
    return {
        "status": "PASS" if core else "FAIL",
        "simple_controls_superior": simple,
        "joint_noninferior": joint_ok,
        "joint_relative_loss_ci_high": joint_ci_high,
        "cfg_shuffle": {
            "delta": delta_cfg,
            "retained_gain_ratio": cfg_ratio,
            "passed": cfg_ok,
        },
        "slrh_shuffle": {
            "delta": delta_slrh,
            "retained_gain_ratio": slrh_ratio,
            "passed": slrh_ok,
        },
        "diagonal": {
            "relative_mse_vs_full": diagonal_relative,
            "correction_variance_fraction": diagonal_fraction,
            "decision": diagonal_decision,
        },
        "reverse": {
            "relative_mse_vs_full": reverse_relative,
            "decision": reverse_decision,
        },
    }


def _evaluate_g4(
    edge_measurement: Mapping[str, Any] | None,
    blockers: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    edge_blockers = [
        blocker["reason_code"]
        for blocker in blockers
        if blocker["scope"].lower() in {"edge", "edge_cpu", "jetson", "g4"}
    ]
    if edge_measurement is None:
        return {
            "status": "BLOCKED",
            "reason_codes": edge_blockers or ["NO_REAL_EDGE_MEASUREMENT"],
        }
    if edge_measurement.get("available") is False or not edge_measurement.get(
        "real_edge_target", False
    ):
        return {
            "status": "BLOCKED",
            "reason_codes": edge_blockers or ["REAL_EDGE_TARGET_UNAVAILABLE"],
        }

    thresholds = config["gates"]["G4"]
    state_bytes = _finite(
        edge_measurement.get("serialized_adapter_state_bytes"),
        label="edge.serialized_adapter_state_bytes",
    )
    apn_latency = _finite(edge_measurement.get("apn_warm_p95_ms"), label="edge.apn_warm_p95_ms")
    full_latency = _finite(edge_measurement.get("full_warm_p95_ms"), label="edge.full_warm_p95_ms")
    apn_memory = _finite(edge_measurement.get("apn_peak_memory_bytes"), label="edge.apn_peak_memory_bytes")
    full_memory = _finite(edge_measurement.get("full_peak_memory_bytes"), label="edge.full_peak_memory_bytes")
    update_seconds = _finite(edge_measurement.get("update_128_seconds"), label="edge.update_128_seconds")
    mae_increase_upper = _finite(
        edge_measurement.get("relative_mae_increase_ci_upper"),
        label="edge.relative_mae_increase_ci_upper",
    )
    if apn_latency <= 0 or apn_memory <= 0:
        raise ConfirmatoryAggregationError("APN edge denominators must be positive")
    latency_overhead = (full_latency - apn_latency) / apn_latency
    memory_overhead = (full_memory - apn_memory) / apn_memory
    checks = {
        "state": state_bytes <= float(thresholds["maximum_state_bytes"]),
        "latency": latency_overhead
        <= float(thresholds["maximum_warm_p95_time_overhead_fraction"]),
        "memory": memory_overhead
        <= float(thresholds["maximum_memory_overhead_fraction"]),
        "update": update_seconds <= float(thresholds["maximum_update_128_seconds"])
        and edge_measurement.get("update_oom") is False,
        "mae_safety": mae_increase_upper <= _MAE_HARM_MARGIN,
    }
    if "net_energy_overhead_fraction" in edge_measurement:
        checks["energy"] = _finite(
            edge_measurement["net_energy_overhead_fraction"],
            label="edge.net_energy_overhead_fraction",
        ) <= 0.10
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "serialized_adapter_state_bytes": int(state_bytes),
        "warm_p95_latency_overhead_fraction": latency_overhead,
        "peak_memory_overhead_fraction": memory_overhead,
        "update_128_seconds": update_seconds,
        "relative_mae_increase_ci_upper": mae_increase_upper,
    }


def aggregate_confirmatory(
    expected_manifest_paths: Sequence[str | Path],
    blocker_records: Sequence[Mapping[str, Any]],
    *,
    config: ResolvedConfig | Mapping[str, Any] | None = None,
    bootstrap_resamples: int | None = None,
    edge_measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the explicitly registered complete runs and evaluate G2--G4.

    Production calls use the configured 50,000 crossed draws and seed
    ``20260721``.  ``bootstrap_resamples`` exists solely for fast synthetic
    tests; the analysis seed cannot be overridden here.
    """

    cfg = _config_data(config)
    configured_draws = int(cfg["bootstrap"]["draws"])
    draws = configured_draws if bootstrap_resamples is None else int(bootstrap_resamples)
    if draws <= 0:
        raise ConfirmatoryAggregationError("bootstrap_resamples must be positive")
    analysis_seed = int(cfg["bootstrap"]["analysis_seed"])
    blockers = _compact_blockers(blocker_records)
    manifests = _load_expected(expected_manifest_paths)
    _validate_manifest_identity(manifests)

    blocks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        blocks[(str(manifest["dataset"]), str(manifest["protocol"]))].append(manifest)

    analyses: list[dict[str, Any]] = []
    for (dataset, protocol), block in sorted(blocks.items()):
        strict = _is_strict(dataset, protocol, cfg)
        reliable = strict and all(_has_reliable_groups(manifest) for manifest in block)
        cells = _block_cells(block, require_hashed_groups=reliable)
        target_audit = _target_support_audit(cells)
        points = pooled_metrics(cells)
        seed_summary = _seed_descriptive(block)
        variants = set(points)
        analysis: dict[str, Any] = {
            "dataset": dataset,
            "protocol": protocol,
            "strict": strict,
            "reliable_group_ids": reliable,
            "inference": (
                "crossed_group_checkpoint" if reliable else "seed_descriptive_only"
            ),
            "counts": {
                "complete_manifests": len(block),
                "groups": target_audit["evaluable_groups"],
                **target_audit,
                "checkpoints": len({cell.checkpoint for cell in cells}),
                "variants": len(variants),
            },
            "variant_metrics": points,
            "seed_descriptive": seed_summary,
            "comparisons": {},
        }
        if reliable and "Full" in variants:
            for reference in sorted(variants.difference({"Full"})):
                analysis["comparisons"][reference] = _comparison(
                    cells,
                    reference=reference,
                    resamples=draws,
                    random_seed=analysis_seed,
                )
        analysis["diagonal_correction_variance_fraction"] = _variant_diagnostic_mean(
            block, "V10", "diagonal_correction_variance_fraction"
        )
        if {"APN", "Full"}.issubset(variants):
            improved, total = _paired_checkpoint_directions(
                cells, baseline="APN", candidate="Full"
            )
            apn_mse = float(points["APN"]["mse"])
            full_mse = float(points["Full"]["mse"])
            gain = (apn_mse - full_mse) / apn_mse
            if reliable:
                primary = analysis["comparisons"]["APN"]["metrics"]
                label = classify_dataset_evidence(
                    relative_mse_gain=gain,
                    mse_effect_ci_low=float(primary["mse"]["effect_ci_low"]),
                    mse_effect_ci_high=float(primary["mse"]["effect_ci_high"]),
                    relative_mae_gain_ci_low=float(
                        primary["mae"]["relative_gain_ci_low"]
                    ),
                    relative_mae_gain_ci_high=float(
                        primary["mae"]["relative_gain_ci_high"]
                    ),
                    improved_checkpoints=improved,
                    checkpoint_count=total,
                    inferential=True,
                    minimum_relative_mse_improvement=float(
                        cfg["gates"]["G3"]["minimum_relative_mse_improvement"]
                    ),
                    minimum_improved_checkpoints=int(
                        cfg["gates"]["G3"]["minimum_improved_paired_seeds"]
                    ),
                )
            else:
                label = classify_dataset_evidence(
                    relative_mse_gain=gain,
                    mse_effect_ci_low=None,
                    mse_effect_ci_high=None,
                    relative_mae_gain_ci_low=None,
                    relative_mae_gain_ci_high=None,
                    improved_checkpoints=improved,
                    checkpoint_count=total,
                    inferential=False,
                )
            analysis["G3"] = {
                "classification": label,
                "relative_mse_gain": gain,
                "improved_checkpoints": improved,
                "checkpoint_count": total,
                "confirmatory_interval_available": reliable,
            }
        else:
            analysis["G3"] = {
                "classification": "safety-inconclusive",
                "reason": "APN/Full pair absent",
                "confirmatory_interval_available": False,
            }
        analyses.append(analysis)

    holm = _apply_holm_families(analyses)
    for analysis in analyses:
        if analysis["inference"] != "crossed_group_checkpoint":
            continue
        comparison = analysis.get("comparisons", {}).get("APN")
        if not isinstance(comparison, Mapping):
            continue
        primary = comparison["metrics"]
        mse = primary["mse"]
        mae = primary["mae"]
        adjusted_p = float(mse["holm_adjusted_p"])
        g3 = analysis["G3"]
        g3["classification"] = classify_dataset_evidence(
            relative_mse_gain=float(g3["relative_mse_gain"]),
            mse_effect_ci_low=float(mse["effect_ci_low"]),
            mse_effect_ci_high=float(mse["effect_ci_high"]),
            relative_mae_gain_ci_low=float(mae["relative_gain_ci_low"]),
            relative_mae_gain_ci_high=float(mae["relative_gain_ci_high"]),
            improved_checkpoints=int(g3["improved_checkpoints"]),
            checkpoint_count=int(g3["checkpoint_count"]),
            inferential=True,
            primary_holm_adjusted_p=adjusted_p,
            minimum_relative_mse_improvement=float(
                cfg["gates"]["G3"]["minimum_relative_mse_improvement"]
            ),
            minimum_improved_checkpoints=int(
                cfg["gates"]["G3"]["minimum_improved_paired_seeds"]
            ),
        )
        g3["primary_holm_adjusted_p"] = adjusted_p

    controls_required = tuple(cfg["gates"]["G2"]["controls_required"])
    joint_margin = float(
        cfg["gates"]["G2"]["joint_noninferiority_relative_mse_margin"]
    )
    for analysis in analyses:
        analysis["G2"] = _evaluate_g2(
            analysis,
            controls_required=controls_required,
            joint_margin=joint_margin,
        )

    g2_applicable = [analysis["G2"] for analysis in analyses if analysis["strict"]]
    if not g2_applicable or all(item["status"] == "BLOCKED" for item in g2_applicable):
        g2_status = "BLOCKED"
    elif all(item["status"] == "PASS" for item in g2_applicable):
        g2_status = "PASS"
    else:
        g2_status = "FAIL"

    confirmatory = [
        analysis for analysis in analyses if analysis["inference"] == "crossed_group_checkpoint"
    ]
    labels = [analysis["G3"]["classification"] for analysis in confirmatory]
    required_strong = math.ceil(
        float(cfg["gates"]["G3"]["strong_fraction"]) * len(confirmatory)
    )
    if not confirmatory:
        g3_status = "BLOCKED"
    elif "harmful" in labels:
        g3_status = "FAIL"
    elif labels.count("strong") >= required_strong:
        g3_status = "PASS"
    else:
        g3_status = "FAIL"

    # Remove an absent diagnostic rather than serializing JSON null as if it
    # were a measured value.  The G2 decision still records when it was needed.
    for analysis in analyses:
        if analysis.get("diagonal_correction_variance_fraction") is None:
            analysis.pop("diagonal_correction_variance_fraction", None)

    return {
        "schema_version": AGGREGATE_SCHEMA,
        "input_audit": {
            "expected_manifest_count": len(expected_manifest_paths),
            "complete_manifest_count": len(manifests),
            "blocker_count": len(blockers),
            "all_expected_complete_and_verified": True,
        },
        "bootstrap": {
            "method": "crossed_group_checkpoint",
            "resamples": draws,
            "analysis_seed": analysis_seed,
            "paired_multiplicities": True,
        },
        "blockers": blockers,
        "holm_families": holm,
        "analyses": analyses,
        "gates": {
            "G2": {"status": g2_status},
            "G3": {
                "status": g3_status,
                "confirmatory_dataset_count": len(confirmatory),
                "strong_count": labels.count("strong"),
                "required_strong_count": required_strong,
                "harmful_count": labels.count("harmful"),
                "safety_inconclusive_count": labels.count("safety-inconclusive"),
            },
            "G4": _evaluate_g4(edge_measurement, blockers, cfg),
        },
    }


__all__ = [
    "AGGREGATE_SCHEMA",
    "ConfirmatoryAggregationError",
    "aggregate_confirmatory",
    "classify_dataset_evidence",
]
