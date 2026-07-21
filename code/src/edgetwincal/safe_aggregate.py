from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import require_within_root
from .schema import atomic_write_json
from .statistics import (
    ErrorCell,
    crossed_cluster_checkpoint_bootstrap,
    holm_adjust,
    pooled_metrics,
)


SAFE_DATASETS = ("beijing_air", "intel_lab")
SAFE_SEEDS = (2024, 2025, 2026, 2027, 2028)
SAFE_MAIN_VARIANTS = ("APN", "Joint", "Full", "Safe")
SAFE_ABLATIONS = (
    "SafeNoBalance",
    "SafeNoRobust",
    "SafeNoBound",
    "SafeNoGate",
)
SAFE_VARIANTS = (*SAFE_MAIN_VARIANTS, *SAFE_ABLATIONS)
SAFE_AGGREGATION_SCHEMA = "edgetwincal.safe-aggregation.v1"


class SafeAggregationError(RuntimeError):
    """Raised when the paired Safe campaign is missing or inconsistent."""


@dataclass(frozen=True)
class SafeGateThresholds:
    harm_margin: float = 0.01
    joint_noninferiority_margin: float = 0.001
    alpha: float = 0.05


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafeAggregationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise SafeAggregationError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _validate_gate_decision(dataset: str, decision: Mapping[str, Any]) -> None:
    if not isinstance(decision.get("enabled"), bool):
        raise SafeAggregationError(f"{dataset} gate decision lacks a boolean enabled field")
    if int(decision.get("checkpoints", -1)) != 5:
        raise SafeAggregationError(f"{dataset} gate must audit five checkpoints")
    if int(decision.get("validation_groups", -1)) < 20:
        raise SafeAggregationError(f"{dataset} gate audited fewer than 20 groups")
    if int(decision.get("validation_cells", -1)) < 400:
        raise SafeAggregationError(f"{dataset} gate audited fewer than 400 cells")
    digest = decision.get("gate_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SafeAggregationError(f"{dataset} gate_sha256 is invalid")


def _validated_manifests(
    manifests: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    variants: Sequence[str],
    gate_decisions: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    expected = {
        (str(dataset), int(seed), str(variant))
        for dataset in datasets
        for seed in seeds
        for variant in variants
    }
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for manifest in manifests:
        if manifest.get("schema_version") != "edgetwincal.safe-evaluation.v1":
            raise SafeAggregationError("Evaluation manifest schema is invalid")
        if manifest.get("status") != "complete":
            raise SafeAggregationError("Aggregation rejects non-complete runs")
        key = (
            str(manifest.get("dataset")),
            int(manifest.get("seed")),
            str(manifest.get("variant")),
        )
        if key in indexed:
            raise SafeAggregationError(f"Duplicate evaluation manifest: {key}")
        indexed[key] = manifest
    missing = sorted(expected.difference(indexed))
    extra = sorted(set(indexed).difference(expected))
    if missing or extra:
        raise SafeAggregationError(
            f"Evaluation matrix differs; missing={missing[:8]}, extra={extra[:8]}"
        )

    for dataset in datasets:
        dataset_rows = [
            indexed[(str(dataset), int(seed), variant)]
            for seed in seeds
            for variant in variants
        ]
        for field in (
            "config_sha256",
            "split_sha256",
            "normalizer_sha256",
            "sample_ids_sha256",
            "group_ids_sha256",
            "mask_sha256",
            "target_sha256",
            "protocol_sha256",
            "gate_sha256",
        ):
            values = {str(row.get(field)) for row in dataset_rows}
            if len(values) != 1 or len(next(iter(values))) != 64:
                raise SafeAggregationError(
                    f"{dataset} {field} differs across variants or seeds"
                )
        if str(dataset_rows[0]["gate_sha256"]) != str(
            gate_decisions[str(dataset)]["gate_sha256"]
        ):
            raise SafeAggregationError(
                f"{dataset} manifest gate_sha256 differs from gate decision"
            )
        if any(
            len(str(row.get("prediction_sha256"))) != 64
            for row in dataset_rows
        ):
            raise SafeAggregationError(
                f"{dataset} prediction_sha256 is invalid"
            )
        if not bool(gate_decisions[str(dataset)]["enabled"]):
            for seed in seeds:
                apn_hash = str(
                    indexed[(str(dataset), int(seed), "APN")][
                        "prediction_sha256"
                    ]
                )
                safe_hash = str(
                    indexed[(str(dataset), int(seed), "Safe")][
                        "prediction_sha256"
                    ]
                )
                if safe_hash != apn_hash:
                    raise SafeAggregationError(
                        f"{dataset}/{seed} disabled gate did not return exact APN"
                    )
        for seed in seeds:
            paired = [
                indexed[(str(dataset), int(seed), variant)] for variant in variants
            ]
            checkpoint = {str(row.get("checkpoint_sha256")) for row in paired}
            if len(checkpoint) != 1 or len(next(iter(checkpoint))) != 64:
                raise SafeAggregationError(
                    f"{dataset}/{seed} variants do not share one checkpoint"
                )
            for field in (
                "config_sha256",
                "split_sha256",
                "normalizer_sha256",
                "sample_ids_sha256",
                "group_ids_sha256",
                "mask_sha256",
            ):
                values = {str(row.get(field)) for row in paired}
                if len(values) != 1 or len(next(iter(values))) != 64:
                    raise SafeAggregationError(
                        f"{dataset}/{seed} paired {field} values differ"
                    )
            reference_counts: dict[str, int] | None = None
            for row in paired:
                cells = row.get("cells")
                if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or not cells:
                    raise SafeAggregationError(f"{dataset}/{seed} has no error cells")
                counts: dict[str, int] = {}
                total_sse = total_sae = 0.0
                total_n = 0
                for cell in cells:
                    if not isinstance(cell, Mapping):
                        raise SafeAggregationError("Error cells must be mappings")
                    group_hash = str(cell.get("group_hash"))
                    if not group_hash or group_hash in counts:
                        raise SafeAggregationError("Group hashes must be nonempty and unique")
                    n = int(cell.get("n", -1))
                    sse = _finite_number(cell.get("sse"), label="cell.sse")
                    sae = _finite_number(cell.get("sae"), label="cell.sae")
                    if n < 0 or sse < 0 or sae < 0 or (n == 0 and (sse or sae)):
                        raise SafeAggregationError("Invalid SSE/SAE/N error cell")
                    counts[group_hash] = n
                    total_sse += sse
                    total_sae += sae
                    total_n += n
                if reference_counts is None:
                    reference_counts = counts
                elif counts != reference_counts:
                    raise SafeAggregationError(
                        f"{dataset}/{seed} variants have unpaired group counts"
                    )
                metrics = row.get("metrics")
                if not isinstance(metrics, Mapping) or total_n <= 0:
                    raise SafeAggregationError("Evaluation metrics are missing")
                mse = _finite_number(metrics.get("mse"), label="metrics.mse")
                mae = _finite_number(metrics.get("mae"), label="metrics.mae")
                if not math.isclose(mse, total_sse / total_n, rel_tol=1e-10, abs_tol=1e-12):
                    raise SafeAggregationError("Manifest MSE differs from cells")
                if not math.isclose(mae, total_sae / total_n, rel_tol=1e-10, abs_tol=1e-12):
                    raise SafeAggregationError("Manifest MAE differs from cells")
    return indexed


def _dataset_cells(
    indexed: Mapping[tuple[str, int, str], Mapping[str, Any]],
    dataset: str,
    seeds: Sequence[int],
    variants: Sequence[str],
    *,
    prefix_dataset: bool = False,
) -> list[ErrorCell]:
    output: list[ErrorCell] = []
    for seed in seeds:
        for variant in variants:
            manifest = indexed[(dataset, int(seed), variant)]
            for cell in manifest["cells"]:
                group = str(cell["group_hash"])
                if prefix_dataset:
                    group = f"{dataset}:{group}"
                output.append(
                    ErrorCell(
                        group_hash=group,
                        checkpoint=str(seed),
                        variant=variant,
                        sse=float(cell["sse"]),
                        sae=float(cell["sae"]),
                        n=int(cell["n"]),
                    )
                )
    return output


def _comparison_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["comparator"]), str(row["metric"])): row
        for row in rows
    }


def _per_seed_rows(
    indexed: Mapping[tuple[str, int, str], Mapping[str, Any]],
    datasets: Sequence[str],
    seeds: Sequence[int],
    variants: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            baseline = indexed[(dataset, int(seed), "APN")]["metrics"]
            for variant in variants:
                metrics = indexed[(dataset, int(seed), variant)]["metrics"]
                mse = float(metrics["mse"])
                mae = float(metrics["mae"])
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": variant,
                        "mse": mse,
                        "mae": mae,
                        "n": int(sum(int(cell["n"]) for cell in indexed[(dataset, int(seed), variant)]["cells"])),
                        "mse_relative_gain_vs_apn": (
                            (float(baseline["mse"]) - mse) / float(baseline["mse"])
                        ),
                        "mae_relative_gain_vs_apn": (
                            (float(baseline["mae"]) - mae) / float(baseline["mae"])
                        ),
                    }
                )
    return rows


def aggregate_safe_campaign(
    manifests: Sequence[Mapping[str, Any]],
    gate_decisions: Mapping[str, Mapping[str, Any]],
    *,
    datasets: Sequence[str] = SAFE_DATASETS,
    seeds: Sequence[int] = SAFE_SEEDS,
    variants: Sequence[str] = SAFE_VARIANTS,
    resamples: int = 50_000,
    random_seed: int = 20_260_721,
    thresholds: SafeGateThresholds = SafeGateThresholds(),
) -> dict[str, Any]:
    """Aggregate the complete paired Safe matrix without model selection."""

    datasets = tuple(str(value) for value in datasets)
    seeds = tuple(int(value) for value in seeds)
    variants = tuple(str(value) for value in variants)
    if len(datasets) < 2 or len(seeds) != 5:
        raise SafeAggregationError("Safe requires at least two targets and exactly five seeds")
    if variants != SAFE_VARIANTS:
        raise SafeAggregationError("Safe variant registry differs from the frozen order")
    if set(gate_decisions) != set(datasets):
        raise SafeAggregationError("Gate decisions do not match target datasets")
    for dataset in datasets:
        _validate_gate_decision(dataset, gate_decisions[dataset])
    indexed = _validated_manifests(
        manifests, datasets=datasets, seeds=seeds, variants=variants,
        gate_decisions=gate_decisions,
    )
    seed_rows = _per_seed_rows(indexed, datasets, seeds, variants)

    analyses: dict[str, dict[str, Any]] = {}
    primary_keys: list[str] = []
    secondary_keys: list[str] = []
    for dataset in datasets:
        cells = _dataset_cells(indexed, dataset, seeds, variants)
        metrics = pooled_metrics(cells)
        apn_rows = crossed_cluster_checkpoint_bootstrap(
            cells,
            baseline="APN",
            comparators=("Safe",),
            resamples=resamples,
            random_seed=random_seed,
        )
        joint_rows = crossed_cluster_checkpoint_bootstrap(
            cells,
            baseline="Joint",
            comparators=("Safe",),
            resamples=resamples,
            random_seed=random_seed,
        )
        diagnostic_rows = crossed_cluster_checkpoint_bootstrap(
            cells,
            baseline="Safe",
            comparators=("Full", *SAFE_ABLATIONS),
            resamples=resamples,
            random_seed=random_seed,
        )
        apn_index = _comparison_index(apn_rows)
        joint_index = _comparison_index(joint_rows)
        diagnostic_index = _comparison_index(diagnostic_rows)
        comparisons: list[dict[str, Any]] = []
        for family, baseline, rows in (
            ("primary", "APN", apn_rows),
            ("secondary", "Joint", joint_rows),
            ("secondary", "Safe", diagnostic_rows),
        ):
            for row in rows:
                item = dict(row)
                item["family"] = family
                item["dataset"] = dataset
                item["comparison_id"] = (
                    f"{dataset}|{baseline}|{item['comparator']}|{item['metric']}"
                )
                comparisons.append(item)
                if item["metric"] == "mse":
                    if family == "primary":
                        primary_keys.append(item["comparison_id"])
                    else:
                        secondary_keys.append(item["comparison_id"])
        safe_seed = [
            row
            for row in seed_rows
            if row["dataset"] == dataset and row["variant"] == "Safe"
        ]
        analyses[dataset] = {
            "dataset": dataset,
            "gate_decision": dict(gate_decisions[dataset]),
            "pooled_metrics": metrics,
            "per_seed_safe": safe_seed,
            "safe_seed_mse_improvements": sum(
                row["mse_relative_gain_vs_apn"] > 0 for row in safe_seed
            ),
            "all_seed_harm_within_margin": all(
                row["mse_relative_gain_vs_apn"] >= -thresholds.harm_margin
                and row["mae_relative_gain_vs_apn"] >= -thresholds.harm_margin
                for row in safe_seed
            ),
            "apn_mse": dict(apn_index[("Safe", "mse")]),
            "apn_mae": dict(apn_index[("Safe", "mae")]),
            "joint_mse": dict(joint_index[("Safe", "mse")]),
            "joint_mae": dict(joint_index[("Safe", "mae")]),
            "diagnostics": {
                f"Safe_vs_{name}_{metric}": dict(row)
                for (name, metric), row in diagnostic_index.items()
            },
            "comparisons": comparisons,
        }

    primary_records = {
        row["comparison_id"]: row
        for analysis in analyses.values()
        for row in analysis["comparisons"]
        if row["comparison_id"] in primary_keys
    }
    primary_adjusted = holm_adjust(
        [float(primary_records[key]["one_sided_p_candidate_not_better"]) for key in primary_keys]
    )
    secondary_records = {
        row["comparison_id"]: row
        for analysis in analyses.values()
        for row in analysis["comparisons"]
        if row["comparison_id"] in secondary_keys
    }
    secondary_adjusted = holm_adjust(
        [float(secondary_records[key]["one_sided_p_candidate_not_better"]) for key in secondary_keys]
    )
    adjusted = dict(zip(primary_keys, primary_adjusted))
    adjusted.update(zip(secondary_keys, secondary_adjusted))
    for analysis in analyses.values():
        for row in analysis["comparisons"]:
            if row["comparison_id"] in adjusted:
                row["holm_adjusted_p"] = float(adjusted[row["comparison_id"]])

    positive_targets: list[str] = []
    target_checks: dict[str, dict[str, bool]] = {}
    for dataset, analysis in analyses.items():
        apn_mse = analysis["apn_mse"]
        apn_mae = analysis["apn_mae"]
        joint_mse = analysis["joint_mse"]
        primary_id = f"{dataset}|APN|Safe|mse"
        checks = {
            "safe_gate_enabled": bool(analysis["gate_decision"]["enabled"]),
            "positive_mse_gain": float(apn_mse["relative_gain"]) > 0,
            "mse_ci_excludes_zero": float(apn_mse["relative_gain_ci_low"]) > 0,
            "holm_one_sided_significant": float(adjusted[primary_id]) < thresholds.alpha,
            "at_least_four_seed_gains": int(analysis["safe_seed_mse_improvements"]) >= 4,
            "mse_harm_ucb_within_one_percent": float(apn_mse["relative_gain_ci_low"]) >= -thresholds.harm_margin,
            "mae_harm_ucb_within_one_percent": float(apn_mae["relative_gain_ci_low"]) >= -thresholds.harm_margin,
            "all_seed_harm_within_one_percent": bool(analysis["all_seed_harm_within_margin"]),
            "joint_point_noninferior": float(joint_mse["relative_gain"]) >= -thresholds.joint_noninferiority_margin,
            "joint_ci_noninferior": float(joint_mse["relative_gain_ci_low"]) >= -thresholds.joint_noninferiority_margin,
        }
        target_checks[dataset] = checks
        if all(checks.values()):
            positive_targets.append(dataset)

    global_cells: list[ErrorCell] = []
    for dataset in datasets:
        global_cells.extend(
            _dataset_cells(indexed, dataset, seeds, variants, prefix_dataset=True)
        )
    ablation_checks: dict[str, dict[str, Any]] = {}
    global_ablation_rows: dict[str, Mapping[str, Any]] = {}
    for ablation in SAFE_ABLATIONS:
        rows = crossed_cluster_checkpoint_bootstrap(
            global_cells,
            baseline=ablation,
            comparators=("Safe",),
            resamples=resamples,
            random_seed=random_seed,
        )
        global_ablation_rows[ablation] = _comparison_index(rows)[("Safe", "mse")]
    adjusted_ablation_p = holm_adjust([
        float(
            global_ablation_rows[name][
                "one_sided_p_candidate_not_better"
            ]
        )
        for name in SAFE_ABLATIONS
    ])
    for ablation, adjusted_p in zip(SAFE_ABLATIONS, adjusted_ablation_p):
        mse = global_ablation_rows[ablation]
        required = ablation in {"SafeNoRobust", "SafeNoBound"}
        supported = (
            float(mse["relative_gain"]) > 0
            and float(mse["relative_gain_ci_low"]) > 0
            and float(adjusted_p) < thresholds.alpha
        )
        ablation_checks[ablation] = {
            "relative_gain": float(mse["relative_gain"]),
            "relative_gain_ci_low": float(mse["relative_gain_ci_low"]),
            "one_sided_p": float(mse["one_sided_p_candidate_not_better"]),
            "holm_adjusted_p": float(adjusted_p),
            "required_for_gate": required,
            "supported": supported,
            "diagnostic_only": not required,
        }

    global_checks = {
        "at_least_two_positive_targets": len(positive_targets) >= 2,
        "every_declared_target_positive": set(positive_targets) == set(datasets),
        "all_target_and_seed_harm_within_one_percent": all(
            checks["all_seed_harm_within_one_percent"]
            and checks["mse_harm_ucb_within_one_percent"]
            and checks["mae_harm_ucb_within_one_percent"]
            for checks in target_checks.values()
        ),
        "safe_noninferior_to_joint_every_target": all(
            checks["joint_point_noninferior"] and checks["joint_ci_noninferior"]
            for checks in target_checks.values()
        ),
        "both_modules_supported_by_ablations": all(
            ablation_checks[name]["supported"]
            for name in ("SafeNoRobust", "SafeNoBound")
        ),
        "all_validation_gates_enabled": all(
            bool(gate_decisions[dataset]["enabled"]) for dataset in datasets
        ),
    }
    verdict = "PASS" if all(global_checks.values()) else "ABANDON"
    return {
        "schema_version": SAFE_AGGREGATION_SCHEMA,
        "verdict": verdict,
        "device_timing_authorized": verdict == "PASS",
        "datasets": list(datasets),
        "seeds": list(seeds),
        "variants": list(variants),
        "resamples": int(resamples),
        "random_seed": int(random_seed),
        "thresholds": {
            "harm_margin": thresholds.harm_margin,
            "joint_noninferiority_margin": thresholds.joint_noninferiority_margin,
            "alpha": thresholds.alpha,
        },
        "positive_targets": positive_targets,
        "per_seed_results": seed_rows,
        "target_analyses": analyses,
        "target_checks": target_checks,
        "ablation_checks": ablation_checks,
        "global_checks": global_checks,
    }


def write_safe_aggregation(report: Mapping[str, Any], path: str | Path) -> Path:
    if report.get("schema_version") != SAFE_AGGREGATION_SCHEMA:
        raise SafeAggregationError("Refusing to write an invalid Safe aggregation")
    destination = require_within_root(path)
    return atomic_write_json(destination, report)


def load_evaluation_manifests(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in paths:
        source = require_within_root(path, must_exist=True)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SafeAggregationError(f"Cannot read {source}: {exc}") from exc
        if not isinstance(value, dict):
            raise SafeAggregationError(f"Evaluation manifest is not an object: {source}")
        output.append(value)
    return output
