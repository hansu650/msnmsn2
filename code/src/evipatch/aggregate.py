"""Stage A metric aggregation, paired bootstrap, and automatic kill gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evipatch.paths import assert_within_project, ensure_project_dir, project_path


@dataclass(frozen=True)
class Evaluation:
    path: Path
    metric: dict[str, float]
    pred: np.ndarray
    true: np.ndarray
    mask: np.ndarray
    sample_ids: np.ndarray
    manifest: dict[str, Any]


def load_evaluation(path: Path | str) -> Evaluation:
    """Load and validate one APN evaluation directory."""
    root = assert_within_project(path)
    files = {
        "metric": root / "metric.json",
        "pred": root / "output_pred.npy",
        "true": root / "input_y.npy",
        "mask": root / "input_y_mask.npy",
        "ids": root / "input_sample_ID.npy",
        "manifest": root / "run_manifest.json",
    }
    missing = [name for name, file in files.items() if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation {root} is missing: {missing}")

    metric = json.loads(files["metric"].read_text(encoding="utf-8"))
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    pred = np.load(files["pred"], allow_pickle=False)
    true = np.load(files["true"], allow_pickle=False)
    mask = np.load(files["mask"], allow_pickle=False)
    sample_ids = np.load(files["ids"], allow_pickle=False).reshape(-1)

    if pred.shape != true.shape or pred.shape != mask.shape:
        raise ValueError(
            f"Prediction/target/mask shapes differ at {root}: "
            f"{pred.shape}, {true.shape}, {mask.shape}"
        )
    if pred.shape[0] != sample_ids.shape[0]:
        raise ValueError("sample ID count does not match evaluation batch dimension")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError(f"Non-finite prediction or target values at {root}")
    if not np.isfinite(mask).all():
        raise ValueError(f"Non-finite target mask values at {root}")
    for name in ("MSE", "MAE"):
        value = float(metric[name])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite {name} in {files['metric']}")
        metric[name] = value
    return Evaluation(root, metric, pred, true, mask, sample_ids, manifest)


def masked_errors_per_patient(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    ids: np.ndarray,
) -> pd.DataFrame:
    """Compute target-mask-aware MSE and MAE for every patient/sample ID."""
    if pred.shape != true.shape or pred.shape != mask.shape:
        raise ValueError("pred, true, and mask must have identical shapes")
    if pred.shape[0] != np.asarray(ids).reshape(-1).shape[0]:
        raise ValueError("ids must match the first tensor dimension")

    records: list[dict[str, Any]] = []
    flat_ids = np.asarray(ids).reshape(-1)
    for sample_index, sample_id in enumerate(flat_ids):
        valid = mask[sample_index] > 0
        count = int(valid.sum())
        if count:
            residual = pred[sample_index][valid] - true[sample_index][valid]
            squared_sum = float(np.square(residual).sum())
            absolute_sum = float(np.abs(residual).sum())
        else:
            squared_sum = 0.0
            absolute_sum = 0.0
        records.append(
            {
                "patient_id": int(sample_id),
                "squared_error_sum": squared_sum,
                "absolute_error_sum": absolute_sum,
                "observed_target_count": count,
            }
        )

    frame = pd.DataFrame.from_records(records)
    grouped = (
        frame.groupby("patient_id", as_index=False)
        .agg(
            squared_error_sum=("squared_error_sum", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            observed_target_count=("observed_target_count", "sum"),
        )
        .sort_values("patient_id")
    )
    denominator = grouped["observed_target_count"].replace(0, np.nan)
    grouped["MSE"] = grouped["squared_error_sum"] / denominator
    grouped["MAE"] = grouped["absolute_error_sum"] / denominator
    return grouped


def hierarchical_paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    seed_column: str,
    id_column: str,
    metric: str,
    n_resamples: int,
    rng_seed: int,
) -> dict[str, float | int]:
    """Resample seed blocks, then paired IDs within each sampled seed."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    required = {seed_column, id_column, metric}
    if not required.issubset(left.columns) or not required.issubset(right.columns):
        raise ValueError(f"Both frames must contain {sorted(required)}")
    if left.duplicated([seed_column, id_column]).any():
        raise ValueError("left contains duplicate seed-ID pairs")
    if right.duplicated([seed_column, id_column]).any():
        raise ValueError("right contains duplicate seed-ID pairs")

    paired = left[list(required)].merge(
        right[list(required)],
        on=[seed_column, id_column],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    paired["difference"] = paired[f"{metric}_left"] - paired[f"{metric}_right"]
    paired = paired[np.isfinite(paired["difference"])]
    seeds = np.sort(paired[seed_column].unique())
    if seeds.size == 0:
        raise ValueError("No finite paired observations for bootstrap")

    differences_by_seed = {
        seed: paired.loc[paired[seed_column] == seed, "difference"].to_numpy(float)
        for seed in seeds
    }
    if any(values.size == 0 for values in differences_by_seed.values()):
        raise ValueError("At least one seed block has no paired observations")

    rng = np.random.default_rng(rng_seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled_seeds = rng.choice(seeds, size=seeds.size, replace=True)
        block_means = []
        for seed in sampled_seeds:
            values = differences_by_seed[seed]
            sampled_values = rng.choice(values, size=values.size, replace=True)
            block_means.append(float(sampled_values.mean()))
        estimates[index] = float(np.mean(block_means))

    alpha = 0.025
    point = float(
        np.mean([values.mean() for values in differences_by_seed.values()])
    )
    right_mean = float(
        right.loc[np.isfinite(right[metric])]
        .groupby(seed_column)[metric]
        .mean()
        .mean()
    )
    return {
        "estimate": point,
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "probability_left_better": float(np.mean(estimates < 0)),
        "relative_effect": float(point / right_mean) if right_mean != 0 else np.nan,
        "n_resamples": int(n_resamples),
        "n_seed_blocks": int(seeds.size),
        "n_pairs": int(paired.shape[0]),
    }


def _find_evaluation(
    results_root: Path,
    variant: str,
    seed: int,
    shift: str,
) -> Path:
    name = "native" if shift == "none" else shift
    matches = list(
        (results_root / "stage_a" / variant / str(seed)).glob(
            f"checkpoints/**/eval_{name}/metric.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {variant}/{seed}/{shift} evaluation, found {len(matches)}"
        )
    return assert_within_project(matches[0].parent)


def summarize_stage_a(
    root: Path | str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create per-run, patient-level, and mean/std summary tables."""
    results_root = assert_within_project(root)
    run_records: list[dict[str, Any]] = []
    patient_frames: list[pd.DataFrame] = []

    for variant in config["stage_a"]["variants"]:
        for seed in config["stage_a"]["seeds"]:
            train_manifest_path = (
                results_root / "stage_a" / variant / str(seed) / "train_manifest.json"
            )
            if not train_manifest_path.is_file():
                raise FileNotFoundError(train_manifest_path)
            train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
            for shift in config["stage_a"]["shifts"]:
                evaluation = load_evaluation(
                    _find_evaluation(results_root, variant, seed, shift)
                )
                record = {
                    "variant": variant,
                    "seed": seed,
                    "shift": shift,
                    "MSE": evaluation.metric["MSE"],
                    "MAE": evaluation.metric["MAE"],
                    "parameter_count": train_manifest["parameter_count"],
                    "train_wall_seconds": train_manifest["process"]["wall_seconds"],
                    "train_peak_gpu_memory_mib": train_manifest["process"][
                        "peak_gpu_memory_mib"
                    ],
                    "inference_wall_seconds": evaluation.manifest["process"][
                        "wall_seconds"
                    ],
                    "inference_peak_gpu_memory_mib": evaluation.manifest["process"][
                        "peak_gpu_memory_mib"
                    ],
                    "evaluation_path": str(evaluation.path),
                }
                run_records.append(record)
                patient = masked_errors_per_patient(
                    evaluation.pred,
                    evaluation.true,
                    evaluation.mask,
                    evaluation.sample_ids,
                )
                patient.insert(0, "shift", shift)
                patient.insert(0, "seed", seed)
                patient.insert(0, "variant", variant)
                patient_frames.append(patient)

    runs = pd.DataFrame.from_records(run_records)
    patients = pd.concat(patient_frames, ignore_index=True)
    summary = (
        runs.groupby(["variant", "shift"], as_index=False)
        .agg(
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            parameter_count_mean=("parameter_count", "mean"),
            train_wall_seconds_mean=("train_wall_seconds", "mean"),
            train_wall_seconds_std=("train_wall_seconds", "std"),
            inference_wall_seconds_mean=("inference_wall_seconds", "mean"),
            inference_wall_seconds_std=("inference_wall_seconds", "std"),
            train_peak_gpu_memory_mib_mean=("train_peak_gpu_memory_mib", "mean"),
        )
        .sort_values(["variant", "shift"])
    )
    macro_by_seed = (
        runs.groupby(["variant", "seed"], as_index=False)[["MSE", "MAE"]].mean()
    )
    macro_summary = (
        macro_by_seed.groupby("variant", as_index=False)
        .agg(
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
        )
        .assign(shift="macro")
    )
    summary = pd.concat([summary, macro_summary], ignore_index=True, sort=False)
    return runs, patients, summary


def _view_patients(patients: pd.DataFrame, variant: str, view: str) -> pd.DataFrame:
    selected = patients[patients["variant"] == variant].copy()
    if view == "macro":
        return (
            selected.groupby(["seed", "patient_id"], as_index=False)[["MSE", "MAE"]]
            .mean()
        )
    return selected[selected["shift"] == view][
        ["seed", "patient_id", "MSE", "MAE"]
    ].copy()


def _bootstrap_table(
    patients: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    statistics = config["statistics"]
    comparisons = ["apn", "raw_count", "shuffled_evidence", "random_features"]
    records: list[dict[str, Any]] = []
    for other in comparisons:
        for view in [*config["stage_a"]["shifts"], "macro"]:
            left = _view_patients(patients, "evipatch_full", view)
            right = _view_patients(patients, other, view)
            for metric in ("MSE", "MAE"):
                result = hierarchical_paired_bootstrap(
                    left,
                    right,
                    "seed",
                    "patient_id",
                    metric,
                    statistics["bootstrap_resamples"],
                    statistics["bootstrap_rng_seed"],
                )
                records.append(
                    {
                        "comparison": f"evipatch_full_vs_{other}",
                        "left": "evipatch_full",
                        "right": other,
                        "view": view,
                        "metric": metric,
                        **result,
                    }
                )
    return pd.DataFrame.from_records(records)


def _summary_value(
    summary: pd.DataFrame,
    variant: str,
    shift: str,
    column: str,
) -> float:
    values = summary.loc[
        (summary["variant"] == variant) & (summary["shift"] == shift), column
    ]
    if len(values) != 1:
        raise ValueError(f"Missing unique summary value for {variant}/{shift}/{column}")
    return float(values.iloc[0])


def _bootstrap_row(
    bootstrap: pd.DataFrame,
    other: str,
    view: str = "macro",
    metric: str = "MSE",
) -> pd.Series:
    rows = bootstrap[
        (bootstrap["comparison"] == f"evipatch_full_vs_{other}")
        & (bootstrap["view"] == view)
        & (bootstrap["metric"] == metric)
    ]
    if len(rows) != 1:
        raise ValueError(f"Missing bootstrap row for full vs {other}/{view}/{metric}")
    return rows.iloc[0]


def decide_gate(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    runs: pd.DataFrame,
    config: dict[str, Any],
    controlled_support_path: Path | str,
    timing_path: Path | str,
) -> dict[str, Any]:
    """Apply every mandatory Stage A condition without manual overrides."""
    thresholds = config["gate"]
    conditions: dict[str, dict[str, Any]] = {}

    controlled_path = assert_within_project(controlled_support_path)
    minimum_pairs = int(config["controlled_support"]["minimum_pairs_per_seed"])
    expected_seeds = [int(seed) for seed in config["stage_a"]["seeds"]]
    controlled_pass = False
    if controlled_path.is_file():
        controlled = pd.read_csv(controlled_path)
        required = {"variant", "seed", "pair_id", "MSE"}
        if not required.issubset(controlled.columns):
            raise ValueError(f"Controlled-support file must contain {sorted(required)}")
        apn_rows = controlled[controlled["variant"] == "apn"]
        pair_counts = {
            seed: int(apn_rows.loc[apn_rows["seed"] == seed, "pair_id"].nunique())
            for seed in expected_seeds
        }
        yield_passed = all(count >= minimum_pairs for count in pair_counts.values())
        means = controlled.groupby("variant")["MSE"].mean()
        if {"apn", "evipatch_full"}.issubset(means.index):
            apn_controlled = float(means["apn"])
            full_controlled = float(means["evipatch_full"])
            controlled_improvement = (
                (apn_controlled - full_controlled) / apn_controlled
                if apn_controlled != 0
                else -np.inf
            )
            controlled_pass = bool(
                yield_passed
                and np.isfinite(controlled_improvement)
                and controlled_improvement
                >= thresholds["controlled_support_relative_improvement_min"]
            )
            controlled_detail = {
                "available": True,
                "yield_passed": yield_passed,
                "pair_counts_by_seed": pair_counts,
                "minimum_pairs_per_seed": minimum_pairs,
                "apn_mse": apn_controlled,
                "full_mse": full_controlled,
                "relative_improvement": controlled_improvement,
                "threshold": thresholds["controlled_support_relative_improvement_min"],
            }
        else:
            controlled_detail = {
                "available": True,
                "yield_passed": yield_passed,
                "pair_counts_by_seed": pair_counts,
                "minimum_pairs_per_seed": minimum_pairs,
                "reason": "missing apn or evipatch_full controlled errors",
                "threshold": thresholds["controlled_support_relative_improvement_min"],
            }
    else:
        controlled_detail = {
            "available": False,
            "yield_passed": False,
            "reason": f"missing {controlled_path}",
            "minimum_pairs_per_seed": minimum_pairs,
            "threshold": thresholds["controlled_support_relative_improvement_min"],
        }
    conditions["controlled_support_improvement"] = {
        "passed": bool(controlled_pass),
        **controlled_detail,
    }
    raw_row = _bootstrap_row(bootstrap, "raw_count")
    conditions["full_beats_raw_macro"] = {
        "passed": bool(raw_row["ci_high"] < 0),
        "estimate_full_minus_raw": float(raw_row["estimate"]),
        "ci_low": float(raw_row["ci_low"]),
        "ci_high": float(raw_row["ci_high"]),
    }

    for control in ("shuffled_evidence", "random_features"):
        row = _bootstrap_row(bootstrap, control)
        conditions[f"full_beats_{control}"] = {
            "passed": bool(row["ci_high"] < 0),
            "estimate_full_minus_control": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }

    native_apn = _summary_value(summary, "apn", "none", "MSE_mean")
    native_full = _summary_value(summary, "evipatch_full", "none", "MSE_mean")
    native_regression = (native_full - native_apn) / native_apn
    conditions["native_mse_regression"] = {
        "passed": bool(native_regression <= thresholds["native_mse_regression_max"]),
        "relative_regression": native_regression,
        "maximum": thresholds["native_mse_regression_max"],
        "apn_mse": native_apn,
        "full_mse": native_full,
    }

    train_unique = runs.drop_duplicates(["variant", "seed"])
    apn_parameters = float(
        train_unique.loc[train_unique["variant"] == "apn", "parameter_count"].mean()
    )
    full_parameters = float(
        train_unique.loc[
            train_unique["variant"] == "evipatch_full", "parameter_count"
        ].mean()
    )
    parameter_overhead = (full_parameters - apn_parameters) / apn_parameters
    conditions["parameter_overhead"] = {
        "passed": bool(parameter_overhead < thresholds["parameter_overhead_max"]),
        "relative_overhead": parameter_overhead,
        "maximum": thresholds["parameter_overhead_max"],
        "apn_parameter_count": apn_parameters,
        "full_parameter_count": full_parameters,
    }

    timing_resolved = assert_within_project(timing_path)
    time_passed = False
    if timing_resolved.is_file():
        timing = json.loads(timing_resolved.read_text(encoding="utf-8"))
        records = {record["variant"]: record for record in timing.get("records", [])}
        if {"apn", "evipatch_full"}.issubset(records):
            apn_record = records["apn"]
            full_record = records["evipatch_full"]
            measured_steps_valid = (
                int(apn_record.get("measured_steps", -1)) == 100
                and int(full_record.get("measured_steps", -1)) == 100
            )
            apn_time = float(apn_record["training_step_mean_ms"])
            full_time = float(full_record["training_step_mean_ms"])
            time_overhead = (
                (full_time - apn_time) / apn_time if apn_time > 0 else np.inf
            )
            time_passed = bool(
                measured_steps_valid
                and np.isfinite(time_overhead)
                and time_overhead < thresholds["time_overhead_max"]
            )
            time_detail = {
                "available": True,
                "measured_steps_valid": measured_steps_valid,
                "apn_step_mean_ms": apn_time,
                "full_step_mean_ms": full_time,
                "relative_overhead": time_overhead,
                "maximum": thresholds["time_overhead_max"],
                "source": str(timing_resolved),
            }
        else:
            time_detail = {
                "available": True,
                "reason": "timing artifact is missing apn or evipatch_full",
                "maximum": thresholds["time_overhead_max"],
                "source": str(timing_resolved),
            }
    else:
        time_detail = {
            "available": False,
            "reason": f"missing {timing_resolved}",
            "maximum": thresholds["time_overhead_max"],
            "source": str(timing_resolved),
        }
    conditions["time_overhead"] = {"passed": time_passed, **time_detail}
    passed = all(condition["passed"] for condition in conditions.values())
    return {
        "verdict": "PASS" if passed else "ABANDON",
        "conditions": conditions,
        "all_mandatory_conditions_passed": passed,
        "controlled_support_path": str(controlled_path),
        "timing_path": str(timing_resolved),
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_report(
    output: Path,
    gate: dict[str, Any],
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# EviPatch 实验结果报告",
        "",
        f"- Stage A verdict: **{gate['verdict']}**",
        "- 本报告是实验与可复现性汇总，不是论文正文。",
        "",
        "## Kill gate",
        "",
    ]
    for name, detail in gate["conditions"].items():
        status = "通过" if detail["passed"] else "失败"
        lines.append(f"- {name}：{status}；{json.dumps(detail, ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "## 三种子汇总",
            "",
            "下表数值为三种子 mean ± std；macro 对 native/MCAR/burst 等权平均。",
            "",
            "| Variant | View | MSE | MAE |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['shift']} | "
            f"{row['MSE_mean']:.6g} ± {row['MSE_std']:.3g} | "
            f"{row['MAE_mean']:.6g} ± {row['MAE_std']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "所有预声明条件同时满足，允许进入条件扩展。"
                if gate["verdict"] == "PASS"
                else "至少一个预声明条件失败，项目按协议标记为 ABANDON；不扩展新数据集或堆叠新模块。"
            ),
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def run_aggregation(config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate all Stage A results, bootstrap pairs, and decide the gate."""
    artifacts = ensure_project_dir(config["project"]["artifacts_root"])
    runs, patients, summary = summarize_stage_a(
        config["project"]["results_root"], config
    )
    bootstrap = _bootstrap_table(patients, config)
    runs.to_csv(artifacts / "stage_a_runs.csv", index=False)
    patients.to_csv(artifacts / "patients.csv", index=False)
    summary.to_csv(artifacts / "stage_a_summary.csv", index=False)
    bootstrap.to_csv(artifacts / "paired_bootstrap.csv", index=False)

    from evipatch.controlled import run_controlled_support

    run_controlled_support(config)
    controlled_path = artifacts / "controlled_support_errors.csv"
    timing_path = artifacts / "timing_100_steps.json"
    gate = decide_gate(
        runs=runs,
        summary=summary,
        bootstrap=bootstrap,
        config=config,
        controlled_support_path=controlled_path,
        timing_path=timing_path,
    )
    (artifacts / "gate_decision.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(artifacts / "REPORT_CN.md", gate, summary)
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_path("code", "configs", "stage_a.json"),
    )
    args = parser.parse_args(argv)
    from evipatch.runner import load_stage_config

    gate = run_aggregation(load_stage_config(args.config))
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
