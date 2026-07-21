from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .paths import PROJECT_ROOT, ensure_directory, require_within_root


SEEDS = (2024, 2025, 2026)
VARIANTS = ("apn", "slrh", "cfg", "full")
METRICS = ("mse", "mae")


def load_runs() -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        run_dir = require_within_root(
            PROJECT_ROOT / "results" / "edgetwincal" / f"seed_{seed}",
            must_exist=True,
        )
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        arrays = np.load(run_dir / "test_outputs.npz")
        runs[seed] = {"metrics": metrics, "arrays": arrays}
    return runs


def patient_rows(runs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, run in runs.items():
        arrays = run["arrays"]
        target = arrays["target"]
        mask = arrays["target_mask"] > 0
        sample_ids = arrays["sample_id"]
        for variant in VARIANTS:
            prediction = arrays[f"prediction_{variant}"]
            for index, sample_id in enumerate(sample_ids):
                selected = mask[index]
                error = prediction[index][selected] - target[index][selected]
                if error.size == 0:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "sample_id": int(sample_id),
                        "mse": float(np.mean(np.square(error), dtype=np.float64)),
                        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
                        "observed_targets": int(error.size),
                    }
                )
    return rows


def summarize(runs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for metric in METRICS:
            values = np.asarray(
                [runs[seed]["metrics"]["test_metrics"][variant][metric] for seed in SEEDS],
                dtype=np.float64,
            )
            baseline = np.asarray(
                [runs[seed]["metrics"]["test_metrics"]["apn"][metric] for seed in SEEDS],
                dtype=np.float64,
            )
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "relative_improvement_vs_apn": float(
                        (baseline.mean() - values.mean()) / baseline.mean()
                    ),
                    **{f"seed_{seed}": float(value) for seed, value in zip(SEEDS, values)},
                }
            )
    return rows


def compact_run_manifest(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep reproducibility metadata while removing machine-local asset paths."""
    assets = metrics.get("assets", {})
    return {
        "schema_version": metrics.get("schema_version"),
        "attempt": metrics.get("attempt"),
        "seed": metrics["seed"],
        "device": metrics.get("device"),
        "passed": metrics["passed"],
        "pass_threshold": metrics.get("pass_threshold"),
        "validation_metrics": metrics["validation_metrics"],
        "test_metrics": metrics["test_metrics"],
        "relative_improvement_vs_apn": metrics["relative_improvement_vs_apn"],
        "fitted_coefficients": metrics.get("fitted_coefficients"),
        "nonzero_coefficients": metrics.get("nonzero_coefficients"),
        "fitting": metrics.get("fitting"),
        "checkpoint_sha256": assets.get("checkpoint_sha256"),
    }


def hierarchical_bootstrap(
    rows: list[dict[str, Any]],
    *,
    resamples: int = 10000,
    random_seed: int = 20260721,
) -> list[dict[str, Any]]:
    lookup: dict[tuple[int, str, str], dict[int, float]] = {}
    for row in rows:
        for metric in METRICS:
            lookup.setdefault((row["seed"], row["variant"], metric), {})[
                row["sample_id"]
            ] = row[metric]
    rng = np.random.default_rng(random_seed)
    output: list[dict[str, Any]] = []
    for comparator in ("apn", "slrh", "cfg"):
        for metric in METRICS:
            differences: list[np.ndarray] = []
            for seed in SEEDS:
                full = lookup[(seed, "full", metric)]
                other = lookup[(seed, comparator, metric)]
                ids = sorted(set(full) & set(other))
                differences.append(
                    np.asarray([full[sample_id] - other[sample_id] for sample_id in ids])
                )
            draws = np.empty(resamples, dtype=np.float64)
            for index in range(resamples):
                sampled_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
                seed_means = []
                for seed_index in sampled_seeds:
                    values = differences[int(seed_index)]
                    patient_indices = rng.integers(0, values.size, size=values.size)
                    seed_means.append(float(values[patient_indices].mean()))
                draws[index] = float(np.mean(seed_means))
            estimate = float(np.mean([values.mean() for values in differences]))
            output.append(
                {
                    "comparison": f"full_minus_{comparator}",
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "resamples": resamples,
                    "random_seed": random_seed,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_aggregation() -> dict[str, Any]:
    runs = load_runs()
    patients = patient_rows(runs)
    summary = summarize(runs)
    bootstrap = hierarchical_bootstrap(patients)
    artifacts = ensure_directory(PROJECT_ROOT / "artifacts")
    write_csv(require_within_root(artifacts / "edgetwincal_patients.csv"), patients)
    write_csv(require_within_root(artifacts / "edgetwincal_summary.csv"), summary)
    write_csv(require_within_root(artifacts / "edgetwincal_bootstrap.csv"), bootstrap)
    payload = {
        "title": "EdgeTwinCal: Dual-Space Calibration for Frozen Irregular-Sensor Digital Twins",
        "track": "Edge Computing, IoT and Digital Twins",
        "baseline": "APN (AAAI 2026), pre-existing checkpoints trained locally with the released implementation, frozen",
        "seeds": list(SEEDS),
        "summary": summary,
        "run_manifests": [
            compact_run_manifest(runs[seed]["metrics"]) for seed in SEEDS
        ],
        "hierarchical_paired_bootstrap": bootstrap,
        "all_seed_kill_tests_passed": all(runs[seed]["metrics"]["passed"] for seed in SEEDS),
    }
    json_path = require_within_root(artifacts / "edgetwincal_results.json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    run_aggregation()
