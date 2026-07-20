from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evipatch.aggregate import (
    decide_gate,
    hierarchical_paired_bootstrap,
    masked_errors_per_patient,
)
from evipatch.runner import load_stage_config


def test_masked_patient_metrics() -> None:
    pred = np.array([[[1.0, 9.0]], [[2.0, 4.0]]])
    true = np.zeros_like(pred)
    mask = np.array([[[1.0, 0.0]], [[1.0, 1.0]]])
    result = masked_errors_per_patient(pred, true, mask, np.array([1, 2]))
    assert result["MSE"].tolist() == [1.0, 10.0]
    assert result["MAE"].tolist() == [1.0, 3.0]


def test_hierarchical_bootstrap_resamples_seed_blocks_and_pairs() -> None:
    left = pd.DataFrame(
        [(seed, patient, 0.8 + patient / 1000) for seed in (2024, 2025, 2026) for patient in range(30)],
        columns=["seed", "patient_id", "MSE"],
    )
    right = left.copy()
    right["MSE"] += 0.1
    result = hierarchical_paired_bootstrap(
        left, right, "seed", "patient_id", "MSE", 1000, 44
    )
    assert result["n_seed_blocks"] == 3
    assert result["n_pairs"] == 90
    assert result["ci_high"] < 0
    assert result["probability_left_better"] == 1.0


def _gate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        [
            {"variant": "apn", "shift": "none", "MSE_mean": 1.0},
            {"variant": "evipatch_full", "shift": "none", "MSE_mean": 0.99},
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "comparison": f"evipatch_full_vs_{other}",
                "view": "macro",
                "metric": "MSE",
                "estimate": -0.1,
                "ci_low": -0.15,
                "ci_high": -0.05,
            }
            for other in ("raw_count", "shuffled_evidence", "random_features")
        ]
    )
    runs = pd.DataFrame(
        [
            {
                "variant": variant,
                "seed": seed,
                "parameter_count": 1000 if variant == "apn" else 1020,
                "train_wall_seconds": 100 if variant == "apn" else 102,
            }
            for variant in ("apn", "evipatch_full")
            for seed in (2024, 2025, 2026)
        ]
    )
    return summary, bootstrap, runs


def _write_timing(path: Path) -> Path:
    path.write_text(
        '{"records": ['
        '{"variant": "apn", "measured_steps": 100, "training_step_mean_ms": 10.0},'
        '{"variant": "evipatch_full", "measured_steps": 100, "training_step_mean_ms": 10.2}'
        ']}'
    )
    return path

def test_gate_passes_only_when_all_conditions_pass(tmp_path: Path) -> None:
    config = load_stage_config()
    config["controlled_support"]["minimum_pairs_per_seed"] = 5
    summary, bootstrap, runs = _gate_inputs()
    controlled = pd.DataFrame(
        [
            {"variant": variant, "seed": seed, "pair_id": pair, "MSE": value}
            for variant, value in (("apn", 1.0), ("evipatch_full", 0.9))
            for seed in (2024, 2025, 2026)
            for pair in range(5)
        ]
    )
    path = tmp_path / "controlled_support_errors.csv"
    controlled.to_csv(path, index=False)
    gate = decide_gate(summary, bootstrap, runs, config, path, _write_timing(tmp_path / "timing.json"))
    assert gate["verdict"] == "PASS"
    assert all(item["passed"] for item in gate["conditions"].values())


def test_missing_controlled_support_fails_closed(tmp_path: Path) -> None:
    config = load_stage_config()
    summary, bootstrap, runs = _gate_inputs()
    gate = decide_gate(
        summary,
        bootstrap,
        runs,
        config,
        tmp_path / "missing_controlled_support.csv",
        _write_timing(tmp_path / "timing.json"),
    )
    assert gate["verdict"] == "ABANDON"
    assert gate["conditions"]["controlled_support_improvement"]["passed"] is False

def test_missing_timing_fails_closed(tmp_path: Path) -> None:
    config = load_stage_config()
    config["controlled_support"]["minimum_pairs_per_seed"] = 1
    summary, bootstrap, runs = _gate_inputs()
    controlled = pd.DataFrame(
        [
            {"variant": variant, "seed": seed, "pair_id": 0, "MSE": value}
            for variant, value in (("apn", 1.0), ("evipatch_full", 0.9))
            for seed in (2024, 2025, 2026)
        ]
    )
    path = tmp_path / "controlled.csv"
    controlled.to_csv(path, index=False)
    gate = decide_gate(
        summary, bootstrap, runs, config, path, tmp_path / "missing_timing.json"
    )
    assert gate["verdict"] == "ABANDON"
    assert gate["conditions"]["time_overhead"]["passed"] is False
