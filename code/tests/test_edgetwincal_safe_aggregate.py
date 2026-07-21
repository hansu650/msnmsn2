from __future__ import annotations

import hashlib

import pytest

from edgetwincal.safe_aggregate import (
    SAFE_VARIANTS,
    SafeAggregationError,
    aggregate_safe_campaign,
)


DATASETS = ("beijing_air", "intel_lab")
SEEDS = (2024, 2025, 2026, 2027, 2028)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _manifests(
    *,
    safe_multiplier: float = 0.90,
    joint_multiplier: float = 0.95,
    gate_enabled: bool = True,
):
    multipliers = {
        "APN": 1.0,
        "Joint": joint_multiplier,
        "Full": 0.94,
        "Safe": safe_multiplier if gate_enabled else 1.0,
        "SafeNoBalance": 0.96,
        "SafeNoRobust": 0.97,
        "SafeNoBound": 0.99,
        "SafeNoGate": 0.92,
    }
    rows = []
    for dataset in DATASETS:
        shared = {
            name: _hash(f"{dataset}|{name}")
            for name in (
                "config",
                "split",
                "normalizer",
                "sample",
                "group",
                "mask",
                "target",
                "protocol",
            )
        }
        gate_hash = _hash(f"{dataset}|gate|{gate_enabled}")
        for seed in SEEDS:
            checkpoint_hash = _hash(f"{dataset}|{seed}|checkpoint")
            apn_prediction_hash = _hash(
                f"{dataset}|{seed}|prediction|APN"
            )
            for variant in SAFE_VARIANTS:
                cells = []
                for group in range(24):
                    n = 100
                    base = 100.0 + group + (seed - 2024)
                    sse = base * multipliers[variant]
                    sae = (base ** 0.5) * multipliers[variant]
                    cells.append(
                        {
                            "group_hash": _hash(f"{dataset}|group|{group}"),
                            "sse": sse,
                            "sae": sae,
                            "n": n,
                        }
                    )
                total_n = sum(cell["n"] for cell in cells)
                prediction_hash = (
                    apn_prediction_hash
                    if variant == "APN"
                    or (variant == "Safe" and not gate_enabled)
                    else _hash(f"{dataset}|{seed}|prediction|{variant}")
                )
                rows.append(
                    {
                        "schema_version": "edgetwincal.safe-evaluation.v1",
                        "status": "complete",
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "checkpoint_sha256": checkpoint_hash,
                        "config_sha256": shared["config"],
                        "split_sha256": shared["split"],
                        "normalizer_sha256": shared["normalizer"],
                        "sample_ids_sha256": shared["sample"],
                        "group_ids_sha256": shared["group"],
                        "mask_sha256": shared["mask"],
                        "target_sha256": shared["target"],
                        "protocol_sha256": shared["protocol"],
                        "gate_sha256": gate_hash,
                        "prediction_sha256": prediction_hash,
                        "cells": cells,
                        "metrics": {
                            "mse": sum(cell["sse"] for cell in cells) / total_n,
                            "mae": sum(cell["sae"] for cell in cells) / total_n,
                        },
                    }
                )
    return rows


def _gates(enabled: bool = True):
    return {
        dataset: {
            "enabled": enabled,
            "checkpoints": 5,
            "validation_groups": 24,
            "validation_cells": 2400,
            "gate_sha256": _hash(f"{dataset}|gate|{enabled}"),
        }
        for dataset in DATASETS
    }


def test_complete_strong_campaign_passes():
    report = aggregate_safe_campaign(
        _manifests(), _gates(), resamples=1000, random_seed=77
    )
    assert report["verdict"] == "PASS"
    assert report["device_timing_authorized"] is True
    assert set(report["positive_targets"]) == set(DATASETS)
    assert len(report["per_seed_results"]) == 2 * 5 * len(SAFE_VARIANTS)
    assert set(report["ablation_checks"]) == {
        "SafeNoBalance", "SafeNoRobust", "SafeNoBound", "SafeNoGate"
    }
    assert all(
        "holm_adjusted_p" in row
        for row in report["ablation_checks"].values()
    )
    assert report["ablation_checks"]["SafeNoBalance"]["diagnostic_only"]
    assert report["ablation_checks"]["SafeNoGate"]["diagnostic_only"]
    assert report["ablation_checks"]["SafeNoRobust"]["required_for_gate"]
    assert report["ablation_checks"]["SafeNoBound"]["required_for_gate"]


def test_disabled_validation_gate_forces_abandon():
    report = aggregate_safe_campaign(
        _manifests(gate_enabled=False), _gates(enabled=False),
        resamples=300, random_seed=7,
    )
    assert report["verdict"] == "ABANDON"
    assert not report["global_checks"]["all_validation_gates_enabled"]


def test_safe_worse_than_joint_fails_noninferiority():
    report = aggregate_safe_campaign(
        _manifests(safe_multiplier=0.98, joint_multiplier=0.90),
        _gates(),
        resamples=300,
        random_seed=8,
    )
    assert report["verdict"] == "ABANDON"
    assert not report["global_checks"]["safe_noninferior_to_joint_every_target"]


def test_missing_or_unpaired_matrix_is_rejected():
    manifests = _manifests()
    manifests.pop()
    with pytest.raises(SafeAggregationError, match="matrix differs"):
        aggregate_safe_campaign(manifests, _gates(), resamples=20)

    manifests = _manifests()



def test_cross_seed_hash_pairing_and_exact_disabled_fallback_are_enforced():
    manifests = _manifests()
    manifests[len(SAFE_VARIANTS)]["target_sha256"] = _hash("changed-target")
    with pytest.raises(SafeAggregationError, match="target_sha256"):
        aggregate_safe_campaign(manifests, _gates(), resamples=20)

    manifests = _manifests(gate_enabled=False)
    safe = next(
        row for row in manifests
        if row["dataset"] == DATASETS[0]
        and row["seed"] == SEEDS[0]
        and row["variant"] == "Safe"
    )
    safe["prediction_sha256"] = _hash("not-apn")
    with pytest.raises(SafeAggregationError, match="exact APN"):
        aggregate_safe_campaign(manifests, _gates(enabled=False), resamples=20)


def test_gate_hash_pairs_decision_and_evaluations():
    manifests = _manifests()
    manifests[0]["gate_sha256"] = _hash("other-gate")
    with pytest.raises(SafeAggregationError, match="gate_sha256"):
        aggregate_safe_campaign(manifests, _gates(), resamples=20)
    manifests[1]["group_ids_sha256"] = _hash("different")
    with pytest.raises(SafeAggregationError, match="group_ids_sha256"):
        aggregate_safe_campaign(manifests, _gates(), resamples=20)
