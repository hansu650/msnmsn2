from __future__ import annotations

import math

import pytest
import torch

from edgetwincal.statistics import (
    accumulate_error_cells,
    classify_safety,
    crossed_cluster_checkpoint_bootstrap,
    holm_adjust,
    pooled_metrics,
)
from edgetwincal.timing import (
    PhaseTimer,
    serialized_state_bytes,
    validate_timing_records,
    warm_inference,
)


def _constant_effect_records() -> list[dict]:
    rows: list[dict] = []
    for group_index in range(7):
        for checkpoint_index in range(3):
            n = 2 + group_index + checkpoint_index
            baseline_mse = 0.8 + 0.03 * group_index + 0.02 * checkpoint_index
            baseline_mae = 0.6 + 0.01 * group_index
            for variant, mse, mae in (
                ("apn", baseline_mse, baseline_mae),
                ("full", baseline_mse - 0.1, baseline_mae - 0.05),
            ):
                rows.append(
                    {
                        "group_hash": f"g{group_index}",
                        "checkpoint": str(2024 + checkpoint_index),
                        "variant": variant,
                        "sse": mse * n,
                        "sae": mae * n,
                        "n": n,
                    }
                )
    return rows


def test_accumulation_is_order_invariant_and_sums_duplicate_windows() -> None:
    records = _constant_effect_records()
    duplicated = records + [
        {
            **records[0],
            "sse": records[0]["sse"] * 0.5,
            "sae": records[0]["sae"] * 0.5,
            "n": records[0]["n"] // 2,
        }
    ]
    forward = accumulate_error_cells(duplicated)
    reverse = accumulate_error_cells(reversed(duplicated))
    assert forward == reverse
    first = next(
        cell
        for cell in forward
        if cell.group_hash == "g0"
        and cell.checkpoint == "2024"
        and cell.variant == "apn"
    )
    assert first.sse > records[0]["sse"]


def test_pooled_micro_uses_sse_over_n_not_macro_mean() -> None:
    cells = accumulate_error_cells(
        [
            {
                "group_hash": "small",
                "checkpoint": "2024",
                "variant": "apn",
                "sse": 1.0,
                "sae": 1.0,
                "n": 1,
            },
            {
                "group_hash": "large",
                "checkpoint": "2024",
                "variant": "apn",
                "sse": 27.0,
                "sae": 9.0,
                "n": 9,
            },
        ]
    )
    metrics = pooled_metrics(cells)["apn"]
    assert metrics["mse"] == pytest.approx(2.8)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["n"] == 10


def test_crossed_bootstrap_known_constant_effect_is_exact() -> None:
    cells = accumulate_error_cells(_constant_effect_records())
    rows = crossed_cluster_checkpoint_bootstrap(
        cells,
        baseline="apn",
        comparators=["full"],
        resamples=2000,
        random_seed=20260721,
        batch_size=200,
    )
    mse = next(row for row in rows if row["metric"] == "mse")
    mae = next(row for row in rows if row["metric"] == "mae")
    assert mse["effect_candidate_minus_baseline"] == pytest.approx(-0.1)
    assert mse["effect_ci_low"] == pytest.approx(-0.1)
    assert mse["effect_ci_high"] == pytest.approx(-0.1)
    assert mae["effect_candidate_minus_baseline"] == pytest.approx(-0.05)
    assert mae["effect_ci_low"] == pytest.approx(-0.05)
    assert mae["effect_ci_high"] == pytest.approx(-0.05)
    assert mse["paired_multiplicities"] is True
    assert mse["groups"] == 7 and mse["checkpoints"] == 3


def test_crossed_bootstrap_excludes_shared_globally_empty_groups() -> None:
    records = _constant_effect_records()
    for checkpoint in ("2024", "2025", "2026"):
        for variant in ("apn", "full"):
            records.append(
                {
                    "group_hash": "empty",
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "sse": 0.0,
                    "sae": 0.0,
                    "n": 0,
                }
            )
    rows = crossed_cluster_checkpoint_bootstrap(
        accumulate_error_cells(records),
        baseline="apn",
        comparators=["full"],
        resamples=100,
        random_seed=20260721,
    )
    mse = next(row for row in rows if row["metric"] == "mse")
    assert mse["groups"] == 7
    assert mse["effect_candidate_minus_baseline"] == pytest.approx(-0.1)


def test_crossed_bootstrap_rejects_variant_target_count_mismatch() -> None:
    records = _constant_effect_records()
    row = next(
        item
        for item in records
        if item["group_hash"] == "g0"
        and item["checkpoint"] == "2024"
        and item["variant"] == "full"
    )
    row["n"] += 1
    with pytest.raises(ValueError, match="identical N"):
        crossed_cluster_checkpoint_bootstrap(
            accumulate_error_cells(records),
            baseline="apn",
            comparators=["full"],
            resamples=10,
        )


def test_crossed_bootstrap_rejects_partially_empty_group() -> None:
    records = _constant_effect_records()
    for row in records:
        if row["group_hash"] == "g0" and row["checkpoint"] == "2024":
            row.update({"sse": 0.0, "sae": 0.0, "n": 0})
    with pytest.raises(ValueError, match="globally empty or positive"):
        crossed_cluster_checkpoint_bootstrap(
            accumulate_error_cells(records),
            baseline="apn",
            comparators=["full"],
            resamples=10,
        )


def test_crossed_bootstrap_rejects_incomplete_pairing() -> None:
    records = _constant_effect_records()
    records.pop()
    cells = accumulate_error_cells(records)
    with pytest.raises(ValueError, match="Incomplete paired"):
        crossed_cluster_checkpoint_bootstrap(
            cells,
            baseline="apn",
            comparators=["full"],
            resamples=10,
        )


def test_holm_and_safety_semantics() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert (
        classify_safety(gain_ci_low=-0.02, gain_ci_high=-0.015, harm_margin=0.01)
        == "harmful"
    )
    assert (
        classify_safety(gain_ci_low=-0.02, gain_ci_high=0.005, harm_margin=0.01)
        == "safety-inconclusive"
    )
    assert (
        classify_safety(gain_ci_low=-0.005, gain_ci_high=0.01, harm_margin=0.01)
        == "not-harmful"
    )


class _TickClock:
    def __init__(self, step: float = 0.001) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def test_phase_timer_synchronizes_cuda_and_labels_cpu_truthfully() -> None:
    clock = _TickClock()
    sync_calls: list[int] = []
    timer = PhaseTimer(clock=clock, synchronize=lambda: sync_calls.append(1))
    with timer.phase("slrh_solve", device="cpu"):
        pass
    assert sync_calls == []
    with timer.phase("feature_extraction", device="cuda:0"):
        pass
    assert len(sync_calls) == 2
    assert timer.records[0].device == "cpu"
    assert timer.records[1].device == "cuda:0"
    assert all(record.wall_seconds > 0 for record in timer.records)


def test_warm_inference_and_timing_schema() -> None:
    clock = _TickClock()
    calls: list[int] = []
    result = warm_inference(
        lambda: calls.append(1),
        device="cpu",
        warmup=2,
        repetitions=4,
        clock=clock,
    )
    assert len(calls) == 6
    assert result["warmup"] == 2
    assert result["repetitions"] == 4
    assert result["p95_ms"] > 0

    required = {
        "apn_load",
        "feature_extraction",
        "cache_read",
        "cache_write",
        "slrh_solve",
        "cfg_solve",
        "validation_selection",
        "serialization",
        "warm_inference",
    }
    records = [
        {"phase": phase, "device": "cpu", "wall_seconds": 0.1}
        for phase in sorted(required)
    ]
    validate_timing_records(records)
    with pytest.raises(ValueError, match="Missing timing phases"):
        validate_timing_records(records[:-1])


def test_serialized_state_bytes_is_finite_positive() -> None:
    size = serialized_state_bytes({"weight": torch.ones(3, 4)})
    assert isinstance(size, int)
    assert size > 0
    assert math.isfinite(float(size))
