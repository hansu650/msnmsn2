from __future__ import annotations

import pytest
import torch

from evipatch.evidence import (
    EVIDENCE_WIDTHS,
    apply_evidence_control,
    compute_evidence,
    evidence_width,
    fixed_random_features,
)


def _case(mask_values: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    bn, length = mask_values.shape
    patches = 3
    weights_raw = torch.full(
        (bn, patches, length), 0.5, dtype=torch.float64, requires_grad=True
    )
    mask = mask_values[:, None, :].to(torch.float64)
    temporal = weights_raw * mask
    times = torch.linspace(0.0, 1.0, length, dtype=torch.float64).view(1, 1, length)
    times = times.expand(bn, -1, -1)
    left = torch.zeros(bn, patches, 1, dtype=torch.float64)
    right = torch.ones(bn, patches, 1, dtype=torch.float64)
    return compute_evidence(temporal, weights_raw, mask, times, left, right), weights_raw


def test_evidence_widths_are_exactly_predeclared() -> None:
    assert EVIDENCE_WIDTHS == {
        "apn": 0,
        "global_ratio": 1,
        "raw_count": 1,
        "soft_mass": 1,
        "evipatch_full": 3,
        "shuffled_evidence": 3,
        "random_features": 3,
    }
    with pytest.raises(ValueError):
        evidence_width("unregistered")


@pytest.mark.parametrize(
    "mask_values",
    [
        torch.zeros(2, 5),
        torch.tensor([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
        torch.ones(2, 5),
    ],
    ids=["empty", "single-point", "dense"],
)
def test_shape_finite_values_and_gradients(mask_values: torch.Tensor) -> None:
    stats, weights_raw = _case(mask_values)
    for key in (
        "soft_mass",
        "effective_support",
        "coverage",
        "raw_count",
        "global_ratio",
    ):
        assert stats[key].shape == (2, 3, 1)
        assert torch.isfinite(stats[key]).all()

    full = apply_evidence_control(stats, "evipatch_full", n_variables=1)
    assert full is not None and full.shape == (2, 3, 3)
    full.sum().backward()
    assert weights_raw.grad is not None
    assert torch.isfinite(weights_raw.grad).all()


def test_empty_and_single_point_coverage_are_zero() -> None:
    empty, _ = _case(torch.zeros(1, 5))
    single, _ = _case(torch.tensor([[0, 0, 1, 0, 0]]))
    assert torch.equal(empty["coverage"], torch.zeros_like(empty["coverage"]))
    assert torch.equal(single["coverage"], torch.zeros_like(single["coverage"]))


def test_scalar_controls_remain_informative_without_layer_norm() -> None:
    one, _ = _case(torch.tensor([[1, 0, 0, 0, 0]]))
    two, _ = _case(torch.tensor([[1, 1, 0, 0, 0]]))
    assert torch.all(two["raw_count"] > one["raw_count"])
    assert torch.all(two["soft_mass"] > one["soft_mass"])


def test_exact_centroid_collision_has_different_evidence() -> None:
    feature = torch.tensor([[[2.0, -1.0]]], dtype=torch.float32)
    duplicated = feature.repeat(1, 2, 1)
    weight_one = torch.tensor([[[0.5]]], dtype=torch.float32)
    weight_two = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    upstream_one = torch.bmm(weight_one, feature) / (
        weight_one.sum(dim=-1, keepdim=True) + 1e-9
    )
    upstream_two = torch.bmm(weight_two, duplicated) / (
        weight_two.sum(dim=-1, keepdim=True) + 1e-9
    )
    torch.testing.assert_close(upstream_one, upstream_two, atol=0, rtol=0)

    time_one = torch.tensor([[[0.5]]])
    time_two = torch.tensor([[[0.5, 0.5]]])
    bounds = torch.tensor([[[0.0]]])
    right = torch.tensor([[[1.0]]])
    stats_one = compute_evidence(
        weight_one, weight_one, torch.ones(1, 1, 1), time_one, bounds, right
    )
    stats_two = compute_evidence(
        weight_two, weight_two, torch.ones(1, 1, 2), time_two, bounds, right
    )
    assert not torch.equal(stats_one["soft_mass"], stats_two["soft_mass"])
    assert not torch.equal(
        stats_one["effective_support"], stats_two["effective_support"]
    )


def test_shuffled_preserves_marginals_and_random_is_fixed() -> None:
    stats, _ = _case(torch.ones(4, 5))
    full = apply_evidence_control(stats, "evipatch_full", n_variables=2)
    shuffled = apply_evidence_control(stats, "shuffled_evidence", n_variables=2)
    assert full is not None and shuffled is not None
    torch.testing.assert_close(
        torch.sort(full.flatten()).values,
        torch.sort(shuffled.flatten()).values,
    )

    table_a = fixed_random_features(2, 3, seed=77)
    table_b = fixed_random_features(2, 3, seed=77)
    torch.testing.assert_close(table_a, table_b)
    random_features = apply_evidence_control(
        stats, "random_features", n_variables=2, random_table=table_a
    )
    assert random_features is not None
    assert random_features.shape == (4, 3, 3)
    assert not random_features.requires_grad
