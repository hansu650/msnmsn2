from __future__ import annotations

import torch

from edgetwincal.controls import (
    fit_bias_only,
    fit_self_affine_with_validation,
)
from edgetwincal.graph import fit_graph_with_validation
from edgetwincal.joint import JointResidualRidge, fit_joint_with_validation
from edgetwincal.ridge import fit_ridge
from edgetwincal.shuffle import (
    shuffle_cross_forecasts,
    shuffle_latent_features,
)
from edgetwincal.variants import (
    fit_diagonal_from_frozen_latent,
    fit_full_with_validation,
    fit_reverse_with_validation,
)


ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def test_ridge_intercept_is_unpenalized_and_constant_safe() -> None:
    features = torch.zeros(20, 1)
    residual = torch.full((20,), 2.75)
    state = fit_ridge(features, residual, alpha=100000.0)
    assert torch.allclose(state.intercept, torch.tensor(2.75, dtype=torch.float64))
    assert torch.count_nonzero(state.weights) == 0
    assert torch.isfinite(state.feature_scale).all()
    assert torch.allclose(
        state.predict_correction(features),
        torch.full((20,), 2.75, dtype=torch.float64),
    )


def test_cfg_anchor_and_frozen_source_are_decoupled() -> None:
    torch.manual_seed(101)
    train_source = torch.randn(600, 1, 4)
    val_source = torch.randn(200, 1, 4)
    train_anchor = torch.randn(600, 1, 4) * 0.1
    val_anchor = torch.randn(200, 1, 4) * 0.1

    def make_target(anchor: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return anchor + 0.7 * torch.roll(source, shifts=1, dims=2)

    train_target = make_target(train_anchor, train_source)
    val_target = make_target(val_anchor, val_source)
    train_mask = torch.ones_like(train_target)
    val_mask = torch.ones_like(val_target)
    graph, audit = fit_graph_with_validation(
        train_anchor,
        train_target,
        train_mask,
        val_anchor,
        val_target,
        val_mask,
        alphas=ALPHAS,
        train_source_forecasts=train_source,
        val_source_forecasts=val_source,
    )
    correct = (graph.apply(val_anchor, val_source) - val_target).square().mean()
    wrong_source = (graph.apply(val_anchor, val_anchor) - val_target).square().mean()
    assert correct < 1e-4
    assert wrong_source > correct * 100
    assert len(audit) == 6
    assert torch.count_nonzero(torch.diagonal(graph.weights, dim1=1, dim2=2)) == 0


def _synthetic_dual_space() -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    torch.manual_seed(111)
    train_base = torch.randn(700, 2, 4)
    val_base = torch.randn(220, 2, 4)
    train_latent = torch.randn(700, 4, 3)
    val_latent = torch.randn(220, 4, 3)
    local_coefficient = torch.tensor([0.25, -0.15, 0.1])

    def target(base: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        local = torch.einsum("bcd,d->bc", latent, local_coefficient).unsqueeze(1)
        return base + local + 0.35 * torch.roll(base, shifts=1, dims=2)

    train_target = target(train_base, train_latent)
    val_target = target(val_base, val_latent)
    return (
        train_base,
        train_latent,
        train_target,
        torch.ones_like(train_target),
    ), (
        val_base,
        val_latent,
        val_target,
        torch.ones_like(val_target),
    )


def test_full_and_reverse_are_explicit_distinct_orders() -> None:
    train, val = _synthetic_dual_space()
    full, full_audit = fit_full_with_validation(
        *train,
        *val,
        latent_alphas=[1.0, 10.0],
        graph_alphas=[1.0, 10.0],
    )
    reverse, reverse_audit = fit_reverse_with_validation(
        *train,
        *val,
        latent_alphas=[1.0, 10.0],
        graph_alphas=[1.0, 10.0],
    )
    full_error = (full.apply(val[0], val[1]) - val[2]).square().mean()
    reverse_error = (reverse.apply(val[0], val[1]) - val[2]).square().mean()
    baseline_error = (val[0] - val[2]).square().mean()
    assert full.order == "slrh_then_cfg"
    assert reverse.order == "cfg_then_slrh"
    assert full_error < baseline_error * 0.01
    assert reverse_error < baseline_error * 0.01
    assert set(full_audit) == {"slrh", "cfg"}
    assert set(reverse_audit) == {"slrh", "cfg"}


def test_diagonal_control_reuses_exact_latent_state() -> None:
    train, val = _synthetic_dual_space()
    full, _ = fit_full_with_validation(
        *train,
        *val,
        latent_alphas=[1.0],
        graph_alphas=[1.0],
    )
    diagonal, audit = fit_diagonal_from_frozen_latent(
        full.latent,
        *train,
        *val,
        graph_alphas=[1.0, 10.0],
    )
    assert diagonal.latent is full.latent
    assert diagonal.graph.include_self is True
    assert len(audit) == 2


def test_bias_and_self_affine_controls_use_training_only() -> None:
    torch.manual_seed(121)
    train_base = torch.randn(500, 2, 3)
    val_base = torch.randn(180, 2, 3)
    train_target = train_base + 1.5 + 0.2 * train_base
    val_target = val_base + 1.5 + 0.2 * val_base
    train_mask = torch.ones_like(train_base)
    val_mask = torch.ones_like(val_base)
    bias = fit_bias_only(train_base, train_target, train_mask)
    affine, audit = fit_self_affine_with_validation(
        train_base,
        train_target,
        train_mask,
        val_base,
        val_target,
        val_mask,
        alphas=ALPHAS,
    )
    bias_error = (bias.apply(val_base) - val_target).square().mean()
    affine_error = (affine.apply(val_base) - val_target).square().mean()
    assert affine_error < bias_error * 0.01
    assert len(audit) == 6


def test_shuffles_are_deterministic_semantic_and_batch_order_invariant() -> None:
    rows, horizon, channels, width = 17, 3, 4, 5
    source = torch.arange(rows * horizon * channels, dtype=torch.float32).reshape(
        rows, horizon, channels
    )
    latent = torch.arange(rows * channels * width, dtype=torch.float32).reshape(
        rows, channels, width
    )
    row_ids = [f"sample-{index}" for index in range(rows)]
    cross_a, audit_a, indices_a = shuffle_cross_forecasts(
        source, row_ids, descriptor="p12|2024|train"
    )
    cross_b, audit_b, indices_b = shuffle_cross_forecasts(
        source, row_ids, descriptor="p12|2024|train"
    )
    latent_a, _, latent_indices = shuffle_latent_features(
        latent, row_ids, descriptor="p12|2024|train"
    )
    assert torch.equal(cross_a, cross_b)
    assert torch.equal(indices_a, indices_b)
    assert audit_a.indices_sha256 == audit_b.indices_sha256
    assert torch.equal(source, torch.arange(source.numel()).reshape_as(source))
    for channel in range(channels):
        permutation = indices_a[channel]
        assert torch.equal(cross_a[:, :, channel], source[permutation, :, channel])
        assert torch.equal(
            latent_a[:, channel], latent[latent_indices[channel], channel]
        )

    reorder = torch.randperm(rows)
    reordered, _, _ = shuffle_cross_forecasts(
        source[reorder],
        [row_ids[index] for index in reorder],
        descriptor="p12|2024|train",
    )
    restored = torch.empty_like(reordered)
    restored[reorder] = reordered
    assert torch.equal(restored, cross_a)


def test_joint_uses_complete_grid_and_p12_matched_slots() -> None:
    train, val = _synthetic_dual_space()
    state, audit = fit_joint_with_validation(
        train[0],
        train[1],
        train[0],
        train[2],
        train[3],
        val[0],
        val[1],
        val[0],
        val[2],
        val[3],
        latent_alphas=ALPHAS,
        graph_alphas=ALPHAS,
    )
    error = (state.apply(val[0], val[1], val[0]) - val[2]).square().mean()
    baseline = (val[0] - val[2]).square().mean()
    assert error < baseline * 0.01
    assert len(audit) == 36
    p12_state = JointResidualRidge(
        1.0,
        1.0,
        torch.zeros(3, 36, 24),
        torch.zeros(3, 36, 36),
        torch.zeros(3, 36, 24),
        torch.ones(3, 36, 24),
        torch.zeros(3, 36, 36),
        torch.ones(3, 36, 36),
        torch.zeros(3, 36),
        torch.ones(3, 36, dtype=torch.int64),
        False,
    )
    assert p12_state.coefficient_slots == 6480


def test_no_observation_cells_remain_finite() -> None:
    base = torch.zeros(8, 2, 3)
    target = torch.ones_like(base)
    mask = torch.zeros_like(base)
    mask[:, 0, 0] = 1
    bias = fit_bias_only(base, target, mask)
    affine, _ = fit_self_affine_with_validation(
        base,
        target,
        mask,
        base,
        target,
        torch.ones_like(mask),
        alphas=[1.0],
    )
    assert torch.isfinite(bias.apply(base)).all()
    assert torch.isfinite(affine.apply(base)).all()
