from __future__ import annotations

import torch

from edgetwincal.robust import (
    apply_robust_joint,
    fit_group_huber_block_ridge,
    fit_robust_joint_adapter,
    group_balanced_weights,
    weighted_median_mad,
    weighted_quantile,
)


def _cell(
    x: torch.Tensor,
    y: torch.Tensor,
    groups: torch.Tensor,
    *,
    squared: bool = False,
    max_iterations: int = 25,
):
    return fit_group_huber_block_ridge(
        x,
        y,
        groups,
        latent_width=x.shape[1],
        alpha_latent=1.0,
        alpha_cross=1.0,
        minimum_rows=1,
        minimum_groups=2,
        squared_loss=squared,
        max_iterations=max_iterations,
    )


def test_group_weights_quantiles_and_group_duplication_invariance() -> None:
    groups = torch.tensor([0, 0, 1, 1, 1, 1])
    weights = group_balanced_weights(groups)
    assert weights.dtype == torch.float64
    assert torch.allclose(
        weights[:2].sum(), torch.tensor(3.0, dtype=torch.float64)
    )
    assert torch.allclose(
        weights[2:].sum(), torch.tensor(3.0, dtype=torch.float64)
    )
    values = torch.tensor([0.0, 1.0, 2.0, 100.0])
    qweights = torch.tensor([1.0, 1.0, 1.0, 0.01])
    assert float(weighted_quantile(values, qweights, 0.5)) == 1.0
    median, scale = weighted_median_mad(values[:3], torch.ones(3))
    assert float(median) == 1.0
    assert float(scale) > 0.0

    x = torch.tensor([[-2.0], [-1.0], [0.5], [1.5], [2.5], [3.5]])
    y = 0.75 * x[:, 0] + 1.25
    base_groups = torch.tensor([0, 0, 1, 1, 2, 2])
    original = _cell(x, y, base_groups, squared=True)
    duplicate = torch.tensor([0, 1, 0, 1, 0, 1, 2, 3, 4, 5])
    repeated = _cell(
        x[duplicate], y[duplicate], base_groups[duplicate], squared=True
    )
    probe = torch.linspace(-3, 4, 31).unsqueeze(1)
    assert original.active and repeated.active
    assert torch.allclose(
        original.predict_correction(probe),
        repeated.predict_correction(probe),
        atol=1e-11,
        rtol=1e-11,
    )


def test_huber_is_more_robust_than_squared_block_ridge() -> None:
    generator = torch.Generator().manual_seed(20260721)
    groups = torch.arange(30).repeat_interleave(10)
    x = torch.randn(300, 1, generator=generator)
    clean = 0.5 + 2.0 * x[:, 0]
    corrupted = clean.clone()
    corrupted[groups == 0] += 120.0
    robust = _cell(x, corrupted, groups)
    squared = _cell(x, corrupted, groups, squared=True)
    assert robust.active and squared.active
    robust_mse = (
        robust.predict_correction(x) - clean.double()
    ).square().mean()
    squared_mse = (
        squared.predict_correction(x) - clean.double()
    ).square().mean()
    assert robust_mse < squared_mse * 0.2
    assert robust.audit.downweighted_fraction > 0.0


def test_fit_is_reorder_invariant() -> None:
    generator = torch.Generator().manual_seed(17)
    groups = torch.arange(24).repeat_interleave(5)
    x = torch.randn(120, 3, generator=generator)
    y = 0.3 + x @ torch.tensor([0.5, -0.75, 1.25])
    permutation = torch.randperm(120, generator=generator)
    left = _cell(x, y, groups)
    right = _cell(x[permutation], y[permutation], groups[permutation])
    probe = torch.randn(19, 3, generator=generator)
    assert left.active and right.active
    assert torch.allclose(
        left.predict_correction(probe),
        right.predict_correction(probe),
        atol=1e-9,
        rtol=1e-9,
    )


def test_constant_empty_insufficient_nonfinite_and_nonconverged_cells_fallback() -> None:
    groups = torch.arange(20).repeat_interleave(6)
    constant = _cell(
        torch.ones(120, 2), torch.full((120,), 3.0), groups
    )
    assert constant.active
    assert torch.allclose(
        constant.predict_correction(torch.ones(4, 2)),
        torch.full((4,), 3.0, dtype=torch.float64),
        atol=1e-10,
    )

    empty = fit_group_huber_block_ridge(
        torch.empty(0, 2),
        torch.empty(0),
        torch.empty(0, dtype=torch.int64),
        latent_width=1,
        alpha_latent=1.0,
        alpha_cross=1.0,
    )
    assert not empty.active and empty.audit.reason == "empty"

    insufficient = fit_group_huber_block_ridge(
        torch.randn(30, 2),
        torch.randn(30),
        torch.zeros(30, dtype=torch.int64),
        latent_width=1,
        alpha_latent=1.0,
        alpha_cross=1.0,
        minimum_rows=1,
        minimum_groups=2,
    )
    assert (
        not insufficient.active
        and insufficient.audit.reason == "insufficient_groups"
    )

    bad_x = torch.randn(120, 2)
    bad_x[0, 0] = float("nan")
    nonfinite = _cell(bad_x, torch.randn(120), groups)
    assert not nonfinite.active and nonfinite.audit.reason == "nonfinite_input"

    generator = torch.Generator().manual_seed(9)
    x = torch.randn(120, 2, generator=generator)
    y = x[:, 0] + torch.where(groups == 0, 100.0, 0.0)
    stopped = _cell(x, y, groups, max_iterations=1)
    assert not stopped.active and stopped.audit.reason == "nonconvergence"
    assert torch.count_nonzero(stopped.weights) == 0


def test_joint_adapter_has_finite_shapes_zero_diagonal_and_no_cap() -> None:
    generator = torch.Generator().manual_seed(44)
    rows, horizon, channels, latent_width = 120, 2, 3, 2
    groups = torch.arange(30).repeat_interleave(4)
    latent = torch.randn(
        rows, channels, latent_width, generator=generator, dtype=torch.float64
    )
    base = torch.randn(
        rows, horizon, channels, generator=generator, dtype=torch.float64
    )
    target = base.clone()
    for h in range(horizon):
        for channel in range(channels):
            other = (channel + 1) % channels
            target[:, h, channel] += (
                0.4 * latent[:, channel, 0] + 0.25 * base[:, h, other]
            )
    mask = torch.ones_like(base, dtype=torch.bool)
    mask[:, 1, 2] = False
    adapter = fit_robust_joint_adapter(
        latent,
        base,
        target,
        mask,
        groups,
        alpha_latent=1.0,
        alpha_cross=1.0,
        minimum_rows=100,
        minimum_groups=20,
        squared_loss=True,
    )
    prediction = apply_robust_joint(adapter, latent, base)
    assert prediction.shape == base.shape
    assert prediction.dtype == base.dtype
    assert torch.isfinite(prediction).all()
    assert adapter.audit.cells == horizon * channels
    assert adapter.audit.zero_cells == 1
    assert adapter.cells[1][2].audit.reason == "empty"
    assert torch.equal(prediction[:, 1, 2], base[:, 1, 2])

    state = adapter.state_dict()
    for name in (
        "weights",
        "feature_center",
        "feature_scale",
        "intercept",
        "residual_scale",
    ):
        assert torch.isfinite(state[name]).all()
    assert "correction_cap" not in state

    correction = prediction - base
    changed = base.clone()
    changed[:, 0, 0] += 10_000.0
    changed_prediction = adapter.apply(latent, changed)
    changed_correction = changed_prediction - changed
    assert torch.allclose(
        correction[:, 0, 0], changed_correction[:, 0, 0]
    )
    assert not adapter.audit.observation_weighted
    assert adapter.audit.squared_loss


def test_observation_weight_switch_and_apply_rejects_nonfinite() -> None:
    groups = torch.arange(20).repeat_interleave(6)
    latent = torch.randn(120, 2, 1)
    base = torch.randn(120, 1, 2)
    target = base + latent[:, :, :1].permute(0, 2, 1)
    adapter = fit_robust_joint_adapter(
        latent,
        base,
        target,
        torch.ones_like(base, dtype=torch.bool),
        groups,
        alpha_latent=1.0,
        alpha_cross=1.0,
        minimum_rows=1,
        minimum_groups=2,
        observation_weighted=True,
        squared_loss=True,
    )
    assert adapter.audit.observation_weighted
    assert all(
        cell.audit.penalty_scale == 1.0 for cell in adapter.iter_cells()
    )
    bad = base.clone()
    bad[0, 0, 0] = float("inf")
    try:
        adapter.apply(latent, bad)
    except ValueError as exc:
        assert "NaN or infinity" in str(exc)
    else:
        raise AssertionError("non-finite apply input was accepted")
