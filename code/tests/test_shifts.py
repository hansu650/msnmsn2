from __future__ import annotations

import torch

from evipatch.shifts import (
    apply_observation_shift,
    exact_burst_mask,
    exact_mcar_mask,
    stable_uniform,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(4 * 11 * 3, dtype=torch.float32).reshape(4, 11, 3)
    mask = torch.ones_like(x)
    mask[0, :2, 1] = 0
    mask[1, 8:, 2] = 0
    times = torch.linspace(0, 1, 11).view(1, 11, 1).expand(4, -1, -1)
    ids = torch.tensor([91, 7, 42, 1003], dtype=torch.float32)
    return x, mask, times, ids


def test_stable_uniform_is_repeatable() -> None:
    ids = torch.tensor([7, 91])
    positions = torch.arange(20)
    first = stable_uniform(ids, 3, positions, 2024)
    second = stable_uniform(ids, 3, positions, 2024)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.all((first >= 0) & (first < 1))


def test_mcar_is_exact_and_independent_of_batch_order() -> None:
    _, mask, _, ids = _inputs()
    shifted = exact_mcar_mask(mask, ids, 0.3, 2024)
    expected = torch.floor((mask > 0).sum(dim=1) * 0.3).to(torch.int64)
    actual = (mask > 0).sum(dim=1) - (shifted > 0).sum(dim=1)
    assert torch.equal(actual, expected)

    order = torch.tensor([3, 0, 2, 1])
    reordered = exact_mcar_mask(mask[order], ids[order], 0.3, 2024)
    torch.testing.assert_close(shifted[order], reordered, atol=0, rtol=0)


def test_burst_matches_mcar_counts_and_is_circular_contiguous() -> None:
    _, mask, times, ids = _inputs()
    mcar = exact_mcar_mask(mask, ids, 0.3, 2025)
    burst = exact_burst_mask(mask, times, ids, 0.3, 2025)
    mcar_deleted = (mask > 0).sum(dim=1) - (mcar > 0).sum(dim=1)
    burst_deleted = (mask > 0).sum(dim=1) - (burst > 0).sum(dim=1)
    assert torch.equal(mcar_deleted, burst_deleted)

    for batch_index in range(mask.shape[0]):
        for channel in range(mask.shape[-1]):
            observed = torch.nonzero(mask[batch_index, :, channel] > 0).flatten()
            deleted = torch.nonzero(
                (mask[batch_index, :, channel] > 0)
                & (burst[batch_index, :, channel] == 0)
            ).flatten()
            ranks = sorted(int(torch.where(observed == value)[0]) for value in deleted)
            if len(ranks) <= 1:
                continue
            gaps = [
                (ranks[(i + 1) % len(ranks)] - ranks[i]) % len(observed)
                for i in range(len(ranks))
            ]
            assert sum(gap != 1 for gap in gaps) <= 1


def test_apply_shift_never_modifies_targets_and_audits_counts() -> None:
    x, mask, times, ids = _inputs()
    target = torch.randn(4, 3, 3)
    target_before = target.clone()
    shifted_x, shifted_mask, audit = apply_observation_shift(
        x, mask, times, ids, "burst", 0.3, 2026
    )
    assert torch.equal(target, target_before)
    assert torch.equal(audit["shift_requested"], audit["shift_actual"])
    assert torch.equal(
        audit["shift_original_observed"] - audit["shift_remaining_observed"],
        audit["shift_actual"],
    )
    assert torch.equal(shifted_x[shifted_mask == 0], torch.zeros_like(shifted_x[shifted_mask == 0]))
