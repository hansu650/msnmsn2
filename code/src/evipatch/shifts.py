"""Deterministic, exact-count observation shifts for Stage A evaluation."""

from __future__ import annotations

import math

import torch
from torch import Tensor


_SHIFT_MODULUS = 2_147_483_647


def stable_uniform(
    sample_ids: Tensor,
    channel: int,
    positions: Tensor,
    seed: int,
) -> Tensor:
    """Hash identifiers and positions into deterministic values in [0, 1).

    The result is independent of Python hash randomization, process lifetime, and
    batch order. The returned shape is sample_ids.numel() by positions.numel().
    """
    if channel < 0:
        raise ValueError("channel must be non-negative")
    sample_terms = sample_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1, 1)
    position_terms = positions.detach().to(device="cpu", dtype=torch.int64).reshape(1, -1)
    hashed = (
        sample_terms * 1_000_003
        + (int(channel) + 1) * 9_176
        + (position_terms + 1) * 1_013_904_223
        + (int(seed) + 1) * 69_069
    ) % _SHIFT_MODULUS
    hashed = (hashed * 48_271 + 12_345) % _SHIFT_MODULUS
    hashed = (hashed * 40_692 + 1) % _SHIFT_MODULUS
    return hashed.to(torch.float64) / float(_SHIFT_MODULUS)


def _validate_mask_inputs(mask: Tensor, sample_ids: Tensor, rate: float) -> None:
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape [B, L, N], got {tuple(mask.shape)}")
    if sample_ids.numel() != mask.shape[0]:
        raise ValueError(
            f"sample_ids has {sample_ids.numel()} entries for batch size {mask.shape[0]}"
        )
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")


def _requested_counts(mask: Tensor, rate: float) -> Tensor:
    observed = (mask > 0).sum(dim=1, dtype=torch.int64)
    return torch.floor(observed.to(torch.float64) * float(rate)).to(torch.int64)


def exact_mcar_mask(
    mask: Tensor,
    sample_ids: Tensor,
    rate: float,
    seed: int,
) -> Tensor:
    """Remove floor(rate * observed) positions in every sample-channel row."""
    _validate_mask_inputs(mask, sample_ids, rate)
    result = (mask > 0).clone()
    requested = _requested_counts(mask, rate)
    batch_size, _, n_channels = mask.shape

    for batch_index in range(batch_size):
        sample_id = sample_ids.reshape(-1)[batch_index : batch_index + 1]
        for channel in range(n_channels):
            count = int(requested[batch_index, channel].item())
            if count == 0:
                continue
            observed_positions = torch.nonzero(
                result[batch_index, :, channel], as_tuple=False
            ).flatten()
            scores = stable_uniform(sample_id, channel, observed_positions, seed).flatten()
            selected = observed_positions[torch.argsort(scores, stable=True)[:count]]
            result[batch_index, selected, channel] = False

    return result.to(dtype=mask.dtype, device=mask.device)


def _times_for_sample(times: Tensor, batch_index: int, channel: int) -> Tensor:
    if times.ndim == 2:
        return times[batch_index]
    if times.ndim != 3:
        raise ValueError(
            f"times must have shape [B, L], [B, L, 1], or [B, L, N], got {tuple(times.shape)}"
        )
    if times.shape[-1] == 1:
        return times[batch_index, :, 0]
    return times[batch_index, :, channel]


def exact_burst_mask(
    mask: Tensor,
    times: Tensor,
    sample_ids: Tensor,
    rate: float,
    seed: int,
) -> Tensor:
    """Remove an exact-count circular contiguous run in chronological order."""
    _validate_mask_inputs(mask, sample_ids, rate)
    if times.shape[0] != mask.shape[0] or times.shape[1] != mask.shape[1]:
        raise ValueError(
            f"times shape {tuple(times.shape)} is incompatible with mask {tuple(mask.shape)}"
        )
    if times.ndim == 3 and times.shape[-1] not in (1, mask.shape[-1]):
        raise ValueError(
            f"times last dimension must be 1 or {mask.shape[-1]}, got {times.shape[-1]}"
        )

    result = (mask > 0).clone()
    requested = _requested_counts(mask, rate)
    batch_size, _, n_channels = mask.shape

    for batch_index in range(batch_size):
        sample_id = sample_ids.reshape(-1)[batch_index : batch_index + 1]
        for channel in range(n_channels):
            count = int(requested[batch_index, channel].item())
            if count == 0:
                continue
            observed_positions = torch.nonzero(
                result[batch_index, :, channel], as_tuple=False
            ).flatten()
            row_times = _times_for_sample(times, batch_index, channel)
            order = torch.argsort(row_times[observed_positions], stable=True)
            chronological_positions = observed_positions[order]
            n_observed = int(chronological_positions.numel())
            start_score = stable_uniform(
                sample_id,
                channel,
                torch.tensor([n_observed], dtype=torch.int64),
                seed,
            ).item()
            start = min(int(math.floor(start_score * n_observed)), n_observed - 1)
            circular_ranks = (
                torch.arange(count, dtype=torch.int64) + start
            ) % n_observed
            selected = chronological_positions[circular_ranks]
            result[batch_index, selected, channel] = False

    return result.to(dtype=mask.dtype, device=mask.device)


def apply_observation_shift(
    x: Tensor,
    x_mask: Tensor,
    x_mark: Tensor,
    sample_ids: Tensor,
    mode: str,
    rate: float,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Apply a test-history-only shift and return a per-row deletion audit."""
    if x.shape != x_mask.shape:
        raise ValueError(f"x and x_mask must match, got {tuple(x.shape)} and {tuple(x_mask.shape)}")
    _validate_mask_inputs(x_mask, sample_ids, rate)
    if mode == "none":
        shifted_mask = x_mask.clone()
    elif mode == "mcar":
        shifted_mask = exact_mcar_mask(x_mask, sample_ids, rate, seed)
    elif mode == "burst":
        shifted_mask = exact_burst_mask(x_mask, x_mark, sample_ids, rate, seed)
    else:
        raise ValueError(f"Unknown observation shift {mode!r}; expected none, mcar, or burst")

    original_observed = (x_mask > 0).sum(dim=1, dtype=torch.int64)
    remaining_observed = (shifted_mask > 0).sum(dim=1, dtype=torch.int64)
    actual = original_observed - remaining_observed
    requested = (
        torch.zeros_like(actual)
        if mode == "none"
        else _requested_counts(x_mask, rate).to(device=actual.device)
    )
    if not torch.equal(actual.cpu(), requested.cpu()):
        raise RuntimeError(
            f"shift {mode} deleted counts that differ from the exact request"
        )

    shifted_x = torch.where(shifted_mask > 0, x, torch.zeros_like(x))
    audit = {
        "shift_requested": requested,
        "shift_actual": actual,
        "shift_original_observed": original_observed,
        "shift_remaining_observed": remaining_observed,
    }
    return shifted_x, shifted_mask, audit
