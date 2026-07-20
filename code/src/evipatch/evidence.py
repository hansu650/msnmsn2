"""Evidence statistics and predeclared controls for EviPatch.

All statistics operate on the temporal weights already computed by APN/TAPA.
No feature is normalized with LayerNorm; scalar controls use monotone log1p
transforms so that one-dimensional ablations remain informative.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor


EVIDENCE_WIDTHS: Final[dict[str, int]] = {
    "apn": 0,
    "global_ratio": 1,
    "raw_count": 1,
    "soft_mass": 1,
    "evipatch_full": 3,
    "shuffled_evidence": 3,
    "random_features": 3,
}


def evidence_width(mode: str) -> int:
    """Return the evidence width for a predeclared Stage A variant."""
    try:
        return EVIDENCE_WIDTHS[mode]
    except KeyError as exc:
        allowed = ", ".join(EVIDENCE_WIDTHS)
        raise ValueError(f"Unknown EviPatch mode {mode!r}; expected one of: {allowed}") from exc


def _as_patch_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must have 3 dimensions, got {tuple(mask.shape)}")
    if mask.shape[0] != reference.shape[0] or mask.shape[-1] != reference.shape[-1]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is incompatible with weights "
            f"{tuple(reference.shape)}"
        )
    if mask.shape[1] not in (1, reference.shape[1]):
        raise ValueError(
            f"mask patch dimension must be 1 or {reference.shape[1]}, got {mask.shape[1]}"
        )
    return mask.to(dtype=reference.dtype).expand_as(reference)


def compute_evidence(
    temporal_weights: Tensor,
    weights_raw: Tensor,
    mask: Tensor,
    times: Tensor,
    left: Tensor,
    right: Tensor,
    global_ratio: Tensor | None = None,
    eps: float = 1e-9,
) -> dict[str, Tensor]:
    """Compute mass, effective support, coverage, count, and diagnostics.

    Expected shapes are BN x P x L for weights, BN x 1/P x L for mask,
    BN x 1 x L for times, and BN x P x 1 for window bounds.
    """
    if temporal_weights.ndim != 3:
        raise ValueError(
            "temporal_weights must have shape [B*N, P, L], "
            f"got {tuple(temporal_weights.shape)}"
        )
    if weights_raw.shape != temporal_weights.shape:
        raise ValueError(
            f"weights_raw {tuple(weights_raw.shape)} must match temporal_weights "
            f"{tuple(temporal_weights.shape)}"
        )
    bn, patches, length = temporal_weights.shape
    if times.shape != (bn, 1, length):
        raise ValueError(f"times must have shape {(bn, 1, length)}, got {tuple(times.shape)}")
    if left.shape != (bn, patches, 1) or right.shape != (bn, patches, 1):
        raise ValueError(
            f"bounds must have shape {(bn, patches, 1)}, got "
            f"left={tuple(left.shape)}, right={tuple(right.shape)}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    patch_mask = _as_patch_mask(mask, temporal_weights)
    mass_raw = temporal_weights.sum(dim=-1, keepdim=True)
    sum_sq_weights = temporal_weights.square().sum(dim=-1, keepdim=True)
    has_mass = mass_raw > 0

    effective_support_raw = mass_raw.square() / (sum_sq_weights + eps)
    effective_support_raw = torch.where(
        has_mass, effective_support_raw, torch.zeros_like(effective_support_raw)
    )

    safe_mass = torch.where(has_mass, mass_raw, torch.ones_like(mass_raw))
    weighted_time_mean = (
        temporal_weights * times
    ).sum(dim=-1, keepdim=True) / safe_mass
    centered_times = times - weighted_time_mean
    temporal_variance = (
        temporal_weights * centered_times.square()
    ).sum(dim=-1, keepdim=True) / safe_mass
    temporal_variance = temporal_variance.clamp_min(0)
    window_width = (right - left).clamp_min(eps)
    positive_variance = temporal_variance > 0
    coverage_raw = torch.where(
        has_mass & positive_variance,
        torch.sqrt(temporal_variance + eps) / window_width,
        torch.zeros_like(temporal_variance),
    )

    inside = (times >= left) & (times <= right)
    hard_count_raw = (inside.to(patch_mask.dtype) * patch_mask).sum(
        dim=-1, keepdim=True
    )

    if global_ratio is None:
        global_ratio_tensor = patch_mask[:, :1, :].mean(dim=-1, keepdim=True)
        global_ratio_tensor = global_ratio_tensor.expand(-1, patches, -1)
    else:
        global_ratio_tensor = global_ratio.to(
            device=temporal_weights.device, dtype=temporal_weights.dtype
        )
        if global_ratio_tensor.ndim == 2:
            global_ratio_tensor = global_ratio_tensor.unsqueeze(-1)
        if global_ratio_tensor.shape == (bn, 1, 1):
            global_ratio_tensor = global_ratio_tensor.expand(-1, patches, -1)
        if global_ratio_tensor.shape != (bn, patches, 1):
            raise ValueError(
                f"global_ratio must broadcast to {(bn, patches, 1)}, "
                f"got {tuple(global_ratio_tensor.shape)}"
            )

    return {
        "mass_raw": mass_raw,
        "soft_mass": torch.log1p(mass_raw),
        "sum_sq_weights": sum_sq_weights,
        "effective_support_raw": effective_support_raw,
        "effective_support": torch.log1p(effective_support_raw),
        "weighted_time_mean": weighted_time_mean,
        "temporal_variance": temporal_variance,
        "coverage_raw": coverage_raw,
        "coverage": torch.log1p(coverage_raw),
        "hard_count_raw": hard_count_raw,
        "raw_count": torch.log1p(hard_count_raw),
        "global_ratio": global_ratio_tensor,
        "window_width": window_width,
        "has_mass": has_mass,
        "weights_raw": weights_raw,
    }


def fixed_random_features(
    n_variables: int,
    n_patches: int,
    width: int = 3,
    seed: int = 1729,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create a fixed Gaussian variable-by-patch feature table."""
    if n_variables <= 0 or n_patches <= 0 or width <= 0:
        raise ValueError("random feature dimensions must all be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    table = torch.randn(
        n_variables, n_patches, width, generator=generator, dtype=dtype
    )
    return table.to(device=device) if device is not None else table


def apply_evidence_control(
    stats: dict[str, Tensor],
    mode: str,
    n_variables: int,
    random_table: Tensor | None = None,
) -> Tensor | None:
    """Select or construct the signature for one predeclared Stage A mode."""
    width = evidence_width(mode)
    if width == 0:
        return None
    if mode == "global_ratio":
        return stats["global_ratio"]
    if mode == "raw_count":
        return stats["raw_count"]
    if mode == "soft_mass":
        return stats["soft_mass"]

    full = torch.cat(
        [stats["soft_mass"], stats["effective_support"], stats["coverage"]], dim=-1
    )
    if mode == "evipatch_full":
        return full
    if mode == "shuffled_evidence":
        if full.shape[0] <= 1:
            raise ValueError("shuffled_evidence requires at least two sample-variable rows")
        return torch.roll(full, shifts=1, dims=0)
    if mode == "random_features":
        if random_table is None:
            raise ValueError("random_features mode requires a fixed random table")
        if random_table.ndim != 3 or random_table.shape[-1] != width:
            raise ValueError(
                f"random_table must have shape [N, P, {width}], "
                f"got {tuple(random_table.shape)}"
            )
        bn, patches, _ = full.shape
        if n_variables <= 0 or bn % n_variables != 0:
            raise ValueError(
                f"flattened row count {bn} is not divisible by n_variables={n_variables}"
            )
        if random_table.shape[:2] != (n_variables, patches):
            raise ValueError(
                f"random_table must have shape {(n_variables, patches, width)}, "
                f"got {tuple(random_table.shape)}"
            )
        batch_size = bn // n_variables
        return (
            random_table.to(device=full.device, dtype=full.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(bn, patches, width)
        )
    raise AssertionError(f"Unhandled mode after validation: {mode}")
