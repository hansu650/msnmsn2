from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .ridge import fit_ridge, masked_micro_mse, require_finite, validate_alphas


def _validate_forecast_tensors(
    base: Tensor, target: Tensor | None = None, mask: Tensor | None = None
) -> None:
    if base.ndim != 3:
        raise ValueError("forecast tensors must have shape [B,H,C]")
    require_finite("base forecast", base)
    if target is not None and target.shape != base.shape:
        raise ValueError("target shape mismatch")
    if mask is not None and mask.shape != base.shape:
        raise ValueError("mask shape mismatch")


@dataclass(frozen=True)
class ResidualBias:
    bias: Tensor
    observations: Tensor

    def apply(self, base: Tensor) -> Tensor:
        _validate_forecast_tensors(base)
        if tuple(self.bias.shape) != tuple(base.shape[1:]):
            raise ValueError("Bias state does not match forecast shape")
        return (base.double() + self.bias).to(base.dtype)

    def state_dict(self) -> dict[str, Tensor]:
        return {"bias": self.bias, "observations": self.observations}


def fit_bias_only(base: Tensor, target: Tensor, mask: Tensor) -> ResidualBias:
    _validate_forecast_tensors(base, target, mask)
    base64, target64 = base.double(), target.double()
    observed_mask = mask > 0
    _, horizon, channels = base.shape
    bias = torch.zeros(horizon, channels, dtype=torch.float64)
    counts = torch.zeros(horizon, channels, dtype=torch.int64)
    for h in range(horizon):
        for channel in range(channels):
            observed = observed_mask[:, h, channel]
            count = int(observed.sum())
            counts[h, channel] = count
            if count:
                bias[h, channel] = (
                    target64[observed, h, channel] - base64[observed, h, channel]
                ).mean()
    require_finite("bias", bias)
    return ResidualBias(bias=bias, observations=counts)


@dataclass(frozen=True)
class SelfAffineResidual:
    alpha: float
    weight: Tensor
    source_mean: Tensor
    source_scale: Tensor
    intercept: Tensor
    observations: Tensor

    def apply(self, base: Tensor) -> Tensor:
        _validate_forecast_tensors(base)
        if tuple(self.weight.shape) != tuple(base.shape[1:]):
            raise ValueError("Self-affine state does not match forecast shape")
        standardized = (base.double() - self.source_mean) / self.source_scale
        correction = standardized * self.weight + self.intercept
        output = base.double() + correction
        require_finite("self-affine output", output)
        return output.to(base.dtype)

    def state_dict(self) -> dict[str, Tensor | float]:
        return {
            "alpha": self.alpha,
            "weight": self.weight,
            "source_mean": self.source_mean,
            "source_scale": self.source_scale,
            "intercept": self.intercept,
            "observations": self.observations,
        }


def _fit_self_affine(
    base: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    alpha: float,
) -> SelfAffineResidual:
    _validate_forecast_tensors(base, target, mask)
    base64, target64 = base.double(), target.double()
    observed_mask = mask > 0
    _, horizon, channels = base.shape
    weight = torch.zeros(horizon, channels, dtype=torch.float64)
    mean = torch.zeros_like(weight)
    scale = torch.ones_like(weight)
    intercept = torch.zeros_like(weight)
    counts = torch.zeros(horizon, channels, dtype=torch.int64)
    for h in range(horizon):
        for channel in range(channels):
            observed = observed_mask[:, h, channel]
            design = base64[observed, h, channel].unsqueeze(-1)
            residual = (
                target64[observed, h, channel] - base64[observed, h, channel]
            )
            solution = fit_ridge(design, residual, alpha=alpha)
            weight[h, channel] = solution.weights[0]
            mean[h, channel] = solution.feature_mean[0]
            scale[h, channel] = solution.feature_scale[0]
            intercept[h, channel] = solution.intercept
            counts[h, channel] = solution.observations
    return SelfAffineResidual(
        float(alpha), weight, mean, scale, intercept, counts
    )


def fit_self_affine_with_validation(
    train_base: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_base: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    alphas: Iterable[float],
) -> tuple[SelfAffineResidual, list[dict[str, float]]]:
    candidates = validate_alphas(alphas)
    best: SelfAffineResidual | None = None
    best_mse = float("inf")
    audit: list[dict[str, float]] = []
    for alpha in candidates:
        state = _fit_self_affine(
            train_base, train_target, train_mask, alpha=alpha
        )
        mse = masked_micro_mse(state.apply(val_base), val_target, val_mask)
        audit.append({"alpha": alpha, "val_mse": mse})
        if mse < best_mse:
            best, best_mse = state, mse
    if best is None:
        raise RuntimeError("No self-affine candidate was fitted")
    return best, audit
