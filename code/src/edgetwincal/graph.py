from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .ridge import fit_ridge, masked_micro_mse, require_finite, validate_alphas


def _validate_inputs(
    base_to_correct: Tensor,
    source_forecasts: Tensor,
    target: Tensor | None = None,
    mask: Tensor | None = None,
) -> None:
    if base_to_correct.ndim != 3 or source_forecasts.ndim != 3:
        raise ValueError("base_to_correct and source_forecasts must have shape [B,H,C]")
    if base_to_correct.shape != source_forecasts.shape:
        raise ValueError("anchor and source forecasts must have identical shapes")
    require_finite("base_to_correct", base_to_correct)
    require_finite("source_forecasts", source_forecasts)
    if target is not None and target.shape != base_to_correct.shape:
        raise ValueError("target shape does not match forecasts")
    if mask is not None and mask.shape != base_to_correct.shape:
        raise ValueError("mask shape does not match forecasts")


@dataclass
class CrossForecastGraph:
    alpha: float
    weights: Tensor
    source_mean: Tensor
    source_std: Tensor
    intercept: Tensor
    include_self: bool = False

    def apply(
        self,
        base_to_correct: Tensor,
        source_forecasts: Tensor | None = None,
    ) -> Tensor:
        source = base_to_correct if source_forecasts is None else source_forecasts
        _validate_inputs(base_to_correct, source)
        output = base_to_correct.double().clone()
        source64 = source.double()
        _, horizon, channels = base_to_correct.shape
        expected = (horizon, channels, channels)
        if self.weights.shape != expected:
            raise ValueError("Graph state does not match forecast shape")
        for h in range(horizon):
            for target_channel in range(channels):
                standardized = (
                    source64[:, h] - self.source_mean[h, target_channel]
                ) / self.source_std[h, target_channel]
                output[:, h, target_channel] += (
                    standardized @ self.weights[h, target_channel]
                    + self.intercept[h, target_channel]
                )
        require_finite("graph output", output)
        return output.to(base_to_correct.dtype)

    def state_dict(self) -> dict[str, Tensor | float | bool]:
        return {
            "alpha": self.alpha,
            "weights": self.weights,
            "source_mean": self.source_mean,
            "source_std": self.source_std,
            "intercept": self.intercept,
            "include_self": self.include_self,
        }


def _fit(
    base_to_correct: Tensor,
    source_forecasts: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    alpha: float,
    include_self: bool,
) -> CrossForecastGraph:
    _validate_inputs(base_to_correct, source_forecasts, target, mask)
    base64 = base_to_correct.double()
    source64 = source_forecasts.double()
    target64 = target.double()
    observed_mask = mask > 0
    _, horizon, channels = base64.shape
    weights = torch.zeros(horizon, channels, channels, dtype=torch.float64)
    means = torch.zeros_like(weights)
    stds = torch.ones_like(weights)
    intercept = torch.zeros(horizon, channels, dtype=torch.float64)
    all_channels = torch.arange(channels)

    for h in range(horizon):
        for target_channel in range(channels):
            observed = observed_mask[:, h, target_channel]
            if include_self:
                source_channels = all_channels
            else:
                source_channels = all_channels[all_channels != target_channel]
            design = source64[observed, h][:, source_channels]
            residual = (
                target64[observed, h, target_channel]
                - base64[observed, h, target_channel]
            )
            solution = fit_ridge(design, residual, alpha=alpha)
            weights[h, target_channel, source_channels] = solution.weights
            means[h, target_channel, source_channels] = solution.feature_mean
            stds[h, target_channel, source_channels] = solution.feature_scale
            intercept[h, target_channel] = solution.intercept

    return CrossForecastGraph(
        alpha=float(alpha),
        weights=weights,
        source_mean=means,
        source_std=stds,
        intercept=intercept,
        include_self=bool(include_self),
    )


def fit_graph_with_validation(
    train_base: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_base: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    alphas: Iterable[float],
    train_source_forecasts: Tensor | None = None,
    val_source_forecasts: Tensor | None = None,
    include_self: bool = False,
) -> tuple[CrossForecastGraph, list[dict[str, float]]]:
    """Fit one global graph alpha using validation micro MSE.

    Omitting source tensors preserves the legacy CFG-only interface. Confirmatory
    Full must pass frozen APN forecasts explicitly while using SLRH output as the
    base_to_correct anchor.
    """

    train_source = train_base if train_source_forecasts is None else train_source_forecasts
    val_source = val_base if val_source_forecasts is None else val_source_forecasts
    _validate_inputs(train_base, train_source, train_target, train_mask)
    _validate_inputs(val_base, val_source, val_target, val_mask)
    candidates = validate_alphas(alphas)
    best: CrossForecastGraph | None = None
    best_mse = float("inf")
    audit: list[dict[str, float]] = []
    for alpha in candidates:
        graph = _fit(
            train_base,
            train_source,
            train_target,
            train_mask,
            alpha=alpha,
            include_self=include_self,
        )
        prediction = graph.apply(val_base, val_source)
        mse = masked_micro_mse(prediction, val_target, val_mask)
        audit.append({"alpha": alpha, "val_mse": mse})
        if mse < best_mse:
            best = graph
            best_mse = mse
    if best is None:
        raise RuntimeError("No graph candidate was fitted")
    return best, audit
