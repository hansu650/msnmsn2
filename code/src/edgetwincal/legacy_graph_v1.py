from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    selected = mask > 0
    if not torch.any(selected):
        raise ValueError("No observed targets")
    return float((prediction[selected] - target[selected]).double().square().mean())


@dataclass
class CrossForecastGraph:
    alpha: float
    weights: Tensor
    source_mean: Tensor
    source_std: Tensor
    intercept: Tensor

    def apply(self, prediction: Tensor) -> Tensor:
        if prediction.ndim != 3:
            raise ValueError("prediction must have shape [B,H,N]")
        source = prediction.double()
        output = source.clone()
        horizon, channels = prediction.shape[1:]
        if self.weights.shape != (horizon, channels, channels):
            raise ValueError("Graph state does not match prediction shape")
        for h in range(horizon):
            for target_channel in range(channels):
                standardized = (
                    source[:, h] - self.source_mean[h, target_channel]
                ) / self.source_std[h, target_channel]
                output[:, h, target_channel] += (
                    standardized @ self.weights[h, target_channel]
                    + self.intercept[h, target_channel]
                )
        return output.to(prediction.dtype)

    def state_dict(self) -> dict[str, Tensor | float]:
        return {
            "alpha": self.alpha,
            "weights": self.weights,
            "source_mean": self.source_mean,
            "source_std": self.source_std,
            "intercept": self.intercept,
        }


def _fit(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    alpha: float,
) -> CrossForecastGraph:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    prediction = prediction.double()
    target = target.double()
    mask = mask > 0
    _, horizon, channels = prediction.shape
    weights = torch.zeros(horizon, channels, channels, dtype=torch.float64)
    means = torch.zeros_like(weights)
    stds = torch.ones_like(weights)
    intercept = torch.zeros(horizon, channels, dtype=torch.float64)
    all_channels = torch.arange(channels)
    for h in range(horizon):
        for target_channel in range(channels):
            observed = mask[:, h, target_channel]
            if int(observed.sum()) < 3:
                continue
            source_channels = all_channels[all_channels != target_channel]
            design = prediction[observed, h][:, source_channels]
            residual = (
                target[observed, h, target_channel]
                - prediction[observed, h, target_channel]
            )
            mean = design.mean(0)
            std = design.std(0).clamp_min(1e-6)
            normalized = (design - mean) / std
            centered = residual - residual.mean()
            identity = torch.eye(normalized.shape[1], dtype=torch.float64)
            coefficient = torch.linalg.solve(
                normalized.T @ normalized + alpha * identity,
                normalized.T @ centered,
            )
            weights[h, target_channel, source_channels] = coefficient
            means[h, target_channel, source_channels] = mean
            stds[h, target_channel, source_channels] = std
            intercept[h, target_channel] = residual.mean()
    return CrossForecastGraph(float(alpha), weights, means, stds, intercept)


def fit_graph_with_validation(
    train_prediction: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_prediction: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    alphas: Iterable[float],
) -> tuple[CrossForecastGraph, list[dict[str, float]]]:
    candidates = [float(value) for value in alphas]
    if not candidates:
        raise ValueError("At least one alpha is required")
    best: CrossForecastGraph | None = None
    best_mse = float("inf")
    audit: list[dict[str, float]] = []
    for alpha in candidates:
        graph = _fit(train_prediction, train_target, train_mask, alpha=alpha)
        mse = _masked_mse(graph.apply(val_prediction), val_target, val_mask)
        audit.append({"alpha": alpha, "val_mse": mse})
        if mse < best_mse:
            best, best_mse = graph, mse
    if best is None:
        raise RuntimeError("No graph candidate was fitted")
    return best, audit
