from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    selected = mask > 0
    return float((prediction[selected] - target[selected]).double().square().mean())


@dataclass
class SensorLatentResidualHead:
    alpha: float
    weights: Tensor
    feature_mean: Tensor
    feature_std: Tensor
    intercept: Tensor

    def apply(self, base_prediction: Tensor, features: Tensor) -> Tensor:
        if base_prediction.ndim != 3 or features.ndim != 3:
            raise ValueError("base_prediction and features must be rank three")
        output = base_prediction.double().clone()
        hidden = features.double()
        _, horizon, channels = base_prediction.shape
        for h in range(horizon):
            for channel in range(channels):
                normalized = (
                    hidden[:, channel] - self.feature_mean[h, channel]
                ) / self.feature_std[h, channel]
                output[:, h, channel] += (
                    normalized @ self.weights[h, channel]
                    + self.intercept[h, channel]
                )
        return output.to(base_prediction.dtype)

    def state_dict(self) -> dict[str, Tensor | float]:
        return {
            "alpha": self.alpha,
            "weights": self.weights,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "intercept": self.intercept,
        }


def _fit(
    base_prediction: Tensor,
    features: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    alpha: float,
) -> SensorLatentResidualHead:
    base_prediction = base_prediction.double()
    features = features.double()
    target = target.double()
    mask = mask > 0
    _, horizon, channels = base_prediction.shape
    feature_dim = features.shape[-1]
    weights = torch.zeros(horizon, channels, feature_dim, dtype=torch.float64)
    means = torch.zeros_like(weights)
    stds = torch.ones_like(weights)
    intercept = torch.zeros(horizon, channels, dtype=torch.float64)
    identity = torch.eye(feature_dim, dtype=torch.float64)
    for h in range(horizon):
        for channel in range(channels):
            observed = mask[:, h, channel]
            if int(observed.sum()) < 3:
                continue
            design = features[observed, channel]
            residual = (
                target[observed, h, channel]
                - base_prediction[observed, h, channel]
            )
            mean = design.mean(0)
            std = design.std(0).clamp_min(1e-6)
            normalized = (design - mean) / std
            centered = residual - residual.mean()
            coefficient = torch.linalg.solve(
                normalized.T @ normalized + alpha * identity,
                normalized.T @ centered,
            )
            weights[h, channel] = coefficient
            means[h, channel] = mean
            stds[h, channel] = std
            intercept[h, channel] = residual.mean()
    return SensorLatentResidualHead(float(alpha), weights, means, stds, intercept)


def fit_latent_head_with_validation(
    train_base: Tensor,
    train_features: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_base: Tensor,
    val_features: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    alphas: Iterable[float],
) -> tuple[SensorLatentResidualHead, list[dict[str, float]]]:
    candidates = [float(value) for value in alphas]
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("alphas must be positive")
    best: SensorLatentResidualHead | None = None
    best_mse = float("inf")
    audit: list[dict[str, float]] = []
    for alpha in candidates:
        head = _fit(
            train_base,
            train_features,
            train_target,
            train_mask,
            alpha=alpha,
        )
        prediction = head.apply(val_base, val_features)
        mse = _masked_mse(prediction, val_target, val_mask)
        audit.append({"alpha": alpha, "val_mse": mse})
        if mse < best_mse:
            best, best_mse = head, mse
    if best is None:
        raise RuntimeError("No latent residual head was fitted")
    return best, audit
