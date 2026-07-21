from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


SCALE_FLOOR = 1e-6


def require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def validate_alphas(values: Iterable[float]) -> tuple[float, ...]:
    alphas = tuple(float(value) for value in values)
    if not alphas:
        raise ValueError("At least one alpha is required")
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in alphas):
        raise ValueError("All alphas must be finite and positive")
    if len(set(alphas)) != len(alphas):
        raise ValueError("Alpha candidates must be unique")
    return alphas


def masked_micro_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    selected = mask > 0
    if not torch.any(selected):
        raise ValueError("No observed validation targets")
    error = prediction[selected].double() - target[selected].double()
    require_finite("masked errors", error)
    return float(error.square().sum() / error.numel())


@dataclass(frozen=True)
class RidgeSolution:
    alpha: float
    weights: Tensor
    feature_mean: Tensor
    feature_scale: Tensor
    intercept: Tensor
    observations: int

    def predict_correction(self, features: Tensor) -> Tensor:
        features64 = features.double()
        if features64.shape[-1] != self.weights.numel():
            raise ValueError("Feature width does not match ridge state")
        require_finite("features", features64)
        normalized = (features64 - self.feature_mean) / self.feature_scale
        return normalized @ self.weights + self.intercept

    def state_dict(self) -> dict[str, Tensor | float | int]:
        return {
            "alpha": self.alpha,
            "weights": self.weights,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "intercept": self.intercept,
            "observations": self.observations,
        }


def fit_ridge(
    features: Tensor,
    residual: Tensor,
    *,
    alpha: float,
    scale_floor: float = SCALE_FLOOR,
) -> RidgeSolution:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    if features.ndim != 2 or residual.ndim != 1:
        raise ValueError("features must be [rows,width] and residual must be [rows]")
    if features.shape[0] != residual.shape[0]:
        raise ValueError("features and residual row counts differ")

    design = features.double()
    response = residual.double()
    require_finite("ridge features", design)
    require_finite("ridge residual", response)
    rows, width = design.shape
    mean = torch.zeros(width, dtype=torch.float64, device=design.device)
    scale = torch.ones(width, dtype=torch.float64, device=design.device)
    weights = torch.zeros(width, dtype=torch.float64, device=design.device)
    intercept = torch.zeros((), dtype=torch.float64, device=design.device)
    if rows == 0:
        return RidgeSolution(float(alpha), weights, mean, scale, intercept, 0)

    mean = design.mean(dim=0)
    scale = design.std(dim=0, unbiased=False).clamp_min(scale_floor)
    intercept = response.mean()
    if rows >= 2 and width:
        normalized = (design - mean) / scale
        centered = response - intercept
        gram = normalized.T @ normalized
        rhs = normalized.T @ centered
        penalty = torch.eye(width, dtype=torch.float64, device=design.device) * alpha
        weights = torch.linalg.solve(gram + penalty, rhs)
    require_finite("ridge weights", weights)
    return RidgeSolution(
        float(alpha), weights, mean, scale, intercept, int(rows)
    )


def select_by_validation(
    candidates: Iterable[tuple[object, Tensor]],
    target: Tensor,
    mask: Tensor,
) -> tuple[object, list[dict[str, float]]]:
    best_state: object | None = None
    best_mse = float("inf")
    audit: list[dict[str, float]] = []
    for state, prediction in candidates:
        alpha = float(getattr(state, "alpha"))
        mse = masked_micro_mse(prediction, target, mask)
        audit.append({"alpha": alpha, "val_mse": mse})
        if mse < best_mse:
            best_state = state
            best_mse = mse
    if best_state is None:
        raise RuntimeError("No ridge candidate was evaluated")
    return best_state, audit
