from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import torch
from torch import Tensor

from .ridge import SCALE_FLOOR, masked_micro_mse, require_finite, validate_alphas


@dataclass(frozen=True)
class JointResidualRidge:
    alpha_latent: float
    alpha_graph: float
    latent_weights: Tensor
    cross_weights: Tensor
    latent_mean: Tensor
    latent_scale: Tensor
    cross_mean: Tensor
    cross_scale: Tensor
    intercept: Tensor
    observations: Tensor
    include_self: bool = False

    def apply(
        self,
        base_to_correct: Tensor,
        latent_features: Tensor,
        source_forecasts: Tensor,
    ) -> Tensor:
        if base_to_correct.ndim != 3 or source_forecasts.shape != base_to_correct.shape:
            raise ValueError("forecast tensors must have identical [B,H,C] shapes")
        if latent_features.ndim != 3:
            raise ValueError("latent_features must have shape [B,C,D]")
        if latent_features.shape[:2] != (
            base_to_correct.shape[0],
            base_to_correct.shape[2],
        ):
            raise ValueError("latent feature batch/channel dimensions do not match")
        require_finite("joint base", base_to_correct)
        require_finite("joint latent", latent_features)
        require_finite("joint source", source_forecasts)
        output = base_to_correct.double().clone()
        latent64 = latent_features.double()
        source64 = source_forecasts.double()
        _, horizon, channels = base_to_correct.shape
        all_channels = torch.arange(channels)
        for h in range(horizon):
            for target_channel in range(channels):
                source_channels = (
                    all_channels
                    if self.include_self
                    else all_channels[all_channels != target_channel]
                )
                latent_z = (
                    latent64[:, target_channel] - self.latent_mean[h, target_channel]
                ) / self.latent_scale[h, target_channel]
                cross_z = (
                    source64[:, h, source_channels]
                    - self.cross_mean[h, target_channel, source_channels]
                ) / self.cross_scale[h, target_channel, source_channels]
                correction = (
                    latent_z @ self.latent_weights[h, target_channel]
                    + cross_z @ self.cross_weights[h, target_channel, source_channels]
                    + self.intercept[h, target_channel]
                )
                output[:, h, target_channel] += correction
        require_finite("joint output", output)
        return output.to(base_to_correct.dtype)

    @property
    def coefficient_slots(self) -> int:
        _, channels, _ = self.cross_weights.shape
        cross_per_target = channels if self.include_self else channels - 1
        return int(
            self.latent_weights.numel()
            + self.intercept.numel()
            + self.cross_weights.shape[0] * channels * cross_per_target
        )

    def state_dict(self) -> dict[str, Tensor | float | bool | int]:
        return {
            "alpha_latent": self.alpha_latent,
            "alpha_graph": self.alpha_graph,
            "latent_weights": self.latent_weights,
            "cross_weights": self.cross_weights,
            "latent_mean": self.latent_mean,
            "latent_scale": self.latent_scale,
            "cross_mean": self.cross_mean,
            "cross_scale": self.cross_scale,
            "intercept": self.intercept,
            "observations": self.observations,
            "include_self": self.include_self,
            "coefficient_slots": self.coefficient_slots,
        }


def _fit_joint(
    base: Tensor,
    latent: Tensor,
    source: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    alpha_latent: float,
    alpha_graph: float,
    include_self: bool,
    scale_floor: float = SCALE_FLOOR,
) -> JointResidualRidge:
    if base.ndim != 3 or source.shape != base.shape:
        raise ValueError("base/source must have identical [B,H,C] shapes")
    if target.shape != base.shape or mask.shape != base.shape:
        raise ValueError("target/mask shapes do not match base")
    if latent.ndim != 3 or latent.shape[:2] != (base.shape[0], base.shape[2]):
        raise ValueError("latent must have shape [B,C,D]")
    for name, tensor in (
        ("base", base),
        ("latent", latent),
        ("source", source),
        ("target", target),
    ):
        require_finite(name, tensor)

    base64, latent64, source64, target64 = (
        base.double(),
        latent.double(),
        source.double(),
        target.double(),
    )
    observed_mask = mask > 0
    _, horizon, channels = base.shape
    latent_dim = latent.shape[-1]
    latent_weights = torch.zeros(horizon, channels, latent_dim, dtype=torch.float64)
    cross_weights = torch.zeros(horizon, channels, channels, dtype=torch.float64)
    latent_mean = torch.zeros_like(latent_weights)
    latent_scale = torch.ones_like(latent_weights)
    cross_mean = torch.zeros_like(cross_weights)
    cross_scale = torch.ones_like(cross_weights)
    intercept = torch.zeros(horizon, channels, dtype=torch.float64)
    observations = torch.zeros(horizon, channels, dtype=torch.int64)
    all_channels = torch.arange(channels)

    for h in range(horizon):
        for target_channel in range(channels):
            observed = observed_mask[:, h, target_channel]
            source_channels = (
                all_channels
                if include_self
                else all_channels[all_channels != target_channel]
            )
            latent_design = latent64[observed, target_channel]
            cross_design = source64[observed, h][:, source_channels]
            response = (
                target64[observed, h, target_channel]
                - base64[observed, h, target_channel]
            )
            rows = response.numel()
            observations[h, target_channel] = rows
            if rows == 0:
                continue
            latent_mu = latent_design.mean(0)
            latent_sd = latent_design.std(0, unbiased=False).clamp_min(scale_floor)
            cross_mu = cross_design.mean(0)
            cross_sd = cross_design.std(0, unbiased=False).clamp_min(scale_floor)
            response_mean = response.mean()
            latent_mean[h, target_channel] = latent_mu
            latent_scale[h, target_channel] = latent_sd
            cross_mean[h, target_channel, source_channels] = cross_mu
            cross_scale[h, target_channel, source_channels] = cross_sd
            intercept[h, target_channel] = response_mean
            if rows < 2:
                continue
            latent_z = (latent_design - latent_mu) / latent_sd
            cross_z = (cross_design - cross_mu) / cross_sd
            design = torch.cat([latent_z, cross_z], dim=1)
            centered = response - response_mean
            penalties = torch.cat(
                [
                    torch.full((latent_dim,), alpha_latent, dtype=torch.float64),
                    torch.full((source_channels.numel(),), alpha_graph, dtype=torch.float64),
                ]
            )
            coefficient = torch.linalg.solve(
                design.T @ design + torch.diag(penalties),
                design.T @ centered,
            )
            latent_weights[h, target_channel] = coefficient[:latent_dim]
            cross_weights[h, target_channel, source_channels] = coefficient[latent_dim:]

    return JointResidualRidge(
        float(alpha_latent),
        float(alpha_graph),
        latent_weights,
        cross_weights,
        latent_mean,
        latent_scale,
        cross_mean,
        cross_scale,
        intercept,
        observations,
        bool(include_self),
    )


def fit_joint_with_validation(
    train_base: Tensor,
    train_latent: Tensor,
    train_source: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_base: Tensor,
    val_latent: Tensor,
    val_source: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    latent_alphas: Iterable[float],
    graph_alphas: Iterable[float],
    include_self: bool = False,
) -> tuple[JointResidualRidge, list[dict[str, float]]]:
    latent_grid = validate_alphas(latent_alphas)
    graph_grid = validate_alphas(graph_alphas)
    best: JointResidualRidge | None = None
    best_mse = float("inf")
    audit: list[dict[str, float]] = []
    for alpha_latent, alpha_graph in product(latent_grid, graph_grid):
        state = _fit_joint(
            train_base,
            train_latent,
            train_source,
            train_target,
            train_mask,
            alpha_latent=alpha_latent,
            alpha_graph=alpha_graph,
            include_self=include_self,
        )
        prediction = state.apply(val_base, val_latent, val_source)
        mse = masked_micro_mse(prediction, val_target, val_mask)
        audit.append(
            {
                "alpha_latent": alpha_latent,
                "alpha_graph": alpha_graph,
                "val_mse": mse,
            }
        )
        if mse < best_mse:
            best, best_mse = state, mse
    if best is None:
        raise RuntimeError("No joint candidate was fitted")
    return best, audit
