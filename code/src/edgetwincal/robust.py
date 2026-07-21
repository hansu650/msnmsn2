"""Group-balanced robust joint residual fitting for EdgeTwinCal-Safe.

This module is independent of the sealed EdgeTwinCal implementations. Fitting
is deterministic CPU float64; invalid data or an unreliable cell produces an
explicit zero-correction state rather than a partially fitted adapter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
from torch import Tensor


MAD_NORMALIZER = 1.4826
DEFAULT_HUBER_DELTA = 1.345
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_TOLERANCE = 1e-8
DEFAULT_SCALE_FLOOR = 1e-6
DEFAULT_FEATURE_CLIP = 5.0


@dataclass(frozen=True)
class RobustCellAudit:
    status: str
    reason: str
    rows: int
    groups: int
    width: int
    iterations: int
    converged: bool
    robust_scale: float
    downweighted_fraction: float
    observation_weighted: bool
    squared_loss: bool
    penalty_scale: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustCellState:
    latent_width: int
    weights: Tensor
    feature_center: Tensor
    feature_scale: Tensor
    intercept: Tensor
    residual_scale: Tensor
    feature_clip: float
    audit: RobustCellAudit

    @property
    def active(self) -> bool:
        return self.audit.status == "fitted"

    def predict_correction(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.weights.numel():
            raise ValueError("features must have shape [N, cell_width]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("apply features contain NaN or infinity")
        x = features.double()
        center = self.feature_center.to(device=x.device)
        scale = self.feature_scale.to(device=x.device)
        weights = self.weights.to(device=x.device)
        z = ((x - center) / scale).clamp(-self.feature_clip, self.feature_clip)
        return z @ weights + self.intercept.to(device=x.device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "latent_width": self.latent_width,
            "weights": self.weights,
            "feature_center": self.feature_center,
            "feature_scale": self.feature_scale,
            "intercept": self.intercept,
            "residual_scale": self.residual_scale,
            "feature_clip": self.feature_clip,
            "audit": self.audit.as_dict(),
        }


@dataclass(frozen=True)
class RobustAdapterAudit:
    cells: int
    fitted_cells: int
    zero_cells: int
    fallback_reasons: tuple[tuple[str, int], ...]
    observation_weighted: bool
    squared_loss: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustJointAdapter:
    horizon: int
    channels: int
    latent_width: int
    alpha_latent: float
    alpha_cross: float
    feature_clip: float
    cells: tuple[tuple[RobustCellState, ...], ...]
    audit: RobustAdapterAudit

    def iter_cells(self) -> Iterator[RobustCellState]:
        for row in self.cells:
            yield from row

    def apply(self, latent: Tensor, base_forecasts: Tensor) -> Tensor:
        return apply_robust_joint(self, latent, base_forecasts)

    def state_dict(self) -> dict[str, Any]:
        states = list(self.iter_cells())
        width = self.latent_width + self.channels - 1
        weights = torch.stack([state.weights for state in states]).reshape(
            self.horizon, self.channels, width
        )
        centers = torch.stack([state.feature_center for state in states]).reshape(
            self.horizon, self.channels, width
        )
        scales = torch.stack([state.feature_scale for state in states]).reshape(
            self.horizon, self.channels, width
        )
        intercepts = torch.stack([state.intercept for state in states]).reshape(
            self.horizon, self.channels
        )
        residual_scales = torch.stack(
            [state.residual_scale for state in states]
        ).reshape(self.horizon, self.channels)
        return {
            "horizon": self.horizon,
            "channels": self.channels,
            "latent_width": self.latent_width,
            "alpha_latent": self.alpha_latent,
            "alpha_cross": self.alpha_cross,
            "feature_clip": self.feature_clip,
            "weights": weights,
            "feature_center": centers,
            "feature_scale": scales,
            "intercept": intercepts,
            "residual_scale": residual_scales,
            "cell_audits": [
                [cell.audit.as_dict() for cell in row] for row in self.cells
            ],
            "audit": self.audit.as_dict(),
        }


def _groups_cpu(group_ids: Tensor, *, allow_empty: bool = False) -> Tensor:
    if not isinstance(group_ids, Tensor) or group_ids.ndim != 1:
        raise ValueError("group_ids must be a rank-one tensor")
    if group_ids.numel() == 0 and allow_empty:
        return group_ids.detach().to(device="cpu", dtype=torch.int64)
    if group_ids.numel() == 0:
        raise ValueError("group_ids must not be empty")
    if group_ids.dtype == torch.bool or group_ids.dtype.is_floating_point:
        raise ValueError("group_ids must use an integer dtype")
    return group_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()


def group_balanced_weights(group_ids: Tensor) -> Tensor:
    """Return CPU float64 row weights N/(G*n_g)."""

    groups = _groups_cpu(group_ids)
    _, inverse, counts = torch.unique(
        groups, sorted=True, return_inverse=True, return_counts=True
    )
    rows = groups.numel()
    n_groups = counts.numel()
    return float(rows) / (
        float(n_groups) * counts[inverse].to(dtype=torch.float64)
    )


def weighted_quantile(values: Tensor, weights: Tensor, q: float) -> Tensor:
    """Deterministic lower weighted quantile of finite one-dimensional values."""

    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must lie in [0,1]")
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be matching rank-one tensors")
    value = values.detach().to(device="cpu", dtype=torch.float64)
    weight = weights.detach().to(device="cpu", dtype=torch.float64)
    valid = weight > 0
    if not bool(valid.any()):
        raise ValueError("weighted quantile requires positive total weight")
    value, weight = value[valid], weight[valid]
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("weighted quantile inputs must be finite")
    order = torch.argsort(value, stable=True)
    ordered_value, ordered_weight = value[order], weight[order]
    cumulative = torch.cumsum(ordered_weight, dim=0)
    threshold = float(q) * float(cumulative[-1])
    index = int(
        torch.searchsorted(
            cumulative, torch.tensor(threshold, dtype=torch.float64)
        )
    )
    return ordered_value[min(index, ordered_value.numel() - 1)]


def weighted_median_mad(
    values: Tensor,
    weights: Tensor,
    *,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
) -> tuple[Tensor, Tensor]:
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    median = weighted_quantile(values, weights, 0.5)
    mad = weighted_quantile((values.double().cpu() - median).abs(), weights, 0.5)
    return median, (MAD_NORMALIZER * mad).clamp_min(scale_floor)


def _zero_cell(
    width: int,
    latent_width: int,
    feature_clip: float,
    *,
    reason: str,
    rows: int,
    groups: int,
    observation_weighted: bool,
    squared_loss: bool,
    iterations: int = 0,
    robust_scale: float = 0.0,
    penalty_scale: float = 1.0,
) -> RobustCellState:
    audit = RobustCellAudit(
        "zero_fallback",
        reason,
        int(rows),
        int(groups),
        int(width),
        int(iterations),
        False,
        float(robust_scale),
        0.0,
        bool(observation_weighted),
        bool(squared_loss),
        float(penalty_scale),
    )
    return RobustCellState(
        int(latent_width),
        torch.zeros(width, dtype=torch.float64),
        torch.zeros(width, dtype=torch.float64),
        torch.ones(width, dtype=torch.float64),
        torch.zeros((), dtype=torch.float64),
        torch.zeros((), dtype=torch.float64),
        float(feature_clip),
        audit,
    )


def _location_scale(
    features: Tensor, weights: Tensor, scale_floor: float
) -> tuple[Tensor, Tensor]:
    width = features.shape[1]
    center = torch.empty(width, dtype=torch.float64)
    scale = torch.empty(width, dtype=torch.float64)
    for column in range(width):
        center[column], scale[column] = weighted_median_mad(
            features[:, column], weights, scale_floor=scale_floor
        )
    return center, scale


def _block_solve(
    design: Tensor,
    response: Tensor,
    weights: Tensor,
    *,
    latent_width: int,
    alpha_latent: float,
    alpha_cross: float,
    penalty_scale: float,
) -> tuple[Tensor, Tensor]:
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise RuntimeError("non-positive effective weight")
    width = design.shape[1]
    x_mean = (weights[:, None] * design).sum(0) / total
    y_mean = (weights * response).sum() / total
    centered_x = design - x_mean
    centered_y = response - y_mean
    penalty = torch.cat(
        [
            torch.full((latent_width,), alpha_latent, dtype=torch.float64),
            torch.full((width - latent_width,), alpha_cross, dtype=torch.float64),
        ]
    ) * float(penalty_scale)
    gram = centered_x.T @ (weights[:, None] * centered_x)
    rhs = centered_x.T @ (weights * centered_y)
    coefficient = (
        torch.linalg.solve(gram + torch.diag(penalty), rhs)
        if width
        else torch.empty(0, dtype=torch.float64)
    )
    intercept = y_mean - x_mean @ coefficient
    return coefficient, intercept


def fit_group_huber_block_ridge(
    features: Tensor,
    residual: Tensor,
    groups: Tensor,
    *,
    latent_width: int,
    alpha_latent: float,
    alpha_cross: float,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
    feature_clip: float = DEFAULT_FEATURE_CLIP,
    minimum_rows: int | None = None,
    minimum_groups: int = 20,
    observation_weighted: bool = False,
    squared_loss: bool = False,
) -> RobustCellState:
    """Fit one robust block-ridge output cell or return an audited zero state."""

    if features.ndim != 2 or residual.ndim != 1:
        raise ValueError("features must be [N,P] and residual must be [N]")
    if features.shape[0] != residual.numel() or groups.numel() != residual.numel():
        raise ValueError("feature, residual, and group row counts differ")
    width = features.shape[1]
    if not 0 <= latent_width <= width:
        raise ValueError("latent_width is outside the feature width")
    constants = (
        alpha_latent,
        alpha_cross,
        huber_delta,
        tolerance,
        scale_floor,
        feature_clip,
    )
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in constants):
        raise ValueError("penalties and numerical constants must be finite and positive")
    if max_iterations <= 0 or minimum_groups < 1:
        raise ValueError("iteration and group minimums must be positive")
    required_rows = max(100, 4 * width) if minimum_rows is None else int(minimum_rows)
    if required_rows < 1:
        raise ValueError("minimum_rows must be positive")

    rows = int(residual.numel())
    if rows == 0:
        return _zero_cell(
            width,
            latent_width,
            feature_clip,
            reason="empty",
            rows=0,
            groups=0,
            observation_weighted=observation_weighted,
            squared_loss=squared_loss,
        )
    group_ids = _groups_cpu(groups)
    group_count = int(torch.unique(group_ids).numel())
    penalty_scale = 1.0 if observation_weighted else float(rows) / group_count
    if rows < required_rows:
        return _zero_cell(
            width,
            latent_width,
            feature_clip,
            reason="insufficient_rows",
            rows=rows,
            groups=group_count,
            observation_weighted=observation_weighted,
            squared_loss=squared_loss,
            penalty_scale=penalty_scale,
        )
    if group_count < minimum_groups:
        return _zero_cell(
            width,
            latent_width,
            feature_clip,
            reason="insufficient_groups",
            rows=rows,
            groups=group_count,
            observation_weighted=observation_weighted,
            squared_loss=squared_loss,
            penalty_scale=penalty_scale,
        )

    design = features.detach().to(device="cpu", dtype=torch.float64).contiguous()
    response = residual.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(design).all()) or not bool(torch.isfinite(response).all()):
        return _zero_cell(
            width,
            latent_width,
            feature_clip,
            reason="nonfinite_input",
            rows=rows,
            groups=group_count,
            observation_weighted=observation_weighted,
            squared_loss=squared_loss,
            penalty_scale=penalty_scale,
        )
    base_weights = (
        torch.ones(rows, dtype=torch.float64)
        if observation_weighted
        else group_balanced_weights(group_ids)
    )
    try:
        center, scale = _location_scale(design, base_weights, scale_floor)
        _, residual_scale = weighted_median_mad(
            response, base_weights, scale_floor=scale_floor
        )
        standardized = ((design - center) / scale).clamp(
            -feature_clip, feature_clip
        )
        coefficient, intercept = _block_solve(
            standardized,
            response,
            base_weights,
            latent_width=latent_width,
            alpha_latent=alpha_latent,
            alpha_cross=alpha_cross,
            penalty_scale=penalty_scale,
        )
    except (RuntimeError, ValueError):
        return _zero_cell(
            width,
            latent_width,
            feature_clip,
            reason="solve_failed",
            rows=rows,
            groups=group_count,
            observation_weighted=observation_weighted,
            squared_loss=squared_loss,
            penalty_scale=penalty_scale,
        )

    iterations = 1
    converged = bool(squared_loss)
    final_robust = torch.ones(rows, dtype=torch.float64)
    if not squared_loss:
        old = torch.cat([intercept.reshape(1), coefficient])
        cutoff = float(huber_delta) * float(residual_scale)
        for iterations in range(1, max_iterations + 1):
            error = response - (standardized @ coefficient + intercept)
            final_robust = torch.minimum(
                torch.ones_like(error),
                torch.full_like(error, cutoff)
                / error.abs().clamp_min(scale_floor),
            )
            effective = base_weights * final_robust
            effective = effective * (float(rows) / float(effective.sum()))
            try:
                new_coefficient, new_intercept = _block_solve(
                    standardized,
                    response,
                    effective,
                    latent_width=latent_width,
                    alpha_latent=alpha_latent,
                    alpha_cross=alpha_cross,
                    penalty_scale=penalty_scale,
                )
            except RuntimeError:
                return _zero_cell(
                    width,
                    latent_width,
                    feature_clip,
                    reason="solve_failed",
                    rows=rows,
                    groups=group_count,
                    observation_weighted=observation_weighted,
                    squared_loss=squared_loss,
                    iterations=iterations,
                    robust_scale=float(residual_scale),
                    penalty_scale=penalty_scale,
                )
            new = torch.cat([new_intercept.reshape(1), new_coefficient])
            change = torch.linalg.vector_norm(new - old)
            reference = max(1.0, float(torch.linalg.vector_norm(old)))
            coefficient, intercept, old = new_coefficient, new_intercept, new
            if float(change) <= tolerance * reference:
                converged = True
                break
    tensors = (coefficient, intercept, center, scale, residual_scale)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        return _zero_cell(
            width,
            latent_width,
            feature_clip,
            reason="nonfinite_state",
            rows=rows,
            groups=group_count,
            observation_weighted=observation_weighted,
            squared_loss=squared_loss,
            iterations=iterations,
            robust_scale=float(residual_scale),
            penalty_scale=penalty_scale,
        )
    if not converged:
        return _zero_cell(
            width,
            latent_width,
            feature_clip,
            reason="nonconvergence",
            rows=rows,
            groups=group_count,
            observation_weighted=observation_weighted,
            squared_loss=squared_loss,
            iterations=iterations,
            robust_scale=float(residual_scale),
            penalty_scale=penalty_scale,
        )
    audit = RobustCellAudit(
        "fitted",
        "ok",
        rows,
        group_count,
        width,
        iterations,
        True,
        float(residual_scale),
        float((final_robust < 1.0).double().mean()),
        bool(observation_weighted),
        bool(squared_loss),
        penalty_scale,
    )
    return RobustCellState(
        int(latent_width),
        coefficient,
        center,
        scale,
        intercept,
        residual_scale,
        float(feature_clip),
        audit,
    )


def _joint_features(
    latent: Tensor,
    base_forecasts: Tensor,
    horizon: int,
    target_channel: int,
) -> Tensor:
    channels = base_forecasts.shape[2]
    indices = torch.arange(channels, device=base_forecasts.device)
    other = indices[indices != target_channel]
    return torch.cat(
        [latent[:, target_channel], base_forecasts[:, horizon, other]], dim=1
    )


def fit_robust_joint_adapter(
    latent: Tensor,
    base_forecasts: Tensor,
    target: Tensor,
    mask: Tensor,
    group_ids: Tensor,
    *,
    alpha_latent: float,
    alpha_cross: float,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
    feature_clip: float = DEFAULT_FEATURE_CLIP,
    minimum_rows: int | None = None,
    minimum_groups: int = 20,
    observation_weighted: bool = False,
    squared_loss: bool = False,
) -> RobustJointAdapter:
    """Fit all [H,C] cells using target latent plus zero-diagonal forecasts."""

    if latent.ndim != 3 or base_forecasts.ndim != 3:
        raise ValueError("latent and base_forecasts must be [N,C,D] and [N,H,C]")
    if target.shape != base_forecasts.shape or mask.shape != base_forecasts.shape:
        raise ValueError("target and mask must match base_forecasts")
    rows, horizon, channels = base_forecasts.shape
    if latent.shape[:2] != (rows, channels) or group_ids.shape != (rows,):
        raise ValueError("latent/group dimensions do not match forecasts")
    if rows <= 0 or horizon <= 0 or channels <= 0:
        raise ValueError("adapter dimensions must be positive")
    _groups_cpu(group_ids)

    latent64 = latent.detach().to(device="cpu", dtype=torch.float64).contiguous()
    base64 = base_forecasts.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    target64 = target.detach().to(device="cpu", dtype=torch.float64).contiguous()
    observed = mask.detach().to(device="cpu") > 0
    groups64 = group_ids.detach().to(
        device="cpu", dtype=torch.int64
    ).contiguous()
    cell_rows: list[tuple[RobustCellState, ...]] = []
    for h in range(horizon):
        current: list[RobustCellState] = []
        for channel in range(channels):
            selected = observed[:, h, channel]
            features = _joint_features(latent64, base64, h, channel)[selected]
            residual = (
                target64[:, h, channel] - base64[:, h, channel]
            )[selected]
            current.append(
                fit_group_huber_block_ridge(
                    features,
                    residual,
                    groups64[selected],
                    latent_width=latent.shape[2],
                    alpha_latent=alpha_latent,
                    alpha_cross=alpha_cross,
                    huber_delta=huber_delta,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    scale_floor=scale_floor,
                    feature_clip=feature_clip,
                    minimum_rows=minimum_rows,
                    minimum_groups=minimum_groups,
                    observation_weighted=observation_weighted,
                    squared_loss=squared_loss,
                )
            )
        cell_rows.append(tuple(current))
    cells = tuple(cell_rows)
    all_cells = [cell for row in cells for cell in row]
    reasons = Counter(
        cell.audit.reason for cell in all_cells if cell.audit.status != "fitted"
    )
    fitted = sum(cell.active for cell in all_cells)
    audit = RobustAdapterAudit(
        len(all_cells),
        fitted,
        len(all_cells) - fitted,
        tuple(sorted(reasons.items())),
        bool(observation_weighted),
        bool(squared_loss),
    )
    return RobustJointAdapter(
        horizon,
        channels,
        latent.shape[2],
        float(alpha_latent),
        float(alpha_cross),
        float(feature_clip),
        cells,
        audit,
    )


def apply_robust_joint(
    state: RobustJointAdapter,
    latent: Tensor,
    base_forecasts: Tensor,
) -> Tensor:
    if latent.ndim != 3 or base_forecasts.ndim != 3:
        raise ValueError("latent and base_forecasts must be rank three")
    rows, horizon, channels = base_forecasts.shape
    expected = (state.horizon, state.channels, state.latent_width)
    if (horizon, channels, latent.shape[2]) != expected:
        raise ValueError("apply tensors do not match robust adapter state")
    if latent.shape[:2] != (rows, channels):
        raise ValueError("apply tensors do not match robust adapter state")
    if not bool(torch.isfinite(latent).all()) or not bool(
        torch.isfinite(base_forecasts).all()
    ):
        raise ValueError("apply tensors contain NaN or infinity")
    output = base_forecasts.double().clone()
    latent64, base64 = latent.double(), base_forecasts.double()
    for h in range(horizon):
        for channel in range(channels):
            features = _joint_features(latent64, base64, h, channel)
            output[:, h, channel] += state.cells[h][
                channel
            ].predict_correction(features)
    return output.to(dtype=base_forecasts.dtype)


__all__ = [
    "RobustAdapterAudit",
    "RobustCellAudit",
    "RobustCellState",
    "RobustJointAdapter",
    "apply_robust_joint",
    "fit_group_huber_block_ridge",
    "fit_robust_joint_adapter",
    "group_balanced_weights",
    "weighted_median_mad",
    "weighted_quantile",
]
