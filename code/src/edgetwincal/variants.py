from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import Tensor

from .graph import CrossForecastGraph, fit_graph_with_validation
from .latent import SensorLatentResidualHead, fit_latent_head_with_validation


@dataclass(frozen=True)
class SequentialAdapter:
    order: str
    latent: SensorLatentResidualHead
    graph: CrossForecastGraph

    def apply(
        self,
        base_prediction: Tensor,
        latent_features: Tensor,
        *,
        graph_source: Tensor | None = None,
    ) -> Tensor:
        source = base_prediction if graph_source is None else graph_source
        if self.order == "slrh_then_cfg":
            local = self.latent.apply(base_prediction, latent_features)
            return self.graph.apply(local, source)
        if self.order == "cfg_then_slrh":
            cross = self.graph.apply(base_prediction, source)
            return self.latent.apply(cross, latent_features)
        raise ValueError(f"Unknown sequential order: {self.order}")

    def state_dict(self) -> dict:
        return {
            "order": self.order,
            "latent": self.latent.state_dict(),
            "graph": self.graph.state_dict(),
        }


def fit_full_with_validation(
    train_base: Tensor,
    train_latent: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_base: Tensor,
    val_latent: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    latent_alphas: Iterable[float],
    graph_alphas: Iterable[float],
    include_self: bool = False,
    train_graph_source: Tensor | None = None,
    val_graph_source: Tensor | None = None,
) -> tuple[SequentialAdapter, dict[str, list[dict[str, float]]]]:
    train_source = train_base if train_graph_source is None else train_graph_source
    val_source = val_base if val_graph_source is None else val_graph_source
    latent, latent_audit = fit_latent_head_with_validation(
        train_base,
        train_latent,
        train_target,
        train_mask,
        val_base,
        val_latent,
        val_target,
        val_mask,
        alphas=latent_alphas,
    )
    train_anchor = latent.apply(train_base, train_latent)
    val_anchor = latent.apply(val_base, val_latent)
    graph, graph_audit = fit_graph_with_validation(
        train_anchor,
        train_target,
        train_mask,
        val_anchor,
        val_target,
        val_mask,
        alphas=graph_alphas,
        train_source_forecasts=train_source,
        val_source_forecasts=val_source,
        include_self=include_self,
    )
    return (
        SequentialAdapter("slrh_then_cfg", latent, graph),
        {"slrh": latent_audit, "cfg": graph_audit},
    )


def fit_reverse_with_validation(
    train_base: Tensor,
    train_latent: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_base: Tensor,
    val_latent: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    latent_alphas: Iterable[float],
    graph_alphas: Iterable[float],
) -> tuple[SequentialAdapter, dict[str, list[dict[str, float]]]]:
    graph, graph_audit = fit_graph_with_validation(
        train_base,
        train_target,
        train_mask,
        val_base,
        val_target,
        val_mask,
        alphas=graph_alphas,
        train_source_forecasts=train_base,
        val_source_forecasts=val_base,
        include_self=False,
    )
    train_anchor = graph.apply(train_base, train_base)
    val_anchor = graph.apply(val_base, val_base)
    latent, latent_audit = fit_latent_head_with_validation(
        train_anchor,
        train_latent,
        train_target,
        train_mask,
        val_anchor,
        val_latent,
        val_target,
        val_mask,
        alphas=latent_alphas,
    )
    return (
        SequentialAdapter("cfg_then_slrh", latent, graph),
        {"cfg": graph_audit, "slrh": latent_audit},
    )


def fit_diagonal_from_frozen_latent(
    latent: SensorLatentResidualHead,
    train_base: Tensor,
    train_latent: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    val_base: Tensor,
    val_latent: Tensor,
    val_target: Tensor,
    val_mask: Tensor,
    *,
    graph_alphas: Iterable[float],
) -> tuple[SequentialAdapter, list[dict[str, float]]]:
    train_anchor = latent.apply(train_base, train_latent)
    val_anchor = latent.apply(val_base, val_latent)
    graph, audit = fit_graph_with_validation(
        train_anchor,
        train_target,
        train_mask,
        val_anchor,
        val_target,
        val_mask,
        alphas=graph_alphas,
        train_source_forecasts=train_base,
        val_source_forecasts=val_base,
        include_self=True,
    )
    return SequentialAdapter("slrh_then_cfg", latent, graph), audit
