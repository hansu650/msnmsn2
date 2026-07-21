from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ErrorCell:
    group_hash: str
    checkpoint: str
    variant: str
    sse: float
    sae: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def accumulate_error_cells(
    records: Iterable[Mapping[str, Any]],
) -> list[ErrorCell]:
    totals: dict[tuple[str, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    for row in records:
        key = (
            str(row["group_hash"]),
            str(row["checkpoint"]),
            str(row["variant"]),
        )
        sse = float(row["sse"])
        sae = float(row["sae"])
        n = int(row["n"])
        if not np.isfinite(sse) or not np.isfinite(sae):
            raise ValueError("SSE/SAE must be finite")
        if sse < 0 or sae < 0 or n < 0:
            raise ValueError("SSE/SAE/N must be nonnegative")
        if n == 0 and (sse != 0 or sae != 0):
            raise ValueError("Zero-count cells must have zero errors")
        totals[key][0] += sse
        totals[key][1] += sae
        totals[key][2] += n
    cells = [
        ErrorCell(group, checkpoint, variant, values[0], values[1], int(values[2]))
        for (group, checkpoint, variant), values in sorted(totals.items())
    ]
    if not cells:
        raise ValueError("No error cells supplied")
    return cells


def error_cells_from_arrays(
    *,
    group_hashes: Sequence[str],
    checkpoint: str | int,
    variant: str,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> list[ErrorCell]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask shapes differ")
    if prediction.shape[0] != len(group_hashes):
        raise ValueError("group hash count does not match array rows")
    records: list[dict[str, Any]] = []
    for index, group_hash in enumerate(group_hashes):
        selected = mask[index] > 0
        error = prediction[index][selected].astype(np.float64) - target[index][
            selected
        ].astype(np.float64)
        records.append(
            {
                "group_hash": str(group_hash),
                "checkpoint": str(checkpoint),
                "variant": str(variant),
                "sse": float(np.square(error).sum(dtype=np.float64)),
                "sae": float(np.abs(error).sum(dtype=np.float64)),
                "n": int(error.size),
            }
        )
    return accumulate_error_cells(records)


def _paired_cube(
    cells: Sequence[ErrorCell],
    variants: Sequence[str],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    groups = sorted({cell.group_hash for cell in cells})
    checkpoints = sorted({cell.checkpoint for cell in cells})
    variant_names = list(dict.fromkeys(str(value) for value in variants))
    if not groups or not checkpoints or not variant_names:
        raise ValueError("groups, checkpoints, and variants must be non-empty")
    group_index = {value: index for index, value in enumerate(groups)}
    checkpoint_index = {value: index for index, value in enumerate(checkpoints)}
    variant_index = {value: index for index, value in enumerate(variant_names)}
    shape = (len(variant_names), len(groups), len(checkpoints))
    sse = np.full(shape, np.nan, dtype=np.float64)
    sae = np.full(shape, np.nan, dtype=np.float64)
    n = np.full(shape, np.nan, dtype=np.float64)
    for cell in cells:
        if cell.variant not in variant_index:
            continue
        index = (
            variant_index[cell.variant],
            group_index[cell.group_hash],
            checkpoint_index[cell.checkpoint],
        )
        if np.isfinite(n[index]):
            raise ValueError("Duplicate cells must be accumulated before inference")
        sse[index], sae[index], n[index] = cell.sse, cell.sae, cell.n
    if not np.isfinite(n).all():
        missing = int(np.isnan(n).sum())
        raise ValueError(f"Incomplete paired cell cube: {missing} cells missing")
    if not np.array_equal(n, np.broadcast_to(n[:1], n.shape)):
        raise ValueError("Paired variants must have identical N for every group/checkpoint")

    globally_empty = np.all(n == 0, axis=(0, 2))
    fully_evaluable = np.all(n > 0, axis=(0, 2))
    if np.any(~(globally_empty | fully_evaluable)):
        raise ValueError(
            "Each paired group must be globally empty or positive in every checkpoint"
        )
    if not np.any(fully_evaluable):
        raise ValueError("Crossed inference has no evaluable groups")
    if np.any(globally_empty):
        groups = [group for group, keep in zip(groups, fully_evaluable) if keep]
        sse = sse[:, fully_evaluable, :]
        sae = sae[:, fully_evaluable, :]
        n = n[:, fully_evaluable, :]
    return groups, checkpoints, sse, sae, n


def pooled_metrics(cells: Sequence[ErrorCell]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for cell in cells:
        totals[cell.variant][0] += cell.sse
        totals[cell.variant][1] += cell.sae
        totals[cell.variant][2] += cell.n
    output: dict[str, dict[str, float | int]] = {}
    for variant, (sse, sae, n_float) in sorted(totals.items()):
        n = int(n_float)
        if n <= 0:
            raise ValueError(f"Variant {variant} has no observed targets")
        output[variant] = {
            "mse": float(sse / n),
            "mae": float(sae / n),
            "sse": float(sse),
            "sae": float(sae),
            "n": n,
        }
    return output


def crossed_cluster_checkpoint_bootstrap(
    cells: Sequence[ErrorCell] | Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    comparators: Sequence[str],
    resamples: int = 50_000,
    random_seed: int = 20_260_721,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    if resamples <= 0 or batch_size <= 0:
        raise ValueError("resamples and batch_size must be positive")
    normalized = (
        list(cells)
        if cells and isinstance(cells[0], ErrorCell)  # type: ignore[index]
        else accumulate_error_cells(cells)  # type: ignore[arg-type]
    )
    variants = [baseline, *comparators]
    if len(set(variants)) != len(variants):
        raise ValueError("Baseline and comparators must be unique")
    groups, checkpoints, sse, sae, n = _paired_cube(normalized, variants)
    point = pooled_metrics(normalized)
    variant_index = {variant: index for index, variant in enumerate(variants)}
    baseline_index = variant_index[baseline]
    rng = np.random.default_rng(random_seed)

    draws: dict[tuple[str, str, str], np.ndarray] = {}
    for comparator in comparators:
        for metric in ("mse", "mae"):
            draws[(comparator, metric, "effect")] = np.empty(
                resamples, dtype=np.float64
            )
            draws[(comparator, metric, "relative_gain")] = np.empty(
                resamples, dtype=np.float64
            )

    write_at = 0
    group_probabilities = np.full(len(groups), 1.0 / len(groups))
    checkpoint_probabilities = np.full(
        len(checkpoints), 1.0 / len(checkpoints)
    )
    while write_at < resamples:
        count = min(batch_size, resamples - write_at)
        group_multiplicity = rng.multinomial(
            len(groups), group_probabilities, size=count
        ).astype(np.float64)
        checkpoint_multiplicity = rng.multinomial(
            len(checkpoints), checkpoint_probabilities, size=count
        ).astype(np.float64)
        weighted_n = np.einsum(
            "bg,vgk,bk->bv",
            group_multiplicity,
            n,
            checkpoint_multiplicity,
            optimize=True,
        )
        if np.any(weighted_n <= 0):
            raise RuntimeError("Bootstrap draw has no observations")
        for metric, numerator in (("mse", sse), ("mae", sae)):
            weighted_error = np.einsum(
                "bg,vgk,bk->bv",
                group_multiplicity,
                numerator,
                checkpoint_multiplicity,
                optimize=True,
            )
            sampled = weighted_error / weighted_n
            sampled_baseline = sampled[:, baseline_index]
            for comparator in comparators:
                sampled_candidate = sampled[:, variant_index[comparator]]
                effect = sampled_candidate - sampled_baseline
                relative_gain = (
                    sampled_baseline - sampled_candidate
                ) / sampled_baseline
                draws[(comparator, metric, "effect")][
                    write_at : write_at + count
                ] = effect
                draws[(comparator, metric, "relative_gain")][
                    write_at : write_at + count
                ] = relative_gain
        write_at += count

    output: list[dict[str, Any]] = []
    for comparator in comparators:
        for metric in ("mse", "mae"):
            baseline_point = float(point[baseline][metric])
            candidate_point = float(point[comparator][metric])
            effect_point = candidate_point - baseline_point
            relative_point = (baseline_point - candidate_point) / baseline_point
            effect_draws = draws[(comparator, metric, "effect")]
            relative_draws = draws[(comparator, metric, "relative_gain")]
            output.append(
                {
                    "baseline": baseline,
                    "comparator": comparator,
                    "metric": metric,
                    "baseline_point": baseline_point,
                    "candidate_point": candidate_point,
                    "effect_candidate_minus_baseline": effect_point,
                    "effect_ci_low": float(np.quantile(effect_draws, 0.025)),
                    "effect_ci_high": float(np.quantile(effect_draws, 0.975)),
                    "relative_gain": relative_point,
                    "relative_gain_ci_low": float(
                        np.quantile(relative_draws, 0.025)
                    ),
                    "relative_gain_ci_high": float(
                        np.quantile(relative_draws, 0.975)
                    ),
                    "one_sided_p_candidate_not_better": float(
                        (1 + np.count_nonzero(effect_draws >= 0))
                        / (resamples + 1)
                    ),
                    "resamples": int(resamples),
                    "random_seed": int(random_seed),
                    "groups": len(groups),
                    "checkpoints": len(checkpoints),
                    "paired_multiplicities": True,
                }
            )
    return output


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("p-values must be a finite one-dimensional sequence")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def classify_safety(
    *,
    gain_ci_low: float,
    gain_ci_high: float,
    harm_margin: float,
) -> str:
    """Classify harm without conflating uncertainty with demonstrated harm."""

    if gain_ci_high < -abs(harm_margin):
        return "harmful"
    if gain_ci_low < -abs(harm_margin):
        return "safety-inconclusive"
    return "not-harmful"
