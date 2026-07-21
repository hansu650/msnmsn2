from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .ridge import require_finite


@dataclass(frozen=True)
class ShuffleAudit:
    kind: str
    descriptor_sha256: str
    indices_sha256: str
    rows: int
    channels: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "descriptor_sha256": self.descriptor_sha256,
            "indices_sha256": self.indices_sha256,
            "rows": self.rows,
            "channels": self.channels,
        }


def _canonical_ids(row_ids: Sequence[object], rows: int) -> tuple[str, ...]:
    if len(row_ids) != rows:
        raise ValueError("row_ids length does not match tensor rows")
    values = tuple(str(value) for value in row_ids)
    if len(set(values)) != len(values):
        raise ValueError("row_ids must be unique for order-independent shuffling")
    return values


def deterministic_permutation(
    row_ids: Sequence[object],
    descriptor: str,
) -> Tensor:
    """Return a one-to-one row permutation invariant to input row order."""

    ids = _canonical_ids(row_ids, len(row_ids))
    if not descriptor:
        raise ValueError("descriptor must be non-empty")
    destination = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(
            f"{descriptor}|destination|{ids[index]}".encode("utf-8")
        ).digest(),
    )
    sources = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(
            f"{descriptor}|source|{ids[index]}".encode("utf-8")
        ).digest(),
    )
    permutation = torch.empty(len(ids), dtype=torch.int64)
    for destination_index, source_index in zip(destination, sources):
        permutation[destination_index] = source_index
    return permutation


def _audit(kind: str, descriptor: str, permutations: Tensor) -> ShuffleAudit:
    descriptor_hash = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
    indices_hash = hashlib.sha256(
        permutations.contiguous().cpu().numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return ShuffleAudit(
        kind=kind,
        descriptor_sha256=descriptor_hash,
        indices_sha256=indices_hash,
        rows=int(permutations.shape[-1]),
        channels=int(permutations.shape[0]),
    )


def shuffle_cross_forecasts(
    source_forecasts: Tensor,
    row_ids: Sequence[object],
    *,
    descriptor: str,
) -> tuple[Tensor, ShuffleAudit, Tensor]:
    if source_forecasts.ndim != 3:
        raise ValueError("source_forecasts must have shape [B,H,C]")
    require_finite("source_forecasts", source_forecasts)
    rows, _, channels = source_forecasts.shape
    ids = _canonical_ids(row_ids, rows)
    output = source_forecasts.clone()
    permutations = torch.empty(channels, rows, dtype=torch.int64)
    for source_channel in range(channels):
        permutation = deterministic_permutation(
            ids, f"{descriptor}|cross-source={source_channel}"
        )
        permutations[source_channel] = permutation
        output[:, :, source_channel] = source_forecasts[
            permutation.to(source_forecasts.device), :, source_channel
        ]
    return output, _audit("cross_forecast", descriptor, permutations), permutations


def shuffle_latent_features(
    latent_features: Tensor,
    row_ids: Sequence[object],
    *,
    descriptor: str,
) -> tuple[Tensor, ShuffleAudit, Tensor]:
    if latent_features.ndim != 3:
        raise ValueError("latent_features must have shape [B,C,D]")
    require_finite("latent_features", latent_features)
    rows, channels, _ = latent_features.shape
    ids = _canonical_ids(row_ids, rows)
    output = latent_features.clone()
    permutations = torch.empty(channels, rows, dtype=torch.int64)
    for target_channel in range(channels):
        permutation = deterministic_permutation(
            ids, f"{descriptor}|latent-target={target_channel}"
        )
        permutations[target_channel] = permutation
        output[:, target_channel] = latent_features[
            permutation.to(latent_features.device), target_channel
        ]
    return output, _audit("latent", descriptor, permutations), permutations
