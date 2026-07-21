"""Leakage-controlled split and normalization primitives for EdgeTwinCal.

This module deliberately has no dependency on APN or a dataset loader.  Group
membership is decided before any statistic is fitted, and every public manifest
contains only salted identifiers.  Raw patient/station/sample identifiers stay
inside the running process and must never be serialized from these objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SPLIT_NAMES = ("train", "val", "test")
SCALE_FLOOR = 1e-6
LOCKED_GROUP_SALTS = MappingProxyType(
    {
        "p12": "edgetwincal-msn2026-p12-v1",
        "physionet2012": "edgetwincal-msn2026-p12-v1",
        "mimic3": "edgetwincal-msn2026-mimic3-v1",
        "mimic-iii": "edgetwincal-msn2026-mimic3-v1",
        "ushcn": "edgetwincal-msn2026-ushcn-v1",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _canonical_identifier(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("Identifiers must be finite")
    text = str(value)
    if not text:
        raise ValueError("Identifiers must not be empty")
    return text


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalize_asset_hashes(
    data_asset_hashes: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, digest in sorted((data_asset_hashes or {}).items()):
        name_text = str(name)
        digest_text = str(digest).lower()
        if not name_text:
            raise ValueError("Data asset hash names must not be empty")
        if len(digest_text) != 64 or any(c not in "0123456789abcdef" for c in digest_text):
            raise ValueError(f"Invalid SHA256 for data asset {name_text!r}")
        normalized[name_text] = digest_text
    return normalized


def salted_identifier_hash(identifier: Any, salt: str) -> str:
    """Hash the canonical UTF-8 string ``<salt>:<identifier>`` with SHA256."""

    if not salt:
        raise ValueError("A non-empty locked salt is required")
    return _sha256_text(f"{salt}:{_canonical_identifier(identifier)}")


def resolve_locked_group_salt(dataset_id: str, locked_salt: str | None = None) -> str:
    """Resolve a protocol salt, requiring an explicit salt for unknown datasets."""

    key = dataset_id.strip().lower()
    expected = LOCKED_GROUP_SALTS.get(key)
    if locked_salt is None:
        if expected is None:
            raise ValueError(
                f"Dataset {dataset_id!r} has no registered salt; pass locked_salt explicitly"
            )
        return expected
    if not locked_salt:
        raise ValueError("locked_salt must not be empty")
    if expected is not None and locked_salt != expected:
        raise ValueError(
            f"Dataset {dataset_id!r} must use its registered salt {expected!r}"
        )
    return locked_salt


@dataclass(frozen=True)
class GroupSplit:
    """An in-memory raw-ID lookup plus a privacy-safe public manifest."""

    dataset_id: str
    protocol_id: str
    salt: str
    _assignments: Mapping[str, str]
    _public_hashes: Mapping[str, tuple[str, ...]]
    _manifest: Mapping[str, Any]

    def split_for(self, group_id: Any) -> str:
        canonical = _canonical_identifier(group_id)
        try:
            return self._assignments[canonical]
        except KeyError as exc:
            raise KeyError(f"Unknown group identifier: {canonical!r}") from exc

    def indices_for(self, group_ids: Sequence[Any], split: str) -> np.ndarray:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}")
        return np.flatnonzero(
            np.asarray([self.split_for(group_id) == split for group_id in group_ids])
        )

    def public_hash_for(self, group_id: Any) -> str:
        """Return a stable pseudonym without exposing the raw identifier."""

        canonical = _canonical_identifier(group_id)
        if canonical not in self._assignments:
            raise KeyError(f"Unknown group identifier: {canonical!r}")
        return salted_identifier_hash(canonical, self.salt)

    @property
    def split_hash(self) -> str:
        return str(self._manifest["split_hash"])

    @property
    def manifest_hash(self) -> str:
        return str(self._manifest["manifest_hash"])

    def public_manifest(self) -> dict[str, Any]:
        """Return a JSON-safe deep copy containing no raw group identifiers."""

        return json.loads(_canonical_json(dict(self._manifest)))


def hash_group_split(
    group_ids: Iterable[Any],
    *,
    dataset_id: str,
    protocol_id: str,
    locked_salt: str | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    code_hash: str | None = None,
    data_asset_hashes: Mapping[str, str] | None = None,
) -> GroupSplit:
    """Assign unique groups by ascending salted SHA256 with floor 80/10/rest.

    The result is invariant to input order and duplicate rows.  ``ratios`` is
    accepted to make the allocation explicit, but msn2026_v1 intentionally
    permits only the locked 80/10/10 protocol.
    """

    if len(ratios) != 3 or not np.allclose(ratios, (0.8, 0.1, 0.1), atol=0.0):
        raise ValueError("msn2026_v1 requires the locked 80/10/10 ratios")
    if not dataset_id or not protocol_id:
        raise ValueError("dataset_id and protocol_id are required")
    salt = resolve_locked_group_salt(dataset_id, locked_salt)
    canonical_groups = sorted({_canonical_identifier(value) for value in group_ids})
    if not canonical_groups:
        raise ValueError("At least one group identifier is required")

    ordered = sorted(
        ((salted_identifier_hash(group, salt), group) for group in canonical_groups),
        key=lambda pair: pair[0],
    )
    n_groups = len(ordered)
    n_train = math.floor(0.8 * n_groups)
    n_val = math.floor(0.1 * n_groups)
    boundaries = (n_train, n_train + n_val)
    split_pairs = {
        "train": ordered[: boundaries[0]],
        "val": ordered[boundaries[0] : boundaries[1]],
        "test": ordered[boundaries[1] :],
    }
    assignments = {
        raw_group: split
        for split, pairs in split_pairs.items()
        for _, raw_group in pairs
    }
    public_hashes = {
        split: tuple(public_hash for public_hash, _ in pairs)
        for split, pairs in split_pairs.items()
    }
    split_payload = {split: list(public_hashes[split]) for split in SPLIT_NAMES}
    split_hash = _sha256_object(split_payload)
    asset_hashes = _normalize_asset_hashes(data_asset_hashes)
    base_manifest: dict[str, Any] = {
        "schema_version": "edgetwincal.protocol.split.v1",
        "dataset_id": dataset_id,
        "protocol_id": protocol_id,
        "allocation": "ascending_sha256_floor_80_10_remainder",
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "group_hash_salt": salt,
        "group_counts": {split: len(public_hashes[split]) for split in SPLIT_NAMES},
        "public_group_hashes": split_payload,
        "group_id_hash": _sha256_object(sorted(hash_ for values in public_hashes.values() for hash_ in values)),
        "split_hash": split_hash,
        "code_hash": (code_hash or _module_sha256()).lower(),
        "data_asset_hashes": asset_hashes,
        "data_asset_hash": _sha256_object(asset_hashes),
    }
    base_manifest["manifest_hash"] = _sha256_object(base_manifest)
    return GroupSplit(
        dataset_id=dataset_id,
        protocol_id=protocol_id,
        salt=salt,
        _assignments=MappingProxyType(assignments),
        _public_hashes=MappingProxyType(public_hashes),
        _manifest=MappingProxyType(base_manifest),
    )


def _normalize_axis_index(axis: int, ndim: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Axis {axis} is out of bounds for an array of dimension {ndim}")
    return normalized


def _feature_view(array: np.ndarray, feature_axis: int) -> tuple[np.ndarray, int]:
    if array.ndim == 0:
        raise ValueError("Values must have a sample dimension")
    if array.ndim == 1:
        return array.reshape(array.shape[0], 1), 0
    axis = _normalize_axis_index(feature_axis, array.ndim)
    if axis == 0:
        raise ValueError("feature_axis cannot be the sample axis")
    return np.moveaxis(array, axis, -1), axis


@dataclass(frozen=True)
class ObservedNormalizer:
    """Feature-wise train-only normalizer with an auditable public manifest."""

    mean: np.ndarray
    scale: np.ndarray
    observed_count: np.ndarray
    feature_axis: int
    scale_floor: float
    _manifest: Mapping[str, Any]

    @property
    def fit_id_hash(self) -> str:
        return str(self._manifest["fit_id_hash"])

    @property
    def manifest_hash(self) -> str:
        return str(self._manifest["manifest_hash"])

    def public_manifest(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._manifest)))

    def _broadcast_stats(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if values.ndim == 1:
            if self.mean.size != 1:
                raise ValueError("Feature count mismatch")
            return self.mean.reshape(()), self.scale.reshape(())
        axis = _normalize_axis_index(self.feature_axis, values.ndim)
        if axis == 0 or values.shape[axis] != self.mean.size:
            raise ValueError("Feature count or axis mismatch")
        shape = [1] * values.ndim
        shape[axis] = self.mean.size
        return self.mean.reshape(shape), self.scale.reshape(shape)

    @staticmethod
    def _checked_mask(values: np.ndarray, observed_mask: Any | None) -> np.ndarray | None:
        if observed_mask is None:
            if not np.isfinite(values).all():
                raise ValueError("Values contain NaN or infinity without an observation mask")
            return None
        try:
            mask = np.broadcast_to(np.asarray(observed_mask, dtype=bool), values.shape)
        except ValueError as exc:
            raise ValueError("observed_mask is not broadcastable to values") from exc
        if np.any(mask & ~np.isfinite(values)):
            raise ValueError("Observed values contain NaN or infinity")
        return mask

    def apply(
        self,
        values: Any,
        observed_mask: Any | None = None,
        *,
        missing_fill: float = 0.0,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        mask = self._checked_mask(array, observed_mask)
        mean, scale = self._broadcast_stats(array)
        transformed = (array - mean) / scale
        if mask is not None:
            transformed = np.where(mask, transformed, float(missing_fill))
        if not np.isfinite(transformed).all():
            raise ValueError("Normalization produced non-finite values")
        return transformed

    transform = apply

    def inverse(
        self,
        normalized: Any,
        observed_mask: Any | None = None,
        *,
        missing_fill: float = 0.0,
    ) -> np.ndarray:
        array = np.asarray(normalized, dtype=np.float64)
        mask = self._checked_mask(array, observed_mask)
        mean, scale = self._broadcast_stats(array)
        restored = array * scale + mean
        if mask is not None:
            restored = np.where(mask, restored, float(missing_fill))
        if not np.isfinite(restored).all():
            raise ValueError("Inverse normalization produced non-finite values")
        return restored

    inverse_transform = inverse


def fit_train_normalizer(
    values: Any,
    observed_mask: Any,
    *,
    sample_ids: Sequence[Any],
    group_ids: Sequence[Any],
    split: GroupSplit,
    feature_axis: int = -1,
    scale_floor: float = SCALE_FLOOR,
    code_hash: str | None = None,
    data_asset_hashes: Mapping[str, str] | None = None,
) -> ObservedNormalizer:
    """Fit feature statistics using only observed rows assigned to ``train``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("Values must have a sample dimension")
    n_samples = array.shape[0]
    if len(sample_ids) != n_samples or len(group_ids) != n_samples:
        raise ValueError("sample_ids and group_ids must match the sample dimension")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    try:
        mask = np.broadcast_to(np.asarray(observed_mask, dtype=bool), array.shape)
    except ValueError as exc:
        raise ValueError("observed_mask is not broadcastable to values") from exc

    train_rows = np.asarray([split.split_for(group_id) == "train" for group_id in group_ids])
    if not train_rows.any():
        raise ValueError("The training split contains no samples")
    moved, normalized_feature_axis = _feature_view(array, feature_axis)
    moved_mask, _ = _feature_view(mask, feature_axis)
    train_values = moved[train_rows].reshape(-1, moved.shape[-1])
    train_mask = moved_mask[train_rows].reshape(-1, moved.shape[-1])
    if np.any(train_mask & ~np.isfinite(train_values)):
        raise ValueError("Observed training values contain NaN or infinity")

    feature_count = train_values.shape[-1]
    means = np.zeros(feature_count, dtype=np.float64)
    scales = np.ones(feature_count, dtype=np.float64)
    counts = train_mask.sum(axis=0, dtype=np.int64)
    for feature in range(feature_count):
        observed = train_values[train_mask[:, feature], feature]
        if observed.size == 0:
            continue
        means[feature] = float(observed.mean(dtype=np.float64))
        raw_scale = float(observed.std(ddof=0, dtype=np.float64))
        scales[feature] = max(raw_scale, float(scale_floor))

    sample_salt = f"{split.salt}:fit-sample-v1"
    fit_hashes_all = sorted(
        salted_identifier_hash(sample_ids[index], sample_salt)
        for index in np.flatnonzero(train_rows)
    )
    fit_hashes_unique = sorted(set(fit_hashes_all))
    asset_hashes = _normalize_asset_hashes(data_asset_hashes)
    base_manifest: dict[str, Any] = {
        "schema_version": "edgetwincal.protocol.normalizer.v1",
        "dataset_id": split.dataset_id,
        "protocol_id": split.protocol_id,
        "fit_split": "train",
        "fit_sample_count": int(train_rows.sum()),
        "fit_id_hashes": fit_hashes_unique,
        "fit_id_hash": _sha256_object(fit_hashes_all),
        "split_hash": split.split_hash,
        "split_manifest_hash": split.manifest_hash,
        "feature_axis": int(normalized_feature_axis),
        "feature_count": int(feature_count),
        "observed_count": counts.astype(int).tolist(),
        "mean": means.tolist(),
        "scale": scales.tolist(),
        "scale_floor": float(scale_floor),
        "no_observation_features": np.flatnonzero(counts == 0).astype(int).tolist(),
        "code_hash": (code_hash or _module_sha256()).lower(),
        "data_asset_hashes": asset_hashes,
        "data_asset_hash": _sha256_object(asset_hashes),
    }
    base_manifest["manifest_hash"] = _sha256_object(base_manifest)
    means.setflags(write=False)
    scales.setflags(write=False)
    counts.setflags(write=False)
    return ObservedNormalizer(
        mean=means,
        scale=scales,
        observed_count=counts,
        feature_axis=normalized_feature_axis,
        scale_floor=float(scale_floor),
        _manifest=MappingProxyType(base_manifest),
    )


def audit_official_fold_station_overlap(
    train_station_ids: Iterable[Any],
    val_station_ids: Iterable[Any],
    test_station_ids: Iterable[Any],
    *,
    dataset_id: str = "ushcn",
    locked_salt: str | None = None,
    code_hash: str | None = None,
    data_asset_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit an official station fold and return privacy-safe overlap evidence."""

    salt = resolve_locked_group_salt(dataset_id, locked_salt)
    raw_sets = {
        "train": {_canonical_identifier(value) for value in train_station_ids},
        "val": {_canonical_identifier(value) for value in val_station_ids},
        "test": {_canonical_identifier(value) for value in test_station_ids},
    }
    public_sets = {
        split: {salted_identifier_hash(value, salt) for value in values}
        for split, values in raw_sets.items()
    }
    pair_names = (("train", "val"), ("train", "test"), ("val", "test"))
    overlaps = {
        f"{left}_{right}": sorted(public_sets[left] & public_sets[right])
        for left, right in pair_names
    }
    disjoint = all(not values for values in overlaps.values())
    base: dict[str, Any] = {
        "schema_version": "edgetwincal.protocol.station_overlap.v1",
        "dataset_id": dataset_id,
        "group_hash_salt": salt,
        "group_counts": {split: len(public_sets[split]) for split in SPLIT_NAMES},
        "public_station_hashes": {
            split: sorted(public_sets[split]) for split in SPLIT_NAMES
        },
        "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "overlap_public_hashes": overlaps,
        "is_group_disjoint": disjoint,
        "decision": "keep_official_fold" if disjoint else "hash_repair_required",
        "code_hash": (code_hash or _module_sha256()).lower(),
        "data_asset_hashes": _normalize_asset_hashes(data_asset_hashes),
    }
    base["audit_hash"] = _sha256_object(base)
    return base


__all__ = [
    "GroupSplit",
    "LOCKED_GROUP_SALTS",
    "ObservedNormalizer",
    "SCALE_FLOOR",
    "audit_official_fold_station_overlap",
    "fit_train_normalizer",
    "hash_group_split",
    "resolve_locked_group_salt",
    "salted_identifier_hash",
]
