"""Leakage-controlled USHCN preparation for EdgeTwinCal.

The released APN task uses fold 0 from ``USHCN_DeBrouwer2019`` and normalizes
time by the maximum timestamp in the protocol frame.  This project-owned layer
keeps that forecasting contract while making station isolation and value
normalization auditable:

* audit the official train/validation/test station keys;
* retain fold 0 when its station sets are disjoint, otherwise repair it with
  the locked salted-SHA256 80/10/10 station assignment;
* fit feature statistics from observed training values only; and
* construct the test partition only after a frozen-ledger token is supplied.

Importing this module never imports APN, instantiates a TSDM task, reads a data
asset, or constructs a test dataset.  Public manifests contain only salted
station hashes and aggregate counts; raw station identifiers remain in private
in-memory mappings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .protocol import (
    GroupSplit,
    ObservedNormalizer,
    SPLIT_NAMES,
    audit_official_fold_station_overlap,
    fit_train_normalizer,
    hash_group_split,
    resolve_locked_group_salt,
    salted_identifier_hash,
)


USHCN_DATASET_ID = "ushcn"
USHCN_PROTOCOL_ID = "ushcn_leakage_controlled_msn2026_v1"
USHCN_OFFICIAL_FOLD = 0
USHCN_SEQUENCE_LENGTH = 150
USHCN_OBSERVATION_CUTOFF = 149.5
USHCN_PREDICTION_STEPS = 3
USHCN_CHANNELS = 5
_PARTITIONS = tuple(SPLIT_NAMES)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _module_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("USHCN station identifiers must be finite")
    result = str(value)
    if not result:
        raise ValueError("USHCN station identifiers must not be empty")
    return result


def _normalise_asset_hashes(
    data_asset_hashes: Mapping[str, str] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_name, raw_digest in sorted((data_asset_hashes or {}).items()):
        name = str(raw_name)
        digest = str(raw_digest).lower()
        if not name:
            raise ValueError("Data asset hash names must not be empty")
        if not _is_sha256(digest):
            raise ValueError(f"Invalid SHA256 for data asset {name!r}")
        result[name] = digest
    return result


def _canonical_official_keys(
    official_split_keys: Mapping[str, Iterable[Any]],
) -> dict[str, tuple[str, ...]]:
    if set(official_split_keys) != set(_PARTITIONS):
        raise ValueError("official_split_keys must contain exactly train, val, and test")
    canonical: dict[str, tuple[str, ...]] = {}
    for partition in _PARTITIONS:
        identifiers = tuple(sorted({_canonical_id(value) for value in official_split_keys[partition]}))
        if not identifiers:
            raise ValueError(f"Official USHCN {partition} station keys must not be empty")
        canonical[partition] = identifiers
    return canonical


def _station_split(
    station_ids: Sequence[str],
    official_keys: Mapping[str, tuple[str, ...]],
    audit: Mapping[str, Any],
    *,
    code_hash: str,
    data_asset_hashes: Mapping[str, str],
) -> GroupSplit:
    """Build one privacy-safe split manifest after the official-fold audit."""

    salt = resolve_locked_group_salt(USHCN_DATASET_ID)
    if bool(audit["is_group_disjoint"]):
        assignments = {
            station: partition
            for partition in _PARTITIONS
            for station in official_keys[partition]
        }
        allocation = "official_fold_0_verified_group_disjoint"
        repaired = False
    else:
        repaired_split = hash_group_split(
            station_ids,
            dataset_id=USHCN_DATASET_ID,
            protocol_id=USHCN_PROTOCOL_ID,
            code_hash=code_hash,
            data_asset_hashes=data_asset_hashes,
        )
        assignments = {station: repaired_split.split_for(station) for station in station_ids}
        allocation = "ascending_sha256_floor_80_10_remainder"
        repaired = True

    public_hashes = {
        partition: tuple(
            sorted(
                salted_identifier_hash(station, salt)
                for station, assigned in assignments.items()
                if assigned == partition
            )
        )
        for partition in _PARTITIONS
    }
    split_payload = {
        partition: list(public_hashes[partition]) for partition in _PARTITIONS
    }
    base: dict[str, Any] = {
        "schema_version": "edgetwincal.ushcn.split.v1",
        "dataset_id": USHCN_DATASET_ID,
        "protocol_id": USHCN_PROTOCOL_ID,
        "official_fold": USHCN_OFFICIAL_FOLD,
        "allocation": allocation,
        "official_fold_audit_hash": str(audit["audit_hash"]),
        "official_fold_decision": str(audit["decision"]),
        "deviation_from_official_fold": repaired,
        "repair_reason": (
            "official_station_overlap" if repaired else "not_applicable"
        ),
        "ratios": (
            {"train": 0.8, "val": 0.1, "test": 0.1}
            if repaired
            else None
        ),
        "group_hash_salt": salt,
        "group_counts": {
            partition: len(public_hashes[partition]) for partition in _PARTITIONS
        },
        "public_group_hashes": split_payload,
        "group_id_hash": _object_hash(
            sorted(value for values in public_hashes.values() for value in values)
        ),
        "split_hash": _object_hash(split_payload),
        "code_hash": code_hash,
        "data_asset_hashes": dict(data_asset_hashes),
        "data_asset_hash": _object_hash(dict(data_asset_hashes)),
    }
    base["manifest_hash"] = _object_hash(base)
    return GroupSplit(
        dataset_id=USHCN_DATASET_ID,
        protocol_id=USHCN_PROTOCOL_ID,
        salt=salt,
        _assignments=MappingProxyType(dict(assignments)),
        _public_hashes=MappingProxyType(public_hashes),
        _manifest=MappingProxyType(base),
    )


@dataclass(frozen=True)
class USHCNTimeScale:
    """APN-compatible time transform, frozen before any partition is built."""

    raw_time_max: float
    sequence_length: int = USHCN_SEQUENCE_LENGTH
    observation_cutoff: float = USHCN_OBSERVATION_CUTOFF
    prediction_steps: int = USHCN_PREDICTION_STEPS

    def __post_init__(self) -> None:
        if not math.isfinite(self.raw_time_max) or self.raw_time_max <= self.observation_cutoff:
            raise ValueError("USHCN raw_time_max must be finite and greater than 149.5")
        if self.sequence_length != USHCN_SEQUENCE_LENGTH:
            raise ValueError("The msn2026_v1 USHCN sequence length is locked to 150")
        if self.observation_cutoff != USHCN_OBSERVATION_CUTOFF:
            raise ValueError("The msn2026_v1 USHCN cutoff is locked to 149.5")
        if self.prediction_steps != USHCN_PREDICTION_STEPS:
            raise ValueError("The msn2026_v1 USHCN horizon is locked to 3 steps")

    @property
    def observation_time(self) -> float:
        return self.observation_cutoff / self.raw_time_max

    def encode(self, times: Sequence[Any] | np.ndarray) -> np.ndarray:
        values = np.asarray(times, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("USHCN times must be one-dimensional")
        if not np.isfinite(values).all():
            raise ValueError("USHCN times must be finite")
        if np.any(values < 0.0) or np.any(values > self.raw_time_max):
            raise ValueError("USHCN times must lie in the frozen [0, raw_time_max] interval")
        return values / self.raw_time_max

    @property
    def manifest_hash(self) -> str:
        return str(self.public_manifest()["manifest_hash"])

    def public_manifest(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "schema_version": "edgetwincal.ushcn.time_scale.v1",
            "kind": "apn_task_global_time_max_frozen_before_partition_construction",
            "formula": "encoded_time=raw_time/raw_time_max",
            "raw_time_max": self.raw_time_max,
            "sequence_length": self.sequence_length,
            "observation_cutoff": self.observation_cutoff,
            "observation_time": self.observation_time,
            "prediction_steps": self.prediction_steps,
        }
        base["manifest_hash"] = _object_hash(base)
        return base


@dataclass(frozen=True)
class USHCNColumnStandardizer:
    """Column-aware wrapper around the generic train-observed normalizer."""

    columns: tuple[str, ...]
    normalizer: ObservedNormalizer = field(repr=False)
    _manifest: Mapping[str, Any] = field(repr=False, default_factory=dict)

    @property
    def mean(self) -> np.ndarray:
        return self.normalizer.mean

    @property
    def scale(self) -> np.ndarray:
        return self.normalizer.scale

    @property
    def manifest_hash(self) -> str:
        return str(self._manifest["manifest_hash"])

    def public_manifest(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._manifest)))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(map(str, frame.columns)) != self.columns:
            raise ValueError("USHCN feature columns or order differ from the frozen normalizer")
        values = frame.to_numpy(dtype=np.float64, copy=True)
        if np.isinf(values).any():
            raise ValueError("USHCN values must not contain infinity")
        transformed = (values - self.mean.reshape(1, -1)) / self.scale.reshape(1, -1)
        transformed[np.isnan(values)] = np.nan
        return pd.DataFrame(transformed, index=frame.index.copy(), columns=frame.columns.copy())


class USHCNInputs(NamedTuple):
    """Input tuple consumed by APN's USHCN collator."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class USHCNSample(NamedTuple):
    """APN-compatible sample carrying only a pseudonymous integer key."""

    key: int
    inputs: USHCNInputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]


class StrictUSHCNTaskDataset(Dataset[USHCNSample]):
    """Project-owned USHCN task dataset matching APN slicing semantics."""

    def __init__(
        self,
        tensors: Sequence[tuple[Tensor, Tensor]],
        *,
        observation_time: float,
        prediction_steps: int,
        sample_keys: Sequence[int],
        manifest: Mapping[str, Any],
    ) -> None:
        if len(tensors) != len(sample_keys):
            raise ValueError("Each USHCN tensor pair must have one pseudonymous sample key")
        if prediction_steps != USHCN_PREDICTION_STEPS:
            raise ValueError("Strict USHCN prediction_steps must be 3")
        self._tensors = tuple((time, values) for time, values in tensors)
        self._sample_keys = tuple(int(key) for key in sample_keys)
        self.observation_time = float(observation_time)
        self.prediction_steps = int(prediction_steps)
        self._manifest = MappingProxyType(dict(manifest))

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, index: int) -> USHCNSample:
        time, values = self._tensors[index]
        first_target = int(torch.count_nonzero(time <= self.observation_time).item())
        history_slice = slice(0, first_target)
        target_slice = slice(first_target, first_target + self.prediction_steps)
        return USHCNSample(
            key=self._sample_keys[index],
            inputs=USHCNInputs(
                time[history_slice],
                values[history_slice],
                time[target_slice],
            ),
            targets=values[target_slice],
            originals=(time, values),
        )

    def public_manifest(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._manifest)))


@dataclass(frozen=True)
class FrozenUSHCNTestLedgerToken:
    """Proof that the fold audit, split, normalizer, and time scale are frozen."""

    dataset_id: str
    protocol_id: str
    registry_hash: str
    official_fold_audit_hash: str
    split_manifest_hash: str
    normalization_manifest_hash: str
    time_scale_manifest_hash: str
    state: str
    token_hash: str

    @classmethod
    def issue(
        cls,
        protocol: "PreparedStrictUSHCN",
        *,
        registry_hash: str,
        state: str = "frozen",
    ) -> "FrozenUSHCNTestLedgerToken":
        registry_hash = registry_hash.lower()
        if not _is_sha256(registry_hash):
            raise ValueError("registry_hash must be a lowercase SHA256")
        if state != "frozen":
            raise ValueError("A USHCN test token can only be issued for a frozen ledger")
        payload = {
            "schema_version": "edgetwincal.ushcn.test_ledger_token.v1",
            "dataset_id": protocol.dataset_id,
            "protocol_id": protocol.protocol_id,
            "registry_hash": registry_hash,
            "official_fold_audit_hash": str(protocol.official_fold_audit["audit_hash"]),
            "split_manifest_hash": protocol.split.manifest_hash,
            "normalization_manifest_hash": protocol.normalizer.manifest_hash,
            "time_scale_manifest_hash": protocol.time_scale.manifest_hash,
            "state": state,
        }
        return cls(
            dataset_id=protocol.dataset_id,
            protocol_id=protocol.protocol_id,
            registry_hash=registry_hash,
            official_fold_audit_hash=payload["official_fold_audit_hash"],
            split_manifest_hash=protocol.split.manifest_hash,
            normalization_manifest_hash=protocol.normalizer.manifest_hash,
            time_scale_manifest_hash=protocol.time_scale.manifest_hash,
            state=state,
            token_hash=_object_hash(payload),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "edgetwincal.ushcn.test_ledger_token.v1",
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "registry_hash": self.registry_hash,
            "official_fold_audit_hash": self.official_fold_audit_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "normalization_manifest_hash": self.normalization_manifest_hash,
            "time_scale_manifest_hash": self.time_scale_manifest_hash,
            "state": self.state,
        }

    def validate_for(self, protocol: "PreparedStrictUSHCN") -> None:
        if self.state != "frozen" or _object_hash(self._payload()) != self.token_hash:
            raise PermissionError("USHCN test ledger token is not a valid frozen token")
        expected = (
            protocol.dataset_id,
            protocol.protocol_id,
            str(protocol.official_fold_audit["audit_hash"]),
            protocol.split.manifest_hash,
            protocol.normalizer.manifest_hash,
            protocol.time_scale.manifest_hash,
        )
        actual = (
            self.dataset_id,
            self.protocol_id,
            self.official_fold_audit_hash,
            self.split_manifest_hash,
            self.normalization_manifest_hash,
            self.time_scale_manifest_hash,
        )
        if actual != expected:
            raise PermissionError("USHCN test ledger token does not match this frozen protocol")

    def public_manifest(self) -> dict[str, Any]:
        return self._payload() | {"token_hash": self.token_hash}


@dataclass(frozen=True)
class PreparedStrictUSHCN:
    """Frozen strict state; partition values are accessed only on construction."""

    dataset_id: str
    protocol_id: str
    official_fold_audit: Mapping[str, Any]
    split: GroupSplit = field(repr=False)
    normalizer: USHCNColumnStandardizer = field(repr=False)
    time_scale: USHCNTimeScale
    _raw_frame: pd.DataFrame = field(repr=False, compare=False)
    _raw_ids: Mapping[str, Any] = field(repr=False, compare=False)
    _sample_keys: Mapping[str, int] = field(repr=False, compare=False)

    def public_manifests(self) -> dict[str, Any]:
        return {
            "official_fold_audit": json.loads(
                _canonical_json(dict(self.official_fold_audit))
            ),
            "split": self.split.public_manifest(),
            "normalization": self.normalizer.public_manifest(),
            "time_scale": self.time_scale.public_manifest(),
        }

    def _require_partition_access(
        self,
        partition: str,
        ledger_token: FrozenUSHCNTestLedgerToken | None,
    ) -> None:
        if partition not in _PARTITIONS:
            raise ValueError(f"Unknown USHCN partition {partition!r}")
        if partition == "test":
            if ledger_token is None:
                raise PermissionError(
                    "Strict USHCN test construction requires a frozen ledger token"
                )
            ledger_token.validate_for(self)

    def build_dataset(
        self,
        partition: str,
        *,
        ledger_token: FrozenUSHCNTestLedgerToken | None = None,
    ) -> StrictUSHCNTaskDataset:
        """Build exactly one partition; test remains inaccessible before freeze."""

        self._require_partition_access(partition, ledger_token)
        canonical_ids = sorted(
            (
                canonical
                for canonical in self._raw_ids
                if self.split.split_for(canonical) == partition
            ),
            key=self.split.public_hash_for,
        )
        tensors: list[tuple[Tensor, Tensor]] = []
        sample_keys: list[int] = []
        public_hashes: list[str] = []
        history_lengths: list[int] = []
        target_lengths: list[int] = []
        for canonical in canonical_ids:
            raw_id = self._raw_ids[canonical]
            station = self._raw_frame.xs(raw_id, level="ID", drop_level=True)
            station = station.sort_index(kind="stable")
            encoded_time = self.time_scale.encode(station.index.to_numpy())
            normalized = self.normalizer.transform(station)
            time_tensor = torch.as_tensor(encoded_time, dtype=torch.float32)
            value_tensor = torch.as_tensor(normalized.to_numpy(), dtype=torch.float32)
            first_target = int(
                torch.count_nonzero(time_tensor <= self.time_scale.observation_time).item()
            )
            tensors.append((time_tensor, value_tensor))
            sample_keys.append(self._sample_keys[canonical])
            public_hashes.append(self.split.public_hash_for(canonical))
            history_lengths.append(first_target)
            target_lengths.append(
                min(self.time_scale.prediction_steps, max(0, len(time_tensor) - first_target))
            )

        manifest_base: dict[str, Any] = {
            "schema_version": "edgetwincal.ushcn.dataset_plan.v1",
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "partition": partition,
            "station_count": len(tensors),
            "public_station_hashes": public_hashes,
            "station_hash": _object_hash(public_hashes),
            "split_manifest_hash": self.split.manifest_hash,
            "normalization_manifest_hash": self.normalizer.manifest_hash,
            "official_fold_audit_hash": str(self.official_fold_audit["audit_hash"]),
            "time_scale": self.time_scale.public_manifest(),
            "feature_count": len(self.normalizer.columns),
            "history_observation_count": {
                "min": min(history_lengths, default=0),
                "max": max(history_lengths, default=0),
            },
            "target_step_count": {
                "min": min(target_lengths, default=0),
                "max": max(target_lengths, default=0),
            },
        }
        manifest_base["manifest_hash"] = _object_hash(manifest_base)
        return StrictUSHCNTaskDataset(
            tensors,
            observation_time=self.time_scale.observation_time,
            prediction_steps=self.time_scale.prediction_steps,
            sample_keys=sample_keys,
            manifest=manifest_base,
        )


def _validate_raw_frame(
    raw_frame: pd.DataFrame,
    *,
    expected_channels: int | None,
) -> pd.DataFrame:
    if not isinstance(raw_frame, pd.DataFrame):
        raise TypeError("raw_frame must be a pandas DataFrame")
    if not isinstance(raw_frame.index, pd.MultiIndex) or raw_frame.index.nlevels != 2:
        raise ValueError("USHCN raw frame must use an (ID, Time) MultiIndex")
    if tuple(raw_frame.index.names) != ("ID", "Time"):
        raise ValueError("USHCN raw frame index levels must be named ID and Time")
    if raw_frame.empty:
        raise ValueError("USHCN raw frame must not be empty")
    if raw_frame.index.has_duplicates:
        raise ValueError("USHCN raw frame must not contain duplicate station/time rows")
    if expected_channels is not None and raw_frame.shape[1] != expected_channels:
        raise ValueError(
            f"Strict USHCN expects {expected_channels} channels, found {raw_frame.shape[1]}"
        )
    if raw_frame.columns.has_duplicates:
        raise ValueError("USHCN feature columns must be unique")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in raw_frame.dtypes):
        raise TypeError("Every USHCN feature column must be numeric")
    values = raw_frame.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise ValueError("USHCN values must not contain infinity")
    times = np.asarray(raw_frame.index.get_level_values("Time"), dtype=np.float64)
    if not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("USHCN times must be finite and non-negative")
    if float(times.max()) <= USHCN_OBSERVATION_CUTOFF:
        raise ValueError("USHCN time range must extend beyond the 149.5 forecast cutoff")
    return raw_frame.sort_index(level=["ID", "Time"], kind="stable").copy(deep=True)


def prepare_strict_ushcn(
    raw_frame: pd.DataFrame,
    official_split_keys: Mapping[str, Iterable[Any]],
    *,
    expected_channels: int | None = USHCN_CHANNELS,
    code_hash: str | None = None,
    data_asset_hashes: Mapping[str, str] | None = None,
    require_train_observation_per_column: bool = True,
) -> PreparedStrictUSHCN:
    """Freeze station split and train-only normalizer without building test.

    ``raw_frame`` is the already-cleaned TSDM USHCN frame.  The caller supplies
    only fold-0 station keys from the official task.  This function performs no
    loader or network operation and does not materialize any partition dataset.
    """

    frame = _validate_raw_frame(raw_frame, expected_channels=expected_channels)
    resolved_code_hash = (code_hash or _module_hash()).lower()
    if not _is_sha256(resolved_code_hash):
        raise ValueError("code_hash must be a lowercase SHA256")
    asset_hashes = _normalise_asset_hashes(data_asset_hashes)

    raw_unique_ids = list(pd.unique(frame.index.get_level_values("ID")))
    canonical_to_raw: dict[str, Any] = {}
    for raw_id in raw_unique_ids:
        canonical = _canonical_id(raw_id)
        if canonical in canonical_to_raw:
            raise ValueError("USHCN station identifiers collide after canonical UTF-8 encoding")
        canonical_to_raw[canonical] = raw_id

    official_keys = _canonical_official_keys(official_split_keys)
    frame_stations = set(canonical_to_raw)
    official_union = set().union(*(set(official_keys[name]) for name in _PARTITIONS))
    if official_union != frame_stations:
        raise ValueError("Official USHCN fold keys must cover exactly the raw-frame stations")

    audit = audit_official_fold_station_overlap(
        official_keys["train"],
        official_keys["val"],
        official_keys["test"],
        dataset_id=USHCN_DATASET_ID,
        code_hash=resolved_code_hash,
        data_asset_hashes=asset_hashes,
    )
    split = _station_split(
        tuple(canonical_to_raw),
        official_keys,
        audit,
        code_hash=resolved_code_hash,
        data_asset_hashes=asset_hashes,
    )

    row_groups = [_canonical_id(value) for value in frame.index.get_level_values("ID")]
    row_samples = [
        f"{group}\x1f{time}\x1f{row}"
        for row, (group, time) in enumerate(
            zip(row_groups, frame.index.get_level_values("Time"), strict=True)
        )
    ]
    values = frame.to_numpy(dtype=np.float64, copy=True)
    observed = np.isfinite(values)
    fitted = fit_train_normalizer(
        values,
        observed,
        sample_ids=row_samples,
        group_ids=row_groups,
        split=split,
        feature_axis=-1,
        code_hash=resolved_code_hash,
        data_asset_hashes=asset_hashes,
    )
    if require_train_observation_per_column and np.any(fitted.observed_count == 0):
        missing = np.flatnonzero(fitted.observed_count == 0).astype(int).tolist()
        raise ValueError(f"USHCN training split has no observations for columns {missing}")

    columns = tuple(map(str, frame.columns))
    normalizer_base: dict[str, Any] = fitted.public_manifest() | {
        "schema_version": "edgetwincal.ushcn.normalizer.v1",
        "columns": list(columns),
        "fit_order": "station_split_before_train_observed_column_fit",
        "missing_policy": "preserve_nan",
        "post_standardization_filter": "none",
    }
    normalizer_base.pop("manifest_hash", None)
    normalizer_base["manifest_hash"] = _object_hash(normalizer_base)
    normalizer = USHCNColumnStandardizer(
        columns=columns,
        normalizer=fitted,
        _manifest=MappingProxyType(normalizer_base),
    )

    time_scale = USHCNTimeScale(
        raw_time_max=float(np.asarray(frame.index.get_level_values("Time"), dtype=np.float64).max())
    )
    ordered_canonical = sorted(canonical_to_raw, key=split.public_hash_for)
    # These consecutive integers are deterministic, exactly representable by
    # APN's float32 sample-ID collator, and disclose no raw station identifier.
    sample_keys = {canonical: rank + 1 for rank, canonical in enumerate(ordered_canonical)}
    return PreparedStrictUSHCN(
        dataset_id=USHCN_DATASET_ID,
        protocol_id=USHCN_PROTOCOL_ID,
        official_fold_audit=MappingProxyType(dict(audit)),
        split=split,
        normalizer=normalizer,
        time_scale=time_scale,
        _raw_frame=frame,
        _raw_ids=MappingProxyType(canonical_to_raw),
        _sample_keys=MappingProxyType(sample_keys),
    )


__all__ = [
    "FrozenUSHCNTestLedgerToken",
    "PreparedStrictUSHCN",
    "StrictUSHCNTaskDataset",
    "USHCN_CHANNELS",
    "USHCN_DATASET_ID",
    "USHCN_OBSERVATION_CUTOFF",
    "USHCN_OFFICIAL_FOLD",
    "USHCN_PREDICTION_STEPS",
    "USHCN_PROTOCOL_ID",
    "USHCN_SEQUENCE_LENGTH",
    "USHCNColumnStandardizer",
    "USHCNInputs",
    "USHCNSample",
    "USHCNTimeScale",
    "prepare_strict_ushcn",
]
