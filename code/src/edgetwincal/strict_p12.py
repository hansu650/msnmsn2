"""Leakage-controlled PhysioNet 2012 preparation for EdgeTwinCal.

The released APN P12 task fits its value standardizer before creating splits.
This project-owned layer makes the opposite order explicit: lock a patient-level
split, fit value statistics from observed training rows only, and build each
``TaskDataset`` partition lazily.  Importing this module never imports APN or
loads data.  The optional APN adapter resolves the vendor class only when the
caller actually asks for it.

The strict protocol fixes the intended 48-hour P12 clock instead of estimating
a time range from any partition.  Integer hours are encoded as ``Time / 48``;
the history cutoff is 35.5 hours and the next three irregular rows are targets,
matching APN's ``seq_len=36`` wrapper and ``TaskDataset`` slicing semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
from types import MappingProxyType
from typing import Any, NamedTuple, Protocol, TypeVar

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .protocol import (
    GroupSplit,
    ObservedNormalizer,
    fit_train_normalizer,
    hash_group_split,
)


P12_DATASET_ID = "p12"
P12_PROTOCOL_ID = "p12_leakage_controlled_msn2026_v1"
P12_HISTORY_HOURS = 36
P12_OBSERVATION_CUTOFF_HOURS = 35.5
P12_PREDICTION_STEPS = 3
P12_CLOCK_HOURS = 48.0
P12_CHANNELS = 36
P12_CLIP_SIGMA = 5.0
P12_EXCLUDED_CONSTANT_COLUMNS = ("MechVent",)
P12_FEATURE_COLUMNS = (
    "pH",
    "PaCO2",
    "PaO2",
    "FiO2",
    "DiasABP",
    "HR",
    "MAP",
    "SysABP",
    "Temp",
    "GCS",
    "Urine",
    "Weight",
    "HCT",
    "BUN",
    "Creatinine",
    "Glucose",
    "HCO3",
    "Mg",
    "Platelets",
    "K",
    "Na",
    "WBC",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "RespRate",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "SaO2",
    "Lactate",
    "Albumin",
    "TroponinT",
    "Cholesterol",
    "TroponinI",
)
P12_RAW_COLUMNS = P12_FEATURE_COLUMNS[:4] + ("MechVent",) + P12_FEATURE_COLUMNS[4:]
_PARTITIONS = ("train", "val", "test")


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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("P12 patient identifiers must be finite")
    result = str(value)
    if not result:
        raise ValueError("P12 patient identifiers must not be empty")
    return result


@dataclass(frozen=True)
class P12TimeScale:
    """Locked, non-fitted time transform for the 48-hour P12 record."""

    clock_hours: float = P12_CLOCK_HOURS
    history_hours: int = P12_HISTORY_HOURS
    observation_cutoff_hours: float = P12_OBSERVATION_CUTOFF_HOURS
    prediction_steps: int = P12_PREDICTION_STEPS

    def __post_init__(self) -> None:
        if self.clock_hours != P12_CLOCK_HOURS:
            raise ValueError("The msn2026_v1 P12 clock is locked to 48 hours")
        if self.history_hours != P12_HISTORY_HOURS:
            raise ValueError("The msn2026_v1 P12 history is locked to 36 hours")
        if self.observation_cutoff_hours != P12_OBSERVATION_CUTOFF_HOURS:
            raise ValueError("The msn2026_v1 P12 cutoff is locked to 35.5 hours")
        if self.prediction_steps != P12_PREDICTION_STEPS:
            raise ValueError("The msn2026_v1 P12 horizon is locked to 3 steps")

    @property
    def observation_time(self) -> float:
        return self.observation_cutoff_hours / self.clock_hours

    def encode(self, times: Sequence[Any] | np.ndarray) -> np.ndarray:
        values = np.asarray(times, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("P12 times must be one-dimensional")
        if not np.isfinite(values).all():
            raise ValueError("P12 times must be finite")
        if np.any(values < 0.0) or np.any(values >= self.clock_hours):
            raise ValueError("P12 times must lie in the locked [0, 48) hour interval")
        return values / self.clock_hours

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "edgetwincal.p12.time_scale.v1",
            "kind": "fixed_protocol_clock_not_fitted",
            "formula": "encoded_time=raw_hour/48",
            "clock_hours": self.clock_hours,
            "history_hours": self.history_hours,
            "observation_cutoff_hours": self.observation_cutoff_hours,
            "observation_time": self.observation_time,
            "prediction_steps": self.prediction_steps,
        }


@dataclass(frozen=True)
class P12ColumnStandardizer:
    """Column-aware wrapper around the generic train-only normalizer."""

    columns: tuple[str, ...]
    normalizer: ObservedNormalizer = field(repr=False)
    clip_sigma: float = P12_CLIP_SIGMA
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
            raise ValueError("P12 feature columns or order differ from the frozen normalizer")
        values = frame.to_numpy(dtype=np.float64, copy=True)
        if np.isinf(values).any():
            raise ValueError("P12 values must not contain infinity")
        transformed = (values - self.mean.reshape(1, -1)) / self.scale.reshape(1, -1)
        transformed[np.isnan(values)] = np.nan
        transformed[(transformed <= -self.clip_sigma) | (transformed >= self.clip_sigma)] = np.nan
        return pd.DataFrame(transformed, index=frame.index.copy(), columns=frame.columns.copy())


class P12Inputs(NamedTuple):
    """Input tuple matching the attributes consumed by APN's P12 collator."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class P12Sample(NamedTuple):
    """Privacy-safe sample matching APN's ``Sample`` attribute contract."""

    key: int
    inputs: P12Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]


_DatasetT = TypeVar("_DatasetT")


class APNTaskDatasetFactory(Protocol[_DatasetT]):
    def __call__(
        self,
        *,
        tensors: list[tuple[Tensor, Tensor]],
        observation_time: float,
        prediction_steps: int,
        idx_list: list[int],
    ) -> _DatasetT: ...


class StrictP12TaskDataset(Dataset[P12Sample]):
    """Project-owned, APN-compatible dataset with no raw patient IDs."""

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
            raise ValueError("Each P12 tensor pair must have one pseudonymous sample key")
        if prediction_steps != P12_PREDICTION_STEPS:
            raise ValueError("Strict P12 prediction_steps must be 3")
        self._tensors = tuple((time, value) for time, value in tensors)
        self._sample_keys = tuple(int(key) for key in sample_keys)
        self.observation_time = float(observation_time)
        self.prediction_steps = int(prediction_steps)
        self._manifest = MappingProxyType(dict(manifest))

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, index: int) -> P12Sample:
        time, values = self._tensors[index]
        first_target = int(torch.count_nonzero(time <= self.observation_time).item())
        history_slice = slice(0, first_target)
        target_slice = slice(first_target, first_target + self.prediction_steps)
        return P12Sample(
            key=self._sample_keys[index],
            inputs=P12Inputs(
                time[history_slice],
                values[history_slice],
                time[target_slice],
            ),
            targets=values[target_slice],
            originals=(time, values),
        )

    def public_manifest(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._manifest)))

    def to_apn_task_dataset(
        self,
        dataset_factory: APNTaskDatasetFactory[_DatasetT] | None = None,
    ) -> _DatasetT:
        """Materialize vendor ``TaskDataset`` only at this explicit call site.

        The caller normally prepares APN's import path first.  Tests and other
        consumers can pass an equivalent factory without importing vendor code.
        """

        if dataset_factory is None:
            module = importlib.import_module("data.dependencies.tsdm.tasks.P12")
            dataset_factory = module.TaskDataset
        return dataset_factory(
            tensors=list(self._tensors),
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
            idx_list=list(self._sample_keys),
        )


@dataclass(frozen=True)
class FrozenP12TestLedgerToken:
    """Deterministic proof that registry, split, and normalizer were frozen."""

    dataset_id: str
    protocol_id: str
    registry_hash: str
    split_manifest_hash: str
    normalization_manifest_hash: str
    state: str
    token_hash: str

    @classmethod
    def issue(
        cls,
        protocol: "PreparedStrictP12",
        *,
        registry_hash: str,
        state: str = "frozen",
    ) -> "FrozenP12TestLedgerToken":
        registry_hash = registry_hash.lower()
        if not _is_sha256(registry_hash):
            raise ValueError("registry_hash must be a lowercase SHA256")
        if state != "frozen":
            raise ValueError("A P12 test token can only be issued for a frozen ledger")
        payload = {
            "schema_version": "edgetwincal.p12.test_ledger_token.v1",
            "dataset_id": protocol.dataset_id,
            "protocol_id": protocol.protocol_id,
            "registry_hash": registry_hash,
            "split_manifest_hash": protocol.split.manifest_hash,
            "normalization_manifest_hash": protocol.normalizer.manifest_hash,
            "state": state,
        }
        return cls(
            dataset_id=protocol.dataset_id,
            protocol_id=protocol.protocol_id,
            registry_hash=registry_hash,
            split_manifest_hash=protocol.split.manifest_hash,
            normalization_manifest_hash=protocol.normalizer.manifest_hash,
            state=state,
            token_hash=_object_hash(payload),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "edgetwincal.p12.test_ledger_token.v1",
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "registry_hash": self.registry_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "normalization_manifest_hash": self.normalization_manifest_hash,
            "state": self.state,
        }

    def validate_for(self, protocol: "PreparedStrictP12") -> None:
        if self.state != "frozen" or _object_hash(self._payload()) != self.token_hash:
            raise PermissionError("P12 test ledger token is not a valid frozen token")
        expected = (
            protocol.dataset_id,
            protocol.protocol_id,
            protocol.split.manifest_hash,
            protocol.normalizer.manifest_hash,
        )
        actual = (
            self.dataset_id,
            self.protocol_id,
            self.split_manifest_hash,
            self.normalization_manifest_hash,
        )
        if actual != expected:
            raise PermissionError("P12 test ledger token does not match this frozen protocol")

    def public_manifest(self) -> dict[str, Any]:
        return self._payload() | {"token_hash": self.token_hash}


@dataclass(frozen=True)
class PreparedStrictP12:
    """Prepared split/normalizer state; partition values remain lazily accessed."""

    dataset_id: str
    protocol_id: str
    split: GroupSplit = field(repr=False)
    normalizer: P12ColumnStandardizer = field(repr=False)
    time_scale: P12TimeScale
    _raw_frame: pd.DataFrame = field(repr=False, compare=False)
    _raw_ids: Mapping[str, Any] = field(repr=False, compare=False)
    _sample_keys: Mapping[str, int] = field(repr=False, compare=False)

    def public_manifests(self) -> dict[str, Any]:
        return {
            "split": self.split.public_manifest(),
            "normalization": self.normalizer.public_manifest(),
            "time_scale": self.time_scale.public_manifest(),
        }

    def _require_partition_access(
        self,
        partition: str,
        ledger_token: FrozenP12TestLedgerToken | None,
    ) -> None:
        if partition not in _PARTITIONS:
            raise ValueError(f"Unknown P12 partition {partition!r}")
        if partition == "test":
            if ledger_token is None:
                raise PermissionError(
                    "Strict P12 test construction requires a frozen ledger token"
                )
            ledger_token.validate_for(self)

    def build_dataset(
        self,
        partition: str,
        *,
        ledger_token: FrozenP12TestLedgerToken | None = None,
    ) -> StrictP12TaskDataset:
        """Build exactly one split; test is inaccessible without a frozen token."""

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
        for canonical in canonical_ids:
            raw_id = self._raw_ids[canonical]
            patient = self._raw_frame.xs(raw_id, level="RecordID", drop_level=True)
            patient = patient.sort_index(kind="stable")
            encoded_time = self.time_scale.encode(patient.index.to_numpy())
            normalized = self.normalizer.transform(patient)
            tensors.append(
                (
                    torch.as_tensor(encoded_time, dtype=torch.float32),
                    torch.as_tensor(normalized.to_numpy(), dtype=torch.float32),
                )
            )
            sample_keys.append(self._sample_keys[canonical])
            public_hashes.append(self.split.public_hash_for(canonical))

        manifest_base: dict[str, Any] = {
            "schema_version": "edgetwincal.p12.dataset_plan.v1",
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "partition": partition,
            "sample_count": len(tensors),
            "public_sample_hashes": public_hashes,
            "sample_hash": _object_hash(public_hashes),
            "split_manifest_hash": self.split.manifest_hash,
            "normalization_manifest_hash": self.normalizer.manifest_hash,
            "time_scale": self.time_scale.public_manifest(),
            "feature_count": len(self.normalizer.columns),
        }
        manifest_base["manifest_hash"] = _object_hash(manifest_base)
        return StrictP12TaskDataset(
            tensors,
            observation_time=self.time_scale.observation_time,
            prediction_steps=self.time_scale.prediction_steps,
            sample_keys=sample_keys,
            manifest=manifest_base,
        )

    def build_apn_dataset(
        self,
        partition: str,
        *,
        ledger_token: FrozenP12TestLedgerToken | None = None,
        dataset_factory: APNTaskDatasetFactory[_DatasetT] | None = None,
    ) -> _DatasetT:
        return self.build_dataset(partition, ledger_token=ledger_token).to_apn_task_dataset(
            dataset_factory
        )


def _validate_raw_frame(
    raw_frame: pd.DataFrame,
    *,
    expected_channels: int | None,
) -> pd.DataFrame:
    if not isinstance(raw_frame, pd.DataFrame):
        raise TypeError("raw_frame must be a pandas DataFrame")
    if not isinstance(raw_frame.index, pd.MultiIndex) or raw_frame.index.nlevels != 2:
        raise ValueError("P12 raw frame must use a (RecordID, Time) MultiIndex")
    if tuple(raw_frame.index.names) != ("RecordID", "Time"):
        raise ValueError("P12 raw frame index levels must be named RecordID and Time")
    if raw_frame.empty:
        raise ValueError("P12 raw frame must not be empty")
    if raw_frame.index.has_duplicates:
        raise ValueError("P12 raw frame must not contain duplicate patient/time rows")
    if raw_frame.columns.has_duplicates:
        raise ValueError("P12 feature columns must be unique")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in raw_frame.dtypes):
        raise TypeError("Every P12 feature column must be numeric")
    values = raw_frame.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise ValueError("P12 values must not contain infinity")
    times = np.asarray(raw_frame.index.get_level_values("Time"), dtype=np.float64)
    P12TimeScale().encode(times)

    # The pinned TSDM source exposes 37 columns. APN's released preprocessing
    # standardizes over the full dataset; MechVent's only observed value is 1,
    # so its zero variance makes the standardized column non-finite and the
    # subsequent all-NaN drop leaves the documented 36-channel tensor. The
    # strict protocol must not rediscover that fact using full-data statistics,
    # therefore the same feature axis is frozen explicitly and audited here.
    columns = tuple(map(str, raw_frame.columns))
    if expected_channels == P12_CHANNELS and columns == P12_RAW_COLUMNS:
        observed_mechvent = raw_frame["MechVent"].dropna().to_numpy(dtype=np.float64)
        if observed_mechvent.size == 0 or not np.all(observed_mechvent == 1.0):
            raise ValueError(
                "P12 MechVent exclusion audit requires non-empty constant observed value 1"
            )
        raw_frame = raw_frame.loc[:, list(P12_FEATURE_COLUMNS)]
        columns = tuple(map(str, raw_frame.columns))

    if expected_channels is not None and raw_frame.shape[1] != expected_channels:
        raise ValueError(
            f"Strict P12 expects {expected_channels} channels, found {raw_frame.shape[1]}"
        )
    if expected_channels == P12_CHANNELS and columns != P12_FEATURE_COLUMNS:
        raise ValueError("Strict P12 columns differ from the frozen 36-channel feature axis")
    return raw_frame.sort_index(level=["RecordID", "Time"], kind="stable").copy(deep=True)


def prepare_strict_p12(
    raw_frame: pd.DataFrame,
    *,
    expected_channels: int | None = P12_CHANNELS,
    code_hash: str | None = None,
    data_asset_hashes: Mapping[str, str] | None = None,
    require_train_observation_per_column: bool = True,
) -> PreparedStrictP12:
    """Lock patient splits, then fit and freeze train-only column statistics.

    ``raw_frame`` is the combined A/B/C time-series frame from TSDM.  This
    function performs no dataset/vendor loading and does not construct any test
    sample.  Passing a frame is intentionally the only data-bearing operation.
    """

    frame = _validate_raw_frame(raw_frame, expected_channels=expected_channels)
    raw_unique_ids = list(pd.unique(frame.index.get_level_values("RecordID")))
    canonical_to_raw: dict[str, Any] = {}
    for raw_id in raw_unique_ids:
        canonical = _canonical_id(raw_id)
        if canonical in canonical_to_raw:
            raise ValueError("P12 patient identifiers collide after canonical UTF-8 encoding")
        canonical_to_raw[canonical] = raw_id

    # Protocol invariant: the complete group assignment is frozen before any
    # value row is selected or any normalization statistic is computed.
    split = hash_group_split(
        canonical_to_raw,
        dataset_id=P12_DATASET_ID,
        protocol_id=P12_PROTOCOL_ID,
        code_hash=code_hash,
        data_asset_hashes=data_asset_hashes,
    )

    row_groups = [_canonical_id(value) for value in frame.index.get_level_values("RecordID")]
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
        code_hash=code_hash,
        data_asset_hashes=data_asset_hashes,
    )
    if require_train_observation_per_column and np.any(fitted.observed_count == 0):
        missing = np.flatnonzero(fitted.observed_count == 0).astype(int).tolist()
        raise ValueError(f"P12 training split has no observations for columns {missing}")

    columns = tuple(map(str, frame.columns))
    normalizer_base: dict[str, Any] = fitted.public_manifest() | {
        "schema_version": "edgetwincal.p12.normalizer.v1",
        "columns": list(columns),
        "source_columns": list(P12_RAW_COLUMNS),
        "excluded_constant_columns": list(P12_EXCLUDED_CONSTANT_COLUMNS),
        "feature_axis_policy": "frozen_to_released_apn_36_channel_axis",
        "fit_order": "patient_split_before_train_only_column_fit",
        "missing_policy": "preserve_nan",
        "post_standardization_filter": "-5 < z < 5",
        "clip_sigma": P12_CLIP_SIGMA,
    }
    normalizer_base.pop("manifest_hash", None)
    normalizer_base["manifest_hash"] = _object_hash(normalizer_base)
    normalizer = P12ColumnStandardizer(
        columns=columns,
        normalizer=fitted,
        _manifest=MappingProxyType(normalizer_base),
    )

    ordered_canonical = sorted(canonical_to_raw, key=split.public_hash_for)
    # Small consecutive integers are deterministic, unique, privacy-safe, and
    # exactly representable by APN's float32 sample_ID collator.
    sample_keys = {canonical: rank + 1 for rank, canonical in enumerate(ordered_canonical)}
    return PreparedStrictP12(
        dataset_id=P12_DATASET_ID,
        protocol_id=P12_PROTOCOL_ID,
        split=split,
        normalizer=normalizer,
        time_scale=P12TimeScale(),
        _raw_frame=frame,
        _raw_ids=MappingProxyType(canonical_to_raw),
        _sample_keys=MappingProxyType(sample_keys),
    )


__all__ = [
    "APNTaskDatasetFactory",
    "FrozenP12TestLedgerToken",
    "P12_CHANNELS",
    "P12_CLOCK_HOURS",
    "P12_EXCLUDED_CONSTANT_COLUMNS",
    "P12_FEATURE_COLUMNS",
    "P12_HISTORY_HOURS",
    "P12_OBSERVATION_CUTOFF_HOURS",
    "P12_PREDICTION_STEPS",
    "P12_RAW_COLUMNS",
    "P12ColumnStandardizer",
    "P12Inputs",
    "P12Sample",
    "P12TimeScale",
    "PreparedStrictP12",
    "StrictP12TaskDataset",
    "prepare_strict_p12",
]
