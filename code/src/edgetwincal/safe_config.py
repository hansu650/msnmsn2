"""Strict immutable configuration for the EdgeTwinCal-Safe campaign."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import PROJECT_ROOT, require_within_root


DEFAULT_SAFE_CONFIG = PROJECT_ROOT / "code" / "configs" / "msn2026" / "safe_v1.json"
SCHEMA_VERSION = "edgetwincal.safe.v1"


class SafeConfigError(ValueError):
    """The Safe protocol is missing, changed, or ambiguous."""


@dataclass(frozen=True)
class SourceSpec:
    landing_url: str
    download_url: str
    filename: str
    expected_sha256: str | None


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    start: datetime
    end: datetime

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp < self.end


@dataclass(frozen=True)
class ValidationGroupSplit:
    group_seconds: int
    hash_name: str
    modulus: int
    val_select_buckets: tuple[int, ...]
    val_safety_buckets: tuple[int, ...]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    title: str
    source: SourceSpec
    value_field: str
    channels: tuple[str | int, ...]
    aggregation: str
    frequency_seconds: int
    history_steps: int
    forecast_steps: int
    stride_steps: int
    interpolation: str
    grouping: str
    id_salt: str
    partitions: tuple[PartitionSpec, ...]
    validation_group_split: ValidationGroupSplit | None

    def partition(self, name: str) -> PartitionSpec:
        lookup = (
            "validation_pool"
            if name in {"val_select", "val_safety"} and self.validation_group_split
            else name
        )
        for partition in self.partitions:
            if partition.name == lookup:
                return partition
        raise KeyError(f"{self.dataset_id} has no partition {name!r}")

    @property
    def test_start(self) -> datetime:
        return self.partition("test").start

    @property
    def test_end(self) -> datetime:
        return self.partition("test").end


@dataclass(frozen=True)
class APNSpec:
    repository: str
    commit: str
    d_model: int
    npatch: int
    te_dim: int
    dropout: float
    batch_size: int
    optimizer: str
    learning_rate: float
    epochs: int
    patience: int
    loss: str


@dataclass(frozen=True)
class SafeExperimentConfig:
    path: Path
    sha256: str
    campaign_id: str
    track: str
    seeds: tuple[int, ...]
    main_variants: tuple[str, ...]
    ablation_variants: tuple[str, ...]
    paths: tuple[tuple[str, str], ...]
    apn: APNSpec
    datasets: tuple[DatasetSpec, ...]
    canonical_payload: bytes

    def dataset(self, dataset_id: str) -> DatasetSpec:
        for spec in self.datasets:
            if spec.dataset_id == dataset_id:
                return spec
        raise KeyError(dataset_id)

    def relative_path(self, name: str) -> Path:
        for key, value in self.paths:
            if key == name:
                return Path(value)
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload.decode("utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    expected, actual = set(keys), set(mapping)
    if actual != expected:
        raise SafeConfigError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise SafeConfigError(f"{label} must be an ISO timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SafeConfigError(f"Invalid {label}: {value!r}") from exc
    if result.tzinfo is not None:
        raise SafeConfigError(f"{label} must be UTC-naive")
    return result


def _source(raw: Mapping[str, Any], label: str) -> SourceSpec:
    _exact(raw, {"landing_url", "download_url", "filename", "expected_sha256"}, label)
    checksum = raw["expected_sha256"]
    if checksum is not None and (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(char not in "0123456789abcdef" for char in checksum)
    ):
        raise SafeConfigError(f"{label}.expected_sha256 is not a lowercase SHA256")
    if any(
        not isinstance(raw[key], str) or not raw[key].startswith("https://")
        for key in ("landing_url", "download_url")
    ):
        raise SafeConfigError(f"{label} URLs must use HTTPS")
    filename = raw["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise SafeConfigError(f"{label}.filename must be a basename")
    return SourceSpec(raw["landing_url"], raw["download_url"], filename, checksum)


def _partitions(raw: Any, label: str) -> tuple[PartitionSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise SafeConfigError(f"{label} must be a non-empty list")
    result: list[PartitionSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise SafeConfigError(f"{label}[{index}] must be an object")
        _exact(item, {"name", "start", "end"}, f"{label}[{index}]")
        start = _timestamp(item["start"], f"{label}[{index}].start")
        end = _timestamp(item["end"], f"{label}[{index}].end")
        if start >= end or (result and result[-1].end != start):
            raise SafeConfigError(f"{label} must be non-empty, contiguous, and ordered")
        result.append(PartitionSpec(str(item["name"]), start, end))
    if len({item.name for item in result}) != len(result):
        raise SafeConfigError(f"{label} contains duplicate names")
    return tuple(result)


def _validation_split(raw: Any, label: str) -> ValidationGroupSplit | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise SafeConfigError(f"{label} must be null or an object")
    _exact(
        raw,
        {"group_seconds", "hash", "modulus", "val_select_buckets", "val_safety_buckets"},
        label,
    )
    result = ValidationGroupSplit(
        int(raw["group_seconds"]),
        str(raw["hash"]),
        int(raw["modulus"]),
        tuple(int(item) for item in raw["val_select_buckets"]),
        tuple(int(item) for item in raw["val_safety_buckets"]),
    )
    select, safety = set(result.val_select_buckets), set(result.val_safety_buckets)
    if (
        result.group_seconds != 10800
        or result.hash_name != "sha256"
        or result.modulus != 2
        or select & safety
        or select | safety != {0, 1}
    ):
        raise SafeConfigError(f"{label} differs from the whole-group SHA256 split")
    return result


def _dataset(dataset_id: str, raw: Mapping[str, Any]) -> DatasetSpec:
    _exact(
        raw,
        {
            "title", "source", "value_field", "channels", "aggregation",
            "frequency_seconds", "history_steps", "forecast_steps", "stride_steps",
            "interpolation", "grouping", "id_salt", "partitions",
            "validation_group_split",
        },
        f"datasets.{dataset_id}",
    )
    channel_raw = raw["channels"]
    if isinstance(channel_raw, list):
        channels: tuple[str | int, ...] = tuple(channel_raw)
    elif isinstance(channel_raw, Mapping):
        _exact(channel_raw, {"integer_range"}, f"datasets.{dataset_id}.channels")
        bounds = channel_raw["integer_range"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise SafeConfigError("integer_range must contain [start, end]")
        channels = tuple(range(int(bounds[0]), int(bounds[1]) + 1))
    else:
        raise SafeConfigError(f"datasets.{dataset_id}.channels is invalid")
    result = DatasetSpec(
        dataset_id, str(raw["title"]), _source(raw["source"], f"datasets.{dataset_id}.source"),
        str(raw["value_field"]), channels, str(raw["aggregation"]),
        int(raw["frequency_seconds"]), int(raw["history_steps"]),
        int(raw["forecast_steps"]), int(raw["stride_steps"]),
        str(raw["interpolation"]), str(raw["grouping"]), str(raw["id_salt"]),
        _partitions(raw["partitions"], f"datasets.{dataset_id}.partitions"),
        _validation_split(
            raw["validation_group_split"],
            f"datasets.{dataset_id}.validation_group_split",
        ),
    )
    if (
        len(set(result.channels)) != len(result.channels)
        or min(
            result.frequency_seconds, result.history_steps,
            result.forecast_steps, result.stride_steps,
        ) <= 0
        or result.interpolation != "none"
    ):
        raise SafeConfigError(f"datasets.{dataset_id} violates the tensor contract")
    return result


_BEIJING_CHANNELS = (
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng",
    "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
)
_BEIJING_PARTITIONS = (
    ("train", "2013-03-01T00:00:00", "2016-01-01T00:00:00"),
    ("val", "2016-01-01T00:00:00", "2016-03-01T00:00:00"),
    ("adapter", "2016-03-01T00:00:00", "2016-05-01T00:00:00"),
    ("val_select", "2016-05-01T00:00:00", "2016-06-01T00:00:00"),
    ("val_safety", "2016-06-01T00:00:00", "2016-09-01T00:00:00"),
    ("test", "2016-09-01T00:00:00", "2017-03-01T00:00:00"),
)
_INTEL_PARTITIONS = (
    ("train", "2004-02-28T00:00:00", "2004-03-16T00:00:00"),
    ("val", "2004-03-16T00:00:00", "2004-03-18T00:00:00"),
    ("adapter", "2004-03-18T00:00:00", "2004-03-22T00:00:00"),
    ("validation_pool", "2004-03-22T00:00:00", "2004-03-27T00:00:00"),
    ("test", "2004-03-27T00:00:00", "2004-04-06T00:00:00"),
)


def _signature(spec: DatasetSpec) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (item.name, item.start.isoformat(), item.end.isoformat())
        for item in spec.partitions
    )


def _validate_frozen(config: SafeExperimentConfig) -> None:
    if (
        config.campaign_id != "edgetwincal_safe_v1"
        or config.track != "Edge Computing, IoT and Digital Twins"
        or config.seeds != (2024, 2025, 2026, 2027, 2028)
        or config.main_variants != ("APN", "Joint", "Full", "Safe")
        or config.ablation_variants
        != ("SafeNoBalance", "SafeNoRobust", "SafeNoBound", "SafeNoGate")
    ):
        raise SafeConfigError("Campaign identity, seed block, or registry changed")
    expected_apn = APNSpec(
        "https://github.com/decisionintelligence/APN.git",
        "f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4",
        24, 12, 8, 0.1, 32, "Adam", 0.01, 200, 10, "masked_mse",
    )
    if config.apn != expected_apn:
        raise SafeConfigError("APN settings are frozen")
    beijing, intel = config.dataset("beijing_air"), config.dataset("intel_lab")
    if (
        beijing.channels != _BEIJING_CHANNELS
        or (beijing.frequency_seconds, beijing.history_steps, beijing.forecast_steps, beijing.stride_steps)
        != (3600, 72, 24, 3)
        or beijing.grouping != "utc_calendar_day"
        or beijing.value_field != "PM2.5"
        or beijing.id_salt != "edgetwincal-safe-msn2026-beijing-air-v1:"
        or _signature(beijing) != _BEIJING_PARTITIONS
        or beijing.validation_group_split is not None
    ):
        raise SafeConfigError("Beijing protocol differs from the frozen design")
    if (
        intel.channels != tuple(range(1, 55))
        or (intel.frequency_seconds, intel.history_steps, intel.forecast_steps, intel.stride_steps)
        != (300, 72, 12, 3)
        or intel.grouping != "utc_3_hour_block"
        or intel.value_field != "temperature"
        or intel.aggregation != "median_5min"
        or intel.id_salt != "edgetwincal-safe-msn2026-intel-lab-v1:"
        or _signature(intel) != _INTEL_PARTITIONS
        or intel.validation_group_split is None
    ):
        raise SafeConfigError("Intel protocol differs from the frozen design")
    raw = config.to_dict()
    expected_auxiliary = {
        "robust_fit": {
            "alpha_latent": [1, 10, 100, 1000, 10000, 100000],
            "alpha_cross": [1, 10, 100, 1000, 10000, 100000],
            "huber_delta": 1.345,
            "max_iterations": 25,
            "tolerance": 1e-8,
            "scale_floor": 1e-6,
            "feature_clip": 5.0,
            "minimum_groups": 20,
            "minimum_rows": 100,
            "rows_per_parameter": 4,
        },
        "safe_envelope": {
            "kappa": [0.25, 0.5, 1.0, 2.0],
            "shrinkage": [0.25, 0.5, 0.75, 1.0],
            "fallback_shrinkage": 0.0,
        },
        "validation_gate": {
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20260721,
            "minimum_groups": 20,
            "minimum_target_cells": 400,
            "apn_harm_ucb": 0.01,
            "joint_macro_harm_ucb": 0.005,
            "minimum_seed_gains": 4,
            "maximum_gain_concentration": 0.25,
        },
        "statistics": {
            "bootstrap_draws": 50000,
            "bootstrap_seed": 20260721,
            "confidence": 0.95,
            "multiple_testing": "holm",
            "joint_noninferiority_margin": 0.001,
        },
    }
    if any(raw[name] != expected for name, expected in expected_auxiliary.items()):
        raise SafeConfigError(
            "Robust, envelope, gate, and statistics values are frozen"
        )



_AUX_KEYS = {
    "robust_fit": {
        "alpha_latent", "alpha_cross", "huber_delta", "max_iterations", "tolerance",
        "scale_floor", "feature_clip", "minimum_groups", "minimum_rows",
        "rows_per_parameter",
    },
    "safe_envelope": {"kappa", "shrinkage", "fallback_shrinkage"},
    "validation_gate": {
        "bootstrap_draws", "bootstrap_seed", "minimum_groups",
        "minimum_target_cells", "apn_harm_ucb", "joint_macro_harm_ucb",
        "minimum_seed_gains", "maximum_gain_concentration",
    },
    "statistics": {
        "bootstrap_draws", "bootstrap_seed", "confidence",
        "multiple_testing", "joint_noninferiority_margin",
    },
}


def load_safe_config(path: str | Path = DEFAULT_SAFE_CONFIG) -> SafeExperimentConfig:
    resolved = require_within_root(path, must_exist=True)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafeConfigError(f"Cannot load Safe config: {resolved}") from exc
    if not isinstance(raw, Mapping):
        raise SafeConfigError("Safe config must be a JSON object")
    _exact(
        raw,
        {
            "schema_version", "campaign", "paths", "apn", "datasets",
            "robust_fit", "safe_envelope", "validation_gate", "statistics",
        },
        "root",
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SafeConfigError(f"Expected schema {SCHEMA_VERSION}")
    campaign = raw["campaign"]
    _exact(
        campaign,
        {
            "id", "track", "seeds", "main_variants", "ablation_variants",
            "test_policy", "old_datasets_policy",
        },
        "campaign",
    )
    if (
        campaign["test_policy"] != "validation_gate_then_once_only"
        or campaign["old_datasets_policy"] != "diagnostic_only"
    ):
        raise SafeConfigError("Test and old-dataset policies are frozen")
    paths = raw["paths"]
    _exact(paths, {"raw", "pretest", "sealed_test", "results", "artifacts", "logs"}, "paths")
    for value in paths.values():
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SafeConfigError(f"Unsafe relative path: {value!r}")
    apn_raw = raw["apn"]
    _exact(
        apn_raw,
        {
            "repository", "commit", "d_model", "npatch", "te_dim", "dropout",
            "batch_size", "optimizer", "learning_rate", "epochs", "patience", "loss",
        },
        "apn",
    )
    for section, keys in _AUX_KEYS.items():
        if not isinstance(raw[section], Mapping):
            raise SafeConfigError(f"{section} must be an object")
        _exact(raw[section], keys, section)
    apn = APNSpec(
        str(apn_raw["repository"]), str(apn_raw["commit"]), int(apn_raw["d_model"]),
        int(apn_raw["npatch"]), int(apn_raw["te_dim"]), float(apn_raw["dropout"]),
        int(apn_raw["batch_size"]), str(apn_raw["optimizer"]),
        float(apn_raw["learning_rate"]), int(apn_raw["epochs"]),
        int(apn_raw["patience"]), str(apn_raw["loss"]),
    )
    dataset_raw = raw["datasets"]
    _exact(dataset_raw, {"beijing_air", "intel_lab"}, "datasets")
    datasets = tuple(
        _dataset(key, dataset_raw[key]) for key in ("beijing_air", "intel_lab")
    )
    canonical = canonical_json_bytes(raw)
    config = SafeExperimentConfig(
        resolved, hashlib.sha256(canonical).hexdigest(), str(campaign["id"]),
        str(campaign["track"]), tuple(int(seed) for seed in campaign["seeds"]),
        tuple(str(item) for item in campaign["main_variants"]),
        tuple(str(item) for item in campaign["ablation_variants"]),
        tuple((str(key), str(value)) for key, value in paths.items()),
        apn, datasets, canonical,
    )
    _validate_frozen(config)
    return config
