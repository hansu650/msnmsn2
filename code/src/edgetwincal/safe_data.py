"""Leakage-resistant data plane for the frozen EdgeTwinCal-Safe holdouts.

Routing decides from timestamps before parsing any test value. Raw test rows are
copied to a physical shard and parsed only after a once-only token is claimed.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import gzip
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import secrets
import statistics
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .paths import PROJECT_ROOT, require_within_root
from .safe_config import DatasetSpec, SafeExperimentConfig, SourceSpec, load_safe_config


@dataclass(frozen=True)
class RawSourceManifest:
    dataset_id: str
    source_relative_path: str
    source_url: str
    landing_url: str
    filename: str
    sha256: str
    size_bytes: int
    expected_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PretestManifest:
    dataset_id: str
    source_sha256: str
    observations_relative_path: str
    sealed_test_relative_path: str
    pretest_rows: int
    sealed_test_rows: int
    observations_sha256: str
    sealed_test_sha256: str
    channel_order: tuple[str | int, ...]
    test_start: str
    test_end: str
    discarded_unusable_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["channel_order"] = list(self.channel_order)
        return result


@dataclass(frozen=True)
class PreparedRows:
    dataset_id: str
    split: str
    channels: tuple[str | int, ...]
    timestamps: tuple[datetime, ...]
    channel_indices: tuple[int, ...]
    values: tuple[float, ...]
    source_sha256: str


@dataclass(frozen=True)
class NormalizerState:
    dataset_id: str
    channel_order: tuple[str | int, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    observed_count: tuple[int, ...]
    fit_split: str
    source_sha256: str
    sha256: str

    def normalize(self, channel_index: int, value: float) -> float:
        return (value - self.mean[channel_index]) / self.scale[channel_index]


@dataclass(frozen=True)
class PretestLoaders:
    train: DataLoader
    val: DataLoader
    adapter: DataLoader
    val_select: DataLoader
    val_safety: DataLoader

    def as_dict(self) -> dict[str, DataLoader]:
        return {
            "train": self.train,
            "val": self.val,
            "adapter": self.adapter,
            "val_select": self.val_select,
            "val_safety": self.val_safety,
        }


@dataclass(frozen=True)
class TestLedger:
    dataset_id: str
    ledger_path: Path
    status: str
    token: str = field(repr=False)


def require_safe_path(root: str | Path, *relative: str | Path) -> Path:
    """Resolve beneath both the workspace and the caller's Safe root."""

    base = require_within_root(root)
    candidate = base.joinpath(*(Path(item) for item in relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Path escapes Safe root: {candidate}") from exc
    require_within_root(candidate)
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    require_within_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_raw_manifest(path: Path) -> RawSourceManifest:
    return RawSourceManifest(**json.loads(path.read_text(encoding="utf-8")))


def _read_pretest_manifest(path: Path) -> PretestManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["channel_order"] = tuple(raw["channel_order"])
    return PretestManifest(**raw)


def _config_and_spec(
    dataset: str | DatasetSpec,
) -> tuple[SafeExperimentConfig, DatasetSpec]:
    config = load_safe_config()
    return config, config.dataset(dataset) if isinstance(dataset, str) else dataset


def manifest_existing_source(
    root: str | Path,
    spec: DatasetSpec,
    source_path: str | Path,
    *,
    enforce_expected_hash: bool = False,
) -> RawSourceManifest:
    """Hash a project-local source for an audited import or synthetic test."""

    base = require_safe_path(root)
    source = require_within_root(source_path, must_exist=True)
    try:
        relative = source.relative_to(base)
    except ValueError as exc:
        raise ValueError("Raw source is outside the supplied Safe root") from exc
    digest = _sha256(source)
    if enforce_expected_hash and spec.source.expected_sha256 != digest:
        raise ValueError("Raw source differs from the frozen official checksum")
    return RawSourceManifest(
        spec.dataset_id,
        relative.as_posix(),
        spec.source.download_url,
        spec.source.landing_url,
        source.name,
        digest,
        source.stat().st_size,
        spec.source.expected_sha256 if enforce_expected_hash else None,
    )


def download_official_dataset(
    root: str | Path,
    dataset: str | DatasetSpec,
    expected_source: SourceSpec | Mapping[str, Any] | None = None,
) -> RawSourceManifest:
    """Download and hash one frozen official object inside the Safe namespace."""

    config, spec = _config_and_spec(dataset)
    source = spec.source
    if expected_source is not None:
        if isinstance(expected_source, Mapping):
            expected_source = SourceSpec(**expected_source)
        if expected_source != source:
            raise ValueError("Caller source descriptor differs from frozen config")
    raw_dir = require_safe_path(root, config.relative_path("raw"), spec.dataset_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = require_safe_path(raw_dir, source.filename)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(f"Unreviewed partial download exists: {temporary}")
        request = Request(source.download_url, headers={"User-Agent": "EdgeTwinCal-Safe/1"})
        try:
            with urlopen(request, timeout=120) as response, temporary.open("xb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            digest = _sha256(temporary)
            if source.expected_sha256 and digest != source.expected_sha256:
                raise ValueError(f"{spec.dataset_id} official SHA256 mismatch")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    digest = _sha256(destination)
    if source.expected_sha256 and digest != source.expected_sha256:
        raise ValueError("Existing official source fails frozen SHA256")
    manifest = RawSourceManifest(
        spec.dataset_id,
        destination.relative_to(require_safe_path(root)).as_posix(),
        source.download_url,
        source.landing_url,
        source.filename,
        digest,
        destination.stat().st_size,
        source.expected_sha256,
    )
    _atomic_bytes(raw_dir / "raw_source_manifest.json", _json_bytes(manifest.to_dict()))
    return manifest


def _data_record(member: str, line: str) -> str:
    return json.dumps(
        {
            "kind": "data",
            "member": member,
            "line_b64": base64.b64encode(line.encode("utf-8")).decode("ascii"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite_float(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def _beijing_timestamp(row: Sequence[str], index: Mapping[str, int]) -> datetime:
    return datetime(
        int(row[index["year"]]), int(row[index["month"]]),
        int(row[index["day"]]), int(row[index["hour"]]),
    )


def _station_from_member(member: str) -> str:
    stem = Path(member).stem
    marker = "PRSA_Data_"
    return (
        stem.split(marker, 1)[1].split("_20130301", 1)[0]
        if marker in stem else stem
    )


def _route_beijing(
    source: Path, spec: DatasetSpec, sealed: io.TextIOBase
) -> tuple[list[tuple[datetime, int, float]], int, int]:
    rows: list[tuple[datetime, int, float]] = []
    sealed_count = 0
    channels = {str(channel): index for index, channel in enumerate(spec.channels)}
    with zipfile.ZipFile(source) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        )
        if not members:
            raise ValueError("Beijing archive contains no station CSV")
        for member in members:
            with archive.open(member) as binary:
                text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                header = next(csv.reader([text.readline()]))
                index = {
                    name.strip().lower(): position for position, name in enumerate(header)
                }
                if not {"year", "month", "day", "hour", "pm2.5"}.issubset(index):
                    raise ValueError(f"Missing Beijing columns in {member}")
                sealed.write(
                    json.dumps(
                        {"kind": "header", "member": member, "header": header},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                fallback = _station_from_member(member)
                for raw_line in text:
                    parsed = next(csv.reader([raw_line]))
                    timestamp = _beijing_timestamp(parsed, index)
                    if timestamp >= spec.test_end:
                        continue
                    station = (
                        parsed[index["station"]].strip()
                        if "station" in index and parsed[index["station"]].strip()
                        else fallback
                    )
                    if station not in channels:
                        raise ValueError(f"Unexpected Beijing station {station!r}")
                    if timestamp >= spec.test_start:
                        sealed.write(_data_record(member, raw_line) + "\n")
                        sealed_count += 1
                        continue
                    value = parsed[index["pm2.5"]].strip()
                    if value and value.upper() not in {"NA", "NAN"}:
                        rows.append(
                            (timestamp, channels[station], _finite_float(value, "PM2.5"))
                        )
    return rows, sealed_count, 0


def _intel_timestamp(parts: Sequence[str]) -> datetime:
    return datetime.fromisoformat(f"{parts[0]}T{parts[1]}")


def _floor_time(timestamp: datetime, seconds: int) -> datetime:
    epoch = datetime(1970, 1, 1)
    elapsed = int((timestamp - epoch).total_seconds())
    return epoch + timedelta(seconds=elapsed - elapsed % seconds)


def _route_intel(
    source: Path, spec: DatasetSpec, sealed: io.TextIOBase
) -> tuple[list[tuple[datetime, int, float]], int, int]:
    cells: dict[tuple[datetime, int], list[float]] = {}
    sealed_count = 0
    discarded_unusable = 0
    channels = {int(channel): index for index, channel in enumerate(spec.channels)}
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rt", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 2:
                raise ValueError("Intel row lacks timestamp fields")
            timestamp = _intel_timestamp(parts)
            if len(parts) < 5:
                if timestamp < spec.test_end:
                    discarded_unusable += 1
                continue
            if timestamp >= spec.test_end:
                continue
            if timestamp >= spec.test_start:
                sealed.write(_data_record("", raw_line) + "\n")
                sealed_count += 1
                continue
            mote = int(parts[3])
            if mote not in channels:
                continue
            key = (_floor_time(timestamp, spec.frequency_seconds), channels[mote])
            cells.setdefault(key, []).append(
                _finite_float(parts[4], "Intel temperature")
            )
    rows = [
        (timestamp, channel, float(statistics.median(values)))
        for (timestamp, channel), values in cells.items()
    ]
    return rows, sealed_count, discarded_unusable


def _write_observations(
    path: Path, rows: Iterable[tuple[datetime, int, float]]
) -> int:
    ordered = sorted(rows, key=lambda item: (item[0], item[1], item[2]))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("timestamp", "channel_index", "value"))
        for timestamp, channel, value in ordered:
            writer.writerow((timestamp.isoformat(), channel, format(value, ".17g")))
        handle.flush()
        os.fsync(handle.fileno())
    return len(ordered)


def prepare_pretest_shards(
    root: str | Path,
    spec: DatasetSpec,
    raw_manifest: RawSourceManifest,
) -> PretestManifest:
    """Timestamp-route first; test values are never numerically parsed here."""

    config = load_safe_config()
    if raw_manifest.dataset_id != spec.dataset_id:
        raise ValueError("Raw source and dataset IDs differ")
    base = require_safe_path(root)
    source = require_safe_path(base, raw_manifest.source_relative_path)
    if not source.is_file() or _sha256(source) != raw_manifest.sha256:
        raise ValueError("Raw source no longer matches its manifest")
    pretest_dir = require_safe_path(base, config.relative_path("pretest"), spec.dataset_id)
    sealed_dir = require_safe_path(base, config.relative_path("sealed_test"), spec.dataset_id)
    pretest_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    observations = require_safe_path(pretest_dir, "observations.csv")
    test_shard = require_safe_path(sealed_dir, "raw_test.jsonl")
    manifest_path = require_safe_path(pretest_dir, "pretest_manifest.json")
    if any(path.exists() for path in (observations, test_shard, manifest_path)):
        raise FileExistsError("Safe shards already exist and cannot be overwritten")
    observations_tmp = observations.with_suffix(".csv.routing")
    test_tmp = test_shard.with_suffix(".jsonl.routing")
    try:
        with test_tmp.open("x", encoding="utf-8", newline="\n") as sealed:
            if spec.dataset_id == "beijing_air":
                rows, sealed_count, discarded_unusable = _route_beijing(source, spec, sealed)
            elif spec.dataset_id == "intel_lab":
                rows, sealed_count, discarded_unusable = _route_intel(source, spec, sealed)
            else:
                raise ValueError(f"No frozen parser for {spec.dataset_id}")
            sealed.flush()
            os.fsync(sealed.fileno())
        pretest_count = _write_observations(observations_tmp, rows)
        os.replace(observations_tmp, observations)
        os.replace(test_tmp, test_shard)
    finally:
        for temporary in (observations_tmp, test_tmp):
            if temporary.exists():
                temporary.unlink()
    manifest = PretestManifest(
        spec.dataset_id,
        raw_manifest.sha256,
        observations.relative_to(base).as_posix(),
        test_shard.relative_to(base).as_posix(),
        pretest_count,
        sealed_count,
        _sha256(observations),
        _sha256(test_shard),
        spec.channels,
        spec.test_start.isoformat(),
        spec.test_end.isoformat(),
        discarded_unusable,
    )
    _atomic_bytes(manifest_path, _json_bytes(manifest.to_dict()))
    return manifest


def load_pretest_manifest(root: str | Path, spec: DatasetSpec) -> PretestManifest:
    config = load_safe_config()
    path = require_safe_path(
        root, config.relative_path("pretest"), spec.dataset_id, "pretest_manifest.json"
    )
    manifest = _read_pretest_manifest(path)
    if (
        manifest.dataset_id != spec.dataset_id
        or manifest.channel_order != spec.channels
        or manifest.test_start != spec.test_start.isoformat()
        or manifest.test_end != spec.test_end.isoformat()
    ):
        raise ValueError("Pre-test manifest differs from frozen spec")
    observations = require_safe_path(root, manifest.observations_relative_path)
    test_shard = require_safe_path(root, manifest.sealed_test_relative_path)
    if (
        _sha256(observations) != manifest.observations_sha256
        or _sha256(test_shard) != manifest.sealed_test_sha256
    ):
        raise ValueError("Prepared shard changed after routing")
    return manifest


def _audit_bucket(spec: DatasetSpec, timestamp: datetime) -> tuple[str, datetime]:
    descriptor = spec.validation_group_split
    if descriptor is None:
        raise ValueError("Dataset has no hashed validation split")
    block = _floor_time(timestamp, descriptor.group_seconds)
    digest = hashlib.sha256(
        f"{spec.id_salt}validation:{block.isoformat()}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % descriptor.modulus
    if bucket in descriptor.val_select_buckets:
        return "val_select", block
    if bucket in descriptor.val_safety_buckets:
        return "val_safety", block
    raise AssertionError("Unassigned validation hash bucket")


def split_for_timestamp(spec: DatasetSpec, timestamp: datetime) -> str | None:
    for partition in spec.partitions:
        if partition.contains(timestamp):
            return (
                _audit_bucket(spec, timestamp)[0]
                if partition.name == "validation_pool" else partition.name
            )
    return None


def load_pretest_rows(
    root: str | Path, spec: DatasetSpec, split: str = "all"
) -> PreparedRows:
    manifest = load_pretest_manifest(root, spec)
    path = require_safe_path(root, manifest.observations_relative_path)
    timestamps: list[datetime] = []
    channels: list[int] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["timestamp", "channel_index", "value"]:
            raise ValueError("Unexpected pre-test observation schema")
        for row in reader:
            timestamp = datetime.fromisoformat(row["timestamp"])
            if split != "all" and split_for_timestamp(spec, timestamp) != split:
                continue
            channel = int(row["channel_index"])
            if not 0 <= channel < len(spec.channels):
                raise ValueError("Channel index outside frozen order")
            timestamps.append(timestamp)
            channels.append(channel)
            values.append(_finite_float(row["value"], "pre-test observation"))
    return PreparedRows(
        spec.dataset_id, split, spec.channels, tuple(timestamps), tuple(channels),
        tuple(values), manifest.source_sha256,
    )


def fit_train_normalizer(pretest_train: PreparedRows) -> NormalizerState:
    if pretest_train.split != "train":
        raise ValueError("Normalizer may be fit only from the train partition")
    cells: list[list[float]] = [[] for _ in pretest_train.channels]
    for channel, value in zip(pretest_train.channel_indices, pretest_train.values):
        cells[channel].append(value)
    if any(not cell for cell in cells):
        raise ValueError(
            f"Train-only normalizer has empty channels: "
            f"{[index for index, cell in enumerate(cells) if not cell]}"
        )
    means = tuple(float(np.mean(cell, dtype=np.float64)) for cell in cells)
    scales = tuple(max(float(np.std(cell, dtype=np.float64)), 1e-6) for cell in cells)
    counts = tuple(len(cell) for cell in cells)
    payload = {
        "dataset_id": pretest_train.dataset_id,
        "channel_order": list(pretest_train.channels),
        "mean": means,
        "scale": scales,
        "observed_count": counts,
        "fit_split": "train",
        "source_sha256": pretest_train.source_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return NormalizerState(
        pretest_train.dataset_id, pretest_train.channels, means, scales, counts,
        "train", pretest_train.source_sha256, digest,
    )


def stable_pseudonymous_id(spec: DatasetSpec, kind: str, value: str) -> int:
    digest = hashlib.sha256(f"{spec.id_salt}{kind}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 53) - 1)


def group_anchor(spec: DatasetSpec, timestamp: datetime) -> datetime:
    if spec.grouping == "utc_calendar_day":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if spec.grouping == "utc_3_hour_block":
        return _floor_time(timestamp, 10800)
    raise ValueError(f"Unknown grouping {spec.grouping!r}")


def iter_target_starts(spec: DatasetSpec, split: str) -> Iterator[datetime]:
    partition = spec.partition(split)
    step = timedelta(seconds=spec.frequency_seconds * spec.stride_steps)
    horizon = timedelta(seconds=spec.frequency_seconds * spec.forecast_steps)
    current = partition.start
    while current + horizon <= partition.end:
        if spec.validation_group_split and split in {"val_select", "val_safety"}:
            assigned, block = _audit_bucket(spec, current)
            block_end = block + timedelta(
                seconds=spec.validation_group_split.group_seconds
            )
            if assigned == split and current + horizon <= block_end:
                yield current
        else:
            yield current
        current += step


class MaskedWindowDataset(Dataset):
    """Dense-shaped masked APN windows over sparse observations."""

    def __init__(
        self,
        spec: DatasetSpec,
        split: str,
        rows: PreparedRows,
        normalizer: NormalizerState,
    ) -> None:
        if (
            rows.dataset_id != spec.dataset_id
            or normalizer.dataset_id != spec.dataset_id
            or rows.channels != spec.channels
            or normalizer.channel_order != spec.channels
            or normalizer.fit_split != "train"
            or rows.source_sha256 != normalizer.source_sha256
        ):
            raise ValueError("Dataset and train-only normalizer identities differ")
        self.spec = spec
        self.split = split
        self.target_starts = tuple(iter_target_starts(spec, split))
        self._lookup = {
            (timestamp, channel): value
            for timestamp, channel, value in zip(
                rows.timestamps, rows.channel_indices, rows.values
            )
        }
        self._normalizer = normalizer

    def __len__(self) -> int:
        return len(self.target_starts)

    def target_timestamp_keys(self) -> set[str]:
        delta = timedelta(seconds=self.spec.frequency_seconds)
        return {
            (start + offset * delta).isoformat()
            for start in self.target_starts
            for offset in range(self.spec.forecast_steps)
        }

    def group_ids(self) -> set[int]:
        return {
            stable_pseudonymous_id(
                self.spec, "group", group_anchor(self.spec, start).isoformat()
            )
            for start in self.target_starts
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.target_starts[index]
        frequency = timedelta(seconds=self.spec.frequency_seconds)
        width = len(self.spec.channels)
        x = np.zeros((self.spec.history_steps, width), dtype=np.float32)
        y = np.zeros((self.spec.forecast_steps, width), dtype=np.float32)
        x_mask, y_mask = np.zeros_like(x), np.zeros_like(y)
        history_start = start - self.spec.history_steps * frequency
        for step in range(self.spec.history_steps):
            timestamp = history_start + step * frequency
            for channel in range(width):
                value = self._lookup.get((timestamp, channel))
                if value is not None:
                    x[step, channel] = self._normalizer.normalize(channel, value)
                    x_mask[step, channel] = 1
        for step in range(self.spec.forecast_steps):
            timestamp = start + step * frequency
            for channel in range(width):
                value = self._lookup.get((timestamp, channel))
                if value is not None:
                    y[step, channel] = self._normalizer.normalize(channel, value)
                    y_mask[step, channel] = 1
        history_scale = float(self.spec.history_steps)
        x_mark = (
            (np.arange(self.spec.history_steps, dtype=np.float32) + 0.5)
            / history_scale
        )[:, None]
        y_mark = (
            (
                self.spec.history_steps
                + np.arange(self.spec.forecast_steps, dtype=np.float32)
                + 0.5
            ) / history_scale
        )[:, None]
        sample_id = stable_pseudonymous_id(self.spec, "sample", start.isoformat())
        group_id = stable_pseudonymous_id(
            self.spec, "group", group_anchor(self.spec, start).isoformat()
        )
        return {
            "x": torch.from_numpy(x),
            "x_mark": torch.from_numpy(x_mark),
            "y": torch.from_numpy(y),
            "y_mark": torch.from_numpy(y_mark),
            "x_mask": torch.from_numpy(x_mask),
            "y_mask": torch.from_numpy(y_mask),
            "sample_id": torch.tensor(sample_id, dtype=torch.int64),
            "group_id": torch.tensor(group_id, dtype=torch.int64),
        }


def as_apn_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    required = {
        "x", "x_mark", "y", "y_mark", "x_mask", "y_mask", "sample_id", "group_id"
    }
    if set(batch) != required:
        raise ValueError(f"Safe batch keys differ: {sorted(set(batch) ^ required)}")
    result = dict(batch)
    result["sample_ID"] = result.pop("sample_id")
    result["group_ID"] = result.pop("group_id")
    return result


def _loader(
    dataset: MaskedWindowDataset, split: str, seed: int, batch_size: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=split == "train",
        generator=generator, num_workers=0, drop_last=False,
    )


def build_pretest_loaders(
    spec: DatasetSpec,
    normalizer: NormalizerState,
    seed: int,
    *,
    root: str | Path = PROJECT_ROOT,
    batch_size: int | None = None,
) -> PretestLoaders:
    size = load_safe_config().apn.batch_size if batch_size is None else int(batch_size)
    rows = load_pretest_rows(root, spec, "all")
    datasets = {
        split: MaskedWindowDataset(spec, split, rows, normalizer)
        for split in ("train", "val", "adapter", "val_select", "val_safety")
    }
    if datasets["val_select"].group_ids() & datasets["val_safety"].group_ids():
        raise ValueError("Validation select and safety groups overlap")
    if (
        datasets["val_select"].target_timestamp_keys()
        & datasets["val_safety"].target_timestamp_keys()
    ):
        raise ValueError("Validation select and safety target timestamps overlap")
    return PretestLoaders(
        *(
            _loader(datasets[split], split, seed, size)
            for split in ("train", "val", "adapter", "val_select", "val_safety")
        )
    )


def _ledger_path(root: str | Path, dataset_id: str) -> Path:
    config = load_safe_config()
    return require_safe_path(
        root, config.relative_path("results"), "protocol", dataset_id, "test_ledger.json"
    )


def freeze_test_ledger(
    root: str | Path,
    dataset: str | DatasetSpec,
    provenance: Mapping[str, Any],
) -> TestLedger:
    _, spec = _config_and_spec(dataset)
    manifest = load_pretest_manifest(root, spec)
    ledger_path = _ledger_path(root, spec.dataset_id)
    claim = ledger_path.with_suffix(".claim")
    if ledger_path.exists() or claim.exists():
        raise FileExistsError("Test ledger already frozen or claimed")
    token = secrets.token_urlsafe(32)
    payload = {
        "schema_version": "edgetwincal.safe-test-ledger.v1",
        "dataset_id": spec.dataset_id,
        "status": "frozen",
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "sealed_test_sha256": manifest.sealed_test_sha256,
        "source_sha256": manifest.source_sha256,
        "provenance": json.loads(json.dumps(provenance, sort_keys=True)),
    }
    _atomic_bytes(ledger_path, _json_bytes(payload))
    return TestLedger(spec.dataset_id, ledger_path, "frozen", token)


def _claim_ledger(
    root: str | Path, spec: DatasetSpec, token: str
) -> tuple[Path, dict[str, Any]]:
    path = _ledger_path(root, spec.dataset_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "frozen":
        raise RuntimeError("Sealed test is not frozen")
    supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(supplied, str(raw.get("token_sha256", ""))):
        raise PermissionError("Invalid sealed-test token")
    claim = path.with_suffix(".claim")
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Sealed test has already been claimed") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(supplied + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    raw["status"] = "opening"
    _atomic_bytes(path, _json_bytes(raw))
    return path, raw


def _decode_raw(record: Mapping[str, Any]) -> str:
    if record.get("kind") != "data":
        raise ValueError("Unexpected sealed data record")
    return base64.b64decode(record["line_b64"], validate=True).decode("utf-8")


def _parse_sealed_test(
    root: str | Path, spec: DatasetSpec, manifest: PretestManifest
) -> list[tuple[datetime, int, float]]:
    path = require_safe_path(root, manifest.sealed_test_relative_path)
    rows: list[tuple[datetime, int, float]] = []
    if spec.dataset_id == "beijing_air":
        headers: dict[str, dict[str, int]] = {}
        channels = {str(channel): index for index, channel in enumerate(spec.channels)}
        with path.open("r", encoding="utf-8") as handle:
            for encoded in handle:
                record = json.loads(encoded)
                member = str(record["member"])
                if record.get("kind") == "header":
                    headers[member] = {
                        str(name).strip().lower(): index
                        for index, name in enumerate(record["header"])
                    }
                    continue
                parsed = next(csv.reader([_decode_raw(record)]))
                index = headers.get(member)
                if index is None:
                    raise ValueError("Missing sealed Beijing header")
                timestamp = _beijing_timestamp(parsed, index)
                station = (
                    parsed[index["station"]].strip()
                    if "station" in index and parsed[index["station"]].strip()
                    else _station_from_member(member)
                )
                value = parsed[index["pm2.5"]].strip()
                if value and value.upper() not in {"NA", "NAN"}:
                    rows.append(
                        (timestamp, channels[station], _finite_float(value, "sealed PM2.5"))
                    )
    elif spec.dataset_id == "intel_lab":
        cells: dict[tuple[datetime, int], list[float]] = {}
        channels = {int(channel): index for index, channel in enumerate(spec.channels)}
        with path.open("r", encoding="utf-8") as handle:
            for encoded in handle:
                parts = _decode_raw(json.loads(encoded)).split()
                mote = int(parts[3])
                if mote not in channels:
                    continue
                key = (
                    _floor_time(_intel_timestamp(parts), spec.frequency_seconds),
                    channels[mote],
                )
                cells.setdefault(key, []).append(
                    _finite_float(parts[4], "sealed Intel temperature")
                )
        rows = [
            (timestamp, channel, float(statistics.median(values)))
            for (timestamp, channel), values in cells.items()
        ]
    else:
        raise ValueError(f"No sealed parser for {spec.dataset_id}")
    return sorted(rows, key=lambda item: (item[0], item[1], item[2]))


def open_sealed_test_loader(
    root: str | Path,
    spec: DatasetSpec,
    token: str,
    normalizer: NormalizerState,
    seed: int,
    *,
    batch_size: int | None = None,
) -> DataLoader:
    """Claim and parse the test shard once, then permanently consume its token."""

    ledger_path, ledger = _claim_ledger(root, spec, token)
    try:
        manifest = load_pretest_manifest(root, spec)
        if ledger["sealed_test_sha256"] != manifest.sealed_test_sha256:
            raise ValueError("Ledger and sealed shard hashes differ")
        test_rows = _parse_sealed_test(root, spec, manifest)
        pretest = load_pretest_rows(root, spec, "all")
        combined = PreparedRows(
            spec.dataset_id,
            "all",
            spec.channels,
            pretest.timestamps + tuple(row[0] for row in test_rows),
            pretest.channel_indices + tuple(row[1] for row in test_rows),
            pretest.values + tuple(row[2] for row in test_rows),
            pretest.source_sha256,
        )
        dataset = MaskedWindowDataset(spec, "test", combined, normalizer)
        size = load_safe_config().apn.batch_size if batch_size is None else int(batch_size)
        loader = _loader(dataset, "test", seed, size)
        ledger["status"] = "consumed"
        ledger["parsed_test_rows"] = len(test_rows)
        _atomic_bytes(ledger_path, _json_bytes(ledger))
        return loader
    except BaseException as exc:
        ledger["status"] = "failed_consumed"
        ledger["failure_type"] = type(exc).__name__
        _atomic_bytes(ledger_path, _json_bytes(ledger))
        raise
