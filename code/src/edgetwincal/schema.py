"""Unified, fail-closed run manifests for EdgeTwinCal msn2026_v1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigError, ResolvedConfig, canonical_json, canonical_sha256, validate_config
from .paths import PROJECT_ROOT, require_within_root


SCHEMA_VERSION = 1
TERMINAL_STATES = frozenset({"complete", "failed"})
ALL_STATES = frozenset({"created", "running", *TERMINAL_STATES})
_HEX = frozenset("0123456789abcdef")
_ATOMIC_REPLACE_ATTEMPTS = 4
_ATOMIC_REPLACE_BASE_DELAY_SECONDS = 0.01


class ManifestError(ValueError):
    """Base class for run-manifest integrity failures."""


class InvalidTransitionError(ManifestError):
    """Raised when a run attempts an illegal state transition."""


class IncompleteRunError(ManifestError):
    """Raised when a non-complete run is offered to aggregation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    )


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{label} is not finite JSON data: {exc}") from exc


def _relative_project_path(path: str | Path, *, must_exist: bool = False) -> tuple[Path, str]:
    try:
        resolved = require_within_root(path, must_exist=must_exist)
    except (FileNotFoundError, ValueError) as exc:
        raise ManifestError(f"Path is unavailable or outside the project: {path}") from exc
    relative = resolved.relative_to(PROJECT_ROOT).as_posix()
    if not relative or relative == ".":
        raise ManifestError("A run artifact may not be the project root")
    return resolved, relative


def sha256_file(path: str | Path) -> str:
    """Hash one project-local regular file without loading it into memory."""

    resolved, _ = _relative_project_path(path, must_exist=True)
    if not resolved.is_file():
        raise ManifestError(f"Required artifact is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_file_record(path: str | Path) -> dict[str, Any]:
    """Build the canonical relative-path/size/SHA record for one artifact."""

    resolved, relative = _relative_project_path(path, must_exist=True)
    if not resolved.is_file():
        raise ManifestError(f"Required artifact is not a regular file: {resolved}")
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
def _replace_with_bounded_windows_retry(source: Path, destination: Path) -> None:
    """Retry only transient Windows sharing/ACL races, then fail closed."""

    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            retryable = os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}
            if not retryable or attempt + 1 == _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(_ATOMIC_REPLACE_BASE_DELAY_SECONDS * (2**attempt))
    raise AssertionError("bounded atomic replace loop exhausted without returning")




def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a project-local JSON file using a same-dir temp file."""

    destination, _ = _relative_project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = _json_copy(payload, label="manifest")
    text = json.dumps(
        materialized,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    _relative_project_path(temporary)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_bounded_windows_retry(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _manifest_reference(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    reference = _json_copy(value, label=label)
    if not isinstance(reference, dict) or not reference:
        raise ManifestError(f"{label} must be a non-empty object")
    recorded = reference.pop("sha256", None)
    calculated = canonical_sha256(reference)
    if recorded is not None and recorded != calculated:
        raise ManifestError(f"{label}.sha256 does not match its canonical content")
    reference["sha256"] = calculated
    return reference


def _validate_required_file_records(
    records: Any,
    *,
    verify_files: bool,
) -> None:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ManifestError("required_files must be a non-empty array")
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ManifestError(f"required_files[{index}] must be an object")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ManifestError(f"required_files[{index}].path must be project-relative")
        resolved, normalized = _relative_project_path(relative, must_exist=verify_files)
        if normalized != Path(relative).as_posix():
            raise ManifestError(f"required_files[{index}].path is not canonical: {relative}")
        if normalized in seen:
            raise ManifestError(f"Duplicate required file: {normalized}")
        seen.add(normalized)
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ManifestError(f"required_files[{index}].bytes must be non-negative")
        digest = record.get("sha256")
        if not _is_sha256(digest):
            raise ManifestError(f"required_files[{index}].sha256 is invalid")
        if verify_files:
            if not resolved.is_file():
                raise ManifestError(f"Required file is missing: {normalized}")
            if resolved.stat().st_size != size:
                raise ManifestError(f"Required file size changed: {normalized}")
            if sha256_file(resolved) != digest:
                raise ManifestError(f"Required file hash changed: {normalized}")


def _validate_cells(cells: Any) -> None:
    if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or not cells:
        raise ManifestError("cells must contain at least one SSE/SAE/N row")
    positive_observations = 0
    required = {"group_hash", "checkpoint_sha256", "variant", "sse", "sae", "n"}
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ManifestError(f"cells[{index}] must be an object")
        missing = sorted(required.difference(cell))
        if missing:
            raise ManifestError(f"cells[{index}] is missing {missing}")
        if not isinstance(cell["group_hash"], str) or not cell["group_hash"]:
            raise ManifestError(f"cells[{index}].group_hash must be salted and non-empty")
        if not _is_sha256(cell["checkpoint_sha256"]):
            raise ManifestError(f"cells[{index}].checkpoint_sha256 is invalid")
        if not isinstance(cell["variant"], str) or not cell["variant"]:
            raise ManifestError(f"cells[{index}].variant is invalid")
        for metric in ("sse", "sae"):
            value = cell[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ManifestError(f"cells[{index}].{metric} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ManifestError(f"cells[{index}].{metric} must be finite and non-negative")
        count = cell["n"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ManifestError(f"cells[{index}].n must be a non-negative integer")
        positive_observations += count
    if positive_observations <= 0:
        raise ManifestError("A complete run must contain at least one observed target")


def _validate_base(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
    status = data.get("status")
    if status not in ALL_STATES:
        raise ManifestError(f"Unknown run status: {status}")
    for key in ("run_id", "dataset", "protocol", "fold", "variant_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ManifestError(f"{key} must be a non-empty string")
    seed = data.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestError("seed must be an integer")
    resolved_config = data.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ManifestError("resolved_config must be an object")
    try:
        validate_config(resolved_config)
    except ConfigError as exc:
        raise ManifestError(f"resolved_config violates the frozen campaign schema: {exc}") from exc
    if data.get("resolved_config_sha256") != canonical_sha256(resolved_config):
        raise ManifestError("resolved_config_sha256 mismatch")
    variant_definition = data.get("variant_definition")
    if not isinstance(variant_definition, Mapping) or not variant_definition:
        raise ManifestError("variant_definition must be a non-empty object")
    if data.get("variant_sha256") != canonical_sha256(variant_definition):
        raise ManifestError("variant_sha256 mismatch")
    argv = data.get("argv")
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise ManifestError("argv must be a non-empty string array")
    if any(not isinstance(argument, str) for argument in argv):
        raise ManifestError("argv entries must be strings")
    if not isinstance(data.get("environment"), Mapping) or not data["environment"]:
        raise ManifestError("environment must be a non-empty object")
    log_path = data.get("log_path")
    if not isinstance(log_path, str) or Path(log_path).is_absolute():
        raise ManifestError("log_path must be project-relative")
    _, normalized_log = _relative_project_path(log_path)
    if normalized_log != Path(log_path).as_posix():
        raise ManifestError("log_path is not canonical")
    completed_phases = data.get("completed_phases")
    if not isinstance(completed_phases, list) or any(
        not isinstance(phase, str) or not phase for phase in completed_phases
    ):
        raise ManifestError("completed_phases must be a string array")
    if len(set(completed_phases)) != len(completed_phases):
        raise ManifestError("completed_phases contains duplicates")
    events = data.get("events")
    if not isinstance(events, list) or not events:
        raise ManifestError("events must be a non-empty array")
    _json_copy(data, label="manifest")


def validate_run_manifest(
    data: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> None:
    """Validate state, hashes, completeness, and optionally artifact bytes."""

    _validate_base(data)
    status = data["status"]
    if status == "failed":
        error = data.get("error")
        if not isinstance(error, Mapping) or not error.get("message"):
            raise ManifestError("A failed run must retain a non-empty error")
        records = data.get("required_files", [])
        if records:
            _validate_required_file_records(records, verify_files=verify_files)
        return
    if status != "complete":
        if data.get("error") is not None:
            raise ManifestError(f"{status} run may not contain an error")
        return

    if data.get("error") is not None:
        raise ManifestError("A complete run must be error-free")
    required_fields = {
        "assets",
        "cache_manifest",
        "split_manifest",
        "normalization_manifest",
        "selected_hyperparameters",
        "timing",
        "cells",
        "metrics",
        "required_files",
    }
    missing = sorted(required_fields.difference(data))
    if missing:
        raise ManifestError(f"Complete run is missing fields: {missing}")
    for field in ("assets", "cache_manifest", "split_manifest", "normalization_manifest"):
        reference = data[field]
        if not isinstance(reference, Mapping) or not reference:
            raise ManifestError(f"{field} must be a non-empty object")
        recorded = reference.get("sha256")
        if not _is_sha256(recorded):
            raise ManifestError(f"{field}.sha256 is invalid")
        content = dict(reference)
        content.pop("sha256", None)
        if canonical_sha256(content) != recorded:
            raise ManifestError(f"{field}.sha256 mismatch")
    if not isinstance(data["selected_hyperparameters"], Mapping):
        raise ManifestError("selected_hyperparameters must be an object")
    timing = data["timing"]
    if not isinstance(timing, Mapping) or not timing.get("segments"):
        raise ManifestError("timing.segments must be non-empty")
    metrics = data["metrics"]
    if not isinstance(metrics, Mapping) or not metrics:
        raise ManifestError("metrics must be a non-empty object")
    _validate_cells(data["cells"])
    _validate_required_file_records(data["required_files"], verify_files=verify_files)
    required_paths = {record["path"] for record in data["required_files"]}
    if data["log_path"] not in required_paths:
        raise ManifestError("log_path must be present in required_files")


@dataclass
class RunManifest:
    """A persisted run state machine with atomic, validated transitions."""

    path: Path
    _data: dict[str, Any]

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        run_id: str,
        dataset: str,
        protocol: str,
        fold: str,
        seed: int,
        variant_id: str,
        variant_definition: Mapping[str, Any],
        resolved_config: ResolvedConfig | Mapping[str, Any],
        argv: Sequence[str],
        environment: Mapping[str, Any],
        log_path: str | Path,
    ) -> "RunManifest":
        destination, _ = _relative_project_path(path)
        if destination.exists():
            raise ManifestError(f"Refusing to overwrite an existing run manifest: {destination}")
        if isinstance(resolved_config, ResolvedConfig):
            config_data = resolved_config.to_dict()
            config_hash = resolved_config.sha256
        else:
            config_data = _json_copy(resolved_config, label="resolved_config")
            config_hash = canonical_sha256(config_data)
        variant_data = _json_copy(variant_definition, label="variant_definition")
        _, relative_log = _relative_project_path(log_path)
        timestamp = _now()
        data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "dataset": dataset,
            "protocol": protocol,
            "fold": fold,
            "seed": seed,
            "variant_id": variant_id,
            "variant_definition": variant_data,
            "variant_sha256": canonical_sha256(variant_data),
            "resolved_config": config_data,
            "resolved_config_sha256": config_hash,
            "argv": list(argv),
            "environment": _json_copy(environment, label="environment"),
            "log_path": relative_log,
            "status": "created",
            "error": None,
            "completed_phases": [],
            "created_at": timestamp,
            "updated_at": timestamp,
            "events": [{"status": "created", "at": timestamp}],
        }
        validate_run_manifest(data, verify_files=False)
        atomic_write_json(destination, data)
        return cls(destination, data)

    @classmethod
    def load(cls, path: str | Path, *, verify_files: bool = False) -> "RunManifest":
        resolved, _ = _relative_project_path(path, must_exist=True)
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"Cannot load run manifest {resolved}: {exc}") from exc
        if not isinstance(data, dict):
            raise ManifestError("Run manifest root must be an object")
        validate_run_manifest(data, verify_files=verify_files)
        return cls(resolved, data)

    @property
    def status(self) -> str:
        return str(self._data["status"])

    @property
    def data(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def _transition(self, expected: str, target: str) -> None:
        if self.status != expected:
            raise InvalidTransitionError(
                f"Cannot transition {self.status} -> {target}; expected {expected}"
            )
        timestamp = _now()
        self._data["status"] = target
        self._data["updated_at"] = timestamp
        self._data["events"].append({"status": target, "at": timestamp})

    def _persist(self, *, verify_files: bool) -> None:
        validate_run_manifest(self._data, verify_files=verify_files)
        atomic_write_json(self.path, self._data)

    def start(self) -> "RunManifest":
        candidate = deepcopy(self._data)
        self._transition("created", "running")
        try:
            self._persist(verify_files=False)
        except Exception:
            self._data = candidate
            raise
        return self

    def mark_phase(self, phase: str) -> "RunManifest":
        if self.status != "running":
            raise InvalidTransitionError("Phases may only be recorded while running")
        if not isinstance(phase, str) or not phase:
            raise ManifestError("phase must be a non-empty string")
        if phase in self._data["completed_phases"]:
            raise ManifestError(f"Phase already recorded: {phase}")
        candidate = deepcopy(self._data)
        self._data["completed_phases"].append(phase)
        self._data["updated_at"] = _now()
        try:
            self._persist(verify_files=False)
        except Exception:
            self._data = candidate
            raise
        return self

    def complete(
        self,
        *,
        assets: Mapping[str, Any],
        cache_manifest: Mapping[str, Any],
        split_manifest: Mapping[str, Any],
        normalization_manifest: Mapping[str, Any],
        selected_hyperparameters: Mapping[str, Any],
        timing: Mapping[str, Any],
        cells: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
        required_files: Iterable[str | Path],
    ) -> "RunManifest":
        if self.status != "running":
            raise InvalidTransitionError(f"Cannot complete a {self.status} run")
        candidate = deepcopy(self._data)
        records = [required_file_record(path) for path in required_files]
        self._data.update(
            {
                "assets": _manifest_reference(assets, label="assets"),
                "cache_manifest": _manifest_reference(cache_manifest, label="cache_manifest"),
                "split_manifest": _manifest_reference(split_manifest, label="split_manifest"),
                "normalization_manifest": _manifest_reference(
                    normalization_manifest, label="normalization_manifest"
                ),
                "selected_hyperparameters": _json_copy(
                    selected_hyperparameters, label="selected_hyperparameters"
                ),
                "timing": _json_copy(timing, label="timing"),
                "cells": _json_copy(cells, label="cells"),
                "metrics": _json_copy(metrics, label="metrics"),
                "required_files": sorted(records, key=lambda record: record["path"]),
            }
        )
        self._transition("running", "complete")
        try:
            self._persist(verify_files=True)
        except Exception:
            self._data = candidate
            raise
        return self

    def fail(
        self,
        error: BaseException | str,
        *,
        required_files: Iterable[str | Path] = (),
    ) -> "RunManifest":
        if self.status != "running":
            raise InvalidTransitionError(f"Cannot fail a {self.status} run")
        candidate = deepcopy(self._data)
        if isinstance(error, BaseException):
            error_record = {"type": type(error).__name__, "message": str(error)}
        else:
            error_record = {"type": "RunFailure", "message": str(error)}
        if not error_record["message"]:
            raise ManifestError("Failure message may not be empty")
        records = [required_file_record(path) for path in required_files]
        self._data["error"] = error_record
        self._data["required_files"] = sorted(records, key=lambda record: record["path"])
        self._transition("running", "failed")
        try:
            self._persist(verify_files=True)
        except Exception:
            self._data = candidate
            raise
        return self


def aggregation_eligibility(path: str | Path) -> tuple[bool, str]:
    """Return explicit aggregation eligibility without silently hiding failure."""

    try:
        manifest = RunManifest.load(path, verify_files=True)
    except ManifestError as exc:
        return False, f"invalid:{exc}"
    if manifest.status != "complete":
        return False, f"status:{manifest.status}"
    return True, "complete_and_verified"


def load_complete_manifest(path: str | Path) -> dict[str, Any]:
    """Load one aggregation input and fail closed on non-complete state."""

    manifest = RunManifest.load(path, verify_files=True)
    if manifest.status != "complete":
        raise IncompleteRunError(
            f"Run {manifest._data['run_id']} has status {manifest.status}; aggregation requires complete"
        )
    return manifest.data


def load_complete_manifests(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load every expected run; any missing, failed, or corrupt member aborts."""

    return [load_complete_manifest(path) for path in paths]
