"""Resolved, immutable configuration for the msn2026 EdgeTwinCal campaign.

This module intentionally imports only the Python standard library and the
project path guard.  In particular, loading configuration or displaying CLI
help never imports APN, torch, or dataset code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .paths import PROJECT_ROOT, require_within_root


DEFAULT_CONFIG = PROJECT_ROOT / "code" / "configs" / "msn2026" / "default.json"
_HEX = frozenset("0123456789abcdef")
_REQUIRED_VARIANTS = frozenset(
    {"APN", "SLRH", "CFG", "Full", "V01", "V02", "V03", "V07", "V08", "V10", "V11", "V12"}
)
_EXPECTED_SEEDS = (2024, 2025, 2026, 2027, 2028)
_EXPECTED_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
_HORIZON_KINDS = frozenset({"fixed_steps", "time_window"})
_PADDING_SOURCE = "frozen_protocol_manifest_or_extraction_contract"


class ConfigError(ValueError):
    """Raised when a campaign configuration violates the frozen protocol."""


def _json_ready(value: Any, *, location: str = "$.") -> Any:
    """Return plain JSON containers while rejecting ambiguous/non-finite data."""

    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"JSON object key at {location} must be a string")
            ready[key] = _json_ready(child, location=f"{location}{key}.")
        return ready
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(child, location=f"{location}[{index}].")
            for index, child in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"Non-finite number at {location}")
        return value
    raise ConfigError(f"Unsupported JSON value {type(value).__name__} at {location}")


def canonical_json(value: Any) -> str:
    """Serialize *value* with the campaign's stable canonical JSON rules."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of canonical UTF-8 JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True)
class ResolvedConfig(Mapping[str, Any]):
    """Immutable resolved configuration and its canonical content hash."""

    _data: Mapping[str, Any]
    sha256: str
    source_path: Path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def canonical_json(self) -> str:
        return canonical_json(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached mutable copy suitable for JSON persistence."""

        return _json_ready(self._data)

    def resolve_path(self, name: str, *, must_exist: bool = False) -> Path:
        """Resolve one configured relative path through the project boundary."""

        paths = self._data.get("paths")
        if not isinstance(paths, Mapping) or name not in paths:
            raise ConfigError(f"Unknown configured path: {name}")
        return require_within_root(str(paths[name]), must_exist=must_exist)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a JSON object")
    return value


def _sequence(parent: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = parent.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{key} must be a JSON array")
    return value


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _HEX for character in value.lower())
    )


def _validate_relative_path(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.anchor or "~" in candidate.parts:
        raise ConfigError(f"{label} must be relative to the project root: {value}")
    if ".." in candidate.parts:
        raise ConfigError(f"{label} may not contain a parent traversal: {value}")
    try:
        require_within_root(candidate)
    except ValueError as exc:
        raise ConfigError(f"{label} escapes the project root: {value}") from exc


def _validate_selection(
    selection: Mapping[str, Any],
    *,
    datasets: Mapping[str, Any],
    seeds: Sequence[int],
    variants: Mapping[str, Any],
) -> None:
    universes: dict[str, tuple[Any, ...]] = {
        "datasets": tuple(datasets),
        "seeds": tuple(seeds),
        "variants": tuple(variants),
    }
    for name, universe in universes.items():
        selected = _sequence(selection, name)
        if not selected:
            raise ConfigError(f"selection.{name} may not be empty")
        if len(set(selected)) != len(selected):
            raise ConfigError(f"selection.{name} contains duplicates")
        unknown = [item for item in selected if item not in universe]
        if unknown:
            raise ConfigError(f"selection.{name} contains unregistered values: {unknown}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate all protocol-critical FIX-01 fields, without touching assets."""

    _json_ready(config)
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    method = _mapping(config, "method")
    if method.get("version") != "msn2026_v1":
        raise ConfigError("method.version must be msn2026_v1")

    project = _mapping(config, "project")
    if not _is_hex_digest(project.get("source_anchor_commit"), 40):
        raise ConfigError("project.source_anchor_commit must be a 40-character git commit")

    apn = _mapping(config, "apn")
    commit = apn.get("commit")
    if not _is_hex_digest(commit, 40):
        raise ConfigError("apn.commit must be a 40-character lowercase hexadecimal commit")
    _validate_relative_path(apn.get("patch"), label="apn.patch")

    campaign = _mapping(config, "campaign")
    seeds = tuple(_sequence(campaign, "seeds"))
    if seeds != _EXPECTED_SEEDS or any(isinstance(seed, bool) for seed in seeds):
        raise ConfigError(f"campaign.seeds must be {list(_EXPECTED_SEEDS)}")

    paths = _mapping(config, "paths")
    if not paths:
        raise ConfigError("paths may not be empty")
    for name, value in paths.items():
        _validate_relative_path(value, label=f"paths.{name}")

    datasets = _mapping(config, "datasets")
    if set(datasets) != {"P12", "HumanActivity", "USHCN", "MIMIC_III"}:
        raise ConfigError("datasets must contain the four locked APN datasets")
    for dataset_id, dataset in datasets.items():
        if not isinstance(dataset, Mapping):
            raise ConfigError(f"datasets.{dataset_id} must be an object")
        _validate_relative_path(dataset.get("storage"), label=f"datasets.{dataset_id}.storage")
        task = _mapping(dataset, "task")
        for name in ("history", "channels"):
            value = task.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"datasets.{dataset_id}.task.{name} must be positive")
        horizon_kind = task.get("horizon_kind")
        if horizon_kind not in _HORIZON_KINDS:
            raise ConfigError(
                f"datasets.{dataset_id}.task.horizon_kind must be fixed_steps or time_window"
            )
        if "horizon" in task:
            raise ConfigError(
                f"datasets.{dataset_id}.task.horizon is ambiguous; use prediction_steps "
                "or forecast_window"
            )
        if horizon_kind == "fixed_steps":
            prediction_steps = task.get("prediction_steps")
            if (
                isinstance(prediction_steps, bool)
                or not isinstance(prediction_steps, int)
                or prediction_steps <= 0
            ):
                raise ConfigError(
                    f"datasets.{dataset_id}.task.prediction_steps must be positive"
                )
            if "forecast_window" in task or "padded_prediction_steps_source" in task:
                raise ConfigError(
                    f"datasets.{dataset_id}.task fixed_steps may not define time-window fields"
                )
        else:
            forecast_window = task.get("forecast_window")
            if (
                isinstance(forecast_window, bool)
                or not isinstance(forecast_window, int)
                or forecast_window <= 0
            ):
                raise ConfigError(
                    f"datasets.{dataset_id}.task.forecast_window must be positive"
                )
            if task.get("padded_prediction_steps_source") != _PADDING_SOURCE:
                raise ConfigError(
                    f"datasets.{dataset_id}.task.padded_prediction_steps_source must be "
                    f"{_PADDING_SOURCE}"
                )
            if "prediction_steps" in task:
                raise ConfigError(
                    f"datasets.{dataset_id}.task time_window may not define prediction_steps"
                )

        expected_kind = "time_window" if dataset_id == "HumanActivity" else "fixed_steps"
        if horizon_kind != expected_kind:
            raise ConfigError(
                f"datasets.{dataset_id}.task.horizon_kind must be {expected_kind}"
            )


    ridge = _mapping(config, "ridge")
    alpha_grid = tuple(float(value) for value in _sequence(ridge, "alpha_grid"))
    joint_grid = tuple(float(value) for value in _sequence(ridge, "joint_alpha_grid"))
    if alpha_grid != _EXPECTED_ALPHA_GRID or joint_grid != _EXPECTED_ALPHA_GRID:
        raise ConfigError(f"ridge grids must be {list(_EXPECTED_ALPHA_GRID)}")
    if float(ridge.get("scale_floor", 0.0)) != 1e-6:
        raise ConfigError("ridge.scale_floor must be 1e-6")
    if ridge.get("intercept_penalized") is not False:
        raise ConfigError("ridge intercept must remain unpenalized")

    variants = _mapping(config, "variants")
    missing_variants = sorted(_REQUIRED_VARIANTS.difference(variants))
    if missing_variants:
        raise ConfigError(f"Missing frozen variants: {missing_variants}")
    for variant_id, definition in variants.items():
        if not isinstance(definition, Mapping) or not definition.get("definition"):
            raise ConfigError(f"variants.{variant_id} requires a definition")

    bootstrap = _mapping(config, "bootstrap")
    if bootstrap.get("draws") != 50000 or bootstrap.get("analysis_seed") != 20260721:
        raise ConfigError("bootstrap must use 50,000 draws and analysis seed 20260721")

    _mapping(config, "protocols")
    _mapping(config, "training")
    _mapping(config, "timing")
    gates = _mapping(config, "gates")
    if set(gates) != {"G0", "G1", "G2", "G3", "G4"}:
        raise ConfigError("gates must contain G0 through G4")

    selection = _mapping(config, "selection")
    _validate_selection(selection, datasets=datasets, seeds=seeds, variants=variants)


def _normalize_requested(value: Any, *, name: str, integer: bool = False) -> list[Any]:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        raw = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise ConfigError(f"{name} selector must be a value or sequence")
    if not raw:
        raise ConfigError(f"{name} selector may not be empty")
    if integer:
        converted: list[int] = []
        for item in raw:
            if isinstance(item, bool):
                raise ConfigError(f"{name} selector contains a boolean")
            try:
                converted.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{name} selector contains a non-integer: {item!r}") from exc
        raw = converted
    return raw


def _apply_selectors(config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    allowed = {"dataset", "datasets", "seed", "seeds", "variant", "variants"}
    unknown_keys = sorted(set(overrides).difference(allowed))
    if unknown_keys:
        raise ConfigError(f"Only dataset/seed/variant selectors may be overridden: {unknown_keys}")

    aliases = (("dataset", "datasets"), ("seed", "seeds"), ("variant", "variants"))
    for singular, plural in aliases:
        if singular in overrides and plural in overrides:
            raise ConfigError(f"Specify only one of {singular} or {plural}")
        key = singular if singular in overrides else plural if plural in overrides else None
        if key is None:
            continue
        requested = _normalize_requested(
            overrides[key], name=plural, integer=plural == "seeds"
        )
        if len(set(requested)) != len(requested):
            raise ConfigError(f"{plural} selector contains duplicates")
        if plural == "datasets":
            universe = list(config["datasets"])
        elif plural == "seeds":
            universe = list(config["campaign"]["seeds"])
        else:
            universe = list(config["variants"])
        unknown = [item for item in requested if item not in universe]
        if unknown:
            raise ConfigError(f"Unknown {plural} selection: {unknown}")
        requested_set = set(requested)
        config["selection"][plural] = [item for item in universe if item in requested_set]


def load_resolved_config(
    path: str | Path = DEFAULT_CONFIG,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    """Load, select, validate, freeze, and hash the single campaign config.

    ``overrides`` is deliberately limited to membership selectors.  It cannot
    change a hyperparameter, protocol, path, split, bootstrap setting, or gate.
    """

    try:
        config_path = require_within_root(path, must_exist=True)
    except (FileNotFoundError, ValueError) as exc:
        raise ConfigError(f"Config path is unavailable or outside the project: {path}") from exc
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot load JSON config {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("Top-level config must be a JSON object")

    resolved = deepcopy(loaded)
    if overrides:
        _apply_selectors(resolved, overrides)
    validate_config(resolved)
    digest = canonical_sha256(resolved)
    return ResolvedConfig(_freeze(resolved), digest, config_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and resolve the frozen EdgeTwinCal msn2026 configuration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--seed", action="append", dest="seeds", type=int)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--print-resolved", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    selectors = {
        key: value
        for key, value in {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "variants": args.variants,
        }.items()
        if value is not None
    }
    try:
        resolved = load_resolved_config(args.config, selectors)
    except ConfigError as exc:
        parser.error(str(exc))
    if args.print_resolved:
        print(json.dumps(resolved.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    print(resolved.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
