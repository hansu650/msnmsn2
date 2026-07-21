"""Config-driven, test-guarded command line interface for msn2026_v1.

The active entry point never imports APN or a dataset module.  Commands that
operate before test opening use only frozen configuration, manifests, and
filesystem metadata.  A future evaluator can enter through the explicit token
gate; this module never opens a test cell implicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .campaign import ProtocolLedger, ProtocolLedgerError
from .config import DEFAULT_CONFIG, ConfigError, ResolvedConfig, load_resolved_config
from .paths import PROJECT_ROOT, require_within_root
from .runtime import BlockedAssetError, ResolvedRunAssets, resolve_run_assets
from .schema import ManifestError, aggregation_eligibility, atomic_write_json


ACTIVE_METHOD_VERSION = "msn2026_v1"
EVALUATION_ARTIFACTS_REQUIRED_EXIT = 4
EVALUATION_FAILED_EXIT = 5
EVALUATOR_NOT_INTEGRATED_EXIT = EVALUATION_ARTIFACTS_REQUIRED_EXIT


class ActiveCliError(RuntimeError):
    """The active CLI cannot safely perform the requested operation."""


def _add_config_arguments(parser: argparse.ArgumentParser, *, selectors: bool) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    if selectors:
        parser.add_argument("--dataset", action="append", dest="datasets")
        parser.add_argument("--seed", action="append", dest="seeds", type=int)
        parser.add_argument("--variant", action="append", dest="variants")


def _selectors(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "datasets": getattr(args, "datasets", None),
            "seeds": getattr(args, "seeds", None),
            "variants": getattr(args, "variants", None),
        }.items()
        if value is not None
    }


def _resolved_config(args: argparse.Namespace) -> ResolvedConfig:
    return load_resolved_config(args.config, _selectors(args))


def _read_json(path: str | Path, *, label: str) -> Any:
    source = require_within_root(path, must_exist=True)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActiveCliError(f"Cannot read {label}: {exc}") from exc


def _project_relative(path: str | Path) -> str:
    resolved = require_within_root(path)
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _safe_block_detail(detail: str) -> str:
    return detail.replace(str(PROJECT_ROOT), "<PROJECT_ROOT>").replace(
        str(PROJECT_ROOT).replace("\\", "/"), "<PROJECT_ROOT>"
    )


def _protocols_for_dataset(
    config: ResolvedConfig,
    dataset_id: str,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    definitions = config["datasets"][dataset_id]
    available = tuple(
        dict.fromkeys(
            (
                str(definitions["release_protocol"]),
                str(definitions["strict_protocol"]),
            )
        )
    )
    if requested is None:
        return available
    known = set(config["protocols"])
    unknown = sorted(set(requested).difference(known))
    if unknown:
        raise ConfigError(f"Unknown protocol selection: {unknown}")
    requested_set = set(requested)
    return tuple(item for item in available if item in requested_set)


def _asset_record(
    assets: ResolvedRunAssets,
    *,
    check_assets: bool,
) -> tuple[str, dict[str, Any]]:
    inputs = assets.required_inputs()
    entries: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    invalid_type: list[str] = []
    for name, path in inputs.items():
        exists = path.exists()
        expected_directory = name in {"dataset_root", "apn_root"}
        valid_type = (
            path.is_dir() if exists and expected_directory
            else path.is_file() if exists
            else False
        )
        entries[name] = {
            "path": _project_relative(path),
            "exists": exists,
            "valid_type": valid_type,
        }
        if not exists:
            missing.append(name)
        elif not valid_type:
            invalid_type.append(name)

    if check_assets and (missing or invalid_type):
        status = "BLOCKED"
    elif all(item["exists"] and item["valid_type"] for item in entries.values()):
        status = "READY"
    else:
        status = "PLANNED"
    return status, {
        "status": status,
        "inputs": entries,
        "missing": sorted(missing),
        "invalid_type": sorted(invalid_type),
    }


def _cmd_audit(args: argparse.Namespace) -> int:
    config = _resolved_config(args)
    rows: list[dict[str, Any]] = []
    requested_protocols = tuple(args.protocols) if args.protocols else None
    for dataset_id in config["selection"]["datasets"]:
        for protocol_id in _protocols_for_dataset(
            config, str(dataset_id), requested_protocols
        ):
            for seed in config["selection"]["seeds"]:
                identity = {
                    "dataset": str(dataset_id),
                    "protocol": protocol_id,
                    "seed": int(seed),
                }
                try:
                    assets = resolve_run_assets(
                        config,
                        str(dataset_id),
                        int(seed),
                        protocol_id,
                        require_existing=False,
                    )
                    status, detail = _asset_record(
                        assets, check_assets=bool(args.check_assets)
                    )
                    rows.append({**identity, **detail})
                except BlockedAssetError as exc:
                    rows.append(
                        {
                            **identity,
                            "status": "BLOCKED",
                            "reason_code": exc.code,
                            "reason": _safe_block_detail(exc.detail),
                        }
                    )

    blocked = sum(row["status"] == "BLOCKED" for row in rows)
    payload = {
        "schema_version": "edgetwincal.active-audit.v1",
        "method_version": ACTIVE_METHOD_VERSION,
        "resolved_config_sha256": config.sha256,
        "assets_checked": bool(args.check_assets),
        "cell_count": len(rows),
        "blocked_count": blocked,
        "cells": rows,
    }
    _emit(payload)
    return 3 if blocked else 0


def _mapping_json(path: str | Path, *, label: str) -> Mapping[str, Any]:
    value = _read_json(path, label=label)
    if not isinstance(value, Mapping):
        raise ActiveCliError(f"{label} must be a JSON object")
    return value


def _cmd_ledger_create(args: argparse.Namespace) -> int:
    config = _resolved_config(args)
    components = _mapping_json(args.components, label="freeze components")
    pretest_checks = _mapping_json(args.pretest_checks, label="pre-test checks")
    ledger = ProtocolLedger.create(
        args.ledger,
        resolved_config=config,
        components=components,
        pretest_checks=pretest_checks,
    )
    _emit(
        {
            "schema_version": "edgetwincal.active-ledger-result.v1",
            "action": "create",
            "ledger": _project_relative(ledger.path),
            "status": ledger.status,
            "resolved_config_sha256": config.sha256,
            "test_opening_count": 0,
        }
    )
    return 0


def _cmd_ledger_freeze(args: argparse.Namespace) -> int:
    ledger = ProtocolLedger.load(args.ledger)
    protocol_sha256 = ledger.freeze()
    _emit(
        {
            "schema_version": "edgetwincal.active-ledger-result.v1",
            "action": "freeze",
            "ledger": _project_relative(ledger.path),
            "status": ledger.status,
            "protocol_sha256": protocol_sha256,
            "test_opening_count": len(ledger.data["test_openings"]),
        }
    )
    return 0


def _registry_values(value: Any) -> Sequence[Any]:
    if isinstance(value, Mapping):
        if set(value) != {"expected_manifests"}:
            raise ActiveCliError(
                "Manifest registry object must contain only expected_manifests"
            )
        value = value["expected_manifests"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ActiveCliError("Manifest registry must be a JSON array")
    return value


def _manifest_paths(args: argparse.Namespace, *, require_nonempty: bool) -> list[Path]:
    raw: list[Any] = list(getattr(args, "manifests", None) or [])
    registry = getattr(args, "registry", None)
    if registry is not None:
        raw.extend(
            _registry_values(_read_json(registry, label="manifest registry"))
        )
    if require_nonempty and not raw:
        raise ActiveCliError(
            "An explicit --manifest or --registry is required; discovery is disabled"
        )
    paths: list[Path] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (str, Path)):
            raise ActiveCliError(f"Manifest registry entry {index} is not a path")
        paths.append(require_within_root(item))
    canonical = [str(path).casefold() for path in paths]
    if len(set(canonical)) != len(canonical):
        raise ActiveCliError("Expected manifest registry contains duplicate paths")
    return paths


def _ledger_status(path: str | Path) -> dict[str, Any]:
    ledger = ProtocolLedger.load(path)
    openings = list(ledger.data.get("test_openings", {}).values())
    return {
        "path": _project_relative(ledger.path),
        "status": ledger.status,
        "resolved_config_sha256": ledger.data["resolved_config_sha256"],
        "protocol_sha256": ledger.data.get("protocol_sha256"),
        "test_opening_count": len(openings),
        "active_test_opening_count": sum(
            record.get("closed_at") is None for record in openings
        ),
        "event_count": len(ledger.data.get("events", [])),
    }


def _cmd_status(args: argparse.Namespace) -> int:
    manifest_paths = _manifest_paths(args, require_nonempty=False)
    if args.ledger is None and not manifest_paths:
        raise ActiveCliError("status requires --ledger, --manifest, or --registry")
    manifest_rows: list[dict[str, Any]] = []
    for path in manifest_paths:
        try:
            eligible, reason = aggregation_eligibility(path)
        except (OSError, ValueError) as exc:
            eligible, reason = False, f"unavailable:{type(exc).__name__}"
        manifest_rows.append(
            {
                "path": _project_relative(path),
                "eligible": eligible,
                "reason": _safe_block_detail(reason),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "edgetwincal.active-status.v1",
        "method_version": ACTIVE_METHOD_VERSION,
        "manifests": manifest_rows,
        "eligible_manifest_count": sum(row["eligible"] for row in manifest_rows),
    }
    if args.ledger is not None:
        payload["ledger"] = _ledger_status(args.ledger)
    _emit(payload)
    return 0 if all(row["eligible"] for row in manifest_rows) else 3


def _blocker_records(path: Path | None) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    value = _read_json(path, label="blocker registry")
    if isinstance(value, Mapping):
        if set(value) != {"blockers"}:
            raise ActiveCliError("Blocker registry object must contain only blockers")
        value = value["blockers"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ActiveCliError("Blocker registry must be a JSON array")
    if any(not isinstance(record, Mapping) for record in value):
        raise ActiveCliError("Every blocker registry entry must be a JSON object")
    return list(value)


def _optional_mapping(path: Path | None, *, label: str) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return _mapping_json(path, label=label)


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from .aggregate_v2 import (
        ConfirmatoryAggregationError,
        aggregate_confirmatory,
    )

    config = load_resolved_config(args.config)
    manifest_paths = _manifest_paths(args, require_nonempty=True)
    blockers = _blocker_records(args.blockers)
    edge = _optional_mapping(args.edge_measurement, label="edge measurement")
    try:
        result = aggregate_confirmatory(
            manifest_paths,
            blockers,
            config=config,
            edge_measurement=edge,
        )
    except ConfirmatoryAggregationError as exc:
        raise ActiveCliError(f"Confirmatory aggregation aborted: {exc}") from exc
    destination = atomic_write_json(args.output, result)
    _emit(
        {
            "schema_version": result["schema_version"],
            "output": _project_relative(destination),
            "input_audit": result["input_audit"],
            "gates": result["gates"],
        }
    )
    return 0
def _cmd_registry(args: argparse.Namespace) -> int:
    """Write the exact variant registry that must be hashed into a ledger."""

    from .campaign_runner import variant_registry

    config = _resolved_config(args)
    payload = variant_registry(config, args.dataset_id, args.protocol_id)
    destination = atomic_write_json(args.output, payload)
    _emit(
        {
            "schema_version": payload["schema_version"],
            "output": _project_relative(destination),
            "dataset": args.dataset_id,
            "protocol": args.protocol_id,
            "variant_count": len(payload["variants"]),
            "variant_registry_sha256": payload["variant_registry_sha256"],
        }
    )
    return 0






def _cmd_pretest_prepare(args: argparse.Namespace) -> int:
    from .campaign_pretest import prepare_pretest_manifests

    config = _resolved_config(args)
    bundle = (
        _mapping_json(args.strict_manifest_bundle, label="strict manifest bundle")
        if args.strict_manifest_bundle is not None
        else None
    )
    result = prepare_pretest_manifests(
        config,
        dataset_id=args.dataset_id,
        protocol_id=args.protocol_id,
        seed=args.preparation_seed,
        fold=args.fold,
        strict_public_bundle=bundle,
        padded_prediction_steps=args.padded_prediction_steps,
    )
    _emit(
        {
            "schema_version": "edgetwincal.pretest-prepare-result.v1",
            "dataset": args.dataset_id,
            "protocol": args.protocol_id,
            "test_constructed": False,
            "protocol_manifest": _project_relative(result.protocol_manifest),
            "split_manifest": _project_relative(result.split_manifest),
            "normalizer_manifest": _project_relative(result.normalizer_manifest),
            "protocol_manifest_sha256": result.protocol_manifest_sha256,
            "split_manifest_sha256": result.split_manifest_sha256,
            "normalizer_manifest_sha256": result.normalizer_manifest_sha256,
        }
    )
    return 0


def _cmd_pretest_fit(args: argparse.Namespace) -> int:
    from .campaign_pretest import fit_and_freeze_registry, load_fitted_registry

    config = _resolved_config(args)
    value = _read_json(args.entries, label="pretest fit entries")
    if not isinstance(value, Mapping) or set(value) != {"entries"}:
        raise ActiveCliError("Pretest fit entries must contain only entries")
    entries = value["entries"]
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ActiveCliError("Pretest fit entries must be a JSON array")
    destination = fit_and_freeze_registry(
        config,
        dataset_id=args.dataset_id,
        protocol_id=args.protocol_id,
        entries=entries,
        output_path=args.output,
        device=args.device,
        require_all_selected_seeds=not args.smoke_single_seed,
    )
    registry = load_fitted_registry(destination)
    _emit(
        {
            "schema_version": registry["schema_version"],
            "output": _project_relative(destination),
            "dataset": registry["dataset"],
            "protocol": registry["protocol"],
            "seed_count": len(registry["seeds"]),
            "variant_registry_sha256": registry["variant_registry_sha256"],
            "test_opened": False,
        }
    )
    return 0

def require_test_access(
    ledger_path: str | Path,
    *,
    cell_id: str,
    token: str,
) -> ProtocolLedger:
    """Validate a token created by exactly one frozen-ledger test opening."""

    ledger = ProtocolLedger.load(ledger_path)
    if ledger.status != "test_active":
        raise ActiveCliError(
            "Evaluation requires an active opening derived from a frozen ledger"
        )
    protocol_sha256 = ledger.data.get("protocol_sha256")
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        raise ActiveCliError("Ledger has no frozen protocol digest")
    opening = ledger.data.get("test_openings", {}).get(cell_id)
    if not isinstance(opening, Mapping):
        raise ActiveCliError("The requested once-only test cell is not open")
    if opening.get("protocol_sha256") != protocol_sha256:
        raise ActiveCliError("Test opening does not match the frozen protocol")
    ledger.validate_test_token(cell_id=cell_id, token=token)
    return ledger


def _cmd_evaluate(args: argparse.Namespace) -> int:
    provided = {
        "fitted_registry": args.fitted_registry,
        "test_cache_registry": args.test_cache_registry,
    }
    if not all(provided.values()):
        ledger = require_test_access(args.ledger, cell_id=args.cell_id, token=args.token)
        _emit(
            {
                "schema_version": "edgetwincal.active-evaluation-gate.v1",
                "status": "BLOCKED",
                "reason_code": "EVALUATION_ARTIFACTS_REQUIRED",
                "ledger": _project_relative(ledger.path),
                "cell_id": args.cell_id,
                "protocol_sha256": ledger.data["protocol_sha256"],
                "test_data_opened_by_cli": False,
                "missing": sorted(name for name, value in provided.items() if not value),
            }
        )
        return EVALUATION_ARTIFACTS_REQUIRED_EXIT

    from .campaign_evaluate import evaluate_campaign_once

    result = evaluate_campaign_once(
        ledger_path=args.ledger,
        cell_id=args.cell_id,
        token=args.token,
        fitted_registry_path=args.fitted_registry,
        test_cache_registry_path=args.test_cache_registry,
        run_root=args.run_root,
        device=args.device,
        argv=tuple(sys.argv),
        require_all_selected_seeds=True,
    )
    _emit(result)
    return 0 if result.get("status") == "complete" else EVALUATION_FAILED_EXIT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "EdgeTwinCal msn2026_v1 control plane. No command implicitly opens test data."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit", help="Resolve registered cells and optionally verify asset metadata."
    )
    _add_config_arguments(audit, selectors=True)
    audit.add_argument("--protocol", action="append", dest="protocols")
    audit.add_argument("--check-assets", action="store_true")
    audit.set_defaults(handler=_cmd_audit)

    ledger = subparsers.add_parser(
        "ledger", help="Create or freeze the pre-test protocol ledger."
    )
    ledger_actions = ledger.add_subparsers(dest="ledger_action", required=True)
    ledger_create = ledger_actions.add_parser(
        "create", help="Create a draft ledger from explicit frozen components."
    )
    _add_config_arguments(ledger_create, selectors=True)
    ledger_create.add_argument("--ledger", type=Path, required=True)
    ledger_create.add_argument("--components", type=Path, required=True)
    ledger_create.add_argument("--pretest-checks", type=Path, required=True)
    ledger_create.set_defaults(handler=_cmd_ledger_create)
    ledger_freeze = ledger_actions.add_parser(
        "freeze", help="Freeze a draft ledger only when every pre-test check passed."
    )
    ledger_freeze.add_argument("--ledger", type=Path, required=True)
    ledger_freeze.set_defaults(handler=_cmd_ledger_freeze)

    status = subparsers.add_parser(
        "status", help="Inspect ledger and explicit run-manifest status without data access."
    )
    status.add_argument("--ledger", type=Path)
    status.add_argument("--manifest", action="append", dest="manifests", type=Path)
    status.add_argument("--registry", type=Path)
    status.set_defaults(handler=_cmd_status)

    aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate an explicit complete-manifest registry fail-closed."
    )
    _add_config_arguments(aggregate, selectors=False)
    aggregate.add_argument("--manifest", action="append", dest="manifests", type=Path)
    aggregate.add_argument("--registry", type=Path)
    aggregate.add_argument("--blockers", type=Path)
    aggregate.add_argument("--edge-measurement", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(handler=_cmd_aggregate)


    pretest = subparsers.add_parser(
        "pretest",
        help="Prepare public manifests and freeze train/validation-fitted variant states.",
    )
    pretest_actions = pretest.add_subparsers(dest="pretest_action", required=True)
    pretest_prepare = pretest_actions.add_parser(
        "prepare",
        help="Create protocol/split/normalizer manifests without constructing test.",
    )
    _add_config_arguments(pretest_prepare, selectors=True)
    pretest_prepare.add_argument("--dataset-id", required=True)
    pretest_prepare.add_argument("--protocol-id", required=True)
    pretest_prepare.add_argument("--preparation-seed", type=int, required=True)
    pretest_prepare.add_argument("--fold", default="fold-0")
    pretest_prepare.add_argument("--strict-manifest-bundle", type=Path)
    pretest_prepare.add_argument("--padded-prediction-steps", type=int)
    pretest_prepare.set_defaults(handler=_cmd_pretest_prepare)

    pretest_fit = pretest_actions.add_parser(
        "fit",
        help="Fit all selected variants from explicit train+val caches and freeze states.",
    )
    _add_config_arguments(pretest_fit, selectors=True)
    pretest_fit.add_argument("--dataset-id", required=True)
    pretest_fit.add_argument("--protocol-id", required=True)
    pretest_fit.add_argument("--entries", type=Path, required=True)
    pretest_fit.add_argument("--output", type=Path, required=True)
    pretest_fit.add_argument(
        "--device", choices=("cpu", "cuda", "cuda:0"), default="cpu"
    )
    pretest_fit.add_argument(
        "--smoke-single-seed",
        action="store_true",
        help="Test-only escape hatch; confirmatory CLI requires every selected seed.",
    )
    pretest_fit.set_defaults(handler=_cmd_pretest_fit)


    registry = subparsers.add_parser(
        "registry",
        help="Freeze the config-derived variant registry before any test opening.",
    )
    _add_config_arguments(registry, selectors=True)
    registry.add_argument("--dataset-id", required=True)
    registry.add_argument("--protocol-id", required=True)
    registry.add_argument("--output", type=Path, required=True)
    registry.set_defaults(handler=_cmd_registry)

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Evaluate the complete frozen multi-seed registry under one once-only "
            "test opening, then consume the opening."
        ),
    )
    evaluate.add_argument("--ledger", type=Path, required=True)
    evaluate.add_argument("--cell-id", required=True)
    evaluate.add_argument("--token", required=True)
    evaluate.add_argument("--fitted-registry", type=Path)
    evaluate.add_argument("--test-cache-registry", type=Path)
    evaluate.add_argument("--run-root", type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    evaluate.set_defaults(handler=_cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BlockedAssetError as exc:
        print(f"BLOCKED[{exc.code}]: {_safe_block_detail(exc.detail)}", file=sys.stderr)
        return 3
    except (
        ActiveCliError,
        ConfigError,
        ProtocolLedgerError,
        ManifestError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


__all__ = (
    "ACTIVE_METHOD_VERSION",
    "EVALUATION_ARTIFACTS_REQUIRED_EXIT",
    "EVALUATION_FAILED_EXIT",
    "EVALUATOR_NOT_INTEGRATED_EXIT",
    "ActiveCliError",
    "require_test_access",
    "build_parser",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
