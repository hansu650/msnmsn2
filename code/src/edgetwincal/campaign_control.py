"""Production once-only control plane for frozen EdgeTwinCal campaigns.

The three state transitions are deliberately separate:

``prepare``
    Hash-verify G0/G1 evidence and compute every ``ProtocolLedger`` freeze
    component from the real fitted registry, protocol files, source tree,
    schemas, patch, and runtime environment.  It never constructs test data.
``freeze``
    Recompute the same material, reject any drift, then create and freeze one
    ledger for exactly one dataset/protocol/fold cell.  It never constructs
    test data.
``execute-once``
    Revalidate the frozen material, load and hash-check all five fitted states,
    open the cell exactly once, extract all five test caches sequentially, and
    evaluate the frozen registry.  A consumed opening is closed and sealed even
    when extraction or evaluation fails.

The command-line facade is read-only unless ``--execute`` is explicitly given.
Raw opening tokens are kept in local variables only; persisted artifacts contain
only the SHA256 already enforced by :class:`edgetwincal.campaign.ProtocolLedger`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .campaign import ProtocolLedger, ProtocolLedgerError
from .config import DEFAULT_CONFIG, ResolvedConfig, canonical_sha256, load_resolved_config
from .paths import PROJECT_ROOT, require_within_root
from .schema import atomic_write_json, sha256_file


PREPARED_CONTROL_SCHEMA = "edgetwincal.once-control-prepared.v1"
PRETEST_EVIDENCE_SCHEMA = "edgetwincal.pretest-evidence.v1"
PRETEST_CHECK_SCHEMA = "edgetwincal.pretest-check.v1"
EXECUTION_RESULT_SCHEMA = "edgetwincal.once-control-result.v1"
CUBLAS_WORKSPACE_VALUE = ":4096:8"
DEFAULT_FOLD = "fold-0"
DEFAULT_SEEDS = (2024, 2025, 2026, 2027, 2028)

RUNNABLE_CELLS: tuple[tuple[str, str], ...] = (
    ("P12", "release_parity"),
    ("HumanActivity", "release_parity"),
    ("USHCN", "release_parity"),
    ("P12", "strict_p12"),
    ("USHCN", "strict_ushcn"),
)
BLOCKED_CELLS: Mapping[tuple[str, str], tuple[str, str]] = {
    ("MIMIC_III", "release_parity"): (
        "missing_author_mapping",
        "released MIMIC-III parity requires the absent author UNIQUE_ID_dict.csv",
    )
}

REQUIRED_G0_CHECKS = (
    "unit_suite",
    "apn_forward_parity",
    "legacy_metric_parity",
)
REQUIRED_G1_CHECKS = (
    "cache_provenance",
    "split_normalization",
    "fitted_registry",
    "root_boundary",
)


class CampaignControlError(RuntimeError):
    """A frozen control artifact is missing, stale, unsafe, or inconsistent."""


class CampaignControlBlocked(CampaignControlError):
    """A declared campaign cell is unavailable for a known external reason."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"BLOCKED[{code}]: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class CellSpec:
    dataset_id: str
    protocol_id: str
    fold: str = DEFAULT_FOLD

    @property
    def cell_id(self) -> str:
        return f"{self.dataset_id}|{self.protocol_id}|{self.fold}"


@dataclass(frozen=True)
class FreezeMaterial:
    components: Mapping[str, Any]
    evidence: Mapping[str, Any]
    registry: Mapping[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignControlError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignControlError(f"{label} must be a JSON object")
    return value


def _relative(path: str | Path, *, must_exist: bool = True) -> str:
    return require_within_root(path, must_exist=must_exist).relative_to(PROJECT_ROOT).as_posix()


def _artifact(path: str | Path) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    if not source.is_file():
        raise CampaignControlError(f"Expected file artifact: {source}")
    return {
        "path": _relative(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _validate_artifact(record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "bytes", "sha256"}:
        raise CampaignControlError(f"{label} must contain path/bytes/sha256 only")
    source = require_within_root(str(record["path"]), must_exist=True)
    if not source.is_file():
        raise CampaignControlError(f"{label} is not a file")
    if int(record["bytes"]) != source.stat().st_size:
        raise CampaignControlError(f"{label} byte size drifted")
    if str(record["sha256"]) != sha256_file(source):
        raise CampaignControlError(f"{label} SHA256 drifted")
    return source


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = require_within_root(path)
    if destination.exists():
        raise CampaignControlError(f"Refusing to overwrite frozen artifact: {destination}")
    return atomic_write_json(destination, value)


def _write_new_text(path: str | Path, value: str) -> Path:
    destination = require_within_root(path)
    if destination.exists():
        raise CampaignControlError(f"Refusing to overwrite frozen artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _canonical_file_set(paths: Sequence[Path]) -> tuple[str, list[dict[str, Any]]]:
    resolved = sorted(
        {require_within_root(path, must_exist=True) for path in paths},
        key=lambda item: item.relative_to(PROJECT_ROOT).as_posix(),
    )
    if not resolved:
        raise CampaignControlError("A frozen source component cannot be empty")
    records = [_artifact(path) for path in resolved]
    return canonical_sha256({"files": records}), records


def _ensure_cublas_workspace() -> Mapping[str, str]:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_VALUE
        observed = CUBLAS_WORKSPACE_VALUE
    if observed != CUBLAS_WORKSPACE_VALUE:
        raise CampaignControlError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with deterministic campaign value "
            f"{CUBLAS_WORKSPACE_VALUE}: {observed}"
        )
    return {"required": CUBLAS_WORKSPACE_VALUE, "observed": observed}


def resolve_cell(dataset_id: str, protocol_id: str, *, fold: str = DEFAULT_FOLD) -> CellSpec:
    key = (str(dataset_id), str(protocol_id))
    if key in BLOCKED_CELLS:
        code, detail = BLOCKED_CELLS[key]
        raise CampaignControlBlocked(code, detail)
    if key not in RUNNABLE_CELLS:
        raise CampaignControlError(f"Unsupported once-only campaign cell: {key}")
    if not fold or "|" in fold:
        raise CampaignControlError("Fold must be a non-empty token without '|'")
    return CellSpec(key[0], key[1], fold)


def default_paths(config: ResolvedConfig, cell: CellSpec) -> Mapping[str, Path]:
    root = require_within_root(
        Path(str(config["paths"]["protocol_root"])) / cell.dataset_id / cell.protocol_id
    )
    evidence = require_within_root(
        Path(str(config["paths"]["artifact_root"]))
        / "pretest"
        / cell.dataset_id
        / cell.protocol_id
        / "pretest_evidence.json"
    )
    return {
        "fitted_registry": root / "fitted_registry.json",
        "pretest_evidence": evidence,
        "prepared_control": root / "once_control_prepared.json",
        "ledger": root / "protocol_ledger.json",
        "test_registry": root / "test_cache_registry.json",
        "execution_result": root / "once_campaign_result.json",
    }


def campaign_plan(
    config: ResolvedConfig,
    *,
    datasets: Sequence[str] | None = None,
    protocols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_filter = None if datasets is None else set(datasets)
    protocol_filter = None if protocols is None else set(protocols)
    rows: list[dict[str, Any]] = []
    for dataset_id, protocol_id in (*RUNNABLE_CELLS, *BLOCKED_CELLS):
        if dataset_filter is not None and dataset_id not in dataset_filter:
            continue
        if protocol_filter is not None and protocol_id not in protocol_filter:
            continue
        cell = CellSpec(dataset_id, protocol_id)
        paths = default_paths(config, cell)
        blocked = BLOCKED_CELLS.get((dataset_id, protocol_id))
        rows.append(
            {
                "cell_id": cell.cell_id,
                "dataset": dataset_id,
                "protocol": protocol_id,
                "status": (
                    f"BLOCKED[{blocked[0]}]" if blocked else "runnable_prepare_freeze_execute_once"
                ),
                "fitted_registry": _relative(paths["fitted_registry"], must_exist=False),
                "pretest_evidence": _relative(paths["pretest_evidence"], must_exist=False),
                "prepared_control": _relative(paths["prepared_control"], must_exist=False),
                "ledger": _relative(paths["ledger"], must_exist=False),
            }
        )
    return rows


def _production_runtime() -> SimpleNamespace:
    """Import torch/vendor-facing modules only after cuBLAS is established."""

    from . import backbone_campaign as backbone
    from .apn_bridge import load_frozen_apn_checkpoint
    from .campaign_evaluate import (
        evaluate_campaign_once,
        extract_test_cache_after_open,
        write_test_cache_registry,
    )
    from .campaign_pretest import load_fitted_registry, load_fitted_states
    from .campaign_runner import load_cache_manifest
    from .fit_cache_campaign import build_provenance_bundle
    from .provenance import CACHE_SCHEMA_VERSION
    from .runtime_v2 import TENSOR_PAYLOAD_SCHEMA_VERSION, collect_environment
    from .timing import REQUIRED_PHASES

    components = backbone._runtime_components()

    def prepare_cell(config: ResolvedConfig, cell: CellSpec) -> Any:
        return backbone.prepare_cell(
            config, backbone.BackboneCell(cell.dataset_id, cell.protocol_id)
        )

    def build_runtime(config: ResolvedConfig, prepared: Any, seed: int) -> Any:
        return backbone._build_train_val_runtime(config, prepared, seed, components)

    def load_checkpoint(bundle: Any, checkpoint: Path, *, device: str) -> Any:
        return load_frozen_apn_checkpoint(
            bundle,
            checkpoint,
            device=device,
            model_factory=components.make_apn_model_factory(bundle),
        )

    return SimpleNamespace(
        prepare_cell=prepare_cell,
        resolve_run_assets=components.resolve_run_assets,
        build_runtime=build_runtime,
        load_checkpoint=load_checkpoint,
        load_fitted_registry=load_fitted_registry,
        load_fitted_states=load_fitted_states,
        load_cache_manifest=load_cache_manifest,
        build_provenance_bundle=build_provenance_bundle,
        extraction_provenance_type=(
            __import__("edgetwincal.runtime_v2", fromlist=["ExtractionProvenance"])
            .ExtractionProvenance
        ),
        collect_environment=collect_environment,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        tensor_payload_schema_version=TENSOR_PAYLOAD_SCHEMA_VERSION,
        timing_required_phases=tuple(sorted(REQUIRED_PHASES)),
        extract_test_cache_after_open=extract_test_cache_after_open,
        write_test_cache_registry=write_test_cache_registry,
        evaluate_campaign_once=evaluate_campaign_once,
    )


def _schema_component(
    *,
    name: str,
    contract: Mapping[str, Any],
    source_names: Sequence[str],
) -> tuple[str, Mapping[str, Any]]:
    files = [require_within_root(PROJECT_ROOT / value, must_exist=True) for value in source_names]
    source_digest, source_records = _canonical_file_set(files)
    evidence = {
        "name": name,
        "contract": dict(contract),
        "source_sha256": source_digest,
        "sources": source_records,
    }
    return canonical_sha256(evidence), evidence


def _source_component() -> tuple[str, Mapping[str, Any]]:
    source_root = require_within_root(PROJECT_ROOT / "code/src/edgetwincal", must_exist=True)
    test_root = require_within_root(PROJECT_ROOT / "code/tests", must_exist=True)
    files = list(source_root.glob("*.py"))
    files.extend(test_root.glob("test_edgetwincal*.py"))
    files.extend(
        [
            require_within_root(PROJECT_ROOT / "code/configs/msn2026/default.json", must_exist=True),
            require_within_root(PROJECT_ROOT / "code/scripts/run_edgetwincal_once.py", must_exist=True),
        ]
    )
    digest, records = _canonical_file_set(files)
    return digest, {"scope": "active_source_config_and_tests", "files": records}


def _validate_registry_and_states(
    config: ResolvedConfig,
    cell: CellSpec,
    fitted_registry_path: str | Path,
    prepared: Any,
    runtime: Any,
    *,
    device: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    registry_path = require_within_root(fitted_registry_path, must_exist=True)
    registry = runtime.load_fitted_registry(registry_path)
    if (registry.get("dataset"), registry.get("protocol")) != (
        cell.dataset_id,
        cell.protocol_id,
    ):
        raise CampaignControlError("Fitted registry identity differs from the selected cell")
    if registry.get("resolved_config_sha256") != config.sha256:
        raise CampaignControlError("Fitted registry resolved-config hash drifted")
    if registry.get("test_opened") is not False:
        raise CampaignControlError("Fitted registry must state test_opened=false")
    rows = list(registry.get("seeds", []))
    seeds = tuple(int(row.get("seed", -1)) for row in rows)
    expected = tuple(int(value) for value in config["selection"]["seeds"])
    if expected != DEFAULT_SEEDS or seeds != expected:
        raise CampaignControlError(
            f"Once-only registry must contain the five ordered seeds: {seeds} != {expected}"
        )
    split_sha = sha256_file(prepared.split_manifest)
    normalizer_sha = sha256_file(prepared.normalizer_manifest)
    cache_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split_manifest_sha256") != split_sha:
            raise CampaignControlError("Fitted registry split hash differs from frozen file")
        if row.get("normalizer_manifest_sha256") != normalizer_sha:
            raise CampaignControlError("Fitted registry normalizer hash differs from frozen file")
        sidecar = require_within_root(row["fit_cache_manifest_path"], must_exist=True)
        manifest = runtime.load_cache_manifest(sidecar)
        if manifest.manifest_digest() != row.get("fit_cache_manifest_sha256"):
            raise CampaignControlError("Fit-cache manifest changed after state fitting")
        if (manifest.dataset_id, int(manifest.seed)) != (cell.dataset_id, int(row["seed"])):
            raise CampaignControlError("Fit-cache identity differs from fitted registry")
        if manifest.split_manifest_sha256 != split_sha:
            raise CampaignControlError("Fit-cache split hash differs from frozen file")
        if manifest.normalizer_manifest_sha256 != normalizer_sha:
            raise CampaignControlError("Fit-cache normalizer hash differs from frozen file")
        runtime.load_fitted_states(config, row, device=device)
        cache_rows.append(
            {
                "seed": int(row["seed"]),
                "fit_cache_manifest": _artifact(sidecar),
                "manifest_digest": manifest.manifest_digest(),
                "state": _artifact(row["state_path"]),
                "state_content_sha256": str(row["state_content_sha256"]),
            }
        )
    return registry, {
        "registry": _artifact(registry_path),
        "variant_registry_sha256": str(registry["variant_registry_sha256"]),
        "ordered_seeds": list(seeds),
        "all_fitted_states_loaded_and_hash_verified": True,
        "fit_cache_rows": cache_rows,
    }


def compute_freeze_material(
    config: ResolvedConfig,
    cell: CellSpec,
    fitted_registry_path: str | Path,
    *,
    runtime: Any | None = None,
    state_device: str = "cpu",
) -> FreezeMaterial:
    """Compute all required ledger components from concrete frozen artifacts."""

    _ensure_cublas_workspace()
    active = runtime or _production_runtime()
    prepared = active.prepare_cell(config, cell)
    protocol_payload = _read_json(prepared.protocol_manifest, label="protocol manifest")
    if protocol_payload.get("test_constructed") is not False:
        raise CampaignControlError("Protocol manifest must state test_constructed=false")
    if (
        protocol_payload.get("dataset"),
        protocol_payload.get("protocol"),
        protocol_payload.get("resolved_config_sha256"),
    ) != (cell.dataset_id, cell.protocol_id, config.sha256):
        raise CampaignControlError("Protocol manifest identity drifted")

    registry, registry_evidence = _validate_registry_and_states(
        config,
        cell,
        fitted_registry_path,
        prepared,
        active,
        device=state_device,
    )
    split_sha = sha256_file(prepared.split_manifest)
    normalizer_sha = sha256_file(prepared.normalizer_manifest)

    cache_sha, cache_evidence = _schema_component(
        name="schema3_fit_and_once_test_cache",
        contract={
            "cache_manifest_schema": int(active.cache_schema_version),
            "tensor_payload_schema": int(active.tensor_payload_schema_version),
            "fit_splits": ["train", "val"],
            "test_splits": ["test"],
        },
        source_names=(
            "code/src/edgetwincal/provenance.py",
            "code/src/edgetwincal/runtime_v2.py",
            "code/src/edgetwincal/campaign_extract.py",
            "code/src/edgetwincal/campaign_evaluate.py",
        ),
    )
    statistics_sha, statistics_evidence = _schema_component(
        name="crossed_group_checkpoint_statistics",
        contract={
            "bootstrap": dict(config["bootstrap"]),
            "G2": dict(config["gates"]["G2"]),
            "G3": dict(config["gates"]["G3"]),
        },
        source_names=(
            "code/src/edgetwincal/statistics.py",
            "code/src/edgetwincal/aggregate_v2.py",
        ),
    )
    timing_sha, timing_evidence = _schema_component(
        name="segmented_timing",
        contract={
            "timing": dict(config["timing"]),
            "required_phases": list(active.timing_required_phases),
            "evaluation_schema": "edgetwincal.timing.v3",
        },
        source_names=(
            "code/src/edgetwincal/timing.py",
            "code/src/edgetwincal/campaign_runner.py",
            "code/src/edgetwincal/campaign_evaluate.py",
        ),
    )
    project_sha, project_evidence = _source_component()
    patch_path = require_within_root(config["apn"]["patch"], must_exist=True)
    patch_sha = sha256_file(patch_path)
    environment_value = {
        "environment": active.collect_environment(),
        "cublas_workspace": dict(_ensure_cublas_workspace()),
        "execution": "single_process_sequential",
    }
    environment_sha = canonical_sha256(environment_value)

    components = {
        "variant_registry_sha256": str(registry["variant_registry_sha256"]),
        "split_manifests": {cell.cell_id: split_sha},
        "normalization_manifests": {cell.cell_id: normalizer_sha},
        "cache_schema_sha256": cache_sha,
        "statistics_sha256": statistics_sha,
        "timing_schema_sha256": timing_sha,
        "project_source_sha256": project_sha,
        "apn_patch_sha256": patch_sha,
        "environment_sha256": environment_sha,
    }
    evidence = {
        "protocol_manifest": _artifact(prepared.protocol_manifest),
        "split_manifest": _artifact(prepared.split_manifest),
        "normalizer_manifest": _artifact(prepared.normalizer_manifest),
        "registry_and_states": registry_evidence,
        "cache_schema": cache_evidence,
        "statistics": statistics_evidence,
        "timing": timing_evidence,
        "project_source": project_evidence,
        "apn_patch": _artifact(patch_path),
        "environment": environment_value,
        "test_constructed": False,
    }
    return FreezeMaterial(components=components, evidence=evidence, registry=registry)


def _finite_nonnegative(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise CampaignControlError(f"{label} must be finite and nonnegative")
    return result


def _validate_check_observations(
    check_id: str,
    observations: Mapping[str, Any],
    *,
    config: ResolvedConfig,
    cell: CellSpec,
    material: FreezeMaterial,
) -> None:
    seeds = list(DEFAULT_SEEDS)
    if check_id == "unit_suite":
        if int(observations.get("exit_code", -1)) != 0:
            raise CampaignControlError("G0 unit suite did not exit successfully")
        if int(observations.get("passed", 0)) <= 0:
            raise CampaignControlError("G0 unit suite contains no passing tests")
        if int(observations.get("failed", -1)) != 0 or int(observations.get("errors", -1)) != 0:
            raise CampaignControlError("G0 unit suite contains failures/errors")
    elif check_id == "apn_forward_parity":
        atol = _finite_nonnegative(observations.get("atol"), label="parity atol")
        rtol = _finite_nonnegative(observations.get("rtol"), label="parity rtol")
        if atol > float(config["gates"]["G0"]["legacy_parity_atol"]):
            raise CampaignControlError("APN parity atol is looser than the frozen gate")
        if rtol > float(config["gates"]["G0"]["legacy_parity_rtol"]):
            raise CampaignControlError("APN parity rtol is looser than the frozen gate")
        if observations.get("allclose") is not True:
            raise CampaignControlError("APN forward parity did not pass")
        _finite_nonnegative(observations.get("max_abs_error"), label="max_abs_error")
        _finite_nonnegative(observations.get("max_rel_error"), label="max_rel_error")
    elif check_id == "legacy_metric_parity":
        error = _finite_nonnegative(observations.get("max_abs_error"), label="metric parity error")
        tolerance = _finite_nonnegative(observations.get("tolerance"), label="metric parity tolerance")
        if tolerance <= 0 or error > tolerance:
            raise CampaignControlError("Legacy metric parity exceeds its evidence tolerance")
    elif check_id == "cache_provenance":
        required = (
            observations.get("cache_schema_version") == 3,
            observations.get("ordered_seeds") == seeds,
            observations.get("all_fit_cache_manifests_verified") is True,
            observations.get("stale_cache_rejected") is True,
            observations.get("corrupt_cache_rejected") is True,
        )
        if not all(required):
            raise CampaignControlError("G1 cache provenance evidence is incomplete")
    elif check_id == "split_normalization":
        split = material.components["split_manifests"][cell.cell_id]
        normalizer = material.components["normalization_manifests"][cell.cell_id]
        required = (
            observations.get("split_manifest_sha256") == split,
            observations.get("normalizer_manifest_sha256") == normalizer,
            observations.get("test_constructed") is False,
        )
        if not all(required):
            raise CampaignControlError("G1 split/normalizer evidence differs from frozen files")
        if cell.protocol_id == "release_parity":
            if observations.get("released_behavior_explicit") is not True:
                raise CampaignControlError("Release behavior must be explicitly labelled")
        elif not (
            observations.get("group_disjoint") is True
            and observations.get("train_only_normalization") is True
        ):
            raise CampaignControlError("Strict G1 split/normalization evidence did not pass")
    elif check_id == "fitted_registry":
        if observations.get("variant_registry_sha256") != material.components["variant_registry_sha256"]:
            raise CampaignControlError("G1 fitted-registry digest differs from frozen registry")
        if observations.get("ordered_seeds") != seeds:
            raise CampaignControlError("G1 fitted-registry seed order differs")
        if observations.get("all_state_hashes_verified") is not True:
            raise CampaignControlError("G1 fitted-state hashes were not verified")
    elif check_id == "root_boundary":
        required = (
            observations.get("escape_rejected") is True,
            observations.get("home_unchanged") is True,
            observations.get("codex_home_unchanged") is True,
            observations.get("single_process") is True,
        )
        if not all(required):
            raise CampaignControlError("G1 root/single-process evidence is incomplete")
    else:  # pragma: no cover - exact check set is validated first.
        raise CampaignControlError(f"Unknown pre-test check: {check_id}")


def _display_argv(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in argv:
        text = str(value)
        candidate = Path(text)
        if candidate.is_absolute():
            try:
                text = candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                pass
        result.append(text)
    return result


def _run_evidence_command(
    name: str,
    argv: Sequence[str],
    *,
    output_directory: Path,
    timeout_seconds: int = 900,
) -> Mapping[str, Any]:
    """Run a shell-free check and persist byte-hashed stdout/stderr evidence."""

    directory = require_within_root(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path = directory / f"{name}.stdout.txt"
    stderr_path = directory / f"{name}.stderr.txt"
    started_at = _now()
    started = time.perf_counter()
    environment = os.environ.copy()
    source_root = str(require_within_root(PROJECT_ROOT / "code/src", must_exist=True))
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not current_pythonpath
        else source_root + os.pathsep + current_pythonpath
    )
    completed = subprocess.run(
        [str(value) for value in argv],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    wall_seconds = time.perf_counter() - started
    _write_new_text(stdout_path, completed.stdout)
    _write_new_text(stderr_path, completed.stderr)
    return {
        "argv": _display_argv(argv),
        "exit_code": int(completed.returncode),
        "started_at": started_at,
        "finished_at": _now(),
        "wall_seconds": float(wall_seconds),
        "stdout": _artifact(stdout_path),
        "stderr": _artifact(stderr_path),
    }


def _junit_counts(path: str | Path) -> Mapping[str, int]:
    source = require_within_root(path, must_exist=True)
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        raise CampaignControlError(f"Cannot parse pytest JUnit evidence: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise CampaignControlError("JUnit evidence contains no test suite")
    leaves = [suite for suite in suites if not suite.findall("./testsuite")]
    counted = leaves or suites
    def total(name: str) -> int:
        return sum(int(float(suite.attrib.get(name, 0))) for suite in counted)

    return {
        "tests": total("tests"),
        "failures": total("failures"),
        "errors": total("errors"),
        "skipped": total("skipped"),
    }


def _apn_forward_parity_observations(config: ResolvedConfig) -> Mapping[str, Any]:
    """Numerically compare pinned upstream APN with patched APN-mode forward."""

    import types
    import torch
    from .apn_bridge import _ensure_vendor_import_path

    _ensure_vendor_import_path()
    from models.APN import Model as PatchedModel

    apn_root = require_within_root(config["paths"]["apn_root"], must_exist=True)
    git_argv = (
        "git", "-c", f"safe.directory={apn_root.as_posix()}", "-C", str(apn_root),
        "show", f"{config['apn']['commit']}:models/APN.py",
    )
    started_at = _now()
    started = time.perf_counter()
    completed = subprocess.run(
        git_argv, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise CampaignControlError(f"Cannot load pinned upstream APN source: {completed.stderr}")
    upstream_module = types.ModuleType("edgetwincal_pinned_upstream_apn")
    exec(compile(completed.stdout, "pinned_upstream_models_APN.py", "exec"), upstream_module.__dict__)
    model_config = SimpleNamespace(
        task_name="short_term_forecast", d_model=24, apn_te_dim=8, enc_in=4,
        apn_npatch=5, apn_nlayer=2, apn_attn_heads=8, dropout=0.1,
        features="M", evipatch_mode="apn", evipatch_random_seed=1729,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        upstream = upstream_module.Model(model_config).eval()
        patched = PatchedModel(model_config).eval()
        migration = patched.load_state_dict(upstream.state_dict(), strict=True)
        torch.manual_seed(99)
        batch_size, history, horizon, channels = 3, 13, 3, 4
        batch = {
            "x": torch.randn(batch_size, history, channels),
            "x_mark": torch.linspace(0, 0.74, history).view(1, history, 1).expand(batch_size, -1, -1),
            "x_mask": (torch.rand(batch_size, history, channels) > 0.2).float(),
            "y": torch.randn(batch_size, horizon, channels),
            "y_mark": torch.linspace(0.75, 0.99, horizon).view(1, horizon, 1).expand(batch_size, -1, -1),
            "y_mask": torch.ones(batch_size, horizon, channels),
        }
        with torch.no_grad():
            upstream_output = upstream(**batch)
            patched_output = patched(**batch)
    reference = upstream_output["pred"]
    actual = patched_output["pred"]
    difference = (actual - reference).abs()
    relative = difference / reference.abs().clamp_min(1.0e-12)
    atol = float(config["gates"]["G0"]["legacy_parity_atol"])
    rtol = float(config["gates"]["G0"]["legacy_parity_rtol"])
    observations = {
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max().item()),
        "max_rel_error": float(relative.max().item()),
        "target_equal": bool(torch.equal(patched_output["true"], upstream_output["true"])),
        "mask_equal": bool(torch.equal(patched_output["mask"], upstream_output["mask"])),
        "state_migration_missing": list(migration.missing_keys),
        "state_migration_unexpected": list(migration.unexpected_keys),
        "deterministic_seeds": {"model": 123, "batch": 99},
        "source_command": {
            "argv": _display_argv(git_argv), "exit_code": int(completed.returncode),
            "started_at": started_at, "finished_at": _now(),
            "wall_seconds": float(wall_seconds),
            "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        },
    }
    if not (
        observations["allclose"] and observations["target_equal"]
        and observations["mask_equal"] and not observations["state_migration_missing"]
        and not observations["state_migration_unexpected"]
    ):
        raise CampaignControlError(f"APN forward parity failed: {observations}")
    return observations


def _legacy_metric_parity_observations() -> Mapping[str, Any]:
    """Re-audit immutable legacy-v1 P12 cache/output/metric parity."""

    import numpy as np
    import torch

    root = require_within_root(PROJECT_ROOT / "results/edgetwincal", must_exist=True)
    tolerance = 1.0e-12
    maximum = 0.0
    rows: list[dict[str, Any]] = []
    for seed in (2024, 2025, 2026):
        cache_path = require_within_root(root / "cache" / f"apn_seed{seed}.pt", must_exist=True)
        output_path = require_within_root(root / f"seed_{seed}" / "test_outputs.npz", must_exist=True)
        metric_path = require_within_root(root / f"seed_{seed}" / "metrics.json", must_exist=True)
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(cache, Mapping) or "test" not in cache:
            raise CampaignControlError(f"Invalid legacy cache for seed {seed}")
        split = cache["test"]
        with np.load(output_path, allow_pickle=False) as output:
            pairs = (
                (split["base_prediction"].cpu().numpy(), output["prediction_apn"]),
                (split["target"].cpu().numpy(), output["target"]),
                (split["mask"].cpu().numpy(), output["target_mask"]),
                (split["sample_id"].cpu().numpy(), output["sample_id"]),
            )
            array_error = max(
                float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                if left.size else 0.0 for left, right in pairs
            )
            mask = output["target_mask"] > 0
            error = (output["prediction_apn"][mask] - output["target"][mask]).astype(np.float64)
            mse = float(np.square(error).mean())
            mae = float(np.abs(error).mean())
            observed = int(error.size)
        metrics = _read_json(metric_path, label=f"legacy metrics seed {seed}")["test_metrics"]["apn"]
        metric_error = max(
            abs(mse - float(metrics["mse"])), abs(mae - float(metrics["mae"])),
            float(abs(observed - int(metrics["observed_targets"]))),
        )
        seed_error = max(array_error, metric_error)
        maximum = max(maximum, seed_error)
        rows.append({
            "seed": seed, "array_max_abs_error": array_error,
            "metric_max_abs_error": metric_error, "observed_targets": observed,
            "cache": _artifact(cache_path), "outputs": _artifact(output_path),
            "metrics": _artifact(metric_path),
        })
    if maximum > tolerance:
        raise CampaignControlError(f"Legacy metric parity exceeded tolerance: {maximum} > {tolerance}")
    return {
        "max_abs_error": maximum,
        "tolerance": tolerance,
        "arithmetic": "saved_float32_subtraction_then_float64_reduction",
        "seeds": rows,
    }


def _split_normalization_observations(
    cell: CellSpec, material: FreezeMaterial
) -> Mapping[str, Any]:
    split_path = _validate_artifact(
        material.evidence["split_manifest"], label="material split manifest"
    )
    normalizer_path = _validate_artifact(
        material.evidence["normalizer_manifest"], label="material normalizer manifest"
    )
    protocol_path = _validate_artifact(
        material.evidence["protocol_manifest"], label="material protocol manifest"
    )
    split = _read_json(split_path, label="split manifest")
    normalizer = _read_json(normalizer_path, label="normalizer manifest")
    protocol = _read_json(protocol_path, label="protocol manifest")
    observations: dict[str, Any] = {
        "split_manifest_sha256": material.components["split_manifests"][cell.cell_id],
        "normalizer_manifest_sha256": material.components["normalization_manifests"][cell.cell_id],
        "test_constructed": protocol.get("test_constructed"),
        "released_behavior_explicit": cell.protocol_id == "release_parity",
        "group_disjoint": None,
        "train_only_normalization": None,
        "support": {
            "protocol": _artifact(protocol_path),
            "split": _artifact(split_path),
            "normalizer": _artifact(normalizer_path),
        },
    }
    if cell.protocol_id != "release_parity":
        public = split.get("public_protocol_manifest", {})
        groups = public.get("public_group_hashes", {})
        if not isinstance(groups, Mapping) or set(groups) != {"train", "val", "test"}:
            raise CampaignControlError("Strict split lacks public hashed group partitions")
        sets = {name: set(values) for name, values in groups.items()}
        disjoint = not (
            sets["train"] & sets["val"]
            or sets["train"] & sets["test"]
            or sets["val"] & sets["test"]
        )
        observations["group_disjoint"] = disjoint
        observations["train_only_normalization"] = (
            normalizer.get("fit_scope") == "observed_train_only"
        )
    return observations


def _root_boundary_observations() -> Mapping[str, Any]:
    home_before = os.environ.get("HOME")
    codex_before = os.environ.get("CODEX_HOME")
    escaped = PROJECT_ROOT.parent / "edgetwincal_outside_sentinel"
    rejected = False
    try:
        require_within_root(escaped)
    except ValueError:
        rejected = True
    return {
        "escape_rejected": rejected,
        "home_unchanged": os.environ.get("HOME") == home_before,
        "codex_home_unchanged": os.environ.get("CODEX_HOME") == codex_before,
        "single_process": True,
        "process_id_recorded": int(os.getpid()),
        "execution_contract": "one_process_sequential_no_worker_pool",
    }


def _check_payload(
    config: ResolvedConfig,
    cell: CellSpec,
    *,
    gate: str,
    check_id: str,
    observations: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema_version": PRETEST_CHECK_SCHEMA,
        "gate": gate,
        "check_id": check_id,
        "status": "PASS",
        "resolved_config_sha256": config.sha256,
        "cell_id": cell.cell_id,
        "observations": dict(observations),
        "recorded_at": _now(),
    }


def generate_pretest_evidence(
    config: ResolvedConfig,
    cell: CellSpec,
    *,
    fitted_registry_path: str | Path,
    output_path: str | Path,
    execute: bool = False,
    runtime: Any | None = None,
) -> Mapping[str, Any]:
    """Generate seven derived, hash-bound G0/G1 check artifacts plus manifest."""

    destination = require_within_root(output_path)
    evidence_root = require_within_root(destination.parent)
    if not execute:
        return {
            "mode": "read_only_evidence_plan",
            "cell_id": cell.cell_id,
            "output": _relative(destination, must_exist=False),
            "checks": {"G0": list(REQUIRED_G0_CHECKS), "G1": list(REQUIRED_G1_CHECKS)},
            "commands": [
                "python -m pytest code/tests -q -p no:cacheprovider --junitxml <support/full_suite.xml>",
                "python -m pytest <cache-reuse/corrupt/stale nodes> -q -p no:cacheprovider --junitxml <support/cache_guards.xml>",
            ],
            "test_constructed": False,
        }
    if destination.exists() or (evidence_root.exists() and any(evidence_root.iterdir())):
        raise CampaignControlError(f"Refusing non-empty evidence destination: {evidence_root}")
    evidence_root.mkdir(parents=True, exist_ok=True)
    support_root = require_within_root(evidence_root / "support")
    checks_root = require_within_root(evidence_root / "checks")
    support_root.mkdir(parents=True, exist_ok=False)
    checks_root.mkdir(parents=True, exist_ok=False)

    material = compute_freeze_material(
        config, cell, fitted_registry_path, runtime=runtime, state_device="cpu"
    )
    python = str(Path(sys.executable).resolve())
    full_junit = support_root / "full_suite.xml"
    full_command = _run_evidence_command(
        "full_suite",
        (
            python, "-m", "pytest", "code/tests", "-q", "-p", "no:cacheprovider",
            "--junitxml", str(full_junit),
        ),
        output_directory=support_root,
    )
    full_counts = _junit_counts(full_junit)
    if full_command["exit_code"] != 0 or full_counts["failures"] or full_counts["errors"]:
        raise CampaignControlError(
            f"Full pre-test suite failed: command={full_command}, counts={full_counts}"
        )

    cache_junit = support_root / "cache_guards.xml"
    cache_command = _run_evidence_command(
        "cache_guards",
        (
            python, "-m", "pytest",
            "code/tests/test_edgetwincal_fit_cache_campaign.py::test_extract_then_reuse_is_train_val_only_and_provenance_drift_fails",
            "code/tests/test_edgetwincal_provenance.py::test_corrupt_payload_is_rejected_before_bytes_are_returned",
            "code/tests/test_edgetwincal_provenance.py::test_stale_cache_reports_the_exact_changed_fields_on_load",
            "-q", "-p", "no:cacheprovider", "--junitxml", str(cache_junit),
        ),
        output_directory=support_root,
    )
    cache_counts = _junit_counts(cache_junit)
    if cache_command["exit_code"] != 0 or cache_counts["failures"] or cache_counts["errors"]:
        raise CampaignControlError(
            f"Cache guard suite failed: command={cache_command}, counts={cache_counts}"
        )

    registry_evidence = material.evidence["registry_and_states"]
    observations: Mapping[str, Mapping[str, Any]] = {
        "unit_suite": {
            "exit_code": int(full_command["exit_code"]),
            "passed": int(full_counts["tests"] - full_counts["skipped"]),
            "failed": int(full_counts["failures"]),
            "errors": int(full_counts["errors"]),
            "skipped": int(full_counts["skipped"]),
            "command": full_command,
            "junit": _artifact(full_junit),
        },
        "apn_forward_parity": _apn_forward_parity_observations(config),
        "legacy_metric_parity": _legacy_metric_parity_observations(),
        "cache_provenance": {
            "cache_schema_version": 3,
            "ordered_seeds": list(DEFAULT_SEEDS),
            "all_fit_cache_manifests_verified": bool(
                registry_evidence["all_fitted_states_loaded_and_hash_verified"]
            ),
            "stale_cache_rejected": cache_command["exit_code"] == 0,
            "corrupt_cache_rejected": cache_command["exit_code"] == 0,
            "command": cache_command,
            "junit": _artifact(cache_junit),
            "fit_cache_rows": registry_evidence["fit_cache_rows"],
        },
        "split_normalization": _split_normalization_observations(cell, material),
        "fitted_registry": {
            "variant_registry_sha256": material.components["variant_registry_sha256"],
            "ordered_seeds": list(DEFAULT_SEEDS),
            "all_state_hashes_verified": bool(
                registry_evidence["all_fitted_states_loaded_and_hash_verified"]
            ),
            "registry": registry_evidence["registry"],
            "states": [row["state"] for row in registry_evidence["fit_cache_rows"]],
        },
        "root_boundary": _root_boundary_observations(),
    }

    references: dict[str, dict[str, Any]] = {"G0": {}, "G1": {}}
    for gate, check_ids in (("G0", REQUIRED_G0_CHECKS), ("G1", REQUIRED_G1_CHECKS)):
        for check_id in check_ids:
            check_path = checks_root / f"{gate}_{check_id}.json"
            _write_new_json(
                check_path,
                _check_payload(
                    config, cell, gate=gate, check_id=check_id,
                    observations=observations[check_id],
                ),
            )
            references[gate][check_id] = _artifact(check_path)
    manifest = {
        "schema_version": PRETEST_EVIDENCE_SCHEMA,
        "resolved_config_sha256": config.sha256,
        "cell_id": cell.cell_id,
        "gates": references,
        "generated_at": _now(),
        "generation_contract": "derived_checks_only_no_user_pass_boolean",
        "test_constructed": False,
    }
    _write_new_json(destination, manifest)
    validated = validate_pretest_evidence(config, cell, destination, material)
    return {
        "mode": "evidence_generated",
        "cell_id": cell.cell_id,
        "manifest": _artifact(destination),
        "checks": references,
        "verified_ledger_pretest_checks": validated["ledger_pretest_checks"],
        "test_constructed": False,
    }


def validate_pretest_evidence(
    config: ResolvedConfig,
    cell: CellSpec,
    evidence_manifest_path: str | Path,
    material: FreezeMaterial,
) -> Mapping[str, Any]:
    """Verify G0/G1 through hash-bound evidence artifacts, never caller booleans."""

    manifest_path = require_within_root(evidence_manifest_path, must_exist=True)
    manifest = _read_json(manifest_path, label="pre-test evidence manifest")
    if manifest.get("schema_version") != PRETEST_EVIDENCE_SCHEMA:
        raise CampaignControlError("Invalid pre-test evidence schema")
    if (
        manifest.get("resolved_config_sha256"),
        manifest.get("cell_id"),
    ) != (config.sha256, cell.cell_id):
        raise CampaignControlError("Pre-test evidence identity differs from the cell")
    gates = manifest.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != {"G0", "G1"}:
        raise CampaignControlError("Pre-test evidence must contain exactly G0 and G1")

    verified: dict[str, Any] = {}
    check_keys: dict[str, bool] = {}
    for gate, required_checks in (("G0", REQUIRED_G0_CHECKS), ("G1", REQUIRED_G1_CHECKS)):
        gate_value = gates[gate]
        if not isinstance(gate_value, Mapping) or set(gate_value) != set(required_checks):
            raise CampaignControlError(f"{gate} evidence check set differs from the contract")
        verified[gate] = {}
        for check_id in required_checks:
            reference = gate_value[check_id]
            if not isinstance(reference, Mapping):
                raise CampaignControlError(f"{gate}.{check_id} reference must be an artifact")
            artifact_path = _validate_artifact(reference, label=f"{gate}.{check_id}")
            check = _read_json(artifact_path, label=f"{gate}.{check_id} artifact")
            if check.get("schema_version") != PRETEST_CHECK_SCHEMA:
                raise CampaignControlError(f"{gate}.{check_id} has an invalid schema")
            if (
                check.get("gate"),
                check.get("check_id"),
                check.get("status"),
                check.get("resolved_config_sha256"),
                check.get("cell_id"),
            ) != (gate, check_id, "PASS", config.sha256, cell.cell_id):
                raise CampaignControlError(f"{gate}.{check_id} identity/status differs")
            observations = check.get("observations")
            if not isinstance(observations, Mapping):
                raise CampaignControlError(f"{gate}.{check_id} lacks observations")
            _validate_check_observations(
                check_id, observations, config=config, cell=cell, material=material
            )
            digest = str(reference["sha256"])
            check_keys[f"{gate}.{check_id}@{digest}"] = True
            verified[gate][check_id] = {
                "artifact": dict(reference),
                "observations_sha256": canonical_sha256(dict(observations)),
            }
    return {
        "manifest": _artifact(manifest_path),
        "verified": verified,
        "ledger_pretest_checks": check_keys,
    }


def prepare_control(
    config: ResolvedConfig,
    cell: CellSpec,
    *,
    fitted_registry_path: str | Path,
    evidence_manifest_path: str | Path,
    output_path: str | Path,
    execute: bool = False,
    runtime: Any | None = None,
) -> Mapping[str, Any]:
    """Build or persist an immutable pre-test control; never touch test data."""

    material = compute_freeze_material(
        config, cell, fitted_registry_path, runtime=runtime, state_device="cpu"
    )
    gate_evidence = validate_pretest_evidence(
        config, cell, evidence_manifest_path, material
    )
    payload = {
        "schema_version": PREPARED_CONTROL_SCHEMA,
        "status": "prepared",
        "cell_id": cell.cell_id,
        "dataset": cell.dataset_id,
        "protocol": cell.protocol_id,
        "fold": cell.fold,
        "resolved_config_sha256": config.sha256,
        "fitted_registry": _artifact(fitted_registry_path),
        "components": dict(material.components),
        "component_evidence": dict(material.evidence),
        "pretest_evidence": dict(gate_evidence),
        "test_constructed": False,
        "token_persisted": False,
    }
    if execute:
        committed = dict(payload)
        committed["prepared_at"] = _now()
        _write_new_json(output_path, committed)
        return committed
    return {
        "mode": "read_only_prepare",
        "output": _relative(output_path, must_exist=False),
        "would_write": payload,
    }


def _load_prepared(path: str | Path) -> dict[str, Any]:
    value = _read_json(path, label="prepared once-control")
    if value.get("schema_version") != PREPARED_CONTROL_SCHEMA or value.get("status") != "prepared":
        raise CampaignControlError("Invalid prepared once-control schema/status")
    if value.get("test_constructed") is not False or value.get("token_persisted") is not False:
        raise CampaignControlError("Prepared once-control violates pre-test/token boundary")
    return value


def _cell_from_prepared(value: Mapping[str, Any]) -> CellSpec:
    cell = resolve_cell(str(value["dataset"]), str(value["protocol"]), fold=str(value["fold"]))
    if value.get("cell_id") != cell.cell_id:
        raise CampaignControlError("Prepared cell_id is inconsistent")
    return cell


def _revalidate_prepared(
    config: ResolvedConfig,
    prepared_path: str | Path,
    *,
    runtime: Any | None,
) -> tuple[dict[str, Any], CellSpec, FreezeMaterial, Mapping[str, Any]]:
    prepared = _load_prepared(prepared_path)
    cell = _cell_from_prepared(prepared)
    if prepared.get("resolved_config_sha256") != config.sha256:
        raise CampaignControlError("Prepared control config hash drifted")
    registry_record = prepared.get("fitted_registry")
    if not isinstance(registry_record, Mapping):
        raise CampaignControlError("Prepared control lacks fitted registry artifact")
    registry_path = _validate_artifact(registry_record, label="fitted registry")
    current = compute_freeze_material(
        config, cell, registry_path, runtime=runtime, state_device="cpu"
    )
    if dict(current.components) != prepared.get("components"):
        raise CampaignControlError("Freeze component drift detected")
    if canonical_sha256(dict(current.evidence)) != canonical_sha256(
        prepared.get("component_evidence")
    ):
        raise CampaignControlError("Freeze component evidence drift detected")
    evidence_record = prepared.get("pretest_evidence", {}).get("manifest")
    if not isinstance(evidence_record, Mapping):
        raise CampaignControlError("Prepared control lacks pre-test evidence manifest")
    evidence_path = _validate_artifact(evidence_record, label="pre-test evidence manifest")
    gate_evidence = validate_pretest_evidence(config, cell, evidence_path, current)
    if dict(gate_evidence) != prepared.get("pretest_evidence"):
        raise CampaignControlError("Pre-test evidence drift detected")
    return prepared, cell, current, gate_evidence


def freeze_control(
    config: ResolvedConfig,
    *,
    prepared_path: str | Path,
    ledger_path: str | Path,
    execute: bool = False,
    runtime: Any | None = None,
) -> Mapping[str, Any]:
    """Validate or commit a cell ledger.  This function has no test API."""

    prepared, cell, material, gates = _revalidate_prepared(
        config, prepared_path, runtime=runtime
    )
    destination = require_within_root(ledger_path)
    if destination.exists():
        raise CampaignControlError(f"Refusing to overwrite protocol ledger: {destination}")
    if not execute:
        return {
            "mode": "read_only_freeze",
            "cell_id": cell.cell_id,
            "ledger": _relative(destination, must_exist=False),
            "components": dict(material.components),
            "pretest_checks": dict(gates["ledger_pretest_checks"]),
            "test_constructed": False,
        }
    ledger = ProtocolLedger.create(
        destination,
        resolved_config=config,
        components=material.components,
        pretest_checks=gates["ledger_pretest_checks"],
    )
    protocol_sha = ledger.freeze()
    return {
        "mode": "frozen",
        "cell_id": cell.cell_id,
        "ledger": _relative(destination),
        "protocol_sha256": protocol_sha,
        "prepared_control_sha256": sha256_file(prepared_path),
        "test_constructed": False,
    }


class _SingleProcessLock:
    def __init__(self, config: ResolvedConfig) -> None:
        self.path = require_within_root(
            Path(str(config["paths"]["run_root"])) / ".once_campaign.lock"
        )
        self.descriptor: int | None = None

    def __enter__(self) -> "_SingleProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise CampaignControlError(f"Another once-only campaign holds {self.path}") from exc
        os.write(self.descriptor, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)
        return False


def _execution_artifacts(
    config: ResolvedConfig,
    cell: CellSpec,
    material: FreezeMaterial,
    runtime: Any,
) -> tuple[list[dict[str, Any]], Path, Path]:
    rows: list[dict[str, Any]] = []
    for registry_row in material.registry["seeds"]:
        seed = int(registry_row["seed"])
        assets = runtime.resolve_run_assets(
            config, cell.dataset_id, seed, cell.protocol_id, require_existing=True
        )
        cache_dir = require_within_root(assets.cache_directory)
        sidecar = require_within_root(cache_dir / "once_test_cache_manifest.json")
        access = require_within_root(cache_dir / "once_test_access_manifest.json")
        for path in (sidecar, access):
            if path.exists():
                raise CampaignControlError(f"Refusing pre-existing once-test artifact: {path}")
        rows.append(
            {
                "seed": seed,
                "registry_row": registry_row,
                "assets": assets,
                "cache_manifest_path": sidecar,
                "access_manifest_path": access,
            }
        )
    paths = default_paths(config, cell)
    test_registry = require_within_root(paths["test_registry"])
    result = require_within_root(paths["execution_result"])
    for path in (test_registry, result):
        if path.exists():
            raise CampaignControlError(f"Refusing pre-existing once-campaign artifact: {path}")
    run_root = require_within_root(Path(str(config["paths"]["run_root"])))
    for seed in DEFAULT_SEEDS:
        for variant in material.registry["seeds"][DEFAULT_SEEDS.index(seed)]["variants"]:
            run_dir = run_root / cell.dataset_id / cell.protocol_id / f"seed_{seed}" / str(variant["variant_id"])
            if run_dir.exists():
                raise CampaignControlError(f"Refusing pre-existing run directory: {run_dir}")
    return rows, test_registry, result


def _validate_execute_readiness(
    config: ResolvedConfig,
    *,
    prepared_path: str | Path,
    ledger_path: str | Path,
    runtime: Any,
) -> tuple[CellSpec, FreezeMaterial, ProtocolLedger, list[dict[str, Any]], Path, Path]:
    prepared, cell, material, gates = _revalidate_prepared(
        config, prepared_path, runtime=runtime
    )
    ledger = ProtocolLedger.load(ledger_path)
    if ledger.status != "frozen":
        raise CampaignControlError(f"Once-only execution requires frozen ledger, got {ledger.status}")
    if ledger.data.get("test_openings"):
        raise CampaignControlError("This cell already has a consumed or active test opening")
    if ledger.data.get("resolved_config_sha256") != config.sha256:
        raise CampaignControlError("Ledger config hash drifted")
    if ledger.data.get("components") != dict(material.components):
        raise CampaignControlError("Ledger freeze components drifted")
    if ledger.data.get("pretest_checks") != dict(gates["ledger_pretest_checks"]):
        raise CampaignControlError("Ledger G0/G1 evidence bindings drifted")
    rows, registry_path, result_path = _execution_artifacts(
        config, cell, material, runtime
    )
    return cell, material, ledger, rows, registry_path, result_path


def _close_and_seal(ledger_path: Path, *, cell_id: str, token: str) -> None:
    ledger = ProtocolLedger.load(ledger_path)
    opening = ledger.data.get("test_openings", {}).get(cell_id)
    if isinstance(opening, Mapping) and opening.get("closed_at") is None:
        ledger.close_test(cell_id=cell_id, token=token)
        ledger = ProtocolLedger.load(ledger_path)
    if ledger.status == "frozen" and ledger.data.get("test_openings"):
        ledger.seal()


def _redacted_failure(exc: BaseException, token: str) -> Mapping[str, Any]:
    message = str(exc).replace(token, "<redacted-token>") if token else str(exc)
    return {"error_type": type(exc).__name__, "error": message}


def execute_cell_once(
    config: ResolvedConfig,
    *,
    prepared_path: str | Path,
    ledger_path: str | Path,
    execute: bool = False,
    extraction_device: str = "cuda:0",
    evaluation_device: str = "cpu",
    runtime: Any | None = None,
    argv: Sequence[str] = ("run_edgetwincal_once.py", "execute-once"),
) -> Mapping[str, Any]:
    """Validate or consume one complete cell under exactly one opening."""

    _ensure_cublas_workspace()
    active = runtime or _production_runtime()
    with _SingleProcessLock(config):
        cell, material, ledger, seed_rows, test_registry, result_path = _validate_execute_readiness(
            config,
            prepared_path=prepared_path,
            ledger_path=ledger_path,
            runtime=active,
        )
        # _revalidate_prepared above loads and hash-checks all five fitted states.
        # No raw token exists yet, so this is necessarily before token validation.
        if not execute:
            return {
                "mode": "read_only_execute_plan",
                "cell_id": cell.cell_id,
                "ordered_seeds": list(DEFAULT_SEEDS),
                "states_loaded_before_token": True,
                "would_open_test_once": True,
                "test_registry": _relative(test_registry, must_exist=False),
                "result": _relative(result_path, must_exist=False),
            }

        ledger_path_resolved = require_within_root(ledger_path, must_exist=True)
        opening = ledger.open_test_once(
            dataset=cell.dataset_id,
            protocol=cell.protocol_id,
            fold=cell.fold,
            split_manifest_sha256=material.components["split_manifests"][cell.cell_id],
            normalization_manifest_sha256=material.components["normalization_manifests"][cell.cell_id],
        )
        token = str(opening["token"])
        entries: list[Mapping[str, Any]] = []
        execution_result: Mapping[str, Any] | None = None
        failure: BaseException | None = None
        try:
            prepared_cell = active.prepare_cell(config, cell)
            provenance = active.build_provenance_bundle(
                cell.dataset_id,
                cell.protocol_id,
                extraction_type=active.extraction_provenance_type,
            )
            for row in seed_rows:
                seed = int(row["seed"])
                bundle, loader_factory, _ = active.build_runtime(
                    config, prepared_cell, seed
                )
                model = active.load_checkpoint(
                    bundle, row["assets"].checkpoint, device=extraction_device
                )
                entry = active.extract_test_cache_after_open(
                    config=config,
                    assets=row["assets"],
                    provenance=provenance.value,
                    model=model,
                    loader_factory=loader_factory,
                    ledger_path=ledger_path_resolved,
                    cell_id=cell.cell_id,
                    token=token,
                    cache_manifest_path=row["cache_manifest_path"],
                    access_manifest_path=row["access_manifest_path"],
                    device=extraction_device,
                )
                entries.append(entry)
                del model
            active.write_test_cache_registry(
                test_registry,
                dataset_id=cell.dataset_id,
                protocol_id=cell.protocol_id,
                cell_id=cell.cell_id,
                entries=entries,
            )
            execution_result = active.evaluate_campaign_once(
                ledger_path=ledger_path_resolved,
                cell_id=cell.cell_id,
                token=token,
                fitted_registry_path=prepared_path
                and require_within_root(
                    _load_prepared(prepared_path)["fitted_registry"]["path"], must_exist=True
                ),
                test_cache_registry_path=test_registry,
                run_root=Path(str(config["paths"]["run_root"])),
                device=evaluation_device,
                argv=tuple(str(value) for value in argv),
                require_all_selected_seeds=True,
            )
            if execution_result.get("status") != "complete":
                raise CampaignControlError(
                    "Once-only evaluator returned a non-complete campaign result"
                )
        except BaseException as exc:  # close/seal even for KeyboardInterrupt/SystemExit.
            failure = exc
        finally:
            _close_and_seal(ledger_path_resolved, cell_id=cell.cell_id, token=token)

        persisted = {
            "schema_version": EXECUTION_RESULT_SCHEMA,
            "status": (
                "failed"
                if failure is not None
                else str((execution_result or {}).get("status", "failed"))
            ),
            "cell_id": cell.cell_id,
            "ordered_seeds": list(DEFAULT_SEEDS),
            "test_opening_consumed": True,
            "ledger_status": ProtocolLedger.load(ledger_path_resolved).status,
            "token_persisted": False,
            "test_cache_registry": (
                _artifact(test_registry) if test_registry.exists() else None
            ),
            "evaluation": execution_result,
            "failure": None if failure is None else _redacted_failure(failure, token),
        }
        serialized = json.dumps(persisted, ensure_ascii=False, sort_keys=True)
        if token in serialized:
            raise CampaignControlError("Raw opening token reached a persisted payload")
        _write_new_json(result_path, persisted)
        if failure is not None:
            raise CampaignControlError(
                f"Once-only cell failed after consuming test: {persisted['failure']}"
            ) from failure
        return persisted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, freeze, or execute the hash-bound EdgeTwinCal once-only test campaign. "
            "All commands are read-only unless --execute is supplied."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command")
    plan = sub.add_parser("plan", help="Show the supported matrix; always read-only")
    plan.add_argument("--dataset", action="append")
    plan.add_argument("--protocol", action="append")

    for name in ("evidence", "prepare", "freeze", "execute-once"):
        child = sub.add_parser(name)
        child.add_argument("--dataset", required=True)
        child.add_argument("--protocol", required=True)
        child.add_argument("--fold", default=DEFAULT_FOLD)
        child.add_argument("--execute", action="store_true")
    evidence = sub.choices["evidence"]
    evidence.add_argument("--fitted-registry", type=Path)
    evidence.add_argument("--output", type=Path)
    prepare = sub.choices["prepare"]
    prepare.add_argument("--fitted-registry", type=Path)
    prepare.add_argument("--pretest-evidence", type=Path)
    prepare.add_argument("--output", type=Path)
    freeze = sub.choices["freeze"]
    freeze.add_argument("--prepared", type=Path)
    freeze.add_argument("--ledger", type=Path)
    once = sub.choices["execute-once"]
    once.add_argument("--prepared", type=Path)
    once.add_argument("--ledger", type=Path)
    once.add_argument("--extraction-device", choices=("cpu", "cuda", "cuda:0"), default="cuda:0")
    once.add_argument("--evaluation-device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_resolved_config(args.config)
        if args.command in (None, "plan"):
            payload: Mapping[str, Any] = {
                "mode": "read_only_plan",
                "resolved_config_sha256": config.sha256,
                "cells": campaign_plan(
                    config,
                    datasets=getattr(args, "dataset", None),
                    protocols=getattr(args, "protocol", None),
                ),
            }
        else:
            cell = resolve_cell(args.dataset, args.protocol, fold=args.fold)
            paths = default_paths(config, cell)
            if args.command == "evidence":
                payload = generate_pretest_evidence(
                    config,
                    cell,
                    fitted_registry_path=args.fitted_registry or paths["fitted_registry"],
                    output_path=args.output or paths["pretest_evidence"],
                    execute=args.execute,
                )
            elif args.command == "prepare":
                payload = prepare_control(
                    config,
                    cell,
                    fitted_registry_path=args.fitted_registry or paths["fitted_registry"],
                    evidence_manifest_path=args.pretest_evidence or paths["pretest_evidence"],
                    output_path=args.output or paths["prepared_control"],
                    execute=args.execute,
                )
            elif args.command == "freeze":
                payload = freeze_control(
                    config,
                    prepared_path=args.prepared or paths["prepared_control"],
                    ledger_path=args.ledger or paths["ledger"],
                    execute=args.execute,
                )
            else:
                payload = execute_cell_once(
                    config,
                    prepared_path=args.prepared or paths["prepared_control"],
                    ledger_path=args.ledger or paths["ledger"],
                    execute=args.execute,
                    extraction_device=args.extraction_device,
                    evaluation_device=args.evaluation_device,
                    argv=tuple(sys.argv if argv is None else argv),
                )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except CampaignControlBlocked as exc:
        print(
            json.dumps(
                {"status": "blocked", "code": exc.code, "detail": exc.detail},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 3
    except (CampaignControlError, ProtocolLedgerError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "BLOCKED_CELLS",
    "CampaignControlBlocked",
    "CampaignControlError",
    "CellSpec",
    "EXECUTION_RESULT_SCHEMA",
    "FreezeMaterial",
    "PREPARED_CONTROL_SCHEMA",
    "PRETEST_CHECK_SCHEMA",
    "PRETEST_EVIDENCE_SCHEMA",
    "RUNNABLE_CELLS",
    "build_arg_parser",
    "campaign_plan",
    "compute_freeze_material",
    "default_paths",
    "generate_pretest_evidence",
    "execute_cell_once",
    "freeze_control",
    "main",
    "prepare_control",
    "resolve_cell",
    "validate_pretest_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main())
