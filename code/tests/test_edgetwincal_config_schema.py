from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import edgetwincal.schema as schema_module
import pytest

from edgetwincal.config import (
    DEFAULT_CONFIG,
    ConfigError,
    canonical_sha256,
    load_resolved_config,
)
from edgetwincal.paths import PROJECT_ROOT
from edgetwincal.schema import (
    IncompleteRunError,
    InvalidTransitionError,
    ManifestError,
    RunManifest,
    validate_run_manifest,
    aggregation_eligibility,
    load_complete_manifest,
    load_complete_manifests,
)


@contextmanager
def _project_scratch():
    base = PROJECT_ROOT / "results" / "edgetwincal_msn2026_v1" / "schema_test_scratch"
    root = base / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_default_config_is_complete_immutable_and_canonical() -> None:
    resolved = load_resolved_config()
    assert resolved["method"]["version"] == "msn2026_v1"
    assert resolved["campaign"]["seeds"] == (2024, 2025, 2026, 2027, 2028)
    assert resolved["bootstrap"]["draws"] == 50000
    assert resolved["bootstrap"]["analysis_seed"] == 20260721
    p12_task = resolved["datasets"]["P12"]["task"]
    activity_task = resolved["datasets"]["HumanActivity"]["task"]
    assert p12_task["horizon_kind"] == "fixed_steps"
    assert p12_task["prediction_steps"] == 3
    assert activity_task["horizon_kind"] == "time_window"
    assert activity_task["forecast_window"] == 300
    assert "prediction_steps" not in activity_task
    assert resolved["datasets"]["USHCN"]["storage"] == (
        "data/tsdm/datasets/USHCN_DeBrouwer2019"
    )
    assert resolved["datasets"]["MIMIC_III"]["availability"] == "required"
    assert resolved.sha256 == canonical_sha256(resolved.to_dict())
    assert resolved.sha256 == canonical_sha256(resolved.to_dict())
    assert len(resolved.sha256) == 64
    with pytest.raises(TypeError):
        resolved["method"]["version"] = "changed"


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    left = {"z": [3, {"b": 2, "a": 1}], "a": "utf8-配置"}
    right = {"a": "utf8-配置", "z": [3, {"a": 1, "b": 2}]}
    assert canonical_sha256(left) == canonical_sha256(right)


def test_cli_selectors_only_select_registered_members() -> None:
    resolved = load_resolved_config(
        overrides={"datasets": ["P12", "USHCN"], "seed": 2026, "variant": "Full"}
    )
    assert resolved["selection"]["datasets"] == ("P12", "USHCN")
    assert resolved["selection"]["seeds"] == (2026,)
    assert resolved["selection"]["variants"] == ("Full",)

    with pytest.raises(ConfigError, match="Unknown datasets"):
        load_resolved_config(overrides={"dataset": "new-test-dataset"})
    with pytest.raises(ConfigError, match="Only dataset/seed/variant"):
        load_resolved_config(overrides={"ridge": [0.0]})
    with pytest.raises(ConfigError, match="duplicates"):
        load_resolved_config(overrides={"seeds": [2024, 2024]})


def test_config_rejects_project_root_escape() -> None:
    with _project_scratch() as scratch:
        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        payload["paths"]["run_root"] = "../outside-msn2"
        invalid = scratch / "invalid.json"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ConfigError, match="parent traversal"):
            load_resolved_config(invalid)

def test_config_rejects_ambiguous_or_mislabeled_horizon_contracts() -> None:
    with _project_scratch() as scratch:
        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        activity = payload["datasets"]["HumanActivity"]["task"]
        activity["horizon"] = 300
        invalid = scratch / "ambiguous_horizon.json"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ConfigError, match="horizon is ambiguous"):
            load_resolved_config(invalid)

        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        payload["datasets"]["P12"]["task"]["horizon_kind"] = "time_window"
        invalid = scratch / "wrong_horizon_kind.json"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ConfigError, match="forecast_window"):
            load_resolved_config(invalid)



def test_config_import_and_help_do_not_import_vendor_or_data_modules() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'code' / 'src')!r}); "
        "from edgetwincal.config import build_arg_parser; "
        "build_arg_parser().format_help(); "
        "assert 'models.APN' not in sys.modules; "
        "assert not any(name.startswith('data.data_provider') for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _new_manifest(root: Path, *, name: str = "run") -> tuple[RunManifest, Path]:
    resolved = load_resolved_config(
        overrides={"dataset": "P12", "seed": 2024, "variant": "Full"}
    )
    log = root / f"{name}.log"
    log.write_text("synthetic train/validation-only smoke\n", encoding="utf-8")
    manifest = RunManifest.create(
        root / f"{name}.json",
        run_id=f"p12-strict-2024-{name}",
        dataset="P12",
        protocol="strict_p12",
        fold="hash-fold-0",
        seed=2024,
        variant_id="Full",
        variant_definition=resolved["variants"]["Full"],
        resolved_config=resolved,
        argv=["python", "synthetic_smoke.py", "--split", "validation"],
        environment={"python": sys.version.split()[0], "device": "cpu"},
        log_path=log,
    )
    return manifest, log

def test_run_schema_rejects_legacy_ambiguous_horizon_even_with_matching_hash() -> None:
    with _project_scratch() as scratch:
        manifest, _ = _new_manifest(scratch)
        data = manifest.data
        activity = data["resolved_config"]["datasets"]["HumanActivity"]["task"]
        activity["horizon"] = activity["forecast_window"]
        data["resolved_config_sha256"] = canonical_sha256(data["resolved_config"])
        with pytest.raises(ManifestError, match="horizon is ambiguous"):
            validate_run_manifest(data, verify_files=False)



def _completion_payload(log: Path) -> dict:
    checkpoint_hash = "a" * 64
    return {
        "assets": {
            "checkpoint_sha256": checkpoint_hash,
            "dataset_asset_hash": "b" * 64,
        },
        "cache_manifest": {"cache_key": "synthetic-validation-cache"},
        "split_manifest": {"split_hash": "c" * 64, "test_opened": False},
        "normalization_manifest": {"fit_split": "train", "fit_id_hash": "d" * 64},
        "selected_hyperparameters": {"alpha_slrh": 100.0, "alpha_cfg": 1000.0},
        "timing": {
            "segments": [
                {
                    "name": "validation_scoring_selection",
                    "device": "cpu",
                    "seconds": 0.01,
                }
            ]
        },
        "cells": [
            {
                "group_hash": "e" * 64,
                "checkpoint_sha256": checkpoint_hash,
                "variant": "Full",
                "sse": 2.0,
                "sae": 3.0,
                "n": 10,
            }
        ],
        "metrics": {"mse": 0.2, "mae": 0.3, "sse": 2.0, "sae": 3.0, "n": 10},
        "required_files": [log],
    }


def test_run_manifest_atomic_complete_transition_and_hash_validation() -> None:
    with _project_scratch() as scratch:
        manifest, log = _new_manifest(scratch)
        assert manifest.status == "created"
        manifest.start().mark_phase("synthetic_validation_smoke")
        manifest.complete(**_completion_payload(log))
        assert manifest.status == "complete"
        loaded = load_complete_manifest(manifest.path)
        assert loaded["resolved_config_sha256"] == canonical_sha256(
            loaded["resolved_config"]
        )
        assert loaded["required_files"][0]["path"].endswith("run.log")
        assert loaded["required_files"][0]["bytes"] == log.stat().st_size
        assert aggregation_eligibility(manifest.path) == (
            True,
            "complete_and_verified",
        )
        assert not list(scratch.glob(".*.tmp"))
        with pytest.raises(InvalidTransitionError):
            manifest.start()


def test_failed_and_running_runs_are_explicitly_excluded() -> None:
    with _project_scratch() as scratch:
        failed, _ = _new_manifest(scratch, name="failed")
        failed.start().mark_phase("feature_extraction").fail(RuntimeError("synthetic failure"))
        assert aggregation_eligibility(failed.path) == (False, "status:failed")
        with pytest.raises(IncompleteRunError, match="status failed"):
            load_complete_manifest(failed.path)

        running, _ = _new_manifest(scratch, name="running")
        running.start()
        assert aggregation_eligibility(running.path) == (False, "status:running")
        with pytest.raises(IncompleteRunError, match="status running"):
            load_complete_manifest(running.path)


def test_invalid_completion_rolls_back_and_is_not_aggregated() -> None:
    with _project_scratch() as scratch:
        manifest, log = _new_manifest(scratch)
        other = scratch / "other.json"
        other.write_text("{}\n", encoding="utf-8")
        manifest.start()
        payload = _completion_payload(log)
        payload["required_files"] = [other]
        with pytest.raises(ManifestError, match="log_path must be present"):
            manifest.complete(**payload)
        assert manifest.status == "running"
        assert RunManifest.load(manifest.path).status == "running"


def test_changed_required_file_invalidates_complete_run() -> None:
    with _project_scratch() as scratch:
        manifest, log = _new_manifest(scratch)
        manifest.start().complete(**_completion_payload(log))
        log.write_text("changed after completion\n", encoding="utf-8")
        eligible, reason = aggregation_eligibility(manifest.path)
        assert eligible is False
        assert reason.startswith("invalid:Required file")


def test_manifest_collection_never_silently_drops_failed_member() -> None:
    with _project_scratch() as scratch:
        complete, complete_log = _new_manifest(scratch, name="complete")
        complete.start().complete(**_completion_payload(complete_log))
        failed, _ = _new_manifest(scratch, name="failed")
        failed.start().fail("expected synthetic failure")
        with pytest.raises(IncompleteRunError, match="status failed"):
            load_complete_manifests([complete.path, failed.path])

def test_atomic_json_retries_transient_windows_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _project_scratch() as scratch:
        destination = scratch / "atomic.json"
        original_replace = schema_module.os.replace
        attempts: list[tuple[Path, Path]] = []
        delays: list[float] = []

        def flaky_replace(source: Path, target: Path) -> None:
            attempts.append((Path(source), Path(target)))
            if len(attempts) < 3:
                error = PermissionError(13, "synthetic transient replace failure")
                error.winerror = 5
                raise error
            original_replace(source, target)

        monkeypatch.setattr(schema_module.os, "replace", flaky_replace)
        monkeypatch.setattr(
            schema_module.time,
            "sleep",
            lambda seconds: delays.append(float(seconds)),
        )
        schema_module.atomic_write_json(destination, {"status": "complete"})

        assert json.loads(destination.read_text(encoding="utf-8")) == {
            "status": "complete"
        }
        assert len(attempts) == 3
        assert delays == [0.01, 0.02]
        assert not list(scratch.glob(".*.tmp"))
