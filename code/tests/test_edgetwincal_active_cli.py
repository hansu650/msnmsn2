from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import edgetwincal.aggregate as active_aggregate
import edgetwincal.experiment as experiment
import edgetwincal.runtime as active_runtime
from edgetwincal.aggregate_v2 import ConfirmatoryAggregationError
from edgetwincal.campaign import ProtocolLedger, ProtocolLedgerError
from edgetwincal.paths import PROJECT_ROOT


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _freeze_components() -> dict:
    return {
        "variant_registry_sha256": _sha("variants"),
        "split_manifests": {"P12|strict_p12": _sha("split")},
        "normalization_manifests": {"P12|strict_p12": _sha("normalizer")},
        "cache_schema_sha256": _sha("cache-schema-3"),
        "statistics_sha256": _sha("statistics"),
        "timing_schema_sha256": _sha("timing"),
        "project_source_sha256": _sha("project"),
        "apn_patch_sha256": _sha("patch"),
        "environment_sha256": _sha("environment"),
    }


def _create_and_freeze_ledger(tmp_path: Path) -> Path:
    components = tmp_path / "components.json"
    pretests = tmp_path / "pretests.json"
    ledger = tmp_path / "protocol-ledger.json"
    components.write_text(json.dumps(_freeze_components()), encoding="utf-8")
    pretests.write_text(
        json.dumps({"G0_unit_and_parity": True, "G1_provenance": True}),
        encoding="utf-8",
    )
    assert (
        experiment.main(
            [
                "ledger",
                "create",
                "--dataset",
                "P12",
                "--seed",
                "2024",
                "--ledger",
                str(ledger),
                "--components",
                str(components),
                "--pretest-checks",
                str(pretests),
            ]
        )
        == 0
    )
    assert experiment.main(["ledger", "freeze", "--ledger", str(ledger)]) == 0
    return ledger


def test_active_facades_expose_v2_only_and_legacy_compactor_is_compatible() -> None:
    assert active_runtime.TENSOR_PAYLOAD_SCHEMA_VERSION == 3
    assert "load_or_create_cache" not in active_runtime.__all__
    assert not hasattr(active_runtime, "load_or_create_cache")
    assert not hasattr(active_runtime, "create_cache")
    with pytest.raises(ConfirmatoryAggregationError, match="non-empty"):
        active_aggregate.run_aggregation([], [])

    metrics = {
        "schema_version": 1,
        "attempt": 5,
        "seed": 2024,
        "device": "cpu",
        "passed": False,
        "validation_metrics": {},
        "test_metrics": {},
        "relative_improvement_vs_apn": {},
        "assets": {
            "checkpoint": r"C:\private\checkpoint.bin",
            "checkpoint_sha256": "abc123",
        },
    }
    compact = active_aggregate.compact_run_manifest(metrics)
    assert compact["checkpoint_sha256"] == "abc123"
    assert "assets" not in compact
    assert r"C:\private" not in json.dumps(compact)


def test_active_help_is_lazy_and_lists_safe_control_plane_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("--help must not load the resolved config")

    apn_before = sys.modules.get("models.APN")
    data_before = {name for name in sys.modules if name.startswith("data.data_provider")}
    monkeypatch.setattr(experiment, "load_resolved_config", forbidden)
    with pytest.raises(SystemExit) as caught:
        experiment.main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    for command in ("audit", "ledger", "status", "aggregate", "pretest", "evaluate"):
        assert command in output
    assert sys.modules.get("models.APN") is apn_before
    assert {name for name in sys.modules if name.startswith("data.data_provider")} == data_before


def test_audit_is_config_driven_and_never_opens_test(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("audit must not open a test cell")

    monkeypatch.setattr(ProtocolLedger, "open_test_once", forbidden)
    code = experiment.main(
        [
            "audit",
            "--dataset",
            "P12",
            "--seed",
            "2024",
            "--protocol",
            "strict_p12",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved_config_sha256"]
    assert payload["cell_count"] == 1
    assert payload["cells"][0]["dataset"] == "P12"
    assert payload["cells"][0]["protocol"] == "strict_p12"
    assert payload["cells"][0]["status"] in {"PLANNED", "READY"}
    assert str(PROJECT_ROOT) not in json.dumps(payload)


def test_ledger_create_freeze_and_status_do_not_create_test_openings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger_path = _create_and_freeze_ledger(tmp_path)
    ledger = ProtocolLedger.load(ledger_path)
    assert ledger.status == "frozen"
    assert ledger.data["test_openings"] == {}

    capsys.readouterr()
    assert experiment.main(["status", "--ledger", str(ledger_path)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["ledger"]["status"] == "frozen"
    assert status["ledger"]["test_opening_count"] == 0
    assert status["ledger"]["active_test_opening_count"] == 0


def test_evaluate_requires_existing_once_only_token_and_does_not_access_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger_path = _create_and_freeze_ledger(tmp_path)
    capsys.readouterr()

    assert (
        experiment.main(
            [
                "evaluate",
                "--ledger",
                str(ledger_path),
                "--cell-id",
                "P12|strict_p12|fold-0",
                "--token",
                "not-a-token",
            ]
        )
        == 2
    )
    assert "not-a-token" not in capsys.readouterr().err

    ledger = ProtocolLedger.load(ledger_path)
    opening = ledger.open_test_once(
        dataset="P12",
        protocol="strict_p12",
        fold="fold-0",
        split_manifest_sha256=_sha("split"),
        normalization_manifest_sha256=_sha("normalizer"),
    )
    with pytest.raises(ProtocolLedgerError, match="already opened"):
        ledger.open_test_once(
            dataset="P12",
            protocol="strict_p12",
            fold="fold-0",
            split_manifest_sha256=_sha("split"),
            normalization_manifest_sha256=_sha("normalizer"),
        )

    capsys.readouterr()
    code = experiment.main(
        [
            "evaluate",
            "--ledger",
            str(ledger_path),
            "--cell-id",
            opening["cell_id"],
            "--token",
            opening["token"],
        ]
    )
    captured = capsys.readouterr()
    assert code == experiment.EVALUATION_ARTIFACTS_REQUIRED_EXIT
    assert opening["token"] not in captured.out
    assert opening["token"] not in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "BLOCKED"
    assert payload["reason_code"] == "EVALUATION_ARTIFACTS_REQUIRED"
    assert payload["test_data_opened_by_cli"] is False
    after = ProtocolLedger.load(ledger_path)
    assert after.status == "test_active"
    assert after.data["test_openings"][opening["cell_id"]]["closed_at"] is None


def test_aggregate_requires_explicit_complete_registry_and_writes_nothing_on_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing-run.json"
    registry = tmp_path / "registry.json"
    output = tmp_path / "aggregate.json"
    registry.write_text(
        json.dumps({"expected_manifests": [str(missing)]}),
        encoding="utf-8",
    )
    code = experiment.main(
        [
            "aggregate",
            "--registry",
            str(registry),
            "--output",
            str(output),
        ]
    )
    assert code == 2
    assert not output.exists()
    assert "aggregation aborted" in capsys.readouterr().err.lower()


def test_status_marks_missing_explicit_manifest_ineligible_without_discovery(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    code = experiment.main(["status", "--manifest", str(missing)])
    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["eligible_manifest_count"] == 0
    assert payload["manifests"][0]["eligible"] is False
    assert payload["manifests"][0]["reason"].startswith(("invalid:", "unavailable:"))
    assert str(PROJECT_ROOT) not in json.dumps(payload)


def test_active_source_has_no_legacy_schema2_cache_entrypoint() -> None:
    source = (PROJECT_ROOT / "code" / "src" / "edgetwincal" / "experiment.py").read_text(
        encoding="utf-8"
    )
    assert "load_or_create_cache" not in source
    assert "schema_version == 2" not in source
    legacy = (
        PROJECT_ROOT / "code" / "src" / "edgetwincal" / "legacy_experiment_v1.py"
    ).read_text(encoding="utf-8")
    assert "legacy_runtime_v1" in legacy
    assert "legacy_graph_v1" in legacy
