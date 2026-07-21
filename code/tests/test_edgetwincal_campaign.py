from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from edgetwincal.campaign import ProtocolLedger, ProtocolLedgerError
from edgetwincal.config import load_resolved_config
from edgetwincal.paths import PROJECT_ROOT


def _components() -> dict:
    digest = "a" * 64
    return {
        "variant_registry_sha256": digest,
        "split_manifests": {"P12|strict_p12|fold0": digest},
        "normalization_manifests": {"P12|strict_p12|fold0": digest},
        "cache_schema_sha256": digest,
        "statistics_sha256": digest,
        "timing_schema_sha256": digest,
        "project_source_sha256": digest,
        "apn_patch_sha256": digest,
        "environment_sha256": digest,
    }


@pytest.fixture
def ledger_path() -> Path:
    path = PROJECT_ROOT / "results" / f".pytest_campaign_{uuid4().hex}.json"
    yield path
    path.unlink(missing_ok=True)


def test_ledger_refuses_freeze_until_every_pretest_passes(
    ledger_path: Path,
) -> None:
    config = load_resolved_config()
    ledger = ProtocolLedger.create(
        ledger_path,
        resolved_config=config,
        components=_components(),
        pretest_checks={"unit_tests": True, "apn_parity": False},
    )
    with pytest.raises(ProtocolLedgerError, match="have not passed"):
        ledger.freeze()


def test_once_only_open_is_persisted_before_access_and_cannot_repeat(
    ledger_path: Path,
) -> None:
    config = load_resolved_config()
    path = ledger_path
    ledger = ProtocolLedger.create(
        path,
        resolved_config=config,
        components=_components(),
        pretest_checks={"unit_tests": True, "apn_parity": True, "integrity": True},
    )
    protocol_hash = ledger.freeze()
    opening = ledger.open_test_once(
        dataset="P12",
        protocol="strict_p12",
        fold="fold0",
        split_manifest_sha256="a" * 64,
        normalization_manifest_sha256="a" * 64,
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["status"] == "test_active"
    assert persisted["test_openings"][opening["cell_id"]]["token_sha256"]
    assert opening["token"] not in path.read_text(encoding="utf-8")
    assert opening["protocol_sha256"] == protocol_hash
    with pytest.raises(ProtocolLedgerError, match="already opened"):
        ledger.open_test_once(
            dataset="P12",
            protocol="strict_p12",
            fold="fold0",
            split_manifest_sha256="a" * 64,
            normalization_manifest_sha256="a" * 64,
        )
    with pytest.raises(ProtocolLedgerError, match="Invalid"):
        ledger.close_test(cell_id=opening["cell_id"], token="wrong")
    ledger.close_test(cell_id=opening["cell_id"], token=opening["token"])
    ledger.seal()
    assert ProtocolLedger.load(path).status == "sealed"


def test_ledger_rejects_noncanonical_component_set(ledger_path: Path) -> None:
    config = load_resolved_config()
    components = _components()
    components.pop("statistics_sha256")
    with pytest.raises(ProtocolLedgerError, match="components differ"):
        ProtocolLedger.create(
            ledger_path,
            resolved_config=config,
            components=components,
            pretest_checks={"unit_tests": True},
        )
