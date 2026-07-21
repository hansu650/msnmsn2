from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ResolvedConfig, canonical_sha256
from .paths import require_within_root
from .schema import atomic_write_json


LEDGER_SCHEMA = "edgetwincal.protocol-ledger.v1"
REQUIRED_FREEZE_COMPONENTS = {
    "variant_registry_sha256",
    "split_manifests",
    "normalization_manifests",
    "cache_schema_sha256",
    "statistics_sha256",
    "timing_schema_sha256",
    "project_source_sha256",
    "apn_patch_sha256",
    "environment_sha256",
}


class ProtocolLedgerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolLedgerError(f"Cannot read protocol ledger: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != LEDGER_SCHEMA:
        raise ProtocolLedgerError("Invalid protocol ledger schema")
    return value


def _validate_components(components: Mapping[str, Any]) -> None:
    if set(components) != REQUIRED_FREEZE_COMPONENTS:
        missing = sorted(REQUIRED_FREEZE_COMPONENTS - set(components))
        extra = sorted(set(components) - REQUIRED_FREEZE_COMPONENTS)
        raise ProtocolLedgerError(
            f"Freeze components differ; missing={missing}, extra={extra}"
        )
    for name in (
        "variant_registry_sha256",
        "cache_schema_sha256",
        "statistics_sha256",
        "timing_schema_sha256",
        "project_source_sha256",
        "apn_patch_sha256",
        "environment_sha256",
    ):
        if not _is_sha256(components[name]):
            raise ProtocolLedgerError(f"{name} must be a SHA256")
    for name in ("split_manifests", "normalization_manifests"):
        mapping = components[name]
        if not isinstance(mapping, Mapping) or not mapping:
            raise ProtocolLedgerError(f"{name} must be a non-empty mapping")
        if any(not _is_sha256(value) for value in mapping.values()):
            raise ProtocolLedgerError(f"{name} values must be SHA256 digests")


@dataclass
class ProtocolLedger:
    path: Path
    data: dict[str, Any]

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        resolved_config: ResolvedConfig,
        components: Mapping[str, Any],
        pretest_checks: Mapping[str, bool],
    ) -> "ProtocolLedger":
        destination = require_within_root(path)
        if destination.exists():
            raise ProtocolLedgerError(f"Refusing to overwrite ledger: {destination}")
        _validate_components(components)
        if not pretest_checks or any(
            not isinstance(value, bool) for value in pretest_checks.values()
        ):
            raise ProtocolLedgerError("pretest_checks must be a non-empty bool mapping")
        created = _now()
        data = {
            "schema_version": LEDGER_SCHEMA,
            "status": "draft",
            "resolved_config_sha256": resolved_config.sha256,
            "resolved_config": resolved_config.to_dict(),
            "components": dict(components),
            "pretest_checks": dict(pretest_checks),
            "protocol_sha256": None,
            "test_openings": {},
            "created_at": created,
            "updated_at": created,
            "events": [{"event": "created", "at": created}],
        }
        atomic_write_json(destination, data)
        return cls(destination, data)

    @classmethod
    def load(cls, path: str | Path) -> "ProtocolLedger":
        source = require_within_root(path, must_exist=True)
        return cls(source, _load(source))

    @property
    def status(self) -> str:
        return str(self.data["status"])

    def _persist(self) -> None:
        self.data["updated_at"] = _now()
        atomic_write_json(self.path, self.data)

    def freeze(self) -> str:
        if self.status != "draft":
            raise ProtocolLedgerError(f"Cannot freeze ledger in state {self.status}")
        failed = sorted(
            name for name, passed in self.data["pretest_checks"].items() if not passed
        )
        if failed:
            raise ProtocolLedgerError(f"Pre-test checks have not passed: {failed}")
        committed = {
            "resolved_config_sha256": self.data["resolved_config_sha256"],
            "components": self.data["components"],
            "pretest_checks": self.data["pretest_checks"],
        }
        digest = canonical_sha256(committed)
        self.data["protocol_sha256"] = digest
        self.data["status"] = "frozen"
        self.data["events"].append(
            {"event": "frozen", "at": _now(), "protocol_sha256": digest}
        )
        self._persist()
        return digest

    def open_test_once(
        self,
        *,
        dataset: str,
        protocol: str,
        fold: str,
        split_manifest_sha256: str,
        normalization_manifest_sha256: str,
    ) -> dict[str, str]:
        if self.status not in {"frozen", "test_active"}:
            raise ProtocolLedgerError(f"Cannot open test in state {self.status}")
        if not _is_sha256(split_manifest_sha256) or not _is_sha256(
            normalization_manifest_sha256
        ):
            raise ProtocolLedgerError("Test opening manifests must be SHA256 digests")
        cell_id = f"{dataset}|{protocol}|{fold}"
        if cell_id in self.data["test_openings"]:
            raise ProtocolLedgerError(f"Test cell was already opened: {cell_id}")
        token = secrets.token_hex(32)
        record = {
            "dataset": dataset,
            "protocol": protocol,
            "fold": fold,
            "split_manifest_sha256": split_manifest_sha256,
            "normalization_manifest_sha256": normalization_manifest_sha256,
            "protocol_sha256": self.data["protocol_sha256"],
            "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
            "opened_at": _now(),
            "closed_at": None,
        }
        self.data["test_openings"][cell_id] = record
        self.data["status"] = "test_active"
        self.data["events"].append(
            {"event": "test_opened", "at": record["opened_at"], "cell_id": cell_id}
        )
        self._persist()
        return {
            "cell_id": cell_id,
            "token": token,
            "protocol_sha256": str(self.data["protocol_sha256"]),
        }

    def validate_test_token(self, *, cell_id: str, token: str) -> None:
        record = self.data["test_openings"].get(cell_id)
        if record is None:
            raise ProtocolLedgerError(f"Unknown test cell: {cell_id}")
        actual = hashlib.sha256(token.encode("ascii")).hexdigest()
        if not secrets.compare_digest(actual, record["token_sha256"]):
            raise ProtocolLedgerError("Invalid test-open token")
        if record["closed_at"] is not None:
            raise ProtocolLedgerError("Test cell is already closed")

    def close_test(self, *, cell_id: str, token: str) -> None:
        self.validate_test_token(cell_id=cell_id, token=token)
        closed = _now()
        self.data["test_openings"][cell_id]["closed_at"] = closed
        self.data["events"].append(
            {"event": "test_closed", "at": closed, "cell_id": cell_id}
        )
        if all(
            record["closed_at"] is not None
            for record in self.data["test_openings"].values()
        ):
            self.data["status"] = "frozen"
        self._persist()

    def seal(self) -> None:
        if self.status != "frozen":
            raise ProtocolLedgerError("All active test cells must close before sealing")
        if not self.data["test_openings"]:
            raise ProtocolLedgerError("Cannot seal a ledger with no test opening")
        self.data["status"] = "sealed"
        self.data["events"].append({"event": "sealed", "at": _now()})
        self._persist()
