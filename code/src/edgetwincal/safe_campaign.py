from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import canonical_sha256
from .paths import PROJECT_ROOT, require_within_root
from .schema import atomic_write_json


SAFE_LEDGER_SCHEMA = "edgetwincal.safe-campaign-test-ledger.v1"
SAFE_EVALUATION_SCHEMA = "edgetwincal.safe-evaluation.v1"


class SafeCampaignError(RuntimeError):
    """Raised when a Safe campaign transition or paired artifact is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafeCampaignError(f"Cannot read campaign JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SafeCampaignError(f"Campaign JSON root is not an object: {path}")
    return value


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + value.numpy().tobytes(order="C")).hexdigest()


def _atomic_torch_save(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


@dataclass
class SafeTestLedger:
    path: Path
    data: dict[str, Any]

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        config_sha256: str,
        components: Mapping[str, str],
        gate_decisions: Mapping[str, Mapping[str, Any]],
    ) -> "SafeTestLedger":
        destination = require_within_root(path)
        if destination.exists():
            raise SafeCampaignError(f"Refusing to overwrite ledger: {destination}")
        if not _is_sha256(config_sha256):
            raise SafeCampaignError("config_sha256 is invalid")
        if not components or any(not _is_sha256(value) for value in components.values()):
            raise SafeCampaignError("Every frozen component must be a SHA256")
        if len(gate_decisions) < 2:
            raise SafeCampaignError("At least two frozen target gate decisions are required")
        copied_gates = json.loads(json.dumps(gate_decisions, sort_keys=True))
        for dataset, decision in copied_gates.items():
            if not isinstance(decision, Mapping) or not isinstance(decision.get("enabled"), bool):
                raise SafeCampaignError(f"{dataset} lacks a final gate decision")
            if not _is_sha256(decision.get("gate_sha256")):
                raise SafeCampaignError(f"{dataset} gate hash is invalid")
            if int(decision.get("checkpoints", -1)) != 5:
                raise SafeCampaignError(f"{dataset} gate did not audit five checkpoints")
        timestamp = _now()
        data = {
            "schema_version": SAFE_LEDGER_SCHEMA,
            "status": "draft",
            "config_sha256": config_sha256,
            "components": dict(sorted(components.items())),
            "gate_decisions": copied_gates,
            "expected_datasets": sorted(copied_gates),
            "protocol_sha256": None,
            "test_openings": {},
            "events": [{"event": "created", "at": timestamp}],
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        atomic_write_json(destination, data)
        return cls(destination, data)

    @classmethod
    def load(cls, path: str | Path) -> "SafeTestLedger":
        source = require_within_root(path, must_exist=True)
        data = _load_json(source)
        if data.get("schema_version") != SAFE_LEDGER_SCHEMA:
            raise SafeCampaignError("Safe test ledger schema is invalid")
        return cls(source, data)

    @property
    def status(self) -> str:
        return str(self.data["status"])

    def _persist(self) -> None:
        self.data["updated_at"] = _now()
        atomic_write_json(self.path, self.data)

    def freeze(self) -> str:
        if self.status != "draft":
            raise SafeCampaignError(f"Cannot freeze a {self.status} ledger")
        committed = {
            "config_sha256": self.data["config_sha256"],
            "components": self.data["components"],
            "gate_decisions": self.data["gate_decisions"],
            "expected_datasets": self.data["expected_datasets"],
        }
        digest = canonical_sha256(committed)
        self.data["protocol_sha256"] = digest
        self.data["status"] = "frozen"
        self.data["events"].append(
            {"event": "frozen", "at": _now(), "protocol_sha256": digest}
        )
        self._persist()
        return digest

    def open_dataset_once(self, dataset: str) -> str:
        if self.status not in {"frozen", "test_active"}:
            raise SafeCampaignError(f"Cannot open test in ledger state {self.status}")
        if dataset not in self.data["expected_datasets"]:
            raise SafeCampaignError(f"Unknown Safe target: {dataset}")
        if dataset in self.data["test_openings"]:
            raise SafeCampaignError(f"Test target was already opened: {dataset}")
        token = secrets.token_hex(32)
        record = {
            "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
            "opened_at": _now(),
            "closed_at": None,
        }
        self.data["test_openings"][dataset] = record
        self.data["status"] = "test_active"
        self.data["events"].append(
            {"event": "test_opened", "dataset": dataset, "at": record["opened_at"]}
        )
        self._persist()
        return token

    def validate_token(self, dataset: str, token: str) -> None:
        record = self.data["test_openings"].get(dataset)
        if not isinstance(record, Mapping):
            raise SafeCampaignError(f"Test target is not open: {dataset}")
        actual = hashlib.sha256(str(token).encode("ascii")).hexdigest()
        if not secrets.compare_digest(actual, str(record["token_sha256"])):
            raise SafeCampaignError("Invalid test-open token")
        if record["closed_at"] is not None:
            raise SafeCampaignError(f"Test target is already closed: {dataset}")

    def close_dataset(self, dataset: str, token: str) -> None:
        self.validate_token(dataset, token)
        closed = _now()
        self.data["test_openings"][dataset]["closed_at"] = closed
        self.data["events"].append(
            {"event": "test_closed", "dataset": dataset, "at": closed}
        )
        if all(
            isinstance(record, Mapping) and record.get("closed_at") is not None
            for record in self.data["test_openings"].values()
        ):
            self.data["status"] = "frozen"
        self._persist()

    def seal(self) -> None:
        if self.status != "frozen":
            raise SafeCampaignError(f"Cannot seal a {self.status} ledger")
        expected = set(self.data["expected_datasets"])
        opened = set(self.data["test_openings"])
        if opened != expected:
            raise SafeCampaignError(
                f"Cannot seal incomplete targets; missing={sorted(expected - opened)}"
            )
        if any(record.get("closed_at") is None for record in self.data["test_openings"].values()):
            raise SafeCampaignError("Cannot seal while a test target remains open")
        self.data["status"] = "sealed"
        self.data["events"].append({"event": "sealed", "at": _now()})
        self._persist()


def group_error_cells(
    *,
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    group_ids: Tensor,
    dataset: str,
    group_salt: str,
) -> list[dict[str, Any]]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise SafeCampaignError("Prediction, target, and mask shapes differ")
    if prediction.ndim != 3 or group_ids.ndim != 1 or len(group_ids) != len(prediction):
        raise SafeCampaignError("Evaluation tensors violate [N,H,C]/[N] contracts")
    selected = mask > 0
    if not torch.isfinite(prediction[selected]).all() or not torch.isfinite(target[selected]).all():
        raise SafeCampaignError("Observed prediction/target values must be finite")
    totals: dict[str, list[float]] = {}
    prediction64 = prediction.detach().cpu().double()
    target64 = target.detach().cpu().double()
    selected_cpu = selected.detach().cpu()
    for row, raw_group in enumerate(group_ids.detach().cpu().tolist()):
        digest = hashlib.sha256(
            f"{group_salt}{dataset}:{int(raw_group)}".encode("utf-8")
        ).hexdigest()
        observed = selected_cpu[row]
        error = prediction64[row][observed] - target64[row][observed]
        bucket = totals.setdefault(digest, [0.0, 0.0, 0.0])
        bucket[0] += float(torch.square(error).sum().item())
        bucket[1] += float(torch.abs(error).sum().item())
        bucket[2] += int(error.numel())
    return [
        {
            "group_hash": group,
            "sse": float(values[0]),
            "sae": float(values[1]),
            "n": int(values[2]),
        }
        for group, values in sorted(totals.items())
    ]


def _metrics(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sse = sum(float(cell["sse"]) for cell in cells)
    sae = sum(float(cell["sae"]) for cell in cells)
    n = sum(int(cell["n"]) for cell in cells)
    if n <= 0 or not math.isfinite(sse) or not math.isfinite(sae):
        raise SafeCampaignError("Evaluation has no finite observed targets")
    return {"mse": sse / n, "mae": sae / n, "sse": sse, "sae": sae, "n": n}


def build_evaluation_manifest(
    *,
    dataset: str,
    seed: int,
    variant: str,
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    sample_ids: Tensor,
    group_ids: Tensor,
    group_salt: str,
    checkpoint_sha256: str,
    config_sha256: str,
    split_sha256: str,
    normalizer_sha256: str,
    selected_hyperparameters: Mapping[str, Any],
    timing: Mapping[str, Any],
    argv: Sequence[str],
    environment: Mapping[str, Any],
    protocol_sha256: str,
    gate_sha256: str,
    private_payload_path: str | Path | None = None,
) -> dict[str, Any]:
    for name, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("config_sha256", config_sha256),
        ("split_sha256", split_sha256),
        ("normalizer_sha256", normalizer_sha256),
        ("protocol_sha256", protocol_sha256),
        ("gate_sha256", gate_sha256),
    ):
        if not _is_sha256(value):
            raise SafeCampaignError(f"{name} is invalid")
    if sample_ids.ndim != 1 or len(sample_ids) != len(prediction):
        raise SafeCampaignError("sample_ids do not match evaluation rows")
    cells = group_error_cells(
        prediction=prediction,
        target=target,
        mask=mask,
        group_ids=group_ids,
        dataset=dataset,
        group_salt=group_salt,
    )
    required_files: list[dict[str, Any]] = []
    if private_payload_path is not None:
        payload_path = _atomic_torch_save(
            private_payload_path,
            {
                "prediction": prediction.detach().cpu(),
                "target": target.detach().cpu(),
                "mask": (mask > 0).detach().cpu(),
                "sample_id": sample_ids.detach().cpu(),
                "group_id": group_ids.detach().cpu(),
            },
        )
        digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        required_files.append(
            {
                "path": payload_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": digest,
                "bytes": payload_path.stat().st_size,
                "private": True,
            }
        )
    return {
        "schema_version": SAFE_EVALUATION_SCHEMA,
        "status": "complete",
        "dataset": str(dataset),
        "seed": int(seed),
        "variant": str(variant),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "normalizer_sha256": normalizer_sha256,
        "sample_ids_sha256": tensor_sha256(sample_ids),
        "group_ids_sha256": tensor_sha256(group_ids),
        "mask_sha256": tensor_sha256(mask > 0),
        "protocol_sha256": protocol_sha256,
        "gate_sha256": gate_sha256,
        "selected_hyperparameters": json.loads(
            json.dumps(selected_hyperparameters, sort_keys=True, allow_nan=False)
        ),
        "prediction_sha256": tensor_sha256(prediction),
        "target_sha256": tensor_sha256(target),
        "timing": json.loads(json.dumps(timing, sort_keys=True, allow_nan=False)),
        "argv": [str(value) for value in argv],
        "environment": json.loads(json.dumps(environment, sort_keys=True, allow_nan=False)),
        "metrics": _metrics(cells),
        "cells": cells,
        "required_files": required_files,
        "completed_at_utc": _now(),
    }


def write_evaluation_manifest(
    manifest: Mapping[str, Any], path: str | Path
) -> Path:
    if manifest.get("schema_version") != SAFE_EVALUATION_SCHEMA:
        raise SafeCampaignError("Refusing to write an invalid evaluation manifest")
    destination = require_within_root(path)
    if destination.exists():
        raise SafeCampaignError(f"Refusing to overwrite evaluation: {destination}")
    return atomic_write_json(destination, manifest)
