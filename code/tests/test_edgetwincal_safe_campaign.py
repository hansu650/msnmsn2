from __future__ import annotations

import hashlib

import pytest
import torch

from edgetwincal.safe_campaign import (
    SafeCampaignError,
    SAFE_LEDGER_SCHEMA,
    SafeTestLedger,
    build_evaluation_manifest,
    write_evaluation_manifest,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _gates():
    return {
        dataset: {
            "enabled": True,
            "checkpoints": 5,
            "validation_groups": 20,
            "validation_cells": 400,
            "gate_sha256": _hash(dataset),
        }
        for dataset in ("beijing_air", "intel_lab")
    }


def test_test_ledger_opens_each_target_once_and_seals(tmp_path, monkeypatch):
    from edgetwincal import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("edgetwincal.safe_campaign.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("edgetwincal.safe_campaign.require_within_root", lambda value: (tmp_path / value).resolve() if not __import__("pathlib").Path(value).is_absolute() else __import__("pathlib").Path(value).resolve())
    monkeypatch.setattr("edgetwincal.schema.PROJECT_ROOT", tmp_path)
    ledger = SafeTestLedger.create(
        "results/ledger.json",
        config_sha256=_hash("config"),
        components={"source": _hash("source")},
        gate_decisions=_gates(),
    )
    ledger.freeze()
    for dataset in ("beijing_air", "intel_lab"):
        token = ledger.open_dataset_once(dataset)
        with pytest.raises(SafeCampaignError, match="already opened"):
            ledger.open_dataset_once(dataset)
        with pytest.raises(SafeCampaignError, match="Invalid"):
            ledger.validate_token(dataset, "bad")
        ledger.close_dataset(dataset, token)
    ledger.seal()
    assert ledger.status == "sealed"
    assert SAFE_LEDGER_SCHEMA == "edgetwincal.safe-campaign-test-ledger.v1"
    assert ledger.data["schema_version"] == SAFE_LEDGER_SCHEMA
    assert all("token" not in str(event) for event in ledger.data["events"])


def test_evaluation_manifest_has_paired_cells_and_private_arrays(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr("edgetwincal.safe_campaign.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "edgetwincal.safe_campaign.require_within_root",
        lambda value: (
            Path(value).resolve()
            if Path(value).is_absolute()
            else (tmp_path / value).resolve()
        ),
    )
    monkeypatch.setattr("edgetwincal.schema.PROJECT_ROOT", tmp_path)
    prediction = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    target = torch.tensor([[[0.0], [2.0]], [[2.0], [6.0]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    manifest = build_evaluation_manifest(
        dataset="beijing_air",
        seed=2024,
        variant="APN",
        prediction=prediction,
        target=target,
        mask=mask,
        sample_ids=torch.tensor([10, 11]),
        group_ids=torch.tensor([7, 8]),
        group_salt="safe:",
        checkpoint_sha256=_hash("checkpoint"),
        config_sha256=_hash("config"),
        split_sha256=_hash("split"),
        normalizer_sha256=_hash("normalizer"),
        selected_hyperparameters={},
        timing={"segments": [{"name": "inference", "seconds": 0.1}]},
        argv=["safe", "test"],
        environment={"python": "test"},
        protocol_sha256=_hash("protocol"),
        gate_sha256=_hash("gate"),
        private_payload_path="results/private.pt",
    )
    assert manifest["metrics"]["n"] == 4
    assert manifest["metrics"]["mse"] == pytest.approx(1.5)
    assert len(manifest["prediction_sha256"]) == 64
    assert len(manifest["target_sha256"]) == 64
    assert len(manifest["cells"]) == 2
    destination = write_evaluation_manifest(manifest, "results/run.json")
    assert destination.is_file()
    with pytest.raises(SafeCampaignError, match="overwrite"):
        write_evaluation_manifest(manifest, "results/run.json")


def test_observed_nonfinite_prediction_is_rejected(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr("edgetwincal.safe_campaign.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "edgetwincal.safe_campaign.require_within_root",
        lambda value: (
            Path(value).resolve()
            if Path(value).is_absolute()
            else (tmp_path / value).resolve()
        ),
    )
    prediction = torch.tensor([[[float("nan")]]])
    with pytest.raises(SafeCampaignError, match="finite"):
        build_evaluation_manifest(
            dataset="intel_lab",
            seed=2024,
            variant="Safe",
            prediction=prediction,
            target=torch.zeros_like(prediction),
            mask=torch.ones_like(prediction, dtype=torch.bool),
            sample_ids=torch.tensor([1]),
            group_ids=torch.tensor([1]),
            group_salt="safe:",
            checkpoint_sha256=_hash("checkpoint"),
            config_sha256=_hash("config"),
            split_sha256=_hash("split"),
            normalizer_sha256=_hash("normalizer"),
            selected_hyperparameters={},
            timing={"segments": [{"name": "inference", "seconds": 0.1}]},
            argv=["safe"],
            environment={"python": "test"},
            protocol_sha256=_hash("protocol"),
            gate_sha256=_hash("gate"),
        )
