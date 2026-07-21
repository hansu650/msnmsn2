from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest
import torch

import edgetwincal.fit_cache_campaign as fit_campaign
from edgetwincal import backbone_campaign as backbone
from edgetwincal.campaign_runner import load_cache_manifest
from edgetwincal.config import (
    ResolvedConfig,
    canonical_sha256,
    load_resolved_config,
)
from edgetwincal.paths import PROJECT_ROOT
from edgetwincal.runtime_v2 import (
    ExtractionProvenance,
    read_tensor_cache,
    resolve_run_assets,
    write_tensor_cache,
)
from edgetwincal.schema import atomic_write_json


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _config(tmp_path: Path) -> ResolvedConfig:
    base = load_resolved_config(
        overrides={"dataset": "USHCN", "seed": 2024}
    ).to_dict()
    prefix = _relative(tmp_path)
    base["paths"].update(
        {
            "checkpoint_root": f"{prefix}/checkpoints",
            "protocol_root": f"{prefix}/protocol",
            "cache_root": f"{prefix}/cache",
            "run_root": f"{prefix}/runs",
            "log_root": f"{prefix}/logs",
            "artifact_root": f"{prefix}/artifacts",
        }
    )
    base["datasets"]["USHCN"]["storage"] = f"{prefix}/dataset"
    base["selection"]["datasets"] = ["USHCN"]
    base["selection"]["seeds"] = [2024]
    return ResolvedConfig(base, canonical_sha256(base), tmp_path / "config.json")


def _assets_and_prepared(tmp_path: Path, config: ResolvedConfig):
    cell = backbone.BackboneCell("USHCN", "release_parity")
    assets = resolve_run_assets(
        config, "USHCN", 2024, "release_parity", require_existing=False
    )
    assets.dataset_root.mkdir(parents=True, exist_ok=True)
    assets.apn_config.parent.mkdir(parents=True, exist_ok=True)
    assets.checkpoint.write_bytes(b"frozen-checkpoint")
    assets.apn_config.write_text("seed: 2024\n", encoding="utf-8")
    protocol = {
        "resolved_config_sha256": config.sha256,
        "dataset": "USHCN",
        "protocol": "release_parity",
        "test_constructed": False,
        "task_contract": {"padded_prediction_steps": 3},
    }
    split = {"schema_version": "unit.split.v1"}
    normalizer = {"schema_version": "unit.normalizer.v1"}
    for path, value in (
        (assets.protocol_manifest, protocol),
        (assets.split_manifest, split),
        (assets.normalizer_manifest, normalizer),
    ):
        atomic_write_json(path, value)
    prepared = backbone.PreparedCell(
        cell,
        assets.protocol_manifest,
        assets.split_manifest,
        assets.normalizer_manifest,
    )
    return assets, prepared


def _splits(config: ResolvedConfig):
    task = config["datasets"]["USHCN"]["task"]
    hyper = config["datasets"]["USHCN"]["apn_hyperparameters"]
    horizon = int(task["prediction_steps"])
    channels = int(task["channels"])
    latent = int(hyper["d_model"])
    te_dim = int(hyper["te_dim"])
    output = {}
    for index, (name, rows) in enumerate((("train", 5), ("val", 3))):
        generator = torch.Generator().manual_seed(900 + index)
        output[name] = {
            "features": torch.randn(rows, channels, latent, generator=generator),
            "time_encoding": torch.randn(
                rows, channels, horizon, te_dim, generator=generator
            ),
            "base_prediction": torch.randn(
                rows, horizon, channels, generator=generator
            ),
            "target": torch.randn(rows, horizon, channels, generator=generator),
            "mask": torch.ones(rows, horizon, channels, dtype=torch.bool),
            "sample_id": torch.arange(index * 100, index * 100 + rows),
            "group_id": torch.arange(index * 1000, index * 1000 + rows),
        }
    return output


def _provenance(loader: str = "loader") -> fit_campaign.ProvenanceBundle:
    value = ExtractionProvenance(
        project_commit="1" * 40,
        dataset_raw_sha256=_sha("raw"),
        dataset_processed_sha256=_sha("processed"),
        loader_source_sha256=_sha(loader),
        extractor_source_sha256=_sha("extractor"),
    )
    return fit_campaign.ProvenanceBundle(
        value,
        {
            "project_head": value.project_commit,
            "dataset_raw": {"sha256": value.dataset_raw_sha256, "files": []},
            "dataset_processed": {
                "sha256": value.dataset_processed_sha256,
                "files": [],
            },
            "loader_source": {
                "sha256": value.loader_source_sha256,
                "files": [],
            },
            "extractor_source": {
                "sha256": value.extractor_source_sha256,
                "files": [],
            },
        },
    )


def test_plan_is_ordered_and_mimic_is_explicitly_blocked(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = fit_campaign.campaign_plan(
        config,
        datasets=["USHCN"],
        protocols=["release_parity", "strict_ushcn"],
        seeds=[2024],
    )
    assert [(row["dataset_id"], row["protocol_id"]) for row in plan] == [
        ("USHCN", "release_parity"),
        ("USHCN", "strict_ushcn"),
    ]

    default = load_resolved_config()
    mimic = fit_campaign.campaign_plan(
        default,
        datasets=["MIMIC_III"],
        protocols=["release_parity"],
        seeds=[2024],
    )
    assert mimic[0]["status"] == "BLOCKED[missing_author_mapping]"


def test_composite_hash_binds_relative_path_and_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    digest, records = fit_campaign.composite_sha256([second, first])
    assert len(digest) == 64
    assert [record["path"] for record in records] == sorted(
        [record["path"] for record in records]
    )
    first.write_bytes(b"changed")
    changed, _ = fit_campaign.composite_sha256([first, second])
    assert changed != digest


def test_extraction_config_requires_exact_native_or_registered_legacy_diff(
    tmp_path: Path,
) -> None:
    import yaml

    frozen_path = tmp_path / "configs.yaml"
    base = {"d_model": 24, "dropout": 0.1}
    frozen_path.write_text(
        yaml.safe_dump(base, sort_keys=True), encoding="utf-8"
    )
    native_identity = {
        "verification_mode": "native_train_manifest",
        "dataset_id": "P12",
        "protocol_id": "release_parity",
        "seed": 2024,
    }
    exact = fit_campaign.validate_extraction_config_compatibility(
        frozen_path,
        base,
        checkpoint_identity=native_identity,
        contracts={"train": {}},
    )
    assert exact["mode"] == "exact_native_config"

    legacy_identity = {
        "verification_mode": "verified_legacy_import",
        "dataset_id": "P12",
        "protocol_id": "release_parity",
        "seed": 2024,
    }
    with pytest.raises(
        fit_campaign.FitCacheCampaignError,
        match="unregistered differences",
    ):
        fit_campaign.validate_extraction_config_compatibility(
            frozen_path,
            {**base, "d_model": 25},
            checkpoint_identity=legacy_identity,
            contracts={"train": {}},
        )


def test_extract_then_reuse_is_train_val_only_and_provenance_drift_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    assets, prepared = _assets_and_prepared(tmp_path, config)
    splits = _splits(config)
    calls: list[str] = []

    class Factory:
        def build_test_loader(self, *args, **kwargs):
            raise AssertionError("test loader must never be constructed")

    bundle = types.SimpleNamespace(
        config={"seed": 2024},
        dataset_id="USHCN",
        protocol_id="release_parity",
        seed=2024,
    )
    factory = Factory()

    def fake_prepare_fit_cache(**kwargs):
        calls.append("prepare_fit_cache")
        assert kwargs["train_backbone"] is False
        assert kwargs["loader_factory"] is factory
        cache, manifest = write_tensor_cache(
            config,
            assets,
            splits,
            kwargs["provenance"],
            stem=fit_campaign.FIT_CACHE_STEM,
        )
        sidecar = atomic_write_json(
            kwargs["cache_manifest_path"], manifest.to_dict()
        )
        return {
            "splits_opened": ["train", "val"],
            "test_opened": False,
            "cache_path": cache,
            "cache_manifest_path": sidecar,
            "cache_manifest_sha256": manifest.manifest_digest(),
        }

    runtime = types.SimpleNamespace(
        resolve_run_assets=lambda *args, **kwargs: assets,
        build_train_val_runtime=lambda *args, **kwargs: (
            bundle,
            factory,
            {"train": {"split": "train"}, "val": {"split": "val"}},
        ),
        prepare_fit_cache=fake_prepare_fit_cache,
        load_cache_manifest=load_cache_manifest,
        read_tensor_cache=read_tensor_cache,
        ExtractionProvenance=ExtractionProvenance,
    )
    monkeypatch.setattr(
        fit_campaign,
        "_validate_frozen_backbone",
        lambda *args, **kwargs: {
            "checkpoint": {"sha256": _sha("checkpoint")},
            "configs_yaml": {"sha256": _sha("config")},
            "checkpoint_identity": {
                "verification_mode": "native_train_manifest",
                "dataset_id": "USHCN",
                "protocol_id": "release_parity",
                "seed": 2024,
            },
        },
    )
    monkeypatch.setattr(
        fit_campaign.backbone,
        "write_or_validate_vendor_config",
        lambda *args, **kwargs: assets.apn_config,
    )

    first = fit_campaign.prepare_or_reuse_seed(
        config,
        prepared,
        2024,
        device="cpu",
        runtime=runtime,
        provenance=_provenance(),
        cublas={"observed": backbone.CUBLAS_WORKSPACE_VALUE},
    )
    assert first.outcome == "extracted"
    assert calls == ["prepare_fit_cache"]
    assert first.entry["checkpoint"].endswith("pytorch_model.bin")

    second = fit_campaign.prepare_or_reuse_seed(
        config,
        prepared,
        2024,
        device="cpu",
        runtime=runtime,
        provenance=_provenance(),
        cublas={"observed": backbone.CUBLAS_WORKSPACE_VALUE},
    )
    assert second.outcome == "reused"
    assert calls == ["prepare_fit_cache"]

    with pytest.raises(
        fit_campaign.FitCacheCampaignError,
        match="control drift",
    ):
        fit_campaign.prepare_or_reuse_seed(
            config,
            prepared,
            2024,
            device="cpu",
            runtime=runtime,
            provenance=_provenance("changed-loader"),
            cublas={"observed": backbone.CUBLAS_WORKSPACE_VALUE},
        )


def test_partial_cache_without_control_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    assets, prepared = _assets_and_prepared(tmp_path, config)
    assets.cache_directory.mkdir(parents=True, exist_ok=True)
    (assets.cache_directory / "train_validation_features.orphan.cache").write_bytes(
        b"orphan"
    )
    runtime = types.SimpleNamespace(
        resolve_run_assets=lambda *args, **kwargs: assets,
    )
    monkeypatch.setattr(
        fit_campaign,
        "_validate_frozen_backbone",
        lambda *args, **kwargs: {},
    )
    with pytest.raises(
        fit_campaign.FitCacheCampaignError,
        match="Partial fit-cache",
    ):
        fit_campaign.prepare_or_reuse_seed(
            config,
            prepared,
            2024,
            device="cpu",
            runtime=runtime,
            provenance=_provenance(),
            cublas={"observed": backbone.CUBLAS_WORKSPACE_VALUE},
        )


def test_source_exposes_no_test_construction_api() -> None:
    source = (
        PROJECT_ROOT / "code/src/edgetwincal/fit_cache_campaign.py"
    ).read_text(encoding="utf-8")
    assert "build_test_loader" not in source
    assert "test_ledger_token" not in source
    assert "train_backbone=False" in source



def test_run_campaign_writes_pretest_fit_entries_in_frozen_seed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    assets, prepared = _assets_and_prepared(tmp_path, config)
    cache = tmp_path / "ready.cache"
    sidecar = tmp_path / "ready.json"
    control = tmp_path / "ready-control.json"
    cache.write_bytes(b"cache")
    sidecar.write_text("{}\n", encoding="utf-8")
    control.write_text("{}\n", encoding="utf-8")
    entry = {
        "seed": 2024,
        "fit_cache": _relative(cache),
        "fit_cache_manifest": _relative(sidecar),
        "checkpoint": _relative(assets.checkpoint),
    }

    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG", backbone.CUBLAS_WORKSPACE_VALUE
    )
    monkeypatch.setattr(
        fit_campaign.backbone,
        "prepare_cell",
        lambda config, cell: prepared,
    )
    monkeypatch.setattr(
        fit_campaign,
        "build_provenance_bundle",
        lambda *args, **kwargs: _provenance(),
    )
    monkeypatch.setattr(
        fit_campaign,
        "prepare_or_reuse_seed",
        lambda *args, **kwargs: fit_campaign.FitCacheResult(
            "USHCN",
            "release_parity",
            2024,
            "reused",
            entry,
            control,
        ),
    )
    runtime = types.SimpleNamespace(ExtractionProvenance=ExtractionProvenance)
    results = fit_campaign.run_campaign(
        config,
        datasets=["USHCN"],
        protocols=["release_parity"],
        seeds=[2024],
        device="cpu",
        runtime=runtime,
    )
    assert [result.seed for result in results] == [2024]
    entries_path = (
        Path(config["paths"]["protocol_root"])
        / "USHCN"
        / "release_parity"
        / "fit_entries.json"
    )
    assert json.loads(entries_path.read_text(encoding="utf-8")) == {
        "entries": [entry]
    }
