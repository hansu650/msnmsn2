from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import edgetwincal.experiment as experiment
import edgetwincal.campaign_extract as campaign_extract

from edgetwincal.apn_bridge import (
    APNConfigBundle,
    FrozenAPNTestLedgerToken,
    StrictLoaderFactory,
)
from edgetwincal.campaign import ProtocolLedger, REQUIRED_FREEZE_COMPONENTS
from edgetwincal.campaign_extract import prepare_fit_cache
from edgetwincal.campaign_evaluate import (
    TEST_ACCESS_SCHEMA,
    _strict_dataset_token,
    evaluate_campaign_once,
    write_test_cache_registry,
)
from edgetwincal.campaign_pretest import (
    fit_and_freeze_registry,
    load_fitted_registry,
    load_fitted_states,
    prepare_pretest_manifests,
)
from edgetwincal.campaign_runner import CachedAPNDecoder, _apply_variant
from edgetwincal.config import ResolvedConfig, canonical_sha256, load_resolved_config
from edgetwincal.paths import PROJECT_ROOT
from edgetwincal.runtime_v2 import (
    ExtractionProvenance,
    resolve_run_assets,
    write_tensor_cache,
)
from edgetwincal.schema import atomic_write_json, sha256_file
from edgetwincal.strict_p12 import prepare_strict_p12


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _isolated_config(tmp_path: Path, dataset: str, protocol: str) -> ResolvedConfig:
    base = load_resolved_config(
        overrides={"dataset": dataset, "seed": 2024}
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
    base["datasets"][dataset]["storage"] = f"{prefix}/dataset"
    base["selection"]["datasets"] = [dataset]
    base["selection"]["seeds"] = [2024]
    return ResolvedConfig(base, canonical_sha256(base), tmp_path / "config.json")


def _make_assets(config: ResolvedConfig, dataset: str, protocol: str):
    assets = resolve_run_assets(
        config, dataset, 2024, protocol, require_existing=False
    )
    assets.dataset_root.mkdir(parents=True, exist_ok=True)
    assets.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    assets.checkpoint.write_bytes(b"synthetic-frozen-checkpoint")
    assets.apn_config.write_text("synthetic: true\n", encoding="utf-8")
    return assets


def _splits(config: ResolvedConfig, dataset: str):
    task = config["datasets"][dataset]["task"]
    hyper = config["datasets"][dataset]["apn_hyperparameters"]
    horizon = int(task["prediction_steps"])
    channels = int(task["channels"])
    latent = int(hyper["d_model"])
    te_dim = int(hyper["te_dim"])
    output = {}
    for split_index, (split, rows) in enumerate(
        (("train", 10), ("val", 6), ("test", 5))
    ):
        generator = torch.Generator().manual_seed(700 + split_index)
        features = torch.randn(rows, channels, latent, generator=generator)
        base = torch.randn(rows, horizon, channels, generator=generator)
        correction = 0.25 * features[:, :, 0].unsqueeze(1).expand(-1, horizon, -1)
        target = base + correction
        start = 1000 * (split_index + 1)
        output[split] = {
            "features": features,
            "time_encoding": torch.randn(
                rows, channels, horizon, te_dim, generator=generator
            ),
            "base_prediction": base,
            "target": target,
            "mask": torch.ones(rows, horizon, channels, dtype=torch.bool),
            "sample_id": torch.arange(start, start + rows, dtype=torch.int64),
            "group_id": torch.arange(start + 100, start + 100 + rows, dtype=torch.int64),
        }
    return output


def _decoder_parity_splits(
    config: ResolvedConfig,
    dataset: str,
    decoder: CachedAPNDecoder,
):
    task = config["datasets"][dataset]["task"]
    hyper = config["datasets"][dataset]["apn_hyperparameters"]
    horizon = int(task["prediction_steps"])
    channels = int(task["channels"])
    latent = int(hyper["d_model"])
    te_dim = int(hyper["te_dim"])
    output = {}
    decoder.eval()
    for split_index, (split, rows) in enumerate(
        (("train", 10), ("val", 6), ("test", 5))
    ):
        generator = torch.Generator().manual_seed(1700 + split_index)
        features = torch.randn(rows, channels, latent, generator=generator)
        time_encoding = torch.randn(
            rows, channels, horizon, te_dim, generator=generator
        )
        with torch.no_grad():
            base = decoder(
                {"features": features, "time_encoding": time_encoding}
            )
        correction = 0.1 * features[:, :, 0].unsqueeze(1).expand(
            -1, horizon, -1
        )
        start = 1000 * (split_index + 1)
        output[split] = {
            "features": features,
            "time_encoding": time_encoding,
            "base_prediction": base,
            "target": base + correction,
            "mask": torch.ones(rows, horizon, channels, dtype=torch.bool),
            "sample_id": torch.arange(start, start + rows, dtype=torch.int64),
            "group_id": torch.arange(
                start + 100, start + 100 + rows, dtype=torch.int64
            ),
        }
    return output


def _provenance() -> ExtractionProvenance:
    return ExtractionProvenance(
        project_commit="1" * 40,
        dataset_raw_sha256=_sha("raw"),
        dataset_processed_sha256=_sha("processed"),
        loader_source_sha256=_sha("loader"),
        extractor_source_sha256=_sha("extractor"),
    )


def _components(
    *, registry_sha: str, cell_id: str, split_sha: str, normalizer_sha: str
):
    scalar = _sha("frozen-component")
    values = {name: scalar for name in REQUIRED_FREEZE_COMPONENTS}
    values["variant_registry_sha256"] = registry_sha
    values["split_manifests"] = {cell_id: split_sha}
    values["normalization_manifests"] = {cell_id: normalizer_sha}
    return values


def test_pretest_fit_and_single_opening_campaign_e2e(tmp_path: Path) -> None:
    config = _isolated_config(tmp_path, "USHCN", "release_parity")
    prepared = prepare_pretest_manifests(
        config,
        dataset_id="USHCN",
        protocol_id="release_parity",
        seed=2024,
    )
    assets = _make_assets(config, "USHCN", "release_parity")
    tensors = _splits(config, "USHCN")
    fit_cache, fit_manifest = write_tensor_cache(
        config,
        assets,
        {"train": tensors["train"], "val": tensors["val"]},
        _provenance(),
        stem="fit",
    )
    fit_sidecar = atomic_write_json(
        tmp_path / "fit_manifest.json", fit_manifest.to_dict()
    )
    test_cache, test_manifest = write_tensor_cache(
        config,
        assets,
        {"test": tensors["test"]},
        _provenance(),
        stem="test",
    )
    test_sidecar = atomic_write_json(
        tmp_path / "test_manifest.json", test_manifest.to_dict()
    )

    fitted_path = fit_and_freeze_registry(
        config,
        dataset_id="USHCN",
        protocol_id="release_parity",
        entries=[
            {
                "seed": 2024,
                "fit_cache": fit_cache,
                "fit_cache_manifest": fit_sidecar,
            }
        ],
        output_path=tmp_path / "fitted_registry.json",
        require_all_selected_seeds=True,
    )
    fitted_registry = load_fitted_registry(fitted_path)
    cell_id = "USHCN|release_parity|fold-0"
    ledger = ProtocolLedger.create(
        tmp_path / "ledger.json",
        resolved_config=config,
        components=_components(
            registry_sha=fitted_registry["variant_registry_sha256"],
            cell_id=cell_id,
            split_sha=prepared.split_manifest_sha256,
            normalizer_sha=prepared.normalizer_manifest_sha256,
        ),
        pretest_checks={"unit_tests": True, "cache_provenance": True},
    )
    ledger.freeze()
    opening = ledger.open_test_once(
        dataset="USHCN",
        protocol="release_parity",
        fold="fold-0",
        split_manifest_sha256=prepared.split_manifest_sha256,
        normalization_manifest_sha256=prepared.normalizer_manifest_sha256,
    )
    access_path = atomic_write_json(
        tmp_path / "test_access.json",
        {
            "schema_version": TEST_ACCESS_SCHEMA,
            "cell_id": cell_id,
            "dataset": "USHCN",
            "protocol": "release_parity",
            "seed": 2024,
            "protocol_sha256": opening["protocol_sha256"],
            "token_sha256": hashlib.sha256(opening["token"].encode("ascii")).hexdigest(),
            "cache_path": _relative(test_cache),
            "cache_manifest_path": _relative(test_sidecar),
            "cache_manifest_sha256": test_manifest.manifest_digest(),
        },
    )
    test_registry = write_test_cache_registry(
        tmp_path / "test_registry.json",
        dataset_id="USHCN",
        protocol_id="release_parity",
        cell_id=cell_id,
        entries=[
            {
                "seed": 2024,
                "test_cache": _relative(test_cache),
                "test_cache_manifest": _relative(test_sidecar),
                "test_access_manifest": _relative(access_path),
            }
        ],
    )
    result = evaluate_campaign_once(
        ledger_path=ledger.path,
        cell_id=cell_id,
        token=opening["token"],
        fitted_registry_path=fitted_path,
        test_cache_registry_path=test_registry,
        run_root=tmp_path / "campaign_runs",
        require_all_selected_seeds=True,
    )
    assert result["status"] == "complete"
    assert len(result["run_manifests"]) == 4
    assert result["test_opening_consumed"] is True
    assert ProtocolLedger.load(ledger.path).status == "frozen"
    for relative in result["run_manifests"]:
        manifest = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["completed_phases"] == [
            "frozen_train_validation_state_loaded",
            "once_opened_test_evaluation",
        ]


def _p12_frame() -> pd.DataFrame:
    patients = [f"PRIVATE-{index:02d}" for index in range(20)]
    times = (0, 12, 35, 36, 37, 38)
    index = pd.MultiIndex.from_product(
        [patients, times], names=("RecordID", "Time")
    )
    values = np.arange(len(index) * 2, dtype=np.float64).reshape(len(index), 2)
    return pd.DataFrame(values, index=index, columns=("a", "b"))


def test_generic_opening_safely_adapts_to_real_strict_p12_token(
    tmp_path: Path,
) -> None:
    strict = prepare_strict_p12(
        _p12_frame(),
        expected_channels=2,
        code_hash=_sha("strict-code"),
        data_asset_hashes={"synthetic": _sha("strict-data")},
    )
    config = _isolated_config(tmp_path, "P12", "strict_p12")
    prepared = prepare_pretest_manifests(
        config,
        dataset_id="P12",
        protocol_id="strict_p12",
        seed=2024,
        strict_public_bundle=strict.public_manifests(),
    )
    assets = _make_assets(config, "P12", "strict_p12")
    cell_id = "P12|strict_p12|fold-0"
    registry_sha = _sha("strict-fitted-registry")
    ledger = ProtocolLedger.create(
        tmp_path / "strict_ledger.json",
        resolved_config=config,
        components=_components(
            registry_sha=registry_sha,
            cell_id=cell_id,
            split_sha=prepared.split_manifest_sha256,
            normalizer_sha=prepared.normalizer_manifest_sha256,
        ),
        pretest_checks={"strict_protocol": True},
    )
    ledger.freeze()
    opening = ledger.open_test_once(
        dataset="P12",
        protocol="strict_p12",
        fold="fold-0",
        split_manifest_sha256=prepared.split_manifest_sha256,
        normalization_manifest_sha256=prepared.normalizer_manifest_sha256,
    )
    generic = FrozenAPNTestLedgerToken.from_protocol_ledger(
        ledger, cell_id=cell_id, token=opening["token"]
    )
    bundle = APNConfigBundle(
        types.SimpleNamespace(batch_size=2, enc_in=2),
        "P12",
        "strict_p12",
        2024,
        {},
    )
    factory = StrictLoaderFactory(bundle, strict)
    specialized = _strict_dataset_token(
        loader_factory=factory,
        assets=assets,
        ledger=ledger,
        opening=ledger.data["test_openings"][cell_id],
    )
    generic.validate_for("P12", "strict_p12")
    loader = factory.build_test_loader(
        test_ledger_token=specialized, purpose="extraction"
    )
    batch = next(iter(loader))
    assert batch["sample_ID"].numel() == 2
    assert all("PRIVATE-" not in str(value) for value in specialized.public_manifest().values())



def test_strict_registry_roundtrips_all_controls_without_test_data(
    tmp_path: Path,
) -> None:
    config = _isolated_config(tmp_path, "USHCN", "strict_ushcn")
    prepare_pretest_manifests(
        config,
        dataset_id="USHCN",
        protocol_id="strict_ushcn",
        seed=2024,
        strict_public_bundle={
            "split": {
                "schema_version": "synthetic.strict-ushcn-split.v1",
                "group_counts": {"train": 10, "val": 6, "test": 5},
            },
            "normalization": {
                "schema_version": "synthetic.strict-ushcn-normalizer.v1",
                "fit_scope": "observed_train_only",
            },
        },
    )
    assets = _make_assets(config, "USHCN", "strict_ushcn")
    hyper = config["datasets"]["USHCN"]["apn_hyperparameters"]
    decoder = CachedAPNDecoder(
        int(hyper["d_model"]),
        int(hyper["te_dim"]),
        float(hyper["dropout"]),
    )
    torch.save(decoder.state_dict(), assets.checkpoint)
    tensors = _decoder_parity_splits(config, "USHCN", decoder)
    fit_cache, fit_manifest = write_tensor_cache(
        config,
        assets,
        {"train": tensors["train"], "val": tensors["val"]},
        _provenance(),
        stem="strict_fit",
    )
    fit_sidecar = atomic_write_json(
        tmp_path / "strict_fit_manifest.json", fit_manifest.to_dict()
    )
    fitted_path = fit_and_freeze_registry(
        config,
        dataset_id="USHCN",
        protocol_id="strict_ushcn",
        entries=[
            {
                "seed": 2024,
                "fit_cache": fit_cache,
                "fit_cache_manifest": fit_sidecar,
                "checkpoint": assets.checkpoint,
            }
        ],
        output_path=tmp_path / "strict_fitted_registry.json",
        require_all_selected_seeds=True,
    )
    registry = load_fitted_registry(fitted_path)
    loaded = load_fitted_states(config, registry["seeds"][0])
    assert loaded.variant_ids == (
        "APN",
        "SLRH",
        "CFG",
        "Full",
        "V01",
        "V02",
        "V03",
        "V07",
        "V08",
        "V10",
        "V11",
        "V12",
    )
    for variant in loaded.variant_ids:
        prediction = _apply_variant(loaded, variant, tensors["val"])
        assert prediction.shape == tensors["val"]["target"].shape
        assert torch.isfinite(prediction).all()



def test_prepare_fit_cache_has_no_test_loader_surface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _isolated_config(tmp_path, "USHCN", "release_parity")
    prepare_pretest_manifests(
        config,
        dataset_id="USHCN",
        protocol_id="release_parity",
        seed=2024,
    )
    assets = _make_assets(config, "USHCN", "release_parity")
    tensors = _splits(config, "USHCN")
    bundle = APNConfigBundle(
        types.SimpleNamespace(),
        "USHCN",
        "release_parity",
        2024,
        {},
    )

    class FitOnlyFactory:
        def __init__(self):
            self.bundle = bundle
            self.test_calls = 0

        def build_test_loader(self, *args, **kwargs):
            self.test_calls += 1
            raise AssertionError("pre-test fit must never construct test")

    factory = FitOnlyFactory()
    model = object()

    def fake_load(*args, **kwargs):
        return model

    def fake_collect(actual_model, actual_factory, *, device):
        assert actual_model is model
        assert actual_factory is factory
        assert device == "cpu"
        return {"train": tensors["train"], "val": tensors["val"]}

    monkeypatch.setattr(
        campaign_extract, "load_frozen_apn_checkpoint", fake_load
    )
    monkeypatch.setattr(campaign_extract, "collect_apn_fit_splits", fake_collect)
    result = prepare_fit_cache(
        config=config,
        assets=assets,
        provenance=_provenance(),
        bundle=bundle,
        loader_factory=factory,
        cache_manifest_path=tmp_path / "prepared_fit_manifest.json",
        device="cpu",
        train_backbone=False,
    )
    assert result["splits_opened"] == ["train", "val"]
    assert result["test_opened"] is False
    assert result["checkpoint_trained"] is False
    assert factory.test_calls == 0



def test_pretest_cli_prepares_and_freezes_fit_registry(
    tmp_path: Path,
    capsys,
) -> None:
    config = _isolated_config(tmp_path, "USHCN", "release_parity")
    config_path = atomic_write_json(tmp_path / "config.json", config.to_dict())
    code = experiment.main(
        [
            "pretest",
            "prepare",
            "--config",
            str(config_path),
            "--dataset",
            "USHCN",
            "--seed",
            "2024",
            "--dataset-id",
            "USHCN",
            "--protocol-id",
            "release_parity",
            "--preparation-seed",
            "2024",
        ]
    )
    assert code == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["test_constructed"] is False

    assets = _make_assets(config, "USHCN", "release_parity")
    tensors = _splits(config, "USHCN")
    fit_cache, fit_manifest = write_tensor_cache(
        config,
        assets,
        {"train": tensors["train"], "val": tensors["val"]},
        _provenance(),
        stem="cli_fit",
    )
    fit_sidecar = atomic_write_json(
        tmp_path / "cli_fit_manifest.json", fit_manifest.to_dict()
    )
    entries = atomic_write_json(
        tmp_path / "cli_entries.json",
        {
            "entries": [
                {
                    "seed": 2024,
                    "fit_cache": str(fit_cache),
                    "fit_cache_manifest": str(fit_sidecar),
                }
            ]
        },
    )
    output = tmp_path / "cli_fitted_registry.json"
    code = experiment.main(
        [
            "pretest",
            "fit",
            "--config",
            str(config_path),
            "--dataset",
            "USHCN",
            "--seed",
            "2024",
            "--dataset-id",
            "USHCN",
            "--protocol-id",
            "release_parity",
            "--entries",
            str(entries),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seed_count"] == 1
    assert payload["test_opened"] is False
    assert output.exists()
