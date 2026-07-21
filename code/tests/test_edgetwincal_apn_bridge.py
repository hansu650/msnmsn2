from __future__ import annotations

import os
import shutil
import types
import uuid
from collections import namedtuple
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from edgetwincal.apn_bridge import (
    APNBridgeError,
    APNConfigBundle,
    APNTestAccessError,
    FrozenAPNTestLedgerToken,
    ReleaseParityLoaderFactory,
    StrictLoaderFactory,
    apn_forward_callback,
    build_vendor_config,
    collect_apn_fit_splits,
    collect_apn_split,
    configure_vendor_environment,
    extract_apn_batch,
    load_frozen_apn_checkpoint,
    make_apn_model_factory,
    smoke_apn_fit_loaders,
)
from edgetwincal.campaign import REQUIRED_FREEZE_COMPONENTS, ProtocolLedger
from edgetwincal.config import load_resolved_config
from edgetwincal.paths import PROJECT_ROOT


Inputs = namedtuple("Inputs", "t x t_target")
Sample = namedtuple("Sample", "key inputs targets originals")


def _sample(key: int, *, history: int = 4, horizon: int = 2, channels: int = 3):
    time = torch.linspace(0.0, 0.7, history)
    target_time = torch.linspace(0.8, 0.9, horizon)
    values = torch.arange(history * channels, dtype=torch.float32).reshape(
        history, channels
    )
    targets = torch.arange(horizon * channels, dtype=torch.float32).reshape(
        horizon, channels
    )
    return Sample(key, Inputs(time, values, target_time), targets, (time, values))


class _ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _TinyCore(nn.Module):
    def __init__(self, channels: int = 3, latent: int = 3, te_dim: int = 2):
        super().__init__()
        self.N = channels
        self.latent = latent
        self.te_dim = te_dim
        self.batch_size = None
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.decoder = nn.Linear(latent + te_dim, 1)

    def LearnableTE(self, time):
        return torch.cat([time, time.square()], dim=-1)

    def IMTS_Model_Logic(self, x_with_te, mask, time):
        del time
        batch = int(self.batch_size)
        channels = self.N
        observed = mask.sum(dim=1).clamp_min(1.0)
        value = (x_with_te[:, :, :1] * mask).sum(dim=1) / observed
        encoded = x_with_te[:, :, 1:].mean(dim=1)
        latent = torch.cat([value, encoded], dim=-1) * (1.0 + self.scale)
        return latent.view(batch, channels, self.latent)


class _TinyAPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyCore()

    def forward(self, x, x_mark, x_mask, **kwargs):
        extracted = extract_apn_batch(
            self,
            {
                "x": x,
                "x_mark": x_mark,
                "x_mask": x_mask,
                "y": kwargs["y"],
                "y_mark": kwargs["y_mark"],
                "y_mask": kwargs["y_mask"],
                "sample_ID": kwargs["sample_ID"],
            },
            dataset_id="P12",
            protocol_id="strict_p12",
            split="train",
            device=x.device,
        )
        return {
            "pred": extracted["base_prediction"].to(x.device),
            "true": kwargs["y"],
            "mask": kwargs["y_mask"],
        }


def _bundle(protocol: str = "strict_p12") -> APNConfigBundle:
    config = types.SimpleNamespace(
        batch_size=2,
        enc_in=3,
        seq_len_max_irr=None,
        pred_len_max_irr=None,
    )
    return APNConfigBundle(config, "P12", protocol, 2024, {})


def _batch() -> dict[str, torch.Tensor]:
    dataset = _ListDataset([_sample(11), _sample(12)])
    factory = _Prepared(dataset, dataset)
    strict = StrictLoaderFactory(_bundle(), factory)
    return next(iter(strict.build_fit_loader("train")))


class _Prepared:
    dataset_id = "p12"
    protocol_id = "strict_p12"

    def __init__(self, train: Dataset, val: Dataset, test: Dataset | None = None):
        self.datasets = {
            "train": train,
            "val": val,
            "test": test if test is not None else val,
        }
        self.calls: list[tuple[str, object | None]] = []
        self.allowed_token = object()

    def build_dataset(self, partition: str, *, ledger_token=None):
        self.calls.append((partition, ledger_token))
        if partition == "test" and ledger_token is not self.allowed_token:
            raise PermissionError("frozen strict token required")
        return self.datasets[partition]


def _ledger_token() -> tuple[FrozenAPNTestLedgerToken, Path]:
    resolved = load_resolved_config(
        overrides={"dataset": "P12", "seed": 2024, "variant": "APN"}
    )
    scratch = (
        PROJECT_ROOT
        / "results"
        / "edgetwincal_msn2026_v1"
        / "apn_bridge_test_scratch"
        / uuid.uuid4().hex
    )
    scratch.mkdir(parents=True, exist_ok=False)
    scalar = "1" * 64
    components = {
        name: ({"P12": scalar} if name.endswith("manifests") else scalar)
        for name in REQUIRED_FREEZE_COMPONENTS
    }
    ledger = ProtocolLedger.create(
        scratch / "ledger.json",
        resolved_config=resolved,
        components=components,
        pretest_checks={"unit_test": True},
    )
    ledger.freeze()
    opening = ledger.open_test_once(
        dataset="P12",
        protocol="release_parity",
        fold="seed2024",
        split_manifest_sha256=scalar,
        normalization_manifest_sha256=scalar,
    )
    token = FrozenAPNTestLedgerToken.from_protocol_ledger(
        ledger, cell_id=opening["cell_id"], token=opening["token"]
    )
    return token, scratch


def test_module_import_is_vendor_lazy_and_environment_changes_only_two_keys(monkeypatch):
    before_home = os.environ.get("HOME")
    before_codex = os.environ.get("CODEX_HOME")
    result = configure_vendor_environment()
    assert set(result) == {"EVIPATCH_PROJECT_ROOT", "EVIPATCH_TSDM_ROOT"}
    assert Path(result["EVIPATCH_PROJECT_ROOT"]) == PROJECT_ROOT
    assert Path(result["EVIPATCH_TSDM_ROOT"]) == PROJECT_ROOT / "data" / "tsdm"
    assert os.environ.get("HOME") == before_home
    assert os.environ.get("CODEX_HOME") == before_codex
    with pytest.raises((ValueError, APNBridgeError)):
        configure_vendor_environment(tsdm_root=PROJECT_ROOT.parent / "msn")


def test_vendor_config_uses_registry_and_overwrites_absolute_template_paths():
    resolved = load_resolved_config(
        overrides={"dataset": "P12", "seed": 2024, "variant": "APN"}
    )
    bundle = build_vendor_config(
        resolved,
        dataset_id="P12",
        protocol_id="strict_p12",
        seed=2024,
        config_type=types.SimpleNamespace,
    )
    assert bundle.config.dataset_name == "P12"
    assert bundle.config.seq_len == 36
    assert bundle.config.pred_len == 3
    assert bundle.config.enc_in == 36
    assert bundle.config.evipatch_mode == "apn"
    assert bundle.config.skip_test_after_train == 1
    assert Path(bundle.config.dataset_root_path).is_relative_to(PROJECT_ROOT)
    assert Path(bundle.config.checkpoints).is_relative_to(PROJECT_ROOT)
    assert Path(bundle.config.checkpoints).name == "seed_2024"
    assert bundle.public_audit()["home_overridden"] is False


def test_model_factory_imports_only_when_invoked_and_checks_seed():
    calls = []

    class VendorModel(_TinyAPN):
        def __init__(self, config):
            super().__init__()
            self.config = config

    def importer(name):
        calls.append(name)
        return types.SimpleNamespace(Model=VendorModel)

    factory = make_apn_model_factory(_bundle(), importer=importer)
    assert calls == []
    assert isinstance(factory(2024), VendorModel)
    assert calls == ["models.APN"]
    with pytest.raises(APNBridgeError):
        factory(2025)


def test_release_factory_never_accepts_test_through_training_api_and_labels_scan():
    bundle = _bundle("release_parity")
    calls = []
    dataset = _ListDataset([_sample(1), _sample(2)])

    def provider(config, split):
        calls.append(split)
        if split == "train":
            config.seq_len_max_irr = 4
            config.pred_len_max_irr = 2
        loader = DataLoader(dataset, batch_size=2)
        return dataset, loader

    factory = ReleaseParityLoaderFactory(bundle, provider=provider)
    with pytest.raises(APNTestAccessError):
        factory("test", 2024)
    with pytest.raises(APNTestAccessError):
        factory.build_test_loader(test_ledger_token=None)  # type: ignore[arg-type]
    factory("train", 2024)
    factory("val", 2024)
    assert calls == ["train", "val"]
    contract = factory.contract("train").public_manifest()
    assert contract["scans_test_during_train_construction"] is True
    assert contract["padding_fit_scope"].startswith("all_partitions_including_test")
    assert contract["padded_prediction_steps"] == 2

    token, scratch = _ledger_token()
    try:
        factory.build_test_loader(test_ledger_token=token)
        assert calls == ["train", "val", "test"]
        assert "token=" not in repr(token)
        assert token.public_manifest()["token_sha256"] not in {"", None}
    finally:
        shutil.rmtree(scratch)


def test_strict_factory_freezes_padding_from_train_val_and_never_touches_test():
    train = _ListDataset([_sample(1, history=3), _sample(2, history=4)])
    val = _ListDataset([_sample(3, history=5)])
    test = _ListDataset([_sample(4, history=5)])
    prepared = _Prepared(train, val, test)
    factory = StrictLoaderFactory(_bundle(), prepared)
    assert [partition for partition, _ in prepared.calls] == ["train", "val"]
    contract = factory.contract("train").public_manifest()
    assert contract["scans_test_during_train_construction"] is False
    assert contract["padding_fit_scope"] == "train_and_validation_only_before_test_opening"
    assert contract["padded_history_steps"] == 5
    assert contract["padded_prediction_steps"] == 2
    assert factory.bundle.config.seq_len_max_irr == 5
    assert factory.bundle.config.pred_len_max_irr == 2

    train_batch = next(iter(factory.build_fit_loader("train")))
    assert train_batch["x"].shape == (2, 5, 3)
    assert train_batch["y"].shape == (2, 2, 3)
    assert train_batch["sample_ID"].dtype == torch.int64
    with pytest.raises(APNTestAccessError):
        factory.build_test_loader(test_ledger_token=None)
    factory.build_test_loader(test_ledger_token=prepared.allowed_token)
    assert [partition for partition, _ in prepared.calls] == ["train", "val", "test"]


def test_strict_test_longer_than_frozen_train_val_fails_closed_on_iteration():
    fit = _ListDataset([_sample(1, history=4)])
    test = _ListDataset([_sample(2, history=5)])
    prepared = _Prepared(fit, fit, test)
    factory = StrictLoaderFactory(_bundle(), prepared)
    loader = factory.build_test_loader(test_ledger_token=prepared.allowed_token)
    with pytest.raises(APNBridgeError, match="exceeds padding"):
        next(iter(loader))


def test_apn_forward_callback_rejects_test_and_preserves_masked_triplet():
    model = _TinyAPN()
    batch = _batch()
    output = apn_forward_callback(model, batch, "train")
    assert output["pred"].shape == batch["y"].shape
    assert torch.equal(output["true"], batch["y"])
    assert torch.equal(output["mask"], batch["y_mask"])
    with pytest.raises(APNBridgeError):
        apn_forward_callback(model, batch, "test")  # type: ignore[arg-type]


def test_extract_shapes_match_runtime_v2_and_ids_are_private_stable():
    model = _TinyAPN().eval()
    batch = _batch()
    first = extract_apn_batch(
        model,
        batch,
        dataset_id="P12",
        protocol_id="strict_p12",
        split="train",
        device="cpu",
    )
    second = extract_apn_batch(
        model,
        batch,
        dataset_id="P12",
        protocol_id="strict_p12",
        split="train",
        device="cpu",
    )
    assert first["features"].shape == (2, 3, 3)
    assert first["time_encoding"].shape == (2, 3, 2, 2)
    assert first["base_prediction"].shape == (2, 2, 3)
    assert first["target"].shape == (2, 2, 3)
    assert first["mask"].shape == (2, 2, 3)
    assert first["mask"].dtype == torch.bool
    assert first["sample_id"].dtype == torch.int64
    assert first["group_id"].dtype == torch.int64
    assert torch.equal(first["sample_id"], second["sample_id"])
    assert not set(first["sample_id"].tolist()).intersection({11, 12})
    with pytest.raises(APNTestAccessError):
        extract_apn_batch(
            model,
            batch,
            dataset_id="P12",
            protocol_id="strict_p12",
            split="test",
            device="cpu",
        )


def test_collect_fit_splits_has_no_test_surface_and_rejects_duplicate_ids():
    dataset = _ListDataset([_sample(11), _sample(12)])
    prepared = _Prepared(dataset, dataset)
    factory = StrictLoaderFactory(_bundle(), prepared)
    model = _TinyAPN().eval()
    splits = collect_apn_fit_splits(model, factory, device="cpu")
    assert set(splits) == {"train", "val"}
    assert all(value["sample_id"].shape == (2,) for value in splits.values())
    assert [partition for partition, _ in prepared.calls] == ["train", "val"]

    batch = _batch()
    with pytest.raises(APNBridgeError, match="collide or repeat"):
        collect_apn_split(
            model,
            [batch, batch],
            dataset_id="P12",
            protocol_id="strict_p12",
            split="train",
            device="cpu",
        )
    with pytest.raises(APNTestAccessError):
        collect_apn_split(
            model,
            [],
            dataset_id="P12",
            protocol_id="strict_p12",
            split="test",
            device="cpu",
        )


def test_train_val_forward_smoke_contract_cannot_open_test():
    dataset = _ListDataset([_sample(21), _sample(22)])
    prepared = _Prepared(dataset, dataset)
    factory = StrictLoaderFactory(_bundle(), prepared)
    manifest = smoke_apn_fit_loaders(_TinyAPN(), factory, device="cpu")
    assert manifest["splits_opened"] == ["train", "val"]
    assert manifest["test_opened"] is False
    assert set(manifest["splits"]) == {"train", "val"}
    assert all(cell["finite_prediction"] for cell in manifest["splits"].values())
    assert [partition for partition, _ in prepared.calls] == ["train", "val"]

def test_checkpoint_load_is_weights_only_strict_and_frozen(tmp_path):
    source = _TinyAPN()
    checkpoint = tmp_path / "pytorch_model.bin"
    torch.save(source.state_dict(), checkpoint)
    loaded = load_frozen_apn_checkpoint(
        _bundle(),
        checkpoint,
        device="cpu",
        model_factory=lambda seed: _TinyAPN(),
    )
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for name, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
