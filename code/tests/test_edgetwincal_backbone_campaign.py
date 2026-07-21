from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import edgetwincal.backbone_campaign as backbone_campaign
from edgetwincal.backbone_campaign import (
    CUBLAS_WORKSPACE_VALUE,
    BackboneBlockedError,
    BackboneCampaignError,
    BackboneCell,
    PreparedCell,
    campaign_plan,
    ensure_cublas_workspace_config,
    prepare_cell,
    select_cells,
    select_seeds,
    train_or_reuse_cell,
    validate_reusable_checkpoint,
    vendor_config_snapshot,
)
from edgetwincal.config import canonical_sha256, load_resolved_config
from edgetwincal.paths import PROJECT_ROOT


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _checkpoint_manifest(
    config,
    cell: BackboneCell,
    seed: int,
    checkpoint: Path,
    argv: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "action": "train_apn_backbone",
        "seed": seed,
        "resolved_config": config.to_dict(),
        "resolved_config_sha256": config.sha256,
        "argv": argv
        or [
            "run_edgetwincal_backbones.py",
            "--dataset",
            cell.dataset_id,
            "--protocol",
            cell.protocol_id,
            "--seed",
            str(seed),
            "--execute",
        ],
        "training_policy": {
            "optimizer": "Adam",
            "loss": "MSE",
            "epochs": 200,
            "patience": 10,
            "validation_interval": 1,
            "learning_rate": config["datasets"][cell.dataset_id][
                "apn_hyperparameters"
            ]["learning_rate"],
        },
        "checkpoint": {
            "path": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _hash(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "weights_only": True,
            "atomic_replace": True,
            "state_sha256": "0" * 64,
        },
    }


def _protocol_files(root: Path, config, cell: BackboneCell) -> PreparedCell:
    split = {
        "schema_version": "unit.split.v1",
        "dataset": cell.dataset_id,
        "protocol": cell.protocol_id,
    }
    normalizer = {
        "schema_version": "unit.normalizer.v1",
        "dataset": cell.dataset_id,
        "protocol": cell.protocol_id,
    }
    protocol = {
        "schema_version": "unit.protocol.v1",
        "resolved_config_sha256": config.sha256,
        "dataset": cell.dataset_id,
        "protocol": cell.protocol_id,
        "test_constructed": False,
        "task_contract": {"padded_prediction_steps": 3},
        "split_content_sha256": canonical_sha256(split),
        "normalizer_content_sha256": canonical_sha256(normalizer),
    }
    paths = {
        "protocol": root / "protocol_manifest.json",
        "split": root / "split_manifest.json",
        "normalizer": root / "normalizer_manifest.json",
    }
    _write_json(paths["protocol"], protocol)
    _write_json(paths["split"], split)
    _write_json(paths["normalizer"], normalizer)
    return PreparedCell(
        cell,
        paths["protocol"],
        paths["split"],
        paths["normalizer"],
    )


class _Contract:
    def __init__(self, split: str) -> None:
        self.split = split

    def public_manifest(self):
        return {
            "requested_split": self.split,
            "padded_prediction_steps": 3,
            "scans_test_during_train_construction": False,
        }


class _TrainValOnlyFactory:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    def build_fit_loader(self, split: str, *, purpose: str):
        assert split in {"train", "val"}
        self.calls.append(("build_fit_loader", split))
        return [split]

    def contract(self, split: str):
        assert split in {"train", "val"}
        return _Contract(split)

    def __call__(self, split: str, seed: int):
        assert split in {"train", "val"}
        self.calls.append(("trainer_loader", split))
        return [split]

    def build_test_loader(self, *args, **kwargs):  # pragma: no cover - sentinel only.
        raise AssertionError("test loader must never be constructed")


class _Bundle:
    def __init__(self, run_dir: Path, seed: int) -> None:
        self.config = {
            "dataset_root_path": str(PROJECT_ROOT / "data" / "tsdm"),
            "checkpoints": str(run_dir),
            "seed_base": seed,
            "seq_len_max_irr": 36,
            "pred_len_max_irr": 3,
        }
        self.seed = seed

    def public_audit(self):
        return {
            "schema_version": "unit.bundle.v1",
            "seed": self.seed,
            "test_automatic_after_train": False,
        }


def _fake_components(root: Path, config, cell: BackboneCell, calls):
    run_dir = root / "checkpoint"
    assets = SimpleNamespace(checkpoint=run_dir / "pytorch_model.bin")

    def build_vendor_config(*args, seed: int, **kwargs):
        return _Bundle(run_dir, seed)

    def factory_type(bundle, *args):
        return _TrainValOnlyFactory(calls)

    def trainer(**kwargs):
        seed = kwargs["seed"]
        loader_factory = kwargs["loader_factory"]
        loader_factory("train", seed)
        loader_factory("val", seed)
        output = Path(kwargs["output_dir"])
        checkpoint = output / "pytorch_model.bin"
        manifest = output / "train_manifest.json"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        _write_json(
            manifest,
            _checkpoint_manifest(config, cell, seed, checkpoint, list(kwargs["argv"])),
        )
        return SimpleNamespace(
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            checkpoint_sha256=_hash(checkpoint),
        )

    return SimpleNamespace(
        resolve_run_assets=lambda *args, **kwargs: assets,
        build_vendor_config=build_vendor_config,
        ReleaseParityLoaderFactory=factory_type,
        StrictLoaderFactory=factory_type,
        make_apn_model_factory=lambda bundle: object(),
        apn_forward_callback=object(),
        train_apn_train_val=trainer,
    )


def test_default_plan_is_ordered_and_does_not_include_mimic() -> None:
    cells = select_cells()
    assert [(cell.dataset_id, cell.protocol_id) for cell in cells] == [
        ("P12", "release_parity"),
        ("P12", "strict_p12"),
        ("USHCN", "release_parity"),
        ("USHCN", "strict_ushcn"),
        ("HumanActivity", "release_parity"),
    ]
    assert select_seeds([2028, 2024, 2026]) == (2024, 2026, 2028)
    assert len(campaign_plan(seeds=[2024])) == 5
    with pytest.raises(BackboneCampaignError, match="duplicates"):
        select_seeds([2024, 2024])


def test_mimic_release_is_explicitly_blocked_before_runtime_import() -> None:
    config = load_resolved_config()
    with pytest.raises(BackboneBlockedError) as caught:
        prepare_cell(config, BackboneCell("MIMIC_III", "release_parity"))
    assert caught.value.code == "missing_author_mapping"
    assert "UNIQUE_ID_dict.csv" in caught.value.detail


def test_cublas_workspace_is_set_and_conflicts_fail_closed() -> None:
    clean: dict[str, str] = {}
    audit = ensure_cublas_workspace_config(clean)
    assert clean["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_VALUE
    assert audit["set_by_control_plane"] is True
    assert ensure_cublas_workspace_config(clean)["set_by_control_plane"] is False
    with pytest.raises(BackboneCampaignError, match="exactly"):
        ensure_cublas_workspace_config({"CUBLAS_WORKSPACE_CONFIG": ":16:8"})


def test_read_only_import_and_plan_do_not_import_heavy_runtime() -> None:
    script = (
        "import json,sys; "
        f"sys.path.insert(0,{str(PROJECT_ROOT / 'code' / 'src')!r}); "
        "import edgetwincal.backbone_campaign as b; "
        "b.main(['--dataset','P12','--seed','2024']); "
        "print(json.dumps({k:(k in sys.modules) for k in "
        "['torch','pandas','models.APN']}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout.strip().splitlines()[-1])
    # The historical package initializer exports torch-backed adapters. The
    # safe wrapper establishes CUBLAS_WORKSPACE_CONFIG before importing it;
    # planning itself must still avoid vendor and dataset imports.
    assert audit["pandas"] is False
    assert audit["models.APN"] is False


def test_reusable_checkpoint_hash_config_policy_and_cell_are_fail_closed(tmp_path) -> None:
    config = load_resolved_config()
    cell = BackboneCell("P12", "release_parity")
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"verified-checkpoint")
    manifest = tmp_path / "train_manifest.json"
    _write_json(manifest, _checkpoint_manifest(config, cell, 2024, checkpoint))

    validated = validate_reusable_checkpoint(
        config=config,
        cell=cell,
        seed=2024,
        checkpoint_path=checkpoint,
        train_manifest_path=manifest,
    )
    assert validated["checkpoint"]["sha256"] == _hash(checkpoint)
    identity = backbone_campaign.validate_frozen_checkpoint_identity(
        config=config,
        cell=cell,
        seed=2024,
        checkpoint_path=checkpoint,
        train_manifest_path=manifest,
    )
    assert identity["verification_mode"] == "native_train_manifest"

    checkpoint.write_bytes(b"mutated")
    with pytest.raises(BackboneCampaignError, match="checkpoint.sha256"):
        validate_reusable_checkpoint(
            config=config,
            cell=cell,
            seed=2024,
            checkpoint_path=checkpoint,
            train_manifest_path=manifest,
        )


def test_vendor_config_rejects_absolute_path_outside_project() -> None:
    outside = Path("C:/Windows/Temp/outside")
    with pytest.raises(ValueError, match="escapes project root"):
        vendor_config_snapshot({"dataset_root_path": str(outside)})


def test_train_then_reuse_uses_only_train_and_val_and_writes_pointer_manifest(
    tmp_path, monkeypatch
) -> None:
    config = load_resolved_config()
    cell = BackboneCell("P12", "release_parity")
    prepared = _protocol_files(tmp_path / "protocol", config, cell)
    calls: list[tuple[str, str]] = []
    components = _fake_components(tmp_path, config, cell, calls)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_VALUE)
    monkeypatch.setattr(
        backbone_campaign,
        "_backbone_log_path",
        lambda config, cell, seed: tmp_path / f"seed_{seed}.log",
    )

    trained = train_or_reuse_cell(
        config,
        prepared,
        2024,
        device="cpu",
        components=components,
    )
    assert trained.outcome == "trained"
    assert calls == [
        ("build_fit_loader", "train"),
        ("build_fit_loader", "val"),
        ("trainer_loader", "train"),
        ("trainer_loader", "val"),
    ]
    control = json.loads(trained.manifest_path.read_text(encoding="utf-8"))
    assert control["test_constructed"] is False
    assert control["allowed_loader_splits"] == ["train", "val"]
    assert control["cublas_workspace"]["observed"] == CUBLAS_WORKSPACE_VALUE
    assert control["artifacts"]["configs_yaml"]["path"].endswith("configs.yaml")
    assert control["artifacts"]["train_manifest"]["path"].endswith(
        "train_manifest.json"
    )
    assert control["artifacts"]["log"]["status"] == "recorded"

    calls.clear()
    reused = train_or_reuse_cell(
        config,
        prepared,
        2024,
        device="cpu",
        components=components,
    )
    assert reused.outcome == "reused"
    assert calls == [
        ("build_fit_loader", "train"),
        ("build_fit_loader", "val"),
    ]


def test_partial_checkpoint_state_is_never_overwritten(tmp_path, monkeypatch) -> None:
    config = load_resolved_config()
    cell = BackboneCell("P12", "release_parity")
    prepared = _protocol_files(tmp_path / "protocol", config, cell)
    calls: list[tuple[str, str]] = []
    components = _fake_components(tmp_path, config, cell, calls)
    checkpoint = tmp_path / "checkpoint" / "pytorch_model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"orphan")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_VALUE)
    monkeypatch.setattr(
        backbone_campaign,
        "_backbone_log_path",
        lambda config, cell, seed: tmp_path / f"seed_{seed}.log",
    )

    with pytest.raises(BackboneCampaignError, match="Partial checkpoint state"):
        train_or_reuse_cell(config, prepared, 2024, device="cpu", components=components)
    assert checkpoint.read_bytes() == b"orphan"



def test_pinned_legacy_import_never_builds_loader_or_forges_train_manifest(
    tmp_path, monkeypatch
) -> None:
    config = load_resolved_config()
    seed = 2024
    cell = BackboneCell("P12", "release_parity")
    prepared = _protocol_files(tmp_path / "protocol", config, cell)
    calls: list[tuple[str, str]] = []
    components = _fake_components(tmp_path, config, cell, calls)

    legacy_root = tmp_path / "legacy" / str(seed)
    source_dir = legacy_root / "checkpoints" / "P12" / "APN" / "iter0"
    source_dir.mkdir(parents=True)
    source_checkpoint = source_dir / "pytorch_model.bin"
    source_configs = source_dir / "configs.yaml"
    source_checkpoint.write_bytes(b"legacy-checkpoint")
    source_configs.write_bytes(b"legacy-config\n")

    target_dir = tmp_path / "checkpoint"
    target_dir.mkdir(parents=True)
    target_checkpoint = target_dir / "pytorch_model.bin"
    target_configs = target_dir / "configs.yaml"
    target_checkpoint.write_bytes(source_checkpoint.read_bytes())
    target_configs.write_bytes(source_configs.read_bytes())

    log_path = tmp_path / "legacy_logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"official train log\n")
    monkeypatch.setattr(
        backbone_campaign, "_legacy_stage_a_root", lambda observed_seed: legacy_root
    )
    monkeypatch.setattr(
        backbone_campaign,
        "_legacy_stage_a_log_path",
        lambda observed_seed: log_path,
    )

    argv = [str(PROJECT_ROOT / ".conda" / "python.exe"), "main.py"]
    for flag, value in backbone_campaign._expected_legacy_p12_flags(
        config, seed
    ).items():
        argv.extend([flag, value])
    manifest = {
        "schema_version": 1,
        "project_root": str(PROJECT_ROOT),
        "variant": "apn",
        "seed": seed,
        "shift": "none",
        "action": "train",
        "process": {
            "argv": argv,
            "cwd": str(PROJECT_ROOT / "vendor" / "APN"),
            "log_path": str(log_path),
            "started_at": "2026-07-20T00:00:00+00:00",
            "ended_at": "2026-07-20T00:01:00+00:00",
            "wall_seconds": 60.0,
            "peak_gpu_memory_mib": 1.0,
            "peak_cuda_allocated_mib": 1.0,
            "peak_cuda_reserved_mib": 2.0,
            "gpu_memory_source": "torch.cuda",
            "return_code": 0,
        },
        "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": _hash(source_checkpoint),
        "parameter_count": 6701,
        "provenance": {
            "python": "3.11.13 | unit",
            "project_git_commit": backbone_campaign.LEGACY_PROJECT_COMMIT,
            "apn_commit": backbone_campaign.LEGACY_APN_COMMIT,
            "apn_patch_sha256": backbone_campaign.LEGACY_APN_PATCH_SHA256,
            "torch_cuda_gpu": {
                "torch": "2.6.0+cu124",
                "cuda_runtime": "12.4",
                "cuda_available": True,
                "gpu": "NVIDIA GeForce RTX 4090",
            },
        },
        "artifacts": [
            {
                "path": source_configs.relative_to(PROJECT_ROOT)
                .as_posix()
                .replace("/", "\\"),
                "bytes": source_configs.stat().st_size,
                "sha256": _hash(source_configs),
            }
        ],
    }
    legacy_manifest = legacy_root / "train_manifest.json"
    _write_json(legacy_manifest, manifest)
    monkeypatch.setattr(
        backbone_campaign,
        "LEGACY_P12_RELEASE_IDENTITIES",
        {
            seed: {
                "checkpoint_sha256": _hash(source_checkpoint),
                "checkpoint_bytes": source_checkpoint.stat().st_size,
                "configs_sha256": _hash(source_configs),
                "configs_bytes": source_configs.stat().st_size,
                "train_manifest_sha256": _hash(legacy_manifest),
                "train_manifest_bytes": legacy_manifest.stat().st_size,
                "log_sha256": _hash(log_path),
                "log_bytes": log_path.stat().st_size,
            }
        },
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_VALUE)

    imported = train_or_reuse_cell(
        config,
        prepared,
        seed,
        device="cpu",
        reuse_only=True,
        components=components,
    )
    assert imported.outcome == "verified_legacy_import"
    assert calls == []
    assert target_checkpoint.read_bytes() == b"legacy-checkpoint"
    assert not (target_dir / "train_manifest.json").exists()
    proof = json.loads(
        (target_dir / "verified_legacy_import.json").read_text(encoding="utf-8")
    )
    assert proof["test_constructed"] is False
    assert proof["loaders_constructed"] is False
    assert proof["new_train_manifest"]["status"] == "absent_by_design"
    control = json.loads(imported.manifest_path.read_text(encoding="utf-8"))
    assert control["outcome"] == "verified_legacy_import"
    assert control["allowed_loader_splits"] == []
    identity = backbone_campaign.validate_frozen_checkpoint_identity(
        config=config,
        cell=cell,
        seed=seed,
        checkpoint_path=target_checkpoint,
        configs_yaml_path=target_configs,
    )
    assert identity["verification_mode"] == "verified_legacy_import"

    log_path.write_bytes(b"tampered\n")
    with pytest.raises(BackboneCampaignError, match="training log identity mismatch"):
        train_or_reuse_cell(
            config,
            prepared,
            seed,
            device="cpu",
            reuse_only=True,
            components=components,
        )
    assert calls == []
