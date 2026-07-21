"""End-to-end control plane for the frozen EdgeTwinCal-Safe campaign.

Only the test command (and the explicit all pipeline at its test step) can
construct the sealed-test loader. Every earlier command is train/validation-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import sys
import time

import torch
from torch import Tensor, nn

from . import safe_data
from .apn_bridge import (
    APNConfigBundle,
    apn_forward_callback,
    collect_apn_split,
    load_frozen_apn_checkpoint,
)
from .apn_training import train_apn_train_val
from .paths import PROJECT_ROOT, require_within_root
from .safe import SafePolicy
from .safe_aggregate import (
    SAFE_VARIANTS,
    aggregate_safe_campaign,
    load_evaluation_manifests,
    write_safe_aggregation,
)
from .safe_campaign import (
    SafeTestLedger,
    build_evaluation_manifest,
    tensor_sha256,
    write_evaluation_manifest,
)
from .safe_config import (
    DEFAULT_SAFE_CONFIG,
    DatasetSpec,
    SafeExperimentConfig,
    load_safe_config,
)
from .safe_runner import (
    SafeFitPolicy,
    SafeSeedStates,
    decide_validation_safety_gate,
    fit_safe_seed,
)


SAFE_PROTOCOL_ID = "safe_v1"
STATE_SCHEMA = "edgetwincal.safe-seed-state.v1"
CACHE_SCHEMA = "edgetwincal.safe-fit-cache.v1"
GATE_SCHEMA = "edgetwincal.safe-validation-gate.v1"
_PINNED_PATCH_SHA256 = "00d8d59221d1580ee2b718365325bd69945dc2c103b0c23d7f93f9365e301746"
_PINNED_APN_MODEL_SHA256 = "d183568c45993e8a291fc3c3225fc9b10336724f8b8e9a748a1f0c83b435be6b"


NORMALIZER_SCHEMA = "edgetwincal.safe-normalizer.v1"
TEST_DATA_COMMANDS = frozenset({"test", "all"})
_PRETEST_COMMAND_NAMES = frozenset(
    {"download", "prepare-pretest", "smoke", "train", "fit", "gate", "aggregate"}
)
_VENDOR_TEMPLATE = (
    PROJECT_ROOT
    / "vendor"
    / "APN"
    / "configs"
    / "APN"
    / "APN_P12_apn_seed2024"
    / "P12.yaml"
)


class SafeExperimentError(RuntimeError):
    """The Safe campaign cannot make a valid irreversible transition."""


class SafeAPNLoader:
    """Re-iterable adapter from project-owned Safe batches to APN batches."""

    def __init__(self, loader: Iterable[Mapping[str, Tensor]]) -> None:
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            yield safe_data.as_apn_batch(batch)

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


def commands_that_may_open_test() -> frozenset[str]:
    return TEST_DATA_COMMANDS


def _sha256_file(path: str | Path) -> str:
    source = require_within_root(path, must_exist=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_new_bytes(path: str | Path, payload: bytes) -> Path:
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _write_new_bytes(path, payload)


def _write_new_torch(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        torch.save(dict(value), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _read_json(path: str | Path) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SafeExperimentError(f"JSON artifact is not an object: {source}")
    return value


def _results_path(
    config: SafeExperimentConfig, *parts: str | Path
) -> Path:
    return safe_data.require_safe_path(
        PROJECT_ROOT, config.relative_path("results"), *parts
    )


def _normalizer_path(config: SafeExperimentConfig, dataset: str) -> Path:
    return _results_path(config, "pretest", dataset, "normalizer.json")


def _backbone_dir(
    config: SafeExperimentConfig, dataset: str, seed: int
) -> Path:
    return _results_path(config, "backbones", dataset, f"seed_{seed}")


def _cache_path(config: SafeExperimentConfig, dataset: str, seed: int) -> Path:
    return _results_path(
        config, "private", "fit_cache", dataset, f"seed_{seed}.pt"
    )


def _state_path(config: SafeExperimentConfig, dataset: str, seed: int) -> Path:
    return _results_path(
        config, "private", "states", dataset, f"seed_{seed}.pt"
    )


def _fit_manifest_path(
    config: SafeExperimentConfig, dataset: str, seed: int
) -> Path:
    return _results_path(config, "fit", dataset, f"seed_{seed}.json")


def _gate_path(config: SafeExperimentConfig, dataset: str) -> Path:
    return _results_path(config, "gate", f"{dataset}.json")


def _evaluation_path(
    config: SafeExperimentConfig, dataset: str, seed: int, variant: str
) -> Path:
    return _results_path(
        config, "evaluations", dataset, f"seed_{seed}", f"{variant}.json"
    )


def _private_evaluation_path(
    config: SafeExperimentConfig, dataset: str, seed: int, variant: str
) -> Path:
    return _results_path(
        config,
        "private",
        "test_arrays",
        dataset,
        f"seed_{seed}",
        f"{variant}.pt",
    )


def _private_test_windows_path(
    config: SafeExperimentConfig, dataset: str
) -> Path:
    return _results_path(
        config, "private", "test_windows", f"{dataset}.pt"
    )


def _private_test_windows_manifest_path(
    config: SafeExperimentConfig, dataset: str
) -> Path:
    return _results_path(
        config, "private", "test_windows", f"{dataset}.json"
    )


def _campaign_ledger_path(config: SafeExperimentConfig) -> Path:
    return _results_path(config, "protocol", "campaign_ledger.json")


def _aggregate_path(config: SafeExperimentConfig) -> Path:
    return _results_path(config, "aggregate", "report.json")


def _raw_manifest_path(
    config: SafeExperimentConfig, dataset: str
) -> Path:
    return safe_data.require_safe_path(
        PROJECT_ROOT,
        config.relative_path("raw"),
        dataset,
        "raw_source_manifest.json",
    )


def _normalizer_payload(
    normalizer: safe_data.NormalizerState, config_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": NORMALIZER_SCHEMA,
        "config_sha256": config_sha256,
        "dataset_id": normalizer.dataset_id,
        "channel_order": list(normalizer.channel_order),
        "mean": list(normalizer.mean),
        "scale": list(normalizer.scale),
        "observed_count": list(normalizer.observed_count),
        "fit_split": normalizer.fit_split,
        "source_sha256": normalizer.source_sha256,
        "normalizer_sha256": normalizer.sha256,
    }


def _load_normalizer(
    config: SafeExperimentConfig, spec: DatasetSpec
) -> safe_data.NormalizerState:
    raw = _read_json(_normalizer_path(config, spec.dataset_id))
    if (
        raw.get("schema_version") != NORMALIZER_SCHEMA
        or raw.get("config_sha256") != config.sha256
        or raw.get("dataset_id") != spec.dataset_id
        or tuple(raw.get("channel_order", ())) != spec.channels
        or raw.get("fit_split") != "train"
    ):
        raise SafeExperimentError("Train-only normalizer identity changed")
    state = safe_data.NormalizerState(
        str(raw["dataset_id"]),
        tuple(raw["channel_order"]),
        tuple(float(value) for value in raw["mean"]),
        tuple(float(value) for value in raw["scale"]),
        tuple(int(value) for value in raw["observed_count"]),
        str(raw["fit_split"]),
        str(raw["source_sha256"]),
        str(raw["normalizer_sha256"]),
    )
    rebuilt = safe_data.fit_train_normalizer(
        safe_data.load_pretest_rows(PROJECT_ROOT, spec, "train")
    )
    if state != rebuilt:
        raise SafeExperimentError(
            "Stored normalizer differs from train-only recomputation"
        )
    return state


def _load_raw_manifest(
    config: SafeExperimentConfig, spec: DatasetSpec
) -> safe_data.RawSourceManifest:
    path = _raw_manifest_path(config, spec.dataset_id)
    manifest = safe_data.RawSourceManifest(**_read_json(path))
    source = safe_data.require_safe_path(
        PROJECT_ROOT, manifest.source_relative_path
    )
    if (
        manifest.dataset_id != spec.dataset_id
        or not source.is_file()
        or _sha256_file(source) != manifest.sha256
        or (
            spec.source.expected_sha256 is not None
            and manifest.sha256 != spec.source.expected_sha256
        )
    ):
        raise SafeExperimentError("Raw source manifest is stale or mismatched")
    return manifest


def _ensure_downloaded(
    config: SafeExperimentConfig, spec: DatasetSpec
) -> safe_data.RawSourceManifest:
    path = _raw_manifest_path(config, spec.dataset_id)
    if path.exists():
        return _load_raw_manifest(config, spec)
    return safe_data.download_official_dataset(PROJECT_ROOT, spec)


def _ensure_prepared(
    config: SafeExperimentConfig, spec: DatasetSpec
) -> tuple[safe_data.PretestManifest, safe_data.NormalizerState]:
    pretest_path = safe_data.require_safe_path(
        PROJECT_ROOT,
        config.relative_path("pretest"),
        spec.dataset_id,
        "pretest_manifest.json",
    )
    if pretest_path.exists():
        manifest = safe_data.load_pretest_manifest(PROJECT_ROOT, spec)
    else:
        manifest = safe_data.prepare_pretest_shards(
            PROJECT_ROOT, spec, _load_raw_manifest(config, spec)
        )
    normalizer_path = _normalizer_path(config, spec.dataset_id)
    if normalizer_path.exists():
        normalizer = _load_normalizer(config, spec)
    else:
        normalizer = safe_data.fit_train_normalizer(
            safe_data.load_pretest_rows(PROJECT_ROOT, spec, "train")
        )
        _write_new_json(
            normalizer_path, _normalizer_payload(normalizer, config.sha256)
        )
    if normalizer.source_sha256 != manifest.source_sha256:
        raise SafeExperimentError("Normalizer and routed source hashes differ")
    return manifest, normalizer


def _verify_vendor_provenance(
    config: SafeExperimentConfig,
) -> dict[str, str]:
    vendor = require_within_root(
        PROJECT_ROOT / "vendor" / "APN", must_exist=True
    )
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={vendor.as_posix()}",
            "-C",
            str(vendor),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    patch = require_within_root(
        PROJECT_ROOT / "patches" / "apn_evipatch.patch",
        must_exist=True,
    )
    model = require_within_root(
        vendor / "models" / "APN.py", must_exist=True
    )
    patch_sha = _sha256_file(patch)
    model_sha = _sha256_file(model)
    if (
        commit != config.apn.commit
        or patch_sha != _PINNED_PATCH_SHA256
        or model_sha != _PINNED_APN_MODEL_SHA256
    ):
        raise SafeExperimentError(
            "Pinned vendor commit, APN patch, or patched model hash changed"
        )
    return {
        "vendor_commit": commit,
        "apn_patch_sha256": patch_sha,
        "patched_model_sha256": model_sha,
    }


def build_safe_vendor_config(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    seed: int,
    *,
    config_type: Any | None = None,
) -> APNConfigBundle:
    """Build APN-only settings from the pinned P12 YAML."""

    if seed not in config.seeds:
        raise SafeExperimentError(f"Unregistered Safe seed: {seed}")
    template = require_within_root(_VENDOR_TEMPLATE, must_exist=True)
    provenance = _verify_vendor_provenance(config)
    import yaml

    raw = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SafeExperimentError("Pinned APN P12 template is not a mapping")
    output_dir = _backbone_dir(config, spec.dataset_id, seed)
    raw.update(
        {
            "task_name": "short_term_forecast",
            "is_training": 1,
            "model_id": f"APN_{spec.dataset_id}_safe_v1_seed{seed}",
            "model_name": "APN",
            "checkpoints": str(output_dir),
            "ablation_name": "",
            "dataset_name": spec.dataset_id,
            "dataset_root_path": str(
                safe_data.require_safe_path(
                    PROJECT_ROOT,
                    config.relative_path("pretest"),
                    spec.dataset_id,
                )
            ),
            "features": "M",
            "seq_len": spec.history_steps,
            "label_len": 0,
            "pred_len": spec.forecast_steps,
            "enc_in": len(spec.channels),
            "dec_in": len(spec.channels),
            "c_out": len(spec.channels),
            "train_epochs": config.apn.epochs,
            "patience": config.apn.patience,
            "val_interval": 1,
            "loss": "MSE",
            "lr_scheduler": "DelayedStepDecayLR",
            "batch_size": config.apn.batch_size,
            "learning_rate": config.apn.learning_rate,
            "d_model": config.apn.d_model,
            "dropout": config.apn.dropout,
            "apn_npatch": config.apn.npatch,
            "apn_te_dim": config.apn.te_dim,
            "evipatch_mode": "apn",
            "evipatch_eval_name": "",
            "observation_shift": "none",
            "shift_rate": 0.0,
            "shift_seed": seed,
            "seed_base": seed,
            "evipatch_random_seed": 1729,
            "num_workers": 0,
            "itr": 1,
            "save_arrays": 0,
            "load_checkpoints_test": 0,
            "test_all": 0,
            "test_train_time": 0,
            "test_inference_time": 0,
            "test_gpu_memory": 0,
            "test_dataset_statistics": 0,
            "skip_test_after_train": 1,
            "train_val_loader_shuffle": None,
            "train_val_loader_drop_last": None,
        }
    )
    if config_type is None:
        vendor = require_within_root(
            PROJECT_ROOT / "vendor" / "APN", must_exist=True
        )
        source = require_within_root(
            PROJECT_ROOT / "code" / "src", must_exist=True
        )
        for path in (source, vendor):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        config_type = importlib.import_module("utils.ExpConfigs").ExpConfigs
    vendor_config = config_type(**raw)
    audit = MappingProxyType(
        {
            "schema_version": "edgetwincal.safe-apn-config.v1",
            "config_sha256": config.sha256,
            "dataset_id": spec.dataset_id,
            "protocol_id": SAFE_PROTOCOL_ID,
            "seed": seed,
            "template": template.relative_to(PROJECT_ROOT).as_posix(),
            "history_steps": spec.history_steps,
            "forecast_steps": spec.forecast_steps,
            "channels": len(spec.channels),
            **provenance,
            "apn_mode": "apn",
            "test_loader_constructed": False,
            "home_overridden": False,
        }
    )
    return APNConfigBundle(
        vendor_config, spec.dataset_id, SAFE_PROTOCOL_ID, seed, audit
    )


def _model_factory(bundle: APNConfigBundle):
    def build(seed: int) -> nn.Module:
        if seed != bundle.seed:
            raise SafeExperimentError("APN model seed differs from bundle")
        vendor = require_within_root(
            PROJECT_ROOT / "vendor" / "APN", must_exist=True
        )
        source = require_within_root(
            PROJECT_ROOT / "code" / "src", must_exist=True
        )
        for path in (source, vendor):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        model = importlib.import_module("models.APN").Model(bundle.config)
        if not isinstance(model, nn.Module):
            raise SafeExperimentError("Pinned APN returned a non-module")
        return model

    return build


def _device(value: str | None) -> torch.device:
    selected = value or ("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device(selected)
    if result.type == "cuda" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in {":4096:8", ":16:8"}:
        raise SafeExperimentError(
            "CUDA requires an early deterministic CUBLAS_WORKSPACE_CONFIG"
        )
    if result.type == "cuda" and not torch.cuda.is_available():
        raise SafeExperimentError("CUDA was requested but is unavailable")
    if result.type not in {"cpu", "cuda"}:
        raise SafeExperimentError("Safe APN device must be cpu or cuda")
    return result


def _selected_datasets(
    config: SafeExperimentConfig, selection: str
) -> tuple[DatasetSpec, ...]:
    if selection == "all":
        return config.datasets
    return (config.dataset(selection),)


def _selected_seeds(
    config: SafeExperimentConfig, selection: str
) -> tuple[int, ...]:
    if selection == "all":
        return config.seeds
    seed = int(selection)
    if seed not in config.seeds:
        raise SafeExperimentError(f"Unregistered seed {seed}")
    return (seed,)


def _pretest_loaders(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    normalizer: safe_data.NormalizerState,
    seed: int,
) -> safe_data.PretestLoaders:
    return safe_data.build_pretest_loaders(
        spec,
        normalizer,
        seed,
        root=PROJECT_ROOT,
        batch_size=config.apn.batch_size,
    )


def _checkpoint_artifacts(
    config: SafeExperimentConfig, spec: DatasetSpec, seed: int
) -> tuple[Path, Path, dict[str, Any]]:
    directory = _backbone_dir(config, spec.dataset_id, seed)
    checkpoint = directory / "pytorch_model.bin"
    manifest_path = directory / "train_manifest.json"
    manifest = _read_json(manifest_path)
    recorded = manifest.get("checkpoint", {})
    resolved = manifest.get("resolved_config", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("seed") != seed
        or resolved.get("safe_config_sha256") != config.sha256
        or resolved.get("dataset_id") != spec.dataset_id
        or not checkpoint.is_file()
        or recorded.get("sha256") != _sha256_file(checkpoint)
    ):
        raise SafeExperimentError("APN checkpoint resume validation failed")
    return checkpoint, manifest_path, manifest


def _train_cell(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    seed: int,
    *,
    device: torch.device,
    argv: Sequence[str],
) -> dict[str, Any]:
    directory = _backbone_dir(config, spec.dataset_id, seed)
    checkpoint = directory / "pytorch_model.bin"
    manifest = directory / "train_manifest.json"
    if checkpoint.exists() or manifest.exists():
        _, _, loaded = _checkpoint_artifacts(config, spec, seed)
        return loaded
    _, normalizer = _ensure_prepared(config, spec)
    loaders = _pretest_loaders(config, spec, normalizer, seed)
    bundle = build_safe_vendor_config(config, spec, seed)
    resolved = {
        "schema_version": "edgetwincal.safe-apn-train.v1",
        "safe_config_sha256": config.sha256,
        "dataset_id": spec.dataset_id,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": seed,
        "apn_config": bundle.public_audit(),
        "normalizer_sha256": normalizer.sha256,
        "fit_splits": ["train", "val"],
        "test_constructed": False,
    }
    result = train_apn_train_val(
        output_dir=directory,
        seed=seed,
        learning_rate=config.apn.learning_rate,
        resolved_config=resolved,
        model_factory=_model_factory(bundle),
        train_loader=SafeAPNLoader(loaders.train),
        validation_loader=SafeAPNLoader(loaders.val),
        forward_callback=apn_forward_callback,
        device=device,
        argv=argv,
    )
    return _read_json(result.manifest_path)


def _collect_pretest_fit_cache(
    model: nn.Module,
    loaders: safe_data.PretestLoaders,
    *,
    spec: DatasetSpec,
    device: torch.device,
) -> dict[str, dict[str, Tensor]]:
    return {
        name: collect_apn_split(
            model,
            SafeAPNLoader(loader),
            dataset_id=spec.dataset_id,
            protocol_id=SAFE_PROTOCOL_ID,
            split="val",
            device=device,
        )
        for name, loader in (
            ("adapter", loaders.adapter),
            ("val_select", loaders.val_select),
            ("val_safety", loaders.val_safety),
        )
    }


def _load_fit_cache(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    seed: int,
    checkpoint_sha256: str,
) -> tuple[dict[str, dict[str, Tensor]], str]:
    path = _cache_path(config, spec.dataset_id, seed)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != CACHE_SCHEMA
        or raw.get("config_sha256") != config.sha256
        or raw.get("dataset_id") != spec.dataset_id
        or raw.get("seed") != seed
        or raw.get("checkpoint_sha256") != checkpoint_sha256
        or raw.get("normalizer_sha256")
        != _load_normalizer(config, spec).sha256
    ):
        raise SafeExperimentError("Private fit cache identity changed")
    splits = raw.get("splits")
    if not isinstance(splits, dict) or set(splits) != {
        "adapter",
        "val_select",
        "val_safety",
    }:
        raise SafeExperimentError("Private fit cache split registry changed")
    return splits, _sha256_file(path)


def _load_seed_state(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    seed: int,
    checkpoint_sha256: str | None = None,
) -> tuple[SafeSeedStates, dict[str, Any], str]:
    path = _state_path(config, spec.dataset_id, seed)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cache_path = _cache_path(config, spec.dataset_id, seed)
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != STATE_SCHEMA
        or raw.get("config_sha256") != config.sha256
        or raw.get("dataset_id") != spec.dataset_id
        or raw.get("seed") != seed
        or (
            checkpoint_sha256 is not None
            and raw.get("checkpoint_sha256") != checkpoint_sha256
        )
        or raw.get("normalizer_sha256")
        != _load_normalizer(config, spec).sha256
        or (
            cache_path.exists()
            and raw.get("cache_sha256") != _sha256_file(cache_path)
        )
        or not isinstance(raw.get("state"), SafeSeedStates)
    ):
        raise SafeExperimentError("Safe seed state identity changed")
    state_hash = _sha256_file(path)
    manifest_path = _fit_manifest_path(config, spec.dataset_id, seed)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if (
            manifest.get("state_sha256") != state_hash
            or manifest.get("cache_sha256") != raw.get("cache_sha256")
            or manifest.get("normalizer_sha256")
            != raw.get("normalizer_sha256")
            or manifest.get("checkpoint_sha256")
            != raw.get("checkpoint_sha256")
        ):
            raise SafeExperimentError(
                "Safe state and fit manifest hashes differ"
            )
    return raw["state"], raw, state_hash


def _safe_policy(config: SafeExperimentConfig) -> SafePolicy:
    raw = config.to_dict()
    envelope = raw["safe_envelope"]
    gate = raw["validation_gate"]
    return SafePolicy(
        cap_grid=tuple(float(value) for value in envelope["kappa"]),
        shrink_grid=tuple(float(value) for value in envelope["shrinkage"]),
        selection_min_groups=int(gate["minimum_groups"]),
        selection_min_observations=int(gate["minimum_target_cells"]),
        dataset_min_groups=int(gate["minimum_groups"]),
        dataset_min_observations=int(gate["minimum_target_cells"]),
        required_checkpoints=5,
        required_improved_checkpoints=int(gate["minimum_seed_gains"]),
        selection_point_loss_margin=0.0,
        selection_joint_loss_margin=float(gate["joint_macro_harm_ucb"]),
        validation_relative_loss_margin=float(gate["apn_harm_ucb"]),
        validation_joint_loss_margin=float(gate["joint_macro_harm_ucb"]),
        final_relative_loss_margin=float(gate["apn_harm_ucb"]),
        final_joint_noninferiority_margin=float(
            raw["statistics"]["joint_noninferiority_margin"]
        ),
        max_positive_gain_concentration=float(
            gate["maximum_gain_concentration"]
        ),
        validation_bootstrap_resamples=int(gate["bootstrap_draws"]),
        dataset_bootstrap_resamples=int(gate["bootstrap_draws"]),
        random_seed=int(gate["bootstrap_seed"]),
    )


def _fit_policy(config: SafeExperimentConfig) -> SafeFitPolicy:
    raw = config.to_dict()["robust_fit"]
    return SafeFitPolicy(
        latent_alphas=tuple(float(value) for value in raw["alpha_latent"]),
        cross_alphas=tuple(float(value) for value in raw["alpha_cross"]),
        huber_delta=float(raw["huber_delta"]),
        max_iterations=int(raw["max_iterations"]),
        tolerance=float(raw["tolerance"]),
        scale_floor=float(raw["scale_floor"]),
        feature_clip=float(raw["feature_clip"]),
        minimum_rows=None,
        minimum_groups=int(raw["minimum_groups"]),
        gate=_safe_policy(config),
    )

def _fit_cell(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    seed: int,
    *,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint, _, _ = _checkpoint_artifacts(config, spec, seed)
    checkpoint_hash = _sha256_file(checkpoint)
    _, normalizer = _ensure_prepared(config, spec)
    cache_path = _cache_path(config, spec.dataset_id, seed)
    if cache_path.exists():
        splits, cache_hash = _load_fit_cache(
            config, spec, seed, checkpoint_hash
        )
    else:
        bundle = build_safe_vendor_config(config, spec, seed)
        model = load_frozen_apn_checkpoint(
            bundle,
            checkpoint,
            device=device,
            model_factory=_model_factory(bundle),
        )
        loaders = _pretest_loaders(config, spec, normalizer, seed)
        splits = _collect_pretest_fit_cache(
            model, loaders, spec=spec, device=device
        )
        _write_new_torch(
            cache_path,
            {
                "schema_version": CACHE_SCHEMA,
                "config_sha256": config.sha256,
                "dataset_id": spec.dataset_id,
                "seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "normalizer_sha256": normalizer.sha256,
                "splits": splits,
            },
        )
        cache_hash = _sha256_file(cache_path)
    state_path = _state_path(config, spec.dataset_id, seed)
    if state_path.exists():
        _, state_wrapper, state_hash = _load_seed_state(
            config, spec, seed, checkpoint_hash
        )
    else:
        fitted = fit_safe_seed(
            splits["adapter"],
            splits["val_select"],
            policy=_fit_policy(config),
        )
        selected = (
            asdict(fitted.selection.selected)
            if fitted.selection.selected is not None
            else None
        )
        state_wrapper = {
            "schema_version": STATE_SCHEMA,
            "config_sha256": config.sha256,
            "dataset_id": spec.dataset_id,
            "seed": seed,
            "checkpoint_sha256": checkpoint_hash,
            "normalizer_sha256": normalizer.sha256,
            "cache_sha256": cache_hash,
            "minimum_rows_policy": "max(100,4*p)",
            "minimum_rows_argument": None,
            "selected_candidate": selected,
            "state": fitted,
        }
        _write_new_torch(state_path, state_wrapper)
        state_hash = _sha256_file(state_path)
    public = {
        "schema_version": "edgetwincal.safe-fit-manifest.v1",
        "status": "complete",
        "config_sha256": config.sha256,
        "dataset_id": spec.dataset_id,
        "seed": seed,
        "checkpoint_sha256": checkpoint_hash,
        "normalizer_sha256": normalizer.sha256,
        "cache_sha256": cache_hash,
        "state_sha256": state_hash,
        "selection": state_wrapper["state"].selection.as_dict(),
        "robust_alpha_latent": state_wrapper["state"].robust.alpha_latent,
        "robust_alpha_cross": state_wrapper["state"].robust.alpha_cross,
        "minimum_rows_policy": "max(100,4*p)",
        "test_constructed": False,
    }
    manifest_path = _fit_manifest_path(config, spec.dataset_id, seed)
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing != public:
            raise SafeExperimentError(
                "Fit manifest differs from valid artifacts"
            )
        return existing
    _write_new_json(manifest_path, public)
    return public


def _load_gate(
    config: SafeExperimentConfig, spec: DatasetSpec
) -> dict[str, Any]:
    value = _read_json(_gate_path(config, spec.dataset_id))
    if (
        value.get("schema_version") != GATE_SCHEMA
        or value.get("config_sha256") != config.sha256
        or value.get("dataset_id") != spec.dataset_id
        or not isinstance(value.get("enabled"), bool)
        or int(value.get("checkpoints", -1)) != len(config.seeds)
        or not isinstance(value.get("gate_sha256"), str)
        or len(value["gate_sha256"]) != 64
    ):
        raise SafeExperimentError("Validation gate artifact is invalid")
    return value


def _gate_dataset(
    config: SafeExperimentConfig, spec: DatasetSpec
) -> dict[str, Any]:
    path = _gate_path(config, spec.dataset_id)
    if path.exists():
        return _load_gate(config, spec)
    states: dict[int, SafeSeedStates] = {}
    safety: dict[int, Mapping[str, Tensor]] = {}
    state_hashes: dict[str, str] = {}
    for seed in config.seeds:
        checkpoint, _, _ = _checkpoint_artifacts(config, spec, seed)
        checkpoint_hash = _sha256_file(checkpoint)
        state, _, state_hash = _load_seed_state(
            config, spec, seed, checkpoint_hash
        )
        cache, _ = _load_fit_cache(
            config, spec, seed, checkpoint_hash
        )
        states[seed] = state
        safety[seed] = cache["val_safety"]
        state_hashes[str(seed)] = state_hash
    decision, manifest = decide_validation_safety_gate(
        states,
        safety,
        dataset_id=spec.dataset_id,
        group_salt=spec.id_salt,
        policy=_safe_policy(config),
    )
    payload = {
        "schema_version": GATE_SCHEMA,
        "status": "complete",
        "config_sha256": config.sha256,
        "dataset_id": spec.dataset_id,
        "state_sha256": state_hashes,
        **manifest,
        "decision": decision.as_dict(),
        "test_constructed": False,
    }
    _write_new_json(path, payload)
    return payload


def _split_sha256(split: Mapping[str, Tensor]) -> str:
    return _canonical_sha256(
        {
            "sample_id": tensor_sha256(split["sample_id"]),
            "group_id": tensor_sha256(split["group_id"]),
            "mask": tensor_sha256(split["mask"] > 0),
        }
    )


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "home_overridden": False,
        "codex_home_overridden": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _component_hashes(
    config: SafeExperimentConfig,
) -> dict[str, str]:
    components = {"safe_config": config.sha256}
    for spec in config.datasets:
        components[f"{spec.dataset_id}:pretest"] = _sha256_file(
            safe_data.require_safe_path(
                PROJECT_ROOT,
                config.relative_path("pretest"),
                spec.dataset_id,
                "pretest_manifest.json",
            )
        )
        components[f"{spec.dataset_id}:normalizer"] = _sha256_file(
            _normalizer_path(config, spec.dataset_id)
        )
        components[f"{spec.dataset_id}:gate_file"] = _sha256_file(
            _gate_path(config, spec.dataset_id)
        )
        for seed in config.seeds:
            checkpoint, _, _ = _checkpoint_artifacts(config, spec, seed)
            components[
                f"{spec.dataset_id}:{seed}:checkpoint"
            ] = _sha256_file(checkpoint)
            components[f"{spec.dataset_id}:{seed}:state"] = _sha256_file(
                _state_path(config, spec.dataset_id, seed)
            )
    return components


def _recovery_token_path(
    config: SafeExperimentConfig, dataset: str
) -> Path:
    return _results_path(
        config, "private", "test_recovery", f"{dataset}.json"
    )


def _data_ledger_path(
    config: SafeExperimentConfig, dataset: str
) -> Path:
    return _results_path(
        config, "protocol", dataset, "test_ledger.json"
    )


def _test_artifact_paths(
    config: SafeExperimentConfig,
) -> list[Path]:
    paths: list[Path] = []
    for spec in config.datasets:
        paths.extend(
            [
                _data_ledger_path(config, spec.dataset_id),
                _data_ledger_path(config, spec.dataset_id).with_suffix(
                    ".claim"
                ),
                _private_test_windows_path(config, spec.dataset_id),
                _private_test_windows_manifest_path(
                    config, spec.dataset_id
                ),
                _recovery_token_path(config, spec.dataset_id),
            ]
        )
        for seed in config.seeds:
            for variant in SAFE_VARIANTS:
                paths.extend(
                    [
                        _evaluation_path(
                            config, spec.dataset_id, seed, variant
                        ),
                        _private_evaluation_path(
                            config, spec.dataset_id, seed, variant
                        ),
                    ]
                )
    return paths


def _load_or_create_campaign_ledger(
    config: SafeExperimentConfig,
    gates: Mapping[str, Mapping[str, Any]],
) -> SafeTestLedger:
    path = _campaign_ledger_path(config)
    components = _component_hashes(config)
    if not path.exists():
        occupied = [
            str(candidate)
            for candidate in _test_artifact_paths(config)
            if candidate.exists()
        ]
        if occupied:
            raise SafeExperimentError(
                "Test artifacts exist without a campaign ledger: "
                + ", ".join(occupied[:4])
            )
        ledger = SafeTestLedger.create(
            path,
            config_sha256=config.sha256,
            components=components,
            gate_decisions=gates,
        )
        ledger.freeze()
        return ledger
    ledger = SafeTestLedger.load(path)
    if ledger.status == "sealed":
        raise SafeExperimentError(
            "Completed once-only Safe test cannot be run again"
        )
    if (
        ledger.status not in {"frozen", "test_active"}
        or ledger.data.get("config_sha256") != config.sha256
        or ledger.data.get("components") != dict(sorted(components.items()))
        or ledger.data.get("gate_decisions")
        != json.loads(json.dumps(gates, sort_keys=True))
        or not isinstance(ledger.data.get("protocol_sha256"), str)
    ):
        raise SafeExperimentError(
            "Crash-resume campaign ledger no longer matches frozen inputs"
        )
    return ledger


def _write_recovery_token(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    protocol_sha256: str,
    token: str,
) -> Path:
    return _write_new_json(
        _recovery_token_path(config, spec.dataset_id),
        {
            "schema_version": "edgetwincal.safe-test-recovery.v1",
            "config_sha256": config.sha256,
            "dataset_id": spec.dataset_id,
            "protocol_sha256": protocol_sha256,
            "token": token,
        },
    )


def _load_recovery_token(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    protocol_sha256: str,
    ledger: SafeTestLedger,
) -> str:
    raw = _read_json(
        _recovery_token_path(config, spec.dataset_id)
    )
    if (
        raw.get("schema_version")
        != "edgetwincal.safe-test-recovery.v1"
        or raw.get("config_sha256") != config.sha256
        or raw.get("dataset_id") != spec.dataset_id
        or raw.get("protocol_sha256") != protocol_sha256
        or not isinstance(raw.get("token"), str)
    ):
        raise SafeExperimentError("Crash-recovery token journal is invalid")
    token = raw["token"]
    ledger.validate_token(spec.dataset_id, token)
    return token


def _load_test_windows_cache(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    normalizer: safe_data.NormalizerState,
    *,
    protocol_sha256: str,
    gate_sha256: str,
) -> list[Mapping[str, Tensor]]:
    path = _private_test_windows_path(config, spec.dataset_id)
    manifest = _read_json(
        _private_test_windows_manifest_path(config, spec.dataset_id)
    )
    if (
        manifest.get("schema_version")
        != "edgetwincal.safe-test-windows-manifest.v1"
        or manifest.get("config_sha256") != config.sha256
        or manifest.get("dataset_id") != spec.dataset_id
        or manifest.get("normalizer_sha256") != normalizer.sha256
        or manifest.get("protocol_sha256") != protocol_sha256
        or manifest.get("gate_sha256") != gate_sha256
        or manifest.get("cache_sha256") != _sha256_file(path)
    ):
        raise SafeExperimentError("Test-window cache hash manifest changed")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version")
        != "edgetwincal.safe-test-windows.v1"
        or raw.get("config_sha256") != config.sha256
        or raw.get("dataset_id") != spec.dataset_id
        or raw.get("normalizer_sha256") != normalizer.sha256
        or raw.get("protocol_sha256") != protocol_sha256
        or raw.get("gate_sha256") != gate_sha256
        or not isinstance(raw.get("batches"), list)
        or not raw["batches"]
    ):
        raise SafeExperimentError("Immutable test-window cache is invalid")
    data_ledger = _read_json(
        _data_ledger_path(config, spec.dataset_id)
    )
    provenance = data_ledger.get("provenance", {})
    if (
        data_ledger.get("status") != "consumed"
        or data_ledger.get("dataset_id") != spec.dataset_id
        or provenance.get("config_sha256") != config.sha256
        or provenance.get("normalizer_sha256") != normalizer.sha256
        or provenance.get("campaign_protocol_sha256")
        != protocol_sha256
        or provenance.get("gate_sha256") != gate_sha256
    ):
        raise SafeExperimentError(
            "Consumed data ledger and test-window cache differ"
        )
    for batch in raw["batches"]:
        if not isinstance(batch, Mapping) or set(batch) != {
            "x",
            "x_mark",
            "y",
            "y_mark",
            "x_mask",
            "y_mask",
            "sample_id",
            "group_id",
        }:
            raise SafeExperimentError("Cached test batch schema changed")
        if any(not isinstance(value, Tensor) for value in batch.values()):
            raise SafeExperimentError("Cached test batch contains non-tensors")
    return raw["batches"]




def _selected_hyperparameters(
    state: SafeSeedStates, variant: str, gate: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "variant": variant,
        "dataset_gate_enabled": bool(gate["enabled"]),
        "gate_sha256": gate["gate_sha256"],
        "robust_alpha_latent": state.robust.alpha_latent,
        "robust_alpha_cross": state.robust.alpha_cross,
        "selection": state.selection.as_dict(),
        "raw_candidate": (
            asdict(state.raw_spec) if state.raw_spec is not None else None
        ),
    }


def _private_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "private": True,
    }


def _validate_private_payload(
    path: Path,
    prediction: Tensor,
    split: Mapping[str, Tensor],
) -> None:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "prediction": prediction.detach().cpu(),
        "target": split["target"].detach().cpu(),
        "mask": (split["mask"] > 0).detach().cpu(),
        "sample_id": split["sample_id"].detach().cpu(),
        "group_id": split["group_id"].detach().cpu(),
    }
    if not isinstance(raw, dict) or set(raw) != set(expected):
        raise SafeExperimentError("Private evaluation payload schema changed")
    for name, value in expected.items():
        if not isinstance(raw[name], Tensor) or not torch.equal(
            raw[name], value
        ):
            raise SafeExperimentError(
                f"Private evaluation payload differs for {name}"
            )


def _write_or_resume_evaluation(
    *,
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    seed: int,
    variant: str,
    state: SafeSeedStates,
    gate: Mapping[str, Any],
    split: Mapping[str, Tensor],
    prediction: Tensor,
    checkpoint_sha256: str,
    normalizer: safe_data.NormalizerState,
    protocol_sha256: str,
    timing: Mapping[str, Any],
    argv: Sequence[str],
) -> bool:
    manifest_path = _evaluation_path(
        config, spec.dataset_id, seed, variant
    )
    private_path = _private_evaluation_path(
        config, spec.dataset_id, seed, variant
    )
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        required = manifest.get("required_files")
        if (
            manifest.get("schema_version")
            != "edgetwincal.safe-evaluation.v1"
            or manifest.get("status") != "complete"
            or manifest.get("dataset") != spec.dataset_id
            or manifest.get("seed") != seed
            or manifest.get("variant") != variant
            or manifest.get("checkpoint_sha256") != checkpoint_sha256
            or manifest.get("config_sha256") != config.sha256
            or manifest.get("normalizer_sha256") != normalizer.sha256
            or manifest.get("protocol_sha256") != protocol_sha256
            or manifest.get("gate_sha256") != gate["gate_sha256"]
            or not isinstance(required, list)
            or len(required) != 1
            or required[0] != _private_file_record(private_path)
        ):
            raise SafeExperimentError(
                "Completed evaluation failed crash-resume validation"
            )
        _validate_private_payload(private_path, prediction, split)
        return False
    if private_path.exists():
        _validate_private_payload(private_path, prediction, split)
        private_argument: Path | None = None
    else:
        private_argument = private_path
    manifest = build_evaluation_manifest(
        dataset=spec.dataset_id,
        seed=seed,
        variant=variant,
        prediction=prediction,
        target=split["target"],
        mask=split["mask"],
        sample_ids=split["sample_id"],
        group_ids=split["group_id"],
        group_salt=spec.id_salt,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config.sha256,
        split_sha256=_split_sha256(split),
        normalizer_sha256=normalizer.sha256,
        selected_hyperparameters=_selected_hyperparameters(
            state, variant, gate
        ),
        timing=timing,
        argv=argv,
        environment=_environment(),
        protocol_sha256=protocol_sha256,
        gate_sha256=gate["gate_sha256"],
        private_payload_path=private_argument,
    )
    if private_argument is None:
        manifest["required_files"] = [
            _private_file_record(private_path)
        ]
    write_evaluation_manifest(manifest, manifest_path)
    return True
def _test_campaign(
    config: SafeExperimentConfig,
    *,
    device: torch.device,
    argv: Sequence[str],
) -> dict[str, Any]:
    """Open raw test once; crash-resume only from immutable window caches."""

    gates = {
        spec.dataset_id: _load_gate(config, spec)
        for spec in config.datasets
    }
    ledger = _load_or_create_campaign_ledger(config, gates)
    protocol_sha256 = str(ledger.data["protocol_sha256"])
    manifests_written = 0
    resumed = bool(ledger.data.get("test_openings"))
    for spec in config.datasets:
        opening = ledger.data["test_openings"].get(spec.dataset_id)
        if isinstance(opening, Mapping) and opening.get(
            "closed_at"
        ) is not None:
            for seed in config.seeds:
                for variant in SAFE_VARIANTS:
                    manifest = _read_json(
                        _evaluation_path(
                            config, spec.dataset_id, seed, variant
                        )
                    )
                    private = _private_evaluation_path(
                        config, spec.dataset_id, seed, variant
                    )
                    if manifest.get("required_files") != [
                        _private_file_record(private)
                    ]:
                        raise SafeExperimentError(
                            "Closed target has an invalid private artifact"
                        )
            continue
        normalizer = _load_normalizer(config, spec)
        if isinstance(opening, Mapping):
            campaign_token = _load_recovery_token(
                config, spec, protocol_sha256, ledger
            )
            materialized_batches = _load_test_windows_cache(
                config,
                spec,
                normalizer,
                protocol_sha256=protocol_sha256,
                gate_sha256=gates[spec.dataset_id]["gate_sha256"],
            )
        else:
            if (
                _data_ledger_path(config, spec.dataset_id).exists()
                or _private_test_windows_manifest_path(
                    config, spec.dataset_id
                ).exists()
                or _private_test_windows_path(
                    config, spec.dataset_id
                ).exists()
                or _recovery_token_path(
                    config, spec.dataset_id
                ).exists()
            ):
                raise SafeExperimentError(
                    "Unopened target already has test artifacts"
                )
            campaign_token = ledger.open_dataset_once(spec.dataset_id)
            _write_recovery_token(
                config, spec, protocol_sha256, campaign_token
            )
            data_ledger = safe_data.freeze_test_ledger(
                PROJECT_ROOT,
                spec,
                {
                    "campaign_protocol_sha256": protocol_sha256,
                    "gate_sha256": gates[spec.dataset_id][
                        "gate_sha256"
                    ],
                    "config_sha256": config.sha256,
                    "normalizer_sha256": normalizer.sha256,
                },
            )
            test_loader = safe_data.open_sealed_test_loader(
                PROJECT_ROOT,
                spec,
                data_ledger.token,
                normalizer,
                config.seeds[0],
                batch_size=config.apn.batch_size,
            )
            materialized_batches = list(test_loader)
            if not materialized_batches:
                raise SafeExperimentError(
                    f"{spec.dataset_id} sealed test loader is empty"
                )
            cache_path = _private_test_windows_path(
                config, spec.dataset_id
            )
            _write_new_torch(
                cache_path,
                {
                    "schema_version": (
                        "edgetwincal.safe-test-windows.v1"
                    ),
                    "config_sha256": config.sha256,
                    "dataset_id": spec.dataset_id,
                    "normalizer_sha256": normalizer.sha256,
                    "protocol_sha256": protocol_sha256,
                    "gate_sha256": gates[spec.dataset_id][
                        "gate_sha256"
                    ],
                    "batches": materialized_batches,
                },
            )
            _write_new_json(
                _private_test_windows_manifest_path(
                    config, spec.dataset_id
                ),
                {
                    "schema_version": (
                        "edgetwincal.safe-test-windows-manifest.v1"
                    ),
                    "config_sha256": config.sha256,
                    "dataset_id": spec.dataset_id,
                    "normalizer_sha256": normalizer.sha256,
                    "protocol_sha256": protocol_sha256,
                    "gate_sha256": gates[spec.dataset_id][
                        "gate_sha256"
                    ],
                    "cache_sha256": _sha256_file(cache_path),
                },
            )
            materialized_batches = _load_test_windows_cache(
                config,
                spec,
                normalizer,
                protocol_sha256=protocol_sha256,
                gate_sha256=gates[spec.dataset_id]["gate_sha256"],
            )
        wrapped_loader = SafeAPNLoader(materialized_batches)
        for seed in config.seeds:
            checkpoint, _, _ = _checkpoint_artifacts(
                config, spec, seed
            )
            checkpoint_hash = _sha256_file(checkpoint)
            bundle = build_safe_vendor_config(config, spec, seed)
            model = load_frozen_apn_checkpoint(
                bundle,
                checkpoint,
                device=device,
                model_factory=_model_factory(bundle),
            )
            extraction_started = time.perf_counter()
            split = collect_apn_split(
                model,
                wrapped_loader,
                dataset_id=spec.dataset_id,
                protocol_id=SAFE_PROTOCOL_ID,
                split="test",
                device=device,
                test_ledger_token=campaign_token,
            )
            extraction_seconds = (
                time.perf_counter() - extraction_started
            )
            state, _, _ = _load_seed_state(
                config, spec, seed, checkpoint_hash
            )
            apply_started = time.perf_counter()
            predictions = state.prediction_map(
                split,
                dataset_gate_enabled=bool(
                    gates[spec.dataset_id]["enabled"]
                ),
            )
            apply_seconds = time.perf_counter() - apply_started
            timing = {
                "kind": (
                    "statistical_evaluation_not_device_benchmark"
                ),
                "apn_extraction_seconds": extraction_seconds,
                "all_adapter_variants_seconds": apply_seconds,
                "device_timing_authorized": False,
                "crash_resume": resumed,
            }
            for variant in SAFE_VARIANTS:
                manifests_written += int(
                    _write_or_resume_evaluation(
                        config=config,
                        spec=spec,
                        seed=seed,
                        variant=variant,
                        state=state,
                        gate=gates[spec.dataset_id],
                        split=split,
                        prediction=predictions[variant],
                        checkpoint_sha256=checkpoint_hash,
                        normalizer=normalizer,
                        protocol_sha256=protocol_sha256,
                        timing=timing,
                        argv=argv,
                    )
                )
        ledger.close_dataset(spec.dataset_id, campaign_token)
        journal = _recovery_token_path(config, spec.dataset_id)
        if journal.exists():
            journal.unlink()
    ledger.seal()
    return {
        "status": "complete",
        "protocol_sha256": protocol_sha256,
        "manifests_written": manifests_written,
        "ledger_status": ledger.status,
        "crash_resumed": resumed,
    }


def _aggregate_campaign(
    config: SafeExperimentConfig,
) -> dict[str, Any]:
    ledger = SafeTestLedger.load(_campaign_ledger_path(config))
    if ledger.status != "sealed":
        raise SafeExperimentError("Safe campaign ledger is not sealed")
    paths = [
        _evaluation_path(config, spec.dataset_id, seed, variant)
        for spec in config.datasets
        for seed in config.seeds
        for variant in SAFE_VARIANTS
    ]
    gates = {
        spec.dataset_id: _load_gate(config, spec)
        for spec in config.datasets
    }
    report = aggregate_safe_campaign(
        load_evaluation_manifests(paths),
        gates,
        datasets=tuple(spec.dataset_id for spec in config.datasets),
        seeds=config.seeds,
        variants=SAFE_VARIANTS,
        resamples=int(
            config.to_dict()["statistics"]["bootstrap_draws"]
        ),
        random_seed=int(
            config.to_dict()["statistics"]["bootstrap_seed"]
        ),
    )
    destination = _aggregate_path(config)
    if destination.exists():
        existing = _read_json(destination)
        if existing != report:
            raise SafeExperimentError("Existing aggregation differs")
        return existing
    write_safe_aggregation(report, destination)
    return report


def require_device_timing_authorized(
    report: Mapping[str, Any] | str | Path,
) -> Mapping[str, Any]:
    """Reject CPU/Jetson timing unless the statistical gate is PASS."""

    value = (
        _read_json(report)
        if isinstance(report, (str, Path))
        else dict(report)
    )
    if (
        value.get("verdict") != "PASS"
        or value.get("device_timing_authorized") is not True
    ):
        raise SafeExperimentError(
            "Real CPU/Jetson timing is unreachable unless the Safe report is PASS"
        )
    return value


def _smoke_cell(
    config: SafeExperimentConfig,
    spec: DatasetSpec,
    seed: int,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _, normalizer = _ensure_prepared(config, spec)
    loaders = _pretest_loaders(config, spec, normalizer, seed)
    bundle = build_safe_vendor_config(config, spec, seed)
    model = _model_factory(bundle)(seed).to(device).eval()
    summaries: dict[str, Any] = {}
    for name, loader in (("train", loaders.train), ("val", loaders.val)):
        first_pass = next(iter(SafeAPNLoader(loader)))
        second_pass = next(iter(SafeAPNLoader(loader)))
        if set(first_pass) != set(second_pass):
            raise SafeExperimentError("Safe APN loader is not re-iterable")
        moved = {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in first_pass.items()
        }
        with torch.no_grad():
            output = apn_forward_callback(
                model, moved, name
            )  # type: ignore[arg-type]
        prediction = output["pred"]
        if not torch.isfinite(prediction).all():
            raise SafeExperimentError("APN smoke prediction is non-finite")
        summaries[name] = {
            "shape": list(prediction.shape),
            "observed_targets": int(
                (moved["y_mask"] > 0).sum().item()
            ),
            "reiterable": True,
        }
    return {
        "dataset_id": spec.dataset_id,
        "seed": seed,
        "config_sha256": config.sha256,
        "splits": summaries,
        "test_constructed": False,
    }

def _cmd_download(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    return {
        spec.dataset_id: _ensure_downloaded(config, spec).to_dict()
        for spec in _selected_datasets(config, args.dataset)
    }


def _cmd_prepare(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    output = {}
    for spec in _selected_datasets(config, args.dataset):
        manifest, normalizer = _ensure_prepared(config, spec)
        output[spec.dataset_id] = {
            "pretest_manifest": manifest.to_dict(),
            "normalizer_sha256": normalizer.sha256,
            "test_constructed": False,
        }
    return output


def _cmd_smoke(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    device = _device(args.device)
    return {
        f"{spec.dataset_id}|{seed}": _smoke_cell(
            config, spec, seed, device=device
        )
        for spec in _selected_datasets(config, args.dataset)
        for seed in _selected_seeds(config, args.seed)
    }


def _cmd_train(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    device = _device(args.device)
    return {
        f"{spec.dataset_id}|{seed}": _train_cell(
            config,
            spec,
            seed,
            device=device,
            argv=sys.argv if args.argv is None else args.argv,
        )
        for spec in _selected_datasets(config, args.dataset)
        for seed in _selected_seeds(config, args.seed)
    }


def _cmd_fit(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    device = _device(args.device)
    return {
        f"{spec.dataset_id}|{seed}": _fit_cell(
            config, spec, seed, device=device
        )
        for spec in _selected_datasets(config, args.dataset)
        for seed in _selected_seeds(config, args.seed)
    }


def _cmd_gate(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    return {
        spec.dataset_id: _gate_dataset(config, spec)
        for spec in _selected_datasets(config, args.dataset)
    }


def _cmd_test(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    return _test_campaign(
        config,
        device=_device(args.device),
        argv=sys.argv if args.argv is None else args.argv,
    )


def _cmd_aggregate(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    del args
    return _aggregate_campaign(config)


def _cmd_all(
    args: argparse.Namespace, config: SafeExperimentConfig
) -> Any:
    _cmd_download(args, config)
    _cmd_prepare(args, config)
    _cmd_smoke(args, config)
    _cmd_train(args, config)
    _cmd_fit(args, config)
    gates = _cmd_gate(args, config)
    tested = _cmd_test(args, config)
    report = _cmd_aggregate(args, config)
    return {"gates": gates, "test": tested, "report": report}


def _add_selection(
    parser: argparse.ArgumentParser,
    *,
    dataset: bool = True,
    seed: bool = False,
    device: bool = False,
) -> None:
    if dataset:
        parser.add_argument(
            "--dataset",
            default="all",
            choices=("all", "beijing_air", "intel_lab"),
        )
    if seed:
        parser.add_argument(
            "--seed",
            default="all",
            choices=(
                "all",
                "2024",
                "2025",
                "2026",
                "2027",
                "2028",
            ),
        )
    if device:
        parser.add_argument("--device", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_edgetwincal_safe.py",
        description=(
            "Frozen EdgeTwinCal-Safe train/validation/test campaign"
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_SAFE_CONFIG))
    subparsers = parser.add_subparsers(
        dest="command", required=True
    )
    download = subparsers.add_parser("download")
    _add_selection(download)
    prepare = subparsers.add_parser("prepare-pretest")
    _add_selection(prepare)
    smoke = subparsers.add_parser("smoke")
    _add_selection(smoke, seed=True, device=True)
    train = subparsers.add_parser("train")
    _add_selection(train, seed=True, device=True)
    fit = subparsers.add_parser("fit")
    _add_selection(fit, seed=True, device=True)
    gate = subparsers.add_parser("gate")
    _add_selection(gate)
    test = subparsers.add_parser("test")
    _add_selection(test, dataset=False, device=True)
    subparsers.add_parser("aggregate")
    all_parser = subparsers.add_parser("all")
    _add_selection(all_parser, seed=True, device=True)
    return parser


_HANDLERS = {
    "download": _cmd_download,
    "prepare-pretest": _cmd_prepare,
    "smoke": _cmd_smoke,
    "train": _cmd_train,
    "fit": _cmd_fit,
    "gate": _cmd_gate,
    "test": _cmd_test,
    "aggregate": _cmd_aggregate,
    "all": _cmd_all,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else None
    try:
        config = load_safe_config(args.config)
        payload = _HANDLERS[args.command](args, config)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "command": getattr(args, "command", None),
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            payload, sort_keys=True, indent=2, default=str
        )
    )
    return 0


__all__ = [
    "SafeAPNLoader",
    "SafeExperimentError",
    "build_parser",
    "build_safe_vendor_config",
    "commands_that_may_open_test",
    "main",
    "require_device_timing_authorized",
]
