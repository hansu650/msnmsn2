from __future__ import annotations

import hashlib
import importlib
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import yaml

from .paths import PROJECT_ROOT, ensure_directory, require_within_root


VENDOR_ROOT = require_within_root(PROJECT_ROOT / "vendor" / "APN", must_exist=True)
CODE_SRC = require_within_root(PROJECT_ROOT / "code" / "src", must_exist=True)


@dataclass(frozen=True)
class RunAssets:
    seed: int
    config: Path
    checkpoint: Path


@dataclass
class Metrics:
    mse: float
    mae: float
    observed_targets: int


def prepare_imports() -> None:
    for path in (str(CODE_SRC), str(VENDOR_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["EVIPATCH_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["EVIPATCH_TSDM_ROOT"] = str(
        require_within_root(PROJECT_ROOT / "data" / "tsdm", must_exist=True)
    )


def load_config(config_path: str | Path) -> Any:
    prepare_imports()
    from utils.ExpConfigs import ExpConfigs

    path = require_within_root(config_path, must_exist=True)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["evipatch_mode"] = "apn"
    raw["observation_shift"] = "none"
    raw["shift_rate"] = 0.0
    raw["is_training"] = 1
    raw["num_workers"] = 0
    return ExpConfigs(**raw)


def load_frozen_apn(config: Any, checkpoint_path: str | Path, device: torch.device) -> Any:
    prepare_imports()
    from models.APN import Model

    checkpoint = require_within_root(checkpoint_path, must_exist=True)
    model = Model(config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def extract_apn_features(
    model: Any,
    x: Tensor,
    x_mark: Tensor,
    x_mask: Tensor,
    y_mark: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    submodel = model.model
    batch, _, variables = x.shape
    submodel.batch_size = batch
    time_features = x_mark[:, :, [0]]
    x_stacked = x.permute(0, 2, 1).reshape(batch * variables, x.shape[1], 1)
    mask_stacked = x_mask.permute(0, 2, 1).reshape(batch * variables, x.shape[1], 1)
    time_stacked = (
        time_features.repeat(1, 1, variables)
        .permute(0, 2, 1)
        .reshape(batch * variables, x.shape[1], 1)
    )
    encoded_time = submodel.LearnableTE(time_stacked)
    features = submodel.IMTS_Model_Logic(
        torch.cat([x_stacked, encoded_time], dim=-1),
        mask_stacked,
        time_stacked,
    )
    prediction_times = y_mark[:, :, [0]]
    horizon = prediction_times.shape[1]
    expanded_times = prediction_times.view(batch, 1, horizon, 1).repeat(
        1, variables, 1, 1
    )
    prediction_time_encoding = submodel.LearnableTE(expanded_times)
    decoder_input = torch.cat(
        [features.unsqueeze(2).expand(-1, -1, horizon, -1), prediction_time_encoding],
        dim=-1,
    )
    prediction = submodel.decoder(decoder_input).squeeze(-1).permute(0, 2, 1)
    return features, prediction, prediction_time_encoding


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_assets(seed: int) -> RunAssets:
    root = require_within_root(
        PROJECT_ROOT / "results" / "stage_a" / "apn" / str(seed), must_exist=True
    )
    checkpoints = list(root.rglob("pytorch_model.bin"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one APN checkpoint for seed {seed}, found {len(checkpoints)}")
    checkpoint = require_within_root(checkpoints[0], must_exist=True)
    return RunAssets(
        seed=seed,
        checkpoint=checkpoint,
        config=require_within_root(checkpoint.with_name("configs.yaml"), must_exist=True),
    )


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def collect_split(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Tensor]:
    collected: dict[str, list[Tensor]] = {
        "features": [],
        "time_encoding": [],
        "base_prediction": [],
        "target": [],
        "mask": [],
        "sample_id": [],
    }
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        features, prediction, time_encoding = extract_apn_features(
            model, batch["x"], batch["x_mark"], batch["x_mask"], batch["y_mark"]
        )
        horizon = prediction.shape[1]
        collected["features"].append(features.cpu())
        collected["time_encoding"].append(time_encoding.cpu())
        collected["base_prediction"].append(prediction.cpu())
        collected["target"].append(batch["y"][:, :horizon].cpu())
        collected["mask"].append(batch["y_mask"][:, :horizon].cpu())
        collected["sample_id"].append(batch["sample_ID"].cpu().to(torch.int64))
    return {key: torch.cat(value) for key, value in collected.items()}


def create_cache(assets: RunAssets, cache_path: Path, device: torch.device) -> dict[str, Any]:
    config = load_config(assets.config)
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        prepare_imports()
        from data.data_provider.data_factory import data_provider

        importlib.import_module(f"data.data_provider.datasets.{config.dataset_name}")
    finally:
        sys.argv = original_argv
    set_determinism(assets.seed)
    train_set, train_loader = data_provider(config, "train")
    val_set, val_loader = data_provider(config, "val")
    test_set, test_loader = data_provider(config, "test")
    model = load_frozen_apn(config, assets.checkpoint, device)
    started = time.perf_counter()
    cache = {
        "metadata": {
            "schema_version": 2,
            "seed": assets.seed,
            "checkpoint": str(assets.checkpoint),
            "checkpoint_sha256": file_sha256(assets.checkpoint),
            "config": str(assets.config),
            "split_samples": {
                "train": len(train_set),
                "val": len(val_set),
                "test": len(test_set),
            },
        },
        "train": collect_split(model, train_loader, device),
        "val": collect_split(model, val_loader, device),
        "test": collect_split(model, test_loader, device),
    }
    cache["metadata"]["cache_wall_seconds"] = time.perf_counter() - started
    torch.save(cache, cache_path)
    return cache


def load_or_create_cache(
    assets: RunAssets,
    output_root: Path,
    device: torch.device,
    rebuild: bool,
) -> tuple[dict[str, Any], Path]:
    cache_path = require_within_root(
        ensure_directory(output_root / "cache") / f"apn_seed{assets.seed}.pt"
    )
    if cache_path.exists() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("metadata", {}).get("schema_version") == 2:
            return cache, cache_path
    return create_cache(assets, cache_path, device), cache_path


def masked_metrics(prediction: Tensor, target: Tensor, mask: Tensor) -> Metrics:
    selected = mask > 0
    count = int(selected.sum())
    if count == 0:
        raise RuntimeError("No observed forecast targets")
    error = prediction[selected] - target[selected]
    return Metrics(
        mse=float(error.double().square().mean()),
        mae=float(error.double().abs().mean()),
        observed_targets=count,
    )


def relative_improvement(baseline: Metrics, candidate: Metrics) -> dict[str, float]:
    return {
        "mse": (baseline.mse - candidate.mse) / baseline.mse,
        "mae": (baseline.mae - candidate.mae) / baseline.mae,
    }
