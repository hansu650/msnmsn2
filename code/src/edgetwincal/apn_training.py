"""Deterministic, train/validation-only APN backbone training.

This module is deliberately owned by EdgeTwinCal rather than ``vendor/APN``.
It has no test-loader argument and imports the released APN implementation only
inside the optional lazy factory helpers.  The optimization contract mirrors
the released APN code: Adam, masked MSE, 200 epochs, patience 10, validation at
every epoch, and DelayedStepDecayLR.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import torch
from torch import Tensor, nn

from .paths import PROJECT_ROOT, require_within_root


LOCKED_EPOCHS = 200
LOCKED_PATIENCE = 10
LOCKED_VALIDATION_INTERVAL = 1
LOCKED_OPTIMIZER = "Adam"
LOCKED_LOSS = "MSE"
LOCKED_SCHEDULER = "DelayedStepDecayLR"
_TRAIN_STAGE = 1


class APNTrainingError(RuntimeError):
    """Raised when a train/validation-only APN run violates its contract."""


Loader: TypeAlias = Iterable[Any] | Callable[[], Iterable[Any]]
Split: TypeAlias = Literal["train", "val"]


class ModelFactory(Protocol):
    def __call__(self, seed: int) -> nn.Module: ...


class LoaderFactory(Protocol):
    def __call__(self, split: Split, seed: int) -> Loader: ...


class ForwardCallback(Protocol):
    def __call__(self, model: nn.Module, batch: Any, stage: Split) -> Any: ...


@dataclass(frozen=True)
class APNTrainingResult:
    """Paths and hashes for one completed APN backbone training run."""

    model: nn.Module
    checkpoint_path: Path
    manifest_path: Path
    checkpoint_sha256: str
    best_state_sha256: str
    best_epoch: int
    best_validation_mse: float
    epochs_ran: int
    stopped_early: bool

    def audit_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("model")
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


@dataclass
class _RNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: list[Tensor] | None
    deterministic_enabled: bool
    deterministic_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _capture_rng_state() -> _RNGState:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    return _RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=cuda_state,
        deterministic_enabled=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=warn_only,
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
    )


def _seed_deterministically(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _restore_rng_state(state: _RNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)
    torch.use_deterministic_algorithms(
        state.deterministic_enabled, warn_only=state.deterministic_warn_only
    )
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.backends.cudnn.benchmark = state.cudnn_benchmark


def delayed_step_decay_factor(scheduler_epoch: int) -> float:
    """Return the exact multiplier used by APN's DelayedStepDecayLR."""

    if scheduler_epoch < 0:
        raise ValueError("scheduler_epoch must be nonnegative")
    return 1.0 if scheduler_epoch < 2 else 0.8 ** (scheduler_epoch - 2)


def make_delayed_step_decay_lr(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Construct APN's released ``DelayedStepDecayLR`` schedule."""

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=delayed_step_decay_factor
    )


def lazy_vendor_model_factory(
    config: Any,
    *,
    module_name: str = "models.APN",
    class_name: str = "Model",
) -> ModelFactory:
    """Create an APN model factory without importing vendor code yet.

    The caller is responsible for making the pinned ``vendor/APN`` tree
    importable.  Import happens only when the returned factory is invoked,
    after the train and validation loaders have been created.
    """

    if not module_name or not class_name:
        raise ValueError("module_name and class_name must be nonempty")

    def build(seed: int) -> nn.Module:
        del seed
        module = importlib.import_module(module_name)
        model_type = getattr(module, class_name)
        model = model_type(config)
        if not isinstance(model, nn.Module):
            raise TypeError("Vendor APN model factory did not return torch.nn.Module")
        return model

    return build


def lazy_vendor_loader_factory(
    config: Any,
    *,
    module_name: str = "data.data_provider.data_factory",
    function_name: str = "data_provider",
) -> LoaderFactory:
    """Create a released-code train/val loader factory with delayed import.

    The returned callable accepts only the type-level ``Split`` contract used
    by :func:`train_apn_train_val`; a runtime guard also rejects every split
    other than ``train`` and ``val``.
    """

    if not module_name or not function_name:
        raise ValueError("module_name and function_name must be nonempty")

    def build(split: Split, seed: int) -> Loader:
        del seed
        if split not in ("train", "val"):
            raise APNTrainingError(f"Forbidden APN training split: {split!r}")
        module = importlib.import_module(module_name)
        provider = getattr(module, function_name)
        supplied = provider(config, split)
        if isinstance(supplied, tuple) and len(supplied) == 2:
            return supplied[1]
        return supplied

    return build


def _materialize_loader(loader: Loader) -> Iterable[Any]:
    return loader() if callable(loader) else loader


def _resolve_loaders(
    *,
    seed: int,
    loader_factory: LoaderFactory | None,
    train_loader: Loader | None,
    validation_loader: Loader | None,
) -> tuple[Loader, Loader]:
    if loader_factory is not None:
        if train_loader is not None or validation_loader is not None:
            raise ValueError(
                "Use loader_factory or explicit train/validation loaders, not both"
            )
        # These are deliberately the only two split strings in this module.
        return loader_factory("train", seed), loader_factory("val", seed)
    if train_loader is None or validation_loader is None:
        raise ValueError(
            "Both train_loader and validation_loader are required without loader_factory"
        )
    return train_loader, validation_loader


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _batch_target_and_mask(batch: Any) -> tuple[Tensor | None, Tensor | None]:
    if isinstance(batch, Mapping):
        target = batch.get("target", batch.get("targets", batch.get("y")))
        mask = batch.get(
            "target_mask", batch.get("y_mask", batch.get("mask"))
        )
        return (
            target if isinstance(target, Tensor) else None,
            mask if isinstance(mask, Tensor) else None,
        )
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        target = batch[1]
        mask = batch[2] if len(batch) >= 3 else None
        return (
            target if isinstance(target, Tensor) else None,
            mask if isinstance(mask, Tensor) else None,
        )
    return None, None


def _default_forward(model: nn.Module, batch: Any, stage: Split) -> Any:
    if isinstance(batch, Mapping):
        if all(key in batch for key in ("x", "x_mark", "x_mask")):
            # Released APN accepts y/y_mask plus these audit-only stage kwargs.
            return model(
                exp_stage="train" if stage == "train" else "val",
                train_stage=_TRAIN_STAGE,
                **batch,
            )
        inputs = batch.get("inputs", batch.get("input", batch.get("x")))
        if inputs is None:
            raise TypeError(
                "Default mapping batches require inputs/input/x; provide forward_callback"
            )
        return model(inputs)
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return model(batch[0])
    raise TypeError(
        "Default batches require (inputs, target[, mask]) or a mapping; "
        "provide forward_callback for another APN interface"
    )


def _output_triplet(output: Any, batch: Any) -> tuple[Tensor, Tensor, Tensor | None]:
    fallback_target, fallback_mask = _batch_target_and_mask(batch)
    prediction: Tensor | None = None
    target: Tensor | None = fallback_target
    mask: Tensor | None = fallback_mask

    if isinstance(output, Tensor):
        prediction = output
    elif isinstance(output, Mapping):
        for key in ("pred", "prediction", "predictions", "forecast", "output"):
            if isinstance(output.get(key), Tensor):
                prediction = output[key]
                break
        for key in ("true", "target", "targets", "y"):
            if isinstance(output.get(key), Tensor):
                target = output[key]
                break
        for key in ("mask", "target_mask", "y_mask"):
            if isinstance(output.get(key), Tensor):
                mask = output[key]
                break
    elif isinstance(output, (tuple, list)) and len(output) in (2, 3):
        prediction = output[0] if isinstance(output[0], Tensor) else None
        target = output[1] if isinstance(output[1], Tensor) else None
        if len(output) == 3:
            mask = output[2] if isinstance(output[2], Tensor) else None

    if prediction is None or target is None:
        raise TypeError(
            "Forward output must expose prediction and target tensors so the "
            "trainer can enforce MSE"
        )
    if mask is not None and not isinstance(mask, Tensor):
        raise TypeError("Target mask must be a tensor")
    return prediction, target, mask


def _masked_sse_and_weight(
    prediction: Tensor, target: Tensor, mask: Tensor | None
) -> tuple[Tensor, float]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    if mask is None:
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Unmasked predictions and targets must be finite")
        residual = prediction - target
        return residual.square().sum(), float(target.numel())
    if mask.shape != target.shape:
        raise ValueError(f"Mask/target shape mismatch: {mask.shape} != {target.shape}")
    if not torch.isfinite(mask).all() or torch.any(mask < 0):
        raise ValueError("Target mask must be finite and nonnegative")
    observed = mask > 0
    if torch.any(observed):
        if not torch.isfinite(prediction[observed]).all() or not torch.isfinite(
            target[observed]
        ).all():
            raise ValueError("Observed predictions and targets must be finite")
    residual = torch.where(observed, prediction - target, torch.zeros_like(target))
    residual = residual * mask.to(dtype=residual.dtype)
    weight = float(mask.sum().detach().item())
    return residual.square().sum(), weight


def _run_forward(
    model: nn.Module,
    batch: Any,
    stage: Split,
    callback: ForwardCallback | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    output = (
        _default_forward(model, batch, stage)
        if callback is None
        else callback(model, batch, stage)
    )
    return _output_triplet(output, batch)


def _validation_micro_mse(
    model: nn.Module,
    loader: Loader,
    *,
    device: torch.device,
    callback: ForwardCallback | None,
) -> tuple[float, int]:
    model.eval()
    total_sse = 0.0
    total_weight = 0.0
    batches = 0
    with torch.no_grad():
        for raw_batch in _materialize_loader(loader):
            batch = _move_to_device(raw_batch, device)
            prediction, target, mask = _run_forward(model, batch, "val", callback)
            sse, weight = _masked_sse_and_weight(prediction, target, mask)
            if weight <= 0:
                continue
            total_sse += float(sse.detach().item())
            total_weight += weight
            batches += 1
    if batches == 0 or total_weight <= 0:
        raise APNTrainingError("Validation loader contains no observed targets")
    value = total_sse / total_weight
    if not math.isfinite(value):
        raise APNTrainingError("Validation micro MSE is not finite")
    return float(value), batches


def _clone_cpu_state(model: nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        state[name] = (
            value.detach().cpu().contiguous().clone()
            if isinstance(value, Tensor)
            else copy.deepcopy(value)
        )
    return state


def _tensor_payload(value: Tensor) -> bytes:
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, Tensor):
            continue
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = _tensor_payload(value)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(state: Mapping[str, Any], destination: Path) -> None:
    destination = require_within_root(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = require_within_root(
        destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
    )
    try:
        with temporary.open("xb") as handle:
            # File-object ZIP serialization is independent of the temporary name
            # and stable across separately cloned but tensor-identical states.
            torch.save(state, handle, _use_new_zipfile_serialization=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    raise TypeError(f"Configuration value is not JSON serializable: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination = require_within_root(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    temporary = require_within_root(
        destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_artifact_name(name: str, *, label: str) -> str:
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"{label} must be a plain file name")
    return name


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _relative_project_path(path: Path) -> str:
    return require_within_root(path, must_exist=True).relative_to(PROJECT_ROOT).as_posix()


def train_apn_train_val(
    *,
    output_dir: str | Path,
    seed: int,
    learning_rate: float,
    resolved_config: Any,
    model_factory: ModelFactory,
    loader_factory: LoaderFactory | None = None,
    train_loader: Loader | None = None,
    validation_loader: Loader | None = None,
    forward_callback: ForwardCallback | None = None,
    device: str | torch.device | None = None,
    argv: Sequence[str] | None = None,
    checkpoint_name: str = "pytorch_model.bin",
    manifest_name: str = "train_manifest.json",
) -> APNTrainingResult:
    """Train one APN backbone without constructing or accepting a test loader.

    Either ``loader_factory`` or both explicit loaders must be supplied.  A
    loader factory is invoked exactly twice, with ``"train"`` and ``"val"``.
    ``forward_callback`` may adapt an APN wrapper, but it must return
    predictions/targets[/mask]; the trainer always computes masked MSE itself.

    Epoch count, patience and validation interval are intentionally absent
    from the signature so a campaign cannot tune them after inspecting data.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    checkpoint_name = _validate_artifact_name(
        checkpoint_name, label="checkpoint_name"
    )
    manifest_name = _validate_artifact_name(manifest_name, label="manifest_name")
    run_dir = require_within_root(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = require_within_root(run_dir / checkpoint_name)
    manifest_path = require_within_root(run_dir / manifest_name)
    config_snapshot = _jsonable(resolved_config)
    config_json = _canonical_json(config_snapshot)
    config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    argv_snapshot = [str(item) for item in (sys.argv if argv is None else argv)]
    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type not in ("cpu", "cuda"):
        raise ValueError("APN training device must be cpu or cuda")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA APN training requested but CUDA is unavailable")

    rng_state = _capture_rng_state()
    started_at = _utc_now()
    wall_started = time.perf_counter()
    setup_started = wall_started
    serialization_seconds = 0.0
    model: nn.Module | None = None
    best_state: dict[str, Any] | None = None
    best_state_sha256 = ""
    curve: list[dict[str, Any]] = []
    stopped_early = False

    try:
        _seed_deterministically(seed)
        # APN may set dynamic config fields while constructing its datasets, so
        # both loaders are intentionally resolved before the model factory.
        resolved_train_loader, resolved_validation_loader = _resolve_loaders(
            seed=seed,
            loader_factory=loader_factory,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        model = model_factory(seed)
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return torch.nn.Module")
        model.to(resolved_device)
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise APNTrainingError("APN model has no trainable parameters")
        trainable_names = [name for name, _ in trainable]
        trainable_count = sum(parameter.numel() for _, parameter in trainable)
        total_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.Adam(
            [parameter for _, parameter in trainable], lr=learning_rate
        )
        if type(optimizer) is not torch.optim.Adam:
            raise APNTrainingError("Locked APN optimizer is Adam")
        scheduler = make_delayed_step_decay_lr(optimizer)
        if resolved_device.type == "cuda":
            _sync(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
        setup_seconds = time.perf_counter() - setup_started

        best_validation_mse = float("inf")
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(1, LOCKED_EPOCHS + 1):
            epoch_started = time.perf_counter()
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            model.train()
            train_sse = 0.0
            train_weight = 0.0
            train_batches = 0
            skipped_train_batches = 0
            for raw_batch in _materialize_loader(resolved_train_loader):
                batch = _move_to_device(raw_batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                prediction, target, mask = _run_forward(
                    model, batch, "train", forward_callback
                )
                sse, weight = _masked_sse_and_weight(prediction, target, mask)
                if weight <= 0:
                    skipped_train_batches += 1
                    continue
                loss = sse / weight
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise APNTrainingError("Training MSE must be a finite scalar")
                if not loss.requires_grad:
                    raise APNTrainingError(
                        "Training MSE has no gradient; check the APN forward callback"
                    )
                loss.backward()
                for name, parameter in trainable:
                    if parameter.grad is not None and not torch.isfinite(
                        parameter.grad
                    ).all():
                        raise APNTrainingError(
                            f"Non-finite gradient in trainable parameter {name!r}"
                        )
                optimizer.step()
                train_sse += float(sse.detach().item())
                train_weight += weight
                train_batches += 1
            if train_batches == 0 or train_weight <= 0:
                raise APNTrainingError("Training loader contains no observed targets")
            train_mse = train_sse / train_weight

            validation_started = time.perf_counter()
            validation_mse, validation_batches = _validation_micro_mse(
                model,
                resolved_validation_loader,
                device=resolved_device,
                callback=forward_callback,
            )
            validation_seconds = time.perf_counter() - validation_started
            improved = validation_mse < best_validation_mse
            if improved:
                best_validation_mse = validation_mse
                best_epoch = epoch
                stale_epochs = 0
                best_state = _clone_cpu_state(model)
                best_state_sha256 = _state_sha256(best_state)
                save_started = time.perf_counter()
                _atomic_torch_save(best_state, checkpoint_path)
                serialization_seconds += time.perf_counter() - save_started
            else:
                stale_epochs += 1

            scheduler.step()
            next_learning_rate = float(optimizer.param_groups[0]["lr"])
            _sync(resolved_device)
            curve.append(
                {
                    "epoch": epoch,
                    "train_micro_mse": float(train_mse),
                    "validation_micro_mse": float(validation_mse),
                    "learning_rate": learning_rate_used,
                    "next_learning_rate": next_learning_rate,
                    "train_batches": train_batches,
                    "skipped_train_batches": skipped_train_batches,
                    "validation_batches": validation_batches,
                    "improved": improved,
                    "stale_epochs": stale_epochs,
                    "validation_wall_seconds": float(validation_seconds),
                    "epoch_wall_seconds": float(time.perf_counter() - epoch_started),
                }
            )
            if stale_epochs >= LOCKED_PATIENCE:
                stopped_early = True
                break

        if best_state is None or best_epoch <= 0:
            raise APNTrainingError("No finite validation checkpoint was selected")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        if _state_sha256(model.state_dict()) != best_state_sha256:
            raise APNTrainingError("Reloaded best checkpoint state hash mismatch")
        if not checkpoint_path.is_file():
            raise APNTrainingError("Atomic best checkpoint is missing")
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        _sync(resolved_device)
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda"
            else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(resolved_device))
            if resolved_device.type == "cuda"
            else 0
        )
        wall_seconds = time.perf_counter() - wall_started
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "action": "train_apn_backbone",
            "seed": seed,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "device": str(resolved_device),
            "training_policy": {
                "optimizer": LOCKED_OPTIMIZER,
                "optimizer_class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}",
                "loss": LOCKED_LOSS,
                "scheduler": LOCKED_SCHEDULER,
                "scheduler_formula": "1.0 if scheduler_epoch < 2 else 0.8 ** (scheduler_epoch - 2)",
                "learning_rate": learning_rate,
                "epochs": LOCKED_EPOCHS,
                "patience": LOCKED_PATIENCE,
                "validation_interval": LOCKED_VALIDATION_INTERVAL,
                "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            },
            "resolved_config": config_snapshot,
            "resolved_config_sha256": config_sha256,
            "argv": argv_snapshot,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "total_parameter_count": total_count,
            "best_epoch": best_epoch,
            "best_validation_micro_mse": float(best_validation_mse),
            "epochs_ran": len(curve),
            "stopped_early": stopped_early,
            "checkpoint": {
                "path": _relative_project_path(checkpoint_path),
                "sha256": checkpoint_sha256,
                "state_sha256": best_state_sha256,
                "bytes": checkpoint_path.stat().st_size,
                "atomic_replace": True,
                "weights_only": True,
            },
            "determinism": {
                "python_numpy_torch_seed": seed,
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "caller_rng_state_restored": True,
            },
            "timing": {
                "wall_seconds": float(wall_seconds),
                "setup_seconds": float(setup_seconds),
                "checkpoint_serialization_seconds": float(serialization_seconds),
                "device_synchronized": resolved_device.type == "cuda",
            },
            "peak_cuda_memory": {
                "allocated_bytes": peak_allocated,
                "reserved_bytes": peak_reserved,
            },
            "curve": curve,
        }
        _atomic_write_json(manifest, manifest_path)
        return APNTrainingResult(
            model=model,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            checkpoint_sha256=checkpoint_sha256,
            best_state_sha256=best_state_sha256,
            best_epoch=best_epoch,
            best_validation_mse=float(best_validation_mse),
            epochs_ran=len(curve),
            stopped_early=stopped_early,
        )
    finally:
        _restore_rng_state(rng_state)


# Concise public alias for campaign code.
train_apn = train_apn_train_val


__all__ = [
    "APNTrainingError",
    "APNTrainingResult",
    "LOCKED_EPOCHS",
    "LOCKED_LOSS",
    "LOCKED_OPTIMIZER",
    "LOCKED_PATIENCE",
    "LOCKED_SCHEDULER",
    "LOCKED_VALIDATION_INTERVAL",
    "delayed_step_decay_factor",
    "lazy_vendor_loader_factory",
    "lazy_vendor_model_factory",
    "make_delayed_step_decay_lr",
    "train_apn",
    "train_apn_train_val",
]
