from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
from torch import Tensor, nn


DEFAULT_DECODER_PREFIXES = ("model.model.decoder.", "model.decoder.")


class DecoderRefitError(RuntimeError):
    """Raised when a decoder-only refit would violate the frozen-state contract."""


@dataclass(frozen=True)
class MicroMSEContribution:
    """Unreduced validation error used to compute a true micro average."""

    sse: float
    observations: int

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.sse)) or float(self.sse) < 0:
            raise ValueError("validation SSE must be finite and nonnegative")
        if int(self.observations) < 0:
            raise ValueError("validation observations must be nonnegative")


@dataclass(frozen=True)
class DecoderRefitResult:
    """Audit record for a global decoder-only hyperparameter selection.

    The supplied model is mutated to the selected candidate's best validation
    state. ``curve`` contains one record per learning-rate/weight-decay pair.
    """

    selected_config: dict[str, float | int]
    curve: tuple[dict[str, Any], ...]
    trainable_parameter_names: tuple[str, ...]
    decoder_prefix: str
    initial_state_hash: str
    initial_frozen_hash: str
    final_frozen_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainStep(Protocol):
    def __call__(self, model: nn.Module, batch: Any) -> Tensor | Mapping[str, Any]: ...


class ValidationStep(Protocol):
    def __call__(self, model: nn.Module, batch: Any) -> Any: ...


Loader: TypeAlias = Iterable[Any] | Callable[[], Iterable[Any]]


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for name, value in state.items():
        cloned[name] = value.detach().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
    return cloned


def _tensor_bytes(value: Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def _hash_tensor_items(items: Iterable[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items, key=lambda item: item[0]):
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        payload = _tensor_bytes(value)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _state_hash(state: Mapping[str, Any]) -> str:
    tensors = [(name, value) for name, value in state.items() if isinstance(value, Tensor)]
    return _hash_tensor_items(tensors)


def _frozen_hash(state: Mapping[str, Any], decoder_prefix: str) -> str:
    tensors = [
        (name, value)
        for name, value in state.items()
        if isinstance(value, Tensor) and not name.startswith(decoder_prefix)
    ]
    return _hash_tensor_items(tensors)


def _assert_frozen_byte_identical(
    initial_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    decoder_prefix: str,
) -> None:
    initial_names = {
        name
        for name, value in initial_state.items()
        if isinstance(value, Tensor) and not name.startswith(decoder_prefix)
    }
    final_names = {
        name
        for name, value in final_state.items()
        if isinstance(value, Tensor) and not name.startswith(decoder_prefix)
    }
    if initial_names != final_names:
        missing = sorted(initial_names - final_names)
        added = sorted(final_names - initial_names)
        raise DecoderRefitError(
            f"Frozen state keys changed (missing={missing}, added={added})"
        )
    changed = [
        name
        for name in sorted(initial_names)
        if initial_state[name].dtype != final_state[name].dtype
        or tuple(initial_state[name].shape) != tuple(final_state[name].shape)
        or _tensor_bytes(initial_state[name]) != _tensor_bytes(final_state[name])
    ]
    if changed:
        raise DecoderRefitError(
            f"Non-decoder tensors changed during decoder-only refit: {changed}"
        )


def _resolve_decoder_prefix(
    model: nn.Module, prefixes: Iterable[str]
) -> tuple[str, tuple[str, ...]]:
    names = tuple(name for name, _ in model.named_parameters())
    normalized = tuple(dict.fromkeys(str(prefix) for prefix in prefixes))
    if not normalized or any(not prefix.endswith(".") for prefix in normalized):
        raise ValueError("decoder prefixes must be nonempty and end with '.'")
    matches = {
        prefix: tuple(name for name in names if name.startswith(prefix))
        for prefix in normalized
    }
    active = [(prefix, matched) for prefix, matched in matches.items() if matched]
    if not active:
        raise DecoderRefitError(
            "No APN decoder parameters found; expected one of " + ", ".join(normalized)
        )
    if len(active) != 1:
        raise DecoderRefitError(
            f"Ambiguous APN decoder prefixes: {[prefix for prefix, _ in active]}"
        )
    prefix, matched = active[0]
    return prefix, matched


def _decoder_module(model: nn.Module, decoder_prefix: str) -> nn.Module:
    module_name = decoder_prefix[:-1]
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise DecoderRefitError(f"Decoder module {module_name!r} is not registered")
    return modules[module_name]


def _configure_trainable_parameters(
    model: nn.Module, trainable_names: tuple[str, ...]
) -> list[nn.Parameter]:
    allowed = set(trainable_names)
    selected: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
        parameter.grad = None
        if name in allowed:
            selected.append(parameter)
    if not selected:
        raise DecoderRefitError("The decoder has no trainable parameters")
    return selected


def _materialize_loader(loader: Loader) -> Iterable[Any]:
    return loader() if callable(loader) else loader


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _prediction_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("prediction", "predictions", "forecast", "output"):
            if isinstance(output.get(key), Tensor):
                return output[key]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, Tensor):
                return item
    raise TypeError("Model output does not contain a prediction tensor")


def _unpack_supervised_batch(batch: Any) -> tuple[Any, Tensor, Tensor | None]:
    if isinstance(batch, Mapping):
        inputs = batch.get("inputs", batch.get("input", batch.get("x")))
        target = batch.get("target", batch.get("targets", batch.get("y")))
        mask = batch.get("mask", batch.get("target_mask"))
        if inputs is None or not isinstance(target, Tensor):
            raise TypeError("Default steps require mapping inputs and tensor targets")
        if mask is not None and not isinstance(mask, Tensor):
            raise TypeError("target mask must be a tensor")
        return inputs, target, mask
    if isinstance(batch, (tuple, list)) and len(batch) in (2, 3):
        inputs, target = batch[0], batch[1]
        mask = batch[2] if len(batch) == 3 else None
        if not isinstance(target, Tensor) or (mask is not None and not isinstance(mask, Tensor)):
            raise TypeError("Default steps require tensor targets and masks")
        return inputs, target, mask
    raise TypeError(
        "Default steps require (inputs, target[, mask]) or a mapping; provide callbacks otherwise"
    )


def _observed_squared_error(
    prediction: Tensor, target: Tensor, mask: Tensor | None
) -> tuple[Tensor, int]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes do not match")
    if mask is None:
        observed = torch.ones_like(target, dtype=torch.bool)
    else:
        if mask.shape != target.shape:
            raise ValueError("target mask shape does not match target")
        observed = mask > 0
    count = int(observed.sum().item())
    if count == 0:
        return prediction.new_zeros(()), 0
    residual = prediction[observed] - target[observed]
    if not torch.isfinite(residual).all():
        raise ValueError("observed predictions/targets must be finite")
    return residual.square().sum(), count


def _default_train_step(model: nn.Module, batch: Any, device: torch.device) -> Tensor:
    inputs, target, mask = _unpack_supervised_batch(_move(batch, device))
    prediction = _prediction_tensor(model(inputs))
    sse, observations = _observed_squared_error(prediction, target, mask)
    if observations == 0:
        raise ValueError("training batch contains no observed targets")
    return sse / observations


def _default_validation_step(
    model: nn.Module, batch: Any, device: torch.device
) -> MicroMSEContribution:
    inputs, target, mask = _unpack_supervised_batch(_move(batch, device))
    prediction = _prediction_tensor(model(inputs))
    sse, observations = _observed_squared_error(prediction, target, mask)
    return MicroMSEContribution(float(sse.item()), observations)


def _loss_from_callback(value: Any) -> Tensor:
    if isinstance(value, Mapping):
        value = value.get("loss")
    elif isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise TypeError("train_step must return a scalar loss tensor or {'loss': tensor}")
    if not torch.isfinite(value):
        raise ValueError("training loss must be finite")
    return value


def _contribution_from_callback(value: Any) -> MicroMSEContribution:
    if isinstance(value, MicroMSEContribution):
        return value
    if isinstance(value, Mapping):
        if "sse" in value and ("observations" in value or "n" in value):
            return MicroMSEContribution(
                float(value["sse"]), int(value.get("observations", value.get("n")))
            )
        prediction = value.get("prediction", value.get("predictions"))
        target = value.get("target", value.get("targets"))
        mask = value.get("mask", value.get("target_mask"))
        if isinstance(prediction, Tensor) and isinstance(target, Tensor):
            sse, observations = _observed_squared_error(prediction, target, mask)
            return MicroMSEContribution(float(sse.item()), observations)
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(
            isinstance(item, (int, float))
            or (isinstance(item, Tensor) and item.numel() == 1)
            for item in value
        ):
            return MicroMSEContribution(float(value[0]), int(value[1]))
        if len(value) in (2, 3) and isinstance(value[0], Tensor) and isinstance(value[1], Tensor):
            mask = value[2] if len(value) == 3 else None
            sse, observations = _observed_squared_error(value[0], value[1], mask)
            return MicroMSEContribution(float(sse.item()), observations)
    raise TypeError(
        "validation_step must return SSE/N or prediction/target[/mask], not a reduced batch mean"
    )


def _validation_micro_mse(
    model: nn.Module,
    loader: Loader,
    validation_step: ValidationStep | None,
    device: torch.device,
) -> float:
    model.eval()
    total_sse = 0.0
    total_observations = 0
    with torch.no_grad():
        for batch in _materialize_loader(loader):
            contribution = (
                _default_validation_step(model, batch, device)
                if validation_step is None
                else _contribution_from_callback(validation_step(model, batch))
            )
            total_sse += contribution.sse
            total_observations += contribution.observations
    if total_observations <= 0:
        raise ValueError("validation loader contains no observed targets")
    mse = total_sse / total_observations
    if not math.isfinite(mse):
        raise ValueError("validation micro MSE is not finite")
    return float(mse)


@dataclass
class _RNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: list[Tensor] | None


def _capture_rng_state() -> _RNGState:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return _RNGState(
        random.getstate(), np.random.get_state(), torch.random.get_rng_state(), cuda_state
    )


def _restore_rng_state(state: _RNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


def _validate_grid(values: Iterable[float], *, name: str, positive: bool) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError(f"{name} grid may not be empty")
    if len(set(grid)) != len(grid):
        raise ValueError(f"{name} grid contains duplicates")
    for value in grid:
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            relation = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} values must be finite and {relation}")
    return grid


def fit_decoder_only(
    model: nn.Module,
    train_loader: Loader,
    validation_loader: Loader,
    *,
    initial_state: Mapping[str, Any] | None = None,
    learning_rates: Iterable[float] = (1e-4, 3e-4, 1e-3),
    weight_decays: Iterable[float] = (0.0, 1e-4),
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    train_step: TrainStep | None = None,
    validation_step: ValidationStep | None = None,
    device: str | torch.device | None = None,
    decoder_prefixes: Iterable[str] = DEFAULT_DECODER_PREFIXES,
) -> DecoderRefitResult:
    """Refit only the APN decoder and select one LR/WD pair by validation MSE.

    Every candidate reloads the exact same ``initial_state`` and receives a
    fresh AdamW optimizer.  The validation callback must expose unreduced SSE
    and observation count (or predictions, targets, and an optional mask), so
    selection is by dataset-level micro MSE rather than a mean of batch means.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if max_epochs <= 0 or max_epochs > 100 or patience <= 0:
        raise ValueError("max_epochs must be in [1, 100] and patience must be positive")
    if not math.isfinite(float(min_delta)) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    lr_grid = _validate_grid(learning_rates, name="learning-rate", positive=True)
    wd_grid = _validate_grid(weight_decays, name="weight-decay", positive=False)
    decoder_prefix, trainable_names = _resolve_decoder_prefix(model, decoder_prefixes)

    reference = _clone_state(model.state_dict() if initial_state is None else initial_state)
    try:
        model.load_state_dict(reference, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise DecoderRefitError("initial_state is incompatible with the model") from exc
    initial_state_hash = _state_hash(reference)
    initial_frozen_hash = _frozen_hash(reference, decoder_prefix)
    if device is None:
        first_parameter = next(model.parameters(), None)
        resolved_device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    else:
        resolved_device = torch.device(device)
        model.to(resolved_device)
        reference = _clone_state(model.state_dict())
        initial_state_hash = _state_hash(reference)
        initial_frozen_hash = _frozen_hash(reference, decoder_prefix)

    decoder = _decoder_module(model, decoder_prefix)
    common_rng = _capture_rng_state()
    curve: list[dict[str, Any]] = []
    selected_state: dict[str, Any] | None = None
    selected_mse = float("inf")
    selected_config: dict[str, float | int] | None = None

    try:
        for learning_rate in lr_grid:
            for weight_decay in wd_grid:
                model.load_state_dict(reference, strict=True)
                _restore_rng_state(common_rng)
                parameters = _configure_trainable_parameters(model, trainable_names)
                optimizer = torch.optim.AdamW(
                    parameters, lr=learning_rate, weight_decay=weight_decay
                )
                candidate_initial_hash = _state_hash(model.state_dict())
                if candidate_initial_hash != initial_state_hash:
                    raise DecoderRefitError("Candidate did not start from the common initial state")

                best_state: dict[str, Any] | None = None
                best_mse = float("inf")
                best_epoch = 0
                stale_epochs = 0
                epoch_curve: list[dict[str, float | int]] = []
                stopped_early = False

                for epoch in range(1, max_epochs + 1):
                    model.eval()
                    decoder.train()
                    train_sse = 0.0
                    train_observations = 0
                    batches = 0
                    for batch in _materialize_loader(train_loader):
                        optimizer.zero_grad(set_to_none=True)
                        loss = (
                            _default_train_step(model, batch, resolved_device)
                            if train_step is None
                            else _loss_from_callback(train_step(model, batch))
                        )
                        loss.backward()
                        optimizer.step()
                        train_sse += float(loss.detach().item())
                        train_observations += 1
                        batches += 1
                    if batches == 0:
                        raise ValueError("training loader produced no batches")

                    validation_mse = _validation_micro_mse(
                        model, validation_loader, validation_step, resolved_device
                    )
                    epoch_curve.append(
                        {
                            "epoch": epoch,
                            "train_mean_batch_loss": train_sse / train_observations,
                            "validation_micro_mse": validation_mse,
                        }
                    )
                    if validation_mse < best_mse - min_delta:
                        best_mse = validation_mse
                        best_epoch = epoch
                        best_state = _clone_state(model.state_dict())
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= patience:
                            stopped_early = True
                            break

                if best_state is None:
                    raise DecoderRefitError("Decoder candidate produced no valid validation state")
                _assert_frozen_byte_identical(reference, best_state, decoder_prefix)
                curve.append(
                    {
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "candidate_initial_state_hash": candidate_initial_hash,
                        "best_epoch": best_epoch,
                        "best_validation_micro_mse": best_mse,
                        "epochs_ran": len(epoch_curve),
                        "stopped_early": stopped_early,
                        "epochs": epoch_curve,
                    }
                )
                if best_mse < selected_mse:
                    selected_mse = best_mse
                    selected_state = best_state
                    selected_config = {
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "best_epoch": best_epoch,
                        "validation_micro_mse": best_mse,
                    }

        if selected_state is None or selected_config is None:
            raise DecoderRefitError("No decoder-only candidate was selected")
        model.load_state_dict(selected_state, strict=True)
        _configure_trainable_parameters(model, trainable_names)
        model.eval()
        final_state = model.state_dict()
        _assert_frozen_byte_identical(reference, final_state, decoder_prefix)
        final_frozen_hash = _frozen_hash(final_state, decoder_prefix)
        if final_frozen_hash != initial_frozen_hash:
            raise DecoderRefitError("Frozen-state hash changed during decoder-only refit")
        return DecoderRefitResult(
            selected_config=selected_config,
            curve=tuple(curve),
            trainable_parameter_names=trainable_names,
            decoder_prefix=decoder_prefix,
            initial_state_hash=initial_state_hash,
            initial_frozen_hash=initial_frozen_hash,
            final_frozen_hash=final_frozen_hash,
        )
    except Exception:
        model.load_state_dict(reference, strict=True)
        _configure_trainable_parameters(model, trainable_names)
        model.eval()
        raise
