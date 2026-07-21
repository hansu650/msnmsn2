from __future__ import annotations

import inspect
import json
import random
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from edgetwincal.apn_training import (
    LOCKED_EPOCHS,
    LOCKED_PATIENCE,
    LOCKED_VALIDATION_INTERVAL,
    delayed_step_decay_factor,
    lazy_vendor_model_factory,
    train_apn_train_val,
)
from edgetwincal.paths import PROJECT_ROOT


class _TinyAPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.decoder = nn.Linear(3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.tanh(self.encoder(inputs)))


class _SentinelTestLoader:
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        raise AssertionError("The once-only test loader was opened during training")


class _TrainValFactory:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(20260721)
        inputs = torch.randn(12, 2, generator=generator)
        targets = 0.6 * inputs[:, :1] - 0.25 * inputs[:, 1:]
        self.train = [
            (inputs[index : index + 4], targets[index : index + 4])
            for index in range(0, 8, 4)
        ]
        self.val = [(inputs[8:], torch.ones_like(targets[8:]))]
        self.test = _SentinelTestLoader()
        self.calls: list[tuple[str, int]] = []

    def __call__(self, split: str, seed: int):
        self.calls.append((split, seed))
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        if split == "test":
            return self.test
        raise AssertionError(f"Unexpected split {split!r}")


def _constant_validation_forward(
    model: nn.Module, batch: object, stage: str
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    if stage == "val":
        # A model-independent validation MSE of exactly 1.0 exercises the
        # locked patience without changing the loss implemented by the trainer.
        return torch.zeros_like(targets), targets
    return model(inputs), targets


@contextmanager
def _project_scratch(label: str):
    base = (
        PROJECT_ROOT
        / "results"
        / "edgetwincal_msn2026_v1"
        / "apn_training_test_scratch"
    )
    root = base / f"{label}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _run(root: Path, seed: int):
    loaders = _TrainValFactory()
    result = train_apn_train_val(
        output_dir=root,
        seed=seed,
        learning_rate=0.05,
        resolved_config={
            "dataset": "synthetic-train-val-only",
            "protocol": "unit-test",
            "seed": seed,
        },
        model_factory=lambda supplied_seed: _TinyAPN(),
        loader_factory=loaders,
        forward_callback=_constant_validation_forward,
        device="cpu",
        argv=["train_apn.py", "--seed", str(seed)],
    )
    return result, loaders


def test_train_val_only_early_stopping_adam_scheduler_and_atomic_manifest() -> None:
    assert "test_loader" not in inspect.signature(train_apn_train_val).parameters
    assert LOCKED_EPOCHS == 200
    assert LOCKED_PATIENCE == 10
    assert LOCKED_VALIDATION_INTERVAL == 1

    with _project_scratch("contract") as root:
        result, loaders = _run(root, 2024)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        assert loaders.calls == [("train", 2024), ("val", 2024)]
        assert loaders.test.iterations == 0
        assert result.stopped_early
        assert result.best_epoch == 1
        assert result.best_validation_mse == pytest.approx(1.0)
        assert result.epochs_ran == 1 + LOCKED_PATIENCE
        assert result.checkpoint_path.name == "pytorch_model.bin"
        assert result.checkpoint_path.is_file()
        assert not list(root.glob(".*.tmp"))

        policy = manifest["training_policy"]
        assert policy["optimizer"] == "Adam"
        assert policy["optimizer_class"] == "torch.optim.adam.Adam"
        assert policy["loss"] == "MSE"
        assert policy["scheduler"] == "DelayedStepDecayLR"
        assert policy["epochs"] == 200
        assert policy["patience"] == 10
        assert policy["validation_interval"] == 1
        assert policy["weight_decay"] == 0.0
        assert manifest["argv"] == ["train_apn.py", "--seed", "2024"]
        assert manifest["trainable_parameter_names"] == [
            "encoder.weight",
            "encoder.bias",
            "decoder.weight",
            "decoder.bias",
        ]
        assert manifest["peak_cuda_memory"] == {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        }

        curve = manifest["curve"]
        assert len(curve) == 11
        # LambdaLR initializes at scheduler epoch 0 and APN steps it after every
        # epoch: the first three training epochs use the base learning rate.
        assert [entry["learning_rate"] for entry in curve[:4]] == pytest.approx(
            [0.05, 0.05, 0.05, 0.04]
        )
        assert [entry["next_learning_rate"] for entry in curve[:3]] == pytest.approx(
            [0.05, 0.05, 0.04]
        )
        assert [entry["stale_epochs"] for entry in curve[:3]] == [0, 1, 2]
        assert all(entry["validation_micro_mse"] == 1.0 for entry in curve)
        assert delayed_step_decay_factor(0) == 1.0
        assert delayed_step_decay_factor(2) == 1.0
        assert delayed_step_decay_factor(3) == pytest.approx(0.8)

        checkpoint_state = torch.load(
            result.checkpoint_path, map_location="cpu", weights_only=True
        )
        for name, tensor in result.model.state_dict().items():
            assert torch.equal(tensor.detach().cpu(), checkpoint_state[name]), name
        assert manifest["checkpoint"]["sha256"] == result.checkpoint_sha256
        assert manifest["checkpoint"]["state_sha256"] == result.best_state_sha256
        assert manifest["checkpoint"]["atomic_replace"] is True


def test_seed_runs_are_reproducible_isolated_and_restore_caller_rng() -> None:
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    with _project_scratch("seeds") as root:
        first, first_loaders = _run(root / "seed2024_first", 2024)
        different, different_loaders = _run(root / "seed2025", 2025)
        repeated, repeated_loaders = _run(root / "seed2024_repeated", 2024)

        assert first.best_state_sha256 == repeated.best_state_sha256
        assert first.checkpoint_sha256 == repeated.checkpoint_sha256
        assert different.best_state_sha256 != first.best_state_sha256
        assert different.checkpoint_sha256 != first.checkpoint_sha256
        assert first_loaders.test.iterations == 0
        assert different_loaders.test.iterations == 0
        assert repeated_loaders.test.iterations == 0

    expected_torch = torch.rand(4, generator=torch.Generator().set_state(torch_state))
    actual_torch = torch.rand(4)
    assert torch.equal(actual_torch, expected_torch)
    expected_numpy = np.random.RandomState()
    expected_numpy.set_state(numpy_state)
    assert np.array_equal(np.random.random(4), expected_numpy.random_sample(4))
    expected_python = random.Random()
    expected_python.setstate(python_state)
    assert [random.random() for _ in range(4)] == [
        expected_python.random() for _ in range(4)
    ]


def test_vendor_factory_is_lazy_until_training_constructs_the_model() -> None:
    module_name = "edgetwincal_test_vendor_module_that_does_not_exist"
    assert module_name not in sys.modules
    factory = lazy_vendor_model_factory(
        {"synthetic": True}, module_name=module_name, class_name="Model"
    )
    assert callable(factory)
    assert module_name not in sys.modules
