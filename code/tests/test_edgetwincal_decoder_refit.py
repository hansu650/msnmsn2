from __future__ import annotations

import pytest
import torch
from torch import nn

from edgetwincal.decoder_refit import (
    DecoderRefitError,
    MicroMSEContribution,
    fit_decoder_only,
)


class _APNCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.decoder = nn.Linear(3, 1)
        self.register_buffer("encoder_calibration", torch.tensor([1.25]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(self.encoder(inputs))
        return self.decoder(latent) * self.encoder_calibration


class _OneLevelAPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _APNCore()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class _MiddleWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _APNCore()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class _TwoLevelAPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _MiddleWrapper()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def _data() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(20260721)
    inputs = torch.randn(24, 2, generator=generator)
    targets = (0.7 * inputs[:, :1]) - (0.4 * inputs[:, 1:])
    return [
        (inputs[index : index + 6], targets[index : index + 6])
        for index in range(0, inputs.shape[0], 6)
    ]


def _training_loss(model: nn.Module, batch: object) -> torch.Tensor:
    inputs, targets = batch
    return (model(inputs) - targets).square().mean()


def _constant_validation(model: nn.Module, batch: object) -> MicroMSEContribution:
    del model, batch
    return MicroMSEContribution(sse=7.0, observations=7)


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_six_candidates_share_initial_state_and_preserve_frozen_tensors() -> None:
    torch.manual_seed(41)
    model = _OneLevelAPN()
    initial = _clone_state(model)

    result = fit_decoder_only(
        model,
        _data(),
        [None],
        initial_state=initial,
        learning_rates=(1e-4, 3e-4, 1e-3),
        weight_decays=(0.0, 1e-4),
        max_epochs=20,
        patience=2,
        train_step=_training_loss,
        validation_step=_constant_validation,
    )

    assert len(result.curve) == 6
    assert result.trainable_parameter_names == (
        "model.decoder.weight",
        "model.decoder.bias",
    )
    assert result.decoder_prefix == "model.decoder."
    assert result.initial_frozen_hash == result.final_frozen_hash
    assert {entry["candidate_initial_state_hash"] for entry in result.curve} == {
        result.initial_state_hash
    }
    assert all(entry["stopped_early"] for entry in result.curve)
    assert all(entry["epochs_ran"] == 3 for entry in result.curve)
    assert result.selected_config == {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "best_epoch": 1,
        "validation_micro_mse": 1.0,
    }

    final = model.state_dict()
    for name, initial_tensor in initial.items():
        if not name.startswith("model.decoder."):
            assert torch.equal(final[name], initial_tensor), name
    assert not torch.equal(final["model.decoder.weight"], initial["model.decoder.weight"])
    assert tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad) == (
        "model.decoder.weight",
        "model.decoder.bias",
    )


def test_nested_apn_decoder_name_is_recorded_exactly() -> None:
    torch.manual_seed(43)
    model = _TwoLevelAPN()
    result = fit_decoder_only(
        model,
        _data(),
        [None],
        learning_rates=(1e-3,),
        weight_decays=(0.0,),
        max_epochs=1,
        patience=1,
        train_step=_training_loss,
        validation_step=_constant_validation,
    )

    assert result.decoder_prefix == "model.model.decoder."
    assert result.trainable_parameter_names == (
        "model.model.decoder.weight",
        "model.model.decoder.bias",
    )
    assert result.initial_frozen_hash == result.final_frozen_hash


def test_nondecoder_mutation_fails_and_restores_initial_state() -> None:
    torch.manual_seed(47)
    model = _OneLevelAPN()
    initial = _clone_state(model)

    def mutating_step(candidate: nn.Module, batch: object) -> torch.Tensor:
        with torch.no_grad():
            candidate.model.encoder.weight.add_(1.0)
        return _training_loss(candidate, batch)

    with pytest.raises(DecoderRefitError, match="Non-decoder tensors changed"):
        fit_decoder_only(
            model,
            _data(),
            [None],
            learning_rates=(1e-3,),
            weight_decays=(0.0,),
            max_epochs=1,
            patience=1,
            train_step=mutating_step,
            validation_step=_constant_validation,
        )

    for name, initial_tensor in initial.items():
        assert torch.equal(model.state_dict()[name], initial_tensor), name
