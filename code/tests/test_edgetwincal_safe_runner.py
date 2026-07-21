from __future__ import annotations

import hashlib

import torch

from edgetwincal.safe import CandidateAudit, CandidateSpec, SafeSelection
from edgetwincal.safe_runner import (
    SAFE_RUNNER_VARIANTS,
    SafeSeedStates,
    _alpha_candidate_better,
)


class _Correction:
    def __init__(self, correction):
        self.correction = correction

    def apply(self, *args, **kwargs):
        base = args[0] if len(args) == 3 else args[1]
        return base + self.correction


class _Robust:
    horizon = 1
    channels = 2

    def __init__(self, correction):
        self.correction = correction
        cell = type("Cell", (), {"residual_scale": torch.tensor(1.0)})()
        self.cells = ((cell, cell),)

    def apply(self, latent, base):
        return base + self.correction


def _selection():
    spec = CandidateSpec(1.0, 0.5)
    audit = CandidateAudit(spec, True, -0.1, {}, 0.1, 0.1, ())
    return SafeSelection(
        False,
        False,
        spec,
        "selected",
        (audit,),
        {},
        hashlib.sha256(b"p").hexdigest(),
        hashlib.sha256(b"s").hexdigest(),
        hashlib.sha256(b"a").hexdigest(),
    )


def _split():
    return {
        "features": torch.zeros(3, 2, 1),
        "base_prediction": torch.zeros(3, 1, 2),
        "target": torch.ones(3, 1, 2),
        "mask": torch.ones(3, 1, 2, dtype=torch.bool),
        "sample_id": torch.tensor([1, 2, 3]),
        "group_id": torch.tensor([10, 11, 12]),
    }


def test_prediction_registry_and_dataset_fallback_are_exact():
    correction = torch.full((3, 1, 2), 0.4)
    state = SafeSeedStates(
        _Correction(correction),
        _Correction(correction),
        _Robust(correction),
        _Robust(correction * 2),
        _Robust(correction * 3),
        _selection(),
        CandidateSpec(1.0, 0.5),
        (),
    )
    enabled = state.prediction_map(_split(), dataset_gate_enabled=True)
    assert tuple(enabled) == SAFE_RUNNER_VARIANTS
    assert torch.equal(enabled["Safe"], torch.full((3, 1, 2), 0.2))
    assert torch.equal(enabled["SafeNoBound"], correction)

    disabled = state.prediction_map(_split(), dataset_gate_enabled=False)
    assert torch.equal(disabled["Safe"], disabled["APN"])
    assert disabled["Safe"].data_ptr() != disabled["APN"].data_ptr()
    assert not torch.equal(disabled["SafeNoGate"], disabled["APN"])



def test_alpha_selection_uses_minimax_tolerance_then_safety_and_regularization():
    weak = (True, -0.10000, CandidateSpec(1.0, 0.50), 1.0, 1.0)
    safer = (True, -0.09995, CandidateSpec(0.5, 0.25), 1.0, 1.0)
    assert _alpha_candidate_better(safer, weak, tie_tolerance=1e-4)
    stronger = (True, -0.09995, CandidateSpec(0.5, 0.25), 100.0, 100.0)
    assert _alpha_candidate_better(stronger, safer, tie_tolerance=1e-4)
    worse = (True, -0.09, CandidateSpec(0.25, 0.25), 1000.0, 1000.0)
    assert not _alpha_candidate_better(worse, stronger, tie_tolerance=1e-4)
