from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from evipatch.controlled import match_support_pairs, patch_units, score_frozen_pairs


def _state() -> dict[str, torch.Tensor]:
    return {
        "model.model.te_scale.weight": torch.tensor([[1.0]]),
        "model.model.te_scale.bias": torch.tensor([0.0]),
        "model.model.te_periodic.weight": torch.tensor([[2.0]]),
        "model.model.te_periodic.bias": torch.tensor([0.1]),
        "model.model.patching.delta_left_params": torch.zeros(1, 2),
        "model.model.patching.raw_log_width_params": torch.full((1, 2), float(np.log(0.5))),
        "model.model.patching.tau_params": torch.tensor([-3.0]),
    }


def test_patch_units_reconstructs_finite_max_support_descriptor() -> None:
    x = np.array(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[1.0], [2.0], [0.0], [0.0]],
        ]
    )
    x_mark = np.broadcast_to(
        np.linspace(0.1, 0.9, 4)[None, :, None], (2, 4, 1)
    ).copy()
    x_mask = np.array(
        [
            [[1.0], [1.0], [1.0], [1.0]],
            [[1.0], [1.0], [0.0], [0.0]],
        ]
    )
    target_mask = np.ones((2, 2, 1))
    units = patch_units(
        x, x_mark, x_mask, target_mask, np.array([11, 22]), _state(), 2024
    )
    assert units.shape[0] == 2
    assert units["patient_id"].tolist() == [11, 22]
    assert (units["effective_support"] > 0).all()
    assert np.isfinite(units.filter(like="descriptor_").to_numpy()).all()


def test_matching_is_disjoint_and_respects_support_contrast() -> None:
    units = pd.DataFrame(
        [
            {"seed": 2024, "patient_id": 1, "channel": 0, "patch": 0, "effective_support": 1.0, "descriptor_0": 0.00, "descriptor_1": 0.00},
            {"seed": 2024, "patient_id": 2, "channel": 0, "patch": 1, "effective_support": 3.0, "descriptor_0": 0.01, "descriptor_1": 0.00},
            {"seed": 2024, "patient_id": 3, "channel": 0, "patch": 0, "effective_support": 1.2, "descriptor_0": 1.00, "descriptor_1": 1.00},
            {"seed": 2024, "patient_id": 4, "channel": 0, "patch": 1, "effective_support": 4.0, "descriptor_0": 1.01, "descriptor_1": 1.00},
        ]
    )
    pairs = match_support_pairs(
        units,
        2024,
        max_centroid_rms=0.1,
        min_effective_support_ratio=2.0,
        min_effective_support_difference=1.0,
    )
    assert len(pairs) == 2
    patients = pairs[["low_patient_id", "high_patient_id"]].to_numpy().reshape(-1)
    assert np.unique(patients).size == patients.size
    assert (pairs["support_ratio"] >= 2.0).all()
    assert (pairs["support_difference"] >= 1.0).all()


def test_frozen_pairs_score_identical_ids_for_each_variant() -> None:
    pairs = pd.DataFrame(
        [
            {
                "seed": 2024,
                "pair_id": "p0",
                "channel": 0,
                "low_patient_id": 10,
                "high_patient_id": 20,
                "low_patch": 0,
                "high_patch": 1,
                "centroid_rms": 0.01,
                "low_effective_support": 1.0,
                "high_effective_support": 3.0,
                "support_ratio": 3.0,
                "support_difference": 2.0,
            }
        ]
    )
    target = np.zeros((2, 2, 1))
    mask = np.ones_like(target)
    common = {"sample_ids": np.array([10, 20]), "target": target, "target_mask": mask}
    evaluations = {
        "apn": {**common, "pred": np.ones_like(target)},
        "evipatch_full": {**common, "pred": np.full_like(target, 0.5)},
    }
    errors = score_frozen_pairs(pairs, evaluations, 2024)
    assert errors["pair_id"].nunique() == 1
    assert errors.set_index("variant").loc["apn", "MSE"] == 1.0
    assert errors.set_index("variant").loc["evipatch_full", "MSE"] == 0.25
