from __future__ import annotations

import json

import pytest
import torch

from edgetwincal.aggregate import compact_run_manifest
from edgetwincal.graph import fit_graph_with_validation
from edgetwincal.latent import fit_latent_head_with_validation
from edgetwincal.paths import PROJECT_ROOT, require_within_root


def test_latent_head_recovers_sensor_specific_residual() -> None:
    torch.manual_seed(23)
    train_base = torch.randn(500, 2, 4)
    val_base = torch.randn(180, 2, 4)
    train_features = torch.randn(500, 4, 3)
    val_features = torch.randn(180, 4, 3)
    coefficient = torch.tensor([0.3, -0.2, 0.1])

    def target(base: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        residual = torch.einsum("bnd,d->bn", features, coefficient).unsqueeze(1)
        return base + residual

    train_target = target(train_base, train_features)
    val_target = target(val_base, val_features)
    train_mask = torch.ones_like(train_target)
    val_mask = torch.ones_like(val_target)
    head, audit = fit_latent_head_with_validation(
        train_base,
        train_features,
        train_target,
        train_mask,
        val_base,
        val_features,
        val_target,
        val_mask,
        alphas=[0.1, 1.0, 10.0],
    )
    baseline = (val_base - val_target).square().mean()
    corrected = (head.apply(val_base, val_features) - val_target).square().mean()
    assert corrected < baseline * 0.01
    assert len(audit) == 3


def test_two_modules_recover_complementary_residuals() -> None:
    torch.manual_seed(29)
    train_base = torch.randn(600, 2, 4)
    val_base = torch.randn(200, 2, 4)
    train_features = torch.randn(600, 4, 3)
    val_features = torch.randn(200, 4, 3)
    coefficient = torch.tensor([0.2, -0.1, 0.15])

    def target(base: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        local = torch.einsum("bnd,d->bn", features, coefficient).unsqueeze(1)
        cross = 0.25 * torch.roll(base, shifts=1, dims=2)
        return base + local + cross

    train_target = target(train_base, train_features)
    val_target = target(val_base, val_features)
    train_mask = torch.ones_like(train_target)
    val_mask = torch.ones_like(val_target)
    head, _ = fit_latent_head_with_validation(
        train_base,
        train_features,
        train_target,
        train_mask,
        val_base,
        val_features,
        val_target,
        val_mask,
        alphas=[0.1, 1.0],
    )
    latent_train = head.apply(train_base, train_features)
    latent_val = head.apply(val_base, val_features)
    graph, _ = fit_graph_with_validation(
        latent_train,
        train_target,
        train_mask,
        latent_val,
        val_target,
        val_mask,
        alphas=[0.1, 1.0, 10.0],
        train_source_forecasts=train_base,
        val_source_forecasts=val_base,
    )
    baseline = (val_base - val_target).square().mean()
    latent_error = (latent_val - val_target).square().mean()
    full_error = (graph.apply(latent_val, val_base) - val_target).square().mean()
    assert latent_error < baseline
    assert full_error < latent_error * 0.1
    diagonal = torch.diagonal(graph.weights, dim1=1, dim2=2)
    assert torch.count_nonzero(diagonal) == 0


def test_project_boundary_rejects_parent_escape() -> None:
    with pytest.raises(ValueError, match="escapes project root"):
        require_within_root(PROJECT_ROOT.parent / "outside")


def test_modules_remain_finite_with_sparse_targets() -> None:
    prediction = torch.zeros(8, 1, 3)
    target = torch.ones_like(prediction)
    mask = torch.zeros_like(prediction)
    mask[:3, :, :] = 1
    graph, _ = fit_graph_with_validation(
        prediction, target, mask, prediction, target, mask, alphas=[1.0]
    )
    assert torch.isfinite(graph.apply(prediction)).all()


def test_compact_manifest_excludes_machine_local_paths() -> None:
    metrics = {
        "schema_version": 1,
        "attempt": 5,
        "seed": 2024,
        "device": "cuda:0",
        "passed": True,
        "pass_threshold": 0.01,
        "validation_metrics": {"apn": {"mse": 0.3}},
        "test_metrics": {"apn": {"mse": 0.31}},
        "relative_improvement_vs_apn": {"full": {"mse": 0.01}},
        "fitted_coefficients": {"slrh": 2700, "cfg": 3888},
        "nonzero_coefficients": {"slrh": 2700, "cfg": 3886},
        "fitting": {
            "selected_alpha_slrh": 100.0,
            "selected_alpha_full_cfg": 1000.0,
            "total_wall_seconds": 1.4,
        },
        "assets": {
            "checkpoint": "C:\\Users\\someone\\checkpoint.bin",
            "checkpoint_sha256": "abc123",
            "cache": "C:\\Users\\someone\\cache.pt",
        },
    }
    manifest = compact_run_manifest(metrics)
    encoded = json.dumps(manifest)
    assert manifest["checkpoint_sha256"] == "abc123"
    assert manifest["fitted_coefficients"]["cfg"] == 3888
    assert "assets" not in manifest
    assert "C:\\\\" not in encoded
