from __future__ import annotations

import subprocess
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from evipatch.evidence import EVIDENCE_WIDTHS
from models.APN import Model as PatchedModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APN_ROOT = PROJECT_ROOT / "vendor" / "APN"


def _upstream_module() -> types.ModuleType:
    source = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={APN_ROOT.as_posix()}",
            "-C",
            str(APN_ROOT),
            "show",
            "HEAD:models/APN.py",
        ],
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType("upstream_apn_for_parity")
    exec(compile(source, "upstream_models_APN.py", "exec"), module.__dict__)
    return module


def _configs(mode: str = "apn") -> SimpleNamespace:
    return SimpleNamespace(
        task_name="short_term_forecast",
        d_model=24,
        apn_te_dim=8,
        enc_in=4,
        apn_npatch=5,
        apn_nlayer=2,
        apn_attn_heads=8,
        dropout=0.1,
        features="M",
        evipatch_mode=mode,
        evipatch_random_seed=1729,
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(99)
    batch_size, history_length, pred_length, variables = 3, 13, 3, 4
    return {
        "x": torch.randn(batch_size, history_length, variables),
        "x_mark": torch.linspace(0, 0.74, history_length)
        .view(1, history_length, 1)
        .expand(batch_size, -1, -1),
        "x_mask": (torch.rand(batch_size, history_length, variables) > 0.2).float(),
        "y": torch.randn(batch_size, pred_length, variables),
        "y_mark": torch.linspace(0.75, 0.99, pred_length)
        .view(1, pred_length, 1)
        .expand(batch_size, -1, -1),
        "y_mask": torch.ones(batch_size, pred_length, variables),
    }


def test_patched_apn_full_forward_matches_upstream_after_state_migration() -> None:
    upstream_class = _upstream_module().Model
    torch.manual_seed(123)
    upstream = upstream_class(_configs()).eval()
    patched = PatchedModel(_configs("apn")).eval()
    migration = patched.load_state_dict(upstream.state_dict(), strict=True)
    assert migration.missing_keys == []
    assert migration.unexpected_keys == []

    batch = _batch()
    with torch.no_grad():
        upstream_output = upstream(**batch)
        patched_output = patched(**batch)
    torch.testing.assert_close(
        patched_output["pred"],
        upstream_output["pred"],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.equal(patched_output["true"], upstream_output["true"])
    assert torch.equal(patched_output["mask"], upstream_output["mask"])


def test_all_variants_keep_downstream_width_and_expected_projection_width() -> None:
    for mode, width in EVIDENCE_WIDTHS.items():
        model = PatchedModel(_configs(mode))
        patching = model.model.patching
        assert patching.projection_layer.in_features == 1 + 8 + width
        assert patching.projection_layer.out_features == 24
        if mode == "random_features":
            assert patching.random_feature_table is not None
            assert patching.random_feature_table.requires_grad is False
            parameter_names = dict(model.named_parameters())
            assert not any("random_feature_table" in name for name in parameter_names)
        else:
            assert patching.random_feature_table is None


def test_full_forward_has_finite_gradients_for_sparse_history() -> None:
    model = PatchedModel(_configs("evipatch_full"))
    batch = _batch()
    batch["x_mask"].zero_()
    batch["x_mask"][:, 3, :] = 1
    output = model(**batch)["pred"]
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
