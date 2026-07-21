from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from edgetwincal.safe_config import load_safe_config
import edgetwincal.safe_experiment as cli


def test_safe_help_is_lazy_and_lists_the_frozen_pipeline(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "download",
        "prepare-pretest",
        "smoke",
        "train",
        "fit",
        "gate",
        "test",
        "aggregate",
        "all",
    ):
        assert command in output


def test_vendor_config_is_apn_only_and_uses_new_dataset_dimensions():
    config = load_safe_config()
    spec = config.dataset("intel_lab")
    bundle = cli.build_safe_vendor_config(
        config,
        spec,
        2024,
        config_type=lambda **values: SimpleNamespace(**values),
    )
    vendor = bundle.config
    assert vendor.dataset_name == "intel_lab"
    assert vendor.seq_len == 72
    assert vendor.pred_len == 12
    assert vendor.enc_in == vendor.dec_in == vendor.c_out == 54
    assert vendor.apn_npatch == 12
    assert vendor.evipatch_mode == "apn"
    assert vendor.skip_test_after_train == 1
    assert bundle.public_audit()["test_loader_constructed"] is False


def test_only_test_and_all_may_reach_the_sealed_reader():
    assert cli.commands_that_may_open_test() == frozenset({"test", "all"})
    for name in cli._PRETEST_COMMAND_NAMES:
        source = inspect.getsource(cli._HANDLERS[name])
        assert "open_sealed_test_loader" not in source
    source = inspect.getsource(cli)
    assert source.count("safe_data.open_sealed_test_loader(") == 1


def test_safe_loader_is_reiterable_and_preserves_group_identity():
    batch = {
        "x": torch.zeros(2, 3, 4),
        "x_mark": torch.zeros(2, 3, 1),
        "y": torch.zeros(2, 1, 4),
        "y_mark": torch.zeros(2, 1, 1),
        "x_mask": torch.ones(2, 3, 4),
        "y_mask": torch.ones(2, 1, 4),
        "sample_id": torch.tensor([1, 2]),
        "group_id": torch.tensor([7, 8]),
    }
    loader = cli.SafeAPNLoader([batch])
    first = next(iter(loader))
    second = next(iter(loader))
    assert first["group_ID"].tolist() == [7, 8]
    assert second["group_ID"].tolist() == [7, 8]
    assert "sample_id" not in first and "group_id" not in first


def test_device_timing_is_unreachable_until_pass(tmp_path: Path):
    abandoned = tmp_path / "abandon.json"
    abandoned.write_text(
        json.dumps(
            {"verdict": "ABANDON", "device_timing_authorized": False}
        ),
        encoding="utf-8",
    )
    with pytest.raises(cli.SafeExperimentError, match="unreachable"):
        cli.require_device_timing_authorized(abandoned)

    passed = {"verdict": "PASS", "device_timing_authorized": True}
    assert cli.require_device_timing_authorized(passed) == passed


def test_fit_policy_keeps_dynamic_minimum_row_formula():
    policy = cli._fit_policy(load_safe_config())
    assert policy.minimum_rows is None


def test_entrypoint_sets_deterministic_cublas_before_safe_import():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_edgetwincal_safe.py"
    )
    expression = (
        "import os,runpy;"
        "os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None);"
        f"runpy.run_path({str(script)!r});"
        "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", expression],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert completed.stdout.strip() == ":4096:8"


def test_entrypoint_rejects_incompatible_cublas_value():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_edgetwincal_safe.py"
    )
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = "invalid"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
    )
    assert completed.returncode != 0
    assert "CUBLAS_WORKSPACE_CONFIG" in completed.stderr


def test_crash_resume_reads_only_immutable_window_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = SimpleNamespace(
        sha256="a" * 64,
        relative_path=lambda name: Path("results"),
    )
    spec = SimpleNamespace(dataset_id="beijing_air")
    normalizer = SimpleNamespace(sha256="b" * 64)
    protocol_sha256 = "c" * 64
    gate_sha256 = "d" * 64
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli.safe_data,
        "require_safe_path",
        lambda root, *parts: Path(root).joinpath(*parts).resolve(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("crash resume must not read sealed raw data")

    monkeypatch.setattr(
        cli.safe_data, "open_sealed_test_loader", forbidden
    )
    batch = {
        "x": torch.zeros(1, 2, 1),
        "x_mark": torch.zeros(1, 2, 1),
        "y": torch.zeros(1, 1, 1),
        "y_mark": torch.zeros(1, 1, 1),
        "x_mask": torch.ones(1, 2, 1),
        "y_mask": torch.ones(1, 1, 1),
        "sample_id": torch.tensor([1]),
        "group_id": torch.tensor([2]),
    }
    cache = cli._private_test_windows_path(config, spec.dataset_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "edgetwincal.safe-test-windows.v1",
            "config_sha256": config.sha256,
            "dataset_id": spec.dataset_id,
            "normalizer_sha256": normalizer.sha256,
            "protocol_sha256": protocol_sha256,
            "gate_sha256": gate_sha256,
            "batches": [batch],
        },
        cache,
    )
    cache_manifest = cli._private_test_windows_manifest_path(
        config, spec.dataset_id
    )
    cache_manifest.write_text(
        json.dumps(
            {
                "schema_version": (
                    "edgetwincal.safe-test-windows-manifest.v1"
                ),
                "config_sha256": config.sha256,
                "dataset_id": spec.dataset_id,
                "normalizer_sha256": normalizer.sha256,
                "protocol_sha256": protocol_sha256,
                "gate_sha256": gate_sha256,
                "cache_sha256": cli._sha256_file(cache),
            }
        ),
        encoding="utf-8",
    )
    ledger = cli._data_ledger_path(config, spec.dataset_id)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "status": "consumed",
                "dataset_id": spec.dataset_id,
                "provenance": {
                    "config_sha256": config.sha256,
                    "normalizer_sha256": normalizer.sha256,
                    "campaign_protocol_sha256": protocol_sha256,
                    "gate_sha256": gate_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = cli._load_test_windows_cache(
        config,
        spec,
        normalizer,
        protocol_sha256=protocol_sha256,
        gate_sha256=gate_sha256,
    )
    assert loaded[0]["group_id"].item() == 2
