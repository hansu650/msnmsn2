from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evipatch.runner import (
    build_apn_command,
    load_stage_config,
    run_checked,
    stage_lock,
)


def test_config_and_train_argv_contract() -> None:
    config = load_stage_config()
    command = build_apn_command(config, "evipatch_full", "train", seed=2025)
    joined = " ".join(command)
    assert "--evipatch_mode evipatch_full" in joined
    assert "--seed_base 2025" in joined
    assert "--skip_test_after_train 1" in joined
    assert "--is_training 1" in joined
    assert "--observation_shift none" in joined
    assert "AdamW" not in joined


def test_evaluation_argv_reuses_checkpoint_and_seed_shift(tmp_path: Path) -> None:
    config = load_stage_config()
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"test")
    command = build_apn_command(
        config,
        "raw_count",
        "evaluate",
        checkpoint=checkpoint,
        shift="burst",
        seed=2026,
    )
    joined = " ".join(command)
    assert f"--checkpoints_test {checkpoint.parent}" in joined
    assert "--shift_rate 0.3" in joined
    assert "--shift_seed 2026" in joined
    assert "--evipatch_eval_name burst" in joined
    assert "--save_arrays 1" in joined


def test_run_checked_records_argv_wall_time_and_local_log(tmp_path: Path) -> None:
    config = load_stage_config()
    log_path = tmp_path / "subprocess.log"
    result = run_checked(
        [config["project"]["python"], "-c", "print('runner-ok')"],
        log_path,
        os.environ.copy(),
        cwd=Path(config["project"]["root"]),
    )
    assert result["return_code"] == 0
    assert result["wall_seconds"] >= 0
    assert result["argv"][2] == "print('runner-ok')"
    assert "runner-ok" in log_path.read_text(encoding="utf-8")


def test_stage_lock_rejects_second_dispatcher() -> None:
    config = load_stage_config()
    with stage_lock(config):
        with pytest.raises(RuntimeError, match="lock already exists"):
            with stage_lock(config):
                pass
