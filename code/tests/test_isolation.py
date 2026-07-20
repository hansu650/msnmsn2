from __future__ import annotations

import os
import subprocess
from pathlib import Path

from evipatch.paths import PROJECT_ROOT
from evipatch.runner import load_stage_config


def _environment(tsdm_root: Path) -> dict[str, str]:
    config = load_stage_config()
    environment = os.environ.copy()
    environment["EVIPATCH_PROJECT_ROOT"] = str(PROJECT_ROOT)
    environment["EVIPATCH_TSDM_ROOT"] = str(tsdm_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "code" / "src"), config["project"]["apn_root"]]
    )
    environment["PYTHONUTF8"] = "1"
    return environment


def test_tsdm_rejects_storage_outside_project() -> None:
    config = load_stage_config()
    completed = subprocess.run(
        [
            config["project"]["python"],
            "-c",
            "from data.dependencies.tsdm.config import BASEDIR; print(BASEDIR)",
        ],
        cwd=config["project"]["apn_root"],
        env=_environment(Path(r"C:\tmp\evipatch_forbidden")),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode != 0
    assert "escapes EviPatch project root" in completed.stderr


def test_p12_collate_applies_shift_only_to_history_and_saves_audit(
    tmp_path: Path,
) -> None:
    config = load_stage_config()
    script = """
import sys
from types import SimpleNamespace
import torch
sys.argv = ['p12_contract_test']
from data.data_provider.datasets import P12
# Simulate a Windows spawned worker whose module-global config was not mutated.
P12.configs.seq_len_max_irr = None
P12.configs.pred_len_max_irr = None
P12.configs.observation_shift = 'none'
runtime_configs = SimpleNamespace(
    seq_len_max_irr=4,
    pred_len_max_irr=2,
    seq_len=36,
    missing_rate=0.0,
    is_training=0,
    observation_shift='mcar',
    shift_rate=0.3,
    shift_seed=2024,
)
x_mark = torch.tensor([0.1, 0.2, 0.3, 0.4])
x = torch.arange(4 * 36, dtype=torch.float32).reshape(4, 36)
y_mark = torch.tensor([0.8, 0.9])
y = torch.arange(2 * 36, dtype=torch.float32).reshape(2, 36)
sample = SimpleNamespace(inputs=(x_mark, x, y_mark), targets=y, key=77)
batch = P12.collate_fn([sample], runtime_configs=runtime_configs)
assert torch.equal(batch['y'][0], y)
assert torch.equal(batch['y_mask'], torch.ones_like(batch['y_mask']))
assert torch.equal(batch['shift_requested'], batch['shift_actual'])
assert torch.equal(batch['shift_actual'], torch.ones(1, 36, dtype=torch.int64))
assert torch.equal((batch['x_mask'] == 0).sum(dim=1), torch.ones(1, 36, dtype=torch.int64))
print('p12-collate-ok')
"""
    completed = subprocess.run(
        [config["project"]["python"], "-c", script],
        cwd=tmp_path,
        env=_environment(tmp_path / "tsdm"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "p12-collate-ok" in completed.stdout
