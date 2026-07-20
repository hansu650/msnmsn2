"""Isolated APN experiment orchestration and provenance capture."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from evipatch.evidence import EVIDENCE_WIDTHS
from evipatch.paths import (
    PROJECT_ROOT,
    assert_isolated_environment_paths,
    assert_within_project,
    ensure_project_dir,
    project_path,
)


DEFAULT_CONFIG = project_path("code", "configs", "stage_a.json")
EXPECTED_APN_COMMIT = "f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4"
CUDA_PEAK_MARKER = "EVIPATCH_CUDA_PEAK_JSON="


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_capture(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_output(repo: Path, *args: str) -> str | None:
    return _run_capture(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args]
    )


def sha256_file(path: Path | str) -> str:
    resolved = assert_within_project(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and strictly validate the centralized experiment matrix."""
    config_path = assert_within_project(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    project = config["project"]
    configured_root = Path(project["root"]).resolve()
    if os.path.normcase(str(configured_root)) != os.path.normcase(str(PROJECT_ROOT)):
        raise ValueError(
            f"Configured project root {configured_root} does not match {PROJECT_ROOT}"
        )
    mutable_paths = {
        key: value
        for key, value in project.items()
        if key.endswith("_root") and key != "root"
    }
    mutable_paths["python"] = project["python"]
    mutable_paths["apn_root"] = project["apn_root"]
    assert_isolated_environment_paths(mutable_paths)

    upstream = config["upstream"]
    if upstream["commit"] != EXPECTED_APN_COMMIT:
        raise ValueError(f"Unexpected APN commit: {upstream['commit']}")
    assert_within_project(upstream["patch"])

    stage = config["stage_a"]
    if stage["variants"] != list(EVIDENCE_WIDTHS):
        raise ValueError(
            f"Stage A variants must be exactly {list(EVIDENCE_WIDTHS)}, "
            f"got {stage['variants']}"
        )
    if stage["seeds"] != [2024, 2025, 2026]:
        raise ValueError("Stage A seeds must be exactly 2024, 2025, and 2026")
    if stage["shifts"] != ["none", "mcar", "burst"]:
        raise ValueError("Stage A shifts must be none, mcar, and burst")
    if stage["hyperparameters"]["optimizer"] != "Adam":
        raise ValueError("Official Stage A optimizer must remain Adam")
    if not stage.get("sequential_gpu_only", False):
        raise ValueError("Stage A must explicitly require sequential single-GPU execution")
    if config["smoke"]["timing_steps"] != 100:
        raise ValueError("Stage A timing gate must use exactly 100 measured steps")
    controlled = config["controlled_support"]
    if controlled["source_variant"] != "apn":
        raise ValueError("Controlled-support pairing must use APN as its source")
    if not (0 < controlled["max_centroid_rms"] <= 1):
        raise ValueError("Controlled-support centroid threshold is invalid")
    if controlled["min_effective_support_ratio"] <= 1:
        raise ValueError("Controlled-support ratio must exceed one")
    if controlled["min_effective_support_difference"] <= 0:
        raise ValueError("Controlled-support difference must be positive")
    if controlled["minimum_pairs_per_seed"] <= 0:
        raise ValueError("Controlled-support minimum pair yield must be positive")

    python_path = Path(project["python"])
    apn_root = Path(project["apn_root"])
    if not python_path.is_file():
        raise FileNotFoundError(f"Configured Python does not exist: {python_path}")
    if not (apn_root / "main.py").is_file():
        raise FileNotFoundError(f"Configured APN checkout is incomplete: {apn_root}")
    actual_commit = _git_output(apn_root, "rev-parse", "HEAD")
    if actual_commit != upstream["commit"]:
        raise RuntimeError(f"APN checkout is at {actual_commit}, expected {upstream['commit']}")
    return config


def _common_apn_arguments(
    config: dict[str, Any],
    variant: str,
    seed: int,
) -> list[str]:
    stage = config["stage_a"]
    hp = stage["hyperparameters"]
    project = config["project"]
    run_root = assert_within_project(
        Path(project["results_root"]) / "stage_a" / variant / str(seed)
    )
    model_id = f"APN_P12_{variant}_seed{seed}"
    arguments: list[str] = [
        "--model_id",
        model_id,
        "--model_name",
        "APN",
        "--dataset_root_path",
        str(Path(project["tsdm_root"]) / "datasets" / "Physionet2012"),
        "--dataset_name",
        stage["dataset_name"],
        "--task_name",
        hp["task_name"],
        "--features",
        hp["features"],
        "--seq_len",
        str(hp["seq_len"]),
        "--label_len",
        str(hp["label_len"]),
        "--pred_len",
        str(hp["pred_len"]),
        "--enc_in",
        str(hp["enc_in"]),
        "--dec_in",
        str(hp["dec_in"]),
        "--c_out",
        str(hp["c_out"]),
        "--loss",
        hp["loss"],
        "--train_epochs",
        str(hp["train_epochs"]),
        "--patience",
        str(hp["patience"]),
        "--val_interval",
        str(hp["val_interval"]),
        "--itr",
        "1",
        "--batch_size",
        str(hp["batch_size"]),
        "--learning_rate",
        str(hp["learning_rate"]),
        "--lr_scheduler",
        hp["lr_scheduler"],
        "--d_model",
        str(hp["d_model"]),
        "--dropout",
        str(hp["dropout"]),
        "--apn_npatch",
        str(hp["apn_npatch"]),
        "--apn_te_dim",
        str(hp["apn_te_dim"]),
        "--num_workers",
        str(hp["num_workers"]),
        "--gpu_id",
        str(stage["gpu_id"]),
        "--use_gpu",
        "1",
        "--use_multi_gpu",
        "0",
        "--wandb",
        "0",
        "--checkpoints",
        str(run_root / "checkpoints"),
        "--evipatch_mode",
        variant,
        "--evipatch_random_seed",
        str(stage["random_feature_seed"]),
        "--seed_base",
        str(seed),
    ]
    return arguments


def build_apn_command(
    config: dict[str, Any],
    variant: str,
    action: str,
    checkpoint: Path | None = None,
    shift: str = "none",
    seed: int | None = None,
) -> list[str]:
    """Build an APN argv list without shell interpolation."""
    stage = config["stage_a"]
    if variant not in stage["variants"]:
        raise ValueError(f"Unknown Stage A variant: {variant}")
    if seed not in stage["seeds"]:
        raise ValueError(f"Seed must be one of {stage['seeds']}, got {seed}")
    if shift not in stage["shifts"]:
        raise ValueError(f"Shift must be one of {stage['shifts']}, got {shift}")
    if action not in {"train", "evaluate", "timing"}:
        raise ValueError("action must be train, evaluate, or timing")

    project = config["project"]
    command = [project["python"], "main.py"]
    command.extend(_common_apn_arguments(config, variant, int(seed)))
    if action == "train":
        if shift != "none" or checkpoint is not None:
            raise ValueError("Training only supports native history and no checkpoint input")
        command.extend(
            [
                "--is_training",
                "1",
                "--skip_test_after_train",
                "1",
                "--observation_shift",
                "none",
                "--shift_rate",
                "0",
                "--shift_seed",
                str(seed),
            ]
        )
    elif action == "timing":
        if shift != "none" or checkpoint is not None:
            raise ValueError("Timing only supports native history and no checkpoint input")
        command.extend(
            [
                "--is_training",
                "0",
                "--test_train_time",
                "1",
                "--load_checkpoints_test",
                "0",
                "--observation_shift",
                "none",
                "--shift_rate",
                "0",
                "--shift_seed",
                str(seed),
            ]
        )
    else:
        if checkpoint is None:
            raise ValueError("Evaluation requires a checkpoint file")
        checkpoint_path = assert_within_project(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        eval_name = "native" if shift == "none" else shift
        command.extend(
            [
                "--is_training",
                "0",
                "--checkpoints_test",
                str(checkpoint_path.parent),
                "--load_checkpoints_test",
                "1",
                "--save_arrays",
                "1",
                "--evipatch_eval_name",
                eval_name,
                "--observation_shift",
                shift,
                "--shift_rate",
                str(0.0 if shift == "none" else stage["shift_rate"]),
                "--shift_seed",
                str(seed),
            ]
        )
    return command


def _query_process_gpu_memory_mib(process_id: int) -> float:
    output = _run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return 0.0
    total = 0.0
    for row in csv.reader(output.splitlines()):
        if len(row) != 2:
            continue
        try:
            pid = int(row[0].strip())
            memory = float(row[1].strip())
        except ValueError:
            continue
        if pid == process_id:
            total += memory
    return total

def _parse_cuda_peak_marker(line: str) -> dict[str, float] | None:
    _, separator, payload = line.partition(CUDA_PEAK_MARKER)
    if not separator:
        return None
    try:
        raw = json.loads(payload.strip())
        parsed = {
            "allocated_mib": float(raw["allocated_mib"]),
            "reserved_mib": float(raw["reserved_mib"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if any(not math.isfinite(value) or value < 0 for value in parsed.values()):
        return None
    return parsed



def run_checked(
    command: list[str],
    log_path: Path | str,
    env: dict[str, str],
    *,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Run one process, stream a local log, and capture wall/peak GPU usage."""
    resolved_log = assert_within_project(log_path)
    ensure_project_dir(resolved_log.parent)
    resolved_cwd = (
        assert_within_project(cwd, allow_root=True)
        if cwd is not None
        else project_path("vendor", "APN")
    )
    executable = assert_within_project(command[0])
    if not executable.is_file():
        raise FileNotFoundError(executable)

    start_wall = time.perf_counter()
    started_at = _utc_now()
    child_cuda_peak: dict[str, float] | None = None
    with resolved_log.open("w", encoding="utf-8", newline="") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=resolved_cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def stream_output() -> None:
            nonlocal child_cuda_peak
            assert process.stdout is not None
            for line in process.stdout:
                parsed_peak = _parse_cuda_peak_marker(line)
                if parsed_peak is not None:
                    child_cuda_peak = parsed_peak
                log_handle.write(line)
                log_handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        output_thread = threading.Thread(target=stream_output, daemon=True)
        output_thread.start()
        peak_gpu_memory_mib = 0.0
        while process.poll() is None:
            peak_gpu_memory_mib = max(
                peak_gpu_memory_mib, _query_process_gpu_memory_mib(process.pid)
            )
            time.sleep(0.2)
        output_thread.join()
        return_code = int(process.returncode)

    if child_cuda_peak is not None:
        peak_cuda_allocated_mib = child_cuda_peak["allocated_mib"]
        peak_cuda_reserved_mib = child_cuda_peak["reserved_mib"]
        peak_gpu_memory_mib = peak_cuda_allocated_mib
        gpu_memory_source = "torch.cuda"
    else:
        peak_cuda_allocated_mib = None
        peak_cuda_reserved_mib = None
        gpu_memory_source = (
            "nvidia-smi-process" if peak_gpu_memory_mib > 0 else "unavailable"
        )

    result = {
        "argv": command,
        "cwd": str(resolved_cwd),
        "log_path": str(resolved_log),
        "started_at": started_at,
        "ended_at": _utc_now(),
        "wall_seconds": time.perf_counter() - start_wall,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "peak_cuda_allocated_mib": peak_cuda_allocated_mib,
        "peak_cuda_reserved_mib": peak_cuda_reserved_mib,
        "gpu_memory_source": gpu_memory_source,
        "return_code": return_code,
    }
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return result


def discover_checkpoints(result_root: Path | str, variant: str) -> list[Path]:
    """Discover one checkpoint for each completed seed, sorted by seed path."""
    root = assert_within_project(result_root)
    checkpoints = [
        path
        for path in root.glob(f"{variant}/*/checkpoints/**/pytorch_model.bin")
        if path.is_file()
    ]
    def checkpoint_seed(path: Path) -> int:
        relative = path.relative_to(root)
        if len(relative.parts) < 2 or relative.parts[0] != variant:
            raise RuntimeError(f"Unexpected checkpoint path layout: {path}")
        return int(relative.parts[1])

    return sorted(checkpoints, key=lambda path: (checkpoint_seed(path), str(path)))


def _discover_one_checkpoint(run_root: Path) -> Path:
    checkpoints = list((run_root / "checkpoints").glob("**/pytorch_model.bin"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint below {run_root}, found {len(checkpoints)}"
        )
    return assert_within_project(checkpoints[0])


def _count_checkpoint_parameters(checkpoint: Path) -> int:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    excluded_suffixes = ("patch_pos_enc.pe", "random_feature_table")
    return int(
        sum(
            tensor.numel()
            for key, tensor in state.items()
            if not key.endswith(excluded_suffixes)
        )
    )


def _environment_for_run(config: dict[str, Any], seed: int) -> dict[str, str]:
    project = config["project"]
    environment = os.environ.copy()
    code_src = str(project_path("code", "src"))
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        code_src if not old_pythonpath else os.pathsep.join([code_src, old_pythonpath])
    )
    environment["EVIPATCH_TSDM_ROOT"] = str(
        assert_within_project(project["tsdm_root"])
    )
    environment["EVIPATCH_PROJECT_ROOT"] = str(PROJECT_ROOT)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["PYTHONUTF8"] = "1"
    return environment


def collect_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Collect reproducibility metadata without writing outside the project."""
    project = config["project"]
    python = project["python"]
    apn_root = Path(project["apn_root"])
    pip_freeze = _run_capture([python, "-m", "pip", "freeze"])
    torch_info_text = _run_capture(
        [
            python,
            "-c",
            (
                "import json,torch; print(json.dumps({"
                "'torch':torch.__version__,"
                "'cuda_runtime':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),"
                "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
                "'capability':torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None"
                "}))"
            ),
        ]
    )
    patch_path = Path(config["upstream"]["patch"])
    root_head = _git_output(PROJECT_ROOT, "rev-parse", "HEAD")
    root_status = _git_output(PROJECT_ROOT, "status", "--short")
    apn_status = _git_output(apn_root, "status", "--short")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pip_freeze": [] if not pip_freeze else pip_freeze.splitlines(),
        "torch_cuda_gpu": None if not torch_info_text else json.loads(torch_info_text),
        "nvidia_smi": _run_capture(["nvidia-smi"]),
        "project_git_commit": root_head,
        "project_git_status": root_status,
        "apn_commit": _git_output(apn_root, "rev-parse", "HEAD"),
        "apn_git_status": apn_status,
        "apn_patch_sha256": sha256_file(patch_path) if patch_path.is_file() else None,
    }


def write_run_manifest(
    path: Path | str,
    *,
    config: dict[str, Any],
    variant: str,
    seed: int,
    shift: str,
    action: str,
    process_result: dict[str, Any],
    checkpoint: Path | None = None,
    artifacts: list[Path] | None = None,
) -> Path:
    """Write a complete run manifest and hashes for selected outputs."""
    manifest_path = assert_within_project(path)
    ensure_project_dir(manifest_path.parent)
    artifact_records = []
    for artifact in sorted(artifacts or []):
        resolved = assert_within_project(artifact)
        if resolved.is_file():
            artifact_records.append(
                {
                    "path": str(resolved.relative_to(PROJECT_ROOT)),
                    "bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
    manifest = {
        "schema_version": 1,
        "project_root": str(PROJECT_ROOT),
        "variant": variant,
        "seed": seed,
        "shift": shift,
        "action": action,
        "process": process_result,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": (
            None if checkpoint is None or not checkpoint.is_file() else sha256_file(checkpoint)
        ),
        "parameter_count": (
            None if checkpoint is None or not checkpoint.is_file()
            else _count_checkpoint_parameters(checkpoint)
        ),
        "provenance": collect_provenance(config),
        "artifacts": artifact_records,
        "written_at": _utc_now(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def _evaluation_directory(checkpoint: Path, shift: str) -> Path:
    name = "native" if shift == "none" else shift
    return assert_within_project(checkpoint.parent / f"eval_{name}")


def _run_root(config: dict[str, Any], variant: str, seed: int) -> Path:
    return assert_within_project(
        Path(config["project"]["results_root"]) / "stage_a" / variant / str(seed)
    )


def train_one(
    config: dict[str, Any],
    variant: str,
    seed: int,
    *,
    resume: bool = True,
) -> Path:
    run_root = _run_root(config, variant, seed)
    manifest_path = run_root / "train_manifest.json"
    if resume and manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(recorded["checkpoint"])
        if checkpoint.is_file() and recorded["process"]["return_code"] == 0:
            return assert_within_project(checkpoint)

    ensure_project_dir(run_root)
    command = build_apn_command(config, variant, "train", seed=seed)
    process_result = run_checked(
        command,
        Path(config["project"]["logs_root"])
        / "stage_a"
        / variant
        / str(seed)
        / "train.log",
        _environment_for_run(config, seed),
        cwd=config["project"]["apn_root"],
    )
    checkpoint = _discover_one_checkpoint(run_root)
    write_run_manifest(
        manifest_path,
        config=config,
        variant=variant,
        seed=seed,
        shift="none",
        action="train",
        process_result=process_result,
        checkpoint=checkpoint,
        artifacts=[checkpoint.parent / "configs.yaml"],
    )
    return checkpoint


def evaluate_one(
    config: dict[str, Any],
    variant: str,
    seed: int,
    shift: str,
    checkpoint: Path,
    *,
    resume: bool = True,
) -> Path:
    eval_dir = _evaluation_directory(checkpoint, shift)
    manifest_path = eval_dir / "run_manifest.json"
    if resume and manifest_path.is_file() and (eval_dir / "metric.json").is_file():
        return eval_dir

    command = build_apn_command(
        config,
        variant,
        "evaluate",
        checkpoint=checkpoint,
        shift=shift,
        seed=seed,
    )
    process_result = run_checked(
        command,
        Path(config["project"]["logs_root"])
        / "stage_a"
        / variant
        / str(seed)
        / f"evaluate_{shift}.log",
        _environment_for_run(config, seed),
        cwd=config["project"]["apn_root"],
    )
    required_names = [
        "metric.json",
        "input_x.npy",
        "input_x_mark.npy",
        "input_x_mask.npy",
        "input_y.npy",
        "input_y_mask.npy",
        "input_sample_ID.npy",
        "output_pred.npy",
        "input_shift_requested.npy",
        "input_shift_actual.npy",
        "input_shift_original_observed.npy",
        "input_shift_remaining_observed.npy",
    ]
    required = [eval_dir / name for name in required_names]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Evaluation {eval_dir} is missing required outputs: {missing}")
    write_run_manifest(
        manifest_path,
        config=config,
        variant=variant,
        seed=seed,
        shift=shift,
        action="evaluate",
        process_result=process_result,
        checkpoint=checkpoint,
        artifacts=required,
    )
    return eval_dir


@contextlib.contextmanager
def stage_lock(config: dict[str, Any]) -> Iterator[None]:
    """Prevent two Stage A dispatchers from sharing the single GPU."""
    lock_path = assert_within_project(
        Path(config["project"]["results_root"]) / "stage_a" / ".stage_a.lock"
    )
    ensure_project_dir(lock_path.parent)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Stage A lock already exists: {lock_path}. "
            "Confirm no run is active before removing a stale lock."
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\nstarted={_utc_now()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def run_stage_a(config: dict[str, Any]) -> None:
    """Run all 21 trainings and 63 evaluations in a strict serial order."""
    with stage_lock(config):
        for variant in config["stage_a"]["variants"]:
            for seed in config["stage_a"]["seeds"]:
                checkpoint = train_one(config, variant, seed)
                for shift in config["stage_a"]["shifts"]:
                    evaluate_one(config, variant, seed, shift, checkpoint)


def run_smoke(config: dict[str, Any]) -> None:
    """Run the project test suite; timing scripts are invoked by run_smoke.ps1."""
    ensure_project_dir(config["project"]["results_root"])
    pytest_temp = project_path("results", "pytest_tmp")
    command = [
        config["project"]["python"],
        "-m",
        "pytest",
        str(project_path("code", "tests")),
        "-q",
        "--basetemp",
        str(pytest_temp),
        "-p",
        "no:cacheprovider",
    ]
    run_checked(
        command,
        Path(config["project"]["logs_root"]) / "smoke" / "pytest.log",
        _environment_for_run(config, config["stage_a"]["seeds"][0]),
        cwd=PROJECT_ROOT,
    )


def _parse_training_timing(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Average training step time .*?:\s*([0-9.]+) ms.*?([0-9.]+) ms",
        text,
    )
    if match is None:
        raise RuntimeError(f"Could not parse 100-step timing from {log_path}")
    return float(match.group(1)), float(match.group(2))


def run_timing_matrix(config: dict[str, Any]) -> Path:
    """Benchmark exactly 100 measured real-P12 optimizer steps per variant."""
    artifacts = ensure_project_dir(config["project"]["artifacts_root"])
    output = artifacts / "timing_100_steps.json"
    records: dict[str, Any] = {}
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        records = {record["variant"]: record for record in existing.get("records", [])}

    with stage_lock(config):
        for variant in config["stage_a"]["variants"]:
            if variant in records:
                continue
            seed = config["stage_a"]["seeds"][0]
            log_path = (
                Path(config["project"]["logs_root"])
                / "smoke"
                / f"timing_{variant}.log"
            )
            process_result = run_checked(
                build_apn_command(config, variant, "timing", seed=seed),
                log_path,
                _environment_for_run(config, seed),
                cwd=config["project"]["apn_root"],
            )
            mean_ms, std_ms = _parse_training_timing(assert_within_project(log_path))
            records[variant] = {
                "variant": variant,
                "seed": seed,
                "warmup_steps": config["smoke"]["warmup_steps"],
                "measured_steps": config["smoke"]["timing_steps"],
                "training_step_mean_ms": mean_ms,
                "training_step_std_ms": std_ms,
                "process": process_result,
            }
            output.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "records": [
                            records[name]
                            for name in config["stage_a"]["variants"]
                            if name in records
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
    if len(records) != len(config["stage_a"]["variants"]):
        raise RuntimeError("The seven-variant timing matrix is incomplete")
    return output



def _status(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for variant in config["stage_a"]["variants"]:
        for seed in config["stage_a"]["seeds"]:
            root = _run_root(config, variant, seed)
            checkpoints = list((root / "checkpoints").glob("**/pytorch_model.bin"))
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint": len(checkpoints) == 1,
                    "native": any(root.glob("checkpoints/**/eval_native/metric.json")),
                    "mcar": any(root.glob("checkpoints/**/eval_mcar/metric.json")),
                    "burst": any(root.glob("checkpoints/**/eval_burst/metric.json")),
                }
            )
    return {"rows": rows, "complete": all(all(row[k] for k in ("checkpoint", "native", "mcar", "burst")) for row in rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "validate",
            "smoke",
            "timing",
            "train-variant",
            "evaluate-variant",
            "stage-a",
            "controlled",
            "aggregate",
            "package",
            "status",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=list(EVIDENCE_WIDTHS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shift", choices=["none", "mcar", "burst"], default="none")
    args = parser.parse_args(argv)
    config = load_stage_config(args.config)

    if args.action == "validate":
        print(json.dumps({"valid": True, "project_root": str(PROJECT_ROOT)}, indent=2))
    elif args.action == "smoke":
        run_smoke(config)
    elif args.action == "timing":
        print(run_timing_matrix(config))
    elif args.action == "train-variant":
        if args.variant is None or args.seed is None:
            parser.error("train-variant requires --variant and --seed")
        print(train_one(config, args.variant, args.seed))
    elif args.action == "evaluate-variant":
        if args.variant is None or args.seed is None:
            parser.error("evaluate-variant requires --variant and --seed")
        checkpoint = _discover_one_checkpoint(_run_root(config, args.variant, args.seed))
        print(evaluate_one(config, args.variant, args.seed, args.shift, checkpoint))
    elif args.action == "stage-a":
        run_stage_a(config)
    elif args.action == "controlled":
        from evipatch.controlled import run_controlled_support

        print(json.dumps(run_controlled_support(config), indent=2, ensure_ascii=False))
    elif args.action == "aggregate":
        from evipatch.aggregate import run_aggregation

        run_aggregation(config)
    elif args.action == "package":
        from evipatch.package import package_project

        print(package_project(config))
    elif args.action == "status":
        print(json.dumps(_status(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
