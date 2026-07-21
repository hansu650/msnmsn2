"""Train/validation-only APN backbone campaign control plane.

The module itself is intentionally lazy: asking for ``--help`` or a read-only
plan does not import APN, pandas, or a dataset module.  The safe script wrapper
sets the deterministic CUDA environment before importing the package.  Real
work starts only after ``--execute``.  Every runnable cell is restricted to
train and validation loaders; there is no test-loader argument or callback in
this control plane.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

from .config import DEFAULT_CONFIG, ResolvedConfig, canonical_sha256, load_resolved_config
from .paths import PROJECT_ROOT, require_within_root


CUBLAS_WORKSPACE_VALUE = ":4096:8"
DEFAULT_SEEDS = (2024, 2025, 2026, 2027, 2028)
DEFAULT_DATASETS = ("P12", "USHCN", "HumanActivity")
CELL_PROTOCOLS: Mapping[str, tuple[str, ...]] = {
    "P12": ("release_parity", "strict_p12"),
    "USHCN": ("release_parity", "strict_ushcn"),
    "HumanActivity": ("release_parity",),
    "MIMIC_III": ("release_parity",),
}
HUMAN_ACTIVITY_PADDED_PREDICTION_STEPS = 11
CONTROL_MANIFEST = "backbone_campaign_manifest.json"
TRAIN_MANIFEST = "train_manifest.json"
LEGACY_IMPORT_MANIFEST = "verified_legacy_import.json"
CHECKPOINT = "pytorch_model.bin"
VENDOR_CONFIG = "configs.yaml"
LEGACY_PROJECT_COMMIT = "0bf46fc9d6cb00f70ffe110df8d48d4c3a592037"
LEGACY_APN_COMMIT = "f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4"
LEGACY_APN_PATCH_SHA256 = (
    "00d8d59221d1580ee2b718365325bd69945dc2c103b0c23d7f93f9365e301746"
)
LEGACY_P12_RELEASE_IDENTITIES: Mapping[int, Mapping[str, Any]] = {
    2024: {
        "checkpoint_sha256": "927b3e540b7450d37897fd1a78dbbbaa1ff93152d1456c350d22f9f1cc41e3d4",
        "checkpoint_bytes": 36083,
        "configs_sha256": "f7b1ab2e975e155364fef9b92c8e75b1da0b86246460b9f5cf02170cb98a0b17",
        "configs_bytes": 3289,
        "train_manifest_sha256": "6b08d14a7297c72cc35b1248992b88b521c10dcb420c55b3bc8be82a298cd7cc",
        "train_manifest_bytes": 8199,
        "log_sha256": "71502029c81a7b01fd83e4df0edc32e353c84a8357729dd8e527c5b4c864a145",
        "log_bytes": 732065,
    },
    2025: {
        "checkpoint_sha256": "945ef93c856c73ef59a39d6c27baaf7599cbebff9b42b55b45381ac255ed6bb9",
        "checkpoint_bytes": 36083,
        "configs_sha256": "83d2deda21aaa0e4797018496759891ca8f25a4950a7466d39e139cc472b9cb6",
        "configs_bytes": 3289,
        "train_manifest_sha256": "f76cd5ddcba5b91b6445432d5f8b4e0e3def31f5adfd7e4218b66423f2ebe249",
        "train_manifest_bytes": 8199,
        "log_sha256": "1de79ae763acf68055182e45e7f2a55e6e0a721665ae0d19fff1a5b884792e4d",
        "log_bytes": 1063553,
    },
    2026: {
        "checkpoint_sha256": "36303d6b1f2fa449e35e93bb9bfcc7446641a78580367a97c9c45847117ac641",
        "checkpoint_bytes": 36083,
        "configs_sha256": "c0abb407632c6e67fabf85b80d8f586dc52be698c181667a3483b3f41c95c06e",
        "configs_bytes": 3289,
        "train_manifest_sha256": "e5391d0704c8c121432fb167f53d0a5f1720bc319a425663f2a8d0b137971852",
        "train_manifest_bytes": 8200,
        "log_sha256": "d0b15dda8fc81b57dcb8e838a6c62f86d010bd5260e52d8f59ef5887b02823bd",
        "log_bytes": 1230939,
    },
}


class BackboneCampaignError(RuntimeError):
    """A backbone campaign cell is unsafe, stale, or internally inconsistent."""


class BackboneBlockedError(BackboneCampaignError):
    """A declared external blocker prevents a cell from running."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"BLOCKED[{self.code}]: {self.detail}")


@dataclass(frozen=True)
class BackboneCell:
    dataset_id: str
    protocol_id: str

    @property
    def key(self) -> str:
        return f"{self.dataset_id}:{self.protocol_id}"


@dataclass(frozen=True)
class PreparedCell:
    cell: BackboneCell
    protocol_manifest: Path
    split_manifest: Path
    normalizer_manifest: Path
    strict_prepared: object | None = None


@dataclass(frozen=True)
class BackboneResult:
    dataset_id: str
    protocol_id: str
    seed: int
    outcome: str
    checkpoint_sha256: str | None
    manifest_path: Path | None
    detail: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "seed": self.seed,
            "outcome": self.outcome,
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_path": (
                None
                if self.manifest_path is None
                else _relative(self.manifest_path)
            ),
            "detail": self.detail,
        }


def ensure_cublas_workspace_config(
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    """Set and audit the exact deterministic CUDA workspace contract."""

    target = os.environ if environ is None else environ
    previous = target.get("CUBLAS_WORKSPACE_CONFIG")
    if previous is None:
        target["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_VALUE
        changed = True
    elif previous != CUBLAS_WORKSPACE_VALUE:
        raise BackboneCampaignError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{CUBLAS_WORKSPACE_VALUE!r}; observed {previous!r}"
        )
    else:
        changed = False
    observed = target.get("CUBLAS_WORKSPACE_CONFIG")
    if observed != CUBLAS_WORKSPACE_VALUE:
        raise BackboneCampaignError("Failed to establish deterministic cuBLAS workspace")
    return {
        "required": CUBLAS_WORKSPACE_VALUE,
        "observed": observed,
        "set_by_control_plane": changed,
    }


def _relative(path: str | Path) -> str:
    resolved = require_within_root(path, must_exist=True)
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: str | Path) -> str:
    source = require_within_root(path, must_exist=True)
    if not source.is_file():
        raise BackboneCampaignError(f"Expected a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackboneCampaignError(f"Cannot read JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise BackboneCampaignError(f"JSON artifact must be an object: {source}")
    return value


def _atomic_write_text(path: str | Path, text: str) -> Path:
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return _atomic_write_text(
        path,
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
    )


def _yaml_ready(value: Any, *, location: str = "config") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _yaml_ready(child, location=f"{location}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _yaml_ready(child, location=f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise BackboneCampaignError(
        f"Vendor config contains a non-serializable {type(value).__name__} at {location}"
    )


def vendor_config_snapshot(config: Any) -> dict[str, Any]:
    """Return the exact YAML-safe vendor config after loader construction."""

    if isinstance(config, Mapping):
        raw = dict(config)
    elif hasattr(config, "__dict__"):
        raw = dict(vars(config))
    else:
        raise BackboneCampaignError("Vendor config has no auditable mapping state")
    snapshot = _yaml_ready(raw)
    for key, value in snapshot.items():
        if not isinstance(value, str) or not value:
            continue
        name = key.lower()
        if not any(token in name for token in ("path", "root", "checkpoint")):
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            require_within_root(candidate)
    return snapshot


def write_or_validate_vendor_config(
    path: str | Path,
    snapshot: Mapping[str, Any],
    *,
    create_if_missing: bool,
) -> Path:
    """Write one exact configs.yaml, or reject any existing drift."""

    import yaml

    destination = require_within_root(path)
    expected = dict(snapshot)
    if destination.exists():
        try:
            current = yaml.safe_load(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise BackboneCampaignError(f"Cannot read existing vendor config: {exc}") from exc
        if current != expected:
            raise BackboneCampaignError(
                f"Existing configs.yaml differs from the reconstructed train/val config: {destination}"
            )
        return destination
    if not create_if_missing:
        raise BackboneCampaignError(f"Reusable checkpoint is missing configs.yaml: {destination}")
    payload = yaml.safe_dump(expected, sort_keys=True, allow_unicode=True)
    return _atomic_write_text(destination, payload)


def _artifact(path: str | Path) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    return {
        "path": _relative(source),
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _argv_value(argv: Sequence[Any], flag: str) -> str:
    values = [str(value) for value in argv]
    positions = [index for index, value in enumerate(values) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        raise BackboneCampaignError(f"Training argv must contain exactly one {flag}")
    return values[positions[0] + 1]


def validate_reusable_checkpoint(
    *,
    config: ResolvedConfig,
    cell: BackboneCell,
    seed: int,
    checkpoint_path: str | Path,
    train_manifest_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless checkpoint bytes and their training manifest agree."""

    checkpoint = require_within_root(checkpoint_path, must_exist=True)
    manifest_path = require_within_root(train_manifest_path, must_exist=True)
    manifest = _load_json(manifest_path)
    required = {
        "schema_version": 1,
        "status": "complete",
        "action": "train_apn_backbone",
        "seed": seed,
        "resolved_config_sha256": config.sha256,
    }
    mismatches = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    snapshot = manifest.get("resolved_config")
    if not isinstance(snapshot, Mapping) or canonical_sha256(snapshot) != config.sha256:
        mismatches["resolved_config"] = "content hash differs from the active frozen config"
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        mismatches["checkpoint"] = "missing checkpoint record"
    else:
        expected_relative = checkpoint.relative_to(PROJECT_ROOT).as_posix()
        checks = {
            "path": expected_relative,
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "weights_only": True,
            "atomic_replace": True,
        }
        for key, expected in checks.items():
            if checkpoint_record.get(key) != expected:
                mismatches[f"checkpoint.{key}"] = {
                    "expected": expected,
                    "observed": checkpoint_record.get(key),
                }
    policy = manifest.get("training_policy")
    expected_policy = {
        "optimizer": "Adam",
        "loss": "MSE",
        "epochs": int(config["training"]["epochs"]),
        "patience": int(config["training"]["patience"]),
        "validation_interval": int(config["training"]["validation_interval"]),
        "learning_rate": float(
            config["datasets"][cell.dataset_id]["apn_hyperparameters"]["learning_rate"]
        ),
    }
    if not isinstance(policy, Mapping):
        mismatches["training_policy"] = "missing"
    else:
        for key, expected in expected_policy.items():
            if policy.get(key) != expected:
                mismatches[f"training_policy.{key}"] = {
                    "expected": expected,
                    "observed": policy.get(key),
                }
    argv = manifest.get("argv")
    try:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            raise BackboneCampaignError("Training manifest argv is not a sequence")
        argv_identity = {
            "dataset": _argv_value(argv, "--dataset"),
            "protocol": _argv_value(argv, "--protocol"),
            "seed": int(_argv_value(argv, "--seed")),
        }
        expected_identity = {
            "dataset": cell.dataset_id,
            "protocol": cell.protocol_id,
            "seed": seed,
        }
        if argv_identity != expected_identity:
            mismatches["argv_identity"] = {
                "expected": expected_identity,
                "observed": argv_identity,
            }
    except (BackboneCampaignError, TypeError, ValueError) as exc:
        mismatches["argv"] = str(exc)
    if mismatches:
        raise BackboneCampaignError(
            "Reusable checkpoint provenance mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return manifest




def _write_or_validate_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create deterministic proof metadata once, or reject any later drift."""

    destination = require_within_root(path)
    expected = dict(payload)
    if destination.exists():
        if _load_json(destination) != expected:
            raise BackboneCampaignError(f"Existing proof manifest drifted: {destination}")
        return destination
    return _atomic_write_json(destination, expected)


def _legacy_stage_a_root(seed: int) -> Path:
    return require_within_root(PROJECT_ROOT / "results" / "stage_a" / "apn" / str(seed))


def _legacy_stage_a_manifest_path(seed: int) -> Path:
    return require_within_root(_legacy_stage_a_root(seed) / TRAIN_MANIFEST)


def _legacy_stage_a_log_path(seed: int) -> Path:
    return require_within_root(
        PROJECT_ROOT / "logs" / "stage_a" / "apn" / str(seed) / "train.log"
    )


def _expected_legacy_p12_flags(
    config: ResolvedConfig, seed: int
) -> dict[str, str]:
    task = config["datasets"]["P12"]["task"]
    hyper = config["datasets"]["P12"]["apn_hyperparameters"]
    channels = int(task["channels"])
    return {
        "--model_id": f"APN_P12_apn_seed{seed}",
        "--model_name": "APN",
        "--dataset_root_path": str(
            PROJECT_ROOT / "data" / "tsdm" / "datasets" / "Physionet2012"
        ),
        "--dataset_name": "P12",
        "--task_name": "short_term_forecast",
        "--features": str(config["training"]["features"]),
        "--seq_len": str(int(task["history"])),
        "--label_len": "0",
        "--pred_len": str(int(task["prediction_steps"])),
        "--enc_in": str(channels),
        "--dec_in": str(channels),
        "--c_out": str(channels),
        "--loss": str(config["training"]["loss"]),
        "--train_epochs": str(int(config["training"]["epochs"])),
        "--patience": str(int(config["training"]["patience"])),
        "--val_interval": str(int(config["training"]["validation_interval"])),
        "--itr": "1",
        "--batch_size": str(int(hyper["batch_size"])),
        "--learning_rate": str(float(hyper["learning_rate"])),
        "--lr_scheduler": "DelayedStepDecayLR",
        "--d_model": str(int(hyper["d_model"])),
        "--dropout": str(float(hyper["dropout"])),
        "--apn_npatch": str(int(hyper["npatch"])),
        "--apn_te_dim": str(int(hyper["te_dim"])),
        "--num_workers": "0",
        "--gpu_id": "0",
        "--use_gpu": "1",
        "--use_multi_gpu": "0",
        "--wandb": "0",
        "--checkpoints": str(_legacy_stage_a_root(seed) / "checkpoints"),
        "--evipatch_mode": "apn",
        "--evipatch_random_seed": "1729",
        "--seed_base": str(seed),
        "--is_training": "1",
        "--skip_test_after_train": "1",
        "--observation_shift": "none",
        "--shift_rate": "0",
        "--shift_seed": str(seed),
    }


def _validate_pinned_artifact(
    *,
    label: str,
    path: str | Path,
    identity: Mapping[str, Any],
    sha_key: str,
    bytes_key: str,
) -> dict[str, Any]:
    artifact = _artifact(path)
    expected = {
        "sha256": str(identity[sha_key]),
        "bytes": int(identity[bytes_key]),
    }
    observed = {key: artifact[key] for key in expected}
    if observed != expected:
        raise BackboneCampaignError(
            f"Legacy {label} identity mismatch: "
            + json.dumps(
                {"expected": expected, "observed": observed},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return artifact


def validate_legacy_p12_release_import(
    *,
    config: ResolvedConfig,
    cell: BackboneCell,
    seed: int,
    checkpoint_path: str | Path,
    configs_yaml_path: str | Path,
) -> dict[str, Any]:
    """Verify and describe one pinned pre-control-plane P12 release checkpoint."""

    if cell != BackboneCell("P12", "release_parity"):
        raise BackboneCampaignError("Legacy import is restricted to P12 release_parity")
    identity = LEGACY_P12_RELEASE_IDENTITIES.get(seed)
    if identity is None:
        raise BackboneCampaignError(f"No pinned legacy identity exists for seed {seed}")
    if config["apn"]["commit"] != LEGACY_APN_COMMIT:
        raise BackboneCampaignError("Active APN commit is incompatible with legacy evidence")

    target_checkpoint = require_within_root(checkpoint_path, must_exist=True)
    target_configs = require_within_root(configs_yaml_path, must_exist=True)
    target_checkpoint_artifact = _validate_pinned_artifact(
        label="target checkpoint",
        path=target_checkpoint,
        identity=identity,
        sha_key="checkpoint_sha256",
        bytes_key="checkpoint_bytes",
    )
    target_configs_artifact = _validate_pinned_artifact(
        label="target configs",
        path=target_configs,
        identity=identity,
        sha_key="configs_sha256",
        bytes_key="configs_bytes",
    )

    legacy_manifest_path = _legacy_stage_a_manifest_path(seed)
    legacy_manifest_artifact = _validate_pinned_artifact(
        label="train manifest",
        path=legacy_manifest_path,
        identity=identity,
        sha_key="train_manifest_sha256",
        bytes_key="train_manifest_bytes",
    )
    legacy = _load_json(legacy_manifest_path)
    expected_top = {
        "schema_version": 1,
        "variant": "apn",
        "seed": seed,
        "shift": "none",
        "action": "train",
        "parameter_count": 6701,
    }
    mismatches: dict[str, Any] = {
        key: {"expected": expected, "observed": legacy.get(key)}
        for key, expected in expected_top.items()
        if legacy.get(key) != expected
    }

    source_checkpoint_raw = legacy.get("checkpoint")
    if not isinstance(source_checkpoint_raw, str):
        mismatches["checkpoint"] = "missing source checkpoint path"
        source_checkpoint = None
    else:
        try:
            source_checkpoint = require_within_root(
                source_checkpoint_raw, must_exist=True
            )
        except (FileNotFoundError, ValueError) as exc:
            mismatches["checkpoint"] = str(exc)
            source_checkpoint = None
    if source_checkpoint is not None:
        expected_root = require_within_root(
            _legacy_stage_a_root(seed) / "checkpoints", must_exist=True
        )
        try:
            source_checkpoint.relative_to(expected_root)
        except ValueError:
            mismatches["checkpoint.path"] = "source checkpoint escaped legacy seed root"
        if legacy.get("checkpoint_sha256") != identity["checkpoint_sha256"]:
            mismatches["checkpoint_sha256"] = {
                "expected": identity["checkpoint_sha256"],
                "observed": legacy.get("checkpoint_sha256"),
            }
        source_checkpoint_artifact = _validate_pinned_artifact(
            label="source checkpoint",
            path=source_checkpoint,
            identity=identity,
            sha_key="checkpoint_sha256",
            bytes_key="checkpoint_bytes",
        )
        source_configs = require_within_root(
            source_checkpoint.parent / VENDOR_CONFIG, must_exist=True
        )
        source_configs_artifact = _validate_pinned_artifact(
            label="source configs",
            path=source_configs,
            identity=identity,
            sha_key="configs_sha256",
            bytes_key="configs_bytes",
        )
        if source_checkpoint.read_bytes() != target_checkpoint.read_bytes():
            mismatches["checkpoint_copy"] = "target bytes differ from legacy source"
        if source_configs.read_bytes() != target_configs.read_bytes():
            mismatches["configs_copy"] = "target bytes differ from legacy source"
    else:
        source_checkpoint_artifact = {}
        source_configs_artifact = {}
        source_configs = None

    process = legacy.get("process")
    if not isinstance(process, Mapping):
        mismatches["process"] = "missing process record"
        process = {}
    if process.get("return_code") != 0:
        mismatches["process.return_code"] = process.get("return_code")
    wall_seconds = process.get("wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or float(wall_seconds) <= 0
    ):
        mismatches["process.wall_seconds"] = wall_seconds
    if process.get("cwd") != str(PROJECT_ROOT / "vendor" / "APN"):
        mismatches["process.cwd"] = process.get("cwd")

    expected_log_path = _legacy_stage_a_log_path(seed)
    try:
        observed_log_path = require_within_root(
            str(process.get("log_path")), must_exist=True
        )
    except (FileNotFoundError, ValueError) as exc:
        mismatches["process.log_path"] = str(exc)
        observed_log_path = None
    if (
        observed_log_path is not None
        and observed_log_path.resolve() != expected_log_path.resolve()
    ):
        mismatches["process.log_path"] = {
            "expected": str(expected_log_path),
            "observed": str(observed_log_path),
        }
    log_artifact = _validate_pinned_artifact(
        label="training log",
        path=expected_log_path,
        identity=identity,
        sha_key="log_sha256",
        bytes_key="log_bytes",
    )

    argv = process.get("argv")
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        mismatches["process.argv"] = "missing argv sequence"
        argv = []
    else:
        expected_flags = _expected_legacy_p12_flags(config, seed)
        for flag, expected in expected_flags.items():
            try:
                observed = _argv_value(argv, flag)
            except BackboneCampaignError as exc:
                mismatches[f"process.argv.{flag}"] = str(exc)
                continue
            if observed != expected:
                mismatches[f"process.argv.{flag}"] = {
                    "expected": expected,
                    "observed": observed,
                }

    provenance = legacy.get("provenance")
    if not isinstance(provenance, Mapping):
        mismatches["provenance"] = "missing provenance"
        provenance = {}
    expected_provenance = {
        "project_git_commit": LEGACY_PROJECT_COMMIT,
        "apn_commit": LEGACY_APN_COMMIT,
        "apn_patch_sha256": LEGACY_APN_PATCH_SHA256,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            mismatches[f"provenance.{key}"] = {
                "expected": expected,
                "observed": provenance.get(key),
            }
    if not str(provenance.get("python", "")).startswith("3.11.13"):
        mismatches["provenance.python"] = provenance.get("python")
    torch_gpu = provenance.get("torch_cuda_gpu")
    if not isinstance(torch_gpu, Mapping):
        mismatches["provenance.torch_cuda_gpu"] = "missing"
        torch_gpu = {}
    expected_runtime = {
        "torch": "2.6.0+cu124",
        "cuda_runtime": "12.4",
        "cuda_available": True,
        "gpu": "NVIDIA GeForce RTX 4090",
    }
    for key, expected in expected_runtime.items():
        if torch_gpu.get(key) != expected:
            mismatches[f"provenance.torch_cuda_gpu.{key}"] = {
                "expected": expected,
                "observed": torch_gpu.get(key),
            }

    config_records = legacy.get("artifacts")
    if source_configs is not None:
        expected_config_record = {
            "path": source_configs.relative_to(PROJECT_ROOT).as_posix().replace(
                "/", "\\"
            ),
            "bytes": source_configs_artifact["bytes"],
            "sha256": source_configs_artifact["sha256"],
        }
        if (
            not isinstance(config_records, list)
            or len(config_records) != 1
            or config_records[0] != expected_config_record
        ):
            mismatches["artifacts"] = {
                "expected": [expected_config_record],
                "observed": config_records,
            }

    if mismatches:
        raise BackboneCampaignError(
            "Legacy P12 release provenance mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    provenance_summary = {
        key: provenance[key]
        for key in (
            "python",
            "project_git_commit",
            "apn_commit",
            "apn_patch_sha256",
        )
    }
    provenance_summary["torch_cuda_gpu"] = dict(torch_gpu)
    return {
        "schema_version": "edgetwincal.verified_legacy_import.v1",
        "status": "complete",
        "outcome": "verified_legacy_import",
        "dataset_id": cell.dataset_id,
        "protocol_id": cell.protocol_id,
        "seed": seed,
        "resolved_config_sha256": config.sha256,
        "test_constructed": False,
        "loaders_constructed": False,
        "new_train_manifest": {
            "status": "absent_by_design",
            "path": (
                target_checkpoint.parent / TRAIN_MANIFEST
            ).relative_to(PROJECT_ROOT).as_posix(),
        },
        "validation_checks": [
            "pinned_checkpoint_sha256_and_size",
            "source_target_checkpoint_byte_identity",
            "pinned_configs_sha256_and_size",
            "source_target_configs_byte_identity",
            "pinned_legacy_train_manifest_sha256_and_size",
            "pinned_training_log_sha256_and_size",
            "official_train_only_argv",
            "successful_process_record",
            "project_apn_patch_and_runtime_provenance",
        ],
        "target_artifacts": {
            "checkpoint": target_checkpoint_artifact,
            "configs_yaml": target_configs_artifact,
        },
        "source_artifacts": {
            "train_manifest": legacy_manifest_artifact,
            "checkpoint": source_checkpoint_artifact,
            "configs_yaml": source_configs_artifact,
            "log": log_artifact,
        },
        "legacy_training": {
            "action": legacy["action"],
            "variant": legacy["variant"],
            "parameter_count": legacy["parameter_count"],
            "argv": [str(value) for value in argv],
            "process": {
                key: process.get(key)
                for key in (
                    "started_at",
                    "ended_at",
                    "wall_seconds",
                    "peak_gpu_memory_mib",
                    "peak_cuda_allocated_mib",
                    "peak_cuda_reserved_mib",
                    "gpu_memory_source",
                    "return_code",
                )
            },
            "provenance": provenance_summary,
        },
    }


def _legacy_control_payload(
    *,
    config: ResolvedConfig,
    prepared: PreparedCell,
    seed: int,
    cublas: Mapping[str, Any],
    checkpoint: Path,
    configs_yaml: Path,
    train_manifest: Path,
    proof_path: Path,
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "edgetwincal.backbone_campaign.v1",
        "status": "complete",
        "outcome": "verified_legacy_import",
        "dataset_id": prepared.cell.dataset_id,
        "protocol_id": prepared.cell.protocol_id,
        "seed": seed,
        "resolved_config_sha256": config.sha256,
        "cublas_workspace": dict(cublas),
        "test_constructed": False,
        "loaders_constructed": False,
        "allowed_loader_splits": [],
        "loader_contracts": {},
        "protocol_manifests": {
            "protocol": _artifact(prepared.protocol_manifest),
            "split": _artifact(prepared.split_manifest),
            "normalizer": _artifact(prepared.normalizer_manifest),
        },
        "artifacts": {
            "checkpoint": _artifact(checkpoint),
            "configs_yaml": _artifact(configs_yaml),
            "train_manifest": {
                "status": "absent_by_design",
                "path": train_manifest.relative_to(PROJECT_ROOT).as_posix(),
            },
            "verified_legacy_import": _artifact(proof_path),
            "legacy_training_evidence": dict(proof["source_artifacts"]),
        },
        "checkpoint_sha256": _sha256(checkpoint),
    }



def validate_frozen_checkpoint_identity(
    *,
    config: ResolvedConfig,
    cell: BackboneCell,
    seed: int,
    checkpoint_path: str | Path,
    train_manifest_path: str | Path | None = None,
    configs_yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only unified identity validator for native and pinned legacy backbones."""

    checkpoint = require_within_root(checkpoint_path, must_exist=True)
    train_manifest = require_within_root(
        train_manifest_path or checkpoint.parent / TRAIN_MANIFEST
    )
    if train_manifest.is_file():
        native = validate_reusable_checkpoint(
            config=config,
            cell=cell,
            seed=seed,
            checkpoint_path=checkpoint,
            train_manifest_path=train_manifest,
        )
        configs = require_within_root(
            configs_yaml_path or checkpoint.parent / VENDOR_CONFIG
        )
        return {
            "schema_version": "edgetwincal.frozen_checkpoint_identity.v1",
            "status": "verified",
            "verification_mode": "native_train_manifest",
            "dataset_id": cell.dataset_id,
            "protocol_id": cell.protocol_id,
            "seed": seed,
            "resolved_config_sha256": config.sha256,
            "checkpoint": _artifact(checkpoint),
            "configs_yaml": (
                _artifact(configs)
                if configs.is_file()
                else {
                    "status": "not_required_for_identity",
                    "path": configs.relative_to(PROJECT_ROOT).as_posix(),
                }
            ),
            "training_evidence": {
                "train_manifest": _artifact(train_manifest),
                "manifest_checkpoint_record": dict(native["checkpoint"]),
            },
        }

    if (
        cell == BackboneCell("P12", "release_parity")
        and seed in LEGACY_P12_RELEASE_IDENTITIES
    ):
        configs = require_within_root(
            configs_yaml_path or checkpoint.parent / VENDOR_CONFIG,
            must_exist=True,
        )
        expected_proof = validate_legacy_p12_release_import(
            config=config,
            cell=cell,
            seed=seed,
            checkpoint_path=checkpoint,
            configs_yaml_path=configs,
        )
        proof_path = require_within_root(
            checkpoint.parent / LEGACY_IMPORT_MANIFEST, must_exist=True
        )
        observed_proof = _load_json(proof_path)
        if observed_proof != expected_proof:
            raise BackboneCampaignError(
                f"Legacy import proof drifted from source evidence: {proof_path}"
            )
        return {
            "schema_version": "edgetwincal.frozen_checkpoint_identity.v1",
            "status": "verified",
            "verification_mode": "verified_legacy_import",
            "dataset_id": cell.dataset_id,
            "protocol_id": cell.protocol_id,
            "seed": seed,
            "resolved_config_sha256": config.sha256,
            "checkpoint": _artifact(checkpoint),
            "configs_yaml": _artifact(configs),
            "training_evidence": {
                "verified_legacy_import": _artifact(proof_path),
                "source_artifacts": dict(expected_proof["source_artifacts"]),
            },
        }

    raise BackboneCampaignError(
        f"Frozen checkpoint has no verified training evidence: {cell.key}/seed_{seed}"
    )
def _strict_asset_paths(dataset_id: str) -> Mapping[str, Path]:
    if dataset_id == "P12":
        base = PROJECT_ROOT / "data" / "tsdm"
        return {
            "processed_set_a_sparse_tar": base / "datasets/Physionet2012/Physionet2012-set-A-sparse.tar",
            "processed_set_b_sparse_tar": base / "datasets/Physionet2012/Physionet2012-set-B-sparse.tar",
            "processed_set_c_sparse_tar": base / "datasets/Physionet2012/Physionet2012-set-C-sparse.tar",
            "raw_set_a_tar_gz": base / "rawdata/Physionet2012/set-a.tar.gz",
            "raw_set_b_tar_gz": base / "rawdata/Physionet2012/set-b.tar.gz",
            "raw_set_c_tar_gz": base / "rawdata/Physionet2012/set-c.tar.gz",
        }
    if dataset_id == "USHCN":
        base = PROJECT_ROOT / "data" / "tsdm"
        return {
            "processed_ushcn_parquet": base / "datasets/USHCN_DeBrouwer2019/USHCN_DeBrouwer2019.parquet",
            "raw_small_chunked_sporadic_csv": base / "rawdata/USHCN_DeBrouwer2019/small_chunked_sporadic.csv",
        }
    raise BackboneCampaignError(f"No strict raw-frame contract for {dataset_id}")


def _verified_strict_identity(
    dataset_id: str,
    split_manifest: Path,
    source_file: Path,
) -> tuple[str, dict[str, str]]:
    actual_assets = {
        name: _sha256(path) for name, path in _strict_asset_paths(dataset_id).items()
    }
    actual_code = _sha256(source_file)
    if split_manifest.exists():
        wrapper = _load_json(split_manifest)
        public = wrapper.get("public_protocol_manifest")
        if not isinstance(public, Mapping):
            raise BackboneCampaignError("Frozen strict split lacks public_protocol_manifest")
        if public.get("code_hash") != actual_code:
            raise BackboneCampaignError("Strict source hash differs from the frozen split")
        if public.get("data_asset_hashes") != actual_assets:
            raise BackboneCampaignError("Strict data-asset hashes differ from the frozen split")
    return actual_code, actual_assets


def _load_p12_frame() -> Any:
    import pandas as pd

    frames = []
    for name in ("a", "b", "c"):
        path = require_within_root(
            PROJECT_ROOT
            / "data/tsdm/datasets/Physionet2012"
            / f"Physionet2012-set-{name.upper()}-sparse.tar",
            must_exist=True,
        )
        with tarfile.open(path, mode="r") as archive:
            try:
                member = archive.getmember("series.feather")
                extracted = archive.extractfile(member)
            except (KeyError, tarfile.TarError) as exc:
                raise BackboneCampaignError(f"Invalid P12 sparse archive {path}: {exc}") from exc
            if extracted is None:
                raise BackboneCampaignError(f"P12 archive has no readable series.feather: {path}")
            frames.append(pd.read_feather(io.BytesIO(extracted.read())))
    frame = pd.concat(frames, axis=0, copy=False)
    if {"RecordID", "Time"}.issubset(frame.columns):
        frame = frame.set_index(["RecordID", "Time"], verify_integrity=True)
    if frame.index.names != ["RecordID", "Time"]:
        raise BackboneCampaignError(f"Unexpected P12 frame index: {frame.index.names}")
    return frame.sort_index(level=["RecordID", "Time"], kind="stable")


def _load_ushcn_frame() -> Any:
    import pandas as pd

    path = require_within_root(
        PROJECT_ROOT
        / "data/tsdm/datasets/USHCN_DeBrouwer2019/USHCN_DeBrouwer2019.parquet",
        must_exist=True,
    )
    frame = pd.read_parquet(path)
    if frame.index.names != ["ID", "Time"]:
        raise BackboneCampaignError(f"Unexpected USHCN frame index: {frame.index.names}")
    return frame


def _ushcn_fold_zero_keys(frame: Any) -> dict[str, Any]:
    """Reproduce only official fold-0 keys; never construct a partition dataset."""

    import numpy as np
    from sklearn.model_selection import train_test_split

    ids = frame.reset_index()["ID"].unique()
    state = np.random.get_state()
    try:
        np.random.seed(432)
        train, test = train_test_split(ids, test_size=0.1)
        train, val = train_test_split(train, test_size=0.1)
    finally:
        np.random.set_state(state)
    return {"train": train, "val": val, "test": test}


def _verify_pretest_manifest_set(prepared: PreparedCell, config: ResolvedConfig) -> None:
    protocol = _load_json(prepared.protocol_manifest)
    if protocol.get("resolved_config_sha256") != config.sha256:
        raise BackboneCampaignError("Protocol manifest resolved-config hash drifted")
    if protocol.get("dataset") != prepared.cell.dataset_id:
        raise BackboneCampaignError("Protocol manifest dataset drifted")
    if protocol.get("protocol") != prepared.cell.protocol_id:
        raise BackboneCampaignError("Protocol manifest protocol drifted")
    if protocol.get("test_constructed") is not False:
        raise BackboneCampaignError("Backbone protocol must state test_constructed=false")
    split = _load_json(prepared.split_manifest)
    normalizer = _load_json(prepared.normalizer_manifest)
    if protocol.get("split_content_sha256") != canonical_sha256(split):
        raise BackboneCampaignError("Frozen split content hash drifted")
    if protocol.get("normalizer_content_sha256") != canonical_sha256(normalizer):
        raise BackboneCampaignError("Frozen normalizer content hash drifted")


def prepare_cell(config: ResolvedConfig, cell: BackboneCell) -> PreparedCell:
    """Prepare/verify one protocol without constructing any test partition."""

    if cell.dataset_id == "MIMIC_III" and cell.protocol_id == "release_parity":
        raise BackboneBlockedError(
            "missing_author_mapping",
            "released MIMIC-III parity requires the absent author UNIQUE_ID_dict.csv mapping",
        )
    if cell.protocol_id not in CELL_PROTOCOLS.get(cell.dataset_id, ()):
        raise BackboneCampaignError(f"Unsupported backbone cell: {cell.key}")

    from .campaign_pretest import prepare_pretest_manifests
    from .runtime_v2 import resolve_run_assets

    seed = DEFAULT_SEEDS[0]
    assets = resolve_run_assets(
        config, cell.dataset_id, seed, cell.protocol_id, require_existing=False
    )
    strict_prepared: object | None = None
    if cell.protocol_id == "release_parity":
        padded = (
            HUMAN_ACTIVITY_PADDED_PREDICTION_STEPS
            if cell.dataset_id == "HumanActivity"
            else None
        )
        manifests = prepare_pretest_manifests(
            config,
            dataset_id=cell.dataset_id,
            protocol_id=cell.protocol_id,
            seed=seed,
            padded_prediction_steps=padded,
        )
    elif cell.protocol_id == "strict_p12":
        from . import strict_p12 as strict_module

        code_hash, asset_hashes = _verified_strict_identity(
            "P12", assets.split_manifest, Path(strict_module.__file__)
        )
        strict_prepared = strict_module.prepare_strict_p12(
            _load_p12_frame(), code_hash=code_hash, data_asset_hashes=asset_hashes
        )
        manifests = prepare_pretest_manifests(
            config,
            dataset_id=cell.dataset_id,
            protocol_id=cell.protocol_id,
            seed=seed,
            strict_public_bundle=strict_prepared.public_manifests(),
        )
    elif cell.protocol_id == "strict_ushcn":
        from . import strict_ushcn as strict_module

        code_hash, asset_hashes = _verified_strict_identity(
            "USHCN", assets.split_manifest, Path(strict_module.__file__)
        )
        frame = _load_ushcn_frame()
        strict_prepared = strict_module.prepare_strict_ushcn(
            frame,
            _ushcn_fold_zero_keys(frame),
            code_hash=code_hash,
            data_asset_hashes=asset_hashes,
        )
        manifests = prepare_pretest_manifests(
            config,
            dataset_id=cell.dataset_id,
            protocol_id=cell.protocol_id,
            seed=seed,
            strict_public_bundle=strict_prepared.public_manifests(),
        )
    else:  # pragma: no cover - guarded by CELL_PROTOCOLS.
        raise BackboneCampaignError(f"Unsupported protocol: {cell.protocol_id}")

    prepared = PreparedCell(
        cell=cell,
        protocol_manifest=manifests.protocol_manifest,
        split_manifest=manifests.split_manifest,
        normalizer_manifest=manifests.normalizer_manifest,
        strict_prepared=strict_prepared,
    )
    _verify_pretest_manifest_set(prepared, config)
    return prepared


def _runtime_components() -> SimpleNamespace:
    """Import torch/APN-facing code only after cuBLAS has been established."""

    from .apn_bridge import (
        ReleaseParityLoaderFactory,
        StrictLoaderFactory,
        apn_forward_callback,
        build_vendor_config,
        make_apn_model_factory,
    )
    from .apn_training import train_apn_train_val
    from .runtime_v2 import resolve_run_assets

    return SimpleNamespace(
        ReleaseParityLoaderFactory=ReleaseParityLoaderFactory,
        StrictLoaderFactory=StrictLoaderFactory,
        apn_forward_callback=apn_forward_callback,
        build_vendor_config=build_vendor_config,
        make_apn_model_factory=make_apn_model_factory,
        train_apn_train_val=train_apn_train_val,
        resolve_run_assets=resolve_run_assets,
    )


def _build_train_val_runtime(
    config: ResolvedConfig,
    prepared: PreparedCell,
    seed: int,
    components: SimpleNamespace,
) -> tuple[Any, Any, Mapping[str, Any]]:
    bundle = components.build_vendor_config(
        config,
        dataset_id=prepared.cell.dataset_id,
        protocol_id=prepared.cell.protocol_id,
        seed=seed,
        num_workers=0,
    )
    if prepared.cell.protocol_id == "release_parity":
        factory = components.ReleaseParityLoaderFactory(bundle)
    else:
        if prepared.strict_prepared is None:
            raise BackboneCampaignError("Strict cell has no reconstructed prepared object")
        factory = components.StrictLoaderFactory(bundle, prepared.strict_prepared)

    # These are the only loader APIs used by this module.  Materializing both
    # before model creation freezes released/strict padding from train+val.
    factory.build_fit_loader("train", purpose="training")
    factory.build_fit_loader("val", purpose="training")
    contracts = {
        split: factory.contract(split).public_manifest() for split in ("train", "val")
    }
    for contract in contracts.values():
        if contract.get("requested_split") not in {"train", "val"}:
            raise BackboneCampaignError("Loader contract escaped train/validation")
    protocol = _load_json(prepared.protocol_manifest)
    expected_steps = int(protocol["task_contract"]["padded_prediction_steps"])
    observed_steps = contracts["train"].get("padded_prediction_steps")
    if observed_steps is not None and int(observed_steps) != expected_steps:
        raise BackboneCampaignError(
            f"Frozen prediction padding is {expected_steps}, loader reconstructed {observed_steps}"
        )
    return bundle, factory, contracts


class _Tee(TextIO):
    def __init__(self, left: TextIO, right: TextIO) -> None:
        self.left = left
        self.right = right

    def write(self, value: str) -> int:
        self.left.write(value)
        self.right.write(value)
        return len(value)

    def flush(self) -> None:
        self.left.flush()
        self.right.flush()


@contextlib.contextmanager
def _captured_log(path: Path, header: Mapping[str, Any]):
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(header), sort_keys=True) + "\n")
        stream.flush()
        with contextlib.redirect_stdout(_Tee(sys.stdout, stream)), contextlib.redirect_stderr(
            _Tee(sys.stderr, stream)
        ):
            yield


def _control_payload(
    *,
    config: ResolvedConfig,
    prepared: PreparedCell,
    seed: int,
    outcome: str,
    cublas: Mapping[str, Any],
    bundle: Any,
    contracts: Mapping[str, Any],
    checkpoint: Path,
    configs_yaml: Path,
    train_manifest: Path,
    log_path: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "edgetwincal.backbone_campaign.v1",
        "status": "complete",
        "outcome": outcome,
        "dataset_id": prepared.cell.dataset_id,
        "protocol_id": prepared.cell.protocol_id,
        "seed": seed,
        "resolved_config_sha256": config.sha256,
        "cublas_workspace": dict(cublas),
        "test_constructed": False,
        "allowed_loader_splits": ["train", "val"],
        "loader_contracts": dict(contracts),
        "vendor_config_audit": bundle.public_audit(),
        "protocol_manifests": {
            "protocol": _artifact(prepared.protocol_manifest),
            "split": _artifact(prepared.split_manifest),
            "normalizer": _artifact(prepared.normalizer_manifest),
        },
        "artifacts": {
            "checkpoint": _artifact(checkpoint),
            "configs_yaml": _artifact(configs_yaml),
            "train_manifest": _artifact(train_manifest),
            "log": (
                {"status": "not_recorded_by_pre_control_plane", "path": None}
                if log_path is None
                else {"status": "recorded", **_artifact(log_path)}
            ),
        },
    }
    payload["checkpoint_sha256"] = payload["artifacts"]["checkpoint"]["sha256"]
    return payload


def _backbone_log_path(
    config: ResolvedConfig, cell: BackboneCell, seed: int
) -> Path:
    return require_within_root(
        Path(config["paths"]["log_root"])
        / "backbones"
        / cell.dataset_id
        / cell.protocol_id
        / f"seed_{seed}.log"
    )


def train_or_reuse_cell(
    config: ResolvedConfig,
    prepared: PreparedCell,
    seed: int,
    *,
    device: str = "cuda:0",
    reuse_only: bool = False,
    argv: Sequence[str] | None = None,
    components: SimpleNamespace | None = None,
) -> BackboneResult:
    """Train or validate exactly one cell, using train and validation only."""

    if seed not in DEFAULT_SEEDS:
        raise BackboneCampaignError(f"Unregistered seed: {seed}")
    cublas = ensure_cublas_workspace_config()
    if components is None:
        from .runtime_v2 import resolve_run_assets

        asset_resolver = resolve_run_assets
    else:
        asset_resolver = components.resolve_run_assets
    assets = asset_resolver(
        config,
        prepared.cell.dataset_id,
        seed,
        prepared.cell.protocol_id,
        require_existing=False,
    )
    run_dir = require_within_root(assets.checkpoint.parent)
    checkpoint = require_within_root(run_dir / CHECKPOINT)
    train_manifest = require_within_root(run_dir / TRAIN_MANIFEST)
    configs_yaml = require_within_root(run_dir / VENDOR_CONFIG)
    control_manifest = require_within_root(run_dir / CONTROL_MANIFEST)
    legacy_proof = require_within_root(run_dir / LEGACY_IMPORT_MANIFEST)
    log_path = _backbone_log_path(config, prepared.cell, seed)

    checkpoint_exists = checkpoint.exists()
    train_manifest_exists = train_manifest.exists()
    if checkpoint_exists and not train_manifest_exists:
        if (
            prepared.cell == BackboneCell("P12", "release_parity")
            and seed in LEGACY_P12_RELEASE_IDENTITIES
            and configs_yaml.is_file()
        ):
            proof = validate_legacy_p12_release_import(
                config=config,
                cell=prepared.cell,
                seed=seed,
                checkpoint_path=checkpoint,
                configs_yaml_path=configs_yaml,
            )
            _write_or_validate_json(legacy_proof, proof)
            payload = _legacy_control_payload(
                config=config,
                prepared=prepared,
                seed=seed,
                cublas=cublas,
                checkpoint=checkpoint,
                configs_yaml=configs_yaml,
                train_manifest=train_manifest,
                proof_path=legacy_proof,
                proof=proof,
            )
            _write_or_validate_json(control_manifest, payload)
            return BackboneResult(
                prepared.cell.dataset_id,
                prepared.cell.protocol_id,
                seed,
                "verified_legacy_import",
                str(proof["target_artifacts"]["checkpoint"]["sha256"]),
                control_manifest,
            )
        raise BackboneCampaignError(
            f"Partial checkpoint state for {prepared.cell.key}/seed_{seed}; refusing overwrite"
        )
    if train_manifest_exists and not checkpoint_exists:
        raise BackboneCampaignError(
            f"Partial checkpoint state for {prepared.cell.key}/seed_{seed}; refusing overwrite"
        )
    if legacy_proof.exists():
        raise BackboneCampaignError(
            f"Unexpected legacy proof beside a native train manifest: {legacy_proof}"
        )

    runtime = components or _runtime_components()
    bundle, loader_factory, contracts = _build_train_val_runtime(
        config, prepared, seed, runtime
    )
    snapshot = vendor_config_snapshot(bundle.config)

    cell_argv = list(
        argv
        or (
            "run_edgetwincal_backbones.py",
            "--dataset",
            prepared.cell.dataset_id,
            "--protocol",
            prepared.cell.protocol_id,
            "--seed",
            str(seed),
            "--execute",
        )
    )
    if checkpoint_exists:
        write_or_validate_vendor_config(
            configs_yaml, snapshot, create_if_missing=True
        )
        manifest = validate_reusable_checkpoint(
            config=config,
            cell=prepared.cell,
            seed=seed,
            checkpoint_path=checkpoint,
            train_manifest_path=train_manifest,
        )
        recorded_log: Path | None = log_path if log_path.is_file() else None
        payload = _control_payload(
            config=config,
            prepared=prepared,
            seed=seed,
            outcome="reused",
            cublas=cublas,
            bundle=bundle,
            contracts=contracts,
            checkpoint=checkpoint,
            configs_yaml=configs_yaml,
            train_manifest=train_manifest,
            log_path=recorded_log,
        )
        if control_manifest.exists():
            current = _load_json(control_manifest)
            stable_current = current | {"outcome": "reused"}
            if stable_current != payload:
                raise BackboneCampaignError("Existing backbone control manifest drifted")
        else:
            _atomic_write_json(control_manifest, payload)
        return BackboneResult(
            prepared.cell.dataset_id,
            prepared.cell.protocol_id,
            seed,
            "reused",
            str(manifest["checkpoint"]["sha256"]),
            control_manifest,
        )

    partial = [path for path in (configs_yaml, control_manifest, log_path) if path.exists()]
    if partial:
        raise BackboneCampaignError(
            "Partial backbone metadata exists without a complete checkpoint: "
            + ", ".join(map(str, partial))
        )
    if reuse_only:
        raise BackboneBlockedError(
            "missing_verified_checkpoint",
            f"{prepared.cell.key}/seed_{seed} has no reusable checkpoint",
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_vendor_config(configs_yaml, snapshot, create_if_missing=True)
    header = {
        "schema_version": "edgetwincal.backbone_log.v1",
        "dataset_id": prepared.cell.dataset_id,
        "protocol_id": prepared.cell.protocol_id,
        "seed": seed,
        "argv": cell_argv,
        "resolved_config_sha256": config.sha256,
        "cublas_workspace": cublas,
        "test_constructed": False,
    }
    try:
        with _captured_log(log_path, header):
            result = runtime.train_apn_train_val(
                output_dir=run_dir,
                seed=seed,
                learning_rate=float(
                    config["datasets"][prepared.cell.dataset_id]["apn_hyperparameters"][
                        "learning_rate"
                    ]
                ),
                resolved_config=config.to_dict(),
                model_factory=runtime.make_apn_model_factory(bundle),
                loader_factory=loader_factory,
                forward_callback=runtime.apn_forward_callback,
                device=device,
                argv=cell_argv,
            )
    except Exception as exc:
        failure = {
            "schema_version": "edgetwincal.backbone_campaign.v1",
            "status": "failed",
            "dataset_id": prepared.cell.dataset_id,
            "protocol_id": prepared.cell.protocol_id,
            "seed": seed,
            "resolved_config_sha256": config.sha256,
            "cublas_workspace": cublas,
            "test_constructed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "configs_yaml": _artifact(configs_yaml),
            "log": _artifact(log_path),
        }
        _atomic_write_json(control_manifest, failure)
        raise

    if result.checkpoint_path.resolve() != checkpoint.resolve():
        raise BackboneCampaignError("Trainer checkpoint path differs from the frozen asset registry")
    if result.manifest_path.resolve() != train_manifest.resolve():
        raise BackboneCampaignError("Trainer manifest path differs from the frozen asset registry")
    validate_reusable_checkpoint(
        config=config,
        cell=prepared.cell,
        seed=seed,
        checkpoint_path=checkpoint,
        train_manifest_path=train_manifest,
    )
    payload = _control_payload(
        config=config,
        prepared=prepared,
        seed=seed,
        outcome="trained",
        cublas=cublas,
        bundle=bundle,
        contracts=contracts,
        checkpoint=checkpoint,
        configs_yaml=configs_yaml,
        train_manifest=train_manifest,
        log_path=log_path,
    )
    _atomic_write_json(control_manifest, payload)
    return BackboneResult(
        prepared.cell.dataset_id,
        prepared.cell.protocol_id,
        seed,
        "trained",
        result.checkpoint_sha256,
        control_manifest,
    )


@contextlib.contextmanager
def _campaign_lock(config: ResolvedConfig):
    path = require_within_root(
        Path(config["paths"]["run_root"]) / ".backbone_campaign.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BackboneCampaignError(f"Another backbone campaign holds {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        if path.exists():
            path.unlink()


def select_cells(
    datasets: Sequence[str] | None = None,
    protocols: Sequence[str] | None = None,
) -> tuple[BackboneCell, ...]:
    selected_datasets = tuple(datasets or DEFAULT_DATASETS)
    if len(set(selected_datasets)) != len(selected_datasets):
        raise BackboneCampaignError("Dataset selection contains duplicates")
    unknown = [name for name in selected_datasets if name not in CELL_PROTOCOLS]
    if unknown:
        raise BackboneCampaignError(f"Unknown datasets: {unknown}")
    requested_protocols = None if protocols is None else set(protocols)
    cells: list[BackboneCell] = []
    for dataset in CELL_PROTOCOLS:
        if dataset not in selected_datasets:
            continue
        matched = [
            protocol
            for protocol in CELL_PROTOCOLS[dataset]
            if requested_protocols is None or protocol in requested_protocols
        ]
        if not matched:
            raise BackboneCampaignError(
                f"No selected protocol is supported for dataset {dataset}"
            )
        cells.extend(BackboneCell(dataset, protocol) for protocol in matched)
    return tuple(cells)


def select_seeds(seeds: Sequence[int] | None = None) -> tuple[int, ...]:
    requested = tuple(DEFAULT_SEEDS if seeds is None else seeds)
    if len(set(requested)) != len(requested):
        raise BackboneCampaignError("Seed selection contains duplicates")
    unknown = [seed for seed in requested if seed not in DEFAULT_SEEDS]
    if unknown:
        raise BackboneCampaignError(f"Unregistered seeds: {unknown}")
    selected = set(requested)
    return tuple(seed for seed in DEFAULT_SEEDS if seed in selected)


def campaign_plan(
    *,
    datasets: Sequence[str] | None = None,
    protocols: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": cell.dataset_id,
            "protocol_id": cell.protocol_id,
            "seed": seed,
            "status": (
                "BLOCKED[missing_author_mapping]"
                if cell.dataset_id == "MIMIC_III"
                else "planned_train_or_verified_reuse"
            ),
        }
        for cell in select_cells(datasets, protocols)
        for seed in select_seeds(seeds)
    ]


def run_campaign(
    config: ResolvedConfig,
    *,
    datasets: Sequence[str] | None = None,
    protocols: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    device: str = "cuda:0",
    reuse_only: bool = False,
) -> list[BackboneResult]:
    """Execute selected cells sequentially under one project-local GPU lock."""

    ensure_cublas_workspace_config()
    results: list[BackboneResult] = []
    selected_seeds = select_seeds(seeds)
    with _campaign_lock(config):
        for cell in select_cells(datasets, protocols):
            try:
                prepared = prepare_cell(config, cell)
            except BackboneBlockedError as exc:
                for seed in selected_seeds:
                    results.append(
                        BackboneResult(
                            cell.dataset_id,
                            cell.protocol_id,
                            seed,
                            "blocked",
                            None,
                            None,
                            str(exc),
                        )
                    )
                continue
            for seed in selected_seeds:
                results.append(
                    train_or_reuse_cell(
                        config,
                        prepared,
                        seed,
                        device=device,
                        reuse_only=reuse_only,
                    )
                )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute deterministic APN backbone training using only train/val APIs. "
            "Without --execute this command is read-only and imports no vendor/data code."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", choices=tuple(CELL_PROTOCOLS))
    parser.add_argument(
        "--protocol",
        action="append",
        choices=tuple(sorted({item for values in CELL_PROTOCOLS.values() for item in values})),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually verify/reuse checkpoints or train missing cells sequentially.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_resolved_config(args.config)
        if not args.execute:
            output: Mapping[str, Any] = {
                "mode": "read_only_plan",
                "resolved_config_sha256": config.sha256,
                "cublas_workspace_required": CUBLAS_WORKSPACE_VALUE,
                "cells": campaign_plan(
                    datasets=args.dataset,
                    protocols=args.protocol,
                    seeds=args.seed,
                ),
            }
            code = 0
        else:
            results = run_campaign(
                config,
                datasets=args.dataset,
                protocols=args.protocol,
                seeds=args.seed,
                device=args.device,
                reuse_only=args.reuse_only,
            )
            output = {
                "mode": "execute",
                "resolved_config_sha256": config.sha256,
                "results": [result.public_dict() for result in results],
            }
            code = 3 if any(result.outcome == "blocked" for result in results) else 0
    except BackboneBlockedError as exc:
        output = {"status": "blocked", "code": exc.code, "detail": exc.detail}
        code = 3
    except (BackboneCampaignError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return code


__all__ = [
    "BackboneBlockedError",
    "BackboneCampaignError",
    "BackboneCell",
    "BackboneResult",
    "PreparedCell",
    "CUBLAS_WORKSPACE_VALUE",
    "DEFAULT_SEEDS",
    "campaign_plan",
    "ensure_cublas_workspace_config",
    "prepare_cell",
    "run_campaign",
    "select_cells",
    "select_seeds",
    "train_or_reuse_cell",
    "validate_frozen_checkpoint_identity",
    "validate_legacy_p12_release_import",
    "validate_reusable_checkpoint",
    "vendor_config_snapshot",
    "write_or_validate_vendor_config",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
