"""Train/validation feature-cache campaign for frozen EdgeTwinCal backbones.

Planning is read-only and lazy. Execution establishes the deterministic cuBLAS
contract before importing PyTorch/APN code, validates the frozen backbone and
protocol manifests, and calls only the train/validation cache extraction API.
No test token, test loader, or test split is accepted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import backbone_campaign as backbone
from .config import (
    DEFAULT_CONFIG,
    ResolvedConfig,
    canonical_sha256,
    load_resolved_config,
)
from .paths import PROJECT_ROOT, require_within_root


FIT_CACHE_CONTROL_SCHEMA = "edgetwincal.fit-cache-campaign.v1"
FIT_CACHE_CONTROL = "fit_cache_campaign_manifest.json"
FIT_CACHE_SIDECAR = "fit_cache_manifest.json"
FIT_ENTRIES = "fit_entries.json"
FIT_CACHE_STEM = "train_validation_features"

LEGACY_EXTRACTION_CONFIG_DIFFS = frozenset(
    {
        "checkpoints",
        "dataset_root_path",
        "load_checkpoints_test",
        "model_id",
        "pred_len_max_irr",
        "seq_len_max_irr",
        "subfolder_train",
    }
)

_DATA_FILES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "P12": {
        "raw": (
            "data/tsdm/rawdata/Physionet2012/set-a.tar.gz",
            "data/tsdm/rawdata/Physionet2012/set-b.tar.gz",
            "data/tsdm/rawdata/Physionet2012/set-c.tar.gz",
        ),
        "processed": (
            "data/tsdm/datasets/Physionet2012/Physionet2012-set-A-sparse.tar",
            "data/tsdm/datasets/Physionet2012/Physionet2012-set-B-sparse.tar",
            "data/tsdm/datasets/Physionet2012/Physionet2012-set-C-sparse.tar",
        ),
    },
    "USHCN": {
        "raw": (
            "data/tsdm/rawdata/USHCN_DeBrouwer2019/small_chunked_sporadic.csv",
        ),
        "processed": (
            "data/tsdm/datasets/USHCN_DeBrouwer2019/USHCN_DeBrouwer2019.parquet",
        ),
    },
    "HumanActivity": {
        "raw": ("data/tsdm/HumanActivity/raw/ConfLongDemo_JSI.txt",),
        "processed": ("data/tsdm/HumanActivity/processed/data.pt",),
    },
}

_LOADER_FILES: Mapping[str, tuple[str, ...]] = {
    "P12": (
        "vendor/APN/data/data_provider/data_factory.py",
        "vendor/APN/data/data_provider/datasets/P12.py",
        "vendor/APN/data/dependencies/tsdm/tasks/P12.py",
        "vendor/APN/data/dependencies/tsdm/datasets/physionet2012.py",
        "vendor/APN/data/dependencies/tsdm/config/_config.py",
    ),
    "USHCN": (
        "vendor/APN/data/data_provider/data_factory.py",
        "vendor/APN/data/data_provider/datasets/USHCN.py",
        "vendor/APN/data/dependencies/tsdm/tasks/ushcn_debrouwer2019.py",
        "vendor/APN/data/dependencies/tsdm/datasets/ushcn_debrouwer2019.py",
        "vendor/APN/data/dependencies/tsdm/config/_config.py",
    ),
    "HumanActivity": (
        "vendor/APN/data/data_provider/data_factory.py",
        "vendor/APN/data/data_provider/datasets/HumanActivity.py",
        "vendor/APN/data/dependencies/HumanActivity/HumanActivity.py",
    ),
}


class FitCacheCampaignError(backbone.BackboneCampaignError):
    """A fit-cache cell is partial, stale, or outside the frozen campaign."""


@dataclass(frozen=True)
class ProvenanceBundle:
    value: Any
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class FitCacheResult:
    dataset_id: str
    protocol_id: str
    seed: int
    outcome: str
    entry: Mapping[str, Any] | None
    control_manifest: Path | None
    detail: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "seed": self.seed,
            "outcome": self.outcome,
            "entry": None if self.entry is None else dict(self.entry),
            "control_manifest": (
                None
                if self.control_manifest is None
                else _relative(self.control_manifest, must_exist=True)
            ),
            "detail": self.detail,
        }


def _relative(path: str | Path, *, must_exist: bool = False) -> str:
    resolved = require_within_root(path, must_exist=must_exist)
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def _sha256_file(path: str | Path) -> str:
    source = require_within_root(path, must_exist=True)
    if not source.is_file():
        raise FitCacheCampaignError(f"Expected file for hashing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    if not source.is_file():
        raise FitCacheCampaignError(f"Expected frozen file: {source}")
    return {
        "path": _relative(source, must_exist=True),
        "bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def composite_sha256(paths: Sequence[str | Path]) -> tuple[str, list[dict[str, Any]]]:
    """Hash a sorted, path-bound set of project files."""

    if isinstance(paths, (str, bytes)) or not paths:
        raise FitCacheCampaignError("Composite hash requires one or more files")
    resolved = [require_within_root(path, must_exist=True) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise FitCacheCampaignError("Composite hash file list contains duplicates")
    records = [_artifact(path) for path in sorted(resolved, key=lambda item: _relative(item))]
    canonical = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), records


def _git_head() -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise FitCacheCampaignError("Project HEAD is not a full SHA-1")
    return value


def _project_files(names: Sequence[str]) -> tuple[Path, ...]:
    return tuple(require_within_root(PROJECT_ROOT / name, must_exist=True) for name in names)


def _source_paths(dataset_id: str, protocol_id: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    if dataset_id not in _LOADER_FILES:
        raise FitCacheCampaignError(f"No fit-cache source contract for {dataset_id}")
    loader_names = [
        "code/src/edgetwincal/apn_bridge.py",
        "code/src/edgetwincal/backbone_campaign.py",
        *_LOADER_FILES[dataset_id],
    ]
    if protocol_id == "strict_p12":
        loader_names.append("code/src/edgetwincal/strict_p12.py")
    elif protocol_id == "strict_ushcn":
        loader_names.append("code/src/edgetwincal/strict_ushcn.py")
    extractor_names = (
        "code/src/edgetwincal/fit_cache_campaign.py",
        "code/src/edgetwincal/campaign_extract.py",
        "code/src/edgetwincal/apn_bridge.py",
        "code/src/edgetwincal/runtime_v2.py",
        "code/src/edgetwincal/provenance.py",
    )
    return _project_files(loader_names), _project_files(extractor_names)


def build_provenance_bundle(
    dataset_id: str,
    protocol_id: str,
    *,
    extraction_type: Any,
) -> ProvenanceBundle:
    """Build real dataset/source provenance without reading a test partition."""

    if dataset_id == "MIMIC_III":
        raise backbone.BackboneBlockedError(
            "missing_author_mapping",
            "MIMIC-III release parity remains blocked without UNIQUE_ID_dict.csv",
        )
    data = _DATA_FILES.get(dataset_id)
    if data is None:
        raise FitCacheCampaignError(f"No dataset provenance contract for {dataset_id}")
    raw_sha, raw_records = composite_sha256(_project_files(data["raw"]))
    processed_sha, processed_records = composite_sha256(
        _project_files(data["processed"])
    )
    loader_paths, extractor_paths = _source_paths(dataset_id, protocol_id)
    loader_sha, loader_records = composite_sha256(loader_paths)
    extractor_sha, extractor_records = composite_sha256(extractor_paths)
    head = _git_head()
    value = extraction_type(
        project_commit=head,
        dataset_raw_sha256=raw_sha,
        dataset_processed_sha256=processed_sha,
        loader_source_sha256=loader_sha,
        extractor_source_sha256=extractor_sha,
    )
    return ProvenanceBundle(
        value=value,
        audit={
            "project_head": head,
            "dataset_raw": {"sha256": raw_sha, "files": raw_records},
            "dataset_processed": {
                "sha256": processed_sha,
                "files": processed_records,
            },
            "loader_source": {"sha256": loader_sha, "files": loader_records},
            "extractor_source": {
                "sha256": extractor_sha,
                "files": extractor_records,
            },
        },
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FitCacheCampaignError(f"Cannot read JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise FitCacheCampaignError(f"JSON artifact must be an object: {source}")
    return value


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
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
        temporary.unlink(missing_ok=True)
    return destination


def _write_or_validate_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = require_within_root(path)
    if destination.exists():
        if _load_json(destination) != dict(value):
            raise FitCacheCampaignError(
                f"Existing frozen JSON differs from expected content: {destination}"
            )
        return destination
    return _atomic_write_json(destination, value)


def _runtime_components() -> SimpleNamespace:
    """Import torch/APN-facing modules only after cuBLAS is configured."""

    from .campaign_extract import prepare_fit_cache
    from .campaign_runner import load_cache_manifest
    from .runtime_v2 import ExtractionProvenance, read_tensor_cache

    backbone_runtime = backbone._runtime_components()

    def build_train_val_runtime(
        config: ResolvedConfig,
        prepared: backbone.PreparedCell,
        seed: int,
    ):
        return backbone._build_train_val_runtime(
            config, prepared, seed, backbone_runtime
        )

    return SimpleNamespace(
        resolve_run_assets=backbone_runtime.resolve_run_assets,
        build_train_val_runtime=build_train_val_runtime,
        prepare_fit_cache=prepare_fit_cache,
        load_cache_manifest=load_cache_manifest,
        read_tensor_cache=read_tensor_cache,
        ExtractionProvenance=ExtractionProvenance,
    )


def _validate_frozen_backbone(
    config: ResolvedConfig,
    prepared: backbone.PreparedCell,
    seed: int,
    assets: Any,
) -> Mapping[str, Any]:
    expected_paths = {
        "protocol": (prepared.protocol_manifest, assets.protocol_manifest),
        "split": (prepared.split_manifest, assets.split_manifest),
        "normalizer": (prepared.normalizer_manifest, assets.normalizer_manifest),
    }
    for label, (prepared_path, asset_path) in expected_paths.items():
        if prepared_path.resolve() != asset_path.resolve():
            raise FitCacheCampaignError(
                f"Prepared {label} path differs from resolved asset registry"
            )
    configs_yaml = require_within_root(assets.apn_config)
    missing = [
        path
        for path in (
            assets.checkpoint,
            configs_yaml,
            assets.protocol_manifest,
            assets.split_manifest,
            assets.normalizer_manifest,
        )
        if not path.is_file()
    ]
    if missing:
        raise backbone.BackboneBlockedError(
            "missing_verified_checkpoint",
            ", ".join(_relative(path) for path in missing),
        )
    identity = backbone.validate_frozen_checkpoint_identity(
        config=config,
        cell=prepared.cell,
        seed=seed,
        checkpoint_path=assets.checkpoint,
        configs_yaml_path=configs_yaml,
    )
    return {
        "checkpoint": _artifact(assets.checkpoint),
        "configs_yaml": _artifact(configs_yaml),
        "checkpoint_identity": dict(identity),
        "protocol_manifest": _artifact(assets.protocol_manifest),
        "split_manifest": _artifact(assets.split_manifest),
        "normalizer_manifest": _artifact(assets.normalizer_manifest),
        "apn_patch": _artifact(assets.apn_patch),
    }


def validate_extraction_config_compatibility(
    frozen_config_path: str | Path,
    reconstructed: Mapping[str, Any],
    *,
    checkpoint_identity: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind a safe extraction config without rewriting training evidence.

    Native checkpoints require semantic YAML equality. The three pinned P12
    legacy imports may differ only in seven audited operational fields that
    name output locations, disable automatic test execution, or hold
    train-loader-derived padding. Every model/data/training field remains
    exactly equal.
    """

    import yaml

    source = require_within_root(frozen_config_path, must_exist=True)
    try:
        frozen = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FitCacheCampaignError(
            f"Cannot read frozen extraction config: {exc}"
        ) from exc
    if not isinstance(frozen, Mapping):
        raise FitCacheCampaignError("Frozen extraction config must be a mapping")
    current = dict(reconstructed)
    keys = sorted(set(frozen) | set(current))
    differences = {
        key: {"frozen": frozen.get(key), "reconstructed": current.get(key)}
        for key in keys
        if frozen.get(key) != current.get(key)
    }
    common = {
        "schema_version": "edgetwincal.extraction-config-compatibility.v1",
        "frozen_config_sha256": canonical_sha256(dict(frozen)),
        "reconstructed_config_sha256": canonical_sha256(current),
    }
    if not differences:
        return {
            **common,
            "mode": "exact_native_config",
            "allowed_difference_keys": [],
            "differences": {},
        }

    mode = checkpoint_identity.get("verification_mode")
    dataset_id = checkpoint_identity.get("dataset_id")
    protocol_id = checkpoint_identity.get("protocol_id")
    seed = checkpoint_identity.get("seed")
    if (
        mode != "verified_legacy_import"
        or dataset_id != "P12"
        or protocol_id != "release_parity"
        or seed not in backbone.LEGACY_P12_RELEASE_IDENTITIES
    ):
        raise FitCacheCampaignError(
            "Native extraction config differs from its frozen training config: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )
    if set(differences) != LEGACY_EXTRACTION_CONFIG_DIFFS:
        raise FitCacheCampaignError(
            "Legacy extraction config has unregistered differences: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )

    train_contract = contracts.get("train")
    if not isinstance(train_contract, Mapping):
        raise FitCacheCampaignError(
            "Legacy extraction requires a frozen train contract"
        )
    history = train_contract.get("padded_history_steps")
    horizon = train_contract.get("padded_prediction_steps")
    checkpoint_record = checkpoint_identity.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise FitCacheCampaignError(
            "Legacy checkpoint identity lacks its checkpoint record"
        )

    expected_old_checkpoint_root = require_within_root(
        PROJECT_ROOT / "results" / "stage_a" / "apn" / str(seed) / "checkpoints",
        must_exist=True,
    )
    expected_new_checkpoint_root = require_within_root(
        checkpoint_record["path"], must_exist=True
    ).parent
    expected_old_dataset_root = require_within_root(
        PROJECT_ROOT / "data" / "tsdm" / "datasets" / "Physionet2012",
        must_exist=True,
    )
    expected_new_dataset_root = require_within_root(
        PROJECT_ROOT / "data" / "tsdm", must_exist=True
    )
    path_checks = {
        "frozen.checkpoints": (
            require_within_root(frozen["checkpoints"], must_exist=True),
            expected_old_checkpoint_root,
        ),
        "reconstructed.checkpoints": (
            require_within_root(current["checkpoints"], must_exist=True),
            expected_new_checkpoint_root,
        ),
        "frozen.dataset_root_path": (
            require_within_root(frozen["dataset_root_path"], must_exist=True),
            expected_old_dataset_root,
        ),
        "reconstructed.dataset_root_path": (
            require_within_root(current["dataset_root_path"], must_exist=True),
            expected_new_dataset_root,
        ),
    }
    mismatches = {
        name: {"observed": str(observed), "expected": str(expected)}
        for name, (observed, expected) in path_checks.items()
        if observed.resolve() != expected.resolve()
    }
    value_checks = {
        "load_checkpoints_test": (
            frozen["load_checkpoints_test"],
            current["load_checkpoints_test"],
            1,
            0,
        ),
        "model_id": (
            frozen["model_id"],
            current["model_id"],
            f"APN_P12_apn_seed{seed}",
            f"APN_P12_release_parity_seed{seed}",
        ),
        "seq_len_max_irr": (
            frozen["seq_len_max_irr"],
            current["seq_len_max_irr"],
            None,
            history,
        ),
        "pred_len_max_irr": (
            frozen["pred_len_max_irr"],
            current["pred_len_max_irr"],
            None,
            horizon,
        ),
        "subfolder_train": (
            bool(frozen["subfolder_train"]),
            current["subfolder_train"],
            True,
            "",
        ),
    }
    for name, (old, new, expected_old, expected_new) in value_checks.items():
        if old != expected_old or new != expected_new:
            mismatches[name] = {
                "frozen": old,
                "reconstructed": new,
                "expected_frozen": expected_old,
                "expected_reconstructed": expected_new,
            }
    if mismatches:
        raise FitCacheCampaignError(
            "Legacy extraction operational fields violate their frozen contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return {
        **common,
        "mode": "verified_legacy_operational_overlay",
        "allowed_difference_keys": sorted(LEGACY_EXTRACTION_CONFIG_DIFFS),
        "differences": differences,
    }


def _frozen_context(
    config: ResolvedConfig,
    prepared: backbone.PreparedCell,
    seed: int,
    assets: Any,
    provenance: ProvenanceBundle,
    cublas: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": prepared.cell.dataset_id,
        "protocol_id": prepared.cell.protocol_id,
        "seed": seed,
        "resolved_config_sha256": config.sha256,
        "cublas_workspace": {
            "required": backbone.CUBLAS_WORKSPACE_VALUE,
            "observed": cublas["observed"],
        },
        "frozen_artifacts": _validate_frozen_backbone(
            config, prepared, seed, assets
        ),
        "provenance": dict(provenance.audit),
        "test_constructed": False,
        "allowed_splits": ["train", "val"],
    }


def _manifest_mismatches(
    manifest: Any,
    config: ResolvedConfig,
    assets: Any,
    provenance: ProvenanceBundle,
) -> dict[str, Any]:
    expected = {
        "method_version": config["method"]["version"],
        "project_commit": provenance.value.project_commit,
        "apn_commit": config["apn"]["commit"],
        "apn_patch_sha256": _sha256_file(assets.apn_patch),
        "apn_mode": config["apn"]["patch_mode"],
        "checkpoint_sha256": _sha256_file(assets.checkpoint),
        "resolved_config_sha256": config.sha256,
        "dataset_id": assets.dataset_id,
        "protocol_manifest_sha256": _sha256_file(assets.protocol_manifest),
        "split_manifest_sha256": _sha256_file(assets.split_manifest),
        "normalizer_manifest_sha256": _sha256_file(assets.normalizer_manifest),
        "loader_source_sha256": provenance.value.loader_source_sha256,
        "extractor_source_sha256": provenance.value.extractor_source_sha256,
        "dataset_raw_sha256": provenance.value.dataset_raw_sha256,
        "dataset_processed_sha256": provenance.value.dataset_processed_sha256,
        "seed": assets.seed,
    }
    return {
        key: {"expected": value, "observed": getattr(manifest, key, None)}
        for key, value in expected.items()
        if getattr(manifest, key, None) != value
    }


def _validate_cache_artifacts(
    *,
    runtime: SimpleNamespace,
    config: ResolvedConfig,
    assets: Any,
    provenance: ProvenanceBundle,
    cache_path: str | Path,
    sidecar_path: str | Path,
    expected_control: Mapping[str, Any] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    cache = require_within_root(cache_path, must_exist=True)
    sidecar = require_within_root(sidecar_path, must_exist=True)
    manifest = runtime.load_cache_manifest(sidecar)
    mismatches = _manifest_mismatches(manifest, config, assets, provenance)
    split_names = {name.split(".", 1)[0] for name in manifest.shapes}
    if split_names != {"train", "val"}:
        mismatches["splits"] = {
            "expected": ["train", "val"],
            "observed": sorted(split_names),
        }
    if mismatches:
        raise FitCacheCampaignError(
            "Fit-cache provenance drift: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    runtime.read_tensor_cache(cache, manifest)
    cache_record = {
        "cache": _artifact(cache),
        "sidecar": _artifact(sidecar),
        "manifest_digest": manifest.manifest_digest(),
        "payload_sha256": manifest.payload_sha256,
        "splits": ["train", "val"],
    }
    if expected_control is not None and dict(expected_control) != cache_record:
        raise FitCacheCampaignError("Existing fit-cache file hashes drifted")
    return manifest, cache_record


def _entry(seed: int, assets: Any, cache: Path, sidecar: Path) -> dict[str, Any]:
    return {
        "seed": seed,
        "fit_cache": _relative(cache, must_exist=True),
        "fit_cache_manifest": _relative(sidecar, must_exist=True),
        "checkpoint": _relative(assets.checkpoint, must_exist=True),
    }


def prepare_or_reuse_seed(
    config: ResolvedConfig,
    prepared: backbone.PreparedCell,
    seed: int,
    *,
    device: str,
    runtime: SimpleNamespace,
    provenance: ProvenanceBundle,
    cublas: Mapping[str, Any],
    reuse_only: bool = False,
) -> FitCacheResult:
    """Extract or hash-validate one train/validation cache."""

    assets = runtime.resolve_run_assets(
        config,
        prepared.cell.dataset_id,
        seed,
        prepared.cell.protocol_id,
        require_existing=False,
    )
    context = _frozen_context(
        config, prepared, seed, assets, provenance, cublas
    )
    cache_directory = require_within_root(assets.cache_directory)
    sidecar = require_within_root(cache_directory / FIT_CACHE_SIDECAR)
    control_path = require_within_root(cache_directory / FIT_CACHE_CONTROL)

    if control_path.exists():
        control = _load_json(control_path)
        required = {
            "schema_version": FIT_CACHE_CONTROL_SCHEMA,
            "status": "complete",
            "frozen_context": context,
        }
        mismatches = {
            key: {"expected": value, "observed": control.get(key)}
            for key, value in required.items()
            if control.get(key) != value
        }
        if mismatches:
            raise FitCacheCampaignError(
                "Existing fit-cache control drift: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        cache_record = control.get("cache_artifacts")
        compatibility = control.get("extraction_config_compatibility")
        if (
            not isinstance(compatibility, Mapping)
            or compatibility.get("schema_version")
            != "edgetwincal.extraction-config-compatibility.v1"
        ):
            raise FitCacheCampaignError(
                "Existing control lacks extraction config compatibility evidence"
            )
        if not isinstance(cache_record, Mapping):
            raise FitCacheCampaignError("Existing control lacks cache_artifacts")
        cache_artifact = cache_record.get("cache")
        sidecar_artifact = cache_record.get("sidecar")
        if not isinstance(cache_artifact, Mapping) or not isinstance(
            sidecar_artifact, Mapping
        ):
            raise FitCacheCampaignError(
                "Existing control has invalid cache artifact records"
            )
        if not isinstance(cache_artifact.get("path"), str) or not isinstance(
            sidecar_artifact.get("path"), str
        ):
            raise FitCacheCampaignError("Existing control has invalid cache paths")
        cache_path = require_within_root(
            cache_artifact["path"], must_exist=True
        )
        recorded_sidecar = require_within_root(
            sidecar_artifact["path"], must_exist=True
        )
        if recorded_sidecar.resolve() != sidecar.resolve():
            raise FitCacheCampaignError("Fit-cache sidecar path drifted")
        _validate_cache_artifacts(
            runtime=runtime,
            config=config,
            assets=assets,
            provenance=provenance,
            cache_path=cache_path,
            sidecar_path=sidecar,
            expected_control=cache_record,
        )
        return FitCacheResult(
            prepared.cell.dataset_id,
            prepared.cell.protocol_id,
            seed,
            "reused",
            _entry(seed, assets, cache_path, sidecar),
            control_path,
        )

    existing_cache_files = (
        tuple(cache_directory.glob(f"{FIT_CACHE_STEM}.*.cache"))
        if cache_directory.exists()
        else ()
    )
    if sidecar.exists() or existing_cache_files:
        raise FitCacheCampaignError(
            "Partial fit-cache artifacts exist without a complete control manifest"
        )
    if reuse_only:
        raise backbone.BackboneBlockedError(
            "missing_fit_cache",
            f"{prepared.cell.key}/seed_{seed} has no frozen fit cache",
        )

    bundle, loader_factory, contracts = runtime.build_train_val_runtime(
        config, prepared, seed
    )
    snapshot = backbone.vendor_config_snapshot(bundle.config)
    checkpoint_identity = context["frozen_artifacts"]["checkpoint_identity"]
    if not isinstance(checkpoint_identity, Mapping):
        raise FitCacheCampaignError("Frozen context lacks checkpoint identity")
    config_compatibility = validate_extraction_config_compatibility(
        assets.apn_config,
        snapshot,
        checkpoint_identity=checkpoint_identity,
        contracts=contracts,
    )
    result = runtime.prepare_fit_cache(
        config=config,
        assets=assets,
        provenance=provenance.value,
        bundle=bundle,
        loader_factory=loader_factory,
        cache_manifest_path=sidecar,
        device=device,
        train_backbone=False,
        argv=(
            "prepare_edgetwincal_fit_caches.py",
            "--dataset",
            prepared.cell.dataset_id,
            "--protocol",
            prepared.cell.protocol_id,
            "--seed",
            str(seed),
            "--execute",
        ),
    )
    if result.get("test_opened") is not False or result.get("splits_opened") != [
        "train",
        "val",
    ]:
        raise FitCacheCampaignError("Extractor violated the train/validation-only contract")
    cache_path = require_within_root(result["cache_path"], must_exist=True)
    manifest, cache_record = _validate_cache_artifacts(
        runtime=runtime,
        config=config,
        assets=assets,
        provenance=provenance,
        cache_path=cache_path,
        sidecar_path=sidecar,
    )
    if result.get("cache_manifest_sha256") != manifest.manifest_digest():
        raise FitCacheCampaignError("Extractor result manifest digest drifted")
    control = {
        "schema_version": FIT_CACHE_CONTROL_SCHEMA,
        "status": "complete",
        "outcome": "extracted",
        "frozen_context": context,
        "loader_contracts": dict(contracts),
        "cache_artifacts": cache_record,
        "extraction_config_compatibility": config_compatibility,
    }
    _write_or_validate_json(control_path, control)
    return FitCacheResult(
        prepared.cell.dataset_id,
        prepared.cell.protocol_id,
        seed,
        "extracted",
        _entry(seed, assets, cache_path, sidecar),
        control_path,
    )


def _entries_path(config: ResolvedConfig, cell: backbone.BackboneCell) -> Path:
    return require_within_root(
        Path(config["paths"]["protocol_root"])
        / cell.dataset_id
        / cell.protocol_id
        / FIT_ENTRIES
    )


def _campaign_lock(config: ResolvedConfig):
    class _Lock:
        def __init__(self) -> None:
            self.path = require_within_root(
                Path(config["paths"]["run_root"]) / ".fit_cache_campaign.lock"
            )
            self.descriptor: int | None = None

        def __enter__(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError as exc:
                raise FitCacheCampaignError(
                    f"Another fit-cache campaign holds {self.path}"
                ) from exc
            os.write(self.descriptor, str(os.getpid()).encode("ascii"))
            return self

        def __exit__(self, exc_type, exc, traceback):
            if self.descriptor is not None:
                os.close(self.descriptor)
            self.path.unlink(missing_ok=True)
            return False

    return _Lock()


def campaign_plan(
    config: ResolvedConfig,
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
                else "planned_fit_cache_extract_or_verified_reuse"
            ),
            "fit_entries": _relative(_entries_path(config, cell)),
        }
        for cell in backbone.select_cells(datasets, protocols)
        for seed in backbone.select_seeds(seeds)
    ]


def run_campaign(
    config: ResolvedConfig,
    *,
    datasets: Sequence[str] | None = None,
    protocols: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    device: str = "cuda:0",
    reuse_only: bool = False,
    runtime: SimpleNamespace | None = None,
) -> list[FitCacheResult]:
    """Run selected cells and seeds sequentially without any test API."""

    cublas = backbone.ensure_cublas_workspace_config()
    active_runtime = runtime or _runtime_components()
    selected_seeds = backbone.select_seeds(seeds)
    results: list[FitCacheResult] = []
    with _campaign_lock(config):
        for cell in backbone.select_cells(datasets, protocols):
            try:
                prepared = backbone.prepare_cell(config, cell)
                provenance = build_provenance_bundle(
                    cell.dataset_id,
                    cell.protocol_id,
                    extraction_type=active_runtime.ExtractionProvenance,
                )
            except backbone.BackboneBlockedError as exc:
                results.extend(
                    FitCacheResult(
                        cell.dataset_id,
                        cell.protocol_id,
                        seed,
                        "blocked",
                        None,
                        None,
                        str(exc),
                    )
                    for seed in selected_seeds
                )
                continue

            cell_results: list[FitCacheResult] = []
            for seed in selected_seeds:
                try:
                    result = prepare_or_reuse_seed(
                        config,
                        prepared,
                        seed,
                        device=device,
                        runtime=active_runtime,
                        provenance=provenance,
                        cublas=cublas,
                        reuse_only=reuse_only,
                    )
                except backbone.BackboneBlockedError as exc:
                    result = FitCacheResult(
                        cell.dataset_id,
                        cell.protocol_id,
                        seed,
                        "blocked",
                        None,
                        None,
                        str(exc),
                    )
                cell_results.append(result)
                results.append(result)

            all_selected = tuple(int(value) for value in config["selection"]["seeds"])
            ready = {
                result.seed: result.entry
                for result in cell_results
                if result.entry is not None
            }
            if tuple(ready) == all_selected:
                _write_or_validate_json(
                    _entries_path(config, cell),
                    {"entries": [ready[seed] for seed in all_selected]},
                )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or extract frozen APN train/validation feature caches. "
            "Without --execute this command is read-only."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", choices=tuple(backbone.CELL_PROTOCOLS))
    parser.add_argument(
        "--protocol",
        action="append",
        choices=tuple(
            sorted(
                {
                    protocol
                    for values in backbone.CELL_PROTOCOLS.values()
                    for protocol in values
                }
            )
        ),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda:0")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
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
                "cublas_workspace_required": backbone.CUBLAS_WORKSPACE_VALUE,
                "cells": campaign_plan(
                    config,
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
    except backbone.BackboneBlockedError as exc:
        output = {"status": "blocked", "code": exc.code, "detail": exc.detail}
        code = 3
    except (
        FitCacheCampaignError,
        backbone.BackboneCampaignError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return code


__all__ = [
    "FIT_CACHE_CONTROL_SCHEMA",
    "FIT_ENTRIES",
    "FitCacheCampaignError",
    "FitCacheResult",
    "ProvenanceBundle",
    "build_arg_parser",
    "build_provenance_bundle",
    "campaign_plan",
    "composite_sha256",
    "main",
    "prepare_or_reuse_seed",
    "run_campaign",
]


if __name__ == "__main__":
    raise SystemExit(main())
