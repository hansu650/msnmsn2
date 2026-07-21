"""Lazy, provenance-safe runtime boundary for the msn2026 campaign.

This module deliberately does not import APN, a dataset module, NumPy, or
PyTorch at import time.  Resolving CLI help therefore cannot initialize a
vendor package or open a dataset.  Asset checks and tensor deserialization are
explicit operations performed only after a dataset/seed/protocol run has been
selected from :class:`~edgetwincal.config.ResolvedConfig`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .config import DEFAULT_CONFIG, ConfigError, ResolvedConfig, load_resolved_config
from .paths import PROJECT_ROOT
from .provenance import (
    CACHE_SCHEMA_VERSION,
    CacheExpectation,
    CacheManifest,
    StaleCacheError,
    cache_file_path,
    read_cache,
    write_cache_atomic,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only; torch stays lazy.
    from torch import Tensor


TENSOR_PAYLOAD_SCHEMA_VERSION = 3
SPLIT_NAMES = ("train", "val", "test")
FIT_SPLITS = ("train", "val")
TEST_SPLITS = ("test",)
ALLOWED_SPLIT_SETS = frozenset(
    {
        frozenset(FIT_SPLITS),
        frozenset(TEST_SPLITS),
        frozenset(SPLIT_NAMES),
    }
)
SPLIT_TENSOR_KEYS = (
    "features",
    "time_encoding",
    "base_prediction",
    "target",
    "mask",
    "sample_id",
    "group_id",
)


class RuntimeV2Error(RuntimeError):
    """Base class for safe-runtime failures."""


class BlockedAssetError(RuntimeV2Error):
    """A run cannot start because an authorized, registered asset is absent."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"BLOCKED[{code}]: {detail}")


class TensorPayloadError(RuntimeV2Error):
    """A tensor bundle or its serialized representation violates schema 3."""


@dataclass(frozen=True)
class ResolvedRunAssets:
    """All deterministic paths for one selected dataset/seed/protocol run."""

    dataset_id: str
    seed: int
    protocol_id: str
    dataset_root: Path
    apn_root: Path
    apn_patch: Path
    checkpoint: Path
    apn_config: Path
    protocol_manifest: Path
    split_manifest: Path
    normalizer_manifest: Path
    cache_directory: Path

    def required_inputs(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                "dataset_root": self.dataset_root,
                "apn_root": self.apn_root,
                "apn_patch": self.apn_patch,
                "checkpoint": self.checkpoint,
                "apn_config": self.apn_config,
                "protocol_manifest": self.protocol_manifest,
                "split_manifest": self.split_manifest,
                "normalizer_manifest": self.normalizer_manifest,
            }
        )


@dataclass(frozen=True)
class ExtractionProvenance:
    """Caller-supplied identities that cannot be inferred from asset filenames."""

    project_commit: str
    dataset_raw_sha256: str
    dataset_processed_sha256: str
    loader_source_sha256: str
    extractor_source_sha256: str


@dataclass(frozen=True)
class ExtractionShapeContract:
    """Explicit padding contract for a time-window forecasting extraction.

    HumanActivity pred_len is a time duration, not a tensor length.  The
    extractor must therefore freeze the actual padded prediction-step count
    before a cache key is constructed.
    """

    padded_prediction_steps: int

    def __post_init__(self) -> None:
        value = self.padded_prediction_steps
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("padded_prediction_steps must be a positive integer")



@dataclass(frozen=True)
class TensorInventory:
    """Canonical tensor metadata and split-identity hashes for a cache key."""

    shapes: Mapping[str, tuple[int, ...]]
    dtypes: Mapping[str, str]
    mask_sha256: str
    sample_ids_sha256: str
    group_ids_sha256: str


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - campaign environment has torch.
        raise BlockedAssetError("torch_unavailable", "PyTorch is required for tensor caches") from exc
    return torch


def _resolve_below(root: str | Path, relative: str | Path) -> Path:
    root_path = Path(root).resolve(strict=True)
    candidate = Path(relative)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (root_path / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ConfigError(f"Resolved asset escapes runtime root: {resolved}") from exc
    return resolved


def _selected(config: ResolvedConfig, kind: str) -> tuple[Any, ...]:
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise ConfigError("Resolved config has no selection registry")
    values = selection.get(kind)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ConfigError(f"selection.{kind} must be a sequence")
    return tuple(values)


def resolve_run_assets(
    config: ResolvedConfig,
    dataset_id: str,
    seed: int,
    protocol_id: str,
    *,
    root: str | Path = PROJECT_ROOT,
    require_existing: bool = True,
) -> ResolvedRunAssets:
    """Resolve one registered run without searching the filesystem.

    The path layout is configuration-driven and uniform across datasets and
    seeds.  No fallback globbing is allowed because selecting "the only file"
    can silently join the wrong checkpoint to a run.
    """

    if not isinstance(config, ResolvedConfig):
        raise TypeError("config must be ResolvedConfig")
    if dataset_id not in _selected(config, "datasets"):
        raise ConfigError(f"Dataset is not selected in resolved config: {dataset_id}")
    if isinstance(seed, bool) or seed not in _selected(config, "seeds"):
        raise ConfigError(f"Seed is not selected in resolved config: {seed!r}")

    datasets = config.get("datasets")
    protocols = config.get("protocols")
    paths = config.get("paths")
    apn = config.get("apn")
    if not all(isinstance(item, Mapping) for item in (datasets, protocols, paths, apn)):
        raise ConfigError("Resolved config is missing dataset/protocol/path/APN registries")
    dataset = datasets.get(dataset_id)
    if not isinstance(dataset, Mapping):
        raise ConfigError(f"Unknown dataset: {dataset_id}")
    allowed_protocols = {dataset.get("release_protocol"), dataset.get("strict_protocol")}
    if protocol_id not in protocols or protocol_id not in allowed_protocols:
        raise ConfigError(
            f"Protocol {protocol_id!r} is not registered for dataset {dataset_id!r}"
        )
    availability = dataset.get("availability")
    if isinstance(availability, str) and availability.startswith("blocked"):
        raise BlockedAssetError(
            "dataset_authorization",
            f"{dataset_id} availability is {availability}",
        )

    root_path = Path(root).resolve(strict=True)
    dataset_root = _resolve_below(root_path, str(dataset["storage"]))
    apn_root = _resolve_below(root_path, str(paths["apn_root"]))
    apn_patch = _resolve_below(root_path, str(apn["patch"]))
    checkpoint_directory = _resolve_below(
        root_path,
        Path(str(paths["checkpoint_root"])) / dataset_id / protocol_id / f"seed_{seed}",
    )
    protocol_directory = _resolve_below(
        root_path,
        Path(str(paths["protocol_root"])) / dataset_id / protocol_id,
    )
    cache_directory = _resolve_below(
        root_path,
        Path(str(paths["cache_root"])) / dataset_id / protocol_id / str(seed),
    )
    assets = ResolvedRunAssets(
        dataset_id=dataset_id,
        seed=int(seed),
        protocol_id=protocol_id,
        dataset_root=dataset_root,
        apn_root=apn_root,
        apn_patch=apn_patch,
        checkpoint=checkpoint_directory / "pytorch_model.bin",
        apn_config=checkpoint_directory / "configs.yaml",
        protocol_manifest=protocol_directory / "protocol_manifest.json",
        split_manifest=protocol_directory / "split_manifest.json",
        normalizer_manifest=protocol_directory / "normalizer_manifest.json",
        cache_directory=cache_directory,
    )
    if require_existing:
        missing = [
            f"{name}={path}"
            for name, path in assets.required_inputs().items()
            if not path.exists()
        ]
        if missing:
            raise BlockedAssetError("missing_assets", "; ".join(missing))
        not_files = [
            f"{name}={path}"
            for name, path in assets.required_inputs().items()
            if name not in {"dataset_root", "apn_root"} and not path.is_file()
        ]
        not_directories = [
            f"{name}={path}"
            for name, path in assets.required_inputs().items()
            if name in {"dataset_root", "apn_root"} and not path.is_dir()
        ]
        if not_files or not_directories:
            raise BlockedAssetError(
                "invalid_asset_type", "; ".join(not_files + not_directories)
            )
    return assets


def file_sha256(path: str | Path, *, root: str | Path = PROJECT_ROOT) -> str:
    """Hash one required file after an explicit runtime-root boundary check."""

    resolved = _resolve_below(root, path)
    if not resolved.is_file():
        raise BlockedAssetError("missing_source", f"Required file is absent: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_source_hashes(
    loader_source: str | Path,
    extractor_source: str | Path,
    *,
    root: str | Path = PROJECT_ROOT,
) -> Mapping[str, str]:
    """Collect the two source identities required by the frozen cache key."""

    return MappingProxyType(
        {
            "loader_source_sha256": file_sha256(loader_source, root=root),
            "extractor_source_sha256": file_sha256(extractor_source, root=root),
        }
    )


def collect_environment() -> dict[str, Any]:
    """Return environment facts without invoking package managers or vendor code."""

    torch = _torch()
    devices: list[dict[str, Any]] = []
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda) if torch.version.cuda is not None else None,
            "cuda_available": cuda_available,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": devices,
        },
    }


def _dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _canonical_tensor(tensor: Any, *, name: str) -> Any:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TensorPayloadError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise TensorPayloadError(f"{name} must be a dense, non-quantized strided tensor")
    return tensor.detach().to(device="cpu").contiguous().clone()


def _ordered_split_names(splits: Mapping[str, Any]) -> tuple[str, ...]:
    names = frozenset(splits)
    if names not in ALLOWED_SPLIT_SETS:
        allowed = [list(names_) for names_ in (FIT_SPLITS, TEST_SPLITS, SPLIT_NAMES)]
        raise TensorPayloadError(
            f"Tensor payload split set must be one of {allowed}; got {sorted(names)}"
        )
    return tuple(name for name in SPLIT_NAMES if name in names)


def _hash_named_tensors(splits: Mapping[str, Mapping[str, Any]], key: str) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    for split in _ordered_split_names(splits):
        tensor = splits[split][key]
        header = json.dumps(
            {
                "split": split,
                "key": key,
                "shape": list(tensor.shape),
                "dtype": _dtype_name(tensor),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _canonicalize_tensor_splits(
    splits: Mapping[str, Mapping[str, Any]],
    *,
    require_group_disjoint: bool,
) -> dict[str, dict[str, Any]]:
    torch = _torch()
    if not isinstance(splits, Mapping):
        raise TensorPayloadError("Tensor payload splits must be a mapping")
    split_names = _ordered_split_names(splits)
    canonical: dict[str, dict[str, Any]] = {}
    all_sample_ids: set[int] = set()
    group_sets: dict[str, set[int]] = {}
    float_keys = {"features", "time_encoding", "base_prediction", "target"}
    floating_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}

    for split in split_names:
        raw = splits[split]
        if not isinstance(raw, Mapping) or set(raw) != set(SPLIT_TENSOR_KEYS):
            missing = sorted(set(SPLIT_TENSOR_KEYS) - set(raw) if isinstance(raw, Mapping) else set(SPLIT_TENSOR_KEYS))
            extra = sorted(set(raw) - set(SPLIT_TENSOR_KEYS) if isinstance(raw, Mapping) else set())
            raise TensorPayloadError(
                f"{split} tensor keys differ; missing={missing}, extra={extra}"
            )
        current = {
            key: _canonical_tensor(raw[key], name=f"{split}.{key}")
            for key in SPLIT_TENSOR_KEYS
        }
        canonical[split] = current
        n = int(current["sample_id"].shape[0]) if current["sample_id"].ndim == 1 else -1
        if n <= 0:
            raise TensorPayloadError(f"{split}.sample_id must be a non-empty rank-1 tensor")
        if current["group_id"].shape != (n,):
            raise TensorPayloadError(f"{split}.group_id must have shape ({n},)")
        if current["sample_id"].dtype != torch.int64 or current["group_id"].dtype != torch.int64:
            raise TensorPayloadError(f"{split} sample_id and group_id must use int64")
        if current["mask"].dtype != torch.bool:
            raise TensorPayloadError(f"{split}.mask must use bool")
        if any(current[key].dtype not in floating_dtypes for key in float_keys):
            raise TensorPayloadError(f"{split} feature/prediction/target tensors must be floating point")

        prediction_shape = tuple(current["base_prediction"].shape)
        if len(prediction_shape) != 3 or prediction_shape[0] != n:
            raise TensorPayloadError(f"{split}.base_prediction must have shape [N,H,C]")
        if tuple(current["target"].shape) != prediction_shape or tuple(current["mask"].shape) != prediction_shape:
            raise TensorPayloadError(
                f"{split} target and mask shapes must equal base_prediction {prediction_shape}"
            )
        _, horizon, channels = prediction_shape
        features_shape = tuple(current["features"].shape)
        if len(features_shape) != 3 or features_shape[:2] != (n, channels):
            raise TensorPayloadError(f"{split}.features must have shape [N,C,D]")
        time_shape = tuple(current["time_encoding"].shape)
        if len(time_shape) != 4 or time_shape[:3] != (n, channels, horizon):
            raise TensorPayloadError(f"{split}.time_encoding must have shape [N,C,H,E]")
        if not bool(current["mask"].any()):
            raise TensorPayloadError(f"{split}.mask has no observed forecast target")
        for key in ("features", "time_encoding", "base_prediction"):
            if not bool(torch.isfinite(current[key]).all()):
                raise TensorPayloadError(f"{split}.{key} contains NaN or infinity")
        if not bool(torch.isfinite(current["target"][current["mask"]]).all()):
            raise TensorPayloadError(f"{split}.target contains a non-finite observed value")

        sample_ids = [int(value) for value in current["sample_id"].tolist()]
        if len(set(sample_ids)) != n:
            raise TensorPayloadError(f"{split}.sample_id contains duplicates")
        overlap = all_sample_ids.intersection(sample_ids)
        if overlap:
            raise TensorPayloadError(f"sample_id overlaps across splits: {sorted(overlap)[:5]}")
        all_sample_ids.update(sample_ids)
        group_sets[split] = {int(value) for value in current["group_id"].tolist()}

    if require_group_disjoint and len(split_names) > 1:
        overlaps = {
            f"{left}_{right}": group_sets[left] & group_sets[right]
            for index, left in enumerate(split_names)
            for right in split_names[index + 1 :]
        }
        nonempty = {name: sorted(values)[:5] for name, values in overlaps.items() if values}
        if nonempty:
            raise TensorPayloadError(f"Strict-protocol group overlap: {nonempty}")
    return canonical


def tensor_inventory(
    splits: Mapping[str, Mapping[str, Any]],
    *,
    require_group_disjoint: bool = False,
) -> TensorInventory:
    canonical = _canonicalize_tensor_splits(
        splits, require_group_disjoint=require_group_disjoint
    )
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for split in _ordered_split_names(canonical):
        for key in SPLIT_TENSOR_KEYS:
            name = f"{split}.{key}"
            tensor = canonical[split][key]
            shapes[name] = tuple(int(value) for value in tensor.shape)
            dtypes[name] = _dtype_name(tensor)
    return TensorInventory(
        shapes=MappingProxyType(shapes),
        dtypes=MappingProxyType(dtypes),
        mask_sha256=_hash_named_tensors(canonical, "mask"),
        sample_ids_sha256=_hash_named_tensors(canonical, "sample_id"),
        group_ids_sha256=_hash_named_tensors(canonical, "group_id"),
    )


def serialize_tensor_payload(splits: Mapping[str, Mapping[str, Any]]) -> bytes:
    """Serialize validated CPU tensors into a schema-3, weights-only payload."""

    torch = _torch()
    canonical = _canonicalize_tensor_splits(splits, require_group_disjoint=False)
    stream = io.BytesIO()
    torch.save(
        {"schema_version": TENSOR_PAYLOAD_SCHEMA_VERSION, "splits": canonical},
        stream,
    )
    return stream.getvalue()


def deserialize_tensor_payload(payload: bytes | bytearray | memoryview) -> dict[str, dict[str, Any]]:
    """Deserialize only safe tensor/primitives and reject every legacy schema."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    torch = _torch()
    try:
        value = torch.load(io.BytesIO(bytes(payload)), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TensorPayloadError(f"Tensor payload cannot be safely deserialized: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "splits"}:
        raise TensorPayloadError("Tensor payload envelope fields are invalid")
    schema = value.get("schema_version")
    if schema != TENSOR_PAYLOAD_SCHEMA_VERSION:
        suffix = "; schema-2 is legacy-only" if schema == 2 else ""
        raise TensorPayloadError(
            f"Tensor payload schema must be {TENSOR_PAYLOAD_SCHEMA_VERSION}{suffix}"
        )
    return _canonicalize_tensor_splits(value["splits"], require_group_disjoint=False)


def _strict_protocol(config: ResolvedConfig, protocol_id: str) -> bool:
    protocol = config["protocols"][protocol_id]
    return protocol.get("kind") != "released_code"


def _manifest_padded_prediction_steps(path: Path) -> int:
    """Read a frozen HumanActivity padding count without importing data code."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TensorPayloadError(
            f"Cannot read padded_prediction_steps from protocol manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise TensorPayloadError("Protocol manifest must be an object")

    candidates: list[tuple[str, Any]] = []
    if "padded_prediction_steps" in value:
        candidates.append(("padded_prediction_steps", value["padded_prediction_steps"]))
    for section_name in ("task_contract", "padding", "task_horizon", "extraction_contract"):
        section = value.get(section_name)
        if isinstance(section, Mapping) and "padded_prediction_steps" in section:
            candidates.append(
                (
                    f"{section_name}.padded_prediction_steps",
                    section["padded_prediction_steps"],
                )
            )
    if not candidates:
        raise TensorPayloadError(
            "HumanActivity time-window task requires padded_prediction_steps in "
            "the frozen protocol/padding manifest or an explicit extraction contract"
        )

    normalized: list[tuple[str, int]] = []
    for label, raw in candidates:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise TensorPayloadError(f"{label} must be a positive integer")
        normalized.append((label, raw))
    distinct = {item for _, item in normalized}
    if len(distinct) != 1:
        raise TensorPayloadError(
            f"Protocol manifest has conflicting padded_prediction_steps: {normalized}"
        )
    return normalized[0][1]


def _task_prediction_steps(
    task: Mapping[str, Any],
    assets: ResolvedRunAssets,
    extraction_contract: ExtractionShapeContract | None,
) -> int:
    kind = task.get("horizon_kind")
    if kind == "fixed_steps":
        expected = int(task["prediction_steps"])
        if (
            extraction_contract is not None
            and extraction_contract.padded_prediction_steps != expected
        ):
            raise TensorPayloadError(
                "Explicit extraction contract conflicts with fixed prediction_steps"
            )
        return expected
    if kind != "time_window":
        raise TensorPayloadError(f"Unknown task horizon_kind: {kind!r}")
    if extraction_contract is not None:
        return extraction_contract.padded_prediction_steps
    return _manifest_padded_prediction_steps(assets.protocol_manifest)

def _key_fields(
    config: ResolvedConfig,
    assets: ResolvedRunAssets,
    splits: Mapping[str, Mapping[str, Any]],
    provenance: ExtractionProvenance,
    *,
    root: str | Path,
    extraction_contract: ExtractionShapeContract | None = None,
) -> dict[str, Any]:
    inventory = tensor_inventory(
        splits,
        require_group_disjoint=_strict_protocol(config, assets.protocol_id),
    )
    dataset = config["datasets"][assets.dataset_id]
    task = dataset["task"]
    apn_parameters = dataset["apn_hyperparameters"]
    expected_horizon = _task_prediction_steps(task, assets, extraction_contract)
    expected_channels = int(task["channels"])
    expected_d_model = int(apn_parameters["d_model"])
    expected_te_dim = int(apn_parameters["te_dim"])
    for split in _ordered_split_names(splits):
        prediction_shape = inventory.shapes[f"{split}.base_prediction"]
        feature_shape = inventory.shapes[f"{split}.features"]
        time_shape = inventory.shapes[f"{split}.time_encoding"]
        if prediction_shape[1:] != (expected_horizon, expected_channels):
            raise TensorPayloadError(
                f"{split}.base_prediction has H,C={prediction_shape[1:]}, expected "
                f"{(expected_horizon, expected_channels)} from the task horizon contract"
            )
        if feature_shape[1:] != (expected_channels, expected_d_model):
            raise TensorPayloadError(
                f"{split}.features has C,D={feature_shape[1:]}, expected "
                f"{(expected_channels, expected_d_model)} from ResolvedConfig"
            )
        if time_shape[1:] != (expected_channels, expected_horizon, expected_te_dim):
            raise TensorPayloadError(
                f"{split}.time_encoding has C,H,E={time_shape[1:]}, expected "
                f"{(expected_channels, expected_horizon, expected_te_dim)} from the task horizon contract"
            )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method_version": str(config["method"]["version"]),
        "project_commit": provenance.project_commit,
        "apn_commit": str(config["apn"]["commit"]),
        "apn_patch_sha256": file_sha256(assets.apn_patch, root=root),
        "apn_mode": str(config["apn"]["patch_mode"]),
        "checkpoint_sha256": file_sha256(assets.checkpoint, root=root),
        "resolved_config_sha256": config.sha256,
        "dataset_id": assets.dataset_id,
        "dataset_raw_sha256": provenance.dataset_raw_sha256,
        "dataset_processed_sha256": provenance.dataset_processed_sha256,
        "protocol_manifest_sha256": file_sha256(assets.protocol_manifest, root=root),
        "split_manifest_sha256": file_sha256(assets.split_manifest, root=root),
        "sample_ids_sha256": inventory.sample_ids_sha256,
        "group_ids_sha256": inventory.group_ids_sha256,
        "normalizer_manifest_sha256": file_sha256(assets.normalizer_manifest, root=root),
        "loader_source_sha256": provenance.loader_source_sha256,
        "extractor_source_sha256": provenance.extractor_source_sha256,
        "seed": assets.seed,
        "history_length": int(task["history"]),
        "horizon": expected_horizon,
        "shapes": dict(inventory.shapes),
        "dtypes": dict(inventory.dtypes),
        "mask_sha256": inventory.mask_sha256,
    }


def build_cache_expectation(
    config: ResolvedConfig,
    assets: ResolvedRunAssets,
    splits: Mapping[str, Mapping[str, Any]],
    provenance: ExtractionProvenance,
    *,
    root: str | Path = PROJECT_ROOT,
    extraction_contract: ExtractionShapeContract | None = None,
) -> CacheExpectation:
    """Build and validate every extraction-key field before a cache read."""

    return CacheManifest.expectation(
        **_key_fields(
            config,
            assets,
            splits,
            provenance,
            extraction_contract=extraction_contract,
            root=root,
        )
    )


def build_cache_manifest(
    config: ResolvedConfig,
    assets: ResolvedRunAssets,
    splits: Mapping[str, Mapping[str, Any]],
    provenance: ExtractionProvenance,
    payload: bytes | bytearray | memoryview,
    *,
    root: str | Path = PROJECT_ROOT,
    extraction_contract: ExtractionShapeContract | None = None,
) -> CacheManifest:
    """Build a schema-3 manifest committing to provenance and payload bytes."""

    return CacheManifest.build(
        payload=payload,
        **_key_fields(
            config,
            assets,
            splits,
            provenance,
            extraction_contract=extraction_contract,
            root=root,
        ),
    )


def write_tensor_cache(
    config: ResolvedConfig,
    assets: ResolvedRunAssets,
    splits: Mapping[str, Mapping[str, Any]],
    provenance: ExtractionProvenance,
    *,
    root: str | Path = PROJECT_ROOT,
    stem: str = "frozen_features",
    extraction_contract: ExtractionShapeContract | None = None,
) -> tuple[Path, CacheManifest]:
    """Serialize and atomically write one provenance-complete tensor cache."""

    payload = serialize_tensor_payload(splits)
    manifest = build_cache_manifest(
        config,
        assets,
        splits,
        provenance,
        payload,
        extraction_contract=extraction_contract,
        root=root,
    )
    path = cache_file_path(assets.cache_directory, stem, manifest, root=root)
    written = write_cache_atomic(path, manifest, payload, root=root)
    return written, manifest


def read_tensor_cache(
    path: str | Path,
    expected: CacheExpectation | CacheManifest,
    *,
    root: str | Path = PROJECT_ROOT,
) -> tuple[CacheManifest, dict[str, dict[str, Any]]]:
    """Read only after envelope, provenance, payload, and tensor checks pass."""

    manifest, payload = read_cache(path, expected, root=root)
    splits = deserialize_tensor_payload(payload)
    inventory = tensor_inventory(splits)
    actual = {
        "shapes": {name: list(shape) for name, shape in inventory.shapes.items()},
        "dtypes": dict(inventory.dtypes),
        "mask_sha256": inventory.mask_sha256,
        "sample_ids_sha256": inventory.sample_ids_sha256,
        "group_ids_sha256": inventory.group_ids_sha256,
    }
    recorded = {
        "shapes": {name: list(shape) for name, shape in manifest.shapes.items()},
        "dtypes": dict(manifest.dtypes),
        "mask_sha256": manifest.mask_sha256,
        "sample_ids_sha256": manifest.sample_ids_sha256,
        "group_ids_sha256": manifest.group_ids_sha256,
    }
    mismatches = {
        name: (recorded[name], actual[name])
        for name in recorded
        if recorded[name] != actual[name]
    }
    if mismatches:
        raise StaleCacheError(mismatches)
    return manifest, splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve one EdgeTwinCal dataset/seed/protocol asset registry entry."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--check-assets", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_resolved_config(
            args.config, {"dataset": args.dataset, "seed": args.seed}
        )
        assets = resolve_run_assets(
            config,
            args.dataset,
            args.seed,
            args.protocol,
            require_existing=args.check_assets,
        )
    except (ConfigError, BlockedAssetError) as exc:
        print(str(exc), file=sys.stderr)
        return 3 if isinstance(exc, BlockedAssetError) else 2
    print(
        json.dumps(
            {
                "dataset_id": assets.dataset_id,
                "seed": assets.seed,
                "protocol_id": assets.protocol_id,
                "checkpoint": str(assets.checkpoint),
                "cache_directory": str(assets.cache_directory),
                "assets_checked": bool(args.check_assets),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "TENSOR_PAYLOAD_SCHEMA_VERSION",
    "ALLOWED_SPLIT_SETS",
    "FIT_SPLITS",
    "SPLIT_NAMES",
    "SPLIT_TENSOR_KEYS",
    "TEST_SPLITS",
    "RuntimeV2Error",
    "BlockedAssetError",
    "TensorPayloadError",
    "ResolvedRunAssets",
    "ExtractionProvenance",
    "TensorInventory",
    "resolve_run_assets",
    "file_sha256",
    "collect_source_hashes",
    "ExtractionShapeContract",
    "collect_environment",
    "tensor_inventory",
    "serialize_tensor_payload",
    "deserialize_tensor_payload",
    "build_cache_expectation",
    "build_cache_manifest",
    "write_tensor_cache",
    "read_tensor_cache",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
