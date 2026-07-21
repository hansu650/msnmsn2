"""Project-owned boundary around the pinned APN implementation.

The bridge keeps every vendor import behind an explicit call, constructs only
registered APN configurations, and separates the released data behavior from
the leakage-controlled project-owned loaders.  In particular, training APIs
accept only ``train`` and ``val``.  A test loader can be created only through a
separate method carrying a validated frozen-ledger token.

The released APN wrappers inspect all partitions (including test) while they
derive irregular padding lengths, and P12 fits its standardizer on the full
dataset.  Those facts are preserved solely as labelled ``release_parity``
behavior.  Strict loaders use a project-owned collator whose padding limits are
frozen from train and validation samples only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .campaign import ProtocolLedger, ProtocolLedgerError
from .config import ResolvedConfig
from .paths import PROJECT_ROOT, require_within_root


FitSplit = Literal["train", "val"]
AnySplit = Literal["train", "val", "test"]
Purpose = Literal["training", "extraction"]

VENDOR_ROOT = PROJECT_ROOT / "vendor" / "APN"
TSDM_ROOT = PROJECT_ROOT / "data" / "tsdm"
_BASE_CONFIG = (
    VENDOR_ROOT
    / "configs"
    / "APN"
    / "APN_P12_apn_seed2024"
    / "P12.yaml"
)
_FIT_SPLITS = ("train", "val")
_ALL_SPLITS = (*_FIT_SPLITS, "test")
_TOKEN_SENTINEL = object()


class APNBridgeError(RuntimeError):
    """Raised when the project/vendor boundary contract is violated."""


class APNTestAccessError(PermissionError, APNBridgeError):
    """Raised before any unauthorized test dataset or loader is constructed."""


class PreparedStrictDataset(Protocol):
    dataset_id: str
    protocol_id: str

    def build_dataset(
        self, partition: str, *, ledger_token: object | None = None
    ) -> Dataset[Any]: ...


@dataclass(frozen=True)
class APNConfigBundle:
    """Vendor config plus a privacy-safe, immutable construction audit."""

    config: Any = field(repr=False)
    dataset_id: str
    protocol_id: str
    seed: int
    audit: Mapping[str, Any]

    def public_audit(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.audit), sort_keys=True))


@dataclass(frozen=True)
class LoaderContract:
    """Frozen account of what data a loader may inspect."""

    dataset_id: str
    protocol_id: str
    kind: str
    requested_split: str
    value_fit_scope: str
    padding_fit_scope: str
    scans_test_during_train_construction: bool
    sample_identity_quality: str
    padded_history_steps: int | None
    padded_prediction_steps: int | None

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "edgetwincal.apn_loader_contract.v1",
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "kind": self.kind,
            "requested_split": self.requested_split,
            "value_fit_scope": self.value_fit_scope,
            "padding_fit_scope": self.padding_fit_scope,
            "scans_test_during_train_construction": self.scans_test_during_train_construction,
            "sample_identity_quality": self.sample_identity_quality,
            "padded_history_steps": self.padded_history_steps,
            "padded_prediction_steps": self.padded_prediction_steps,
        }


class FrozenAPNTestLedgerToken:
    """Opaque active opening from a frozen :class:`ProtocolLedger`.

    Instances cannot be constructed directly.  The raw opening secret is never
    exposed by ``repr`` or the public manifest, and every use revalidates the
    still-open ledger cell.
    """

    __slots__ = (
        "_ledger",
        "_secret",
        "cell_id",
        "dataset_id",
        "protocol_id",
        "fold",
        "protocol_sha256",
        "token_sha256",
    )

    def __init__(
        self,
        marker: object,
        *,
        ledger: ProtocolLedger,
        secret: str,
        cell_id: str,
        dataset_id: str,
        protocol_id: str,
        fold: str,
        protocol_sha256: str,
    ) -> None:
        if marker is not _TOKEN_SENTINEL:
            raise APNTestAccessError(
                "FrozenAPNTestLedgerToken must come from an active frozen-ledger opening"
            )
        self._ledger = ledger
        self._secret = secret
        self.cell_id = cell_id
        self.dataset_id = dataset_id
        self.protocol_id = protocol_id
        self.fold = fold
        self.protocol_sha256 = protocol_sha256
        self.token_sha256 = hashlib.sha256(secret.encode("ascii")).hexdigest()

    def __repr__(self) -> str:
        return (
            "FrozenAPNTestLedgerToken("
            f"cell_id={self.cell_id!r}, dataset_id={self.dataset_id!r}, "
            f"protocol_id={self.protocol_id!r}, token_sha256={self.token_sha256!r})"
        )

    @classmethod
    def from_protocol_ledger(
        cls,
        ledger: ProtocolLedger,
        *,
        cell_id: str,
        token: str,
    ) -> "FrozenAPNTestLedgerToken":
        if not isinstance(ledger, ProtocolLedger):
            raise TypeError("ledger must be ProtocolLedger")
        ledger.validate_test_token(cell_id=cell_id, token=token)
        if ledger.status != "test_active":
            raise APNTestAccessError("Protocol ledger has no active frozen test opening")
        record = ledger.data["test_openings"].get(cell_id)
        if not isinstance(record, Mapping):
            raise APNTestAccessError("Test opening is absent from the protocol ledger")
        protocol_sha256 = record.get("protocol_sha256")
        if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
            raise APNTestAccessError("Test opening is not tied to a frozen protocol digest")
        return cls(
            _TOKEN_SENTINEL,
            ledger=ledger,
            secret=token,
            cell_id=cell_id,
            dataset_id=str(record["dataset"]),
            protocol_id=str(record["protocol"]),
            fold=str(record["fold"]),
            protocol_sha256=protocol_sha256,
        )

    def validate_for(self, dataset_id: str, protocol_id: str) -> None:
        if (dataset_id, protocol_id) != (self.dataset_id, self.protocol_id):
            raise APNTestAccessError(
                "Frozen test opening does not match the requested dataset/protocol"
            )
        try:
            self._ledger.validate_test_token(cell_id=self.cell_id, token=self._secret)
        except ProtocolLedgerError as exc:
            raise APNTestAccessError(str(exc)) from exc
        if self._ledger.status != "test_active":
            raise APNTestAccessError("Frozen test opening is no longer active")

    def public_manifest(self) -> dict[str, str]:
        return {
            "schema_version": "edgetwincal.apn_test_access.v1",
            "cell_id": self.cell_id,
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "fold": self.fold,
            "protocol_sha256": self.protocol_sha256,
            "token_sha256": self.token_sha256,
        }


def _resolve_project_path(path: str | Path, *, must_exist: bool = False) -> Path:
    return require_within_root(path, must_exist=must_exist)


def configure_vendor_environment(
    *,
    project_root: str | Path = PROJECT_ROOT,
    tsdm_root: str | Path = TSDM_ROOT,
) -> Mapping[str, str]:
    """Set exactly the two project-owned environment variables APN/TSDM need."""

    resolved_project = _resolve_project_path(project_root, must_exist=True)
    if resolved_project != PROJECT_ROOT.resolve(strict=True):
        raise APNBridgeError("APN bridge project_root must be the msn2 project root")
    resolved_tsdm = _resolve_project_path(tsdm_root, must_exist=True)
    try:
        resolved_tsdm.relative_to(resolved_project)
    except ValueError as exc:  # defensive; require_within_root already enforces this.
        raise APNBridgeError("TSDM root escapes the project root") from exc
    home_before = os.environ.get("HOME")
    codex_home_before = os.environ.get("CODEX_HOME")
    os.environ["EVIPATCH_PROJECT_ROOT"] = str(resolved_project)
    os.environ["EVIPATCH_TSDM_ROOT"] = str(resolved_tsdm)
    if os.environ.get("HOME") != home_before or os.environ.get("CODEX_HOME") != codex_home_before:
        raise APNBridgeError("APN bridge must never alter HOME or CODEX_HOME")
    return MappingProxyType(
        {
            "EVIPATCH_PROJECT_ROOT": str(resolved_project),
            "EVIPATCH_TSDM_ROOT": str(resolved_tsdm),
        }
    )


def _ensure_vendor_import_path(apn_root: str | Path = VENDOR_ROOT) -> Path:
    root = _resolve_project_path(apn_root, must_exist=True)
    source = _resolve_project_path(PROJECT_ROOT / "code" / "src", must_exist=True)
    for path in (source, root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    configure_vendor_environment()
    return root


def _install_config_shim(config: Any) -> None:
    """Prevent vendor dataset imports from parsing process argv or writing YAML."""

    module = sys.modules.get("utils.configs")
    if module is None or not getattr(module, "__edgetwincal_shim__", False):
        module = types.ModuleType("utils.configs")
        module.__edgetwincal_shim__ = True
        sys.modules["utils.configs"] = module
    module.configs = config


def _dataset_config(config: ResolvedConfig, dataset_id: str) -> Mapping[str, Any]:
    if not isinstance(config, ResolvedConfig):
        raise TypeError("resolved_config must be ResolvedConfig")
    datasets = config["datasets"]
    if dataset_id not in datasets:
        raise APNBridgeError(f"Unknown APN dataset {dataset_id!r}")
    return datasets[dataset_id]


def build_vendor_config(
    resolved_config: ResolvedConfig,
    *,
    dataset_id: str,
    protocol_id: str,
    seed: int,
    source_yaml: str | Path | None = None,
    num_workers: int = 0,
    config_type: Callable[..., Any] | None = None,
) -> APNConfigBundle:
    """Build a complete APN config from the frozen campaign registry.

    Reading YAML is harmless, but importing ``utils.ExpConfigs`` is delayed
    until this function is called.  Absolute paths embedded in an old YAML are
    always overwritten with boundary-checked msn2 paths.
    """

    dataset = _dataset_config(resolved_config, dataset_id)
    if seed not in tuple(resolved_config["campaign"]["seeds"]):
        raise APNBridgeError(f"Seed {seed} is not registered")
    allowed_protocols = {dataset["release_protocol"], dataset["strict_protocol"]}
    if protocol_id not in allowed_protocols:
        raise APNBridgeError(
            f"Protocol {protocol_id!r} is not registered for {dataset_id!r}"
        )
    template = _resolve_project_path(source_yaml or _BASE_CONFIG, must_exist=True)
    import yaml

    raw = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise APNBridgeError("APN YAML template must contain a mapping")
    task = dataset["task"]
    hp = dataset["apn_hyperparameters"]
    if task.get("horizon_kind") == "fixed_steps":
        vendor_pred_len = int(task["prediction_steps"])
    elif task.get("horizon_kind") == "time_window":
        # APN names this argument pred_len even though HumanActivity interprets
        # it as a time-window width, not a dense number of forecast rows.
        vendor_pred_len = int(task["forecast_window"])
    else:
        raise APNBridgeError(
            f"Unsupported horizon contract for {dataset_id}: {task.get('horizon_kind')!r}"
        )
    storage = _resolve_project_path(dataset["storage"], must_exist=True)
    checkpoint_dir = _resolve_project_path(
        Path(resolved_config["paths"]["checkpoint_root"])
        / dataset_id
        / protocol_id
        / f"seed_{seed}"
    )
    raw.update(
        {
            "task_name": "short_term_forecast",
            "is_training": 1,
            "model_id": f"APN_{dataset_id}_{protocol_id}_seed{seed}",
            "model_name": "APN",
            "checkpoints": str(checkpoint_dir),
            "ablation_name": "",
            "dataset_name": dataset_id,
            "dataset_root_path": str(storage),
            "features": "M",
            "seq_len": int(task["history"]),
            "label_len": 0,
            "pred_len": vendor_pred_len,
            "enc_in": int(task["channels"]),
            "dec_in": int(task["channels"]),
            "c_out": int(task["channels"]),
            "train_epochs": int(resolved_config["training"]["epochs"]),
            "patience": int(resolved_config["training"]["patience"]),
            "val_interval": int(resolved_config["training"]["validation_interval"]),
            "loss": str(resolved_config["training"]["loss"]),
            "lr_scheduler": "DelayedStepDecayLR",
            "batch_size": int(hp["batch_size"]),
            "learning_rate": float(hp["learning_rate"]),
            "d_model": int(hp["d_model"]),
            "dropout": float(hp["dropout"]),
            "apn_npatch": int(hp["npatch"]),
            "apn_te_dim": int(hp["te_dim"]),
            "evipatch_mode": "apn",
            "observation_shift": "none",
            "shift_rate": 0.0,
            "shift_seed": int(seed),
            "seed_base": int(seed),
            "num_workers": int(num_workers),
            "itr": 1,
            "save_arrays": 0,
            "load_checkpoints_test": 0,
            "test_all": 0,
            "test_train_time": 0,
            "test_inference_time": 0,
            "test_gpu_memory": 0,
            "test_dataset_statistics": 0,
            "skip_test_after_train": 1,
            "train_val_loader_shuffle": None,
            "train_val_loader_drop_last": None,
        }
    )
    if config_type is None:
        _ensure_vendor_import_path()
        config_type = getattr(importlib.import_module("utils.ExpConfigs"), "ExpConfigs")
    vendor_config = config_type(**raw)
    audit = MappingProxyType(
        {
            "schema_version": "edgetwincal.apn_config_bundle.v1",
            "dataset_id": dataset_id,
            "protocol_id": protocol_id,
            "seed": int(seed),
            "template": template.relative_to(PROJECT_ROOT).as_posix(),
            "dataset_storage": storage.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_directory": checkpoint_dir.relative_to(PROJECT_ROOT).as_posix(),
            "apn_mode": "apn",
            "test_automatic_after_train": False,
            "home_overridden": False,
        }
    )
    return APNConfigBundle(vendor_config, dataset_id, protocol_id, int(seed), audit)


def make_apn_model_factory(
    bundle: APNConfigBundle,
    *,
    importer: Callable[[str], Any] | None = None,
) -> Callable[[int], nn.Module]:
    """Return an ``apn_training.ModelFactory`` without importing APN yet."""

    if not isinstance(bundle, APNConfigBundle):
        raise TypeError("bundle must be APNConfigBundle")

    def factory(seed: int) -> nn.Module:
        if seed != bundle.seed:
            raise APNBridgeError(
                f"Model factory seed {seed} differs from frozen seed {bundle.seed}"
            )
        _ensure_vendor_import_path()
        module = (importer or importlib.import_module)("models.APN")
        model = module.Model(bundle.config)
        if not isinstance(model, nn.Module):
            raise APNBridgeError("Pinned APN Model did not produce torch.nn.Module")
        return model

    return factory


def apn_forward_callback(
    model: nn.Module,
    batch: Mapping[str, Any],
    stage: FitSplit,
) -> Mapping[str, Tensor]:
    """Adapter matching :func:`edgetwincal.apn_training.train_apn_train_val`."""

    if stage not in _FIT_SPLITS:
        raise APNBridgeError(f"Training callback rejects split {stage!r}")
    if not isinstance(batch, Mapping):
        raise TypeError("APN batches must be mappings")
    required = {"x", "x_mark", "x_mask", "y", "y_mark", "y_mask"}
    missing = sorted(required.difference(batch))
    if missing:
        raise APNBridgeError(f"APN batch is missing fields: {missing}")
    output = model(
        exp_stage=stage,
        train_stage=1,
        **batch,
    )
    if not isinstance(output, Mapping):
        raise APNBridgeError("APN forward output must be a mapping")
    return output


def _loader_from_vendor_result(result: Any) -> tuple[Dataset[Any], Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise APNBridgeError("APN data_provider must return (dataset, loader)")
    dataset, loader = result
    if not hasattr(dataset, "__len__") or not hasattr(loader, "__iter__"):
        raise APNBridgeError("APN data_provider returned invalid dataset/loader")
    return dataset, loader


class ReleaseParityLoaderFactory:
    """Released APN loader with its leakage/scanning behavior explicitly labelled."""

    def __init__(
        self,
        bundle: APNConfigBundle,
        *,
        provider: Callable[[Any, str], Any] | None = None,
        importer: Callable[[str], Any] | None = None,
    ) -> None:
        if bundle.protocol_id != "release_parity":
            raise APNBridgeError("ReleaseParityLoaderFactory requires release_parity")
        self.bundle = bundle
        self._provider = provider
        self._importer = importer
        self._datasets: dict[str, Dataset[Any]] = {}
        self._loaders: dict[tuple[str, str], Any] = {}
        self._train_constructed = False

    def _resolve_provider(self) -> Callable[[Any, str], Any]:
        if self._provider is None:
            _ensure_vendor_import_path()
            _install_config_shim(self.bundle.config)
            module = (self._importer or importlib.import_module)(
                "data.data_provider.data_factory"
            )
            self._provider = module.data_provider
        return self._provider

    def _construct(self, split: AnySplit) -> tuple[Dataset[Any], Any]:
        if split == "val" and not self._train_constructed:
            raise APNBridgeError(
                "Released val loader requires train construction first because APN mutates "
                "its config with all-partition padding lengths"
            )
        result = self._resolve_provider()(self.bundle.config, split)
        dataset, loader = _loader_from_vendor_result(result)
        self._datasets[split] = dataset
        self._loaders[(split, "training")] = loader
        if split == "train":
            self._train_constructed = True
        return dataset, loader

    def _get(self, split: AnySplit, purpose: Purpose) -> Any:
        key = (split, purpose)
        if key in self._loaders:
            return self._loaders[key]
        if split not in self._datasets:
            dataset, released_loader = self._construct(split)
        else:
            dataset = self._datasets[split]
            released_loader = self._loaders[(split, "training")]
        if purpose == "training":
            return released_loader
        extraction = DataLoader(
            dataset,
            batch_size=int(self.bundle.config.batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=getattr(released_loader, "collate_fn", None),
        )
        self._loaders[key] = extraction
        return extraction

    def __call__(self, split: FitSplit, seed: int) -> Iterable[Any]:
        if split not in _FIT_SPLITS:
            raise APNTestAccessError(
                "APN training loader factory accepts only train and val"
            )
        if seed != self.bundle.seed:
            raise APNBridgeError("Loader seed differs from frozen APN config")
        return self._get(split, "training")

    def build_fit_loader(self, split: FitSplit, *, purpose: Purpose = "extraction") -> Any:
        if split not in _FIT_SPLITS:
            raise APNTestAccessError("Fit loader accepts only train and val")
        return self._get(split, purpose)

    def build_test_loader(
        self,
        *,
        test_ledger_token: FrozenAPNTestLedgerToken,
        purpose: Purpose = "extraction",
    ) -> Any:
        if not isinstance(test_ledger_token, FrozenAPNTestLedgerToken):
            raise APNTestAccessError("Released test construction requires a frozen token")
        test_ledger_token.validate_for(self.bundle.dataset_id, self.bundle.protocol_id)
        return self._get("test", purpose)

    def contract(self, split: AnySplit) -> LoaderContract:
        return LoaderContract(
            dataset_id=self.bundle.dataset_id,
            protocol_id=self.bundle.protocol_id,
            kind="released_code_behavior",
            requested_split=split,
            value_fit_scope=(
                "released_wrapper; P12 standardization includes test; dataset-specific"
            ),
            padding_fit_scope="all_partitions_including_test_during_non_val_construction",
            scans_test_during_train_construction=True,
            sample_identity_quality="released_sample_key; not reliable for crossed inference",
            padded_history_steps=_optional_positive_int(
                getattr(self.bundle.config, "seq_len_max_irr", None)
            ),
            padded_prediction_steps=_optional_positive_int(
                getattr(self.bundle.config, "pred_len_max_irr", None)
            ),
        )


def _sample_parts(sample: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    try:
        inputs = sample.inputs
        time = inputs.t
        values = inputs.x
        target_time = inputs.t_target
        targets = sample.targets
        key = int(sample.key)
    except (AttributeError, TypeError, ValueError) as exc:
        raise APNBridgeError("Strict dataset sample is not APN TaskDataset-compatible") from exc
    tensors = (time, values, target_time, targets)
    if any(not isinstance(item, Tensor) for item in tensors):
        raise APNBridgeError("Strict APN sample fields must be tensors")
    return time, values, target_time, targets, key


class ProjectOwnedIrregularCollator:
    """APN-compatible collator with train+validation-only padding metadata."""

    def __init__(self, *, history_steps: int, prediction_steps: int, channels: int) -> None:
        if min(history_steps, prediction_steps, channels) <= 0:
            raise ValueError("Strict padding dimensions must be positive")
        self.history_steps = int(history_steps)
        self.prediction_steps = int(prediction_steps)
        self.channels = int(channels)

    def __call__(self, samples: list[Any]) -> dict[str, Tensor]:
        if not samples:
            raise APNBridgeError("Cannot collate an empty strict APN batch")
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        x_marks: list[Tensor] = []
        y_marks: list[Tensor] = []
        x_masks: list[Tensor] = []
        y_masks: list[Tensor] = []
        keys: list[int] = []
        for sample in samples:
            time, values, target_time, targets, key = _sample_parts(sample)
            if values.ndim != 2 or targets.ndim != 2:
                raise APNBridgeError("Strict APN values/targets must be rank two")
            if values.shape[1] != self.channels or targets.shape[1] != self.channels:
                raise APNBridgeError("Strict APN sample channel count differs from config")
            if len(time) > self.history_steps or len(target_time) > self.prediction_steps:
                raise APNBridgeError(
                    "A strict sample exceeds padding limits frozen from train+validation"
                )
            xs.append(values)
            ys.append(targets)
            x_marks.append(time)
            y_marks.append(target_time)
            x_masks.append(values.isfinite())
            y_masks.append(targets.isfinite())
            keys.append(key)

        # Sentinels force a constant shape without consulting the test split.
        xs.append(torch.zeros(self.history_steps, self.channels))
        ys.append(torch.zeros(self.prediction_steps, self.channels))
        x_marks.append(torch.zeros(self.history_steps))
        y_marks.append(torch.zeros(self.prediction_steps))
        x_masks.append(torch.zeros(self.history_steps, self.channels, dtype=torch.bool))
        y_masks.append(torch.zeros(self.prediction_steps, self.channels, dtype=torch.bool))
        x = pad_sequence(xs, batch_first=True, padding_value=float("nan"))[:-1]
        y = pad_sequence(ys, batch_first=True, padding_value=float("nan"))[:-1]
        x_mark = pad_sequence(x_marks, batch_first=True, padding_value=0.0)[:-1]
        y_mark = pad_sequence(y_marks, batch_first=True, padding_value=0.0)[:-1]
        x_mask = pad_sequence(x_masks, batch_first=True, padding_value=False)[:-1]
        y_mask = pad_sequence(y_masks, batch_first=True, padding_value=False)[:-1]
        sample_ids = torch.tensor(keys, dtype=torch.int64)
        return {
            "x": torch.nan_to_num(x),
            "x_mark": x_mark.unsqueeze(-1).float(),
            "x_mask": x_mask.float(),
            "y": torch.nan_to_num(y),
            "y_mark": y_mark.unsqueeze(-1).float(),
            "y_mask": y_mask.float(),
            "sample_ID": sample_ids,
            "group_ID": sample_ids.clone(),
        }


def _padding_limits(datasets: Iterable[Dataset[Any]], channels: int) -> tuple[int, int]:
    history = 0
    prediction = 0
    samples_seen = 0
    for dataset in datasets:
        for index in range(len(dataset)):
            time, values, target_time, targets, _ = _sample_parts(dataset[index])
            if values.shape[-1] != channels or targets.shape[-1] != channels:
                raise APNBridgeError("Strict train/val sample channel count differs from config")
            history = max(history, len(time))
            prediction = max(prediction, len(target_time))
            samples_seen += 1
    if samples_seen == 0 or history <= 0 or prediction <= 0:
        raise APNBridgeError("Strict train+validation partitions cannot freeze padding")
    return history, prediction


class StrictLoaderFactory:
    """Train/validation-only loader for a prepared leakage-controlled dataset."""

    def __init__(
        self,
        bundle: APNConfigBundle,
        prepared: PreparedStrictDataset,
    ) -> None:
        if bundle.protocol_id == "release_parity":
            raise APNBridgeError("StrictLoaderFactory rejects release_parity")
        self.bundle = bundle
        self.prepared = prepared
        if str(prepared.dataset_id).lower() != bundle.dataset_id.lower():
            raise APNBridgeError("Prepared strict dataset ID differs from APN config")
        self._datasets: dict[str, Dataset[Any]] = {
            split: prepared.build_dataset(split) for split in _FIT_SPLITS
        }
        channels = int(bundle.config.enc_in)
        history, prediction = _padding_limits(self._datasets.values(), channels)
        bundle.config.seq_len_max_irr = history
        bundle.config.pred_len_max_irr = prediction
        self.collator = ProjectOwnedIrregularCollator(
            history_steps=history,
            prediction_steps=prediction,
            channels=channels,
        )
        self._loaders: dict[tuple[str, str], Any] = {}

    def _loader(self, split: AnySplit, purpose: Purpose, dataset: Dataset[Any]) -> DataLoader[Any]:
        key = (split, purpose)
        if key not in self._loaders:
            self._loaders[key] = DataLoader(
                dataset,
                batch_size=int(self.bundle.config.batch_size),
                shuffle=purpose == "training" and split == "train",
                num_workers=0,
                drop_last=purpose == "training" and split == "train",
                collate_fn=self.collator,
            )
        return self._loaders[key]

    def __call__(self, split: FitSplit, seed: int) -> Iterable[Any]:
        if split not in _FIT_SPLITS:
            raise APNTestAccessError("APN training loader factory accepts only train and val")
        if seed != self.bundle.seed:
            raise APNBridgeError("Loader seed differs from frozen APN config")
        return self._loader(split, "training", self._datasets[split])

    def build_fit_loader(self, split: FitSplit, *, purpose: Purpose = "extraction") -> Any:
        if split not in _FIT_SPLITS:
            raise APNTestAccessError("Fit loader accepts only train and val")
        return self._loader(split, purpose, self._datasets[split])

    def build_test_loader(
        self,
        *,
        test_ledger_token: object,
        purpose: Purpose = "extraction",
    ) -> Any:
        if test_ledger_token is None:
            raise APNTestAccessError("Strict test construction requires a frozen token")
        try:
            dataset = self.prepared.build_dataset(
                "test", ledger_token=test_ledger_token
            )
        except PermissionError as exc:
            raise APNTestAccessError(str(exc)) from exc
        return self._loader("test", purpose, dataset)

    def contract(self, split: AnySplit) -> LoaderContract:
        return LoaderContract(
            dataset_id=self.bundle.dataset_id,
            protocol_id=self.bundle.protocol_id,
            kind="leakage_controlled_project_owned",
            requested_split=split,
            value_fit_scope="observed_train_only_normalizer_from_prepared_protocol",
            padding_fit_scope="train_and_validation_only_before_test_opening",
            scans_test_during_train_construction=False,
            sample_identity_quality="privacy_safe_group_key_from_frozen_strict_split",
            padded_history_steps=int(self.collator.history_steps),
            padded_prediction_steps=int(self.collator.prediction_steps),
        )


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    return result if result > 0 else None


def load_frozen_apn_checkpoint(
    bundle: APNConfigBundle,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    model_factory: Callable[[int], nn.Module] | None = None,
) -> nn.Module:
    """Strictly load a weights-only APN checkpoint and freeze every parameter."""

    checkpoint = _resolve_project_path(checkpoint_path, must_exist=True)
    resolved_device = torch.device(device)
    model = (model_factory or make_apn_model_factory(bundle))(bundle.seed)
    state = torch.load(checkpoint, map_location=resolved_device, weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise APNBridgeError("APN checkpoint must be a non-empty state mapping")
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise APNBridgeError("APN checkpoint contains non-tensor state entries")
    model.load_state_dict(state, strict=True)
    model.to(resolved_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _private_id(
    dataset_id: str,
    protocol_id: str,
    split: str,
    raw_id: int,
    *,
    include_split: bool,
) -> int:
    parts = ["edgetwincal-apn-id-v1", dataset_id, protocol_id]
    if include_split:
        parts.append(split)
    parts.append(str(raw_id))
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@torch.no_grad()
def extract_apn_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    dataset_id: str,
    protocol_id: str,
    split: AnySplit,
    device: str | torch.device,
    test_ledger_token: object | None = None,
) -> dict[str, Tensor]:
    """Extract the frozen APN tensors consumed by :mod:`runtime_v2`."""

    if split not in _ALL_SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    if split == "test" and test_ledger_token is None:
        raise APNTestAccessError("APN test feature extraction requires a frozen token")
    resolved_device = torch.device(device)
    moved = _move_batch(batch, resolved_device)
    required = {"x", "x_mark", "x_mask", "y", "y_mark", "y_mask", "sample_ID"}
    missing = sorted(required.difference(moved))
    if missing:
        raise APNBridgeError(f"APN extraction batch is missing fields: {missing}")
    x = moved["x"]
    x_mark = moved["x_mark"]
    x_mask = moved["x_mask"]
    y = moved["y"]
    y_mark = moved["y_mark"]
    y_mask = moved["y_mask"]
    if not all(isinstance(value, Tensor) for value in (x, x_mark, x_mask, y, y_mark, y_mask)):
        raise APNBridgeError("APN extraction fields must be tensors")
    if x.ndim != 3 or x_mark.ndim != 3 or x_mask.shape != x.shape:
        raise APNBridgeError("APN history tensors must have shapes [B,L,C], [B,L,T], [B,L,C]")
    if y.ndim != 3 or y_mark.ndim != 3 or y_mask.shape != y.shape:
        raise APNBridgeError("APN target tensors must have shapes [B,H,C], [B,H,T], [B,H,C]")
    core = getattr(model, "model", None)
    if core is None or not all(
        hasattr(core, name) for name in ("LearnableTE", "IMTS_Model_Logic", "decoder")
    ):
        raise APNBridgeError("Model does not expose the pinned APN latent/decoder interface")
    batch_size, history_steps, channels = x.shape
    if y.shape[0] != batch_size or y.shape[2] != channels:
        raise APNBridgeError("APN history and target batch/channel dimensions differ")
    core.batch_size = batch_size
    history_time = x_mark[:, :, [0]]
    stacked_x = x.permute(0, 2, 1).reshape(batch_size * channels, history_steps, 1)
    stacked_mask = x_mask.permute(0, 2, 1).reshape(
        batch_size * channels, history_steps, 1
    )
    stacked_time = (
        history_time.repeat(1, 1, channels)
        .permute(0, 2, 1)
        .reshape(batch_size * channels, history_steps, 1)
    )
    history_encoding = core.LearnableTE(stacked_time)
    features = core.IMTS_Model_Logic(
        torch.cat([stacked_x, history_encoding], dim=-1),
        stacked_mask,
        stacked_time,
    )
    horizon = y_mark.shape[1]
    prediction_times = y_mark[:, :, [0]].view(batch_size, 1, horizon, 1).repeat(
        1, channels, 1, 1
    )
    time_encoding = core.LearnableTE(prediction_times)
    decoder_input = torch.cat(
        [features.unsqueeze(2).expand(-1, -1, horizon, -1), time_encoding], dim=-1
    )
    prediction = core.decoder(decoder_input).squeeze(-1).permute(0, 2, 1)
    if features.ndim != 3 or features.shape[:2] != (batch_size, channels):
        raise APNBridgeError("APN latent feature shape must be [B,C,D]")
    if time_encoding.ndim != 4 or time_encoding.shape[:3] != (
        batch_size,
        channels,
        horizon,
    ):
        raise APNBridgeError("APN prediction time encoding shape must be [B,C,H,E]")
    if prediction.shape != y.shape:
        raise APNBridgeError(
            f"APN prediction/target shapes differ: {prediction.shape} != {y.shape}"
        )
    raw_sample = moved["sample_ID"]
    if not isinstance(raw_sample, Tensor) or raw_sample.numel() != batch_size:
        raise APNBridgeError("sample_ID must contain exactly one value per sample")
    raw_sample_ids = [int(value) for value in raw_sample.reshape(-1).tolist()]
    raw_group = moved.get("group_ID", raw_sample)
    if not isinstance(raw_group, Tensor) or raw_group.numel() != batch_size:
        raise APNBridgeError("group_ID must contain exactly one value per sample")
    raw_group_ids = [int(value) for value in raw_group.reshape(-1).tolist()]
    sample_ids = torch.tensor(
        [
            _private_id(dataset_id, protocol_id, split, value, include_split=True)
            for value in raw_sample_ids
        ],
        dtype=torch.int64,
    )
    group_ids = torch.tensor(
        [
            _private_id(dataset_id, protocol_id, split, value, include_split=False)
            for value in raw_group_ids
        ],
        dtype=torch.int64,
    )
    return {
        "features": features.detach().cpu().contiguous(),
        "time_encoding": time_encoding.detach().cpu().contiguous(),
        "base_prediction": prediction.detach().cpu().contiguous(),
        "target": y.detach().cpu().contiguous(),
        "mask": (y_mask > 0).detach().cpu().contiguous(),
        "sample_id": sample_ids,
        "group_id": group_ids,
    }


@torch.no_grad()
def collect_apn_split(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    protocol_id: str,
    split: AnySplit,
    device: str | torch.device,
    test_ledger_token: object | None = None,
) -> dict[str, Tensor]:
    """Collect one split without exposing raw sample or group identifiers."""

    if split == "test" and test_ledger_token is None:
        raise APNTestAccessError("APN test collection requires a frozen token")
    model.eval()
    parts: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "features",
            "time_encoding",
            "base_prediction",
            "target",
            "mask",
            "sample_id",
            "group_id",
        )
    }
    for batch in loader:
        extracted = extract_apn_batch(
            model,
            batch,
            dataset_id=dataset_id,
            protocol_id=protocol_id,
            split=split,
            device=device,
            test_ledger_token=test_ledger_token,
        )
        for name, tensor in extracted.items():
            parts[name].append(tensor)
    if not parts["sample_id"]:
        raise APNBridgeError(f"APN {split} loader produced no batches")
    try:
        result = {name: torch.cat(values, dim=0) for name, values in parts.items()}
    except RuntimeError as exc:
        raise APNBridgeError(
            "APN batches have inconsistent frozen tensor shapes; check padding manifest"
        ) from exc
    ids = result["sample_id"].tolist()
    if len(set(ids)) != len(ids):
        raise APNBridgeError("Pseudonymous APN sample IDs collide or repeat within split")
    return result


def collect_apn_fit_splits(
    model: nn.Module,
    loader_factory: ReleaseParityLoaderFactory | StrictLoaderFactory,
    *,
    device: str | torch.device,
) -> dict[str, dict[str, Tensor]]:
    """Collect exactly train and validation; this API has no test argument."""

    return {
        split: collect_apn_split(
            model,
            loader_factory.build_fit_loader(split, purpose="extraction"),
            dataset_id=loader_factory.bundle.dataset_id,
            protocol_id=loader_factory.bundle.protocol_id,
            split=split,
            device=device,
        )
        for split in _FIT_SPLITS
    }


@torch.no_grad()
def smoke_apn_fit_loaders(
    model: nn.Module,
    loader_factory: ReleaseParityLoaderFactory | StrictLoaderFactory,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Run one forward-only train batch and one val batch, never test.

    This is the pre-training integration smoke. Its signature deliberately has
    no split or test-token parameter, so it cannot open the test partition.
    """

    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    summaries: dict[str, Any] = {}
    for split in _FIT_SPLITS:
        loader = loader_factory.build_fit_loader(split, purpose="extraction")
        try:
            raw_batch = next(iter(loader))
        except StopIteration as exc:
            raise APNBridgeError(f"APN {split} loader is empty") from exc
        if not isinstance(raw_batch, Mapping):
            raise APNBridgeError(f"APN {split} loader must return mapping batches")
        batch = _move_batch(raw_batch, resolved_device)
        output = apn_forward_callback(model, batch, split)
        prediction = output.get("pred")
        target = output.get("true")
        mask = output.get("mask")
        if not all(isinstance(value, Tensor) for value in (prediction, target, mask)):
            raise APNBridgeError("APN smoke forward must expose pred/true/mask tensors")
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise APNBridgeError("APN smoke prediction/target/mask shapes differ")
        observed = int((mask > 0).sum().item())
        if observed <= 0:
            raise APNBridgeError(f"APN {split} smoke batch has no observed target")
        finite_prediction = bool(torch.isfinite(prediction).all())
        if not finite_prediction:
            raise APNBridgeError(f"APN {split} smoke prediction is non-finite")
        summaries[split] = {
            "batch_size": int(prediction.shape[0]),
            "prediction_shape": list(prediction.shape),
            "observed_targets": observed,
            "finite_prediction": finite_prediction,
            "loader_contract": loader_factory.contract(split).public_manifest(),
        }
    return {
        "schema_version": "edgetwincal.apn_fit_smoke.v1",
        "dataset_id": loader_factory.bundle.dataset_id,
        "protocol_id": loader_factory.bundle.protocol_id,
        "seed": loader_factory.bundle.seed,
        "splits_opened": ["train", "val"],
        "test_opened": False,
        "splits": summaries,
    }


__all__ = [
    "APNBridgeError",
    "APNConfigBundle",
    "APNTestAccessError",
    "FrozenAPNTestLedgerToken",
    "LoaderContract",
    "ProjectOwnedIrregularCollator",
    "ReleaseParityLoaderFactory",
    "StrictLoaderFactory",
    "apn_forward_callback",
    "build_vendor_config",
    "collect_apn_fit_splits",
    "collect_apn_split",
    "configure_vendor_environment",
    "extract_apn_batch",
    "load_frozen_apn_checkpoint",
    "make_apn_model_factory",
    "smoke_apn_fit_loaders",
]
