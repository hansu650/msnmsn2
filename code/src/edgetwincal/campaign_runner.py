"""Frozen-registry campaign runner for one EdgeTwinCal dataset/checkpoint cell.

The runner deliberately separates the two data phases.  It fits every selected
variant from a schema-3 cache containing exactly ``train`` and ``val`` before it
accepts a once-only test token or reads a test cache.  The test cache must then
contain exactly ``test`` and must match all frozen provenance fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .campaign import ProtocolLedger, ProtocolLedgerError
from .config import ResolvedConfig, canonical_sha256, validate_config
from .controls import fit_bias_only, fit_self_affine_with_validation
from .decoder_refit import DecoderRefitResult, fit_decoder_only
from .graph import CrossForecastGraph, fit_graph_with_validation
from .joint import JointResidualRidge, fit_joint_with_validation
from .latent import SensorLatentResidualHead, fit_latent_head_with_validation
from .paths import PROJECT_ROOT, require_within_root
from .provenance import CacheManifest
from .protocol import salted_identifier_hash
from .runtime_v2 import (
    FIT_SPLITS,
    TEST_SPLITS,
    collect_environment,
    deserialize_tensor_payload,
    file_sha256,
    read_tensor_cache,
    resolve_run_assets,
)
from .schema import RunManifest, atomic_write_json, sha256_file
from .shuffle import shuffle_cross_forecasts, shuffle_latent_features
from .statistics import error_cells_from_arrays, pooled_metrics
from .timing import PhaseTimer, serialized_state_bytes, warm_inference
from .variants import (
    SequentialAdapter,
    fit_diagonal_from_frozen_latent,
    fit_full_with_validation,
    fit_reverse_with_validation,
)


REGISTRY_SCHEMA = "edgetwincal.variant-registry.v1"
FITTED_REGISTRY_SCHEMA = "edgetwincal.fitted-variant-registry.v1"
PRETEST_PROTOCOL_SCHEMA = "edgetwincal.pretest-protocol.v1"
EVALUATION_SCHEMA = "edgetwincal.cell-evaluation.v3"
TIMING_SCHEMA = "edgetwincal.timing.v3"
MAIN_VARIANTS = ("APN", "SLRH", "CFG", "Full")
STRICT_CONTROL_VARIANTS = ("V01", "V02", "V03", "V07", "V08", "V10", "V11", "V12")
STRICT_CONTROL_PROTOCOLS = frozenset({"strict_p12", "strict_ushcn"})


class CampaignRunnerError(ValueError):
    """A frozen campaign cell cannot be evaluated without protocol drift."""


@dataclass(frozen=True)
class PreparedManifestSet:
    protocol_manifest: Path
    split_manifest: Path
    normalizer_manifest: Path
    protocol_manifest_sha256: str
    split_manifest_sha256: str
    normalizer_manifest_sha256: str

def _json_mapping(path: str | Path, *, label: str) -> Mapping[str, Any]:
    source = require_within_root(path, must_exist=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignRunnerError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CampaignRunnerError(f"{label} must be a JSON object")
    return value


def load_cache_manifest(path: str | Path) -> CacheManifest:
    """Load an explicit cache sidecar; cache discovery is intentionally absent."""

    value = _json_mapping(path, label="cache manifest")
    try:
        return CacheManifest.from_dict(value)
    except Exception as exc:
        raise CampaignRunnerError(f"Invalid cache manifest: {exc}") from exc


def write_cache_manifest(path: str | Path, manifest: CacheManifest) -> Path:
    if not isinstance(manifest, CacheManifest):
        raise TypeError("manifest must be CacheManifest")
    return atomic_write_json(path, manifest.to_dict())


def resolved_config_from_ledger(ledger: ProtocolLedger) -> ResolvedConfig:
    raw = ledger.data.get("resolved_config")
    if not isinstance(raw, Mapping):
        raise CampaignRunnerError("Ledger has no resolved configuration")
    validate_config(raw)
    digest = canonical_sha256(raw)
    if digest != ledger.data.get("resolved_config_sha256"):
        raise CampaignRunnerError("Ledger resolved configuration hash mismatch")
    return ResolvedConfig(dict(raw), digest, ledger.path)


def selected_variant_ids(
    config: ResolvedConfig, dataset_id: str, protocol_id: str
) -> tuple[str, ...]:
    if dataset_id not in config["selection"]["datasets"]:
        raise CampaignRunnerError(f"Dataset is not selected: {dataset_id}")
    selected = tuple(str(value) for value in config["selection"]["variants"])
    required = MAIN_VARIANTS + (
        STRICT_CONTROL_VARIANTS if protocol_id in STRICT_CONTROL_PROTOCOLS else ()
    )
    missing = [variant for variant in required if variant not in selected]
    if missing:
        raise CampaignRunnerError(
            f"Frozen selection omits required variants for {protocol_id}: {missing}"
        )
    return required


def variant_registry(
    config: ResolvedConfig, dataset_id: str, protocol_id: str
) -> dict[str, Any]:
    variants = selected_variant_ids(config, dataset_id, protocol_id)
    payload = {
        "schema_version": REGISTRY_SCHEMA,
        "resolved_config_sha256": config.sha256,
        "dataset": dataset_id,
        "protocol": protocol_id,
        "execution": "sequential_single_process_single_device",
        "variants": [
            {"variant_id": variant, "definition": dict(config["variants"][variant])}
            for variant in variants
        ],
    }
    return {**payload, "variant_registry_sha256": canonical_sha256(payload)}


def write_variant_registry(
    path: str | Path,
    config: ResolvedConfig,
    dataset_id: str,
    protocol_id: str,
) -> Path:
    return atomic_write_json(path, variant_registry(config, dataset_id, protocol_id))


def _registry_digest(registry: Mapping[str, Any]) -> str:
    content = dict(registry)
    recorded = content.pop("variant_registry_sha256", None)
    calculated = canonical_sha256(content)
    if recorded != calculated:
        raise CampaignRunnerError("Variant registry self-hash mismatch")
    return calculated


def _assert_phase_manifest(manifest: CacheManifest, phase: str) -> None:
    split_names = {name.split(".", 1)[0] for name in manifest.shapes}
    expected = set(FIT_SPLITS if phase == "fit" else TEST_SPLITS)
    if split_names != expected:
        raise CampaignRunnerError(
            f"{phase} cache split set must be {sorted(expected)}, got {sorted(split_names)}"
        )


_PHASE_VARYING_CACHE_FIELDS = frozenset(
    {"shapes", "dtypes", "mask_sha256", "sample_ids_sha256", "group_ids_sha256"}
)


def _assert_cache_pair(fit: CacheManifest, test: CacheManifest) -> None:
    fit_fields = fit.key_dict()
    test_fields = test.key_dict()
    mismatches = {
        name: (fit_fields[name], test_fields[name])
        for name in fit_fields
        if name not in _PHASE_VARYING_CACHE_FIELDS and fit_fields[name] != test_fields[name]
    }
    if mismatches:
        raise CampaignRunnerError(
            f"Fit/test cache provenance differs: {sorted(mismatches)}"
        )


def _assert_tensor_phase(splits: Mapping[str, Any], expected: tuple[str, ...]) -> None:
    if tuple(splits) != expected:
        raise CampaignRunnerError(
            f"Cache payload splits must be exactly {list(expected)}, got {list(splits)}"
        )


def _strict_protocol(protocol_id: str) -> bool:
    return protocol_id != "release_parity"


def _assert_cross_phase_disjoint(
    fit: Mapping[str, Mapping[str, Tensor]],
    test: Mapping[str, Mapping[str, Tensor]],
    *,
    strict: bool,
) -> None:
    fit_sample = set(int(value) for split in FIT_SPLITS for value in fit[split]["sample_id"].tolist())
    test_sample = set(int(value) for value in test["test"]["sample_id"].tolist())
    if fit_sample.intersection(test_sample):
        raise CampaignRunnerError("Sample IDs overlap between fit and test caches")
    if strict:
        fit_group = set(int(value) for split in FIT_SPLITS for value in fit[split]["group_id"].tolist())
        test_group = set(int(value) for value in test["test"]["group_id"].tolist())
        if fit_group.intersection(test_group):
            raise CampaignRunnerError("Strict group IDs overlap between fit and test caches")


class _CachedDecoderCore(nn.Module):
    def __init__(self, d_model: int, te_dim: int, dropout: float) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model + te_dim, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )


class CachedAPNDecoder(nn.Module):
    """Exact APN decoder topology driven by frozen cached latent/time tensors."""

    def __init__(self, d_model: int, te_dim: int, dropout: float) -> None:
        super().__init__()
        self.model = _CachedDecoderCore(d_model, te_dim, dropout)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        features = inputs["features"]
        time_encoding = inputs["time_encoding"]
        if features.ndim != 3 or time_encoding.ndim != 4:
            raise CampaignRunnerError("Cached decoder inputs have invalid ranks")
        if time_encoding.shape[:2] != features.shape[:2]:
            raise CampaignRunnerError("Cached decoder batch/channel dimensions differ")
        horizon = time_encoding.shape[2]
        decoder_input = torch.cat(
            [features.unsqueeze(2).expand(-1, -1, horizon, -1), time_encoding], dim=-1
        )
        return self.model.decoder(decoder_input).squeeze(-1).permute(0, 2, 1)


def _checkpoint_state(path: Path) -> Mapping[str, Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CampaignRunnerError(f"Cannot safely load APN checkpoint: {exc}") from exc
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise CampaignRunnerError("APN checkpoint is not a non-empty state mapping")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise CampaignRunnerError("APN checkpoint contains non-tensor state entries")
    return state


def _decoder_state_from_checkpoint(path: Path) -> dict[str, Tensor]:
    state = _checkpoint_state(path)
    wanted = ("0.weight", "0.bias", "3.weight", "3.bias")
    result: dict[str, Tensor] = {}
    for suffix in wanted:
        matches = [
            value for name, value in state.items() if name.endswith(f"decoder.{suffix}")
        ]
        if len(matches) != 1:
            raise CampaignRunnerError(
                f"Checkpoint must contain exactly one decoder.{suffix}; found {len(matches)}"
            )
        result[suffix] = matches[0].detach().cpu().contiguous().clone()
    return result


@dataclass(frozen=True)
class DecoderVariant:
    model: CachedAPNDecoder
    audit: Any

    def apply(self, split: Mapping[str, Tensor], *, device: str) -> Tensor:
        resolved = torch.device(device)
        self.model.to(resolved).eval()
        with torch.no_grad():
            prediction = self.model(
                {
                    "features": split["features"].to(resolved),
                    "time_encoding": split["time_encoding"].to(resolved),
                }
            )
        return prediction.detach().cpu()

    def state_dict(self) -> dict[str, Any]:
        if isinstance(self.audit, Mapping):
            selected = self.audit.get("selected_config", self.audit)
            initial_frozen_hash = self.audit.get("initial_frozen_hash", "")
            final_frozen_hash = self.audit.get("final_frozen_hash", "")
        else:
            selected = self.audit.selected_config
            initial_frozen_hash = self.audit.initial_frozen_hash
            final_frozen_hash = self.audit.final_frozen_hash
        return {
            "decoder": {
                name: value.detach().cpu().contiguous()
                for name, value in self.model.state_dict().items()
                if ".decoder." in name
            },
            "selected_config": dict(selected),
            "initial_frozen_hash": str(initial_frozen_hash),
            "final_frozen_hash": str(final_frozen_hash),
        }


@dataclass(frozen=True)
class CrossShuffleVariant:
    latent: SensorLatentResidualHead
    graph: CrossForecastGraph
    descriptor: str

    def apply(self, split: Mapping[str, Tensor]) -> Tensor:
        source, _, _ = shuffle_cross_forecasts(
            split["base_prediction"],
            split["sample_id"].tolist(),
            descriptor=self.descriptor,
        )
        anchor = self.latent.apply(split["base_prediction"], split["features"])
        return self.graph.apply(anchor, source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor,
            "latent": self.latent.state_dict(),
            "graph": self.graph.state_dict(),
        }


@dataclass(frozen=True)
class LatentShuffleVariant:
    latent: SensorLatentResidualHead
    descriptor: str

    def apply(self, split: Mapping[str, Tensor]) -> Tensor:
        latent, _, _ = shuffle_latent_features(
            split["features"],
            split["sample_id"].tolist(),
            descriptor=self.descriptor,
        )
        return self.latent.apply(split["base_prediction"], latent)

    def state_dict(self) -> dict[str, Any]:
        return {"descriptor": self.descriptor, "latent": self.latent.state_dict()}


@dataclass(frozen=True)
class FittedVariants:
    dataset_id: str
    protocol_id: str
    seed: int
    variant_ids: tuple[str, ...]
    states: Mapping[str, Any]
    audits: Mapping[str, Any]
    timing: tuple[Mapping[str, Any], ...]
    device: str


def _decoder_batches(split: Mapping[str, Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "features": split["features"],
                "time_encoding": split["time_encoding"],
            },
            "target": split["target"],
            "mask": split["mask"],
        }
    ]


def _fit_decoder_variant(
    config: ResolvedConfig,
    dataset_id: str,
    fit: Mapping[str, Mapping[str, Tensor]],
    checkpoint_path: Path,
    *,
    device: str,
    seed: int,
) -> DecoderVariant:
    dataset = config["datasets"][dataset_id]
    hyper = dataset["apn_hyperparameters"]
    model = CachedAPNDecoder(
        int(hyper["d_model"]), int(hyper["te_dim"]), float(hyper["dropout"])
    )
    decoder_state = _decoder_state_from_checkpoint(checkpoint_path)
    model.model.decoder.load_state_dict(decoder_state, strict=True)
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    atol = float(config["gates"]["G0"]["legacy_parity_atol"])
    rtol = float(config["gates"]["G0"]["legacy_parity_rtol"])
    parity_batch_size = int(hyper["batch_size"])
    with torch.no_grad():
        for split_name in FIT_SPLITS:
            split = fit[split_name]
            actual = torch.cat(
                [
                    model(
                        {
                            "features": split["features"][start : start + parity_batch_size].to(resolved_device),
                            "time_encoding": split["time_encoding"][start : start + parity_batch_size].to(resolved_device),
                        }
                    )
                    for start in range(0, split["features"].shape[0], parity_batch_size)
                ],
                dim=0,
            )
            expected = split["base_prediction"].to(resolved_device)
            if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
                maximum = float((actual - expected).abs().max().item())
                raise CampaignRunnerError(
                    f"Cached APN decoder parity failed on {split_name}; max_abs={maximum}"
                )
    refit = config["decoder_refit"]
    devices: list[int] = []
    if device.startswith("cuda"):
        devices = [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        audit = fit_decoder_only(
            model,
            _decoder_batches(fit["train"]),
            _decoder_batches(fit["val"]),
            learning_rates=refit["learning_rates"],
            weight_decays=refit["weight_decays"],
            max_epochs=int(refit["max_epochs"]),
            patience=int(refit["patience"]),
            device=device,
            decoder_prefixes=("model.decoder.",),
        )
    return DecoderVariant(model, audit)


def fit_variants(
    config: ResolvedConfig,
    dataset_id: str,
    protocol_id: str,
    seed: int,
    fit_splits: Mapping[str, Mapping[str, Tensor]],
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> FittedVariants:
    """Fit the frozen registry sequentially using only train and validation."""

    _assert_tensor_phase(fit_splits, FIT_SPLITS)
    if device not in {"cpu", "cuda", "cuda:0"}:
        raise CampaignRunnerError("Campaign device must be cpu, cuda, or cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CampaignRunnerError("CUDA was requested but is unavailable")
    variants = selected_variant_ids(config, dataset_id, protocol_id)
    alpha = tuple(float(value) for value in config["ridge"]["alpha_grid"])
    joint_alpha = tuple(float(value) for value in config["ridge"]["joint_alpha_grid"])
    train, val = fit_splits["train"], fit_splits["val"]
    states: dict[str, Any] = {"APN": None}
    audits: dict[str, Any] = {"APN": {"selected": "frozen_checkpoint"}}
    timer = PhaseTimer()

    with timer.phase("slrh_solve", device="cpu"):
        states["SLRH"], audits["SLRH"] = fit_latent_head_with_validation(
            train["base_prediction"], train["features"], train["target"], train["mask"],
            val["base_prediction"], val["features"], val["target"], val["mask"],
            alphas=alpha,
        )
    with timer.phase("cfg_solve", device="cpu"):
        states["CFG"], audits["CFG"] = fit_graph_with_validation(
            train["base_prediction"], train["target"], train["mask"],
            val["base_prediction"], val["target"], val["mask"], alphas=alpha,
            train_source_forecasts=train["base_prediction"],
            val_source_forecasts=val["base_prediction"], include_self=False,
        )
    with timer.phase("validation_selection", device="cpu"):
        states["Full"], audits["Full"] = fit_full_with_validation(
            train["base_prediction"], train["features"], train["target"], train["mask"],
            val["base_prediction"], val["features"], val["target"], val["mask"],
            latent_alphas=alpha, graph_alphas=alpha,
            train_graph_source=train["base_prediction"],
            val_graph_source=val["base_prediction"], include_self=False,
        )

    if protocol_id in STRICT_CONTROL_PROTOCOLS:
        states["V01"] = fit_bias_only(
            train["base_prediction"], train["target"], train["mask"]
        )
        audits["V01"] = {"selected": "closed_form"}
        states["V02"], audits["V02"] = fit_self_affine_with_validation(
            train["base_prediction"], train["target"], train["mask"],
            val["base_prediction"], val["target"], val["mask"], alphas=alpha,
        )
        if checkpoint_path is None:
            raise CampaignRunnerError("Strict V03 requires an explicit APN checkpoint")
        checkpoint = require_within_root(checkpoint_path, must_exist=True)
        with timer.phase("decoder_refit", device=device):
            states["V03"] = _fit_decoder_variant(
                config, dataset_id, fit_splits, checkpoint, device=device, seed=seed
            )
        audits["V03"] = {
            "selected_config": dict(states["V03"].audit.selected_config),
            "curve": list(states["V03"].audit.curve),
        }
        states["V07"], audits["V07"] = fit_reverse_with_validation(
            train["base_prediction"], train["features"], train["target"], train["mask"],
            val["base_prediction"], val["features"], val["target"], val["mask"],
            latent_alphas=alpha, graph_alphas=alpha,
        )
        states["V08"], audits["V08"] = fit_joint_with_validation(
            train["base_prediction"], train["features"], train["base_prediction"],
            train["target"], train["mask"], val["base_prediction"], val["features"],
            val["base_prediction"], val["target"], val["mask"],
            latent_alphas=joint_alpha, graph_alphas=joint_alpha, include_self=False,
        )
        states["V10"], audits["V10"] = fit_diagonal_from_frozen_latent(
            states["Full"].latent,
            train["base_prediction"], train["features"], train["target"], train["mask"],
            val["base_prediction"], val["features"], val["target"], val["mask"],
            graph_alphas=alpha,
        )
        cross_descriptor = f"{config.sha256}|{dataset_id}|{protocol_id}|{seed}|V11"
        train_cross, train_audit, _ = shuffle_cross_forecasts(
            train["base_prediction"], train["sample_id"].tolist(), descriptor=cross_descriptor
        )
        val_cross, val_audit, _ = shuffle_cross_forecasts(
            val["base_prediction"], val["sample_id"].tolist(), descriptor=cross_descriptor
        )
        train_anchor = states["Full"].latent.apply(train["base_prediction"], train["features"])
        val_anchor = states["Full"].latent.apply(val["base_prediction"], val["features"])
        graph, graph_audit = fit_graph_with_validation(
            train_anchor, train["target"], train["mask"], val_anchor, val["target"], val["mask"],
            alphas=alpha, train_source_forecasts=train_cross, val_source_forecasts=val_cross,
            include_self=False,
        )
        states["V11"] = CrossShuffleVariant(states["Full"].latent, graph, cross_descriptor)
        audits["V11"] = {
            "train_shuffle": train_audit.as_dict(),
            "validation_shuffle": val_audit.as_dict(),
            "cfg": graph_audit,
        }
        latent_descriptor = f"{config.sha256}|{dataset_id}|{protocol_id}|{seed}|V12"
        train_latent, train_latent_audit, _ = shuffle_latent_features(
            train["features"], train["sample_id"].tolist(), descriptor=latent_descriptor
        )
        val_latent, val_latent_audit, _ = shuffle_latent_features(
            val["features"], val["sample_id"].tolist(), descriptor=latent_descriptor
        )
        shuffled_head, shuffled_audit = fit_latent_head_with_validation(
            train["base_prediction"], train_latent, train["target"], train["mask"],
            val["base_prediction"], val_latent, val["target"], val["mask"], alphas=alpha,
        )
        states["V12"] = LatentShuffleVariant(shuffled_head, latent_descriptor)
        audits["V12"] = {
            "train_shuffle": train_latent_audit.as_dict(),
            "validation_shuffle": val_latent_audit.as_dict(),
            "slrh": shuffled_audit,
        }

    if tuple(states) != variants:
        raise CampaignRunnerError(
            f"Fitted variant order differs from registry: {tuple(states)} != {variants}"
        )
    return FittedVariants(
        dataset_id, protocol_id, int(seed), variants, states, audits,
        tuple(timer.as_dicts()), device,
    )


def _apply_variant(
    fitted: FittedVariants, variant: str, split: Mapping[str, Tensor]
) -> Tensor:
    state = fitted.states[variant]
    base, features = split["base_prediction"], split["features"]
    if variant == "APN":
        return base.clone()
    if variant == "SLRH":
        return state.apply(base, features)
    if variant == "CFG":
        return state.apply(base, base)
    if variant in {"Full", "V07", "V10"}:
        return state.apply(base, features, graph_source=base)
    if variant in {"V01", "V02"}:
        return state.apply(base)
    if variant == "V03":
        return state.apply(split, device=fitted.device)
    if variant == "V08":
        return state.apply(base, features, base)
    if variant in {"V11", "V12"}:
        return state.apply(split)
    raise CampaignRunnerError(f"No predictor for registered variant {variant}")


def _state_dict(variant: str, state: Any) -> Mapping[str, Any]:
    if variant == "APN":
        return {}
    value = state.state_dict()
    if not isinstance(value, Mapping):
        raise CampaignRunnerError(f"{variant} state_dict must return a mapping")
    return value


def _manifest_cache_reference(fit: CacheManifest, test: CacheManifest) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "fit_key_sha256": fit.digest(),
        "fit_manifest_sha256": fit.manifest_digest(),
        "fit_payload_sha256": fit.payload_sha256,
        "test_key_sha256": test.digest(),
        "test_manifest_sha256": test.manifest_digest(),
        "test_payload_sha256": test.payload_sha256,
    }


def _metrics_and_cells(
    split: Mapping[str, Tensor], prediction: Tensor, *, variant: str,
    checkpoint_sha256: str, salt: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_hashes = [salted_identifier_hash(int(value), salt) for value in split["group_id"].tolist()]
    cells = error_cells_from_arrays(
        group_hashes=group_hashes,
        checkpoint=checkpoint_sha256,
        variant=variant,
        prediction=prediction.detach().cpu().numpy(),
        target=split["target"].detach().cpu().numpy(),
        mask=split["mask"].detach().cpu().numpy(),
    )
    schema_cells = [
        {
            "group_hash": cell.group_hash,
            "checkpoint_sha256": cell.checkpoint,
            "variant": cell.variant,
            "sse": cell.sse,
            "sae": cell.sae,
            "n": cell.n,
        }
        for cell in cells
    ]
    metrics = pooled_metrics(cells)[variant]
    return metrics, schema_cells


def _ledger_bindings(
    ledger: ProtocolLedger,
    config: ResolvedConfig,
    cell_id: str,
    fit_manifest: CacheManifest,
    registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    if ledger.status != "test_active":
        raise CampaignRunnerError("Evaluation requires an active frozen test opening")
    if ledger.data.get("resolved_config_sha256") != config.sha256:
        raise CampaignRunnerError("Ledger/config digest mismatch")
    components = ledger.data.get("components")
    if not isinstance(components, Mapping):
        raise CampaignRunnerError("Ledger freeze components are absent")
    if components.get("variant_registry_sha256") != _registry_digest(registry):
        raise CampaignRunnerError("Frozen variant registry digest mismatch")
    opening = ledger.data.get("test_openings", {}).get(cell_id)
    if not isinstance(opening, Mapping):
        raise CampaignRunnerError("Requested test cell is not open")
    split_map = components.get("split_manifests")
    norm_map = components.get("normalization_manifests")
    if not isinstance(split_map, Mapping) or split_map.get(cell_id) != fit_manifest.split_manifest_sha256:
        raise CampaignRunnerError("Frozen split-manifest digest does not bind this cell")
    if not isinstance(norm_map, Mapping) or norm_map.get(cell_id) != fit_manifest.normalizer_manifest_sha256:
        raise CampaignRunnerError("Frozen normalizer-manifest digest does not bind this cell")
    if opening.get("split_manifest_sha256") != fit_manifest.split_manifest_sha256:
        raise CampaignRunnerError("Test opening split digest differs from fit provenance")
    if opening.get("normalization_manifest_sha256") != fit_manifest.normalizer_manifest_sha256:
        raise CampaignRunnerError("Test opening normalizer digest differs from fit provenance")
    return opening


def _legacy_single_seed_evaluate_disabled(
    *,
    ledger_path: str | Path,
    cell_id: str,
    token: str,
    fit_cache_path: str | Path,
    fit_cache_manifest_path: str | Path,
    test_cache_path: str | Path,
    test_cache_manifest_path: str | Path,
    variant_registry_path: str | Path,
    checkpoint_path: str | Path | None = None,
    run_root: str | Path | None = None,
    device: str = "cpu",
    argv: Sequence[str] = ("edgetwincal", "evaluate"),
) -> dict[str, Any]:
    """Run one cell.  Test files are untouched until all variants are fitted."""
    raise CampaignRunnerError(
        "Single-seed test consumption is disabled; use evaluate_campaign_once "
        "so every frozen checkpoint shares one ledger opening"
    )

    ledger = ProtocolLedger.load(ledger_path)
    config = resolved_config_from_ledger(ledger)
    registry = _json_mapping(variant_registry_path, label="variant registry")
    fit_manifest = load_cache_manifest(fit_cache_manifest_path)
    _assert_phase_manifest(fit_manifest, "fit")
    opening = _ledger_bindings(ledger, config, cell_id, fit_manifest, registry)
    dataset_id = str(opening["dataset"])
    protocol_id = str(opening["protocol"])
    seed = int(fit_manifest.seed)
    expected_registry = variant_registry(config, dataset_id, protocol_id)
    if dict(registry) != expected_registry:
        raise CampaignRunnerError("Variant registry content differs from frozen config")
    if fit_manifest.dataset_id != dataset_id or fit_manifest.resolved_config_sha256 != config.sha256:
        raise CampaignRunnerError("Fit cache identity differs from frozen cell")
    if checkpoint_path is not None:
        checkpoint = require_within_root(checkpoint_path, must_exist=True)
        if file_sha256(checkpoint) != fit_manifest.checkpoint_sha256:
            raise CampaignRunnerError("Explicit checkpoint hash differs from fit cache")
    else:
        checkpoint = None

    fit_timer = PhaseTimer()
    with fit_timer.phase("cache_read", device="cpu"):
        _, fit_splits = read_tensor_cache(fit_cache_path, fit_manifest)
    _assert_tensor_phase(fit_splits, FIT_SPLITS)
    fitted = fit_variants(
        config, dataset_id, protocol_id, seed, fit_splits,
        checkpoint_path=checkpoint, device=device,
    )

    # Token acceptance and every test-side read happen strictly after fitting.
    ledger.validate_test_token(cell_id=cell_id, token=token)
    test_attempted = False
    run_manifests: list[str] = []
    failures: list[dict[str, str]] = []
    try:
        test_attempted = True
        test_manifest = load_cache_manifest(test_cache_manifest_path)
        _assert_phase_manifest(test_manifest, "test")
        _assert_cache_pair(fit_manifest, test_manifest)
        with fit_timer.phase("cache_read", device="cpu"):
            _, test_splits = read_tensor_cache(test_cache_path, test_manifest)
        _assert_tensor_phase(test_splits, TEST_SPLITS)
        _assert_cross_phase_disjoint(
            fit_splits, test_splits, strict=_strict_protocol(protocol_id)
        )
        split = test_splits["test"]
        environment = collect_environment()
        output_root = require_within_root(
            run_root or Path(str(config["paths"]["run_root"]))
        )
        salt = f"edgetwincal-msn2026-result-v1:{dataset_id}:{protocol_id}"
        for variant in fitted.variant_ids:
            run_dir = require_within_root(
                output_root / dataset_id / protocol_id / f"seed_{seed}" / variant
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = require_within_root(run_dir / "run_log.json")
            manifest_path = require_within_root(run_dir / "run_manifest.json")
            manifest = RunManifest.create(
                manifest_path,
                run_id=f"{dataset_id}-{protocol_id}-{seed}-{variant}",
                dataset=dataset_id, protocol=protocol_id,
                fold=str(opening["fold"]), seed=seed, variant_id=variant,
                variant_definition=dict(config["variants"][variant]),
                resolved_config=config, argv=tuple(str(value) for value in argv),
                environment=environment, log_path=log_path,
            ).start()
            try:
                prediction_timer = PhaseTimer()
                with prediction_timer.phase("warm_inference", device=device):
                    prediction = _apply_variant(fitted, variant, split)
                metrics, cells = _metrics_and_cells(
                    split, prediction, variant=variant,
                    checkpoint_sha256=test_manifest.checkpoint_sha256, salt=salt,
                )
                state = _state_dict(variant, fitted.states[variant])
                selected = {
                    "audit": fitted.audits[variant],
                    "adapter_state_bytes": serialized_state_bytes(state),
                }
                timing_segments = [
                    *fit_timer.as_dicts(), *list(fitted.timing), *prediction_timer.as_dicts()
                ]
                log_payload = {
                    "schema_version": EVALUATION_SCHEMA,
                    "status": "complete",
                    "dataset": dataset_id,
                    "protocol": protocol_id,
                    "seed": seed,
                    "variant": variant,
                    "metrics": metrics,
                    "cache_schema_version": 3,
                    "test_access": {
                        "cell_id": cell_id,
                        "protocol_sha256": ledger.data["protocol_sha256"],
                        "token_persisted": False,
                    },
                }
                atomic_write_json(log_path, log_payload)
                manifest.mark_phase("fit_train_validation_only")
                manifest.mark_phase("once_opened_test_evaluation")
                manifest.complete(
                    assets={
                        "checkpoint_sha256": test_manifest.checkpoint_sha256,
                        "apn_commit": test_manifest.apn_commit,
                        "apn_patch_sha256": test_manifest.apn_patch_sha256,
                    },
                    cache_manifest=_manifest_cache_reference(fit_manifest, test_manifest),
                    split_manifest={
                        "schema_version": (
                            "edgetwincal.protocol.split.v1"
                            if _strict_protocol(protocol_id)
                            else "upstream.record-split.v1"
                        ),
                        "group_ids_reliable": _strict_protocol(protocol_id),
                        "group_id_hash": test_manifest.group_ids_sha256,
                        "split_manifest_sha256": test_manifest.split_manifest_sha256,
                    },
                    normalization_manifest={
                        "schema_version": "edgetwincal.protocol.normalizer.v1",
                        "normalizer_manifest_sha256": test_manifest.normalizer_manifest_sha256,
                    },
                    selected_hyperparameters=selected,
                    timing={"schema_version": TIMING_SCHEMA, "segments": timing_segments},
                    cells=cells,
                    metrics=metrics,
                    required_files=(log_path,),
                )
                run_manifests.append(manifest_path.relative_to(PROJECT_ROOT).as_posix())
            except Exception as exc:
                if not log_path.exists():
                    atomic_write_json(
                        log_path,
                        {
                            "schema_version": EVALUATION_SCHEMA,
                            "status": "failed",
                            "variant": variant,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                manifest.fail(exc, required_files=(log_path,))
                run_manifests.append(manifest_path.relative_to(PROJECT_ROOT).as_posix())
                failures.append({"variant": variant, "error": str(exc)})
    finally:
        if test_attempted:
            # Once any test artifact was touched the opening is consumed, even on failure.
            ledger.close_test(cell_id=cell_id, token=token)

    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete" if not failures else "failed",
        "dataset": dataset_id,
        "protocol": protocol_id,
        "seed": seed,
        "variant_registry_sha256": _registry_digest(registry),
        "run_manifests": run_manifests,
        "failures": failures,
        "test_opening_consumed": True,
    }


__all__ = [
    "CampaignRunnerError",
    "CachedAPNDecoder",
    "EVALUATION_SCHEMA",
    "FittedVariants",
    "MAIN_VARIANTS",
    "REGISTRY_SCHEMA",
    "STRICT_CONTROL_VARIANTS",
    "fit_variants",
    "load_cache_manifest",
    "resolved_config_from_ledger",
    "selected_variant_ids",
    "variant_registry",
    "write_cache_manifest",
    "write_variant_registry",
]
