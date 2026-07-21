"""Pre-test preparation and fitted-state freezing for EdgeTwinCal campaigns.

This module creates the three runtime manifests without constructing a test
dataset, fits all selected variants from explicit train+validation schema-3
caches, and persists only safe tensor/primitives.  Its aggregate registry hash
is the value bound to ``ProtocolLedger.components.variant_registry_sha256``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .campaign_runner import (
    FITTED_REGISTRY_SCHEMA,
    PRETEST_PROTOCOL_SCHEMA,
    CampaignRunnerError,
    CrossShuffleVariant,
    DecoderVariant,
    FittedVariants,
    LatentShuffleVariant,
    fit_variants,
    load_cache_manifest,
    selected_variant_ids,
    variant_registry,
)
from .config import ResolvedConfig, canonical_sha256
from .controls import ResidualBias, SelfAffineResidual
from .graph import CrossForecastGraph
from .joint import JointResidualRidge
from .latent import SensorLatentResidualHead
from .paths import PROJECT_ROOT, require_within_root
from .provenance import CacheManifest
from .runtime_v2 import FIT_SPLITS, file_sha256, read_tensor_cache, resolve_run_assets
from .schema import atomic_write_json, sha256_file
from .variants import SequentialAdapter


STATE_SCHEMA = 3
STATE_FILENAME = "fitted_states.pt"


@dataclass(frozen=True)
class PreparedManifestSet:
    protocol_manifest: Path
    split_manifest: Path
    normalizer_manifest: Path
    protocol_manifest_sha256: str
    split_manifest_sha256: str
    normalizer_manifest_sha256: str


def _write_frozen_json(path: Path, value: Mapping[str, Any]) -> Path:
    destination = require_within_root(path)
    if destination.exists():
        try:
            current = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignRunnerError(f"Existing frozen manifest is unreadable: {exc}") from exc
        if current != dict(value):
            raise CampaignRunnerError(f"Refusing to overwrite drifted frozen manifest: {destination}")
        return destination
    return atomic_write_json(destination, value)


def _privacy_audit(value: Any, *, location: str = "$") -> None:
    """Reject common raw-ID/path fields from caller-provided public bundles."""

    forbidden = {
        "raw_id",
        "raw_ids",
        "patient_id",
        "patient_ids",
        "station_id",
        "station_ids",
        "record_id",
        "record_ids",
        "subject_id",
        "subject_ids",
        "absolute_path",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if name in forbidden:
                raise CampaignRunnerError(f"Private field is forbidden in public manifest: {location}.{key}")
            _privacy_audit(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _privacy_audit(child, location=f"{location}[{index}]")
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if ":/" in normalized or normalized.startswith("/"):
            raise CampaignRunnerError(f"Absolute path is forbidden in public manifest at {location}")


def prepare_pretest_manifests(
    config: ResolvedConfig,
    *,
    dataset_id: str,
    protocol_id: str,
    seed: int,
    fold: str = "fold-0",
    strict_public_bundle: Mapping[str, Any] | None = None,
    padded_prediction_steps: int | None = None,
) -> PreparedManifestSet:
    """Create protocol/split/normalizer files before any test construction.

    Released P12/USHCN use fixed task contracts.  Released HumanActivity must
    supply the dynamic padding count frozen by its released loader contract.
    Strict P12/USHCN must supply ``PreparedStrict*.public_manifests()``.
    """

    assets = resolve_run_assets(
        config,
        dataset_id,
        seed,
        protocol_id,
        require_existing=False,
    )
    task = config["datasets"][dataset_id]["task"]
    if task["horizon_kind"] == "time_window":
        if isinstance(padded_prediction_steps, bool) or not isinstance(padded_prediction_steps, int) or padded_prediction_steps <= 0:
            raise CampaignRunnerError(
                "A positive padded_prediction_steps from the frozen loader contract is required"
            )
    elif padded_prediction_steps is not None and int(padded_prediction_steps) != int(task["prediction_steps"]):
        raise CampaignRunnerError("Fixed-step padding differs from the task contract")

    strict = protocol_id != "release_parity"
    if strict and protocol_id not in {"strict_p12", "strict_ushcn"}:
        raise CampaignRunnerError(
            "Pretest strict preparation currently supports strict_p12 and strict_ushcn"
        )
    if strict and strict_public_bundle is None:
        raise CampaignRunnerError("Strict preparation requires a public strict manifest bundle")
    if not strict and strict_public_bundle is not None:
        raise CampaignRunnerError("Released preparation rejects a strict manifest bundle")
    bundle = dict(strict_public_bundle or {})
    _privacy_audit(bundle)

    if strict:
        if not isinstance(bundle.get("split"), Mapping) or not isinstance(bundle.get("normalization"), Mapping):
            raise CampaignRunnerError("Strict bundle must contain split and normalization objects")
        split_payload = {
            "schema_version": "edgetwincal.protocol.split.v1",
            "dataset": dataset_id,
            "protocol": protocol_id,
            "fold": fold,
            "group_ids_reliable": True,
            "public_protocol_manifest": dict(bundle["split"]),
        }
        normalization_payload = {
            "schema_version": "edgetwincal.protocol.normalizer.v1",
            "dataset": dataset_id,
            "protocol": protocol_id,
            "fit_scope": "observed_train_only",
            "public_protocol_manifest": dict(bundle["normalization"]),
        }
    else:
        split_payload = {
            "schema_version": "upstream.record-split.v1",
            "dataset": dataset_id,
            "protocol": protocol_id,
            "fold": fold,
            "group_ids_reliable": False,
            "split_behavior": "released_APN_dataset_wrapper",
            "contains_raw_identifiers": False,
        }
        normalization_payload = {
            "schema_version": "upstream.normalizer-behavior.v1",
            "dataset": dataset_id,
            "protocol": protocol_id,
            "fit_scope": "released_code_behavior",
            "leakage_claim": "descriptive_only_not_strict_inference",
        }

    horizon = int(
        padded_prediction_steps
        if task["horizon_kind"] == "time_window"
        else task["prediction_steps"]
    )
    protocol_payload = {
        "schema_version": PRETEST_PROTOCOL_SCHEMA,
        "method_version": config["method"]["version"],
        "resolved_config_sha256": config.sha256,
        "dataset": dataset_id,
        "protocol": protocol_id,
        "fold": fold,
        "test_constructed": False,
        "task_contract": {
            "history": int(task["history"]),
            "horizon_kind": str(task["horizon_kind"]),
            "padded_prediction_steps": horizon,
            "channels": int(task["channels"]),
        },
        "strict_public_supplements": {
            key: value
            for key, value in bundle.items()
            if key not in {"split", "normalization"}
        },
    }
    protocol_payload["split_content_sha256"] = canonical_sha256(split_payload)
    protocol_payload["normalizer_content_sha256"] = canonical_sha256(normalization_payload)
    _privacy_audit(protocol_payload)
    _privacy_audit(split_payload)
    _privacy_audit(normalization_payload)

    _write_frozen_json(assets.split_manifest, split_payload)
    _write_frozen_json(assets.normalizer_manifest, normalization_payload)
    _write_frozen_json(assets.protocol_manifest, protocol_payload)
    return PreparedManifestSet(
        assets.protocol_manifest,
        assets.split_manifest,
        assets.normalizer_manifest,
        sha256_file(assets.protocol_manifest),
        sha256_file(assets.split_manifest),
        sha256_file(assets.normalizer_manifest),
    )


def _update_hash(digest: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"kind": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    elif isinstance(value, Mapping):
        digest.update(b"M")
        for key in sorted(value):
            _update_hash(digest, str(key))
            _update_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"L")
        for child in value:
            _update_hash(digest, child)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
    else:
        raise CampaignRunnerError(f"Unsafe fitted-state value: {type(value)!r}")


def state_sha256(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, value)
    return digest.hexdigest()


def _state_mapping(fitted: FittedVariants) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for variant in fitted.variant_ids:
        state = fitted.states[variant]
        states[variant] = {} if state is None else dict(state.state_dict())
    return states


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> Path:
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CampaignRunnerError(f"Refusing to overwrite fitted state: {destination}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _relative(path: Path) -> str:
    return require_within_root(path, must_exist=True).relative_to(PROJECT_ROOT).as_posix()


def _selection_summary(variant: str, state: Any) -> dict[str, Any]:
    if variant == "APN":
        return {"kind": "frozen_checkpoint"}
    if variant == "SLRH":
        return {"alpha": float(state.alpha)}
    if variant == "CFG":
        return {"alpha": float(state.alpha), "include_self": bool(state.include_self)}
    if variant in {"Full", "V07", "V10"}:
        return {
            "order": state.order,
            "latent_alpha": float(state.latent.alpha),
            "graph_alpha": float(state.graph.alpha),
            "include_self": bool(state.graph.include_self),
        }
    if variant == "V01":
        return {"kind": "closed_form_bias"}
    if variant == "V02":
        return {"alpha": float(state.alpha)}
    if variant == "V03":
        return {"selected_config": dict(state.audit.selected_config)}
    if variant == "V08":
        return {
            "alpha_latent": float(state.alpha_latent),
            "alpha_graph": float(state.alpha_graph),
            "include_self": bool(state.include_self),
        }
    if variant == "V11":
        return {
            "latent_alpha": float(state.latent.alpha),
            "graph_alpha": float(state.graph.alpha),
            "descriptor_sha256": hashlib.sha256(state.descriptor.encode("utf-8")).hexdigest(),
        }
    if variant == "V12":
        return {
            "latent_alpha": float(state.latent.alpha),
            "descriptor_sha256": hashlib.sha256(state.descriptor.encode("utf-8")).hexdigest(),
        }
    raise CampaignRunnerError(f"No selection summary for {variant}")


def fit_and_freeze_registry(
    config: ResolvedConfig,
    *,
    dataset_id: str,
    protocol_id: str,
    entries: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    device: str = "cpu",
    require_all_selected_seeds: bool = True,
) -> Path:
    """Fit and persist all checkpoint states before a ledger may be frozen."""

    expected_seeds = tuple(int(value) for value in config["selection"]["seeds"])
    supplied = tuple(int(entry["seed"]) for entry in entries)
    if len(set(supplied)) != len(supplied):
        raise CampaignRunnerError("Pretest fit registry contains duplicate seeds")
    if require_all_selected_seeds and supplied != expected_seeds:
        raise CampaignRunnerError(
            f"Pretest fit must list every selected seed in order: {supplied} != {expected_seeds}"
        )
    if any(seed not in expected_seeds for seed in supplied):
        raise CampaignRunnerError("Pretest fit contains a seed outside frozen selection")

    output = require_within_root(output_path)
    if output.exists():
        raise CampaignRunnerError(f"Refusing to overwrite fitted registry: {output}")
    state_root = require_within_root(output.parent / f"{output.stem}_states")
    registry_rows: list[dict[str, Any]] = []
    definition_registry = variant_registry(config, dataset_id, protocol_id)
    for entry in entries:
        seed = int(entry["seed"])
        manifest = load_cache_manifest(entry["fit_cache_manifest"])
        split_names = {name.split(".", 1)[0] for name in manifest.shapes}
        if split_names != set(FIT_SPLITS):
            raise CampaignRunnerError("Pretest fit cache must contain exactly train and val")
        if manifest.seed != seed or manifest.dataset_id != dataset_id:
            raise CampaignRunnerError("Pretest fit cache identity mismatch")
        _, splits = read_tensor_cache(entry["fit_cache"], manifest)
        checkpoint = entry.get("checkpoint")
        if checkpoint is not None:
            checkpoint_path = require_within_root(checkpoint, must_exist=True)
            if file_sha256(checkpoint_path) != manifest.checkpoint_sha256:
                raise CampaignRunnerError("Pretest checkpoint hash differs from cache provenance")
        else:
            checkpoint_path = None
        fitted = fit_variants(
            config,
            dataset_id,
            protocol_id,
            seed,
            splits,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        state_value = {
            "schema_version": STATE_SCHEMA,
            "dataset": dataset_id,
            "protocol": protocol_id,
            "seed": seed,
            "resolved_config_sha256": config.sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "variant_ids": list(fitted.variant_ids),
            "states": _state_mapping(fitted),
        }
        state_digest = state_sha256(state_value)
        state_path = state_root / f"seed_{seed}" / STATE_FILENAME
        _atomic_torch_save(state_path, state_value)
        registry_rows.append(
            {
                "seed": seed,
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "fit_cache_manifest_sha256": manifest.manifest_digest(),
                "split_manifest_sha256": manifest.split_manifest_sha256,
                "normalizer_manifest_sha256": manifest.normalizer_manifest_sha256,
                "fit_cache": _relative(require_within_root(entry["fit_cache"], must_exist=True)),
                "fit_cache_manifest_path": _relative(
                    require_within_root(entry["fit_cache_manifest"], must_exist=True)
                ),
                "state_path": _relative(state_path),
                "state_file_sha256": sha256_file(state_path),
                "state_content_sha256": state_digest,
                "variants": [
                    {
                        "variant_id": variant,
                        "state_sha256": state_sha256(state_value["states"][variant]),
                        "selected_hyperparameters": _selection_summary(
                            variant, fitted.states[variant]
                        ),
                    }
                    for variant in fitted.variant_ids
                ],
            }
        )

    payload = {
        "schema_version": FITTED_REGISTRY_SCHEMA,
        "resolved_config_sha256": config.sha256,
        "dataset": dataset_id,
        "protocol": protocol_id,
        "definition_registry_sha256": definition_registry["variant_registry_sha256"],
        "execution": "sequential_single_process_single_device",
        "test_opened": False,
        "seeds": registry_rows,
    }
    payload["variant_registry_sha256"] = canonical_sha256(payload)
    return atomic_write_json(output, payload)


def load_fitted_registry(path: str | Path) -> Mapping[str, Any]:
    source = require_within_root(path, must_exist=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignRunnerError(f"Cannot read fitted registry: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != FITTED_REGISTRY_SCHEMA:
        raise CampaignRunnerError("Invalid fitted registry schema")
    content = dict(value)
    recorded = content.pop("variant_registry_sha256", None)
    if recorded != canonical_sha256(content):
        raise CampaignRunnerError("Fitted registry self-hash mismatch")
    return value


def _latent(value: Mapping[str, Any]) -> SensorLatentResidualHead:
    return SensorLatentResidualHead(
        float(value["alpha"]), value["weights"], value["feature_mean"],
        value["feature_std"], value["intercept"],
    )


def _graph(value: Mapping[str, Any]) -> CrossForecastGraph:
    return CrossForecastGraph(
        float(value["alpha"]), value["weights"], value["source_mean"],
        value["source_std"], value["intercept"], bool(value["include_self"]),
    )


def _sequential(value: Mapping[str, Any]) -> SequentialAdapter:
    return SequentialAdapter(str(value["order"]), _latent(value["latent"]), _graph(value["graph"]))


def load_fitted_states(
    config: ResolvedConfig,
    registry_row: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> FittedVariants:
    state_path = require_within_root(registry_row["state_path"], must_exist=True)
    if sha256_file(state_path) != registry_row["state_file_sha256"]:
        raise CampaignRunnerError("Fitted state file hash mismatch")
    try:
        value = torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CampaignRunnerError(f"Cannot safely load fitted state: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != STATE_SCHEMA:
        raise CampaignRunnerError("Invalid fitted state schema")
    if state_sha256(value) != registry_row["state_content_sha256"]:
        raise CampaignRunnerError("Fitted state content hash mismatch")
    raw = value["states"]
    states: dict[str, Any] = {"APN": None}
    states["SLRH"] = _latent(raw["SLRH"])
    states["CFG"] = _graph(raw["CFG"])
    states["Full"] = _sequential(raw["Full"])
    if value["protocol"] in {"strict_p12", "strict_ushcn"}:
        states["V01"] = ResidualBias(raw["V01"]["bias"], raw["V01"]["observations"])
        v02 = raw["V02"]
        states["V02"] = SelfAffineResidual(
            float(v02["alpha"]), v02["weight"], v02["source_mean"],
            v02["source_scale"], v02["intercept"], v02["observations"],
        )
        from .campaign_runner import CachedAPNDecoder

        hyper = config["datasets"][value["dataset"]]["apn_hyperparameters"]
        decoder = CachedAPNDecoder(
            int(hyper["d_model"]), int(hyper["te_dim"]), float(hyper["dropout"])
        )
        decoder.load_state_dict(raw["V03"]["decoder"], strict=True)
        states["V03"] = DecoderVariant(decoder, raw["V03"])
        states["V07"] = _sequential(raw["V07"])
        v08 = raw["V08"]
        states["V08"] = JointResidualRidge(
            float(v08["alpha_latent"]), float(v08["alpha_graph"]),
            v08["latent_weights"], v08["cross_weights"], v08["latent_mean"],
            v08["latent_scale"], v08["cross_mean"], v08["cross_scale"],
            v08["intercept"], v08["observations"], bool(v08["include_self"]),
        )
        states["V10"] = _sequential(raw["V10"])
        states["V11"] = CrossShuffleVariant(
            _latent(raw["V11"]["latent"]), _graph(raw["V11"]["graph"]),
            str(raw["V11"]["descriptor"]),
        )
        states["V12"] = LatentShuffleVariant(
            _latent(raw["V12"]["latent"]), str(raw["V12"]["descriptor"])
        )
    variant_ids = tuple(str(item) for item in value["variant_ids"])
    if tuple(states) != variant_ids or variant_ids != selected_variant_ids(
        config, str(value["dataset"]), str(value["protocol"])
    ):
        raise CampaignRunnerError("Fitted state variants differ from frozen selection")
    for row in registry_row["variants"]:
        variant = row["variant_id"]
        actual = {} if states[variant] is None else dict(states[variant].state_dict())
        if state_sha256(actual) != row["state_sha256"]:
            raise CampaignRunnerError(f"Fitted state hash mismatch for {variant}")
    return FittedVariants(
        str(value["dataset"]), str(value["protocol"]), int(value["seed"]),
        variant_ids, states, {}, (), device,
    )


__all__ = [
    "PreparedManifestSet",
    "fit_and_freeze_registry",
    "load_fitted_registry",
    "load_fitted_states",
    "prepare_pretest_manifests",
    "state_sha256",
]
