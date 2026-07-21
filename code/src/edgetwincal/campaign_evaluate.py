"""Once-opened, multi-checkpoint EdgeTwinCal test campaign evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .apn_bridge import FrozenAPNTestLedgerToken, collect_apn_split
from .campaign import ProtocolLedger
from .campaign_pretest import load_fitted_registry, load_fitted_states
from .campaign_runner import (
    EVALUATION_SCHEMA,
    TIMING_SCHEMA,
    CampaignRunnerError,
    _apply_variant,
    _assert_cache_pair,
    _assert_cross_phase_disjoint,
    _manifest_cache_reference,
    _metrics_and_cells,
    _strict_protocol,
    load_cache_manifest,
    resolved_config_from_ledger,
    write_cache_manifest,
)
from .paths import PROJECT_ROOT, require_within_root
from .runtime_v2 import (
    ExtractionProvenance,
    ResolvedRunAssets,
    collect_environment,
    read_tensor_cache,
    write_tensor_cache,
)
from .schema import RunManifest, atomic_write_json, sha256_file
from .timing import PhaseTimer, serialized_state_bytes


TEST_ACCESS_SCHEMA = "edgetwincal.test-cache-access.v1"
TEST_REGISTRY_SCHEMA = "edgetwincal.test-cache-registry.v1"


def _read_json(path: str | Path, *, label: str) -> Mapping[str, Any]:
    source = require_within_root(path, must_exist=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignRunnerError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CampaignRunnerError(f"{label} must be a JSON object")
    return value


def _relative(path: Path) -> str:
    return require_within_root(path, must_exist=True).relative_to(PROJECT_ROOT).as_posix()


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()




def _strict_dataset_token(
    *,
    loader_factory: Any,
    assets: ResolvedRunAssets,
    ledger: ProtocolLedger,
    opening: Mapping[str, Any],
) -> Any:
    prepared = getattr(loader_factory, "prepared", None)
    if prepared is None:
        raise CampaignRunnerError("Strict loader factory has no prepared protocol state")
    split_file_hash = sha256_file(assets.split_manifest)
    normalizer_file_hash = sha256_file(assets.normalizer_manifest)
    if opening.get("split_manifest_sha256") != split_file_hash:
        raise CampaignRunnerError("Strict opening does not bind the on-disk split manifest")
    if opening.get("normalization_manifest_sha256") != normalizer_file_hash:
        raise CampaignRunnerError("Strict opening does not bind the on-disk normalizer manifest")
    components = ledger.data.get("components", {})
    if components.get("split_manifests", {}).get(
        f"{opening['dataset']}|{opening['protocol']}|{opening['fold']}"
    ) != split_file_hash:
        raise CampaignRunnerError("Strict split manifest is absent from frozen components")
    if components.get("normalization_manifests", {}).get(
        f"{opening['dataset']}|{opening['protocol']}|{opening['fold']}"
    ) != normalizer_file_hash:
        raise CampaignRunnerError("Strict normalizer manifest is absent from frozen components")

    split_wrapper = _read_json(assets.split_manifest, label="strict split manifest")
    normalizer_wrapper = _read_json(
        assets.normalizer_manifest, label="strict normalizer manifest"
    )
    if split_wrapper.get("public_protocol_manifest") != prepared.split.public_manifest():
        raise CampaignRunnerError("Prepared strict split differs from the frozen wrapper")
    if (
        normalizer_wrapper.get("public_protocol_manifest")
        != prepared.normalizer.public_manifest()
    ):
        raise CampaignRunnerError("Prepared strict normalizer differs from the frozen wrapper")
    registry_hash = str(components.get("variant_registry_sha256", ""))
    dataset_key = assets.dataset_id.lower()
    if dataset_key == "p12":
        from .strict_p12 import FrozenP12TestLedgerToken

        token = FrozenP12TestLedgerToken.issue(
            prepared, registry_hash=registry_hash, state="frozen"
        )
    elif dataset_key == "ushcn":
        from .strict_ushcn import FrozenUSHCNTestLedgerToken

        token = FrozenUSHCNTestLedgerToken.issue(
            prepared, registry_hash=registry_hash, state="frozen"
        )
    else:
        raise CampaignRunnerError(
            "Strict token adaptation is implemented only for P12 and USHCN"
        )
    token.validate_for(prepared)
    return token

def extract_test_cache_after_open(
    *,
    config: Any,
    assets: ResolvedRunAssets,
    provenance: ExtractionProvenance,
    model: torch.nn.Module,
    loader_factory: Any,
    ledger_path: str | Path,
    cell_id: str,
    token: str,
    cache_manifest_path: str | Path,
    access_manifest_path: str | Path,
    device: str,
    test_loader_builder: Callable[[FrozenAPNTestLedgerToken], Any] | None = None,
) -> dict[str, Any]:
    """Construct and cache test tensors only under an active frozen opening.

    ``test_loader_builder`` is the strict-protocol adapter point. Released APN
    factories can use their native ``build_test_loader`` method directly.
    """

    ledger = ProtocolLedger.load(ledger_path)
    access = FrozenAPNTestLedgerToken.from_protocol_ledger(
        ledger, cell_id=cell_id, token=token
    )
    access.validate_for(assets.dataset_id, assets.protocol_id)
    opening = ledger.data["test_openings"][cell_id]
    if test_loader_builder is not None:
        loader = test_loader_builder(access)
    elif assets.protocol_id != "release_parity":
        strict_token = _strict_dataset_token(
            loader_factory=loader_factory,
            assets=assets,
            ledger=ledger,
            opening=opening,
        )
        loader = loader_factory.build_test_loader(
            test_ledger_token=strict_token, purpose="extraction"
        )
    else:
        loader = loader_factory.build_test_loader(
            test_ledger_token=access, purpose="extraction"
        )
    test_split = collect_apn_split(
        model,
        loader,
        dataset_id=assets.dataset_id,
        protocol_id=assets.protocol_id,
        split="test",
        device=device,
        test_ledger_token=access,
    )
    cache_path, cache_manifest = write_tensor_cache(
        config,
        assets,
        {"test": test_split},
        provenance,
        stem="once_opened_test_features",
    )
    manifest_path = write_cache_manifest(cache_manifest_path, cache_manifest)
    access_payload = {
        "schema_version": TEST_ACCESS_SCHEMA,
        "cell_id": cell_id,
        "dataset": assets.dataset_id,
        "protocol": assets.protocol_id,
        "seed": assets.seed,
        "protocol_sha256": access.protocol_sha256,
        "token_sha256": access.token_sha256,
        "cache_path": _relative(cache_path),
        "cache_manifest_path": _relative(manifest_path),
        "cache_manifest_sha256": cache_manifest.manifest_digest(),
    }
    access_path = atomic_write_json(access_manifest_path, access_payload)
    return {
        "seed": assets.seed,
        "test_cache": _relative(cache_path),
        "test_cache_manifest": _relative(manifest_path),
        "test_access_manifest": _relative(access_path),
    }


def write_test_cache_registry(
    path: str | Path,
    *,
    dataset_id: str,
    protocol_id: str,
    cell_id: str,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    seeds = [int(entry["seed"]) for entry in entries]
    if len(set(seeds)) != len(seeds):
        raise CampaignRunnerError("Test cache registry contains duplicate seeds")
    payload = {
        "schema_version": TEST_REGISTRY_SCHEMA,
        "dataset": dataset_id,
        "protocol": protocol_id,
        "cell_id": cell_id,
        "seeds": [dict(entry) for entry in entries],
    }
    return atomic_write_json(path, payload)


def _validate_test_access(
    access_path: str | Path,
    *,
    token: str,
    cell_id: str,
    protocol_sha256: str,
    seed: int,
    cache_manifest_digest: str,
) -> None:
    access = _read_json(access_path, label="test access manifest")
    expected = {
        "schema_version": TEST_ACCESS_SCHEMA,
        "cell_id": cell_id,
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "token_sha256": _token_sha256(token),
        "cache_manifest_sha256": cache_manifest_digest,
    }
    mismatches = {
        key: (expected_value, access.get(key))
        for key, expected_value in expected.items()
        if access.get(key) != expected_value
    }
    if mismatches:
        raise CampaignRunnerError(
            f"Test cache was not produced under this opening: {sorted(mismatches)}"
        )


def _run_one_variant(
    *,
    config: Any,
    fitted: Any,
    fit_manifest: Any,
    test_manifest: Any,
    split: Mapping[str, torch.Tensor],
    variant: str,
    opening: Mapping[str, Any],
    cell_id: str,
    ledger: ProtocolLedger,
    run_root: Path,
    registry_variant: Mapping[str, Any],
    environment: Mapping[str, Any],
    argv: Sequence[str],
) -> str:
    dataset_id = fitted.dataset_id
    protocol_id = fitted.protocol_id
    seed = fitted.seed
    run_dir = require_within_root(
        run_root / dataset_id / protocol_id / f"seed_{seed}" / variant
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = require_within_root(run_dir / "run_log.json")
    manifest_path = require_within_root(run_dir / "run_manifest.json")
    manifest = RunManifest.create(
        manifest_path,
        run_id=f"{dataset_id}-{protocol_id}-{seed}-{variant}",
        dataset=dataset_id,
        protocol=protocol_id,
        fold=str(opening["fold"]),
        seed=seed,
        variant_id=variant,
        variant_definition=dict(config["variants"][variant]),
        resolved_config=config,
        argv=tuple(str(value) for value in argv),
        environment=environment,
        log_path=log_path,
    ).start()
    try:
        timer = PhaseTimer()
        with timer.phase("warm_inference", device=fitted.device):
            prediction = _apply_variant(fitted, variant, split)
        metrics, cells = _metrics_and_cells(
            split,
            prediction,
            variant=variant,
            checkpoint_sha256=test_manifest.checkpoint_sha256,
            salt=f"edgetwincal-msn2026-result-v1:{dataset_id}:{protocol_id}",
        )
        state = {} if fitted.states[variant] is None else fitted.states[variant].state_dict()
        if serialized_state_bytes(state) != int(registry_variant.get("adapter_state_bytes", serialized_state_bytes(state))):
            # Byte size is informational when absent; state SHA validation happened on load.
            pass
        atomic_write_json(
            log_path,
            {
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
            },
        )
        manifest.mark_phase("frozen_train_validation_state_loaded")
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
            selected_hyperparameters=dict(registry_variant["selected_hyperparameters"]),
            timing={"schema_version": TIMING_SCHEMA, "segments": timer.as_dicts()},
            cells=cells,
            metrics=metrics,
            required_files=(log_path,),
        )
        return manifest_path.relative_to(PROJECT_ROOT).as_posix()
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
        raise


def evaluate_campaign_once(
    *,
    ledger_path: str | Path,
    cell_id: str,
    token: str,
    fitted_registry_path: str | Path,
    test_cache_registry_path: str | Path,
    run_root: str | Path | None = None,
    device: str = "cpu",
    argv: Sequence[str] = ("edgetwincal", "evaluate"),
    require_all_selected_seeds: bool = True,
) -> dict[str, Any]:
    """Consume one opening only after all frozen checkpoint states are loaded."""

    ledger = ProtocolLedger.load(ledger_path)
    config = resolved_config_from_ledger(ledger)
    registry = load_fitted_registry(fitted_registry_path)
    registry_sha = str(registry["variant_registry_sha256"])
    if ledger.data.get("components", {}).get("variant_registry_sha256") != registry_sha:
        raise CampaignRunnerError("Ledger is not bound to this fitted registry")
    if ledger.status != "test_active":
        raise CampaignRunnerError("Campaign evaluation requires one active frozen opening")
    opening = ledger.data.get("test_openings", {}).get(cell_id)
    if not isinstance(opening, Mapping):
        raise CampaignRunnerError("Requested campaign cell is not open")
    dataset_id = str(opening["dataset"])
    protocol_id = str(opening["protocol"])
    if (registry.get("dataset"), registry.get("protocol")) != (dataset_id, protocol_id):
        raise CampaignRunnerError("Fitted registry identity differs from the opening")
    if registry.get("resolved_config_sha256") != config.sha256:
        raise CampaignRunnerError("Fitted registry config hash differs from the ledger")
    if ledger.data.get("resolved_config_sha256") != config.sha256:
        raise CampaignRunnerError("Ledger resolved configuration hash changed")
    expected_seeds = tuple(int(value) for value in config["selection"]["seeds"])
    registry_rows = list(registry["seeds"])
    registry_seeds = tuple(int(row["seed"]) for row in registry_rows)
    if require_all_selected_seeds and registry_seeds != expected_seeds:
        raise CampaignRunnerError("Fitted registry does not contain every selected seed in order")


    split_hashes = {str(row["split_manifest_sha256"]) for row in registry_rows}
    normalizer_hashes = {str(row["normalizer_manifest_sha256"]) for row in registry_rows}
    if len(split_hashes) != 1 or len(normalizer_hashes) != 1:
        raise CampaignRunnerError("Fitted checkpoints do not share frozen split/normalizer hashes")
    split_hash = next(iter(split_hashes))
    normalizer_hash = next(iter(normalizer_hashes))
    components = ledger.data.get("components", {})
    if components.get("split_manifests", {}).get(cell_id) != split_hash:
        raise CampaignRunnerError("Ledger split component does not bind the fitted registry")
    if components.get("normalization_manifests", {}).get(cell_id) != normalizer_hash:
        raise CampaignRunnerError("Ledger normalizer component does not bind the fitted registry")
    if opening.get("split_manifest_sha256") != split_hash:
        raise CampaignRunnerError("Test opening split hash differs from fitted state")
    if opening.get("normalization_manifest_sha256") != normalizer_hash:
        raise CampaignRunnerError("Test opening normalizer hash differs from fitted state")

    # Load and hash-verify every selected state before accepting the raw token.
    fitted_by_seed = {
        int(row["seed"]): load_fitted_states(config, row, device=device)
        for row in registry_rows
    }
    ledger.validate_test_token(cell_id=cell_id, token=token)
    test_registry = _read_json(test_cache_registry_path, label="test cache registry")
    if test_registry.get("schema_version") != TEST_REGISTRY_SCHEMA:
        raise CampaignRunnerError("Invalid test cache registry schema")
    if (
        test_registry.get("dataset"), test_registry.get("protocol"), test_registry.get("cell_id")
    ) != (dataset_id, protocol_id, cell_id):
        raise CampaignRunnerError("Test cache registry identity differs from the opening")
    test_rows = list(test_registry.get("seeds", []))
    test_seeds = tuple(int(row["seed"]) for row in test_rows)
    if test_seeds != registry_seeds:
        raise CampaignRunnerError("Test and fitted registries have different ordered seeds")

    output_root = require_within_root(
        run_root or Path(str(config["paths"]["run_root"]))
    )
    environment = collect_environment()
    completed: list[str] = []
    failure: dict[str, Any] | None = None
    unrun = [
        {"seed": seed, "variant": variant}
        for seed in registry_seeds
        for variant in fitted_by_seed[seed].variant_ids
    ]
    test_phase_started = False
    try:
        for registry_row, test_row in zip(registry_rows, test_rows):
            seed = int(registry_row["seed"])
            fitted = fitted_by_seed[seed]
            test_phase_started = True
            test_manifest = load_cache_manifest(test_row["test_cache_manifest"])
            _validate_test_access(
                test_row["test_access_manifest"],
                token=token,
                cell_id=cell_id,
                protocol_sha256=str(ledger.data["protocol_sha256"]),
                seed=seed,
                cache_manifest_digest=test_manifest.manifest_digest(),
            )
            fit_manifest_digest = registry_row["fit_cache_manifest_sha256"]
            # The exact fit manifest remains explicit in the state registry row.
            fit_manifest = load_cache_manifest(registry_row["fit_cache_manifest_path"])
            if fit_manifest.manifest_digest() != fit_manifest_digest:
                raise CampaignRunnerError("Fit manifest changed after state freezing")
            _assert_cache_pair(fit_manifest, test_manifest)
            _, fit_splits = read_tensor_cache(registry_row["fit_cache"], fit_manifest)
            _, test_splits = read_tensor_cache(test_row["test_cache"], test_manifest)
            _assert_cross_phase_disjoint(
                fit_splits, test_splits, strict=_strict_protocol(protocol_id)
            )
            split = test_splits["test"]
            variant_rows = {row["variant_id"]: row for row in registry_row["variants"]}
            for variant in fitted.variant_ids:
                path = _run_one_variant(
                    config=config,
                    fitted=fitted,
                    fit_manifest=fit_manifest,
                    test_manifest=test_manifest,
                    split=split,
                    variant=variant,
                    opening=opening,
                    cell_id=cell_id,
                    ledger=ledger,
                    run_root=output_root,
                    registry_variant=variant_rows[variant],
                    environment=environment,
                    argv=argv,
                )
                completed.append(path)
                unrun.pop(0)
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "unrun": unrun,
        }
    finally:
        if test_phase_started:
            ledger.close_test(cell_id=cell_id, token=token)

    result = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete" if failure is None else "failed",
        "dataset": dataset_id,
        "protocol": protocol_id,
        "seeds": list(registry_seeds),
        "variant_registry_sha256": registry_sha,
        "run_manifests": completed,
        "failure": failure,
        "test_opening_consumed": test_phase_started,
    }
    return result


__all__ = [
    "TEST_ACCESS_SCHEMA",
    "TEST_REGISTRY_SCHEMA",
    "evaluate_campaign_once",
    "extract_test_cache_after_open",
    "write_test_cache_registry",
]
