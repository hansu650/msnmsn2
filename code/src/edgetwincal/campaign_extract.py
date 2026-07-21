"""Project-owned APN backbone and train/validation cache preparation boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .apn_bridge import (
    apn_forward_callback,
    collect_apn_fit_splits,
    load_frozen_apn_checkpoint,
    make_apn_model_factory,
)
from .apn_training import APNTrainingResult, train_apn_train_val
from .campaign_runner import CampaignRunnerError, write_cache_manifest
from .config import ResolvedConfig
from .paths import require_within_root
from .runtime_v2 import (
    ExtractionProvenance,
    ResolvedRunAssets,
    write_tensor_cache,
)


def prepare_fit_cache(
    *,
    config: ResolvedConfig,
    assets: ResolvedRunAssets,
    provenance: ExtractionProvenance,
    bundle: Any,
    loader_factory: Any,
    cache_manifest_path: str | Path,
    device: str,
    train_backbone: bool,
    argv: Sequence[str] = ("edgetwincal", "prepare-fit-cache"),
) -> dict[str, Any]:
    """Train/load one APN and cache exactly train+validation tensors.

    The signature has no test loader, split selector, ledger token, or test
    callback.  The provided APN loader factory is invoked only through its
    train/validation-only interfaces.
    """

    identity = (
        str(getattr(bundle, "dataset_id", "")),
        str(getattr(bundle, "protocol_id", "")),
        int(getattr(bundle, "seed", -1)),
    )
    if identity != (assets.dataset_id, assets.protocol_id, assets.seed):
        raise CampaignRunnerError("APN bundle identity differs from resolved assets")
    if getattr(loader_factory, "bundle", None) is not bundle:
        raise CampaignRunnerError("Loader factory is not bound to the frozen APN bundle")

    training: APNTrainingResult | None = None
    if train_backbone:
        training = train_apn_train_val(
            output_dir=assets.checkpoint.parent,
            seed=assets.seed,
            learning_rate=float(
                config["datasets"][assets.dataset_id]["apn_hyperparameters"][
                    "learning_rate"
                ]
            ),
            resolved_config=config.to_dict(),
            model_factory=make_apn_model_factory(bundle),
            loader_factory=loader_factory,
            forward_callback=apn_forward_callback,
            device=device,
            argv=argv,
        )
        model = training.model
        if training.checkpoint_path.resolve() != assets.checkpoint.resolve():
            raise CampaignRunnerError("APN trainer checkpoint path differs from asset registry")
    else:
        model = load_frozen_apn_checkpoint(
            bundle,
            assets.checkpoint,
            device=device,
            model_factory=make_apn_model_factory(bundle),
        )

    splits = collect_apn_fit_splits(
        model,
        loader_factory,
        device=device,
    )
    if tuple(splits) != ("train", "val"):
        raise CampaignRunnerError("Fit extraction returned anything other than train and val")
    cache_path, cache_manifest = write_tensor_cache(
        config,
        assets,
        splits,
        provenance,
        stem="train_validation_features",
    )
    sidecar = write_cache_manifest(cache_manifest_path, cache_manifest)
    return {
        "schema_version": "edgetwincal.fit-cache-preparation.v1",
        "dataset": assets.dataset_id,
        "protocol": assets.protocol_id,
        "seed": assets.seed,
        "splits_opened": ["train", "val"],
        "test_opened": False,
        "checkpoint_trained": bool(train_backbone),
        "checkpoint_sha256": cache_manifest.checkpoint_sha256,
        "cache_path": require_within_root(cache_path, must_exist=True),
        "cache_manifest_path": require_within_root(sidecar, must_exist=True),
        "cache_manifest_sha256": cache_manifest.manifest_digest(),
        "training_manifest_path": (
            training.manifest_path if training is not None else None
        ),
    }


__all__ = ["prepare_fit_cache"]
