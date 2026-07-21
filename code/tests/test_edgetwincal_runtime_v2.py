from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
import torch

import edgetwincal.runtime_v2 as runtime_v2
from edgetwincal.config import load_resolved_config
from edgetwincal.provenance import StaleCacheError
from edgetwincal.runtime_v2 import (
    BlockedAssetError,
    ExtractionProvenance,
    ExtractionShapeContract,
    TensorPayloadError,
    build_cache_expectation,
    collect_source_hashes,
    deserialize_tensor_payload,
    read_tensor_cache,
    resolve_run_assets,
    serialize_tensor_payload,
    tensor_inventory,
    write_tensor_cache,
)

HA_PADDED_STEPS = 7


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config(dataset: str, seed: int):
    return load_resolved_config(overrides={"dataset": dataset, "seed": seed})


def _create_assets(tmp_path: Path, dataset: str, seed: int, protocol: str):
    config = _config(dataset, seed)
    planned = resolve_run_assets(
        config,
        dataset,
        seed,
        protocol,
        root=tmp_path,
        require_existing=False,
    )
    planned.dataset_root.mkdir(parents=True, exist_ok=True)
    planned.apn_root.mkdir(parents=True, exist_ok=True)
    protocol_manifest = "{\"protocol\":\"" + protocol + "\"}"
    if dataset == "HumanActivity":
        protocol_manifest = (
            "{\"task_contract\":{\"padded_prediction_steps\":7},\"protocol\":\"" + protocol + "\"}"
        )
    for path, value in {
        planned.apn_patch: "synthetic patch",
        planned.checkpoint: f"checkpoint:{dataset}:{seed}:{protocol}",
        planned.apn_config: "synthetic: true\n",
        planned.protocol_manifest: protocol_manifest,
        planned.split_manifest: f'{{"dataset":"{dataset}","seed":{seed}}}',
        planned.normalizer_manifest: '{"fit":"train_only"}',
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    resolved = resolve_run_assets(
        config,
        dataset,
        seed,
        protocol,
        root=tmp_path,
        require_existing=True,
    )
    return config, resolved


def _splits(
    config,
    dataset: str,
    *,
    padded_prediction_steps: int = HA_PADDED_STEPS,
) -> dict[str, dict[str, torch.Tensor]]:
    task = config["datasets"][dataset]["task"]
    apn_parameters = config["datasets"][dataset]["apn_hyperparameters"]
    if task["horizon_kind"] == "fixed_steps":
        horizon = int(task["prediction_steps"])
    else:
        horizon = padded_prediction_steps
    channels = int(task["channels"])
    d_model = int(apn_parameters["d_model"])
    te_dim = int(apn_parameters["te_dim"])
    result: dict[str, dict[str, torch.Tensor]] = {}
    for split_index, (split, n) in enumerate((("train", 3), ("val", 2), ("test", 2))):
        generator = torch.Generator().manual_seed(9000 + split_index)
        sample_start = 1000 * (split_index + 1)
        result[split] = {
            "features": torch.randn(n, channels, d_model, generator=generator),
            "time_encoding": torch.randn(
                n, channels, horizon, te_dim, generator=generator
            ),
            "base_prediction": torch.randn(
                n, horizon, channels, generator=generator
            ),
            "target": torch.randn(n, horizon, channels, generator=generator),
            "mask": torch.ones(n, horizon, channels, dtype=torch.bool),
            "sample_id": torch.arange(sample_start, sample_start + n, dtype=torch.int64),
            "group_id": torch.full(
                (n,), 10 * (split_index + 1), dtype=torch.int64
            ),
        }
    return result


def _provenance(tmp_path: Path, dataset: str) -> ExtractionProvenance:
    loader = tmp_path / "sources" / f"{dataset.lower()}_loader.py"
    extractor = tmp_path / "sources" / "extractor.py"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text(f"DATASET = {dataset!r}\n", encoding="utf-8")
    extractor.write_text("VERSION = 3\n", encoding="utf-8")
    hashes = collect_source_hashes(loader, extractor, root=tmp_path)
    return ExtractionProvenance(
        project_commit="1" * 40,
        dataset_raw_sha256=_sha(f"{dataset}:raw"),
        dataset_processed_sha256=_sha(f"{dataset}:processed"),
        loader_source_sha256=hashes["loader_source_sha256"],
        extractor_source_sha256=hashes["extractor_source_sha256"],
    )


def test_schema3_tensor_cache_round_trip_is_provenance_complete(tmp_path: Path) -> None:
    config, assets = _create_assets(tmp_path, "P12", 2024, "strict_p12")
    splits = _splits(config, "P12")
    provenance = _provenance(tmp_path, "P12")

    path, written_manifest = write_tensor_cache(
        config, assets, splits, provenance, root=tmp_path
    )
    expected = build_cache_expectation(
        config, assets, splits, provenance, root=tmp_path
    )
    loaded_manifest, loaded = read_tensor_cache(path, expected, root=tmp_path)

    assert loaded_manifest.to_dict() == written_manifest.to_dict()
    assert loaded_manifest.schema_version == 3
    assert loaded_manifest.dataset_id == "P12"
    assert loaded_manifest.seed == 2024
    assert set(loaded_manifest.shapes) == {
        f"{split}.{key}"
        for split in ("train", "val", "test")
        for key in runtime_v2.SPLIT_TENSOR_KEYS
    }
    for split in ("train", "val", "test"):
        for key in runtime_v2.SPLIT_TENSOR_KEYS:
            assert torch.equal(loaded[split][key], splits[split][key])


@pytest.mark.parametrize(
    "split_names", (("train", "val"), ("test",))
)
def test_phase_separated_fit_and_once_opened_test_caches_round_trip(
    tmp_path: Path, split_names: tuple[str, ...]
) -> None:
    config, assets = _create_assets(tmp_path, "P12", 2024, "strict_p12")
    all_splits = _splits(config, "P12")
    splits = {name: all_splits[name] for name in split_names}
    provenance = _provenance(tmp_path, "P12")

    path, written_manifest = write_tensor_cache(
        config, assets, splits, provenance, root=tmp_path
    )
    expected = build_cache_expectation(
        config, assets, splits, provenance, root=tmp_path
    )
    loaded_manifest, loaded = read_tensor_cache(path, expected, root=tmp_path)

    assert set(loaded) == set(split_names)
    assert loaded_manifest.to_dict() == written_manifest.to_dict()
    assert set(loaded_manifest.shapes) == {
        f"{split}.{key}"
        for split in split_names
        for key in runtime_v2.SPLIT_TENSOR_KEYS
    }

def test_stale_checkpoint_is_rejected_before_payload_is_returned(tmp_path: Path) -> None:
    config, assets = _create_assets(tmp_path, "USHCN", 2028, "strict_ushcn")
    splits = _splits(config, "USHCN")
    provenance = _provenance(tmp_path, "USHCN")
    path, _ = write_tensor_cache(config, assets, splits, provenance, root=tmp_path)

    assets.checkpoint.write_text("mutated checkpoint", encoding="utf-8")
    stale_expectation = build_cache_expectation(
        config, assets, splits, provenance, root=tmp_path
    )
    with pytest.raises(StaleCacheError) as caught:
        read_tensor_cache(path, stale_expectation, root=tmp_path)
    assert set(caught.value.mismatches) == {"checkpoint_sha256"}


def test_dataset_and_seed_cannot_cross_reuse_cache(tmp_path: Path) -> None:
    p12_config, p12_assets = _create_assets(
        tmp_path, "P12", 2024, "release_parity"
    )
    ha_config, ha_assets = _create_assets(
        tmp_path, "HumanActivity", 2025, "release_parity"
    )
    p12_splits = _splits(p12_config, "P12")
    ha_splits = _splits(ha_config, "HumanActivity")
    p12_provenance = _provenance(tmp_path, "P12")
    ha_provenance = _provenance(tmp_path, "HumanActivity")

    p12_path, _ = write_tensor_cache(
        p12_config, p12_assets, p12_splits, p12_provenance, root=tmp_path
    )
    ha_path, _ = write_tensor_cache(
        ha_config, ha_assets, ha_splits, ha_provenance, root=tmp_path
    )
    assert p12_path != ha_path

    ha_expected = build_cache_expectation(
        ha_config, ha_assets, ha_splits, ha_provenance, root=tmp_path
    )
    with pytest.raises(StaleCacheError) as caught:
        read_tensor_cache(p12_path, ha_expected, root=tmp_path)
    assert {"dataset_id", "seed", "resolved_config_sha256"}.issubset(
        caught.value.mismatches
    )

def test_human_activity_time_window_uses_frozen_dynamic_padding_contract(
    tmp_path: Path,
) -> None:
    config, assets = _create_assets(
        tmp_path, "HumanActivity", 2024, "release_parity"
    )
    task = config["datasets"]["HumanActivity"]["task"]
    assert task["horizon_kind"] == "time_window"
    assert task["forecast_window"] == 300
    assert "prediction_steps" not in task

    provenance = _provenance(tmp_path, "HumanActivity")
    seven_step_splits = _splits(config, "HumanActivity")
    frozen = build_cache_expectation(
        config, assets, seven_step_splits, provenance, root=tmp_path
    )
    assert frozen.key_dict()["horizon"] == HA_PADDED_STEPS
    assert frozen.key_dict()["shapes"]["train.base_prediction"][1] == HA_PADDED_STEPS

    three_hundred_step_splits = _splits(
        config, "HumanActivity", padded_prediction_steps=300
    )
    with pytest.raises(TensorPayloadError, match="task horizon contract"):
        build_cache_expectation(
            config,
            assets,
            three_hundred_step_splits,
            provenance,
            root=tmp_path,
        )

    assets.protocol_manifest.write_text(
        "{\"protocol\":\"release_parity\"}", encoding="utf-8"
    )
    with pytest.raises(TensorPayloadError, match="requires padded_prediction_steps"):
        build_cache_expectation(
            config, assets, seven_step_splits, provenance, root=tmp_path
        )

    eight_step_splits = _splits(
        config, "HumanActivity", padded_prediction_steps=8
    )
    explicit = build_cache_expectation(
        config,
        assets,
        eight_step_splits,
        provenance,
        extraction_contract=ExtractionShapeContract(8),
        root=tmp_path,
    )
    assert explicit.key_dict()["horizon"] == 8
    assert explicit.digest() != frozen.digest()



def test_missing_assets_are_explicitly_blocked_and_paths_are_pinned(tmp_path: Path) -> None:
    p12 = _config("P12", 2027)
    with pytest.raises(BlockedAssetError, match=r"BLOCKED\[missing_assets\]") as caught:
        resolve_run_assets(
            p12,
            "P12",
            2027,
            "strict_p12",
            root=tmp_path,
            require_existing=True,
        )
    assert "checkpoint=" in str(caught.value)
    assert "dataset_root=" in str(caught.value)
    planned = resolve_run_assets(
        p12,
        "P12",
        2027,
        "strict_p12",
        root=tmp_path,
        require_existing=False,
    )
    assert planned.checkpoint == tmp_path / "results/checkpoints/P12/strict_p12/seed_2027/pytorch_model.bin"

    mimic = _config("MIMIC_III", 2024)
    with pytest.raises(BlockedAssetError, match=r"BLOCKED\[missing_assets\]"):
        resolve_run_assets(
            mimic,
            "MIMIC_III",
            2024,
            "strict_mimic_iii",
            root=tmp_path,
            require_existing=True,
        )


def test_payload_rejects_schema2_wrong_keys_shapes_and_observed_nan() -> None:
    config = _config("USHCN", 2024)
    splits = _splits(config, "USHCN")

    legacy = io.BytesIO()
    torch.save({"schema_version": 2, "splits": splits}, legacy)
    with pytest.raises(TensorPayloadError, match="schema-2 is legacy-only"):
        deserialize_tensor_payload(legacy.getvalue())

    missing_key = {split: dict(values) for split, values in splits.items()}
    del missing_key["train"]["group_id"]
    with pytest.raises(TensorPayloadError, match="tensor keys differ"):
        serialize_tensor_payload(missing_key)

    wrong_shape = {split: dict(values) for split, values in splits.items()}
    wrong_shape["val"]["target"] = wrong_shape["val"]["target"][:, :, :-1]
    with pytest.raises(TensorPayloadError, match="target and mask shapes"):
        tensor_inventory(wrong_shape)

    observed_nan = {split: dict(values) for split, values in splits.items()}
    observed_nan["test"]["target"] = observed_nan["test"]["target"].clone()
    observed_nan["test"]["target"][0, 0, 0] = float("nan")
    with pytest.raises(TensorPayloadError, match="non-finite observed"):
        serialize_tensor_payload(observed_nan)


def test_strict_group_overlap_is_rejected_but_release_inventory_is_descriptive(
    tmp_path: Path,
) -> None:
    strict_config, strict_assets = _create_assets(
        tmp_path, "P12", 2026, "strict_p12"
    )
    splits = _splits(strict_config, "P12")
    provenance = _provenance(tmp_path, "P12")

    wrong_dimensions = {split: dict(values) for split, values in splits.items()}
    wrong_dimensions["train"]["features"] = torch.zeros(
        wrong_dimensions["train"]["features"].shape[0],
        wrong_dimensions["train"]["features"].shape[1],
        99,
    )
    with pytest.raises(TensorPayloadError, match="expected .* from ResolvedConfig"):
        build_cache_expectation(
            strict_config, strict_assets, wrong_dimensions, provenance, root=tmp_path
        )

    splits["test"]["group_id"][0] = splits["train"]["group_id"][0]
    tensor_inventory(splits, require_group_disjoint=False)
    with pytest.raises(TensorPayloadError, match="group overlap"):
        build_cache_expectation(
            strict_config, strict_assets, splits, provenance, root=tmp_path
        )


def test_help_is_lazy_and_does_not_resolve_vendor_or_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("--help must stop before loading config or assets")

    monkeypatch.setattr(runtime_v2, "load_resolved_config", forbidden)
    monkeypatch.setattr(runtime_v2, "resolve_run_assets", forbidden)
    with pytest.raises(SystemExit) as caught:
        runtime_v2.main(["--help"])
    assert caught.value.code == 0
    assert "--dataset" in capsys.readouterr().out


def test_payload_serialization_detaches_and_copies_input_tensors() -> None:
    config = _config("USHCN", 2024)
    splits = _splits(config, "USHCN")
    payload = serialize_tensor_payload(splits)
    splits["train"]["features"].fill_(999.0)

    loaded = deserialize_tensor_payload(payload)
    assert not torch.equal(loaded["train"]["features"], splits["train"]["features"])
    assert loaded["train"]["features"].device.type == "cpu"
    assert loaded["train"]["features"].is_contiguous()
