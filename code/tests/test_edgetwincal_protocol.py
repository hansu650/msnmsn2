from __future__ import annotations

import json

import numpy as np
import pytest

from edgetwincal.protocol import (
    LOCKED_GROUP_SALTS,
    audit_official_fold_station_overlap,
    fit_train_normalizer,
    hash_group_split,
    salted_identifier_hash,
)


ASSET_HASHES = {"synthetic": "a" * 64}
CODE_HASH = "b" * 64


def _p12_split(group_ids: list[str]):
    return hash_group_split(
        group_ids,
        dataset_id="p12",
        protocol_id="p12_strict_v1",
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )


def test_group_split_is_disjoint_deterministic_and_floor_allocated() -> None:
    unique = [f"patient-secret-{index:02d}" for index in range(20)]
    rows = [group for group in unique for _ in range(2)]
    first = _p12_split(rows)
    second = _p12_split(list(reversed(rows)))

    assert first.public_manifest() == second.public_manifest()
    manifest = first.public_manifest()
    assert manifest["group_counts"] == {"train": 16, "val": 2, "test": 2}
    public_sets = {
        name: set(manifest["public_group_hashes"][name])
        for name in ("train", "val", "test")
    }
    assert public_sets["train"].isdisjoint(public_sets["val"])
    assert public_sets["train"].isdisjoint(public_sets["test"])
    assert public_sets["val"].isdisjoint(public_sets["test"])

    expected_order = sorted(
        unique,
        key=lambda group: salted_identifier_hash(group, LOCKED_GROUP_SALTS["p12"]),
    )
    assert all(first.split_for(group) == "train" for group in expected_order[:16])
    assert all(first.split_for(group) == "val" for group in expected_order[16:18])
    assert all(first.split_for(group) == "test" for group in expected_order[18:])
    for group in unique:
        assert len(set(first.split_for(group) for row in rows if row == group)) == 1


def test_generic_split_requires_and_records_an_explicit_locked_salt() -> None:
    groups = [f"subject-{index}" for index in range(10)]
    with pytest.raises(ValueError, match="no registered salt"):
        hash_group_split(groups, dataset_id="new-public-data", protocol_id="strict-v1")

    split = hash_group_split(
        groups,
        dataset_id="new-public-data",
        protocol_id="strict-v1",
        locked_salt="edgetwincal-msn2026-new-public-data-v1",
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )
    assert split.public_manifest()["group_hash_salt"].endswith("new-public-data-v1")
    with pytest.raises(ValueError, match="registered salt"):
        hash_group_split(
            groups,
            dataset_id="p12",
            protocol_id="bad",
            locked_salt="wrong-salt",
        )


def test_normalizer_fits_train_ids_only_and_ignores_test_extremes() -> None:
    groups = [f"patient-private-{index:02d}" for index in range(20)]
    sample_ids = [f"window-private-{index:02d}" for index in range(20)]
    split = _p12_split(groups)
    values = np.zeros((20, 2), dtype=np.float64)
    for index, group in enumerate(groups):
        if split.split_for(group) == "train":
            values[index] = (index, 2.0 * index + 1.0)
        elif split.split_for(group) == "val":
            values[index] = (100.0, 200.0)
        else:
            values[index] = (1.0e9, -1.0e9)
    mask = np.ones_like(values, dtype=bool)

    first = fit_train_normalizer(
        values,
        mask,
        sample_ids=sample_ids,
        group_ids=groups,
        split=split,
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )
    changed = values.copy()
    changed[[split.split_for(group) == "test" for group in groups]] *= 12345.0
    second = fit_train_normalizer(
        changed,
        mask,
        sample_ids=sample_ids,
        group_ids=groups,
        split=split,
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    assert first.fit_id_hash == second.fit_id_hash

    normalizer_manifest = first.public_manifest()
    fit_hashes = set(normalizer_manifest["fit_id_hashes"])
    sample_salt = f"{split.salt}:fit-sample-v1"
    expected_train = {
        salted_identifier_hash(sample_ids[index], sample_salt)
        for index, group in enumerate(groups)
        if split.split_for(group) == "train"
    }
    expected_test = {
        salted_identifier_hash(sample_ids[index], sample_salt)
        for index, group in enumerate(groups)
        if split.split_for(group) == "test"
    }
    assert fit_hashes == expected_train
    assert fit_hashes.isdisjoint(expected_test)
    assert normalizer_manifest["split_hash"] == split.split_hash
    assert normalizer_manifest["code_hash"] == CODE_HASH
    assert normalizer_manifest["data_asset_hashes"] == ASSET_HASHES


def test_public_manifests_never_contain_raw_identifiers() -> None:
    groups = [f"DO-NOT-EXPORT-PATIENT-{index}" for index in range(10)]
    samples = [f"DO-NOT-EXPORT-WINDOW-{index}" for index in range(10)]
    split = _p12_split(groups)
    normalizer = fit_train_normalizer(
        np.arange(20, dtype=np.float64).reshape(10, 2),
        np.ones((10, 2), dtype=bool),
        sample_ids=samples,
        group_ids=groups,
        split=split,
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )
    serialized = json.dumps(
        {"split": split.public_manifest(), "normalizer": normalizer.public_manifest()},
        sort_keys=True,
    )
    assert not any(identifier in serialized for identifier in groups + samples)
    assert split.public_hash_for(groups[0]) in serialized


def test_constant_missing_nan_and_round_trip_are_finite() -> None:
    groups = [f"patient-{index:02d}" for index in range(10)]
    split = _p12_split(groups)
    samples = [f"window-{index:02d}" for index in range(10)]
    values = np.zeros((10, 3), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    for index, group in enumerate(groups):
        if split.split_for(group) == "train":
            values[index, 0] = 7.0
            values[index, 1] = float(index)
            values[index, 2] = np.nan
            mask[index, :2] = True
        else:
            values[index] = (999.0, -999.0, np.nan)
    normalizer = fit_train_normalizer(
        values,
        mask,
        sample_ids=samples,
        group_ids=groups,
        split=split,
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )
    assert normalizer.scale[0] == pytest.approx(1e-6)
    assert normalizer.mean[2] == 0.0
    assert normalizer.scale[2] == 1.0
    assert normalizer.public_manifest()["no_observation_features"] == [2]

    transformed = normalizer.apply(values, mask)
    restored = normalizer.inverse(transformed, mask)
    assert np.isfinite(transformed).all()
    assert np.isfinite(restored).all()
    np.testing.assert_allclose(restored[mask], values[mask], atol=1e-12, rtol=0.0)
    assert np.all(transformed[~mask] == 0.0)
    assert np.all(restored[~mask] == 0.0)

    bad = values.copy()
    bad[np.flatnonzero([split.split_for(group) == "train" for group in groups])[0], 1] = np.nan
    with pytest.raises(ValueError, match="Observed training values"):
        fit_train_normalizer(
            bad,
            mask,
            sample_ids=samples,
            group_ids=groups,
            split=split,
        )


def test_station_overlap_audit_is_privacy_safe_and_selects_repair() -> None:
    disjoint = audit_official_fold_station_overlap(
        ["PRIVATE-STATION-A"],
        ["PRIVATE-STATION-B"],
        ["PRIVATE-STATION-C"],
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )
    assert disjoint["is_group_disjoint"] is True
    assert disjoint["decision"] == "keep_official_fold"

    overlapping = audit_official_fold_station_overlap(
        ["PRIVATE-STATION-A", "PRIVATE-STATION-B"],
        ["PRIVATE-STATION-B"],
        ["PRIVATE-STATION-C", "PRIVATE-STATION-A"],
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )
    assert overlapping["is_group_disjoint"] is False
    assert overlapping["decision"] == "hash_repair_required"
    assert overlapping["overlap_counts"] == {
        "train_val": 1,
        "train_test": 1,
        "val_test": 0,
    }
    serialized = json.dumps(overlapping, sort_keys=True)
    assert "PRIVATE-STATION" not in serialized
