from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from edgetwincal.strict_ushcn import (
    FrozenUSHCNTestLedgerToken,
    USHCN_OBSERVATION_CUTOFF,
    USHCN_PREDICTION_STEPS,
    prepare_strict_ushcn,
)


CODE_HASH = "d" * 64
ASSET_HASHES = {"synthetic-ushcn": "e" * 64}
TIMES = (0.0, 60.0, 149.0, 149.5, 150.0, 151.0, 160.0, 200.0)
STATIONS = tuple(f"RAW-SECRET-STATION-{index:02d}" for index in range(20))


def _synthetic_frame(*, channels: int = 5) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([STATIONS, TIMES], names=("ID", "Time"))
    values = np.empty((len(index), channels), dtype=np.float64)
    for row, (station, time) in enumerate(index):
        station_number = int(str(station).rsplit("-", 1)[-1])
        values[row] = [station_number + float(time) / 100.0 + feature for feature in range(channels)]
    values[1, 1] = np.nan
    return pd.DataFrame(values, index=index, columns=[f"CH_{i}" for i in range(channels)])


def _disjoint_fold() -> dict[str, tuple[str, ...]]:
    return {
        "train": STATIONS[:16],
        "val": STATIONS[16:18],
        "test": STATIONS[18:],
    }


def _overlapping_fold() -> dict[str, tuple[str, ...]]:
    return {
        "train": STATIONS[:16],
        "val": STATIONS[15:18],
        "test": STATIONS[18:],
    }


def _prepare(
    frame: pd.DataFrame,
    fold: dict[str, tuple[str, ...]] | None = None,
):
    return prepare_strict_ushcn(
        frame,
        fold or _disjoint_fold(),
        expected_channels=frame.shape[1],
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )


def test_disjoint_official_fold_zero_is_preserved() -> None:
    prepared = _prepare(_synthetic_frame())
    audit = prepared.public_manifests()["official_fold_audit"]
    split = prepared.split.public_manifest()

    assert audit["is_group_disjoint"] is True
    assert audit["decision"] == "keep_official_fold"
    assert split["allocation"] == "official_fold_0_verified_group_disjoint"
    assert split["deviation_from_official_fold"] is False
    assert split["group_counts"] == {"train": 16, "val": 2, "test": 2}
    for partition, stations in _disjoint_fold().items():
        assert all(prepared.split.split_for(station) == partition for station in stations)


def test_official_overlap_triggers_deterministic_station_hash_repair() -> None:
    frame = _synthetic_frame()
    first = _prepare(frame, _overlapping_fold())
    reversed_fold = {
        name: tuple(reversed(values)) for name, values in _overlapping_fold().items()
    }
    second = _prepare(frame.sample(frac=1.0, random_state=7), reversed_fold)

    audit = first.public_manifests()["official_fold_audit"]
    split = first.split.public_manifest()
    assert audit["is_group_disjoint"] is False
    assert audit["overlap_counts"]["train_val"] == 1
    assert audit["decision"] == "hash_repair_required"
    assert split["allocation"] == "ascending_sha256_floor_80_10_remainder"
    assert split["deviation_from_official_fold"] is True
    assert split["group_counts"] == {"train": 16, "val": 2, "test": 2}
    assert split == second.split.public_manifest()
    assert all(first.split.split_for(station) in {"train", "val", "test"} for station in STATIONS)


def test_test_extremes_cannot_change_train_observed_statistics() -> None:
    original = _synthetic_frame()
    first = _prepare(original)
    changed = original.copy(deep=True)
    changed.loc[(list(_disjoint_fold()["test"]), slice(None)), :] = 1.0e12
    second = _prepare(changed)

    np.testing.assert_array_equal(first.normalizer.mean, second.normalizer.mean)
    np.testing.assert_array_equal(first.normalizer.scale, second.normalizer.scale)
    assert first.normalizer.public_manifest() == second.normalizer.public_manifest()

    train_values = original.loc[(list(_disjoint_fold()["train"]), slice(None)), :].to_numpy()
    np.testing.assert_allclose(first.normalizer.mean, np.nanmean(train_values, axis=0))
    np.testing.assert_allclose(first.normalizer.scale, np.nanstd(train_values, axis=0))
    manifest = first.normalizer.public_manifest()
    assert manifest["fit_order"] == "station_split_before_train_observed_column_fit"
    assert manifest["fit_sample_count"] == 16 * len(TIMES)


def test_task_dataset_matches_apn_time_and_three_step_forecast_contract() -> None:
    prepared = _prepare(_synthetic_frame())
    dataset = prepared.build_dataset("train")
    sample = dataset[0]

    assert len(dataset) == 16
    assert sample.inputs.x.shape == (4, 5)
    assert sample.targets.shape == (USHCN_PREDICTION_STEPS, 5)
    torch.testing.assert_close(
        sample.inputs.t,
        torch.tensor([0.0, 60.0 / 200.0, 149.0 / 200.0, 149.5 / 200.0]),
    )
    torch.testing.assert_close(
        sample.inputs.t_target,
        torch.tensor([150.0 / 200.0, 151.0 / 200.0, 160.0 / 200.0]),
    )
    assert dataset.observation_time == pytest.approx(USHCN_OBSERVATION_CUTOFF / 200.0)
    assert isinstance(sample.key, int) and 1 <= sample.key <= len(STATIONS)


def test_test_dataset_requires_a_matching_frozen_ledger_token() -> None:
    prepared = _prepare(_synthetic_frame())
    with pytest.raises(PermissionError, match="frozen ledger token"):
        prepared.build_dataset("test")
    with pytest.raises(ValueError, match="only be issued"):
        FrozenUSHCNTestLedgerToken.issue(
            prepared,
            registry_hash="f" * 64,
            state="draft",
        )

    token = FrozenUSHCNTestLedgerToken.issue(prepared, registry_hash="f" * 64)
    dataset = prepared.build_dataset("test", ledger_token=token)
    assert len(dataset) == 2

    tampered = replace(token, normalization_manifest_hash="a" * 64)
    with pytest.raises(PermissionError, match="not a valid frozen token"):
        prepared.build_dataset("test", ledger_token=tampered)


def test_public_manifests_and_repr_do_not_expose_station_ids() -> None:
    prepared = _prepare(_synthetic_frame(), _overlapping_fold())
    train_dataset = prepared.build_dataset("train")
    token = FrozenUSHCNTestLedgerToken.issue(prepared, registry_hash="f" * 64)
    serialized = json.dumps(
        {
            "protocol": prepared.public_manifests(),
            "dataset": train_dataset.public_manifest(),
            "token": token.public_manifest(),
        },
        sort_keys=True,
    )

    assert not any(station in serialized for station in STATIONS)
    assert not any(station in repr(prepared) for station in STATIONS)
    assert all(len(value) == 64 for value in train_dataset.public_manifest()["public_station_hashes"])


def test_rejects_incomplete_official_fold_and_wrong_frame_contract() -> None:
    frame = _synthetic_frame()
    incomplete = _disjoint_fold()
    incomplete["test"] = (STATIONS[-1],)
    with pytest.raises(ValueError, match="cover exactly"):
        _prepare(frame, incomplete)

    with pytest.raises(ValueError, match="expects 5 channels"):
        prepare_strict_ushcn(
            frame.iloc[:, :2],
            _disjoint_fold(),
            code_hash=CODE_HASH,
            data_asset_hashes=ASSET_HASHES,
        )

    bad_time = frame.reset_index()
    bad_time["Time"] = bad_time["Time"].clip(upper=149.0)
    bad_time = bad_time.drop_duplicates(["ID", "Time"]).set_index(["ID", "Time"])
    with pytest.raises(ValueError, match="extend beyond"):
        _prepare(bad_time)


def test_split_and_manifest_are_invariant_to_rows_and_fold_key_order() -> None:
    frame = _synthetic_frame()
    first = _prepare(frame)
    reordered_frame = frame.sample(frac=1.0, random_state=19)
    reordered_fold = {
        name: tuple(reversed(values)) for name, values in _disjoint_fold().items()
    }
    second = _prepare(reordered_frame, reordered_fold)

    assert first.split.public_manifest() == second.split.public_manifest()
    assert first.normalizer.public_manifest() == second.normalizer.public_manifest()
    assert first.time_scale.public_manifest() == second.time_scale.public_manifest()
