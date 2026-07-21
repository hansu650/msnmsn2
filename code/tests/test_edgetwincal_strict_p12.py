from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from edgetwincal.strict_p12 import (
    FrozenP12TestLedgerToken,
    P12_FEATURE_COLUMNS,
    P12_OBSERVATION_CUTOFF_HOURS,
    P12_PREDICTION_STEPS,
    P12_RAW_COLUMNS,
    P12TimeScale,
    prepare_strict_p12,
)


CODE_HASH = "b" * 64
ASSET_HASHES = {"synthetic-p12": "a" * 64}
TIMES = (0, 12, 35, 36, 37, 38)


def _synthetic_frame(*, channels: int = 2) -> pd.DataFrame:
    patient_ids = [f"RAW-SECRET-PATIENT-{index:02d}" for index in range(20)]
    index = pd.MultiIndex.from_product(
        [patient_ids, TIMES], names=("RecordID", "Time")
    )
    values = np.empty((len(index), channels), dtype=np.float64)
    for row, (patient, time) in enumerate(index):
        patient_number = int(str(patient).rsplit("-", 1)[-1])
        values[row] = [patient_number + time / 100.0 + feature for feature in range(channels)]
    values[1, 1] = np.nan
    return pd.DataFrame(values, index=index, columns=[f"feature_{i}" for i in range(channels)])


def _prepare(frame: pd.DataFrame):
    return prepare_strict_p12(
        frame,
        expected_channels=frame.shape[1],
        code_hash=CODE_HASH,
        data_asset_hashes=ASSET_HASHES,
    )


def test_test_extremes_cannot_change_train_only_statistics() -> None:
    original = _synthetic_frame()
    first = _prepare(original)
    changed = original.copy(deep=True)
    test_ids = [
        patient
        for patient in original.index.get_level_values("RecordID").unique()
        if first.split.split_for(patient) == "test"
    ]
    changed.loc[(test_ids, slice(None)), :] = 1.0e12
    second = _prepare(changed)

    np.testing.assert_array_equal(first.normalizer.mean, second.normalizer.mean)
    np.testing.assert_array_equal(first.normalizer.scale, second.normalizer.scale)
    assert first.normalizer.public_manifest() == second.normalizer.public_manifest()


def test_split_is_locked_before_fit_and_manifest_names_only_train_rows() -> None:
    frame = _synthetic_frame()
    prepared = _prepare(frame)
    split_manifest = prepared.split.public_manifest()
    normalizer_manifest = prepared.normalizer.public_manifest()

    assert split_manifest["group_counts"] == {"train": 16, "val": 2, "test": 2}
    assert normalizer_manifest["fit_order"] == "patient_split_before_train_only_column_fit"
    assert normalizer_manifest["fit_sample_count"] == 16 * len(TIMES)
    assert normalizer_manifest["split_hash"] == split_manifest["split_hash"]
    assert len(normalizer_manifest["fit_id_hashes"]) == 16 * len(TIMES)

    train_patients = {
        patient
        for patient in frame.index.get_level_values("RecordID").unique()
        if prepared.split.split_for(patient) == "train"
    }
    train_values = frame.loc[(list(train_patients), slice(None)), :].to_numpy()
    np.testing.assert_allclose(prepared.normalizer.mean, np.nanmean(train_values, axis=0))
    np.testing.assert_allclose(prepared.normalizer.scale, np.nanstd(train_values, axis=0))


def test_task_dataset_preserves_apn_history_and_three_target_steps() -> None:
    prepared = _prepare(_synthetic_frame())
    dataset = prepared.build_dataset("train")
    sample = dataset[0]

    assert len(dataset) == 16
    assert sample.inputs.x.shape == (3, 2)
    assert sample.targets.shape == (P12_PREDICTION_STEPS, 2)
    torch.testing.assert_close(
        sample.inputs.t,
        torch.tensor([0.0, 12.0 / 48.0, 35.0 / 48.0]),
    )
    torch.testing.assert_close(
        sample.inputs.t_target,
        torch.tensor([36.0 / 48.0, 37.0 / 48.0, 38.0 / 48.0]),
    )
    assert dataset.observation_time == pytest.approx(P12_OBSERVATION_CUTOFF_HOURS / 48.0)
    assert isinstance(sample.key, int) and 1 <= sample.key <= 20


def test_test_dataset_requires_matching_frozen_ledger_token() -> None:
    prepared = _prepare(_synthetic_frame())
    with pytest.raises(PermissionError, match="frozen ledger token"):
        prepared.build_dataset("test")
    with pytest.raises(ValueError, match="only be issued"):
        FrozenP12TestLedgerToken.issue(
            prepared, registry_hash="c" * 64, state="draft"
        )

    token = FrozenP12TestLedgerToken.issue(prepared, registry_hash="c" * 64)
    dataset = prepared.build_dataset("test", ledger_token=token)
    assert len(dataset) == 2

    tampered = replace(token, registry_hash="d" * 64)
    with pytest.raises(PermissionError, match="not a valid frozen token"):
        prepared.build_dataset("test", ledger_token=tampered)


def test_vendor_adapter_is_delayed_and_receives_privacy_safe_keys() -> None:
    prepared = _prepare(_synthetic_frame())
    captured: dict[str, object] = {}

    class CapturedDataset:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    result = prepared.build_apn_dataset("val", dataset_factory=CapturedDataset)
    assert isinstance(result, CapturedDataset)
    assert captured["prediction_steps"] == 3
    assert captured["observation_time"] == pytest.approx(35.5 / 48.0)
    assert len(captured["tensors"]) == 2  # type: ignore[arg-type]
    assert all(isinstance(key, int) for key in captured["idx_list"])  # type: ignore[union-attr]


def test_public_outputs_and_repr_never_contain_raw_patient_ids() -> None:
    prepared = _prepare(_synthetic_frame())
    train_dataset = prepared.build_dataset("train")
    token = FrozenP12TestLedgerToken.issue(prepared, registry_hash="c" * 64)
    serialized = json.dumps(
        {
            "protocol": prepared.public_manifests(),
            "dataset": train_dataset.public_manifest(),
            "token": token.public_manifest(),
        },
        sort_keys=True,
    )
    raw_ids = [f"RAW-SECRET-PATIENT-{index:02d}" for index in range(20)]
    assert not any(raw_id in serialized for raw_id in raw_ids)
    assert not any(raw_id in repr(prepared) for raw_id in raw_ids)
    assert prepared.time_scale.public_manifest()["kind"] == "fixed_protocol_clock_not_fitted"


def test_rejects_non_p12_shape_and_out_of_clock_time() -> None:
    frame = _synthetic_frame()
    with pytest.raises(ValueError, match="expects 36 channels"):
        prepare_strict_p12(frame)

    bad = frame.reset_index()
    bad.loc[0, "Time"] = 48
    bad = bad.set_index(["RecordID", "Time"])
    with pytest.raises(ValueError, match=r"\[0, 48\)"):
        _prepare(bad)

    scale = P12TimeScale()
    assert scale.observation_time == pytest.approx(35.5 / 48.0)


def test_real_raw_schema_drops_only_audited_constant_mechvent() -> None:
    frame = _synthetic_frame(channels=len(P12_RAW_COLUMNS))
    frame.columns = list(P12_RAW_COLUMNS)
    frame.loc[:, "MechVent"] = np.nan
    frame.loc[frame.index[::2], "MechVent"] = 1.0

    prepared = prepare_strict_p12(frame)
    manifest = prepared.normalizer.public_manifest()

    assert prepared.normalizer.columns == P12_FEATURE_COLUMNS
    assert "Weight" in prepared.normalizer.columns
    assert "MechVent" not in prepared.normalizer.columns
    assert manifest["excluded_constant_columns"] == ["MechVent"]
    assert manifest["source_columns"] == list(P12_RAW_COLUMNS)

    bad = frame.copy()
    bad.loc[bad.index[0], "MechVent"] = 0.0
    with pytest.raises(ValueError, match="MechVent exclusion audit"):
        prepare_strict_p12(bad)
