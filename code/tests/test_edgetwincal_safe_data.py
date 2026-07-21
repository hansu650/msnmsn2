from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import csv
import io
import json
import gzip
from pathlib import Path
import zipfile

import pytest
import torch
from edgetwincal.apn_bridge import _private_id, extract_apn_batch

from edgetwincal.paths import PROJECT_ROOT
from edgetwincal.safe_config import PartitionSpec, load_safe_config
from edgetwincal.safe_data import (
    MaskedWindowDataset,
    as_apn_batch,
    build_pretest_loaders,
    fit_train_normalizer,
    freeze_test_ledger,
    load_pretest_rows,
    manifest_existing_source,
    open_sealed_test_loader,
    prepare_pretest_shards,
    require_safe_path,
    stable_pseudonymous_id,
)


HEADER = [
    "No", "year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "NO2",
    "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station",
]


def _tiny_beijing():
    base = load_safe_config().dataset("beijing_air")
    start = datetime(2013, 1, 1)
    names = ("train", "val", "adapter", "val_select", "val_safety", "test")
    partitions = tuple(
        PartitionSpec(
            name, start + timedelta(days=index), start + timedelta(days=index + 1)
        )
        for index, name in enumerate(names)
    )
    return replace(
        base,
        channels=("Aotizhongxin",),
        frequency_seconds=3600,
        history_steps=2,
        forecast_steps=1,
        stride_steps=1,
        partitions=partitions,
    )


def _csv_member(spec, *, reverse=False, invalid_test=False):
    rows = []
    number = 0
    timestamp = spec.partition("train").start
    while timestamp < spec.partition("test").end:
        number += 1
        value = (
            "SEALED_NOT_NUMERIC"
            if invalid_test and timestamp >= spec.test_start
            else float(number)
        )
        rows.append(
            [
                number, timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
                value, 0, 0, 0, 0, 0, 0, 0, 0, 0, "N", 0, "Aotizhongxin",
            ]
        )
        timestamp += timedelta(hours=1)
    if reverse:
        rows.reverse()
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(rows)
    return output.getvalue()


def _prepare(root: Path, *, reverse=False, invalid_test=False):
    spec = _tiny_beijing()
    raw_dir = root / "incoming"
    raw_dir.mkdir(parents=True)
    source = raw_dir / "synthetic_beijing.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
            _csv_member(spec, reverse=reverse, invalid_test=invalid_test),
        )
    raw_manifest = manifest_existing_source(root, spec, source)
    manifest = prepare_pretest_shards(root, spec, raw_manifest)
    return spec, manifest


def test_root_boundary_rejects_escape(tmp_path):
    assert require_safe_path(tmp_path, "inside").is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="escapes Safe root"):
        require_safe_path(tmp_path, "..", "outside")
    with pytest.raises(ValueError, match="escapes project root"):
        require_safe_path(PROJECT_ROOT.parent / "msn")


def test_timestamp_router_never_parses_test_numeric_values(tmp_path):
    spec, manifest = _prepare(tmp_path, invalid_test=True)
    assert manifest.sealed_test_rows == 24
    train = load_pretest_rows(tmp_path, spec, "train")
    normalizer = fit_train_normalizer(train)
    ledger = freeze_test_ledger(tmp_path, spec, {"config": "synthetic"})
    with pytest.raises(ValueError):
        open_sealed_test_loader(tmp_path, spec, ledger.token, normalizer, 2024)
    state = json.loads(ledger.ledger_path.read_text(encoding="utf-8"))
    assert state["status"] == "failed_consumed"


def test_train_only_normalizer_and_apn_batch_contract(tmp_path):
    spec, _ = _prepare(tmp_path)
    train = load_pretest_rows(tmp_path, spec, "train")
    normalizer = fit_train_normalizer(train)
    assert normalizer.fit_split == "train"
    assert normalizer.observed_count == (24,)
    assert normalizer.mean[0] == pytest.approx(12.5)

    rows = load_pretest_rows(tmp_path, spec, "all")
    item = MaskedWindowDataset(spec, "val", rows, normalizer)[0]
    assert set(item) == {
        "x", "x_mark", "y", "y_mark", "x_mask", "y_mask", "sample_id", "group_id"
    }
    assert item["x"].shape == (2, 1)
    assert item["y"].shape == (1, 1)
    assert item["sample_id"].dtype == torch.int64
    torch.testing.assert_close(item["x_mark"][:, 0], torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(item["y_mark"][:, 0], torch.tensor([1.25]))
    apn = as_apn_batch(item)
    assert "sample_ID" in apn and "sample_id" not in apn
    assert "group_ID" in apn and "group_id" not in apn
    assert torch.equal(apn["group_ID"], item["group_id"])

    class _Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(3, 1, bias=False)

        def LearnableTE(self, times):
            return times

        def IMTS_Model_Logic(self, values, masks, times):
            del masks, times
            return values.mean(dim=1).reshape(self.batch_size, -1, 2)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Core()

    batched = as_apn_batch({key: value.unsqueeze(0) for key, value in item.items()})
    extracted = extract_apn_batch(
        _Model(), batched, dataset_id="beijing_air", protocol_id="safe_v1",
        split="val", device="cpu",
    )
    expected_group = _private_id(
        "beijing_air", "safe_v1", "val", int(item["group_id"]),
        include_split=False,
    )
    assert extracted["group_id"].item() == expected_group


def test_split_loaders_are_disjoint_and_test_is_once_only(tmp_path):
    spec, _ = _prepare(tmp_path)
    normalizer = fit_train_normalizer(load_pretest_rows(tmp_path, spec, "train"))
    loaders = build_pretest_loaders(
        spec, normalizer, 2024, root=tmp_path, batch_size=4
    )
    assert loaders.val_select.dataset.group_ids().isdisjoint(
        loaders.val_safety.dataset.group_ids()
    )
    assert loaders.val_select.dataset.target_timestamp_keys().isdisjoint(
        loaders.val_safety.dataset.target_timestamp_keys()
    )

    ledger = freeze_test_ledger(
        tmp_path, spec, {"normalizer_sha256": normalizer.sha256}
    )
    with pytest.raises(PermissionError):
        open_sealed_test_loader(tmp_path, spec, "wrong-token", normalizer, 2024)
    test_loader = open_sealed_test_loader(
        tmp_path, spec, ledger.token, normalizer, 2024, batch_size=8
    )
    batch = next(iter(test_loader))
    assert batch["x"].shape[-2:] == (2, 1)
    with pytest.raises(RuntimeError):
        open_sealed_test_loader(tmp_path, spec, ledger.token, normalizer, 2024)


def test_ids_and_rows_are_input_reorder_deterministic(tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    spec_a, manifest_a = _prepare(first_root, reverse=False)
    spec_b, manifest_b = _prepare(second_root, reverse=True)
    rows_a = load_pretest_rows(first_root, spec_a, "all")
    rows_b = load_pretest_rows(second_root, spec_b, "all")
    assert list(zip(rows_a.timestamps, rows_a.channel_indices, rows_a.values)) == list(
        zip(rows_b.timestamps, rows_b.channel_indices, rows_b.values)
    )
    start = spec_a.partition("val_select").start
    assert stable_pseudonymous_id(spec_a, "sample", start.isoformat()) == (
        stable_pseudonymous_id(spec_b, "sample", start.isoformat())
    )
    assert manifest_a.observations_sha256 == manifest_b.observations_sha256


def test_intel_whole_three_hour_hash_groups_do_not_overlap():
    from edgetwincal.safe_data import group_anchor, iter_target_starts

    spec = load_safe_config().dataset("intel_lab")
    select = {
        group_anchor(spec, target)
        for target in iter_target_starts(spec, "val_select")
    }
    safety = {
        group_anchor(spec, target)
        for target in iter_target_starts(spec, "val_safety")
    }
    assert select and safety and select.isdisjoint(safety)

def test_intel_timestamp_only_rows_are_audited_and_skipped(tmp_path):
    base = load_safe_config().dataset("intel_lab")
    start = datetime(2004, 2, 28)
    names = ("train", "val", "adapter", "validation_pool", "test")
    partitions = tuple(
        PartitionSpec(
            name, start + timedelta(days=index), start + timedelta(days=index + 1)
        )
        for index, name in enumerate(names)
    )
    spec = replace(
        base, channels=(1,), history_steps=2, forecast_steps=1, stride_steps=1,
        partitions=partitions,
    )
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "synthetic_intel.txt.gz"
    rows = (
        "2004-02-28 00:00:00.000000 1 1 20.0 30.0 40.0 3.0\n"
        "2004-02-28 00:01:00.000000 2\n"
        "2004-03-03 00:00:00.000000 3 1 21.0 31.0 41.0 3.1\n"
        "2004-03-03 00:01:00.000000 4\n"
    )
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.writelines(rows)
    raw = manifest_existing_source(tmp_path, spec, source)
    manifest = prepare_pretest_shards(tmp_path, spec, raw)
    assert manifest.pretest_rows == 1
    assert manifest.sealed_test_rows == 1
    assert manifest.discarded_unusable_rows == 2
    prepared = load_pretest_rows(tmp_path, spec, "train")
    assert prepared.values == (20.0,)
