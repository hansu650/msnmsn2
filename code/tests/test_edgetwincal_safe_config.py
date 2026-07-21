from __future__ import annotations

import json

import pytest

from edgetwincal.safe_config import (
    DEFAULT_SAFE_CONFIG,
    SafeConfigError,
    canonical_sha256,
    load_safe_config,
)


def test_frozen_safe_config_exact_protocol():
    config = load_safe_config()
    assert config.seeds == (2024, 2025, 2026, 2027, 2028)
    assert config.main_variants == ("APN", "Joint", "Full", "Safe")
    assert config.apn.commit == "f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4"
    assert (config.apn.d_model, config.apn.npatch, config.apn.te_dim) == (24, 12, 8)

    beijing = config.dataset("beijing_air")
    assert len(beijing.channels) == 12
    assert (beijing.history_steps, beijing.forecast_steps, beijing.stride_steps) == (
        72, 24, 3
    )
    assert beijing.partition("test").start.isoformat() == "2016-09-01T00:00:00"

    intel = config.dataset("intel_lab")
    assert intel.channels == tuple(range(1, 55))
    assert (intel.history_steps, intel.forecast_steps, intel.stride_steps) == (72, 12, 3)
    assert intel.partition("val_select").name == "validation_pool"


def test_canonical_hash_ignores_mapping_order():
    assert canonical_sha256({"b": 1, "a": [2]}) == canonical_sha256(
        {"a": [2], "b": 1}
    )


def test_strict_config_rejects_unknown_key(tmp_path):
    raw = json.loads(DEFAULT_SAFE_CONFIG.read_text(encoding="utf-8"))
    raw["campaign"]["surprise"] = True
    path = tmp_path / "unsafe_config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SafeConfigError, match="unexpected"):
        load_safe_config(path)


def test_frozen_config_rejects_partition_drift(tmp_path):
    raw = json.loads(DEFAULT_SAFE_CONFIG.read_text(encoding="utf-8"))
    raw["datasets"]["beijing_air"]["partitions"][0]["end"] = "2015-12-31T00:00:00"
    raw["datasets"]["beijing_air"]["partitions"][1]["start"] = "2015-12-31T00:00:00"
    path = tmp_path / "drifted_config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SafeConfigError, match="Beijing protocol"):
        load_safe_config(path)


def test_frozen_config_rejects_auxiliary_drift(tmp_path):
    for section, key, value in (
        ("robust_fit", "alpha_latent", [99]),
        ("safe_envelope", "kappa", [99.0]),
        ("validation_gate", "minimum_target_cells", 1),
        ("statistics", "joint_noninferiority_margin", 0.5),
    ):
        raw = json.loads(DEFAULT_SAFE_CONFIG.read_text(encoding="utf-8"))
        raw[section][key] = value
        path = tmp_path / f"drifted_{section}.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SafeConfigError, match="values are frozen"):
            load_safe_config(path)
