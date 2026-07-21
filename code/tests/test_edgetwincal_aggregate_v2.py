from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import edgetwincal.aggregate_v2 as aggregate_v2
from edgetwincal.aggregate_v2 import (
    ConfirmatoryAggregationError,
    aggregate_confirmatory,
    classify_dataset_evidence,
)
from edgetwincal.schema import IncompleteRunError, ManifestError


VARIANT_MSE = {
    "APN": 1.00,
    "SLRH": 0.90,
    "CFG": 0.88,
    "Full": 0.80,
    "V01": 0.95,
    "V02": 0.94,
    "V03": 0.93,
    "V07": 0.82,
    "V08": 0.8004,
    "V10": 0.805,
    "V11": 0.88,
    "V12": 0.97,
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_manifests(
    *,
    strict: bool,
    variants: tuple[str, ...] = tuple(VARIANT_MSE),
) -> list[dict]:
    manifests: list[dict] = []
    for seed in range(2024, 2029):
        checkpoint = _digest(f"checkpoint-{seed}")
        for variant in variants:
            cells = []
            for group_index in range(7):
                n = 5 + group_index
                common = 0.002 * (seed - 2024) + 0.003 * group_index
                mse = VARIANT_MSE[variant] + common
                mae = 0.70 + 0.5 * (VARIANT_MSE[variant] - 1.0) + common * 0.1
                cells.append(
                    {
                        "group_hash": (
                            _digest(f"group-{group_index}")
                            if strict
                            else f"official-record-{group_index}"
                        ),
                        "checkpoint_sha256": checkpoint,
                        "variant": variant,
                        "sse": mse * n,
                        "sae": mae * n,
                        "n": n,
                    }
                )
            split_manifest = (
                {
                    "schema_version": "edgetwincal.protocol.split.v1",
                    "group_ids_reliable": True,
                    "group_id_hash": _digest("strict-groups"),
                }
                if strict
                else {"schema_version": "upstream.record-split.v1"}
            )
            metrics = {"mse": VARIANT_MSE[variant], "mae": 0.5}
            if variant == "V10":
                metrics["diagonal_correction_variance_fraction"] = 0.3
            manifests.append(
                {
                    "status": "complete",
                    "dataset": "P12",
                    "protocol": "strict_p12" if strict else "release_parity",
                    "fold": "fold-0",
                    "seed": seed,
                    "variant_id": variant,
                    "assets": {"checkpoint_sha256": checkpoint},
                    "split_manifest": split_manifest,
                    "cells": cells,
                    "metrics": metrics,
                }
            )
    return manifests


def _install_loader(monkeypatch: pytest.MonkeyPatch, manifests: list[dict]) -> list[Path]:
    paths = [Path("synthetic") / f"run-{index}.json" for index in range(len(manifests))]
    lookup = {str(path): manifest for path, manifest in zip(paths, manifests)}

    def load(path: str | Path) -> dict:
        return copy.deepcopy(lookup[str(path)])

    monkeypatch.setattr(aggregate_v2, "load_complete_manifest", load)
    return paths


def test_strict_crossed_aggregation_holm_and_mechanism_gates_are_compact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests = _synthetic_manifests(strict=True)
    paths = _install_loader(monkeypatch, manifests)
    private_prefix = "C:" + "\\" + "Users"
    private_path = private_prefix + "\\someone\\private\\inventory.json"
    result = aggregate_confirmatory(
        paths,
        [
            {
                "scope": "edge",
                "status": "BLOCKED",
                "reason_code": "NO_EDGE_CPU",
                "reason": "device absent under " + private_path,
            }
        ],
        bootstrap_resamples=500,
    )

    analysis = result["analyses"][0]
    assert analysis["inference"] == "crossed_group_checkpoint"
    assert analysis["counts"] == {
        "complete_manifests": 60,
        "groups": 7,
        "manifest_groups": 7,
        "evaluable_groups": 7,
        "zero_target_groups_excluded": 0,
        "zero_target_cells_excluded": 0,
        "checkpoints": 5,
        "variants": 12,
    }
    assert analysis["G3"]["classification"] == "strong"
    assert 0 <= analysis["G3"]["primary_holm_adjusted_p"] < 0.05
    assert analysis["G2"]["status"] == "PASS"
    assert all(analysis["G2"]["simple_controls_superior"].values())
    assert analysis["G2"]["joint_noninferior"] is True
    assert analysis["G2"]["cfg_shuffle"]["passed"] is True
    assert analysis["G2"]["slrh_shuffle"]["passed"] is True
    assert analysis["G2"]["diagonal"]["decision"] == "CROSS_SENSOR_CLAIM_NOT_REFUTED"
    assert analysis["G2"]["reverse"]["decision"] == "FULL_ORDER_SUPPORTED"
    assert result["gates"]["G2"]["status"] == "PASS"
    assert result["gates"]["G3"]["status"] == "PASS"
    assert result["gates"]["G4"]["status"] == "BLOCKED"
    assert result["holm_families"]["primary"]["comparison_count"] == 1
    assert result["holm_families"]["secondary"]["comparison_count"] == 10
    assert (
        analysis["comparisons"]["V01"]["metrics"]["mse"]["holm_adjusted_p"]
        < 0.05
    )

    serialized = json.dumps(result, sort_keys=True)
    assert _digest("group-0") not in serialized
    assert _digest("checkpoint-2024") not in serialized
    assert private_prefix not in serialized
    assert "<redacted-local-path>" in result["blockers"][0]["reason"]


def test_official_unreliable_groups_are_seed_descriptive_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests = _synthetic_manifests(strict=False, variants=("APN", "Full"))
    paths = _install_loader(monkeypatch, manifests)
    result = aggregate_confirmatory(paths, [], bootstrap_resamples=20)

    analysis = result["analyses"][0]
    assert analysis["strict"] is False
    assert analysis["reliable_group_ids"] is False
    assert analysis["inference"] == "seed_descriptive_only"
    assert analysis["comparisons"] == {}
    assert analysis["seed_descriptive"]["APN"]["seed_count"] == 5
    assert analysis["G3"]["classification"] == "supportive"
    assert analysis["G3"]["confirmatory_interval_available"] is False
    assert analysis["G2"]["status"] == "NOT_APPLICABLE"
    assert result["gates"]["G3"]["status"] == "BLOCKED"
    assert result["gates"]["G4"]["status"] == "BLOCKED"


def test_aggregate_audits_shared_globally_empty_target_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests = _synthetic_manifests(strict=True, variants=("APN", "Full"))
    for manifest in manifests:
        manifest["cells"].append(
            {
                "group_hash": _digest("globally-empty-group"),
                "checkpoint_sha256": manifest["assets"]["checkpoint_sha256"],
                "variant": manifest["variant_id"],
                "sse": 0.0,
                "sae": 0.0,
                "n": 0,
            }
        )
    paths = _install_loader(monkeypatch, manifests)
    result = aggregate_confirmatory(paths, [], bootstrap_resamples=20)

    counts = result["analyses"][0]["counts"]
    assert counts["manifest_groups"] == 8
    assert counts["groups"] == 7
    assert counts["evaluable_groups"] == 7
    assert counts["zero_target_groups_excluded"] == 1
    assert counts["zero_target_cells_excluded"] == 10


@pytest.mark.parametrize(
    "failure",
    [
        ManifestError("synthetic corrupt manifest"),
        IncompleteRunError("synthetic failed run"),
        FileNotFoundError("synthetic missing manifest"),
    ],
)
def test_any_expected_missing_failed_or_corrupt_run_aborts_without_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    complete = _synthetic_manifests(strict=False, variants=("APN",))[0]

    def load(path: str | Path) -> dict:
        if Path(path).name == "expected-bad.json":
            raise failure
        return copy.deepcopy(complete)

    monkeypatch.setattr(aggregate_v2, "load_complete_manifest", load)
    with pytest.raises(ConfirmatoryAggregationError, match="Expected manifest #1"):
        aggregate_confirmatory(
            [Path("registry") / "expected-good.json", Path("registry") / "expected-bad.json"],
            [],
            bootstrap_resamples=10,
        )


def test_g3_harm_requires_interval_confirmation() -> None:
    inconclusive = classify_dataset_evidence(
        relative_mse_gain=-0.003,
        mse_effect_ci_low=-0.01,
        mse_effect_ci_high=0.01,
        relative_mae_gain_ci_low=-0.01,
        relative_mae_gain_ci_high=0.001,
        improved_checkpoints=2,
        checkpoint_count=5,
        inferential=True,
    )
    harmful = classify_dataset_evidence(
        relative_mse_gain=-0.01,
        mse_effect_ci_low=0.001,
        mse_effect_ci_high=0.02,
        relative_mae_gain_ci_low=-0.02,
        relative_mae_gain_ci_high=-0.003,
        improved_checkpoints=0,
        checkpoint_count=5,
        inferential=True,
    )
    neutral = classify_dataset_evidence(
        relative_mse_gain=-0.001,
        mse_effect_ci_low=-0.01,
        mse_effect_ci_high=0.01,
        relative_mae_gain_ci_low=-0.001,
        relative_mae_gain_ci_high=0.001,
        improved_checkpoints=2,
        checkpoint_count=5,
        inferential=True,
    )
    assert inconclusive == "safety-inconclusive"
    assert harmful == "harmful"
    assert neutral == "neutral"


def test_g3_strong_requires_holm_and_mae_safety() -> None:
    common = {
        "relative_mse_gain": 0.01,
        "mse_effect_ci_low": -0.02,
        "mse_effect_ci_high": -0.001,
        "improved_checkpoints": 5,
        "checkpoint_count": 5,
        "inferential": True,
    }
    assert (
        classify_dataset_evidence(
            **common,
            relative_mae_gain_ci_low=0.001,
            relative_mae_gain_ci_high=0.01,
        )
        == "supportive"
    )
    assert (
        classify_dataset_evidence(
            **common,
            relative_mae_gain_ci_low=0.001,
            relative_mae_gain_ci_high=0.01,
            primary_holm_adjusted_p=0.01,
        )
        == "strong"
    )
    assert (
        classify_dataset_evidence(
            **common,
            relative_mae_gain_ci_low=-0.003,
            relative_mae_gain_ci_high=0.01,
            primary_holm_adjusted_p=0.01,
        )
        == "safety-inconclusive"
    )


def test_duplicate_expected_paths_are_rejected_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aggregate_v2,
        "load_complete_manifest",
        lambda path: pytest.fail("loader must not run for a duplicate registry"),
    )
    with pytest.raises(ConfirmatoryAggregationError, match="duplicates"):
        aggregate_confirmatory(["same.json", "same.json"], [])
