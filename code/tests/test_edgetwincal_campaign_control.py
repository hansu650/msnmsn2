from __future__ import annotations

from pathlib import Path

import pytest

from edgetwincal.campaign_control import (
    CampaignControlBlocked,
    CampaignControlError,
    _redacted_failure,
    _junit_counts,
    build_arg_parser,
    generate_pretest_evidence,
    resolve_cell,
)
from edgetwincal.config import load_resolved_config


def test_evidence_cli_is_explicit_and_defaults_to_read_only() -> None:
    args = build_arg_parser().parse_args(
        [
            "evidence",
            "--dataset",
            "P12",
            "--protocol",
            "release_parity",
        ]
    )
    assert args.command == "evidence"
    assert args.dataset == "P12"
    assert args.protocol == "release_parity"
    assert args.execute is False
    assert args.fitted_registry is None
    assert args.output is None


def test_evidence_dry_run_never_reads_registry_or_writes_files(
    tmp_path: Path,
) -> None:
    config = load_resolved_config()
    destination = tmp_path / "pretest" / "pretest_evidence.json"
    result = generate_pretest_evidence(
        config,
        resolve_cell("P12", "release_parity"),
        fitted_registry_path=tmp_path / "missing_registry.json",
        output_path=destination,
        execute=False,
    )
    assert result["mode"] == "read_only_evidence_plan"
    assert result["test_constructed"] is False
    assert result["checks"] == {
        "G0": [
            "unit_suite",
            "apn_forward_parity",
            "legacy_metric_parity",
        ],
        "G1": [
            "cache_provenance",
            "split_normalization",
            "fitted_registry",
            "root_boundary",
        ],
    }
    assert not destination.exists()
    assert not destination.parent.exists()


def test_evidence_generation_refuses_nonempty_destination_before_checks(
    tmp_path: Path,
) -> None:
    config = load_resolved_config()
    root = tmp_path / "pretest"
    root.mkdir(parents=True)
    (root / "sentinel.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(CampaignControlError, match="non-empty evidence destination"):
        generate_pretest_evidence(
            config,
            resolve_cell("P12", "release_parity"),
            fitted_registry_path=tmp_path / "missing_registry.json",
            output_path=root / "pretest_evidence.json",
            execute=True,
        )
    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "preserve"


def test_failure_redaction_never_persists_raw_token() -> None:
    token = "a-secret-once-only-token"
    payload = _redacted_failure(
        RuntimeError(f"failure while using {token}"),
        token,
    )
    assert payload["error_type"] == "RuntimeError"
    assert token not in payload["error"]
    assert "<redacted-token>" in payload["error"]

def test_junit_counts_sums_leaf_suites_when_root_has_no_totals(
    tmp_path: Path,
) -> None:
    report = tmp_path / "pytest.xml"
    report.write_text(
        "<?xml version='1.0'?>"
        "<testsuites name='pytest tests'>"
        "<testsuite name='first' tests='2' failures='0' errors='0' skipped='1'/>"
        "<testsuite name='second' tests='3' failures='1' errors='0' skipped='0'/>"
        "</testsuites>",
        encoding="utf-8",
    )
    assert _junit_counts(report) == {
        "tests": 5,
        "failures": 1,
        "errors": 0,
        "skipped": 1,
    }


def test_mimic_is_explicitly_blocked_before_any_path_access() -> None:
    with pytest.raises(CampaignControlBlocked, match="missing_author_mapping"):
        resolve_cell("MIMIC_III", "release_parity")
