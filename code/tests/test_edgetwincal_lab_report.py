from __future__ import annotations

import json

from edgetwincal.lab_report import ANALYSIS_DIR, _svg_bar_chart, build_gate_decision


def _aggregate() -> dict:
    return json.loads(
        (ANALYSIS_DIR / "confirmatory_aggregate.json").read_text(encoding="utf-8")
    )


def test_frozen_gate_decision_is_abandon() -> None:
    decision = build_gate_decision(
        _aggregate(),
        {"G0": {"status": "PASS"}, "G1": {"status": "PASS"}},
    )
    assert decision["verdict"] == "ABANDON"
    assert decision["gates"]["G2"]["status"] == "FAIL"
    assert decision["gates"]["G3_strict"]["status"] == "FAIL"
    assert decision["gates"]["G3_release_scope"]["status"] == "FAIL"
    assert decision["gates"]["G4"]["status"] == "BLOCKED"
    assert decision["claim_actions"]["same_test_retuning"] == "PROHIBITED"
    assert decision["claim_actions"]["current_APN_route"].startswith("STOP_")


def test_strict_classifications_and_ci_semantics() -> None:
    aggregate = _aggregate()
    strict = {
        row["dataset"]: row
        for row in aggregate["analyses"]
        if row["strict"]
    }
    assert strict["P12"]["G3"]["classification"] == "strong"
    assert strict["USHCN"]["G3"]["classification"] == "harmful"
    p12 = strict["P12"]["comparisons"]["APN"]["metrics"]["mse"]
    assert p12["effect_ci_high"] < 0
    assert p12["holm_adjusted_p"] < 0.05
    ushcn_mae = strict["USHCN"]["comparisons"]["APN"]["metrics"]["mae"]
    assert ushcn_mae["relative_loss_ci_low"] > 0.002


def test_svg_bar_chart_is_standalone_and_labeled() -> None:
    svg = _svg_bar_chart(
        [
            ("P12", [("APN", 0.293), ("Full", 0.291)]),
            ("USHCN", [("APN", 0.181), ("Full", 0.445)]),
        ],
        title="Strict result",
        y_label="MSE",
    )
    assert svg.startswith("<svg ")
    assert "P12" in svg and "USHCN" in svg
    assert "APN" in svg and "Full" in svg
    assert svg.rstrip().endswith("</svg>")
