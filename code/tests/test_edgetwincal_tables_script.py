from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "scripts" / "build_edgetwincal_tables.mjs"


def test_table_builder_declares_required_inputs_and_outputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "confirmatory_aggregate.json",
        "gate_decision.json",
        "failure_diagnosis.json",
        "EdgeTwinCal_lab_results.xlsx",
        "dataset_variant_summary.csv",
        "seed_summary.csv",
        "paired_comparisons.csv",
        "gate_summary.csv",
    ):
        assert name in source


def test_table_builder_uses_artifact_tool_formulas_and_full_qa() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "@oai/artifact-tool" in source
    assert "openpyxl" not in source
    assert "xlsxwriter" not in source
    assert ".formulas =" in source
    assert "Relative MSE gain vs APN" in source
    assert "workbook.inspect" in source
    assert "workbook.render" in source
    for sheet in (
        "Results",
        "Seed metrics",
        "Paired CIs",
        "Gate audit",
        "Provenance",
    ):
        assert sheet in source


def test_table_builder_documents_ci_and_holm_as_distinct_quantities() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raw crossed group-by-checkpoint percentile 95% CIs" in source
    assert "Holm correction applies to the one-sided bootstrap p-value only" in source
