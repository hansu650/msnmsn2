from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "prepare_mimic_iii.py"
SPEC = importlib.util.spec_from_file_location("prepare_mimic_iii", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
mimic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mimic
SPEC.loader.exec_module(mimic)


def _admission_row(
    subject: int,
    hadm: int,
    admit: str,
    discharge: str,
    *,
    admission_type: str = "EMERGENCY",
    chart: int = 1,
) -> dict:
    return {
        "SUBJECT_ID": subject,
        "HADM_ID": hadm,
        "ADMITTIME": admit,
        "DISCHTIME": discharge,
        "DEATHTIME": np.nan,
        "ADMISSION_TYPE": admission_type,
        "HAS_CHARTEVENTS_DATA": chart,
    }


def test_admission_filter_matches_notebook_contract() -> None:
    admissions = pd.DataFrame(
        [
            _admission_row(1, 101, "2100-01-01 00:00:00", "2100-01-05 00:00:00"),
            _admission_row(2, 201, "2100-01-01 00:00:00", "2100-01-05 00:00:00"),
            _admission_row(2, 202, "2101-01-01 00:00:00", "2101-01-05 00:00:00"),
            _admission_row(3, 301, "2100-01-01 00:00:00", "2100-01-03 00:00:00"),
            _admission_row(4, 401, "2100-01-01 00:00:00", "2100-01-05 00:00:00"),
            _admission_row(
                5,
                501,
                "2100-01-01 00:00:00",
                "2100-01-05 00:00:00",
                chart=0,
            ),
            _admission_row(
                6,
                601,
                "2100-01-01 00:00:00",
                "2100-01-05 00:00:00",
                admission_type="NEWBORN",
            ),
        ]
    )
    patients = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3, 4, 5, 6],
            "DOB": [
                "2070-01-01 00:00:00",
                "2070-01-01 00:00:00",
                "2070-01-01 00:00:00",
                "2092-01-01 00:00:00",
                "2070-01-01 00:00:00",
                "2070-01-01 00:00:00",
            ],
        }
    )

    result = mimic.prepare_admissions(admissions, patients)

    assert result["HADM_ID"].tolist() == [101]
    assert result["ELAPSED_DAYS"].tolist() == [4]
    assert result["DEATHTAG"].tolist() == [0]


def test_admission_age_reproduces_historic_ns_wrap_and_audits_true_age() -> None:
    admissions = pd.DataFrame(
        [
            _admission_row(1, 101, "2117-01-20 12:15:00", "2117-01-24 12:15:00"),
            _admission_row(2, 201, "2100-01-01 00:00:00", "2100-01-05 00:00:00"),
            _admission_row(3, 301, "2100-01-01 00:00:00", "2100-01-05 00:00:00"),
        ]
    )
    patients = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3],
            "DOB": [
                "1817-01-20 00:00:00",
                "2070-01-01 00:00:00",
                "2086-01-01 00:00:00",
            ],
        }
    )
    audit: dict[str, object] = {}

    result = mimic.prepare_admissions(admissions, patients, audit=audit)

    assert result["HADM_ID"].tolist() == [201]
    assert audit == {
        "semantics": "historic_pandas_0.23_int64_ns_wrap",
        "one_visit_rows": 3,
        "stay_3_to_29_days_rows": 3,
        "historic_age_over_15_rows": 1,
        "chart_events_rows": 1,
        "final_rows": 1,
        "true_age_sensitivity": {
            "age_over_15_rows": 2,
            "final_rows": 2,
            "historic_excluded_but_true_age_included_before_chart": 1,
            "historic_excluded_but_true_age_included_final": 1,
            "semantics": "seconds-resolution subtraction; sensitivity only",
        },
    }


def _input_row(**overrides) -> dict:
    row = {
        "SUBJECT_ID": 1,
        "HADM_ID": 101,
        "STARTTIME": "2100-01-01 01:00:00",
        "ENDTIME": "2100-01-01 00:00:00",
        "ITEMID": 1,
        "AMOUNT": -100.0,
        "AMOUNTUOM": "ml",
        "RATE": np.nan,
        "RATEUOM": np.nan,
        "PATIENTWEIGHT": 70.0,
        "ORDERCATEGORYDESCRIPTION": "Continuous IV",
    }
    row.update(overrides)
    return row


def test_input_negative_swap_and_half_hour_expansion() -> None:
    raw = pd.DataFrame([_input_row()])
    labels = pd.DataFrame({"ITEMID": [1], "LABEL": ["Albumin 5%"]})

    cleaned = mimic.clean_input_events(raw, labels)
    expanded = mimic.expand_input_events(cleaned)

    assert cleaned.iloc[0]["STARTTIME"] == pd.Timestamp("2100-01-01 00:00:00")
    assert cleaned.iloc[0]["ENDTIME"] == pd.Timestamp("2100-01-01 01:00:00")
    assert expanded["AMOUNT"].tolist() == pytest.approx([50.0, 50.0])
    assert expanded["CHARTTIME"].tolist() == [
        pd.Timestamp("2100-01-01 00:00:00"),
        pd.Timestamp("2100-01-01 00:30:00"),
    ]


def test_zero_amount_rewritten_shape_retains_upstream_negative_duration() -> None:
    raw = pd.DataFrame(
        [
            _input_row(
                STARTTIME="2146-05-16 20:00:00",
                ENDTIME="2146-05-16 19:37:00",
                AMOUNT=0.0,
                RATE=np.nan,
                RATEUOM=np.nan,
                ORDERCATEGORYDESCRIPTION="Bolus",
            )
        ]
    )
    labels = pd.DataFrame({"ITEMID": [1], "LABEL": ["GT Flush"]})

    cleaned = mimic.clean_input_events(raw, labels)
    audit = mimic.summarize_input_duration_audit(cleaned)
    expanded = mimic.expand_input_events(cleaned)

    assert cleaned["DURATION"].iloc[0] == pd.Timedelta(minutes=-23)
    assert audit == {
        "negative_duration_rows_retained": 1,
        "negative_duration_zero_amount_rows": 1,
        "negative_duration_missing_rate_rows": 1,
        "negative_duration_rows_by_label": {"GT Flush": 1},
        "zero_duration_rows_retained": 0,
    }
    assert expanded["CHARTTIME"].tolist() == [pd.Timestamp("2146-05-16 20:00:00")]
    assert expanded["AMOUNT"].tolist() == [0.0]



def test_input_null_duration_still_fails_closed() -> None:
    raw = pd.DataFrame([_input_row(ENDTIME=np.nan, AMOUNT=0.0)])
    labels = pd.DataFrame({"ITEMID": [1], "LABEL": ["GT Flush"]})

    with pytest.raises(mimic.PreprocessError, match="null duration"):
        mimic.clean_input_events(raw, labels)



def test_input_rate_amount_mismatch_fails_closed() -> None:
    raw = pd.DataFrame(
        [
            _input_row(
                STARTTIME="2100-01-01 00:00:00",
                ENDTIME="2100-01-01 01:00:00",
                AMOUNT=5.0,
                RATE=10.0,
                RATEUOM="mL/hour",
            )
        ]
    )
    labels = pd.DataFrame({"ITEMID": [1], "LABEL": ["Albumin 5%"]})
    cleaned = mimic.clean_input_events(raw, labels)
    with pytest.raises(mimic.PreprocessError, match="rate/amount mismatch"):
        mimic.validate_input_rate_amount(cleaned)

def test_prescription_ranges_and_units_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 1],
            "HADM_ID": [101, 101, 101],
            "STARTDATE": ["2100-01-01 00:00:00"] * 3,
            "DRUG": ["Aspirin", "D5W", "D5W"],
            "DOSE_VAL_RX": ["10-20", "250", "not-a-number"],
            "DOSE_UNIT_RX": ["mg", "ml", "ml"],
        }
    )

    result = mimic.clean_prescriptions(frame)

    assert result["DOSE_VAL_RX"].tolist() == pytest.approx([15.0, 250.0])
    assert result["DOSE_UNIT_RX"].tolist() == ["mg", "mL"]

def test_prescription_date_only_startdate_uses_upstream_midnight_semantics() -> None:
    values = pd.Series(
        ["2175-06-11", "2175-06-12 03:04:05", pd.NA],
        dtype="string",
    )
    audit: dict[str, object] = {}

    parsed = mimic.parse_mimic_datetime(
        values,
        name="prescription STARTDATE",
        audit=audit,
    )

    assert parsed.iloc[0] == pd.Timestamp("2175-06-11 00:00:00")
    assert parsed.iloc[1] == pd.Timestamp("2175-06-12 03:04:05")
    assert pd.isna(parsed.iloc[2])
    assert audit == {
        "total_rows": 3,
        "null_rows": 1,
        "second_resolution_rows": 1,
        "date_only_rows_promoted_to_midnight": 1,
        "date_only_examples": ["2175-06-11"],
        "invalid_rows": 0,
        "invalid_examples": [],
    }

    with pytest.raises(mimic.PreprocessError, match="2175/06/11"):
        mimic.parse_mimic_datetime(
            pd.Series(["2175/06/11"], dtype="string"),
            name="prescription STARTDATE",
        )




def test_output_iserror_check_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "SUBJECT_ID": [1],
            "HADM_ID": [101],
            "CHARTTIME": ["2100-01-01 00:00:00"],
            "ITEMID": [1],
            "VALUE": [1.0],
            "VALUEUOM": ["mL"],
            "ISERROR": [1.0],
        }
    )
    labels = pd.DataFrame({"ITEMID": [1], "LABEL": ["Foley"]})

    with pytest.raises(mimic.PreprocessError, match="ISERROR"):
        mimic.clean_output_events(frame, labels)


def _two_time_frame(value_name: str, label_name: str, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HADM_ID": [10, 10, 10],
            "CHARTTIME": [
                "2100-01-01 00:00:00",
                "2100-01-01 00:00:00",
                "2100-01-01 00:30:00",
            ],
            value_name: [2.0, 4.0, 8.0],
            label_name: [label, label, label],
        }
    )


def test_complete_tensor_uses_origin_specific_aggregation() -> None:
    inputs = _two_time_frame("AMOUNT", "LABEL", "Input A")
    labs = _two_time_frame("VALUENUM", "LABEL", "Lab A")
    outputs = _two_time_frame("VALUE", "LABEL", "Output A")
    prescriptions = _two_time_frame("DOSE_VAL_RX", "DRUG", "Drug A")
    diagnoses = pd.DataFrame({"HADM_ID": [10], "SEQ_NUM": [1], "ICD9_CODE": ["1234"]})

    tensor, labels = mimic.merge_complete_tensor(
        inputs,
        labs,
        outputs,
        prescriptions,
        unique_ids={10: 0},
        diagnoses=diagnoses,
        bin_k=mimic.APN_REFERENCE_BIN_K,
        minimum_observations=1,
    )

    assert tuple(tensor.columns) == mimic.OUTPUT_COLUMNS
    assert tensor.shape == (8, 7)
    assert labels["LABEL"].tolist() == ["Input A", "Lab A", "Output A", "Drug A"]
    assert sorted(tensor["TIME_STAMP"].unique().tolist()) == [0, 1]

    lab_code = int(labels.loc[labels["LABEL"] == "Lab A", "LABEL_CODE"].iloc[0])
    input_code = int(labels.loc[labels["LABEL"] == "Input A", "LABEL_CODE"].iloc[0])
    assert tensor.loc[
        (tensor["LABEL_CODE"] == lab_code) & (tensor["TIME_STAMP"] == 0), "VALUENUM"
    ].iloc[0] == pytest.approx(3.0)
    assert tensor.loc[
        (tensor["LABEL_CODE"] == input_code) & (tensor["TIME_STAMP"] == 0), "VALUENUM"
    ].iloc[0] == pytest.approx(6.0)
    assert np.isfinite(tensor.to_numpy(dtype=float)).all()

def test_apn_mimic_time_binning_rejects_conflicting_upstream_values() -> None:
    inputs = _two_time_frame("AMOUNT", "LABEL", "Input A")
    labs = _two_time_frame("VALUENUM", "LABEL", "Lab A")
    outputs = _two_time_frame("VALUE", "LABEL", "Output A")
    prescriptions = _two_time_frame("DOSE_VAL_RX", "DRUG", "Drug A")
    diagnoses = pd.DataFrame(
        {"HADM_ID": [10], "SEQ_NUM": [1], "ICD9_CODE": ["1234"]}
    )

    with pytest.raises(mimic.PreprocessError, match="requires bin_k=2"):
        mimic.merge_complete_tensor(
            inputs,
            labs,
            outputs,
            prescriptions,
            unique_ids={10: 0},
            diagnoses=diagnoses,
            bin_k=60,
            minimum_observations=1,
        )




def test_nonreference_mapping_is_sorted_and_explicit() -> None:
    def pair(hadm: int, label: str, value_name: str, label_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "HADM_ID": [hadm, hadm],
                "CHARTTIME": ["2100-01-01 00:00:00", "2100-01-01 00:30:00"],
                value_name: [1.0, 2.0],
                label_name: [label, label],
            }
        )

    inputs = pd.concat(
        [pair(20, "Input A", "AMOUNT", "LABEL"), pair(10, "Input A", "AMOUNT", "LABEL")]
    )
    labs = pd.concat(
        [pair(20, "Lab A", "VALUENUM", "LABEL"), pair(10, "Lab A", "VALUENUM", "LABEL")]
    )
    outputs = pd.concat(
        [pair(20, "Output A", "VALUE", "LABEL"), pair(10, "Output A", "VALUE", "LABEL")]
    )
    prescriptions = pd.concat(
        [
            pair(20, "Drug A", "DOSE_VAL_RX", "DRUG"),
            pair(10, "Drug A", "DOSE_VAL_RX", "DRUG"),
        ]
    )
    diagnoses = pd.DataFrame(
        {
            "HADM_ID": [10, 20],
            "SEQ_NUM": [1, 1],
            "ICD9_CODE": ["1234", "5678"],
        }
    )

    tensor, _ = mimic.merge_complete_tensor(
        inputs,
        labs,
        outputs,
        prescriptions,
        unique_ids=None,
        bin_k=mimic.APN_REFERENCE_BIN_K,
        diagnoses=diagnoses,
        reference_mode=False,
        minimum_observations=1,
    )

    # HADM 10 is mapped before HADM 20 regardless of source row order.
    assert sorted(tensor["UNIQUE_ID"].unique().tolist()) == [0, 1]
    assert (tensor["UNIQUE_ID"].value_counts().sort_index().to_numpy() == [8, 8]).all()

def test_nonreference_mode_still_requires_diagnoses() -> None:
    frame = _two_time_frame("AMOUNT", "LABEL", "Input A")
    with pytest.raises(mimic.PreprocessError, match="all preprocessing modes require"):
        mimic.merge_complete_tensor(
            frame,
            frame.rename(columns={"AMOUNT": "VALUENUM"}),
            frame.rename(columns={"AMOUNT": "VALUE"}),
            frame.rename(columns={"AMOUNT": "DOSE_VAL_RX", "LABEL": "DRUG"}),
            unique_ids=None,
            bin_k=mimic.APN_REFERENCE_BIN_K,
            diagnoses=None,
            reference_mode=False,
            minimum_observations=1,
        )



def test_reference_dependencies_fail_before_source_scan(tmp_path: Path) -> None:
    parser = mimic.build_parser()
    args = parser.parse_args(
        [
            "--source-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "complete_tensor.csv"),
            "--diagnoses-icd",
            str(tmp_path / "DIAGNOSES_ICD.csv"),
            "--unique-id-map",
            str(tmp_path / "UNIQUE_ID_dict.csv"),
        ]
    )

    with pytest.raises(mimic.PreprocessError, match="canonical preprocessing requires"):
        mimic.run_pipeline(args)
    assert not (tmp_path / "complete_tensor.csv").exists()


def test_nonreference_filename_cannot_impersonate_canonical(tmp_path: Path) -> None:
    parser = mimic.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "deterministic-nonreference",
            "--source-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "complete_tensor.csv"),
        ]
    )

    with pytest.raises(mimic.PreprocessError, match="must not be named"):
        mimic.run_pipeline(args)



def test_deterministic_mode_preflights_diagnoses_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mimic, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        mimic,
        "verify_pinned_notebooks",
        lambda _vendor_root: {"commit": "synthetic-test"},
    )
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (tmp_path / "source").mkdir()
    parser = mimic.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "deterministic-nonreference",
            "--source-dir",
            str(tmp_path / "source"),
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "tensor.NONREFERENCE.csv"),
            "--diagnoses-icd",
            str(tmp_path / "missing_DIAGNOSES_ICD.csv"),
            "--vendor-root",
            str(vendor),
        ]
    )

    with pytest.raises(
        mimic.PreprocessError,
        match="deterministic-nonreference mode still requires DIAGNOSES_ICD",
    ):
        mimic.run_pipeline(args)
    assert not (tmp_path / "work" / "source_manifest.json").exists()


def test_audit_manifest_tracks_auxiliary_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mimic, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        mimic,
        "verify_pinned_notebooks",
        lambda _vendor_root: {"commit": "synthetic-test"},
    )
    source = tmp_path / "source"
    source.mkdir()
    for filename, columns in mimic.PRIMARY_SOURCES.items():
        pd.DataFrame(columns=columns).to_csv(source / filename, index=False)
    diagnoses = tmp_path / "DIAGNOSES_ICD.csv"
    mapping = tmp_path / "UNIQUE_ID_dict.csv"
    pd.DataFrame(
        {"HADM_ID": [10], "SEQ_NUM": [1], "ICD9_CODE": ["1234"]}
    ).to_csv(diagnoses, index=False)
    pd.DataFrame({"HADM_ID": [10], "UNIQUE_ID": [0]}).to_csv(mapping, index=False)
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    work = tmp_path / "work"

    args = mimic.build_parser().parse_args(
        [
            "--audit-only",
            "--mode",
            "deterministic-nonreference",
            "--source-dir",
            str(source),
            "--work-dir",
            str(work),
            "--output",
            str(tmp_path / "tensor.NONREFERENCE.csv"),
            "--diagnoses-icd",
            str(diagnoses),
            "--unique-id-map",
            str(mapping),
            "--vendor-root",
            str(vendor),
        ]
    )
    result = mimic.run_pipeline(args)
    manifest = json.loads((work / "source_manifest.json").read_text(encoding="utf-8"))

    assert result["status"] == "audit_complete"
    assert set(manifest["auxiliary_sources"]) == {
        "DIAGNOSES_ICD.csv",
        "UNIQUE_ID_dict.csv",
    }
    diagnosis_audit = manifest["auxiliary_sources"]["DIAGNOSES_ICD.csv"]
    assert diagnosis_audit["physical_data_rows"] == 1
    assert diagnosis_audit["sha256"] == hashlib.sha256(diagnoses.read_bytes()).hexdigest()
    assert manifest["reference_dependencies"]["blockers"] == []
    assert manifest["mode_dependencies"] == {
        "required": ["DIAGNOSES_ICD.csv"],
        "blockers": [],
    }
    assert manifest["mode_semantics"]["diagnoses_used"] is True
    assert manifest["mode_semantics"]["cohort_contract"] == (
        "same_primary_diagnosis_filter"
    )
    assert manifest["mode_semantics"]["unique_id_strategy"] == "sorted_HADM_ID"
    assert manifest["time_binning"]["bin_k"] == 2
    assert manifest["time_binning"]["conflicting_upstream_values"] == {
        "gru_ode_bayes_DataMerging_cell_11": 60,
        "pinned_apn_loader_docstring": 10,
        "official_apn_pdf_appendix_statement": "absent",
    }

def test_candidate_validation_checks_schema_shape_and_hash(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    frame = pd.DataFrame(
        [[0, 0, 0, 1.0, 1.5, 0.5, -1.0], [0, 0, 1, 2.0, 1.5, 0.5, 1.0]],
        columns=mimic.OUTPUT_COLUMNS,
    )
    frame.to_csv(candidate, index=True)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert mimic.validate_nonreference_candidate(
        candidate, expected_shape=(2, 7)
    )["shape"] == [2, 7]
    with pytest.raises(mimic.PreprocessError, match="cohort or bin_k drifted"):
        mimic.validate_nonreference_candidate(candidate, expected_shape=(3, 7))

    metadata = mimic.validate_candidate(
        candidate,
        expected_shape=(2, 7),
        expected_sha256=digest,
    )

    assert metadata["shape"] == [2, 7]
    assert metadata["sha256"] == digest
    with pytest.raises(mimic.PreprocessError, match="shape mismatch"):
        mimic.validate_candidate(candidate, expected_shape=(3, 7), expected_sha256=digest)


def test_project_boundary_rejects_external_paths() -> None:
    external = Path("C:" + "/Users" + "/someone/Downloads/outside.csv")
    with pytest.raises(mimic.PreprocessError, match="escapes project root"):
        mimic.assert_inside_project(external)
