#!/usr/bin/env python3
"""Reproduce the pinned GRU-ODE-Bayes MIMIC-III ``complete_tensor.csv``.

This is a non-interactive translation of the effective cells in the following
notebooks from GRU-ODE-Bayes commit
``ddd0b34e884dbee1c09b6a3927d1e9ab10443af8``:

* ``Admissions.ipynb``
* ``Outputs.ipynb``
* ``LabEvents.ipynb``
* ``Prescriptions.ipynb``
* ``DataMerging.ipynb`` (through its ``END OF FILE`` marker)

The script deliberately fails closed.  In particular, the merging notebook
loads ``UNIQUE_ID_dict.csv`` and ``DIAGNOSES_ICD.csv`` even though its README
only describes the eight primary source tables.  The former contains an
unseeded, previously frozen permutation.  Neither dependency can be inferred
from the eight tables while preserving APN/tsdm's canonical SHA256, so both are
required in reference mode.

No source data is downloaded and no environment or test split is opened here.
All paths, including inputs, work files, manifests, and outputs, must resolve
inside this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNED_GRU_ODE_BAYES_COMMIT = "ddd0b34e884dbee1c09b6a3927d1e9ab10443af8"
REFERENCE_SHA256 = "8e884a916d28fd546b898b54e20055d4ad18d9a7abe262e15137080e9feb4fc2"
REFERENCE_SHAPE = (3_082_224, 7)
APN_REFERENCE_BIN_K = 2
REFERENCE_INPUT_PROCESSED_ROWS = 4_540_572
REFERENCE_ADMISSION_STAGE_COUNTS: Mapping[str, int] = {
    "one_visit_rows": 38_983,
    "stay_3_to_29_days_rows": 28_840,
    "historic_age_over_15_rows": 23_495,
    "chart_events_rows": 23_465,
    "final_rows": 23_465,
}
NOTEBOOK_NAMES = (
    "Admissions.ipynb",
    "Outputs.ipynb",
    "LabEvents.ipynb",
    "Prescriptions.ipynb",
    "DataMerging.ipynb",
)
PRIMARY_SOURCES: Mapping[str, tuple[str, ...]] = {
    "ADMISSIONS.csv": (
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "DEATHTIME",
        "ADMISSION_TYPE",
        "HAS_CHARTEVENTS_DATA",
    ),
    "PATIENTS.csv": ("SUBJECT_ID", "DOB"),
    "D_ITEMS.csv": ("ITEMID", "LABEL"),
    "D_LABITEMS.csv": ("ITEMID", "LABEL"),
    "INPUTEVENTS_MV.csv": (
        "SUBJECT_ID",
        "HADM_ID",
        "STARTTIME",
        "ENDTIME",
        "ITEMID",
        "AMOUNT",
        "AMOUNTUOM",
        "RATE",
        "RATEUOM",
        "PATIENTWEIGHT",
        "ORDERCATEGORYDESCRIPTION",
    ),
    "OUTPUTEVENTS.csv": (
        "SUBJECT_ID",
        "HADM_ID",
        "CHARTTIME",
        "ITEMID",
        "VALUE",
        "VALUEUOM",
        "ISERROR",
    ),
    "LABEVENTS.csv": (
        "SUBJECT_ID",
        "HADM_ID",
        "CHARTTIME",
        "ITEMID",
        "VALUE",
        "VALUENUM",
        "VALUEUOM",
    ),
    "PRESCRIPTIONS.csv": (
        "SUBJECT_ID",
        "HADM_ID",
        "STARTDATE",
        "DRUG",
        "DOSE_VAL_RX",
        "DOSE_UNIT_RX",
    ),
}

INPUT_LABELS = (
    "Albumin 5%",
    "Dextrose 5%",
    "Lorazepam (Ativan)",
    "Calcium Gluconate",
    "Midazolam (Versed)",
    "Phenylephrine",
    "Furosemide (Lasix)",
    "Hydralazine",
    "Norepinephrine",
    "Magnesium Sulfate",
    "Nitroglycerin",
    "Insulin - Glargine",
    "Insulin - Humalog",
    "Insulin - Regular",
    "Heparin Sodium",
    "Morphine Sulfate",
    "Potassium Chloride",
    "Packed Red Blood Cells",
    "Gastric Meds",
    "D5 1/2NS",
    "LR",
    "K Phos",
    "Solution",
    "Sterile Water",
    "Metoprolol",
    "Piggyback",
    "OR Crystalloid Intake",
    "OR Cell Saver Intake",
    "PO Intake",
    "GT Flush",
    "KCL (Bolus)",
    "Magnesium Sulfate (Bolus)",
)

OUTPUT_LABELS = (
    "Gastric Gastric Tube",
    "Stool Out Stool",
    "Urine Out Incontinent",
    "Ultrafiltrate Ultrafiltrate",
    "Foley",
    "Void",
    "Condom Cath",
    "Fecal Bag",
    "Ostomy (output)",
    "Chest Tube #1",
    "Chest Tube #2",
    "Jackson Pratt #1",
    "OR EBL",
    "Pre-Admission",
    "TF Residual",
)

LAB_LABELS = (
    "Albumin",
    "Alanine Aminotransferase (ALT)",
    "Alkaline Phosphatase",
    "Anion Gap",
    "Asparate Aminotransferase (AST)",
    "Base Excess",
    "Basophils",
    "Bicarbonate",
    "Bilirubin, Total",
    "Calcium, Total",
    "Calculated Total CO2",
    "Chloride",
    "Creatinine",
    "Eosinophils",
    "Glucose",
    "Hematocrit",
    "Hemoglobin",
    "Lactate",
    "Lymphocytes",
    "MCH",
    "MCHC",
    "MCV",
    "Magnesium",
    "Monocytes",
    "Neutrophils",
    "PT",
    "PTT",
    "Phosphate",
    "Platelet Count",
    "Potassium",
    "RDW",
    "Red Blood Cells",
    "Sodium",
    "Specific Gravity",
    "Urea Nitrogen",
    "White Blood Cells",
    "pCO2",
    "pH",
    "pO2",
)

PRESCRIPTION_LABELS = (
    "Aspirin",
    "Bisacodyl",
    "Docusate Sodium",
    "D5W",
    "Humulin-R Insulin",
    "Potassium Chloride",
    "Magnesium Sulfate",
    "Metoprolol Tartrate",
    "Sodium Chloride 0.9%  Flush",
    "Pantoprazole",
)

OUTPUT_COLUMNS = (
    "UNIQUE_ID",
    "LABEL_CODE",
    "TIME_STAMP",
    "VALUENUM",
    "MEAN",
    "STD",
    "VALUENORM",
)


class PreprocessError(RuntimeError):
    """Raised when the canonical preprocessing contract cannot be proven."""


@dataclass(frozen=True)
class CsvAudit:
    relative_path: str
    bytes: int
    sha256: str
    physical_data_rows: int
    columns: tuple[str, ...]


def assert_inside_project(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve *path* and reject anything outside ``PROJECT_ROOT``."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=must_exist)
    root = PROJECT_ROOT.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreprocessError(f"path escapes project root: {resolved}") from exc
    return resolved


def relative_project_path(path: str | Path) -> str:
    return assert_inside_project(path).relative_to(PROJECT_ROOT).as_posix()


def sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    checked = assert_inside_project(path, must_exist=True)
    digest = hashlib.sha256()
    with checked.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_data_rows(path: Path, *, block_size: int = 8 * 1024 * 1024) -> int:
    """Count physical CSV rows without materializing a multi-gigabyte table.

    MIMIC-III 1.4 tables used here contain one record per physical line.  The
    parsed row counts collected during transformation are checked against this
    value, so a source containing embedded newlines fails closed.
    """

    line_breaks = 0
    last = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            line_breaks += block.count(b"\n")
            last = block[-1:]
    physical_lines = line_breaks + (1 if path.stat().st_size and last != b"\n" else 0)
    if physical_lines < 1:
        raise PreprocessError(f"empty CSV: {relative_project_path(path)}")
    return physical_lines - 1


def _read_header(path: Path) -> tuple[str, ...]:
    try:
        return tuple(pd.read_csv(path, nrows=0).columns.astype(str))
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise PreprocessError(f"cannot parse CSV header: {relative_project_path(path)}") from exc


def audit_csv(path: str | Path, required_columns: Sequence[str]) -> CsvAudit:
    checked = assert_inside_project(path, must_exist=True)
    columns = _read_header(checked)
    missing = sorted(set(required_columns) - set(columns))
    if missing:
        raise PreprocessError(
            f"{relative_project_path(checked)} missing required columns: {missing}"
        )
    return CsvAudit(
        relative_path=relative_project_path(checked),
        bytes=checked.stat().st_size,
        sha256=sha256_file(checked),
        physical_data_rows=_physical_data_rows(checked),
        columns=columns,
    )


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = assert_inside_project(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def verify_pinned_notebooks(vendor_root: str | Path) -> dict[str, Any]:
    root = assert_inside_project(vendor_root, must_exist=True)
    git_command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-C",
        str(root),
        "rev-parse",
        "HEAD",
    ]
    try:
        commit = subprocess.run(
            git_command,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreprocessError("cannot verify pinned GRU-ODE-Bayes checkout") from exc
    if commit != PINNED_GRU_ODE_BAYES_COMMIT:
        raise PreprocessError(
            f"GRU-ODE-Bayes commit mismatch: expected {PINNED_GRU_ODE_BAYES_COMMIT}, got {commit}"
        )

    notebook_root = root / "data_preproc" / "MIMIC"
    notebooks: dict[str, Any] = {}
    for name in NOTEBOOK_NAMES:
        notebook = assert_inside_project(notebook_root / name, must_exist=True)
        document = json.loads(notebook.read_text(encoding="utf-8"))
        code_cells = [cell for cell in document.get("cells", []) if cell.get("cell_type") == "code"]
        if name == "DataMerging.ipynb":
            marker_positions = [
                index
                for index, cell in enumerate(document.get("cells", []))
                if "END OF FILE" in "".join(cell.get("source", [])).upper()
            ]
            if marker_positions != [60]:
                raise PreprocessError(
                    f"unexpected DataMerging END OF FILE marker(s): {marker_positions}"
                )
            effective = [
                cell
                for index, cell in enumerate(document.get("cells", []))
                if cell.get("cell_type") == "code" and index <= marker_positions[0]
            ]
        else:
            effective = code_cells
        notebooks[name] = {
            "relative_path": relative_project_path(notebook),
            "sha256": sha256_file(notebook),
            "effective_code_cells": len(effective),
        }
    return {"commit": commit, "notebooks": notebooks}


def require_columns(frame: pd.DataFrame, required: Iterable[str], *, table: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise PreprocessError(f"{table} missing columns: {missing}")


def parse_mimic_datetime(
    series: pd.Series,
    *,
    name: str,
    audit: dict[str, Any] | None = None,
) -> pd.Series:
    """Parse the two timestamp representations emitted by MIMIC-III.

    Most source tables use second-resolution timestamps. ``PRESCRIPTIONS.csv``
    uses date-only ``STARTDATE`` values, which older pandas accepted even when
    the upstream notebook supplied a second-resolution ``format``. Modern
    pandas is strict, so both representations are parsed explicitly while all
    other non-null strings still fail closed.
    """

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(series, errors="coerce")
        invalid = series.notna() & parsed.isna()
        if audit is not None:
            audit.clear()
            audit.update(
                {
                    "total_rows": int(len(series)),
                    "null_rows": int(series.isna().sum()),
                    "second_resolution_rows": int(series.notna().sum()),
                    "date_only_rows_promoted_to_midnight": 0,
                    "date_only_examples": [],
                    "invalid_rows": int(invalid.sum()),
                    "invalid_examples": [],
                }
            )
    else:
        text = series.astype("string")
        second_resolution = text.str.fullmatch(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", na=False
        )
        date_only = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        parsed_seconds = pd.to_datetime(
            text.where(second_resolution),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        parsed_dates = pd.to_datetime(
            text.where(date_only),
            format="%Y-%m-%d",
            errors="coerce",
        )
        parsed = parsed_seconds.fillna(parsed_dates)
        invalid = text.notna() & parsed.isna()
        if audit is not None:
            audit.clear()
            audit.update(
                {
                    "total_rows": int(len(text)),
                    "null_rows": int(text.isna().sum()),
                    "second_resolution_rows": int(
                        (second_resolution & parsed.notna()).sum()
                    ),
                    "date_only_rows_promoted_to_midnight": int(
                        (date_only & parsed.notna()).sum()
                    ),
                    "date_only_examples": text.loc[date_only]
                    .drop_duplicates()
                    .head(5)
                    .tolist(),
                    "invalid_rows": int(invalid.sum()),
                    "invalid_examples": text.loc[invalid]
                    .drop_duplicates()
                    .head(5)
                    .tolist(),
                }
            )
    if invalid.any():
        examples = series.loc[invalid].astype("string").drop_duplicates().head(5).tolist()
        raise PreprocessError(
            f"invalid {name} timestamps: {int(invalid.sum())}; examples={examples}"
        )
    return parsed


def elapsed_days_without_ns_overflow(
    later: pd.Series, earlier: pd.Series, *, name: str
) -> pd.Series:
    """Return floor elapsed days at MIMIC's native one-second resolution.

    MIMIC-III shifts dates for deidentification and encodes some elderly
    patients with DOBs centuries before admission. Subtracting two otherwise
    valid ``datetime64[ns]`` values can therefore overflow pandas' roughly
    292-year timedelta range. Downcasting the operands to seconds before the
    subtraction preserves every bit of source timestamp precision and the
    original notebook's ``.dt.days`` floor semantics.
    """

    if len(later) != len(earlier):
        raise ValueError(f"{name} timestamp arrays must have equal length")
    later_seconds = later.to_numpy(dtype="datetime64[s]")
    earlier_seconds = earlier.to_numpy(dtype="datetime64[s]")
    valid = ~(np.isnat(later_seconds) | np.isnat(earlier_seconds))
    days = np.full(len(later_seconds), np.nan, dtype=np.float64)
    elapsed_seconds = later_seconds[valid] - earlier_seconds[valid]
    days[valid] = elapsed_seconds // np.timedelta64(1, "D")
    return pd.Series(days, index=later.index, name=name)


def elapsed_days_with_historic_pandas_ns_wrap(
    later: pd.Series, earlier: pd.Series, *, name: str
) -> pd.Series:
    """Reproduce the pinned notebook's legacy int64-nanosecond subtraction.

    The notebook was executed with pandas 0.23.4. Its subtraction wrapped
    MIMIC's de-identification spans of roughly 300 years in signed int64
    nanoseconds. This is implemented explicitly, not delegated to pandas.
    """

    if len(later) != len(earlier):
        raise ValueError(f"{name} timestamp arrays must have equal length")
    later_ns = later.to_numpy(dtype="datetime64[ns]")
    earlier_ns = earlier.to_numpy(dtype="datetime64[ns]")
    valid = ~(np.isnat(later_ns) | np.isnat(earlier_ns))
    days = np.full(len(later_ns), np.nan, dtype=np.float64)
    with np.errstate(over="ignore"):
        wrapped = (
            later_ns[valid].view(np.uint64) - earlier_ns[valid].view(np.uint64)
        ).view(np.int64)
    days[valid] = wrapped // (86_400 * 1_000_000_000)
    return pd.Series(days, index=later.index, name=name)


def prepare_admissions(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    *,
    audit: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Translate the effective filtering cells of ``Admissions.ipynb``."""

    require_columns(admissions, PRIMARY_SOURCES["ADMISSIONS.csv"], table="ADMISSIONS")
    require_columns(patients, PRIMARY_SOURCES["PATIENTS.csv"], table="PATIENTS")
    if admissions["HADM_ID"].isna().any() or patients["SUBJECT_ID"].isna().any():
        raise PreprocessError("null admission or patient identifiers")

    dob = patients[["SUBJECT_ID", "DOB"]].copy()
    dob["DOBTIME"] = parse_mimic_datetime(dob["DOB"], name="DOB")
    counts = admissions.groupby("SUBJECT_ID", sort=True)["HADM_ID"].nunique()
    one_visit = set(counts[counts == 1].index)
    merged = pd.merge(
        dob[["SUBJECT_ID", "DOBTIME"]],
        admissions.loc[admissions["SUBJECT_ID"].isin(one_visit)],
        on="SUBJECT_ID",
    )
    merged["ADMITTIME"] = parse_mimic_datetime(merged["ADMITTIME"], name="ADMITTIME")
    merged["DISCHTIME"] = parse_mimic_datetime(merged["DISCHTIME"], name="DISCHTIME")
    merged["ELAPSED_TIME"] = merged["DISCHTIME"] - merged["ADMITTIME"]
    merged["ELAPSED_DAYS"] = merged["ELAPSED_TIME"].dt.days
    merged["DEATHTAG"] = merged["DEATHTIME"].notna().astype(np.int8)
    historic_age_days = elapsed_days_with_historic_pandas_ns_wrap(
        merged["ADMITTIME"], merged["DOBTIME"], name="HISTORIC_AGE_DAYS"
    )
    true_age_days = elapsed_days_without_ns_overflow(
        merged["ADMITTIME"], merged["DOBTIME"], name="TRUE_AGE_DAYS"
    )
    stay = merged["ELAPSED_DAYS"].between(3, 29, inclusive="both")
    historic_age = historic_age_days / 365.0 > 15
    true_age = true_age_days / 365.0 > 15
    has_chart = merged["HAS_CHARTEVENTS_DATA"] == 1
    not_newborn = merged["ADMISSION_TYPE"] != "NEWBORN"
    keep = stay & historic_age & has_chart & not_newborn
    true_age_keep = stay & true_age & has_chart & not_newborn
    if audit is not None:
        audit.clear()
        audit.update(
            {
                "semantics": "historic_pandas_0.23_int64_ns_wrap",
                "one_visit_rows": int(len(merged)),
                "stay_3_to_29_days_rows": int(stay.sum()),
                "historic_age_over_15_rows": int((stay & historic_age).sum()),
                "chart_events_rows": int((stay & historic_age & has_chart).sum()),
                "final_rows": int(keep.sum()),
                "true_age_sensitivity": {
                    "age_over_15_rows": int((stay & true_age).sum()),
                    "final_rows": int(true_age_keep.sum()),
                    "historic_excluded_but_true_age_included_before_chart": int(
                        (stay & true_age & ~historic_age).sum()
                    ),
                    "historic_excluded_but_true_age_included_final": int(
                        (true_age_keep & ~keep).sum()
                    ),
                    "semantics": "seconds-resolution subtraction; sensitivity only",
                },
            }
        )
    result = merged.loc[keep].copy()
    if result["HADM_ID"].duplicated().any():
        raise PreprocessError("admission filter produced duplicate HADM_ID values")
    return result

def validate_reference_admission_stage_counts(audit: Mapping[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in REFERENCE_ADMISSION_STAGE_COUNTS.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise PreprocessError(
            "admission stage counts differ from pinned notebook outputs: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _drop_above(frame: pd.DataFrame, label: str, column: str, limit: float) -> pd.DataFrame:
    return frame.loc[~((frame["LABEL"] == label) & (frame[column] > limit))].copy()


def clean_input_events(frame: pd.DataFrame, item_labels: pd.DataFrame) -> pd.DataFrame:
    """Apply pre-expansion INPUTEVENTS_MV cleaning from Admissions cells 18--38."""

    require_columns(frame, PRIMARY_SOURCES["INPUTEVENTS_MV.csv"], table="INPUTEVENTS_MV")
    require_columns(item_labels, ("ITEMID", "LABEL"), table="D_ITEMS")
    data = pd.merge(frame, item_labels[["ITEMID", "LABEL"]], on="ITEMID")
    data = data.loc[data["LABEL"].isin(INPUT_LABELS)].copy()

    unit_rules = {
        "Magnesium Sulfate": ("AMOUNTUOM", "grams"),
        "Metoprolol": ("AMOUNTUOM", "mg"),
        "Dextrose 5%": ("RATEUOM", "mL/hour"),
        "Magnesium Sulfate (Bolus)": ("RATEUOM", "mL/hour"),
        "Piggyback": ("RATEUOM", "mL/hour"),
        "Packed Red Blood Cells": ("RATEUOM", "mL/hour"),
    }
    for label, (column, unit) in unit_rules.items():
        data = data.loc[~((data["LABEL"] == label) & (data[column] != unit))].copy()

    data["STARTTIME"] = parse_mimic_datetime(data["STARTTIME"], name="input STARTTIME")
    data["ENDTIME"] = parse_mimic_datetime(data["ENDTIME"], name="input ENDTIME")
    negative = data["AMOUNT"] < 0
    old_start = data.loc[negative, "STARTTIME"].copy()
    data.loc[negative, "STARTTIME"] = data.loc[negative, "ENDTIME"].to_numpy()
    data.loc[negative, "ENDTIME"] = old_start.to_numpy()
    data.loc[negative, "AMOUNT"] = -data.loc[negative, "AMOUNT"]
    data["DURATION"] = data["ENDTIME"] - data["STARTTIME"]
    if data["DURATION"].isna().any():
        raise PreprocessError("input event contains null duration after correction")

    hard_limits = {
        "Calcium Gluconate": 10.0,
        "Gastric Meds": 5000.0,
        "Heparin Sodium": 50000.0,
        "Hydralazine": 200.0,
        "Insulin - Humalog": 100.0,
        "Insulin - Regular": 1000.0,
        "Magnesium Sulfate": 51.0,
    }
    for label, limit in hard_limits.items():
        data = _drop_above(data, label, "AMOUNT", limit)
    return data

def summarize_input_duration_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Count duration edge cases retained by the upstream short-event partition."""

    require_columns(
        frame,
        ("DURATION", "AMOUNT", "RATE", "LABEL"),
        table="cleaned INPUTEVENTS_MV",
    )
    negative = frame["DURATION"] < timedelta(0)
    zero = frame["DURATION"] == timedelta(0)
    labels = frame.loc[negative, "LABEL"].astype(str).value_counts(sort=False)
    return {
        "negative_duration_rows_retained": int(negative.sum()),
        "negative_duration_zero_amount_rows": int(
            (negative & frame["AMOUNT"].eq(0)).sum()
        ),
        "negative_duration_missing_rate_rows": int(
            (negative & frame["RATE"].isna()).sum()
        ),
        "negative_duration_rows_by_label": {
            label: int(count) for label, count in sorted(labels.items())
        },
        "zero_duration_rows_retained": int(zero.sum()),
    }




def filter_group_outliers(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    standard_deviations: float,
) -> pd.DataFrame:
    """Drop only the high-side outliers used by the notebooks."""

    descriptions = frame.groupby(label_column, sort=True)[value_column].agg(["mean", "std"])
    limits = descriptions["mean"] + standard_deviations * descriptions["std"]
    mapped = frame[label_column].map(limits)
    return frame.loc[~(frame[value_column] > mapped)].copy()


def expand_input_events(frame: pd.DataFrame, *, interval_hours: float = 0.5) -> pd.DataFrame:
    """Discretize extended administrations exactly as notebook cells 38--40."""

    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    threshold = timedelta(hours=interval_hours)
    duration = frame["DURATION"]
    rate_missing = frame["RATE"].isna()
    partitions = (
        frame.loc[(duration > threshold) & rate_missing].copy(),
        frame.loc[(duration <= threshold) & rate_missing].copy(),
        frame.loc[(duration > threshold) & ~rate_missing].copy(),
        frame.loc[(duration <= threshold) & ~rate_missing].copy(),
    )
    if sum(len(part) for part in partitions) != len(frame):
        raise PreprocessError("input duration/rate partition is not exhaustive")

    expanded: list[pd.DataFrame] = []
    seconds = 3600.0 * interval_hours
    for index, part in enumerate(partitions):
        if index in (0, 2):
            repeats = np.ceil(part["DURATION"].dt.total_seconds() / seconds).astype(int)
            if (repeats < 1).any():
                raise PreprocessError("invalid input expansion repeat count")
            part["Repeat"] = repeats
            repeated = part.loc[part.index.repeat(repeats)].copy()
            offsets = repeated.groupby(level=0, sort=False).cumcount()
            repeated["CHARTTIME"] = repeated["STARTTIME"] + pd.to_timedelta(
                offsets * interval_hours, unit="h"
            )
            repeated["AMOUNT"] = repeated["AMOUNT"] / repeated["Repeat"]
            expanded.append(repeated)
        else:
            part["CHARTTIME"] = part["STARTTIME"]
            expanded.append(part)
    return pd.concat(expanded, axis=0)


def validate_input_rate_amount(frame: pd.DataFrame) -> None:
    """Run the three rate/amount consistency assertions from notebook cell 37."""

    require_columns(
        frame,
        ("RATE", "RATEUOM", "AMOUNT", "DURATION", "PATIENTWEIGHT"),
        table="cleaned INPUTEVENTS_MV",
    )
    rate_units = frame["RATEUOM"].astype("string")
    duration_seconds = frame["DURATION"].dt.total_seconds()
    has_rate = frame["RATE"].notna()
    checks = (
        (
            "per-hour",
            has_rate & rate_units.str.contains("hour", na=False),
            frame["RATE"] * duration_seconds / 3600.0,
            frame["AMOUNT"],
        ),
        (
            "mL/min",
            has_rate & rate_units.str.contains("mL/min", na=False),
            frame["RATE"] * duration_seconds / 60.0,
            frame["AMOUNT"],
        ),
        (
            "kg/min",
            has_rate & rate_units.str.contains("kg/min", na=False),
            frame["RATE"] * duration_seconds / 60.0 * frame["PATIENTWEIGHT"],
            1000.0 * frame["AMOUNT"],
        ),
    )
    for name, mask, computed, recorded in checks:
        mismatch = mask & ((computed - recorded).abs() > 0.01)
        if mismatch.any():
            raise PreprocessError(
                f"{name} input rate/amount mismatch in {int(mismatch.sum())} rows"
            )


def clean_output_events(frame: pd.DataFrame, item_labels: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, PRIMARY_SOURCES["OUTPUTEVENTS.csv"], table="OUTPUTEVENTS")
    if frame["ISERROR"].notna().any():
        raise PreprocessError("OUTPUTEVENTS contains non-null ISERROR entries")
    data = pd.merge(frame, item_labels[["ITEMID", "LABEL"]], on="ITEMID")
    data = data.loc[data["LABEL"].isin(OUTPUT_LABELS)].copy()
    return data


def finalize_output_events(frame: pd.DataFrame) -> pd.DataFrame:
    data = filter_group_outliers(
        frame,
        label_column="LABEL",
        value_column="VALUE",
        standard_deviations=4.0,
    )
    upper = {
        "Foley": 5500.0,
        "OR EBL": 5000.0,
        "OR Out EBL": 5000.0,
        "OR Urine": 5000.0,
        "Pre-Admission": 5000.0,
        "Pre-Admission Output Pre-Admission Output": 5000.0,
        "Urine Out Foley": 5000.0,
    }
    for label, limit in upper.items():
        data = _drop_above(data, label, "VALUE", limit)
    data = data.loc[
        ~(
            data["LABEL"].isin(("Pre-Admission", "Void"))
            & (data["VALUE"] < 0)
        )
    ].copy()
    return data.dropna(subset=["VALUE"]).copy()


def clean_lab_events(frame: pd.DataFrame, lab_labels: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, PRIMARY_SOURCES["LABEVENTS.csv"], table="LABEVENTS")
    data = pd.merge(frame, lab_labels[["ITEMID", "LABEL"]], on="ITEMID")
    data = data.loc[data["LABEL"].isin(LAB_LABELS)].copy()
    unit_fixes = {
        "Calculated Total CO2": "mEq/L",
        "PT": "sec",
        "pCO2": "mm Hg",
        "pH": "units",
        "pO2": "mm Hg",
    }
    for label, unit in unit_fixes.items():
        data.loc[data["LABEL"] == label, "VALUEUOM"] = unit
    glucose_negative = (
        (data["LABEL"] == "Glucose")
        & data["VALUENUM"].isna()
        & (data["VALUE"] == "NEG")
    )
    data.loc[glucose_negative, "VALUENUM"] = -1.0
    data = data.dropna(subset=["VALUENUM"]).copy()
    data = data.loc[~((data["LABEL"] == "Anion Gap") & (data["VALUENUM"] < 0))].copy()
    data = data.loc[
        ~((data["LABEL"] == "Base Excess") & ~data["VALUENUM"].between(-50, 50))
    ].copy()
    data = data.loc[~((data["LABEL"] == "Hemoglobin") & (data["VALUENUM"] > 25))].copy()
    bad_glucose_hadm = data["HADM_ID"].isin((103500.0, 117066.0))
    data = data.loc[
        ~((data["LABEL"] == "Glucose") & (data["VALUENUM"] > 2000) & bad_glucose_hadm)
    ].copy()
    data = data.loc[~((data["LABEL"] == "Potassium") & (data["VALUENUM"] > 30))].copy()
    return data


def clean_prescriptions(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, PRIMARY_SOURCES["PRESCRIPTIONS.csv"], table="PRESCRIPTIONS")
    data = frame.loc[frame["DRUG"].isin(PRESCRIPTION_LABELS)].copy()
    data = data.dropna(subset=["DOSE_UNIT_RX"]).copy()
    data.loc[(data["DRUG"] == "D5W") & (data["DOSE_UNIT_RX"] == "ml"), "DOSE_UNIT_RX"] = "mL"
    flush = data["DRUG"] == "Sodium Chloride 0.9%  Flush"
    data.loc[flush & (data["DOSE_UNIT_RX"] == "ml"), "DOSE_UNIT_RX"] = "mL"
    allowed_units = {
        "D5W": "mL",
        "Magnesium Sulfate": "gm",
        "Potassium Chloride": "mEq",
        "Bisacodyl": "mg",
        "Humulin-R Insulin": "UNIT",
        "Pantoprazole": "mg",
    }
    for drug, unit in allowed_units.items():
        data = data.loc[~((data["DRUG"] == drug) & (data["DOSE_UNIT_RX"] != unit))].copy()

    raw_dose = data["DOSE_VAL_RX"].astype("string")
    ranges = raw_dose.str.contains("-", na=False)
    range_parts = raw_dose.loc[ranges].str.split("-", n=1, expand=True)
    first = pd.to_numeric(range_parts[0], errors="coerce")
    second_text = range_parts[1].replace("", pd.NA)
    second = pd.to_numeric(second_text, errors="coerce").fillna(first)
    numeric = pd.to_numeric(raw_dose, errors="coerce")
    numeric.loc[ranges] = (first + second) / 2.0
    data["DOSE_VAL_RX"] = numeric
    return data.dropna(subset=["DOSE_VAL_RX"]).copy()


def finalize_prescriptions(frame: pd.DataFrame) -> pd.DataFrame:
    data = filter_group_outliers(
        frame,
        label_column="DRUG",
        value_column="DOSE_VAL_RX",
        standard_deviations=4.0,
    )
    data["CHARTTIME"] = parse_mimic_datetime(data["STARTDATE"], name="prescription STARTDATE")
    data["DRUG"] = data["DRUG"] + " Drug"
    return data


def load_unique_id_map(
    path: str | Path, hadm_ids: Iterable[int] | None = None
) -> dict[int, int]:
    checked = assert_inside_project(path, must_exist=True)
    mapping_frame = pd.read_csv(checked)
    require_columns(mapping_frame, ("HADM_ID", "UNIQUE_ID"), table="UNIQUE_ID_dict")
    if mapping_frame[["HADM_ID", "UNIQUE_ID"]].isna().any().any():
        raise PreprocessError("UNIQUE_ID_dict contains null identifiers")
    mapping_frame = mapping_frame[["HADM_ID", "UNIQUE_ID"]].astype(np.int64)
    if mapping_frame["HADM_ID"].duplicated().any() or mapping_frame["UNIQUE_ID"].duplicated().any():
        raise PreprocessError("UNIQUE_ID_dict must be one-to-one")
    required = (
        set(mapping_frame["HADM_ID"].tolist())
        if hadm_ids is None
        else set(int(value) for value in hadm_ids)
    )
    available = set(mapping_frame["HADM_ID"].tolist())
    missing = sorted(required - available)
    if missing:
        raise PreprocessError(f"UNIQUE_ID_dict misses {len(missing)} retained admissions")
    selected = mapping_frame.loc[mapping_frame["HADM_ID"].isin(required)]
    values = sorted(selected["UNIQUE_ID"].tolist())
    if values != list(range(len(values))):
        raise PreprocessError("retained UNIQUE_ID values are not a contiguous 0..N-1 permutation")
    return dict(zip(selected["HADM_ID"], selected["UNIQUE_ID"], strict=True))


def apply_diagnosis_filter(frame: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Replicate the notebook's inner merge with primary ICD-9 diagnoses."""

    require_columns(diagnoses, ("HADM_ID", "SEQ_NUM", "ICD9_CODE"), table="DIAGNOSES_ICD")
    primary = diagnoses.loc[diagnoses["SEQ_NUM"] == 1, ["HADM_ID", "ICD9_CODE"]].copy()
    if primary["HADM_ID"].duplicated().any():
        raise PreprocessError("DIAGNOSES_ICD has multiple SEQ_NUM=1 rows for an admission")
    code_lengths = primary["ICD9_CODE"].astype(str).str[:3].str.len()
    if (code_lengths != 3).any():
        raise PreprocessError("primary ICD9 code cannot be reduced to exactly three characters")
    # pd.merge creates the RangeIndex that the notebook writes as CSV column 0.
    return pd.merge(frame, primary, on="HADM_ID")


def merge_complete_tensor(
    inputs: pd.DataFrame,
    labs: pd.DataFrame,
    outputs: pd.DataFrame,
    prescriptions: pd.DataFrame,
    *,
    unique_ids: Mapping[int, int] | None,
    diagnoses: pd.DataFrame,
    bin_k: int,
    reference_mode: bool = True,
    minimum_observations: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate DataMerging.ipynb with the pinned APN binning modification."""

    if bin_k != APN_REFERENCE_BIN_K:
        raise PreprocessError(f"APN MIMIC preprocessing requires bin_k={APN_REFERENCE_BIN_K}")

    if minimum_observations < 1:
        raise ValueError("minimum_observations must be positive")
    if diagnoses is None:
        raise PreprocessError("all preprocessing modes require DIAGNOSES_ICD.csv")
    sources = []
    specifications = (
        (inputs, "AMOUNT", "LABEL", "Inputs"),
        (labs, "VALUENUM", "LABEL", "Lab"),
        (outputs, "VALUE", "LABEL", "Outputs"),
        (prescriptions, "DOSE_VAL_RX", "DRUG", "Prescriptions"),
    )
    labels_by_origin: list[set[str]] = []
    for frame, value_column, label_column, origin in specifications:
        require_columns(
            frame,
            ("HADM_ID", "CHARTTIME", value_column, label_column),
            table=origin,
        )
        part = frame[["HADM_ID", "CHARTTIME", value_column, label_column]].copy()
        part = part.rename(columns={value_column: "VALUENUM", label_column: "LABEL"})
        part["Origin"] = origin
        sources.append(part)
        labels_by_origin.append(set(part["LABEL"].dropna().astype(str)))
    union_size = len(set().union(*labels_by_origin))
    sum_size = sum(len(labels) for labels in labels_by_origin)
    if union_size != sum_size:
        raise PreprocessError("labels collide across MIMIC source tables")

    merged = pd.concat(sources, axis=0)
    merged["CHARTTIME"] = parse_mimic_datetime(merged["CHARTTIME"], name="merged CHARTTIME")
    reference = merged.groupby("HADM_ID", sort=True)["CHARTTIME"].min().rename("REF_TIME")
    merged = pd.merge(reference, merged, left_index=True, right_on="HADM_ID")
    delta = merged["CHARTTIME"] - merged["REF_TIME"]
    if (delta < timedelta(0)).any():
        raise PreprocessError("negative merged time offset")
    merged = merged.loc[delta < timedelta(hours=48)].copy()
    delta = merged["CHARTTIME"] - merged["REF_TIME"]

    ordered_labels = list(pd.unique(merged["LABEL"]))
    label_map = dict(zip(ordered_labels, range(len(ordered_labels)), strict=True))
    merged["LABEL_CODE"] = merged["LABEL"].map(label_map).astype(np.int64)
    # The original notebook uses 60, but the pinned APN loader explicitly
    # requires its modified preprocessing with bin_k=2 for canonical identity.
    merged["TIME"] = (
        delta.dt.total_seconds() * bin_k / 3600.0
    ).round().astype(np.int64)
    merged["VALUENUM"] = pd.to_numeric(merged["VALUENUM"], errors="raise")

    aggregated: list[pd.DataFrame] = []
    keys = ["HADM_ID", "TIME", "LABEL_CODE"]
    for origin, operation in (
        ("Lab", "mean"),
        ("Inputs", "sum"),
        ("Outputs", "sum"),
        ("Prescriptions", "sum"),
    ):
        subset = merged.loc[merged["Origin"] == origin, keys + ["VALUENUM"]]
        values = subset.groupby(keys, sort=False, as_index=False)["VALUENUM"].agg(operation)
        aggregated.append(values)
    complete = pd.concat(aggregated, axis=0)
    if complete.duplicated(keys).any():
        raise PreprocessError("duplicate HADM_ID/LABEL_CODE/TIME rows after aggregation")
    counts = complete.groupby("HADM_ID", sort=True)["TIME"].count()
    retained = set(counts[counts >= minimum_observations].index)
    complete = complete.loc[complete["HADM_ID"].isin(retained)].copy()

    complete = apply_diagnosis_filter(complete, diagnoses)

    if unique_ids is None:
        if reference_mode:
            raise PreprocessError("reference mode requires the frozen UNIQUE_ID mapping")
        ordered_hadm = sorted(complete["HADM_ID"].astype(int).unique())
        unique_ids = dict(zip(ordered_hadm, range(len(ordered_hadm)), strict=True))
    missing_mapping = set(complete["HADM_ID"].astype(int)) - set(unique_ids)
    if missing_mapping:
        raise PreprocessError(f"UNIQUE_ID mapping misses {len(missing_mapping)} admissions")
    complete["UNIQUE_ID"] = complete["HADM_ID"].astype(int).map(unique_ids)
    if complete["UNIQUE_ID"].isna().any():
        raise PreprocessError("null UNIQUE_ID after mapping")

    tensor = complete[["UNIQUE_ID", "LABEL_CODE", "TIME", "VALUENUM"]].copy()
    tensor = tensor.rename(columns={"TIME": "TIME_STAMP"})
    means = tensor.groupby("LABEL_CODE", sort=True)["VALUENUM"].mean()
    standard_deviations = tensor.groupby("LABEL_CODE", sort=True)["VALUENUM"].std()
    tensor["MEAN"] = tensor["LABEL_CODE"].map(means)
    tensor["STD"] = tensor["LABEL_CODE"].map(standard_deviations)
    if tensor["STD"].isna().any() or (tensor["STD"] == 0).any():
        raise PreprocessError("undefined or zero feature standard deviation")
    tensor["VALUENORM"] = (tensor["VALUENUM"] - tensor["MEAN"]) / tensor["STD"]
    if not np.isfinite(tensor[list(OUTPUT_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise PreprocessError("non-finite complete tensor values")
    tensor["UNIQUE_ID"] = tensor["UNIQUE_ID"].astype(np.int64)
    tensor["LABEL_CODE"] = tensor["LABEL_CODE"].astype(np.int64)
    tensor["TIME_STAMP"] = tensor["TIME_STAMP"].astype(np.int64)
    label_dictionary = pd.DataFrame(
        {"LABEL": ordered_labels, "LABEL_CODE": list(range(len(ordered_labels)))}
    )
    return tensor[list(OUTPUT_COLUMNS)], label_dictionary


def inspect_candidate(path: str | Path) -> dict[str, Any]:
    candidate = assert_inside_project(path, must_exist=True)
    header = pd.read_csv(candidate, nrows=0, index_col=0)
    if tuple(header.columns) != OUTPUT_COLUMNS:
        raise PreprocessError(f"candidate columns mismatch: {tuple(header.columns)}")
    rows = _physical_data_rows(candidate)
    shape = (rows, len(header.columns))
    digest = sha256_file(candidate)
    return {"shape": list(shape), "sha256": digest, "bytes": candidate.stat().st_size}


def validate_candidate(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] = REFERENCE_SHAPE,
    expected_sha256: str = REFERENCE_SHA256,
) -> dict[str, Any]:
    metadata = inspect_candidate(path)
    shape, digest = tuple(metadata["shape"]), metadata["sha256"]
    if shape != expected_shape:
        raise PreprocessError(f"candidate shape mismatch: expected {expected_shape}, got {shape}")
    if digest != expected_sha256:
        raise PreprocessError(
            f"candidate SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return metadata

def validate_nonreference_candidate(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] = REFERENCE_SHAPE,
) -> dict[str, Any]:
    """Validate schema and cohort/binning shape without claiming reference hash parity."""

    metadata = inspect_candidate(path)
    shape = tuple(metadata["shape"])
    if shape != expected_shape:
        raise PreprocessError(
            "non-reference candidate shape mismatch: "
            f"expected {expected_shape}, got {shape}; cohort or bin_k drifted"
        )
    return metadata




def _iter_chunks(path: Path, columns: Sequence[str], chunksize: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, usecols=list(columns), chunksize=chunksize, low_memory=False)


def _write_frame(path: Path, frame: pd.DataFrame, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a" if append else "w", header=not append, index=False)


def _read_selected(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(columns), low_memory=False)


def _filter_hadm(frame: pd.DataFrame, hadm_ids: set[int]) -> pd.DataFrame:
    return frame.loc[frame["HADM_ID"].isin(hadm_ids)].copy()


def _stage_primary_tables(
    source_dir: Path,
    work_dir: Path,
    admissions: pd.DataFrame,
    *,
    chunksize: int,
) -> tuple[dict[str, Path], dict[str, int], dict[str, Any]]:
    """Chunk raw event tables, retaining only notebook-selected columns/rows."""

    hadm_ids = set(admissions["HADM_ID"].astype(int))
    item_labels = pd.read_csv(source_dir / "D_ITEMS.csv", usecols=["ITEMID", "LABEL"])
    lab_labels = pd.read_csv(source_dir / "D_LABITEMS.csv", usecols=["ITEMID", "LABEL"])
    parsed_rows: dict[str, int] = {name: 0 for name in PRIMARY_SOURCES}
    parsed_rows["ADMISSIONS.csv"] = len(pd.read_csv(source_dir / "ADMISSIONS.csv", usecols=["HADM_ID"]))
    parsed_rows["PATIENTS.csv"] = len(pd.read_csv(source_dir / "PATIENTS.csv", usecols=["SUBJECT_ID"]))
    parsed_rows["D_ITEMS.csv"] = len(item_labels)
    parsed_rows["D_LABITEMS.csv"] = len(lab_labels)
    input_duration_audit: dict[str, Any] = {
        "negative_duration_rows_retained": 0,
        "negative_duration_zero_amount_rows": 0,
        "negative_duration_missing_rate_rows": 0,
        "negative_duration_rows_by_label": {},
        "zero_duration_rows_retained": 0,
    }

    staged: dict[str, Path] = {}

    input_base = work_dir / "INPUTS_base.csv"
    append = False
    for chunk in _iter_chunks(source_dir / "INPUTEVENTS_MV.csv", PRIMARY_SOURCES["INPUTEVENTS_MV.csv"], chunksize):
        parsed_rows["INPUTEVENTS_MV.csv"] += len(chunk)
        selected = clean_input_events(_filter_hadm(chunk, hadm_ids), item_labels)
        if len(selected):
            chunk_audit = summarize_input_duration_audit(selected)
            for key in (
                "negative_duration_rows_retained",
                "negative_duration_zero_amount_rows",
                "negative_duration_missing_rate_rows",
                "zero_duration_rows_retained",
            ):
                input_duration_audit[key] += chunk_audit[key]
            for label, count in chunk_audit[
                "negative_duration_rows_by_label"
            ].items():
                current = input_duration_audit["negative_duration_rows_by_label"]
                current[label] = current.get(label, 0) + count
            _write_frame(input_base, selected, append=append)
            append = True
    if not append:
        raise PreprocessError("no retained INPUTEVENTS_MV rows")
    input_all = pd.read_csv(input_base, low_memory=False)
    input_all["STARTTIME"] = parse_mimic_datetime(input_all["STARTTIME"], name="staged input STARTTIME")
    input_all["ENDTIME"] = parse_mimic_datetime(input_all["ENDTIME"], name="staged input ENDTIME")
    input_all["DURATION"] = input_all["ENDTIME"] - input_all["STARTTIME"]
    input_all = _drop_above(input_all, "D5 1/2NS", "RATE", 1000.0)
    input_all = filter_group_outliers(
        input_all, label_column="LABEL", value_column="RATE", standard_deviations=4.0
    )
    validate_input_rate_amount(input_all)
    input_all = expand_input_events(input_all)
    input_all = filter_group_outliers(
        input_all, label_column="LABEL", value_column="AMOUNT", standard_deviations=5.0
    )
    inputs_path = work_dir / "INPUTS_processed.csv"
    input_all[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "AMOUNT", "LABEL"]].to_csv(
        inputs_path, index=False
    )
    staged["inputs"] = inputs_path
    del input_all
    input_duration_audit["negative_duration_rows_by_label"] = dict(
        sorted(input_duration_audit["negative_duration_rows_by_label"].items())
    )
    input_duration_audit["negative_duration_semantics"] = (
        "retained in the upstream duration<=0.5h partition with CHARTTIME=STARTTIME"
    )
    transformation_audit = {
        "schema_version": 1,
        "input_duration": input_duration_audit,
    }
    atomic_write_json(work_dir / "transformation_audit.json", transformation_audit)

    lab_path = work_dir / "LAB_processed.csv"
    append = False
    for chunk in _iter_chunks(source_dir / "LABEVENTS.csv", PRIMARY_SOURCES["LABEVENTS.csv"], chunksize):
        parsed_rows["LABEVENTS.csv"] += len(chunk)
        selected = clean_lab_events(_filter_hadm(chunk, hadm_ids), lab_labels)
        if len(selected):
            selected = selected[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "VALUENUM", "LABEL"]]
            _write_frame(lab_path, selected, append=append)
            append = True
    if not append:
        raise PreprocessError("no retained LABEVENTS rows")
    staged["labs"] = lab_path

    output_base = work_dir / "OUTPUTS_base.csv"
    append = False
    for chunk in _iter_chunks(source_dir / "OUTPUTEVENTS.csv", PRIMARY_SOURCES["OUTPUTEVENTS.csv"], chunksize):
        parsed_rows["OUTPUTEVENTS.csv"] += len(chunk)
        if chunk["ISERROR"].notna().any():
            raise PreprocessError("OUTPUTEVENTS contains non-null ISERROR entries")
        selected = clean_output_events(_filter_hadm(chunk, hadm_ids), item_labels)
        if len(selected):
            _write_frame(output_base, selected, append=append)
            append = True
    if not append:
        raise PreprocessError("no retained OUTPUTEVENTS rows")
    output_all = finalize_output_events(pd.read_csv(output_base, low_memory=False))
    outputs_path = work_dir / "OUTPUTS_processed.csv"
    output_all[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "VALUE", "LABEL"]].to_csv(
        outputs_path, index=False
    )
    staged["outputs"] = outputs_path
    del output_all

    prescription_base = work_dir / "PRESCRIPTIONS_base.csv"
    append = False
    for chunk in _iter_chunks(source_dir / "PRESCRIPTIONS.csv", PRIMARY_SOURCES["PRESCRIPTIONS.csv"], chunksize):
        parsed_rows["PRESCRIPTIONS.csv"] += len(chunk)
        selected = clean_prescriptions(_filter_hadm(chunk, hadm_ids))
        if len(selected):
            _write_frame(prescription_base, selected, append=append)
            append = True
    if not append:
        raise PreprocessError("no retained PRESCRIPTIONS rows")
    prescription_all = finalize_prescriptions(pd.read_csv(prescription_base, low_memory=False))
    prescriptions_path = work_dir / "PRESCRIPTIONS_processed.csv"
    prescription_all[
        ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "DOSE_VAL_RX", "DRUG"]
    ].to_csv(prescriptions_path, index=False)
    staged["prescriptions"] = prescriptions_path
    return staged, parsed_rows, transformation_audit


def _preflight_reference_dependencies(diagnoses_path: Path, mapping_path: Path) -> None:
    missing = [path for path in (diagnoses_path, mapping_path) if not path.is_file()]
    if missing:
        formatted = ", ".join(relative_project_path(path) for path in missing)
        raise PreprocessError(
            "canonical preprocessing requires notebook dependencies that are not present: "
            f"{formatted}. DIAGNOSES_ICD.csv filters the final rows; UNIQUE_ID_dict.csv "
            "contains the notebook's unseeded frozen patient permutation."
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = assert_inside_project(args.source_dir, must_exist=True)
    output = assert_inside_project(args.output)
    work_dir = assert_inside_project(args.work_dir)
    diagnoses_path = assert_inside_project(args.diagnoses_icd)
    mapping_path = assert_inside_project(args.unique_id_map)
    vendor_root = assert_inside_project(args.vendor_root, must_exist=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_mode = args.mode == "reference"
    if not reference_mode and output.name.lower() == "complete_tensor.csv":
        raise PreprocessError(
            "deterministic-nonreference output must not be named complete_tensor.csv; "
            "use an explicit name containing NONREFERENCE"
        )
    if not reference_mode and "nonreference" not in output.name.lower():
        raise PreprocessError("non-reference output filename must contain NONREFERENCE")
    notebook_manifest = verify_pinned_notebooks(vendor_root)
    if not args.audit_only and reference_mode and output.exists() and not args.force:
        existing = validate_candidate(output)
        return {
            "status": "already_valid_reference",
            "output": relative_project_path(output),
            **existing,
        }
    if not args.audit_only:
        # Check hidden dependencies before hashing or parsing multi-gigabyte tables.
        if reference_mode:
            _preflight_reference_dependencies(diagnoses_path, mapping_path)
        elif not diagnoses_path.is_file():
            raise PreprocessError(
                "deterministic-nonreference mode still requires DIAGNOSES_ICD.csv "
                "to preserve the notebook cohort"
            )
    if (
        not args.audit_only
        and not reference_mode
        and output.exists()
        and not args.force
    ):
        existing = validate_nonreference_candidate(output)
        return {
            "status": "already_valid_deterministic_nonreference",
            "output": relative_project_path(output),
            **existing,
        }

    audits: dict[str, CsvAudit] = {}
    for filename, columns in PRIMARY_SOURCES.items():
        audits[filename] = audit_csv(source_dir / filename, columns)
    source_manifest_path = work_dir / "source_manifest.json"
    auxiliary_audits: dict[str, CsvAudit] = {}
    for name, path, columns in (
        ("DIAGNOSES_ICD.csv", diagnoses_path, ("HADM_ID", "SEQ_NUM", "ICD9_CODE")),
        ("UNIQUE_ID_dict.csv", mapping_path, ("HADM_ID", "UNIQUE_ID")),
    ):
        if path.is_file():
            auxiliary_audits[name] = audit_csv(path, columns)
    dependency_presence = {
        "DIAGNOSES_ICD.csv": diagnoses_path.is_file(),
        "UNIQUE_ID_dict.csv": mapping_path.is_file(),
    }
    dependency_blockers = [
        f"missing:{name}" for name, present in dependency_presence.items() if not present
    ]
    mode_required = (
        ["DIAGNOSES_ICD.csv", "UNIQUE_ID_dict.csv"]
        if reference_mode
        else ["DIAGNOSES_ICD.csv"]
    )
    mode_blockers = [f"missing:{name}" for name in mode_required if not dependency_presence[name]]

    source_manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "project_relative_source_dir": relative_project_path(source_dir),
        "upstream": notebook_manifest,
        "sources": {name: asdict(audit) for name, audit in sorted(audits.items())},
        "auxiliary_sources": {
            name: asdict(audit)
            for name, audit in sorted(auxiliary_audits.items())
        },
        "reference_dependencies": {
            "required": ["DIAGNOSES_ICD.csv", "UNIQUE_ID_dict.csv"],
            "present": dependency_presence,
            "blockers": dependency_blockers,
        },
        "mode_dependencies": {
            "required": mode_required,
            "blockers": mode_blockers,
        },
        "time_binning": {
            "bin_k": APN_REFERENCE_BIN_K,
            "bins_per_hour": APN_REFERENCE_BIN_K,
            "formula": "round(elapsed_seconds * bin_k / 3600)",
            "selection_basis": (
                "pinned APN loader runtime contract plus canonical shape/hash"
            ),
            "conflicting_upstream_values": {
                "gru_ode_bayes_DataMerging_cell_11": 60,
                "pinned_apn_loader_docstring": 10,
                "official_apn_pdf_appendix_statement": "absent",
            },
            "canonical_expected_shape": list(REFERENCE_SHAPE),
            "canonical_expected_sha256": REFERENCE_SHA256,
            "deterministic_validation": "canonical shape only; reference hash forbidden",
        },
        "mode_semantics": {
            "cohort_contract": "same_primary_diagnosis_filter",
            "diagnoses_used": True,
            "unique_id_strategy": (
                "frozen_external_map" if reference_mode else "sorted_HADM_ID"
            ),
            "release_parity_candidate": reference_mode and not dependency_blockers,
            "deterministic_nonreference": not reference_mode,
        },
    }
    atomic_write_json(source_manifest_path, source_manifest)
    if args.audit_only:
        return {
            "status": "audit_complete",
            "source_manifest": relative_project_path(source_manifest_path),
        }


    admissions_raw = pd.read_csv(source_dir / "ADMISSIONS.csv", low_memory=False)
    patients_raw = pd.read_csv(source_dir / "PATIENTS.csv", low_memory=False)
    admission_audit: dict[str, Any] = {}
    admissions = prepare_admissions(admissions_raw, patients_raw, audit=admission_audit)
    validate_reference_admission_stage_counts(admission_audit)
    atomic_write_json(work_dir / "admissions_audit.json", admission_audit)
    admissions_path = work_dir / "Admissions_processed.csv"
    admissions.to_csv(admissions_path, index=True)
    staged, parsed_rows, transformation_audit = _stage_primary_tables(
        source_dir, work_dir, admissions, chunksize=args.chunksize
    )
    transformation_audit["admissions_filter"] = admission_audit
    transformation_audit_path = work_dir / "transformation_audit.json"
    atomic_write_json(transformation_audit_path, transformation_audit)

    for filename, audit in audits.items():
        if parsed_rows[filename] != audit.physical_data_rows:
            raise PreprocessError(
                f"parsed/physical row mismatch for {filename}: "
                f"{parsed_rows[filename]} != {audit.physical_data_rows}"
            )

    inputs = _read_selected(
        staged["inputs"], ("HADM_ID", "CHARTTIME", "AMOUNT", "LABEL")
    )
    labs = _read_selected(staged["labs"], ("HADM_ID", "CHARTTIME", "VALUENUM", "LABEL"))
    outputs = _read_selected(
        staged["outputs"], ("HADM_ID", "CHARTTIME", "VALUE", "LABEL")
    )
    prescriptions = _read_selected(
        staged["prescriptions"], ("HADM_ID", "CHARTTIME", "DOSE_VAL_RX", "DRUG")
    )
    source_timestamp_audit: dict[str, Any] = {}
    for origin, frame in (
        ("Inputs", inputs),
        ("Lab", labs),
        ("Outputs", outputs),
        ("Prescriptions", prescriptions),
    ):
        timestamp_audit: dict[str, Any] = {}
        frame["CHARTTIME"] = parse_mimic_datetime(
            frame["CHARTTIME"],
            name=f"{origin} CHARTTIME",
            audit=timestamp_audit,
        )
        timestamp_audit["null_rows_excluded_by_48h_window"] = timestamp_audit[
            "null_rows"
        ]
        source_timestamp_audit[origin] = timestamp_audit
    transformation_audit["source_timestamps"] = {
        "by_origin": source_timestamp_audit,
        "date_only_semantics": "valid prescription STARTDATE promoted to midnight",
        "null_semantics": "NaT rows excluded by the upstream TIME_STAMP<48h filter",
    }
    atomic_write_json(transformation_audit_path, transformation_audit)

    diagnoses = pd.read_csv(diagnoses_path, low_memory=False)
    mapping = load_unique_id_map(mapping_path) if reference_mode else None
    tensor, label_dictionary = merge_complete_tensor(
        inputs,
        labs,
        outputs,
        prescriptions,
        unique_ids=mapping,
        bin_k=APN_REFERENCE_BIN_K,
        diagnoses=diagnoses,
        minimum_observations=50,
        reference_mode=reference_mode,
    )
    candidate_name = (
        "complete_tensor.reference.candidate.csv"
        if reference_mode
        else "complete_tensor.deterministic_NONREFERENCE.candidate.csv"
    )
    candidate = work_dir / candidate_name

    tensor.to_csv(candidate, index=True)
    label_dictionary.to_csv(work_dir / "label_dict.csv", index=False)

    try:
        validation = (
            validate_candidate(candidate)
            if reference_mode
            else validate_nonreference_candidate(candidate)
        )
    except PreprocessError as exc:
        failure = {
            **source_manifest,
            "transformation_audit": transformation_audit,
            "status": "rejected_reference" if reference_mode else "rejected_nonreference",
            "candidate": relative_project_path(candidate),
            "candidate_shape": list(tensor.shape),
            "candidate_sha256": sha256_file(candidate),
            "error": str(exc),
        }
        atomic_write_json(work_dir / "preprocess_manifest.json", failure)
        raise

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(candidate, temporary_output)
    os.replace(temporary_output, output)
    completion_status = (
        "complete_reference"
        if reference_mode
        else "complete_deterministic_nonreference_not_release_parity"
    )
    final_manifest = {
        **source_manifest,
        "status": completion_status,
        "parsed_rows": parsed_rows,
        "transformation_audit": transformation_audit,
        "output": relative_project_path(output),
        "output_validation": validation,
        "parameters": {
            "chunksize": args.chunksize,
            "minimum_observations": 50,
            "mode": args.mode,
            "bin_k": APN_REFERENCE_BIN_K,
        },
        "release_parity_eligible": reference_mode,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "argv": [str(value) for value in sys.argv],
    }
    atomic_write_json(work_dir / "preprocess_manifest.json", final_manifest)
    return {"status": completion_status, "output": relative_project_path(output), **validation}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or prepare MIMIC-III tensors with explicit reference identity.",
    )
    parser.add_argument(
        "--mode",
        choices=("reference", "deterministic-nonreference"),
        default="reference",
        help=("reference requires the frozen mapping/diagnoses and exact APN hash; "
              "deterministic-nonreference uses sorted HADM_IDs and is never release parity"),
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "tsdm" / "rawdata" / "MIMIC_III_source",
        help="Directory containing the eight MIMIC-III 1.4 source CSV files.",
    )
    parser.add_argument(
        "--diagnoses-icd",
        type=Path,
        default=PROJECT_ROOT / "data" / "tsdm" / "rawdata" / "MIMIC_III_source" / "DIAGNOSES_ICD.csv",
        help="DIAGNOSES_ICD.csv used by DataMerging.ipynb's final inner join.",
    )
    parser.add_argument(
        "--unique-id-map",
        type=Path,
        default=PROJECT_ROOT / "data" / "tsdm" / "rawdata" / "MIMIC_III_source" / "UNIQUE_ID_dict.csv",
        help="Frozen UNIQUE_ID_dict.csv created by the upstream notebook run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "tsdm"
            / "rawdata"
            / "MIMIC_III_DeBrouwer2019"
            / "complete_tensor.csv"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "tsdm" / "work" / "MIMIC_III_DeBrouwer2019",
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=PROJECT_ROOT / "vendor" / "gru_ode_bayes",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Hash/audit source schemas and rows without transforming patient data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an output only after mode-specific validation passes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    try:
        result = run_pipeline(args)
    except (PreprocessError, FileNotFoundError, PermissionError, pd.errors.ParserError) as exc:
        print(f"MIMIC_PREPROCESS_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
