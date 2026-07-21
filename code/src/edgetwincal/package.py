"""Fail-closed, deterministic delivery packaging for EdgeTwinCal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .paths import PROJECT_ROOT, require_within_root


CAMPAIGN_ID = "edgetwincal_msn2026_v1"
EXPECTED_RUN_MANIFESTS = 180
MAX_FILE_BYTES = 100 * 1024 * 1024
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PACKAGE_SCHEMA = "edgetwincal.delivery-package.v1"

_STATIC_FILES = (
    ".gitignore",
    "code/README.md",
    "code/configs/msn2026/default.json",
    "code/requirements.txt",
    "patches/apn_evipatch.patch",
)
_SCRIPT_FILES = (
    "aggregate_edgetwincal.py",
    "apply_apn_patch.ps1",
    "build_edgetwincal_tables.mjs",
    "common.ps1",
    "package_edgetwincal.py",
    "prepare_edgetwincal_fit_caches.py",
    "prepare_mimic_iii.py",
    "render_edgetwincal_results.py",
    "run_edgetwincal.py",
    "run_edgetwincal_backbones.py",
    "run_edgetwincal_once.py",
    "setup_environment.ps1",
)
_TEST_FILES = (
    "test_apn_parity.py",
    "test_edgetwincal_active_cli.py",
    "test_edgetwincal_aggregate_v2.py",
    "conftest.py",
    "test_edgetwincal_apn_bridge.py",
    "test_edgetwincal_backbone_campaign.py",
    "test_edgetwincal_apn_training.py",
    "test_edgetwincal_campaign.py",
    "test_edgetwincal_campaign_control.py",
    "test_edgetwincal_campaign_runner.py",
    "test_edgetwincal_config_schema.py",
    "test_edgetwincal_decoder_refit.py",
    "test_edgetwincal_fit_cache_campaign.py",
    "test_edgetwincal_lab_report.py",
    "test_edgetwincal_mechanisms.py",
    "test_edgetwincal_package.py",
    "test_edgetwincal_protocol.py",
    "test_edgetwincal_provenance.py",
    "test_edgetwincal_runtime_v2.py",
    "test_edgetwincal_statistics_timing.py",
    "test_edgetwincal_strict_p12.py",
    "test_edgetwincal_strict_ushcn.py",
    "test_edgetwincal_tables_script.py",
    "test_mimic_preprocess.py",
)
_ANALYSIS_REQUIRED = (
    "analysis_provenance.json",
    "blockers.json",
    "confirmatory_aggregate.json",
    "dataset_variant_summary.csv",
    "EdgeTwinCal_lab_results.xlsx",
    "failure_diagnosis.json",
    "figures/strict_ablation_gain.svg",
    "figures/strict_main_mse.svg",
    "gate_decision.json",
    "gate_summary.csv",
    "manifest_registry.json",
    "paired_comparisons.csv",
    "pretest_terminal_summary.json",
    "REPORT_CN.md",
    "seed_summary.csv",
)
_PRETEST_CELLS = (
    ("HumanActivity", "release_parity"),
    ("P12", "release_parity"),
    ("P12", "strict_p12"),
    ("USHCN", "release_parity"),
    ("USHCN", "strict_ushcn"),
)
_SAFE_PRETEST_CHECKS = (
    "G0_legacy_metric_parity.json",
    "G0_unit_suite.json",
    "G1_cache_provenance.json",
    "G1_fitted_registry.json",
    "G1_root_boundary.json",
    "G1_split_normalization.json",
)
_PROTOCOL_FILES = (
    "fit_entries.json",
    "fitted_registry.json",
    "once_campaign_result.json",
    "once_control_prepared.json",
    "protocol_ledger.json",
    "protocol_manifest.json",
    "split_manifest.json",
    "test_cache_registry.json",
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".conda",
        ".git",
        ".pytest_cache",
        "__pycache__",
        "cache",
        "caches",
        "checkpoint",
        "checkpoints",
        "credentials",
        "data",
        "datasets",
        "docs/manuscripts",
        "output",
        "packages",
        "secrets",
        "secret",
        "vendor",
    }
)
_FORBIDDEN_SUFFIXES = frozenset(
    {
        ".7z",
        ".bin",
        ".ckpt",
        ".gz",
        ".npy",
        ".npz",
        ".onnx",
        ".p12",
        ".pdf",
        ".pem",
        ".pfx",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
    }
)
_SECRET_FILENAMES = frozenset(
    {
        ".env",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
)
_CONTENT_PATTERNS = (
    ("private Windows home path", re.compile(rb"(?i)\b[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9._ -]+")),
    ("private POSIX home path", re.compile(rb"(?i)(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9._-]+")),
    ("GitHub token", re.compile(rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")),
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer credential", re.compile(rb"(?i)Authorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    (
        "assigned credential",
        re.compile(
            rb"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)"
            rb"\s*[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9+/=_-]{8,}"
        ),
    ),
    ("credential-bearing URL", re.compile(rb"(?i)https?://[^\s/:@]+:[^\s/@]+@")),
)


class PackageError(RuntimeError):
    """Raised when the delivery cannot be built without violating policy."""


@dataclass(frozen=True)
class PackageResult:
    archive: Path
    checksum_csv: Path
    sha256: str
    bytes: int
    source_file_count: int
    member_count: int


def sha256_file(path: str | Path) -> str:
    resolved = require_within_root(path, must_exist=True)
    if not resolved.is_file():
        raise PackageError(f"Not a regular project file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _delivery_root(root: str | Path) -> Path:
    resolved = require_within_root(root, must_exist=True)
    if not resolved.is_dir():
        raise PackageError(f"Delivery root is not a directory: {resolved}")
    return resolved


def _canonical_relative(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise PackageError(f"Package path is not canonical POSIX relative form: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PackageError(f"Unsafe package path: {value!r}")
    return relative


def _resolve_source(root: Path, relative: str, *, required: bool = True) -> Path | None:
    canonical = _canonical_relative(relative)
    candidate = root.joinpath(*canonical.parts)
    try:
        resolved = require_within_root(candidate, must_exist=required)
    except FileNotFoundError:
        if required:
            raise PackageError(f"Required delivery file is missing: {relative}") from None
        return None
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PackageError(f"Delivery source escapes selected root: {relative}") from exc
    if not resolved.exists():
        return None
    if not resolved.is_file():
        raise PackageError(f"Delivery source is not a regular file: {relative}")
    return resolved


def _relative_name(path: Path, root: Path) -> str:
    resolved = require_within_root(path, must_exist=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise PackageError(f"Delivery source is outside selected root: {resolved}") from exc
    _canonical_relative(relative)
    return relative


def _assert_path_policy(relative: str) -> None:
    canonical = _canonical_relative(relative)
    lowered = tuple(part.casefold() for part in canonical.parts)
    if any(part in _FORBIDDEN_PARTS for part in lowered):
        raise PackageError(f"Forbidden delivery path: {relative}")
    if len(lowered) >= 2 and "/".join(lowered[:2]) in _FORBIDDEN_PARTS:
        raise PackageError(f"Forbidden delivery path: {relative}")
    if canonical.suffix.casefold() in _FORBIDDEN_SUFFIXES:
        raise PackageError(f"Forbidden delivery file type: {relative}")
    if canonical.name.casefold() in _SECRET_FILENAMES:
        raise PackageError(f"Suspected secret filename: {relative}")


def _scan_bytes(payload: bytes, *, label: str) -> None:
    for finding, pattern in _CONTENT_PATTERNS:
        if pattern.search(payload):
            raise PackageError(f"Refusing {label}: detected {finding}")


def _scan_named_payload(name: str, payload: bytes) -> None:
    _scan_bytes(payload, label=name)
    if PurePosixPath(name).suffix.casefold() != ".xlsx":
        return
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as workbook:
            total = sum(info.file_size for info in workbook.infolist())
            if total > 4 * MAX_FILE_BYTES:
                raise PackageError(f"Refusing {name}: excessive expanded workbook size")
            for info in workbook.infolist():
                if info.is_dir():
                    continue
                if info.file_size > MAX_FILE_BYTES:
                    raise PackageError(f"Refusing {name}: oversized workbook member")
                _scan_bytes(workbook.read(info), label=f"{name}!{info.filename}")
    except zipfile.BadZipFile as exc:
        raise PackageError(f"Refusing malformed workbook: {name}") from exc


def audit_source_file(
    path: str | Path,
    *,
    root: str | Path = PROJECT_ROOT,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Enforce boundary, type, size, filename, private-path, and secret rules."""

    delivery_root = _delivery_root(root)
    resolved = require_within_root(path, must_exist=True)
    relative = _relative_name(resolved, delivery_root)
    _assert_path_policy(relative)
    size = resolved.stat().st_size
    if size > max_file_bytes:
        raise PackageError(
            f"Delivery source exceeds {max_file_bytes} bytes: {relative} ({size} bytes)"
        )
    payload = resolved.read_bytes()
    _scan_named_payload(relative, payload)
    return {"path": relative, "bytes": size, "sha256": hashlib.sha256(payload).hexdigest()}


def collect_registered_run_files(
    root: str | Path,
    registry_path: str | Path,
    *,
    expected_count: int = EXPECTED_RUN_MANIFESTS,
) -> list[Path]:
    """Resolve exactly the sealed run manifests and their compact run logs."""

    delivery_root = _delivery_root(root)
    registry = require_within_root(registry_path, must_exist=True)
    try:
        registry.relative_to(delivery_root)
    except ValueError as exc:
        raise PackageError("Manifest registry is outside the delivery root") from exc
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError("Manifest registry is unreadable") from exc
    manifest_names = payload.get("expected_manifests")
    if not isinstance(manifest_names, list) or not all(isinstance(item, str) for item in manifest_names):
        raise PackageError("Manifest registry must contain expected_manifests strings")
    if len(manifest_names) != expected_count or len(set(manifest_names)) != expected_count:
        raise PackageError(
            f"Manifest registry must contain exactly {expected_count} unique entries"
        )
    prefix = f"results/{CAMPAIGN_ID}/runs/"
    selected: list[Path] = []
    for relative in sorted(manifest_names):
        canonical = _canonical_relative(relative)
        if not relative.startswith(prefix) or canonical.name != "run_manifest.json":
            raise PackageError(f"Registry contains an out-of-scope run path: {relative}")
        manifest = _resolve_source(delivery_root, relative)
        assert manifest is not None
        try:
            run_payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PackageError(f"Unreadable run manifest: {relative}") from exc
        if run_payload.get("status") != "complete":
            raise PackageError(f"Non-complete run in sealed registry: {relative}")
        log_relative = canonical.parent.joinpath("run_log.json").as_posix()
        run_log = _resolve_source(delivery_root, log_relative)
        assert run_log is not None
        selected.extend((manifest, run_log))
    return selected


def collect_delivery_files(root: str | Path = PROJECT_ROOT) -> list[Path]:
    """Collect the closed whitelist for the final audited lab return."""

    delivery_root = _delivery_root(root)
    selected: dict[str, Path] = {}

    def add(relative: str, *, required: bool = True) -> None:
        source = _resolve_source(delivery_root, relative, required=required)
        if source is not None:
            selected[relative] = source

    for relative in _STATIC_FILES:
        add(relative)
    source_dir = _resolve_source(delivery_root, "code/src/edgetwincal/__init__.py")
    assert source_dir is not None
    for source in sorted(source_dir.parent.glob("*.py")):
        add(source.relative_to(delivery_root).as_posix())
    for name in _SCRIPT_FILES:
        add(f"code/scripts/{name}")
    for name in _TEST_FILES:
        add(f"code/tests/{name}")

    analysis_prefix = f"artifacts/{CAMPAIGN_ID}/analysis"
    for relative in _ANALYSIS_REQUIRED:
        add(f"{analysis_prefix}/{relative}")

    for dataset, protocol in _PRETEST_CELLS:
        prefix = f"artifacts/{CAMPAIGN_ID}/pretest/{dataset}/{protocol}"
        add(f"{prefix}/pretest_evidence.json")
        for name in _SAFE_PRETEST_CHECKS:
            add(f"{prefix}/checks/{name}")
        protocol_prefix = f"results/{CAMPAIGN_ID}/protocol/{dataset}/{protocol}"
        for name in _PROTOCOL_FILES:
            add(f"{protocol_prefix}/{name}")

    registry_path = _resolve_source(delivery_root, f"{analysis_prefix}/manifest_registry.json")
    assert registry_path is not None
    for source in collect_registered_run_files(delivery_root, registry_path):
        selected[_relative_name(source, delivery_root)] = source

    files = [selected[name] for name in sorted(selected)]
    for source in files:
        audit_source_file(source, root=delivery_root)
    return files


def _manifest_records(files: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    records = [audit_source_file(path, root=root) for path in files]
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise PackageError("Duplicate delivery source path")
    return sorted(records, key=lambda record: record["path"])


def _checksums_csv(records: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=("path", "bytes", "sha256"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def _package_manifest(records: Sequence[Mapping[str, Any]], verdict: str) -> bytes:
    payload = {
        "archive_timestamp": "1980-01-01T00:00:00Z",
        "campaign_id": CAMPAIGN_ID,
        "files": list(records),
        "schema_version": PACKAGE_SCHEMA,
        "source_file_count": len(records),
        "verdict": verdict,
    }
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    _canonical_relative(name)
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_zip_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    archive.writestr(_zip_info(name), payload)


def verify_delivery_archive(path: str | Path) -> dict[str, Any]:
    """Run CRC, member-set, recorded-hash, path, size, and content checks."""

    archive_path = require_within_root(path, must_exist=True)
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise PackageError("Archive contains duplicate member names")
            if archive.testzip() is not None:
                raise PackageError("ZIP CRC validation failed")
            if "PACKAGE_MANIFEST.json" not in names or "SHA256SUMS.csv" not in names:
                raise PackageError("Archive is missing its internal audit files")
            try:
                manifest = json.loads(archive.read("PACKAGE_MANIFEST.json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PackageError("Invalid internal package manifest") from exc
            if manifest.get("schema_version") != PACKAGE_SCHEMA:
                raise PackageError("Unexpected internal package schema")
            records = manifest.get("files")
            if not isinstance(records, list):
                raise PackageError("Internal package manifest has no file list")
            expected_names = {"PACKAGE_MANIFEST.json", "SHA256SUMS.csv"}
            record_map: dict[str, Mapping[str, Any]] = {}
            for record in records:
                if not isinstance(record, Mapping):
                    raise PackageError("Malformed internal file record")
                name = record.get("path")
                if not isinstance(name, str):
                    raise PackageError("Malformed internal file path")
                _assert_path_policy(name)
                if name in record_map:
                    raise PackageError(f"Duplicate internal file record: {name}")
                record_map[name] = record
                expected_names.add(name)
            if set(names) != expected_names:
                raise PackageError("Archive member set differs from its manifest")

            checksum_reader = csv.DictReader(
                io.StringIO(archive.read("SHA256SUMS.csv").decode("utf-8"))
            )
            checksum_rows = list(checksum_reader)
            if checksum_reader.fieldnames != ["path", "bytes", "sha256"]:
                raise PackageError("Unexpected internal checksum columns")
            checksum_map = {row["path"]: row for row in checksum_rows}
            if len(checksum_map) != len(checksum_rows) or set(checksum_map) != set(record_map):
                raise PackageError("Internal checksum rows differ from the package manifest")

            for info in infos:
                _canonical_relative(info.filename)
                if info.flag_bits & 0x1:
                    raise PackageError(f"Encrypted archive member: {info.filename}")
                if info.file_size > MAX_FILE_BYTES:
                    raise PackageError(f"Oversized archive member: {info.filename}")
                payload = archive.read(info)
                _scan_named_payload(info.filename, payload)
                if info.filename not in record_map:
                    continue
                record = record_map[info.filename]
                digest = hashlib.sha256(payload).hexdigest()
                try:
                    recorded_bytes = int(record.get("bytes", -1))
                    csv_bytes = int(checksum_map[info.filename]["bytes"])
                except (TypeError, ValueError) as exc:
                    raise PackageError(f"Invalid size record: {info.filename}") from exc
                if recorded_bytes != len(payload) or csv_bytes != len(payload):
                    raise PackageError(f"Archived size mismatch: {info.filename}")
                if record.get("sha256") != digest or checksum_map[info.filename]["sha256"] != digest:
                    raise PackageError(f"Archived SHA-256 mismatch: {info.filename}")
            if manifest.get("source_file_count") != len(record_map):
                raise PackageError("Internal source_file_count mismatch")
    except zipfile.BadZipFile as exc:
        raise PackageError(f"Invalid ZIP archive: {archive_path}") from exc
    return {
        "member_count": len(names),
        "member_hashes_verified": True,
        "source_file_count": len(record_map),
        "testzip_passed": True,
    }

def _replace_with_bounded_retry(source: Path, destination: Path) -> None:
    for attempt in range(4):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            retryable = os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}
            if not retryable or attempt == 3:
                raise
            time.sleep(0.01 * (2**attempt))
    raise AssertionError("bounded replace retry exhausted")



def _atomic_write(path: Path, payload: bytes) -> None:
    destination = require_within_root(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    require_within_root(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_bounded_retry(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_deterministic_archive(
    root: str | Path,
    files: Iterable[Path],
    output: str | Path,
    *,
    verdict: str,
) -> PackageResult:
    """Build and verify a byte-deterministic ZIP from already selected files."""

    delivery_root = _delivery_root(root)
    verdict_normalized = verdict.upper()
    if verdict_normalized not in {"PASS", "ABANDON"}:
        raise PackageError("A terminal PASS or ABANDON verdict is required")
    output_path = require_within_root(output)
    try:
        output_path.relative_to(delivery_root)
    except ValueError as exc:
        raise PackageError("Package output must remain under the selected delivery root") from exc
    if output_path.suffix.casefold() != ".zip":
        raise PackageError("Package output must use the .zip suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _manifest_records(list(files), delivery_root)
    package_manifest = _package_manifest(records, verdict_normalized)
    checksums = _checksums_csv(records)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    require_within_root(temporary)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for record in records:
                source = _resolve_source(delivery_root, str(record["path"]))
                assert source is not None
                info = _zip_info(str(record["path"]))
                with source.open("rb") as source_handle, archive.open(info, mode="w") as target:
                    shutil.copyfileobj(source_handle, target, length=1024 * 1024)
            _write_zip_bytes(archive, "PACKAGE_MANIFEST.json", package_manifest)
            _write_zip_bytes(archive, "SHA256SUMS.csv", checksums)
        verification = verify_delivery_archive(temporary)
        _replace_with_bounded_retry(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    verification = verify_delivery_archive(output_path)
    digest = sha256_file(output_path)
    return PackageResult(
        archive=output_path,
        checksum_csv=output_path.parent / "SHA256SUMS.csv",
        sha256=digest,
        bytes=output_path.stat().st_size,
        source_file_count=len(records),
        member_count=int(verification["member_count"]),
    )


def _external_checksum(result: PackageResult, root: Path) -> bytes:
    relative = _relative_name(result.archive, root)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "path",
            "bytes",
            "sha256",
            "testzip_passed",
            "member_hashes_verified",
            "member_count",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(
        {
            "path": relative,
            "bytes": result.bytes,
            "sha256": result.sha256,
            "testzip_passed": "true",
            "member_hashes_verified": "true",
            "member_count": result.member_count,
        }
    )
    return buffer.getvalue().encode("utf-8")


def create_delivery_package(
    root: str | Path = PROJECT_ROOT,
    *,
    output: str | Path | None = None,
) -> PackageResult:
    """Validate the final gate, collect the whitelist, and emit ZIP plus ZIP SHA CSV."""

    delivery_root = _delivery_root(root)
    gate_path = _resolve_source(
        delivery_root,
        f"artifacts/{CAMPAIGN_ID}/analysis/gate_decision.json",
    )
    assert gate_path is not None
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError("Final gate decision is unreadable") from exc
    verdict = str(gate.get("verdict", "")).upper()
    if verdict not in {"PASS", "ABANDON"}:
        raise PackageError("Final gate decision must be PASS or ABANDON")
    destination = (
        delivery_root
        / "packages"
        / f"EdgeTwinCal_{CAMPAIGN_ID}_{verdict.casefold()}_lab_return.zip"
        if output is None
        else Path(output)
    )
    if not destination.is_absolute():
        destination = delivery_root / destination
    files = collect_delivery_files(delivery_root)
    result = create_deterministic_archive(
        delivery_root,
        files,
        destination,
        verdict=verdict,
    )
    _atomic_write(result.checksum_csv, _external_checksum(result, delivery_root))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = create_delivery_package(output=args.output)
    print(
        json.dumps(
            {
                "archive": result.archive.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": result.bytes,
                "checksum_csv": result.checksum_csv.relative_to(PROJECT_ROOT).as_posix(),
                "member_count": result.member_count,
                "sha256": result.sha256,
                "source_file_count": result.source_file_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
