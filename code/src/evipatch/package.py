"""Filtered, deterministic EviPatch delivery archive builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from evipatch.paths import (
    PROJECT_ROOT,
    assert_within_project,
    ensure_project_dir,
    project_path,
)


_SOURCE_SUFFIXES = {".py", ".json", ".md", ".txt", ".ps1", ".sh"}
_ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".log", ".npy"}
_RESULT_NAMES = {
    "metric.json",
    "run_manifest.json",
    "train_manifest.json",
    "configs.yaml",
    "output_pred.npy",
    "input_y.npy",
    "input_y_mask.npy",
    "input_sample_ID.npy",
    "input_shift_requested.npy",
    "input_shift_actual.npy",
}
_ALWAYS_EXCLUDED_PARTS = {
    ".git",
    ".conda",
    "__pycache__",
    ".pytest_cache",
    "cache",
    "data",
    "packages",
    "vendor",
}
_SECRET_NAMES = {".env", "id_rsa", "id_ed25519", "credentials", "credentials.json"}
_SECRET_SUFFIXES = {".pem", ".key", ".p12"}
_SECRET_PATTERNS = (
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def sha256_file(path: Path | str) -> str:
    resolved = assert_within_project(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_secret(path: Path) -> bool:
    if path.name.lower() in _SECRET_NAMES or path.suffix.lower() in _SECRET_SUFFIXES:
        return True
    if path.suffix.lower() not in _SOURCE_SUFFIXES | {".yaml", ".yml", ".csv", ".log"}:
        return False
    if path.stat().st_size > 10 * 1024 * 1024:
        return False
    data = path.read_bytes()
    return any(pattern.search(data) for pattern in _SECRET_PATTERNS)


def _allowed_source(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in _ALWAYS_EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name == ".gitignore" or path.name == "README.md":
        return True
    if relative.parts[0] == "docs":
        return path.suffix.lower() == ".md"
    if relative.parts[0] == "code" and any(
        part.lower() in {"data", "logs", "results"}
        for part in relative.parts[1:-1]
    ):
        return False
    if relative.parts[0] == "code":
        return path.suffix.lower() in _SOURCE_SUFFIXES
    if relative.parts[0] == "patches":
        return path.suffix.lower() == ".patch"
    return False


def iter_delivery_files(
    project_root: Path | str,
    gate: dict[str, Any],
) -> Iterator[Path]:
    """Yield reproducibility code plus compact Stage A evidence, never checkpoints."""
    root = assert_within_project(project_root, allow_root=True)
    if gate.get("verdict") not in {"PASS", "ABANDON"}:
        raise ValueError("A finalized PASS or ABANDON gate decision is required")

    selected: set[Path] = set()
    top_level_candidates = [
        root / "README.md",
        root / ".gitignore",
        root / "docs",
        root / "code",
        root / "patches",
        root / "artifacts",
        root / "results",
    ]
    candidates: list[Path] = []
    for top_level in top_level_candidates:
        if top_level.is_file():
            candidates.append(top_level)
        elif top_level.is_dir():
            candidates.extend(path for path in top_level.rglob("*") if path.is_file())

    for path in candidates:
        relative = path.relative_to(root)
        if _allowed_source(path, root):
            selected.add(path)
            continue
        if relative.parts and relative.parts[0] == "artifacts":
            if path.suffix.lower() in _ARTIFACT_SUFFIXES:
                selected.add(path)
            continue
        if relative.parts and relative.parts[0] == "results":
            if path.name in _RESULT_NAMES:
                selected.add(path)

    for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
        relative_parts = {part.lower() for part in path.relative_to(root).parts}
        if relative_parts & _ALWAYS_EXCLUDED_PARTS:
            continue
        if _is_secret(path):
            raise RuntimeError(f"Refusing to package suspected secret: {path}")
        yield path


def build_manifest(
    files: Iterable[Path],
    root: Path | str,
) -> list[dict[str, Any]]:
    """Record stable relative paths, byte sizes, and SHA-256 digests."""
    project_root = assert_within_project(root, allow_root=True)
    records = []
    for path in sorted(set(files), key=lambda item: item.relative_to(project_root).as_posix()):
        resolved = assert_within_project(path)
        records.append(
            {
                "path": resolved.relative_to(project_root).as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return records


def _checksums_csv(manifest: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(manifest)
    return buffer.getvalue()


def _zip_write_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def create_archive(
    root: Path | str,
    files: Iterable[Path],
    output: Path | str,
) -> Path:
    """Write a deterministic ZIP and verify every archived source checksum."""
    project_root = assert_within_project(root, allow_root=True)
    output_path = assert_within_project(output)
    ensure_project_dir(output_path.parent)
    selected = list(files)
    manifest = build_manifest(selected, project_root)
    package_manifest = {
        "schema_version": 1,
        "archive_timestamp": "1980-01-01T00:00:00+00:00",
        "project_root_name": project_root.name,
        "files": manifest,
    }
    checksums_text = _checksums_csv(manifest)
    checksum_path = output_path.parent / "SHA256SUMS.csv"
    manifest_path = output_path.parent / "PACKAGE_MANIFEST.json"
    checksum_path.write_text(checksums_text, encoding="utf-8", newline="")
    manifest_path.write_text(
        json.dumps(package_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for record in manifest:
            source = project_root / record["path"]
            _zip_write_bytes(archive, record["path"], source.read_bytes())
        _zip_write_bytes(archive, "SHA256SUMS.csv", checksums_text.encode("utf-8"))
        _zip_write_bytes(
            archive,
            "PACKAGE_MANIFEST.json",
            json.dumps(package_manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        )

    with zipfile.ZipFile(output_path, mode="r") as archive:
        archived_names = set(archive.namelist())
        expected_names = {record["path"] for record in manifest} | {
            "SHA256SUMS.csv",
            "PACKAGE_MANIFEST.json",
        }
        if archived_names != expected_names:
            raise RuntimeError("Archive member set differs from the package manifest")
        for record in manifest:
            digest = hashlib.sha256(archive.read(record["path"])).hexdigest()
            if digest != record["sha256"]:
                raise RuntimeError(f"Archive checksum mismatch: {record['path']}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
    return output_path


def _write_compact_logs(config: dict[str, Any]) -> None:
    raw_root = assert_within_project(config["project"]["logs_root"])
    compact_root = ensure_project_dir(
        Path(config["project"]["artifacts_root"]) / "compact_logs"
    )
    if not raw_root.exists():
        return
    for source in raw_root.rglob("*.log"):
        relative = source.relative_to(raw_root)
        destination = assert_within_project(compact_root / relative)
        ensure_project_dir(destination.parent)
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) <= 320:
            compact = lines
        else:
            omitted = len(lines) - 280
            compact = [
                *lines[:80],
                f"... {omitted} lines omitted from compact delivery log ...",
                *lines[-200:],
            ]
        destination.write_text("\n".join(compact) + "\n", encoding="utf-8")


def package_project(config: dict[str, Any]) -> Path:
    """Build the final delivery after a machine-readable gate decision exists."""
    artifacts = assert_within_project(config["project"]["artifacts_root"])
    gate_path = artifacts / "gate_decision.json"
    if not gate_path.is_file():
        raise FileNotFoundError("Run Stage A aggregation before packaging")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    _write_compact_logs(config)

    max_bytes = int(config["packaging"]["max_included_file_bytes"])
    files = list(iter_delivery_files(PROJECT_ROOT, gate))
    oversized = [path for path in files if path.stat().st_size > max_bytes]
    if oversized:
        details = ", ".join(
            f"{path.relative_to(PROJECT_ROOT)} ({path.stat().st_size} bytes)"
            for path in oversized
        )
        raise RuntimeError(f"Delivery files exceed the configured size cap: {details}")

    verdict = gate["verdict"].lower()
    output = (
        assert_within_project(config["project"]["packages_root"])
        / f"EviPatch_stage_a_{verdict}.zip"
    )
    return create_archive(PROJECT_ROOT, files, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_path("code", "configs", "stage_a.json"),
    )
    args = parser.parse_args(argv)
    from evipatch.runner import load_stage_config

    output = package_project(load_stage_config(args.config))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
