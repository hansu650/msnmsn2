from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
import edgetwincal.package as package_module

from edgetwincal.package import (
    PACKAGE_SCHEMA,
    PackageError,
    audit_source_file,
    collect_registered_run_files,
    create_deterministic_archive,
    create_delivery_package,
    verify_delivery_archive,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_rejects_forbidden_paths_size_private_paths_and_secrets(
    tmp_path: Path,
) -> None:
    safe = _write(tmp_path / "safe" / "result.json", '{"ok": true}\n')
    assert audit_source_file(safe, root=tmp_path)["path"] == "safe/result.json"

    forbidden = _write(tmp_path / "data" / "raw.json", "{}\n")
    with pytest.raises(PackageError, match="Forbidden delivery path"):
        audit_source_file(forbidden, root=tmp_path)

    oversized = _write(tmp_path / "safe" / "large.txt", "1234")
    with pytest.raises(PackageError, match="exceeds 3 bytes"):
        audit_source_file(oversized, root=tmp_path, max_file_bytes=3)

    private_value = "C:" + "\\" + "Users" + "\\" + "someone" + "\\project"
    private_file = _write(tmp_path / "safe" / "private.txt", private_value)
    with pytest.raises(PackageError, match="private Windows home path"):
        audit_source_file(private_file, root=tmp_path)

    token_value = "gh" + "p_" + ("A" * 24)
    token_file = _write(tmp_path / "safe" / "token.txt", token_value)
    with pytest.raises(PackageError, match="GitHub token"):
        audit_source_file(token_file, root=tmp_path)

    workbook = tmp_path / "safe" / "private.xlsx"
    with zipfile.ZipFile(workbook, mode="w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            "C:" + "\\" + "Users" + "\\" + "workbook-owner" + "\\source",
        )
    with pytest.raises(PackageError, match="private Windows home path"):
        audit_source_file(workbook, root=tmp_path)


def test_registered_runs_are_exact_complete_and_include_logs(tmp_path: Path) -> None:
    relative_manifests: list[str] = []
    for seed in (2024, 2025):
        run_dir = (
            tmp_path
            / "results"
            / "edgetwincal_msn2026_v1"
            / "runs"
            / "P12"
            / "strict_p12"
            / f"seed_{seed}"
            / "APN"
        )
        manifest = _write(run_dir / "run_manifest.json", '{"status": "complete"}\n')
        _write(run_dir / "run_log.json", '{"events": []}\n')
        relative_manifests.append(manifest.relative_to(tmp_path).as_posix())
    registry = _write(
        tmp_path / "registry.json",
        json.dumps({"expected_manifests": list(reversed(relative_manifests))}),
    )
    files = collect_registered_run_files(tmp_path, registry, expected_count=2)
    names = [path.name for path in files]
    assert names.count("run_manifest.json") == 2
    assert names.count("run_log.json") == 2

    _write(
        tmp_path.joinpath(*Path(relative_manifests[0]).parts),
        '{"status": "failed"}\n',
    )
    with pytest.raises(PackageError, match="Non-complete run"):
        collect_registered_run_files(tmp_path, registry, expected_count=2)


def test_archive_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    first_source = _write(tmp_path / "safe" / "a.txt", "alpha\n")
    second_source = _write(tmp_path / "safe" / "b.json", '{"value": 2}\n')
    selected = [second_source, first_source]

    first = create_deterministic_archive(
        tmp_path,
        selected,
        tmp_path / "first.zip",
        verdict="ABANDON",
    )
    second = create_deterministic_archive(
        tmp_path,
        list(reversed(selected)),
        tmp_path / "second.zip",
        verdict="ABANDON",
    )
    assert _sha(first.archive) == _sha(second.archive)
    assert first.source_file_count == 2
    audit = verify_delivery_archive(first.archive)
    assert audit == {
        "member_count": 4,
        "member_hashes_verified": True,
        "source_file_count": 2,
        "testzip_passed": True,
    }

    with zipfile.ZipFile(first.archive) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        assert manifest["schema_version"] == PACKAGE_SCHEMA
        assert manifest["verdict"] == "ABANDON"
        rows = list(
            csv.DictReader(
                io.StringIO(archive.read("SHA256SUMS.csv").decode("utf-8"))
            )
        )
        assert [row["path"] for row in rows] == ["safe/a.txt", "safe/b.json"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_archive_rejects_sources_and_outputs_outside_selected_root(
    tmp_path: Path,
) -> None:
    inside = _write(tmp_path / "inside.txt", "inside\n")
    outside = _write(tmp_path.parent / f"outside-{tmp_path.name}.txt", "outside\n")
    try:
        with pytest.raises(PackageError, match="outside selected root"):
            create_deterministic_archive(
                tmp_path,
                [outside],
                tmp_path / "bad.zip",
                verdict="ABANDON",
            )
        with pytest.raises(PackageError, match="output must remain"):
            create_deterministic_archive(
                tmp_path,
                [inside],
                tmp_path.parent / f"bad-{tmp_path.name}.zip",
                verdict="ABANDON",
            )
    finally:
        outside.unlink(missing_ok=True)


def test_delivery_entrypoint_writes_external_zip_verification_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _write(
        tmp_path
        / "artifacts"
        / "edgetwincal_msn2026_v1"
        / "analysis"
        / "gate_decision.json",
        '{"verdict": "ABANDON"}\n',
    )
    source = _write(tmp_path / "safe" / "result.json", '{"complete": true}\n')
    monkeypatch.setattr(package_module, "collect_delivery_files", lambda root: [gate, source])

    result = create_delivery_package(tmp_path)
    assert result.archive.is_file()
    assert result.checksum_csv == tmp_path / "packages" / "SHA256SUMS.csv"
    rows = list(
        csv.DictReader(io.StringIO(result.checksum_csv.read_text(encoding="utf-8")))
    )
    assert rows == [
        {
            "path": result.archive.relative_to(tmp_path).as_posix(),
            "bytes": str(result.bytes),
            "sha256": result.sha256,
            "testzip_passed": "true",
            "member_hashes_verified": "true",
            "member_count": str(result.member_count),
        }
    ]
    assert verify_delivery_archive(result.archive)["member_hashes_verified"] is True
