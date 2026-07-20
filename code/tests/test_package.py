from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from evipatch.package import create_archive, iter_delivery_files
from evipatch.paths import PROJECT_ROOT


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_delivery_selection_excludes_private_and_large_source_trees() -> None:
    files = list(iter_delivery_files(PROJECT_ROOT, {"verdict": "ABANDON"}))
    relative = [path.relative_to(PROJECT_ROOT).as_posix() for path in files]
    assert "code/src/evipatch/package.py" in relative
    assert "patches/apn_evipatch.patch" in relative
    assert not any(name.startswith(".conda/") for name in relative)
    assert not any(name.startswith("vendor/") for name in relative)
    assert not any(name.startswith("data/") for name in relative)
    assert not any(name.endswith("pytorch_model.bin") for name in relative)


def test_archive_is_verified_and_byte_deterministic(tmp_path: Path) -> None:
    selected = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "code" / "configs" / "stage_a.json",
    ]
    first = create_archive(PROJECT_ROOT, selected, tmp_path / "first.zip")
    first_digest = _digest(first)
    second = create_archive(PROJECT_ROOT, selected, tmp_path / "second.zip")
    second_digest = _digest(second)
    assert first_digest == second_digest
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "README.md",
            "code/configs/stage_a.json",
            "SHA256SUMS.csv",
            "PACKAGE_MANIFEST.json",
        }
