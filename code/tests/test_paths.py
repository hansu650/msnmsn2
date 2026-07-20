from __future__ import annotations

from pathlib import Path

import pytest

from evipatch.paths import (
    PROJECT_ROOT,
    assert_isolated_environment_paths,
    assert_within_project,
    ensure_project_dir,
    project_path,
)


def test_project_root_is_exact() -> None:
    assert PROJECT_ROOT == Path(r"C:\Users\qintian\Desktop\msn2")
    assert project_path("results") == PROJECT_ROOT / "results"


@pytest.mark.parametrize(
    "candidate",
    [
        Path(r"C:\Users\qintian\Desktop\msn"),
        Path(r"C:\tmp"),
        Path("..") / "msn",
        Path("..") / "outside",
    ],
)
def test_outside_paths_fail_closed(candidate: Path) -> None:
    with pytest.raises(ValueError):
        assert_within_project(candidate)


def test_root_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError):
        assert_within_project(PROJECT_ROOT)
    assert assert_within_project(PROJECT_ROOT, allow_root=True) == PROJECT_ROOT


def test_project_temp_directory_is_allowed(tmp_path: Path) -> None:
    resolved = assert_within_project(tmp_path)
    assert PROJECT_ROOT in resolved.parents
    child = ensure_project_dir(resolved / "child")
    assert child.is_dir()


def test_bulk_validation_identifies_named_escape() -> None:
    with pytest.raises(ValueError, match="cache"):
        assert_isolated_environment_paths(
            {"results": project_path("results"), "cache": Path(r"C:\tmp")}
        )
