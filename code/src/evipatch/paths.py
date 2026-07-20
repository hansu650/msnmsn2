"""Project-root resolution and fail-closed output path guards."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root() -> Path:
    """Resolve the repository root from this module, never from caller cwd."""
    root = Path(__file__).resolve().parents[3]
    required = (
        root / "docs" / "user_requirements.md",
        root / "code",
        root / "vendor",
    )
    if root.name.lower() != "msn2" or not all(path.exists() for path in required):
        raise RuntimeError(
            f"Could not verify the isolated EviPatch project root from {__file__}: {root}"
        )
    return root


PROJECT_ROOT = resolve_project_root()


def assert_within_project(
    path: Path | str,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve a path against the project root and reject every escape."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    root = PROJECT_ROOT.resolve(strict=True)

    try:
        common = Path(os.path.commonpath([str(root), str(resolved)]))
    except ValueError as exc:
        raise ValueError(f"Path is on a different volume than project root: {path}") from exc

    if os.path.normcase(str(common)) != os.path.normcase(str(root)):
        raise ValueError(f"Path escapes project root {root}: {resolved}")
    if not allow_root and os.path.normcase(str(resolved)) == os.path.normcase(str(root)):
        raise ValueError("The project root itself is not an allowed output target")
    return resolved


def ensure_project_dir(path: Path | str) -> Path:
    """Validate and create a directory below the isolated project root."""
    resolved = assert_within_project(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def project_path(*parts: str, allow_root: bool = False) -> Path:
    """Construct and validate a project-local path."""
    return assert_within_project(PROJECT_ROOT.joinpath(*parts), allow_root=allow_root)


def assert_isolated_environment_paths(paths: dict[str, Path | str]) -> dict[str, Path]:
    """Validate a named collection of mutable paths before a run starts."""
    validated: dict[str, Path] = {}
    for name, value in paths.items():
        try:
            validated[name] = assert_within_project(value)
        except ValueError as exc:
            raise ValueError(f"Mutable path {name!r} is not isolated: {value}") from exc
    return validated
