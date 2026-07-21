from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def require_within_root(path: str | Path, *, must_exist: bool = False) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {resolved}") from exc
    return resolved


def ensure_directory(path: str | Path) -> Path:
    resolved = require_within_root(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
