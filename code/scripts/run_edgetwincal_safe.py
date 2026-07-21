"""CLI entry point for the frozen EdgeTwinCal-Safe campaign."""

from __future__ import annotations

import os
import sys
_CUBLAS_ALLOWED = {":4096:8", ":16:8"}
_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _cublas is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
elif _cublas not in _CUBLAS_ALLOWED:
    raise RuntimeError(
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8"
    )
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
for path in (SOURCE_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from edgetwincal.safe_experiment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
