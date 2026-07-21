"""Safe entry point for the train/validation fit-cache campaign."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"


def _establish_environment() -> None:
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE
    elif current != REQUIRED_CUBLAS_WORKSPACE:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{REQUIRED_CUBLAS_WORKSPACE!r}; observed {current!r}"
        )
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def main() -> int:
    _establish_environment()
    from edgetwincal.fit_cache_campaign import main as campaign_main

    return campaign_main()


if __name__ == "__main__":
    raise SystemExit(main())
