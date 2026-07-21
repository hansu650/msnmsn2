"""Boundary-checked facade for the EdgeTwinCal once-only campaign control."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from edgetwincal.campaign_control import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
