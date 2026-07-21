from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "code" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from edgetwincal.aggregate import run_aggregation


if __name__ == "__main__":
    run_aggregation()
