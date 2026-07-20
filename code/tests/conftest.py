from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_SRC = PROJECT_ROOT / "code" / "src"
APN_ROOT = PROJECT_ROOT / "vendor" / "APN"

for path in (CODE_SRC, APN_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
