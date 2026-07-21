from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "code" / "src"
for path in (SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from edgetwincal.experiment import main


if __name__ == "__main__":
    main()
