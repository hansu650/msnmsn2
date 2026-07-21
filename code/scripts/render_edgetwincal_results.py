"""Generate the sealed EdgeTwinCal laboratory-return artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "code" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from edgetwincal.lab_report import LabReportError, render_lab_return


def main() -> int:
    try:
        result = render_lab_return()
    except LabReportError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
