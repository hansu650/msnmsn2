"""Active msn2026 runtime facade.

Only the provenance-complete schema-3 runtime is exported here.  The former
schema-2 cache implementation lives in :mod:`legacy_runtime_v1` and is not an
active campaign path.
"""

from __future__ import annotations

from . import runtime_v2 as _runtime_v2
from .runtime_v2 import *  # noqa: F401,F403 - deliberate facade


__all__ = tuple(_runtime_v2.__all__)


def main(argv=None) -> int:
    return _runtime_v2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
