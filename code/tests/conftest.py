from __future__ import annotations

import sys
import shutil
import uuid

import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_SRC = PROJECT_ROOT / "code" / "src"
APN_ROOT = PROJECT_ROOT / "vendor" / "APN"

for path in (CODE_SRC, APN_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


@pytest.fixture
def tmp_path():
    """Keep every test artifact inside msn2 and avoid host temp ACL drift."""

    base = PROJECT_ROOT / "results" / "pytest_msn2026_scratch"
    root = base / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root)
