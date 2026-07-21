"""Active fail-closed aggregation facade for the msn2026 campaign."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .aggregate_v2 import (
    AGGREGATE_SCHEMA,
    ConfirmatoryAggregationError,
    aggregate_confirmatory,
    classify_dataset_evidence,
)


def run_aggregation(
    expected_manifest_paths: Sequence[str | Path],
    blocker_records: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Aggregate an explicit manifest registry through the v2 implementation.

    There are intentionally no implicit seeds, result directories, filesystem
    globs, or schema-2 cache fallbacks in this active entry point.
    """
    if not expected_manifest_paths:
        raise ConfirmatoryAggregationError("An explicit non-empty manifest registry is required")

    return aggregate_confirmatory(expected_manifest_paths, blocker_records, **kwargs)


def compact_run_manifest(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compatibility helper for immutable legacy-v1 parity tests only.

    New confirmatory runs use :class:`edgetwincal.schema.RunManifest`.  Importing
    the old helper lazily prevents legacy result discovery from entering the
    active aggregation path.
    """

    from .legacy_aggregate_v1 import compact_run_manifest as legacy_compact

    return legacy_compact(metrics)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared active CLI's explicit ``aggregate`` command."""

    from .experiment import main as experiment_main

    forwarded = list(sys.argv[1:] if argv is None else argv)
    return experiment_main(["aggregate", *forwarded])


__all__ = (
    "AGGREGATE_SCHEMA",
    "ConfirmatoryAggregationError",
    "aggregate_confirmatory",
    "classify_dataset_evidence",
    "run_aggregation",
    "compact_run_manifest",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
