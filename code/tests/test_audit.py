from __future__ import annotations

import numpy as np

from evipatch.audit import audit_shift_views


def _view(
    x: np.ndarray,
    mask: np.ndarray,
    ids: np.ndarray,
    marks: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    requested: np.ndarray,
    original: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "ids": ids,
        "x": x,
        "x_mark": marks,
        "x_mask": mask,
        "y": y,
        "y_mask": y_mask,
        "pred": y.copy(),
        "requested": requested,
        "actual": requested.copy(),
        "original": original,
        "remaining": original - requested,
    }


def _synthetic_views() -> dict[str, dict[str, np.ndarray]]:
    ids = np.array([101, 202], dtype=np.int64)
    marks = np.broadcast_to(np.arange(5, dtype=np.float32)[None, :, None], (2, 5, 1)).copy()
    y = np.arange(4, dtype=np.float32).reshape(2, 1, 2)
    y_mask = np.ones_like(y)
    native_mask = np.array(
        [
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 0]],
            [[1, 1], [1, 1], [1, 1], [0, 1], [0, 1]],
        ],
        dtype=np.float32,
    )
    native_x = (np.arange(20, dtype=np.float32).reshape(2, 5, 2) + 1) * native_mask
    original = native_mask.sum(axis=1).astype(np.int64)
    requested = np.floor(original * 0.4).astype(np.int64)
    zero = np.zeros_like(requested)

    mcar_mask = native_mask.copy()
    mcar_mask[0, [0, 2], 0] = 0
    mcar_mask[0, [1], 1] = 0
    mcar_mask[1, [1], 0] = 0
    mcar_mask[1, [0, 3], 1] = 0
    burst_mask = native_mask.copy()
    burst_mask[0, [2, 3], 0] = 0
    burst_mask[0, [2], 1] = 0
    burst_mask[1, [2], 0] = 0
    burst_mask[1, [2, 3], 1] = 0
    mcar_x = native_x.copy()
    mcar_x[mcar_mask == 0] = 0
    burst_x = native_x.copy()
    burst_x[burst_mask == 0] = 0

    return {
        "none": _view(native_x, native_mask, ids, marks, y, y_mask, zero, original),
        "mcar": _view(mcar_x, mcar_mask, ids, marks, y, y_mask, requested, original),
        "burst": _view(burst_x, burst_mask, ids, marks, y, y_mask, requested, original),
    }


def test_audit_shift_views_accepts_exact_matched_deletions() -> None:
    result = audit_shift_views(_synthetic_views(), rate=0.4)
    assert result["passed"]
    assert all(result["checks"].values())
    assert result["totals"] == {
        "patients": 2,
        "original_observed": 17,
        "mcar_requested": 6,
        "mcar_actual": 6,
        "burst_requested": 6,
        "burst_actual": 6,
    }


def test_audit_shift_views_detects_target_mutation_and_count_mismatch() -> None:
    views = _synthetic_views()
    views["burst"]["y"] = views["burst"]["y"].copy()
    views["burst"]["y"][0, 0, 0] += 1
    views["mcar"]["actual"] = views["mcar"]["actual"].copy()
    views["mcar"]["actual"][0, 0] -= 1
    result = audit_shift_views(views, rate=0.4)
    assert not result["passed"]
    assert not result["checks"]["targets_identical"]
    assert not result["checks"]["mcar_requested_equals_actual"]
    assert not result["checks"]["mcar_burst_counts_matched"]
