"""Fail-closed audit of every Stage A checkpoint, evaluation, and shift view."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evipatch.paths import PROJECT_ROOT, assert_within_project, ensure_project_dir


SHIFT_DIRECTORIES = {"none": "native", "mcar": "mcar", "burst": "burst"}
ARRAY_FILES = {
    "ids": "input_sample_ID.npy",
    "x": "input_x.npy",
    "x_mark": "input_x_mark.npy",
    "x_mask": "input_x_mask.npy",
    "y": "input_y.npy",
    "y_mask": "input_y_mask.npy",
    "pred": "output_pred.npy",
    "requested": "input_shift_requested.npy",
    "actual": "input_shift_actual.npy",
    "original": "input_shift_original_observed.npy",
    "remaining": "input_shift_remaining_observed.npy",
}
CROSS_VARIANT_FILES = tuple(
    ARRAY_FILES[key]
    for key in (
        "ids",
        "x",
        "x_mark",
        "x_mask",
        "y",
        "y_mask",
        "requested",
        "actual",
        "original",
        "remaining",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    missing = [filename for filename in ARRAY_FILES.values() if not (path / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation {path} is missing arrays: {missing}")
    return {
        key: np.load(path / filename, allow_pickle=False, mmap_mode="r")
        for key, filename in ARRAY_FILES.items()
    }


def _metric_from_arrays(view: Mapping[str, np.ndarray]) -> dict[str, float]:
    residual = (view["pred"] - view["y"]) * view["y_mask"]
    count = float(np.sum(view["y_mask"]))
    denominator = count if count > 0 else 1.0
    return {
        "MSE": float(np.sum(np.square(residual)) / denominator),
        "MAE": float(np.sum(np.abs(residual)) / denominator),
    }


def _shape_checks(view: Mapping[str, np.ndarray]) -> dict[str, bool]:
    x = view["x"]
    x_mask = view["x_mask"]
    x_mark = view["x_mark"]
    y = view["y"]
    y_mask = view["y_mask"]
    pred = view["pred"]
    ids = np.asarray(view["ids"]).reshape(-1)
    audit_shape = (x.shape[0], x.shape[2]) if x.ndim == 3 else None
    return {
        "history_rank": x.ndim == 3,
        "history_mask_shape": x.shape == x_mask.shape,
        "time_mark_shape": x_mark.ndim == 3 and x_mark.shape[:2] == x.shape[:2],
        "target_prediction_shape": y.shape == y_mask.shape == pred.shape and y.ndim == 3,
        "sample_id_count": ids.size == x.shape[0],
        "unique_sample_ids": np.unique(ids).size == ids.size,
        "audit_array_shapes": audit_shape is not None
        and all(
            np.asarray(view[key]).shape == audit_shape
            for key in ("requested", "actual", "original", "remaining")
        ),
    }


def audit_shift_views(
    views: Mapping[str, Mapping[str, np.ndarray]],
    rate: float,
) -> dict[str, Any]:
    """Audit native/MCAR/burst arrays for one variant and seed."""
    if set(views) != set(SHIFT_DIRECTORIES):
        raise ValueError(f"Expected views {sorted(SHIFT_DIRECTORIES)}, got {sorted(views)}")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be within [0, 1]")

    native = views["none"]
    mcar = views["mcar"]
    burst = views["burst"]
    expected = np.floor(np.asarray(native["original"]) * rate).astype(np.int64)
    checks: dict[str, bool] = {
        "sample_ids_identical": all(
            np.array_equal(native["ids"], views[shift]["ids"])
            for shift in ("mcar", "burst")
        ),
        "targets_identical": all(
            np.array_equal(native["y"], views[shift]["y"])
            for shift in ("mcar", "burst")
        ),
        "target_masks_identical": all(
            np.array_equal(native["y_mask"], views[shift]["y_mask"])
            for shift in ("mcar", "burst")
        ),
        "time_marks_identical": all(
            np.array_equal(native["x_mark"], views[shift]["x_mark"])
            for shift in ("mcar", "burst")
        ),
        "original_counts_identical": all(
            np.array_equal(native["original"], views[shift]["original"])
            for shift in ("mcar", "burst")
        ),
        "native_zero_requested": bool(np.all(native["requested"] == 0)),
        "native_zero_actual": bool(np.all(native["actual"] == 0)),
        "native_remaining_equals_original": np.array_equal(
            native["remaining"], native["original"]
        ),
        "mcar_exact_floor_request": np.array_equal(mcar["requested"], expected),
        "burst_exact_floor_request": np.array_equal(burst["requested"], expected),
        "mcar_requested_equals_actual": np.array_equal(
            mcar["requested"], mcar["actual"]
        ),
        "burst_requested_equals_actual": np.array_equal(
            burst["requested"], burst["actual"]
        ),
        "mcar_burst_counts_matched": np.array_equal(mcar["actual"], burst["actual"]),
        "remaining_accounting": all(
            np.array_equal(
                np.asarray(views[shift]["original"]) - np.asarray(views[shift]["actual"]),
                views[shift]["remaining"],
            )
            for shift in SHIFT_DIRECTORIES
        ),
        "mask_count_accounting": all(
            np.array_equal(
                np.asarray(views[shift]["x_mask"]).sum(axis=1),
                views[shift]["remaining"],
            )
            for shift in SHIFT_DIRECTORIES
        ),
    }

    native_mask = np.asarray(native["x_mask"]) > 0
    native_x = np.asarray(native["x"])
    for shift in ("mcar", "burst"):
        shifted_mask = np.asarray(views[shift]["x_mask"]) > 0
        shifted_x = np.asarray(views[shift]["x"])
        removed = native_mask & ~shifted_mask
        retained = shifted_mask
        removed_counts = removed.sum(axis=1)
        checks[f"{shift}_does_not_add_observations"] = bool(
            np.all(~shifted_mask | native_mask)
        )
        checks[f"{shift}_removed_mask_count_matches_actual"] = np.array_equal(
            removed_counts, views[shift]["actual"]
        )
        checks[f"{shift}_removed_values_zero"] = bool(np.all(shifted_x[removed] == 0))
        checks[f"{shift}_retained_values_unchanged"] = bool(
            np.array_equal(shifted_x[retained], native_x[retained])
        )

    totals = {
        "patients": int(np.asarray(native["ids"]).reshape(-1).size),
        "original_observed": int(np.asarray(native["original"]).sum()),
        "mcar_requested": int(np.asarray(mcar["requested"]).sum()),
        "mcar_actual": int(np.asarray(mcar["actual"]).sum()),
        "burst_requested": int(np.asarray(burst["requested"]).sum()),
        "burst_actual": int(np.asarray(burst["actual"]).sum()),
    }
    return {"checks": checks, "totals": totals, "passed": all(checks.values())}


def _one_checkpoint(run_root: Path) -> Path:
    matches = list(run_root.glob("checkpoints/**/pytorch_model.bin"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint below {run_root}, found {len(matches)}")
    return assert_within_project(matches[0])


def _process_checks(process: Mapping[str, Any]) -> dict[str, bool]:
    allocated = process.get("peak_cuda_allocated_mib")
    reserved = process.get("peak_cuda_reserved_mib")
    return {
        "return_code_zero": process.get("return_code") == 0,
        "positive_wall_time": float(process.get("wall_seconds", 0)) > 0,
        "torch_cuda_memory_source": process.get("gpu_memory_source") == "torch.cuda",
        "positive_cuda_allocated": allocated is not None and float(allocated) > 0,
        "reserved_not_below_allocated": allocated is not None
        and reserved is not None
        and float(reserved) >= float(allocated),
    }


def _artifact_checks(manifest: Mapping[str, Any]) -> dict[str, bool]:
    records = manifest.get("artifacts", [])
    checks: dict[str, bool] = {"artifact_records_present": bool(records)}
    for index, record in enumerate(records):
        relative = Path(record["path"])
        path = assert_within_project(PROJECT_ROOT / relative)
        checks[f"artifact_{index}_exists"] = path.is_file()
        if path.is_file():
            checks[f"artifact_{index}_size"] = path.stat().st_size == int(record["bytes"])
            checks[f"artifact_{index}_sha256"] = _sha256(path) == record["sha256"]
    return checks


def _record_failures(failures: list[str], prefix: str, checks: Mapping[str, bool]) -> None:
    failures.extend(f"{prefix}: {name}" for name, passed in checks.items() if not passed)


def audit_stage_a(config: dict[str, Any]) -> dict[str, Any]:
    """Audit all formal Stage A outputs and write a machine-readable verdict."""
    stage = config["stage_a"]
    results_root = assert_within_project(config["project"]["results_root"])
    artifacts_root = ensure_project_dir(config["project"]["artifacts_root"])
    patch_path = assert_within_project(config["upstream"]["patch"])
    patch_sha256 = _sha256(patch_path)
    failures: list[str] = []
    runs: list[dict[str, Any]] = []
    cross_variant: list[dict[str, Any]] = []
    reference_hashes: dict[tuple[int, str], dict[str, str]] = {}
    project_commits: set[str] = set()
    observed_patch_hashes: set[str] = set()
    training_count = 0
    evaluation_count = 0

    for variant in stage["variants"]:
        for seed in stage["seeds"]:
            prefix = f"{variant}/{seed}"
            run_root = assert_within_project(results_root / "stage_a" / variant / str(seed))
            checkpoint = _one_checkpoint(run_root)
            train_manifest_path = run_root / "train_manifest.json"
            if not train_manifest_path.is_file():
                raise FileNotFoundError(train_manifest_path)
            train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
            training_count += 1
            train_process_checks = _process_checks(train_manifest["process"])
            train_checks = {
                "manifest_variant": train_manifest.get("variant") == variant,
                "manifest_seed": train_manifest.get("seed") == seed,
                "manifest_action": train_manifest.get("action") == "train",
                "manifest_project_root": os.path.normcase(train_manifest.get("project_root", ""))
                == os.path.normcase(str(PROJECT_ROOT)),
                "checkpoint_path": Path(train_manifest.get("checkpoint", "")) == checkpoint,
                "checkpoint_sha256": train_manifest.get("checkpoint_sha256")
                == _sha256(checkpoint),
                "positive_parameter_count": int(train_manifest.get("parameter_count") or 0) > 0,
                **train_process_checks,
                **_artifact_checks(train_manifest),
            }
            provenance = train_manifest["provenance"]
            train_checks.update(
                {
                    "pinned_apn_commit": provenance.get("apn_commit")
                    == config["upstream"]["commit"],
                    "clean_project_status": provenance.get("project_git_status") == "",
                    "current_patch_sha256": provenance.get("apn_patch_sha256")
                    == patch_sha256,
                }
            )
            project_commits.add(str(provenance.get("project_git_commit")))
            observed_patch_hashes.add(str(provenance.get("apn_patch_sha256")))
            _record_failures(failures, f"{prefix}/train", train_checks)

            views: dict[str, dict[str, np.ndarray]] = {}
            shift_records: dict[str, Any] = {}
            run_hashes: dict[str, dict[str, str]] = {}
            for shift, directory_name in SHIFT_DIRECTORIES.items():
                eval_dir = assert_within_project(checkpoint.parent / f"eval_{directory_name}")
                metric_path = eval_dir / "metric.json"
                manifest_path = eval_dir / "run_manifest.json"
                if not metric_path.is_file() or not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing metric/manifest at {eval_dir}")
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                arrays = _load_arrays(eval_dir)
                views[shift] = arrays
                evaluation_count += 1
                shape_checks = _shape_checks(arrays)
                finite_checks = {
                    f"finite_{key}": bool(np.isfinite(np.asarray(arrays[key])).all())
                    for key in ("ids", "x", "x_mark", "x_mask", "y", "y_mask", "pred")
                }
                recomputed = _metric_from_arrays(arrays)
                metric_checks = {
                    "finite_metric_mse": bool(np.isfinite(float(metric["MSE"]))),
                    "finite_metric_mae": bool(np.isfinite(float(metric["MAE"]))),
                    "metric_mse_recomputed": bool(
                        np.isclose(recomputed["MSE"], float(metric["MSE"]), rtol=1e-6, atol=1e-7)
                    ),
                    "metric_mae_recomputed": bool(
                        np.isclose(recomputed["MAE"], float(metric["MAE"]), rtol=1e-6, atol=1e-7)
                    ),
                }
                eval_checks = {
                    "manifest_variant": manifest.get("variant") == variant,
                    "manifest_seed": manifest.get("seed") == seed,
                    "manifest_shift": manifest.get("shift") == shift,
                    "manifest_action": manifest.get("action") == "evaluate",
                    "checkpoint_sha256": manifest.get("checkpoint_sha256")
                    == train_manifest.get("checkpoint_sha256"),
                    **_process_checks(manifest["process"]),
                    **shape_checks,
                    **finite_checks,
                    **metric_checks,
                    **_artifact_checks(manifest),
                }
                eval_provenance = manifest["provenance"]
                eval_checks.update(
                    {
                        "project_commit_matches_train": eval_provenance.get("project_git_commit")
                        == provenance.get("project_git_commit"),
                        "pinned_apn_commit": eval_provenance.get("apn_commit")
                        == config["upstream"]["commit"],
                        "clean_project_status": eval_provenance.get("project_git_status") == "",
                        "current_patch_sha256": eval_provenance.get("apn_patch_sha256")
                        == patch_sha256,
                    }
                )
                _record_failures(failures, f"{prefix}/{shift}", eval_checks)
                hashes = {
                    filename: _sha256(eval_dir / filename)
                    for filename in CROSS_VARIANT_FILES
                }
                run_hashes[shift] = hashes
                reference_key = (seed, shift)
                if variant == stage["variants"][0]:
                    reference_hashes[reference_key] = hashes
                    hash_checks = {filename: True for filename in CROSS_VARIANT_FILES}
                else:
                    reference = reference_hashes[reference_key]
                    hash_checks = {
                        filename: digest == reference[filename]
                        for filename, digest in hashes.items()
                    }
                _record_failures(failures, f"{prefix}/{shift}/cross_variant", hash_checks)
                cross_variant.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "shift": shift,
                        "reference_variant": stage["variants"][0],
                        "checks": hash_checks,
                        "passed": all(hash_checks.values()),
                    }
                )
                shift_records[shift] = {
                    "metric": {"MSE": float(metric["MSE"]), "MAE": float(metric["MAE"])},
                    "recomputed_metric": recomputed,
                    "peak_cuda_allocated_mib": manifest["process"].get(
                        "peak_cuda_allocated_mib"
                    ),
                    "peak_cuda_reserved_mib": manifest["process"].get(
                        "peak_cuda_reserved_mib"
                    ),
                    "checks": eval_checks,
                    "passed": all(eval_checks.values()),
                }

            shift_audit = audit_shift_views(views, float(stage["shift_rate"]))
            _record_failures(failures, f"{prefix}/shift_audit", shift_audit["checks"])
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": train_manifest["checkpoint_sha256"],
                    "parameter_count": train_manifest["parameter_count"],
                    "train_checks": train_checks,
                    "train_passed": all(train_checks.values()),
                    "shifts": shift_records,
                    "shift_audit": shift_audit,
                    "passed": all(train_checks.values())
                    and all(record["passed"] for record in shift_records.values())
                    and shift_audit["passed"],
                }
            )

    global_checks = {
        "training_count_21": training_count == len(stage["variants"]) * len(stage["seeds"]),
        "evaluation_count_63": evaluation_count
        == len(stage["variants"]) * len(stage["seeds"]) * len(stage["shifts"]),
        "single_project_commit": len(project_commits) == 1,
        "single_patch_hash": observed_patch_hashes == {patch_sha256},
        "all_runs_passed": all(run["passed"] for run in runs),
        "all_cross_variant_checks_passed": all(item["passed"] for item in cross_variant),
    }
    _record_failures(failures, "global", global_checks)
    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "project_root": str(PROJECT_ROOT),
        "training_count": training_count,
        "evaluation_count": evaluation_count,
        "project_git_commits": sorted(project_commits),
        "apn_commit": config["upstream"]["commit"],
        "patch_sha256": patch_sha256,
        "global_checks": global_checks,
        "failures": failures,
        "runs": runs,
        "cross_variant": cross_variant,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    output = assert_within_project(artifacts_root / "stage_a_audit.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Stage A audit failed with {len(failures)} discrepancies; see {output}")
    return result
