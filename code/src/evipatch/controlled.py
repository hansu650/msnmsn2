"""Frozen controlled-support matching and scoring for the Stage A kill gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from evipatch.paths import assert_within_project, ensure_project_dir


PAIR_COLUMNS = [
    "seed",
    "pair_id",
    "channel",
    "low_patient_id",
    "high_patient_id",
    "low_patch",
    "high_patch",
    "centroid_rms",
    "low_effective_support",
    "high_effective_support",
    "support_ratio",
    "support_difference",
]
ERROR_COLUMNS = [
    "variant",
    "seed",
    "pair_id",
    "channel",
    "low_patient_id",
    "high_patient_id",
    "MSE",
    "observed_target_count",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_tensor(state: Mapping[str, torch.Tensor], suffix: str) -> np.ndarray:
    matches = [(key, value) for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        keys = [key for key, _ in matches]
        raise KeyError(f"Expected one checkpoint tensor ending in {suffix!r}, got {keys}")
    return matches[0][1].detach().cpu().numpy().astype(np.float64, copy=False)


def _learned_time_embedding(times: np.ndarray, state: Mapping[str, torch.Tensor]) -> np.ndarray:
    values = times[..., None].astype(np.float64, copy=False)
    scale_weight = _state_tensor(state, "te_scale.weight")
    scale_bias = _state_tensor(state, "te_scale.bias")
    periodic_weight = _state_tensor(state, "te_periodic.weight")
    periodic_bias = _state_tensor(state, "te_periodic.bias")
    scale = values @ scale_weight.T + scale_bias
    periodic = np.sin(values @ periodic_weight.T + periodic_bias)
    return np.concatenate([scale, periodic], axis=-1)


def patch_units(
    x: np.ndarray,
    x_mark: np.ndarray,
    x_mask: np.ndarray,
    target_mask: np.ndarray,
    sample_ids: np.ndarray,
    state: Mapping[str, torch.Tensor],
    seed: int,
    *,
    support_epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Reconstruct APN soft centroids and keep one max-support patch per patient/channel."""
    values = np.asarray(x, dtype=np.float64)
    masks = np.asarray(x_mask, dtype=np.float64)
    marks = np.asarray(x_mark, dtype=np.float64)
    targets = np.asarray(target_mask, dtype=np.float64)
    ids = np.rint(np.asarray(sample_ids).reshape(-1)).astype(np.int64)
    if values.shape != masks.shape or values.ndim != 3:
        raise ValueError("x and x_mask must have identical [B,L,N] shapes")
    if marks.ndim != 3 or marks.shape[:2] != values.shape[:2] or marks.shape[2] < 1:
        raise ValueError("x_mark must have shape [B,L,D>=1]")
    if targets.ndim != 3 or targets.shape[0] != values.shape[0] or targets.shape[2] != values.shape[2]:
        raise ValueError("target_mask must have shape [B,H,N]")
    if ids.size != values.shape[0] or np.unique(ids).size != ids.size:
        raise ValueError("sample IDs must be unique and match the history batch")
    if support_epsilon <= 0:
        raise ValueError("support_epsilon must be positive")

    delta = _state_tensor(state, "patching.delta_left_params")
    raw_width = _state_tensor(state, "patching.raw_log_width_params")
    tau_params = _state_tensor(state, "patching.tau_params")
    batch_size, _, n_variables = values.shape
    if delta.ndim != 2 or delta.shape != raw_width.shape or delta.shape[0] != n_variables:
        raise ValueError("Checkpoint patch geometry does not match history channels")
    n_patches = delta.shape[1]
    base_size = 1.0 / n_patches
    centers = np.linspace(base_size / 2.0, 1.0 - base_size / 2.0, n_patches)
    left = centers[None, :] - base_size / 2.0 + delta
    right = left + np.exp(raw_width) + 1e-6
    taus = np.logaddexp(0.0, tau_params) + 1e-6
    times = marks[:, :, 0]
    time_embedding = _learned_time_embedding(times, state)
    target_counts = (targets > 0).sum(axis=1)

    records: list[dict[str, Any]] = []
    for channel in range(n_variables):
        channel_times = times[:, None, :]
        weights_raw = expit(
            (right[channel][None, :, None] - channel_times) / taus[channel]
        ) * expit(
            (channel_times - left[channel][None, :, None]) / taus[channel]
        )
        temporal = weights_raw * masks[:, None, :, channel]
        mass = temporal.sum(axis=-1)
        squared_mass = np.square(temporal).sum(axis=-1)
        effective = np.divide(
            np.square(mass),
            squared_mass,
            out=np.zeros_like(mass),
            where=squared_mass > support_epsilon,
        )
        features = np.concatenate([values[:, :, channel, None], time_embedding], axis=-1)
        weighted = np.einsum("bpl,bld->bpd", temporal, features, optimize=True)
        centroids = np.divide(
            weighted,
            mass[..., None],
            out=np.zeros_like(weighted),
            where=mass[..., None] > support_epsilon,
        )
        best_patch = np.argmax(effective, axis=1)
        rows = np.arange(batch_size)
        best_support = effective[rows, best_patch]
        best_mass = mass[rows, best_patch]
        best_centroids = centroids[rows, best_patch]
        for row in range(batch_size):
            if (
                best_support[row] <= support_epsilon
                or target_counts[row, channel] <= 0
                or not np.isfinite(best_centroids[row]).all()
            ):
                continue
            record: dict[str, Any] = {
                "seed": int(seed),
                "patient_id": int(ids[row]),
                "channel": int(channel),
                "patch": int(best_patch[row]),
                "effective_support": float(best_support[row]),
                "soft_mass": float(best_mass[row]),
                "target_observed_count": int(target_counts[row, channel]),
            }
            for index, value in enumerate(best_centroids[row]):
                record[f"descriptor_{index}"] = float(value)
            records.append(record)
    return pd.DataFrame.from_records(records)


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, standard)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (values - median) / scale


def match_support_pairs(
    units: pd.DataFrame,
    seed: int,
    *,
    max_centroid_rms: float,
    min_effective_support_ratio: float,
    min_effective_support_difference: float,
) -> pd.DataFrame:
    """Greedily freeze disjoint nearest pairs without reading any model errors."""
    descriptor_columns = sorted(
        [column for column in units.columns if column.startswith("descriptor_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not descriptor_columns:
        raise ValueError("No descriptor columns were provided")
    if max_centroid_rms <= 0 or min_effective_support_ratio <= 1:
        raise ValueError("Controlled-support thresholds are invalid")
    if min_effective_support_difference <= 0:
        raise ValueError("Support difference must be positive")

    output: list[dict[str, Any]] = []
    pair_number = 0
    for channel in sorted(units["channel"].unique()):
        group = (
            units[units["channel"] == channel]
            .sort_values(["patient_id", "patch"])
            .reset_index(drop=True)
        )
        if len(group) < 2:
            continue
        descriptors = group[descriptor_columns].to_numpy(np.float64)
        standardized = _robust_standardize(descriptors)
        supports = group["effective_support"].to_numpy(np.float64)
        candidates: list[tuple[float, int, int]] = []
        for low_index, low_support in enumerate(supports):
            required = max(
                low_support * min_effective_support_ratio,
                low_support + min_effective_support_difference,
            )
            valid = np.flatnonzero(supports >= required)
            if valid.size == 0:
                continue
            deltas = standardized[valid] - standardized[low_index]
            distances = np.sqrt(np.mean(np.square(deltas), axis=1))
            best_position = int(np.argmin(distances))
            distance = float(distances[best_position])
            high_index = int(valid[best_position])
            if distance <= max_centroid_rms:
                candidates.append((distance, low_index, high_index))
        candidates.sort(
            key=lambda item: (
                item[0],
                int(group.iloc[item[1]]["patient_id"]),
                int(group.iloc[item[2]]["patient_id"]),
            )
        )
        used: set[int] = set()
        for distance, low_index, high_index in candidates:
            if low_index in used or high_index in used:
                continue
            low = group.iloc[low_index]
            high = group.iloc[high_index]
            used.update((low_index, high_index))
            pair_id = f"{int(seed)}-{int(channel):02d}-{pair_number:05d}"
            pair_number += 1
            low_support = float(low["effective_support"])
            high_support = float(high["effective_support"])
            output.append(
                {
                    "seed": int(seed),
                    "pair_id": pair_id,
                    "channel": int(channel),
                    "low_patient_id": int(low["patient_id"]),
                    "high_patient_id": int(high["patient_id"]),
                    "low_patch": int(low["patch"]),
                    "high_patch": int(high["patch"]),
                    "centroid_rms": distance,
                    "low_effective_support": low_support,
                    "high_effective_support": high_support,
                    "support_ratio": high_support / low_support,
                    "support_difference": high_support - low_support,
                }
            )
    return pd.DataFrame.from_records(output, columns=PAIR_COLUMNS)


def _load_native_evaluation(path: Path) -> dict[str, np.ndarray]:
    files = {
        "x": "input_x.npy",
        "x_mark": "input_x_mark.npy",
        "x_mask": "input_x_mask.npy",
        "target": "input_y.npy",
        "target_mask": "input_y_mask.npy",
        "sample_ids": "input_sample_ID.npy",
        "pred": "output_pred.npy",
    }
    missing = [name for name in files.values() if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Native evaluation {path} is missing {missing}")
    return {
        name: np.load(path / filename, allow_pickle=False)
        for name, filename in files.items()
    }


def score_frozen_pairs(
    pairs: pd.DataFrame,
    evaluations: Mapping[str, Mapping[str, np.ndarray]],
    seed: int,
) -> pd.DataFrame:
    """Score identical frozen pair IDs for every variant using channel target MSE."""
    records: list[dict[str, Any]] = []
    for variant, arrays in evaluations.items():
        ids = np.rint(np.asarray(arrays["sample_ids"]).reshape(-1)).astype(np.int64)
        if np.unique(ids).size != ids.size:
            raise ValueError(f"{variant}/{seed} contains duplicate sample IDs")
        index_by_id = {int(patient): index for index, patient in enumerate(ids)}
        pred = np.asarray(arrays["pred"], dtype=np.float64)
        target = np.asarray(arrays["target"], dtype=np.float64)
        mask = np.asarray(arrays["target_mask"], dtype=np.float64)
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError(f"{variant}/{seed} prediction/target shapes do not match")
        for pair in pairs.itertuples(index=False):
            patient_ids = (int(pair.low_patient_id), int(pair.high_patient_id))
            if any(patient not in index_by_id for patient in patient_ids):
                raise KeyError(f"{variant}/{seed} is missing a frozen pair patient")
            indices = [index_by_id[patient] for patient in patient_ids]
            channel = int(pair.channel)
            valid = mask[indices, :, channel] > 0
            count = int(valid.sum())
            if count == 0:
                raise ValueError(f"Frozen pair {pair.pair_id} has no target observations")
            residual = pred[indices, :, channel] - target[indices, :, channel]
            mse = float(np.square(residual)[valid].mean())
            records.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "pair_id": pair.pair_id,
                    "channel": channel,
                    "low_patient_id": patient_ids[0],
                    "high_patient_id": patient_ids[1],
                    "MSE": mse,
                    "observed_target_count": count,
                }
            )
    return pd.DataFrame.from_records(records, columns=ERROR_COLUMNS)


def _one_native_path(results_root: Path, variant: str, seed: int) -> Path:
    matches = list(
        (results_root / "stage_a" / variant / str(seed)).glob(
            "checkpoints/**/eval_native/metric.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one native evaluation for {variant}/{seed}, got {len(matches)}")
    return assert_within_project(matches[0].parent)


def _one_checkpoint(results_root: Path, variant: str, seed: int) -> Path:
    matches = list(
        (results_root / "stage_a" / variant / str(seed)).glob(
            "checkpoints/**/pytorch_model.bin"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {variant}/{seed}, got {len(matches)}")
    return assert_within_project(matches[0])


def run_controlled_support(config: dict[str, Any]) -> dict[str, Any]:
    """Construct error-blind APN pairs and score all Stage A variants."""
    settings = config["controlled_support"]
    results_root = assert_within_project(config["project"]["results_root"])
    artifacts = ensure_project_dir(config["project"]["artifacts_root"])
    source_variant = settings["source_variant"]
    all_units: list[pd.DataFrame] = []
    all_pairs: list[pd.DataFrame] = []
    all_errors: list[pd.DataFrame] = []
    seed_audits: list[dict[str, Any]] = []

    for seed in config["stage_a"]["seeds"]:
        source_path = _one_native_path(results_root, source_variant, seed)
        source_arrays = _load_native_evaluation(source_path)
        checkpoint = _one_checkpoint(results_root, source_variant, seed)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        units = patch_units(
            source_arrays["x"],
            source_arrays["x_mark"],
            source_arrays["x_mask"],
            source_arrays["target_mask"],
            source_arrays["sample_ids"],
            state,
            seed,
            support_epsilon=settings["support_epsilon"],
        )
        pairs = match_support_pairs(
            units,
            seed,
            max_centroid_rms=settings["max_centroid_rms"],
            min_effective_support_ratio=settings["min_effective_support_ratio"],
            min_effective_support_difference=settings[
                "min_effective_support_difference"
            ],
        )
        evaluations = {
            variant: _load_native_evaluation(
                _one_native_path(results_root, variant, seed)
            )
            for variant in config["stage_a"]["variants"]
        }
        errors = score_frozen_pairs(pairs, evaluations, seed)
        pair_count = int(len(pairs))
        minimum = int(settings["minimum_pairs_per_seed"])
        seed_audits.append(
            {
                "seed": int(seed),
                "source_variant": source_variant,
                "source_evaluation": str(source_path),
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": _sha256(checkpoint),
                "eligible_units": int(len(units)),
                "matched_pairs": pair_count,
                "minimum_pairs": minimum,
                "yield_passed": pair_count >= minimum,
            }
        )
        all_units.append(units)
        all_pairs.append(pairs)
        all_errors.append(errors)

    units_frame = pd.concat(all_units, ignore_index=True) if all_units else pd.DataFrame()
    pairs_frame = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=PAIR_COLUMNS)
    errors_frame = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame(columns=ERROR_COLUMNS)
    units_frame.to_csv(artifacts / "controlled_support_units.csv", index=False)
    pairs_frame.to_csv(artifacts / "controlled_support_pairs.csv", index=False)
    errors_frame.to_csv(artifacts / "controlled_support_errors.csv", index=False)
    audit = {
        "schema_version": 1,
        "pairing_is_error_blind": True,
        "settings": settings,
        "seeds": seed_audits,
        "all_seeds_meet_minimum": all(item["yield_passed"] for item in seed_audits),
        "total_pairs": int(len(pairs_frame)),
        "total_error_rows": int(len(errors_frame)),
    }
    (artifacts / "controlled_support_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit
