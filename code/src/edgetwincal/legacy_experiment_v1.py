from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from .legacy_runtime_v1 import (
    file_sha256,
    find_assets,
    load_or_create_cache,
    masked_metrics,
    relative_improvement,
    set_determinism,
)
from .paths import PROJECT_ROOT, ensure_directory, require_within_root
from .legacy_graph_v1 import fit_graph_with_validation

from .latent import fit_latent_head_with_validation


def run_experiment(args: argparse.Namespace) -> dict:
    seed = int(args.seed)
    set_determinism(seed)
    started = time.perf_counter()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    assets = find_assets(seed)
    cache_root = ensure_directory(PROJECT_ROOT / "results" / "edgetwincal")
    cache, cache_path = load_or_create_cache(
        assets, cache_root, device, args.rebuild_cache
    )
    train, val, test = cache["train"], cache["val"], cache["test"]

    latent_head, latent_audit = fit_latent_head_with_validation(
        train["base_prediction"], train["features"], train["target"], train["mask"],
        val["base_prediction"], val["features"], val["target"], val["mask"],
        alphas=args.latent_alphas,
    )
    latent_train = latent_head.apply(train["base_prediction"], train["features"])
    latent_val = latent_head.apply(val["base_prediction"], val["features"])

    cfg_only, cfg_audit = fit_graph_with_validation(
        train["base_prediction"], train["target"], train["mask"],
        val["base_prediction"], val["target"], val["mask"],
        alphas=args.graph_alphas,
    )
    full_cfg, full_audit = fit_graph_with_validation(
        latent_train, train["target"], train["mask"],
        latent_val, val["target"], val["mask"],
        alphas=args.graph_alphas,
    )
    validation_predictions = {
        "apn": val["base_prediction"],
        "slrh": latent_val,
        "cfg": cfg_only.apply(val["base_prediction"]),
        "full": full_cfg.apply(latent_val),
    }
    validation_metrics = {
        name: masked_metrics(value, val["target"], val["mask"])
        for name, value in validation_predictions.items()
    }

    latent_test = latent_head.apply(test["base_prediction"], test["features"])
    test_predictions = {
        "apn": test["base_prediction"],
        "slrh": latent_test,
        "cfg": cfg_only.apply(test["base_prediction"]),
        "full": full_cfg.apply(latent_test),
    }
    test_metrics = {
        name: masked_metrics(value, test["target"], test["mask"])
        for name, value in test_predictions.items()
    }
    baseline = test_metrics["apn"]
    improvements = {
        name: relative_improvement(baseline, value)
        for name, value in test_metrics.items()
        if name != "apn"
    }
    passed = (
        improvements["full"]["mse"] >= args.pass_threshold
        and test_metrics["full"].mae <= baseline.mae
        and test_metrics["full"].mse <= min(test_metrics["slrh"].mse, test_metrics["cfg"].mse)
    )

    output_root = ensure_directory(PROJECT_ROOT / "results" / "edgetwincal")
    run_dir = ensure_directory(output_root / f"seed_{seed}")
    state_path = require_within_root(run_dir / "closed_form_modules.pt")
    arrays_path = require_within_root(run_dir / "test_outputs.npz")
    torch.save(
        {
            "slrh": latent_head.state_dict(),
            "cfg_only": cfg_only.state_dict(),
            "full_cfg": full_cfg.state_dict(),
        },
        state_path,
    )
    np.savez_compressed(
        arrays_path,
        **{f"prediction_{name}": value.numpy() for name, value in test_predictions.items()},
        target=test["target"].numpy(),
        target_mask=test["mask"].numpy(),
        sample_id=test["sample_id"].numpy(),
    )
    result = {
        "schema_version": 1,
        "attempt": 5,
        "track": "Edge Computing, IoT and Digital Twins",
        "title": "EdgeTwinCal: Dual-Space Calibration for Frozen Irregular-Sensor Digital Twins",
        "baseline": "APN (AAAI 2026), pre-existing checkpoint trained locally with the released implementation, frozen",
        "seed": seed,
        "device": str(device),
        "modules": {
            "slrh": "Sensor Latent Residual Head on the frozen decoder path",
            "cfg": "Cross-Forecast Graph after decoding",
        },
        "validation_metrics": {name: asdict(value) for name, value in validation_metrics.items()},
        "test_metrics": {name: asdict(value) for name, value in test_metrics.items()},
        "relative_improvement_vs_apn": improvements,
        "pass_threshold": args.pass_threshold,
        "passed": passed,
        "fitted_coefficients": {
            "slrh": int(latent_head.weights.numel() + latent_head.intercept.numel()),
            "cfg": int(
                full_cfg.weights.numel()
                - full_cfg.weights.shape[0] * full_cfg.weights.shape[1]
                + full_cfg.intercept.numel()
            ),
        },
        "nonzero_coefficients": {
            "slrh": int(torch.count_nonzero(latent_head.weights) + torch.count_nonzero(latent_head.intercept)),
            "cfg": int(torch.count_nonzero(full_cfg.weights) + torch.count_nonzero(full_cfg.intercept)),
        },
        "fitting": {
            "latent_validation": latent_audit,
            "cfg_validation": cfg_audit,
            "full_cfg_validation": full_audit,
            "selected_alpha_slrh": latent_head.alpha,
            "selected_alpha_cfg": cfg_only.alpha,
            "selected_alpha_full_cfg": full_cfg.alpha,
            "total_wall_seconds": time.perf_counter() - started,
        },
        "assets": {
            "checkpoint": str(assets.checkpoint),
            "checkpoint_sha256": file_sha256(assets.checkpoint),
            "cache": str(cache_path),
            "module_state": str(state_path),
            "test_outputs": str(arrays_path),
        },
    }
    metrics_path = require_within_root(run_dir / "metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({
        "seed": seed,
        "test_metrics": result["test_metrics"],
        "relative_improvement_vs_apn": improvements,
        "passed": passed,
        "metrics_path": str(metrics_path),
    }, indent=2, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EdgeTwinCal attempt 5")
    parser.add_argument("--seed", type=int, default=2024, choices=[2024, 2025, 2026])
    parser.add_argument("--latent-alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--graph-alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--pass-threshold", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
