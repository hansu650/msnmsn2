"""Build the sealed EdgeTwinCal laboratory return.

Only immutable pre-test evidence, sealed run manifests, and train/validation
fit caches are consumed.  This module cannot construct a test loader and never
reads a test tensor cache.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .campaign import ProtocolLedger
from .campaign_pretest import load_fitted_registry, load_fitted_states
from .campaign_runner import _apply_variant, load_cache_manifest, resolved_config_from_ledger
from .paths import PROJECT_ROOT, require_within_root
from .runtime_v2 import read_tensor_cache
from .schema import atomic_write_json, sha256_file


CAMPAIGN = "edgetwincal_msn2026_v1"
ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / CAMPAIGN / "analysis"
PRETEST_DIR = PROJECT_ROOT / "artifacts" / CAMPAIGN / "pretest"
PROTOCOL_DIR = PROJECT_ROOT / "results" / CAMPAIGN / "protocol"
RUN_DIR = PROJECT_ROOT / "results" / CAMPAIGN / "runs"
CELLS = (
    ("HumanActivity", "release_parity", 20),
    ("P12", "release_parity", 20),
    ("P12", "strict_p12", 60),
    ("USHCN", "release_parity", 20),
    ("USHCN", "strict_ushcn", 60),
)
STRICT_CELLS = (("P12", "strict_p12"), ("USHCN", "strict_ushcn"))


class LabReportError(RuntimeError):
    """A sealed return artifact cannot be generated safely."""


def _load_json(path: str | Path) -> Mapping[str, Any]:
    source = require_within_root(path, must_exist=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabReportError(f"Cannot read JSON {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LabReportError(f"Expected a JSON object: {source}")
    return value


def _relative(path: str | Path) -> str:
    return require_within_root(path, must_exist=False).relative_to(PROJECT_ROOT).as_posix()


def _write_text(path: str | Path, text: str) -> Path:
    destination = require_within_root(path, must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    require_within_root(temporary, must_exist=False)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _hash_reference(path: str | Path) -> dict[str, Any]:
    source = require_within_root(path, must_exist=True)
    return {"path": _relative(source), "bytes": source.stat().st_size, "sha256": sha256_file(source)}


def _analysis(aggregate: Mapping[str, Any], dataset: str, protocol: str) -> Mapping[str, Any]:
    matches = [
        item for item in aggregate.get("analyses", ())
        if item.get("dataset") == dataset and item.get("protocol") == protocol
    ]
    if len(matches) != 1:
        raise LabReportError(f"Expected one analysis for {dataset}/{protocol}, found {len(matches)}")
    return matches[0]


def build_pretest_summary() -> dict[str, Any]:
    """Revalidate the five sealed cells and all 180 explicit run manifests."""

    rows: list[dict[str, Any]] = []
    total = 0
    referenced_checks = 0
    for dataset, protocol, expected in CELLS:
        cell = PROTOCOL_DIR / dataset / protocol
        evidence_path = PRETEST_DIR / dataset / protocol / "pretest_evidence.json"
        ledger_path = cell / "protocol_ledger.json"
        once_path = cell / "once_campaign_result.json"
        prepared_path = cell / "once_control_prepared.json"
        evidence = _load_json(evidence_path)
        ledger = _load_json(ledger_path)
        once = _load_json(once_path)
        prepared = _load_json(prepared_path)
        openings = ledger.get("test_openings")
        if ledger.get("status") != "sealed" or not isinstance(openings, Mapping) or len(openings) != 1:
            raise LabReportError(f"Expected one sealed test opening: {dataset}/{protocol}")
        opening = next(iter(openings.values()))
        if not opening.get("opened_at") or not opening.get("closed_at"):
            raise LabReportError(f"Test opening was not closed: {dataset}/{protocol}")
        events = ledger.get("events", ())
        for event_name in ("test_opened", "test_closed", "sealed"):
            if sum(row.get("event") == event_name for row in events) != 1:
                raise LabReportError(f"Unexpected {event_name} count: {dataset}/{protocol}")
        if once.get("status") != "complete" or once.get("token_persisted") is not False:
            raise LabReportError(f"Once-only result is invalid: {dataset}/{protocol}")
        if prepared.get("token_persisted") is not False or prepared.get("test_constructed") is not False:
            raise LabReportError(f"Prepared control crossed test boundary: {dataset}/{protocol}")

        gates = evidence.get("gates")
        if not isinstance(gates, Mapping) or set(gates) != {"G0", "G1"}:
            raise LabReportError(f"Invalid pre-test evidence: {dataset}/{protocol}")
        check_count = 0
        for gate in ("G0", "G1"):
            if not isinstance(gates[gate], Mapping) or not gates[gate]:
                raise LabReportError(f"Empty {gate} evidence: {dataset}/{protocol}")
            for name, reference in sorted(gates[gate].items()):
                actual = _hash_reference(reference["path"])
                if actual["sha256"] != reference["sha256"] or actual["bytes"] != reference["bytes"]:
                    raise LabReportError(f"Pre-test evidence drift: {dataset}/{protocol}/{gate}/{name}")
                check_count += 1
        referenced_checks += check_count

        evaluation = once.get("evaluation")
        manifests = evaluation.get("run_manifests") if isinstance(evaluation, Mapping) else None
        if not isinstance(manifests, list) or len(manifests) != expected:
            raise LabReportError(f"Expected {expected} run manifests: {dataset}/{protocol}")
        manifest_hashes = []
        for item in manifests:
            run = _load_json(item)
            if (
                run.get("status") != "complete"
                or run.get("dataset") != dataset
                or run.get("protocol") != protocol
            ):
                raise LabReportError(f"Invalid run manifest: {item}")
            manifest_hashes.append(_hash_reference(item)["sha256"])
        total += len(manifests)
        rows.append({
            "dataset": dataset,
            "protocol": protocol,
            "expected_run_manifests": expected,
            "complete_run_manifests": len(manifests),
            "ledger_status": "sealed",
            "test_opening_count": 1,
            "test_closed": True,
            "token_persisted": False,
            "G0": "PASS",
            "G1": "PASS",
            "evidence": _hash_reference(evidence_path),
            "ledger": _hash_reference(ledger_path),
            "once_result": _hash_reference(once_path),
            "check_reference_count": check_count,
            "run_manifest_digest": hashlib.sha256("\n".join(manifest_hashes).encode("ascii")).hexdigest(),
        })
    if total != 180:
        raise LabReportError(f"Expected 180 completed manifests, found {total}")
    return {
        "schema_version": "edgetwincal.pretest-terminal-summary.v1",
        "G0": {"status": "PASS"},
        "G1": {"status": "PASS"},
        "cell_count": len(rows),
        "expected_manifest_count": 180,
        "complete_manifest_count": total,
        "referenced_check_count": referenced_checks,
        "all_test_ledgers_sealed": True,
        "all_tokens_nonpersisted": True,
        "cells": rows,
    }


def build_gate_decision(aggregate: Mapping[str, Any], pretest: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the frozen statistical result into the non-overridable verdict."""

    audit = aggregate.get("input_audit", {})
    if (
        audit.get("expected_manifest_count") != 180
        or audit.get("complete_manifest_count") != 180
        or not audit.get("all_expected_complete_and_verified")
    ):
        raise LabReportError("Formal aggregate does not cover 180 verified manifests")
    release = [item for item in aggregate["analyses"] if item.get("protocol") == "release_parity"]
    required = math.ceil(0.75 * len(release))
    strong = sum(item["G3"]["classification"] == "strong" for item in release)
    release_audit = {
        "status": "FAIL" if strong < required else "PASS",
        "inference": "seed_descriptive_only",
        "runnable_dataset_count": len(release),
        "required_strong_count": required,
        "strong_count": strong,
        "classifications": {item["dataset"]: item["G3"]["classification"] for item in release},
        "reason": "Release group IDs are not reliable; no confirmatory group-level interval exists.",
    }
    gates = {
        "G0": {"status": pretest["G0"]["status"], "scope": "implementation_parity"},
        "G1": {"status": pretest["G1"]["status"], "scope": "protocol_and_provenance"},
        "G2": dict(aggregate["gates"]["G2"]),
        "G3_strict": dict(aggregate["gates"]["G3"]),
        "G3_release_scope": release_audit,
        "G4": dict(aggregate["gates"]["G4"]),
    }
    reasons = []
    for key, reason in (
        ("G2", "G2_MECHANISM_FAIL"),
        ("G3_strict", "G3_STRICT_GENERALIZATION_FAIL"),
        ("G3_release_scope", "G3_RELEASE_BROAD_CLAIM_FAIL"),
        ("G4", "G4_REAL_EDGE_BLOCKED"),
    ):
        if gates[key]["status"] != "PASS":
            reasons.append(reason)
    verdict = "PASS" if all(row["status"] == "PASS" for row in gates.values()) else "ABANDON"
    return {
        "schema_version": "edgetwincal.gate-decision.v1",
        "verdict": verdict,
        "gate_policy": "all declared gates must pass; blocked is not pass",
        "reasons": reasons,
        "gates": gates,
        "evidence": {
            "aggregate": _hash_reference(ANALYSIS_DIR / "confirmatory_aggregate.json"),
            "manifest_registry": _hash_reference(ANALYSIS_DIR / "manifest_registry.json"),
            "blockers": _hash_reference(ANALYSIS_DIR / "blockers.json"),
        },
        "claim_actions": {
            "broad_dataset_claim": "REJECT",
            "real_edge_claim": "BLOCKED_AND_NARROW",
            "P12_strict_result": "REPORT_DATASET_SPECIFIC_POSITIVE",
            "USHCN_strict_result": "REPORT_HARM",
            "paper_conclusion_rewrite": "NOT_AUTHORIZED_BY_CURRENT_HANDOFF",
            "same_test_retuning": "PROHIBITED",
            "current_APN_route": "STOP_AFTER_FIFTH_STRUCTURAL_ATTEMPT",
            "future_baseline": "SWITCH_BASELINE_AND_USE_A_NEW_INDEPENDENT_TARGET",
        },
    }


def _masked_stats(split: Mapping[str, torch.Tensor], prediction: torch.Tensor) -> dict[str, Any]:
    mask = split["mask"].bool()
    target = split["target"]
    base = split["base_prediction"]
    error = (prediction - target)[mask].double()
    residual = (base - target)[mask].double().abs()
    correction = (prediction - base)[mask].double().abs()
    if error.numel() == 0:
        raise LabReportError("Fit diagnostic encountered an empty split")

    def quantiles(values: torch.Tensor) -> dict[str, float]:
        result = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99], dtype=torch.double))
        return {
            "q50": float(result[0]),
            "q90": float(result[1]),
            "q99": float(result[2]),
            "max": float(values.max()),
        }

    return {
        "n": int(error.numel()),
        "sse": float(torch.sum(error.square())),
        "sae": float(torch.sum(error.abs())),
        "mse": float(torch.mean(error.square())),
        "mae": float(torch.mean(error.abs())),
        "base_residual_abs": quantiles(residual),
        "correction_abs": quantiles(correction),
    }


def _validation_group_dominance(
    split: Mapping[str, torch.Tensor],
    predictions: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    groups = split["group_id"].detach().cpu().tolist()
    mask = split["mask"].bool()
    target = split["target"]
    per_variant: dict[str, dict[int, float]] = {}
    for variant, prediction in predictions.items():
        squared = ((prediction - target).double().square() * mask).sum(dim=(1, 2))
        grouped: dict[int, float] = defaultdict(float)
        for group, value in zip(groups, squared.detach().cpu().tolist()):
            grouped[int(group)] += float(value)
        per_variant[variant] = dict(grouped)
    apn = per_variant["APN"]
    top_group = max(apn, key=apn.get)
    apn_total = sum(apn.values())
    result: dict[str, Any] = {
        "group_count": len(apn),
        "top_apn_sse_fraction": apn[top_group] / apn_total if apn_total else None,
    }
    for variant in ("SLRH", "Full"):
        gains = {group: apn[group] - per_variant[variant][group] for group in apn}
        total = sum(gains.values())
        gain_group = max(gains, key=gains.get)
        label = variant.lower()
        result[f"{label}_total_sse_gain"] = total
        result[f"top_{label}_gain_fraction"] = gains[gain_group] / total if total > 0 else None
        result[f"top_{label}_gain_is_top_apn_group"] = gain_group == top_group
    return result


def _fit_diagnosis(dataset: str, protocol: str) -> dict[str, Any]:
    cell = PROTOCOL_DIR / dataset / protocol
    ledger = ProtocolLedger.load(cell / "protocol_ledger.json")
    config = resolved_config_from_ledger(ledger)
    registry = load_fitted_registry(cell / "fitted_registry.json")
    entries = _load_json(cell / "fit_entries.json")["entries"]
    entry_by_seed = {int(row["seed"]): row for row in entries}
    registry_by_seed = {int(row["seed"]): row for row in registry["seeds"]}
    variants = ("APN", "SLRH", "CFG", "Full", "V11", "V12")
    totals: dict[str, dict[str, float]] = {
        split: {
            f"{variant}_{name}": 0.0
            for variant in variants
            for name in ("sse", "sae", "n")
        }
        for split in ("train", "val")
    }
    seed_rows = []
    for seed in sorted(entry_by_seed):
        entry = entry_by_seed[seed]
        manifest = load_cache_manifest(entry["fit_cache_manifest"])
        _, splits = read_tensor_cache(entry["fit_cache"], manifest)
        fitted = load_fitted_states(config, registry_by_seed[seed], device="cpu")
        seed_metrics: dict[str, Any] = {}
        validation_predictions: dict[str, torch.Tensor] = {}
        for split_name in ("train", "val"):
            split = splits[split_name]
            seed_metrics[split_name] = {}
            for variant in variants:
                prediction = _apply_variant(fitted, variant, split).detach().cpu()
                stats = _masked_stats(split, prediction)
                seed_metrics[split_name][variant] = stats
                for name in ("sse", "sae", "n"):
                    totals[split_name][f"{variant}_{name}"] += stats[name]
                if split_name == "val":
                    validation_predictions[variant] = prediction
        seed_rows.append({
            "seed": seed,
            "metrics": seed_metrics,
            "validation_group_dominance": _validation_group_dominance(
                splits["val"], validation_predictions
            ),
            "selected_hyperparameters": {
                row["variant_id"]: row["selected_hyperparameters"]
                for row in registry_by_seed[seed]["variants"]
            },
        })

    pooled: dict[str, Any] = {}
    for split_name in ("train", "val"):
        pooled[split_name] = {}
        for variant in variants:
            n = totals[split_name][f"{variant}_n"]
            pooled[split_name][variant] = {
                "n": int(n),
                "mse": totals[split_name][f"{variant}_sse"] / n,
                "mae": totals[split_name][f"{variant}_sae"] / n,
            }
        apn = pooled[split_name]["APN"]["mse"]
        for variant in variants:
            pooled[split_name][variant]["relative_mse_gain_vs_APN"] = (
                apn - pooled[split_name][variant]["mse"]
            ) / apn
    return {
        "dataset": dataset,
        "protocol": protocol,
        "fit_splits_only": ["train", "val"],
        "test_tensor_cache_read": False,
        "pooled": pooled,
        "seeds": seed_rows,
    }


def _sealed_test_harm_concentration() -> dict[str, Any]:
    """Use completed manifest cells only, never predictions or test caches."""

    rows = []
    for seed in range(2024, 2029):
        manifests = {
            variant: _load_json(
                RUN_DIR / "USHCN" / "strict_ushcn" / f"seed_{seed}" / variant / "run_manifest.json"
            )
            for variant in ("APN", "SLRH", "Full")
        }
        cells = {
            variant: {row["group_hash"]: float(row["sse"]) for row in manifest["cells"]}
            for variant, manifest in manifests.items()
        }
        apn = cells["APN"]
        result: dict[str, Any] = {"seed": seed, "group_count": len(apn)}
        for variant in ("SLRH", "Full"):
            harm = {group: cells[variant][group] - apn[group] for group in apn}
            positive_total = sum(max(value, 0.0) for value in harm.values())
            top = max(harm.values())
            label = variant.lower()
            result[f"{label}_positive_harm_sse"] = positive_total
            result[f"{label}_top_group_positive_harm_fraction"] = (
                top / positive_total if positive_total > 0 else None
            )
        rows.append(result)
    return {
        "source": "sealed_run_manifest_cells_only",
        "test_tensor_cache_read": False,
        "seeds": rows,
    }


def build_failure_diagnosis(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    cells = [_fit_diagnosis(dataset, protocol) for dataset, protocol in STRICT_CELLS]
    return {
        "schema_version": "edgetwincal.failure-diagnosis.v1",
        "verdict": "STRUCTURAL_ROUTE_UNSAFE",
        "not_an_implementation_replay_failure": True,
        "evidence_boundary": {
            "fit_cache_splits_read": ["train", "val"],
            "test_tensor_cache_read": False,
            "test_evidence_read": "sealed run-manifest error cells only",
        },
        "strict_fit_diagnostics": cells,
        "sealed_test_harm_concentration": _sealed_test_harm_concentration(),
        "interpretation": [
            "P12 shows a small, checkpoint-consistent strict gain.",
            "USHCN validation is tiny and heavy-tailed; micro-MSE selection is dominated by a few high-leverage groups.",
            "The unbounded residual correction looks strongly favorable on USHCN validation but reverses on sealed test.",
            "This is a selection-and-generalization failure, not checkpoint or state replay drift.",
        ],
        "next_action": {
            "same_APN_test_retuning": "PROHIBITED",
            "sixth_route_on_opened_tests": "DO_NOT_RUN",
            "baseline_policy": "SWITCH_BASELINE",
            "evaluation_policy": "USE_A_NEW_INDEPENDENT_TARGET",
        },
        "aggregate_sha256": sha256_file(ANALYSIS_DIR / "confirmatory_aggregate.json"),
    }


def _svg_bar_chart(
    rows: Sequence[tuple[str, Sequence[tuple[str, float]]]],
    *,
    title: str,
    y_label: str,
    width: int = 980,
    height: int = 560,
) -> str:
    left, right, top, bottom = 92, 40, 70, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [value for _, series in rows for _, value in series]
    low, high = min(0.0, min(values)), max(0.0, max(values))
    if high == low:
        high = low + 1.0
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_h

    palette = (
        "#2667FF", "#E45756", "#3A9D5D", "#F2A541", "#7B61FF", "#6C757D",
        "#17A2B8", "#D65DB1", "#845EC2", "#008F7A", "#C34A36",
    )
    group_w = plot_w / len(rows)
    max_series = max(len(series) for _, series in rows)
    bar_w = min(42.0, group_w / (max_series + 1))
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        yy = y(value)
        body.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#E8EBF0"/>')
        body.append(f'<text x="{left-10}" y="{yy+5:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#41464F">{value:.3f}</text>')
    body.append(f'<line x1="{left}" y1="{y(0):.2f}" x2="{width-right}" y2="{y(0):.2f}" stroke="#333" stroke-width="1.3"/>')
    for index, (label, series) in enumerate(rows):
        center = left + group_w * (index + 0.5)
        start = center - bar_w * len(series) / 2
        for offset, (_, value) in enumerate(series):
            x = start + offset * bar_w
            bar_top = min(y(value), y(0))
            bar_h = max(1.0, abs(y(value) - y(0)))
            body.append(f'<rect x="{x+3:.2f}" y="{bar_top:.2f}" width="{bar_w-6:.2f}" height="{bar_h:.2f}" fill="{palette[offset % len(palette)]}"/>')
        body.append(f'<text x="{center:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="600">{html.escape(label)}</text>')
    legend = []
    for _, series in rows:
        for name, _ in series:
            if name not in legend:
                legend.append(name)
    legend_y = height - 36
    for index, name in enumerate(legend):
        x = left + index * ((width - left - right) / max(1, len(legend)))
        body.append(f'<rect x="{x:.2f}" y="{legend_y}" width="12" height="12" fill="{palette[index % len(palette)]}"/>')
        body.append(f'<text x="{x+16:.2f}" y="{legend_y+11}" font-family="Arial" font-size="10">{html.escape(name)}</text>')
    body.append(f'<text transform="translate(22 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(y_label)}</text>')
    body.append("</svg>")
    return "\n".join(body) + "\n"


def _write_figures(aggregate: Mapping[str, Any]) -> list[Path]:
    strict = [_analysis(aggregate, *cell) for cell in STRICT_CELLS]
    main_rows = [
        (
            item["dataset"],
            [
                ("APN", float(item["variant_metrics"]["APN"]["mse"])),
                ("Full", float(item["variant_metrics"]["Full"]["mse"])),
            ],
        )
        for item in strict
    ]
    variants = ("SLRH", "CFG", "Full", "V01", "V02", "V03", "V07", "V08", "V10", "V11", "V12")
    gain_rows = []
    for item in strict:
        apn = float(item["variant_metrics"]["APN"]["mse"])
        gain_rows.append((
            item["dataset"],
            [
                (variant, 100.0 * (apn - float(item["variant_metrics"][variant]["mse"])) / apn)
                for variant in variants
            ],
        ))
    figure_dir = ANALYSIS_DIR / "figures"
    main = _write_text(
        figure_dir / "strict_main_mse.svg",
        _svg_bar_chart(main_rows, title="Strict protocols: APN vs Full", y_label="Masked micro MSE"),
    )
    ablation = _write_text(
        figure_dir / "strict_ablation_gain.svg",
        _svg_bar_chart(
            gain_rows,
            title="Strict ablations: relative MSE gain versus APN",
            y_label="Relative MSE gain (%)",
            width=1320,
            height=620,
        ),
    )
    return [main, ablation]


def _pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _report_markdown(
    aggregate: Mapping[str, Any],
    gate: Mapping[str, Any],
    failure: Mapping[str, Any],
    pretest: Mapping[str, Any],
) -> str:
    """Render the audited Chinese lab return from frozen numeric artifacts."""

    p12 = _analysis(aggregate, "P12", "strict_p12")
    ushcn = _analysis(aggregate, "USHCN", "strict_ushcn")
    release = [
        _analysis(aggregate, dataset, "release_parity")
        for dataset in ("P12", "HumanActivity", "USHCN")
    ]
    p12_cmp = p12["comparisons"]["APN"]["metrics"]
    ushcn_cmp = ushcn["comparisons"]["APN"]["metrics"]
    lines = [
        "# EdgeTwinCal 实验室回报（封存确认性实验）",
        "",
        f"**最终结论：{gate['verdict']}。** 本回报只审计实验，不重写论文结论，也不依据已打开的测试集继续调参。",
        "",
        "## 1. 完整性与边界",
        "",
        f"- 五个可运行实验单元均通过 G0/G1，完成并验证 {pretest['complete_manifest_count']}/{pretest['expected_manifest_count']} 个显式 run manifest。",
        "- 每个单元恰好一次 test opening，均已关闭并 seal；token 未持久化。",
        "- 统计使用 group × checkpoint crossed paired bootstrap（50,000 draws，seed 20260721）。",
        "- 95% CI 是未校正 percentile CI；Holm 校正施加在 one-sided bootstrap p 值上，不能称为 Holm-adjusted CI。",
        "",
        "## 2. 严格协议主结果",
        "",
        "| 数据集 | APN MSE | Full MSE | 相对改善 | Full−APN MSE 95% CI | MAE 相对改善 95% CI | 配对 checkpoint | 分类 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| P12 | {p12['variant_metrics']['APN']['mse']:.6f} | "
            f"{p12['variant_metrics']['Full']['mse']:.6f} | {_pct(p12['G3']['relative_mse_gain'])} | "
            f"[{p12_cmp['mse']['effect_ci_low']:.6f}, {p12_cmp['mse']['effect_ci_high']:.6f}] | "
            f"[{_pct(p12_cmp['mae']['relative_gain_ci_low'])}, {_pct(p12_cmp['mae']['relative_gain_ci_high'])}] | "
            f"{p12['G3']['improved_checkpoints']}/5 | {p12['G3']['classification']} |"
        ),
        (
            f"| USHCN | {ushcn['variant_metrics']['APN']['mse']:.6f} | "
            f"{ushcn['variant_metrics']['Full']['mse']:.6f} | {_pct(ushcn['G3']['relative_mse_gain'])} | "
            f"[{ushcn_cmp['mse']['effect_ci_low']:.6f}, {ushcn_cmp['mse']['effect_ci_high']:.6f}] | "
            f"[{_pct(ushcn_cmp['mae']['relative_gain_ci_low'])}, {_pct(ushcn_cmp['mae']['relative_gain_ci_high'])}] | "
            f"{ushcn['G3']['improved_checkpoints']}/5 | {ushcn['G3']['classification']} |"
        ),
        "",
        "P12 的 0.822% 改善在五个 checkpoint 上方向一致，primary Holm-adjusted p=0.005840；这是数据集特定的正结果。USHCN 的 Full MSE 增加 145.48%，且 MAE 增幅区间下界为 0.338%，越过预声明 0.2% harm margin，因此分类为 harmful。",
        "",
        "![严格协议主结果](figures/strict_main_mse.svg)",
        "",
        "![严格协议消融](figures/strict_ablation_gain.svg)",
        "",
        "## 3. 机制与范围门控",
        "",
        "- G2：FAIL。P12 的简单控制和两种 shuffle 通过，但 V08 Joint 非劣性上界 0.4839% 超过 0.1% margin；反向顺序近似持平，不能声称顺序优势。",
        "- USHCN 的简单控制、Joint、两种 shuffle 均不支持预期机制；V07 反向顺序更好，V10 缺少方差诊断。",
        "- G3 strict：FAIL。要求两个严格数据集均 strong，实际为 1 strong + 1 harmful。",
        "- G3 release broad-scope：FAIL。三个 release 数据集只有 checkpoint-level 描述统计，0/3 strong。",
        "- G4：BLOCKED。没有真实 edge CPU/Jetson 测量；RTX 4090 不能替代 edge target。",
        "",
        "## 4. Release 描述性结果",
        "",
        "| 数据集 | APN MSE | Full MSE | 相对改善 | 标签 |",
        "|---|---:|---:|---:|---|",
    ]
    for item in release:
        lines.append(
            f"| {item['dataset']} | {item['variant_metrics']['APN']['mse']:.6f} | "
            f"{item['variant_metrics']['Full']['mse']:.6f} | {_pct(item['G3']['relative_mse_gain'])} | "
            f"{item['G3']['classification']} |"
        )
    ushcn_fit = next(item for item in failure["strict_fit_diagnostics"] if item["dataset"] == "USHCN")
    p12_fit = next(item for item in failure["strict_fit_diagnostics"] if item["dataset"] == "P12")
    lines.extend([
        "",
        "这些 release split 的 group IDs 不可靠，不能把五个 checkpoint 当作独立患者或站点做确认性推断。",
        "",
        "## 5. 失败根因",
        "",
        f"- 严格 P12 validation：APN MSE={p12_fit['pooled']['val']['APN']['mse']:.6f}，Full={p12_fit['pooled']['val']['Full']['mse']:.6f}，表观改善 {_pct(p12_fit['pooled']['val']['Full']['relative_mse_gain_vs_APN'])}。",
        f"- 严格 USHCN validation：APN MSE={ushcn_fit['pooled']['val']['APN']['mse']:.6f}，Full={ushcn_fit['pooled']['val']['Full']['mse']:.6f}，表观改善 {_pct(ushcn_fit['pooled']['val']['Full']['relative_mse_gain_vs_APN'])}；每个 seed 的 validation 有效目标很少且重尾。",
        "- USHCN validation micro-MSE 被少数高杠杆 group 支配，因而偏好幅度不受限的 residual correction；该方向在封存 test 上翻转。",
        "- checkpoint、fit-cache、fitted-state、manifest 哈希和零对角约束均通过审计，故不是状态迁移或实现重放错误。",
        "",
        "## 6. 决策与下一步",
        "",
        "- 按预声明 gate，当前路线判定 ABANDON；不得把 P12 的局部正结果包装为普遍收益。",
        "- 已经是 APN 上的第五个结构性尝试：停止同一 baseline 的第六次路线，不用这些 test 继续迭代。",
        "- 下一轮应切换 baseline，并使用全新的独立 target；可预注册 train-only robust/guarded adapter，但不能回写为本轮补救结果。",
        "- MIMIC-III 缺 author mapping、HumanActivity participant IDs 不可靠、真实 edge target 不可用，继续保持 blocker，不伪造结果。",
        "",
        "## 7. 可追溯文件",
        "",
        "- confirmatory_aggregate.json：正式统计与 crossed bootstrap。",
        "- pretest_terminal_summary.json：G0/G1、once-only 和 180/180 完整性复核。",
        "- gate_decision.json：机器可读 gate 与 ABANDON。",
        "- failure_diagnosis.json：只读取 train/val fit cache 与 sealed manifest cells 的诊断。",
        "- EdgeTwinCal_lab_results.xlsx 及同目录 CSV：表格化结果和 provenance。",
    ])
    return "\n".join(lines) + "\n"


def render_lab_return() -> dict[str, Any]:
    aggregate_path = ANALYSIS_DIR / "confirmatory_aggregate.json"
    aggregate = _load_json(aggregate_path)
    pretest = build_pretest_summary()
    atomic_write_json(ANALYSIS_DIR / "pretest_terminal_summary.json", pretest)
    gate = build_gate_decision(aggregate, pretest)
    atomic_write_json(ANALYSIS_DIR / "gate_decision.json", gate)
    failure = build_failure_diagnosis(aggregate)
    atomic_write_json(ANALYSIS_DIR / "failure_diagnosis.json", failure)
    figures = _write_figures(aggregate)
    report = _write_text(
        ANALYSIS_DIR / "REPORT_CN.md",
        _report_markdown(aggregate, gate, failure, pretest),
    )
    sources = [
        aggregate_path,
        ANALYSIS_DIR / "manifest_registry.json",
        ANALYSIS_DIR / "blockers.json",
        ANALYSIS_DIR / "pretest_terminal_summary.json",
        ANALYSIS_DIR / "gate_decision.json",
        ANALYSIS_DIR / "failure_diagnosis.json",
        report,
        *figures,
    ]
    for name in (
        "dataset_variant_summary.csv",
        "seed_summary.csv",
        "paired_comparisons.csv",
        "gate_summary.csv",
        "EdgeTwinCal_lab_results.xlsx",
    ):
        table_path = ANALYSIS_DIR / name
        if table_path.is_file():
            sources.append(table_path)
    provenance = {
        "schema_version": "edgetwincal.analysis-provenance.v1",
        "campaign": CAMPAIGN,
        "test_reopened": False,
        "test_tensor_cache_read": False,
        "paper_conclusion_rewritten": False,
        "verdict": gate["verdict"],
        "files": [_hash_reference(path) for path in sources],
    }
    atomic_write_json(ANALYSIS_DIR / "analysis_provenance.json", provenance)
    return provenance


__all__ = [
    "LabReportError",
    "build_failure_diagnosis",
    "build_gate_decision",
    "build_pretest_summary",
    "render_lab_return",
]
