from dataclasses import replace

import numpy as np
import pytest
import torch

from edgetwincal.safe import (
    AblationSemantics,
    SafePolicy,
    apply_bounded_combined_correction,
    apply_safe_selection,
    deterministic_validation_group_split,
    evaluate_dataset_safety_gate,
    select_safe_candidate,
)


def policy(**changes):
    base = SafePolicy(
        selection_min_groups=4,
        selection_min_observations=4,
        dataset_min_groups=8,
        validation_bootstrap_resamples=400,
        dataset_bootstrap_resamples=500,
        max_positive_gain_concentration=1.0,
    )
    return replace(base, **changes)


def test_production_policy_uses_frozen_sample_floors():
    frozen = SafePolicy()
    assert (frozen.selection_min_groups, frozen.selection_min_observations) == (20, 400)
    assert (frozen.dataset_min_groups, frozen.dataset_min_observations) == (20, 400)


def test_combined_cap_then_shrink_is_strict():
    base = torch.tensor([0.0, 0.0], dtype=torch.float64)
    result = apply_bounded_combined_correction(
        base, [torch.tensor([9.0, -9.0]), torch.tensor([4.0, -4.0])],
        residual_scale=torch.tensor([2.0, 4.0]), cap_multiplier=0.5, shrink=0.25,
    )
    assert torch.equal(result, torch.tensor([0.25, -0.5], dtype=torch.float64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shrink": 0.0},
        {"correction_enabled": False},
        {"fallback_to_base": True},
    ],
)
def test_disabled_paths_are_bitwise_apn_clones(kwargs):
    base = torch.tensor([1.125, -3.75], dtype=torch.float64)
    output = apply_bounded_combined_correction(
        base, torch.ones_like(base), residual_scale=1.0,
        cap_multiplier=1.0, shrink=kwargs.pop("shrink", 1.0), **kwargs,
    )
    assert output.data_ptr() != base.data_ptr()
    assert output.numpy().tobytes() == base.numpy().tobytes()


def test_envelope_disabled_retains_raw_combined_correction():
    base = torch.tensor([1.0, 2.0])
    output = apply_bounded_combined_correction(
        base, [torch.tensor([5.0, -4.0]), torch.tensor([2.0, 1.0])],
        residual_scale=0.01, cap_multiplier=0.25, shrink=0.25,
        envelope_enabled=False,
    )
    assert torch.equal(output, torch.tensor([8.0, -1.0]))


def test_group_split_is_disjoint_balanced_and_order_invariant():
    groups = ["g5", "g1", "g4", "g2", "g3", "g0"]
    left = deterministic_validation_group_split(groups, salt="frozen")
    right = deterministic_validation_group_split(list(reversed(groups)), salt="frozen")
    assert left == right
    assert set(left.selection_groups).isdisjoint(left.safety_groups)
    assert set(left.selection_groups) | set(left.safety_groups) == set(groups)
    assert abs(len(left.selection_groups) - len(left.safety_groups)) <= 1


def candidate_inputs(groups=8):
    base = torch.zeros(groups, 1, dtype=torch.float64)
    target = torch.ones_like(base)
    mask = torch.ones_like(base, dtype=torch.bool)
    return base, target, mask, [f"g{i}" for i in range(groups)]


def test_candidate_tie_order_prefers_smallest_shrink_then_cap():
    base, target, mask, groups = candidate_inputs()
    decision = select_safe_candidate(
        base, torch.zeros_like(base), base.clone(), target, mask, groups,
        residual_scale=1.0, policy=policy(),
    )
    assert not decision.fallback_to_apn
    assert decision.selected is not None
    assert decision.selected.shrink == 0.25
    assert decision.selected.cap_multiplier == 0.25
    output = apply_safe_selection(
        base, torch.zeros_like(base), decision, residual_scale=1.0
    )
    assert torch.equal(output, base)


def test_insufficient_groups_and_failed_confidence_fall_back_bitwise():
    base, target, mask, groups = candidate_inputs(3)
    insufficient = select_safe_candidate(
        base, torch.ones_like(base), base, target, mask, groups,
        residual_scale=1.0, policy=policy(),
    )
    assert insufficient.fallback_to_apn
    base, target, mask, groups = candidate_inputs(10)
    target[-1] = -2.0
    uncertain = select_safe_candidate(
        base, torch.ones_like(base), base, target, mask, groups,
        residual_scale=1.0,
        policy=policy(validation_relative_loss_margin=0.0),
        semantics=AblationSemantics(leave_one_group_out=False),
    )
    assert uncertain.fallback_to_apn
    assert uncertain.reason in {"validation_confidence_failed", "no_feasible_candidate"}
    output = apply_safe_selection(
        base, torch.ones_like(base), uncertain, residual_scale=1.0
    )
    assert output.numpy().tobytes() == base.numpy().tobytes()


def test_no_envelope_selection_keeps_raw_not_apn():
    base, target, mask, groups = candidate_inputs()
    semantics = AblationSemantics.no_safety_envelope()
    decision = select_safe_candidate(
        base, torch.full_like(base, 2.0), base, target, mask, groups,
        residual_scale=0.01, policy=policy(), semantics=semantics,
    )
    assert decision.raw_retention and not decision.fallback_to_apn
    output = apply_safe_selection(
        base, torch.full_like(base, 2.0), decision,
        residual_scale=0.01, semantics=semantics,
    )
    assert torch.equal(output, torch.full_like(base, 2.0))


def cells(safe_by_checkpoint, groups=24, include_original=True):
    rows = []
    for checkpoint, safe_mse in enumerate(safe_by_checkpoint):
        for group in range(groups):
            variants = {"APN": (1.0, 1.0), "Joint": (0.97, 0.985), "Safe": (safe_mse, np.sqrt(safe_mse))}
            if include_original:
                variants["Original"] = (1.04, 1.02)
            for variant, (mse, mae) in variants.items():
                rows.append(
                    {
                        "group_hash": f"g{group}", "checkpoint": f"s{checkpoint}",
                        "variant": variant, "sse": mse * 10,
                        "sae": mae * 10, "n": 10,
                    }
                )
    return rows


def test_stable_five_checkpoint_gate_passes_and_audits_original():
    decision = evaluate_dataset_safety_gate(
        cells([0.94, 0.945, 0.95, 0.955, 0.96]), dataset_id="P12",
        original_variant="Original", policy=policy(),
    )
    assert decision.passed
    assert decision.deploy_variant == "Safe"
    assert decision.improved_checkpoints == 5
    assert decision.original_variant == "Original"
    assert len(decision.audit_sha256) == 64
    assert decision.ledger_fields() == {
        "enabled": True,
        "checkpoints": 5,
        "validation_groups": 24,
        "validation_cells": 1200,
        "gate_sha256": decision.audit_sha256,
    }


def test_anomalous_checkpoint_and_insufficient_groups_fall_back():
    anomalous = evaluate_dataset_safety_gate(
        cells([0.90, 0.90, 0.90, 0.90, 1.05]), dataset_id="P12",
        policy=policy(),
    )
    assert not anomalous.passed
    assert anomalous.deploy_variant == "APN"
    assert "checkpoint_harm" in anomalous.reasons
    small = evaluate_dataset_safety_gate(
        cells([0.94] * 5, groups=3), dataset_id="P12", policy=policy()
    )
    assert not small.passed
    assert small.reasons == ("insufficient_groups",)
    sparse = cells([0.94] * 5)
    for row in sparse:
        row["n"] = 1
        row["sse"] /= 10
        row["sae"] /= 10
    too_few_cells = evaluate_dataset_safety_gate(
        sparse, dataset_id="P12", policy=policy(dataset_min_groups=20)
    )
    assert not too_few_cells.passed
    assert too_few_cells.reasons == ("insufficient_observations",)


def test_crossed_confidence_failure_falls_back():
    rows = cells([0.98] * 5)
    for row in rows:
        if row["variant"] == "Safe":
            group = int(row["group_hash"][1:])
            row["sse"] = (0.25 if group < 12 else 1.70) * row["n"]
            row["sae"] = np.sqrt(row["sse"] / row["n"]) * row["n"]
    decision = evaluate_dataset_safety_gate(
        rows, dataset_id="P12", policy=policy(dataset_bootstrap_resamples=1200)
    )
    assert not decision.passed
    assert "efficacy_ci" in decision.reasons


def test_ablation_semantics_have_distinct_auditable_meanings():
    full = AblationSemantics.full()
    no_fit = AblationSemantics.no_robust_fit()
    no_gate = AblationSemantics.no_safety_envelope()
    assert full.sha256 != no_fit.sha256 != no_gate.sha256
    assert not no_fit.robust_fit and not no_fit.group_equalization
    assert no_gate.robust_fit and not no_gate.safety_envelope



def test_group_statistics_use_group_mse_and_macro_leave_one_out():
    from edgetwincal.safe import GroupMetric, _loo_gain, positive_gain_concentration

    baseline = (
        GroupMetric("large", 1000.0, 0.0, 1000),
        GroupMetric("small", 10.0, 0.0, 1),
        GroupMetric("third", 1.0, 0.0, 1),
    )
    candidate = (
        GroupMetric("large", 900.0, 0.0, 1000),
        GroupMetric("small", 9.0, 0.0, 1),
        GroupMetric("third", 1.0, 0.0, 1),
    )
    assert positive_gain_concentration(baseline, candidate) == pytest.approx(
        1.0 / 1.1
    )
    assert _loo_gain(baseline, candidate) == pytest.approx(0.05)


def test_dataset_gate_uses_one_shared_crossed_draw_for_all_metrics():
    frozen = policy()
    decision = evaluate_dataset_safety_gate(
        cells([0.94, 0.945, 0.95, 0.955, 0.96]),
        dataset_id="P12",
        policy=frozen,
    )
    assert decision.passed
    assert {
        (row["reference"], row["metric"])
        for row in decision.bootstrap_audit
    } == {
        (reference, metric)
        for reference in ("APN", "Joint")
        for metric in ("micro_mse", "group_macro_mse", "micro_mae")
    }
    assert all(
        row["shared_group_checkpoint_multiplicities"]
        and row["random_seed"] == frozen.random_seed
        for row in decision.bootstrap_audit
    )


def test_group_mse_gain_concentration_is_rejected():
    rows = cells([0.98] * 5)
    for row in rows:
        if row["variant"] == "Safe":
            group = int(row["group_hash"][1:])
            mse = 0.50 if group == 0 else 1.0
            row["sse"] = mse * row["n"]
            row["sae"] = np.sqrt(mse) * row["n"]
    decision = evaluate_dataset_safety_gate(
        rows,
        dataset_id="P12",
        policy=policy(
            max_positive_gain_concentration=0.25,
            dataset_bootstrap_resamples=1000,
        ),
    )
    assert not decision.passed
    assert "group_mse_gain_concentration" in decision.reasons
