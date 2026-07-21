<!--
Modified: 2026-07-21
Changes: Updated the active EdgeTwinCal route with the formal ablation rerun,
checkpoint provenance, and exploratory evidence boundary.
-->

## Current Active Route: EdgeTwinCal

> Status: exploratory three-seed main experiment and mechanism ablation complete.

**Target.** IEEE MSN 2026, track `Edge Computing, IoT and Digital Twins`.

**Question.** Can a frozen channel-independent irregular forecaster be improved
without backbone retraining by decomposing its residual into sensor-local latent
information and cross-sensor forecast information?

**Method.** EdgeTwinCal freezes pre-existing checkpoints trained locally with the
released APN implementation and fits two sequential closed-form modules. The
Sensor Latent Residual Head (SLRH) reads each sensor's
APN latent representation on the decoder path. The Cross-Forecast Graph (CFG)
then predicts the remaining residual from other sensors' intermediate forecasts
with a zero diagonal. Both ridge penalties are selected on validation only.

**Evidence.** On PhysioNet 2012 across seeds 2024/2025/2026, frozen APN obtains
MSE 0.312331 +/- 0.000512 and MAE 0.365334 +/- 0.000863. EdgeTwinCal obtains
MSE 0.309058 +/- 0.000494 and MAE 0.362978 +/- 0.000335, relative improvements
of 1.048% and 0.645%. SLRH and CFG alone improve MSE by 0.553% and 0.591%.
A descriptive patient-macro hierarchical paired bootstrap gives full-minus-APN
MSE -0.003793 with 95% CI [-0.005619, -0.002130].

**Scope.** The manuscript reports only the PhysioNet main experiment and
APN/SLRH/CFG/full ablation. Values for other methods are quoted from the APN
paper for context and are not presented as reproduced comparisons. No APN or
external baseline is retrained. Because five structural attempts inspected the
same PhysioNet test set, the final route is an exploratory pilot; confirmation
requires an untouched dataset or holdout. Every paired variant inherits the
released P12 preprocessing behavior.

**Research questions.** RQ1 asks whether frozen dual-space calibration improves
the paired APN checkpoints. RQ2 asks whether SLRH and CFG are complementary.
RQ3 asks how consistently the descriptive gain appears across seeds and patients.

---

# EviPatch Idea and Experiment Report
> Generated: 2026-07-20 | Status: CONFIRMED_FOR_STAGE_A
> Scope: idea refinement and experiment design only; no manuscript is generated.

---

## Part 1 Topic Overview

### 1 Motivation

APN transforms an irregular multivariate history into regular patch tokens using Time-Aware Patch Aggregation (TAPA). For each variable and learned soft window, it computes a normalized weighted average of the observed value and its learned time embedding. This design preserves the weighted centroid but discards the denominator after normalization. Consequently, histories with the same centroid but different evidence quantity can become indistinguishable to every downstream APN component.

The issue is narrower than claiming that APN loses all sampling-density information. Average time embeddings can retain some information about where observations occur, but they cannot in general recover total soft mass. Under an independent-noise latent-state model, the Bayes-optimal forecast depends on the number of observations even when the sample mean is identical, so the discarded quantity can be forecast-relevant.

> The research question is deliberately falsifiable: does preserving local evidence statistics add information beyond a simple count baseline on real irregular sensor data?

**Why this research is necessary:**

- **Application necessity**: IoT and clinical sensors exhibit sparse, bursty, and throttled observation processes. A predictor that cannot condition on evidence quantity cannot learn when the same apparent mean is weakly or strongly supported.
- **Theoretical necessity**: normalized pooling creates a provable equivalence class over histories with different total weight. APN's downstream query and decoder receive no separate mass or mask-count path in official commit `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`.
- **Timing necessity**: APN provides a current, lightweight, reproducible AAAI 2026 baseline whose claimed adaptation to information density makes the information-loss question directly testable.

### 2 Research Questions

#### RQ1: Core Question

**Does preserving patch-local observation evidence improve irregular sensor forecasting under native and observation-shift conditions?**

- **Corresponding gap**: TAPA discards its normalized pooling denominator.
- **Novelty boundary**: EviPatch does not change patch granularity, query aggregation, decoder, or loss.
- **Corresponding experiment**: Stage A compares APN and EviPatch on PhysioNet native, matched 30% MCAR, and 30% burst evaluation.

#### RQ2: Mechanism Question

**Are soft mass, effective support, and temporal coverage useful beyond raw patch count and projection-width controls?**

- **Relationship to RQ1**: this determines whether the idea is evidence preservation or merely feature-count augmentation.
- **Corresponding experiment**: raw-count, soft-mass, shuffled-evidence, and fixed-random-feature controls; controlled representation-collision tests; coverage recoverability probe after the main gate.

#### RQ3: Boundary Question

**If the mechanism survives PhysioNet, does it generalize across biomechanics and climate sensing?**

- **Corresponding experiment**: conditional Stage B on HumanActivity and USHCN; not run unless the Stage A gate passes.

### 3 Key Works

| Short Name | Venue | Year | Core Contribution | Role Here |
|---|---:|---:|---|---|
| APN | AAAI | 2026 | Adaptive soft-window aggregation for IMTS forecasting | Core baseline and diagnosed computation path |
| t-PatchGNN | ICML | 2024 | Patch and graph modeling for irregular series | Closest patch-based comparison after Stage A |
| TimeMosaic | AAAI | 2026 | Instance-wise adaptive granularity and segment prediction | Explicit novelty boundary; its mechanism is not adopted |
| ContiMask | NeurIPS | 2025 | Observation-structure-aware irregular perturbation | Motivates mechanism-specific shifts |
| Time-IMM | NeurIPS D&B | 2025 | Cause-driven irregularity benchmark | Optional post-gate external stress test |

#### APN (AAAI 2026) [1]

APN learns variable- and patch-specific soft temporal windows, normalizes their weighted sums into patch centroids, aggregates patches with a learnable query, and decodes future values with a shallow MLP.

> Borrowing value: official architecture, dataset loaders, splits, optimizer implementation, and evaluation metrics are preserved for paired comparisons.

#### t-PatchGNN (ICML 2024) [2]

t-PatchGNN models intra-patch irregular observations and inter-patch relations with time-aware patching and graph components.

> Borrowing value: establishes variable observation support as a central IMTS property and supplies a post-gate comparison implemented in the APN repository.

#### TimeMosaic (AAAI 2026) [3]

TimeMosaic adapts granularity and forecasting segments for regular time series.

> Boundary value: EviPatch must not add sample-conditioned boundaries or horizon-conditioned decoding.

---

## Part 2 Idea Design

### 1 Introduction

EviPatch is a minimal evidence-preserving extension of APN. It keeps TAPA's soft windows, normalized feature centroid, query module, decoder, training objective, and data splits unchanged. The only architectural change is to expose statistics already derivable from TAPA's temporal weights to the patch projection. The project is conditional: if the full signature cannot beat a raw-count control under a predeclared PhysioNet gate, the idea is abandoned rather than expanded.

### 2 Related Works

#### 2.1 Irregular Forecasting and Patching

Recent irregular forecasting methods use continuous-time dynamics, graphs, attention, or temporal patches. APN is selected because its soft-window normalization yields an exact and locally testable information-loss question while remaining lightweight enough for controlled multi-seed experiments.

#### 2.2 Observation Processes

Irregularity can arise from random thinning, bursts, event-triggered sampling, or system constraints. EviPatch does not claim to identify these mechanisms causally. It only gives the predictor local evidence descriptors and tests them under separately reported shifts.

#### 2.3 Research Gap

No work in the bounded review directly diagnoses APN's discarded soft-mass denominator and evaluates mass/support/coverage against count, shuffled-semantic, and equal-width controls. This is provisional novelty, not an exhaustive global novelty claim.

### 3 Method

#### 3.1 Evidence-Preserving TAPA

For temporal weight `a_i = alpha_ip * m_i`, APN computes a weighted centroid:

$$
\mu_p = \frac{\sum_i a_i z_i}{\sum_i a_i + \epsilon}.
$$

EviPatch additionally computes:

$$
s_p=\log(1+\sum_i a_i), \qquad
e_p=\log\left(1+\frac{(\sum_i a_i)^2}{\sum_i a_i^2+\epsilon}\right),
$$

$$
c_p=\log\left(1+\frac{\sqrt{\sum_i a_i(t_i-\bar t_p)^2/(\sum_i a_i+\epsilon)}}{t_p^{right}-t_p^{left}+\epsilon}\right).
$$

The full token input is `[mu_p, s_p, e_p, c_p]`, followed by the original linear projection, FFN, and residual normalization.

> Refinement from the source idea: the evidence vector is not passed through per-token LayerNorm. LayerNorm on a one-dimensional soft-mass or count ablation is identically zero and would make the comparison invalid. Monotone `log1p` transforms are used consistently instead; a training-only normalization sensitivity can be added after the gate.

#### 3.2 Controls

| Variant | Added information | Evidence width |
|---|---|---:|
| `apn` | None | 0 |
| `global_ratio` | Per-sample/channel observed ratio repeated over patches | 1 |
| `raw_count` | `log1p` hard count inside each learned window | 1 |
| `soft_mass` | `s_p` only | 1 |
| `evipatch_full` | `s_p`, `e_p`, `c_p` | 3 |
| `shuffled_evidence` | Full signature cyclically misaligned across sample-channel rows | 3 |
| `random_features` | Fixed Gaussian features indexed by variable and patch | 3 |

`random_features` has the same projection width as the full method, while `shuffled_evidence` preserves the marginal distribution but destroys semantic alignment.

#### 3.3 Structural Tests

- Exact duplication collision: fixed centroid and timestamps, different support, original APN output unchanged within numerical tolerance.
- Evidence sensitivity: the same pair produces different EviPatch signatures.
- No-side-channel audit: changing support while fixing the centroid cannot alter the APN query/decoder input.
- Gradient and finite-value checks for empty, single-observation, dense, and zero-width-limit patches.

#### 3.4 Baseline and Metrics

APN is the core baseline; t-PatchGNN is a conditional post-gate comparison. MSE and MAE follow the official repository. Robustness is measured as relative error inflation from native evaluation. Efficiency reporting includes parameters, peak VRAM, wall time, and inference latency.

---

## Part 3 Experiment Design

### 0 Baseline Experiment Survey

#### 0.1 APN (AAAI 2026) [1]

**Paper**: Rethinking Irregular Time Series Forecasting: A Simple Yet Effective Baseline
**Code**: `decisionintelligence/APN`, audited commit `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`

| Item | Official paper | Audited code |
|---|---|---|
| Datasets | PhysioNet, MIMIC, HumanActivity, USHCN | Same loaders; MIMIC needs credentials |
| Split | 80/10/10 stated | P12 is approximately 81/9/10 because validation is 10% of the remaining 90% |
| Optimizer | AdamW stated | Adam implemented and recommended by README note |
| Normalization | Not specified as test-inclusive | P12 encoder explicitly fits the concatenated full dataset |
| Epochs/patience | 200 / 10 | P12 script: 200 / 10 |
| Seeds | Mean ± std | Script default `itr=5`, seeds 2024 onward |
| Metrics | MSE, MAE | Masked MSE, masked MAE |

> Stage A preserves the audited code behavior for all variants. Split or normalization corrections are separate sensitivity experiments and may not be credited to EviPatch.

#### 0.2 Field Convention Synthesis

Standard benchmarks are PhysioNet 2012, HumanActivity, MIMIC, and USHCN. MSE and MAE are standard forecast metrics; mean and standard deviation over repeated seeds are expected. Component ablations, parameter sensitivity, and efficiency analysis recur in APN. For this idea, semantic controls are more important than adding unrelated model baselines before the mechanism survives.

### Data and Code Availability Summary

| Item | Status | Notes |
|---|---|---|
| APN code | Available | Official repository cloned locally; no clear top-level license detected, so it remains uncommitted and is reproduced by commit plus patch |
| PhysioNet 2012 | Available | Public automatic download; forced to project-local TSDM storage |
| HumanActivity | Available | Public automatic download; conditional Stage B |
| USHCN | Available | Public automatic download; conditional Stage B |
| MIMIC | Not used | Credentialed access excluded by user requirement |

### 1 Datasets

| Dataset | Variables | Samples reported by APN | Usage |
|---|---:|---:|---|
| PhysioNet 2012 | 36 | 11,981 | Stage A primary |
| HumanActivity | 12 | 1,359 | Conditional Stage B |
| USHCN | 5 | 1,114 | Conditional Stage B |

All raw, processed, and cached data must stay under `C:\Users\qintian\Desktop\msn2`. MIMIC is excluded.

### 2 Experiment Design

#### 2.1 Smoke and Contract Tests

**Purpose**: validate tensor shapes, exact APN parity for `apn` mode, finite gradients, deterministic controls, output-path isolation, and 100-step runtime before any multi-seed run.

**Pass condition**: all unit tests pass; APN-patched `apn` mode matches the original forward output after state-dict transfer within `atol=1e-6, rtol=1e-5`; no write occurs outside the project root.

#### 2.2 Stage A PhysioNet Kill-Test

**Training**: seven variants, seeds 2024/2025/2026, official P12 hyperparameters (`seq_len=36`, `pred_len=3`, `d_model=24`, `lr=0.03`, `batch=32`, `dropout=0.1`, `patches=20`, `TE=8`, Adam, at most 200 epochs, patience 10). Every variant changes only the evidence input and corresponding projection width.

**Evaluation views using each native checkpoint**:

- Native history.
- Deterministic exact-rate 30% MCAR history thinning.
- Deterministic exact-rate 30% contiguous burst loss, matched to MCAR deletion count per sample-variable whenever feasible.
- Controlled real-data support pairs constructed after checkpointing by matching patch value/time-embedding centroids while varying effective support; exact synthetic collision pairs are mandatory even if real matches are scarce.

The controlled-support pairing is fixed before reading any variant's errors:

1. For each seed, use only its APN checkpoint and native test histories to compute every learned patch's normalized value plus learned-time-embedding centroid and effective support.
2. Keep the maximum-support patch for each patient/channel, robustly standardize centroid coordinates within seed/channel, and greedily form disjoint pairs without reusing a patient/channel unit.
3. Require centroid RMS distance at most 0.35, effective-support ratio at least 2.0, and absolute effective-support difference at least 1.0.
4. Require at least 100 matched pairs per seed; lower yield fails the controlled-support gate closed rather than loosening thresholds post hoc.
5. Score the same frozen pair IDs for every variant using masked channel-level target MSE averaged over both patients. Save pair membership, centroid distance, support contrast, yield, and all exclusions.

**Primary metrics**: masked MSE, masked MAE, relative error inflation, and patient-level errors. Save predictions and masks for paired analysis.

**Statistical analysis**: report mean ± standard deviation over seeds and a hierarchical paired bootstrap that resamples seed blocks and patient IDs. Report effect sizes and 95% intervals; do not infer success from a single best seed.

**Predeclared kill gate**: continue only if all conditions hold:

1. `evipatch_full` improves the controlled-support primary error by at least 5% relative to APN.
2. `evipatch_full` is better than `raw_count` on the predeclared native/MCAR/burst macro-average, with the paired 95% interval for the error difference excluding zero.
3. `shuffled_evidence` and `random_features` do not reproduce the full method's gain; full must beat each control with the paired macro-MSE 95% interval excluding zero.
4. Native MSE does not regress by more than 1% relative to APN.
5. Full's parameter-count and predeclared 100-step training-time overheads are each strictly below 5% relative to APN.

If any mandatory condition fails, status becomes `ABANDON` and Stage B is not run.

#### 2.3 Conditional Stage B

If Stage A passes, run APN, raw count, full, and shuffled evidence on HumanActivity and USHCN for the same three seeds and native/MCAR/burst evaluations. Add t-PatchGNN on PhysioNet native and burst if runtime permits. Further trigger/throttle mechanisms, coverage probes, and leakage-free preprocessing are post-gate extensions and must be labeled separately.

#### 2.4 Efficiency

For APN, raw count, and full EviPatch, record parameter count, peak allocated CUDA memory, training step latency after warm-up, evaluation latency, checkpoint size, and total wall time. The full method must remain below 5% parameter and time overhead.

### 3 Resource Estimate

| Phase | Runs | Estimated time on RTX 4090 |
|---|---:|---:|
| Environment/data preparation | 1 | 1–3 h, network-dependent |
| Smoke and 100-step timing | 7 variants | 0.5–1.5 h |
| Stage A training | 7 × 3 seeds = 21 | 5–10 h after data preparation |
| Stage A evaluation/statistics | 63 evaluation views plus controls | 1–3 h |
| Conditional Stage B | 4 × 2 datasets × 3 seeds = 24 | 6–12 h |

> Estimates are provisional until the 100-step benchmark. The gate, not the estimate, determines whether Stage B runs.

## References

[1] Liu, Xvyuan, et al. “Rethinking Irregular Time Series Forecasting: A Simple Yet Effective Baseline.” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 28, 2026, pp. 23873–23881. https://doi.org/10.1609/aaai.v40i28.39563.

[2] Zhang, Weijia, et al. “t-PatchGNN: An Efficient Graph Neural Network for Temporally Irregular Multivariate Time Series.” *Proceedings of the 41st International Conference on Machine Learning*, 2024.

[3] “TimeMosaic.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2026. Exact bibliographic metadata remains to be rechecked before manuscript use.

## Pending Verification

- [ ] Confirm TimeMosaic's final AAAI bibliographic metadata before any paper citation.
- [ ] Measure real Stage A runtime before retaining Stage B workload estimates.
- [ ] Quantify the number and tolerance distribution of real controlled support-pair matches before using that analysis as a claim.
