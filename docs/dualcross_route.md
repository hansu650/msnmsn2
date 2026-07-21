# DualCross attempt series

> Current route override: 2026-07-21. This document supersedes the abandoned
> EviPatch route for new experiments without altering its frozen audit record.

## Scope

- Target track: **Edge Computing, IoT and Digital Twins**.
- Baseline: official APN (AAAI 2026 Oral) checkpoints; APN is not retrained.
- External comparison: use values reported by APN and other top-conference papers;
  do not reproduce additional baselines.
- Runtime: one PhysioNet main experiment plus mechanism ablations, designed to
  finish within one day on the existing RTX 4090.
- Baseline policy: keep APN for five distinct structural attempts before any
  baseline switch. Hyperparameter-only changes do not count as attempts.

## Attempt ledger

### Attempt 1: CorePatch -- failed

A shared aggregate-redistribute core was appended to frozen per-channel APN
features. Seed 2024 test MSE changed from 0.312922 to 0.312412 (0.163% relative)
and MAE from 0.366014 to 0.365009 (0.274% relative). This is below the frozen 1%
MSE threshold and is recorded as a failure, despite the small positive change.

Diagnosis: a single global core compresses all 36 sensors into the same summary;
it lacks target-channel-specific routing. Training loss decreased, but validation
improvement saturated early.

### Attempt 2: DualCross -- frozen design

Structural problem: APN processes every sensor independently through patching,
query aggregation, and decoding. Another sensor cannot influence the target
sensor forecast.

The unified solution introduces two modules at two different locations while
freezing all APN weights:

1. **Cross-Sensor Feature Interaction (CSFI)** is inserted between APN's channel
   representation and its frozen decoder. Diagonal-masked low-rank attention
   routes sample-specific information from other sensors and produces a residual
   feature update.
2. **Shrinkage Residual Graph (SRG)** is placed after the decoder. It predicts each
   target sensor's residual from the other sensors' forecasts using a strongly
   regularized directed graph fit in closed form on the training split.

The main variant is CSFI+SRG. Ablations are APN, CSFI only, and SRG only. APN is
loaded from an existing checkpoint produced by its official implementation and remains frozen in every variant.

## Attempt-2 decision rule

Seed 2024 is the kill-test. The attempt passes only if the combined method reduces
masked test MSE by at least 1% relative to the same frozen APN checkpoint, does not
worsen MAE, and is no worse than the stronger single-module ablation. If it passes,
run seeds 2025 and 2026 plus the four-way ablation. If it fails, record the result
and proceed to structural attempt 3 on APN without test-set tuning.

## Validation-only diagnosis

A training-only ridge fit evaluated on validation (never on the attempt-2 test
targets) showed that cross-sensor residuals are predictable: with alpha=1000,
cross-only calibration improved validation MSE by 1.127%, versus about 0.413% for
self-only calibration. This observation selected target-specific cross-sensor
routing; it did not select a test result.

### Attempt 2 result: failed

On seed 2024, CSFI, SRG, and the combined method improved test MSE by 0.232%,
0.464%, and 0.678%, respectively. The combined method beat both single modules
and improved MAE by 0.509%, but missed the frozen 1% MSE threshold. No attempt-2
hyperparameter was changed after observing the test result.

### Attempt 3: pseudo-observation bridge -- validation kill

The input-side module fitted cross-sensor pseudo-observations at synchronous
events and injected them into frozen TAPA with low-confidence masks; the planned
second module was a support-gated output fusion. Training data had 7.24 observed
sensors per active event and 99.3% multi-sensor active events, yet the best
validation setting (ridge alpha 10000, pseudo-mask weight 0.03) worsened MSE by
8.75%. The route was killed before test evaluation and before implementing the
dependent output module.

### Attempt 4: VarGraph -- frozen design

APN uses one decoder for all 36 variables without an explicit variable identity.
VarGraph addresses decoder homogeneity and cross-sensor omission at two locations:

1. **Variable-Identity FiLM (VIF)** applies channel-specific scale and shift to the
   frozen APN representation immediately before its frozen shared decoder.
2. **Latent-Conditioned Residual Graph (LRG)** operates after the decoder and fits
   each target residual from that target's frozen latent state plus all same-horizon
   sensor forecasts using validation-selected ridge shrinkage.

Validation-only screening gave 0.563% MSE improvement for VIF and 1.500% for the
local-latent plus cross-forecast residual design. The formal ablations are APN,
VIF, LRG, and VIF+LRG; the seed-2024 test remains unseen for this attempt until the
implementation and unit tests pass.

### Attempt 4 result: failed

VIF, LRG, and VIF+LRG improved seed-2024 test MSE by 0.238%, 1.116%, and
1.076%. Although both LRG variants crossed 1%, the combined method was slightly
worse than LRG alone (0.309554 versus 0.309429 MSE), violating the frozen
two-module rule. The route is therefore recorded as failed without retuning VIF.

### Attempt 5: EdgeTwinCal -- frozen design

The final APN attempt decomposes residual adaptation sequentially across two
locations while keeping the full APN checkpoint immutable:

1. **Sensor Latent Residual Head (SLRH)** is a sensor- and horizon-specific linear
   readout from APN's frozen latent representation, parallel to the frozen decoder.
2. **Cross-Forecast Graph (CFG)** is a diagonal-free directed graph after that
   forecast, fitted only on the remaining residual from other sensors' predictions.

Both modules use closed-form ridge fitting on training data and choose shrinkage
on validation. Validation-only screening improved MSE by 0.816% for SLRH and
1.546% for SLRH+CFG; the selected validation alphas were 100 and 1000. Formal
ablations are APN, SLRH, CFG, and the combined EdgeTwinCal. Seed-2024 test remains
unseen for attempt 5 until implementation tests pass.

### Attempt 5 result: passed

The frozen seed-2024 kill-test passed without post-test retuning: APN MSE changed
from 0.312922 to 0.309617 (1.056% reduction), MAE also improved, and the combined
method beat both single-module ablations. The same frozen protocol then passed on
seeds 2025 and 2026 with 1.005% and 1.082% MSE reductions.

Across the three existing official-implementation checkpoints, APN obtained 0.312331 +/- 0.000512 MSE and
EdgeTwinCal obtained 0.309058 +/- 0.000494, a 1.048% relative reduction. SLRH and
CFG alone improved MSE by 0.553% and 0.591%, while their combination exceeded
both. A 10,000-replicate seed-block and patient-level paired bootstrap estimated
full-minus-APN MSE at -0.003793 with 95% CI [-0.005619, -0.002130]. This closes the
five-attempt APN policy with a successful route; no baseline switch is required.
