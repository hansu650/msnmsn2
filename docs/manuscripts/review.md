# EdgeTwinCal manuscript review

> Reviewed: 2026-07-21
> Scope: ResearchPilot G.7 claim--evidence audit after the formal ablation rerun.

## Claim--evidence alignment

| Major claim | Direct evidence | Status |
|---|---|---|
| Released APN processes channels independently and uses a shallow decoder. | APN Sec. 3.2--3.5 and Eqs. (7)--(10); released `models/APN.py`. | Supported as a public-architecture fact. |
| Frozen APN has no explicit sensor--horizon residual-specialization or cross-sensor forecast-correction route. | Structural inference from the APN computation path; worded explicitly as our analysis, not an APN-author claim. | Supported and bounded. |
| SLRH and CFG capture complementary residual spaces. | SLRH and CFG individually reduce target-micro MSE by 0.553% and 0.591%; full reduces it by 1.048% and beats both. | Supported for this exploratory PhysioNet study. |
| EdgeTwinCal improves the paired frozen checkpoints. | Three seed-wise reductions of 1.056%, 1.005%, and 1.082%; aggregate APN 0.312331 +/- 0.000512 versus full 0.309058 +/- 0.000494. | Supported descriptively. |
| The patient-level effect is negative. | 10,000-resample hierarchical patient-macro full-minus-APN MSE -0.003793, 95% CI [-0.005619, -0.002130], over 1,185 patients per seed. | Supported descriptively; not corrected for route selection. |
| Adaptation is fast from cached representations. | Final rerun metrics record 1.40--1.42 seconds per checkpoint on an RTX 4090. | Supported for cached server-side fitting only. |
| Edge-device deployment is established. | No edge-device timing or energy measurement exists. | Not claimed; moved to future validation. |

## High-priority review outcomes

- Resolved checkpoint provenance: these are pre-existing checkpoints trained locally
  with the released APN implementation, not checkpoints published by APN's authors.
- Separated APN Table 2 paper values from the three-checkpoint paired evaluation
  and restored the omitted Warpformer row.
- Disclosed the released P12 behavior: approximately 81/9/10 splitting,
  train/validation drop-last, and full-data standardization.
- Distinguished target-micro MSE in the main table from patient-macro bootstrap
  effects and reported the 1,185-patient estimand.
- Labeled the final route exploratory because five structural attempts inspected
  the same PhysioNet test set.
- Removed confirmatory and edge-hardware claims unsupported by the current study.

## Remaining evidence needed for a stronger submission

1. Confirm EdgeTwinCal on an untouched dataset or newly reserved holdout.
2. Measure latency, energy, and memory on representative edge hardware.
3. Repeat under leakage-free train-only standardization if claiming general
   performance rather than released-implementation-compatible adaptation.
4. Test whether the static CFG generalizes beyond the single clinical dataset.

## Reverse outline

1. Introduction: frozen irregular forecasting needs cheap recalibration.
2. Structural diagnosis: APN's efficient public path lacks two explicit residual
   correction routes under frozen deployment.
3. Insight: decompose the residual into local latent and cross-forecast spaces.
4. Method: SLRH handles the local component; CFG handles the remaining cross-channel component.
5. Main experiment: paired three-checkpoint results improve target-micro MSE/MAE.
6. Ablation: both modules contribute and the full route is strongest on MSE.
7. Limitations: this is a post-selection pilot under released-code preprocessing.
