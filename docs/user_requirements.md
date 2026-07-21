# User Requirements

## EdgeTwinCal-Safe Final Confirmatory Override (2026-07-21, latest)

This section supersedes the earlier instruction to stop after the fifth APN
route only for one final, independently evaluated safety route. It does not
reopen, retune, or reinterpret any sealed `msn2026_v1` test result.

- Keep the APN baseline and pinned upstream commit unchanged. Do not switch
  backbones and do not modify APN''s query, decoder, loss, or training objective.
- Preserve APN, the original sequential EdgeTwinCal, and the existing Joint
  Ridge as immutable comparison implementations. EdgeTwinCal-Safe adds only
  two method modules: group-balanced robust residual fitting, and a bounded
  validation-only deployment safety envelope with exact APN fallback.
- Old P12 and USHCN results are diagnostic evidence only. Their opened tests,
  caches, and outcomes must never participate in hyperparameter selection,
  candidate ranking, gate thresholds, or implementation iteration.
- Use two genuinely new sensor-network targets with unopened chronological
  holdouts: UCI Beijing Multi-Site Air Quality and the official Intel Berkeley
  Lab wireless-sensor data. Freeze source hashes, preprocessing, split times,
  normalization, windowing, model settings, variant registry, statistics, and
  safety thresholds before either holdout is constructed or evaluated.
- Run paired APN checkpoints for seeds 2024--2028. Compare APN, Joint Ridge,
  original EdgeTwinCal, and EdgeTwinCal-Safe fairly on identical checkpoints,
  windows, masks, and pseudonymous group IDs. Retain every dataset and seed,
  including failures and Safe fallbacks.
- Report masked micro MSE/MAE, group summaries, every seed, crossed paired
  group-by-checkpoint bootstrap intervals, effect sizes, and ablations for
  robust fitting, group balancing, amplitude bounds, and the deployment gate.
- Final success requires at least two new targets to be positive, no target to
  regress by more than 1% in MSE or MAE, and Safe to meet the predeclared 0.1%
  MSE non-inferiority margin versus Joint Ridge. Any failure yields `ABANDON`;
  no post-test threshold change, sixth retry, or extra module is allowed.
- Measure real CPU/Jetson latency and memory only after the efficacy gate passes.
  Missing Jetson hardware is `BLOCKED`; workstation results cannot be labeled
  Jetson or edge-device evidence.
- Work on the isolated `lab/edgetwincal-safe` branch and under a new
  `edgetwincal_safe_v1` namespace. Preserve the sealed `msn2026_v1` package,
  all old results, and unrelated user changes. All mutable paths remain inside
  `C:\Users\qintian\Desktop\msn2`; never set `HOME` or `CODEX_HOME`.
- This request authorizes the complete ResearchPilot C/D/E/F workflow without
  repeated confirmation: freeze the design, implement, test, download the two
  public datasets, train sequentially, open each new holdout once, aggregate,
  apply the gate, document the result, and commit/push the isolated branch.

## Current Lab Handoff Override (2026-07-21, latest)

The user-provided directory
`C:\Users\qintian\Downloads\EdgeTwinCal_Lab_Experiment_Handoff_20260721`
is the current experiment contract. It is read-only input; every generated file
must remain under `C:\Users\qintian\Desktop\msn2`. This section supersedes both
the earlier EdgeTwinCal pilot scope and the historical EviPatch scope below.

- Work in the current Codex task and on an isolated Git branch; do not create a
  second task, do not touch the sibling `Desktop\msn`, and do not set `HOME` or
  `CODEX_HOME`.
- Keep APN at upstream commit
  `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`. APN is the only locally evaluated
  backbone. Values for other methods come only from APN Table 2 and are labeled
  as reported; APN Table 3 is neither copied nor reproduced.
- Treat the existing three-checkpoint P12 result as an exploratory pilot. Before
  opening any new test split, implement and pass FIX-01 through FIX-06, freeze
  configs/splits/normalizers/variant registry/alpha grids/statistics, and record
  the freeze hashes.
- Run release-parity and leakage-controlled campaigns separately. Strict P12 and
  USHCN are mandatory when assets exist; HumanActivity should use participant-
  held-out folds; MIMIC-III is `BLOCKED` without legal authorization.
- Target five paired APN checkpoints (2024--2028). Run APN/SLRH/CFG/Full on every
  runnable release campaign. On strict P12 and USHCN also run the predeclared
  controls V01, V02, V03, V07, V08, V10, V11, and V12.
- Preserve the intended sequential semantics: SLRH corrects frozen APN forecasts;
  CFG then corrects the SLRH output while its zero-diagonal features remain other-
  sensor frozen APN forecasts. Ridge intercepts are unpenalized and one global
  alpha is selected per dataset/checkpoint/variant on validation micro MSE from
  `{1,10,100,1000,10000,100000}`.
- New test data is opened once only after the G0/G1 pre-test gates pass. Test data
  is never used for tuning, route selection, or protocol repair.
- Statistical inference uses group x checkpoint crossed paired bootstrap with
  50,000 draws and seed 20260721, paired multiplicities across variants, effect
  sizes/95% intervals, and Holm correction for declared families.
- Report segmented timing truthfully. CPU closed-form solves must not be labeled
  GPU work. Edge claims require a real edge CPU/Jetson measurement; the desktop
  RTX 4090 is not an edge substitute. Unavailable hardware is `BLOCKED` and the
  claim is narrowed.
- No paper conclusion rewrite is part of this handoff. Deliver the audited lab
  return package only, excluding datasets, caches, checkpoints, NPZ files,
  `vendor/APN`, PDFs, environments, secrets, and private absolute paths.
- Missing HumanActivity/USHCN/MIMIC assets and seeds 2027/2028 are blockers, not
  permission to invent results. Complete all runnable work and record each
  unavailable cell explicitly.

## Earlier EdgeTwinCal Pilot Override (2026-07-21, superseded)

The latest user instructions supersede the historical EviPatch-only scope below.

- Current method: **EdgeTwinCal**, after five distinct structural attempts on APN.
- Target track: **Edge Computing, IoT and Digital Twins** at IEEE MSN 2026.
- Baseline policy: reuse the existing APN (AAAI 2026) checkpoints previously
  trained locally with the released implementation and keep APN frozen; do not
  retrain APN or reproduce additional baselines.
- Experiments: PhysioNet 2012 main result and the APN/SLRH/CFG/full ablation only,
  with seeds 2024/2025/2026. External comparison values may be quoted from papers.
- Manuscript: create a concise English IEEE-conference LaTeX draft using the user-
  supplied `IEEE-conference-template-062824`; the draft need not be polished.
- Writing example: use the supplied previous `submission.pdf` for structure/style,
  but do not repeat its visual time-series anomaly-detection topic or Big Data and
  AI track.
- Cleanup: remove failed-route code and results created in the current work while
  preserving the final EdgeTwinCal route and the immutable historical audit.
- Continue to enforce the `msn2` root boundary and never read/write the sibling
  `msn` project.

The remaining sections are retained as historical EviPatch requirements and do
not override this current-route section.

## Scope

- Project root: `C:\Users\qintian\Desktop\msn2` only.
- Do not modify, import files from, or write outputs into `C:\Users\qintian\Desktop\msn` or its existing `msnmsn` repository.
- Improve and test the EviPatch idea; do not write the paper manuscript.
- Follow the staged kill-test protocol in `MSN2026_EviPatch_idea.md` before any expansion.
- Final delivery must include reproducible code, experiment logs, statistical summaries, checksums, and one result archive.
- Intermediate code and appropriate compact results may be pushed only to `hansu650/msnmsn2`.

## Experiment Requirements

- Core baseline: official APN (AAAI 2026).
- First run a smoke test, then the PhysioNet Stage A kill-test with three seeds.
- Preserve the official split and optimizer unless a documented compatibility fix is necessary.
- Include decisive controls: APN, global observed ratio, raw patch count, soft mass, full EviPatch, shuffled evidence, and equal-parameter random features.
- Stop if full EviPatch does not significantly outperform raw patch count under the predeclared gate.
- Run no MIMIC experiments requiring credentials.
- Expand to HumanActivity and USHCN only if Stage A passes.

## Environment and Isolation

- Create a fresh Conda prefix environment at `C:\Users\qintian\Desktop\msn2\.conda\envs\evipatch`.
- Store datasets, caches, checkpoints, logs, and result archives under the project root.
- Add all large/generated/private artifacts to `.gitignore`; never commit the Conda environment or raw datasets.
- Guard scripts against output paths outside the project root.

## Document Preferences

- Language: Chinese for user-facing summaries; English is acceptable for code and machine-readable experiment artifacts.
- No manuscript generation.
- Keep research documentation focused on idea refinement, implementation, reproducibility, and experimental evidence.
