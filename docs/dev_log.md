# Dev Log — EviPatch
> Created: 2026-07-20 | Last updated: 2026-07-21
> Linked implementation guide: `docs/implementation.md`
> This file is append-only. Every code change appends a log entry.

## Project Overview

| Item | Detail |
|---|---|
| Research direction | Evidence-preserving adaptive patching for irregular sensor forecasting |
| Implementation strategy | Project-owned modules plus patch against pinned APN |
| Framework | PyTorch 2.6.0 + CUDA 12.4 |
| Environment | `.conda/envs/evipatch` prefix under project root |
| Git repository | `git@github.com:hansu650/msnmsn2.git` |
| Push scope | Whole project except ignored environments, baseline source, data, caches, checkpoints, and large outputs |
| Auto-run | Fast tests and full Stage A authorized; Stage B conditional on gate |

## Project Architecture

```text
code/src/evipatch/ → patched vendor/APN → project-local data/results/logs → aggregate → gate → package
```

## Implementation Progress

| Module | File | Status | Completed | Notes |
|---|---|---|---|---|
| Init | `code/README.md`, notebook placeholder | ✅ Done | 2026-07-20 | Project scope documented |
| Environment | `code/requirements.txt`, setup script | ✅ Done | 2026-07-21 | Existing CUDA environment verified; no rebuild performed |
| Config | `code/configs/stage_a.json` | ✅ Done | 2026-07-21 | Seven variants, three seeds, three views, gates, and local roots validated |
| Evidence | `code/src/evipatch/evidence.py` | ✅ Done | 2026-07-21 | Syntax, shape, finite-value, and gradient checks passed |
| Shifts | `code/src/evipatch/shifts.py` | ✅ Done | 2026-07-21 | Exact-count MCAR/burst and batch-order determinism checks passed |
| Paths | `code/src/evipatch/paths.py` | ✅ Done | 2026-07-21 | Module-derived root and fail-closed containment checks passed |
| Runner | `code/src/evipatch/runner.py` | ✅ Done | 2026-07-21 | Validate/timing/controlled/status and serial-dispatch contracts passed |
| Aggregation | `code/src/evipatch/aggregate.py`, `controlled.py` | ✅ Done | 2026-07-21 | Frozen support pairing, patient metrics, bootstrap, and fail-closed gate passed |
| Packaging | `code/src/evipatch/package.py` | ✅ Done | 2026-07-21 | Filter, checksum, secret scan, and ZIP verification passed |
| Baseline patch | `patches/apn_evipatch.patch` | ✅ Done | 2026-07-21 | Final 11-file clean replay/compile and APN parity passed |
| Tests | `code/tests/` | ✅ Done | 2026-07-21 | 40 passed |
| Scripts | `code/scripts/` | ✅ Done | 2026-07-21 | Boundary-checked setup, patch, smoke, Stage A, monitor, and package entry points |
| Stage A | results and gate | ⬜ TODO | — | |

Status: ⬜ TODO / 🔄 WIP / ✅ Done (run-verified) / ❌ Blocked

## Dev Log Entries

### 2026-07-20 23:20 — Isolated project initialized

- **Completed**: cloned empty target repository, pinned APN upstream locally, created the project-local Conda prefix, recorded the confirmed design and implementation guide, and created code/notebook README files.
- **Issues**: HTTPS GitHub cloning was unavailable; APN paper/code disagree on optimizer and split details; P12 code fits normalization on full data.
- **Solutions**: used authenticated SSH; Stage A preserves audited code behavior for paired parity and records leakage-free preprocessing as a separate post-gate sensitivity.

### 2026-07-21 00:29 - Implementation-plan audit before coding

- **Completed**: verified the existing Conda prefix, CUDA runtime, RTX 4090, pinned APN commit, clean vendor worktree, and all mandatory project paths; expanded `docs/implementation.md` to cover explicit seed selection, collision-free evaluation names, and persistence of shift-deletion audit arrays.
- **Reason**: upstream APN hard-codes the seed origin and names evaluations only to minute precision, which cannot provide an unambiguous one-seed/one-shift manifest or guarantee that native/MCAR/burst outputs do not collide.
- **Scope**: orchestration and audit output only; APN query, decoder, loss, patch granularity, split, optimizer, and native model computation remain unchanged.
- **How to Run check**: no command changed yet; command documentation will be added after the runner and scripts are executable.

### 2026-07-21 00:36 - Evidence module implemented

- **Completed**: added the `evipatch` package initializer and implemented all seven evidence-width modes, differentiable soft mass/effective support/coverage statistics, hard-count and global-ratio controls, deterministic cyclic shuffled evidence, and a fixed non-trainable Gaussian feature table.
- **Validation**: `py_compile` passed; a synthetic full-signature forward/backward check returned shape `[2, 3, 3]` with finite outputs and finite gradients.
- **Issue**: the first new-file unified diff was truncated to one line by the Windows pipeline; the immediate syntax check caught it before any downstream use.
- **Solution**: replaced the file with an automatically line-counted unified diff and reran syntax plus tensor checks successfully.
- **How to Run check**: no user command added; this module is exercised through the forthcoming pytest smoke script.

### 2026-07-21 00:41 - Deterministic shift module implemented

- **Completed**: implemented sample-ID/channel/seed keyed stable hashing, exact per-sample/channel MCAR deletion, exact matched-count chronological burst deletion, zeroing of removed history values, and requested/actual deletion audits.
- **Validation**: `py_compile` passed; a reordered-batch check produced identical sample-specific MCAR masks, and MCAR/burst each removed exactly three of ten points in every synthetic row.
- **Isolation/target check**: the API accepts only history tensors and returns no target tensor, preventing target modification by construction.
- **How to Run check**: no standalone command added; formal cases will run through the smoke pytest suite.

### 2026-07-21 00:44 - Path isolation module implemented

- **Completed**: implemented module-derived project-root discovery, relative-path anchoring to `msn2`, resolved-path containment checks, guarded directory creation, and bulk mutable-path validation.
- **Validation**: syntax/import checks passed; `results` resolved inside the project while the sibling `msn`, `..\\msn`, and `C:\\tmp` were all rejected.
- **Boundary behavior**: the project root itself is rejected as an output target unless a caller explicitly opts in, reducing broad accidental writes.
- **How to Run check**: no standalone command added; boundary behavior will be covered by pytest and every runner/package entry point.

### 2026-07-21 00:47 - Stage A configuration implemented

- **Completed**: centralized all seven variants, seeds 2024/2025/2026, native/MCAR/burst views, official P12 hyperparameters, fixed-random control seed, bootstrap settings, five kill-gate families, conditional Stage B scope, and packaging exclusions.
- **Validation**: JSON parsing, seven-variant cardinality, exact seed list, Adam selection, and every configured mutable path passed project-boundary validation.
- **Protocol note**: shift randomness is keyed by the checkpoint seed, so paired variants for the same seed see identical deletions while seed blocks remain independent.
- **How to Run check**: no command added yet; `runner.py` will expose this config through a single CLI.

### 2026-07-21 00:56 - Isolated sequential runner implemented

- **Completed**: implemented strict config/upstream validation, shell-free APN argv construction, one-seed training, three independent evaluation views, resumable manifests, checkpoint discovery, SHA-256/provenance capture, process wall time, sampled process GPU memory, a single-GPU lock, and a fixed serial 7 × 3 dispatcher.
- **Validation**: `py_compile`, `runner validate`, `runner status`, and train-argv contract checks passed; status correctly reports all 21 checkpoint slots and 63 evaluation slots as pending.
- **Review fix**: checkpoint sorting now derives the seed from the path relative to the Stage A root instead of a brittle absolute path offset.
- **How to Run check**: added validate, status, smoke, Stage A, aggregation, and packaging commands below; GPU/data commands remain pending integration smoke.

### 2026-07-21 01:08 - Aggregation and automatic gate implemented

- **Completed**: implemented strict evaluation loading, patient-level masked MSE/MAE, three-seed mean ± std, equal-weight native/MCAR/burst macro summaries, seed-block plus patient-ID hierarchical paired bootstrap, effect sizes/95% intervals, all mandatory gate conditions, and a concise Chinese experiment report.
- **Validation**: syntax passed; synthetic masked errors matched hand calculations, and a three-seed/60-pair bootstrap recovered an exact -0.2 paired MSE effect with a wholly negative 95% interval.
- **Fail-closed behavior**: a missing or malformed controlled-support result file makes the gate fail instead of silently omitting condition 1; any failed condition yields ABANDON, not post-hoc CAUTION.
- **How to Run check**: the existing `runner aggregate` command now maps to this implementation and writes CSV/JSON/REPORT_CN artifacts.

### 2026-07-21 01:14 - Filtered result packaging implemented

- **Completed**: implemented targeted source/artifact/result selection, secret signatures, per-file size caps, compact-log generation, stable SHA-256 manifests, fixed ZIP timestamps, checkpoint/environment/data/vendor exclusions, and post-write ZIP/member checksum verification.
- **Validation**: syntax passed; a project-local smoke archive included code plus both package metadata files, passed `ZipFile.testzip`, and contained no `.conda`, vendor source, or checkpoint member. All temporary smoke outputs were then removed from the verified `packages` directory.
- **Issue**: the unprivileged Windows sandbox assigned an unusable ACL to its first Python-created temporary directory.
- **Solution**: reran the same project-local write under the required sandbox approval, verified the archive, and cleaned only the four explicitly resolved temporary targets.
- **How to Run check**: the existing `runner package` command is current; it requires a finalized `artifacts/gate_decision.json`.

### 2026-07-21 01:21 - Smoke iteration: numeric and Windows compatibility fixes

- **Diagnosis**: first smoke produced 27 passes, seven fixture errors because the pytest base parent did not exist, and one single-point coverage failure caused by epsilon perturbing the weighted timestamp mean. The corrected second run reached 34/35; its sole failure was bundled tsdm referencing NumPy extended-precision names absent on Windows.
- **Changes**: `evidence.py` now divides timestamp moments by exact mass when non-empty and a safe unit denominator only when empty; `runner.py` creates the project-local results parent before pytest; the APN patch aliases only missing `float128/complex256` names to NumPy `longdouble/clongdouble`.
- **Scope**: numeric stability, test isolation, and import compatibility only; no model component, optimizer, split, loss, query, decoder, or claim changed.
- **Validation**: the third full smoke completed with 35 passed in 8.03 seconds.
- **Document sync**: `docs/implementation.md` updated before the tsdm compatibility code; How to Run command unchanged.

### 2026-07-21 01:23 - Reproducible APN patch finalized

- **Completed**: generated `patches/apn_evipatch.patch` from `git diff` against APN commit `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`; SHA-256 at this validation point is `EAC04E2C859C22D7138FD80E39529E29B3F534A5BC95A882ADF37BCCFFE779A2`.
- **Validation**: reverse-check passed on the modified local clone; a separate clean detached worktree accepted the patch, passed `git diff --check` and patched-file compilation, then was removed. Full APN-mode forward state migration has maximum absolute error 0.
- **Licensing boundary**: vendor/APN remains ignored and uncommitted; only upstream URL/commit plus the patch will be delivered.
- **How to Run check**: patch application will be exposed by `code/scripts/apply_apn_patch.ps1` in the next implementation step.

### 2026-07-21 01:24 - Formal smoke test suite completed

- **Completed**: added 35 tests covering evidence shapes, empty/single/dense inputs, finite gradients, scalar informativeness, exact centroid collision, seven projection widths, fixed random features, full-model APN parity, deterministic exact shifts, target preservation, path/TSDM isolation, runner contracts, patient/bootstrap statistics, gate fail-closed behavior, and deterministic package verification.
- **Result**: `35 passed in 8.03s`; raw output is saved at `logs/smoke/pytest.log`.
- **How to Run check**: `runner smoke` is now run-verified and keeps pytest temporary data under `results/pytest_tmp`.

### 2026-07-21 01:31 - Exact timing runner completed

- **Completed**: added a resumable seven-variant `timing` action and corrected APN's existing timing helper so its `test_steps=100` argument caps both training and inference measurements after warm-up.
- **Scope**: benchmark control only; the real P12 loader, official Adam optimizer, loss, batch size, model computation, and training protocol are unchanged.
- **Validation**: runner compilation, timing argv inspection, and the focused runner/APN parity suite passed (7 tests); the real-GPU 100-step matrix remains the next smoke action.
- **How to Run check**: added `runner timing`; `run_smoke.ps1` runs it after pytest unless `-SkipTiming` is supplied.

### 2026-07-21 01:31 - Isolated operation scripts completed

- **Completed**: added shared project-boundary enforcement plus environment verification, idempotent pinned-APN patching, smoke/timing, strict serial Stage A, status monitoring, and aggregate/package PowerShell entry points.
- **Isolation**: scripts derive `msn2` from `PSScriptRoot`, reject paths outside it, and set only `PYTHONPATH`, `EVIPATCH_PROJECT_ROOT`, and `EVIPATCH_TSDM_ROOT`; they never set `HOME` or `CODEX_HOME`.
- **Validation**: all seven scripts passed PowerShell parser checks; environment/pip/CUDA validation reported Python 3.11.13, torch 2.6.0+cu124, CUDA available, RTX 4090, and a clean dependency check; patch application correctly reported already-applied.
- **Patch update**: the regenerated nine-file APN patch, now including exact timing caps, has SHA-256 `F60F622B6E934E360E93521731873A9BE5F4777740E81F13C05C8BFB7F47A997`; clean detached-worktree replay, `diff --check`, and compilation passed.
- **How to Run check**: the script commands below are now the preferred entry points.

### 2026-07-21 01:47 - Real-P12 smoke: cache and Windows worker diagnosis

- **Diagnosis 1**: the first official tsdm clean returned `(metadata, series)` in memory, while its persisted deserializer returns `(series, metadata)` as the P12 task expects. The first run therefore populated the isolated cache and ended with `KeyError: Time`; a fresh process verified all A/B/C first frames have the expected `RecordID, Time` index.
- **Diagnosis 2**: the next run computed `seq_len_max_irr=35` in the main process, but Windows DataLoader workers use `spawn`, re-imported the module-global config, and saw `None`. The same loss of runtime config would also discard requested test shifts.
- **Change**: documented the compatibility contract first, then bound the active P12 `ExpConfigs` into `collate_fn` with a pickle-safe `partial`; the collate now reads padding, training/evaluation, seed, and shift values from this explicit runtime object.
- **Scope**: Windows process transport only. P12 samples/split/normalization, target tensors, APN model, Adam optimizer, and all Stage A hyperparameters are unchanged.
- **Validation**: compilation and a spawned-worker simulation with intentionally invalid module-global padding passed (`2 passed`); real P12 timing is retried next.
- **How to Run check**: no command changed; `runner timing` resumes from completed variant records and overwrites the failed variant log.

### 2026-07-21 02:14 - Seven-variant real-P12 timing completed

- **Result**: after 10 warm-up steps, exactly 100 optimizer steps produced APN/global-ratio/raw-count/soft-mass/full/shuffled/random means of 5.592/5.230/5.253/5.622/5.767/5.972/5.392 ms.
- **Gate evidence**: full's predeclared step-time overhead versus APN is 3.13%, below the strict 5% threshold; the immutable records, argv, wall times, and raw logs are saved in `artifacts/timing_100_steps.json` and `logs/smoke/timing_*.log`.
- **Protocol**: measurements used the real official P12 train loader, Adam, MSE, batch size 32, and a single RTX 4090 sequentially. No timing was rerun or selected based on favorable values.
- **How to Run check**: `runner timing` now resumes a complete seven-record artifact without rerunning completed variants.

### 2026-07-21 02:14 - Controlled-support analysis and strict gate completed

- **Completed**: added error-blind APN-checkpoint reconstruction of learned patch value/time-embedding centroids and effective support, max-support patient/channel units, robust within-channel matching, disjoint frozen pair IDs, identical all-variant scoring, pair/yield audits, and fail-closed minimum-yield enforcement.
- **Frozen thresholds**: centroid RMS ≤ 0.35, support ratio ≥ 2, support difference ≥ 1, and at least 100 pairs per seed; thresholds are centralized in `stage_a.json` and were documented before implementation.
- **Gate correction**: time overhead now comes only from the real 100-step artifact, not end-to-end training wall time that can contain initial data preparation; missing/malformed controlled or timing evidence produces ABANDON.
- **Validation**: controlled reconstruction/matching/scoring and aggregation gate tests passed (8 tests); native evaluations now save `x_mark` plus complete requested/actual/original/remaining shift audits required by this analysis.
- **How to Run check**: added `runner controlled`; `runner aggregate` regenerates frozen pairs before gate evaluation.

### 2026-07-21 02:14 - Clean-cache P12 reproduction finalized

- **Completed**: made bundled tsdm's first-clean PhysioNet tuple order match its own persisted deserializer and passed active runtime configs into P12 collate workers through a pickle-safe partial.
- **Reason**: an empty cache previously failed once with `KeyError: Time`; Windows spawned workers then lost runtime padding/shift values. Both are baseline process-transport defects, not experimental changes.
- **Validation**: clean/deserialize order, Windows-worker padding/shift transport, target preservation, and isolation contracts passed; the lowercase upstream module path was restored after a Windows case-only patch artifact was detected.
- **Final patch**: the 11-file patch SHA-256 is `8DB0BF3CE05715D8C4B131D4F80E4DF7F13EBD8C4C36A9B2D3BEA841B3E01CD5`; reverse-check and clean detached-worktree apply, `diff --check`, and compilation passed.
- **How to Run check**: no new setup step is required; `run_smoke.ps1` works from an empty project-local data cache.

### 2026-07-21 02:14 - Expanded formal smoke completed

- **Result**: `40 passed in 7.32s`, including the original parity/collision/gradient/shift/path/package contracts plus controlled-support, timing-gate, Windows-worker, and clean-cache regressions.
- **How to Run check**: raw output is current at `logs/smoke/pytest.log`.

### 2026-07-21 02:19 - Pre-commit delivery gate passed

- **Validation**: the final pre-commit full suite completed with `40 passed in 7.60s`; the current raw output is staged from `logs/smoke/pytest.log`.
- **Git boundary**: 48 files totaling 276,170 bytes passed staged `diff --check`, forbidden-path checks, and the 100 MB limit; no environment, data, vendor source, result tree, checkpoint, cache, or package was tracked.
- **Formatting**: added explicit LF rules for source/docs/patch/log artifacts and CRLF for PowerShell; nested patch whitespace is exempted only from top-level patch-file false positives.
- **How to Run check**: commands are unchanged.

## 2026-07-21 02:26 CST — Windows Stage A loader iteration

- **ResearchPilot phase**: `research[F]-iteration`; diagnosed an execution bottleneck after the first incomplete APN/2024 attempt, before any training manifest or evaluation result was accepted.
- **Observation**: with the official `num_workers=10`, Windows `spawn` startup consumed roughly 30 seconds for the training iterator and 27 seconds for validation in every epoch, while the model steps themselves completed in only a few seconds.
- **Decision**: set the Stage A runtime-only loader setting to `num_workers=0`. This preserves the official P12 split/preprocessing, batch size, shuffled sampler, seed policy, model, loss, Adam optimizer, schedule, and all model hyperparameters; it only removes repeated worker-process startup.
- **Recovery**: terminate the exact project-owned runner/child/worker PIDs, remove only the incomplete `results/stage_a/apn/2024` tree and stale Stage A lock, then restart the matrix from APN/2024. No completed checkpoint, manifest, metric, or evaluation is discarded.
- **Validation required before restart**: boundary validation plus the full unit/smoke suite, followed by direct confirmation that the first restarted epoch no longer shows the worker-spawn delay.
- **How to Run check**: commands remain unchanged; `stage-a` reads the revised boundary-checked config.

## 2026-07-21 02:36 CST — Child-side CUDA peak audit design

- **ResearchPilot phases**: `research[F]-iteration` diagnosed the first completed APN/2024 training manifest; `research[E]-coding` implements the bounded instrumentation fix.
- **Failure evidence**: training and native evaluation completed, but WDDM exposed no process memory through `nvidia-smi --query-compute-apps`, leaving `peak_gpu_memory_mib=0.0` despite CUDA execution.
- **Design**: the patched APN child resets CUDA peak counters immediately before `main()`, synchronizes in `finally`, and prints one JSON marker containing `torch.cuda.max_memory_allocated` and `max_memory_reserved` in MiB. This is measurement-only and cannot affect model outputs or optimization.
- **Runner contract**: `run_checked` parses the child marker and records allocated/reserved peaks plus `gpu_memory_source=torch.cuda`; its existing process-scoped `nvidia-smi` value remains a fallback for generic subprocesses without the marker.
- **Recovery**: stop before accepting later variants, remove the incomplete Stage A tree, regenerate `patches/apn_evipatch.patch`, run unit/parity/smoke validation, commit the fix, then restart from APN/2024 so every manifest uses the same measurement contract.
- **How to Run check**: commands remain unchanged; the metric is emitted and captured automatically by every train/evaluate action.

## 2026-07-21 02:50 CST — CUDA peak audit run-verified

- **Targeted regression**: `code/tests/test_runner.py` completed with `5 passed`; child markers override the WDDM fallback and generic subprocesses retain nullable child fields.
- **Real APN evidence**: a diagnostic native evaluation of the discarded APN/2024 checkpoint reported `31.36474609375 MiB` peak allocated and `48.0 MiB` peak reserved with `gpu_memory_source=torch.cuda` in its manifest.
- **Patch replay artifact**: regenerated the same 11-file `patches/apn_evipatch.patch`; SHA-256 is `00D8D59221D1580EE2B718365325BD69945DC2C103B0C23D7F93F9365E301746`.
- **Full validation**: boundary validation, exact APN parity, clean patch replay, evidence/shift/path tests, and the new memory contract completed with `41 passed in 7.88s`.
- **Recovery completed**: removed only the diagnostic `results/stage_a` and `logs/stage_a` trees after boundary verification; the formal matrix will restart from APN/2024 at the next commit.
- **How to Run check**: commands are unchanged.

## 2026-07-21 03:48 CST — Stage A complete; kill gate ABANDON

- **Matrix completion**: all seven variants × three seeds trained sequentially on one RTX 4090; `runner status` verified 21/21 checkpoints and 63/63 native/MCAR/burst views, and the Stage A lock was released.
- **Controlled-support yield**: error-blind APN geometry produced 936/940/932 frozen pairs for seeds 2024/2025/2026, all above the predeclared minimum of 100; 2,808 pairs generated 19,656 seven-variant error rows.
- **Gate evidence**: full improved controlled-support MSE by `-0.1199%` versus the required `+5%`; full-minus-raw macro MSE was `0.0003818` with 95% CI `[-0.0005169, 0.0013172]`; full was significantly worse than shuffled with `0.0013241` and 95% CI `[0.0003639, 0.0023545]`; full also failed to significantly beat random features.
- **Passed constraints only**: native regression was `0.3608%` (≤1%), parameter overhead `1.0745%` (<5%), and frozen 100-step time overhead `3.1295%` (<5%).
- **Decision**: verdict is `ABANDON`. Per the frozen protocol, do not run HumanActivity, USHCN, or t-PatchGNN and do not add or tune model modules; proceed only with audit, failure analysis, and packaging.
- **Audit implementation plan**: `research[E]-coding` adds a fail-closed `runner audit` action for all manifests, arrays, metrics, shifts, CUDA fields, provenance, and cross-variant mask equality; no training/evaluation result is changed.
- **How to Run check**: add `runner audit` between `stage-a` and `controlled`; aggregation/package commands remain unchanged.

## 2026-07-21 04:04 CST — Full audit and failure report verified

- **Audit result**: `runner audit` returned `PASS` for 21/21 training manifests, 63/63 evaluation views, 21/21 run-level checks, and 63/63 cross-variant fingerprints with zero discrepancies; all formal runs use project commit `0bf46fc9d6cb00f70ffe110df8d48d4c3a592037` and patch SHA-256 `00D8D59221D1580EE2B718365325BD69945DC2C103B0C23D7F93F9365E301746`.
- **Shift evidence**: every variant/seed preserves 1,199 sample IDs and 281,425 original history observations; exact MCAR and matched burst each request and remove 69,711 points, with unchanged targets/target masks/time marks and zeroed history-only removals.
- **Aggregation contract**: `run_aggregation` now requires the audit to pass before regenerating controlled-support, patient/bootstrap, gate, and report artifacts.
- **Report**: `artifacts/REPORT_CN.md` now records readable gate evidence, audit provenance, five bounded failure interpretations, and the frozen decision to stop all conditional extensions.
- **Validation**: audit/aggregate/controlled targeted tests completed with `10 passed`; the final full boundary/parity/shift/package suite completed with `43 passed in 10.68s`.
- **Known issue resolved**: controlled-support yield is no longer unknown; all seeds exceed the predeclared minimum by more than 9×.
- **How to Run check**: `runner audit`, `runner aggregate`, and `runner package` are current below.

## 2026-07-21 04:12 CST — Final delivery freeze

- **Outcome freeze**: README and `REPORT_CN.md` state the same `ABANDON` verdict and explicitly record that HumanActivity, USHCN, and t-PatchGNN were not run.
- **Required outputs**: the delivery selection contains source/config/tests/scripts, the 11-file APN patch, compact logs, 21 training + 63 evaluation manifests, 63 prediction/target/target-mask/sample-ID sets, deletion audits, statistics, gate, full audit, and Chinese report.
- **Archive dry run**: 693 members and 691 manifest records passed member-set, per-member SHA-256, `testzip`, formal-count, compact-log-coverage, and forbidden-member checks; no `.conda`, dataset, vendor source, checkpoint, cache, secret, or `packages/` member was present.
- **Git boundary**: push only code/docs plus compact JSON/CSV/Markdown results; exclude packages, environments, raw/processed data, full vendor source, checkpoints, caches, raw arrays, and raw Stage A logs. Recheck every candidate for the 100 MB limit before commit.
- **How to Run check**: after this documentation freeze, rerun `runner package` once and verify its external ZIP SHA-256 without changing packaged inputs.

## 2026-07-21 04:16 CST — Post-freeze regression passed

- **Final code validation**: after the audit-gated aggregation and enhanced report generator were frozen, the complete boundary/parity/evidence/shift/runner/aggregate/package suite completed with `43 passed in 12.56s`.
- **Delivery rule**: regenerate the archive once more only to capture this current compact smoke log; make no further packaged-input edits afterward.

## Known Issues

- [ ] Official APN repository has no clearly detected top-level license; do not commit or package its source.
- [ ] Real controlled support-pair yield is unknown until PhysioNet is processed.

## How to Run

### Environment Setup

```powershell
conda activate C:\Users\qintian\Desktop\msn2\.conda\envs\evipatch
```

Verify the existing environment without rebuilding it; install only genuinely missing dependencies when explicitly needed:

```powershell
& .\code\scripts\setup_environment.ps1
# Optional only for a confirmed missing dependency:
& .\code\scripts\setup_environment.ps1 -InstallMissing
```

The boundary-checked wrappers are preferred for normal operation:

```powershell
& .\code\scripts\apply_apn_patch.ps1
& .\code\scripts\run_smoke.ps1
& .\code\scripts\monitor_stage_a.ps1
& .\code\scripts\run_stage_a.ps1
& .\code\scripts\package_results.ps1
```

### Experiment Commands

Set the project-owned source path for the current PowerShell session, then invoke the runner with the environment's absolute Python executable:

```powershell
$evipatchPython = 'C:\Users\qintian\Desktop\msn2\.conda\envs\evipatch\python.exe'
$env:PYTHONPATH = 'C:\Users\qintian\Desktop\msn2\code\src'
& $evipatchPython -m evipatch.runner validate
& $evipatchPython -m evipatch.runner status
& $evipatchPython -m evipatch.runner smoke
& $evipatchPython -m evipatch.runner timing
& $evipatchPython -m evipatch.runner stage-a
& $evipatchPython -m evipatch.runner audit
& $evipatchPython -m evipatch.runner controlled
& $evipatchPython -m evipatch.runner aggregate
& $evipatchPython -m evipatch.runner package
```

- **`validate`**: checks the seven-variant matrix, fixed seeds, local mutable paths, Python executable, APN checkout, and pinned upstream commit; writes nothing.
- **`status`**: prints the completion state of every seed checkpoint and native/MCAR/burst evaluation; writes nothing.
- **`smoke`**: runs the project pytest suite and writes `logs/smoke/pytest.log`.
- **`timing`**: benchmarks exactly 100 measured real-P12 optimizer steps after warm-up for all seven variants and writes `artifacts/timing_100_steps.json`.
- **`stage-a`**: acquires `results/stage_a/.stage_a.lock`, then trains 21 checkpoints and evaluates 63 views strictly sequentially; outputs checkpoints under `results/`, raw logs under `logs/`, arrays/metrics beside each checkpoint, and one JSON manifest per process.
- **`audit`**: fails closed unless all 21/63 manifests, arrays, metric recomputations, shift audits, cross-variant masks, CUDA peaks, hashes, and provenance checks pass; writes `artifacts/stage_a_audit.json`.
- **`controlled`**: freezes error-blind real-data support pairs from each APN checkpoint/native history and scores identical pair IDs for all variants.
- **`aggregate`**: produces patient errors, seed summaries, hierarchical paired bootstrap results, and `gate_decision.json` under `artifacts/`.
- **`package`**: creates the final filtered ZIP and `SHA256SUMS.csv` under `packages/`.

## 2026-07-21 -- New track-aligned attempt series

- **ResearchPilot phases**: re-entered exploration through implementation after the
  frozen EviPatch `ABANDON` verdict; the current route is documented separately in
  `docs/dualcross_route.md` so the old audit record remains immutable.
- **Scope override**: target `Edge Computing, IoT and Digital Twins`; use official
  frozen APN checkpoints, reproduce no additional baseline, and run only one main
  experiment plus mechanism ablations. APN remains the baseline for five distinct
  structural attempts before any possible switch.
- **Attempt 1**: implemented and tested a 410-parameter shared CorePatch adapter.
  Its seed-2024 MSE improved only 0.163% (0.312922 to 0.312412), below the frozen
  1% threshold, so it is recorded as failed rather than promoted.
- **Iteration diagnosis**: training/validation-only ridge analysis found 1.127%
  validation improvement from strongly shrunk cross-sensor residual prediction,
  compared with roughly 0.413% from self-only calibration. No attempt-2 test
  target was inspected during this design decision.
- **Attempt 2 frozen design**: DualCross combines CSFI before the frozen decoder
  and SRG after the decoder. Planned ablations are APN, CSFI, SRG, and CSFI+SRG.
- **How to Run**: attempt 2 will use a dedicated boundary-checked runner; its exact
  command is appended only after the implementation passes unit tests.

### 2026-07-21 -- DualCross implementation smoke passed

- **Code**: added an isolated `dualcross` package, a dedicated runner and frozen
  config. CSFI uses diagonal-masked low-rank attention before a copied frozen APN
  decoder; SRG fits a directed diagonal-free ridge graph after decoding, with its
  shrinkage selected only on validation.
- **Tests**: DualCross plus retained attempt-1 regression tests completed with
  `7 passed`; coverage includes output shape, zero attention diagonal, finite
  gradients, decoder freezing/parity, synthetic graph recovery, and root escape.
- **How to Run**:

  ```powershell
  $env:PYTHONPATH = 'C:\Users\qintian\Desktop\msn2\code\src'
  & 'C:\Users\qintian\Desktop\msn2\.conda\envs\evipatch\python.exe' `
    .\code\scripts\run_dualcross.py --seed 2024
  ```

### 2026-07-21 -- Five-attempt APN search completed

- **ResearchPilot phases**: `research[F]-iteration` recorded each failure without
  test-set retuning; `research[B]-idea` and `research[C]-experiment` froze the next
  structural hypothesis before implementation.
- **Attempt 2**: DualCross reached a 0.678% seed-2024 MSE reduction but missed the
  predeclared 1% threshold.
- **Attempt 3**: the pseudo-observation bridge worsened validation MSE by 8.75%
  and was terminated before test evaluation.
- **Attempt 4**: VarGraph's strongest single module reduced MSE by 1.116%, but the
  combined two-module route achieved only 1.076% and therefore failed the frozen
  requirement that the complete method beat both ablations.
- **Attempt 5**: EdgeTwinCal passed the seed-2024 kill-test and then seeds 2025 and
  2026. The complete ledger, including frozen decision rules, is retained in
  `docs/dualcross_route.md`.

### 2026-07-21 -- EdgeTwinCal main experiment and ablation frozen

- **ResearchPilot phases**: `research[D]-implementation`, `research[E]-coding`,
  and `research[F]-iteration` produced a self-contained final package around the
  unchanged existing checkpoint produced by the official APN implementation.
- **Modules**: Sensor Latent Residual Head (SLRH) reads frozen per-sensor latents
  in parallel with the shared decoder; Cross-Forecast Graph (CFG) applies a
  diagonal-free directed correction after the intermediate forecast. Both are
  closed-form ridge fits, with penalties selected on validation only.
- **Main result**: three-seed APN MSE is 0.312331 +/- 0.000512 and EdgeTwinCal MSE
  is 0.309058 +/- 0.000494, a 1.048% reduction. MAE improves by 0.645%.
- **Ablation**: SLRH, CFG, and full reduce MSE by 0.553%, 0.591%, and 1.048%,
  respectively. Full beats both single modules for every reported aggregate MSE.
- **Inference**: a 10,000-replicate seed-block and patient-level paired bootstrap
  gives full-minus-APN MSE -0.003793 with 95% CI [-0.005619, -0.002130].
- **Cost**: the final cached closed-form fits take 1.40--1.42 seconds per seed on
  one RTX 4090. Each complete audit/ablation state is 0.256 MB.
- **Cleanup**: deleted only the failed CorePatch, DualCross, and VarGraph source,
  configs, scripts, tests, and result trees. Historical audit documents and the
  successful EdgeTwinCal results were preserved; the old `msn` directory was not
  accessed.

### 2026-07-21 -- IEEE paper draft compiled and visually reviewed

- **ResearchPilot phases**: `research[G.0]-plan` through `research[G.6]-conclusion`
  produced a concise English draft; `research[G.7]-review` compiled and rendered
  the supplied IEEE conference format for page-level inspection.
- **Track framing**: the draft targets `Edge Computing, IoT and Digital Twins` and
  presents fast recalibration of a frozen irregular-sensor digital twin, rather
  than repeating the prior Big Data and AI anomaly-detection submission.
- **Comparison policy**: no additional baseline was reproduced. Published APN,
  t-PatchGNN, and GraFITi values are clearly labeled as paper-reported context;
  causal claims use only the paired frozen-checkpoint experiment and ablations.
- **Artifacts**: source is `docs/manuscripts/paper.tex`, references are in
  `docs/manuscripts/references.bib`, and the compiled four-page draft is
  `docs/manuscripts/paper.pdf`. Reproducible figures are generated by
  `notebooks/figures.ipynb`.

### Current EdgeTwinCal How to Run

```powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path

# Main experiment and four-way ablation on the existing frozen checkpoint cache.
& $edgeTwinPython .\code\scripts\run_edgetwincal.py --seed 2024
& $edgeTwinPython .\code\scripts\run_edgetwincal.py --seed 2025
& $edgeTwinPython .\code\scripts\run_edgetwincal.py --seed 2026

# Three-seed summary and hierarchical paired bootstrap.
& $edgeTwinPython .\code\scripts\aggregate_edgetwincal.py

# Final implementation tests.
& $edgeTwinPython -m pytest -q .\code\tests\test_edgetwincal.py

# IEEE draft (Tectonic downloads only standard TeX assets on first use).
Set-Location .\docs\manuscripts
& tectonic paper.tex
```

### 2026-07-21 -- Formal ablation rerun and APN-aligned manuscript revision

- **ResearchPilot phases**: `research[G.2]-experiments`,
  `research[G.3]-abstract`, and `research[G.7]-review`; the PDF workflow was
  used to compare the supplied submission, the official APN paper, and the latest
  rendered draft.
- **Rerun**: APN/SLRH/CFG/full were recomputed sequentially for seeds
  2024/2025/2026 from the frozen feature caches. The three-seed command plus
  aggregation finished in 15.7 seconds and reproduced every MSE/MAE value.
- **Ablation visualization**: `notebooks/figures.ipynb` now generates a two-panel
  figure containing MSE/MAE improvement bars with across-seed error bars and
  seed-wise paired MSE curves. The exact mean +/- standard deviation values remain
  in Table II.
- **Baseline alignment**: Table I now mirrors APN Table 2's mean +/- standard
  deviation convention, separates published five-seed context from our paired
  evaluation, and includes Warpformer, t-PatchGNN, GraFITi, and APN without
  reproducing them.
- **Provenance correction**: the reused APN artifacts are described as pre-existing
  checkpoints trained locally with the released implementation, not checkpoints
  published by APN's authors.
- **Protocol disclosure**: the draft records the released P12 implementation's
  approximate 81/9/10 split, train/validation drop-last, and full-data
  standardization. All four paired variants inherit the same behavior.
- **Review correction**: because five structural routes inspected the same test
  set, the study is labeled exploratory; the patient-macro bootstrap estimand is
  distinguished from target-micro MSE, and edge-device claims are removed.
- **Paper output**: the revised four-page IEEE PDF compiles without overfull boxes.

### 2026-07-21 -- EdgeTwinCal final consistency check

- **Metadata**: result manifests now distinguish 3,888 free CFG coefficient slots
  from 3,886 nonzero fitted coefficients and identify the checkpoint provenance
  accurately.
- **Regression**: the final EdgeTwinCal tests completed with `4 passed in 1.65s`;
  all JSON and Notebook artifacts parse successfully.
- **PDF verification**: the latest `paper.pdf` is a four-page US-letter document;
  every page was rendered after the final edit and no clipping, overlap, or
  overfull box remains. The byte-identical delivery copy is
  `output/pdf/EdgeTwinCal_draft.pdf`.
- **Repository check**: `git diff --check` passed and all failed-route source and
  result directories remain absent.

### 2026-07-21 -- GitHub synchronization bundle prepared

- **Current route**: updated the repository landing page to make EdgeTwinCal the
  active project while retaining EviPatch only as a historical negative audit.
- **Baseline provenance**: added `docs/baselines/APN_AAAI2026.md` with the
  official article, publisher PDF, DOI, GitHub repository, and pinned commit.
- **Versioned deliverables**: final source, tests, configuration, compact
  three-seed CSV/JSON results, patient-level paired data, figures, Notebook,
  IEEE source/template/class/bibliography, and compiled draft are selected.
- **Deliberate exclusions**: publisher-owned APN PDF, complete `vendor/APN`
  checkout, local caches/checkpoints/data/environments, private absolute paths,
  and secrets. (This continuation repairs the previously truncated final line;
  no historical entry was rewritten.)

### 2026-07-21 -- Confirmatory lab handoff accepted and M0 audited

- **ResearchPilot phases**: entered research[D]-implementation, followed by
  research[E]-coding and research[F]-iteration only after the active design is
  frozen. No manuscript rewrite is authorized by the handoff.
- **Authority**: the read-only
  EdgeTwinCal_Lab_Experiment_Handoff_20260721 bundle was verified 18/18 against
  SHA256SUMS.txt. Its protocol now supersedes the earlier three-seed pilot scope.
- **Isolation**: all writes remain under msn2; the sibling msn, HOME, and
  CODEX_HOME remain untouched. Work continues on branch
  lab/msn2026-full-benchmark.
- **Cancelled download**: removed the assistant-created ignored cache file
  cache/reference_papers/APN_AAAI2026.pdf after the user said no download was
  needed. The official source remains linked; no publisher PDF is committed.
- **Assets**: P12 data and APN checkpoints 2024--2026 are present. Seeds
  2027/2028, HumanActivity, USHCN, MIMIC-III, and real edge hardware are absent
  and are recorded as BLOCKED until legally/physically available.
- **Environment**: Python 3.11.13, PyTorch 2.6.0+cu124, CUDA available on one RTX
  4090. APN is pinned at f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4.
- **Legacy G0 evidence**: existing five EdgeTwinCal tests passed in 1.82 s.
  Cache predictions/targets/masks/sample IDs match saved outputs exactly, and
  recomputed APN metrics differ by at most 5.55e-17. The runner was deliberately
  not called because it overwrites legacy outputs.
- **Handoff hash note**: project compact results use CRLF and handoff references
  use LF. After newline normalization all four files are text-identical; no
  result drift exists.
- **Protocol repairs**: fixed the active requirements/idea/design boundary,
  separated legacy_v1 from msn2026_v1, defined correct CFG anchor/source
  semantics, restricted crossed inference to reliable group IDs, split harmful
  from safety-inconclusive, and renamed legacy pass semantics conceptually.
- **Implementation plan**: FIX-01 config and FIX-06 schema first, then cache
  provenance, strict split/normalization, generic ridge and controls, segmented
  timing, crossed inference, complete pre-test gate, frozen registry, and only
  then once-only test evaluation.

### Confirmatory campaign How to Run (pre-test phase)

```powershell
$edgeTwinPython = (Resolve-Path ''.\.conda\envs\evipatch\python.exe'').Path
$env:PYTHONPATH = (Resolve-Path ''.\code\src'').Path

# Safe pre-test suite. This must pass before any confirmatory test command exists.
& $edgeTwinPython -m pytest -q .\code\tests\test_edgetwincal.py

# The msn2026_v1 commands will require one resolved config and a frozen protocol
# ledger. Until FIX-01--06 are implemented and G0/G1 pass, do not open new test.
```

### 2026-07-21 -- FIX-01 through FIX-06 foundation and mechanism controls

- **FIX-01 / configuration**: added code/configs/msn2026/default.json and
  edgetwincal/config.py. The only overrides are registered dataset/seed/variant
  selectors; paths, five seeds, six-point ridge grids, protocols, variants,
  bootstrap, timing, and gates are locked. Current default config SHA256 is
  6228502eacd2bbce02621281a2e8e8056728afc7629b28cd5424ed154d2a7e35.
- **FIX-06 / run schema**: added edgetwincal/schema.py with atomic
  created/running/complete/failed manifests, required-file hashing, and fail-
  closed aggregation eligibility.
- **FIX-02 / provenance**: added edgetwincal/provenance.py with every handoff key
  field, digest filenames, payload hash/length, exact stale comparison, partial
  and corrupt rejection, a scoped lock, and same-directory atomic replacement.
  Sixty-four mutation/corruption tests passed.
- **FIX-03 / protocol**: added edgetwincal/protocol.py with salted deterministic
  group splits, train-only observed normalizers, privacy-safe manifests, and
  USHCN station-overlap audit. Six synthetic leakage tests passed without
  loading a real test split.
- **Shared ridge semantics**: added ridge.py and refactored latent.py. Feature
  scales use a 1e-6 floor; intercepts are unpenalized; validation micro MSE
  chooses one alpha globally.
- **CFG semantic fix**: archived the coupled legacy implementation as
  legacy_graph_v1.py. The active graph.py accepts separate base_to_correct and
  source_forecasts. Full therefore uses an SLRH anchor with frozen APN sources.
- **Controls**: added controls.py, joint.py, shuffle.py, and variants.py for
  Bias-only, Self-affine, Reverse, complete 6x6 Joint, Full-Diagonal with exact
  SLRH reuse, CrossShuffle, and LatentShuffle. The P12 Joint coefficient-slot
  invariant is 6,480.
- **V03**: added decoder_refit.py. Each of the six AdamW LR x weight-decay
  candidates restarts from the identical checkpoint; only exact decoder names
  are trainable and every non-decoder tensor/buffer is byte-checked.
- **FIX-05**: added timing.py with CPU/CUDA labels, synchronization boundaries,
  phase records, warm inference, and serialized-state measurement.
- **FIX-04**: added statistics.py with SSE/SAE/N cells, pooled micro metrics,
  shared-multiplicity crossed group x checkpoint bootstrap, Holm adjustment,
  and distinct harmful versus safety-inconclusive labels.
- **Once-only test guard**: added campaign.py. A persisted frozen ledger must
  commit config, registries, manifests, code, statistics, timing, patch, and
  environment hashes before it can issue a one-use test token.
- **Tests so far**: mechanism controls 8 passed; statistics/timing 8 passed;
  campaign ledger 3 passed; APN upstream/patched parity passed at atol 1e-8 and
  rtol 1e-7. All work used synthetic or legacy-parity inputs; no new test split
  was opened.


### 2026-07-21 -- Strict USHCN protocol preparation

- Added `edgetwincal/strict_ushcn.py`. It audits the official TSDM/APN fold-0
  station keys, preserves a group-disjoint official fold, and otherwise applies
  the locked salted-SHA256 floor 80/10/remainder station repair.
- Value normalization is fitted only from finite observations belonging to
  frozen training stations. APN's `seq_len=150` task boundary is retained as a
  `149.5` cutoff, times use the task-compatible frozen global-time-maximum
  transform, and exactly the next three irregular rows form the forecast target.
- Public audit, split, normalization, time, dataset, and ledger manifests contain
  salted station hashes and counts only. Strict test construction requires a
  token binding the frozen registry, fold audit, split, normalizer, and time
  scale hashes.
- Added eight synthetic tests covering official-fold retention, deterministic
  overlap repair, station non-overlap, train-only statistics, test-extreme
  invariance, APN slicing, frozen-token enforcement, and raw-ID exclusion. The
  combined protocol/P12/USHCN strict suite passes 21 tests. No vendor task was
  instantiated, no dataset was read, and no real test split was opened.

#### Strict USHCN How to Run (synthetic pre-test only)

```powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path
& $edgeTwinPython -m pytest -q -p no:cacheprovider `
  .\code\tests\test_edgetwincal_protocol.py `
  .\code\tests\test_edgetwincal_strict_p12.py `
  .\code\tests\test_edgetwincal_strict_ushcn.py
```
### 2026-07-21 -- Active control plane, phase caches, and data acquisition

- **Legacy isolation**: moved the exploratory runner/runtime/aggregator behind
  explicit `legacy_*_v1` modules. The active CLI is config-driven and provides
  audit, ledger create/freeze, status, explicit-registry aggregation, and a
  token-gated evaluator boundary. The evaluator remains fail-closed until the
  real campaign integration is complete.
- **Train/validation-only APN trainer**: added a project-owned deterministic
  Adam trainer with the released APN scheduler, masked micro-MSE validation,
  patience 10, atomic best-state writes, curve/timing/CUDA manifests, and no
  test-loader argument.
- **Phase-separated schema-3 caches**: the active runtime now supports a
  train+validation fit cache before protocol freeze and a separate test cache
  only after the once-opened campaign gate. The combined three-split form
  remains accepted solely for complete parity/audit bundles. Arbitrary partial
  split sets are rejected.
- **Regression repair**: fixed the accidental indentation of the explicit CFG
  source forecasts in the legacy unit test; the original five tests pass again.
  The active --help test now checks that module state is unchanged rather than
  incorrectly assuming no earlier parity test imported APN in the same process.
- **Public assets**: downloaded and verified HumanActivity from the UCI URL
  pinned by APN (25 participant records; loader MD5 matched) and USHCN from the
  pinned GRU-ODE-Bayes source (raw SHA256
  `671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf`,
  semantic shape 350665 x 5). The local USHCN parquet byte hash differs from
  the old reference because parquet serialization is version-dependent; raw
  identity and semantic shape are therefore recorded separately.
- **Authorized MIMIC acquisition**: after the user confirmed access authority,
  downloaded only the nine GRU-ODE-Bayes-required MIMIC-III v1.4 tables from
  the specified Hugging Face mirror into ignored project storage. Their headers
  and standard row counts match v1.4 (for example LABEVENTS 27,854,055 data
  rows and INPUTEVENTS_MV 3,618,991), and every file has a local SHA256 record.
- **MIMIC fail-closed finding**: the upstream DataMerging notebook additionally
  consumes a saved `UNIQUE_ID_dict.csv` created by an unseeded random shuffle.
  It is absent from APN, GRU-ODE-Bayes, and the mirror. A deterministic strict
  derivative may be built, but it cannot be labeled released-code parity unless
  the final APN reference shape and SHA256 both match.


### 2026-07-21 -- Frozen multi-seed campaign runner integrated

- **ResearchPilot phases**: research[E]-coding implemented the frozen campaign
  boundary; research[F]-iteration exercised release and strict synthetic
  campaigns and repaired state-replay and CLI exit contracts.
- **Pre-test manifests**: `campaign_pretest.prepare_pretest_manifests` writes
  protocol, split, and normalizer manifests without constructing test data.
  `fit_and_freeze_registry` consumes only explicit schema-3 train/validation
  caches, fits the selected variants sequentially, stores tensor/primitive state
  with `torch.save`, and binds the total fitted-registry SHA256 for the ledger.
- **Backbone/cache boundary**: `campaign_extract.prepare_fit_cache` either loads
  the registered APN checkpoint or runs the train/validation-only APN trainer,
  then extracts exactly `train` and `val`. Its API exposes no test loader,
  split selector, or test token.
- **Once-only evaluation**: `campaign_evaluate.extract_test_cache_after_open`
  requires an active ledger token before constructing a test loader and writes a
  token-bound access sidecar. `evaluate_campaign_once` hash-verifies every
  fitted seed/state before accepting the token, evaluates all seeds and variants
  sequentially under one opening, records every unrun cell on failure, and
  closes the opening only after the whole campaign or a consumed-test failure.
- **Strict adapter**: generic `FrozenAPNTestLedgerToken` openings are converted
  to the dataset-specific P12/USHCN token only after the prepared strict public
  split/normalizer manifests and their ledger hashes match.
- **Controls**: strict P12/USHCN state replay covers APN, SLRH, CFG, Full, V01,
  V02, V03, V07, V08, V10, V11, and V12. V03 uses the exact cached APN decoder
  topology and safe `weights_only=True` reload.
- **CLI**: active `experiment.py` now exposes `pretest prepare`, `pretest
  fit`, and complete multi-seed `evaluate`; missing fitted/test registries
  return the explicit artifacts-required exit code without consuming test data.
- **Validation**: the focused APN bridge/trainer/campaign/runtime/strict suite
  passed 57 tests; the complete project suite passed 231 tests in 20.28 seconds.
  The real APN checkout already exists at the pinned commit
  `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`, so it was not downloaded again.

#### Frozen campaign How to Run

```powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path
$config = '.\code\configs\msn2026\default.json'

# 1. Create public protocol/split/normalizer manifests without opening test.
& $edgeTwinPython -m edgetwincal.experiment pretest prepare `
  --config $config --dataset USHCN --seed 2024 `
  --dataset-id USHCN --protocol-id release_parity --preparation-seed 2024

# 2. After project-owned train/validation cache extraction, fit every selected
# seed and variant from the explicit entries file and freeze their states.
& $edgeTwinPython -m edgetwincal.experiment pretest fit `
  --config $config --dataset USHCN `
  --dataset-id USHCN --protocol-id release_parity `
  --entries .\results\edgetwincal_msn2026_v1\protocol\USHCN\release_parity\fit_entries.json `
  --output .\results\edgetwincal_msn2026_v1\protocol\USHCN\release_parity\fitted_registry.json `
  --device cpu

# 3. Create/freeze the ledger with the fitted registry SHA, open the cell once,
# and extract each seed's test cache through extract_test_cache_after_open.
# The token is passed in memory/CLI only and is never persisted in run artifacts.

# 4. Evaluate the complete frozen seed registry sequentially and consume opening.
& $edgeTwinPython -m edgetwincal.experiment evaluate `
  --ledger .\results\edgetwincal_msn2026_v1\protocol\protocol_ledger.json `
  --cell-id 'USHCN|release_parity|fold-0' --token '<ONCE_ONLY_TOKEN>' `
  --fitted-registry .\results\edgetwincal_msn2026_v1\protocol\USHCN\release_parity\fitted_registry.json `
  --test-cache-registry .\results\edgetwincal_msn2026_v1\protocol\USHCN\release_parity\test_cache_registry.json `
  --device cpu

# Regression suite used for the integration freeze.
& $edgeTwinPython -m pytest code\tests -q -p no:cacheprovider
```


### 2026-07-21 -- Frozen train/validation fit-cache campaign control plane

- **ResearchPilot phases**: research[E]-coding added an isolated orchestration
  layer; research[F]-iteration verified strict cache reuse, provenance drift
  rejection, partial-artifact failure, deterministic ordering, and the absence
  of a test-construction surface.
- **Files**: added `code/src/edgetwincal/fit_cache_campaign.py`,
  `code/scripts/prepare_edgetwincal_fit_caches.py`, and
  `code/tests/test_edgetwincal_fit_cache_campaign.py`. The existing campaign
  runner and backbone campaign were not modified.
- **Backbone contract**: every cache cell reuses
  `backbone_campaign.prepare_cell`, the exact frozen `pytorch_model.bin`,
  `configs.yaml`, and `train_manifest.json`. Both the 22 newly trained
  manifests and three legacy-imported P12 manifests enter through the same
  `validate_reusable_checkpoint` contract; there is no legacy cache shortcut.
- **Real provenance**: schema-3 cache manifests bind the current project HEAD,
  composite raw-data and processed-data hashes, and composite loader/extractor
  source hashes. Composite hashes include stable project-relative paths, file
  sizes, and individual SHA256 values.
- **Isolation and determinism**: all paths pass the msn2 root guard. The wrapper
  establishes `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing APN/PyTorch
  runtime code and rejects a conflicting value.
- **No-test boundary**: execution constructs only `train` and `val` through
  the existing fit-loader runtime, then calls
  `campaign_extract.prepare_fit_cache(train_backbone=False)`. The module has
  no test-loader or test-ledger-token API. MIMIC-III release parity remains
  explicitly `BLOCKED[missing_author_mapping]`.
- **Reuse semantics**: a complete per-seed control manifest commits the frozen
  assets, provenance, cache envelope, cache sidecar, payload hash, and manifest
  digest. Reuse re-reads and hash-validates the schema-3 payload. Any source,
  dataset, checkpoint, config, manifest, sidecar, or cache drift fails closed;
  orphan cache files are never overwritten.
- **Pretest handoff**: after all five frozen seeds of a cell are ready, the
  control plane writes exactly `{"entries": [...]}` to the cell-local
  `fit_entries.json`, directly consumable by `experiment pretest fit`.
- **Validation**: focused tests passed `6 passed in 1.91s`; the safe wrapper
  `--help` and a read-only reordered-seed plan both returned exit code 0.
  Neither validation command used `--execute`, so no real cache extraction or
  test access occurred.
- **Resource estimate**: expected aggregate cache storage is approximately
  1.0--1.2 GiB for 25 cells (P12 dominates); allow 1.5 GiB free space for
  envelopes and temporary atomic writes. Expected sequential extraction time on
  the RTX 4090 is roughly 45--90 minutes, with actual time dependent on
  HumanActivity padding and disk hashing.

#### Fit-cache Campaign How to Run

```powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path

# Read-only 25-cell plan: five datasets/protocol cells x five seeds.
& $edgeTwinPython .\code\scripts\prepare_edgetwincal_fit_caches.py

# Full sequential extraction/reuse campaign. This includes P12 release/strict,
# USHCN release/strict, and HumanActivity release; MIMIC is deliberately absent.
& $edgeTwinPython .\code\scripts\prepare_edgetwincal_fit_caches.py `
  --dataset P12 --dataset USHCN --dataset HumanActivity `
  --protocol release_parity --protocol strict_p12 --protocol strict_ushcn `
  --seed 2024 --seed 2025 --seed 2026 --seed 2027 --seed 2028 `
  --device cuda:0 --execute

# Validation-only reuse audit after extraction; missing cache is BLOCKED and
# every existing cache is rehashed.
& $edgeTwinPython .\code\scripts\prepare_edgetwincal_fit_caches.py `
  --dataset P12 --dataset USHCN --dataset HumanActivity `
  --protocol release_parity --protocol strict_p12 --protocol strict_ushcn `
  --seed 2024 --seed 2025 --seed 2026 --seed 2027 --seed 2028 `
  --device cuda:0 --reuse-only --execute

# Example direct handoff into pretest fitting for one complete cell.
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path
& $edgeTwinPython -m edgetwincal.experiment pretest fit `
  --dataset-id P12 --protocol-id release_parity `
  --entries .\results\edgetwincal_msn2026_v1\protocol\P12\release_parity\fit_entries.json `
  --output .\results\edgetwincal_msn2026_v1\protocol\P12\release_parity\fitted_registry.json `
  --device cpu

# Focused regression suite.
& $edgeTwinPython -m pytest `
  .\code\tests\test_edgetwincal_fit_cache_campaign.py `
  -q -p no:cacheprovider
```


### 2026-07-21 -- APN backbone train/validation control plane and legacy import

- **ResearchPilot phases**: research[E]-coding implemented the isolated
  train/validation control plane; research[F]-iteration added real-manifest
  reconstruction, fail-closed reuse, deterministic CUDA auditing, and the
  verified legacy compatibility path.
- **Files**: added code/src/edgetwincal/backbone_campaign.py,
  code/scripts/run_edgetwincal_backbones.py, and
  code/tests/test_edgetwincal_backbone_campaign.py.
- **Matrix**: the ordered default covers P12 release/strict, USHCN
  release/strict, and HumanActivity release for seeds 2024--2028. MIMIC-III
  release remains BLOCKED[missing_author_mapping] until the released author
  UNIQUE_ID_dict.csv exists.
- **No-test boundary**: preparation and training expose only train/validation
  APIs; production code contains no test-loader call. Every control artifact
  records test_constructed=false. Legacy verification constructs no loader at
  all and records loaders_constructed=false.
- **Checkpoint reuse**: native checkpoints must pass
  validate_reusable_checkpoint, including exact config content/hash, argv,
  Adam/MSE/epoch/patience policy, and atomic weights-only checkpoint identity.
  An orphan checkpoint, manifest, config, control file, or log is never
  overwritten.
- **Legacy P12 compatibility correction**: release-parity seeds 2024--2026 are
  imported as verified_legacy_import, not retrained and not represented by a
  fabricated new train_manifest.json. Each proof pins the copied checkpoint,
  copied configs.yaml, original Stage A train manifest, original training log,
  official train-only argv, successful process record, and project/APN/patch/
  Python/PyTorch/CUDA/GPU provenance.
- **Unified downstream API**:
  validate_frozen_checkpoint_identity(config=..., cell=..., seed=...,
  checkpoint_path=..., train_manifest_path=None, configs_yaml_path=None) is
  read-only. Native results use verification_mode=native_train_manifest;
  legacy P12 results require the sibling verified_legacy_import.json, recompute
  the complete source proof, and use
  verification_mode=verified_legacy_import. Cache preparation must use this
  API rather than require a sibling native train manifest unconditionally.
- **Real reuse-only audit**: P12 release seeds 2024, 2025, and 2026 all returned
  verified_legacy_import twice with checkpoint SHA256 values
  927b3e540b7450d37897fd1a78dbbbaa1ff93152d1456c350d22f9f1cc41e3d4,
  945ef93c856c73ef59a39d6c27baaf7599cbebff9b42b55b45381ac255ed6bb9,
  and
  36303d6b1f2fa449e35e93bb9bfcc7446641a78580367a97c9c45847117ac641.
  No training or test access occurred and no new train manifest was created.
- **Validation**: py_compile passed; the backbone plus fit-cache combination
  passed 15 tests, the related runtime/bridge/trainer/strict/cache group passed
  62 tests, and the full EdgeTwinCal selection passed 185 tests with 62
  deselected. The only warning was pytest's inability to update its optional
  root cache; test execution itself passed.

#### Backbone Campaign How to Run

~~~powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path

# Read-only ordered plan; imports no APN or dataset module and starts no training.
& $edgeTwinPython .\code\scripts\run_edgetwincal_backbones.py

# Audit the three pinned legacy P12 release checkpoints. This may create or
# validate only verified_legacy_import.json and backbone_campaign_manifest.json.
& $edgeTwinPython .\code\scripts\run_edgetwincal_backbones.py --dataset P12 --protocol release_parity --seed 2024 --seed 2025 --seed 2026 --execute --reuse-only

# Focused regression, including native/legacy unified identity verification.
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path
& $edgeTwinPython -m pytest .\code\tests\test_edgetwincal_backbone_campaign.py .\code\tests\test_edgetwincal_fit_cache_campaign.py -q -p no:cacheprovider
~~~

### 2026-07-21 -- Unified backbone identity integrated into fit-cache extraction

- Replaced the fit-cache control plane's unconditional sibling
  `train_manifest.json` requirement with the read-only
  `validate_frozen_checkpoint_identity` API.
- The 22 native checkpoints still require and validate their original training
  manifests. P12 release seeds 2024--2026 instead revalidate
  `verified_legacy_import.json` plus the pinned Stage A manifest, log,
  checkpoint, config, APN patch, and runtime provenance; no replacement
  training manifest is created.
- No loader or test split was constructed during this integration.
- `py_compile` passed and the combined backbone/fit-cache regression suite
  passed 15/15.

### 2026-07-21 -- Sealed confirmatory return, gate decision, and package

- **ResearchPilot phases**: research[E]-coding supplied the closed result/report
  and deterministic packaging entrypoints; research[F]-iteration audited the
  crossed statistics, gate semantics, zero-target handling, failure mechanism,
  workbook formulas, and archive verification. No paper-writing phase was
  entered.
- **Closed experiment state**: five runnable cells are complete:
  HumanActivity release 20/20, P12 release 20/20, USHCN release 20/20,
  strict P12 60/60, and strict USHCN 60/60. All five ledgers contain exactly
  one closed opening, are sealed, and record token_persisted=false.
- **Pre-test evidence**: G0 and G1 are PASS in every cell. The terminal audit
  rehashed every referenced check and all 180 explicit run manifests.
- **Statistical correction**: globally empty P12 target groups are retained in
  raw SSE/SAE/N but excluded from crossed multiplicities. Strong classification
  is assigned only after Holm and requires the primary adjusted p below .05;
  MAE safety takes precedence. Intervals remain raw percentile 95% CIs while
  Holm applies only to one-sided bootstrap p-values.
- **Formal result**: G2 FAIL, strict G3 FAIL (P12 strong, USHCN harmful),
  release broad-scope FAIL, and G4 BLOCKED without a real edge target. The
  machine verdict is ABANDON.
- **Failure diagnosis**: train/validation replay shows P12 APN/Full validation
  MSE 0.309850/0.307801, while USHCN shows 1.010812/0.548789 on only 334
  effective targets per checkpoint. A few heavy-tailed groups dominate the
  apparent USHCN validation gain; the unbounded correction reverses on the
  sealed test. State/cache/checkpoint hashes and replay invariants pass, so this
  is selection/generalization failure rather than implementation drift.
- **Stop rule**: this is the fifth APN structural route. Same-test retuning and a
  sixth APN route are prohibited. Future work must switch baseline and use a
  new independent target.
- **Artifacts**: `lab_report.py` and `render_edgetwincal_results.py` generate
  `gate_decision.json`, `failure_diagnosis.json`, the Chinese `REPORT_CN.md`,
  two SVGs, and provenance. `build_edgetwincal_tables.mjs` uses
  `@oai/artifact-tool` to generate four CSVs and a five-sheet XLSX. The workbook
  render/inspect pass covered every sheet and found zero formula errors.
- **Packaging**: `package.py` uses a closed whitelist, rejects data/environments/
  caches/checkpoints/NPZ/PT/PDF/vendor/manuscripts/secrets/private absolute
  paths and files over 100 MB, writes deterministic ZIP metadata, and verifies
  CRC plus every member SHA256.

#### Sealed lab return How to Run

```powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path

& $edgeTwinPython .\code\scripts\render_edgetwincal_results.py
& $edgeTwinPython -m pytest .\code\tests -q -p no:cacheprovider
& $edgeTwinPython .\code\scripts\package_edgetwincal.py
```

The table builder additionally requires the Codex bundled Node runtime with a
local `node_modules` junction containing `@oai/artifact-tool`. It is executed
from an ignored project-local QA directory and writes the final CSV/XLSX files
directly under `artifacts/edgetwincal_msn2026_v1/analysis`.

### 2026-07-21 23:00 CST ? EdgeTwinCal-Safe final confirmatory override accepted

- **ResearchPilot phases**: Re-opened Phase C experiment design and Phase D implementation design under the user''s explicit final override; Phase E/F will implement and diagnose the frozen route. No manuscript phase is active.
- **Scope**: Keep the existing APN backbone and compare paired `APN`, `Joint`, original `EdgeTwinCal`, and `EdgeTwinCal-Safe`. Safe adds group-balanced Huber residual fitting, a bounded/shrunk correction, and a validation-only dataset gate with exact APN fallback.
- **New untouched targets**: Predeclared Beijing multi-site PM2.5 and Intel Lab mote-temperature forecasting with five APN seeds each. Existing P12/USHCN/MIMIC outcomes remain diagnostic only and cannot tune the new route.
- **Isolation**: Created the independent `edgetwincal_safe_v1` namespace on branch `lab/edgetwincal-safe`; the sealed `msn2026_v1` code, ledgers, results, and packages remain immutable.
- **How to Run**: No Safe command exists yet. Commands will be added here and to `code/README.md` immediately after the frozen design is implemented and smoke-tested.

### 2026-07-21 23:05 CST ? Safe idea and untouched-target protocol frozen

- **Changed**: `docs/idea_report.md` now defines GRJF and BVSE as the two Safe modules, the four paired main methods, four mechanism ablations, all fixed optimization/envelope grids, the independent validation safety gate, and exact APN fallback.
- **Data protocol**: Froze Beijing 12-station PM2.5 and Intel 54-mote temperature timelines, windows, group identities, five paired seeds, train-only normalization, and once-only sealed tests before downloading or inspecting candidate values.
- **Inference and kill gate**: Froze crossed group-by-checkpoint statistics, two-target positivity, the 1% harm ceiling, 0.1% Joint non-inferiority margin, conditional device timing, and unconditional `ABANDON` on efficacy failure.
- **How to Run**: Still unchanged; implementation commands do not exist yet.

### 2026-07-21 23:12 CST ? Safe implementation design frozen

- **Changed**: Replaced the active header of `docs/implementation.md` with the independent `edgetwincal_safe_v1` guide while retaining the sealed `msn2026_v1` implementation as a provenance appendix.
- **Contracts**: Froze project-owned modules, tensor/data APIs, physical test-shard tokening, APN pairing, group-Huber block ridge, bounded envelope, validation gate, eight-variant registry, state machine, statistics, and fail-closed device timing.
- **Execution size**: Declared 10 APN training manifests and 80 paired evaluation manifests across two targets, five seeds, four main methods, and four ablations.
- **How to Run**: Added the required Safe CLI contract to the implementation guide. It is declarative until the CLI and smoke tests are implemented.

### 2026-07-21 23:24 CST ? Safe paired aggregator implemented

- **Code**: Added `safe_aggregate.py` with strict 2-target x 5-seed x 8-variant completeness/pairing checks, pooled MSE/MAE, per-seed rows, crossed group-by-checkpoint bootstrap, Holm families, the 1% harm ceiling, 0.1% Joint non-inferiority, two-module ablation support, and fail-closed PASS/ABANDON timing authorization.
- **Tests**: Added synthetic complete-pass, disabled-gate, Joint-inferiority, missing-matrix, and paired-provenance rejection coverage in `test_edgetwincal_safe_aggregate.py`.
- **How to Run**: The aggregate API is available; CLI wiring remains pending.

### 2026-07-21 23:27 CST ? Safe aggregator syntax repair

- **Fixed**: Corrected one malformed f-string subscript introduced by the Windows unified-patch transport in `safe_aggregate.py`. No experiment semantics changed.
- **How to Run**: Re-running the focused aggregate tests next.

### 2026-07-21 23:38 CST ? Safe once-only ledger and evaluation artifacts implemented

- **Code**: Added `safe_campaign.py` with an independent frozen ledger, hashed in-memory-only test tokens, per-target once-only open/close/seal transitions, canonical tensor hashes, group-level SSE/SAE/N, private prediction payloads, and non-overwriting evaluation manifests.
- **Tests**: Added ledger transition/token secrecy, paired metric/cell/private-array, overwrite refusal, and non-finite prediction coverage.
- **How to Run**: Low-level campaign APIs are available; the end-to-end CLI remains pending.

### 2026-07-21 23:40 CST ? Windows-safe atomic tensor flush

- **Fixed**: Opened the temporary torch payload in read/write binary mode before `fsync`, resolving Windows error 9 during private evaluation-array serialization.
- **How to Run**: Re-running the focused campaign tests.

### 2026-07-21 23:40 CST - BVSE Safe selection and dataset gate implemented

- **Code**: Added validation-only deterministic group splitting, group metrics, gain-concentration auditing, bounded combined corrections, exact APN clone fallback, frozen cap/shrink candidate selection, and ablation semantics in `safe.py`.
- **Gate**: Added the five-checkpoint crossed group-by-checkpoint bootstrap gate with APN harm ceilings, Joint non-inferiority, anomalous-checkpoint rejection, auditable hashes, and configurable resample counts.
- **Tests**: Added focused coverage for cap/shrink order, bitwise fallbacks, order-invariant splits, tie order, confidence and group failures, no-envelope raw retention, and stable/anomalous five-seed gates.
- **How to Run**: `python -m pytest code/tests/test_edgetwincal_safe_gate.py -q`.
### 2026-07-21 23:20 CST ? GRJF robust core implemented (focused verification pending)

- **Added**: New isolated code/src/edgetwincal/robust.py and code/tests/test_edgetwincal_safe_robust.py; no sealed EdgeTwinCal module was modified.
- **Contract**: Deterministic CPU float64 group-balanced Huber block ridge with N/(G*n_g) weights, weighted median/MAD scaling, separate latent/cross penalties, unpenalized intercept, feature-z clipping, zero-diagonal joint features, and explicit audited zero-cell fallbacks.
- **Ablations**: observation_weighted changes only row weighting and squared_loss changes only the robust loss; BVSE correction caps remain outside this module.
- **Verification**: Focused synthetic coverage was added for duplication/outlier/reorder behavior, degenerate and invalid cells, adapter shapes, finite state, zero diagonal, and both ablation switches. The focused pytest command is the next action.
- **How to Run**: No user-facing Safe CLI exists yet, so the run manual remains unchanged.
### 2026-07-21 23:42 CST ? GRJF focused verification passed

- **Result**: .conda/envs/evipatch/python.exe -m pytest code/tests/test_edgetwincal_safe_robust.py -q -p no:cacheprovider completed with 6 passed in 1.58 s.
- **Review**: Python compilation and git diff --check passed. The only test adjustment changed the zero-diagonal precision probe to float64; solver behavior was unchanged.
- **Status**: GRJF core is run-verified. BVSE caps and deployment gates remain outside robust.py as designed.
- **How to Run**: The focused developer command above is reproducible; no user-facing Safe CLI was added by this module.

### 2026-07-21 23:45 CST - Safe production floors and ledger contract

- **Protocol alignment**: Raised production Safe selection and dataset-gate defaults to at least 20 groups and 400 observed cells; low thresholds remain explicit test-only overrides.
- **Audit API**: Dataset gate decisions now retain actual checkpoint/group/cell counts and expose `ledger_fields()` with enabled, checkpoints, validation_groups, validation_cells, and gate_sha256.
- **Tests**: Added frozen-default, insufficient-cell, and exact ledger-mapping coverage.
- **How to Run**: `python -m pytest code/tests/test_edgetwincal_safe_gate.py -q`.

### 2026-07-21 23:47 CST ? Safe paired-state runner implemented

- **Code**: Added `safe_runner.py` to fit ordinary Joint, original Full, the full robust alpha grid, and controlled no-balance/no-robust states from adapter-train plus `val_select` only. It applies the frozen eight-variant registry and builds the five-checkpoint `val_safety` decision without any test API.
- **Pairing**: All variants share frozen base/latent/source tensors; ablations reuse the selected main penalties and envelope so each changes only its declared mechanism. Dataset-gate rejection makes `Safe` an exact APN clone while retaining `SafeNoGate`.
- **Tests**: Added focused variant-order, bounded correction, raw ablation, and exact fallback coverage.
- **How to Run**: Runner is library-wired; CLI integration remains pending.

### 2026-07-21 23:52 CST - Frozen Safe config and holdout isolation data plane

- **Config**: Added strict immutable `safe_v1.json` parsing for the two official sources, exact Beijing/Intel chronological protocols, five seeds, frozen APN settings, method registry, and exact-value robust/envelope/gate/statistics thresholds.
- **Data isolation**: Added root-bounded manifests, timestamp-first physical pre-test/test routing, train-only normalization, deterministic whole-group validation routing, pseudonymous IDs, APN history-axis time marks, preserved APN-facing group IDs, masked loaders, and a token-hashed once-only test ledger. Pre-test preparation never parses numeric test values.
- **Tests**: Focused synthetic checks cover strict schema/value drift rejection, path escape, split/group/target disjointness, train-only normalization, reorder-stable rows and IDs, exact APN markers and extracted group identity, test-reader inaccessibility, invalid tokens, and once-only consumption; 11 passed.
- **How to Run**: `.\.conda\envs\evipatch\python.exe -m pytest code\tests\test_edgetwincal_safe_config.py code\tests\test_edgetwincal_safe_data.py -q -p no:cacheprovider`.

### 2026-07-21 23:51 CST ? Gate deployment semantics aligned

- **Fixed**: `DatasetSafetyDecision.enabled` now means the Safe deployment actually passed the validation gate; a rejected gate records `enabled=false` and exact APN fallback instead of merely indicating that an audit ran.
- **Runner**: Uses the gate object's canonical `ledger_fields()` rather than independently reconstructing checkpoint/group/cell counts, while retaining a local cross-check count.
- **How to Run**: Re-running Safe gate and runner tests.

### 2026-07-22 00:02 CST ? Official Safe sources downloaded and hashed

- **ResearchPilot Phase E action**: After config/data/solver/gate synthetic tests passed, downloaded only the two predeclared official raw objects into the ignored `data/edgetwincal_safe_v1/raw` namespace. No sealed-test value was parsed or opened.
- **Beijing source**: 7,959,991 bytes, SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`, exactly matching the frozen expected digest.
- **Intel source**: 34,422,518 bytes, observed SHA256 `d99288c8f406ca6604d359ceaa0d8adfffa79e7095061a1e27dc4399f48c7225`; the config intentionally had no upstream-published expected digest, so the observed digest is frozen in its raw manifest.
- **Isolation**: Both manifests and objects are under `msn2`; neither raw object is tracked or package-eligible.
- **How to Run**: The forthcoming CLI `download` command wraps the same verified data API.

### 2026-07-22 00:09 CST ? Intel official incomplete-row handling frozen

- **Observed source condition**: The official Intel object contains 526 timestamp-only three-field records among 2,313,682 lines; they contain no mote or temperature and cannot contribute a measurement. The first run stopped before writing an Intel shard.
- **Code**: Timestamp routing now audits and skips only rows with a valid date/time but fewer than five fields. It still fails on a row lacking timestamp fields, parses no missing test value, and records `discarded_unusable_rows` in the pre-test manifest. Beijing records zero discarded rows.
- **Tests**: Added a synthetic pre-test and sealed-test malformed-row case proving usable rows remain, unusable rows are counted, and no value is fabricated.
- **How to Run**: Re-run the focused Safe data suite, then retry Intel pre-test preparation; existing Beijing prepared files remain immutable.

### 2026-07-22 00:11 CST ? Intel discard audit constructor repair

- **Fixed**: Moved the new discarded-row counter into the `PretestManifest` constructor after the first focused collection exposed an indentation error. Protocol behavior is unchanged.
- **How to Run**: Re-running `test_edgetwincal_safe_data.py` before touching the official Intel shard.

### 2026-07-22 00:13 CST ? Intel malformed-row fixture boundary corrected

- **Test only**: Moved the synthetic sealed rows from the exclusive test end timestamp to the actual test start. This now exercises one usable sealed row plus one unusable timestamp-only sealed row.
- **How to Run**: Re-running the focused data suite.

### 2026-07-22 00:17 CST ? Pre-test shards and feasibility audit complete

- **Isolation result**: Beijing pre-test/sealed raw counts are 360,966/52,128; Intel counts are 357,440/213,875 with 526 timestamp-only unusable records audited. The sealed shards remain token-unopened and their numeric fields have not been parsed.
- **Beijing windows**: train 8,281 (1,036 groups), APN val 473 (60), adapter 481 (61), val_select 241 (31), val_safety 729 (92). Target cells are respectively 2,335,246; 135,006; 134,629; 67,794; 205,267. Train-only normalizer SHA256 is `d90a294712c05187b4eedb82e7c9f5a10f0b36a4fdd9416cfb17e1045ea15690`.
- **Intel windows**: train 1,629 (136 groups), APN val 189 (16), adapter 381 (32), val_select 180 (20), val_safety 180 (20). Target cells are 917,192; 80,532; 205,382; 83,760; 84,286. Train-only normalizer SHA256 is `73121077393cea72bb593548ce9b2a9a206a2039c09689a1c4ebc1c5fa12bf05`.
- **No tuning**: Intel per-mote train counts are heterogeneous (35--4,498), but the predeclared 1--54 channel set and all thresholds remain unchanged; sparse output cells will use the already frozen zero-correction fallback.
- **How to Run**: Next actions are protocol-conformance fixes identified by the integration audit, full synthetic smoke, then sequential APN training.

### 2026-07-22 00:22 CST ? Real APN train-only 100-step smoke passed

- **Device**: Official patched APN in `apn` mode ran on NVIDIA GeForce RTX 4090 with finite gradients; no checkpoint or test loader was created.
- **Beijing**: 4,949 parameters, 100 Adam steps in 2.958 s (33.81 step/s), peak allocated CUDA memory 41,328,640 bytes; masked train loss moved 0.9124 -> 0.8039 over 902,214 observed target cells.
- **Intel**: 7,007 parameters, 100 Adam steps in 5.699 s (17.55 step/s), peak allocated CUDA memory 98,489,856 bytes; masked train loss moved 0.9762 -> 0.3360 over 1,801,371 observed target cells.
- **Forecast**: Even a full 200-epoch upper bound for all ten small APN checkpoints is compatible with the one-day budget; early stopping should reduce it further. This timing does not authorize or claim edge latency.
- **How to Run**: The smoke will be exposed by the Safe CLI; formal training waits for the remaining protocol-conformance tests.

### 2026-07-22 00:41 CST -- EdgeTwinCal-Safe end-to-end CLI and crash-only recovery

- **ResearchPilot Phase E/F action**: Added the independent `edgetwincal_safe_v1` control plane and entry script without changing the pinned APN baseline or sealed `msn2026_v1` campaign.
- **APN training**: `train` wraps every project-owned batch through `as_apn_batch`, preserves `group_ID`, uses a re-iterable loader for all epochs, and calls `train_apn_train_val`. Checkpoints live at `results/edgetwincal_safe_v1/backbones/<dataset>/seed_<seed>/`; seeds and datasets execute sequentially.
- **Intentional new-data adaptation**: Safe uses the pre-frozen global masked micro-MSE validation early-stopping rule because it matches the new campaign primary metric and all four compared methods share each checkpoint. This is not claimed as byte-for-byte P12 training parity.
- **Deterministic CUDA**: The entry script sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing PyTorch when absent and fails closed on incompatible pre-existing values. The value is recorded in training/evaluation provenance; `HOME` and `CODEX_HOME` remain untouched.
- **Vendor provenance**: Every APN model construction verifies upstream commit `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`, `patches/apn_evipatch.patch` SHA256, and patched `vendor/APN/models/APN.py` SHA256 before use.
- **Pre-test fitting**: `fit` extracts only `adapter`/`val_select`/`val_safety` under constant cross-seed protocol ID `safe_v1`, persists private tensor caches and `SafeSeedStates`, and cross-checks config/checkpoint/normalizer/cache/state/manifest hashes. Robust row threshold remains `max(100, 4*p)` (`minimum_rows=None`).
- **Validation gate**: Standalone `gate` loads five persisted states and `val_safety` caches only; it has no sealed-test token or reader path.
- **Once-only test with crash recovery**: A normal first run freezes both campaign/data ledgers, consumes each raw sealed shard once, immediately materializes an immutable private test-window cache, and binds its SHA256 to dataset/config/normalizer/protocol/gate hashes. A crash-active ledger resumes only from that cache and fills missing seed-by-variant manifests; complete artifacts are validated and not rewritten. A sealed campaign rejects another test.
- **Aggregation/timing**: `aggregate` requires the sealed 2-by-5-by-8 paired matrix. `require_device_timing_authorized` makes CPU/Jetson timing unreachable unless the final report is `PASS`.
- **Verification**: The focused CLI/config/data suite passed 21 tests; official Beijing/Intel CPU APN smokes were finite and did not open a test loader. The complete Safe suite subsequently passed 50 tests.
- **How to Run**:
  - `.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py smoke --dataset beijing_air --seed 2024 --device cuda:0`
  - Replace `smoke` with `train`, then repeat `train` sequentially for all five seeds on both datasets.
  - Run `fit` for the same ten dataset/seed pairs, followed by `gate`, `test --device cuda:0`, and `aggregate`.



### 2026-07-22 00:55 CST ? EdgeTwinCal-Safe protocol-conformance audit closed

- **Group-robust selection**: Positive-gain concentration now uses per-group MSE rather than SSE, and leave-one-group-out evaluates the remaining groups'' macro-MSE. Candidate and alpha ties use the frozen `1e-4` tolerance, then lower shrink, lower cap, and stronger block regularization; raw validation micro-MSE no longer selects alpha.
- **Validation-only dataset gate**: One RNG now supplies shared group and checkpoint multiplicities for APN/Joint/Safe micro-MSE, group-macro-MSE, and micro-MAE. Deployment requires APN point non-degradation and one-sided 1% harm bounds, reliable positive APN micro-MSE gain, Joint macro-MSE point/UCB within 0.5%, at least four of five checkpoint gains, nonnegative macro leave-one-group-out gain, concentration at most 0.25, and the retained per-checkpoint 1% harm guard. Failure remains an exact APN clone.
- **Provenance and aggregation**: Evaluation manifests now hash predictions and targets. Aggregation pairs config/split/normalizer/sample/group/mask/target/protocol/gate hashes across all variants and seeds, verifies the gate-decision hash, and proves disabled Safe predictions hash-identically match APN. The campaign ledger schema is now `edgetwincal.safe-campaign-test-ledger.v1`, distinct from the data ledger. All four ablations are globally reported with one Holm family; only NoRobust and NoBound are required mechanism gates, while NoBalance and NoGate remain diagnostics.
- **Verification**: All 55 Safe config/data/robust/gate/runner/campaign/aggregate/CLI tests passed. The sole warning is pytest''s inability to create its optional workspace cache and does not affect results.
- **How to Run**: `$env:PYTHONPATH='code/src'; .\.conda\envs\evipatch\python.exe -m pytest code/tests/test_edgetwincal_safe_config.py code/tests/test_edgetwincal_safe_data.py code/tests/test_edgetwincal_safe_robust.py code/tests/test_edgetwincal_safe_gate.py code/tests/test_edgetwincal_safe_runner.py code/tests/test_edgetwincal_safe_campaign.py code/tests/test_edgetwincal_safe_aggregate.py code/tests/test_edgetwincal_safe_cli.py -q`

### 2026-07-22 -- EdgeTwinCal-Safe ten-checkpoint training complete

- **ResearchPilot Phase F action**: Completed the frozen two-dataset by five-seed APN matrix sequentially on one RTX 4090. The command output is retained at `logs/edgetwincal_safe_v1/train_all.log` (359,460 bytes); no test loader or sealed value was opened.
- **Integrity**: All 10 manifests report `status=complete`; every checkpoint SHA256 matches its manifest, every run records the same Safe config and train-only normalizer, and every `test_constructed` flag is false.
- **Beijing**: Seeds 2024--2028 stopped after 18/21/20/17/14 epochs with best validation masked micro-MSE 0.623929/0.620892/0.624780/0.621528/0.623685. Wall times were 118.7/140.3/133.4/112.0/93.6 seconds; each model has 4,949 parameters.
- **Intel**: Seeds 2024--2028 stopped after 40/22/78/24/18 epochs with best validation masked micro-MSE 14.011420/13.007040/26.851669/13.148166/12.891948. Wall times were 125.6/68.7/247.4/75.1/56.4 seconds; each model has 7,007 parameters. The high seed-2026 variance is retained and will be handled by the predeclared five-checkpoint validation gate, not tuned away.
- **How to Run**: `.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py train --dataset all --seed all --device cuda:0`; completed cells resume only after checkpoint/manifest hash validation. Next run the matching `fit`, then `gate`; physical test remains closed until both validation gates and ledgers validate.

### 2026-07-22 -- Safe adapter matrix and validation-only gates complete

- **ResearchPilot Phase F action**: Completed all 10 adapter cells with the frozen 6-by-6 alpha grid, bounded-envelope grid, Joint/Full controls, and four ablation states. Ten fit caches, ten states, and ten public fit manifests are present and hash-consistent; `logs/edgetwincal_safe_v1/fit_all.log` retains the command output.
- **Beijing selection**: All five checkpoints selected feasible Safe candidates. Their `val_select` MSE gains versus APN were 2.223%, 2.460%, 6.439%, 2.132%, and 1.378%; no checkpoint fell back.
- **Intel selection**: Seeds 2024--2027 returned `no_feasible_candidate` and exact APN fallback. Seed 2028 selected `cap=2, shrink=1` with 1.356% `val_select` MSE gain. All failures and raw candidate audits are retained.
- **Dataset gate**: Beijing passed on disjoint `val_safety`: pooled Safe/APN MSE 0.189561/0.194690 (2.634% relative gain; shared-bootstrap gain CI lower bound 0.686%), 5/5 checkpoint gains, and Safe also beat Joint by 7.313%. Intel failed with only 1/5 gains, unreliable efficacy CI, and material Joint inferiority; deployment is an exact APN clone with reasons `apn_mse_gain_unreliable`, `efficacy_ci`, `joint_macro_point`, `joint_macro_ucb`, and `checkpoint_consistency`.
- **Decision implication**: The final requirement of positive results on two new targets can no longer pass. Complete the predeclared once-only test matrix and statistics to retain all outcomes, but do not run CPU/Jetson timing and do not alter the method.
- **How to Run**: `.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py fit --dataset all --seed all --device cuda:0`, followed by `.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py gate`. Both commands are complete and resume only after hash validation; the next and only test-opening command is `test --device cuda:0`.

### 2026-07-22 -- Once-only test, paired statistics, and ABANDON

- **ResearchPilot Phase F action**: After both validation decisions were frozen, opened each physical test shard exactly once, materialized immutable recovery caches, and completed the full two-dataset by five-seed by eight-variant matrix. The campaign ledger is sealed, 80/80 manifests completed, no crash resume occurred, and protocol SHA256 is `0ad544dbaef79fb07c16aa894d99eeeab42b5df2993ba6bd4c85098cb1fb6854`.
- **Beijing test**: APN/Joint/Full/Safe pooled MSE is 0.886283/0.874270/0.874231/0.890569; Safe is 0.484% worse than APN (95% relative-gain CI -1.289% to +0.208%) and 1.864% worse than Joint (95% CI -3.033% to -0.542%). Seed 2026 harms APN by 1.282%, breaching the 1% ceiling.
- **Intel test**: Gate-disabled Safe exactly equals APN at MSE 434.928686 and MAE 18.427934. Joint has MSE 427.355942, while the paired Safe-vs-Joint CI is too wide for non-inferiority. All five exact-fallback hashes passed.
- **Final decision**: `ABANDON`. Two positive targets, per-target/seed harm, Joint non-inferiority, enabled gates, and both required module checks all failed. No further method change, new baseline, CPU/Jetson timing, paper expansion, or ZIP packaging is authorized.
- **Outputs**: Added compact, non-private JSON/CSV/Markdown artifacts under `artifacts/edgetwincal_safe_v1`; raw data, checkpoints, fit caches, private predictions, and ledgers remain ignored under `data/` and `results/`.

### 2026-07-22 -- Compact result workbook and delivery audit complete

- **Spreadsheet artifact**: Added `EdgeTwinCal-Safe_results.xlsx` with Overview, Summary, PerSeed, PairedStats, and Ablations sheets. Formula inspection matched zero errors, all five sheets were rendered for visual QA, and the workbook contains no macros, external links, or raw identifiers.
- **Public integrity**: Rebuilt the concise result record as valid UTF-8, generated `SHA256SUMS.csv` for all ten payload files, and independently reconciled every JSON/CSV/XLSX row against the aggregate report.
- **Isolation audit**: The public artifact directory contains no raw data, checkpoint, prediction array, test cache, sample identifier, credential, absolute user path, or file larger than 100 MB.
- **Decision**: The final verdict remains `ABANDON`; device timing, paper expansion, and ZIP packaging remain intentionally blocked.
- **How to Run**: `.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py aggregate`; inspect `artifacts\edgetwincal_safe_v1\RESULTS.md` and validate payload hashes with `SHA256SUMS.csv`.
