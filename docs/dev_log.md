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
