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
& $evipatchPython -m evipatch.runner controlled
& $evipatchPython -m evipatch.runner aggregate
& $evipatchPython -m evipatch.runner package
```

- **`validate`**: checks the seven-variant matrix, fixed seeds, local mutable paths, Python executable, APN checkout, and pinned upstream commit; writes nothing.
- **`status`**: prints the completion state of every seed checkpoint and native/MCAR/burst evaluation; writes nothing.
- **`smoke`**: runs the project pytest suite and writes `logs/smoke/pytest.log`.
- **`timing`**: benchmarks exactly 100 measured real-P12 optimizer steps after warm-up for all seven variants and writes `artifacts/timing_100_steps.json`.
- **`stage-a`**: acquires `results/stage_a/.stage_a.lock`, then trains 21 checkpoints and evaluates 63 views strictly sequentially; outputs checkpoints under `results/`, raw logs under `logs/`, arrays/metrics beside each checkpoint, and one JSON manifest per process.
- **`controlled`**: freezes error-blind real-data support pairs from each APN checkpoint/native history and scores identical pair IDs for all variants.
- **`aggregate`**: produces patient errors, seed summaries, hierarchical paired bootstrap results, and `gate_decision.json` under `artifacts/`.
- **`package`**: creates the final filtered ZIP and `SHA256SUMS.csv` under `packages/`.
