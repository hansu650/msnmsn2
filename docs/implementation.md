# Implementation Guide — EviPatch Stage A
> Generated: 2026-07-20 | Strategy: strong baseline patch | Status: CONFIRMED
> Linked design: `docs/idea_report.md` Part 3

## 1 Original Project Information

| Item | Value |
|---|---|
| Baseline | APN official repository |
| Audited commit | `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4` |
| Framework | PyTorch 2.6.0 + CUDA 12.4 |
| Python | 3.11.13 |
| Local baseline path | `vendor/APN` (gitignored) |
| Modification strategy | Project-owned modules plus a reproducible patch applied to the pinned baseline |

The baseline is not copied into this repository because no clear top-level license was detected. Reproduction uses the upstream URL, commit hash, and `patches/apn_evipatch.patch`.

## 2 Rewrite Scope

Only the following baseline areas are patched:

1. `models/APN.py`: append evidence features before the existing projection.
2. `utils/configs.py` and `utils/ExpConfigs.py`: add evidence and observation-shift arguments.
3. `data/data_provider/data_factory.py` and `datasets/P12.py`: pass runtime configs safely to Windows workers and apply deterministic test-only masks.
4. `data/dependencies/tsdm/config/_config.py`: honor the explicit project-local storage root.
5. `main.py`: accept an explicit seed base so every native checkpoint is trained by a separate, auditable process.
6. `exp/exp_main.py`: use collision-free evaluation names and retain deletion-audit arrays alongside predictions and targets.
7. `data/dependencies/tsdm/datasets/physionet2012.py`: make first-clean tuple order match the same class's persisted deserializer.
8. `data/dependencies/tsdm/utils/types/dtypes.py`: on Windows only, provide missing NumPy extended-precision attribute aliases required by bundled tsdm imports.
9. `utils/tools.py`: honor its existing `test_steps=100` argument so seven-variant timing is exactly 100 measured optimization steps after warm-up.

The query module, decoder, MSE loss, Adam optimizer, early stopping, and dataset split remain unchanged for Stage A parity. These two orchestration-only additions are required to map every saved checkpoint and evaluation view to one seed and one run manifest without relying on minute-resolution directory inference.

## 3 Complete Project Tree

```text
msn2/
├── .conda/envs/evipatch/              # local environment; gitignored
├── code/
│   ├── configs/stage_a.json           # variants, seeds, shifts, APN hyperparameters
│   ├── scripts/
│   │   ├── setup_environment.ps1      # dependency installation and verification
│   │   ├── apply_apn_patch.ps1        # clone/pin/apply patch idempotently
│   │   ├── run_smoke.ps1              # unit/parity/100-step checks
│   │   ├── run_stage_a.ps1            # sequential training and evaluation
│   │   ├── monitor_stage_a.ps1        # compact progress reader
│   │   └── package_results.ps1        # manifest, checksums, zip
│   ├── src/evipatch/
│   │   ├── __init__.py
│   │   ├── evidence.py                # evidence signatures and controls
│   │   ├── shifts.py                  # deterministic exact-rate test shifts
│   │   ├── paths.py                   # project-root write guards
│   │   ├── runner.py                  # commands, provenance, checkpoint discovery
│   │   ├── controlled.py              # frozen controlled-support pairing and scoring
│   │   ├── aggregate.py               # metrics and hierarchical bootstrap
│   │   └── package.py                 # delivery archive builder
│   ├── tests/
│   │   ├── test_evidence.py
│   │   ├── test_shifts.py
│   │   ├── test_paths.py
│   │   └── test_apn_parity.py
│   ├── README.md
│   └── requirements.txt               # excludes torch family
├── data/                               # project-local raw/processed data; gitignored
├── results/                            # checkpoints and evaluation arrays; gitignored
├── logs/                               # process and monitoring logs; gitignored
├── artifacts/                          # compact summaries; gitignored by default
├── packages/                           # final archive; gitignored
├── docs/
│   ├── source/MSN2026_EviPatch_idea.md
│   ├── user_requirements.md
│   ├── idea_report.md
│   ├── implementation.md
│   └── dev_log.md
├── patches/apn_evipatch.patch
├── vendor/APN/                         # pinned upstream clone; gitignored
├── .gitignore
└── README.md
```

## 4 Per-File Function Table

| File | Functions/classes | Input | Output | Called by |
|---|---|---|---|---|
| `evidence.py` | `evidence_width`, `compute_evidence`, `apply_evidence_control`, `fixed_random_features` | TAPA weights, times, bounds, mask, mode | Evidence tensor `[B*N,P,K]` plus diagnostics | patched APN model, tests |
| `shifts.py` | `stable_uniform`, `exact_mcar_mask`, `exact_burst_mask`, `apply_observation_shift` | values/mask/times/sample IDs/rate/seed | shifted values and mask; deletion audit | patched P12 collate, tests |
| `paths.py` | `resolve_project_root`, `assert_within_project`, `ensure_project_dir` | candidate paths | validated absolute path or exception | runner, package, scripts |
| `runner.py` | `load_stage_config`, `build_apn_command`, `run_checked`, `discover_checkpoints`, `write_run_manifest`, `main` | stage config and action | subprocess results and JSON provenance | PowerShell scripts |
| `controlled.py` | `patch_units`, `match_support_pairs`, `score_frozen_pairs`, `run_controlled_support` | APN checkpoint and native evaluation arrays | pair audit CSV/JSON and controlled errors | aggregation/gate |
| `aggregate.py` | `load_evaluation`, `masked_errors_per_patient`, `hierarchical_paired_bootstrap`, `summarize_stage_a`, `decide_gate`, `main` | result tree and config | CSV/JSON/Markdown summaries and gate decision | Stage A runner, packaging |
| `package.py` | `iter_delivery_files`, `sha256_file`, `build_manifest`, `create_archive`, `main` | project root and summary status | reproducible ZIP and checksums | packaging script |
| `test_evidence.py` | collision, shape, finite-gradient, control tests | synthetic tensors | pytest result | smoke script |
| `test_shifts.py` | exact-rate/determinism/burst-locality tests | synthetic masks | pytest result | smoke script |
| `test_paths.py` | root-boundary tests | temporary paths | pytest result | smoke script |
| `test_apn_parity.py` | state-transfer forward parity | original and patched APN | max absolute/relative difference | smoke script |
| `stage_a.json` | experiment matrix | N/A | centralized settings | runner |
| patched `main.py` | explicit per-process seed selection | `seed_base`, `itr` | deterministic training iteration(s) | APN CLI |
| patched `exp_main.py` | evaluation naming and shift audit persistence | evaluation name and batch audit tensors | non-colliding evaluation directory and saved arrays | APN CLI |

## 5 Function-Level Design

### 5.1 `evidence.py`

`evidence_width(mode: str) -> int`

- Returns `0` for `apn`, `1` for `global_ratio`, `raw_count`, or `soft_mass`, and `3` for `evipatch_full`, `shuffled_evidence`, or `random_features`.
- Rejects unknown modes before model construction.

`compute_evidence(temporal_weights: Tensor, weights_raw: Tensor, mask: Tensor, times: Tensor, left: Tensor, right: Tensor, global_ratio: Tensor | None, eps: float = 1e-9) -> dict[str, Tensor]`

- Shapes: weights/mask `[B*N,P,L]`, times `[B*N,1,L]`, bounds `[B*N,P,1]`.
- Computes unmodified mass before adding numerical epsilon, squared-weight sum, effective support, weighted time mean/variance, normalized coverage, and hard count inside learned bounds.
- Empty patches produce zero finite signatures.
- Returns transformed and raw diagnostics without detaching the differentiable mass/support/coverage path.

`apply_evidence_control(stats: dict[str, Tensor], mode: str, n_variables: int, random_table: Tensor | None = None) -> Tensor | None`

- Selects the predeclared signature.
- `shuffled_evidence` applies a deterministic cyclic roll across flattened sample-channel rows, preserving marginal values and tensor width.
- `random_features` indexes a registered fixed Gaussian table `[N,P,3]` and adds no trainable parameters.
- Does not normalize a one-dimensional signature with LayerNorm.

### 5.2 `shifts.py`

`stable_uniform(sample_ids: Tensor, channel: int, positions: Tensor, seed: int) -> Tensor`

- Implements a deterministic integer hash converted to `[0,1)`; results do not depend on batch order or Python hash randomization.

`exact_mcar_mask(mask: Tensor, sample_ids: Tensor, rate: float, seed: int) -> Tensor`

- For every sample-variable row, ranks observed positions by stable hash and removes exactly `floor(rate*n_observed)` points.
- Never removes padded or already-missing entries.

`exact_burst_mask(mask: Tensor, times: Tensor, sample_ids: Tensor, rate: float, seed: int) -> Tensor`

- Chooses a deterministic start rank and removes one circularly selected contiguous run in chronological observed-position order.
- Removes the same count as `exact_mcar_mask` for each sample-variable, establishing matched deletion rate.

`apply_observation_shift(x: Tensor, x_mask: Tensor, x_mark: Tensor, sample_ids: Tensor, mode: str, rate: float, seed: int) -> tuple[Tensor, Tensor, dict]`

- Supports `none`, `mcar`, and `burst` in Stage A.
- Sets removed values to zero because APN gates them with the returned mask.
- Returns a deletion audit including requested/actual counts per sample-variable.

### 5.3 `paths.py`

`resolve_project_root() -> Path` resolves the repository from the module path, not the caller's current directory.

`assert_within_project(path: Path | str, *, allow_root: bool = False) -> Path` resolves symlinks/relative segments and rejects paths outside `C:\Users\qintian\Desktop\msn2`.

`ensure_project_dir(path: Path | str) -> Path` validates then creates a directory.

### 5.4 `runner.py`

`load_stage_config(path: Path) -> dict` validates variants, seed list, shifts, absolute roots, and the pinned APN commit.

`build_apn_command(config: dict, variant: str, action: str, checkpoint: Path | None = None, shift: str = "none", seed: int | None = None) -> list[str]` returns an argument list without shell interpolation.

`run_checked(command: list[str], log_path: Path, env: dict[str, str]) -> dict` runs a process, streams stdout/stderr to a project-local log, records wall time, return code, CUDA metadata, and fails on nonzero exit.

`discover_checkpoints(result_root: Path, variant: str) -> list[Path]` requires exactly one checkpoint per configured seed and sorts by `iterN`.

`write_run_manifest(...) -> Path` records git hashes, commands, dependency snapshot, data paths, GPU, timestamps, and artifact hashes.

`main(argv: list[str] | None = None) -> int` exposes validation, smoke/timing, train/evaluate, strict serial Stage A, controlled-support, aggregation, packaging, and status actions.

### 5.5 `controlled.py`

`patch_units(...)` reconstructs APN's learned soft windows and normalized value/time-embedding centroids from the APN checkpoint and saved native histories, then retains one maximum-support patch per patient/channel.

`match_support_pairs(...)` robustly standardizes centroid coordinates within seed/channel and greedily freezes disjoint nearest pairs under the predeclared distance/support/yield thresholds without reading model errors.

`score_frozen_pairs(...)` applies identical pair IDs to all variants and computes masked channel-level target MSE; `run_controlled_support(...)` writes pair, error, and yield audits.

### 5.6 `aggregate.py`

`load_evaluation(path: Path) -> Evaluation` loads `metric.json`, saved prediction/target/mask/sample-ID arrays, and run manifest; it validates shape and finite values.

`masked_errors_per_patient(pred: ndarray, true: ndarray, mask: ndarray, ids: ndarray) -> DataFrame` computes patient MSE/MAE using only target-mask entries.

`hierarchical_paired_bootstrap(left: DataFrame, right: DataFrame, seed_column: str, id_column: str, metric: str, n_resamples: int, rng_seed: int) -> dict` resamples seeds, then patient IDs within seeds, and returns mean paired difference, 95% percentile interval, and probability of improvement.

`summarize_stage_a(root: Path, config: dict) -> tuple[DataFrame, DataFrame]` produces per-run and aggregate tables for native/MCAR/burst and macro averages.

`decide_gate(summary: DataFrame, bootstrap: DataFrame, config: dict) -> dict` implements every condition in `idea_report.md` without manual overrides and returns only `PASS` or `ABANDON` with machine-readable reasons. Time overhead comes from the frozen real-P12 100-step artifact, not end-to-end wall time.

### 5.7 `package.py`

`iter_delivery_files(project_root: Path, gate: dict) -> Iterator[Path]` includes code, configs, patch, docs, compact logs, manifests, summaries, tests, and selected predictions; excludes environments, raw data, caches, and large checkpoints unless explicitly requested.

`sha256_file(path: Path) -> str` streams SHA-256.

`build_manifest(files: Iterable[Path], root: Path) -> list[dict]` records relative path, size, and hash in stable sorted order.

`create_archive(root: Path, files: Iterable[Path], output: Path) -> Path` writes a deterministic ZIP under `packages/`.

## 6 Baseline Patch Details

### `models/APN.py`

- Add `evipatch_mode` and registered fixed-random table to `AttentionPatchAggregation.__init__`.
- Set projection input width to `1 + te_dim + evidence_width(mode)`.
- Preserve original `sum_weights` denominator exactly for the centroid.
- Call project-owned evidence functions and concatenate only the selected signature.
- In `apn` mode, the forward graph and state-dict shapes are identical to upstream.

### `utils/configs.py` and `utils/ExpConfigs.py`

- Add `evipatch_mode`, `observation_shift`, `shift_rate`, `shift_seed`, `seed_base`, and `evipatch_eval_name`.
- Defaults are `apn`, `none`, `0.0`, `0`, `2024`, and an empty evaluation name, so upstream behavior is preserved except that the formerly hard-coded seed origin is now explicit.

### `main.py` and `exp/exp_main.py`

- Build the iteration seed list from `seed_base` while preserving the official 2024 default.
- Allow the runner to assign a stable evaluation directory name such as `eval_native`, `eval_mcar`, or `eval_burst` within an explicit checkpoint directory.
- Save `shift_requested` and `shift_actual` arrays when present; the target, target mask, predictions, and sample IDs continue to use the upstream array-saving path.

### `P12.py`

- After padding and before return, apply a shift only when `is_training == 0` and mode is not `none`.
- Accept the active `ExpConfigs` as an explicit collate argument rather than relying on a module-global object inside DataLoader workers.
- Use `sample_ID`-keyed deterministic masking.
- The target and target mask are never modified.

### `data_factory.py`

- Bind the active P12 `ExpConfigs` into its collate function with a pickle-safe `functools.partial`. This is equivalent under Linux/fork and preserves runtime padding, seed, and shift values under Windows/spawn.
- Stage A sets `num_workers=0` on Windows. The official value of 10 repeatedly spawns worker processes for every train/validation iterator and makes each epoch spend tens of seconds in process startup; the zero-worker setting changes only data-loading execution, not the split, collation, shuffle sampler, model, loss, optimizer, or hyperparameters that define the comparison.

### TSDM `_config.py`

- Read `EVIPATCH_TSDM_ROOT`; if missing, fail under the EviPatch runner rather than silently writing to the user profile.
- Direct raw, processed, log, and model directories under `data/tsdm`.
### TSDM Windows dtype compatibility

- If NumPy does not expose `float128` or `complex256` (Windows), alias only those missing names to `longdouble` and `clongdouble` before bundled tsdm builds its dtype tables.
- Preserve the native NumPy attributes unchanged on platforms that provide them; this is an import compatibility fix, not a numeric preprocessing change.

### APN timing helper

- Cap the existing training-step benchmark at its declared `test_steps` argument instead of silently consuming every remaining dataloader batch.
- Keep the official optimizer, loss, batch size, and real P12 train loader unchanged during the seven-variant timing smoke.


## 7 Data Preparation

PhysioNet is downloaded by the pinned APN/tsdm code into `data/tsdm/rawdata/Physionet2012` and processed under `data/tsdm/datasets/Physionet2012`. The runner sets only the task-specific `EVIPATCH_TSDM_ROOT`; it does not repurpose `HOME` or `CODEX_HOME`.

Before download, the runner verifies that every resolved storage path is inside the project. A download manifest records source URL, upstream hash when available, final size, and local SHA-256.

## 8 Results Format

| Artifact | Format | Required fields |
|---|---|---|
| `run_manifest.json` | JSON | variant, seed, shift, full command argv, git hashes, environment, GPU, start/end, return code |
| `metric.json` | JSON | masked MSE, masked MAE |
| `patients.csv` | CSV | variant, seed, shift, patient ID, MSE, MAE, observed target count |
| `stage_a_runs.csv` | CSV | per-run metric, timing, parameters, peak memory |
| `stage_a_summary.csv` | CSV | mean, std, relative change, macro average |
| `paired_bootstrap.csv` | CSV | comparison, metric, estimate, CI low/high, resamples |
| `controlled_support_pairs.csv` | CSV | seed, pair ID, patient/channel/patch membership, centroid distance, support contrast |
| `controlled_support_errors.csv` | CSV | frozen pair ID, variant, seed, masked pair MSE |
| `timing_100_steps.json` | JSON | seven variants, warm-up/measured steps, mean/std optimizer-step time, argv |
| `gate_decision.json` | JSON | verdict, each condition, evidence paths, timestamp |
| `REPORT_CN.md` | Markdown | concise Chinese interpretation with no unsupported claims |
| `SHA256SUMS.csv` | CSV | relative path, bytes, SHA-256 |

## 9 Implementation and Run Order

1. Create and verify the local Conda environment.
2. Implement project-owned evidence, shift, path, runner, aggregation, and packaging modules.
3. Apply the patch to the pinned APN clone and generate `patches/apn_evipatch.patch` from the upstream diff.
4. Run unit tests and exact upstream parity.
5. Download/process PhysioNet inside the project.
6. Run 100-step smoke timing for all seven variants.
7. Train Stage A sequentially for three seeds per variant.
8. Evaluate the same checkpoints on native, exact 30% MCAR, and matched 30% burst.
9. Aggregate, bootstrap, and execute the automatic gate.
10. Run conditional Stage B only on `PASS`.
11. Package and verify checksums; push only compact, authorized artifacts.

## 10 Validation

- ✅ Experiment coverage: every Stage A variant, shift, metric, control, gate, and packaging output has a supporting function or script.
- ✅ Logic consistency: APN centroid remains `[B*N,P,1+Dte]`; evidence is `[B*N,P,K]`; projection input is `[B*N,P,1+Dte+K]`; downstream hidden width is unchanged.
- ✅ Completeness: every file in the planned tree has a responsibility and implementation contract.
- ✅ Environment: project-local Conda prefix selected; requirements file excludes the torch family; CUDA torch is installed separately.
- ✅ Device: RTX 4090 (24 GB) detected; single-GPU sequential execution selected.
- ✅ Dataset: public PhysioNet automatic download selected; MIMIC excluded.
- ✅ Run strategy: smoke → three-seed Stage A → automatic gate → conditional expansion.
- ✅ Isolation: all mutable paths guarded under `C:\Users\qintian\Desktop\msn2`.
