# Active Implementation Guide ? EdgeTwinCal msn2026_v1 Lab Campaign
> Updated: 2026-07-21 | Status: design frozen before coding
> Authority: EdgeTwinCal_Lab_Experiment_Handoff_20260721

## 1 Scope and invariants

All writes are rooted at C:\Users\qintian\Desktop\msn2. The Downloads handoff
is read-only. The sibling Desktop\msn, HOME, and CODEX_HOME are out of scope.
APN remains pinned at f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4. The
project method version is msn2026_v1; legacy_v1 remains immutable parity evidence.

The new implementation must not open a new test split until G0 and G1 are
recorded PASS. Train and validation data, plus synthetic fixtures, are the only
data allowed during implementation and tuning. Any test-derived pass threshold
in the old runner is removed from the confirmatory path.

## 2 Implementation strategy

The APN backbone remains frozen. Project-owned modules provide configuration,
protocol manifests, provenance-safe cached feature extraction, generic ridge
solvers, EdgeTwinCal variants, unified run records, segmented timing, and crossed
inference. Release parity and strict audit are separate campaigns and directories.
The earlier EviPatch patch may remain applied locally only in APN mode after
forward parity; it is not part of the EdgeTwinCal method.

The most important semantic repair is a two-input CFG interface:

- base_to_correct is the target forecast to which a CFG correction is added.
- source_forecasts is the tensor used to construct graph features.
- CFG-only passes frozen APN for both inputs.
- Full passes SLRH output as base_to_correct and frozen APN as source_forecasts.
- Reverse first fits CFG on frozen APN and then fits SLRH on the CFG output.
- Full-Diagonal reuses the exact frozen SLRH state and changes only CFG self-edge
  availability.

## 3 Project-owned tree

- code/configs/msn2026/default.json: single resolved experiment source.
- code/src/edgetwincal/config.py: validation, canonicalization, config hash.
- code/src/edgetwincal/schema.py: atomic run manifest and completeness checks.
- code/src/edgetwincal/provenance.py: cache keys/manifests and stale rejection.
- code/src/edgetwincal/protocol.py: group splits and train-only normalization.
- code/src/edgetwincal/ridge.py: shared standardization and ridge objectives.
- code/src/edgetwincal/latent.py: SLRH using the shared ridge implementation.
- code/src/edgetwincal/graph.py: decoupled CFG anchor/source implementation.
- code/src/edgetwincal/controls.py: V01 Bias, V02 Self-Affine, V10 Diagonal.
- code/src/edgetwincal/joint.py: V08 block-penalty joint ridge.
- code/src/edgetwincal/shuffle.py: V11/V12 deterministic permutations.
- code/src/edgetwincal/decoder_refit.py: V03 decoder-only AdamW refit.
- code/src/edgetwincal/timing.py: phase/device-aware measurement.
- code/src/edgetwincal/statistics.py: cell table, crossed bootstrap, Holm.
- code/src/edgetwincal/campaign.py: pre-test freeze and once-only orchestration.
- code/tests/test_edgetwincal_*.py: one focused suite per FIX/control.
- results/edgetwincal_legacy_v1: immutable pilot namespace.
- results/edgetwincal_msn2026_v1: new private campaign state.
- artifacts/edgetwincal_msn2026_v1: compact non-private outputs.
- packages: final audited return archive.

## 4 Function and contract table

| Module | Primary contract | Required output |
|---|---|---|
| config.py | load_resolved_config(path, overrides) | immutable mapping plus canonical SHA256 |
| schema.py | RunManifest start/complete/fail/validate | atomic JSON with file hashes and status |
| provenance.py | CacheManifest build/validate/digest | exact field mismatch or corruption rejection |
| protocol.py | hash_group_split, fit_train_normalizer | disjoint split and normalization manifests |
| ridge.py | fit_ridge_grid, apply_ridge | unpenalized intercept and validation-selected alpha |
| latent.py | fit/apply SLRH | corrected forecast, state, curve, timing |
| graph.py | fit/apply CFG(anchor, source) | zero/optional diagonal graph correction |
| controls.py | fit/apply Bias/Affine/Diagonal | matched train/validation-only controls |
| joint.py | fit_joint_grid | complete 6 by 6 validation surface |
| shuffle.py | deterministic_permutation | indices hash and shuffled feature source only |
| decoder_refit.py | fit_decoder_only | trainable names, curves, hashes, timings |
| timing.py | PhaseTimer, warm_inference | synchronized phase records with true device |
| statistics.py | cells, crossed_bootstrap, holm | micro effects, 95% CI, corrected decisions |
| campaign.py | freeze, validate, execute | registry hash and once-only test ledger |

## 5 Tensor and objective logic

Let B be batch size, H forecast horizon, C sensors, and D frozen latent width.

- Forecast, target, and target mask have shape [B,H,C].
- APN latent features have shape [B,C,D].
- SLRH predicts one residual vector of H values for each target sensor from its
  own D-vector; training rows are mask-filtered target positions.
- CFG creates for each target sensor and horizon a feature vector from frozen APN
  forecasts of source sensors. Zero diagonal excludes the target sensor; V10
  explicitly includes it.
- Shuffles permute rows, never target/mask values. CrossShuffle uses one source
  permutation per source sensor and split, shared by every target and horizon.
  LatentShuffle uses one permutation per target sensor and split, shared across
  horizons.
- All feature standardizers are fitted on training rows only, use a 1e-6 scale
  floor, and are serialized with fit-row hashes.
- Ridge minimizes summed squared error plus alpha times squared slopes. Intercepts
  are never penalized. A single alpha is chosen per dataset, checkpoint, variant
  using validation micro MSE.
- Joint ridge standardizes latent and graph blocks separately and evaluates all
  36 ordered alpha pairs.

Empty-mask targets return finite zero corrections and explicit no-observation
diagnostics. NaN or infinite features, targets, coefficients, or metrics fail the
run before it can be marked complete.

## 6 FIX-01: resolved configuration

default.json contains schema/method versions, project/APN commits, datasets,
protocols, seeds 2024--2028, hyperparameters, variant registry, ridge grids,
bootstrap settings, timing settings, gates, and paths relative to the project
root. CLI seed/dataset/variant overrides may only select members already present
in the resolved config. Canonical JSON uses sorted keys and stable separators.
Every run stores the complete resolved object and its SHA256. Import and --help
must not require vendor/APN or data assets.

## 7 FIX-02: provenance-safe cache

A cache key includes project commit, APN commit, APN patch hash/mode, checkpoint
SHA256, resolved-config hash, raw and processed dataset hashes, protocol/split
manifest hash, sample/group ID hash, normalization manifest hash, loader and
extractor source hashes, seed, shapes, dtypes, and mask hash. The digest is part
of the filename. Loading validates every field and payload hash; stale, partial,
or corrupt caches are rejected. Writes use a temporary file followed by an
atomic replace and a same-directory lock. A legacy schema-2 cache can be read
only by the isolated legacy parity command, never relabeled as msn2026_v1.

## 8 FIX-03: leakage-controlled protocol

Strict group membership is chosen before normalization. P12 group hashes use the
locked patient salt and ascending SHA256 allocation with floor 80/10/10.
Normalization sees only observed training values. Manifests record dataset and
protocol IDs, counts, salted public group hashes, internal fit-ID hash, split
hash, feature statistics, scale floor, code hash, and data asset hashes. Raw IDs
never enter public artifacts. Tests prove group disjointness, train-only fit,
test-extreme invariance, deterministic hashes, and no raw-ID leakage.

USHCN first audits station overlap in the official fold; it keeps a disjoint fold
or applies the locked station hash repair. HumanActivity never infers participant
identity from reset sample IDs. MIMIC-III is blocked unless legal access is
present and documented.

## 9 FIX-04: confirmatory inference

Evaluation emits one row per group, checkpoint, and variant with SSE, SAE, and N;
duplicate windows aggregate by summation. Point estimates are pooled sums divided
by pooled N. Each of 50,000 draws independently samples multiplicities over the
global group set and checkpoint set, then applies the same crossed products to
all variants. Seed 20260721 is fixed. The output includes absolute and relative
effects, percentile 95% intervals, patient/group macro descriptions, improvement
fractions, and Holm-adjusted comparison families. Datasets without reliable group
IDs receive seed-level summaries, not crossed patient claims.

A known-answer constant-effect fixture must return its exact point effect and a
degenerate interval; unequal-N, duplicate-row, row-order, and paired-multiplicity
tests protect the estimand.

## 10 FIX-05: segmented timing

The timing schema separates APN load, feature extraction, cache read/write, SLRH
candidate solves, CFG candidate solves, validation scoring/selection,
serialization, and warm batch-1 inference. Each record names CPU or CUDA and
contains warmup/repetition settings. CUDA phases synchronize before start and
after end. Closed-form solves on CPU are labeled CPU. Peak CUDA allocation and
process RSS are separate. RTX 4090 results are workstation results. Edge gates
remain blocked until a real edge CPU or Jetson record exists.

## 11 FIX-06: unified run schema

RunManifest transitions created -> running -> complete or failed. Complete runs
must contain the resolved config/hash, assets, cache manifest, split and
normalizer manifests, variant definition/hash, selected hyperparameters,
segmented timing, SSE/SAE/N cells, metrics, required-file hashes, environment,
argv, log path, and error-free status. Failures retain the error and completed
phases. Aggregation accepts only complete manifests whose required files and
hashes validate. It never silently drops a failed seed.

## 12 Variant registry

| ID | Definition | Special validation |
|---|---|---|
| APN | frozen forecast | no fitted state |
| SLRH | target latent ridge | one alpha curve |
| CFG | zero-diagonal other-sensor forecast ridge | anchor=source=APN |
| Full | SLRH then CFG | CFG anchor=SLRH; source=APN |
| V01 | residual Bias-only | train observed rows only |
| V02 | per target/horizon Self-Affine | unpenalized intercept |
| V03 | decoder-only refit | non-decoder weights byte-identical |
| V07 | CFG then SLRH | stage order explicit |
| V08 | joint latent+cross forecast ridge | full 36-cell grid |
| V10 | Full-Diagonal | exact SLRH state reused |
| V11 | Full-CrossShuffle | only CFG source rows shuffled |
| V12 | SLRH-LatentShuffle | only latent rows shuffled |

V03 initializes from the same checkpoint and enables gradients only for
model.model.decoder parameters. It uses AdamW, up to 100 epochs, patience 10,
learning rates 1e-4/3e-4/1e-3 and weight decay 0/1e-4, selected on validation.

## 13 Campaign sequence

1. M0 asset audit, existing five tests, legacy cache/output metric parity, and
   immutable legacy hash manifest.
2. Implement FIX-01 and FIX-06 skeletons.
3. Implement FIX-02 and exhaustive stale-cache tests.
4. Implement FIX-03 and split/normalization tests.
5. Repair CFG semantics, factor generic ridge, and implement static controls.
6. Implement decoder refit and frozen-parameter tests.
7. Integrate FIX-05 timing and FIX-04 statistics.
8. Run every pre-test unit/integration test and APN mode forward parity at
   atol 1e-8, rtol 1e-7.
9. Freeze resolved config, variant registry, splits, normalizers, code/patch
   hashes, statistics settings, and a signed protocol ledger.
10. Train or locate paired APN checkpoints using train/validation only; failures
    stay in the matrix.
11. Open each new test once, run the frozen registry without tuning, then close
    the once-only ledger.
12. Aggregate gates, render compact tables/figures, benchmark available hardware,
    and build the audited return ZIP.

GPU jobs run sequentially on the single RTX 4090. CPU-only unit work may run
concurrently only when it cannot contend with or alter a GPU campaign.

## 14 Coverage and gates

G0 requires all tests, legacy metric parity, and APN forward parity. G1 requires
no leakage, overlap, stale cache, pairing, missing run, or provenance failure.
G2 evaluates order, joint, capacity, diagonal, and shuffle explanations. G3
classifies strict datasets as strong/supportive/neutral/harmful or
safety-inconclusive; broad claims require at least ceiling(0.75D) strong datasets
and none harmful. G4 requires state at most 1 MiB, warm batch-1 p95 overhead at
most 10%, memory overhead at most 10%, and update-128 at most 60 seconds on a
real edge target. Unavailable data/checkpoints/hardware are explicit BLOCKED
cells and narrow the claim.

## 15 Pre-coding checklist

- Workspace boundary: verified msn2 only.
- Git branch: lab/msn2026-full-benchmark.
- Python/PyTorch: 3.11.13 and 2.6.0+cu124.
- GPU: one RTX 4090 available; no edge device detected.
- APN commit: pinned and locally patched; APN-mode parity required.
- Data: P12 assets present; HumanActivity, USHCN, and MIMIC-III absent.
- Checkpoints: 2024--2026 present and finite; 2027/2028 absent.
- Existing tests: five passed.
- Legacy parity: cache/output arrays identical and metric error <= 5.55e-17.
- Run strategy: project-root commands, sequential GPU, no new test before freeze.
- Documentation: this guide, idea report, user requirements, append-only dev log,
  and code README are the synchronized How to Run sources.

The absent assets prevent the full four-dataset/five-checkpoint campaign today,
but they do not block implementing and validating the complete protocol or
running the P12 cells whose required assets exist.

---

# Archived Implementation Guide ? EviPatch Stage A


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

`run_checked(command: list[str], log_path: Path, env: dict[str, str]) -> dict` runs a process, streams stdout/stderr to a project-local log, records wall time and return code, parses child-reported `torch.cuda` peak allocated/reserved memory, falls back to process-scoped `nvidia-smi` only when the marker is absent, and fails on nonzero exit.

`discover_checkpoints(result_root: Path, variant: str) -> list[Path]` requires exactly one checkpoint per configured seed and sorts by `iterN`.

`write_run_manifest(...) -> Path` records git hashes, commands, dependency snapshot, data paths, GPU, timestamps, and artifact hashes.

`main(argv: list[str] | None = None) -> int` exposes validation, smoke/timing, train/evaluate, strict serial Stage A, full result audit, controlled-support, aggregation, packaging, and status actions.

### 5.5 `audit.py`

`audit_shift_views(views: Mapping[str, Mapping[str, ndarray]], rate: float) -> dict` verifies within-run sample/target/time invariance, exact `floor(rate*n)` requests, requested/actual equality, matched MCAR/burst counts, remaining-count accounting, history-mask counts, zeroed removed values, and unchanged retained values.

`audit_stage_a(config: dict) -> dict` checks all 21 training manifests and 63 evaluation views, recomputes masked MSE/MAE, requires child-side nonzero CUDA peaks, checks checkpoint hashes and provenance, and verifies every variant uses byte-identical seed/shift data and masks; it writes `artifacts/stage_a_audit.json` and fails closed on any discrepancy.

### 5.6 `controlled.py`

`patch_units(...)` reconstructs APN's learned soft windows and normalized value/time-embedding centroids from the APN checkpoint and saved native histories, then retains one maximum-support patch per patient/channel.

`match_support_pairs(...)` robustly standardizes centroid coordinates within seed/channel and greedily freezes disjoint nearest pairs under the predeclared distance/support/yield thresholds without reading model errors.

`score_frozen_pairs(...)` applies identical pair IDs to all variants and computes masked channel-level target MSE; `run_controlled_support(...)` writes pair, error, and yield audits.

### 5.7 `aggregate.py`

`load_evaluation(path: Path) -> Evaluation` loads `metric.json`, saved prediction/target/mask/sample-ID arrays, and run manifest; it validates shape and finite values.

`masked_errors_per_patient(pred: ndarray, true: ndarray, mask: ndarray, ids: ndarray) -> DataFrame` computes patient MSE/MAE using only target-mask entries.

`hierarchical_paired_bootstrap(left: DataFrame, right: DataFrame, seed_column: str, id_column: str, metric: str, n_resamples: int, rng_seed: int) -> dict` resamples seeds, then patient IDs within seeds, and returns mean paired difference, 95% percentile interval, and probability of improvement.

`summarize_stage_a(root: Path, config: dict) -> tuple[DataFrame, DataFrame]` produces per-run and aggregate tables for native/MCAR/burst and macro averages.

`decide_gate(summary: DataFrame, bootstrap: DataFrame, config: dict) -> dict` implements every condition in `idea_report.md` without manual overrides and returns only `PASS` or `ABANDON` with machine-readable reasons. Time overhead comes from the frozen real-P12 100-step artifact, not end-to-end wall time.

`run_aggregation(config: dict) -> dict` first requires the full audit to pass, then regenerates controlled-support/statistical artifacts and writes a Chinese report with readable gate evidence, audit provenance, failure-cause interpretation, and the frozen stop/extension decision.

### 5.8 `package.py`

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
- Reset `torch.cuda` peak counters immediately before the run and emit one machine-readable allocated/reserved-memory marker in a `finally` block, so Windows WDDM runs retain true child-process CUDA peaks even when `nvidia-smi` omits compute-process memory.
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
| `run_manifest.json` | JSON | variant, seed, shift, full command argv, git hashes, environment, GPU, peak CUDA allocated/reserved MiB and source, start/end, return code |
| `stage_a_audit.json` | JSON | 21/63 completeness, metric recomputation, shift/target/cross-variant checks, CUDA/provenance checks, failures |
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

---

## Active sealed-return implementation (final)

The EdgeTwinCal confirmatory campaign is complete and immutable. The active
result/report surface is:

- `aggregate_v2.py` and `statistics.py`: 180-manifest audit, pooled micro
  metrics, group-by-checkpoint crossed paired bootstrap, raw percentile 95% CI,
  and Holm-adjusted one-sided bootstrap p-values.
- `lab_report.py`: rehash G0/G1 evidence, verify exactly one closed/sealed
  opening per cell, build the non-overridable gate, replay only train/validation
  fit caches for diagnosis, read test-side evidence only from sealed manifest
  error cells, and render the Chinese report/SVG/provenance.
- `build_edgetwincal_tables.mjs`: write the four traceable CSV tables and the
  Results, Seed metrics, Paired CIs, Gate audit, and Provenance workbook sheets
  through `@oai/artifact-tool`.
- `package.py`: collect the closed delivery whitelist, scan private paths and
  secrets (including expanded XLSX XML), reject forbidden/oversized sources,
  create a deterministic ZIP, and verify CRC/member SHA256.

The final gate is ABANDON: G2 mechanism FAIL; strict G3 FAIL with P12 strong and
USHCN harmful; release broad-scope FAIL because group IDs are unreliable; G4
BLOCKED because no real edge CPU/Jetson target exists. The P12 result may be
reported only as dataset-specific. The current APN route stops after attempt
five, and the already opened tests cannot be used to select a replacement.

### Final output contract

`artifacts/edgetwincal_msn2026_v1/analysis` contains the formal aggregate,
manifest registry, blockers, terminal pre-test audit, gate decision, failure
diagnosis, analysis provenance, Chinese report, two SVG figures, four CSV
tables, and `EdgeTwinCal_lab_results.xlsx`. `packages` contains only the final
ZIP and its external verification CSV. Datasets, environments, caches,
checkpoints, NPZ/PT/PDF files, `vendor/APN`, manuscripts, secrets, private
absolute paths, and files above 100 MB are forbidden.

### Final How to Run

```powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path

# Regenerate the sealed JSON/Markdown/SVG return without opening test data.
& $edgeTwinPython .\code\scripts\render_edgetwincal_results.py

# Validate the complete project-owned implementation.
& $edgeTwinPython -m pytest .\code\tests -q -p no:cacheprovider

# Create and self-verify the filtered deterministic archive.
& $edgeTwinPython .\code\scripts\package_edgetwincal.py
```

For workbook regeneration, run `build_edgetwincal_tables.mjs` from an ignored
project-local temporary directory whose `node_modules` is a junction to the
Codex bundled dependency runtime, pass the repository root explicitly, render
all five sheets to temporary PNGs, inspect the declared ranges, and require a
zero-result formula-error scan before exporting XLSX.
