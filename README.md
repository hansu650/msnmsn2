# EdgeTwinCal Experimental Repository

This repository contains the reproducible pilot implementation, ablations, compact
results, figures, and IEEE conference draft for:

> **EdgeTwinCal: Dual-Space Calibration for Frozen Irregular-Sensor Digital Twins**

Target track: **IEEE MSN 2026 - Edge Computing, IoT and Digital Twins**.

EdgeTwinCal keeps pre-existing APN checkpoints frozen and fits two lightweight,
closed-form residual modules:

- **SLRH** (Sensor Latent Residual Head) reads sensor-local corrections from the
  frozen APN latent representation.
- **CFG** (Cross-Forecast Graph) models the remaining residual from other sensors'
  intermediate forecasts with a zero diagonal.

## Pilot result

On PhysioNet 2012, across locally trained APN checkpoints for seeds 2024, 2025,
and 2026, frozen APN obtains MSE `0.312331 +/- 0.000512`; EdgeTwinCal obtains
`0.309058 +/- 0.000494`, a `1.048%` relative reduction. This is an exploratory
result because five structural routes were screened on the same test set; an
untouched holdout is required for confirmation.

## Repository map

- `code/src/edgetwincal/`: final implementation
- `code/configs/edgetwincal.json`: experiment configuration
- `code/scripts/`: run and aggregation entry points
- `code/tests/test_edgetwincal.py`: unit and contract tests
- `artifacts/edgetwincal_*.{csv,json}`: compact three-seed results and paired
  bootstrap outputs
- `notebooks/figures.ipynb`: reproducible paper figures
- `docs/manuscripts/paper.tex`: IEEE LaTeX source
- `docs/manuscripts/paper.pdf`: compiled four-page draft
- `docs/manuscripts/EXPLANATION_CN.md`: Chinese explanation of the paper
- `docs/baselines/APN_AAAI2026.md`: official APN paper/code links and provenance
- `docs/dualcross_route.md`: five-attempt route ledger

## Quick verification

From the repository root, using the existing project-local environment:

```powershell
$python = '.\.conda\envs\evipatch\python.exe'
& $python -m pytest code\tests\test_edgetwincal.py -q
& $python code\scripts\aggregate_edgetwincal.py
```

Raw data, cached APN representations, checkpoints, prediction arrays, the local
Conda environment, and the complete upstream APN source are intentionally not
versioned. See `docs/baselines/APN_AAAI2026.md` for the pinned upstream source.

## Historical EviPatch audit

The earlier EviPatch route failed its predeclared Stage A gate and is retained as
an immutable negative-result audit. Its verdict is **ABANDON**; it is not the
current manuscript route. Historical summaries already tracked in `artifacts/`
and `docs/` should not be interpreted as EdgeTwinCal results.
