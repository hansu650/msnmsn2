# EdgeTwinCal sealed laboratory return

This directory contains the project-owned implementation and audit tooling for
the frozen EdgeTwinCal MSN 2026 campaign. APN remains pinned at upstream commit
`f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`; the ignored `vendor/APN` checkout
is not shipped. Reproduction uses `patches/apn_evipatch.patch` in APN-parity
mode plus the source under `code/src/edgetwincal`.

The confirmatory campaign is closed. Its five runnable cells contain 180/180
complete run manifests, each protocol ledger is sealed after exactly one test
opening, and no test token was persisted. Test caches must not be reopened.
Post-hoc failure diagnosis reads only train/validation fit caches and already
sealed run-manifest error cells.

The formal verdict is `ABANDON`: G2 mechanism evidence failed, strict G3 has one
strong P12 result and one harmful USHCN result, release-parity evidence is
descriptive only, and G4 is blocked without a real edge target. This was the
fifth structural route on APN; do not use the opened tests for a sixth route.

## Reproduce the compact return

Run from the repository root with the existing project-local environment:

```powershell
$edgeTwinPython = (Resolve-Path '.\.conda\envs\evipatch\python.exe').Path
$env:PYTHONPATH = (Resolve-Path '.\code\src').Path

# Rebuild gate, failure diagnosis, Chinese report, and SVG figures.
& $edgeTwinPython .\code\scripts\render_edgetwincal_results.py

# Build CSV/XLSX tables with the Codex bundled Artifact Tool. The script itself
# gives a clear error when that runtime is unavailable.
& node .\code\scripts\build_edgetwincal_tables.mjs --repo-root (Resolve-Path '.').Path

# Regression suite.
& $edgeTwinPython -m pytest .\code\tests -q -p no:cacheprovider

# Build and verify the deterministic filtered delivery archive.
& $edgeTwinPython .\code\scripts\package_edgetwincal.py
```

The table builder must be run with a Node resolution path that contains
`@oai/artifact-tool`; the Codex workspace runtime supplies it. The committed
XLSX and CSV artifacts are already generated and do not require Node merely to
inspect the result.

## Compact outputs

The audited outputs live in
`artifacts/edgetwincal_msn2026_v1/analysis`: formal aggregate, gate decision,
failure diagnosis, Chinese lab return, two SVG figures, four CSV tables, and
`EdgeTwinCal_lab_results.xlsx`. The package command includes only whitelisted
code, tests, compact manifests/logs, protocol evidence, and analysis artifacts.
