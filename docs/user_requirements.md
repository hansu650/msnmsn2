# User Requirements

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
