# EviPatch Experimental Repository

This repository is an isolated implementation and experimental audit of EviPatch, an evidence-preserving extension to APN's temporal adaptive patch aggregation.

The project follows a predeclared kill-test: smoke-test the implementation, then compare EviPatch against APN and decisive count-based controls on PhysioNet with three seeds. Expansion to other datasets is conditional on passing the gate recorded in `docs/idea_report.md`.

No paper manuscript is generated in this repository.

## Stage A outcome

Stage A is complete and the frozen verdict is **ABANDON**. EviPatch full did not reach the controlled-support improvement threshold, did not significantly beat `raw_count`, and was significantly worse than `shuffled_evidence`; therefore the conditional HumanActivity, USHCN, and t-PatchGNN extensions were not run.

- Chinese result and failure analysis: `artifacts/REPORT_CN.md`
- Machine-readable gate: `artifacts/gate_decision.json`
- Full 21-training/63-evaluation audit: `artifacts/stage_a_audit.json`
- Three-seed summaries and paired bootstrap: `artifacts/stage_a_summary.csv`, `artifacts/paired_bootstrap.csv`
