# APN Baseline Reference and Provenance

EdgeTwinCal uses APN as its frozen irregular-time-series forecasting backbone.
The baseline is:

> X. Liu, X. Qiu, X. Wu, Z. Li, C. Guo, J. Hu, and B. Yang,
> "Rethinking Irregular Time Series Forecasting: A Simple Yet Effective
> Baseline," *Proceedings of the AAAI Conference on Artificial Intelligence*,
> vol. 40, no. 28, pp. 23873--23881, 2026.

Official resources:

- Article page: <https://ojs.aaai.org/index.php/AAAI/article/view/39563>
- Publisher PDF: <https://ojs.aaai.org/index.php/AAAI/article/download/39563/43524>
- DOI: <https://doi.org/10.1609/aaai.v40i28.39563>
- Official GitHub repository: <https://github.com/decisionintelligence/APN>
- Audited upstream commit: `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`
- APN model at that commit:
  <https://github.com/decisionintelligence/APN/blob/f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4/models/APN.py>
- P12 script at that commit:
  <https://github.com/decisionintelligence/APN/blob/f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4/scripts/APN/P12.sh>

## What is and is not stored here

The publisher PDF is linked rather than re-hosted because its article page bears
the publisher's copyright notice. The complete `vendor/APN` checkout is also not
committed because no confirmed top-level redistribution license was found during
the audit. Reproduction therefore uses the official repository at the pinned
commit plus this repository's documented local integration and existing patch:

- `patches/apn_evipatch.patch` preserves the earlier EviPatch audit changes.
- EdgeTwinCal itself is implemented outside `vendor/APN` and consumes cached
  outputs from pre-existing checkpoints trained locally with the released APN
  implementation.

No author-published checkpoint is claimed. The three paired checkpoints used in
the pilot were trained locally from the released implementation for seeds 2024,
2025, and 2026, then kept frozen for every EdgeTwinCal ablation.

## Protocol compatibility notes

The released P12 implementation differs from the paper description in details
that matter for interpretation: it produces an approximately 81/9/10 split,
drops incomplete training and validation batches, fits the standardizer before
the split, and uses Adam in code. Every paired EdgeTwinCal variant inherits the
same behavior for checkpoint compatibility. The manuscript reports these facts
and does not call the experiment a leakage-free re-evaluation.
