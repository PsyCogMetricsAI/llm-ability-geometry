# Limitations and preserved failures

- **Historical claims are not restored by cache replay.** The package reproduces
  the accepted C17 cache-based analyses; it does not claim that all 147 historical
  claims were recovered. Preregistration/protocol-adherence evidence remains
  unrecovered (`HISTORICAL_LIMIT` records in the accepted ledger).
- **Replay scope.** Results were produced through checkpoint continuation, not one uninterrupted regeneration from original model inference.
- **Pythia CPU/GPU failures preserved**: 41/43 strict comparisons fail at 1e-5 and
  are reported as failures, never relaxed.
- **Gemma window evidence**: 200 responses, 198 nonempty targets used; the
  short-window overlap and finalnorm constraints are retained.
- **Old six-model identity is unknown**; the new window evidence is a new analysis,
  not a historical six-model recovery.
- **New 28-cell grid analyses are new** (regularized 2PL MAP primary + Rasch
  comparator); they do not restore the historical custom 2PL MML-EM/EAP
  implementation, and `converged=false` metadata is preserved.
- **Bounded probes are descriptive**, not prediction/causal experiments.
- **Prior-16 LID / F-POOLING / empirical NBIAS / G-detail** become available only
  through the accepted supplement drivers; unsupported branches is
  explicitly uncomputable rather than silently filled.
- **No weights/credentials** are shipped; optional GPU source reruns require
  authorized acquisition plus hash verification. `/tmp` staging must be
  re-materialized from the declared digests.
