# Statistical source navigation

These files preserve the scientific implementations underlying the
paper-specific comparisons. `SOURCE_MAP.json` records public file hashes. Private machine paths and internal producer identities have been removed. They supplement the frozen replay engine in `../reproduction/code/`.

- `theta_mirt_panel51.R`: historical code/math mirt fits and EAP scores.
- `study1_reuse_analyze.py`, `study1_reuse_bind.py`: saved code/math panels and adjustment.
- `study1_accuracy_v2.py`: separately corrected accuracy-conditioned diagnostics.
- `study2_reuse_fit.py`: science girth fitting/scoring branch.
- `new_medical_repair*.py`: regularized 2PL and Rasch alternatives.
- `new_medical_associations.py`: medical comparisons and undefined residual handling.
- `study1_reuse_retention.py`: separate disjoint-item protocol.
- `window_cloud_metrics.py`: new fixed-version window sensitivity.

They retain original project-relative/historical paths and dependencies. They
are source references, **not new standalone quick-start commands**. Use
[the reproduction guide](../docs/REPRODUCING.md) for supported entrypoints and
the required archived input tree. No estimator or frozen parameter was changed.
