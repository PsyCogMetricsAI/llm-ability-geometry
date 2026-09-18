# T3 Reproduction Pipeline

Independent reproduction of the paper's main association results
(Table 1: code/math; Table 2: native science) from shipped response
matrices and geometry composites.

## What it does

1. **Re-fits IRT theta** from the 0/1 response matrices (not from saved
   theta) using the same estimators as the paper:
   - Code/math: R `mirt` 1-factor 2PL, `NCYCLES=2000`, EAP scores
   - Science: Python `girth` `twopl_mml` + `ability_eap` on the
     `fit_mask` subset (447 of 448 items)

2. **Verifies** that the re-fit theta exactly matches the shipped reference
   theta (max absolute difference, rank correlation).

3. **Computes the paper's association statistics** and compares each to the
   published Table 1 and Table 2 values at display precision (3 decimal
   places):
   - `raw_rho`: Pearson correlation of average-tie ranks of (geometry, theta)
   - `partial_rho | Z`: partial rank correlation controlling for
     [log10_params, d_model, ordinal_family_code, mean_logprob]
   - 95% CI: family-block bootstrap, seed 20260605, 10000 replicates

## Environment

### R (for code/math theta)

- R >= 4.3 with `mirt` >= 1.47
- On R < 4.5, `mirt` needs an archived `Deriv`:
  ```
  install.packages("https://cran.r-project.org/src/contrib/Archive/Deriv/Deriv_4.1.6.tar.gz",
                    repos = NULL, type = "source")
  install.packages("mirt")
  ```
- Set `R_LIBS` to the library path containing `mirt`.

### Python (for science theta + associations)

- Python 3.10+ with:
  - `girth` 0.8.0
  - `numpy` >= 2.0
  - `scipy` >= 1.10

## Running

From the `reader/` directory:

```bash
PYTHON=/path/to/python \
R_LIBS=/path/to/rlib \
bash run_t3.sh
```

Outputs go to `reader/outputs/`:

```
outputs/
  theta_codemath/    # Re-fit code/math theta CSVs + reliability
  theta_science/     # Re-fit science theta JSON
  associations/      # results.json with all statistics
```

## Files

```
reader/
  README.md                              # This file
  run_t3.sh                              # Orchestrator script
  scripts/
    fit_theta_codemath.R                 # R mirt re-fit for code & math
    fit_theta_science.py                 # girth re-fit for science
    compute_associations.py              # Association math + comparison
  inputs/                                # Shipped data (read-only)
    response_matrix_code.csv             # 51 models x 170 items
    response_matrix_math.csv             # 51 models x 1333 items
    response_matrix_science.npz          # 448 items x 43 models
    geometry_composite_50.json           # Shipped geometry composites
    covariates_50.json                   # Scale, family, fluency controls
    science_features_native.json         # 7 geometry indicators + controls
    theta_reference_panel51.json         # Reference theta for verification
    theta_reference_science.json         # Reference science theta
    SHA256SUMS                           # Input integrity checksums
```

## Scope — what this pipeline does and does not establish

This T3 pipeline is the **recommended end-to-end reproduction of the paper's main results**. Starting from the shipped 0/1 response matrices and derived per-model geometry vectors, it **independently re-fits the IRT ability (θ)** and **recomputes** the raw, covariate-adjusted, and family-block-interval associations, reproducing **Table 1 (code/math)** and **Table 2 (native science)** to the paper's displayed precision.

It establishes:
- the reported θ scores are reproducible by re-running the exact estimators (not merely re-displayed);
- the reported associations and their family-block intervals follow from θ + the geometry summaries under the documented method.

It does **not** cover (by design):
- **Model inference or activation extraction** — no model weights are used; the geometry summaries are shipped as derived vectors, not recomputed from hidden states.
- The **accuracy-conditioned intervals**, medical estimators, calibration, disjoint-item, window and other supplementary analyses — those are the accepted saved results (see the table exporter, level B) or the optional full cached replay (T2, see `T2_FULL_REPLAY.md`).

### Important: native vs rarefied science
Table 2 uses the **native** (unrarefied) activation-cloud indicators, shipped here in `inputs/science_features_native.json`. The archived full-replay graph (T2) also contains a separate **rarefied** 28-cell science grid; its numbers differ (e.g. PR partial ≈ −0.652 rarefied vs **−0.637 native**) and are a **sensitivity analysis, not Table 2**. Do not substitute one for the other.
