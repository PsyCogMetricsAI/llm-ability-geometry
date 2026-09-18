#!/usr/bin/env python3
"""Compute Table 1 (code/math) and Table 2 (native science) associations.

Usage:
    python scripts/compute_associations.py inputs/ outputs/theta_codemath/ \
           outputs/theta_science/ outputs/associations/

Reads:
  - inputs/geometry_composite_50.json
  - inputs/covariates_50.json
  - inputs/science_features_native.json
  - outputs/theta_codemath/theta_2pl_{code,math}.csv  (re-fit theta)
  - outputs/theta_science/theta_science.json           (re-fit theta)
  - inputs/theta_reference_panel51.json                (for verification)
  - inputs/theta_reference_science.json                (for verification)

Writes:
  - outputs/associations/results.json
  - stdout: comparison table with MATCH/MISMATCH
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.stats import rankdata

SEED = 20260605
N_BOOT = 10000


# ── core statistical functions ─────────────────────────────────────────

def pearson(x, y):
    """Pearson correlation of two arrays."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    a, b = a - a.mean(), b - b.mean()
    denom = math.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom else float("nan")


def design(c):
    """Build design matrix: intercept + ranked covariates."""
    c = np.asarray(c, float)
    if c.ndim == 1:
        c = c[:, None]
    return np.column_stack(
        [np.ones(len(c))] + [rankdata(c[:, j]) for j in range(c.shape[1])]
    )


def rho(x, y, c=None):
    """Partial rank correlation (Pearson of OLS residuals of ranks)."""
    a, b = rankdata(x), rankdata(y)
    if c is not None:
        d = design(c)
        a = a - d @ np.linalg.lstsq(d, a, rcond=None)[0]
        b = b - d @ np.linalg.lstsq(d, b, rcond=None)[0]
    return pearson(a, b)


def family_block_bootstrap(x, y, c, families, seed, n_boot):
    """Family-block bootstrap with percentile CI.

    Resample families WITH replacement; within each drawn family, keep all
    its models. Re-rank inside each replicate. Skip invalid draws.
    """
    rng = np.random.default_rng(seed)
    family_order = sorted(set(families.tolist()))
    n_families = len(family_order)
    blocks = [np.where(families == f)[0] for f in family_order]

    values = np.full(n_boot, np.nan)
    for i in range(n_boot):
        choices = rng.choice(n_families, size=n_families, replace=True)
        idx = np.concatenate([blocks[k] for k in choices])
        xb, yb = x[idx], y[idx]
        if len(idx) < 3 or len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        cb = None if c is None else c[idx]
        if cb is not None:
            cc = cb[:, None] if cb.ndim == 1 else cb
            if any(len(np.unique(cc[:, j])) < 2 for j in range(cc.shape[1])):
                continue
            d = design(cc)
            if np.linalg.matrix_rank(d) < d.shape[1]:
                continue
        values[i] = rho(xb, yb, cb)

    finite = values[np.isfinite(values)]
    if len(finite):
        ci = np.percentile(finite, [2.5, 97.5]).tolist()
    else:
        ci = [None, None]
    n_valid = int(len(finite))
    return ci, n_valid


# ── loading helpers ────────────────────────────────────────────────────

def load_refit_theta_codemath(theta_dir):
    """Load re-fit theta from R output CSVs. Returns dict[domain][model] = theta."""
    result = {}
    for dom in ("code", "math"):
        path = os.path.join(theta_dir, f"theta_2pl_{dom}.csv")
        with open(path) as f:
            reader = csv.DictReader(f)
            result[dom] = {row["model"]: float(row["theta"]) for row in reader}
    return result


def load_refit_theta_science(theta_dir):
    """Load re-fit science theta. Returns list of 43 floats."""
    path = os.path.join(theta_dir, "theta_science.json")
    with open(path) as f:
        data = json.load(f)
    return data["theta"]


# ── main ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python compute_associations.py <input_dir> "
            "<theta_codemath_dir> <theta_science_dir> <output_dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    theta_cm_dir = sys.argv[2]
    theta_sci_dir = sys.argv[3]
    output_dir = sys.argv[4]
    os.makedirs(output_dir, exist_ok=True)

    # ── Load geometry composite ────────────────────────────────────────
    geo_data = json.load(
        open(os.path.join(input_dir, "geometry_composite_50.json"))
    )
    geo_models = geo_data["models"]

    # ── Load covariates ────────────────────────────────────────────────
    cov_data = json.load(
        open(os.path.join(input_dir, "covariates_50.json"))
    )
    cov_models = cov_data["models"]

    # model order = covariates order (50 models, the analysis panel)
    model_order = list(cov_models.keys())
    n = len(model_order)
    assert n == 50, f"Expected 50 models, got {n}"

    # ── Load re-fit theta (code/math) ──────────────────────────────────
    refit_theta = load_refit_theta_codemath(theta_cm_dir)

    # ── Load reference theta for verification ──────────────────────────
    ref_theta = json.load(
        open(os.path.join(input_dir, "theta_reference_panel51.json"))
    )

    # ── Verify re-fit theta matches reference ──────────────────────────
    print("=" * 70)
    print("THETA VERIFICATION: code/math re-fit vs reference")
    print("=" * 70)
    for dom in ("code", "math"):
        diffs = []
        for m in model_order:
            refit_val = refit_theta[dom][m]
            ref_val = ref_theta[dom][m]
            diffs.append(abs(refit_val - ref_val))
        max_diff = max(diffs)
        # rank correlation
        refit_arr = np.array([refit_theta[dom][m] for m in model_order])
        ref_arr = np.array([ref_theta[dom][m] for m in model_order])
        rank_rho = pearson(rankdata(refit_arr), rankdata(ref_arr))
        status = "PASS" if max_diff < 1e-6 else "WARN"
        print(
            f"  [{dom}] max_abs_diff = {max_diff:.2e}, "
            f"rank_rho = {rank_rho:.10f}  [{status}]"
        )

    # ── Table 1: code and math ─────────────────────────────────────────
    families_list = np.array(
        [cov_models[m]["code"]["family"] for m in model_order]
    )

    table1_targets = {
        "code": {
            "raw": -0.453,
            "partial": -0.232,
            "ci_lo": -0.494,
            "ci_hi": 0.077,
        },
        "math": {
            "raw": 0.329,
            "partial": 0.523,
            "ci_lo": 0.255,
            "ci_hi": 0.799,
        },
    }

    table1_results = {}
    for dom in ("code", "math"):
        x = np.array(
            [geo_models[m][dom]["geometry_composite_zpc1"] for m in model_order]
        )
        y = np.array([refit_theta[dom][m] for m in model_order])
        c = np.array(
            [
                [
                    cov_models[m][dom]["scale_log10_params"],
                    cov_models[m][dom]["d_model"],
                    cov_models[m][dom]["family_code"],
                    cov_models[m][dom]["fluency_mean_logprob"],
                ]
                for m in model_order
            ]
        )

        raw_val = rho(x, y)
        partial_val = rho(x, y, c)
        ci, n_valid = family_block_bootstrap(
            x, y, c, families_list, SEED, N_BOOT
        )

        table1_results[dom] = {
            "raw_rho": raw_val,
            "partial_rho": partial_val,
            "ci": ci,
            "n_valid": n_valid,
        }

    # ── Table 2: native science ────────────────────────────────────────
    sci_data = json.load(
        open(os.path.join(input_dir, "science_features_native.json"))
    )
    sci_families = np.array(sci_data["families"])
    sci_covariates = np.array(sci_data["covariates"])  # (43, 4)

    # Load re-fit science theta
    refit_sci_theta = np.array(load_refit_theta_science(theta_sci_dir))

    # Verify against reference
    ref_sci = json.load(
        open(os.path.join(input_dir, "theta_reference_science.json"))
    )
    ref_sci_theta = np.array(ref_sci["theta"])
    sci_max_diff = float(np.max(np.abs(refit_sci_theta - ref_sci_theta)))
    sci_rank_rho = pearson(rankdata(refit_sci_theta), rankdata(ref_sci_theta))
    sci_status = "PASS" if sci_max_diff < 1e-6 else "WARN"
    print(
        f"  [science] max_abs_diff = {sci_max_diff:.2e}, "
        f"rank_rho = {sci_rank_rho:.10f}  [{sci_status}]"
    )
    print("=" * 70)

    table2_indicators = [
        ("eff_rank_pr", "PR/eff_rank_pr"),
        ("rankme", "RankMe"),
        ("stable_rank", "Stable rank"),
        ("spectral_alpha", "Spectral decay"),
        ("vn_entropy", "VN entropy"),
        ("isoscore", "IsoScore"),
        ("twoNN_id", "TwoNN"),
    ]

    table2_targets = {
        "eff_rank_pr": {"raw": -0.607, "partial": -0.637, "ci_lo": -0.763, "ci_hi": -0.308},
        "rankme": {"raw": -0.416, "partial": -0.341, "ci_lo": -0.690, "ci_hi": 0.034},
        "stable_rank": {"raw": -0.561, "partial": -0.642, "ci_lo": -0.756, "ci_hi": -0.300},
        "spectral_alpha": {"raw": 0.545, "partial": 0.587, "ci_lo": 0.258, "ci_hi": 0.738},
        "vn_entropy": {"raw": -0.611, "partial": -0.579, "ci_lo": -0.698, "ci_hi": -0.202},
        "isoscore": {"raw": -0.651, "partial": -0.634, "ci_lo": -0.735, "ci_hi": -0.375},
        "twoNN_id": {"raw": 0.151, "partial": 0.243, "ci_lo": -0.174, "ci_hi": 0.551},
    }

    table2_results = {}
    for ind_key, ind_label in table2_indicators:
        x = np.array(sci_data["indicators"][ind_key])
        y = refit_sci_theta
        c = sci_covariates

        raw_val = rho(x, y)
        partial_val = rho(x, y, c)
        ci, n_valid = family_block_bootstrap(
            x, y, c, sci_families, SEED, N_BOOT
        )

        table2_results[ind_key] = {
            "label": ind_label,
            "raw_rho": raw_val,
            "partial_rho": partial_val,
            "ci": ci,
            "n_valid": n_valid,
        }

    # ── Comparison table ───────────────────────────────────────────────
    def fmt3(v):
        """Format to 3 decimal places for display precision."""
        return f"{v:.3f}"

    def match3(computed, target):
        """Check if computed matches target at 3 decimal places."""
        return round(computed, 3) == round(target, 3)

    print()
    print("=" * 90)
    print("TABLE 1: Code and Math (geometry_composite vs theta)")
    print("=" * 90)
    print(f"{'Domain':<8} {'Stat':<10} {'Target':>8} {'Computed':>10} {'Match':>8}")
    print("-" * 90)

    all_match = True
    for dom in ("code", "math"):
        r = table1_results[dom]
        t = table1_targets[dom]
        for stat_name, target_key, computed_val in [
            ("raw", "raw", r["raw_rho"]),
            ("partial", "partial", r["partial_rho"]),
            ("CI_lo", "ci_lo", r["ci"][0]),
            ("CI_hi", "ci_hi", r["ci"][1]),
        ]:
            target_val = t[target_key]
            m = match3(computed_val, target_val)
            if not m:
                all_match = False
            print(
                f"{dom:<8} {stat_name:<10} {fmt3(target_val):>8} "
                f"{fmt3(computed_val):>10} {'MATCH' if m else 'MISMATCH':>8}"
            )

    print()
    print("=" * 90)
    print("TABLE 2: Native Science (geometry indicators vs theta)")
    print("=" * 90)
    print(f"{'Indicator':<20} {'Stat':<10} {'Target':>8} {'Computed':>10} {'Match':>8}")
    print("-" * 90)

    for ind_key, ind_label in table2_indicators:
        r = table2_results[ind_key]
        t = table2_targets[ind_key]
        for stat_name, target_key, computed_val in [
            ("raw", "raw", r["raw_rho"]),
            ("partial", "partial", r["partial_rho"]),
            ("CI_lo", "ci_lo", r["ci"][0]),
            ("CI_hi", "ci_hi", r["ci"][1]),
        ]:
            target_val = t[target_key]
            m = match3(computed_val, target_val)
            if not m:
                all_match = False
            print(
                f"{ind_label:<20} {stat_name:<10} {fmt3(target_val):>8} "
                f"{fmt3(computed_val):>10} {'MATCH' if m else 'MISMATCH':>8}"
            )

    print()
    print("=" * 90)
    if all_match:
        print("OVERALL: ALL MATCH at display precision (3 decimal places)")
    else:
        print("OVERALL: SOME MISMATCH detected — see above")
    print("=" * 90)

    # ── Save results ───────────────────────────────────────────────────
    full_results = {
        "table1": {
            dom: {
                "raw_rho": table1_results[dom]["raw_rho"],
                "partial_rho": table1_results[dom]["partial_rho"],
                "ci": table1_results[dom]["ci"],
                "n_valid_bootstrap": table1_results[dom]["n_valid"],
            }
            for dom in ("code", "math")
        },
        "table2": {
            ind_key: {
                "label": table2_results[ind_key]["label"],
                "raw_rho": table2_results[ind_key]["raw_rho"],
                "partial_rho": table2_results[ind_key]["partial_rho"],
                "ci": table2_results[ind_key]["ci"],
                "n_valid_bootstrap": table2_results[ind_key]["n_valid"],
            }
            for ind_key, _ in table2_indicators
        },
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "all_match_display_precision": all_match,
    }

    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
