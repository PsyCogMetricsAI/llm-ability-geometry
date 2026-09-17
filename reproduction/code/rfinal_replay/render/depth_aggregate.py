"""Fresh recomputation of the G4 depth summary from per-model/per-layer rows.

This module re-implements the frozen G4 descriptive aggregation rule
(ext_P2_geo_20260914/h4/code/build_profiles.py, protocol h4/G4_PROTOCOL.md)
so that the depth profiles figure and any freshly computed summary are
regenerated from per-model/per-layer measurements instead of consuming the
archived ``depth_summary.csv`` aggregate as a computational input.

Frozen rule
-----------
* each model contributes one ``(delta = (n_layers - layer_index)/n_layers)``
  per stored layer, sorted shallow -> deep;
* fixed grid ``d`` in {0.05, 0.10, ..., 1.00};
* a model is eligible at grid depth ``d`` iff its stored depth interval covers
  ``d`` (no extrapolation);
* ``r2_b``, ``r2_eps`` and ``beta`` are linearly interpolated per model
  (``numpy.interp``);
* each grid row holds ``n_models`` = number of eligible models,
  ``median_r2_b`` / ``median_r2_eps`` = medians over eligible models,
  ``rho_H2`` = Spearman rho between ``-beta`` and ``s`` over eligible models
  (average ranks, scipy-compatible);
* the per-metric peak is the first (shallowest) grid row attaining the maximum.
"""
from __future__ import annotations

import numpy as np

SUMMARY_COLUMNS = ("relative_depth", "n_models", "median_r2_b", "rho_H2", "median_r2_eps")
PEAK_METRICS = ("median_r2_b", "rho_H2", "median_r2_eps")


def grid_depths(n: int = 20):
    return [round(i / n, 10) for i in range(1, n + 1)]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Fractional (average) ranks for ties; matches scipy.stats.rankdata default."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_rho(a, b) -> float:
    ra, rb = _average_ranks(np.asarray(a, dtype=float)), _average_ranks(np.asarray(b, dtype=float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _per_model_arrays(profile_rows):
    arrays = {}
    for row in profile_rows:
        model = row["model_id"]
        rec = {
            "relative_depth": float(row["relative_depth"]),
            "r2_b": float(row["r2_b"]),
            "beta": float(row["beta"]),
            "r2_eps": float(row["r2_eps"]),
        }
        arrays.setdefault(model, {"s": float(row["s"]), "rows": []})["rows"].append(rec)
    for model, blob in arrays.items():
        blob["rows"].sort(key=lambda r: r["relative_depth"])
        depths = [r["relative_depth"] for r in blob["rows"]]
        if len(set(depths)) != len(depths):
            raise ValueError(f"duplicate relative_depth rows for model {model}")
    return arrays


def recompute_summary(profile_rows, depths=None):
    """Return the fresh G4 summary rows (list of dicts, float values)."""
    arrays = _per_model_arrays(profile_rows)
    if not arrays:
        raise ValueError("depth_profiles input contains no rows")
    models = sorted(arrays)
    grid = grid_depths() if depths is None else list(depths)
    summary = []
    for d in grid:
        eligible = [m for m in models
                    if arrays[m]["rows"][0]["relative_depth"] <= d <= arrays[m]["rows"][-1]["relative_depth"]]
        if not eligible:
            raise ValueError(f"no eligible model at grid depth {d}")
        bs, eps, betas = [], [], []
        for m in eligible:
            xs = [r["relative_depth"] for r in arrays[m]["rows"]]
            bs.append(float(np.interp(d, xs, [r["r2_b"] for r in arrays[m]["rows"]])))
            eps.append(float(np.interp(d, xs, [r["r2_eps"] for r in arrays[m]["rows"]])))
            betas.append(float(np.interp(d, xs, [r["beta"] for r in arrays[m]["rows"]])))
        rho = spearman_rho(-np.asarray(betas), [arrays[m]["s"] for m in eligible])
        summary.append({
            "relative_depth": float(d),
            "n_models": len(eligible),
            "median_r2_b": float(np.median(bs)),
            "rho_H2": rho,
            "median_r2_eps": float(np.median(eps)),
        })
    return summary


def recompute_peaks(summary):
    """Per-metric grid maxima; first (shallowest) row wins ties."""
    peaks = []
    for metric in PEAK_METRICS:
        best = summary[0]
        for row in summary[1:]:
            if row[metric] > best[metric]:
                best = row
        peaks.append({"metric": metric, "relative_depth": best["relative_depth"],
                      "value": best[metric], "n_models": best["n_models"]})
    return peaks


def compare_rows(fresh, archive, rel_tol: float = 1e-6, abs_tol: float = 1e-9):
    """Compare fresh summary rows with an archived aggregate (comparison only).

    Returns a report dict; never raises on mismatch and never feeds values back
    into the computation.
    """
    report = {"columns": {}, "row_count_fresh": len(fresh), "row_count_archive": len(archive),
              "tolerance": {"abs": abs_tol, "rel": rel_tol}, "max_abs_diff": 0.0,
              "within_tolerance": True, "mismatch_cells": []}
    if len(fresh) != len(archive):
        report["within_tolerance"] = False
        report["mismatch_cells"].append({"column": "*", "row": None, "detail": "row count differs"})
        return report
    for col in SUMMARY_COLUMNS:
        worst = 0.0
        for i, (f, a) in enumerate(zip(fresh, archive)):
            fv, av = float(f[col]), float(a[col])
            diff = abs(fv - av)
            if diff > abs_tol + rel_tol * abs(av):
                report["within_tolerance"] = False
                report["mismatch_cells"].append({"column": col, "row": i,
                                                 "fresh": fv, "archive": av, "abs_diff": diff})
            worst = max(worst, diff)
        report["columns"][col] = {"max_abs_diff": worst}
        report["max_abs_diff"] = max(report["max_abs_diff"], worst)
    return report
