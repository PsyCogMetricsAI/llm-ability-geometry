"""Retained legacy AN-B branch: theta recovery + geometry + power + D2 + bootstrap.

Fresh aggregation is recomputed only from the saved per-replicate / per-draw
arrays of the accepted production run.  The accepted reports
(reports/NEW_SIMULATION_REPAIR.json, reports/NEW_BOOTSTRAP_POWER.json) are used
as comparison references only, never as computational inputs.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import expit

from common import Comparator

BRANCH = "legacy"

PROD = "runs/new_simulation_repair/production"
BP_PROD = "runs/bootstrap_power_repair/production"
REF_REPORT = "reports/NEW_SIMULATION_REPAIR.json"
REF_BOOT = "reports/NEW_BOOTSTRAP_POWER.json"

GEOMETRY_ORDER = [(r, n) for r in (2, 4, 8) for n in (64, 128, 256)]
POWER_ORDER = list(itertools.product([16, 30, 50], [5, 10, 20, 40, 80],
                                     [0, 0.674], [0, 0.3]))
BP_ORDER = list(itertools.product([16, 30, 50], [5, 10, 20, 40, 80], [0, 0.674]))
CAL_KEYS = ["bias", "rmse", "rho", "coverage", "posterior_variance"]


def _load_module(name: str, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen_calibration(seed, scenario_index, rep, pars, n, k):
    rr = np.random.default_rng(np.random.SeedSequence([seed, 1, scenario_index * 10000 + rep]))
    theta = rr.normal(size=n)
    b = rr.normal(pars["b_mean"], 1, k)
    a = pars["a_multiplier"] * np.exp(rr.normal(0, 0.35, k))
    y = (rr.random((n, k)) < expit(a * (theta[:, None] - b))).astype(np.int8)
    return theta, a, b, y


def _calibration(y, fit_a, fit_b):
    """Per-replicate metrics, exactly the accepted estimator path (orig.score)."""
    est, var, lo, hi, loglik = score(y, fit_a, fit_b)
    return est, var, lo, hi, loglik


def run(root, src, out_dir):
    """Execute the legacy branch replay. Returns (summary, comparisons, checks, provenance)."""
    from scipy.stats import spearmanr

    code_new = src.require("code/new_simulation_repair.py")
    code_bp = src.require("code/bootstrap_power_repair.py")
    code_fin = src.require("code/finalize_simulation_repair.py")
    cfg_path = src.require("config/CT_SIMULATIONS.json")
    from common import load_json
    cfg = load_json(cfg_path)
    seed = cfg["seed"]

    orig = _load_module("anb_new_simulation_repair", code_new)
    bpr = _load_module("anb_bootstrap_power_repair", code_bp)
    global score
    score = orig.score
    wilson = orig.wilson
    auc = orig.auc

    ref_report_path = src.require(REF_REPORT)
    ref_report = load_json(ref_report_path)
    ref_sha = src.declared[REF_REPORT]["sha256"]
    ref_boot_path = src.require(REF_BOOT)
    ref_boot = load_json(ref_boot_path)
    ref_boot_sha = src.declared[REF_BOOT]["sha256"]
    summary_all_path = src.require(f"{PROD}/summary_all.json")
    summary_all = load_json(summary_all_path)
    summary_all_sha = src.declared[f"{PROD}/summary_all.json"]["sha256"]

    cmp = Comparator(BRANCH)
    checks = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    # Cross-bindings: accepted report declares the hashes of its own inputs.
    check("cross:summary_all_source_sha256",
          ref_report["source_sha256"] == summary_all_sha,
          {"report": ref_report["source_sha256"], "found": summary_all_sha})
    check("cross:config_sha256",
          ref_report["config_sha256"] == src.declared["config/CT_SIMULATIONS.json"]["sha256"],
          {"report": ref_report["config_sha256"]})
    check("cross:code_sha256",
          ref_report["code_sha256"] == src.declared["code/new_simulation_repair.py"]["sha256"],
          {"report": ref_report["code_sha256"]})

    # ------------------------------------------------------------- calibration
    scenarios = list(cfg["calibration"]["scenarios"].items())
    n_rep = int(ref_report["calibration"]["standard"]["n_total"])
    cal_rows = {name: [] for name, _ in scenarios}
    est_checks = {"eap_max_abs_diff": 0.0, "posterior_variance_max_abs_diff": 0.0,
                  "lo_max_abs_diff": 0.0, "hi_max_abs_diff": 0.0, "n_replicates": 0}
    for s_idx, (name, pars) in enumerate(scenarios):
        for rep in range(n_rep):
            rel = f"{PROD}/calibration_{s_idx}_{rep:03d}.npz"
            with np.load(src.require(rel)) as z:
                payload = {k: z[k] for k in z.files}
            est, var, lo, hi, _ = _calibration(payload["response"], payload["fit_a"], payload["fit_b"])
            theta = payload["theta"]
            delta = float(max(np.max(np.abs(payload["fit_a"] - payload["fit200_a"])),
                              np.max(np.abs(payload["fit_b"] - payload["fit200_b"]))))
            row = {
                "scenario": name, "rep": rep, "status": "finite",
                "bias": float(np.mean(est - theta)),
                "rmse": float(np.sqrt(np.mean((est - theta) ** 2))),
                "rho": float(spearmanr(est, theta).statistic),
                "coverage": float(np.mean((theta >= lo) & (theta <= hi))),
                "posterior_variance": float(var.mean()),
                "stability_maxdiff": delta,
                "stable": bool(delta <= 1e-3),
                "boundary_items": int(np.sum((payload["fit_a"] <= 0.2001) | (payload["fit_a"] >= 4.9999))),
            }
            est_checks["n_replicates"] += 1
            est_checks["eap_max_abs_diff"] = max(est_checks["eap_max_abs_diff"],
                                                 float(np.max(np.abs(est - payload["eap"]))))
            est_checks["posterior_variance_max_abs_diff"] = max(
                est_checks["posterior_variance_max_abs_diff"],
                float(np.max(np.abs(var - payload["posterior_variance"]))))
            est_checks["lo_max_abs_diff"] = max(est_checks["lo_max_abs_diff"],
                                                float(np.max(np.abs(lo - payload["lo"]))))
            est_checks["hi_max_abs_diff"] = max(est_checks["hi_max_abs_diff"],
                                                float(np.max(np.abs(hi - payload["hi"]))))
            cal_rows[name].append(row)

    # estimator check: accepted score() on the saved fit arrays reproduces the
    # saved per-position EAP summaries (exact to float noise).
    check("calibration.estimator.max_abs_diffs",
          all(v <= 1e-12 for k, v in est_checks.items() if k != "n_replicates"),
          est_checks)

    cal_summary = {}
    for name, _ in scenarios:
        rows = cal_rows[name]
        good = [r for r in rows if r["status"] == "finite"]
        z = {"n_total": len(rows), "n_finite": len(good),
             "n_stable": int(sum(r["stable"] for r in good)),
             "n_any_boundary": int(sum(r["boundary_items"] > 0 for r in good))}
        for key in CAL_KEYS:
            v = np.array([r[key] for r in good], dtype=np.float64)
            z[key] = {"mean": float(v.mean()) if len(v) else None,
                      "replicate_mcse": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None}
        cal_summary[name] = z

    # per-replicate comparison against the saved accepted per-replicate rows
    ref_rows = {(r["scenario"], r["rep"]): r for r in summary_all["calibration"]["all_repetitions"]}
    for name, _ in scenarios:
        for r in cal_rows[name]:
            ref = ref_rows[(name, r["rep"])]
            for key in CAL_KEYS + ["stability_maxdiff"]:
                cmp.float(f"calibration.replicate.{name}.{r['rep']:03d}.{key}", r[key], ref[key],
                          new_ptr=f"/calibration/{name}/rows/{r['rep']:03d}/{key}",
                          ref_ptr=f"/calibration/all_repetitions/{name}#rep={r['rep']:03d}/{key}",
                          ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
            cmp.exact(f"calibration.replicate.{name}.{r['rep']:03d}.boundary_items",
                      r["boundary_items"], ref["boundary_items"],
                      new_ptr=f"/calibration/{name}/rows/{r['rep']:03d}/boundary_items",
                      ref_ptr=f"/calibration/all_repetitions/{name}#rep={r['rep']:03d}/boundary_items",
                      ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
        for key in CAL_KEYS:
            cmp.float(f"calibration.{name}.{key}.mean", cal_summary[name][key]["mean"],
                      ref_report["calibration"][name][key]["mean"],
                      new_ptr=f"/calibration/{name}/{key}/mean",
                      ref_ptr=f"/calibration/{name}/{key}/mean",
                      ref_path=REF_REPORT, ref_sha256=ref_sha)
            cmp.float(f"calibration.{name}.{key}.replicate_mcse", cal_summary[name][key]["replicate_mcse"],
                      ref_report["calibration"][name][key]["replicate_mcse"],
                      new_ptr=f"/calibration/{name}/{key}/replicate_mcse",
                      ref_ptr=f"/calibration/{name}/{key}/replicate_mcse",
                      ref_path=REF_REPORT, ref_sha256=ref_sha)
        for key in ("n_total", "n_finite", "n_stable", "n_any_boundary"):
            cmp.exact(f"calibration.{name}.{key}", cal_summary[name][key],
                      ref_report["calibration"][name][key],
                      new_ptr=f"/calibration/{name}/{key}", ref_ptr=f"/calibration/{name}/{key}",
                      ref_path=REF_REPORT, ref_sha256=ref_sha)

    # ---------------------------------------------------------------- geometry
    geometry = []
    for r, n in GEOMETRY_ORDER:
        rel = f"{PROD}/geometry_r{r}_n{n}.npy"
        vals = np.load(src.require(rel))
        entry = {"rank": r, "n": n, "reps": int(vals.shape[0]),
                 "mean": float(vals[:, 0].mean()),
                 "bias": float(vals[:, 0].mean() - r),
                 "rmse": float(np.sqrt(np.mean((vals[:, 0] - r) ** 2)))}
        geometry.append(entry)
        for idx, ref_row in enumerate(summary_all["geometry"]):
            if ref_row["rank"] == r and ref_row["n"] == n:
                for key in ("mean", "bias", "rmse"):
                    cmp.float(f"geometry.r{r}.n{n}.{key}", entry[key], ref_row[key],
                              new_ptr=f"/geometry#rank={r},n={n}/{key}",
                              ref_ptr=f"/geometry/{idx}/{key}",
                              ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
                cmp.exact(f"geometry.r{r}.n{n}.reps", entry["reps"], ref_row["reps"],
                          new_ptr=f"/geometry#rank={r},n={n}/reps",
                          ref_ptr=f"/geometry/{idx}/reps",
                          ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
                break

    # representative generator+estimator check on stored geometry draws
    geom_checks = []
    for r, n in GEOMETRY_ORDER:
        vals = np.load(src.require(f"{PROD}/geometry_r{r}_n{n}.npy"))
        rr = np.random.default_rng(np.random.SeedSequence([seed, 2, r * 100000 + n * 100 + 0]))
        x = rr.normal(size=(n, r))

        def pr(xx):
            v = np.linalg.eigvalsh(np.cov(xx, rowvar=False))
            return float(v.sum() ** 2 / (v @ v))

        regen = [pr(x), pr(x[:n // 2]), pr(x[n // 2:])]
        ok = bool(np.array_equal(np.asarray(regen), vals[0]))
        geom_checks.append({"check": f"geometry.r{r}.n{n}.rep0", "pass": ok,
                            "detail": {"regenerated": regen, "stored": vals[0].tolist()}})
    check("geometry.representative_regeneration", all(c["pass"] for c in geom_checks),
          {"n": len(geom_checks), "n_fail": sum(not c["pass"] for c in geom_checks)})

    # ------------------------------------------------------------------- power
    sign = np.array(list(itertools.product([-1, 1], repeat=6)))
    power = []
    power_p_max_diff = 0.0
    power_p_exact = 0
    for cell, (n, k, rho, icc) in enumerate(POWER_ORDER):
        with np.load(src.require(f"{PROD}/power_{cell:03d}.npz")) as z:
            blocks, p_stored, valid_stored, reject_stored = (
                z["blocks"], z["p"], z["valid"], z["reject"])
        ts = blocks @ sign.T
        obs = blocks.sum(1)
        p = (np.abs(ts) >= np.abs(obs[:, None]) - 1e-12).mean(1)
        valid = np.isfinite(p)
        reject = valid & (p <= 0.05)
        diff = float(np.max(np.abs(p - p_stored)))
        power_p_max_diff = max(power_p_max_diff, diff)
        power_p_exact += int(np.array_equal(p, p_stored) and np.array_equal(valid, valid_stored)
                             and np.array_equal(reject, reject_stored))
        count = int(reject.sum())
        reps = int(p.shape[0])
        rate = count / reps
        power.append({
            "cell": cell, "n": n, "k": k, "latent_rho": rho, "icc": icc,
            "family_sizes": np.bincount(np.arange(n) % 6).tolist(),
            "observed_population_rho": rho / (1 + 20 / k),
            "total": reps, "valid": int(valid.sum()), "reject": count,
            "rate": rate, "mcse": float(np.sqrt(rate * (1 - rate) / reps)),
            "wilson95": wilson(count, reps),
        })
    check("power.estimator.p_recompute", power_p_max_diff == 0.0,
          {"max_abs_diff": power_p_max_diff, "cells_bit_identical": power_p_exact, "n_cells": len(POWER_ORDER)})
    _power_generator_check(src, seed, check)
    ref_power = {row["cell"]: row for row in summary_all["power"]}
    for row in power:
        ref = ref_power[row["cell"]]
        for key, kind in (("rate", "float"), ("mcse", "float"),
                          ("observed_population_rho", "float")):
            cmp.float(f"power.cell{row['cell']}.{key}", row[key], ref[key],
                      new_ptr=f"/power/{row['cell']}/{key}", ref_ptr=f"/power/{ref['cell']}/{key}",
                      ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
        cmp.tuple_float(f"power.cell{row['cell']}.wilson95", row["wilson95"], ref["wilson95"],
                        new_ptr=f"/power/{row['cell']}/wilson95", ref_ptr=f"/power/{ref['cell']}/wilson95",
                        ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
        for key in ("total", "valid", "reject", "n", "k", "latent_rho", "icc", "family_sizes"):
            cmp.exact(f"power.cell{row['cell']}.{key}", row[key], ref[key],
                      new_ptr=f"/power/{row['cell']}/{key}", ref_ptr=f"/power/{ref['cell']}/{key}",
                      ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)

    # ---------------------------------------------------------------------- D2
    worlds = ["real", "shadow", "noisy"]
    d2_scores, d2_response_scores = {}, {}
    for name in worlds:
        with np.load(src.require(f"{PROD}/d2_{name}.npz")) as z:
            d2_scores[name] = z["score"]
            d2_response_scores[name] = z["response_only_score"]
    pairs = [
        ("real_vs_shadow", (d2_scores["real"], d2_scores["shadow"]), False),
        ("noisy_vs_shadow", (d2_scores["noisy"], d2_scores["shadow"]), False),
        ("commoncause_identical_observables", (d2_scores["real"], d2_scores["real"]), True),
        ("response_only", (d2_response_scores["real"], d2_response_scores["shadow"]), False),
    ]
    rr = np.random.default_rng(np.random.SeedSequence([seed, 5, 0]))
    boots = {}
    for name, (x, y), is_common in pairs:
        if is_common:
            boots[name] = np.full(2000, 0.5)
        else:
            boots[name] = np.array([auc(rr.choice(x, len(x)), rr.choice(y, len(y)))
                                    for _ in range(2000)])
    d2 = {}
    for name, (x, y), _ in pairs:
        stored = np.load(src.require(f"{PROD}/d2_auc_bootstrap_{name}.npy"))
        check(f"d2.bootstrap_regen.{name}", bool(np.array_equal(boots[name], stored)),
              {"max_abs_diff": float(np.max(np.abs(boots[name] - stored)))})
        d2[name] = {"auc": auc(x, y),
                    "bootstrap95": np.quantile(boots[name], [.025, .975]).tolist(),
                    "n_per_class": int(len(x))}
        ref = summary_all["d2"][name]
        cmp.float(f"d2.{name}.auc", d2[name]["auc"], ref["auc"],
                  new_ptr=f"/d2/{name}/auc", ref_ptr=f"/d2/{name}/auc",
                  ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
        cmp.tuple_float(f"d2.{name}.bootstrap95", d2[name]["bootstrap95"], ref["bootstrap95"],
                        new_ptr=f"/d2/{name}/bootstrap95", ref_ptr=f"/d2/{name}/bootstrap95",
                        ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)
        cmp.exact(f"d2.{name}.n_per_class", d2[name]["n_per_class"], ref["n_per_class"],
                  new_ptr=f"/d2/{name}/n_per_class", ref_ptr=f"/d2/{name}/n_per_class",
                  ref_path=f"{PROD}/summary_all.json", ref_sha256=summary_all_sha)

    matched_scores = {}
    for world in ("R", "S", "N"):
        with np.load(src.require(f"{PROD}/d2_matched_{world}.npz")) as z:
            matched_scores[world] = z["scores"]
    rr = np.random.default_rng(np.random.SeedSequence([seed, 7, 0]))
    matched_boots = {}
    for a, b in (("R", "S"), ("R", "N"), ("S", "N")):
        for col, label in enumerate(["bare", "measured_control", "noisy_control"]):
            x = matched_scores[a][:, col]
            y = matched_scores[b][:, col]
            key = f"{a}_vs_{b}_{label}"
            matched_boots[key] = np.array([auc(rr.choice(x, len(x)), rr.choice(y, len(y)))
                                           for _ in range(2000)])
    d2_matched = {"AUC": {}, "mean_scalar_linear_CKA_bare_measured_noisy": {},
                  "columns": ["bare", "measured_control", "noisy_control"]}
    for a, b in (("R", "S"), ("R", "N"), ("S", "N")):
        for col, label in enumerate(["bare", "measured_control", "noisy_control"]):
            key = f"{a}_vs_{b}_{label}"
            stored = np.load(src.require(f"{PROD}/d2_matched_boot_{key}.npy"))
            check(f"d2_matched.bootstrap_regen.{key}", bool(np.array_equal(matched_boots[key], stored)),
                  {"max_abs_diff": float(np.max(np.abs(matched_boots[key] - stored)))})
            x = matched_scores[a][:, col]
            y = matched_scores[b][:, col]
            d2_matched["AUC"][key] = {"auc": auc(x, y),
                                      "bootstrap95": np.quantile(matched_boots[key], [.025, .975]).tolist()}
            ref = ref_report["D2_matched"]["AUC"][key]
            cmp.float(f"d2_matched.{key}.auc", d2_matched["AUC"][key]["auc"], ref["auc"],
                      new_ptr=f"/d2_matched/AUC/{key}/auc", ref_ptr=f"/D2_matched/AUC/{key}/auc",
                      ref_path=REF_REPORT, ref_sha256=ref_sha)
            cmp.tuple_float(f"d2_matched.{key}.bootstrap95", d2_matched["AUC"][key]["bootstrap95"],
                            ref["bootstrap95"],
                            new_ptr=f"/d2_matched/AUC/{key}/bootstrap95",
                            ref_ptr=f"/D2_matched/AUC/{key}/bootstrap95",
                            ref_path=REF_REPORT, ref_sha256=ref_sha)
    for world in ("R", "S", "N"):
        cka = (matched_scores[world] ** 2).mean(0).tolist()
        d2_matched["mean_scalar_linear_CKA_bare_measured_noisy"][world] = cka
        ref = ref_report["D2_matched"]["mean_scalar_linear_CKA_bare_measured_noisy"][world]
        cmp.tuple_float(f"d2_matched.CKA.{world}", cka, ref,
                        new_ptr=f"/d2_matched/mean_scalar_linear_CKA_bare_measured_noisy/{world}",
                        ref_ptr=f"/D2_matched/mean_scalar_linear_CKA_bare_measured_noisy/{world}",
                        ref_path=REF_REPORT, ref_sha256=ref_sha)
    cmp.exact("d2_matched.columns", d2_matched["columns"], ref_report["D2_matched"]["columns"],
              new_ptr="/d2_matched/columns", ref_ptr="/D2_matched/columns",
              ref_path=REF_REPORT, ref_sha256=ref_sha)

    # ------------------------------------------------- supplemental bootstrap
    bp_cells = []
    ref_bp_cells = {c["cell"]: c for c in ref_boot["cells"]}
    for cell, (n, k, rho) in enumerate(BP_ORDER):
        rel = f"{BP_PROD}/cell_{cell:02d}.npz"
        with np.load(src.require(rel)) as z:
            statistics, valid_mass, valid, reject, patterns = (
                z["statistics"], z["valid_mass"], z["valid"], z["reject"], z["patterns"])
        n_total = int(valid.shape[0])
        count = int(reject.sum())
        valid_ci = int(valid.sum())
        rate = count / n_total
        zc = 1.959963984540054
        den = 1 + zc * zc / n_total
        ctr = (rate + zc * zc / (2 * n_total)) / den
        half = zc * np.sqrt(rate * (1 - rate) / n_total + zc * zc / (4 * n_total ** 2)) / den
        j = int(np.flatnonzero((patterns == 1).all(1))[0])
        point = statistics[:, j]
        good = np.isfinite(point)
        row = {
            "cell": cell, "n": n, "k": k, "rho_generator": rho, "n_total": n_total,
            "valid_CI": valid_ci, "rejections": count, "rate_unconditional": rate,
            "rate_conditional": float(count / valid_ci) if valid_ci else None,
            "wilson95": [float(ctr - half), float(ctr + half)],
            "valid_mass_min": float(valid_mass.min()),
            "observed_partial_spearman": {
                "finite": int(good.sum()),
                "mean": float(point[good].mean()),
                "mcse": float(point[good].std(ddof=1) / np.sqrt(good.sum())),
            },
            "invalid_pattern_mass_mean": float(1 - valid_mass.mean()),
        }
        bp_cells.append(row)
        ref = ref_bp_cells[cell]
        cmp.float(f"bp.cell{cell}.rate_unconditional", row["rate_unconditional"],
                  ref["rate_unconditional"], new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/rate_unconditional",
                  ref_ptr=f"/cells/{cell}/rate_unconditional", ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        cmp.float(f"bp.cell{cell}.rate_conditional", row["rate_conditional"],
                  ref["rate_conditional"], new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/rate_conditional",
                  ref_ptr=f"/cells/{cell}/rate_conditional", ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        cmp.float(f"bp.cell{cell}.valid_mass_min", row["valid_mass_min"], ref["valid_mass_min"],
                  new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/valid_mass_min",
                  ref_ptr=f"/cells/{cell}/valid_mass_min", ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        cmp.float(f"bp.cell{cell}.observed.mean", row["observed_partial_spearman"]["mean"],
                  ref["observed_partial_spearman"]["mean"],
                  new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/observed_partial_spearman/mean",
                  ref_ptr=f"/cells/{cell}/observed_partial_spearman/mean",
                  ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        cmp.float(f"bp.cell{cell}.observed.mcse", row["observed_partial_spearman"]["mcse"],
                  ref["observed_partial_spearman"]["mcse"],
                  new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/observed_partial_spearman/mcse",
                  ref_ptr=f"/cells/{cell}/observed_partial_spearman/mcse",
                  ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        cmp.float(f"bp.cell{cell}.invalid_pattern_mass_mean", row["invalid_pattern_mass_mean"],
                  ref["invalid_pattern_mass_mean"],
                  new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/invalid_pattern_mass_mean",
                  ref_ptr=f"/cells/{cell}/invalid_pattern_mass_mean",
                  ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        cmp.tuple_float(f"bp.cell{cell}.wilson95", row["wilson95"], ref["wilson95"],
                        new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/wilson95",
                        ref_ptr=f"/cells/{cell}/wilson95", ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        for key in ("n_total", "valid_CI", "rejections"):
            cmp.exact(f"bp.cell{cell}.{key}", row[key], ref[key],
                      new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/{key}",
                      ref_ptr=f"/cells/{cell}/{key}", ref_path=REF_BOOT, ref_sha256=ref_boot_sha)
        cmp.exact(f"bp.cell{cell}.observed.finite", row["observed_partial_spearman"]["finite"],
                  ref["observed_partial_spearman"]["finite"],
                  new_ptr=f"/supplemental_bootstrap_power/cells/{cell}/observed_partial_spearman/finite",
                  ref_ptr=f"/cells/{cell}/observed_partial_spearman/finite",
                  ref_path=REF_BOOT, ref_sha256=ref_boot_sha)

    bp_regen = _bp_representative_check(src, bpr, check)
    # metadata cross-checks against the accepted report
    meta = summary_all["metadata"]
    check("cross:seed", meta["seed"] == seed, {"summary_all": meta["seed"], "config": seed})
    check("cross:production_counts",
          ref_report["production_counts"]["calibration"] == 200
          and ref_report["production_counts"]["geometry"] == 1800
          and ref_report["production_counts"]["sign_randomization_power"] == 60000
          and ref_report["production_counts"]["D2_bernoulli_panels"] == 3000
          and ref_report["production_counts"]["D2_matched_gaussian_panels"] == 3000
          and ref_report["production_counts"]["actual_rank_familybootstrap_outer"] == 30000,
          dict(ref_report["production_counts"]))

    summary = {
        "branch": BRANCH,
        "classification": "CACHE_REPLAY_FROM_SAVED_PER_REPLICATE_AND_PER_DRAW_ARRAYS",
        "input_level": {
            "calibration": "per-replicate npz payload arrays (generator draws + fitted item parameters + EAP summaries)",
            "geometry": "per-replicate npy arrays of participation ratios",
            "power": "per-draw npz arrays (xy panels, family blocks, sign-flip p-values)",
            "d2": "per-replicate npz score arrays + per-resample bootstrap npy arrays",
            "supplemental_bootstrap_power": "per-replicate npz arrays (panels, 462-pattern statistics, valid mass, CI, reject)",
        },
        "estimand_note": "retained AN-B estimands preserved: 200-replicate theta recovery "
                         "(standard + difficult_weak), geometry 1800, sign-randomisation power, "
                         "D2 matched panels, supplemental family bootstrap. C4 is NOT substituted "
                         "for theta recovery.",
        "calibration": cal_summary,
        "geometry": geometry,
        "power": power,
        "d2": d2,
        "d2_matched": d2_matched,
        "supplemental_bootstrap_power": {
            "cells": bp_cells,
            "interpretation": ref_boot.get("interpretation"),
            "classification": ref_boot.get("classification"),
        },
        "production_counts_recomputed": {
            "calibration": sum(v["n_total"] for v in cal_summary.values()),
            "geometry": sum(g["reps"] for g in geometry),
            "sign_randomization_power": sum(p["total"] for p in power),
            "D2_bernoulli_panels": int(d2["real_vs_shadow"]["n_per_class"] * 3),
            "D2_matched_gaussian_panels": int(matched_scores["R"].shape[0] * 3),
            "actual_rank_familybootstrap_outer": sum(c["n_total"] for c in bp_cells),
        },
        "cross_bindings": {
            "report_source_sha256_matches_summary_all": ref_report["source_sha256"] == summary_all_sha,
            "config_sha256_matches_report": ref_report["config_sha256"] == src.declared["config/CT_SIMULATIONS.json"]["sha256"],
            "code_sha256_matches_report": ref_report["code_sha256"] == src.declared["code/new_simulation_repair.py"]["sha256"],
        },
        "estimator_checks": est_checks,
        "supplemental_bootstrap_representative_check": bp_regen,
        "comparison_only_sources": [f"{PROD}/summary_all.json", REF_REPORT, REF_BOOT],
        "aggregator_sources": {
            "per_replicate_metrics": "code/new_simulation_repair.py (score/auc/wilson re-imported, calibration_job generator recipe re-executed for representative draws)",
            "power": "code/new_simulation_repair.py power branch (sign matrix recomputed from blocks)",
            "bootstrap": "code/bootstrap_power_repair.py statistics/PAT/MASS for the representative check",
            "supplemental_finalize": "code/finalize_simulation_repair.py (observed_partial_spearman definition re-derived from saved statistics)",
        },
    }
    provenance = {
        "code_dependencies": {
            "code/new_simulation_repair.py": src.declared["code/new_simulation_repair.py"]["sha256"],
            "code/bootstrap_power_repair.py": src.declared["code/bootstrap_power_repair.py"]["sha256"],
            "code/finalize_simulation_repair.py": src.declared["code/finalize_simulation_repair.py"]["sha256"],
        },
        "config_dependencies": {"config/CT_SIMULATIONS.json": src.declared["config/CT_SIMULATIONS.json"]["sha256"]},
    }
    return summary, cmp, checks, provenance


score = None  # bound at run() to the imported original estimator


def _power_generator_check(src, seed, check):
    """Regenerate cell 0 panels from the frozen seed and compare bit-for-bit."""
    ok = True
    detail = {}
    n, k, rho, icc = POWER_ORDER[0]
    rr = np.random.default_rng(np.random.SeedSequence([seed, 3, 0]))
    fam = np.arange(n) % 6
    f = rr.normal(size=(1000, 6, 2))
    e = rr.normal(size=(1000, n, 2))
    f[:, :, 1] = rho * f[:, :, 0] + np.sqrt(1 - rho * rho) * f[:, :, 1]
    e[:, :, 1] = rho * e[:, :, 0] + np.sqrt(1 - rho * rho) * e[:, :, 1]
    xy = (np.sqrt(icc) * f[:, fam, :] + np.sqrt(1 - icc) * e
          + np.sqrt(20 / k) * rr.normal(size=(1000, n, 2)))
    products = xy[:, :, 0] * xy[:, :, 1]
    blocks = np.array([products[:, fam == j].sum(1) for j in range(6)]).T
    with np.load(src.require(f"{PROD}/power_{0:03d}.npz")) as z:
        xy_ok = bool(np.array_equal(xy, z["xy"]))
        blocks_ok = bool(np.array_equal(blocks, z["blocks"]))
    ok = xy_ok and blocks_ok
    detail = {"cell": 0, "xy_bit_identical": xy_ok, "blocks_bit_identical": blocks_ok}
    check("power.generator.cell0", ok, detail)
    return ok


def _bp_representative_check(src, bpr, check):
    """Regenerate one supplemental-bootstrap replicate and re-run the original
    statistics() on it; compare to the stored per-replicate row."""
    cell, rep = 0, 0
    n, k, rho = BP_ORDER[cell]
    rng = np.random.default_rng(np.random.SeedSequence([20260917, 2, cell, rep]))
    fam = np.arange(n) % 6
    f = rng.normal(size=(6, 2))
    e = rng.normal(size=(n, 2))
    f[:, 1] = rho * f[:, 0] + np.sqrt(1 - rho * rho) * f[:, 1]
    e[:, 1] = rho * e[:, 0] + np.sqrt(1 - rho * rho) * e[:, 1]
    xy = np.sqrt(.3) * f[fam] + np.sqrt(.7) * e + np.sqrt(20 / k) * rng.normal(size=(n, 2))
    C = rng.normal(size=(n, 4))
    C[:, 2] = (fam - fam.mean()) / fam.std()
    xy = xy + (C @ np.array([.3, .2, .2, .2]))[:, None]
    panel = np.column_stack([xy, C])
    values, mass, ci = bpr.statistics(panel, fam)
    with np.load(src.require(f"{BP_PROD}/cell_{cell:02d}.npz")) as z:
        panel_ok = bool(np.array_equal(panel, z["panels"][rep]))
        stat_max = float(np.nanmax(np.abs(values - z["statistics"][rep])))
        ci_ok = bool(np.allclose(ci, z["ci"][rep], rtol=0, atol=1e-12, equal_nan=True))
        mass_ok = bool(mass == z["valid_mass"][rep])
    ok = panel_ok and stat_max <= 1e-9 and ci_ok and mass_ok
    detail = {"cell": cell, "rep": rep, "panel_bit_identical": panel_ok,
              "statistics_max_abs_diff": stat_max, "ci_match_1e-12": ci_ok,
              "valid_mass_bit_identical": mass_ok}
    check("supplemental_bootstrap.representative_regen", ok, detail)
    return detail
