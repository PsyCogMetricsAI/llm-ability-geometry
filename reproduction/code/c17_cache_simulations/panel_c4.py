"""C4 panel-calibration branch: aggregate saved per-replicate records.

No redraw of the 5000 x 10000 inner family bootstrap: the replay aggregates the
hash-declared per-replicate records with the accepted aggregation code and
re-runs the original generator/estimator for a bounded representative set.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np

from common import Comparator, SourceError, load_json

BRANCH = "panel"
RUN_REL = "runs/c17_panel_calibration_v1/production_v1"
REF_REPORT = "reports/C17_PANEL_CALIBRATION_v1.json"
CELL_FIELDS = [
    ("R", "exact"), ("B", "exact"), ("rho_P", "float"), ("lambda", "float"),
    ("rho_S_target", "float"), ("mean_partial_rho", "float"),
    ("median_partial_rho", "float"), ("bias_vs_rho_S", "float"),
    ("sd_partial_rho", "float"), ("mc_se_mean_partial_rho", "float"),
    ("rejection_controlled_ci", "float"), ("mc_se_rejection", "float"),
    ("coverage_rho_S", "float"), ("coverage_conditional_valid", "float"),
    ("coverage_conditional_valid_R", "exact"),
    ("rejection_permutation_p", "float"),
    ("invalid_replicate_share", "float"), ("n_invalid_replicates", "exact"),
    ("n_partial_invalid_replicates", "exact"),
    ("mean_invalid_draws_per_replicate", "float"),
    ("mean_invalid_share_inner", "float"), ("max_invalid_share_inner", "float"),
    ("mean_n_valid", "float"), ("min_n_valid", "exact"),
    ("mean_ci_width", "float"), ("mean_permutation_total", "float"),
    ("min_permutation_total", "exact"),
    ("n_rank_borderline_redecided", "exact"), ("n_perm_borderline_redecided", "exact"),
    ("min_perm_margin", "float"),
]
ROLLUP_FIELDS = [
    ("rho_P", "float"), ("lambda", "float"), ("rho_S_target", "float"),
    ("n_replicate_runs", "exact"), ("mean_partial_rho", "float"),
    ("bias_vs_rho_S", "float"), ("sd_partial_rho", "float"),
    ("mc_se_mean", "float"), ("rejection_controlled_ci", "float"),
    ("coverage_rho_S_all500", "float"), ("rejection_permutation_p", "float"),
    ("n_invalid_replicates", "exact"),
    ("mean_invalid_draws_per_replicate", "float"),
]
NULL_SCENARIOS = ("N0", "N03", "N06")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _relocated_panel(src, C):
    """Read the real 50-position / 13-family panel from the selected tree.

    Structure-verbatim copy of c17_common.load_panel, but every read goes
    through the hash-declared manifest of the selected analysis root, so a
    relocated tree never falls back to the signed producer's absolute paths.
    """
    rho = load_json(src.require("../Imports/geometry/rho_inputs_50.json"))
    cov = load_json(src.require("../Imports/geometry/covariates_50.json"))
    tags = list(rho["models"].keys())
    if tags != list(cov["models"].keys()):
        raise SourceError("panel model order mismatch between relocated bound inputs")
    fams = [cov["models"][t]["code"]["family"] for t in tags]
    fam_sorted = sorted(set(fams))
    family_counts = {f: fams.count(f) for f in fam_sorted}
    fam_idx = C.family_codes(fams)
    domains = {}
    for dom in C.DOMAINS:
        Cmat = np.column_stack([
            [float(cov["models"][t][dom][name]) for t in tags] for name in C.COV_NAMES
        ]).astype(np.float64)
        x_real = np.array([rho["models"][t][dom]["geometry_composite_zpc1"]
                           for t in tags], dtype=np.float64)
        y_real = np.array([rho["models"][t][dom]["theta_hat_2pl_eap"]
                           for t in tags], dtype=np.float64)
        domains[dom] = {"C": Cmat, "x_real": x_real, "y_real": y_real}
    return {
        "tags": tags, "n_models": len(tags), "fams": fams,
        "fam_sorted": fam_sorted, "n_fam": len(fam_sorted),
        "family_counts": family_counts, "fam_idx": fam_idx, "domains": domains,
    }


def run(root, src, out_dir):
    code_dir = src.require("code/c17_panel_calibration/c17_runner.py").parent
    for rel in ("code/c17_panel_calibration/c17_common.py",
                "code/c17_panel_calibration/c17_kernel.py",
                "code/c17_panel_calibration/c17_reference.py",
                "code/c17_panel_calibration/c17_runner.py",
                "code/c17_panel_calibration/c17_equivalence.py",
                "code/c17_panel_calibration/c17_runner_preflight_tests.py",
                "code/c17_panel_calibration/c17_selfcheck.py",
                "code/c17_panel_calibration/c17_stage1_report.py",
                "code/c17_panel_calibration/c17_timing_pilot.py",
                "code/c17_panel_calibration/postrun/c17_production_selfcheck.py",
                "code/c17_panel_calibration/postrun/c17_stage2_report.py"):
        src.require(rel)
    cfg = src.require("config/CT_C17_PANEL_CALIBRATION_v2.json")
    ref_path = src.require(REF_REPORT)
    ref = load_json(ref_path)
    ref_sha = src.declared[REF_REPORT]["sha256"]

    sys.path.insert(0, str(code_dir))
    import c17_common as C
    import c17_runner as RUN
    import c17_reference as REF
    stage2 = _mod("c17_stage2_report_replay", code_dir / "postrun" / "c17_stage2_report.py")

    # Explicit per-process path binding: the signed producer modules keep their
    # original absolute defaults, so every path the replay can touch is rebound
    # to the selected tree before use.  Producer source files stay untouched.
    relocated_s5 = src.require(
        "../Imports/geometry/offline_analyze_docs_20260607/s5_convergence_analyze.py")
    path_bindings = {
        "c17_common.R": str(root), "c17_common.RUNS_DIR": str(root / "runs"),
        "c17_common.REPORTS_DIR": str(root / "reports"),
        "c17_common.GEO": str(root.parent / "Imports" / "geometry"),
        "c17_common.S5_PATH": str(relocated_s5),
        "c17_common.CONFIG_PATH": str(root / "config" / "CT_C17_PANEL_CALIBRATION_v2.json"),
        "c17_runner.PROD_DIR": str(root / RUN_REL),
        "c17_reference.S5_PATH": str(relocated_s5),
    }
    C.R = str(root)
    C.RUNS_DIR = str(root / "runs")
    C.REPORTS_DIR = str(root / "reports")
    C.GEO = str(root.parent / "Imports" / "geometry")
    C.S5_PATH = str(relocated_s5)
    C.CONFIG_PATH = str(root / "config" / "CT_C17_PANEL_CALIBRATION_v2.json")
    RUN.PROD_DIR = str(root / RUN_REL)
    REF.S5_PATH = str(relocated_s5)

    cmp = Comparator(BRANCH)
    checks = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    # ------------------------------------------------------------- integrity
    run_start = ref["hashes"]["computation_code_at_run_start"]
    postrun = ref["hashes"]["postrun_tools"]
    mismatch = []
    for rel, expected in run_start.items():
        rel_full = f"code/c17_panel_calibration/{rel}"
        found = src.declared[rel_full]["sha256"]
        if found != expected:
            mismatch.append({"path": rel_full, "expected": expected, "found": found})
    for rel, expected in postrun.items():
        rel_full = f"code/c17_panel_calibration/postrun/{rel}"
        found = src.declared[rel_full]["sha256"]
        if found != expected:
            mismatch.append({"path": rel_full, "expected": expected, "found": found})
    check("integrity.computation_code_matches_run_start", not mismatch, {"mismatch": mismatch})
    check("integrity.kernel_matches_reference_digest",
          src.declared["code/c17_panel_calibration/c17_kernel.py"]["sha256"]
          == ref["root_stage1_release"]["kernel_sha256"], None)
    check("integrity.contract_matches_report",
          src.declared["config/CT_C17_PANEL_CALIBRATION_v2.json"]["sha256"]
          == ref["design_authority"]["contract_sha256"], None)
    record_hashes = ref["hashes"]["record_files"]
    bad_records = []
    for name, expected in record_hashes.items():
        found = src.declared[f"{RUN_REL}/{name}"]["sha256"]
        if found != expected:
            bad_records.append({"file": name, "expected": expected, "found": found})
    check("integrity.record_file_hashes", not bad_records, {"mismatch": bad_records})

    # ------------------------------------------------------ recompute summary
    runs_dir = str(root / RUN_REL)
    completeness = RUN.verify_sweep_completeness(runs_dir)
    records = RUN.load_records(runs_dir)
    cells = RUN._cell_summary_rows(records)
    rollup = stage2.scenario_rollup(records)
    check("completeness.canonical_5000", bool(completeness["complete"]) and len(records) == 5000,
          {"complete": completeness["complete"], "n_records": len(records),
           "n_unique_ids": completeness["n_unique_ids"]})
    panel = _relocated_panel(src, C)
    design = {
        "R_per_cell": int(C.R_PER_CELL), "B_inner": int(C.B_INNER),
        "n_cells": len(C.SCENARIOS) * len(C.DOMAINS),
        "n_scenarios": len(C.SCENARIOS), "n_domains": len(C.DOMAINS),
        "n_positions": int(panel["n_models"]), "n_families": int(panel["n_fam"]),
        "replicates": int(C.R_PER_CELL) * len(C.SCENARIOS) * len(C.DOMAINS),
        "outer_seed_base": int(C.OUTER_SEED_BASE), "inner_seed": int(C.INNER_SEED),
    }
    check("design.matches_frozen_contract",
          design["R_per_cell"] == ref["design_authority"]["R_per_cell"]
          and design["B_inner"] == ref["design_authority"]["B_inner"]
          and design["n_cells"] == len(ref["design_authority"]["cells"])
          and design["n_scenarios"] == len(ref["design_authority"]["scenarios"])
          and design["n_domains"] == len(ref["design_authority"]["domains"])
          and design["n_positions"] == 50 and design["n_families"] == 13,
          design)

    null_cells = {
        c["cell_id"]: {
            "scenario": c["scenario"], "domain": c["domain"], "R": c["R"],
            "mean_partial_rho": c["mean_partial_rho"],
            "rejection_controlled_ci": c["rejection_controlled_ci"],
            "rejection_wilson95": c["rejection_wilson95"],
            "coverage_rho_S": c["coverage_rho_S"],
            "rejection_permutation_p": c["rejection_permutation_p"],
            "rejection_permutation_wilson95": c["rejection_permutation_wilson95"],
            "n_invalid_replicates": c["n_invalid_replicates"],
        }
        for c in cells if c["scenario"] in NULL_SCENARIOS
    }
    check("null_cells.six_null_domain_cells", len(null_cells) == 6, sorted(null_cells))

    # ------------------------------------------------------------- comparator
    ref_cells = {c["cell_id"]: c for c in ref["cell_outcomes"]}
    cmp.exact("cells.id_order", [c["cell_id"] for c in cells],
              [c["cell_id"] for c in ref["cell_outcomes"]],
              new_ptr="/cells/*/cell_id", ref_ptr="/cell_outcomes/*/cell_id",
              ref_path=REF_REPORT, ref_sha256=ref_sha)
    for idx, c in enumerate(cells):
        j = next(k for k, rc in enumerate(ref["cell_outcomes"]) if rc["cell_id"] == c["cell_id"])
        rc = ref["cell_outcomes"][j]
        for key, kind in CELL_FIELDS:
            ptr_new = f"/cells/{idx}/{key}"
            ptr_ref = f"/cell_outcomes/{j}/{key}"
            if kind == "float":
                cmp.float(f"cell.{c['cell_id']}.{key}", c[key], rc[key],
                          new_ptr=ptr_new, ref_ptr=ptr_ref, ref_path=REF_REPORT, ref_sha256=ref_sha)
            else:
                cmp.exact(f"cell.{c['cell_id']}.{key}", c[key], rc[key],
                          new_ptr=ptr_new, ref_ptr=ptr_ref, ref_path=REF_REPORT, ref_sha256=ref_sha)
        for key in ("rejection_wilson95", "coverage_wilson95",
                    "coverage_conditional_valid_wilson95",
                    "rejection_permutation_wilson95"):
            cmp.tuple_float(f"cell.{c['cell_id']}.{key}", c[key], rc[key],
                            new_ptr=f"/cells/{idx}/{key}", ref_ptr=f"/cell_outcomes/{j}/{key}",
                            ref_path=REF_REPORT, ref_sha256=ref_sha)
    for scen, r in rollup.items():
        for key, kind in ROLLUP_FIELDS:
            if kind == "float":
                cmp.float(f"rollup.{scen}.{key}", r[key],
                          ref["scenario_rollup_both_domains"][scen][key],
                          new_ptr=f"/scenario_rollup_both_domains/{scen}/{key}",
                          ref_ptr=f"/scenario_rollup_both_domains/{scen}/{key}",
                          ref_path=REF_REPORT, ref_sha256=ref_sha)
            else:
                cmp.exact(f"rollup.{scen}.{key}", r[key],
                          ref["scenario_rollup_both_domains"][scen][key],
                          new_ptr=f"/scenario_rollup_both_domains/{scen}/{key}",
                          ref_ptr=f"/scenario_rollup_both_domains/{scen}/{key}",
                          ref_path=REF_REPORT, ref_sha256=ref_sha)
        for key in ("rejection_wilson95", "coverage_rho_S_all500_wilson95",
                    "rejection_permutation_wilson95"):
            cmp.tuple_float(f"rollup.{scen}.{key}", r[key],
                            ref["scenario_rollup_both_domains"][scen][key],
                            new_ptr=f"/scenario_rollup_both_domains/{scen}/{key}",
                            ref_ptr=f"/scenario_rollup_both_domains/{scen}/{key}",
                            ref_path=REF_REPORT, ref_sha256=ref_sha)

    # ----------------------------------------- representative original replay
    rep_checks = []
    rec_by_cell = {}
    for r in records:
        rec_by_cell.setdefault(r["cell_id"], {})[int(r["replicate_index"])] = r
    from scipy.stats import rankdata
    for c in cells:
        cid = c["cell_id"]
        rec = rec_by_cell[cid][0]
        dom = c["domain"]
        Cmat = panel["domains"][dom]["C"]
        x, y = C.draw_replicate(c["scenario_index"], rec["domain_index"], 0,
                                c["rho_P"], c["lambda"], panel["fam_idx"],
                                panel["n_fam"], panel["n_models"])
        seed_ok = list(rec["outer_seed"]) == [C.OUTER_SEED_BASE, c["scenario_index"],
                                              rec["domain_index"], 0]
        Z = np.column_stack([np.ones(panel["n_models"]), rankdata(Cmat, axis=0)])
        XY = np.column_stack([rankdata(x), rankdata(y)])
        E = XY - Z @ np.linalg.lstsq(Z, XY, rcond=None)[0]
        point = float(np.corrcoef(E.T)[0, 1])
        diff = abs(point - rec["partial_rho"])
        rep_checks.append({"check": f"panel.{cid}.rep0.point_estimate",
                           "pass": bool(seed_ok and diff <= 1e-12),
                           "detail": {"seed_match": seed_ok, "abs_diff": diff}})
    check("panel.representative_point_estimates",
          all(rc["pass"] for rc in rep_checks),
          {"n": len(rep_checks), "n_fail": sum(not rc["pass"] for rc in rep_checks)})

    s5 = REF.load_s5(str(relocated_s5))
    families = np.array(panel["fams"])
    reference_replays = []
    for cid, rep in (("code_N0", 137), ("math_A06", 137), ("code_A03", 499)):
        c = next(x for x in cells if x["cell_id"] == cid)
        rec = rec_by_cell[cid][rep]
        Cmat = panel["domains"][c["domain"]]["C"]
        x, y = C.draw_replicate(c["scenario_index"], rec["domain_index"], rep,
                                c["rho_P"], c["lambda"], panel["fam_idx"],
                                panel["n_fam"], panel["n_models"])
        ci = s5.signed_family_block_cluster_ci(x, y, Cmat, families, b_boot=10000, seed=20260605)
        pv = s5.permutation_p(x, y, Cmat, b_perm=10000, seed=20260605)
        lo_ok = abs(float(ci[0]) - rec["controlled_ci_lo"]) <= 1e-9
        hi_ok = abs(float(ci[1]) - rec["controlled_ci_hi"]) <= 1e-9
        excl_ok = bool(ci[2]) == bool(rec["controlled_ci_excludes_zero"])
        nvalid_ok = int(ci[3]) == int(rec["controlled_ci_n_valid"])
        p_ok = abs(float(pv) - rec["permutation_p"]) <= 1e-12
        pcount_ok = rec["permutation_p"] == (rec["permutation_hits"] + 1) / (rec["permutation_total"] + 1)
        ok = lo_ok and hi_ok and excl_ok and nvalid_ok and p_ok and pcount_ok
        reference_replays.append({"cell": cid, "rep": rep, "pass": bool(ok),
                                  "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                                  "excludes_zero": bool(ci[2]), "n_valid": int(ci[3]),
                                  "p": float(pv),
                                  "detail": {"lo_ok": lo_ok, "hi_ok": hi_ok,
                                             "excludes_ok": excl_ok, "nvalid_ok": nvalid_ok,
                                             "p_ok": p_ok, "pcount_ok": pcount_ok}})
    check("panel.representative_full_s5_reference",
          all(r["pass"] for r in reference_replays),
          {"n": len(reference_replays)})

    summary = {
        "branch": BRANCH,
        "node": "C4",
        "classification": "CACHE_REPLAY_AGGREGATE_FROM_SAVED_PER_REPLICATE_RECORDS",
        "input_level": "per-replicate JSONL records (saved controlled family-bootstrap CI endpoints, "
                       "legacy model-permutation p, invalidity/guard bookkeeping); the "
                       "5000 x 10000 inner bootstrap is NOT redrawn",
        "design": design,
        "cells": cells,
        "scenario_rollup_both_domains": rollup,
        "null_cells": null_cells,
        "inference_layers_separate": {
            "controlled_family_bootstrap_ci": "columns controlled_ci_lo/hi, controlled_ci_excludes_zero, "
                                              "rejection_controlled_ci, coverage_rho_S",
            "legacy_model_permutation": "columns permutation_p/hits/total, rejection_permutation_p; "
                                        "kept separate from the controlled CI readouts",
            "note": "no combined/mixed readout is produced",
        },
        "completeness": completeness,
        "n_records": len(records),
        "representative_checks": {"point_estimates": rep_checks,
                                  "full_s5_reference_replays": reference_replays},
        "path_bindings": path_bindings,
        "path_binding_note": "signed producer modules keep their original absolute "
                             "defaults; every path this replay can read is rebound to the "
                             "selected analysis tree before use, and the panel/S5 reads go "
                             "through the hash-declared source manifest",
        "aggregator_source": "code/c17_panel_calibration/c17_runner.py::_cell_summary_rows + "
                             "code/c17_panel_calibration/postrun/c17_stage2_report.py::scenario_rollup "
                             "(imported, no source writes)",
    }
    provenance = {
        "code_dependencies": {
            f"code/c17_panel_calibration/{k}": v for k, v in run_start.items()
        },
        "postrun_dependencies": {
            f"code/c17_panel_calibration/postrun/{k}": v for k, v in postrun.items()
        },
        "config_dependencies": {"config/CT_C17_PANEL_CALIBRATION_v2.json":
                                src.declared["config/CT_C17_PANEL_CALIBRATION_v2.json"]["sha256"]},
    }
    return summary, cmp, checks, provenance
