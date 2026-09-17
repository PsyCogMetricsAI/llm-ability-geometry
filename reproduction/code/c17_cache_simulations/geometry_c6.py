"""C6 geometry branch: recompute moments and the 14 stratified bootstrap CIs
from the canonical saved draw rows / per-draw metric arrays.

The frozen library functions (finite moments, Wilson, Spearman, the stratified
within-config bootstrap) are imported from code/c17_geometry_validation; the
saved summaries are comparison references only and are never aggregated into
the fresh output.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import Comparator, SourceError, load_json, relocate_original_path, sha256_file

BRANCH = "geometry"
RUN_ROOT = "runs/c17_geometry_validation_v1"
WORK = f"{RUN_ROOT}/attempts/C6.attempt-2/work"
REF = "reports/C17_GEOMETRY_RESULTS_v1.json"
STAGE_SIG = {"twonn_clean": "twonn", "noise": "noise", "boundary": "boundary",
             "stability": "stability"}
STAGE_COUNTS = {"twonn_clean": 3600, "noise": 2400, "boundary": 100, "stability": 1800}


def _boot_task(args):
    from c17_geometry_validation.common import stratified_bootstrap_spearman
    strata, seed_entropy, n_resamples = args
    return stratified_bootstrap_spearman(strata, seed_entropy, n_resamples)


def _reason_counts(rows, getter):
    counter = Counter()
    for r in rows:
        reason = getter(r)
        if reason:
            counter[reason] += 1
    return dict(sorted(counter.items()))


def _pairs_from_clouds(rows, metric):
    out = []
    for r in rows:
        a = r["half_a"]["metrics"][metric]["value"]
        b = r["half_b"]["metrics"][metric]["value"]
        if a is None or b is None:
            out.append((float("nan"), float("nan")))
        else:
            out.append((float(a), float(b)))
    return out


def _num(v):
    return float("nan") if v is None else float(v)


def run(root, src, out_dir):
    for rel in ("code/c17_geometry_validation/__init__.py",
                "code/c17_geometry_validation/common.py",
                "code/c17_geometry_validation/contract.py",
                "code/c17_geometry_validation/library.py",
                "code/c17_geometry_validation/mc.py",
                "code/c17_geometry_validation/aggregate.py",
                "code/c17_geometry_validation/report.py",
                "code/c17_geometry_validation/finalize.py",
                "code/c17_geometry_validation/determinism.py"):
        src.require(rel)
    ref_path = src.require(REF)
    ref = load_json(ref_path)
    ref_sha = src.declared[REF]["sha256"]
    contract_path = src.require("config/CT_C17_GEOMETRY_v2.json")

    sys.path.insert(0, str(src.require("code/c17_geometry_validation/__init__.py").parent.parent))
    from c17_geometry_validation import contract as CT

    # Explicit per-process path binding (relocated tree, signed source untouched):
    # every path constant the imported modules can read is rebound before the
    # library / mc / aggregate modules are imported (library.py copies
    # contract.LIBRARY_PATH at import time).
    relocated_lib = src.require(
        "../Imports/geometry/route_c_mc_dispatch_20260608/activation_indicators_lib.py")
    relocated_fixtures_npz = src.require("reports/C5_GEOMETRY_FIXTURES_v1.npz")
    relocated_fixtures_json = src.require("reports/C5_GEOMETRY_FIXTURES_v1.json")
    path_bindings = {
        "c17_geometry_validation.contract.CONTRACT_PATH": str(contract_path),
        "c17_geometry_validation.contract.LIBRARY_PATH": str(relocated_lib),
        "c17_geometry_validation.contract.FIXTURE_CONTAINER": str(relocated_fixtures_npz),
        "c17_geometry_validation.contract.FIXTURE_MANIFEST": str(relocated_fixtures_json),
    }
    CT.CONTRACT_PATH = contract_path
    CT.LIBRARY_PATH = str(relocated_lib)
    CT.FIXTURE_CONTAINER = relocated_fixtures_npz
    CT.FIXTURE_MANIFEST = relocated_fixtures_json

    from c17_geometry_validation import library as LIB
    from c17_geometry_validation import mc as MC
    from c17_geometry_validation import aggregate as AG
    from c17_geometry_validation import common as GC

    LIB.LIBRARY_PATH = str(relocated_lib)

    cmp = Comparator(BRANCH)
    checks = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    contract = load_json(contract_path)
    if sha256_file(contract_path) != CT.CONTRACT_SHA256 or \
            contract.get("contract_id") != "CT_C17_GEOMETRY_v2":
        raise SourceError(
            f"selected-tree geometry contract does not match the code-declared hash: {contract_path}")
    check("integrity.contract_sha256_declared_in_code",
          GC.sha256_file(contract_path) == CT.CONTRACT_SHA256,
          {"declared_in_code": CT.CONTRACT_SHA256})
    binding_rows, external_bindings = [], []
    for orig, expected in contract["bindings"].items():
        relocated = relocate_original_path(orig, root)
        if relocated is None:
            external_bindings.append({"path": orig, "sha256": expected,
                                      "reason": "outside the selected project anchors; "
                                                "not read by this replay"})
            continue
        rel = Path(os.path.relpath(relocated, root)).as_posix()
        if rel not in src.declared:
            external_bindings.append({"path": orig, "sha256": expected,
                                      "reason": "binding is inside the project but not part of "
                                                "the replay source manifest; not read by this "
                                                "replay (no hidden fallback)"})
            continue
        path = src.require(rel)
        actual = sha256_file(path)
        binding_rows.append({"source": orig, "resolved": str(path),
                             "expected_sha256": expected, "actual_sha256": actual,
                             "match": actual == expected,
                             "inside_selected_project": src.policy._inside(path)})
    bad_bindings = [b for b in binding_rows
                    if not b["match"] or not b["inside_selected_project"]]
    check("integrity.relocated_contract_bindings",
          not bad_bindings and len(binding_rows) >= 5,
          {"n_bindings": len(binding_rows), "n_bad": len(bad_bindings),
           "n_external_not_read": len(external_bindings),
           "bad": bad_bindings[:5]})
    for key in ("twonn_summary_sha256", "noise_summary_sha256",
                "stability_summary_sha256", "boundary_summary_sha256"):
        rel = f"{WORK}/{key.replace('_summary_sha256', '')}/summary.json"
        if key.startswith("twonn"):
            rel = f"{WORK}/twonn_clean/summary.json"
        found = src.require(rel)
        check(f"integrity.{key}", src.declared[rel]["sha256"] == ref["artifact_hashes"][key],
              {"declared_in_report": ref["artifact_hashes"][key]})

    sigs, rows_by_stage = {}, {}
    for stage, kind in STAGE_SIG.items():
        sig_path = src.require(f"{WORK}/{stage}/stage_signature.json")
        sigs[stage] = load_json(sig_path)
        data_rel = f"{WORK}/{stage}/clouds.jsonl" if stage == "stability" else \
            f"{WORK}/{stage}/draws.jsonl"
        rows = GC.read_jsonl(src.require(data_rel))
        rows_by_stage[kind] = rows
        validated = MC.validate_draws(rows, kind, sigs[stage])
        expected = STAGE_COUNTS[stage]
        check(f"completeness.{stage}", len(rows) == expected and validated["n_draws"] == expected,
              {"n": len(rows), "expected": expected})

    twonn_rows = rows_by_stage["twonn"]
    noise_rows = rows_by_stage["noise"]
    boundary_rows = rows_by_stage["boundary"]
    clouds = rows_by_stage["stability"]

    # ------------------------------------------------------------- B2 (twonn)
    ref_twonn = load_json(src.require(f"{WORK}/twonn_clean/summary.json"))
    twonn_cells = []
    classification = Counter()
    for cell in CT.b2_cells():
        cr = [r for r in twonn_rows if int(r["cell_id"]) == cell["cell_id"]]
        two_nn = GC.finite_summary([_num(r["twoNN_id"]) for r in cr], target=float(cell["D"]))
        two_nn["t7"] = AG._t7(two_nn, float(cell["D"]))
        classification[two_nn["t7"]["classification"]] += 1
        twonn_cells.append({
            "cell_id": cell["cell_id"], "dist": cell["dist"], "D": cell["D"], "n": cell["n"],
            "n_total": len(cr),
            "twoNN_id": two_nn,
            "twoNN_id_mle": GC.finite_summary([_num(r["twoNN_id_mle"]) for r in cr],
                                              target=float(cell["D"])),
            "spectral_secondary": {
                m: GC.finite_summary([_num(r["metrics_other"][m]["value"]) for r in cr])
                for m in LIB.SPECTRAL_ORDER},
            "status_counts": dict(sorted(Counter(r["status"] for r in cr).items())),
            "reason_counts": _reason_counts(
                cr, lambda r: None if r["twoNN_id"] is not None else r["reason"]),
        })
    ref_cells_tw = {c["cell_id"]: c for c in ref_twonn["cells"]}
    for c in twonn_cells:
        rc = ref_cells_tw[c["cell_id"]]
        base_n, base_r = f"/twonn_clean/cells/{c['cell_id']}", f"/cells/{c['cell_id']}"
        for key in ("n_total",):
            cmp.exact(f"twonn.{c['cell_id']}.{key}", c[key], rc[key],
                      new_ptr=f"{base_n}/{key}", ref_ptr=f"{base_r}/{key}",
                      ref_path=f"{WORK}/twonn_clean/summary.json",
                      ref_sha256=ref["artifact_hashes"]["twonn_summary_sha256"])
        for block, key in ((c["twoNN_id"], "twoNN_id"), (c["twoNN_id_mle"], "twoNN_id_mle")):
            for f in ("n_total", "n_finite", "mean", "sd", "mcse_mean", "ci95_low",
                      "ci95_high", "failure_rate", "q025", "q50", "q975",
                      "rmse_finite", "rmse_all_r", "target", "bias"):
                cmp.float(f"twonn.{c['cell_id']}.{key}.{f}", block[f], rc[key][f],
                          new_ptr=f"{base_n}/{key}/{f}", ref_ptr=f"{base_r}/{key}/{f}",
                          ref_path=f"{WORK}/twonn_clean/summary.json",
                          ref_sha256=ref["artifact_hashes"]["twonn_summary_sha256"])
        cmp.exact(f"twonn.{c['cell_id']}.t7", c["twoNN_id"]["t7"]["classification"],
                  rc["twoNN_id"]["t7"]["classification"],
                  new_ptr=f"{base_n}/twoNN_id/t7/classification",
                  ref_ptr=f"{base_r}/twoNN_id/t7/classification",
                  ref_path=f"{WORK}/twonn_clean/summary.json",
                  ref_sha256=ref["artifact_hashes"]["twonn_summary_sha256"])
        cmp.exact(f"twonn.{c['cell_id']}.status_counts", c["status_counts"], rc["status_counts"],
                  new_ptr=f"{base_n}/status_counts", ref_ptr=f"{base_r}/status_counts",
                  ref_path=f"{WORK}/twonn_clean/summary.json",
                  ref_sha256=ref["artifact_hashes"]["twonn_summary_sha256"])
        for m, block in c["spectral_secondary"].items():
            for f in ("n_total", "n_finite", "mean", "sd", "mcse_mean"):
                cmp.float(f"twonn.{c['cell_id']}.spectral.{m}.{f}", block[f],
                          rc["spectral_secondary"][m][f],
                          new_ptr=f"{base_n}/spectral_secondary/{m}/{f}",
                          ref_ptr=f"{base_r}/spectral_secondary/{m}/{f}",
                          ref_path=f"{WORK}/twonn_clean/summary.json",
                          ref_sha256=ref["artifact_hashes"]["twonn_summary_sha256"])
    twonn_summary = {
        "cells": twonn_cells,
        "classification_counts": dict(sorted(classification.items())),
        "failed_cells": [c["cell_id"] for c in twonn_cells if c["twoNN_id"]["n_finite"] == 0],
        "n_finite_total": int(sum(c["twoNN_id"]["n_finite"] for c in twonn_cells)),
        "n_finite_by_metric": {
            **{m: int(sum(c["spectral_secondary"][m]["n_finite"] for c in twonn_cells))
               for m in LIB.SPECTRAL_ORDER},
            "twoNN_id": int(sum(c["twoNN_id"]["n_finite"] for c in twonn_cells)),
            "twoNN_id_mle": int(sum(c["twoNN_id_mle"]["n_finite"] for c in twonn_cells)),
        },
        "n_total": int(sum(c["n_total"] for c in twonn_cells)),
        "labels_by_class": {
            k: sorted(f"{c['dist']} D={c['D']} n={c['n']}" for c in twonn_cells
                      if c["twoNN_id"]["t7"]["classification"] == k)
            for k in ("RECOVERS", "INTERMEDIATE", "DEGRADES")},
    }
    by_class_ref = ref["claimed_findings"][1]["basis"]["by_class"]
    for k, v in by_class_ref.items():
        cmp.exact(f"twonn.by_class.{k}", twonn_summary["labels_by_class"].get(k, []),
                  sorted(v), new_ptr=f"/twonn_clean/labels_by_class/{k}",
                  ref_ptr=f"/claimed_findings/1/basis/by_class/{k}",
                  ref_path=REF, ref_sha256=ref_sha)

    # ------------------------------------------------------------- B3 (noise)
    ref_noise = load_json(src.require(f"{WORK}/noise/summary.json"))
    targets_all = CT.b3_targets(contract)
    noise_cells = []
    for cell in CT.b3_cells():
        cr = [r for r in noise_rows if int(r["cell_id"]) == cell["cell_id"]]
        targets = targets_all[cell["key"]]
        rankme_target = targets["rankme_target_n64"] if cell["n"] == 64 else targets["rankme_target_n256"]
        targets_echo = {
            "eff_rank_pr": targets["eff_rank_pr"], "rankme": rankme_target,
            "rankme_asymptotic": targets["rankme_asymptotic"],
            "stable_rank": targets["stable_rank"],
            "vn_entropy": targets["vn_entropy_nats"],
            "spectral_alpha": targets["spectral_alpha_logrank_ols"],
            "isoscore": targets["isoscore"],
        }
        metrics = {m: GC.finite_summary([_num(r["metrics"][m]["value"]) for r in cr],
                                        target=float(targets_echo[m]))
                   for m in LIB.SPECTRAL_ORDER}
        two_nn = GC.finite_summary([_num(r["twoNN_id"]) for r in cr])
        noise_cells.append({
            "cell_id": cell["cell_id"], "r": cell["r"], "sigma": cell["sigma"],
            "n": cell["n"], "n_total": len(cr), "targets": targets_echo,
            "metrics": metrics, "twoNN_id_descriptive": two_nn,
            "status_counts": dict(sorted(Counter(r["status"] for r in cr).items())),
        })
    ref_cells_noise = {c["cell_id"]: c for c in ref_noise["cells"]}
    for c in noise_cells:
        rc = ref_cells_noise[c["cell_id"]]
        base_n, base_r = f"/noise/cells/{c['cell_id']}", f"/cells/{c['cell_id']}"
        cmp.exact(f"noise.{c['cell_id']}.n_total", c["n_total"], rc["n_total"],
                  new_ptr=f"{base_n}/n_total", ref_ptr=f"{base_r}/n_total",
                  ref_path=f"{WORK}/noise/summary.json",
                  ref_sha256=ref["artifact_hashes"]["noise_summary_sha256"])
        for m, block in c["metrics"].items():
            for f in ("n_total", "n_finite", "mean", "sd", "mcse_mean", "ci95_low",
                      "ci95_high", "failure_rate", "q025", "q50", "q975",
                      "rmse_finite", "rmse_all_r", "bias"):
                cmp.float(f"noise.{c['cell_id']}.{m}.{f}", block[f], rc["metrics"][m][f],
                          new_ptr=f"{base_n}/metrics/{m}/{f}", ref_ptr=f"{base_r}/metrics/{m}/{f}",
                          ref_path=f"{WORK}/noise/summary.json",
                          ref_sha256=ref["artifact_hashes"]["noise_summary_sha256"])
        cmp.float(f"noise.{c['cell_id']}.twoNN.mean", c["twoNN_id_descriptive"]["mean"],
                  rc["twoNN_id_descriptive"]["mean"],
                  new_ptr=f"{base_n}/twoNN_id_descriptive/mean",
                  ref_ptr=f"{base_r}/twoNN_id_descriptive/mean",
                  ref_path=f"{WORK}/noise/summary.json",
                  ref_sha256=ref["artifact_hashes"]["noise_summary_sha256"])
        cmp.exact(f"noise.{c['cell_id']}.status_counts", c["status_counts"], rc["status_counts"],
                  new_ptr=f"{base_n}/status_counts", ref_ptr=f"{base_r}/status_counts",
                  ref_path=f"{WORK}/noise/summary.json",
                  ref_sha256=ref["artifact_hashes"]["noise_summary_sha256"])
    noise_maxima = {}
    for m in LIB.SPECTRAL_ORDER:
        biases = [c["metrics"][m]["bias"] for c in noise_cells if c["metrics"][m]["bias"] is not None]
        rmses = [c["metrics"][m]["rmse_finite"] for c in noise_cells
                 if c["metrics"][m]["rmse_finite"] is not None]
        noise_maxima[m] = {
            "max_abs_bias": max(abs(b) for b in biases) if biases else None,
            "max_rmse_finite": max(rmses) if rmses else None,
            "n_cells": sum(1 for c in noise_cells if c["metrics"][m]["bias"] is not None),
        }

    # ---------------------------------------------------------- B4 (boundary)
    ref_boundary = load_json(src.require(f"{WORK}/boundary/summary.json"))
    boundary_cells = []
    for cell in CT.b4_cells():
        cr = [r for r in boundary_rows if int(r["cell_id"]) == cell["cell_id"]]
        per_metric = {}
        for m in LIB.METRIC_ORDER:
            values = [_num(r["metrics"][m]["value"]) for r in cr]
            block = GC.finite_summary(values)
            block["reason_counts"] = _reason_counts(
                cr, lambda r, mm=m: r["metrics"][mm]["reason"]
                if r["metrics"][mm]["value"] is None else None)
            per_metric[m] = block
        boundary_cells.append({
            "cell_id": cell["cell_id"], "scenario": cell["scenario"],
            "n": cell["n"], "d": cell["d"], "n_total": len(cr),
            "status_counts": dict(sorted(Counter(r["status"] for r in cr).items())),
            "metrics": per_metric,
        })
    ref_cells_b = {c["cell_id"]: c for c in ref_boundary["randomized_cells"]}
    for c in boundary_cells:
        rc = ref_cells_b[c["cell_id"]]
        base_n, base_r = f"/boundary/randomized_cells/{c['cell_id']}", f"/randomized_cells/{c['cell_id']}"
        cmp.exact(f"boundary.{c['cell_id']}.status_counts", c["status_counts"], rc["status_counts"],
                  new_ptr=f"{base_n}/status_counts", ref_ptr=f"{base_r}/status_counts",
                  ref_path=f"{WORK}/boundary/summary.json",
                  ref_sha256=ref["artifact_hashes"]["boundary_summary_sha256"])
        for m, block in c["metrics"].items():
            for f in ("n_total", "n_finite", "mean", "failure_rate", "sd"):
                cmp.float(f"boundary.{c['cell_id']}.{m}.{f}", block[f], rc["metrics"][m][f],
                          new_ptr=f"{base_n}/metrics/{m}/{f}", ref_ptr=f"{base_r}/metrics/{m}/{f}",
                          ref_path=f"{WORK}/boundary/summary.json",
                          ref_sha256=ref["artifact_hashes"]["boundary_summary_sha256"])

    # -------------------------------------------------------- B5 (stability)
    ref_stab = load_json(src.require(f"{WORK}/stability/summary.json"))
    spectral_cfg, twonn_cfg = CT.b5_configs()
    ensembles, boot_tasks = [], []
    for name, cfgs in (("spectral_mixed", spectral_cfg), ("twonn_mixed", twonn_cfg)):
        cfg_rows = {c["cell_id"]: [r for r in clouds if int(r["config_index"]) == c["cell_id"]]
                    for c in cfgs}
        per_metric = {}
        for m in LIB.METRIC_ORDER:
            strata = [_pairs_from_clouds(cfg_rows[c["cell_id"]], m) for c in cfgs]
            flat = [p for s in strata for p in s]
            n_total = len(flat)
            n_finite = sum(1 for a, b in flat if math.isfinite(a) and math.isfinite(b))
            boot_tasks.append((name, m, strata))
            gap, n_zero = [], 0
            for a, b in flat:
                if math.isfinite(a) and math.isfinite(b):
                    if abs(a) <= 0.0:
                        n_zero += 1
                        continue
                    gap.append(abs(a - b) / abs(a))
            per_metric[m] = {
                "metric": m, "n_pairs_total": n_total, "n_pairs_finite": n_finite,
                "n_pairs_invalid": n_total - n_finite,
                "failure_rate": (n_total - n_finite) / n_total if n_total else None,
                "failure_rate_wilson95": GC.wilson_interval(n_total - n_finite, n_total)
                if n_total else None,
                "gap_abs_rel": {**GC.median_iqr(gap), "n_zero_reference_skipped": n_zero},
                "per_config_spearman": [
                    {"config": c["label"],
                     "n_finite_pairs": sum(1 for p in strata[i]
                                           if math.isfinite(p[0]) and math.isfinite(p[1])),
                     "spearman": GC.spearman([p[0] for p in strata[i]],
                                             [p[1] for p in strata[i]])}
                    for i, c in enumerate(cfgs)],
            }
        ensembles.append({"ensemble": name, "clouds_total": sum(len(v) for v in cfg_rows.values()),
                          "per_metric": per_metric})
    # 14 stratified bootstrap intervals: frozen seeds through the frozen helper.
    jobs = [(t[2], list(CT.B5_BOOTSTRAP_SEED), int(CT.B5_BOOTSTRAP_RESAMPLES))
            for t in boot_tasks]
    with ProcessPoolExecutor(max_workers=2) as pool:
        boot_results = list(pool.map(_boot_task, jobs))
    for (name, m, _), boot in zip(boot_tasks, boot_results):
        next(e for e in ensembles if e["ensemble"] == name)["per_metric"][m]["bootstrap"] = boot
    n_intervals = sum(1 for e in ensembles for m in e["per_metric"]
                      if "bootstrap" in e["per_metric"][m])
    check("stability.n_bootstrap_intervals", n_intervals == 14, {"n": n_intervals})

    ref_ens = {e["ensemble"]: e for e in ref_stab["ensembles"]}
    for e in ensembles:
        re_ = ref_ens[e["ensemble"]]
        cmp.exact(f"stability.{e['ensemble']}.clouds_total", e["clouds_total"], re_["clouds_total"],
                  new_ptr=f"/stability/ensembles/{e['ensemble']}/clouds_total",
                  ref_ptr=f"/ensembles/{e['ensemble']}/clouds_total",
                  ref_path=f"{WORK}/stability/summary.json",
                  ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])
        for m, block in e["per_metric"].items():
            rb = re_["per_metric"][m]
            base_n = f"/stability/ensembles/{e['ensemble']}/per_metric/{m}"
            base_r = f"/ensembles/{e['ensemble']}/per_metric/{m}"
            for f in ("n_pairs_total", "n_pairs_finite", "n_pairs_invalid", "failure_rate"):
                cmp.float(f"stability.{e['ensemble']}.{m}.{f}", block[f], rb[f],
                          new_ptr=f"{base_n}/{f}", ref_ptr=f"{base_r}/{f}",
                          ref_path=f"{WORK}/stability/summary.json",
                          ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])
            cmp.float(f"stability.{e['ensemble']}.{m}.pooled_spearman",
                      block["bootstrap"]["pooled_spearman"], rb["bootstrap"]["pooled_spearman"],
                      new_ptr=f"{base_n}/bootstrap/pooled_spearman",
                      ref_ptr=f"{base_r}/bootstrap/pooled_spearman",
                      ref_path=f"{WORK}/stability/summary.json",
                      ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])
            for f in ("ci95_low", "ci95_high", "bootstrap_mean"):
                cmp.float(f"stability.{e['ensemble']}.{m}.bootstrap.{f}",
                          block["bootstrap"][f], rb["bootstrap"][f],
                          new_ptr=f"{base_n}/bootstrap/{f}", ref_ptr=f"{base_r}/bootstrap/{f}",
                          ref_path=f"{WORK}/stability/summary.json",
                          ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])
            for f in ("n_resamples", "n_bootstrap_valid", "n_bootstrap_null",
                      "counts_preserved_per_stratum", "seed_entropy"):
                cmp.exact(f"stability.{e['ensemble']}.{m}.bootstrap.{f}",
                          block["bootstrap"][f], rb["bootstrap"][f],
                          new_ptr=f"{base_n}/bootstrap/{f}", ref_ptr=f"{base_r}/bootstrap/{f}",
                          ref_path=f"{WORK}/stability/summary.json",
                          ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])
            for i, pc in enumerate(block["per_config_spearman"]):
                cmp.exact(f"stability.{e['ensemble']}.{m}.per_config.{i}.config",
                          pc["config"], rb["per_config_spearman"][i]["config"],
                          new_ptr=f"{base_n}/per_config_spearman/{i}/config",
                          ref_ptr=f"{base_r}/per_config_spearman/{i}/config",
                          ref_path=f"{WORK}/stability/summary.json",
                          ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])
                cmp.float(f"stability.{e['ensemble']}.{m}.per_config.{i}.spearman",
                          pc["spearman"], rb["per_config_spearman"][i]["spearman"],
                          new_ptr=f"{base_n}/per_config_spearman/{i}/spearman",
                          ref_ptr=f"{base_r}/per_config_spearman/{i}/spearman",
                          ref_path=f"{WORK}/stability/summary.json",
                          ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])
            for f in ("n_finite", "median", "q025", "q975"):
                cmp.float(f"stability.{e['ensemble']}.{m}.gap.{f}", block["gap_abs_rel"][f],
                          rb["gap_abs_rel"][f], new_ptr=f"{base_n}/gap_abs_rel/{f}",
                          ref_ptr=f"{base_r}/gap_abs_rel/{f}",
                          ref_path=f"{WORK}/stability/summary.json",
                          ref_sha256=ref["artifact_hashes"]["stability_summary_sha256"])

    # ------------------------------------------- comparator vs accepted report
    for m in LIB.METRIC_ORDER:
        ev = ref["per_metric_evidence"][m]["c6_evidence"]
        new_ev = {}
        for e in ensembles:
            new_ev[e["ensemble"]] = e["per_metric"][m]["bootstrap"]
        for ens_name in ("spectral_mixed", "twonn_mixed"):
            ref_block = ev["stability"][ens_name]
            b = new_ev[ens_name]
            cmp.float(f"report.{m}.{ens_name}.pooled_spearman", b["pooled_spearman"],
                      ref_block["pooled_spearman"],
                      new_ptr=f"/stability/ensembles/{ens_name}/per_metric/{m}/bootstrap/pooled_spearman",
                      ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/stability/{ens_name}/pooled_spearman",
                      ref_path=REF, ref_sha256=ref_sha)
            cmp.tuple_float(f"report.{m}.{ens_name}.ci95",
                            [b["ci95_low"], b["ci95_high"]], ref_block["ci95"],
                            new_ptr=f"/stability/ensembles/{ens_name}/per_metric/{m}/bootstrap%5Bci95%5D",
                            ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/stability/{ens_name}/ci95",
                            ref_path=REF, ref_sha256=ref_sha)
            new_n_finite = next(e for e in ensembles if e["ensemble"] == ens_name)[
                "per_metric"][m]["n_pairs_finite"]
            cmp.exact(f"report.{m}.{ens_name}.n_pairs_finite", new_n_finite,
                      ref_block["n_pairs_finite"],
                      new_ptr=f"/stability/ensembles/{ens_name}/per_metric/{m}/n_pairs_finite",
                      ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/stability/{ens_name}/n_pairs_finite",
                      ref_path=REF, ref_sha256=ref_sha)
        nt = ev.get("noise_targets")
        nm = noise_maxima.get(m)
        if nt is not None and nm is not None and nm["max_abs_bias"] is not None:
            cmp.float(f"report.{m}.noise.max_abs_bias", nm["max_abs_bias"],
                      nt["max_abs_bias"],
                      new_ptr=f"/noise/maxima/{m}/max_abs_bias",
                      ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/noise_targets/max_abs_bias",
                      ref_path=REF, ref_sha256=ref_sha)
            cmp.float(f"report.{m}.noise.max_rmse_finite", nm["max_rmse_finite"],
                      nt["max_rmse_finite"],
                      new_ptr=f"/noise/maxima/{m}/max_rmse_finite",
                      ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/noise_targets/max_rmse_finite",
                      ref_path=REF, ref_sha256=ref_sha)
            cmp.exact(f"report.{m}.noise.n_cells", nm["n_cells"], nt["n_cells"],
                      new_ptr=f"/noise/maxima/{m}/n_cells",
                      ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/noise_targets/n_cells",
                      ref_path=REF, ref_sha256=ref_sha)
        else:
            cmp.skip(f"report.{m}.noise", f"/per_metric_evidence/{m}/c6_evidence/noise_targets",
                     REF, "no target-based noise maximum for this metric")
        cmp.exact(f"report.{m}.twonn_clean.n_finite",
                  twonn_summary["n_finite_by_metric"][m],
                  ev["twonn_clean"]["n_finite"],
                  new_ptr=f"/twonn_clean/n_finite_by_metric/{m}",
                  ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/twonn_clean/n_finite",
                  ref_path=REF, ref_sha256=ref_sha)
        cmp.exact(f"report.{m}.boundary.n_finite",
                  int(sum(c["metrics"][m]["n_finite"] for c in boundary_cells)),
                  ev["boundary_randomized"]["n_finite"],
                  new_ptr=f"/boundary/randomized_cells/*/metrics/{m}/n_finite",
                  ref_ptr=f"/per_metric_evidence/{m}/c6_evidence/boundary_randomized/n_finite",
                  ref_path=REF, ref_sha256=ref_sha)
    check("report.14_intervals_compared", True,
          {"n_intervals": 14, "note": "each interval compared via pooled_spearman + ci95 pointers"})

    # ------------------------------------------- representative regeneration
    rep_checks = []
    for cell in CT.b2_cells():
        rows = [r for r in twonn_rows if int(r["cell_id"]) == cell["cell_id"]]
        rep = 137
        row = next(r for r in rows if int(r["rep"]) == rep)
        x = MC.gen_b2(cell, rep)
        rec, _ = MC._b2_record(cell, rep, x)
        diffs = []
        if rec["data_sha256"] != row["data_sha256"]:
            diffs.append(("data_sha256", None, None))
        for mm in LIB.SPECTRAL_ORDER:
            diffs.append((mm, _num(rec["metrics_other"][mm]["value"]),
                          row["metrics_other"][mm]["value"]))
        diffs.append(("twoNN_id", _num(rec["twoNN_id"]), row["twoNN_id"]))
        bad = [d for d in diffs if d[1] is not None and d[2] is not None
               and abs(d[1] - float(d[2])) > 1e-9]
        rep_checks.append({"check": f"twonn.cell{cell['cell_id']}.rep137",
                           "pass": not bad and rec["data_sha256"] == row["data_sha256"],
                           "detail": {"n_metric_diffs_over_1e-9": len(bad)}})
    for cell in CT.b3_cells():
        rows = [r for r in noise_rows if int(r["cell_id"]) == cell["cell_id"]]
        rep = 137
        row = next(r for r in rows if int(r["rep"]) == rep)
        targets = targets_all[cell["key"]]
        x = MC.gen_b3(cell, rep)
        rec, _ = MC._b3_record(cell, rep, x, targets)
        bad = [m for m in LIB.SPECTRAL_ORDER
               if (_num(rec["metrics"][m]["value"]) is not None
                   and row["metrics"][m]["value"] is not None
                   and abs(_num(rec["metrics"][m]["value"]) - float(row["metrics"][m]["value"])) > 1e-9)]
        rep_checks.append({"check": f"noise.cell{cell['cell_id']}.rep137",
                           "pass": not bad and rec["data_sha256"] == row["data_sha256"],
                           "detail": {"metrics_over_1e-9": bad}})
    for cell in CT.b4_cells():
        rows = [r for r in boundary_rows if int(r["cell_id"]) == cell["cell_id"]]
        rep = 17
        row = next(r for r in rows if int(r["rep"]) == rep)
        x = MC.gen_b4(cell, rep)
        rec, _ = MC._b4_record(cell, rep, x)
        bad = [m for m in LIB.METRIC_ORDER
               if (_num(rec["metrics"][m]["value"]) is not None
                   and row["metrics"][m]["value"] is not None
                   and abs(_num(rec["metrics"][m]["value"]) - float(row["metrics"][m]["value"])) > 1e-9)]
        rep_checks.append({"check": f"boundary.cell{cell['cell_id']}.rep17",
                           "pass": not bad and rec["data_sha256"] == row["data_sha256"],
                           "detail": {"metrics_over_1e-9": bad}})
    cfg_all = spectral_cfg + twonn_cfg
    for cfg in cfg_all:
        ens = "spectral" if cfg["cell_id"] < 6 else "twonn"
        cloud = 137
        row = next(r for r in clouds if int(r["config_index"]) == cfg["cell_id"]
                   and int(r["cloud"]) == cloud)
        x, half_a, half_b, split_note = MC.gen_b5(ens, cfg, cloud)
        rec, _ = MC._b5_record(ens, cfg, cloud, x, half_a, half_b, split_note)
        bad_parts = []
        for part in ("full", "half_a", "half_b"):
            for m in LIB.METRIC_ORDER:
                a = _num(rec[part]["metrics"][m]["value"])
                b = row[part]["metrics"][m]["value"]
                if a is not None and b is not None and abs(a - float(b)) > 1e-9:
                    bad_parts.append(f"{part}:{m}")
        rep_checks.append({"check": f"stability.config{cfg['cell_id']}.cloud137",
                           "pass": not bad_parts and rec["full"]["data_sha256"] == row["full"]["data_sha256"],
                           "detail": {"metrics_over_1e-9": bad_parts}})
    check("representative_regeneration", all(c["pass"] for c in rep_checks),
          {"n": len(rep_checks), "n_fail": sum(not c["pass"] for c in rep_checks),
           "checks": rep_checks})

    summary = {
        "branch": BRANCH,
        "node": "C6",
        "classification": "CACHE_REPLAY_FROM_SAVED_DRAW_ROWS_AND_PER_DRAW_METRIC_ARRAYS",
        "input_level": "per-draw metric rows (draws.jsonl / clouds.jsonl) + per-draw metric arrays "
                       "(arrays/*.npz); the 7900 draws are NOT regenerated, only representative "
                       "fixed draws are re-executed through the frozen generators",
        "n_draws": {"twonn_clean": len(twonn_rows), "noise": len(noise_rows),
                    "boundary": len(boundary_rows), "stability_clouds": len(clouds),
                    "total": len(twonn_rows) + len(noise_rows) + len(boundary_rows) + len(clouds)},
        "twonn_clean": twonn_summary,
        "noise": {"cells": noise_cells, "maxima": noise_maxima,
                  "failed_cells": [c["cell_id"] for c in noise_cells
                                   if c["twoNN_id_descriptive"]["n_finite"] == 0]},
        "boundary": {"randomized_cells": boundary_cells,
                     "n_total": sum(c["n_total"] for c in boundary_cells),
                     "fixtures_note": "deterministic fixture statuses are a separate fixture-stage "
                                      "artifact (work/fixtures/C5_FIXTURE_RESULTS_v1.json); the "
                                      "randomized draw cache replayed here carries every B4 number "
                                      "reported in the accepted results report"},
        "stability": {"ensembles": ensembles,
                      "bootstrap": {"design": "stratified within each fixed config; quota-preserving",
                                    "n_resamples": int(CT.B5_BOOTSTRAP_RESAMPLES),
                                    "seed_entropy": list(CT.B5_BOOTSTRAP_SEED)},
                      "n_intervals": n_intervals},
        "statistics_contract": {
            "moments": "finite-conditional mean/sd/MCSE=sd/sqrt(n_finite)/95% CI; null with reason",
            "denominators": "all-R with per-metric n_finite; failure_rate with Wilson 95%",
            "quantiles": "require n_finite>=5",
            "t7": "n_finite>=2 and finite-conditional CI else UNDEFINED_INSUFFICIENT_FINITE",
        },
        "representative_checks": rep_checks,
        "path_bindings": path_bindings,
        "contract_bindings_verified": binding_rows,
        "contract_bindings_not_read": external_bindings,
        "path_binding_note": "the selected-tree geometry contract is read from the declared "
                             "manifest (code-declared sha256 enforced); every in-project "
                             "binding is relocated by anchor and hash-verified; bindings "
                             "outside the project or absent from the manifest are never read",
        "generator_source": "code/c17_geometry_validation/mc.py (gen_b2/gen_b3/gen_b4/gen_b5, "
                            "_b2/_b3/_b4/_b5 records) re-executed for fixed representative draws",
        "aggregator_source": "code/c17_geometry_validation/common.py (finite_summary, spearman, "
                             "stratified_bootstrap_spearman, median_iqr, wilson_interval) "
                             "+ aggregate.py._t7",
    }
    provenance = {
        "code_dependencies": {
            f"code/c17_geometry_validation/{k}": src.declared[f"code/c17_geometry_validation/{k}"]["sha256"]
            for k in ("__init__.py", "common.py", "contract.py", "library.py", "mc.py",
                      "aggregate.py", "report.py", "finalize.py", "determinism.py")
        },
        "config_dependencies": {"config/CT_C17_GEOMETRY_v2.json":
                                src.declared["config/CT_C17_GEOMETRY_v2.json"]["sha256"]},
        "contract_id": contract.get("contract_id"),
    }
    return summary, cmp, checks, provenance
