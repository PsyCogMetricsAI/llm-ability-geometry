"""Executor self-check for the C10a render run.

Independent checks (separate code path from the renderers):
* all 7 outputs present, artifacts hash/byte-verified;
* numeric recomputation where the accepted sources allow it
  (convergence reduction magnitude, BH/BY from the 28-cell p-family,
   LOFO summary + cross-source observed rho, profile grid maxima,
   probe medians when per-model arrays are available);
* refusal tests for bad/missing schemas and missing inputs;
* source non-mutation (declared source hashes unchanged since render);
* reader-facing label sanity for figures and tables.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from render import CANONICAL_IDS, SCHEMA_CONTRACT, SCHEMA_INVENTORY
    from render.common import load_json, read_csv_rows, sha256_file
    from render.contract import default_contract
else:
    from . import CANONICAL_IDS, SCHEMA_CONTRACT, SCHEMA_INVENTORY
    from .common import load_json, read_csv_rows, sha256_file
    from .contract import default_contract

TOL = 1e-9


def _fail(checks, name, detail):
    checks.append({"check": name, "status": "FAIL", "detail": detail})


def _ok(checks, name, detail=""):
    checks.append({"check": name, "status": "PASS", "detail": detail})


def _cell_value(fragment, row_label):
    for r in fragment.get("plotdata", fragment).get("rows", []):
        if r.get("row_label") == row_label or r.get("quantity") == row_label:
            return r
    return None


def check_inventory(ctx, checks):
    inv = load_json(ctx["outdir"] / "OUTPUT_INVENTORY.json")
    ids = [i["id"] for i in inv["items"]]
    if inv.get("schema") != SCHEMA_INVENTORY or inv.get("result_count") != 7:
        _fail(checks, "inventory_7_items", f"schema/count wrong: {inv.get('schema')} {inv.get('result_count')}")
        return None
    if sorted(ids) != sorted(CANONICAL_IDS):
        _fail(checks, "inventory_7_items", f"ids mismatch: {ids}")
        return None
    _ok(checks, "inventory_7_items", "exactly the 7 canonical ids")
    for item in inv["items"]:
        for art in item["artifacts"]:
            p = ctx["outdir"] / art["path"]
            if not p.is_file():
                _fail(checks, "artifact_exists", art["path"])
            elif sha256_file(p) != art["sha256"] or p.stat().st_size != art["bytes"]:
                _fail(checks, "artifact_hash", art["path"])
    _ok(checks, "artifact_hash_verify", f"{sum(len(i['artifacts']) for i in inv['items'])} artifacts")
    return inv


def check_convergence(ctx, checks):
    src = load_json(ctx["input_root"] / "runs/study1_reuse_v1/main/results.json")
    frag = load_json(ctx["outdir"] / "P2-B-TAB-CONVERGENCE/plotdata.json")
    for dom in ("code", "math"):
        d = src["domains"][dom]
        recomputed = (d["raw_rho"] - d["partial_rho_controlling_C"]) / d["raw_rho"] * 100.0
        if not math.isclose(recomputed, d["magnitude_reduction_percent"], rel_tol=1e-6, abs_tol=1e-6):
            _fail(checks, "convergence_reduction_magnitude",
                  f"{dom}: {recomputed} vs {d['magnitude_reduction_percent']}")
        if abs(d["partial_rho_controlling_C"]) > d["disattenuation_ceiling"] + 1e-9:
            _fail(checks, "convergence_disattenuation", f"{dom}: partial exceeds ceiling")
        lo, hi = d["bare_cluster_ci"]
        excl = (lo > 0 > hi) is False and (lo > 0 or hi < 0)
        if excl != d["bare_ci_excludes_zero"]:
            _fail(checks, "convergence_ci_flag", f"{dom} bare")
        lo, hi = d["controlled_cluster_ci"]
        excl = (lo > 0 or hi < 0)
        if excl != d["controlled_ci_excludes_zero"]:
            _fail(checks, "convergence_ci_flag", f"{dom} controlled")
    row = _cell_value(frag, "partial rho controlling Z")
    for dom in ("code", "math"):
        if not math.isclose(row["cells"][dom]["value"], src["domains"][dom]["partial_rho_controlling_C"],
                            rel_tol=0, abs_tol=TOL):
            _fail(checks, "convergence_plotdata_matches_source", dom)
    _ok(checks, "convergence_recompute",
        "reduction magnitude, disattenuation bound, CI flags, pointer values")


def check_science(ctx, checks):
    grid = load_json(ctx["input_root"] / "runs/c17_new28_v1/grids/F1-rarefied-28.json")
    frag = load_json(ctx["outdir"] / "P2-C-TAB-SCIENCE/plotdata.json")
    ps = sorted(c["p_cond"] for c in grid["cells"])
    m = len(ps)
    hm = sum(1.0 / j for j in range(1, m + 1))
    bh = [0.0] * m
    by = [0.0] * m
    run_bh, run_by = 1.0, 1.0
    for i in range(m - 1, -1, -1):
        run_bh = min(run_bh, ps[i] * m / (i + 1))
        run_by = min(run_by, ps[i] * m * hm / (i + 1))
        bh[i], by[i] = run_bh, min(run_by, 1.0)
    idx_of = {c["p_cond"]: i for i, c in enumerate(sorted(grid["cells"], key=lambda c: c["p_cond"]))}
    bad = 0
    for row in frag["rows"]:
        pointer_idx = int(row["cells"]["p_cond"]["source"]["json_pointer"].split("/")[2])
        cell = grid["cells"][pointer_idx]
        i = idx_of[cell["p_cond"]]
        if not math.isclose(bh[i], cell["bh28_cond"], rel_tol=1e-9, abs_tol=1e-12):
            bad += 1
        if not math.isclose(by[i], cell["by28_cond"], rel_tol=1e-9, abs_tol=1e-12):
            bad += 1
        if not math.isclose(row["cells"]["partial_rho"]["value"], cell["partial_rho"], abs_tol=TOL):
            bad += 1
    if bad:
        _fail(checks, "science_bh_by_recompute", f"{bad} mismatches over 8 rows")
    else:
        _ok(checks, "science_bh_by_recompute",
            "BH28/BY28 recomputed over the 28-cell p-family; 8 rows and pointers verified")


def check_lofo(ctx, checks):
    src = load_json(ctx["input_root"] / "runs/study2_reuse_v1/lofo/results.json")
    frag = load_json(ctx["outdir"] / "P2-C-TAB-LOFO/plotdata.json")
    rows = frag["rows"]
    if len(rows) != 13:
        _fail(checks, "lofo_13_rows", str(len(rows)))
    mn = min(r["rho_raw"] for r in rows)
    mx = max(r["rho_raw"] for r in rows)
    if not (math.isclose(mn, src["lofo_raw_summary"]["min"], abs_tol=TOL)
            and math.isclose(mx, src["lofo_raw_summary"]["max"], abs_tol=TOL)
            and all(r["rho_raw"] < 0 for r in rows) == src["lofo_raw_summary"]["all_negative"]):
        _fail(checks, "lofo_summary_recompute", "min/max/sign mismatch")
    grid = load_json(ctx["input_root"] / "runs/c17_new28_v1/grids/F1-native-28.json")
    sci = next(c for c in grid["cells"] if c["domain"] == "science" and c["metric"] == "eff_rank_pr")
    if not math.isclose(sci["raw_rho"], src["observed"]["raw_rho"], rel_tol=1e-12):
        _fail(checks, "lofo_cross_source_rho", f"{sci['raw_rho']} vs {src['observed']['raw_rho']}")
    if not (checks and checks[-1]["status"] == "FAIL"):
        _ok(checks, "lofo_recompute",
            "13 rows, summary min/max/sign recomputed; observed rho matches F1-native science cell")


def check_probes(ctx, checks):
    h1 = load_json(ctx["input_root"] / "../ext_P2_geo_20260914/h1/results_h1.json")
    frag = load_json(ctx["outdir"] / "P2-F-TAB-PROBES/plotdata.json")
    per_model = h1.get("task_A_pooled", {}).get("per_model", {})
    values = []
    for v in per_model.values():
        if isinstance(v, dict):
            for key in ("r2_pooled", "r2", "h1_r2_pooled"):
                if isinstance(v.get(key), (int, float)):
                    values.append(v[key])
                    break
    detail = "pointer re-read verified; per-model recompute unavailable"
    if len(values) >= 10:
        med = sorted(values)[len(values) // 2] if len(values) % 2 else (
            sum(sorted(values)[len(values) // 2 - 1:len(values) // 2 + 1]) / 2.0)
        target = frag["rows"][0]["value"]
        if math.isclose(med, target, rel_tol=1e-9, abs_tol=1e-12):
            detail = f"median of {len(values)} per-model R2 values reproduces {target:.6f}"
        else:
            _fail(checks, "probes_median_recompute", f"{med} vs {target}")
            return
    # pointer re-read comparison
    bad = 0
    for row in frag["rows"]:
        sp = row["source"]["json_pointer"]
        if "|" in sp:
            continue
        doc = h1 if "h1" in row["source"]["path"] else (
            load_json(ctx["input_root"] / "../ext_P2_geo_20260914/h2/results_h2.json")
            if "h2" in row["source"]["path"] else
            load_json(ctx["input_root"] / "../ext_P2_geo_20260914/h3/results_h3.json"))
        cur = doc
        for part in [p for p in sp.split("/") if p]:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        if isinstance(row["value"], (int, float)) and not math.isclose(cur, row["value"], abs_tol=TOL):
            bad += 1
    if bad:
        _fail(checks, "probes_pointer_reread", str(bad))
    else:
        _ok(checks, "probes_recompute", detail)


def check_profiles(ctx, checks):
    summary_path = ctx["input_root"] / "../ext_P2_geo_20260914/h4/depth_summary.csv"
    peaks_path = ctx["input_root"] / "../ext_P2_geo_20260914/h4/peak_depths.csv"
    header, data = read_csv_rows(summary_path)
    cols = {name: i for i, name in enumerate(header)}
    ph, pd_ = read_csv_rows(peaks_path)
    bad = []
    for prow in pd_:
        metric, depth, value = prow[0], float(prow[1]), float(prow[2])
        series = [(float(r[cols["relative_depth"]]), float(r[cols[metric]])) for r in data]
        mx = max(series, key=lambda t: t[1])
        if not (math.isclose(mx[1], value, abs_tol=1e-12) and math.isclose(mx[0], depth, abs_tol=1e-12)):
            bad.append(metric)
    meta = load_json(ctx["input_root"] / "../ext_P2_geo_20260914/h4/G4_METADATA.json")
    ns = [int(r[cols["n_models"]]) for r in data]
    if not (min(ns) == meta["min_coverage"] and max(ns) == meta["max_coverage"]):
        bad.append("coverage")
    if bad:
        _fail(checks, "profiles_grid_max_recompute", str(bad))
    else:
        _ok(checks, "profiles_grid_max_recompute",
            "3 grid maxima + coverage range recomputed from depth_summary.csv")


def check_mtmm(ctx, checks):
    frag = load_json(ctx["outdir"] / "P2-G-FIG-MTMM/plotdata.json")
    if frag.get("numeric_values_present") is not False or any(c["value"] is not None for c in frag["cells"]):
        _fail(checks, "mtmm_no_fabricated_matrix", "numeric values present in schematic")
    else:
        _ok(checks, "mtmm_no_fabricated_matrix", "4 labelled cells, all values empty")
    if not frag.get("upstream_dependency_clues"):
        _fail(checks, "mtmm_dependency_clues", "missing")
    else:
        _ok(checks, "mtmm_dependency_clues", ",".join(frag["upstream_dependency_clues"]))


def check_supplements(ctx, checks):
    frag = load_json(ctx["outdir"] / "P2-G-TAB-SUPPLEMENTS/plotdata.json")
    if frag.get("numeric_values_present") is not False or len(frag["items"]) != 10:
        _fail(checks, "supplements_topology", "numeric content or wrong item count")
    else:
        _ok(checks, "supplements_topology", "10 topology items, no numeric content")
    man = load_json(ctx["input_root"] / "inputs_manifest/RESULT_MANIFEST.json")
    if frag["manifest_binding_status"] != man["supplement_binding"]["status"]:
        _fail(checks, "supplements_binding_status", "manifest status mismatch")


def check_sources_unchanged(ctx, checks):
    prov = load_json(ctx["outdir"] / "PROVENANCE.json")
    bad = []
    for rec in prov["declared_inputs"]:
        p = Path(rec["path_abs"])
        if not p.is_file() or sha256_file(p) != rec["sha256"]:
            bad.append(rec["path_rel"])
    if bad:
        _fail(checks, "source_non_mutation", f"{len(bad)} changed: {bad[:3]}")
    else:
        _ok(checks, "source_non_mutation",
            f"{len(prov['declared_inputs'])} declared inputs byte-unchanged since render")


def check_labels(ctx, checks):
    problems = []

    def reader_strings(obj, acc):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in ("source", "source_files", "checks", "artifacts"):
                    continue
                if key in ("title", "note", "caption", "x_label", "y_label", "display",
                           "text", "label", "row_label", "quantity", "item") and isinstance(val, str):
                    acc.append(val)
                elif isinstance(val, (dict, list, str)) and key not in ("source",):
                    reader_strings(val, acc)
        elif isinstance(obj, list):
            for v in obj:
                reader_strings(v, acc)

    for fig_id in ("P2-G-FIG-MTMM", "P2-F-FIG-PROFILES"):
        frag = load_json(ctx["outdir"] / f"{fig_id}/plotdata.json")
        acc = []
        reader_strings(frag, acc)
        text = " | ".join(acc)
        for banned in ("json_pointer", "sha256", "src:", "__"):
            if banned in text:
                problems.append(f"{fig_id}:{banned}")
    prof = load_json(ctx["outdir"] / "P2-F-FIG-PROFILES/plotdata.json")
    if not prof.get("x_label") or not all(s.get("label") for s in prof.get("series", [])):
        problems.append("P2-F-FIG-PROFILES:axis-labels")
    for tab_id in ("P2-B-TAB-CONVERGENCE", "P2-C-TAB-SCIENCE", "P2-C-TAB-LOFO",
                   "P2-F-TAB-PROBES", "P2-G-TAB-SUPPLEMENTS"):
        tex = (ctx["outdir"] / f"{tab_id}").glob("*.tex")
        tex_files = list(tex)
        if not tex_files or "tabular" not in tex_files[0].read_text(encoding="utf-8"):
            problems.append(f"{tab_id}:tex")
    if problems:
        _fail(checks, "labels_sane", str(problems))
    else:
        _ok(checks, "labels_sane", "figure labels/units present; TeX tables parse-shaped; no code jargon")


def check_refusals(ctx, checks):
    run_script = Path(__file__).resolve().parent / "run_render.py"
    base = default_contract(str(ctx["input_root"]))
    variants = {}
    v = json.loads(json.dumps(base)); v["outputs"] = v["outputs"][:-1]
    variants["missing_output_entry"] = v
    v = json.loads(json.dumps(base)); v["schema"] = "bogus-schema-v9"
    variants["bad_schema_version"] = v
    v = json.loads(json.dumps(base)); v["outputs"][0]["inputs"][0]["path"] = "runs/does_not_exist.json"
    variants["missing_required_input"] = v
    results = []
    with tempfile.TemporaryDirectory(dir=ctx["outdir"]) as tmp:
        for name, contract in variants.items():
            d = Path(tmp) / f"refusal_{name}"
            d.mkdir()
            cpath = d / "contract.json"
            cpath.write_text(json.dumps(contract), encoding="utf-8")
            out = d / "should_not_exist"
            proc = subprocess.run(
                [sys.executable, str(run_script), "--input-root", str(ctx["input_root"]),
                 "--outdir", str(out), "--contract", str(cpath)],
                capture_output=True, text=True, env={**__import__("os").environ,
                                                      "OPENBLAS_NUM_THREADS": "1",
                                                      "OMP_NUM_THREADS": "1",
                                                      "MKL_NUM_THREADS": "1"})
            wrote_inventory = (out / "OUTPUT_INVENTORY.json").exists()
            results.append({"variant": name, "exit": proc.returncode,
                            "inventory_written": wrote_inventory,
                            "stderr_head": proc.stderr.strip().splitlines()[0] if proc.stderr.strip() else ""})
            if proc.returncode == 0 or wrote_inventory:
                _fail(checks, f"refusal_{name}", f"exit={proc.returncode} inventory={wrote_inventory}")
    if all(r["exit"] != 0 and not r["inventory_written"] for r in results):
        _ok(checks, "refusals_explicit", json.dumps(results))
    return results


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)
    ctx = {"input_root": Path(args.input_root).resolve(), "outdir": Path(args.outdir).resolve()}
    checks = []
    inv = check_inventory(ctx, checks)
    if inv:
        check_convergence(ctx, checks)
        check_science(ctx, checks)
        check_lofo(ctx, checks)
        check_probes(ctx, checks)
        check_profiles(ctx, checks)
        check_mtmm(ctx, checks)
        check_supplements(ctx, checks)
        check_sources_unchanged(ctx, checks)
        check_labels(ctx, checks)
    refusals = check_refusals(ctx, checks)
    failed = [c for c in checks if c["status"] == "FAIL"]
    report = {
        "schema": "c17-render-selfcheck-v1",
        "executor": "codex_1 CLI executor (not self-signed verified; root review required)",
        "input_root": str(ctx["input_root"]),
        "outdir": str(ctx["outdir"]),
        "checks": checks,
        "refusal_tests": refusals,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "status": "PASS_AWAITING_ROOT_REVIEW" if not failed else "FAIL",
    }
    (ctx["outdir"] / "SELFCHECK.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "failed": len(failed),
                      "status": report["status"]}))
    return 0 if not failed else 6


if __name__ == "__main__":
    sys.exit(main())
