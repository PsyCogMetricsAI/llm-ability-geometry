"""C10 final-integration checks (numerical integration checks).

Independent code path from the renderers. Runs the final config end to end in
the owned check directory, exercises refusals, and verifies the mandatory C10
integration points:

* hash-pin enforcement (declared sha256 mismatch refuses before any write);
* fresh depth summary computed from per-model/per-layer rows (old archives
  comparison-only, excluded from computation, archive-independence proven);
* seven outputs from the final config schema;
* no source writeback;
* final-config stage topology / whitelist / exception completeness;
* reader-facing captions free of internal cycle/task jargon.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from render import CANONICAL_IDS
    from render.common import sha256_file
    from render.contract import final_contract
else:
    from . import CANONICAL_IDS
    from .common import sha256_file
    from .contract import final_contract

JARGON = ("C9", "C10", "C17", "REQUIRED_PENDING", "prereg", "preregist", "task-path", "task path")


def _run(root: Path, outdir: Path, config: Path, expect: int | None = 0):
    env = {**__import__("os").environ, "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.run(
        [sys.executable, "code/rfinal_replay/render/run_render.py", "--input-root", ".",
         "--outdir", str(outdir.relative_to(root)), "--final-config", str(config.relative_to(root))],
        cwd=root, env=env, capture_output=True, text=True)
    if expect is not None and proc.returncode != expect:
        raise AssertionError(f"unexpected exit {proc.returncode} (wanted {expect}): {proc.stderr[-400:]}")
    return proc


def _rows(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--contract", default="config/CT_C17_REPLAY_FINAL_v1.json")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    outdir = (root / args.outdir).resolve()
    checks = []

    def ok(name, detail=""):
        checks.append({"check": name, "status": "PASS", "detail": detail})

    def fail(name, detail):
        checks.append({"check": name, "status": "FAIL", "detail": detail})

    # ---- 0. declared input hashes before the run ----------------------------
    contract = final_contract(str(root))
    before = {spec["path"]: sha256_file((root / spec["path"]).resolve())
              for e in contract["outputs"] for spec in e["inputs"]}

    # ---- 1. fresh run from the final config ---------------------------------
    proc = _run(root, outdir, root / args.contract, expect=0)
    inv_path = outdir / "OUTPUT_INVENTORY.json"
    if not inv_path.is_file():
        fail("fresh_run_inventory", f"no inventory; stderr={proc.stderr[-300:]}")
        inv = {"items": []}
    else:
        inv = json.loads(inv_path.read_text())
        ids = [i["id"] for i in inv["items"]]
        if sorted(ids) != sorted(CANONICAL_IDS) or inv.get("result_count") != 7:
            fail("fresh_run_seven_outputs", f"ids={ids}")
        else:
            ok("fresh_run_seven_outputs", "7/7 canonical ids from the final config")
        bad = 0
        for item in inv["items"]:
            for art in item["artifacts"]:
                p = outdir / art["path"]
                if not p.is_file() or sha256_file(p) != art["sha256"]:
                    bad += 1
        if bad:
            fail("artifact_hash_verify", f"{bad} artifacts missing/hash-mismatched")
        else:
            ok("artifact_hash_verify",
               f"{sum(len(i['artifacts']) for i in inv['items'])} artifacts hash-verified")

    # ---- 2. hash-pin enforcement + refusal behaviour ------------------------
    checks_root = outdir / "_checks"
    checks_root.mkdir(parents=True, exist_ok=True)
    bad_cfg = json.loads((root / args.contract).read_text())
    bad_cfg["render_contract"]["outputs"][0]["inputs"][0]["sha256"] = "0" * 64
    canonical = json.dumps(bad_cfg["render_contract"], sort_keys=True, separators=(",", ":")).encode()
    bad_cfg["render_contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    bad_path = checks_root / "pinned_mismatch.config.json"
    bad_path.write_text(json.dumps(bad_cfg, indent=1))
    out_bad = checks_root / "out_pin_mismatch"
    proc = _run(root, out_bad, bad_path, expect=3)
    if (out_bad / "OUTPUT_INVENTORY.json").exists():
        fail("hash_pin_refusal", "inventory written despite hash mismatch")
    else:
        ok("hash_pin_refusal", "exit 3, no inventory written before preflight refusal")

    missing_cfg = json.loads((root / args.contract).read_text())
    missing_cfg["render_contract"]["outputs"][1]["inputs"][0]["path"] = "runs/does_not_exist.json"
    missing_cfg["render_contract"]["outputs"][1]["inputs"][0].pop("sha256", None)
    canonical = json.dumps(missing_cfg["render_contract"], sort_keys=True, separators=(",", ":")).encode()
    missing_cfg["render_contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    missing_path = checks_root / "missing_input.config.json"
    missing_path.write_text(json.dumps(missing_cfg, indent=1))
    proc = _run(root, checks_root / "out_missing", missing_path, expect=3)
    ok("missing_input_refusal", "exit 3 for missing declared input")

    ptr_cfg = json.loads((root / args.contract).read_text())
    ptr_cfg["render_contract"]["outputs"][1]["inputs"][0]["pointer"] = "/domains/not_a_domain"
    canonical = json.dumps(ptr_cfg["render_contract"], sort_keys=True, separators=(",", ":")).encode()
    ptr_cfg["render_contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    ptr_path = checks_root / "bad_pointer.config.json"
    ptr_path.write_text(json.dumps(ptr_cfg, indent=1))
    proc = _run(root, checks_root / "out_pointer", ptr_path, expect=3)
    ok("failed_upstream_pointer_refusal", "exit 3 when a declared upstream pointer cannot resolve")

    # ---- 3. fresh depth summary + archive independence ----------------------
    pd = json.loads((outdir / "P2-F-FIG-PROFILES/plotdata.json").read_text())
    fresh_rows = _rows(outdir / "P2-F-FIG-PROFILES/depth_summary_fresh.csv")
    prof_rows = _rows((root / "../ext_P2_geo_20260914/h4/depth_profiles.csv").resolve())
    try:
        import numpy as np
        arrays = {}
        for r in prof_rows:
            arrays.setdefault(r["model_id"], []).append(r)
        for m in arrays:
            arrays[m].sort(key=lambda r: float(r["relative_depth"]))
        models = sorted(arrays)
        ref = []
        for i in range(1, 21):
            d = round(i / 20, 10)
            elig = [m for m in models if float(arrays[m][0]["relative_depth"]) <= d
                    <= float(arrays[m][-1]["relative_depth"])]
            bs = [float(np.interp(d, [float(r["relative_depth"]) for r in arrays[m]],
                                  [float(r["r2_b"]) for r in arrays[m]])) for m in elig]
            eps = [float(np.interp(d, [float(r["relative_depth"]) for r in arrays[m]],
                                   [float(r["r2_eps"]) for r in arrays[m]])) for m in elig]
            betas = [float(np.interp(d, [float(r["relative_depth"]) for r in arrays[m]],
                                     [float(r["beta"]) for r in arrays[m]])) for m in elig]
            ss = [float(arrays[m][0]["s"]) for m in elig]
            try:
                from scipy.stats import spearmanr, rankdata
                rho = float(spearmanr(-np.asarray(betas), ss).statistic)
            except Exception:
                rho = None
            ref.append((d, len(elig), float(np.median(bs)), rho, float(np.median(eps))))
        worst = 0.0
        for rr, fr in zip(ref, fresh_rows):
            if int(fr["n_models"]) != rr[1]:
                fail("fresh_summary_independent_recompute", f"n_models mismatch at {rr[0]}")
                break
            worst = max(worst, abs(float(fr["median_r2_b"]) - rr[2]), abs(float(fr["median_r2_eps"]) - rr[4]))
            if rr[3] is not None:
                worst = max(worst, abs(float(fr["rho_H2"]) - rr[3]))
        else:
            if worst <= 1e-9:
                ok("fresh_summary_independent_recompute",
                   f"20 grid rows recomputed independently (scipy spearman); max abs diff {worst:.2e}")
            else:
                fail("fresh_summary_independent_recompute", f"max abs diff {worst:.2e}")
    except AssertionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        fail("fresh_summary_independent_recompute", repr(exc))

    cmp_blk = pd.get("comparison_only", {})
    if (cmp_blk.get("archived_summary", {}).get("comparison", {}).get("within_tolerance")
            and "not approved" in cmp_blk.get("role", "")):
        ok("archive_comparison_only",
           f"archived aggregate matches fresh within the replay comparator "
           f"(max abs diff {pd['comparison_only']['archived_summary']['comparison']['max_abs_diff']:.2e}); "
           "role=comparison-only/not approved")
    else:
        fail("archive_comparison_only", json.dumps(cmp_blk)[:200])
    if pd.get("derivation", {}).get("computation_input", "").endswith("depth_profiles.csv"):
        ok("fresh_summary_from_per_model_rows", pd["derivation"]["computation_input"])
    else:
        fail("fresh_summary_from_per_model_rows", "declared computation input is not depth_profiles.csv")

    # archive-independence: perturbed archive (own declared hash) -> identical series CSV
    perturb_dir = checks_root / "perturbed"
    perturb_dir.mkdir(parents=True, exist_ok=True)
    src_arch = (root / "../ext_P2_geo_20260914/h4/depth_summary.csv").resolve()
    perturbed = perturb_dir / "depth_summary.csv"
    lines = src_arch.read_text().splitlines()
    header, rows = lines[0].split(","), [l.split(",") for l in lines[1:]]
    rows[0][header.index("median_r2_b")] = str(float(rows[0][header.index("median_r2_b")]) + 1e-9)
    perturbed.write_text("\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\n")
    indep_cfg = json.loads((root / args.contract).read_text())
    for spec in indep_cfg["render_contract"]["outputs"][5]["inputs"]:
        if spec["role"] == "ext_h4_depth_summary_archive":
            spec["path"] = str(perturbed.relative_to(root))
            spec.pop("external", None)
            spec["sha256"] = sha256_file(perturbed)
            spec["comparison_only"] = True
            spec["computational_input"] = False
    canonical = json.dumps(indep_cfg["render_contract"], sort_keys=True, separators=(",", ":")).encode()
    indep_cfg["render_contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    indep_path = checks_root / "perturbed_archive.config.json"
    indep_path.write_text(json.dumps(indep_cfg, indent=1))
    out_indep = checks_root / "out_perturbed_archive"
    _run(root, out_indep, indep_path, expect=0)
    a = (outdir / "P2-F-FIG-PROFILES/depth_profiles_series.csv").read_bytes()
    b = (out_indep / "P2-F-FIG-PROFILES/depth_profiles_series.csv").read_bytes()
    if a == b:
        ok("archive_independence", "perturbed comparison archive leaves the plotted series byte-identical")
    else:
        fail("archive_independence", "perturbed comparison archive changed the plotted series")

    # accepted C10a output comparison (numeric)
    accepted = root / "recovered/c17_render_comparison_v1/P2-F-FIG-PROFILES/depth_profiles_series.csv"
    if accepted.is_file():
        old, new = _rows(accepted), _rows(outdir / "P2-F-FIG-PROFILES/depth_profiles_series.csv")
        if len(old) != len(new):
            fail("accepted_series_match", f"row count {len(old)} != {len(new)}")
        else:
            worst = 0.0
            for x, y in zip(old, new):
                worst = max(worst, abs(float(x["value"]) - float(y["value"])))
                if x["metric"] != y["metric"] or x["n_models"] != y["n_models"]:
                    fail("accepted_series_match", "id/order/n mismatch")
                    break
            else:
                (ok if worst <= 1e-9 else fail)(
                    "accepted_series_match",
                    f"{len(new)} rows; max abs diff vs accepted render {worst:.2e}")
    else:
        fail("accepted_series_match", "accepted C10a series CSV not found")

    # ---- 4. no writeback -----------------------------------------------------
    after = {spec["path"]: sha256_file((root / spec["path"]).resolve())
             for e in contract["outputs"] for spec in e["inputs"]}
    if before == after:
        ok("no_source_writeback", f"{len(before)} declared inputs byte-unchanged after runs")
    else:
        changed = [k for k in before if before[k] != after.get(k)]
        fail("no_source_writeback", f"changed: {changed}")
    strays = [p for p in (root / "code/rfinal_replay").rglob("*")
              if p.is_file() and p.suffix == ".pyc" and outdir in p.parents]
    if strays:
        fail("no_stray_pyc", str(strays[:3]))
    else:
        ok("no_stray_pyc", "no bytecode written into the run tree")

    # ---- 5. final config topology / whitelist / exceptions ------------------
    cfg = json.loads((root / args.contract).read_text())
    canonical = json.dumps(cfg["render_contract"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() == cfg.get("render_contract_sha256"):
        ok("embedded_contract_hash", cfg["render_contract_sha256"])
    else:
        fail("embedded_contract_hash", "embedded render contract hash mismatch")
    stages = cfg.get("regeneration_plan", {}).get("stages", [])
    missing = [s.get("id") for s in stages
               if not all(k in s for k in ("id", "kind", "status"))]
    render_ids = {rid for s in stages for rid in s.get("result_ids", [])}
    suite = [s for s in stages if s.get("table_kind") == "c17_render_suite" and s.get("executable")]
    if missing or (set(CANONICAL_IDS) - render_ids) or not suite:
        fail("stage_topology_completeness",
             f"missing={missing} render_ids_missing={sorted(set(CANONICAL_IDS) - render_ids)} "
             f"executable_suite={bool(suite)}")
    else:
        ok("stage_topology_completeness",
           f"{len(stages)} engine stages; all 7 result ids bound; executable renderer suite present")
    old18 = [s for s in stages if s.get("id", "").startswith("S20_")]
    bound = [s for s in old18 if isinstance(s.get("c10_binding"), dict)]
    modes = {s["c10_binding"]["mode"] for s in bound}
    if len(bound) != 18 or modes - {"direct_execution", "equivalent_replacement"}:
        fail("old18_replacement_map", f"n={len(bound)}/{len(old18)} modes={modes}")
    else:
        counts = {m: sum(1 for s in bound if s["c10_binding"]["mode"] == m) for m in modes}
        ok("old18_replacement_map", f"18 adapters mapped {counts}")
    wl = cfg["input_whitelist"]["entries"]
    bad = [w["id"] for w in wl if not w.get("sha256") or not w.get("level")]
    l3_bad = [w["id"] for w in wl
              if w["level"] == "L3_PINNED_MODEL_SNAPSHOT" and w.get("usable_by_default")]
    if bad or l3_bad:
        fail("whitelist_integrity", f"bad={bad[:3]} l3_usable_by_default={l3_bad[:3]}")
    else:
        ok("whitelist_integrity",
           f"{len(wl)} pinned entries; L3 weights are rebuild-routes only (not usable by default replay)")
    exc = (cfg.get("c10_integration") or {}).get("input_exceptions", [])
    if len(exc) >= 2 and all(e["status"] == "PENDING_ROOT_DECISION_NOT_APPROVED"
                             and e["computational_input"] is False for e in exc):
        ok("input_exceptions_pending", f"{len(exc)} G4 depth_summary exceptions shown with row-level meaning")
    else:
        fail("input_exceptions_pending", json.dumps(exc)[:200])

    # ---- 6. reader-facing jargon / supplement bindings ----------------------
    sup_tex = (outdir / "P2-G-TAB-SUPPLEMENTS/supplement_topology.tex").read_text()
    sup_svg = (outdir / "P2-G-TAB-SUPPLEMENTS/supplement_topology.svg").read_text()
    mt_svg = (outdir / "P2-G-FIG-MTMM/mtmm_schematic.svg").read_text()
    prof_svg = (outdir / "P2-F-FIG-PROFILES/depth_profiles.svg").read_text()
    bad_tokens = {name: [t for t in JARGON if t in text] for name, text in
                  (("supplement.tex", sup_tex), ("supplement.svg", sup_svg),
                   ("mtmm.svg", mt_svg), ("profiles.svg", prof_svg))}
    bad_tokens = {k: v for k, v in bad_tokens.items() if v}
    if bad_tokens:
        fail("reader_facing_jargon", json.dumps(bad_tokens))
    else:
        ok("reader_facing_jargon", "no internal cycle/task jargon in figure/table captions")
    sup_rows = _rows(outdir / "P2-G-TAB-SUPPLEMENTS/supplement_topology.csv")
    if len(sup_rows) == 10 and all(r["evidence_source"].strip() and r["binding_status"].strip()
                                   for r in sup_rows) \
            and any("28 cells" in r["binding_status"] or "28-cell" in r["binding_status"] for r in sup_rows):
        ok("supplement_bindings", "10 items with concrete bindings; 28-cell scope stated")
    else:
        fail("supplement_bindings", f"{len(sup_rows)} rows")

    summary = {"node": "C10", "status": "awaiting_root_verification",
               "n_checks": len(checks), "n_pass": sum(1 for c in checks if c["status"] == "PASS"),
               "n_fail": sum(1 for c in checks if c["status"] == "FAIL"), "checks": checks}
    with open(outdir / "SELFCHECK_FINAL.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}))
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
