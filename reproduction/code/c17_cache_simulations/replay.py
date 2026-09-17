#!/usr/bin/env python3
"""C17 C10s cached-simulation replay (runnable, standalone).

    python code/c17_cache_simulations/replay.py --analysis-root R --out NEW_DIR \
        [--branch all|legacy|panel|geometry] [--expected-manifest PATH]
    python code/c17_cache_simulations/replay.py manifest --analysis-root R \
        --write code/c17_cache_simulations/SOURCE_MANIFEST_v1.json

Reads only hash-declared per-replicate / per-draw cache files, regenerates the
accepted numeric summary statistics, compares them against the accepted reports
and saves summaries only inside NEW_DIR (refuses a non-empty NEW_DIR).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # never leave bytecode side-effects in source trees

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import common  # noqa: E402  (pins BLAS env before numpy)
from common import (BRANCHES, OutputError, PathAudit, PathPolicy, SourceError, Sources,
                    load_declared_manifest, rel_to_root, sanitize, sha256_file, utc_now,
                    write_json)  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_SOURCE = 3
EXIT_OUTPUT = 4
EXIT_COMPARISON = 5


def _import_branch(name):
    import importlib
    return importlib.import_module({"legacy": "legacy_anb", "panel": "panel_c4",
                                    "geometry": "geometry_c6"}[name])


def discover_sources(root: Path) -> list[dict]:
    """The declared universe of replay sources (path, role, level)."""
    out: list[dict] = []

    def add(rel, role, level):
        out.append({"path": rel, "role": role, "level": level})

    # ---- replay code (self-describing)
    for fn in ("__init__.py", "common.py", "legacy_anb.py", "panel_c4.py",
               "geometry_c6.py", "replay.py", "negative_tests.py"):
        add(f"code/c17_cache_simulations/{fn}", "code", "code-dependency")
    # ---- legacy AN-B cache
    add("config/CT_SIMULATIONS.json", "config", "config")
    for s in (0, 1):
        for rep in range(100):
            add(f"runs/new_simulation_repair/production/calibration_{s}_{rep:03d}.npz",
                "input", "per-replicate-array")
    for r in (2, 4, 8):
        for n in (64, 128, 256):
            add(f"runs/new_simulation_repair/production/geometry_r{r}_n{n}.npy",
                "input", "per-replicate-array")
    for cell in range(60):
        add(f"runs/new_simulation_repair/production/power_{cell:03d}.npz",
            "input", "per-draw-array")
    for world in ("real", "shadow", "noisy"):
        add(f"runs/new_simulation_repair/production/d2_{world}.npz",
            "input", "per-replicate-array")
    for world in ("R", "S", "N"):
        add(f"runs/new_simulation_repair/production/d2_matched_{world}.npz",
            "input", "per-replicate-array")
    for key in ("real_vs_shadow", "noisy_vs_shadow",
                "commoncause_identical_observables", "response_only"):
        add(f"runs/new_simulation_repair/production/d2_auc_bootstrap_{key}.npy",
            "input", "bootstrap-resample-draw")
    for a, b in (("R", "S"), ("R", "N"), ("S", "N")):
        for label in ("bare", "measured_control", "noisy_control"):
            add(f"runs/new_simulation_repair/production/d2_matched_boot_{a}_vs_{b}_{label}.npy",
                "input", "bootstrap-resample-draw")
    add("runs/new_simulation_repair/production/summary_all.json",
        "comparator", "saved-summary-comparator-only")
    add("runs/bootstrap_power_repair/production/summary.json",
        "comparator", "saved-summary-comparator-only")
    for cell in range(30):
        add(f"runs/bootstrap_power_repair/production/cell_{cell:02d}.npz",
            "input", "per-draw-array")
        add(f"runs/bootstrap_power_repair/production/cell_{cell:02d}.json",
            "comparator", "saved-summary-comparator-only")
    for fn in ("new_simulation_repair.py", "finalize_simulation_repair.py",
               "bootstrap_power_repair.py"):
        add(f"code/{fn}", "code", "code-dependency")
    add("reports/NEW_SIMULATION_REPAIR.json", "comparator", "accepted-report-comparator-only")
    add("reports/NEW_BOOTSTRAP_POWER.json", "comparator", "accepted-report-comparator-only")

    # ---- C4 panel calibration cache
    add("config/CT_C17_PANEL_CALIBRATION_v2.json", "config", "config")
    for dom in ("code", "math"):
        for scen in ("N0", "N03", "N06", "A03", "A06"):
            add(f"runs/c17_panel_calibration_v1/production_v1/{dom}_{scen}.jsonl",
                "input", "per-replicate-record")
    add("runs/c17_panel_calibration_v1/production_v1/cell_summary.json",
        "comparator", "saved-summary-comparator-only")
    add("runs/c17_panel_calibration_v1/production_v1/per_replicate_table.csv",
        "comparator", "saved-summary-comparator-only")
    add("runs/c17_panel_calibration_v1/production_v1/aggregate_manifest.json",
        "cross-check", "run-manifest")
    for fn in ("c17_common.py", "c17_kernel.py", "c17_reference.py", "c17_runner.py",
               "c17_equivalence.py", "c17_runner_preflight_tests.py", "c17_selfcheck.py",
               "c17_stage1_report.py", "c17_timing_pilot.py",
               "postrun/c17_production_selfcheck.py", "postrun/c17_stage2_report.py"):
        add(f"code/c17_panel_calibration/{fn}", "code", "code-dependency")
    add("reports/C17_PANEL_CALIBRATION_v1.json", "comparator",
        "accepted-report-comparator-only")
    for rel in ("rho_inputs_50.json", "covariates_50.json", "theta_hat_panel51.json",
                "g5no_gates_code.json", "g5no_gates_math.json"):
        add(f"../Imports/geometry/{rel}", "design-input", "frozen-source")
    add("../Imports/geometry/offline_analyze_docs_20260607/s5_convergence_analyze.py",
        "design-input", "frozen-source")

    # ---- C6 geometry validation cache
    add("config/CT_C17_GEOMETRY_v2.json", "config", "config")
    for stage in ("twonn_clean", "noise", "boundary"):
        add(f"runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/{stage}/draws.jsonl",
            "input", "per-draw-metric-row")
        add(f"runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/{stage}/stage_signature.json",
            "input", "stage-signature")
        add(f"runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/{stage}/summary.json",
            "comparator", "saved-summary-comparator-only")
    add("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/stability/clouds.jsonl",
        "input", "per-draw-metric-row")
    add("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/stability/stage_signature.json",
        "input", "stage-signature")
    add("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/stability/summary.json",
        "comparator", "saved-summary-comparator-only")
    arrays_dir = root / ("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work")
    for stage in ("twonn_clean", "noise", "boundary", "stability"):
        adir = arrays_dir / stage / "arrays"
        if adir.is_dir():
            for f in sorted(adir.glob("*.npz")):
                add(f"runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/{stage}/arrays/{f.name}",
                    "input", "per-draw-array")
    for fn in ("__init__.py", "common.py", "contract.py", "library.py", "mc.py",
               "aggregate.py", "report.py", "finalize.py", "determinism.py"):
        add(f"code/c17_geometry_validation/{fn}", "code", "code-dependency")
    add("runs/c17_geometry_validation_v1/OUTPUT_HASHES.json", "cross-check", "run-manifest")
    add("runs/c17_geometry_validation_v1/RUN_MANIFEST_v1.json", "cross-check", "run-manifest")
    add("reports/C17_GEOMETRY_RESULTS_v1.json", "comparator",
        "accepted-report-comparator-only")
    add("reports/C5_GEOMETRY_FIXTURES_v1.npz", "design-input", "frozen-source")
    add("reports/C5_GEOMETRY_FIXTURES_v1.json", "design-input", "frozen-source")
    add("../Imports/geometry/route_c_mc_dispatch_20260608/activation_indicators_lib.py",
        "design-input", "frozen-source")
    return out


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / rel)


def cmd_manifest(args) -> int:
    root = Path(args.analysis_root).resolve()
    dest = Path(args.write)
    if dest.exists() and not args.force:
        print(f"refusing to overwrite existing manifest: {dest} (use --force)", file=sys.stderr)
        return EXIT_OUTPUT
    entries = []
    missing = []
    for e in discover_sources(root):
        p = _resolve(root, e["path"])
        if not p.exists():
            if not args.allow_missing:
                missing.append(e["path"])
                continue
        entries.append({"path": e["path"], "sha256": sha256_file(p) if p.exists() else None,
                        "role": e["role"], "level": e["level"]})
    if missing:
        print("refusing to write a manifest with missing sources:\n  "
              + "\n  ".join(missing), file=sys.stderr)
        return EXIT_SOURCE
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, {
        "schema": "c17-cache-simulation-source-manifest-v1",
        "generated_utc": utc_now(),
        "analysis_root": str(root),
        "path_convention": "paths are relative to the analysis root (R); '../' entries are "
                           "P-level frozen design inputs shared by the accepted C4/C6 runs",
        "n_sources": len(entries),
        "sources": entries,
    })
    print(f"wrote {dest} with {len(entries)} sources")
    return EXIT_OK


def cmd_verify(args) -> int:
    """Verify every declared source against its sha256; write nothing."""
    root = Path(args.analysis_root).resolve()
    manifest_path = Path(args.manifest) if args.manifest else HERE / "SOURCE_MANIFEST_v1.json"
    entries = load_declared_manifest(manifest_path)
    bad, n = [], 0
    for e in entries:
        p = _resolve(root, e["path"])
        if not p.exists():
            bad.append({"path": e["path"], "reason": "missing"})
            continue
        actual = sha256_file(p)
        n += 1
        if actual != e["sha256"]:
            bad.append({"path": e["path"], "reason": "sha256_mismatch",
                        "declared": e["sha256"], "found": actual})
    print(json.dumps({"manifest": str(manifest_path), "n_sources": len(entries),
                      "n_verified": n, "n_bad": len(bad), "bad": bad[:10]}, indent=1))
    return EXIT_OK if not bad else EXIT_SOURCE


def cmd_replay(args) -> int:
    started = time.perf_counter()
    root = Path(args.analysis_root).resolve()
    out = Path(args.out)
    if out.exists():
        if not out.is_dir():
            raise OutputError(f"output path exists and is not a directory: {out}")
        if any(out.iterdir()):
            raise OutputError(f"output directory is not empty (refusing to overwrite): {out}")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise OutputError(f"output directory is not empty: {out}")

    manifest_path = Path(args.expected_manifest) if args.expected_manifest else \
        HERE / "SOURCE_MANIFEST_v1.json"
    entries = load_declared_manifest(manifest_path)
    policy = PathPolicy(root)
    src = Sources(root, entries, policy)
    original_project = HERE.parents[2]
    audit = PathAudit(HERE, (root, root.parent), out, original_project=original_project)
    audit.install()

    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    results = {}
    deps = {"replay_code": {}, "config": {}, "branch_code": {}}
    for fn in ("common.py", "legacy_anb.py", "panel_c4.py", "geometry_c6.py", "replay.py"):
        deps["replay_code"][f"code/c17_cache_simulations/{fn}"] = sha256_file(HERE / fn)
    deps["manifest"] = {"path": rel_to_root(manifest_path, root),
                        "sha256": sha256_file(manifest_path)}

    status = EXIT_OK
    for branch in branches:
        mod = _import_branch(branch)
        t0 = time.perf_counter()
        summary, cmp, checks, provenance = mod.run(root, src, out)
        wall = time.perf_counter() - t0
        cmp_doc = cmp.dump()
        checks_doc = {"branch": branch,
                      "n_checks": len(checks),
                      "n_fail": sum(1 for c in checks if not c["pass"]),
                      "failures": [c for c in checks if not c["pass"]],
                      "checks": checks}
        files = {
            "summary": write_json(out / f"{branch}.summary.json", summary),
            "comparison": write_json(out / f"{branch}.comparison.json", cmp_doc),
            "checks": write_json(out / f"{branch}.checks.json", checks_doc),
        }
        ok = (cmp_doc["n_fail"] == 0) and (checks_doc["n_fail"] == 0)
        results[branch] = {
            "status": "pass" if ok else "fail",
            "wall_seconds": wall,
            "n_comparisons": cmp_doc["n_compared"], "n_comparison_fail": cmp_doc["n_fail"],
            "n_checks": checks_doc["n_checks"], "n_check_fail": checks_doc["n_fail"],
            "outputs": files,
        }
        deps["branch_code"].update(provenance.get("code_dependencies", {}))
        deps["config"].update(provenance.get("config_dependencies", {}))
        if not ok:
            status = EXIT_COMPARISON

    sources_doc = {
        "schema": "c17-cache-simulation-source-manifest-v1",
        "note": "sources consumed by this run (verified sha256 before any statistic)",
        "n_sources_consumed": len(src.used),
        "n_sources_verified": src.checked,
        "sources": src.used_entries(),
    }
    write_json(out / "source_manifest.json", sources_doc)
    write_json(out / "code_config_dependencies.json", deps)
    audit_doc = audit.report()
    audit_doc["schema"] = "c17-cache-simulation-path-audit-v1"
    audit_doc["analysis_root"] = str(root)
    audit_doc["selected_project"] = str(root.parent)
    audit_doc["original_project_reference"] = str(original_project)
    write_json(out / "path_audit.json", audit_doc)
    write_json(out / "commands.json", {
        "argv": [sys.executable] + sys.argv,
        "cwd": os.getcwd(),
        "analysis_root": str(root),
        "branch": args.branch,
        "exit_code_expected": status,
        "equivalent_integration_command":
            f"python code/c17_cache_simulations/replay.py --analysis-root {root} "
            f"--out <NEW_DIR> --branch {args.branch}",
    })
    receipt = {
        "schema": "c17-cache-simulation-replay-receipt-v1",
        "utc": utc_now(),
        "analysis_root": str(root),
        "branch": args.branch,
        "status": "pass" if status == EXIT_OK else "fail",
        "exit_code": status,
        "wall_seconds": time.perf_counter() - started,
        "sources_verified": src.checked,
        "branches": results,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
            "scipy": __import__("scipy").__version__,
            "blas_threads": {k: os.environ.get(k) for k in
                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "comparator": "abs(new-ref) <= 1e-9 + 1e-6*abs(ref); counts/ids/null-masks exact",
        "path_audit": {k: audit.report()["counts"][k] for k in
                       ("selected_tree_reads", "original_tree_reads",
                        "reads_outside_selected_tree", "writes_outside_out_dir")},
        "writeback": "none: outputs were written only under the requested --out directory",
    }
    receipt["output_files"] = sorted(p.name for p in out.iterdir()) + ["run_receipt.json"]
    write_json(out / "run_receipt.json", receipt)
    print(f"[replay] branch={args.branch} status={receipt['status']} "
          f"sources_verified={src.checked} outputs={len(receipt['output_files'])} dir={out}")
    return status


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")
    man = sub.add_parser("manifest", help="write the declared source manifest with hashes")
    man.add_argument("--analysis-root", default=str(ROOT_DEFAULT))
    man.add_argument("--write", required=True)
    man.add_argument("--force", action="store_true")
    man.add_argument("--allow-missing", action="store_true")
    ver = sub.add_parser("verify", help="verify declared sources against the manifest (read-only)")
    ver.add_argument("--analysis-root", default=str(ROOT_DEFAULT))
    ver.add_argument("--manifest")
    ap.add_argument("--analysis-root", default=str(ROOT_DEFAULT))
    ap.add_argument("--out")
    ap.add_argument("--branch", choices=("all",) + BRANCHES, default="all")
    ap.add_argument("--expected-manifest")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command == "manifest":
        return cmd_manifest(args)
    if args.command == "verify":
        return cmd_verify(args)
    if not args.out:
        print("error: --out NEW_DIR is required", file=sys.stderr)
        return EXIT_USAGE
    try:
        return cmd_replay(args)
    except OutputError as exc:
        print(f"[output-error] {exc}", file=sys.stderr)
        return EXIT_OUTPUT
    except SourceError as exc:
        print(f"[source-error] {exc}", file=sys.stderr)
        return EXIT_SOURCE


if __name__ == "__main__":
    raise SystemExit(main())
