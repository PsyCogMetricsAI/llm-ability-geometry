#!/usr/bin/env python3
"""Portability test: run the replay against a relocated, renamed project tree.

The relocated tree is a real copy (no writable hardlinks, no symlinked code or
config) whose project basename differs from the original project.  The run must
succeed while every scientific read resolves inside the relocated tree, and the
dynamic path audit must show zero reads of the original project tree.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPLAY = HERE / "replay.py"

RELOCATED_PROJECT_NAME = "C10S_RELOCATED_PROJECT"

COPY_ITEMS = [
    ("code/c17_cache_simulations", True),
    ("code/c17_panel_calibration", True),
    ("code/c17_geometry_validation", True),
    ("code/new_simulation_repair.py", False),
    ("code/finalize_simulation_repair.py", False),
    ("code/bootstrap_power_repair.py", False),
    ("config/CT_SIMULATIONS.json", False),
    ("config/CT_C17_PANEL_CALIBRATION_v2.json", False),
    ("config/CT_C17_GEOMETRY_v2.json", False),
    ("runs/new_simulation_repair/production", True),
    ("runs/bootstrap_power_repair/production", True),
    ("runs/c17_panel_calibration_v1/production_v1", True),
    ("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/twonn_clean", True),
    ("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/noise", True),
    ("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/boundary", True),
    ("runs/c17_geometry_validation_v1/attempts/C6.attempt-2/work/stability", True),
    ("runs/c17_geometry_validation_v1/OUTPUT_HASHES.json", False),
    ("runs/c17_geometry_validation_v1/RUN_MANIFEST_v1.json", False),
    ("reports/C17_PANEL_CALIBRATION_v1.json", False),
    ("reports/C17_GEOMETRY_RESULTS_v1.json", False),
    ("reports/NEW_SIMULATION_REPAIR.json", False),
    ("reports/NEW_BOOTSTRAP_POWER.json", False),
    ("reports/C5_GEOMETRY_FIXTURES_v1.npz", False),
    ("reports/C5_GEOMETRY_FIXTURES_v1.json", False),
    ("../Imports/geometry/rho_inputs_50.json", False),
    ("../Imports/geometry/covariates_50.json", False),
    ("../Imports/geometry/theta_hat_panel51.json", False),
    ("../Imports/geometry/g5no_gates_code.json", False),
    ("../Imports/geometry/g5no_gates_math.json", False),
    ("../Imports/geometry/route_c_mc_dispatch_20260608/activation_indicators_lib.py", False),
    ("../Imports/geometry/offline_analyze_docs_20260607/s5_convergence_analyze.py", False),
]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-root", required=True)
    ap.add_argument("--checkdir", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    root = Path(args.analysis_root).resolve()
    check_dir = Path(args.checkdir).resolve()
    log_dir = check_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    reloc = check_dir / "relocated" / RELOCATED_PROJECT_NAME
    rp = reloc / root.name
    if reloc.exists():
        raise SystemExit(f"refusing: relocated tree already exists: {reloc}")
    reloc.mkdir(parents=True)
    for rel, is_dir in COPY_ITEMS:
        src = (root / rel).resolve() if not rel.startswith("..") else (root / rel).resolve()
        dst = (rp / rel).resolve() if not rel.startswith("..") else (rp / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        # plain recursive copy: no ownership/mode preservation attempts (the
        # relocated tree must be an independent read-only-content copy)
        cp = subprocess.run(["cp", "-r", str(src), str(dst)], capture_output=True, text=True)
        if cp.returncode != 0:
            raise SystemExit(f"copy failed for {rel}: {cp.stderr}")
    out = check_dir / "relocated_out"
    log = log_dir / "relocated_run.log"
    argv = [args.python, str(REPLAY), "--analysis-root", str(rp), "--out", str(out),
            "--branch", "all",
            "--expected-manifest", str(rp / "code" / "c17_cache_simulations" /
                                       "SOURCE_MANIFEST_v1.json")]
    proc = subprocess.run(argv, capture_output=True, text=True)
    text = (f"$ {' '.join(argv)}\n--- exit {proc.returncode} ---\n{proc.stdout}{proc.stderr}")
    log.write_text(text, encoding="utf-8")
    checks = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    check("original_tree_still_exists", root.exists(), {"original_root": str(root)})
    check("different_destination_project_basename",
          reloc.name != root.parent.name,
          {"relocated_project": reloc.name, "original_project": root.parent.name})
    check("replay_exit_zero", proc.returncode == 0, {"exit": proc.returncode,
                                                     "log": str(log)})
    audit = json.loads((out / "path_audit.json").read_text()) if \
        (out / "path_audit.json").exists() else {}
    check("audit_no_original_tree_reads", audit.get("counts", {}).get("original_tree_reads") == 0,
          audit.get("counts"))
    check("audit_no_writes_outside_out_dir",
          audit.get("counts", {}).get("writes_outside_out_dir") == 0, audit.get("counts"))
    check("audit_selected_tree_reads_positive",
          (audit.get("counts", {}).get("selected_tree_reads") or 0) > 100,
          audit.get("counts"))
    sm = json.loads((out / "source_manifest.json").read_text()) if \
        (out / "source_manifest.json").exists() else {"sources": []}
    outside = [s["path"] for s in sm.get("sources", [])
               if s.get("resolved_path") and not (
                   s["resolved_path"].startswith(str(rp.resolve()) + "/")
                   or s["resolved_path"].startswith(str(reloc.resolve()) + "/"))]
    check("all_consumed_sources_resolve_in_relocated_tree", not outside,
          {"n_sources": len(sm.get("sources", [])), "outside": outside[:5]})
    receipt = json.loads((out / "run_receipt.json").read_text()) if \
        (out / "run_receipt.json").exists() else {}
    check("receipt_all_branches_pass",
          receipt.get("status") == "pass" and
          all(b.get("status") == "pass" for b in receipt.get("branches", {}).values()),
          {k: v.get("status") for k, v in receipt.get("branches", {}).items()})
    doc = {
        "schema": "c17-cache-simulation-relocation-test-v1",
        "original_root": str(root), "relocated_root": str(rp),
        "relocated_project_name": reloc.name,
        "original_project_name": root.parent.name,
        "argv": argv, "exit": proc.returncode,
        "log": str(log), "log_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "n_checks": len(checks), "n_fail": sum(1 for c in checks if not c["pass"]),
        "checks": checks,
    }
    (check_dir / "relocation_results.json").write_text(json.dumps(doc, indent=1) + "\n",
                                                       encoding="utf-8")
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']}")
    return 0 if doc["n_fail"] == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
