#!/usr/bin/env python3
"""Meaningful negative tests for the cached-simulation replay.

  1. source-missing : a declared cache file absent under --analysis-root  -> exit 3
  2. hash-mismatch  : declared sha256 altered for a consumed cache file   -> exit 3
  3. non-empty out  : replay refuses a non-empty output directory         -> exit 4
  4. bad branch     : argparse rejects an unknown branch                  -> exit 2

Run under the pinned replay env:
  python code/c17_cache_simulations/negative_tests.py --analysis-root R --checkdir DIR
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPLAY = HERE / "replay.py"


def run_case(name, argv, expected, log_dir):
    log = log_dir / f"negative_{name}.log"
    proc = subprocess.run(argv, capture_output=True, text=True)
    text = (f"$ {' '.join(argv)}\n--- exit {proc.returncode} "
            f"(expected {expected}) ---\n{proc.stdout}{proc.stderr}")
    log.write_text(text, encoding="utf-8")
    return {
        "case": name,
        "argv": argv,
        "expected_exit": expected,
        "exit": proc.returncode,
        "pass": proc.returncode == expected,
        "log": str(log),
        "log_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "stderr_tail": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else None,
    }


LEGACY_PREFIX = [
    "config/CT_SIMULATIONS.json",
    "code/new_simulation_repair.py",
    "code/finalize_simulation_repair.py",
    "code/bootstrap_power_repair.py",
]


def build_missing_source_root(manifest, root, farm, skip_rel):
    """Real-copy mini tree holding every legacy read before the omitted source."""
    farm.mkdir(parents=True, exist_ok=True)
    todo = list(LEGACY_PREFIX)
    for e in manifest["sources"]:
        rel = e["path"]
        if rel.startswith("runs/new_simulation_repair/production/calibration_") or \
                rel.startswith("runs/new_simulation_repair/production/geometry_"):
            todo.append(rel)
    n = 0
    for rel in todo:
        if rel == skip_rel:
            continue
        target = farm / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(root / rel, target)
            n += 1
    return n


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-root", required=True)
    ap.add_argument("--checkdir", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    root = Path(args.analysis_root).resolve()
    check = Path(args.checkdir).resolve()
    log_dir = check / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    neg = check / "negative"
    neg.mkdir(parents=True, exist_ok=True)
    manifest_path = HERE / "SOURCE_MANIFEST_v1.json"
    manifest = json.loads(manifest_path.read_text())

    missing_rel = "runs/new_simulation_repair/production/power_000.npz"
    missing_root = neg / "analysis_root_missing_source"
    tag = 1
    while missing_root.exists() and any(missing_root.rglob("*")):
        tag += 1
        missing_root = neg / f"analysis_root_missing_source_{tag}"
    out_missing = neg / f"out_source_missing_{tag}"
    n_links = build_missing_source_root(manifest, root, missing_root, missing_rel)
    results = []
    results.append(run_case(
        "source_missing",
        [args.python, str(REPLAY), "--analysis-root", str(missing_root),
         "--out", str(out_missing), "--branch", "legacy"],
        3, log_dir))
    results[-1]["detail"] = {"farm_copies": n_links, "omitted_source": missing_rel,
                             "expect_message": "declared source missing on disk"}
    results[-1]["message_ok"] = bool(
        results[-1]["stderr_tail"] and
        "declared source missing on disk" in (results[-1]["stderr_tail"] or ""))
    results[-1]["pass"] = results[-1]["pass"] and results[-1]["message_ok"]

    bad_manifest = neg / f"manifest_hash_mismatch_{tag}.json"
    bad = json.loads(manifest_path.read_text())
    hits = 0
    for e in bad["sources"]:
        if e["path"] == "config/CT_SIMULATIONS.json":
            e["sha256"] = ("0" if e["sha256"][0] != "0" else "1") + e["sha256"][1:]
            hits += 1
    if hits != 1:
        raise SystemExit("manifest corruption setup failed")
    bad_manifest.write_text(json.dumps(bad), encoding="utf-8")
    results.append(run_case(
        "hash_mismatch",
        [args.python, str(REPLAY), "--analysis-root", str(root),
         "--out", str(neg / f"out_hash_mismatch_{tag}"), "--branch", "legacy",
         "--expected-manifest", str(bad_manifest)],
        3, log_dir))
    results[-1]["detail"] = {"corrupted_entry": "config/CT_SIMULATIONS.json",
                             "expect_message": "sha256 mismatch"}
    results[-1]["message_ok"] = bool(
        results[-1]["stderr_tail"] and
        "sha256 mismatch" in (results[-1]["stderr_tail"] or ""))
    results[-1]["pass"] = results[-1]["pass"] and results[-1]["message_ok"]

    # lexical escape: a declared path leaving the selected project must be refused
    escape_manifest = neg / f"manifest_lexical_escape_{tag}.json"
    esc = json.loads(manifest_path.read_text())
    for e in esc["sources"]:
        if e["path"] == "../Imports/geometry/rho_inputs_50.json":
            e["path"] = "../../../etc/hostname"
    escape_manifest.write_text(json.dumps(esc), encoding="utf-8")
    results.append(run_case(
        "lexical_escape",
        [args.python, str(REPLAY), "--analysis-root", str(root),
         "--out", str(neg / f"out_lexical_escape_{tag}"), "--branch", "panel",
         "--expected-manifest", str(escape_manifest)],
        3, log_dir))
    results[-1]["detail"] = {"expect_message": "escapes the selected project root"}
    results[-1]["message_ok"] = bool(
        results[-1]["stderr_tail"] and
        "escapes the selected project root" in (results[-1]["stderr_tail"] or ""))
    results[-1]["pass"] = results[-1]["pass"] and results[-1]["message_ok"]

    # symlink escape: a declared in-tree path that resolves outside the project
    sym_root = neg / f"analysis_root_symlink_escape_{tag}"
    sym_root.mkdir(parents=True, exist_ok=True)
    probe_rel = "runs/new_simulation_repair/production/power_000.npz"
    probe = sym_root / probe_rel
    probe.parent.mkdir(parents=True, exist_ok=True)
    if not probe.exists():
        os.symlink(root / probe_rel, probe)
    results.append(run_case(
        "symlink_escape",
        [args.python, str(REPLAY), "--analysis-root", str(sym_root),
         "--out", str(neg / f"out_symlink_escape_{tag}"), "--branch", "legacy"],
        3, log_dir))
    results[-1]["detail"] = {"symlink": str(probe), "target": str(root / probe_rel),
                             "expect_message": "resolves outside the selected tree"}
    results[-1]["message_ok"] = bool(
        results[-1]["stderr_tail"] and
        "resolves outside the selected tree" in (results[-1]["stderr_tail"] or ""))
    results[-1]["pass"] = results[-1]["pass"] and results[-1]["message_ok"]

    nonempty = neg / "out_nonempty"
    nonempty.mkdir(parents=True, exist_ok=True)
    (nonempty / "keep.txt").write_text("occupied\n", encoding="utf-8")
    results.append(run_case(
        "nonempty_out",
        [args.python, str(REPLAY), "--analysis-root", str(root),
         "--out", str(nonempty), "--branch", "legacy"],
        4, log_dir))

    results.append(run_case(
        "bad_branch",
        [args.python, str(REPLAY), "--analysis-root", str(root),
         "--out", str(neg / "out_bad_branch"), "--branch", "bogus"],
        2, log_dir))

    doc = {
        "schema": "c17-cache-simulation-negative-tests-v1",
        "analysis_root": str(root),
        "python": args.python,
        "n_cases": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "cases": results,
    }
    (check / "negative_results.json").write_text(json.dumps(doc, indent=1) + "\n",
                                                 encoding="utf-8")
    for r in results:
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['case']} exit={r['exit']} "
              f"expected={r['expected_exit']}")
    return 0 if doc["n_pass"] == doc["n_cases"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
