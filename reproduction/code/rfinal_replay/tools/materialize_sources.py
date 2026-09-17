#!/usr/bin/env python3
"""Materialize the C17 replay roots with digest verification.

Topology (matches the engine contract analysis.parent == project):

    <dest>/project/                                  <- PROJECT_ROOT
    <dest>/project/repro_paper2_20260915/            <- ANALYSIS_ROOT

Packaged code/config/comparison/locks are copied into the analysis mirror first
(package-code override, pinned), then every whitelist input referenced by an
active stage of the relocated config is materialized from --source-project /
--source-analysis and verified against its declared hash.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from pathlib import Path

PKG = Path(__file__).resolve().parents[3]
ANALYSIS_NAME = "repro_paper2_20260915"
OVERLAY_MAP = {"code": "code", "config": "config",
               "comparison": "recovered/c17_render_comparison_v1"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def roots(dest: Path):
    return dest / "project", dest / "project" / ANALYSIS_NAME


def load_config():
    return json.loads((PKG / "config/CT_C17_REPLAY_FINAL_v1.relocated.json").read_text())


def plan_entries(config):
    """Entries referenced by each active stage (input_refs, input_globs, adapter refs)."""
    entries = {e["id"]: e for e in config["input_whitelist"]["entries"]}
    selected = {}
    for stage in config["regeneration_plan"]["stages"]:
        sid = stage["id"]
        refs = list(stage.get("input_refs", []))
        refs += [i for i, e in entries.items() if sid in (e.get("consumed_by") or [])]
        adapter = stage.get("adapter") or {}
        refs += list(adapter.get("input_refs", []))
        for pattern in list(stage.get("input_globs", [])) + list(adapter.get("input_globs", [])):
            refs += sorted(i for i in entries if fnmatch.fnmatchcase(i, pattern))
        for ref in refs:
            e = entries.get(ref)
            if e is None:
                continue
            row = selected.setdefault(ref, {"id": e["id"], "root": e.get("root"), "path": e["path"],
                                            "sha256": e["sha256"], "level": e.get("level"),
                                            "consumed_by": []})
            row["consumed_by"] = sorted(set(row["consumed_by"] + [sid]))
    return list(selected.values())


def target_for(project: Path, analysis: Path, entry) -> Path:
    return (project / entry["path"]) if entry["root"] == "project" else (analysis / entry["path"])


def overlay_count():
    return sum(1 for rel in OVERLAY_MAP for p in (PKG / rel).rglob("*") if p.is_file())


def package_overlay(analysis: Path):
    copied = 0
    for rel, target_rel in OVERLAY_MAP.items():
        src_root = PKG / rel
        if not src_root.is_dir():
            continue
        for p in sorted(src_root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                dst = analysis / target_rel / p.relative_to(src_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
                copied += 1
    return copied


def verify(project: Path, analysis: Path, entries):
    missing, bad = [], []
    for e in entries:
        target = target_for(project, analysis, e)
        if not target.is_file():
            missing.append({"id": e["id"], "path": str(target)})
        elif sha(target) != e["sha256"]:
            bad.append({"id": e["id"], "path": str(target), "declared": e["sha256"],
                        "actual": sha(target)})
    return missing, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-project")
    ap.add_argument("--source-analysis")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-entries", type=int, default=None, help="bounded test mode")
    ap.add_argument("--plan-check", action="store_true",
                    help="independently resolve the plan and report unresolved stage inputs")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)
    project, analysis = roots(dest)
    config = load_config()
    entries = plan_entries(config)
    if args.max_entries is not None:
        entries = entries[:args.max_entries]
    stage_ids = [s["id"] for s in config["regeneration_plan"]["stages"]]
    per_stage = {sid: sum(1 for e in entries if sid in e["consumed_by"]) for sid in stage_ids}
    coverage = {"entries": len(entries), "per_stage": per_stage,
                "active_stages_complete": all(per_stage[sid] > 0 for sid in stage_ids)}
    if args.plan_check:
        import sys as _sys
        _sys.path.insert(0, str(PKG / "code"))
        from rfinal_replay import stages  # noqa: PLC0415
        plan = stages.build_plan(config, project, analysis, project / "runs/_plancheck")
        unresolved, overlay_ok, declared_ok = [], 0, 0
        for row in plan:
            for item in row["inputs"]:
                target = Path(item["path"])
                rel_analysis = str(target.relative_to(analysis)) if str(target).startswith(str(analysis)) else None
                rel_project = str(target.relative_to(project)) if str(target).startswith(str(project)) else None
                in_overlay = any((PKG / key / rel).is_file() for key, rel in
                                 (("code", rel_analysis), ("config", rel_analysis))
                                 if rel_analysis) or (rel_analysis or "").startswith("recovered/c17_render_comparison_v1")
                declared = any(e["path"] == (rel_analysis or rel_project) or
                               e["path"].endswith("/" + (rel_project or "")) for e in entries)
                if in_overlay:
                    overlay_ok += 1
                elif declared:
                    declared_ok += 1
                else:
                    unresolved.append({"stage": row["id"], "id": item["id"], "path": item["path"]})
        print(json.dumps({"mode": "plan-check", "stages": len(plan),
                          "resolved_inputs": overlay_ok + declared_ok,
                          "package_overlay": overlay_ok, "declared_materialized": declared_ok,
                          "unresolved": unresolved[:5], "n_unresolved": len(unresolved)}))
        return 1 if unresolved else 0
    if args.verify_only:
        missing, bad = verify(project, analysis, entries)
        print(json.dumps({"mode": "verify-only", "project_root": str(project),
                          "analysis_root": str(analysis), "coverage": coverage,
                          "missing": len(missing), "hash_mismatch": len(bad),
                          "problems": (missing + bad)[:5]}))
        return 1 if (missing or bad) else 0
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "dest": str(dest), "project_root": str(project),
                          "analysis_root": str(analysis), "coverage": coverage,
                          "package_overlay_files": overlay_count()}))
        return 0
    if not args.source_project or not args.source_analysis:
        raise SystemExit("--source-project and --source-analysis are required unless --dry-run/--verify-only")
    source_project, source_analysis = Path(args.source_project), Path(args.source_analysis)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"dest not empty: {dest}")
    project.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    overlay = package_overlay(analysis)
    copied = verified = 0
    problems = []
    for e in entries:
        target = target_for(project, analysis, e)
        if target.is_file() and sha(target) == e["sha256"]:
            verified += 1  # package-overlay or previously materialized resource wins
            continue
        root = e["root"]
        rel = e["path"]
        if root == "project" and rel.startswith(ANALYSIS_NAME + "/"):
            root, rel = "analysis", rel[len(ANALYSIS_NAME) + 1:]
        src = (source_project if root == "project" else source_analysis) / rel
        if not src.is_file():
            problems.append({"id": e["id"], "error": "missing source", "path": str(src)})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        copied += 1
        if sha(target) != e["sha256"]:
            problems.append({"id": e["id"], "error": "hash mismatch", "path": str(target),
                             "declared": e["sha256"], "actual": sha(target)})
        else:
            verified += 1
    print(json.dumps({"mode": "materialize", "project_root": str(project),
                      "analysis_root": str(analysis), "package_overlay_files": overlay,
                      "copied": copied, "verified": verified, "coverage": coverage,
                      "problems": problems[:5], "n_problems": len(problems)}))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
