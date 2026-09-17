"""Declared R6 regeneration plan: stage resolution and executability.

Stage kinds:

``input_audit``              verify the allowlist, levels and forbidden patterns.
``frozen_consumer_sandbox``  run a byte-identical frozen consumer inside a sandbox
                             under the fresh output root (no write into the signed tree).
``derived_table``            render a table from inputs produced earlier in *this*
                             run (fresh outputs only; never an earlier run's file).
``node_render``              one of the seven unfinished downstream figure/table nodes.
``external_producer``        a branch producer of retained numbers executed through
                             its declared path-safe adapter inside a fresh sandbox.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from . import guards

try:  # keep the module importable without the adapter engine (older callers)
    from . import adapters as _adapters
except Exception:  # pragma: no cover - defensive
    _adapters = None


def expand_input_refs(stage: dict, entries: dict[str, dict]) -> list[str]:
    resolved: list[str] = []
    for entry_id in stage.get("input_refs", []):
        if entry_id not in entries:
            raise guards.ReplayRefusal(f"stage {stage['id']}: unknown input ref {entry_id}")
        resolved.append(entry_id)
    for pattern in stage.get("input_globs", []):
        matches = sorted(entry_id for entry_id in entries if fnmatch.fnmatchcase(entry_id, pattern))
        if not matches:
            raise guards.ReplayRefusal(f"stage {stage['id']}: input glob matches nothing: {pattern}")
        resolved.extend(matches)
    seen: set[str] = set()
    unique: list[str] = []
    for entry_id in resolved:
        if entry_id not in seen:
            seen.add(entry_id)
            unique.append(entry_id)
    return unique


def build_plan(config: dict, project_root: Path, analysis_root: Path, out_root: Path) -> list[dict]:
    """Resolve every declared stage into a path-resolved, hash-carrying plan row."""
    entries = {entry["id"]: entry for entry in config["input_whitelist"]["entries"]}
    plan: list[dict] = []
    for stage in config["regeneration_plan"]["stages"]:
        if stage.get("kind") == "external_producer" and _adapters is not None and stage.get("adapter"):
            refs = _adapters.expand_stage_inputs(stage, entries)
        else:
            refs = expand_input_refs(stage, entries)
        resolved_inputs = []
        for entry_id in refs:
            entry = entries[entry_id]
            path = (
                Path(entry["path"]).expanduser().resolve()
                if entry["root"] == "absolute"
                else (project_root if entry["root"] == "project" else analysis_root) / entry["path"]
            )
            resolved_inputs.append(
                {
                    "id": entry_id,
                    "level": entry["level"],
                    "role": entry.get("role", ""),
                    "path": str(path),
                    "expected_sha256": entry["sha256"],
                }
            )
        code_ref = stage.get("consumer_code_ref")
        code_entry = entries.get(code_ref) if code_ref else None
        code_path = None
        if code_entry is not None:
            code_path = (
                Path(code_entry["path"]).expanduser().resolve()
                if code_entry["root"] == "absolute"
                else (project_root if code_entry["root"] == "project" else analysis_root)
                / code_entry["path"]
            )
        executable = bool(stage.get("executable"))
        if executable and code_ref is None and stage["kind"] == "frozen_consumer_sandbox":
            raise guards.ReplayRefusal(f"stage {stage['id']}: frozen consumer stage lacks consumer_code_ref")
        adapter = stage.get("adapter") if stage.get("kind") == "external_producer" else None
        fresh_inputs = list(stage.get("fresh_input_stages", []))
        if adapter:
            fresh_inputs += [item["stage"] for item in adapter.get("fresh_inputs", [])]
        plan.append(
            {
                "id": stage["id"],
                "kind": stage["kind"],
                "title": stage.get("title", ""),
                "node": stage.get("node"),
                "result_ids": list(stage.get("result_ids", [])),
                "status": stage.get("status", "UNKNOWN"),
                "executable": executable,
                "blocked_reason": stage.get("blocked_reason"),
                "consumer_code": str(code_path) if code_path else None,
                "consumer_code_sha256": code_entry["sha256"] if code_entry else None,
                "consumer_code_ref": stage.get("consumer_code_ref"),
                "adapter": adapter,
                "coverage_expectation": (adapter or {}).get("coverage_expectation"),
                "timeout_seconds": (adapter or {}).get("timeout_seconds"),
                "resource_policy": (adapter or {}).get("resource_policy"),
                "argv": list(stage.get("argv", [])),
                "run_dir": stage.get("run_dir"),
                "table_kind": stage.get("table_kind"),
                "render_fresh_source_map": list(stage.get("render_fresh_source_map", [])),
                "require_fresh": bool(stage.get("require_fresh")),
                "reference_dir": stage.get("reference_dir"),
                "comparison_bundle": stage.get("comparison_bundle"),
                "inputs": resolved_inputs,
                "fresh_input_stages": fresh_inputs,
                "outputs": [
                    {"path": item["path"], "role": item.get("role", "")}
                    for item in stage.get("outputs", [])
                ],
                "notes": stage.get("notes", ""),
            }
        )
    return plan


def plan_summary(plan: list[dict]) -> dict:
    ready = [row["id"] for row in plan if row["executable"]]
    pending = [
        {"id": row["id"], "kind": row["kind"], "reason": row["blocked_reason"]}
        for row in plan
        if not row["executable"]
    ]
    return {
        "n_stages": len(plan),
        "n_executable": len(ready),
        "n_pending": len(pending),
        "n_external_producers": sum(1 for row in plan if row["kind"] == "external_producer"),
        "n_external_producers_executable": sum(
            1 for row in plan if row["kind"] == "external_producer" and row["executable"]
        ),
        "executable": ready,
        "pending": pending,
    }


def render_plan_text(
    plan: list[dict],
    *,
    project_root: Path,
    analysis_root: Path,
    out_root: Path,
    config_path: Path,
    verify_inputs: str,
) -> str:
    lines = [
        "RFINAL CLEAN REPLAY - RESOLVED PLAN (dry run; nothing is written outside the process)",
        f"  project-root : {project_root}",
        f"  analysis-root: {analysis_root}",
        f"  config       : {config_path}",
        f"  output-root  : {out_root}",
        f"  verify-inputs: {verify_inputs}",
        "",
    ]
    for row in plan:
        state = "READY" if row["executable"] else "PENDING"
        lines.append(f"[{state}] {row['id']} ({row['kind']}) - {row['title']}")
        if row["result_ids"]:
            lines.append(f"         result_ids: {','.join(row['result_ids'])}")
        if row["consumer_code"]:
            lines.append(f"         consumer  : {row['consumer_code']} sha256={row['consumer_code_sha256']}")
        if row["inputs"]:
            lines.append(f"         inputs    : {len(row['inputs'])} allowlisted file(s)")
            for item in row["inputs"][:3]:
                lines.append(f"           - {item['id']} [{item['level']}] {item['path']}")
            if len(row["inputs"]) > 3:
                lines.append(f"           ... {len(row['inputs']) - 3} more")
        if row["fresh_input_stages"]:
            lines.append(f"         fresh-from: {','.join(row['fresh_input_stages'])}")
        if row.get("coverage_expectation"):
            lines.append(f"         coverage  : {row['coverage_expectation']}")
        if row.get("resource_policy"):
            lines.append(f"         resources : {row['resource_policy']}")
        for item in row["outputs"]:
            lines.append(f"         output    : {out_root / item['path']} ({item['role']})")
        if row["blocked_reason"]:
            lines.append(f"         blocked   : {row['blocked_reason']}")
        lines.append("")
    summary = plan_summary(plan)
    lines.append(
        f"SUMMARY: {summary['n_executable']} executable / {summary['n_pending']} pending; "
        "dry run wrote nothing."
    )
    return "\n".join(lines)
