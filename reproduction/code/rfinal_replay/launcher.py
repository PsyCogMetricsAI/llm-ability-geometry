#!/usr/bin/env python3
"""RFINAL clean-replay launcher (paper2, round repro_paper2_20260915).

Usage (dry run, writes nothing outside a temporary directory you choose):

    python3 code/rfinal_replay/launcher.py \
        --project-root <PROJECT_ROOT> \
        --analysis-root <ANALYSIS_ROOT> \
        --out <NEW_OUTPUT_DIR> \
        --dry-run --verify-inputs none

Execution requires --execute, the public replay-ready configuration, frozen tolerances and a new empty output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow ``python3 code/rfinal_replay/launcher.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rfinal_replay import guards, manifest, stages  # type: ignore
    from rfinal_replay import sandbox_runner  # type: ignore
    from rfinal_replay import adapters  # type: ignore
else:  # pragma: no cover
    from . import adapters, guards, manifest, sandbox_runner, stages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFINAL clean-replay launcher (dry-run by default)")
    parser.add_argument("--project-root", required=False, help="project root P (explicit; never inferred from __file__)")
    parser.add_argument("--analysis-root", required=False, help="analysis root R (explicit; never inferred from __file__)")
    parser.add_argument("--config", default=None, help="config path (default: <analysis-root>/config/RFINAL_REPLAY_DRAFT.json)")
    parser.add_argument("--out", default=None, help="target output directory (must not exist or must be empty)")
    parser.add_argument("--stage", action="append", default=None, help="limit execution to these stage ids (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan and write nothing")
    parser.add_argument("--execute", action="store_true", help="execute ready stages (requires authorization + frozen tolerances)")
    parser.add_argument(
        "--verify-inputs",
        choices=("none", "lazy", "full"),
        default=None,
        help="none: plan only; lazy: verify inputs of selected stages; full: verify every whitelist entry",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    parser.add_argument(
        "--allow-large-hash",
        action="store_true",
        help="allow hashing whitelist entries larger than 1 GiB (declared multi-GB weight shards are not hashed by default)",
    )
    parser.add_argument("--selftest", action="store_true", help="run the executor self-tests and exit")
    parser.add_argument("--print-config-summary", action="store_true", help="print whitelist/plan counts and exit")
    parser.add_argument(
        "--mode",
        choices=("replay", "engine-check"),
        default="replay",
        help="replay: the authorized R6 clean replay; engine-check: declared adapter/refusal checks only",
    )
    parser.add_argument("--max-stages", type=int, default=None, help="stop after N selected stages (engine checks)")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="refuse to report success unless every executable stage in scope produced outputs",
    )
    return parser.parse_args(argv)


def default_config_path(analysis_root: Path) -> Path:
    return analysis_root / "config" / "RFINAL_REPLAY_DRAFT.json"


def default_out_root(analysis_root: Path, config: dict) -> Path:
    out_rel = config["root_contract"]["output_root_default"]
    return (analysis_root / out_rel).resolve()


def resolve_output_root(requested: str | None, analysis_root: Path, config: dict) -> Path:
    if requested:
        out = Path(requested).expanduser()
        if not out.is_absolute():
            out = Path.cwd() / out
        return out.resolve()
    return default_out_root(analysis_root, config)


def compute_status(config: dict, plan: list[dict]) -> dict:
    return {
        "authorized": config["execution_authorization"]["status"]
        == config["execution_authorization"]["required_status_for_run"],
        "tolerances_frozen": not any(
            isinstance(value, str) and value.startswith("TO_BE_FROZEN_BY_ROOT_BEFORE_R6")
            for value in config["tolerances"].values()
        ),
        "plan": stages.plan_summary(plan),
    }


def verify_stage_inputs(allow: guards.InputAllowlist, stage: dict, mode: str) -> dict[str, str]:
    if mode == "none":
        return {item["id"]: item["expected_sha256"] for item in stage["inputs"]}
    return allow.verify_many([item["id"] for item in stage["inputs"]], force=(mode == "full"))


def derive_execution_ctx(out_root: Path, project_root: Path | None = None,
                         analysis_root: Path | None = None) -> dict:
    """Execution roots for derived stages; never resolved from __file__."""
    if analysis_root is None:
        analysis_root = Path(out_root).resolve().parents[1]
    if project_root is None:
        project_root = Path(analysis_root).resolve().parent
    return {"out_root": Path(out_root).resolve(), "project_root": Path(project_root).resolve(),
            "analysis_root": Path(analysis_root).resolve()}


def run_derived_table(stage: dict, out_root: Path, fresh_registry: dict[str, list[dict]],
                      config: dict | None = None, mode: str = "replay",
                      ctx: dict | None = None) -> dict:
    """Render a table from freshly produced artifacts of this same run only."""
    ctx = ctx or derive_execution_ctx(out_root)
    if stage["table_kind"] == "c17_depth_summary_fresh":
        return _run_depth_summary_fresh(stage, out_root)
    if stage["table_kind"] == "c17_render_suite":
        return _run_c17_render_suite(stage, out_root, fresh_registry, config, mode, ctx)
    if stage["table_kind"] == "c17_result_validation":
        return _run_c17_result_validation(stage, out_root, fresh_registry, config, ctx)
    if stage["table_kind"] == "c17_claim_map":
        return _run_c17_claim_map(stage, out_root, fresh_registry, config, ctx)
    if stage["table_kind"] != "window_chi_table":
        raise guards.ReplayRefusal(f"stage {stage['id']}: unsupported derived table kind {stage['table_kind']!r}")
    rows = []
    for stage_id in stage["fresh_input_stages"]:
        for record in fresh_registry.get(stage_id, []):
            path = Path(record["path"])
            if guards.sha256_file(path) != record["sha256"]:
                raise guards.ReplayRefusal(f"fresh artifact changed mid-run: {path}")
            metrics = json.loads(path.read_text())
            if "windows" not in metrics or "used_ids" not in metrics:
                raise guards.ReplayRefusal(f"fresh metrics missing window fields: {path}")
            rows.append(
                {
                    "source_stage": stage_id,
                    "run_dir": metrics.get("run_dir", record.get("source_run_dir", "")),
                    "n_used": metrics.get("n_used"),
                    "used_ids_sha256": guards.canonical_sha256(metrics.get("used_ids", [])),
                    "window_chi": {key: value["chi"] for key, value in metrics["windows"].items()},
                    "max_pairwise_rel_diff": metrics.get("max_pairwise_rel_diff"),
                }
            )
    if not rows:
        raise guards.ReplayRefusal(f"stage {stage['id']}: no fresh metrics available (run the upstream stages first)")
    table = {
        "schema": "rfinal-window-chi-table-v1",
        "producer": "code/rfinal_replay/launcher.py",
        "created_utc": manifest.utc_now(),
        "rule": "regenerated only from metrics.json files produced earlier in this same fresh run",
        "rows": sorted(rows, key=lambda row: row["source_stage"]),
    }
    output_path = out_root / stage["outputs"][0]["path"]
    digest = manifest.write_json_atomic(output_path, table)
    return {"path": str(output_path), "sha256": digest, "role": stage["outputs"][0]["role"]}


def _run_depth_summary_fresh(stage: dict, out_root: Path) -> dict:
    """Fresh G4 summary computed from per-model/per-layer rows (no archive input)."""
    import csv as _csv
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rfinal_replay.render import depth_aggregate  # noqa: PLC0415

    src = Path(stage["inputs"][0]["path"])
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    fresh = depth_aggregate.recompute_summary(rows)
    peaks = depth_aggregate.recompute_peaks(fresh)
    out = out_root / stage["outputs"][0]["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh, lineterminator="\n")
        writer.writerow(list(depth_aggregate.SUMMARY_COLUMNS))
        writer.writerows([[r[c] for c in depth_aggregate.SUMMARY_COLUMNS] for r in fresh])
    extra = None
    if len(stage["outputs"]) > 1:
        extra_path = out_root / stage["outputs"][1]["path"]
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        with open(extra_path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.writer(fh, lineterminator="\n")
            writer.writerow(["metric", "relative_depth", "value", "n_models"])
            writer.writerows([[p["metric"], p["relative_depth"], p["value"], p["n_models"]] for p in peaks])
        extra = {"path": str(extra_path), "sha256": guards.sha256_file(extra_path)}
    return {"path": str(out), "sha256": guards.sha256_file(out),
            "role": stage["outputs"][0]["role"], "extra_outputs": [extra] if extra else []}


def _adapt_fresh_input(item: dict, stage: dict, out_root: Path, suite_dir: Path,
                       fresh_registry: dict[str, list[dict]] | None = None):
    """Build a renderer-ready artifact from fresh in-run cache-driver outputs."""
    import hashlib
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rfinal_replay import fresh_adapters  # noqa: PLC0415

    kind = item["adapter"]
    src_stage = item["stage_id"]
    sandbox_out = out_root / "sandbox" / src_stage / "out"
    records = (fresh_registry or {}).get(src_stage, [])
    if kind == "grid":
        pattern = f"F1_MAP__{item['grid_stage']}__*.json"
        unique_records = {str(Path(r["path"])): r for r in records}
        cells = sorted(Path(r["path"]) for r in unique_records.values()
                       if Path(r["path"]).name.startswith(pattern.split("*")[0])
                       and Path(r["path"]).suffix == ".json"
                       and "/cells/" in str(r["path"]).replace("\\", "/"))
        if not cells:
            cells = sorted((sandbox_out / "new28" / "cells").glob(pattern))
        if not cells:
            raise guards.ReplayRefusal(f"no fresh {item['grid_stage']} cell files under {sandbox_out}")
        doc = fresh_adapters.assemble_grid(cells, family="F1", stage=item["grid_stage"],
                                           expected_cells=28)
        path = suite_dir / "adapted" / f"F1-{item['grid_stage']}-28.json"
        digest = fresh_adapters.write_json(path, doc)
        return {"path": str(path), "sha256": digest, "kind": kind,
                "source_files": [str(c) for c in cells]}
    if kind == "probes":
        agg = next((Path(r["path"]) for r in records
                    if Path(r["path"]).name == "probe_aggregates.json"), None)
        if agg is None:
            agg = sandbox_out / "probes" / "probe_aggregates.json"
        if not Path(agg).is_file():
            raise guards.ReplayRefusal(f"fresh probe aggregates missing: {agg}")
        docs = fresh_adapters.build_probe_docs(agg)
        path = suite_dir / "adapted" / f"probes_{item['part']}.json"
        digest = fresh_adapters.write_json(path, docs[item["part"]])
        return {"path": str(path), "sha256": digest, "kind": kind, "source": str(agg)}
    raise guards.ReplayRefusal(f"unknown fresh adapter {kind!r}")


STAGE_ROOT_LAYOUT = {
    "S40_cache_simulations": "sandbox/S40_cache_simulations/out",
    "S41_cache_observations": "sandbox/S41_cache_observations/out",
    "S42_cache_baseline": "sandbox/S42_cache_baseline/out",
    "S43_supplements_a": "sandbox/S43_supplements_a/out",
    "S44_supplements_b": "sandbox/S44_supplements_b/out",
    "S14_c17_depth_summary_fresh": ".",
    "S15_c17_render_suite": "render/outputs",
}
CLAIM_MAP_COUNTS = {"records_total": 147, "records_mandatory": 139, "bindings_verified": 697,
                    "records_with_verified_bindings": 142, "explicit_nonnumeric_records": 5,
                    "pending_records": 0}


def claim_map_roots(stage: dict, out_root: Path, ctx: dict, harness_roots: dict | None,
                    replay: bool):
    """Explicit declared topology; replay requires roots + real receipts under the run root."""
    analysis_root = ctx["analysis_root"]
    roots = {}
    for sid in stage["stage_root_ids"]:
        if harness_roots is not None:
            root = Path(harness_roots[sid]).resolve()
            if not root.is_dir():
                raise guards.ReplayRefusal(f"claim map harness root missing for {sid}: {root}")
            roots[sid] = str(root)
            continue
        rel = STAGE_ROOT_LAYOUT[sid]
        root = (out_root / rel).resolve() if rel != "." else out_root.resolve()
        if not str(root).startswith(str(out_root.resolve())):
            raise guards.ReplayRefusal(f"claim map: stage root escapes the run root: {sid}")
        if not root.is_dir():
            raise guards.ReplayRefusal(f"claim map: declared stage root missing (no aliases): {sid} {root}")
        receipt = out_root / "receipts" / f"{sid}.json"
        if replay and not receipt.is_file():
            raise guards.ReplayRefusal(f"claim map: real receipt missing for {sid}: {receipt}")
        if replay and rel != "." and not any(root.iterdir()):
            raise guards.ReplayRefusal(f"claim map: empty stage root for {sid}")
        roots[sid] = str(root)
    return roots


def _run_c17_claim_map(stage: dict, out_root: Path, fresh_registry: dict[str, list[dict]],
                       config: dict | None, ctx: dict, harness_roots: dict | None = None,
                       replay: bool = True) -> dict:
    """Claim-map validation over explicitly declared current-run roots (no source writes)."""
    import subprocess as _subprocess
    import sys as _sys

    analysis_root = ctx["analysis_root"]
    roots = claim_map_roots(stage, out_root, ctx, harness_roots, replay)
    # registry hashes must be valid for every declared root in both modes
    allowed_bases = [out_root.resolve()] if replay else [Path(r).resolve() for r in (harness_roots or {}).values()]
    for sid, root in roots.items():
        records = fresh_registry.get(sid) or []
        if not records:
            raise guards.ReplayRefusal(f"claim map: no registry records for {sid}")
        for rec in records:
            path = Path(rec["path"])
            if not path.is_file() or guards.sha256_file(path) != rec["sha256"]:
                raise guards.ReplayRefusal(f"claim map: registry hash invalid for {sid}: {path}")
            if not any(str(path.resolve()).startswith(str(base) + os.sep) or path.resolve() == base
                       for base in allowed_bases):
                raise guards.ReplayRefusal(f"claim map: registry record outside the declared roots for "
                                           f"{sid}: {path}")
    stage_outputs_p = out_root / "claim_map/stage_outputs.json"
    stage_outputs_p.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json_atomic(stage_outputs_p, {
        "schema": "c17-stage-outputs-v1",
        "note": "explicit declared current-run stage roots" if replay else "harness fixture roots",
        "runs": roots, "stages": {sid: {"run": sid, "subpath": "."} for sid in roots}})
    out_dir = out_root / "claim_map/run"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [_sys.executable, str(analysis_root / stage["validator"]),
           "--mapping", str(analysis_root / stage["mapping"]),
           "--stage-outputs", str(stage_outputs_p), "--out", str(out_dir), "--require-complete"]
    proc = _subprocess.run(cmd, cwd=str(analysis_root), capture_output=True, text=True)
    summary_p = out_dir / "summary.json"
    validation_p = out_dir / "validation.jsonl"
    if proc.returncode != 0 or not summary_p.is_file():
        raise guards.ReplayRefusal(f"stage {stage['id']}: claim-map validator refused exit={proc.returncode}: "
                                   + (proc.stderr or proc.stdout)[-300:])
    summary = json.loads(summary_p.read_text())
    for key, want in CLAIM_MAP_COUNTS.items():
        if summary.get(key) != want:
            raise guards.ReplayRefusal(f"stage {stage['id']}: claim-map {key}={summary.get(key)} != {want}")
    if summary.get("failures"):
        raise guards.ReplayRefusal(f"stage {stage['id']}: claim-map failures: "
                                   + json.dumps(summary["failures"][:3]))
    if not validation_p.is_file() or validation_p.stat().st_size == 0:
        raise guards.ReplayRefusal(f"stage {stage['id']}: validation.jsonl missing/empty")
    payload = {"schema": "c17-claim-map-stage-v1", "stage": stage["id"], "roots": roots,
               "evidence_level": "replay" if replay else "bounded harness (labelled fixture receipts)",
               "counts": {k: summary[k] for k in CLAIM_MAP_COUNTS},
               "outputs": [{"path": str(summary_p), "sha256": guards.sha256_file(summary_p)},
                           {"path": str(validation_p), "sha256": guards.sha256_file(validation_p)}]}
    out = out_root / stage["outputs"][0]["path"]
    digest = manifest.write_json_atomic(out, payload)
    return {"path": str(out), "sha256": digest, "role": stage["outputs"][0]["role"],
            "extra_outputs": payload["outputs"]}


def _run_c17_result_validation(stage: dict, out_root: Path, fresh_registry: dict[str, list[dict]],
                               config: dict | None, ctx: dict) -> dict:
    """Non-vacuous validation of the fresh CSV bundle against the pinned comparison bundle."""
    import csv as _csv
    import os as _os

    analysis_root = ctx["analysis_root"]
    records = fresh_registry.get("S15_c17_render_suite", [])
    bundle_rec = next((r for r in records if Path(r["path"]).name == "RENDER_BUNDLE.json"), None)
    if bundle_rec is None:
        raise guards.ReplayRefusal(f"stage {stage['id']}: RENDER_BUNDLE.json not found in the fresh registry")
    bundle = json.loads(Path(bundle_rec["path"]).read_text())
    if bundle.get("schema") != "c17-render-bundle-v1" or not bundle.get("artifacts"):
        raise guards.ReplayRefusal(f"stage {stage['id']}: render bundle schema/artifacts invalid")
    bundle_path = Path(bundle_rec["path"])
    if guards.sha256_file(bundle_path) != bundle_rec["sha256"]:
        raise guards.ReplayRefusal(f"stage {stage['id']}: render bundle changed mid-run")
    fresh_outdir = Path(bundle["render_outdir"])
    cmp_rel = (stage.get("comparison_bundle")
               or (config or {}).get("c10_integration", {}).get("comparison_bundle")
               or "config/CT_C17_RENDER_COMPARISON_v1.json")
    cfg_path = analysis_root / cmp_rel
    cmp_cfg = json.loads(cfg_path.read_text())
    ref_dir = analysis_root / cmp_cfg["bundle_dir"]
    results, failures, n_compared = [], [], 0

    def fail(fname, column, row, detail):
        failures.append({"file": fname, "column": column, "row": row, "detail": detail})

    def resolve_source(rec_path: str):
        cand = Path(rec_path)
        if cand.is_absolute():
            return cand if cand.is_file() else None
        bases = [fresh_outdir, fresh_outdir.parent.parent, analysis_root]
        for base in bases:
            p = (base / cand).resolve()
            if p.is_file():
                return p
        return None

    for spec in cmp_cfg["files"]:
        fname, rel = spec["result_id"], spec["path"]
        ref_file, fresh_file = ref_dir / rel, fresh_outdir / rel
        row = {"file": rel, "n_rows": spec["n_rows"], "comparisons": 0, "provenance_rows": 0,
               "status": "PENDING"}
        if not ref_file.is_file():
            raise guards.ReplayRefusal(f"stage {stage['id']}: comparison reference missing: {ref_file}")
        if guards.sha256_file(ref_file) != spec["sha256"]:
            raise guards.ReplayRefusal(f"stage {stage['id']}: comparison reference tampered: {ref_file}")
        if not fresh_file.is_file():
            raise guards.ReplayRefusal(f"stage {stage['id']}: fresh output missing: {fresh_file}")
        with open(ref_file, newline="", encoding="utf-8") as fh:
            ref_rows = list(_csv.DictReader(fh))
        with open(fresh_file, newline="", encoding="utf-8") as fh:
            fresh_rows = list(_csv.DictReader(fh))
        if list(ref_rows[0].keys() if ref_rows else []) != spec["columns"]:
            raise guards.ReplayRefusal(f"stage {stage['id']}: reference columns changed: {rel}")
        if list(fresh_rows[0].keys() if fresh_rows else []) != spec["columns"]:
            raise guards.ReplayRefusal(f"stage {stage['id']}: fresh columns differ from spec: {rel}")
        if len(ref_rows) != spec["n_rows"] or len(fresh_rows) != spec["n_rows"]:
            raise guards.ReplayRefusal(
                f"stage {stage['id']}: row count mismatch for {rel}: "
                f"ref={len(ref_rows)} fresh={len(fresh_rows)} pinned={spec['n_rows']}")

        def is_null(v):
            return v is None or str(v).strip().lower() in ("", "nan", "null", "na", "--")

        for i, (r_ref, r_new) in enumerate(zip(ref_rows, fresh_rows)):
            for col, policy in spec["column_policies"].items():
                a, b = r_ref.get(col, ""), r_new.get(col, "")
                if policy == "provenance_traceability":
                    if col in ("source_path", "source_sha256"):
                        continue  # handled per-row below
                    if col == "source_pointer":
                        # presence in the referenced fresh JSON is verified per row below
                        continue
                    if col == "evidence_source":
                        if not str(b).strip():
                            fail(fname, col, i, "empty evidence_source")
                        row["comparisons"] += 1
                    continue
                if policy == "exact_int":
                    if is_null(a) and is_null(b):
                        continue
                    try:
                        ok = int(str(a)) == int(str(b))
                    except ValueError:
                        ok = str(a) == str(b)
                    row["comparisons"] += 1
                    if not ok:
                        fail(fname, col, i, f"exact_int {a!r} != {b!r}")
                elif policy == "exact_text":
                    if is_null(a) and is_null(b):
                        continue
                    row["comparisons"] += 1
                    if str(a) != str(b):
                        fail(fname, col, i, f"exact_text {a!r} != {b!r}")
                elif policy == "numeric":
                    if is_null(a) or is_null(b):
                        row["comparisons"] += 1
                        if is_null(a) != is_null(b):
                            fail(fname, col, i, f"null mask {a!r} vs {b!r}")
                        continue
                    try:
                        f_ref, f_new = float(a), float(b)
                    except ValueError:
                        fail(fname, col, i, f"unparseable numeric {a!r} vs {b!r}")
                        continue
                    row["comparisons"] += 1
                    if not (math.isfinite(f_ref) and math.isfinite(f_new)):
                        fail(fname, col, i, f"nonfinite numeric ref={f_ref!r} fresh={f_new!r}")
                        continue
                    if not (abs(f_new - f_ref) <= 1e-9 + 1e-6 * abs(f_ref)):
                        fail(fname, col, i, f"numeric {f_ref!r} vs {f_new!r}")
            # provenance traceability for every row of provenance columns
            if "source_path" in spec["columns"] and "source_sha256" in spec["columns"]:
                rec_path, rec_sha = r_new.get("source_path", ""), r_new.get("source_sha256", "")
                rec_ptr = r_new.get("source_pointer", "") if "source_pointer" in spec["columns"] else ""
                if not str(rec_path).strip() or not str(rec_sha).strip():
                    fail(fname, "source_path", i,
                         f"mandatory provenance columns empty: path={rec_path!r} sha={rec_sha!r}")
                else:
                    resolved = resolve_source(rec_path)
                    if resolved is None:
                        fail(fname, "source_path", i, f"fresh source missing: {rec_path}")
                    elif guards.sha256_file(resolved) != rec_sha:
                        fail(fname, "source_path", i, f"fresh source hash mismatch: {rec_path}")
                    else:
                        row["provenance_rows"] += 1
                        n_compared += 1
                        if str(rec_ptr).strip():
                            try:
                                doc = json.loads(Path(resolved).read_text())
                            except Exception as exc:
                                fail(fname, "source_pointer", i, f"source not readable JSON: {exc}")
                            else:
                                okptr = True
                                for one_ptr in str(rec_ptr).split("|"):
                                    cur = doc
                                    for part in [x for x in one_ptr.split("/") if x]:
                                        if isinstance(cur, list):
                                            cur = cur[int(part)]
                                        elif isinstance(cur, dict) and part in cur:
                                            cur = cur[part]
                                        else:
                                            okptr = False
                                            break
                                    if not okptr:
                                        break
                                if not okptr:
                                    fail(fname, "source_pointer", i,
                                         f"pointer {rec_ptr!r} not present in fresh source {rec_path}")
                                else:
                                    n_compared += 1
            if "evidence_source" in spec["columns"]:
                text = str(r_new.get("evidence_source", ""))
                for token in text.replace(";", " ").split():
                    if (token.count(".") == 1 and "/" in token
                            and token.endswith((".json", ".csv", ".tex"))):
                        if resolve_source(token) is None:
                            fail(fname, "evidence_source", i, f"unresolved evidence path: {token}")
                        else:
                            n_compared += 1
        n_compared += row["comparisons"]
        if row["comparisons"] + row["provenance_rows"] == 0:
            raise guards.ReplayRefusal(f"stage {stage['id']}: zero comparisons for required file {rel}")
        row["status"] = "PASS" if not any(f["file"] == rel for f in failures) else "FAIL"
        results.append(row)
    if n_compared == 0:
        raise guards.ReplayRefusal(f"stage {stage['id']}: validation made zero comparisons")
    if failures:
        raise guards.ReplayRefusal(
            f"stage {stage['id']}: validation failures ({len(failures)}): "
            + json.dumps(failures[:5]))
    payload = {"schema": "c17-result-validation-v1", "stage": stage["id"],
               "created_utc": manifest.utc_now(),
               "comparator": "abs(a-b) <= 1e-9 + 1e-6*abs(reference); ints/text exact; null masks exact",
               "comparison_bundle": {"path": str(cfg_path.relative_to(analysis_root)),
                                     "dir": str(ref_dir.relative_to(analysis_root)),
                                     "files": len(cmp_cfg["files"])},
               "fresh_outdir": str(fresh_outdir),
               "results": results, "n_files": len(results), "n_compared": n_compared, "n_failed": 0,
               "provenance_columns": "source_path/source_sha256 verified against the current fresh files; "
                                     "not compared literally with previous-run paths"}
    out = out_root / stage["outputs"][0]["path"]
    digest = manifest.write_json_atomic(out, payload)
    return {"path": str(out), "sha256": digest, "role": stage["outputs"][0]["role"]}


def _run_c17_render_suite(stage: dict, out_root: Path, fresh_registry: dict[str, list[dict]],
                          config: dict | None, mode: str = "replay", ctx: dict | None = None) -> dict:
    """Run the seven-output renderer inside the run root from declared inputs.

    Declared ``render_fresh_source_map`` entries substitute freshly produced
    artifacts of this run for the pinned C10 bindings *when available*; entries
    without a fresh artifact fall back to the pinned accepted source and are
    recorded as such in the bundle (no silent substitution).
    """
    import os as _os
    import subprocess as _subprocess
    import sys as _sys

    if not config or "render_contract" not in config:
        raise guards.ReplayRefusal(f"stage {stage['id']}: config carries no render_contract")
    ctx = ctx or derive_execution_ctx(out_root)
    analysis_root = ctx["analysis_root"]
    project_root = ctx["project_root"]
    suite_dir = (out_root / stage["outputs"][0]["path"]).parent
    suite_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(json.dumps(config["render_contract"]))
    applied, pinned = [], []
    pending_schema = []
    for item in stage.get("render_fresh_source_map", []):
        rec = None
        if item.get("adapter"):
            try:
                rec = _adapt_fresh_input(item, stage, out_root, suite_dir, fresh_registry)
            except Exception as exc:
                pending_schema.append({**item, "reason": f"fresh adapter failed: {exc}"})
                continue
        if item.get("schema_compatible") is False and not rec:
            pending_schema.append({**item, "reason": "fresh producer output needs a schema adapter before "
                                                     "it can replace this renderer input"})
            continue
        # preserve an adapter-produced record; consult the registry only when no
        # adapter record exists (root independence check C10i.root-render-check)
        recs = fresh_registry.get(item["stage_id"], []) if rec is None else []
        if rec is None and item.get("artifact"):
            rec = next((r for r in recs if Path(r["path"]).name == item["artifact"]), None)
            if rec is None and recs:
                pinned.append({**item, "reason": "declared artifact name not produced by the fresh stage"})
                continue
        elif rec is None and recs:
            rec = recs[-1]
        if rec is None:
            pinned.append({**item, "reason": "fresh artifact not produced in this run"})
            continue
        for entry in contract["outputs"]:
            if entry["id"] != item.get("result_id", entry["id"]):
                continue
            for spec in entry["inputs"]:
                if spec.get("role") == item["role"]:
                    spec["path"] = _os.path.relpath(rec["path"], out_root).replace(_os.sep, "/")
                    spec["sha256"] = rec["sha256"]
                    spec["external"] = True
                    spec["runtime_relative"] = True
                    if item.get("computational_input") is False:
                        spec["comparison_only"] = True
                        spec["computational_input"] = False
                    applied.append({**item, "sha256": rec["sha256"]})
    # Re-point every remaining accepted path relative to the run root (escapes allowed
    # because the sources are hash-pinned and declared in the whitelist).
    for entry in contract["outputs"]:
        for spec in entry["inputs"]:
            if spec.get("runtime_relative"):
                continue
            base = project_root if spec.get("external") and not spec["path"].startswith("..") else analysis_root
            abs_path = (base / spec["path"]).resolve()
            spec["path"] = _os.path.relpath(abs_path, out_root).replace(_os.sep, "/")
            spec["external"] = True
    if stage.get("require_fresh") and mode == "replay":
        missing = ([{**p, "kind": "pinned_fallback"} for p in pinned] + list(pending_schema))
        if missing:
            raise guards.ReplayRefusal(
                "stage %s: replay refuses missing fresh upstream for render inputs: %s"
                % (stage["id"], json.dumps([{k: m.get(k) for k in ("result_id", "role", "stage_id", "reason")}
                                            for m in missing])))
    contract_path = suite_dir / "CONTRACT_FRESH.json"
    digest = manifest.write_json_atomic(contract_path, contract)
    render_out = suite_dir / "outputs"
    def _renderer_python():
        import os as _env_os
        env_cfg = (config or {}).get("renderer_environment") or {}
        lock_path = (analysis_root / env_cfg.get("environment_lock", "")) if env_cfg.get("environment_lock") else None
        lock = json.loads(lock_path.read_text()) if lock_path and lock_path.is_file() else None
        if lock is None:
            raise guards.ReplayRefusal("renderer environment lock missing; refusing to guess an interpreter")
        want = lock.get("runtime") or {}
        candidates = []
        override = _env_os.environ.get("RFINAL_RENDER_PYTHON")
        if override:
            candidates.append(override)
        candidates.extend(env_cfg.get("interpreter_candidates") or [])
        for cand in candidates:
            if not cand:
                continue
            probe = _subprocess.run(
                [cand, "-c", "import json,sys,matplotlib,numpy;print(json.dumps([sys.version.split()[0],"
                             "matplotlib.__version__,numpy.__version__]))"], capture_output=True, text=True)
            if probe.returncode != 0:
                continue
            got = json.loads(probe.stdout)
            if (got[0] == want.get("python") and got[1] == want.get("matplotlib")
                    and got[2] == want.get("numpy")):
                return cand, {"python": got[0], "matplotlib": got[1], "numpy": got[2],
                              "executable": str(cand), "override": bool(override)}
        raise guards.ReplayRefusal("no configured interpreter matches the pinned renderer lock versions")

    render_python, render_runtime = _renderer_python()
    cmd = [render_python, str(analysis_root / "code/rfinal_replay/render/run_render.py"),
           "--input-root", str(out_root), "--outdir", str(render_out),
           "--contract", str(contract_path)]
    env = dict(_os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = _subprocess.run(cmd, cwd=str(analysis_root), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise guards.ReplayRefusal(
            f"stage {stage['id']}: render suite refused/failed exit={proc.returncode}: "
            + (proc.stderr or "")[-400:])
    inventory = json.loads((render_out / "OUTPUT_INVENTORY.json").read_text())
    bundle = {
        "schema": "c17-render-bundle-v1",
        "stage": stage["id"],
        "created_utc": manifest.utc_now(),
        "contract": {"path": str(contract_path), "sha256": digest, "contract_sha256": guards.canonical_sha256(contract)},
        "render_outdir": str(render_out),
        "result_count": inventory.get("result_count"),
        "result_ids": [item["id"] for item in inventory.get("items", [])],
        "artifacts": [{**art, "result_id": item["id"]}
                      for item in inventory.get("items", []) for art in item.get("artifacts", [])],
        "renderer_runtime": render_runtime,
        "substitutions_applied": applied,
        "substitutions_pinned_fallback": pinned,
        "schema_adapter_pending": pending_schema,
        "require_fresh": bool(stage.get("require_fresh")),
    }
    bundle_path = suite_dir / "RENDER_BUNDLE.json"
    digest = manifest.write_json_atomic(bundle_path, bundle)
    return {"path": str(bundle_path), "sha256": digest, "role": stage["outputs"][0]["role"],
            "extra_outputs": [{"path": str(render_out / "OUTPUT_INVENTORY.json"),
                               "sha256": guards.sha256_file(render_out / "OUTPUT_INVENTORY.json")}]}


def execute(
    *,
    config: dict,
    config_path: Path,
    plan: list[dict],
    project_root: Path,
    analysis_root: Path,
    out_root: Path,
    selected: list[str] | None,
    verify_inputs: str,
    empty_check: dict,
    mode: str = "replay",
    max_stages: int | None = None,
) -> dict:
    if mode == "replay":
        guards.assert_authorized(config)
        guards.assert_tolerances_frozen(config)
        guards.assert_pattern_exceptions_approved(config)
    elif mode != "engine-check":
        raise guards.ReplayRefusal(f"unknown execution mode: {mode!r}")
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = "rfinal-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    before = guards.snapshot_tree([analysis_root / name for name in ("runs", "code", "config", "reports", "verifier")])
    manifest.write_json_atomic(
        out_root / "RUN_STARTED.json",
        {
            "run_id": run_id,
            "config": str(config_path),
            "project_root": str(project_root),
            "analysis_root": str(analysis_root),
            "output_root": str(out_root),
            "selected_stages": selected,
            "verify_inputs": verify_inputs,
            "empty_output_assertion": empty_check,
            "mode": mode,
            "max_stages": max_stages,
        },
    )
    allow = guards.InputAllowlist(config, project_root, analysis_root)
    entries = {entry["id"]: entry for entry in config["input_whitelist"]["entries"]}
    enforced = plan
    if mode == "engine-check":
        check_plan = config.get("engineering_check_plan") or config["regeneration_plan"].get(
            "engineering_check_plan", {}
        )
        eligible = set(check_plan.get("eligible_stages", []))
        if not eligible:
            raise guards.ReplayRefusal(
                "engine-check mode requires a declared engineering_check_plan.eligible_stages list"
            )
        enforced = [row for row in plan if row["id"] in eligible]
    executable_plan = [row for row in enforced if row["executable"] and (selected is None or row["id"] in selected)]
    if selected is not None:
        unknown = set(selected) - {row["id"] for row in plan}
        if unknown:
            raise guards.ReplayRefusal(f"unknown stage ids: {sorted(unknown)}")
        if mode == "engine-check":
            refused = sorted(set(selected) - {row["id"] for row in enforced})
            if refused:
                raise guards.ReplayRefusal(
                    "engine-check mode refuses stages outside engineering_check_plan.eligible_stages: "
                    + ", ".join(refused)
                )
    if max_stages is not None:
        executable_plan = executable_plan[:max_stages]

    env_executable = None
    if any(row["kind"] == "external_producer" for row in executable_plan):
        from . import environment as environment_module

        recorded = config["environment_lock"].get("cpu_replay", {}).get("interpreter")
        if recorded:
            candidate = Path(recorded)
            if not candidate.is_absolute():
                candidate = Path(analysis_root) / candidate
        else:
            candidate = environment_module.cpu_replay_python(analysis_root)
        if not candidate.is_file():
            raise guards.ReplayRefusal(
                "external producers require the isolated CPU replay environment; "
                f"interpreter not found: {candidate} (refusing to run producers under the host environment)"
            )
        expected_pins = config["environment_lock"].get("cpu_replay", {}).get("pins") or {}
        probe = adapters.verify_isolated_environment(
            executable=candidate, expected_pins=expected_pins, analysis_root=analysis_root
        )
        if probe.get("status") != "PASS":
            raise guards.ReplayRefusal(
                "isolated CPU replay environment failed its import/version probe: "
                + json.dumps(probe.get("problems") or probe.get("error"))
            )
        env_executable = candidate
        manifest.write_json_atomic(
            out_root / "environment_probe.json",
            {"executable": str(candidate), "probe": probe},
        )

    fresh_registry: dict[str, list[dict]] = {}
    receipts: list[dict] = []
    outputs: list[dict] = []
    code_paths = [Path(row["consumer_code"]) for row in plan if row["consumer_code"]]
    code_paths.extend(sorted(Path(__file__).resolve().parent.glob("*.py")))
    for row in executable_plan:
        if row["kind"] == "input_audit":
            verified = allow.verify_many([item["id"] for item in row["inputs"]], force=True)
            receipt = {"stage": row["id"], "kind": row["kind"], "verified_inputs": verified}
            path = out_root / row["outputs"][0]["path"]
            receipt["sha256"] = manifest.write_json_atomic(path, {"schema": "rfinal-input-audit-v1", "verified": verified})
            fresh_registry.setdefault(row["id"], []).append({"path": str(path), "sha256": receipt["sha256"]})
            outputs.append({"stage": row["id"], "path": str(path), "sha256": receipt["sha256"], "role": row["outputs"][0]["role"]})
        elif row["kind"] == "frozen_consumer_sandbox":
            verified = allow.verify_many([item["id"] for item in row["inputs"]], force=True)
            if row.get("consumer_code_ref"):
                verified[row["consumer_code_ref"]] = allow.verify(entries[row["consumer_code_ref"]], force=True)
            run_dir = row["run_dir"]
            input_paths = [Path(item["path"]) for item in row["inputs"]]
            receipt = sandbox_runner.run_stage(
                stage={**row, "argv": row["argv"]},
                out_root=out_root,
                project_root=project_root,
                analysis_root=analysis_root,
                consumer_code=Path(row["consumer_code"]),
                input_paths=input_paths,
            )
            receipt["verified_inputs"] = verified
            produced = out_root / "sandbox" / row["id"] / "runs" / run_dir / "metrics.json"
            if not produced.is_file():
                raise guards.ReplayRefusal(f"stage {row['id']}: expected fresh artifact missing: {produced}")
            target = out_root / row["outputs"][0]["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = guards.sha256_file(produced)
            target.write_bytes(produced.read_bytes())
            if guards.sha256_file(target) != digest:
                raise guards.ReplayRefusal(f"stage {row['id']}: copied artifact hash mismatch")
            fresh_registry.setdefault(row["id"], []).append({"path": str(produced), "sha256": digest})
            outputs.append({"stage": row["id"], "path": str(target), "sha256": digest, "role": row["outputs"][0]["role"]})
            receipt["output"] = {"path": str(target), "sha256": digest}
        elif row["kind"] == "derived_table":
            stage_with_kind = {**row, "table_kind": row.get("table_kind")}
            produced = run_derived_table(stage_with_kind, out_root, fresh_registry, config=config,
                                         mode=mode,
                                         ctx={"out_root": out_root, "project_root": project_root,
                                              "analysis_root": analysis_root})
            fresh_registry.setdefault(row["id"], []).append({"path": produced["path"], "sha256": produced["sha256"]})
            for extra in produced.get("extra_outputs") or []:
                if extra:
                    fresh_registry.setdefault(row["id"], []).append(extra)
            outputs.append({"stage": row["id"], **produced})
            receipt = {"stage": row["id"], "kind": row["kind"], "output": produced}
        elif row["kind"] == "external_producer":
            receipt = adapters.run_external_stage(
                stage=row,
                out_root=out_root,
                project_root=project_root,
                analysis_root=analysis_root,
                allow=allow,
                fresh_registry=fresh_registry,
                entries=entries,
                env_executable=env_executable,
                access_log_dir=out_root / "logs",
            )
            for item in receipt["outputs"]:
                # single registration ownership: run_external_stage already registered the
                # canonical records in fresh_registry; do not append them a second time
                outputs.append(
                    {
                        "stage": row["id"],
                        "path": item["collected_path"],
                        "sha256": item["sha256"],
                        "role": item["role"],
                        "result_ids": item.get("result_ids", []),
                    }
                )
        else:
            raise guards.ReplayRefusal(f"stage {row['id']}: kind {row['kind']} is not executable yet")
        receipts.append(receipt)
        manifest.write_json_atomic(out_root / "receipts" / f"{row['id']}.json", receipt)

    after = guards.snapshot_tree([analysis_root / name for name in ("runs", "code", "config", "reports", "verifier")])
    snapshot_diff = guards.classify_snapshot_diff(before, after)
    out_prefix = str(Path(out_root).resolve()) + os.sep
    concurrent_additions = [
        row for row in snapshot_diff["added"] if not str(Path(row).resolve()).startswith(out_prefix)
    ]
    # Modifications/removals of paths this run declared as inputs are integrity
    # failures.  Other changes in a shared tree are recorded as concurrent
    # activity: the producers themselves cannot write there (audit hook) and the
    # engine writes only under the output root.
    declared_input_paths = {str(Path(item["path"]).resolve()) for rows in [row["inputs"] for row in plan] for item in rows}
    outside_changes = list(snapshot_diff["modified"]) + list(snapshot_diff["removed"])
    input_integrity = []
    concurrent_changes = []
    for row in outside_changes:
        head = row.split(": ", 1)[0]
        if head in declared_input_paths:
            input_integrity.append(row)
        else:
            concurrent_changes.append(row)
    if input_integrity:
        raise guards.ReplayRefusal(
            "declared-input integrity failure while running: " + "; ".join(input_integrity)
        )
    stage_rows = [
        {
            "id": row["id"],
            "status": "COMPLETE" if any(item["stage"] == row["id"] for item in outputs) else "SKIPPED",
            "kind": row["kind"],
            "blocked_reason": row["blocked_reason"],
            "coverage_expectation": row.get("coverage_expectation"),
            "inputs": [{"id": item["id"], "path": item["path"], "expected_sha256": item["expected_sha256"]} for item in row["inputs"]],
        }
        for row in plan
        if (selected is None or row["id"] in selected)
        and (mode != "engine-check" or row["id"] in eligible)
    ]
    all_stages_complete = all(
        any(item["stage"] == row["id"] for item in outputs) for row in plan if row["executable"]
    )
    if mode == "engine-check":
        status = "ENGINE_CHECK_PARTIAL_NOT_REPLAY"
    elif all_stages_complete and all(row["executable"] for row in plan):
        status = "COMPLETE"
    else:
        status = "COMPLETE_WITH_PENDING_STAGES"
    inputs_manifest = [
        {
            "id": entry["id"],
            "level": entry["level"],
            "role": entry.get("role", ""),
            "path": str(allow.resolve(entry)),
            "declared_sha256": entry["sha256"],
            "verified_sha256": allow.verified.get(entry["id"]),
            "verified": entry["id"] in allow.verified,
        }
        for entry in entries.values()
        if entry["id"] in allow.verified
    ]
    final_manifest = manifest.build_manifest(
        run_id=run_id,
        mode="execute",
        project_root=project_root,
        analysis_root=analysis_root,
        config_path=config_path,
        config=config,
        output_root=out_root,
        empty_output_check=empty_check,
        inputs=inputs_manifest,
        code=manifest.code_hashes(sorted(set(code_paths))),
        environment=manifest.environment_snapshot(
            (config["environment_lock"].get("primary") or {}).get("executable")
            or config["environment_lock"].get("cpu_replay", {}).get("interpreter")
            or sys.executable
        ),
        stages=stage_rows,
        outputs=outputs,
        status=status,
        extra_guards={
            "signed_tree_snapshot_unchanged": not outside_changes,
            "signed_tree_modified_or_removed": outside_changes,
            "declared_input_integrity_failures": input_integrity,
            "concurrent_modifications_observed": concurrent_changes[:50],
            "concurrent_modifications_count": len(concurrent_changes),
            "concurrent_additions_observed": concurrent_additions[:50],
            "concurrent_additions_count": len(concurrent_additions),
            "receipts": len(receipts),
        },
    )
    final_manifest["execution_mode"] = mode
    final_manifest["engine_check_scope"] = {
        "eligible_stages": sorted(eligible) if mode == "engine-check" else None,
        "note": (
            "engine-check runs exercise declared adapters on legal lightweight paths and refusal "
            "behaviour; they never claim the frozen R6 replay or the downstream scientific nodes"
        ),
    }
    manifest.write_json_atomic(out_root / "run_manifest.json", final_manifest)
    return final_manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        from importlib import import_module

        selftest = import_module(f"{__package__}.selftest" if __package__ else "rfinal_replay.selftest")
        return selftest.main()
    if not args.project_root or not args.analysis_root:
        raise guards.ReplayRefusal("--project-root and --analysis-root are required (explicit roots only)")
    project_root = guards.resolve_root(args.project_root, "project-root")
    analysis_root = guards.resolve_root(args.analysis_root, "analysis-root")
    config_path = Path(args.config).expanduser().resolve() if args.config else default_config_path(analysis_root)
    config = guards.load_config(config_path)
    out_root = resolve_output_root(args.out, analysis_root, config)
    plan = stages.build_plan(config, project_root, analysis_root, out_root)
    verify_inputs = args.verify_inputs or ("lazy" if args.execute else "none")

    if args.print_config_summary:
        summary = {
            "config": str(config_path),
            "status": config["status"],
            "whitelist_counts_by_level": guards.InputAllowlist(config, project_root, analysis_root).counts_by_level(),
            "plan": stages.plan_summary(plan),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.execute and args.dry_run:
        raise guards.ReplayRefusal("choose either --dry-run or --execute")
    if not args.execute and not args.dry_run:
        args.dry_run = True

    if args.dry_run:
        allow = guards.InputAllowlist(config, project_root, analysis_root)
        verification: dict[str, str] = {}
        skipped_large: list[dict] = []
        if verify_inputs == "full":
            for entry in allow.entries.values():
                size = entry.get("bytes") or (allow.resolve(entry).stat().st_size if allow.resolve(entry).is_file() else 0)
                if entry.get("hash_source") and size > guards.MAX_INLINE_HASH_BYTES and not args.allow_large_hash:
                    skipped_large.append({"id": entry["id"], "bytes": size, "hash_source": entry["hash_source"]})
                    continue
                verification[entry["id"]] = allow.verify(entry, force=True, allow_large=args.allow_large_hash)
        elif verify_inputs == "lazy":
            for row in plan:
                if row["executable"]:
                    verification.update(
                        allow.verify_many([item["id"] for item in row["inputs"]], force=False, allow_large=args.allow_large_hash)
                    )
        empty_check = guards.assert_empty_output_dir(out_root)
        if args.json:
            payload = {
                "mode": "dry-run",
                "project_root": str(project_root),
                "analysis_root": str(analysis_root),
                "config": str(config_path),
                "output_root": str(out_root),
                "output_check": empty_check,
                "verify_inputs": verify_inputs,
                "verified_inputs": verification,
                "skipped_large_manifest_digest": skipped_large,
                "status": compute_status(config, plan),
                "plan": plan,
                "note": "dry run wrote nothing; no directory was created",
            }
            print(json.dumps(payload, indent=2))
        else:
            print(
                stages.render_plan_text(
                    plan,
                    project_root=project_root,
                    analysis_root=analysis_root,
                    out_root=out_root,
                    config_path=config_path,
                    verify_inputs=verify_inputs,
                )
            )
            print(json.dumps({"output_check": empty_check, "status": compute_status(config, plan)}, indent=2))
        if out_root.exists():
            if not out_root.is_dir() or any(out_root.iterdir()):
                raise guards.ReplayRefusal(f"dry run changed the target directory: {out_root}")
        return 0

    empty_check = guards.assert_empty_output_dir(out_root)
    final_manifest = execute(
        config=config,
        config_path=config_path,
        plan=plan,
        project_root=project_root,
        analysis_root=analysis_root,
        out_root=out_root,
        selected=args.stage,
        verify_inputs=verify_inputs,
        empty_check=empty_check,
        mode=args.mode,
        max_stages=args.max_stages,
    )
    if args.require_complete:
        incomplete = [
            row["id"]
            for row in final_manifest["stages"]
            if row["status"] != "COMPLETE" and row["blocked_reason"] is None
        ]
        skipped = [row["id"] for row in final_manifest["stages"] if row["status"] == "SKIPPED"]
        if final_manifest["status"] == "COMPLETE_WITH_PENDING_STAGES" and not skipped:
            raise guards.ReplayRefusal(
                "--require-complete: execution ended with pending stages: "
                + ", ".join(row["id"] for row in plan if not row["executable"])
            )
        if incomplete:
            raise guards.ReplayRefusal(f"--require-complete: stages without outputs: {incomplete}")
    print(json.dumps({"status": final_manifest["status"], "output_root": final_manifest["output_root"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except guards.ReplayRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        raise SystemExit(2)
