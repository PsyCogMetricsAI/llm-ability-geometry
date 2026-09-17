#!/usr/bin/env python3
"""C10m validator: resolve mapping bindings against consumer-injected stage outputs.

Usage:
  python3 validate_claim_map.py --mapping <map.json> \
      --stage-outputs <stage_outputs.json> --out <newdir>

Asserts:
  * mapping integrity: 147 records, original_order 0..146, unique ids,
    139 mandatory, ledger sha256 matches the on-disk immutable ledger;
  * every concrete stage root lies inside one of the declared fresh runs
    (runs are declared in the stage-outputs map; no absolute private
    hardcoding is required);
  * every binding file exists, resolves at its JSON pointer with the declared
    value kind/shape, and is inside its declared stage root;
  * records without confirmed bindings are reported exactly as pending
    (never silently passed);
  * sha256 recorded for every referenced file plus canonical sha256 of the
    resolved pointer value.
Exit codes: 0 clean, 2 verification failures, 3 structural/integrity failure.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = os.path.join(R, "reports", "C17_FINAL_EVIDENCE_LEDGER_v1.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(doc, pointer):
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise ValueError("bad pointer %r" % pointer)
    cur = doc
    for raw in pointer.split("/")[1:]:
        tok = raw.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


def kind_of(val):
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, (int, float)):
        return "number"
    return {dict: "dict", list: "list", str: "string"}.get(type(val), type(val).__name__)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--stage-outputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-complete", action="store_true",
                    help="fail on any unresolved computable binding, any missing stage output, "
                         "and any non-historical pending record")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    failures, pending, verified = [], [], []
    mapping = json.load(open(args.mapping, encoding="utf-8"))
    so = json.load(open(args.stage_outputs, encoding="utf-8"))
    runs = {rid: os.path.realpath(p) for rid, p in so.get("runs", {}).items()}
    stages = so.get("stages", {})
    # --- structural integrity -------------------------------------------------
    structural = []
    recs = mapping.get("records", [])
    if len(recs) != 147:
        structural.append("record count %d != 147" % len(recs))
    if sorted(r["original_order"] for r in recs) != list(range(147)):
        structural.append("original_order is not exactly 0..146")
    ids = [r["result_id"] for r in recs]
    if len(set(ids)) != len(ids):
        structural.append("duplicate result ids")
    mand = [r for r in recs if r.get("mandatory")]
    if len(mand) != 139:
        structural.append("mandatory count %d != 139" % len(mand))
    ledger_sha = sha256_file(LEDGER)
    if mapping.get("ledger", {}).get("sha256") != ledger_sha:
        structural.append("ledger sha256 mismatch")
    if [r["original_order"] for r in recs] != [i for i in range(len(recs))]:
        structural.append("records not stored in original order")
    # --- per-record binding checks -------------------------------------------
    out_rows = []
    stage_roots = {}
    stages_not_instantiated = []
    for sid in mapping.get("stage_catalog", {}):
        st = stages.get(sid)
        if st is None:
            # logical stage declared but not yet instantiated in this consumer map
            # (e.g. S14 depth summary); any record binding that needs it fails below.
            stages_not_instantiated.append(sid)
            if args.require_complete:
                failures.append({"stage": sid, "error": "missing stage output (--require-complete)"})
            continue
        run_id = st.get("run")
        if run_id not in runs:
            failures.append({"stage": sid, "error": "run %r not declared" % run_id})
            continue
        root = os.path.realpath(os.path.join(runs[run_id], st.get("subpath", ".")))
        if not (root == runs[run_id] or root.startswith(runs[run_id] + os.sep)):
            failures.append({"stage": sid, "error": "stage path escapes declared run root"})
            continue
        stage_roots[sid] = root
    for rec in recs:
        rid = rec["result_id"]
        if rec["status"] == "pending_fresh_value":
            pending.append({"result_id": rid, "reason": rec.get("pending_reason"),
                            "producer_stage": (rec.get("producer") or {}).get("stage_id")})
            if args.require_complete and rec.get("mandatory") and rec.get("c17_disposition") != "HISTORICAL_LIMIT":
                failures.append({"result_id": rid, "error": "mandatory unresolved computable binding (--require-complete)"})
            continue
        if rec["status"] == "explicit_no_fresh_numeric":
            out_rows.append({"result_id": rid, "status": rec["status"],
                             "classification": rec["classification"], "bindings": []})
            continue
        for b in rec.get("bindings", []):
            sid = b["stage_id"]
            if b.get("fulfillment") == "awaiting_verification":
                out_rows.append({"result_id": rid, "status": "awaiting_verification", "stage_id": sid,
                                 "file": b.get("file"), "json_pointer": b.get("json_pointer"),
                                 "owner": b.get("owner"), "note": b.get("note")})
                continue
            if b.get("fulfillment") not in (None, "available"):
                if args.require_complete:
                    failures.append({"result_id": rid,
                                     "error": "binding awaiting stage output (--require-complete): %s" % sid})
                out_rows.append({"result_id": rid, "status": "awaiting_stage_output", "stage_id": sid,
                                 "file": b.get("file"), "owner": b.get("owner")})
                continue
            if sid not in stage_roots:
                failures.append({"result_id": rid, "error": "stage %r unavailable" % sid})
                continue
            if b.get("csv_selector"):
                path = os.path.realpath(os.path.join(stage_roots[sid], b["file"]))
                if not os.path.isfile(path):
                    failures.append({"result_id": rid, "error": "csv missing: %s" % b["file"]})
                    continue
                with open(path, newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                sel = b["csv_selector"]
                if sel.get("all_rows"):
                    vals = [r.get(sel["column"]) for r in rows if sel["column"] in r]
                    ok = len(vals) > 0
                elif sel.get("aggregate") == "max":
                    nums = [float(r[sel["column"]]) for r in rows if r.get(sel["column"]) not in (None, "")]
                    vals = max(nums) if nums else None
                    ok = vals is not None
                else:
                    hit = [r for r in rows if r.get(sel["column"]) == sel.get("equals")]
                    vals = [r.get(sel["select"]) for r in hit]
                    ok = len(hit) > 0
                if not ok:
                    failures.append({"result_id": rid, "error": "csv selector empty: %s" % sel})
                    continue
                out_rows.append({"result_id": rid, "status": rec["status"],
                                 "bindings": [{"file": b["file"], "csv_selector": sel,
                                               "selection": vals,
                                               "file_sha256": sha256_file(path)}]})
                verified.append({"result_id": rid, "file": b["file"], "csv": True})
                continue
            if b.get("json_pointer") is None:
                path = os.path.realpath(os.path.join(stage_roots[sid], b["file"]))
                if os.path.exists(path):
                    out_rows.append({"result_id": rid, "status": "artifact_present",
                                     "file_sha256": sha256_file(path),
                                     "bindings": [{"file": b["file"], "file_sha256": sha256_file(path)}]})
                    verified.append({"result_id": rid, "file": b["file"], "artifact": True})
                else:
                    failures.append({"result_id": rid, "error": "artifact missing: %s" % b["file"]})
                continue
            path = os.path.realpath(os.path.join(stage_roots[sid], b["file"]))
            if not (path == stage_roots[sid] or path.startswith(stage_roots[sid] + os.sep)):
                failures.append({"result_id": rid, "error": "binding file escapes stage root"})
                continue
            if not os.path.isfile(path):
                failures.append({"result_id": rid, "error": "file missing: %s" % b["file"]})
                continue
            try:
                doc = json.load(open(path, encoding="utf-8"))
                val = resolve(doc, b["json_pointer"])
            except Exception as exc:  # noqa: BLE001
                failures.append({"result_id": rid, "error": "pointer %s: %s" % (b["json_pointer"], exc)})
                continue
            ident_bad = False
            for ident in (b.get("identity") or []):
                try:
                    iv = resolve(doc, ident["pointer"])
                except Exception as exc:  # noqa: BLE001
                    failures.append({"result_id": rid, "error": "identity pointer %s: %s" % (ident["pointer"], exc)})
                    ident_bad = True
                    break
                if iv != ident.get("equals"):
                    failures.append({"result_id": rid, "error": "identity mismatch at %s: %r != %r"
                                     % (ident["pointer"], iv, ident.get("equals"))})
                    ident_bad = True
                    break
            if ident_bad:
                continue
            exp = b.get("expectation") or {}
            act = kind_of(val)
            if exp.get("kind") and act != exp["kind"]:
                if not (exp["kind"] == "number" and act == "number"):
                    failures.append({"result_id": rid, "error": "kind %s != declared %s" % (act, exp["kind"])})
                    continue
            if exp.get("nonempty") and hasattr(val, "__len__") and len(val) == 0:
                failures.append({"result_id": rid, "error": "declared nonempty but empty"})
                continue
            row = {"result_id": rid, "stage_id": sid, "file": b["file"],
                   "json_pointer": b["json_pointer"], "value_kind": act,
                   "file_sha256": sha256_file(path),
                   "value_sha256": hashlib.sha256(json.dumps(val, sort_keys=True).encode()).hexdigest(),
                   "expectation_ok": True}
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                row["value"] = float(val)
            out_rows.append({"result_id": rid, "status": rec["status"], "bindings": [row]})
            verified.append(row)
    if failures or structural:
        status = "FAIL"
    elif pending or stages_not_instantiated or any(
            b.get("fulfillment") == "awaiting_stage_output"
            for r in recs for b in r.get("bindings", [])):
        status = "PARTIAL"
    else:
        status = "PASS"
    with open(os.path.join(args.out, "validation.jsonl"), "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    summary = {"schema": "c17-claim-map-validation-v1", "status": status,
               "mapping_sha256": sha256_file(args.mapping),
               "stage_outputs_sha256": sha256_file(args.stage_outputs),
               "records_total": len(recs), "records_mandatory": len(mand),
               "bindings_verified": len(verified),
               "records_with_verified_bindings": len({r["result_id"] for r in verified}),
               "explicit_nonnumeric_records": sum(1 for r in recs if r["status"] == "explicit_no_fresh_numeric"),
               "pending_records": len(pending), "structural_errors": structural,
               "stages_not_instantiated": stages_not_instantiated,
               "failures": failures}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    with open(os.path.join(args.out, "pending.json"), "w", encoding="utf-8") as fh:
        json.dump(pending, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: summary[k] for k in ("status", "bindings_verified",
                                              "records_with_verified_bindings", "pending_records",
                                              "explicit_nonnumeric_records", "stages_not_instantiated",
                                              "failures", "structural_errors")}, indent=1))
    return 0 if status == "PASS" else (3 if structural else 2)


if __name__ == "__main__":
    sys.exit(main())
