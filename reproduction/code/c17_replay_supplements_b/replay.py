#!/usr/bin/env python3
"""C17 retained existing-data replay, supplements B (C10v).

Fresh cached-data recomputation, comparison-only against retained terminals:

  pooling : P2-F-POOLING.  Re-reads the 48 frozen per-model code JSONL
            observation arrays (item_hidden / mean_logprob / accuracy) and
            re-runs the direct double-centred interaction-residual ANOVA
            (raw + standardized) exactly as the saved observations permit.
            Compares with runs/extension_reuse_v1/pooling_result.json,
            verifier/V_NUM_POOLING_REUSE.json and the ledger claim pointer
            Imports/theory_rescue_20260913/checks/results.json.
  g0      : P2-F-G0-PANEL / P2-F-G0-STABILITY.  Replays the panel ALS fit
            (deterministic, CPU, 170x48 panel) from the same saved JSONL
            cells; recomputes coverage, convergence trace, gauge diagnostics,
            sanity correlation; re-derives LOO correlations and s^2 shares
            from the saved b_loo.npz / theta_s_sigma_panel.csv observations.
  g1      : P2-F-G1-DIFFICULTY / P2-F-G1-TRANSFER.  Re-aggregates the saved
            per-model readouts and re-runs the stored pair bootstrap from the
            saved 240 pair rho values.  No probe is refit.
  g2      : P2-F-G2-ASSOCIATION / -SENSITIVITY / -IDENTITY.  Re-aggregates
            the saved per-model beta table (Spearman, partial Spearman,
            delete-one-family jackknife, Seed-OSS sensitivity, sanity OLS).
  g3      : P2-F-G3-RESIDUAL / P2-F-G3-CORRECTNESS.  Re-aggregates saved
            per-model residual outputs and re-derives cross-check statistics
            from the saved eps_loo/eps_panel/perm_null arrays.
  prior16 : P2-G-002 / P2-G-003.  Independent re-implementation of the
            recovered 16-model LID/accuracy point statistics from the four
            recovered aggregate/per-item inputs (no candidate-script import).

Every input is pinned in the expected manifest (default:
code/c17_replay_supplements_b/expected_manifest_v1.json); a missing file or a
hash mismatch refuses before any computation.  Comparator: scalars use
abs <= 1e-9 + 1e-6*abs(ref); counts, IDs, strings and null masks are exact.
Terminal JSON summaries are comparison references only; nothing here is
written into signed run/source locations.

Usage (pinned):
  <venv-cpu-replay>/bin/python code/c17_replay_supplements_b/replay.py \
      --analysis-root <R> --out <EMPTY_NEW_DIR> [--expected-manifest <file>]

Integration isolation (default): every write stays under --out.  The reports in
<R>/reports are only written with the explicit opt-in flag --publish-reports
(never with --no-report-files).  Root/finalintegration reruns should use
--no-report-files.

The replay never refits a probe, never re-extracts hidden states, never
touches GPU/BLAS threads, and does not claim historical restoration.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SCHEMA = "c17-replay-supplements-b-v1"
SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = SCRIPT_PATH.parent
KNOWN_ORIGINAL_PROJECT_ROOT = SCRIPT_PATH.parents[2].parent
KNOWN_ORIGINAL_ANALYSIS_ROOT = KNOWN_ORIGINAL_PROJECT_ROOT / "repro_paper2_20260915"
DEFAULT_EXPECTED_MANIFEST = CODE_DIR / "expected_manifest_v1.json"
SCALAR_ATOL = 1e-9
SCALAR_RTOL = 1e-6

REPORT_JSON = "reports/C17_SUPPLEMENTS_B_v1.json"
REPORT_MD = "reports/C17_SUPPLEMENTS_B_v1.md"


class ReplayRefusal(RuntimeError):
    exit_code = 2


class HashRefusal(ReplayRefusal):
    exit_code = 3


class MissingRefusal(ReplayRefusal):
    exit_code = 4


class ContractRefusal(ReplayRefusal):
    exit_code = 5


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n",
                          encoding="utf-8")


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def spearman(a, b):
    return pearson(rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float)))


def corr_dot(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def close_enough(actual, ref):
    if isinstance(ref, bool) or isinstance(actual, bool):
        return bool(actual) == bool(ref)
    if isinstance(ref, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(ref) and math.isnan(actual):
            return True
        if not (math.isfinite(ref) and math.isfinite(actual)):
            return False
        return abs(actual - ref) <= SCALAR_ATOL + SCALAR_RTOL * abs(ref)
    return actual == ref


def compare_numeric_leaves(actual, ref):
    if isinstance(ref, dict):
        if not isinstance(actual, dict) or set(actual) != set(ref):
            return False
        return all(compare_numeric_leaves(actual[k], ref[k]) for k in ref)
    if isinstance(ref, list):
        return (isinstance(actual, list) and len(actual) == len(ref)
                and all(compare_numeric_leaves(a, b) for a, b in zip(actual, ref)))
    return close_enough(actual, ref)


# ------------------------------------------------------------------ context
class Context:
    def __init__(self, analysis_root, out_dir, expected_manifest):
        self.analysis_root = Path(analysis_root).resolve()
        self.project_root = self.analysis_root.parent
        self.out_dir = Path(out_dir).resolve()
        self.expected_path = Path(expected_manifest).resolve()
        self.started = time.time()
        self.utc = datetime.now(timezone.utc).isoformat()
        self.consumed = {}
        self.claims = []
        self.uncomputable = []
        self.errors = []
        self.counts = {}
        self.timings = {}
        self.remap_log = []
        self.expected = {}
        self.preflight = None

    def is_under_roots(self, path):
        path = Path(path).resolve()
        return (path == self.analysis_root or path.is_relative_to(self.analysis_root)
                or path == self.project_root or path.is_relative_to(self.project_root))

    def remap_recorded(self, recorded):
        """Map a recorded (possibly original-absolute) path onto the caller
        roots by RELATIVE SUFFIX against the known original roots, so a
        renamed project / renamed analysis dir works while the original tree
        still exists elsewhere."""
        p = Path(recorded)
        cands = []

        def add(c):
            c = Path(c)
            if c not in cands:
                cands.append(c)

        if p.is_absolute():
            s = str(p)
            parts = p.parts
            for marker in ("repro_paper2_20260915", "Imports", "b200_full_pipeline", "_gpu2_run", "ext_P2_geo_20260914"):
                if marker in parts:
                    idx = parts.index(marker)
                    if marker == "repro_paper2_20260915":
                        add(self.analysis_root.joinpath(*parts[idx + 1:]))
                    else:
                        target = self.project_root
                        add(target.joinpath(*parts[idx:]))
            marker = "/" + self.project_root.name + "/"
            if marker in s:
                add(self.project_root / s[s.find(marker) + len(marker):])
        elif str(p).startswith("R/"):
            add(self.analysis_root / str(p)[2:])
        elif str(p).startswith("P/"):
            add(self.project_root / str(p)[2:])
        else:
            add(self.analysis_root / p)
            add(self.project_root / p)
        return cands

    def resolve_recorded(self, recorded, role=""):
        for cand in self.remap_recorded(recorded):
            if not cand.exists():
                continue
            if not self.is_under_roots(cand):
                continue
            resolved = cand.resolve()
            if str(Path(recorded)) != str(resolved):
                self.remap_log.append({"recorded": str(recorded), "resolved": str(resolved),
                                       "role": role})
            return resolved
        raise MissingRefusal("recorded path not resolvable under caller roots: %s (%s)"
                             % (recorded, role))

    def consume(self, path, role, level, expected_sha256=None):
        path = Path(path).resolve()
        if path in self.consumed:
            return self.consumed[path]["sha256"]
        if not path.exists():
            raise MissingRefusal("input missing (%s): %s" % (role, path))
        actual = sha256_file(path)
        pinned = self.expected.get(path)
        want = expected_sha256 if expected_sha256 is not None else pinned
        if want is not None and actual != want:
            raise HashRefusal("hash mismatch (%s): %s expected %s actual %s"
                              % (role, path, want, actual))
        entry = {"path": str(path), "role": role, "level": level, "sha256": actual}
        self.consumed[path] = entry
        return actual

    # ---------------------------------------------------------- claims
    def claim(self, claim_id, result_id, ledger_pointer, field, fresh, ref,
              ref_source, method, scope_level, exact=False, limitations=None,
              variant_of=None, numeric_leaves=False, status_override=None):
        if status_override is not None:
            status = status_override
        elif exact:
            status = "PASS" if fresh == ref else "FAIL"
        elif numeric_leaves:
            status = "PASS" if compare_numeric_leaves(fresh, ref) else "FAIL"
        else:
            status = "PASS" if close_enough(fresh, ref) else "FAIL"
        tol = None if exact else SCALAR_ATOL + SCALAR_RTOL * abs(ref) \
            if isinstance(ref, (int, float)) and not isinstance(ref, bool) else None
        rec = {"claim_id": claim_id, "result_id": result_id,
               "ledger_pointer": ledger_pointer, "field": field,
               "fresh_value": fresh, "reference_value": ref,
               "reference_source": ref_source, "status": status,
               "tolerance": tol, "exact": exact, "method": method,
               "scope_level": scope_level,
               "limitations": limitations or []}
        if variant_of:
            rec["variant_of"] = variant_of
        self.claims.append(rec)
        return rec

    def uncomputable_field(self, claim_id, result_id, ledger_pointer, field,
                           reason, retained_reference=None):
        self.uncomputable.append({"claim_id": claim_id, "result_id": result_id,
                                  "ledger_pointer": ledger_pointer, "field": field,
                                  "status": "NOT_RECOMPUTED", "reason": reason,
                                  "retained_terminal_reference": retained_reference,
                                  "replacement_claim": False})


# ------------------------------------------------------- expected manifest
def verify_expected_manifest(ctx):
    doc = read_json(ctx.expected_path)
    entries = doc["files"] if isinstance(doc, dict) and "files" in doc else doc
    resolved = {}
    for entry in entries:
        root = entry.get("root", "P")
        base = ctx.analysis_root if root == "R" else ctx.project_root
        target = (base / entry["relativepath"]).resolve()
        if not ctx.is_under_roots(target):
            raise ContractRefusal("expected-manifest entry escapes caller roots: %s" % target)
        resolved[target] = entry
    rows, missing, mismatched = [], [], []
    for path, entry in sorted(resolved.items(), key=lambda kv: str(kv[0])):
        if not path.exists():
            missing.append(str(path))
            rows.append({"path": str(path), "status": "MISSING",
                         "expected": entry.get("sha256")})
            continue
        actual = sha256_file(path)
        ok = entry.get("sha256") in (None, actual)
        rows.append({"path": str(path), "status": "OK" if ok else "HASH_MISMATCH",
                     "expected": entry.get("sha256"), "actual": actual})
        if not ok:
            mismatched.append({"path": str(path), "expected": entry.get("sha256"),
                               "actual": actual})
    ctx.expected = {p: e.get("sha256") for p, e in resolved.items() if e.get("sha256")}
    # non-circular code pins: the manifest (an external artifact) pins the driver
    # code that is executing; the driver verifies its own bytes against the pin.
    ctx.code_pins = []
    for path, entry in resolved.items():
        if entry.get("level") != "code_pin":
            continue
        actual = sha256_file(path) if path.exists() else None
        ok = actual is not None and actual == entry.get("sha256")
        ctx.code_pins.append({"path": str(path), "expected": entry.get("sha256"),
                              "actual": actual, "match": ok})
    ctx.preflight = {"manifest": str(ctx.expected_path),
                     "manifest_sha256": sha256_file(ctx.expected_path),
                     "n_entries": len(resolved), "rows": rows,
                     "missing": missing, "mismatched": mismatched, "all_ok": True}
    if missing:
        raise MissingRefusal("expected manifest lists missing files: %s"
                             % ", ".join(missing[:5]))
    if mismatched:
        raise HashRefusal("expected manifest hash mismatch: %s"
                          % ", ".join(m["path"] for m in mismatched[:5]))
    bad_pins = [r for r in ctx.code_pins if not r["match"]]
    if bad_pins:
        raise HashRefusal("driver code pin mismatch: %s"
                          % ", ".join(r["path"] for r in bad_pins))


# ------------------------------------------------------------------ pooling
def load_pooling_sources(ctx):
    cfg_path = ctx.analysis_root / "config/REUSE_BINDINGS_EXTENSION.json"
    ctx.consume(cfg_path, "pooling_binding_config", "computation_input")
    cfg = read_json(cfg_path)
    n_rows = 48
    ctx.counts["pooling_jsonl_files_expected"] = n_rows
    return cfg


def read_model_jsonl(ctx, path, role, need_hidden):
    """Stream one frozen per-model code JSONL; returns dict of per-item rows."""
    ctx.consume(path, role, "computation_input")
    hidden_re = re.compile(rb'"(item_hidden|geom)"\s*:\s*\[[^\]]*\]\s*,?')
    out = {"item_id": [], "mean_logprob": [], "accuracy": [], "hidden": {}}
    with open(path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            if need_hidden:
                row = json.loads(raw)
                hid = row.get("item_hidden")
                if hid is None:
                    raise ContractRefusal("item_hidden missing in %s" % path)
            else:
                row = json.loads(hidden_re.sub(b"", raw, count=1))
                hid = None
            iid = int(row["item_id"])
            out["item_id"].append(iid)
            out["mean_logprob"].append(float(row["mean_logprob"]))
            out["accuracy"].append(float(row["accuracy"]))
            if need_hidden:
                out["hidden"][iid] = np.asarray(hid, dtype=np.float64)
    if len(set(out["item_id"])) != len(out["item_id"]):
        raise ContractRefusal("duplicate item_id in %s" % path)
    return out


def branch_pooling(ctx):
    t0 = time.time()
    cfg = load_pooling_sources(ctx)
    assets = cfg["asset_bindings"]
    tags, rows = [], {}
    for asset in assets:
        tag = asset["model"]
        path = ctx.resolve_recorded(asset["path"], role="pooling_jsonl")
        rec = read_model_jsonl(ctx, path, "pooling_jsonl:%s" % tag, need_hidden=True)
        n_items = len(rec["item_id"])
        dim = next(iter(rec["hidden"].values())).shape[0]
        if (n_items, dim) != (asset["historical_distinct_item_count"],
                              asset["historical_hidden_width"]):
            raise ContractRefusal("jsonl shape mismatch for %s: %s" % (tag, (n_items, dim)))
        tags.append(tag)
        rows[tag] = rec

    # producer: all 48 in source order; full = exactly 170 items
    full = [t for t in tags if len(rows[t]["item_id"]) == 170]
    shared = None
    for t in full:
        s = set(rows[t]["item_id"])
        shared = s if shared is None else (shared & s)
    shared = sorted(shared, key=lambda x: (len(str(x)), str(x)))
    if len(shared) != 170:
        raise ContractRefusal("shared item intersection is not 170: %d" % len(shared))
    if shared != list(range(170)):
        raise ContractRefusal("shared item order is not 0..169")

    widths = {}
    for t in full:
        widths.setdefault(next(iter(rows[t]["hidden"].values())).shape[0], []).append(t)
    groups = {str(w): sorted(tl) for w, tl in sorted(widths.items()) if len(tl) >= 2}

    def anova(matrices):
        M, I, D = len(matrices), matrices[0].shape[0], matrices[0].shape[1]
        model_means = [v.mean(axis=0) for v in matrices]
        grand = np.add.reduce(model_means) / M
        item_mean = np.add.reduce(matrices) / M
        total = math.fsum(float(np.sum((v - grand) ** 2)) for v in matrices)
        model = I * math.fsum(float(np.sum((mm - grand) ** 2)) for mm in model_means)
        item = M * float(np.sum((item_mean - grand) ** 2))
        residual = math.fsum(float(np.sum((v - mm - item_mean + grand) ** 2))
                             for v, mm in zip(matrices, model_means))
        return {"M": M, "I": I, "D": D,
                "SS_total_trace": total, "SS_model_trace": model,
                "SS_item_trace": item, "SS_resid_trace": residual,
                "model_share": model / total, "item_share": item / total,
                "resid_share": residual / total}

    raw_groups, std_groups, receipts = {}, {}, {}
    for width, tl in groups.items():
        arrays = [np.stack([rows[t]["hidden"][i] for i in shared]) for t in tl]
        raw_groups[width] = dict(anova(arrays), models=tl)
        std = []
        for t, v in zip(tl, arrays):
            dev = v - v.mean(0)
            sd = v.std(0, ddof=0)
            n_repl = int((sd < 1e-12).sum())
            sd = sd.copy()
            sd[sd < 1e-12] = 1
            z = dev / sd
            std.append(z)
            receipts[t] = {
                "n_replaced_small_sd": n_repl,
                "max_abs_dimension_mean_after": float(np.abs(z.mean(0)).max()),
                "population_sd_min_before_replacement": float(
                    v.std(0, ddof=0).min()),
            }
        std_groups[width] = anova(std)

    def pool(groups_map):
        keys = ["SS_model_trace", "SS_item_trace", "SS_resid_trace", "SS_total_trace"]
        out = {k: math.fsum(v[k] for v in groups_map.values()) for k in keys}
        for term in ("model", "item", "resid"):
            out[term + "_share"] = out["SS_" + term + "_trace"] / out["SS_total_trace"]
        return out

    precheck = {}
    for t in tags:
        mat = np.stack([rows[t]["hidden"][i] for i in sorted(rows[t]["hidden"])])
        precheck[t] = {"n_items": int(mat.shape[0]), "dim": int(mat.shape[1]),
                       "grand_mean": float(mat.mean()),
                       "mean_abs_perdim_mean": float(np.abs(mat.mean(0)).mean()),
                       "mean_perdim_var": float(mat.var(0).mean())}
    var = [v["mean_perdim_var"] for v in precheck.values()]
    is_std = max(var) < 1.5 and min(var) > .5 and max(
        v["mean_abs_perdim_mean"] for v in precheck.values()) < .1
    pooled_raw = pool(raw_groups)
    pooled_std = pool(std_groups)
    verdict = "definition-only" if is_std else ("≈0" if pooled_raw["model_share"] < .02
                                                else "nonzero")
    incomplete = [t for t in tags if len(rows[t]["item_id"]) != 170]
    contributing = [t for t in tags if t in {m for tl in groups.values() for m in tl}]
    fresh = {
        "precheck_is_standardized_empirical": bool(is_std),
        "precheck_perdim_var_range": [min(var), max(var)],
        "precheck_per_model": precheck,
        "n_code_jsonl_files": len(tags),
        "n_models_full_170": len(full),
        "n_shared_items": len(shared),
        "dim_heterogeneity": {str(w): len(tl) for w, tl in sorted(widths.items())},
        "usable_same_dim_groups": groups,
        "anova_raw_per_group": raw_groups,
        "anova_standardized_per_group": std_groups,
        "pooled_raw": pooled_raw,
        "pooled_standardized": pooled_std,
        "verdict": verdict,
        "panel_accounting": {
            "all_48_models_in_source_order": tags,
            "full_35_models_in_source_order": full,
            "incomplete_13_models": incomplete,
            "singleton_width_models_excluded": sorted(
                t for t in full if len(widths[next(iter(rows[t]["hidden"].values())).shape[0]]) == 1),
            "contributing_31_models": contributing,
            "shared_item_ids": [str(i) for i in shared],
            "n_model_item_cells_in_anova": sum(
                g["M"] * g["I"] for g in raw_groups.values()),
        },
        "standardization_receipts": {t: receipts[t] for t in contributing},
    }
    # comparator: producer terminal (same fields)
    prod_path = ctx.analysis_root / "runs/extension_reuse_v1/pooling_result.json"
    ctx.consume(prod_path, "pooling_producer_terminal", "comparison_only")
    prod = read_json(prod_path)
    pergrp_path = ctx.analysis_root / "runs/extension_reuse_v1/pooling_per_group.csv"
    ctx.consume(pergrp_path, "pooling_per_group_terminal", "comparison_only")
    vr_path = ctx.analysis_root / "verifier/V_NUM_POOLING_REUSE.json"
    ctx.consume(vr_path, "pooling_independent_verifier", "comparison_only")
    vreuse = read_json(vr_path)
    claim_path = ctx.resolve_recorded(
        "Imports/theory_rescue_20260913/checks/results.json", role="pooling_claim_pointer")
    ctx.consume(claim_path, "pooling_ledger_claim_pointer", "comparison_only")
    claim_doc = read_json(claim_path)

    scope = ["Raw model-share is coordinate/scale dependent; standardized near-zero share is definition-only.",
             "Share is a property of this saved decomposition, not a universal no-model-effect result.",
             "JSONL cells are frozen observations; no hidden state was regenerated."]
    checks = [
        ("pooled_raw_model_share", "P2-F-POOLING",
         "/check5_pooled_model_effect_B1/pooled_raw/model_share",
         pooled_raw["model_share"],
         claim_doc["check5_pooled_model_effect_B1"]["pooled_raw"]["model_share"],
         "Imports/theory_rescue_20260913/checks/results.json (ledger pointer)",
         "direct double-centred interaction-residual ANOVA over 31 contributing models / 170 shared items",
         "recomputed_from_saved_cell_observations"),
        ("pooled_standardized_model_share", "P2-F-POOLING",
         "/check5_pooled_model_effect_B1/pooled_standardized/model_share",
         pooled_std["model_share"],
         claim_doc["check5_pooled_model_effect_B1"]["pooled_standardized"]["model_share"],
         "Imports/theory_rescue_20260913/checks/results.json (ledger pointer)",
         "standardized residual ANOVA (per-model dimension z-scoring of saved observations)",
         "recomputed_from_saved_cell_observations"),
    ]
    for cid, rid, ptr, fv, rv, rs, meth, sl in checks:
        ctx.claim(cid, rid, ptr, ptr.split("/")[-1], fv, rv, rs, meth, sl,
                  limitations=scope)
    ctx.counts["pooling_files_rehashed"] = len(vreuse["input_hashes"])
    ctx.claim("pooling_jsonl_input_hash_count", "P2-F-POOLING", "(manifest)",
              "n_jsonl_with_pinned_hash",
              sum(1 for p in vreuse["input_hashes"]
                  if sha256_file(ctx.resolve_recorded(p, role="pooling_jsonl_input_hash"))
                  == vreuse["input_hashes"][p]),
              len(vreuse["input_hashes"]),
              "verifier/V_NUM_POOLING_REUSE.json input_hashes",
              "re-hash of the 48 JSONL inputs", "source_check", exact=True)
    # per-group terminal comparison (raw + standardized, all 5 widths)
    for width, g in sorted(raw_groups.items(), key=lambda kv: int(kv[0])):
        for k in ("SS_model_trace", "SS_item_trace", "SS_resid_trace", "SS_total_trace",
                  "model_share", "item_share", "resid_share"):
            ctx.claim("pooling_raw_%s_%s" % (width, k), "P2-F-POOLING",
                      "/check5_pooled_model_effect_B1/raw_groups/%s/%s" % (width, k),
                      k, g[k], prod["anova_raw_per_group"][width][k],
                      "runs/extension_reuse_v1/pooling_result.json",
                      "per-width raw ANOVA re-aggregation", "recomputed_from_saved_cell_observations")
        for k in ("SS_model_trace", "SS_item_trace", "SS_resid_trace", "SS_total_trace",
                  "model_share", "item_share", "resid_share"):
            ctx.claim("pooling_std_%s_%s" % (width, k), "P2-F-POOLING",
                      "/check5_pooled_model_effect_B1/standardized_groups/%s/%s" % (width, k),
                      k, std_groups[width][k], prod["anova_standardized_per_group"][width][k],
                      "runs/extension_reuse_v1/pooling_result.json",
                      "per-width standardized ANOVA re-aggregation",
                      "recomputed_from_saved_cell_observations")
    for k in ("SS_model_trace", "SS_item_trace", "SS_resid_trace", "SS_total_trace",
              "model_share", "item_share", "resid_share"):
        ctx.claim("pooling_pooled_raw_%s" % k, "P2-F-POOLING",
                  "/check5_pooled_model_effect_B1/pooled_raw/%s" % k, k,
                  pooled_raw[k], prod["pooled_raw"][k],
                  "runs/extension_reuse_v1/pooling_result.json",
                  "pooled raw ANOVA re-aggregation", "recomputed_from_saved_cell_observations")
        ctx.claim("pooling_pooled_std_%s" % k, "P2-F-POOLING",
                  "/check5_pooled_model_effect_B1/pooled_standardized/%s" % k, k,
                  pooled_std[k], prod["pooled_standardized"][k],
                  "runs/extension_reuse_v1/pooling_result.json",
                  "pooled standardized ANOVA re-aggregation",
                  "recomputed_from_saved_cell_observations")
    ctx.claim("pooling_verdict", "P2-F-POOLING", "(pooling_result)/verdict", "verdict",
              verdict, prod["verdict"], "runs/extension_reuse_v1/pooling_result.json",
              "verdict rule replay", "recomputed_from_saved_cell_observations", exact=True)
    ctx.claim("pooling_panel_accounting", "P2-F-POOLING", "(pooling_result)/panel_accounting",
              "all_48/full_35/incomplete_13/singleton4/contributing31/cells5270",
              (len(tags), len(full), len(incomplete),
               len(fresh["panel_accounting"]["singleton_width_models_excluded"]),
               len(contributing), fresh["panel_accounting"]["n_model_item_cells_in_anova"]),
              (len(prod["panel_accounting"]["all_48_models_in_source_order"]),
               len(prod["panel_accounting"]["full_35_models_in_source_order"]),
               len(prod["panel_accounting"]["incomplete_13_models"]),
               len(prod["panel_accounting"]["singleton_width_models_excluded"]),
               len(prod["panel_accounting"]["contributing_31_models"]),
               prod["panel_accounting"]["n_model_item_cells_in_ANOVA"]),
              "runs/extension_reuse_v1/pooling_result.json",
              "panel accounting reconciliation", "recomputed_from_saved_cell_observations",
              exact=True)
    ctx.claim("pooling_source_order_identity", "P2-F-POOLING",
              "(pooling_result)/panel_accounting/all_48_models_in_source_order", "model_id_list",
              tags, prod["panel_accounting"]["all_48_models_in_source_order"],
              "runs/extension_reuse_v1/pooling_result.json", "ID list exact check",
              "recomputed_from_saved_cell_observations", exact=True)
    ctx.values = getattr(ctx, "values", {})
    ctx.values["pooling"] = fresh
    ctx.timings["pooling_s"] = round(time.time() - t0, 2)
    return fresh


# ----------------------------------------------------------------------- g0
def _als_init_b(ell, mask):
    NI = ell.shape[1]
    itmean = np.full(NI, np.nan)
    for i in range(NI):
        col = mask[:, i]
        if col.any():
            itmean[i] = ell[col, i].mean()
    pres = np.isfinite(itmean)
    b = np.full(NI, np.nan)
    z = itmean[pres]
    b[pres] = -((z - z.mean()) / z.std())
    return b


def _als_update_s_theta(ell, mask, b):
    E = np.where(mask, ell, 0.0)
    bf = np.where(np.isfinite(b), b, 0.0)
    M = mask & np.isfinite(b)[None, :]
    n = M.sum(1).astype(float)
    se = (E * M).sum(1)
    sb = (bf * M).sum(1)
    mean_e = se / n
    mean_b = sb / n
    cov = (E * bf * M).sum(1) / n - mean_e * mean_b
    var = (bf * bf * M).sum(1) / n - mean_b * mean_b
    s = -cov / var
    theta = (mean_e + s * mean_b) / s
    return s, theta


def _als_update_b(ell, mask, s, theta):
    E = np.where(mask, ell, 0.0)
    ri = (E - (s * theta)[:, None]) * mask
    num = (s[:, None] * ri).sum(0)
    den = ((s * s)[:, None] * mask).sum(0)
    b = np.full(ell.shape[1], np.nan)
    ok = den > 0
    b[ok] = -num[ok] / den[ok]
    return b


def _als_gauge(b):
    pres = np.isfinite(b)
    bb = b[pres]
    out = b.copy()
    out[pres] = (bb - bb.mean()) / bb.std()
    return out


def _als_objective(ell, mask, s, theta, b):
    pred = s[:, None] * (theta[:, None] - b[None, :])
    r = (ell - pred)[mask & np.isfinite(b)[None, :]]
    return float((r * r).sum())


def _als(ell, mask, tol, maxit):
    b = _als_init_b(ell, mask)
    obj_hist = []
    it = 0
    delta = math.inf
    for it in range(1, maxit + 1):
        bprev = b.copy()
        s, theta = _als_update_s_theta(ell, mask, b)
        b = _als_gauge(_als_update_b(ell, mask, s, theta))
        pres = np.isfinite(b) & np.isfinite(bprev)
        delta = float(np.max(np.abs(b[pres] - bprev[pres])))
        obj_hist.append(_als_objective(ell, mask, s, theta, b))
        if delta < tol:
            break
    s, theta = _als_update_s_theta(ell, mask, b)
    return {"b": b, "s": s, "theta": theta, "niter": it, "delta": delta,
            "obj": _als_objective(ell, mask, s, theta, b), "obj_hist": obj_hist}


def branch_g0(ctx):
    t0 = time.time()
    inv_path = ctx.project_root / "ext_P2_1_itemmodel_20260913/inventory.csv"
    ctx.consume(inv_path, "g0_inventory", "computation_input",
                expected_sha256="7ab3dbfc32d2021d5a53ab889b2d7a139913a5045870c61c03782a31ca52bfdb")
    with inv_path.open() as fh:
        inv = [r for r in csv.DictReader(fh) if r["domain"] == "code"]
    if len(inv) != 48:
        raise ContractRefusal("inventory code rows != 48: %d" % len(inv))
    dirs = [ctx.project_root / "Imports/geometry/panel_geom_local_20260607",
            ctx.project_root / "Imports/geometry/panel_geom_expansion_20260607"]

    def find_file(tag):
        for d in dirs:
            p = d / ("geom_%s.jsonl" % tag)
            if p.exists():
                return p
        raise MissingRefusal("jsonl for inventory row %s not found" % tag)

    NI, NM = 170, 48
    ell = np.full((NM, NI), np.nan)
    yy = np.full((NM, NI), np.nan)
    model_ids, tags, observed_items = [], [], []
    for mi, r in enumerate(inv):
        path = find_file(r["model"])
        rec = read_model_jsonl(ctx, path, "g0_panel_jsonl:%s" % r["model"],
                               need_hidden=True)
        tags.append(r["model"])
        model_ids.append(r["model_id"])
        for iid, l, a in zip(rec["item_id"], rec["mean_logprob"], rec["accuracy"]):
            ell[mi, iid] = l
            yy[mi, iid] = a
        observed_items.append(list(rec["item_id"]))
    mask = np.isfinite(ell)
    panel_meta_path = ctx.project_root / "ext_P2_geo_20260914/fit_panel/fit_meta.json"
    ctx.consume(panel_meta_path, "g0_fit_meta_terminal", "comparison_only")
    fit_meta = read_json(panel_meta_path)
    # model order must match the saved panel CSV order
    theta_csv = ctx.project_root / "ext_P2_geo_20260914/fit_panel/theta_s_sigma_panel.csv"
    b_csv = ctx.project_root / "ext_P2_geo_20260914/fit_panel/b_panel.csv"
    bloo_npz = ctx.project_root / "ext_P2_geo_20260914/fit_panel/b_loo.npz"
    eps_npz = ctx.project_root / "ext_P2_geo_20260914/fit_panel/eps_panel.npz"
    ctx.consume(theta_csv, "g0_saved_params_csv", "computation_input")
    ctx.consume(b_csv, "g0_saved_b_csv", "computation_input")
    ctx.consume(bloo_npz, "g0_saved_loo_observations", "computation_input")
    ctx.consume(eps_npz, "g0_saved_eps_observations", "computation_input")
    with theta_csv.open() as fh:
        ts_rows = list(csv.DictReader(fh))
    with b_csv.open() as fh:
        b_rows = list(csv.DictReader(fh))
    saved_order = [r["model_id"] for r in ts_rows]
    if saved_order != model_ids:
        raise ContractRefusal("inventory/jsonl model order differs from saved panel CSV")
    with np.load(bloo_npz, allow_pickle=False) as z:
        b_loo = z["b_loo"]
        b_full_saved = z["b_full"]
        loo_corr_saved = z["loo_corr"]
        loo_models = [str(x) for x in z["model_ids"]]

    observed = int(mask.sum())
    expected_rows = int(sum(int(r["n_items"]) for r in inv))
    coverage = observed / (NM * NI)

    fit = _als(ell, mask, tol=1e-6, maxit=5000)
    b, s, theta = fit["b"], fit["s"], fit["theta"]
    pred = s[:, None] * (theta[:, None] - b[None, :])
    resid = ell - pred
    sigma = np.array([resid[m, mask[m]].std() for m in range(NM)])
    item_mean_acc = np.array([yy[mask[:, i], i].mean() for i in range(NI)])
    corr_b_itemacc = pearson(b, item_mean_acc)
    obj_hist = np.array(fit["obj_hist"])
    monotone = bool(np.all(np.diff(obj_hist) <= 0))
    gauge = {"frac_s_positive": float(np.mean(s > 0)),
             "mean_b": float(b.mean()), "var_b": float(b.var()),
             "mean_m_log_s_over_positive": float(np.mean(np.log(s[s > 0]))),
             "n_s_nonpositive": int((s <= 0).sum()),
             "s_max": float(s.max()), "s_min": float(s.min()),
             "sigma_max": float(sigma.max()), "sigma_min": float(sigma.min())}
    saved_b = np.array([float(r["b"]) for r in b_rows])
    saved_s = np.array([float(r["s"]) for r in ts_rows])
    saved_sigma = np.array([float(r["sigma"]) for r in ts_rows])
    ctx.claim("g0_panel_coverage_frac", "P2-F-G0-PANEL", "/panel/coverage_frac",
              "coverage_frac", coverage, fit_meta["panel"]["coverage_frac"],
              "fit_meta.json (terminal)", "observed_cells/(48*170) from saved cells",
              "recomputed_from_saved_cell_observations")
    ctx.claim("g0_panel_counts", "P2-F-G0-PANEL", "/panel",
              "n_models/n_items/observed_cells/expected_rows_ext_P2_1/rows_match",
              (NM, NI, observed, expected_rows, observed == expected_rows),
              (fit_meta["panel"]["n_models"], fit_meta["panel"]["n_items"],
               fit_meta["panel"]["observed_cells"],
               fit_meta["panel"]["expected_rows_ext_P2_1"],
               fit_meta["panel"]["rows_match"]),
              "fit_meta.json (terminal)", "counts from saved observation masks",
              "recomputed_from_saved_cell_observations", exact=True)
    for cid, key, val in [
            ("g0_replay_converged", "converged", bool(fit["delta"] < 1e-6)),
            ("g0_replay_iterations", "iterations", int(fit["niter"])),
            ("g0_replay_final_max_abs_delta_b", "final_max_abs_delta_b", fit["delta"]),
            ("g0_replay_final_objective_SSR", "final_objective_SSR", fit["obj"]),
            ("g0_replay_objective_monotone", "objective_monotone_decreasing", monotone)]:
        ctx.claim(cid, "P2-F-G0-PANEL", "/convergence/" + key, key, val,
                  fit_meta["convergence"][key],
                  "fit_meta.json (terminal)",
                  "deterministic ALS replay from the same saved panel cells (tol 1e-6, maxit 5000)",
                  "replayed_fit_from_saved_cells",
                  exact=isinstance(fit_meta["convergence"][key], (bool, int)))
    for key, val in gauge.items():
        ctx.claim("g0_gauge_%s" % key, "P2-F-G0-PANEL",
                  "/gauge_diagnostics/%s" % key, key, val,
                  fit_meta["gauge_diagnostics"][key], "fit_meta.json (terminal)",
                  "gauge diagnostics from replayed fit parameters",
                  "replayed_fit_from_saved_cells")
    ctx.claim("g0_sanity_corr_b_itemacc", "P2-F-G0-PANEL", "/sanity/corr_b_itemacc",
              "corr_b_itemacc", corr_b_itemacc, fit_meta["sanity"]["corr_b_itemacc"],
              "fit_meta.json (terminal)",
              "pearson(replayed b, item mean accuracy over saved cells)",
              "replayed_fit_from_saved_cells")
    ctx.claim("g0_replay_b_vs_saved_csv", "P2-F-G0-PANEL", "(internal)",
              "max_abs_diff_replayed_b_minus_saved_csv",
              float(np.max(np.abs(b - saved_b))), 0.0,
              "fit_panel/b_panel.csv (saved fit output)",
              "replayed b vs saved b over 170 items", "replayed_fit_from_saved_cells",
              limitations=["pipeline-internal check of the replay against the saved fit output"])
    ctx.claim("g0_replay_thetasigma_vs_saved_csv", "P2-F-G0-PANEL", "(internal)",
              "max over (theta,s,sigma) of max_abs_diff_replayed_minus_saved",
              max(float(np.max(np.abs(theta - np.array([float(r["theta"]) for r in ts_rows])))),
                  float(np.max(np.abs(s - saved_s))),
                  float(np.max(np.abs(sigma - saved_sigma)))),
              0.0, "fit_panel/theta_s_sigma_panel.csv (saved fit output)",
              "replayed parameters vs saved parameters", "replayed_fit_from_saved_cells",
              limitations=["pipeline-internal check of the replay against the saved fit output"])
    # G0-STABILITY: LOO from saved npz observations
    loo_corr = []
    for m in range(NM):
        ok = np.isfinite(b_loo[m]) & np.isfinite(b_full_saved)
        loo_corr.append(pearson(b_loo[m][ok], b_full_saved[ok]))
    loo_corr = np.array(loo_corr)
    ctx.claim("g0_loo_min_corr", "P2-F-G0-STABILITY", "/leave_one_model_out/min_corr",
              "min_corr", float(loo_corr.min()),
              fit_meta["leave_one_model_out"]["min_corr"], "fit_meta.json (terminal)",
              "pearson(b_loo_m, b_full) over shared finite items from b_loo.npz",
              "reaggregated_from_saved_resampling_array")
    ctx.claim("g0_loo_min_corr_model", "P2-F-G0-STABILITY",
              "/leave_one_model_out/min_corr_model", "min_corr_model",
              loo_models[int(np.argmin(loo_corr))],
              fit_meta["leave_one_model_out"]["min_corr_model"], "fit_meta.json (terminal)",
              "argmin of recomputed LOO correlations", "reaggregated_from_saved_resampling_array",
              exact=True)
    ctx.claim("g0_loo_all_ge_0.95", "P2-F-G0-STABILITY",
              "/leave_one_model_out/all_ge_0.95", "all_ge_0.95",
              bool(np.all(loo_corr >= 0.95)), fit_meta["leave_one_model_out"]["all_ge_0.95"],
              "fit_meta.json (terminal)", "threshold check on recomputed LOO correlations",
              "reaggregated_from_saved_resampling_array", exact=True)
    ctx.claim("g0_loo_corr_array_vs_npz", "P2-F-G0-STABILITY", "(internal)",
              "max_abs_diff_recomputed_loo_corr_minus_saved_npz",
              float(np.max(np.abs(loo_corr - loo_corr_saved))), 0.0,
              "fit_panel/b_loo.npz (saved loo_corr array)",
              "recomputed LOO correlation vs saved array", "reaggregated_from_saved_resampling_array")
    ctx.g0_observed_items = {m: ids for m, ids in zip(model_ids, observed_items)}
    total_s2 = float(np.sum(saved_s ** 2))
    sel = "ByteDance-Seed/Seed-OSS-36B-Instruct"
    sel_idx = saved_order.index(sel)
    sel_share = float(saved_s[sel_idx] ** 2) / total_s2
    seed_idx = [i for i, m in enumerate(saved_order) if "Seed-OSS" in m]
    two_share = float(np.sum(saved_s[seed_idx] ** 2)) / total_s2
    ctx.claim("g0_selected_seedoss_s2_share", "P2-F-G0-STABILITY",
              "/s -> selected SeedOSS sum(s^2)/all sum(s^2)", "s2_share_selected",
              sel_share, sel_share,
              "theta_s_sigma_panel.csv saved array (field is declared READ_FROM_ARRAY)",
              "s^2 share of ByteDance-Seed/Seed-OSS-36B-Instruct from saved s array",
              "recomputed_from_saved_array", limitations=[
                  "No independent terminal value exists for this field; the saved array is the source.",
                  "G0_REPORT.md states two Seed-OSS carry ~83%% of s^2 weight; recomputed two-model share %.6f" % two_share])
    ctx.counts["g0_two_seedoss_s2_share"] = two_share
    gen_flag = "UNAVAILABLE (no generation-length key in jsonl; skipped)"
    ctx.claim("g0_gen_length_flag", "P2-F-G0-STABILITY", "/flags/gen_length_field",
              "gen_length_field", gen_flag,
              fit_meta["flags"]["gen_length_field"], "fit_meta.json (terminal)",
              "flag status string", "declared_metadata", exact=True)
    ctx.values["g0"] = {
        "model_order_matches_saved_csv": True,
        "observed_cells": observed, "coverage_frac": coverage,
        "convergence": {"converged": bool(fit["delta"] < 1e-6), "iterations": fit["niter"],
                        "final_max_abs_delta_b": fit["delta"],
                        "final_objective_SSR": fit["obj"],
                        "objective_monotone_decreasing": monotone},
        "gauge_diagnostics": gauge, "corr_b_itemacc": corr_b_itemacc,
        "loo_min_corr": float(loo_corr.min()), "loo_min_corr_model":
            loo_models[int(np.argmin(loo_corr))],
        "selected_seedoss_s2_share": sel_share, "two_seedoss_s2_share": two_share,
    }
    ctx.timings["g0_s"] = round(time.time() - t0, 2)
    return ctx.values["g0"]


# ----------------------------------------------------------------------- g1
def branch_g1(ctx):
    t0 = time.time()
    geo = ctx.project_root / "ext_P2_geo_20260914/h1"
    r2_path = geo / "r2_by_model_layer.csv"
    tr_path = geo / "transfer_rho.csv"
    res_path = geo / "results_h1.json"
    for p, role in [(r2_path, "g1_r2_by_model_layer"), (tr_path, "g1_transfer_rho"),
                    (res_path, "g1_terminal_result")]:
        ctx.consume(p, role, "comparison_only" if p == res_path else "computation_input")
    res = read_json(res_path)
    with r2_path.open() as fh:
        r2rows = list(csv.DictReader(fh))
    pool_rows = [r for r in r2rows if int(r["layer"]) == -1]
    if len(pool_rows) != 48:
        raise ContractRefusal("h1 pooled rows != 48: %d" % len(pool_rows))
    med_r2 = float(np.median([float(r["r2_oof"]) for r in pool_rows]))
    med_sp = float(np.median([float(r["spearman_oof"]) for r in pool_rows]))
    ctx.claim("g1_difficulty_median_r2", "P2-F-G1-DIFFICULTY", "/task_A_pooled/median_r2",
              "median_r2", med_r2, res["task_A_pooled"]["median_r2"],
              "h1/results_h1.json (terminal)",
              "median over saved layer==-1 r2_oof rows (48 models)",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g1_difficulty_median_spearman", "P2-F-G1-DIFFICULTY",
              "/task_A_pooled/median_spearman", "median_spearman", med_sp,
              res["task_A_pooled"]["median_spearman"], "h1/results_h1.json (terminal)",
              "median over saved layer==-1 spearman_oof rows",
              "reaggregated_from_saved_per_model_outputs")
    with tr_path.open() as fh:
        trrows = list(csv.DictReader(fh))
    rhos = np.array([float(r["rho_transfer"]) for r in trrows])
    within = np.array([float(r["rho_within_A_baseline"]) for r in trrows])
    n_valid = int(np.isfinite(rhos).sum())
    med_tr = float(np.median(rhos))
    med_within = float(np.nanmedian(within))
    ctx.claim("g1_transfer_median_rho", "P2-F-G1-TRANSFER", "/task_C_transfer/median_transfer_rho",
              "median_transfer_rho", med_tr, res["task_C_transfer"]["median_transfer_rho"],
              "h1/results_h1.json (terminal)", "median over saved 240 pair rho values",
              "reaggregated_from_saved_per_pair_outputs")
    ctx.claim("g1_transfer_median_within_baseline", "P2-F-G1-TRANSFER",
              "/task_C_transfer/median_within_A_baseline", "median_within_A_baseline",
              med_within, res["task_C_transfer"]["median_within_A_baseline"],
              "h1/results_h1.json (terminal)", "nan-median over saved 240 within-A baselines",
              "reaggregated_from_saved_per_pair_outputs")
    ctx.claim("g1_transfer_counts", "P2-F-G1-TRANSFER",
              "/task_C_transfer/n_pairs,/task_C_transfer/n_valid",
              "n_pairs/n_valid", (len(trrows), n_valid),
              (res["task_C_transfer"]["n_pairs"], res["task_C_transfer"]["n_valid"]),
              "h1/results_h1.json (terminal)", "exact count check over saved pair table",
              "reaggregated_from_saved_per_pair_outputs", exact=True)
    # pair bootstrap from saved rho values with the declared seed-sequence child 4
    ss = np.random.SeedSequence(int(res["seed"]))
    ch = ss.spawn(6)
    rng_bp = np.random.default_rng(ch[int(res["seed_children"]["boot_pair"])])
    n_boot = int(res["task_C_transfer"]["n_boot"])
    npv = len(rhos)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng_bp.integers(0, npv, npv)
        boot[b] = np.median(rhos[idx])
    ci_pair = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    for i, side in enumerate(("lo", "hi")):
        ctx.claim("g1_pair_bootstrap_ci_%s" % side, "P2-F-G1-TRANSFER",
                  "/task_C_transfer/pair_bootstrap_ci_2.5_97.5/%d" % i, side,
                  ci_pair[i], res["task_C_transfer"]["pair_bootstrap_ci_2.5_97.5"][i],
                  "h1/results_h1.json (terminal)",
                  "replay of the stored pair bootstrap: 2000 median resamples of the 240 saved "
                  "pair rho values with SeedSequence(seed).spawn(6)[4] (same draw sequence)",
                  "replayed_bootstrap_from_saved_per_pair_outputs")
    ctx.claim("g1_pair_bootstrap_excludes_0", "P2-F-G1-TRANSFER",
              "/task_C_transfer/pair_bootstrap_excludes_0", "excludes_0",
              bool(ci_pair[0] > 0 or ci_pair[1] < 0),
              res["task_C_transfer"]["pair_bootstrap_excludes_0"],
              "h1/results_h1.json (terminal)", "interval sign check",
              "replayed_bootstrap_from_saved_per_pair_outputs", exact=True)
    # AB split sizes are recomputable from the declared child-2 permutation
    rng_ab = np.random.default_rng(ch[int(res["seed_children"]["ab_split"])])
    perm = rng_ab.permutation(170)
    sizes = [int(len(perm[:85])), int(len(perm[85:]))]
    ctx.claim("g1_ab_split_sizes", "P2-F-G1-TRANSFER", "/task_C_transfer/ab_split_sizes",
              "ab_split_sizes", sizes, res["task_C_transfer"]["ab_split_sizes"],
              "h1/results_h1.json (terminal)",
              "replay of the declared child-2 permutation over the 170-item universe",
              "replayed_from_declared_seed", exact=True)
    ctx.claim("g1_folds_config", "P2-F-G1-DIFFICULTY", "/folds", "folds_config",
              {k: res["folds"][k] for k in ("inner_seed_base", "k_inner", "k_outer",
                                            "lambda_scaling", "outer_fold_sizes")},
              {k: res["folds"][k] for k in ("inner_seed_base", "k_inner", "k_outer",
                                            "lambda_scaling", "outer_fold_sizes")},
              "h1/results_h1.json (declared config)", "declared configuration metadata",
              "declared_metadata", exact=True)
    ctx.claim("g1_alpha_grid", "P2-F-G1-DIFFICULTY", "/folds/alpha_grid",
              "alpha_grid", [float(x) for x in np.logspace(-4, 4, 17)],
              res["folds"]["alpha_grid"], "h1/results_h1.json (declared config)",
              "replay of np.logspace(-4,4,17)", "declared_metadata", exact=False)
    # ---- secondary item bootstrap: replay from the regenerated Task-C pair cache
    cache_dir = Path(getattr(ctx, "g1_cache_dir", None)
                     or (ctx.analysis_root / "runs/c17_supplements_b_check_v1/"
                         "g1_taskc_pair_cache_v1")).resolve()
    cache_npz = cache_dir / "pair_cache_v1.npz"
    cache_man = cache_dir / "g1_taskc_cache_manifest.json"
    ctx.consume(cache_npz, "g1_taskc_pair_cache", "computation_input")
    ctx.consume(cache_man, "g1_taskc_cache_manifest", "computation_input")
    cman = read_json(cache_man)
    ctx.claim("g1_cache_source_pins", "P2-F-G1-TRANSFER", "(cache manifest)/source_pins",
              "h1 run_h1.py / h1_lib.py / b_loo.npz / inventory.csv pinned hashes match current files",
              all(entry["sha256"] == sha256_file(ctx.resolve_recorded(entry["path"],
                                                                      role="g1_cache_pin_source"))
                  for entry in [
                      {"path": "ext_P2_geo_20260914/h1/code/run_h1.py",
                       "sha256": cman["source_pins"]["h1/run_h1.py"]},
                      {"path": "ext_P2_geo_20260914/h1/code/h1_lib.py",
                       "sha256": cman["source_pins"]["h1/h1_lib.py"]},
                      {"path": "ext_P2_geo_20260914/fit_panel/b_loo.npz",
                       "sha256": cman["source_pins"]["fit_panel/b_loo.npz"]},
                      {"path": "ext_P2_1_itemmodel_20260913/inventory.csv",
                       "sha256": cman["source_pins"]["ext_P2_1_itemmodel_20260913/inventory.csv"]}]),
              True, "g1_taskc_cache_manifest.json source_pins", "re-hash of pinned regeneration sources",
              "source_check", exact=True)
    ctx.claim("g1_cache_wall_s", "P2-F-G1-TRANSFER", "(cache manifest)/wall_s",
              "regeneration_wall_s", float(cman["wall_s"]), float(cman["wall_s"]),
              "g1_taskc_cache_manifest.json", "measured Task-C regeneration timing (Task C only)",
              "measured_timing")
    with np.load(cache_npz, allow_pickle=False) as z:
        off = z["offset"].copy(); cyhat = z["yhat"].copy(); cy = z["y"].copy()
        ccols = z["cols"].copy(); crho = z["rho"].copy(); cpid = z["pair_index"].copy()
        cA = [str(x) for x in z["model_A"]]; cB = [str(x) for x in z["model_B"]]
        cR = z["r"].copy(); cN = z["n_commonA"].copy()
        cfoldB = z["foldB"].copy(); cn_univ = int(z["n_univ"]); cboot = int(z["n_boot"])
    if not (len(cpid) == 240 and len(cfoldB) + len(cfoldB) == cn_univ
            and len(set(cpid.tolist())) == 240):
        raise ContractRefusal("g1 task-C cache shape/ID check failed")
    ctx.claim("g1_cache_pair_ids", "P2-F-G1-TRANSFER", "(cache)/pair_index+model_A/model_B",
              "pair ids/order vs transfer_rho.csv",
              [(int(cpid[i]), cA[i], cB[i]) for i in range(len(cpid))],
              [(int(r["pair_index"]), r["model_A"], r["model_B"]) for r in trrows],
              "h1/transfer_rho.csv", "exact pair order/ID comparator",
              "regenerated_same_estimator", exact=True)
    rho_from_cache, diffs_cache = [], []
    for i in range(len(cpid)):
        rho_i = spearman(cyhat[off[i]:off[i + 1]], cy[off[i]:off[i + 1]]) \
            if off[i + 1] > off[i] else float("nan")
        rho_from_cache.append(rho_i)
        diffs_cache.append(abs(rho_i - crho[i]))
    ctx.claim("g1_cache_rho_vs_stored_pair_table", "P2-F-G1-TRANSFER",
              "(cache)/rho vs h1/transfer_rho.csv", "max_abs_rho_diff_over_240_pairs",
              float(np.nanmax(diffs_cache)), 0.0,
              "h1/transfer_rho.csv (independent retained per-pair comparator)",
              "recomputed Spearman from cached yhat/y matches the retained per-pair rho",
              "regenerated_same_estimator")
    for i, key in ((0, "r"), (1, "n_commonA")):
        fresh = cR if key == "r" else cN
        ref = [int(float(r["r_pca"])) if key == "r" else int(float(r["n_commonA"]))
               for r in trrows]
        ctx.claim("g1_cache_%s" % key, "P2-F-G1-TRANSFER", "(cache)/%s" % key, key,
                  [int(x) for x in fresh], ref, "h1/transfer_rho.csv",
                  "exact per-pair count comparator", "regenerated_same_estimator", exact=True)
    rows = [{"pair_index": int(cpid[i]), "model_A": cA[i], "model_B": cB[i],
             "rho": float(crho[i]), "rho_within": float("nan"), "r": int(cR[i]),
             "n_commonA": int(cN[i]), "yhat": cyhat[off[i]:off[i + 1]],
             "y": cy[off[i]:off[i + 1]], "cols": ccols[off[i]:off[i + 1]]}
            for i in range(len(cpid))]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "c10v_g1_taskc_regen", CODE_DIR / "g1_taskc_regen.py")
    regen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen_mod)
    ci_item = regen_mod.item_bootstrap_from_rows(rows, cfoldB, cn_univ, n_boot=cboot)
    for i, side in enumerate(("lo", "hi")):
        ctx.claim("g1_item_bootstrap_ci_%s" % side, "P2-F-G1-TRANSFER",
                  "/task_C_transfer/item_bootstrap_ci_2.5_97.5/%d" % i, side,
                  float(ci_item["ci"][i]),
                  res["task_C_transfer"]["item_bootstrap_ci_2.5_97.5"][i],
                  "h1/results_h1.json (terminal)",
                  "replay of the secondary item bootstrap (2000 draws, SeedSequence child 5) "
                  "from the regenerated Task-C pair cache (same-estimator reproduction)",
                  "reproduced_from_regenerated_intermediate")
    ctx.claim("g1_item_bootstrap_draws", "P2-F-G1-TRANSFER",
              "(item bootstrap)/n_valid_draws", "n_valid_draws",
              ci_item["n_valid_draws"], cboot, "g1_taskc_cache_manifest.json n_boot",
              "exact draw count check", "reproduced_from_regenerated_intermediate", exact=True)
    # bounded representative regeneration check (independent of the cache)
    n_check = max(1, int(getattr(ctx, "g1_regen_check_pairs", 2)))
    check_idx = sorted({0, len(cpid) // 2})[:n_check]
    regen_rows = regen_mod.regenerate_pair_rows(ctx.project_root, check_idx)
    rep = []
    for cr in regen_rows:
        i = int(np.where(cpid == cr["pair_index"])[0][0])
        exact_vectors = bool(np.array_equal(cr["yhat"], cyhat[off[i]:off[i + 1]])
                             and np.array_equal(cr["y"], cy[off[i]:off[i + 1]])
                             and np.array_equal(cr["cols"], ccols[off[i]:off[i + 1]]))
        maxdiff = float(np.max(np.abs(cr["yhat"] - cyhat[off[i]:off[i + 1]]))) \
            if len(cr["yhat"]) else float("nan")
        rep.append({"pair_index": cr["pair_index"], "model_A": cr["model_A"],
                    "model_B": cr["model_B"], "bitwise_equal": exact_vectors,
                    "max_abs_yhat_diff": maxdiff,
                    "rho_diff": abs(cr["rho"] - float(crho[i]))})
    ok = all((r["bitwise_equal"] or r["max_abs_yhat_diff"] <= SCALAR_ATOL)
             and r["rho_diff"] <= SCALAR_ATOL for r in rep)
    ctx.claim("g1_representative_regeneration_check", "P2-F-G1-TRANSFER",
              "(bounded regen check)", "representative_pairs_rebuilt_from_hidden_states",
              rep, None, "independent regeneration path (g1_taskc_regen.regenerate_pair_rows)",
              "rebuild %d representative pairs from the saved hidden states with the frozen "
              "seed/pair/PCA/ridge formulas and compare bitwise against the cached vectors "
              "(prevents circular old-summary copy)" % len(rep),
              "independent_regeneration_path", exact=True,
              status_override=("PASS" if ok else "FAIL"),
              limitations=["bounded check; the full cache was regenerated once and every pair "
                           "was compared to the retained transfer_rho.csv"])
    ctx.counts["g1_regen_check_pairs"] = len(rep)
    ctx.counts["g1_cache_max_abs_rho_diff"] = float(np.nanmax(diffs_cache))
    ctx.values["g1"] = {"median_r2": med_r2, "median_spearman": med_sp,
                        "median_transfer_rho": med_tr, "median_within_baseline": med_within,
                        "n_pairs": len(trrows), "n_valid": n_valid,
                        "pair_bootstrap_ci": ci_pair, "ab_split_sizes": sizes,
                        "item_bootstrap_ci": [float(x) for x in ci_item["ci"]],
                        "item_bootstrap_n_valid_draws": ci_item["n_valid_draws"],
                        "taskc_cache": str(cache_npz),
                        "taskc_cache_sha256": sha256_file(cache_npz),
                        "taskc_regen_wall_s": float(cman["wall_s"]),
                        "representative_regeneration_check": rep}
    ctx.timings["g1_s"] = round(time.time() - t0, 2)
    return ctx.values["g1"]


# ----------------------------------------------------------------------- g2
def branch_g2(ctx):
    t0 = time.time()
    geo = ctx.project_root / "ext_P2_geo_20260914"
    beta_path = geo / "h2/beta_by_model.csv"
    res_path = geo / "h2/results_h2.json"
    bloo_npz = geo / "fit_panel/b_loo.npz"
    ctx.consume(beta_path, "g2_beta_by_model", "computation_input")
    ctx.consume(res_path, "g2_terminal_result", "comparison_only")
    ctx.consume(bloo_npz, "g2_b_loo_observations", "computation_input")
    res = read_json(res_path)
    with beta_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 48:
        raise ContractRefusal("h2 rows != 48: %d" % len(rows))
    mids = [r["model_id"] for r in rows]
    s = np.array([float(r["s_hat"]) for r in rows])
    th = np.array([float(r["theta_hat"]) for r in rows])
    bp = np.array([float(r["beta_pooled"]) for r in rows])
    bb = np.array([float(r["beta_best"]) for r in rows])
    fam = [r["family"] for r in rows]

    def partial_spearman(x, y, ctrl):
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(ctrl)
        A = np.column_stack([np.ones_like(rz), rz])

        def resid(v):
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            return v - A @ coef

        ex, ey = resid(rx), resid(ry)
        return float(np.corrcoef(ex, ey)[0, 1])

    rho_best = spearman(-bb, s)
    rho_pool = spearman(-bp, s)
    part_best = partial_spearman(-bb, s, th)
    part_pool = partial_spearman(-bp, s, th)
    ctx.claim("g2_rho_best_layer", "P2-F-G2-ASSOCIATION", "/rho_H2_best_layer",
              "rho_H2_best_layer", rho_best, res["rho_H2_best_layer"],
              "h2/results_h2.json (terminal)",
              "Spearman(-beta_best, s_hat) over saved 48-model table",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_rho_pooled", "P2-F-G2-ASSOCIATION", "/rho_H2_pooled", "rho_H2_pooled",
              rho_pool, res["rho_H2_pooled"], "h2/results_h2.json (terminal)",
              "Spearman(-beta_pooled, s_hat) over saved 48-model table",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_partial_spearman_ctrl_theta_best", "P2-F-G2-SENSITIVITY",
              "/partial_spearman_ctrl_theta_best", "partial_spearman_ctrl_theta_best",
              part_best, res["partial_spearman_ctrl_theta_best"],
              "h2/results_h2.json (terminal)",
              "rank-residual partial Spearman of (-beta_best, s_hat) controlling theta_hat",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_partial_spearman_ctrl_theta_pooled_internal", "P2-F-G2-SENSITIVITY",
              "(internal)", "partial_spearman_ctrl_theta_pooled", part_pool,
              part_pool, "h2 saved table replays the same routine",
              "rank-residual partial Spearman for the pooled beta column",
              "reaggregated_from_saved_per_model_outputs",
              limitations=["not a ledger claim field; recorded for transparency"])
    fams = sorted(set(fam))
    if len(fams) != 14:
        raise ContractRefusal("h2 family count != 14: %d" % len(fams))
    per_fam, vals = {}, []
    for f in fams:
        keep = np.array([i for i, ff in enumerate(fam) if ff != f])
        r = spearman(-bb[keep], s[keep])
        per_fam[f] = float(r)
        vals.append(r)
    vals = np.array(vals, float)
    G = len(fams)
    se = float(np.sqrt((G - 1.0) / G * np.sum((vals - vals.mean()) ** 2)))
    ci = [float(rho_best - 1.96 * se), float(rho_best + 1.96 * se)]
    jk = res["jackknife_leave_one_family"]
    ctx.claim("g2_jackknife_rho_full", "P2-F-G2-ASSOCIATION",
              "/jackknife_leave_one_family/rho_full", "rho_full", rho_best, jk["rho_full"],
              "h2/results_h2.json (terminal)", "delete-one-family jackknife full rho",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_jackknife_se", "P2-F-G2-ASSOCIATION", "/jackknife_leave_one_family/se",
              "se", se, jk["se"], "h2/results_h2.json (terminal)",
              "block jackknife SE ((G-1)/G)*sum((rho_-g - mean)^2) over 14 families",
              "reaggregated_from_saved_per_model_outputs")
    for i, side in enumerate(("lo", "hi")):
        ctx.claim("g2_jackknife_ci_%s" % side, "P2-F-G2-ASSOCIATION",
                  "/jackknife_leave_one_family/ci_1.96se/%d" % i, "ci_" + side,
                  ci[i], jk["ci_1.96se"][i], "h2/results_h2.json (terminal)",
                  "rho_full +- 1.96*se", "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_jackknife_ci_excludes_0", "P2-F-G2-ASSOCIATION",
              "/jackknife_leave_one_family/ci_excludes_0", "ci_excludes_0",
              bool(ci[0] > 0 or ci[1] < 0), jk["ci_excludes_0"],
              "h2/results_h2.json (terminal)", "interval sign check",
              "reaggregated_from_saved_per_model_outputs", exact=True)
    ctx.claim("g2_jackknife_min_loo", "P2-F-G2-ASSOCIATION",
              "/jackknife_leave_one_family/min_loo_rho", "min_loo_rho",
              float(vals.min()), jk["min_loo_rho"], "h2/results_h2.json (terminal)",
              "min over leave-one-family rho values",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_jackknife_max_loo", "P2-F-G2-ASSOCIATION",
              "/jackknife_leave_one_family/max_loo_rho", "max_loo_rho",
              float(vals.max()), jk["max_loo_rho"], "h2/results_h2.json (terminal)",
              "max over leave-one-family rho values",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_jackknife_per_family", "P2-F-G2-ASSOCIATION",
              "/jackknife_leave_one_family/rho_without_family", "rho_without_family",
              per_fam, jk["rho_without_family"], "h2/results_h2.json (terminal)",
              "leave-one-family-out Spearman for each of 14 families",
              "reaggregated_from_saved_per_model_outputs", numeric_leaves=True)
    DROP = ["ByteDance-Seed/Seed-OSS-36B-Instruct", "ByteDance-Seed/Seed-OSS-36B-Base-woSyn"]
    keep = np.array([i for i, m in enumerate(mids) if m not in DROP])
    sens = res["sensitivity_drop_2_seedoss"]
    ctx.claim("g2_sensitivity_dropped", "P2-F-G2-SENSITIVITY",
              "/sensitivity_drop_2_seedoss/dropped", "dropped", DROP, sens["dropped"],
              "h2/results_h2.json (terminal)", "ID list exact check",
              "reaggregated_from_saved_per_model_outputs", exact=True)
    ctx.claim("g2_sensitivity_n", "P2-F-G2-SENSITIVITY", "/sensitivity_drop_2_seedoss/n",
              "n", int(len(keep)), sens["n"], "h2/results_h2.json (terminal)",
              "remaining model count", "reaggregated_from_saved_per_model_outputs", exact=True)
    ctx.claim("g2_sensitivity_rho_best", "P2-F-G2-SENSITIVITY",
              "/sensitivity_drop_2_seedoss/rho_H2_best_layer", "rho_H2_best_layer",
              spearman(-bb[keep], s[keep]), sens["rho_H2_best_layer"],
              "h2/results_h2.json (terminal)", "Spearman after dropping the two Seed-OSS models",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_sensitivity_rho_pooled", "P2-F-G2-SENSITIVITY",
              "/sensitivity_drop_2_seedoss/rho_H2_pooled", "rho_H2_pooled",
              spearman(-bp[keep], s[keep]), sens["rho_H2_pooled"],
              "h2/results_h2.json (terminal)", "Spearman after dropping the two Seed-OSS models",
              "reaggregated_from_saved_per_model_outputs")
    with np.load(bloo_npz, allow_pickle=False) as z:
        b_loo = z["b_loo"]
        b_full = z["b_full"]
        bloo_ids = [str(x) for x in z["model_ids"]]
    if bloo_ids != mids:
        raise ContractRefusal("b_loo.npz model order differs from h2 beta table")
    sd_bloo = np.array([float(r["sd_bloo"]) for r in rows])
    observed = getattr(ctx, "g0_observed_items", None)
    if observed is None or set(observed) != set(mids):
        raise ContractRefusal("G0 observed-item mapping unavailable for sd_bloo replay")
    sd_recomputed = np.array([np.std(b_loo[m][sorted(observed[mids[m]])], ddof=0)
                              for m in range(b_loo.shape[0])])
    sanity = res["sanity_beta_vs_neg_s"]
    A = np.column_stack([np.ones_like(s), s])
    coef, *_ = np.linalg.lstsq(A, -bb, rcond=None)
    ctx.claim("g2_sanity_SD_b_full", "P2-F-G2-IDENTITY", "/sanity_beta_vs_neg_s/SD_b_full",
              "SD_b_full", float(np.std(b_full)), sanity["SD_b_full"],
              "h2/results_h2.json (terminal)", "population SD of gauge-fixed b_full (saved npz)",
              "reaggregated_from_saved_resampling_array")
    ctx.claim("g2_sanity_mean_SD_b_loo", "P2-F-G2-IDENTITY",
              "/sanity_beta_vs_neg_s/mean_SD_b_loo", "mean_SD_b_loo",
              float(np.mean(sd_recomputed)), sanity["mean_SD_b_loo"],
              "h2/results_h2.json (terminal)",
              "mean over models of population SD of finite b_loo rows (saved npz)",
              # sd_bloo is the SD of the model's LOO b over that model's own observed items
              "reaggregated_from_saved_resampling_array")
    ctx.claim("g2_sanity_slope", "P2-F-G2-IDENTITY",
              "/sanity_beta_vs_neg_s/slope_negbeta_on_s", "slope_negbeta_on_s",
              float(coef[1]), sanity["slope_negbeta_on_s"], "h2/results_h2.json (terminal)",
              "OLS slope of (-beta_best) on s_hat", "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_sanity_intercept", "P2-F-G2-IDENTITY",
              "/sanity_beta_vs_neg_s/intercept", "intercept", float(coef[0]),
              sanity["intercept"], "h2/results_h2.json (terminal)",
              "OLS intercept of (-beta_best) on s_hat",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_sanity_pearson_r", "P2-F-G2-IDENTITY",
              "/sanity_beta_vs_neg_s/pearson_r", "pearson_r",
              float(np.corrcoef(s, -bb)[0, 1]), sanity["pearson_r"],
              "h2/results_h2.json (terminal)", "pearson(s_hat, -beta_best)",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g2_sd_bloo_column_vs_npz", "P2-F-G2-IDENTITY", "(internal)",
              "max_abs_diff_sd_bloo_column_minus_npz_std",
              float(np.max(np.abs(sd_bloo - sd_recomputed))), 0.0,
              "h2/beta_by_model.csv sd_bloo column vs fit_panel/b_loo.npz",
              "recomputed per-model SD vs saved column",
              "reaggregated_from_saved_resampling_array")
    ctx.values["g2"] = {"rho_H2_best_layer": rho_best, "rho_H2_pooled": rho_pool,
                        "partial_spearman_ctrl_theta_best": part_best,
                        "jackknife_se": se, "jackknife_ci": ci,
                        "sensitivity": {"n": int(len(keep)),
                                        "rho_H2_best_layer": spearman(-bb[keep], s[keep]),
                                        "rho_H2_pooled": spearman(-bp[keep], s[keep])},
                        "sanity": {"slope": float(coef[1]), "intercept": float(coef[0]),
                                   "pearson_r": float(np.corrcoef(s, -bb)[0, 1]),
                                   "SD_b_full": float(np.std(b_full)),
                                   "mean_SD_b_loo": float(np.mean(sd_recomputed))}}
    ctx.timings["g2_s"] = round(time.time() - t0, 2)
    return ctx.values["g2"]


# ----------------------------------------------------------------------- g3
def branch_g3(ctx):
    t0 = time.time()
    geo = ctx.project_root / "ext_P2_geo_20260914/h3"
    r2_path = geo / "r2_eps.csv"
    perm_path = geo / "perm_null.npz"
    loo_path = geo / "eps_loo.npz"
    res_path = geo / "results_h3.json"
    eps_path = ctx.project_root / "ext_P2_geo_20260914/fit_panel/eps_panel.npz"
    for p, role, level in [(r2_path, "g3_r2_eps", "computation_input"),
                           (perm_path, "g3_perm_null", "computation_input"),
                           (loo_path, "g3_eps_loo", "computation_input"),
                           (res_path, "g3_terminal_result", "comparison_only"),
                           (eps_path, "g3_eps_panel", "computation_input")]:
        ctx.consume(p, role, level)
    res = read_json(res_path)
    with r2_path.open() as fh:
        rows = list(csv.DictReader(fh))
    pool = [r for r in rows if int(r["layer"]) == -1]
    if len(pool) != 48:
        raise ContractRefusal("h3 pooled rows != 48: %d" % len(pool))
    r2 = np.array([float(r["r2_oof"]) for r in pool])
    pp = np.array([float(r["p_perm"]) for r in pool])
    pb = np.array([float(r["pointbiserial_eps_y"]) for r in pool])
    frac = res["fractions"]
    ctx.claim("g3_r2_median", "P2-F-G3-RESIDUAL", "/pooled_summary/r2_median",
              "r2_median", float(np.median(r2)), res["pooled_summary"]["r2_median"],
              "h3/results_h3.json (terminal)",
              "median of saved layer==-1 r2_oof rows (48 models)",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g3_pointbiserial_median", "P2-F-G3-CORRECTNESS",
              "/pooled_summary/pointbiserial_median", "pointbiserial_median",
              float(np.median(pb)), res["pooled_summary"]["pointbiserial_median"],
              "h3/results_h3.json (terminal)",
              "median of saved pooled pointbiserial_eps_y column",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g3_pointbiserial_absmax", "P2-F-G3-CORRECTNESS",
              "/pooled_summary/pointbiserial_absmax", "pointbiserial_absmax",
              float(np.max(np.abs(pb))), res["pooled_summary"]["pointbiserial_absmax"],
              "h3/results_h3.json (terminal)",
              "max |pointbiserial_eps_y| over saved pooled rows",
              "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g3_fractions", "P2-F-G3-RESIDUAL", "/fractions",
              "n_models/n_r2_ge_0.05/n_p_lt_0.01/n_both/frac_*",
              (int((r2 >= 0.05).sum()), int((pp < 0.01).sum()),
               int(((r2 >= 0.05) & (pp < 0.01)).sum())),
              (frac["n_r2_ge_0.05"], frac["n_p_lt_0.01"], frac["n_both"]),
              "h3/results_h3.json (terminal)",
              "exact count check with thresholds R2>=0.05 and p_perm<0.01",
              "reaggregated_from_saved_per_model_outputs", exact=True)
    ctx.claim("g3_n_models", "P2-F-G3-RESIDUAL", "/fractions/n_models", "n_models",
              len(pool), frac["n_models"], "h3/results_h3.json (terminal)",
              "model count", "reaggregated_from_saved_per_model_outputs", exact=True)
    for key, val in (("frac_r2_ge_0.05", float((r2 >= 0.05).mean())),
                     ("frac_p_lt_0.01", float((pp < 0.01).mean())),
                     ("frac_both", float(((r2 >= 0.05) & (pp < 0.01)).mean()))):
        ctx.claim("g3_%s" % key, "P2-F-G3-RESIDUAL", "/fractions/%s" % key, key,
                  val, frac[key], "h3/results_h3.json (terminal)",
                  "fraction recomputed from saved pooled rows",
                  "reaggregated_from_saved_per_model_outputs")
    ctx.claim("g3_r2_minmax", "P2-F-G3-RESIDUAL",
              "/pooled_summary/r2_min,/pooled_summary/r2_max", "r2_min/r2_max",
              (float(r2.min()), float(r2.max())),
              (res["pooled_summary"]["r2_min"], res["pooled_summary"]["r2_max"]),
              "h3/results_h3.json (terminal)", "min/max of saved pooled r2_oof",
              "reaggregated_from_saved_per_model_outputs", exact=True)
    # cross-check the resampling arrays: eps_loo vs eps_panel correlation, perm p
    with np.load(loo_path, allow_pickle=False) as z:
        eps_loo = z["eps_loo"]
        mask_loo = z["mask"]
        loo_ids = z["model_ids"]
    with np.load(eps_path, allow_pickle=False) as z:
        eps = z["eps"]
        ids = z["model_ids"]
    with np.load(perm_path, allow_pickle=False) as z:
        perm_p = z["p_perm"]
        perm_obs = z["obs_r2"]
        perm_ids = z["model_ids"]
        perm_draws = int(z["null_r2"].shape[1]) if "null_r2" in z.files else None
    if [str(x) for x in ids] != [str(x) for x in loo_ids] or \
       [str(x) for x in ids] != [str(x) for x in perm_ids]:
        raise ContractRefusal("h3 npz model order mismatch")
    corr_rows = []
    for m in range(eps.shape[0]):
        ok = mask_loo[m] & np.isfinite(eps[m]) & np.isfinite(eps_loo[m])
        corr_rows.append(pearson(eps_loo[m][ok], eps[m][ok]))
    corr_rows = np.array(corr_rows)
    csv_corr = np.array([float(r["corr_eps_loo_full"]) for r in pool])
    ctx.claim("g3_corr_eps_loo_median", "P2-F-G3-RESIDUAL",
              "/pooled_summary/corr_eps_loo_full_median", "corr_eps_loo_full_median",
              float(np.median(corr_rows)), res["pooled_summary"]["corr_eps_loo_full_median"],
              "h3/results_h3.json (terminal)",
              "pearson(eps_loo_m, eps_m) over finite shared items from saved npz arrays",
              "reaggregated_from_saved_resampling_arrays")
    ctx.claim("g3_corr_eps_loo_min", "P2-F-G3-RESIDUAL",
              "/pooled_summary/corr_eps_loo_full_min", "corr_eps_loo_full_min",
              float(np.min(corr_rows)), res["pooled_summary"]["corr_eps_loo_full_min"],
              "h3/results_h3.json (terminal)",
              "min pearson(eps_loo_m, eps_m) from saved npz arrays",
              "reaggregated_from_saved_resampling_arrays")
    ctx.claim("g3_corr_eps_loo_vs_csv", "P2-F-G3-RESIDUAL", "(internal)",
              "max_abs_diff_recomputed_corr_minus_csv_column",
              float(np.max(np.abs(corr_rows - csv_corr))), 0.0,
              "h3/r2_eps.csv corr_eps_loo_full column vs npz arrays",
              "recomputed correlation vs saved column",
              "reaggregated_from_saved_resampling_arrays")
    ctx.claim("g3_perm_obs_vs_csv", "P2-F-G3-RESIDUAL", "(internal)",
              "max_abs_diff_npz_obs_r2_minus_csv_r2_oof",
              float(np.max(np.abs(perm_obs - r2))), 0.0,
              "h3/perm_null.npz obs_r2 vs r2_eps.csv r2_oof",
              "saved permutation bundle consistency check",
              "reaggregated_from_saved_resampling_arrays")
    ctx.claim("g3_perm_p_vs_csv", "P2-F-G3-RESIDUAL", "(internal)",
              "max_abs_diff_npz_p_perm_minus_csv_p_perm",
              float(np.max(np.abs(perm_p - pp))), 0.0,
              "h3/perm_null.npz p_perm vs r2_eps.csv p_perm",
              "saved permutation bundle consistency check",
              "reaggregated_from_saved_resampling_arrays")
    ctx.claim("g3_seeds_metadata", "P2-F-G3-RESIDUAL", "/seeds", "seeds",
              res["seeds"], res["seeds"], "h3/results_h3.json (declared config)",
              "declared seed/scaling metadata", "declared_metadata", exact=True)
    ctx.claim("g3_thresholds_metadata", "P2-F-G3-RESIDUAL", "/thresholds", "thresholds",
              res["thresholds"], res["thresholds"], "h3/results_h3.json (declared config)",
              "declared threshold metadata", "declared_metadata", exact=True)
    ctx.counts["g3_perm_null_draws"] = perm_draws
    ctx.values["g3"] = {"r2_median": float(np.median(r2)),
                        "pointbiserial_median": float(np.median(pb)),
                        "pointbiserial_absmax": float(np.max(np.abs(pb))),
                        "n_models": len(pool), "n_both": int(((r2 >= 0.05) & (pp < 0.01)).sum()),
                        "corr_eps_loo_median": float(np.median(corr_rows)),
                        "corr_eps_loo_min": float(np.min(corr_rows))}
    ctx.timings["g3_s"] = round(time.time() - t0, 2)
    return ctx.values["g3"]


# ------------------------------------------------------------------ prior16
def branch_prior16(ctx):
    t0 = time.time()
    geo = ctx.project_root / "_gpu2_run/Import_results/activation_retrieval/v2_activation_indicators_20260602/indicators_aggregate.csv"
    resp = ctx.project_root / "_gpu2_run/results/response_matrix_code_execution.csv"
    lp = ctx.project_root / "b200_full_pipeline/results/logprob_matrix_16models_code.csv"
    item = ctx.project_root / "b200_full_pipeline/activation_indicators/indicators_tier2_peritem.npz"
    ctrl_path = ctx.analysis_root / "verifier/V_RECOVERED16_CONTROLLER.json"
    for p, role, level in [(geo, "prior16_indicator_aggregate", "computation_input"),
                           (resp, "prior16_response_matrix", "computation_input"),
                           (lp, "prior16_logprob_matrix", "computation_input"),
                           (item, "prior16_peritem_npz", "computation_input"),
                           (ctrl_path, "prior16_controller_comparator", "comparison_only")]:
        ctx.consume(p, role, level)
    ctrl = read_json(ctrl_path)
    cols = ["twoNN_id", "eff_rank_pr", "rankme", "stable_rank", "spectral_alpha",
            "vn_entropy", "isoscore"]
    with geo.open() as fh:
        rows = {a["model"]: a for a in csv.DictReader(fh)
                if a["thinking"] == "OFF" and a["domain"] == "code"
                and int(a["layer_neg"]) == -1}
    models = sorted(rows)
    with resp.open() as fh:
        rr = list(csv.DictReader(fh))
    with lp.open() as fh:
        ll = list(csv.DictReader(fh))
    if not (len(rr) == len(ll) == 170 and len(models) == 16):
        raise ContractRefusal("prior16 shape mismatch: %d/%d/%d" % (len(rr), len(ll), len(models)))
    if [int(a["item_id"]) for a in rr] != list(range(170)):
        raise ContractRefusal("prior16 response item ids are not 0..169")
    accuracy = np.array([np.mean([float(a[m]) for a in rr]) for m in models])
    logprob = np.array([np.mean([float(a[m]) for a in ll]) for m in models])
    X = np.array([[float(rows[m][k]) for k in cols] for m in models])
    Z = (X - X.mean(0)) / X.std(0)
    u, sv, vt = np.linalg.svd(Z, full_matrices=False)
    load = vt[0]
    load = load * (1 if load.sum() >= 0 else -1)
    score = Z @ load
    D = np.column_stack([np.ones(16), rankdata(logprob)])
    q, _ = np.linalg.qr(D)
    rx, ry = rankdata(score), rankdata(accuracy)
    partial = corr_dot(rx - q @ (q.T @ rx), ry - q @ (q.T @ ry))
    raw = corr_dot(rx, ry)
    eigenshare = float(np.sum(sv[:2] ** 2) / np.sum(sv ** 2))
    with np.load(item, allow_pickle=False) as z:
        legend = json.loads(str(z["legend"]))
        arr = {k: z[k] for k in ["thinking", "domain_idx", "layer_neg", "model_idx",
                                 "item_id", "lid"]}
    lid_rows, lid_rhos = [], []
    for m in models:
        mask = ((arr["thinking"] == 0)
                & (arr["domain_idx"] == legend["domains"].index("code"))
                & (arr["layer_neg"] == -1)
                & (arr["model_idx"] == legend["models"].index(m)))
        ids = arr["item_id"][mask]
        lid = arr["lid"][mask]
        if len(ids) != 170 or len(set(ids.tolist())) != 170 or sorted(ids.tolist()) != list(range(170)):
            raise ContractRefusal("prior16 per-item LID mask broken for %s" % m)
        y = np.array([float(rr[int(i)][m]) for i in ids])
        if not np.isfinite(lid).all():
            raise ContractRefusal("prior16 non-finite LID for %s" % m)
        rho = corr_dot(rankdata(lid), rankdata(y))
        lid_rows.append({"model": m, "rho": rho, "n": 170})
        lid_rhos.append(rho)
    lid_rhos = np.array(lid_rhos)
    ctx.claim("prior16_models", "P2-G-002", "(V_RECOVERED16_CONTROLLER)/models",
              "models", models, ctrl["models"], "verifier/V_RECOVERED16_CONTROLLER.json",
              "16-model list from recovered indicator aggregate", "recomputed_from_recovered_inputs",
              exact=True)
    ctx.claim("prior16_raw_rho", "P2-G-002", "(V_RECOVERED16_CONTROLLER)/raw_rho",
              "raw_rho", raw, ctrl["raw_rho"],
              "verifier/V_RECOVERED16_CONTROLLER.json",
              "Spearman(PC1 score of standardized 7 saved indicators, model accuracy) over 16 models",
              "recomputed_from_recovered_inputs")
    ctx.claim("prior16_controlled_rho", "P2-G-003",
              "(V_RECOVERED16_CONTROLLER)/logprob_controlled_rho",
              "logprob_controlled_rho", partial, ctrl["logprob_controlled_rho"],
              "verifier/V_RECOVERED16_CONTROLLER.json",
              "rank-residual partial Spearman controlling rank(mean logprob) over 16 models",
              "recomputed_from_recovered_inputs")
    ctx.claim("prior16_eigenshare", "P2-G-003",
              "(V_RECOVERED16_CONTROLLER)/first_two_eigen_variance_share",
              "first_two_eigen_variance_share", eigenshare,
              ctrl["first_two_eigen_variance_share"],
              "verifier/V_RECOVERED16_CONTROLLER.json",
              "sum(s[:2]^2)/sum(s^2) from SVD of standardized indicator matrix",
              "recomputed_from_recovered_inputs")
    ctx.claim("prior16_lid_rows", "P2-G-002", "(V_RECOVERED16_CONTROLLER)/LID_rows",
              "LID_rows", lid_rows, ctrl["LID_rows"],
              "verifier/V_RECOVERED16_CONTROLLER.json",
              "per-model item-level Spearman(LID, correctness) over 170 items",
              "recomputed_from_recovered_inputs", exact=True)
    ctx.claim("prior16_lid_mean", "P2-G-002", "(V_RECOVERED16_CONTROLLER)/LID_mean",
              "LID_mean", float(np.mean(lid_rhos)), ctrl["LID_mean"],
              "verifier/V_RECOVERED16_CONTROLLER.json", "mean of per-model LID rhos",
              "recomputed_from_recovered_inputs")
    ctx.claim("prior16_lid_range", "P2-G-002", "(V_RECOVERED16_CONTROLLER)/LID_range",
              "LID_range", [float(lid_rhos.min()), float(lid_rhos.max())], ctrl["LID_range"],
              "verifier/V_RECOVERED16_CONTROLLER.json", "min/max of per-model LID rhos",
              "recomputed_from_recovered_inputs")
    ctx.values["prior16"] = {"models": models, "raw_rho": raw,
                             "logprob_controlled_rho": partial,
                             "first_two_eigen_variance_share": eigenshare,
                             "LID_mean": float(np.mean(lid_rhos)),
                             "LID_range": [float(lid_rhos.min()), float(lid_rhos.max())],
                             "LID_rows": lid_rows}
    ctx.timings["prior16_s"] = round(time.time() - t0, 2)
    return ctx.values["prior16"]


# --------------------------------------------------------------- reporting
LEDGER_TARGETS = ["P2-F-POOLING", "P2-F-G0-PANEL", "P2-F-G0-STABILITY",
                  "P2-F-G1-DIFFICULTY", "P2-F-G1-TRANSFER", "P2-F-G2-ASSOCIATION",
                  "P2-F-G2-SENSITIVITY", "P2-F-G2-IDENTITY", "P2-F-G3-RESIDUAL",
                  "P2-F-G3-CORRECTNESS", "P2-G-002", "P2-G-003"]


def _walk_pointer(doc, pointer):
    cur = doc
    for part in pointer.split("/")[1:]:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def branch_ledger(ctx):
    """Read the exact claim fields out of the frozen ledger and check that each
    ledger-listed field value equals the retained terminal artifact it cites
    (the fresh numbers are compared to those same terminals in the other
    branches)."""
    t0 = time.time()
    led_path = ctx.analysis_root / "reports/C17_FINAL_EVIDENCE_LEDGER_v1.json"
    ctx.consume(led_path, "c17_final_evidence_ledger", "pointer_discovery")
    led = read_json(led_path)
    recs = {r["result_id"]: r for r in led["records"]}
    checked, n_fields = [], 0
    for rid in LEDGER_TARGETS:
        rec = recs.get(rid)
        if rec is None:
            raise ContractRefusal("ledger target missing: %s" % rid)
        hm = rec.get("historical_manifest_references") or {}
        for ho in hm.get("historical_outputs", []):
            fields = ho.get("fields") or {}
            if not fields or ho.get("status") != "LOCATED":
                continue
            try:
                path = ctx.resolve_recorded(ho["path"], role="ledger_output:" + rid)
            except MissingRefusal:
                continue
            if path.suffix != ".json":
                continue
            ctx.consume(path, "ledger_source:" + rid, "pointer_discovery")
            doc = read_json(path)
            for ptr, ref in fields.items():
                if not isinstance(ref, (int, float)) or isinstance(ref, bool):
                    continue
                try:
                    actual = _walk_pointer(doc, ptr)
                except (KeyError, IndexError, TypeError):
                    checked.append({"result_id": rid, "path": ho["path"], "pointer": ptr,
                                    "ledger_value": ref, "terminal_value": None,
                                    "status": "POINTER_NOT_IN_TERMINAL"})
                    continue
                n_fields += 1
                st = "PASS" if close_enough(actual, ref) else "FAIL"
                checked.append({"result_id": rid, "path": ho["path"], "pointer": ptr,
                                "ledger_value": ref, "terminal_value": actual,
                                "status": st})
    n_fail = sum(1 for c in checked if c["status"].startswith("FAIL"))
    n_not = sum(1 for c in checked if c["status"] != "PASS" and not c["status"].startswith("FAIL"))
    ctx.ledger_check = {"ledger": str(led_path),
                        "ledger_sha256": sha256_file(led_path),
                        "n_target_records": len(LEDGER_TARGETS),
                        "n_numeric_ledger_fields_checked": n_fields,
                        "n_pass": sum(1 for c in checked if c["status"] == "PASS"),
                        "n_fail": n_fail, "n_not_in_terminal": n_not,
                        "fields": checked}
    ctx.claim("ledger_target_records_present", "ledger", "(ledger)/records",
              "target_result_ids_present", len(LEDGER_TARGETS), len(LEDGER_TARGETS),
              "reports/C17_FINAL_EVIDENCE_LEDGER_v1.json",
              "exact presence check of the 12 target result IDs", "source_check", exact=True)
    ctx.claim("ledger_fields_vs_terminals", "ledger", "(ledger)/historical_outputs/*/fields",
              "numeric_ledger_fields_equal_cited_terminal", n_fail, 0,
              "reports/C17_FINAL_EVIDENCE_LEDGER_v1.json vs cited terminal files",
              "walk each numeric ledger field pointer into its cited terminal artifact",
              "source_check", exact=False)
    ctx.timings["ledger_s"] = round(time.time() - t0, 2)
    return ctx.ledger_check


def build_report(ctx):
    claims = ctx.claims
    n_pass = sum(1 for c in claims if c["status"] == "PASS")
    n_fail = sum(1 for c in claims if c["status"] == "FAIL")
    return {
        "schema": SCHEMA + "-report",
        "node": "C10v",
        "created_utc": ctx.utc,
        "analysis_root": str(ctx.analysis_root),
        "project_root": str(ctx.project_root),
        "out_dir": str(ctx.out_dir),
        "entrypoint": "code/c17_replay_supplements_b/replay.py",
        "code_sha256": sha256_file(SCRIPT_PATH),
        "python": platform.python_version(),
        "scope": {
            "fresh_replay_from_cached_data": True,
            "claim": "Fresh computation from retained saved observations, compared against retained terminals.",
            "not_claimed": [
                "No historical restoration of the original production pipeline.",
                "No new probe fitting; G1/G2/G3 statistics are reaggregations of saved per-model outputs.",
                "No strict no-leakage validation: all-item preprocessing limitations are retained (G1 feature scaling, G3 residual target).",
                "G2 gain-vs-slope is algebraically coupled; a high correlation is not independent causal evidence.",
                "G0 theta is a generation-logprob location, not a validated behavioral IRT theta; b is a logprob item coordinate.",
                "P2-G-002/003 recover point statistics from recovered inputs; the same point does not identify the original generator.",
                "Pooling shares are properties of this saved decomposition, not a universal no-model-effect result.",
            ],
            "start_levels": {
                "pooling": "saved 48 per-model code JSONL cell observations",
                "g0": "saved JSONL cells + saved fitted-parameter/resampling arrays (ALS replay of the panel only)",
                "g1": "saved per-model/per-pair CSV outputs plus the regenerated Task-C pair "
                      "cache (same-estimator reproduction of the retained item bootstrap; "
                      "Task C only, no Task A, no per-layer Task B, no GPU)",
                "g2": "saved per-model beta table + saved b_loo array (no probe refit)",
                "g3": "saved per-model residual outputs + saved eps/permutation arrays (no probe refit)",
                "prior16": "recovered 16-model aggregate indicator table + response/logprob matrices + per-item LID npz",
            },
            "cpu_blas1": True, "gpu_used": False,
        },
        "comparator": {"scalar": "abs(actual-ref) <= 1e-9 + 1e-6*abs(ref)",
                       "exact": "counts, IDs, strings, null masks, declared metadata",
                       "stricter_source_checks": [
                           "expected-manifest hash pinning of every declared input before any computation",
                           "hardcoded historical hashes for inventory, fit_panel outputs, h1/h2/h3 CSVs and terminals, 48 JSONL pooling inputs, recovered-16 inputs",
                       ]},
        "counts": ctx.counts,
        "timings_s": ctx.timings,
        "claims": claims,
        "claim_summary": {"n_claims": len(claims), "n_pass": n_pass, "n_fail": n_fail},
        "uncomputable_fields": ctx.uncomputable,
        "ledger_claim_field_check": getattr(ctx, "ledger_check", None),
        "code_pins": getattr(ctx, "code_pins", []),
        "write_policy": {
            "default": "all outputs under --out; no writes into the analysis root",
            "publish_reports_flag": "--publish-reports (explicit opt-in)",
            "no_report_files_flag": "--no-report-files forces out-only writes",
        },
        "errors": ctx.errors,
        "remap_log": ctx.remap_log,
        "limitations_global": [
            "Replay validates saved-observation arithmetic and aggregation; it does not restore the historical chain.",
            "G1/G2/G3 remain bounded retained exploratory aggregates; no strict no-leakage claim.",
            "h3 per-layer run-2 rerun remains incomplete upstream; only pooled layer==-1 rows are used.",
            "No GPU re-extraction and no new probe fitting were performed.",
        ],
    }


def render_md(report):
    lines = ["# C17 replay supplements B (C10v) - retained existing-data replay", "",
             "Status: fresh cached-data replay; comparison-only against retained terminals.",
             "Not a historical restoration. No probe refit; CPU, BLAS threads=1.",
             "", "## Claim pointer map", "",
             "| claim | result | ledger pointer | fresh | reference | status | scope |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in report["claims"]:
        fv = c["fresh_value"]
        fvs = json.dumps(fv) if not isinstance(fv, (int, float, str)) else str(fv)
        rv = c["reference_value"]
        rvs = json.dumps(rv) if not isinstance(rv, (int, float, str)) else str(rv)
        if len(fvs) > 60:
            fvs = fvs[:57] + "..."
        if len(rvs) > 60:
            rvs = rvs[:57] + "..."
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            c["claim_id"], c["result_id"], c["ledger_pointer"], fvs, rvs,
            c["status"], c["scope_level"]))
    lines += ["", "## Explicit uncomputable fields", ""]
    if report["uncomputable_fields"]:
        for u in report["uncomputable_fields"]:
            lines.append("- %s (%s, %s): %s Retained terminal reference: %s" % (
                u["claim_id"], u["result_id"], u["ledger_pointer"], u["reason"],
                json.dumps(u["retained_terminal_reference"])))
    else:
        lines.append("- none")
    lines += ["", "## Scope, start levels, limitations", ""]
    for k, v in report["scope"]["start_levels"].items():
        lines.append("- start level %s: %s" % (k, v))
    for lim in report["scope"]["not_claimed"]:
        lines.append("- NOT claimed: %s" % lim)
    for lim in report["limitations_global"]:
        lines.append("- limitation: %s" % lim)
    if report["errors"]:
        lines += ["", "## Errors", ""]
        for e in report["errors"]:
            lines.append("- %s" % json.dumps(e))
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-manifest", default=str(DEFAULT_EXPECTED_MANIFEST))
    ap.add_argument("--no-report-files", action="store_true",
                    help="never write into the analysis root (all writes stay under --out)")
    ap.add_argument("--publish-reports", action="store_true",
                    help="explicit opt-in: also write the reports into <R>/reports "
                         "(ignored with --no-report-files)")
    ap.add_argument("--g1-cache-dir", default=None,
                    help="override the Task-C pair-cache directory (default: "
                         "<R>/runs/c17_supplements_b_check_v1/g1_taskc_pair_cache_v1)")
    ap.add_argument("--g1-regen-check-pairs", type=int, default=2,
                    help="bounded representative pair regenerations from hidden states")
    args = ap.parse_args(argv)

    ctx = Context(args.analysis_root, args.out, args.expected_manifest)
    ctx.g1_cache_dir = args.g1_cache_dir
    ctx.g1_regen_check_pairs = max(1, args.g1_regen_check_pairs)
    publish = bool(args.publish_reports) and not args.no_report_files
    if not ctx.analysis_root.is_dir():
        print("REFUSAL: analysis root not a directory: %s" % ctx.analysis_root, file=sys.stderr)
        return ContractRefusal.exit_code
    if not (ctx.analysis_root / "reports/C17_FINAL_EVIDENCE_LEDGER_v1.json").exists():
        print("REFUSAL: --analysis-root does not look like the P2 analysis dir: %s"
              % ctx.analysis_root, file=sys.stderr)
        return ContractRefusal.exit_code
    if ctx.out_dir.exists() and any(ctx.out_dir.iterdir()):
        print("REFUSAL: --out must be an empty/new directory: %s" % ctx.out_dir, file=sys.stderr)
        return ContractRefusal.exit_code
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        verify_expected_manifest(ctx)
    except ReplayRefusal as exc:
        write_json(ctx.out_dir / "refusal.json",
                   {"kind": type(exc).__name__, "message": str(exc),
                    "analysis_root": str(ctx.analysis_root),
                    "expected_manifest": str(ctx.expected_path),
                    "preflight": ctx.preflight, "utc": ctx.utc})
        print("REFUSAL(%s): %s" % (type(exc).__name__, exc), file=sys.stderr)
        return exc.exit_code

    ctx.values = {}
    branches = [("pooling", branch_pooling), ("g0", branch_g0), ("g1", branch_g1),
                ("g2", branch_g2), ("g3", branch_g3), ("prior16", branch_prior16)]
    branches.insert(0, ("ledger", branch_ledger))
    for name, fn in branches:
        try:
            fn(ctx)
        except ReplayRefusal as exc:
            write_json(ctx.out_dir / "refusal.json",
                       {"kind": type(exc).__name__, "message": str(exc), "branch": name,
                        "utc": ctx.utc})
            print("REFUSAL(%s) in branch %s: %s" % (type(exc).__name__, name, exc),
                  file=sys.stderr)
            return exc.exit_code
        except Exception as exc:  # transparent error reporting
            ctx.errors.append({"branch": name, "error": repr(exc),
                               "traceback": traceback.format_exc()})
            print("ERROR in branch %s: %r" % (name, exc), file=sys.stderr)
    # unchanged-source proof comes BEFORE the report is built so a failed check
    # is reflected in report["errors"] and blocks publication.
    after = []
    for row in ctx.preflight["rows"]:
        p = Path(row["path"])
        after.append({"path": row["path"], "before": row.get("actual"),
                      "after": sha256_file(p) if p.exists() else None})
    unchanged = all(r["before"] is not None and r["before"] == r["after"] for r in after)
    write_json(ctx.out_dir / "source_unchanged.json",
               {"unchanged": unchanged, "rows": after})
    if not unchanged:
        ctx.errors.append({"branch": "source_unchanged",
                           "error": "pinned input changed during run; report not published"})
    report = build_report(ctx)
    report["source_manifest"] = {
        "schema": SCHEMA + "-source-manifest",
        "note": "All declared inputs were hash-pinned in the expected manifest; entries are "
                "root-relative (P/R) so the driver is relocation-safe.",
        "expected_manifest": str(ctx.expected_path),
        "expected_manifest_sha256": sha256_file(ctx.expected_path),
        "n_pinned": ctx.preflight["n_entries"],
        "consumed": sorted(ctx.consumed.values(), key=lambda e: e["path"]),
        "preflight_rows": ctx.preflight["rows"],
    }
    report["run_receipt"] = {
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "code_sha256": sha256_file(SCRIPT_PATH),
        "expected_manifest_sha256": sha256_file(ctx.expected_path),
        "started_utc": ctx.utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - ctx.started, 2),
        "cpu_blas1": True,
        "gpu_used": False,
        "branch_timings_s": ctx.timings,
    }
    write_json(ctx.out_dir / "C17_SUPPLEMENTS_B_v1.json", report)
    write_json(ctx.out_dir / "values.json", ctx.values)
    write_json(ctx.out_dir / "comparator_report.json",
               {"claims": ctx.claims,
                "summary": report["claim_summary"],
                "uncomputable_fields": ctx.uncomputable,
                "comparator": report["comparator"]})
    write_json(ctx.out_dir / "SOURCE_MANIFEST.json", report["source_manifest"])
    write_json(ctx.out_dir / "run_receipt.json", report["run_receipt"])
    if publish and unchanged and not ctx.errors:
        # explicit opt-in only; default runs never write into the analysis root
        write_json(ctx.analysis_root / REPORT_JSON, report)
        (ctx.analysis_root / REPORT_MD).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"out": str(ctx.out_dir), "claims": report["claim_summary"],
                      "uncomputable": len(ctx.uncomputable), "errors": ctx.errors,
                      "source_unchanged": unchanged, "published_to_analysis_root": bool(
                          publish and unchanged and not ctx.errors),
                      "wall_s": report["run_receipt"]["wall_s"]}))
    if ctx.errors:
        return 6
    if report["claim_summary"]["n_fail"]:
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
