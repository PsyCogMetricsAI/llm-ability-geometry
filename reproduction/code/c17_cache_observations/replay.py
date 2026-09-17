#!/usr/bin/env python3
"""C17 cached-observation replay (C10o): recompute paper-2 C17 outputs from saved
per-model/per-item observations and stored draw arrays; never re-extract, never
re-fit, never touch a signed run dir.

Branches
  new28   : rebuild CT_C17_NEW28_v1 point associations from the contract-bound
            saved model inputs (features/theta/covariates/observations) and the
            inference summaries from the stored permutation/bootstrap draw
            arrays; reassemble F1/F2 native+rarefied 28-cell grids and the
            declared combined-56 secondary summary (BH/BY recomputed from the
            fresh p values).
  windows : recompute the six-model window clouds chi (W8/W32/W128) from the
            saved per-item window vectors with the frozen consumer arithmetic
            (code/window_cloud_metrics.py), plus effective lengths and the
            cross-model rank rho.  No GPU, no weights, no writes to signed runs.
  probes  : freshly aggregate the bounded retained G1/G2/G3 probe medians /
            rhos / counts consumed by P2-F-TAB-PROBES from the per-model numeric
            observation tables (CSV), not from terminal result JSONs.

Everything is comparison-only against the signed references; the comparator
uses abs <= 1e-9 + 1e-6*abs(ref) for scalars and exact equality for counts,
IDs, status strings and null masks.  The window consumer's scientific 0.2
stability flag is NOT used as a numeric tolerance anywhere.

Usage:
  python code/c17_cache_observations/replay.py --analysis-root <R> --out <NEW_DIR>
         [--branch all|new28|windows|probes] [--representative N] [--workers N]
         [--no-report-files] [--selftest]
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ANALYSIS_ROOT = SCRIPT_PATH.parents[2]
SCHEMA = "c17-cache-observations-replay-v1"

# Known original roots as recorded (absolute) inside the signed contract / run
# manifests at freeze time.  Remapping is done by RELATIVE SUFFIX against these
# prefixes, so a relocated project may have any basename.
KNOWN_ORIGINAL_PROJECT_ROOT = SCRIPT_PATH.parents[2].parent
KNOWN_ORIGINAL_ANALYSIS_ROOT = KNOWN_ORIGINAL_PROJECT_ROOT / "repro_paper2_20260915"

# ---------------------------------------------------------------- frozen pins
CONTRACT_SHA256 = "2b83b1dca91b5e9d3143177dc863a85c4cfc17c5dfc5aa80021f5be8ae2d097a"
PRIMITIVE_SHA256 = "8ac43d46c2738bd993850a07cc3ad1c33a4ede41a02d273c40e05ca7947eb614"
GRID_MODULE_SHA256 = "51a8486a3055f9b37ca2c3ba5b7077f3ba16d5d8f042fa503cbb8fdf580c6cac"
NEW28_RUNNER_SHA256 = "42282ecf2f729e3102aac8e5832ae96ec43f881f58f8b8307f5cfc6053e72c8c"

DOMAINS = ["science", "medical", "code", "math"]
STAGES = ["native", "rarefied"]
METRICS = ["twoNN_id", "eff_rank_pr", "rankme", "stable_rank", "spectral_alpha",
           "vn_entropy", "isoscore"]
METRIC_ORDER_SOURCE = "H4BC_INPUT_ARRAYS.json.metric_order"
F1 = "F1_MAP"
F2 = "F2_RASCH"
FAMILIES = [F1, F2]

B_TOTAL = 10000
SEED_PERMUTATION = 20260605
SEED_FAMILY_BOOTSTRAP = 20260605
EXCEED_SLACK = 1e-12
MAX_INVALID_CONDITIONAL = 500
MIN_CI_VALID = 9000
CI_PCTS = (2.5, 97.5)
ALPHA_LEVELS = (0.05, 0.10)
N_ITEMS_CONTEXT = {"science": 448, "medical": 1273, "code": 591, "math": 1333}
REASON_CODES = {0: "valid", 1: "constant_input", 2: "rank_deficient_design",
                3: "zero_residual_rank", 4: "zero_residual_norm"}

WINDOW_MODELS = ["smol", "qwen14", "qwen32", "pythia", "gemma", "llama"]
WINDOW_SIZES = [8, 32, 128]
WINDOW_K = 10
SCALAR_ATOL = 1e-9
SCALAR_RTOL = 1e-6


class ReplayRefusal(RuntimeError):
    exit_code = 2


class HashRefusal(ReplayRefusal):
    exit_code = 3


class MissingRefusal(ReplayRefusal):
    exit_code = 4


class ContractRefusal(ReplayRefusal):
    exit_code = 5



def provenance_identity(recorded_path):
    """Canonical root-relative identity for a recorded provenance path.

    Applies the same declared-root mapping to both the fresh and the reference
    field so a relocated analysis root compares equal, while the recorded
    sha256 for every observation entry is still compared exactly.
    """
    text = str(recorded_path)
    analysis_name = Path(str(KNOWN_ORIGINAL_ANALYSIS_ROOT)).name
    # already-tokenised values: P/<analysis-name>/rest is the same identity as R/rest;
    # a staged absolute path is reduced to the suffix after the analysis directory name.
    if text.startswith("P/" + analysis_name + "/"):
        return "R/" + text[len("P/" + analysis_name + "/"):]
    if text.startswith(("P/", "R/")):
        text = ({"P": "P", "R": "R"}[text[0]]) + "/" + text[2:]
        return text
    marker = "/" + analysis_name + "/"
    if marker in text:
        return "R/" + text.rsplit(marker, 1)[1]
    p = Path(text)
    for known, token in ((KNOWN_ORIGINAL_PROJECT_ROOT, "P"), (KNOWN_ORIGINAL_ANALYSIS_ROOT, "R")):
        known_s = str(known)
        if p.is_absolute() and (text == known_s or text.startswith(known_s + os.sep)):
            return token + "/" + str(Path(text).relative_to(known_s))
    parts = p.parts
    for marker in ("Imports", "b200_full_pipeline", "_gpu2_run", "ext_P2_geo_20260914",
                   "runs", "config", "reports", "code", "verifier", "inputs_manifest"):
        if marker in parts:
            idx = len(parts) - 1 - parts[::-1].index(marker)
            token = "P" if marker in ("Imports", "b200_full_pipeline", "_gpu2_run") else "R"
            return token + "/" + "/".join(parts[idx:])
    return text

def load_expected_inputs(path, analysis_root):
    """Load an expected-source manifest ({"files": [...]} or a bare list) and
    resolve every entry against the caller-supplied roots: entries are
    {relativepath, root (R/P), sha256}; a resolved abspath is also accepted."""
    doc = read_json(path)
    entries = doc["files"] if isinstance(doc, dict) and "files" in doc else doc
    analysis_root = Path(analysis_root).resolve()
    project_root = analysis_root.parent
    out = {}
    for entry in entries:
        root = entry.get("root", "R")
        if entry.get("relativepath"):
            target = (analysis_root if root == "R" else project_root) / entry["relativepath"]
        elif entry.get("abspath"):
            target = Path(entry["abspath"])
            if not target.is_absolute():
                target = (analysis_root if root == "R" else project_root) / target
        else:
            base = analysis_root if root == "R" else project_root
            target = base / entry["relativepath"]
        target = target.resolve()
        if not (target == analysis_root or target.is_relative_to(analysis_root)
                or target == project_root or target.is_relative_to(project_root)):
            raise ReplayRefusal("expected-inputs entry escapes the caller roots: %s" % target)
        out[target] = entry["sha256"]
    return out


def verify_expected_inputs(expected, consumed=None):
    """Pre-flight verification for a supplied expected-source manifest: every
    pinned file must exist and match; tamper refuses before any computation."""
    rows, missing, mismatched = [], [], []
    for path, sha in sorted(expected.items(), key=lambda kv: str(kv[0])):
        if not path.exists():
            missing.append(str(path))
            rows.append({"path": str(path), "expected": sha, "status": "MISSING"})
            continue
        actual = sha256_file(path)
        ok = actual == sha
        rows.append({"path": str(path), "expected": sha, "actual": actual,
                     "status": "OK" if ok else "HASH_MISMATCH"})
        if not ok:
            mismatched.append({"path": str(path), "expected": sha, "actual": actual})
    if missing:
        raise MissingRefusal("expected-inputs manifest lists missing files: %s"
                             % ", ".join(missing[:5]))
    if mismatched:
        raise HashRefusal("expected-inputs manifest hash mismatch: %s"
                          % ", ".join(m["path"] for m in mismatched[:5]))
    return {"n_entries": len(expected), "rows": rows,
            "missing": missing, "mismatched": mismatched, "all_ok": True}


# ---------------------------------------------------------------------- utils
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Context:
    def __init__(self, analysis_root, out_dir, branch, representative, workers,
                 report_json, report_md, contract_path=None, expected_inputs=None):
        self.analysis_root = Path(analysis_root).resolve()
        self.project_root = self.analysis_root.parent
        self.out_dir = Path(out_dir).resolve()
        self.branch = branch
        self.representative = representative
        self.workers = max(1, min(2, workers))
        self.report_json = report_json
        self.report_md = report_md
        self.contract_path_override = contract_path
        self.expected_inputs = expected_inputs or {}
        self.consumed = []
        self._consumed_seen = {}
        self.path_skips = []
        self.started = time.time()
        self.refusals = []

    # -- path handling / portability -------------------------------------
    def _is_under_caller_roots(self, path):
        path = Path(path).resolve()
        return (path == self.analysis_root or path.is_relative_to(self.analysis_root)
                or path == self.project_root or path.is_relative_to(self.project_root))

    def _remap_candidates(self, recorded):
        """Prefer candidates under the caller-supplied roots, mapped by relative
        suffix from the known original roots (basename-independent), then the
        basename-marker fallback, and only then the recorded path itself (which
        is used solely if it resolves inside the caller roots)."""
        p = Path(recorded)
        cands = []

        def add(cand):
            cand = Path(cand)
            if cand not in cands:
                cands.append(cand)

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
                idx = s.find(marker)
                add(self.project_root / s[idx + len(marker):])
            add(p)
        elif str(p).startswith("R/"):
            add(self.analysis_root / str(p)[2:])
        elif str(p).startswith("P/"):
            add(self.project_root / str(p)[2:])
        else:
            add(self.analysis_root / p)
            add(self.project_root / p)
        return cands

    def resolve_recorded(self, recorded, note=""):
        """Resolve a recorded path portably against the given analysis root.

        Candidates under the caller-supplied R/P win over the recorded absolute
        path; the recorded path is used only when it resolves inside the caller
        roots.  Candidate containment is checked with Path.is_relative_to (never
        string prefix), so sibling-prefix paths are refused.  Out-of-root
        existing candidates are skipped, not fatal, and only a full miss refuses.
        """
        skipped = []
        for cand in self._remap_candidates(recorded):
            if not cand.exists():
                continue
            resolved = cand.resolve()
            if not self._is_under_caller_roots(resolved):
                skipped.append(str(resolved))
                continue
            if note and Path(recorded).is_absolute() and str(resolved) != str(Path(recorded).resolve()):
                self.refusals.append({"kind": "path_remapped", "recorded": str(recorded),
                                      "resolved": str(resolved), "note": note})
            return resolved
        if skipped:
            self.path_skips.append({"recorded": str(recorded), "out_of_root": skipped,
                                    "note": note})
        raise MissingRefusal(
            "recorded path unavailable under analysis root %s: %s (out-of-root candidates "
            "refused/skipped: %s)" % (self.analysis_root, recorded, skipped or "none"))

    # -- consumed-file registry ------------------------------------------
    def consume(self, path, role, level, root=None, expected_sha256=None, note=""):
        path = Path(path).resolve()
        cached = self._consumed_seen.get(path)
        if cached is not None:
            if expected_sha256 is not None and cached["sha256"] != expected_sha256:
                raise HashRefusal("hash mismatch for %s (role=%s): expected %s, actual %s"
                                  % (path, role, expected_sha256, cached["sha256"]))
            return cached["sha256"]
        if not path.exists():
            raise MissingRefusal("input file missing (role=%s): %s" % (role, path))
        pinned = self.expected_inputs.get(path)
        if pinned is not None and expected_sha256 is None:
            expected_sha256 = pinned
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise HashRefusal("hash mismatch for %s (role=%s): expected %s, actual %s"
                              % (path, role, expected_sha256, digest))
        if pinned is not None and digest != pinned:
            raise HashRefusal("expected-inputs manifest mismatch for %s (role=%s): expected %s, "
                              "actual %s" % (path, role, pinned, digest))
        if root is None:
            if str(path).startswith(str(self.analysis_root)):
                root = "R"
                rel = str(path.relative_to(self.analysis_root))
            else:
                root = "P"
                rel = str(path.relative_to(self.project_root))
        else:
            base = self.analysis_root if root == "R" else self.project_root
            rel = str(path.relative_to(base))
        if path not in self._consumed_seen:
            entry = {"relativepath": rel, "root": root, "abspath": str(path),
                     "sha256": digest, "role": role, "level": level}
            if note:
                entry["note"] = note
            self._consumed_seen[path] = entry
            self.consumed.append(entry)
        return digest

    def write_json(self, relpath, obj):
        target = (self.out_dir / relpath).resolve()
        if target != self.out_dir and self.out_dir not in target.parents:
            raise ReplayRefusal("write outside the fresh output root: %s" % target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(obj, indent=2, allow_nan=False, sort_keys=False) + "\n",
                          encoding="utf-8")
        return target

    def write_text(self, relpath, text):
        target = (self.out_dir / relpath).resolve()
        if target != self.out_dir and self.out_dir not in target.parents:
            raise ReplayRefusal("write outside the fresh output root: %s" % target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target


def _stats_arrays(entry):
    return {k: v for k, v in entry.items() if isinstance(v, (int, float, str))}


def rel_diff(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


class Comparator:
    """Signed-reference comparator: scalars tol = 1e-9 + 1e-6*|ref|; everything
    else (counts, IDs, statuses, null masks, orderings) exact."""

    def __init__(self):
        self.rows = []
        self.ref_files = {}

    def ref(self, ctx, path, pointer, role, level="comparison_only"):
        path = Path(path).resolve()
        digest = ctx.consume(path, role, level)
        self.ref_files[str(path)] = digest
        return path, digest

    def scalar(self, branch, name, fresh, ref, pointer, ref_sha=None):
        if fresh is None or ref is None:
            ok = fresh is None and ref is None
            self.rows.append({"branch": branch, "field": name, "kind": "null_mask",
                              "fresh": fresh, "reference": ref, "pass": ok,
                              "reference_pointer": pointer, "reference_sha256": ref_sha})
            return ok
        tol = SCALAR_ATOL + SCALAR_RTOL * abs(float(ref))
        delta = abs(float(fresh) - float(ref))
        ok = bool(delta <= tol)
        self.rows.append({"branch": branch, "field": name, "kind": "scalar",
                          "fresh": float(fresh), "reference": float(ref),
                          "abs_diff": delta, "tolerance": tol, "pass": ok,
                          "reference_pointer": pointer, "reference_sha256": ref_sha})
        return ok

    def exact(self, branch, name, fresh, ref, pointer, ref_sha=None, kind="exact"):
        ok = fresh == ref
        self.rows.append({"branch": branch, "field": name, "kind": kind,
                          "fresh": fresh, "reference": ref, "pass": bool(ok),
                          "reference_pointer": pointer, "reference_sha256": ref_sha})
        return ok

    def array(self, branch, name, fresh, ref, pointer, ref_sha=None):
        fresh = [None if v is None else float(v) for v in fresh]
        ref = [None if v is None else float(v) for v in ref]
        if len(fresh) != len(ref):
            self.rows.append({"branch": branch, "field": name, "kind": "array",
                              "fresh_len": len(fresh), "reference_len": len(ref),
                              "pass": False, "reference_pointer": pointer,
                              "reference_sha256": ref_sha})
            return False
        worst = 0.0
        ok = True
        for a, b in zip(fresh, ref):
            if a is None or b is None:
                if a is not b:
                    ok = False
                continue
            tol = SCALAR_ATOL + SCALAR_RTOL * abs(b)
            delta = abs(a - b)
            worst = max(worst, delta)
            if delta > tol:
                ok = False
        self.rows.append({"branch": branch, "field": name, "kind": "array",
                          "n": len(fresh), "max_abs_diff": worst, "pass": bool(ok),
                          "reference_pointer": pointer, "reference_sha256": ref_sha})
        return ok

    def summary(self):
        failed = [r for r in self.rows if not r["pass"]]
        return {"n_comparisons": len(self.rows), "n_pass": len(self.rows) - len(failed),
                "n_fail": len(failed),
                "failed_fields": [{"branch": r["branch"], "field": r["field"],
                                   "kind": r["kind"],
                                   "fresh": r.get("fresh"), "reference": r.get("reference"),
                                   "reference_pointer": r.get("reference_pointer")}
                                  for r in failed[:40]],
                "tolerance_rule": "scalar abs <= 1e-9 + 1e-6*abs(ref); counts/IDs/"
                                  "statuses/null masks exact",
                "window_scientific_flag_0.2_used_as_tolerance": False}


# ------------------------------------------------- frozen numeric primitives
def pearson(x, y):
    a, b = np.asarray(x, float), np.asarray(y, float)
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def design(c):
    c = np.asarray(c, float)
    if c.ndim == 1:
        c = c[:, None]
    return np.column_stack([np.ones(len(c))] + [rankdata(c[:, j]) for j in range(c.shape[1])])


def rho(x, y, c=None):
    a, b = rankdata(x), rankdata(y)
    if c is not None:
        d = design(c)
        a = a - d @ np.linalg.lstsq(d, a, rcond=None)[0]
        b = b - d @ np.linalg.lstsq(d, b, rcond=None)[0]
    return pearson(a, b)


def draws_for_panel(n_families, n, b_total, seed_perm, seed_boot):
    rng = np.random.default_rng(seed_boot)
    family_draws = np.array([rng.choice(n_families, size=n_families, replace=True)
                             for _ in range(b_total)])
    rng = np.random.default_rng(seed_perm)
    permutation_indices = np.array([rng.permutation(n) for _ in range(b_total)])
    return family_draws, permutation_indices


def _nan_reason(d, values):
    rd = np.linalg.matrix_rank(d)
    for v in values:
        if np.linalg.matrix_rank(np.column_stack([d, v])) == rd:
            return 3
    return 4


def bootstrap_reasoned(x, y, c, families, draws):
    family_order = sorted(set(families.tolist()))
    blocks = [np.where(families == f)[0] for f in family_order]
    values = np.full(len(draws), np.nan)
    reasons = np.zeros(len(draws), dtype=np.int8)
    for row, choices in enumerate(draws):
        idx = np.concatenate([blocks[k] for k in choices])
        xb, yb = x[idx], y[idx]
        if len(idx) < 3 or len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            reasons[row] = 1
            continue
        cb = None if c is None else c[idx]
        if cb is not None:
            cc = cb[:, None] if cb.ndim == 1 else cb
            if any(len(np.unique(cc[:, j])) < 2 for j in range(cc.shape[1])):
                reasons[row] = 1
                continue
            d = design(cc)
            if np.linalg.matrix_rank(d) < d.shape[1]:
                reasons[row] = 2
                continue
        else:
            d = design(np.arange(len(xb)))[:, :1] * 0.0
        value = rho(xb, yb, cb)
        if np.isfinite(value):
            values[row] = value
        else:
            values[row] = np.nan
            d = design(cb) if cb is not None else design(np.zeros(len(xb)))
            reasons[row] = _nan_reason(d, (rankdata(xb), rankdata(yb)))
    return values, reasons


def permutation_reasoned(x, y, c, draws):
    observed = rho(x, y, c)
    values = np.full(len(draws), np.nan)
    reasons = np.zeros(len(draws), dtype=np.int8)
    d = design(c) if c is not None else None
    for k, idx in enumerate(draws):
        value = rho(x, y[idx], c)
        if np.isfinite(value):
            values[k] = value
        else:
            values[k] = np.nan
            if d is None:
                reasons[k] = _nan_reason(design(np.zeros(len(x))),
                                         (rankdata(x), rankdata(y[idx])))
            else:
                reasons[k] = _nan_reason(d, (rankdata(x), rankdata(y[idx])))
    return observed, values, reasons


def reason_counts(reasons):
    out = {name: 0 for name in REASON_CODES.values()}
    for code, name in REASON_CODES.items():
        out[name] = int(np.sum(reasons == code))
    return out


def percentile_ci(values, reasons):
    finite = values[np.isfinite(values)]
    counts = reason_counts(reasons)
    if len(finite) < MIN_CI_VALID:
        return {"lo": None, "hi": None, "n_valid": int(len(finite)),
                "n_invalid": int(len(values) - len(finite)), "reasons": counts,
                "null_reason": "B_valid_below_%d" % MIN_CI_VALID}
    lo, hi = np.percentile(finite, CI_PCTS).tolist()
    return {"lo": float(lo), "hi": float(hi), "n_valid": int(len(finite)),
            "n_invalid": int(len(values) - len(finite)), "reasons": counts,
            "null_reason": None}


def bh_stepup(p):
    p = np.array(p, float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        out[i] = min(prev, 1.0)
    return out.tolist()


def by_adjust(p):
    m = len(p)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return [min(1.0, v * harmonic) for v in bh_stepup(p)]


def cell_gate(x, y, c):
    checks = {
        "x_finite": bool(np.isfinite(x).all()),
        "y_finite": bool(np.isfinite(y).all()),
        "controls_finite": bool(np.isfinite(c).all()),
        "x_two_distinct": bool(len(np.unique(x)) >= 2),
        "y_two_distinct": bool(len(np.unique(y)) >= 2),
    }
    checks["design_rank"] = int(np.linalg.matrix_rank(design(c))) if checks["controls_finite"] else None
    checks["design_rank_ok"] = checks["design_rank"] == design(c).shape[1]
    checks["pass"] = all(checks[k] for k in
                         ("x_finite", "y_finite", "controls_finite", "x_two_distinct",
                          "y_two_distinct", "design_rank_ok"))
    return checks


def rasch_canonicalization(rasch_Y, map_Y, rasch_theta, map_theta):
    total = np.asarray(rasch_Y, float).sum(axis=0)
    levels = np.unique(total)
    spreads = np.array([np.ptp(rasch_theta[total == level]) for level in levels])
    group_means = np.array([rasch_theta[total == level].mean() for level in levels])
    rank_corr = float(np.corrcoef(rankdata(rasch_theta), rankdata(total))[0, 1])
    checks = {
        "Y_identical_rasch_vs_map": bool(np.array_equal(rasch_Y, map_Y)),
        "unique_totals": int(len(levels)),
        "unique_totals_equals_n": bool(len(levels) == len(total)),
        "max_within_total_EAP_spread": float(spreads.max()) if len(spreads) else None,
        "group_means_strictly_increasing": bool(np.all(np.diff(group_means) > 0)),
        "rank_correlation_total_vs_rasch_EAP_ranks": rank_corr,
        "n_theta_map_finite": int(np.isfinite(map_theta).sum()),
        "n_theta_rasch_finite": int(np.isfinite(rasch_theta).sum()),
    }
    ok = (checks["Y_identical_rasch_vs_map"] and checks["unique_totals_equals_n"]
          and checks["max_within_total_EAP_spread"] is not None
          and checks["max_within_total_EAP_spread"] < 1e-8
          and checks["group_means_strictly_increasing"]
          and abs(rank_corr - 1.0) < 1e-12
          and checks["n_theta_map_finite"] == len(total)
          and checks["n_theta_rasch_finite"] == len(total))
    checks["status"] = "PASS" if ok else "FAIL"
    checks["total_scores"] = total.astype(int).tolist()
    return checks


def posterior_reliability(theta, variance):
    """Original producer formula (new_medical_repair*.py, study2_reuse_fit.py):
    rel = var(theta, ddof=1) / (var(theta, ddof=1) + mean(variance)).
    Regenerated on the accepted saved score arrays; no recorded aggregate is
    copied into a fresh value."""
    theta = np.asarray(theta, float)
    variance = np.asarray(variance, float)
    var_theta = float(np.var(theta, ddof=1))
    return float(var_theta / (var_theta + float(np.mean(variance))))


# ------------------------------------------------------------------- new28
# Declared input roles for the CT_C17_NEW28_v1 computation inputs.  The signed
# contract is unchanged; this producer records metadata/aggregate-only reads
# differently: only statistics sources feed numbers; provenance records are
# never used as a numerical calculation input (no old terminal aggregate is
# copied into a fresh value).
NEW28_INPUT_ROLES = {
    "features_H4BC": ("statistics_source:geometry_features", "computation_input"),
    "theta_panel51": ("statistics_source:code_math_theta", "computation_input"),
    "covariates_50": ("statistics_source:covariates", "computation_input"),
    "science_fit_arrays": ("statistics_source:science_theta_and_posterior_grids",
                           "computation_input"),
    "medical_map_A_npz": ("statistics_source:medical_MAP_theta_variance",
                          "computation_input"),
    "medical_rasch_A_npz": ("statistics_source:medical_Rasch_theta_variance",
                            "computation_input"),
    "medical_hist_matrix_npz": ("identity_pin:A45_response_matrix", "declared_identity"),
    "observations_receipt": ("metadata_index:observation_hash_lock", "provenance_only"),
    "science_fit_json": ("provenance:science_fit_status_counts", "provenance_only"),
    "science_fit_receipt": ("provenance:science_fit_output_hashes", "provenance_only"),
    "medical_refined_summary": ("provenance:medical_acceptance_flags_recorded_aggregates",
                               "provenance_only"),
    "medical_repair_summary": ("provenance:medical_rasch_acceptance_recorded_aggregates",
                              "provenance_only"),
    "medical_hist_fit_json": ("provenance:historical_failure_record", "provenance_only"),
    "medical_hist_inputs_json": ("metadata:medical_tags_item_counts", "provenance_only"),
    "machinery_permute_module": ("code_dependency:frozen_rho_permute_bootstrap",
                                 "code_dependency"),
    "machinery_grid_module": ("code_dependency:frozen_grid_draw_scheme", "code_dependency"),
}
DEFAULT_DECLARED_ROLE = ("declared_identity_pin", "declared_identity")


class New28:
    def __init__(self, ctx):
        self.ctx = ctx
        self.run_root = ctx.analysis_root / "runs/c17_new28_v1"

    def load_contract(self):
        path = self.ctx.contract_path_override or (self.ctx.analysis_root / "config/CT_C17_NEW28_v1.json")
        path = Path(path)
        if not path.exists():
            raise MissingRefusal("contract not found: %s" % path)
        digest = self.ctx.consume(path, "new28_contract", "computation_input",
                                  root="R" if str(path).startswith(str(self.ctx.analysis_root)) else "P")
        if self.ctx.contract_path_override is None and digest != CONTRACT_SHA256:
            raise HashRefusal("CT_C17_NEW28_v1.json changed: expected %s, actual %s"
                              % (CONTRACT_SHA256, digest))
        contract = read_json(path)
        self.contract_sha = digest
        self.contract = contract
        self.contract_path = path
        for name in ("machinery_permute_module", "machinery_grid_module"):
            desc = contract["sources"]["computation_inputs"][name]
            p = self.ctx.resolve_recorded(desc["path"], note=name)
            role, level = NEW28_INPUT_ROLES.get(name, DEFAULT_DECLARED_ROLE)
            self.ctx.consume(p, "new28_input:%s:%s" % (name, role), level,
                             expected_sha256=desc["sha256"])
        return contract

    def verify_inputs(self):
        contract = self.contract
        verified = []
        for name, desc in contract["sources"]["computation_inputs"].items():
            p = self.ctx.resolve_recorded(desc["path"], note="new28:%s" % name)
            role, level = NEW28_INPUT_ROLES.get(name, DEFAULT_DECLARED_ROLE)
            digest = self.ctx.consume(p, "new28_input:%s:%s" % (name, role), level,
                                      expected_sha256=desc["sha256"])
            verified.append({"name": name, "path": str(p), "sha256": digest,
                             "role": role, "level": level,
                             "pin_verified_only": level == "declared_identity",
                             "declared_role": desc.get("role", "")})
        return verified

    def observation_rows(self, domain, tags):
        cin = self.contract["sources"]["computation_inputs"]
        receipt_path = self.ctx.resolve_recorded(cin["observations_receipt"]["path"],
                                                 note="observation receipt")
        locked = {row["cell"]: row for row in read_json(receipt_path)["cells"]}
        rows, refs = [], []
        obs_dir = self.ctx.analysis_root / "runs/study2_reuse_v1/observations"
        for tag in tags:
            key = tag + "|" + domain
            if key not in locked:
                raise ContractRefusal("observation missing for %s" % key)
            path = (obs_dir / ("%s__%s.json" % (tag, domain))).resolve()
            recorded = Path(locked[key]["output"])
            if recorded.name != path.name:
                raise ContractRefusal("observation path identity failure for %s: %s"
                                      % (key, recorded))
            digest = self.ctx.consume(path, "new28_observation:%s" % key, "computation_input",
                                      expected_sha256=locked[key]["sha256"])
            rows.append(read_json(path))
            refs.append({"cell": key, "path": str(path), "sha256": digest})
        return rows, refs

    def build_domain_inputs(self):
        ctx, cin = self.ctx, self.contract["sources"]["computation_inputs"]
        feature_file = self.ctx.resolve_recorded(cin["features_H4BC"]["path"], note="features")
        features = read_json(feature_file)
        if features["metric_order"] != METRICS:
            raise ContractRefusal("metric order changed in %s" % feature_file)
        theta_path = self.ctx.resolve_recorded(cin["theta_panel51"]["path"], note="theta51")
        theta_panel = read_json(theta_path)
        cov_path = self.ctx.resolve_recorded(cin["covariates_50"]["path"], note="covariates")
        covariates = read_json(cov_path)["models"]
        science_fit_path = self.ctx.resolve_recorded(cin["science_fit_json"]["path"], note="science_fit")
        science_fit = read_json(science_fit_path)
        science_receipt_path = self.ctx.resolve_recorded(cin["science_fit_receipt"]["path"],
                                                         note="science_receipt")
        science_receipt = read_json(science_receipt_path)
        for name, digest in science_receipt["outputs"].items():
            p = Path(science_receipt_path).parent / name
            if not p.exists():
                raise MissingRefusal("science fit receipt listed a missing file: %s" % p)
            self.ctx.consume(p, "new28_science_receipt_output:%s" % name, "computation_input",
                             expected_sha256=digest)
        mr_path = self.ctx.resolve_recorded(cin["medical_refined_summary"]["path"], note="mr_summary")
        medical_refined = read_json(mr_path)["variants"]["medical_A"]
        mp_path = self.ctx.resolve_recorded(cin["medical_repair_summary"]["path"], note="mp_summary")
        medical_repair = read_json(mp_path)["variants"]["medical_A"]
        map_npz_path = self.ctx.resolve_recorded(cin["medical_map_A_npz"]["path"], note="map_npz")
        rasch_npz_path = self.ctx.resolve_recorded(cin["medical_rasch_A_npz"]["path"], note="rasch_npz")
        map_scores = np.load(map_npz_path)
        rasch_scores = np.load(rasch_npz_path)
        hist_fit_path = self.ctx.resolve_recorded(cin["medical_hist_fit_json"]["path"], note="hist_fit")
        hist = read_json(hist_fit_path)

        out = {}
        for domain in DOMAINS:
            dom = features["domains"][domain]
            tags = list(dom["tags"])
            rows, obs_refs = self.observation_rows(domain, tags)
            native = np.array(dom["native"], float)
            rarefied = np.array(dom["rarefied"], float)
            if native.shape != (len(tags), 7) or rarefied.shape != (len(tags), 7):
                raise ContractRefusal("feature shape failure: %s" % domain)
            controls, families = [], []
            for tag, row in zip(tags, rows):
                ref = covariates[tag].get("code") or covariates[tag]["math"]
                if domain in ("code", "math"):
                    d_model = covariates[tag][domain]["d_model"]
                    fluency = covariates[tag][domain]["fluency_mean_logprob"]
                else:
                    d_model = row["d_model"]
                    fluency = row["fluency_nonnull_mean"]
                controls.append([ref["scale_log10_params"], d_model, ref["family_code"], fluency])
                families.append(ref["family"])
            controls = np.array(controls, float)
            entry = {"domain": domain, "tags": tags, "native": native, "rarefied": rarefied,
                     "controls": controls, "families": np.array(families),
                     "n_models": len(tags), "n_families": len(set(families)),
                     "n_items_context": N_ITEMS_CONTEXT[domain],
                     "rel_native": list(dom["rel_native"]), "rel_rarefied": list(dom["rel_rarefied"]),
                     "comment": {"features": str(feature_file), "covariates": str(cov_path)},
                     "observation_refs": obs_refs,
                     "feature_file": str(feature_file),
                     "feature_sha256": cin["features_H4BC"]["sha256"],
                     "covariates_path": str(cov_path),
                     "covariates_sha256": cin["covariates_50"]["sha256"],
                     "control_columns": ["scale_log10_params", "d_model", "family_code", "fluency"]}
            if domain in ("code", "math"):
                if not set(tags) <= set(theta_panel[domain].keys()):
                    raise ContractRefusal("%s theta key-set mismatch" % domain)
                theta = np.array([theta_panel[domain][t] for t in tags], float)
                entry.update({
                    "theta": theta, "theta_variant": "panel51_mirt_%s" % domain,
                    "theta_source": {"path": str(theta_path),
                                     "locator": "['%s'][tag] (order = feature tags)" % domain,
                                     "sha256": cin["theta_panel51"]["sha256"]},
                    "rel_th": float(theta_panel["rel"][domain]),
                    "source_converged": bool(theta_panel["converged"][domain]),
                    "n_items_theta_source": int(theta_panel["n_items"][domain]),
                    "n_items_used_after_fit_mask": None})
            elif domain == "science":
                sci_inputs_path = Path(science_fit_path).parent / "inputs.json"
                if not sci_inputs_path.exists():
                    raise MissingRefusal("science fit inputs.json missing: %s" % sci_inputs_path)
                self.ctx.consume(sci_inputs_path, "new28_metadata:science_fit_inputs_tags",
                                 "provenance_only")
                fit_tags = read_json(sci_inputs_path)["tags"]
                if fit_tags != tags or science_fit["tags"] != tags:
                    raise ContractRefusal("science tag order mismatch")
                sci_arrays_path = self.ctx.resolve_recorded(cin["science_fit_arrays"]["path"],
                                                           note="science_arrays")
                sci_arrays = np.load(sci_arrays_path)
                theta = np.asarray(sci_arrays["theta"], float)
                recorded_rel = float(science_fit["rel_th"])
                rel_regen = posterior_reliability(sci_arrays["posterior_mean_grid61"],
                                                  sci_arrays["posterior_var_grid61"])
                entry.update({
                    "theta": theta, "theta_variant": "science_A girth 2PL EAP",
                    "theta_source": {"path": str(sci_arrays_path),
                                     "locator": "['theta']", "sha256": cin["science_fit_arrays"]["sha256"]},
                    "rel_th": rel_regen,
                    "rel_th_provenance": {
                        "source": "regenerated: posterior_reliability("
                                  "posterior_mean_grid61, posterior_var_grid61) from "
                                  "science_fit_arrays.npz",
                        "recorded_aggregate_crosscheck": {
                            "path": str(science_fit_path), "pointer": "/rel_th",
                            "recorded_value": recorded_rel,
                            "regenerated_value": rel_regen,
                            "abs_diff": abs(recorded_rel - rel_regen),
                            "recorded_value_used_as_input": False}},
                    "source_status": science_fit["status"],
                    "n_items_theta_source": int(science_fit["n_items_total"]),
                    "n_items_used_after_fit_mask": int(science_fit["n_items_fit"])})
            else:
                med_inputs_path = self.ctx.resolve_recorded(cin["medical_hist_inputs_json"]["path"],
                                                            note="medical_fit_inputs")
                med_inputs = read_json(med_inputs_path)
                fit_tags = med_inputs["tags"]
                if not (medical_refined["tags"] == tags == medical_repair["tags"]
                        == fit_tags == hist["tags"]):
                    raise ContractRefusal("medical tag order mismatch across features, "
                                          "fit inputs.json, refined/repair summaries and the "
                                          "historical failure record")
                map_fit = next(f for f in medical_refined["fits"] if f["label"] == "medical_A_map_1.0")
                rasch_fit = next(f for f in medical_repair["fits"] if f["label"] == "medical_A_rasch_1.0")
                if not (map_fit["score_acceptance_candidate"] and rasch_fit["score_acceptance_candidate"]):
                    raise ContractRefusal("medical score acceptance flags lost")
                hist_nonfinite = int(sum(1 for v in hist["theta"]
                                         if v is None or not isinstance(v, (int, float))
                                         or not math.isfinite(v)))
                if hist["status"] != "FAILED_CRITERION_NONFINITE_THETA" or hist_nonfinite != 24:
                    raise ContractRefusal("historical medical fit record changed")
                y_map = np.asarray(map_scores["theta"], float)
                y_rasch = np.asarray(rasch_scores["theta"], float)
                canon = rasch_canonicalization(np.asarray(rasch_scores["Y"]),
                                               np.asarray(map_scores["Y"]), y_rasch, y_map)
                if canon["status"] != "PASS":
                    raise ContractRefusal("Rasch equivalence check failed: %s" % canon)
                rel_map_regen = posterior_reliability(map_scores["theta"], map_scores["variance"])
                rel_rasch_regen = posterior_reliability(rasch_scores["theta"],
                                                        rasch_scores["variance"])
                recorded_rel_map = float(map_fit["posterior_reliability"])
                recorded_rel_rasch = float(rasch_fit["posterior_reliability"])
                entry.update({
                    "theta_map": y_map,
                    "theta_rasch": np.asarray(rankdata(canon["total_scores"], method="average"), float),
                    "theta_variant_map": "medical_A_map_1.0 (refined regularized 2PL MAP EAP, A45)",
                    "theta_variant_rasch": "medical_A_rasch_1.0 canonical exact total-score ranks (A45)",
                    "theta_source_map": {"path": str(map_npz_path), "locator": "['theta']",
                                         "sha256": cin["medical_map_A_npz"]["sha256"]},
                    "theta_source_rasch": {"path": str(rasch_npz_path),
                                           "locator": "rankdata(['Y'].sum(axis=0), method='average')",
                                           "sha256": cin["medical_rasch_A_npz"]["sha256"]},
                    "rasch_equivalence": {k: v for k, v in canon.items() if k != "total_scores"},
                    "rel_th_map": rel_map_regen,
                    "rel_th_rasch": rel_rasch_regen,
                    "rel_th_provenance": {
                        "source": "regenerated: posterior_reliability(theta, variance) from the "
                                  "accepted medical fit NPZ intermediates (A45)",
                        "recorded_aggregate_crosscheck": {
                            "map": {"path": str(mr_path), "pointer": "/variants/medical_A/fits/"
                                                              "medical_A_map_1.0/posterior_reliability",
                                    "recorded_value": recorded_rel_map,
                                    "regenerated_value": rel_map_regen,
                                    "abs_diff": abs(recorded_rel_map - rel_map_regen),
                                    "recorded_value_used_as_input": False},
                            "rasch": {"path": str(mp_path), "pointer": "/variants/medical_A/fits/"
                                                                "medical_A_rasch_1.0/posterior_reliability",
                                      "recorded_value": recorded_rel_rasch,
                                      "regenerated_value": rel_rasch_regen,
                                      "abs_diff": abs(recorded_rel_rasch - rel_rasch_regen),
                                      "recorded_value_used_as_input": False}}},
                    "n_items_theta_source": int(hist["n_items_total"]),
                    "n_items_used_after_fit_mask": int(medical_refined["shape"][0])})
            for array_name in ("theta", "theta_map", "theta_rasch", "native", "rarefied", "controls"):
                if array_name in entry and not np.isfinite(np.asarray(entry[array_name], float)).all():
                    raise ContractRefusal("non-finite bound array %s in %s" % (array_name, domain))
            out[domain] = entry
        return out

    def jobs(self, inputs):
        jobs = []
        for stage in STAGES:
            for domain in DOMAINS:
                for mi in range(len(METRICS)):
                    jobs.append(self._job(inputs, F1, stage, domain, mi))
                    if domain == "medical":
                        jobs.append(self._job(inputs, F2, stage, domain, mi))
        return jobs

    def _job(self, inputs, family, stage, domain, metric_index, shared=None):
        entry = inputs[domain]
        metric = METRICS[metric_index]
        theta_key = "theta"
        if domain == "medical":
            theta_key = "theta_map" if family == F1 else "theta_rasch"
        theta_variant = entry.get("theta_variant") or (
            entry["theta_variant_map"] if family == F1 else entry["theta_variant_rasch"])
        theta_source = entry.get("theta_source") or (
            entry["theta_source_map"] if family == F1 else entry["theta_source_rasch"])
        rel_th = entry.get("rel_th")
        if domain == "medical":
            rel_th = entry["rel_th_map"] if family == F1 else entry["rel_th_rasch"]
        key = "%s__%s__%s__%s" % (family, stage, domain, metric)
        n_items = {
            "context": entry["n_items_context"],
            "theta_source": entry.get("n_items_theta_source"),
            "used_after_fit_mask": entry.get("n_items_used_after_fit_mask"),
            "note": "context = source/context count from the frozen parameters; theta_source and "
                    "used_after_fit_mask are the actual fitted counts where recorded",
        }
        return {
            "key": key, "family": family, "stage": stage, "domain": domain, "metric": metric,
            "x": entry[stage][:, metric_index].copy(),
            "y": np.asarray(entry[theta_key], float).copy(),
            "controls": entry["controls"], "families": entry["families"],
            "theta_variant": theta_variant, "theta_source": theta_source,
        "rel_geo": entry["rel_" + stage][metric_index], "rel_th": rel_th,
            "rel_th_provenance": entry.get("rel_th_provenance"),
            "n_items": n_items, "b_total": B_TOTAL,
            "seed_permutation": SEED_PERMUTATION, "seed_family_bootstrap": SEED_FAMILY_BOOTSTRAP,
            "shared_computation_from": shared, "draws_dir": self.run_root / "draws",
            "draws_file": "draws/%s.npz" % key,
            "hashes": {
                "feature_file": entry["feature_file"],
                "feature_sha256": entry["feature_sha256"],
                "covariates": entry["covariates_path"],
                "covariates_sha256": entry["covariates_sha256"],
                "theta_source_path": theta_source["path"],
                "theta_source_sha256": theta_source["sha256"],
                "observations": entry["observation_refs"],
                "cached_replay_code_sha256": sha256_file(SCRIPT_PATH),
                "machinery_primitive_sha256": PRIMITIVE_SHA256,
                "machinery_grid_sha256": GRID_MODULE_SHA256,
                "contract_sha256": self.contract_sha,
                "reference_runner_sha256": NEW28_RUNNER_SHA256,
            },
        }

    def summarize_from_stored(self, job, stored):
        """Inference summary recomputed from the stored draw/statistic arrays."""
        x = np.asarray(stored["x"], float)
        y = np.asarray(stored["theta"], float)
        controls = np.asarray(stored["controls"], float)
        families = np.asarray(stored["families"], dtype=str)
        perm = np.asarray(stored["permutation"], float)
        perm_reasons = np.asarray(stored["permutation_reasons"], dtype=np.int8)
        cond = np.asarray(stored["controlled_bootstrap"], float)
        cond_reasons = np.asarray(stored["controlled_reasons"], dtype=np.int8)
        bare = np.asarray(stored["bare_bootstrap"], float)
        bare_reasons = np.asarray(stored["bare_reasons"], dtype=np.int8)
        observed = rho(x, y, controls)
        finite = np.isfinite(perm)
        b_valid = int(finite.sum())
        b_invalid = int(len(perm) - b_valid)
        exceed = int(np.sum(np.abs(perm[finite]) >= abs(observed) - EXCEED_SLACK))
        p_lower = (exceed + 1) / (job["b_total"] + 1)
        p_upper = (exceed + (job["b_total"] - b_valid) + 1) / (job["b_total"] + 1)
        if b_valid == 0 or b_invalid > MAX_INVALID_CONDITIONAL:
            status, p_cond, p_exact = "DEGENERATE_DENOMINATOR", None, None
        else:
            p_cond = (exceed + 1) / (b_valid + 1)
            p_exact = "%d/%d" % (exceed + 1, b_valid + 1)
            status = "VALID_DENOMINATOR" if b_invalid == 0 else "CONDITIONAL_VALID"
        return {"observed": float(observed), "B_valid": b_valid, "E_exceed": exceed,
                "p_lower": float(p_lower), "p_upper": float(p_upper),
                "p_cond": p_cond, "p_cond_exact": p_exact, "status": status,
                "ci_controlled": percentile_ci(cond, cond_reasons),
                "ci_bare": percentile_ci(bare, bare_reasons),
                "B_invalid_reasons": reason_counts(perm_reasons),
                "n_families": len(set(families.tolist())),
                "permutation_draw_array_length": int(len(stored["permutation_indices"])),
                "bootstrap_draw_array_length": int(len(stored["family_draws"]))}

    def run(self, comparator):
        t0 = time.time()
        ctx = self.ctx
        self.load_contract()
        verified_inputs = self.verify_inputs()
        inputs = self.build_domain_inputs()
        jobs = self.jobs(inputs)

        cells = {}
        cell_notes = {}
        stored_mismatch = []
        draws_verified = []
        for job in jobs:
            key = job["key"]
            src_key = job["shared_computation_from"] or key
            draws_path = self.run_root / "draws" / (src_key + ".npz")
            draws_path = ctx.resolve_recorded(str(draws_path), note="draws:%s" % src_key)
            digest = ctx.consume(draws_path, "new28_stored_draws:%s" % src_key,
                                 "computation_input")
            stored = np.load(draws_path)
            x = job["x"].astype(float).copy()
            sd = x.std(ddof=0)
            x = (x - x.mean()) / sd if sd > 0 else x - x.mean()
            mismatches = []
            if not np.array_equal(np.asarray(stored["x"], float), x):
                mismatches.append("x")
            if not np.array_equal(np.asarray(stored["theta"], float), np.asarray(job["y"], float)):
                mismatches.append("theta")
            if not np.array_equal(np.asarray(stored["controls"], float),
                                  np.asarray(job["controls"], float)):
                mismatches.append("controls")
            if not np.array_equal(np.asarray(stored["families"], dtype=str),
                                  np.asarray(job["families"], dtype=str)):
                mismatches.append("families")
            if mismatches:
                stored_mismatch.append({"key": key, "draws_file": src_key,
                                        "fields": mismatches})
            rec = {
                "key": key, "family": job["family"], "stage": job["stage"],
                "domain": job["domain"], "metric": job["metric"],
                "n_models": int(len(job["y"])), "n_families": None,
                "theta_variant": job["theta_variant"], "theta_source": job["theta_source"],
                "hashes": job["hashes"], "rel_geo": job["rel_geo"], "rel_th": job["rel_th"],
                "rel_th_provenance": job.get("rel_th_provenance"),
                "n_items": job["n_items"], "B_total": job["b_total"],
                "draws_file": "draws/%s.npz" % src_key,
                "shared_computation_from": job["shared_computation_from"],
                "seed_permutation": job["seed_permutation"],
                "seed_family_bootstrap": job["seed_family_bootstrap"],
                "draws_sha256": digest,
                "statistic_arrays": {"permutation": "draws/%s.npz[permutation]" % src_key,
                                     "controlled_bootstrap":
                                         "draws/%s.npz[controlled_bootstrap]" % src_key,
                                     "bare_bootstrap": "draws/%s.npz[bare_bootstrap]" % src_key},
                "input_gate": cell_gate(x, job["y"].astype(float), job["controls"]),
                "cached_replay": {"source": "stored_draws+rebuilt_inputs",
                                  "stored_inputs_match": not mismatches},
            }
            summary = self.summarize_from_stored(job, stored)
            raw_rho = float(rho(x, job["y"].astype(float)))
            rec.update({
                "raw_rho": raw_rho, "partial_rho": summary["observed"],
                "ci_controlled": summary["ci_controlled"], "ci_bare": summary["ci_bare"],
                "B_valid": summary["B_valid"],
                "B_invalid_reasons": summary["B_invalid_reasons"],
                "E_exceed": summary["E_exceed"], "p_cond": summary["p_cond"],
                "p_cond_exact": summary["p_cond_exact"],
                "p_lower": summary["p_lower"], "p_upper": summary["p_upper"],
                "bh28_cond": None, "by28_cond": None, "bh28_worst": None,
                "bh28_best": None, "by28_worst": None, "by28_best": None,
                "status": summary["status"],
                "notes": "" if summary["B_valid"] == job["b_total"]
                         else "invalid permutation draws recorded with reason codes; bounds reported",
                "permutation_draw_array_length": summary["permutation_draw_array_length"],
                "permutation_statistic_array_length": int(len(stored["permutation"])),
                "bootstrap_draw_array_length": summary["bootstrap_draw_array_length"],
                "bootstrap_statistic_array_length": int(len(stored["controlled_bootstrap"])),
                "n_families": summary["n_families"],
            })
            if job["shared_computation_from"]:
                rec["notes"] = ("non-medical F1/F2 identical inputs: shared computation and draw arrays "
                                "by hash; no duplicate 21-cell recomputation (root resolution A)")
            cells[key] = rec
            if not rec["cached_replay"]["stored_inputs_match"]:
                cell_notes.setdefault("stored_input_mismatch", []).append(key)
            draws_verified.append({"key": key, "draws": src_key, "sha256": digest})

        # F2 non-medical cells are the F1 computation bound by hash (root resolution A):
        # no duplicate recomputation, identical fresh values, explicit provenance.
        for stage in STAGES:
            for domain in ("science", "code", "math"):
                for metric in METRICS:
                    f1_key = "%s__%s__%s__%s" % (F1, stage, domain, metric)
                    f2_key = "%s__%s__%s__%s" % (F2, stage, domain, metric)
                    shared = json.loads(json.dumps(cells[f1_key]))
                    shared["key"] = f2_key
                    shared["family"] = F2
                    shared["shared_computation_from"] = f1_key
                    shared["notes"] = ("non-medical F1/F2 identical inputs: shared computation and "
                                       "draw arrays by hash; no duplicate 21-cell recomputation "
                                       "(root resolution A)")
                    cells[f2_key] = shared

        grids = {}
        for family in FAMILIES:
            for stage in STAGES:
                grids["%s__%s" % (family, stage)] = self.assemble_grid(cells, family, stage)
        combined = {}
        for family in FAMILIES:
            combined[family] = self.combined_secondary(grids, family)

        checks = self.representative_checks(inputs, jobs)

        # fresh outputs
        for key, rec in sorted(cells.items()):
            ctx.write_json("new28/cells/%s.json" % key, rec)
        grid_files = {"%s-%s-28.json" % (family.split("_")[0], stage): grids["%s__%s" % (family, stage)]
                      for family in FAMILIES for stage in STAGES}
        for name, grid in grid_files.items():
            ctx.write_json("new28/grids/%s" % name, grid)
        ctx.write_json("new28/grids/combined56_secondary.json", combined)
        check = {
            "schema": "c17-cache-observations-new28-v1",
            "code_sha256": sha256_file(SCRIPT_PATH),
            "contract": {"path": str(self.contract_path), "sha256": self.contract_sha},
            "verified_computation_inputs": verified_inputs,
            "source_roles": {
                name: {"role": role, "level": level}
                for name, (role, level) in sorted(NEW28_INPUT_ROLES.items())},
            "source_role_policy": [
                "statistics sources (level computation_input) are the only numerical inputs: "
                "geometry features, theta panels/arrays, covariates, accepted medical NPZ "
                "intermediates.",
                "provenance_only reads (refined/repair summaries, science fit JSON/receipt, "
                "historical failure record, observation hash-lock receipt, fit inputs tags) "
                "provide metadata/status/acceptance flags only; no aggregate from them feeds a "
                "fresh value.",
                "posterior reliabilities in fresh cells are regenerated from the saved "
                "posterior mean/variance (science) and theta/variance (medical) arrays with the "
                "original producer formula; recorded aggregate values are cross-checked and "
                "marked recorded_value_used_as_input=false.",
                "declared_identity pins are hash-verified only and are not read as statistics.",
                "The signed contract CT_C17_NEW28_v1.json is unchanged; this producer records "
                "metadata-only reads differently.",
            ],
            "n_cell_records": len(cells),
            "n_stored_draw_files": len({d["draws"] for d in draws_verified}),
            "stored_input_mismatches": stored_mismatch,
            "representative_full_generator_checks": checks,
            "grids": {name: {"release_status": g["release_status"], "m": g["m"],
                             "missing_cells": g["missing_cells"],
                             "crossing_counts_at_0.05": g["crossing_counts_at_0.05"]}
                      for name, g in grid_files.items()},
            "combined56_secondary": {f: {"m": combined[f]["m"], "complete": combined[f]["complete"],
                                         "crossing_counts_at_0.05": combined[f]["crossing_counts_at_0.05"]}
                                     for f in FAMILIES},
            "boundaries": [
                "Medical F1 uses the ACCEPTED upstream regularized-2PL MAP scores (A45) as saved inputs; "
                "they are not fresh fits and are not relabelled obsolete.",
                "source_converged=false for code/math panel51 theta is preserved as a code/math "
                "convergence limitation, not silently upgraded.",
                "BH/BY are descriptive multiplicity summaries only; no calibrated FDR guarantee.",
                "Invalid permutation draws keep the frozen reason codes and worst/best bounds.",
            ],
            "seconds": round(time.time() - t0, 3),
        }
        ctx.write_json("new28/new28_check.json", check)
        self.compare_cells(cells, comparator)
        self.compare_grids(grid_files, comparator)
        self.compare_combined(combined, comparator)
        return check

    def ref_cell(self, ctx, key):
        path = ctx.analysis_root / "runs/c17_new28_v1/cells" / (key + ".json")
        return path, ctx.consume(path, "new28_reference_cell:%s" % key, "comparison_only")

    def ref_grid(self, ctx, name):
        path = ctx.analysis_root / "runs/c17_new28_v1/grids" / name
        return path, ctx.consume(path, "new28_reference_grid:%s" % name, "comparison_only")

    def ref_combined(self, ctx):
        path = ctx.analysis_root / "runs/c17_new28_v1/grids/combined56_secondary.json"
        sha = ctx.consume(path, "new28_reference_combined56", "comparison_only")
        return path, sha

    def compare_combined(self, combined, comparator):
        path, sha = self.ref_combined(self.ctx)
        ref = read_json(path)
        base = "R:runs/c17_new28_v1/grids/combined56_secondary.json"
        for family in FAMILIES:
            for field in ("bh56_cond", "by56_cond", "bh56_worst", "bh56_best"):
                comparator.array("new28", "combined56/%s/%s" % (family, field),
                                 combined[family].get(field), ref[family].get(field),
                                 "%s#/%s/%s" % (base, family, field), sha)
            for field in ("release_status", "complete", "m", "cell_order",
                          "crossing_counts_at_0.05", "crossing_counts_at_0.10"):
                comparator.exact("new28", "combined56/%s/%s" % (family, field),
                                 combined[family].get(field), ref[family].get(field),
                                 "%s#/%s/%s" % (base, family, field), sha)

    def compare_cells(self, cells, comparator):
        ctx = self.ctx
        for key, rec in sorted(cells.items()):
            path, sha = self.ref_cell(ctx, key)
            ref = read_json(path)
            base = "R:runs/c17_new28_v1/cells/%s.json" % key
            for field in ("raw_rho", "partial_rho", "p_cond", "p_lower", "p_upper",
                          "rel_geo", "rel_th"):
                comparator.scalar("new28", "%s/%s" % (key, field), rec.get(field),
                                  ref.get(field), "%s#/%s" % (base, field), sha)
            for field in ("B_total", "B_valid", "E_exceed", "status", "p_cond_exact",
                          "n_models", "n_families", "permutation_draw_array_length",
                          "permutation_statistic_array_length", "bootstrap_draw_array_length",
                          "bootstrap_statistic_array_length", "draws_file",
                          "shared_computation_from", "theta_variant"):
                comparator.exact("new28", "%s/%s" % (key, field), rec.get(field),
                                 ref.get(field), "%s#/%s" % (base, field), sha)
            comparator.exact("new28", "%s/B_invalid_reasons" % key, rec["B_invalid_reasons"],
                             ref["B_invalid_reasons"], "%s#/B_invalid_reasons" % base, sha)
            comparator.exact("new28", "%s/input_gate" % key, rec["input_gate"],
                             ref["input_gate"], "%s#/input_gate" % base, sha)
            for ci in ("ci_controlled", "ci_bare"):
                for f in ("lo", "hi"):
                    comparator.scalar("new28", "%s/%s/%s" % (key, ci, f),
                                      rec[ci].get(f), ref[ci].get(f),
                                      "%s#/%s/%s" % (base, ci, f), sha)
                for f in ("n_valid", "n_invalid", "null_reason"):
                    comparator.exact("new28", "%s/%s/%s" % (key, ci, f),
                                     rec[ci].get(f), ref[ci].get(f),
                                     "%s#/%s/%s" % (base, ci, f), sha)
                comparator.exact("new28", "%s/%s/reasons" % (key, ci),
                                 rec[ci]["reasons"], ref[ci]["reasons"],
                                 "%s#/%s/reasons" % (base, ci), sha)
            for f in ("feature_sha256", "covariates_sha256", "theta_source_sha256"):
                comparator.exact("new28", "%s/hashes/%s" % (key, f),
                                 rec["hashes"][f], ref["hashes"].get(f),
                                 "%s#/hashes/%s" % (base, f), sha)
            # path-equivalence fix: compare the declared root-relative identity of the
            # provenance path on both sides (cell + sha256 still compared exactly, no
            # item dropped, no digest relaxed, arrays never compared to themselves).
            comparator.exact(
                "new28", "%s/hashes/observations" % key,
                [{"cell": o.get("cell"), "sha256": o.get("sha256"),
                  "path": provenance_identity(o.get("path"))}
                 for o in rec["hashes"]["observations"]],
                [{"cell": o.get("cell"), "sha256": o.get("sha256"),
                  "path": provenance_identity(o.get("path"))}
                 for o in ref["hashes"]["observations"]],
                "%s#/hashes/observations" % base, sha)
            comparator.exact("new28", "%s/draws_sha256" % key, rec["draws_sha256"],
                             ref["draws_sha256"], "%s#/draws_sha256" % base, sha)

    def compare_grids(self, grid_files, comparator):
        ctx = self.ctx
        for name, grid in sorted(grid_files.items()):
            path, sha = self.ref_grid(ctx, name)
            ref = read_json(path)
            base = "R:runs/c17_new28_v1/grids/%s" % name
            for field in ("release_status", "m", "cell_order", "missing_cells"):
                comparator.exact("new28", "grid:%s/%s" % (name, field), grid.get(field),
                                 ref.get(field), "%s#/%s" % (base, field), sha)
            for field in ("bh_cond", "by_cond", "bh_worst", "bh_best", "by_worst", "by_best"):
                comparator.array("new28", "grid:%s/%s" % (name, field), grid.get(field),
                                 ref.get(field), "%s#/%s" % (base, field), sha)
            for field in ("crossing_counts_at_0.05", "crossing_counts_at_0.10"):
                comparator.exact("new28", "grid:%s/%s" % (name, field), grid.get(field),
                                 ref.get(field), "%s#/%s" % (base, field), sha)
            comparator.exact("new28", "grid:%s/cell_keys" % name,
                             [c["key"] for c in grid["cells"]],
                             [c["key"] for c in ref["cells"]],
                             "%s#/cells/key" % base, sha)

    def assemble_grid(self, cells, family, stage):
        rows = [cells["%s__%s__%s__%s" % (family, stage, domain, metric)]
                for domain in DOMAINS for metric in METRICS]
        complete = all(row["p_cond"] is not None for row in rows)
        bh_cond = bh_stepup([row["p_cond"] for row in rows]) if complete else None
        by_cond = by_adjust([row["p_cond"] for row in rows]) if complete else None
        bh_worst = bh_stepup([row["p_upper"] for row in rows]) if complete else None
        by_worst = by_adjust([row["p_upper"] for row in rows]) if complete else None
        bh_best = bh_stepup([row["p_lower"] for row in rows]) if complete else None
        by_best = by_adjust([row["p_lower"] for row in rows]) if complete else None
        for i, row in enumerate(rows):
            row["bh28_cond"] = bh_cond[i] if bh_cond else None
            row["by28_cond"] = by_cond[i] if by_cond else None
            row["bh28_worst"] = bh_worst[i] if bh_worst else None
            row["by28_worst"] = by_worst[i] if by_worst else None
            row["bh28_best"] = bh_best[i] if bh_best else None
            row["by28_best"] = by_best[i] if by_best else None
        crossing = {}
        for alpha in ALPHA_LEVELS:
            entry = {"bh_cond": None, "by_cond": None, "robust_both_bounds": None}
            if bh_cond:
                entry["bh_cond"] = int(sum(1 for v in bh_cond if v < alpha))
                entry["by_cond"] = int(sum(1 for v in by_cond if v < alpha))
            if bh_best and bh_worst:
                entry["robust_both_bounds"] = int(sum(1 for b, w in zip(bh_best, bh_worst)
                                                     if b < alpha and w < alpha))
            crossing["%.2f" % alpha] = entry
        return {
            "family": family, "stage": stage, "m": len(rows),
            "release_status": "RELEASED_COMPLETE_28" if complete else "BOUNDS_ONLY_NOT_RELEASED",
            "cells": rows,
            "cell_order": "domain-major (%s) then metric order (%s)" % (",".join(DOMAINS),
                                                                       METRIC_ORDER_SOURCE),
            "bh_cond": bh_cond, "by_cond": by_cond, "bh_worst": bh_worst, "bh_best": bh_best,
            "by_worst": by_worst, "by_best": by_best,
            "crossing_counts_at_0.05": crossing["0.05"],
            "crossing_counts_at_0.10": crossing["0.10"],
            "missing_cells": [row["key"] for row in rows if row["p_cond"] is None],
            "dependency_note": ("BH validity conditions (independence/PRDS) are not established here; "
                                "BY (BH x H_m, capped at 1) is the arbitrary-dependence summary; none of "
                                "these values is released as a calibrated FDR guarantee."),
        }

    def combined_secondary(self, grids, family):
        rows = grids["%s__native" % family]["cells"] + grids["%s__rarefied" % family]["cells"]
        complete = all(row["p_cond"] is not None for row in rows)
        bh56 = bh_stepup([row["p_cond"] for row in rows]) if complete else None
        by56 = by_adjust([row["p_cond"] for row in rows]) if complete else None
        bh56_worst = bh_stepup([row["p_upper"] for row in rows]) if complete else None
        bh56_best = bh_stepup([row["p_lower"] for row in rows]) if complete else None
        crossing = {}
        for alpha in ALPHA_LEVELS:
            entry = {"bh_cond": None, "by_cond": None, "robust_both_bounds": None}
            if bh56:
                entry["bh_cond"] = int(sum(1 for v in bh56 if v < alpha))
                entry["by_cond"] = int(sum(1 for v in by56 if v < alpha))
            if bh56_best and bh56_worst:
                entry["robust_both_bounds"] = int(sum(1 for b, w in zip(bh56_best, bh56_worst)
                                                     if b < alpha and w < alpha))
            crossing["%.2f" % alpha] = entry
        return {
            "family": family, "m": len(rows), "release_status": "SECONDARY_NO_SELECTION",
            "complete": complete, "cell_order": [row["key"] for row in rows],
            "bh56_cond": bh56, "by56_cond": by56, "bh56_worst": bh56_worst, "bh56_best": bh56_best,
            "crossing_counts_at_0.05": crossing["0.05"],
            "crossing_counts_at_0.10": crossing["0.10"],
            "dependency_note": ("combined native+rarefied 56-cell summary, declared secondary only; the two "
                                "stages are two coordinate systems over the same models and are never pooled "
                                "into a primary family; never used for selection; no calibrated FDR claim."),
        }

    def representative_checks(self, inputs, jobs):
        """Full inference-generator re-run on a bounded representative subset:
        regenerate the stored draw arrays from the seeds, re-run the frozen
        permutation/bootstrap loops on the freshly rebuilt inputs, and compare
        the regenerated statistic arrays to the stored ones bitwise."""
        from concurrent.futures import ProcessPoolExecutor

        wanted = ["F1_MAP__native__science__twoNN_id", "F1_MAP__rarefied__code__spectral_alpha",
                  "F1_MAP__native__medical__rankme", "F2_RASCH__native__medical__isoscore"]
        wanted = wanted[:max(0, self.ctx.representative)]
        job_by_key = {j["key"]: j for j in jobs}
        payloads = []
        for key in wanted:
            job = job_by_key[key]
            stored_path = self.run_root / "draws" / (key + ".npz")
            payloads.append({"key": key,
                             "x": job["x"], "y": job["y"], "controls": job["controls"],
                             "families": job["families"], "b_total": job["b_total"],
                             "seed_permutation": job["seed_permutation"],
                             "seed_family_bootstrap": job["seed_family_bootstrap"],
                             "draws_path": str(self.ctx.resolve_recorded(str(stored_path)))})
        if not payloads:
            inherit = getattr(self.ctx, "inherit_representative_from", None)
            if inherit:
                path = Path(inherit)
                if not path.exists():
                    raise MissingRefusal("inherited representative-check evidence missing: %s" % path)
                sha = self.ctx.consume(path, "inherited_representative_checks",
                                       "comparison_only")
                doc = read_json(path)
                code_sha = doc.get("code_sha256")
                run_manifest = path.parents[1] / "run_manifest.json"
                if code_sha is None and run_manifest.exists():
                    m_sha = self.ctx.consume(run_manifest, "inherited_representative_run_manifest",
                                             "comparison_only")
                    code_sha = read_json(run_manifest).get("code_sha256")
                else:
                    m_sha = None
                return [{"inherited_from": str(path), "inherited_sha256": sha,
                         "inherited_code_sha256": code_sha,
                         "inherited_run_manifest": (str(run_manifest) if m_sha else None),
                         "inherited_run_manifest_sha256": m_sha,
                         "checks": doc.get("representative_full_generator_checks"),
                         "note": "previous full-generator checks preserved with their exact "
                                 "source/code scope; not re-run in this bounded correction run"}]
            return []
        with ProcessPoolExecutor(max_workers=min(self.ctx.workers, len(payloads))) as pool:
            return list(pool.map(_representative_worker, payloads))


def _representative_worker(payload):
    """Module-level worker: regenerate the draw arrays and statistic arrays for
    one cell from the frozen seeds and compare bitwise with the stored arrays."""
    t0 = time.time()
    key = payload["key"]
    x = np.asarray(payload["x"], float).copy()
    sd = x.std(ddof=0)
    x = (x - x.mean()) / sd if sd > 0 else x - x.mean()
    y = np.asarray(payload["y"], float)
    controls = np.asarray(payload["controls"], float)
    families = np.asarray(payload["families"], dtype=str)
    n = len(y)
    family_draws, permutation_indices = draws_for_panel(
        len(set(families.tolist())), n, payload["b_total"],
        payload["seed_permutation"], payload["seed_family_bootstrap"])
    observed, perm_values, perm_reasons = permutation_reasoned(x, y, controls,
                                                               permutation_indices)
    cond_values, cond_reasons = bootstrap_reasoned(x, y, controls, families, family_draws)
    labels = {f: i for i, f in enumerate(sorted(set(families.tolist())))}
    bare = np.array([labels[f] for f in families], float)
    bare_values, bare_reasons = bootstrap_reasoned(x, y, bare, families, family_draws)
    stored = np.load(payload["draws_path"])
    comparisons = {}
    for name, fresh_arr, stored_arr in (
            ("family_draws", family_draws, stored["family_draws"]),
            ("permutation_indices", permutation_indices, stored["permutation_indices"]),
            ("x", x, stored["x"]),
            ("permutation", perm_values, stored["permutation"]),
            ("permutation_reasons", perm_reasons, stored["permutation_reasons"]),
            ("controlled_bootstrap", cond_values, stored["controlled_bootstrap"]),
            ("controlled_reasons", cond_reasons, stored["controlled_reasons"]),
            ("bare_bootstrap", bare_values, stored["bare_bootstrap"]),
            ("bare_reasons", bare_reasons, stored["bare_reasons"])):
        fresh_arr = np.asarray(fresh_arr)
        stored_arr = np.asarray(stored_arr)
        equal = (fresh_arr.shape == stored_arr.shape
                 and np.array_equal(fresh_arr, stored_arr, equal_nan=True))
        max_abs = None
        if fresh_arr.dtype.kind == "f" and stored_arr.dtype.kind == "f":
            with np.errstate(invalid="ignore"):
                diff = np.abs(fresh_arr.astype(float) - stored_arr.astype(float))
            finite = diff[np.isfinite(diff)]
            max_abs = float(finite.max()) if finite.size else None
        comparisons[name] = {"bitwise_equal": bool(equal), "max_abs_diff": max_abs}
    return {"key": key, "observed_partial_rho": float(observed),
            "seconds": round(time.time() - t0, 3), "arrays": comparisons,
            "all_bitwise_equal": all(v["bitwise_equal"] for v in comparisons.values())}


# ------------------------------------------------------------------ windows
def window_chi(run_dir, used_ids, width):
    """Frozen consumer arithmetic (code/window_cloud_metrics.py) re-executed
    read-only on the saved per-item window vectors."""
    X = np.stack([np.load(run_dir / ("item_%s.npz" % i))["W%d" % width] for i in used_ids])
    neighbors = []
    for j in range(X.shape[1]):
        D = cdist(X[:, j].astype("float64"), X[:, j].astype("float64"))
        np.fill_diagonal(D, np.inf)
        neighbors.append(np.argsort(D, axis=1)[:, :WINDOW_K])
    pairs = [float(np.mean([len(set(neighbors[j][i]) & set(neighbors[j + 1][i])) / WINDOW_K
                            for i in range(len(used_ids))]))
             for j in range(len(neighbors) - 1)]
    return {"chi": float(np.mean(pairs)), "pair_chis": pairs, "shape": list(X.shape)}


def run_windows(ctx, comparator):
    t0 = time.time()
    consumer_path = ctx.analysis_root / "code/window_cloud_metrics.py"
    ctx.consume(consumer_path, "frozen_window_consumer_code", "code_dependency")
    lengths_path = ctx.analysis_root / "code/window_effective_lengths.py"
    lengths_sha = ctx.consume(lengths_path, "frozen_effective_lengths_code", "code_dependency")
    summary_path = ctx.analysis_root / "reports/RFINAL_WINDOW_SUMMARY_v1.json"
    ctx.consume(summary_path, "window_run_dir_discovery", "pointer_discovery")
    summary = read_json(summary_path)
    checks_path = ctx.analysis_root / "reports/C17_WINDOW_CHECKS_v1.json"
    ctx.consume(checks_path, "window_length_coverage_reference", "pointer_discovery")
    checks = read_json(checks_path)

    per_model = {}
    chi_vectors = {w: [] for w in WINDOW_SIZES}
    order = []
    for name in WINDOW_MODELS:
        spec = summary["models"][name]
        run_dir = ctx.resolve_recorded(str(ctx.analysis_root / spec["run_dir"]),
                                       note="window run dir %s" % name)
        run_meta_path = run_dir / "run_200.json"
        run_meta = read_json(run_meta_path)
        ctx.consume(run_meta_path, "window_run_meta:%s" % name, "computation_input")
        used = [r for r in run_meta["records"] if "skip" not in r]
        used_ids = [str(r["id"]) for r in used]
        skipped = [{"id": str(r["id"]), "skip": r["skip"]} for r in run_meta["records"] if "skip" in r]
        model_entry = {
            "model": name, "repo": run_meta.get("repo"), "revision": run_meta.get("revision"),
            "run_dir": spec["run_dir"], "n_records": len(run_meta["records"]),
            "n_used": len(used), "used_ids": used_ids,
            "skipped_ids": [s["id"] for s in skipped],
            "skipped_reasons": sorted({(s["skip"] if isinstance(s["skip"], str) else
                                        s["skip"].get("reason", "skip")) for s in skipped}),
            "layer_convention": {
                "selected_hidden_state_indices_example": run_meta["records"][0].get(
                    "selected_hidden_state_indices"),
                "note": ("per-item W8/W32/W128 pools of the saved selected hidden-state layers "
                         "(run-specific deepest-last order); chi pairs are consecutive entries of "
                         "that stored order"),
            },
            "windows": {},
            "effective_lengths": None,
        }
        for width in WINDOW_SIZES:
            for rid in used_ids:
                ctx.consume(run_dir / ("item_%s.npz" % rid),
                            "window_item_npz:%s:W%d" % (name, width), "computation_input")
            res = window_chi(run_dir, used_ids, width)
            model_entry["windows"][str(width)] = res
            chi_vectors[width].append(res["chi"])
        vals = [model_entry["windows"][str(w)]["chi"] for w in WINDOW_SIZES]
        model_entry["max_pairwise_rel_diff"] = float(max(
            rel_diff(a, b) for a in vals for b in vals))
        lens = np.array([r["seq_len"] - r["prompt_len"] for r in used])
        eff = {
            "model": run_meta.get("repo"), "run": spec["run_dir"],
            "n_requested": run_meta.get("n_requested"), "n_used": len(used),
            "skipped_ids": [s["id"] for s in skipped],
            "answer_token_length_quantiles": np.quantile(lens, [0, .25, .5, .75, 1]).tolist(),
            "n_le8": int((lens <= 8).sum()), "n_le32": int((lens <= 32).sum()),
            "n_le128": int((lens <= 128).sum()),
            "n_lt8": int((lens < 8).sum()), "n_lt32": int((lens < 32).sum()),
            "n_lt128": int((lens < 128).sum()),
            "actual_effective_lengths": {str(w): np.minimum(w, lens).tolist()
                                         for w in WINDOW_SIZES},
            "used_ids": used_ids,
            "n_actual_window8_differs_from128": int((lens > 8).sum()),
            "n_actual_window32_differs_from128": int((lens > 32).sum()),
            "run_sha256": sha256_file(run_meta_path),
            "code_sha256": lengths_sha,
        }
        model_entry["effective_lengths"] = eff
        per_model[name] = model_entry
        order.append(name)
        ctx.write_json("windows/models/%s.json" % name, model_entry)
        ctx.write_json("windows/effective_lengths/%s.json" % name, eff)

    per_width = {}
    for width in WINDOW_SIZES:
        vec = chi_vectors[width]
        per_width["W%d" % width] = {
            "chi_by_model": {m: per_model[m]["windows"][str(width)]["chi"] for m in order},
            "max_cross_model_rel_diff": float(max(rel_diff(a, b) for a in vec for b in vec)),
            "mean": float(np.mean(vec)), "min": float(np.min(vec)), "max": float(np.max(vec)),
        }
    rho_w8_w128 = float(spearmanr(chi_vectors[8], chi_vectors[128]).statistic)
    cross = {"per_width": per_width, "spearman_rho_W8_vs_W128_across_models": rho_w8_w128,
             "same_width_policy": ("recomputed from the saved per-item window vectors; not a "
                                   "historical gate result")}

    # comparator: signed RFINAL summary and each run's saved metrics.json
    base = "R:reports/RFINAL_WINDOW_SUMMARY_v1.json"
    sm_sha = sha256_file(summary_path)
    for name in order:
        entry = per_model[name]
        ref = summary["models"][name]
        for width in WINDOW_SIZES:
            comparator.scalar("windows", "%s/chi/%d" % (name, width),
                              entry["windows"][str(width)]["chi"], ref["chi"][str(width)],
                              "%s#/models/%s/chi/%d" % (base, name, width), sm_sha)
        comparator.exact("windows", "%s/n_used" % name, entry["n_used"], ref["n_used"],
                         "%s#/models/%s/n_used" % (base, name), sm_sha)
        comparator.exact("windows", "%s/skipped_ids" % name, entry["skipped_ids"],
                         ref["skipped_ids"], "%s#/models/%s/skipped_ids" % (base, name), sm_sha)
        comparator.scalar("windows", "%s/max_pairwise_rel_diff" % name,
                          entry["max_pairwise_rel_diff"], ref["max_pairwise_rel_diff"],
                          "%s#/models/%s/max_pairwise_rel_diff" % (base, name), sm_sha)
        run_dir = ctx.resolve_recorded(str(ctx.analysis_root / ref["run_dir"]))
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            m_sha = ctx.consume(metrics_path, "window_saved_metrics:%s" % name, "comparison_only")
            metrics = read_json(metrics_path)
            for width in WINDOW_SIZES:
                comparator.scalar("windows", "savedmetrics:%s/chi/%d" % (name, width),
                                  entry["windows"][str(width)]["chi"],
                                  metrics["windows"][str(width)]["chi"],
                                  "R:%s/metrics.json#/windows/%d/chi" % (ref["run_dir"], width),
                                  m_sha)
            comparator.exact("windows", "savedmetrics:%s/used_ids" % name, entry["used_ids"],
                             [str(i) for i in metrics["used_ids"]],
                             "R:%s/metrics.json#/used_ids" % ref["run_dir"], m_sha)
            comparator.exact("windows", "savedmetrics:%s/n_used" % name, entry["n_used"],
                             metrics["n_used"],
                             "R:%s/metrics.json#/n_used" % ref["run_dir"], m_sha)
        eff_path = run_dir / "effective_lengths.json"
        if eff_path.exists():
            e_sha = ctx.consume(eff_path, "window_saved_effective_lengths:%s" % name,
                                "comparison_only")
            eff_ref = read_json(eff_path)
            comparator.exact("windows", "effective:%s/n_used" % name, entry["n_used"],
                             eff_ref["n_used"],
                             "R:%s/effective_lengths.json#/n_used" % ref["run_dir"], e_sha)
            comparator.exact("windows", "effective:%s/used_ids" % name, entry["used_ids"],
                             [str(i) for i in eff_ref["used_ids"]],
                             "R:%s/effective_lengths.json#/used_ids" % ref["run_dir"], e_sha)
            for f in ("n_le8", "n_le32", "n_le128",
                      "n_actual_window8_differs_from128", "n_actual_window32_differs_from128"):
                comparator.exact("windows", "effective:%s/%s" % (name, f),
                                 entry["effective_lengths"][f], eff_ref[f],
                                 "R:%s/effective_lengths.json#/%s" % (ref["run_dir"], f), e_sha)
            comparator.array("windows", "effective:%s/quantiles" % name,
                             entry["effective_lengths"]["answer_token_length_quantiles"],
                             eff_ref["answer_token_length_quantiles"],
                             "R:%s/effective_lengths.json#/answer_token_length_quantiles"
                             % ref["run_dir"], e_sha)
        else:
            model_entry["effective_lengths"]["saved_reference"] = "ABSENT"

    for width in WINDOW_SIZES:
        ref_w = summary["cross_model"]["per_width"]["W%d" % width]
        for f in ("max_cross_model_rel_diff", "mean", "min", "max"):
            comparator.scalar("windows", "cross/W%d/%s" % (width, f), per_width["W%d" % width][f],
                              ref_w[f], "%s#/cross_model/per_width/W%d/%s" % (base, width, f),
                              sm_sha)
        for m in order:
            comparator.scalar("windows", "cross/W%d/chi/%s" % (width, m),
                              per_width["W%d" % width]["chi_by_model"][m],
                              ref_w["chi_by_model"][m],
                              "%s#/cross_model/per_width/W%d/chi_by_model/%s" % (base, width, m),
                              sm_sha)
    comparator.scalar("windows", "cross/spearman_rho_W8_vs_W128", rho_w8_w128,
                      summary["cross_model"]["spearman_rho_W8_vs_W128_across_models"],
                      "%s#/cross_model/spearman_rho_W8_vs_W128_across_models" % base, sm_sha)

    # independent length-coverage reference from C17_WINDOW_CHECKS_v1.json
    ck_sha = sha256_file(checks_path)
    xcheck = checks.get("cross_window_review_crosscheck") or {}
    saved_meta = checks.get("six_model_length_coverage") or {}
    n_xcheck = 0
    for name in order:
        if name in xcheck:
            review = xcheck[name]["review"]
            for width in WINDOW_SIZES:
                comparator.exact("windows", "c7check:%s/review_lt%d" % (name, width),
                                 per_model[name]["effective_lengths"]["n_lt%d" % width],
                                 int(review[str(width)]),
                                 "R:reports/C17_WINDOW_CHECKS_v1.json#/cross_window_"
                                 "review_crosscheck/%s/review/%d" % (name, width), ck_sha)
                n_xcheck += 1
        if name in saved_meta and "lengths_saved_meta" in saved_meta[name]:
            meta = saved_meta[name]["lengths_saved_meta"]
            comparator.exact("windows", "c7check:%s/n_used_saved_meta" % name,
                             per_model[name]["n_used"], int(meta["n_used"]),
                             "R:reports/C17_WINDOW_CHECKS_v1.json#/six_model_length_coverage/"
                             "%s/lengths_saved_meta/n_used" % name, ck_sha)
            for width in WINDOW_SIZES:
                comparator.exact("windows", "c7check:%s/saved_meta_lt%d" % (name, width),
                                 per_model[name]["effective_lengths"]["n_lt%d" % width],
                                 int(meta["answer_lt_%d" % width]),
                                 "R:reports/C17_WINDOW_CHECKS_v1.json#/six_model_length_coverage/"
                                 "%s/lengths_saved_meta/answer_lt_%d" % (name, width), ck_sha)
                n_xcheck += 1
    out = {
        "schema": "c17-cache-observations-windows-v1",
        "models": per_model,
        "cross_model": cross,
        "frozen_consumer": {
            "code": "R:code/window_cloud_metrics.py",
            "code_sha256": sha256_file(consumer_path),
            "arithmetic": ("K=%d Jaccard neighbour-overlap over consecutive stored layer entries, "
                           "euclidean cdist on float64 casts of the saved float32 pools" % WINDOW_K),
            "read_only_note": ("saved run dirs consumed read-only; nothing written or hardlinked "
                               "inside them; saved metrics.json / effective_lengths.json are "
                               "comparison-only"),
        },
        "length_coverage_reference_comparisons": n_xcheck,
        "boundaries": [
            "Pythia new GPU analysis is not CPU-equivalent; the strict CPU/GPU 41/43 failure is "
            "historical and not relaxed here.",
            "Gemma ids 33/44 are empty-target skips by the accepted selection rule (n_used=198).",
            "Llama 200/200 used; answer-length tails heavy (max 1027); window stability is not "
            "generalisable beyond the saved answers.",
            "No GPU and no model weights were used; only saved per-item vectors/meta/token_ids.",
            "The 0.2 scientific stability flag of the original consumer is reported as a "
            "per-model flag only and is never used as the comparator tolerance.",
        ],
        "checks_report_sha256": ck_sha,
        "seconds": round(time.time() - t0, 3),
    }
    ctx.write_json("windows/window_replay.json", out)
    return out


# ------------------------------------------------------------------- probes
def run_probes(ctx, comparator):
    t0 = time.time()
    P = ctx.project_root
    h1_r2 = ctx.resolve_recorded(str(P / "ext_P2_geo_20260914/h1/r2_by_model_layer.csv"))
    h1_tr = ctx.resolve_recorded(str(P / "ext_P2_geo_20260914/h1/transfer_rho.csv"))
    h2_beta = ctx.resolve_recorded(str(P / "ext_P2_geo_20260914/h2/beta_by_model.csv"))
    h3_r2 = ctx.resolve_recorded(str(P / "ext_P2_geo_20260914/h3/r2_eps.csv"))
    for path, role in ((h1_r2, "probe_h1_r2_by_model_layer"),
                       (h1_tr, "probe_h1_transfer_rho"),
                       (h2_beta, "probe_h2_beta_by_model"),
                       (h3_r2, "probe_h3_r2_eps")):
        ctx.consume(path, role, "computation_input")

    import csv

    def read_csv(path):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    r2_rows = read_csv(h1_r2)
    tr_rows = read_csv(h1_tr)
    beta_rows = read_csv(h2_beta)
    eps_rows = read_csv(h3_r2)

    pooled_r2 = [float(r["r2_oof"]) for r in r2_rows if r["layer"] == "-1"]
    transfer = [float(r["rho_transfer"]) for r in tr_rows]
    s_hat = np.array([float(r["s_hat"]) for r in beta_rows], float)
    neg_beta_best = np.array([-float(r["beta_best"]) for r in beta_rows], float)
    neg_beta_pooled = np.array([-float(r["beta_pooled"]) for r in beta_rows], float)
    eps_pooled = [r for r in eps_rows if r["layer"] == "-1"]
    r2_eps = [float(r["r2_oof"]) for r in eps_pooled]
    p_perm = [float(r["p_perm"]) for r in eps_pooled]
    n_both = int(sum(1 for p, r2 in zip(p_perm, r2_eps) if p < 0.01 and r2 >= 0.05))

    agg = {
        "schema": "c17-cache-observations-probes-v1",
        "role": "fresh aggregation over per-model numeric observation tables for P2-F-TAB-PROBES",
        "rows": {
            "difficulty_readout_median_r2": {
                "value": float(np.median(pooled_r2)), "n_models": len(pooled_r2),
                "source": "P:ext_P2_geo_20260914/h1/r2_by_model_layer.csv",
                "rule": "median over model rows with layer==-1 (pooled representation) of r2_oof"},
            "aligned_transfer_median_rho": {
                "value": float(np.median(transfer)), "n_pairs": len(transfer),
                "source": "P:ext_P2_geo_20260914/h1/transfer_rho.csv",
                "rule": "median of rho_transfer over all recorded pairs"},
            "gain_slope_best_layer_rho": {
                "value": float(spearmanr(neg_beta_best, s_hat).statistic),
                "n_models": len(beta_rows),
                "source": "P:ext_P2_geo_20260914/h2/beta_by_model.csv",
                "rule": "Spearman(-beta_best, s_hat) over models"},
            "gain_slope_pooled_rho": {
                "value": float(spearmanr(neg_beta_pooled, s_hat).statistic),
                "n_models": len(beta_rows),
                "source": "P:ext_P2_geo_20260914/h2/beta_by_model.csv",
                "rule": "Spearman(-beta_pooled, s_hat) over models"},
            "residual_readout_median_r2": {
                "value": float(np.median(r2_eps)), "n_models": len(r2_eps),
                "source": "P:ext_P2_geo_20260914/h3/r2_eps.csv",
                "rule": "median of layer==-1 r2_oof rows"},
            "residual_threshold_count": {
                "value": n_both, "n_models": len(eps_pooled),
                "source": "P:ext_P2_geo_20260914/h3/r2_eps.csv",
                "rule": "count of layer==-1 models with p_perm<0.01 AND r2_oof>=0.05"},
        },
        "boundaries": [
            "Bounded retained G1/G2/G3 exploratory probes; aggregation only, no new fits and no "
            "new prediction experiment.",
            "These numbers have not passed a strict no-leakage validation (implementation-level "
            "caveat retained).",
            "The h3 per-layer run-2 rerun is recorded as incomplete upstream; only the pooled "
            "layer==-1 rows are aggregated here.",
            "G4depth summaries are another worker's scope and are not touched.",
        ],
        "seconds": round(time.time() - t0, 3),
    }
    ctx.write_json("probes/probe_aggregates.json", agg)

    refs = {
        "difficulty_readout_median_r2": (P / "ext_P2_geo_20260914/h1/results_h1.json",
                                         "/task_A_pooled/median_r2"),
        "aligned_transfer_median_rho": (P / "ext_P2_geo_20260914/h1/results_h1.json",
                                        "/task_C_transfer/median_transfer_rho"),
        "gain_slope_best_layer_rho": (P / "ext_P2_geo_20260914/h2/results_h2.json",
                                      "/rho_H2_best_layer"),
        "gain_slope_pooled_rho": (P / "ext_P2_geo_20260914/h2/results_h2.json", "/rho_H2_pooled"),
        "residual_readout_median_r2": (P / "ext_P2_geo_20260914/h3/results_h3.json",
                                       "/pooled_summary/r2_median"),
        "residual_threshold_count": (P / "ext_P2_geo_20260914/h3/results_h3.json",
                                     "/fractions/n_both"),
    }
    for row, (path, pointer) in refs.items():
        sha = ctx.consume(path, "probe_terminal_result_reference:%s" % row, "comparison_only")
        doc = read_json(path)
        node = doc
        for part in [p for p in pointer.split("/") if p]:
            node = node[part]
        base = "P:" + str(path.relative_to(P))
        if row == "residual_threshold_count":
            comparator.exact("probes", "%s/count" % row, agg["rows"][row]["value"], node,
                             "%s#%s" % (base, pointer), sha)
            comparator.exact("probes", "%s/n_models" % row, agg["rows"][row]["n_models"],
                             read_json(path)["fractions"]["n_models"],
                             "%s#/fractions/n_models" % base, sha)
        else:
            comparator.scalar("probes", row, agg["rows"][row]["value"], node,
                              "%s#%s" % (base, pointer), sha)
    return agg


# --------------------------------------------------------------- selftest
def run_portability_selftest(ctx):
    """Relocation fixture: build a minimal project tree with a DIFFERENT project
    and analysis basename under the fresh out dir, while the original project
    root still exists, and prove that (a) recorded absolute paths are remapped
    by relative suffix onto the caller roots (fixture bytes with fixture hashes
    verify, originals ignored), and (b) a sibling-prefix path is refused."""
    t0 = time.time()
    fixture_project = ctx.out_dir / "portability/ProjRenamedFixture"
    fixture_analysis = fixture_project / "AnalysisRenamedFixture"
    (fixture_analysis / "runs/study2_reuse_v1/features").mkdir(parents=True, exist_ok=True)
    (fixture_analysis / "code").mkdir(parents=True, exist_ok=True)

    fixture_features = fixture_analysis / "runs/study2_reuse_v1/features/H4BC_INPUT_ARRAYS.json"
    real_features = KNOWN_ORIGINAL_ANALYSIS_ROOT / "runs/study2_reuse_v1/features/H4BC_INPUT_ARRAYS.json"
    feature_bytes = json.dumps({"fixture": "relocated-copy",
                                "real_sha256_would_differ": True}, indent=1) + "\n"
    fixture_features.write_text(feature_bytes, encoding="utf-8")
    fixture_features_sha = sha256_file(fixture_features)

    # machinery modules copied byte-identically so the fixture contract can pin them
    for name in ("study1_reuse_analyze.py", "study2_reuse_inference.py"):
        src = KNOWN_ORIGINAL_ANALYSIS_ROOT / "code" / name
        (fixture_analysis / "code" / name).write_bytes(src.read_bytes())

    contract = {
        "sources": {"computation_inputs": {
            "features_H4BC": {
                "path": str(real_features),  # recorded ORIGINAL absolute path (still exists)
                "sha256": fixture_features_sha},
            "machinery_permute_module": {
                "path": str(KNOWN_ORIGINAL_ANALYSIS_ROOT / "code/study1_reuse_analyze.py"),
                "sha256": sha256_file(fixture_analysis / "code/study1_reuse_analyze.py")},
            "machinery_grid_module": {
                "path": str(KNOWN_ORIGINAL_ANALYSIS_ROOT / "code/study2_reuse_inference.py"),
                "sha256": sha256_file(fixture_analysis / "code/study2_reuse_inference.py")}}}}
    (fixture_analysis / "config").mkdir(parents=True, exist_ok=True)
    contract_path = fixture_analysis / "config/CT_C17_NEW28_v1.json"
    contract_path.write_text(json.dumps(contract, indent=1) + "\n", encoding="utf-8")

    checks = []
    fx_ctx = Context(fixture_analysis, ctx.out_dir / "portability/fixture_out", "new28",
                     0, 1, None, None, contract_path=contract_path)
    (ctx.out_dir / "portability/fixture_out").mkdir(parents=True, exist_ok=True)
    try:
        engine = New28(fx_ctx)
        engine.load_contract()
        verified = engine.verify_inputs()
        by_name = {v["name"]: v for v in verified}
        used_fixture = by_name["features_H4BC"]["path"].startswith(str(fixture_project))
        originals_exist = real_features.exists()
        checks.append({
            "label": "relocated_fixture_remap_preferred_over_existing_original",
            "pass": bool(used_fixture and originals_exist),
            "fixture_project_root": str(fixture_project),
            "fixture_analysis_root": str(fixture_analysis),
            "recorded_path": str(real_features),
            "resolved_path": by_name["features_H4BC"]["path"],
            "original_still_exists": originals_exist,
            "note": "recorded absolute path exists outside the caller roots; the remapped "
                    "candidate (different project AND analysis basename) was used and the "
                    "fixture-pinned hash verified"})
    except ReplayRefusal as exc:
        checks.append({"label": "relocated_fixture_remap_preferred_over_existing_original",
                       "pass": False, "raised": type(exc).__name__, "message": str(exc)[:300]})

    # sibling-prefix escape: a real file whose path string starts with the caller
    # project root plus a suffix must be refused, never read.
    sibling_dir = ctx.out_dir / "portability/ProjRenamedFixtureSibling"
    sibling_dir.mkdir(parents=True, exist_ok=True)
    sibling_file = sibling_dir / "escape_probe.json"
    sibling_file.write_text("{}\n", encoding="utf-8")
    probe_ctx = Context(fixture_analysis, ctx.out_dir / "portability/sibling_out", "new28",
                        0, 1, None, None)
    refusal = None
    try:
        probe_ctx.resolve_recorded(str(sibling_file), note="sibling-prefix escape probe")
    except ReplayRefusal as exc:
        refusal = {"raised": type(exc).__name__, "exit_code": exc.exit_code,
                   "message": str(exc)[:300]}
    checks.append({
        "label": "sibling_prefix_escape_refused",
        "pass": refusal is not None,
        "probe_path": str(sibling_file),
        "probe_path_shares_string_prefix_with_project_root": str(sibling_file).startswith(
            str(fixture_project)),
        "refusal": refusal,
        "skips_recorded": probe_ctx.path_skips,
        "note": "Path.is_relative_to semantics; the exists-but-out-of-root file is refused, "
                "never returned"})

    evidence = {"schema": "c17-cache-observations-portability-selftest-v1",
                "checks": checks, "all_pass": all(c["pass"] for c in checks),
                "seconds": round(time.time() - t0, 3)}
    ctx.write_json("portability_selftest.json", evidence)
    return evidence


def run_selftest(ctx, base_contract_path):
    """Bounded negative tests for missing/hash refusal, run inside the fresh
    output dir; never mutates any signed source."""
    t0 = time.time()
    evidence = {"checks": []}
    contract = read_json(base_contract_path)
    victim = "features_H4BC"
    tampered = json.loads(json.dumps(contract))
    tampered["sources"]["computation_inputs"][victim]["sha256"] = "0" * 64
    tp = ctx.write_json("selftest/tampered_contract.json", tampered)
    missing = json.loads(json.dumps(contract))
    missing["sources"]["computation_inputs"][victim]["path"] = str(
        ctx.out_dir / "selftest/does_not_exist.json")
    mp = ctx.write_json("selftest/missing_contract.json", missing)

    for label, path, expected in (("tampered_hash", tp, HashRefusal),
                                  ("missing_input", mp, MissingRefusal)):
        sub = Context(ctx.analysis_root, ctx.out_dir / ("selftest/%s_out" % label), "new28",
                      0, 1, None, None, contract_path=path)
        sub.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            engine = New28(sub)
            engine.load_contract()
            engine.verify_inputs()
            outcome = {"label": label, "raised": None, "pass": False}
        except ReplayRefusal as exc:
            outcome = {"label": label, "raised": type(exc).__name__,
                       "message": str(exc)[:300], "exit_code": exc.exit_code,
                       "pass": isinstance(exc, expected)}
        evidence["checks"].append(outcome)
    t13 = time.time()
    try:
        engine = New28(Context(ctx.analysis_root, ctx.out_dir / "selftest/draws_out", "new28",
                               0, 1, None, None))
        engine.load_contract()
        engine.verify_inputs()
        draws_ok = True
    except ReplayRefusal:
        draws_ok = False
    evidence["checks"].append({
        "label": "real_contract_inputs_verify", "pass": bool(draws_ok),
        "message": "clean contract hash verification must succeed on unmodified inputs"})
    evidence["seconds"] = round(time.time() - t0, 3)
    evidence["all_pass"] = all(c["pass"] for c in evidence["checks"])
    ctx.write_json("selftest/selftest.json", evidence)
    return evidence


# ---------------------------------------------------------------- reporting
def build_report(ctx, branches, comparator, extra):
    summary = comparator.summary()
    compact = {}
    for name, info in branches.items():
        if name == "selftest":
            compact[name] = {"all_pass": info.get("all_pass"),
                             "checks": [{"label": c["label"], "pass": c["pass"],
                                         "raised": c.get("raised")} for c in info["checks"]]}
        elif name == "new28":
            compact[name] = {
                "n_cell_records": info["n_cell_records"],
                "n_stored_draw_files": info["n_stored_draw_files"],
                "stored_input_mismatches": len(info["stored_input_mismatches"]),
                "representative_full_generator_checks": [
                    ({"key": r["key"], "all_bitwise_equal": r["all_bitwise_equal"],
                      "seconds": r.get("seconds")}
                     if "key" in r else
                     {"inherited_from": r.get("inherited_from"),
                      "inherited_sha256": r.get("inherited_sha256"),
                      "inherited_code_sha256": r.get("inherited_code_sha256"),
                      "checks": [{"key": c.get("key"), "all_bitwise_equal": c.get("all_bitwise_equal")}
                                 for c in (r.get("checks") or [])]})
                    for r in info["representative_full_generator_checks"]],
                "grids": info["grids"],
                "combined56_secondary": info["combined56_secondary"],
                "boundaries": info["boundaries"],
            }
        elif name == "windows":
            compact[name] = {
                "per_model": {m: {"repo": e["repo"], "revision": e["revision"],
                                  "run_dir": e["run_dir"], "n_used": e["n_used"],
                                  "skipped_ids": e["skipped_ids"],
                                  "chi": {w: e["windows"][w]["chi"] for w in e["windows"]},
                                  "max_pairwise_rel_diff": e["max_pairwise_rel_diff"],
                                  "saved_effective_lengths": e["effective_lengths"].get(
                                      "saved_reference", "PRESENT")}
                              for m, e in info["models"].items()},
                "cross_model": info["cross_model"],
                "frozen_consumer": info["frozen_consumer"],
                "boundaries": info["boundaries"],
            }
        elif name == "probes":
            compact[name] = {"rows": info["rows"], "boundaries": info["boundaries"]}
        else:
            compact[name] = info
    report = {
        "schema": "c17-cache-observations-report-v1",
        "node": "C10o",
        "status": "CACHED_OBSERVATION_REPLAY_COMPLETE" if summary["n_fail"] == 0
                  else "CACHED_OBSERVATION_REPLAY_WITH_FAILURES",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_root": str(ctx.analysis_root),
        "out_dir": str(ctx.out_dir),
        "branches": compact,
        "comparator": summary,
        "consumed_file_counts": {
            level: sum(1 for c in ctx.consumed if c["level"] == level)
            for level in sorted({c["level"] for c in ctx.consumed})},
        "boundaries": [
            "Medical F1 = accepted upstream regularized-2PL MAP A45 scores as saved inputs; no fresh "
            "fit; not relabelled obsolete. F2 = accepted Rasch A45 canonical total-score ranks.",
            "code/math panel51 theta keeps converged=false as a recorded code/math limitation.",
            "BH/BY are recomputed descriptive multiplicity summaries; no calibrated FDR guarantee "
            "(descriptive-FDR boundary preserved).",
            "Six-model windows are a NEW pinned analysis (historical six-model revision identity "
            "unrecovered); pythia GPU is not CPU-equivalent (41/43 retained).",
            "G1/G2/G3 probes are bounded retained exploratory aggregates; no strict no-leakage claim.",
            "Comparison tolerance: abs <= 1e-9 + 1e-6*abs(ref); counts/IDs/statuses/null masks exact; "
            "the window consumer's 0.2 stability flag is not used as tolerance.",
            "G4depth is handled by another worker; out of scope here.",
        ],
        "unresolved_limits": extra.get("unresolved_limits", []),
        "inputs_unchanged": extra.get("source_unchanged", {}),
    }
    return report


def _regenerate_report(src, report_json, report_md, branch_filter="all"):
    """Report-only mode: re-render the two report files from an existing fresh
    run dir; no computation, no signed-source writes."""
    class _Shim:
        def __init__(self, summary):
            self._summary = summary

        def summary(self):
            return self._summary

    ctx = Context(src.parents[2], src, "all", 0, 1, report_json, report_md)
    ctx.consumed = read_json(src / "consumed_files.json")["files"]
    branches = {
        "selftest": read_json(src / "selftest/selftest.json"),
        "new28": read_json(src / "new28/new28_check.json"),
        "windows": read_json(src / "windows/window_replay.json"),
        "probes": read_json(src / "probes/probe_aggregates.json"),
    }
    if branch_filter != "all":
        branches = {k: v for k, v in branches.items()
                    if k == branch_filter or k == "selftest"}
    comparator = _Shim(read_json(src / "comparator_report.json")["summary"])
    source_unchanged = read_json(src / "source_unchanged.json")
    missing_eff = [m for m, e in branches.get("windows", {}).get("models", {}).items()
                   if "saved_reference" not in e.get("effective_lengths", {})
                   or e["effective_lengths"].get("saved_reference") == "ABSENT"]
    report = build_report(ctx, branches, comparator,
                          {"unresolved_limits": [
                              "Historical six-model panel revision/pipeline identity is unrecovered; "
                              "the window branch is a NEW pinned analysis.",
                              "Pythia GPU window numbers remain non-CPU-equivalent (41/43 retained); "
                              "no equivalence claim is made by this replay.",
                              "No saved effective_lengths.json upstream for: %s (lengths recomputed "
                              "from run_200.json meta and cross-checked to C17_WINDOW_CHECKS_v1)."
                              % (", ".join(sorted(missing_eff)) or "none"),
                              "Report regenerated from run dir %s (report-only mode)." % src],
                           "source_unchanged": source_unchanged})
    Path(report_json).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    Path(report_md).write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report_from": str(src),
                      "report_json": str(report_json), "n_fail":
                          report["comparator"]["n_fail"]}, indent=1))
    return 0


def report_markdown(report):
    lines = ["# C17 cached-observation replay (C10o)",
             "",
             "- status: **%s**" % report["status"],
             "- analysis root: `%s`" % report["analysis_root"],
             "- fresh output dir: `%s`" % report["out_dir"],
             "- created: %s" % report["created_utc"],
             "",
             "## Comparator",
             "",
             "| branch | comparisons | pass | fail |",
             "| --- | --- | --- | --- |"]
    per_branch = {}
    for row in getattr(report, "_rows", []):
        per_branch.setdefault(row["branch"], [0, 0])
        per_branch[row["branch"]][0 if row["pass"] else 1] += 1
    lines.append("| all | %d | %d | %d |" % (report["comparator"]["n_comparisons"],
                                             report["comparator"]["n_pass"],
                                             report["comparator"]["n_fail"]))
    lines += ["", "tolerance: %s" % report["comparator"]["tolerance_rule"], ""]
    if report["comparator"]["n_fail"]:
        lines += ["## Failed fields", ""]
        for f in report["comparator"]["failed_fields"]:
            lines.append("- %s / %s (%s) fresh=%s ref=%s -> %s"
                         % (f["branch"], f["field"], f["kind"], f["fresh"], f["reference"],
                            f["reference_pointer"]))
        lines.append("")
    lines += ["## Branch summary", ""]
    for name, info in report["branches"].items():
        lines.append("- **%s**: %s" % (name, json.dumps(info, sort_keys=True)[:600]))
    lines += ["", "## Boundaries", ""]
    lines += ["- %s" % b for b in report["boundaries"]]
    if report["unresolved_limits"]:
        lines += ["", "## Unresolved limits", ""]
        lines += ["- %s" % u for u in report["unresolved_limits"]]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="C17 cached-observation replay (C10o)")
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    parser.add_argument("--out", required=True, help="fresh output dir (must not exist non-empty)")
    parser.add_argument("--branch", choices=["all", "new28", "windows", "probes"], default="all")
    parser.add_argument("--representative", type=int, default=4,
                        help="bounded number of full inference-generator re-runs (max 4)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-report-files", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--report-from", default=None,
                        help="regenerate the two report files from an existing fresh run dir")
    parser.add_argument("--expected-inputs", default=None,
                        help="expected-source manifest JSON enforced for every consumed file")
    parser.add_argument("--write-expected-inputs", default=None,
                        help="write the consumed-file manifest (for main-engine enforcement)")
    parser.add_argument("--selftest-portability", action="store_true",
                        help="run the relocated-fixture / sibling-prefix escape selftest")
    parser.add_argument("--inherit-representative-from", default=None,
                        help="previous new28_check.json whose full-generator checks are preserved")
    args = parser.parse_args(argv)

    analysis_root = Path(args.analysis_root).resolve()
    if not (analysis_root / "config/CT_C17_NEW28_v1.json").exists():
        raise MissingRefusal("analysis root does not look like R: %s" % analysis_root)
    report_json = analysis_root / "reports/C17_CACHE_OBSERVATIONS_v1.json"
    report_md = analysis_root / "reports/C17_CACHE_OBSERVATIONS_v1.md"
    if args.report_from:
        src = Path(args.report_from).resolve()
        if not (src / "comparator_report.json").exists():
            raise MissingRefusal("not a replay run dir: %s" % src)
        return _regenerate_report(src, report_json, report_md, args.branch)
    out_dir = Path(args.out).resolve()
    if str(out_dir) in ("/tmp", "/") or not str(out_dir).startswith("/"):
        raise ReplayRefusal("refusing output dir: %s" % out_dir)
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise ReplayRefusal("refusing to reuse a non-empty output dir: %s" % out_dir)
    else:
        out_dir.mkdir(parents=True)

    ctx = Context(analysis_root, out_dir, args.branch, min(4, max(0, args.representative)),
                  args.workers,
                  None if args.no_report_files else report_json,
                  None if args.no_report_files else report_md)
    ctx.inherit_representative_from = args.inherit_representative_from
    expected = {}
    if args.expected_inputs:
        expected = load_expected_inputs(args.expected_inputs, analysis_root)
        ctx.expected_inputs = expected
        ctx.consume(Path(args.expected_inputs).resolve(), "expected_inputs_manifest",
                    "comparison_only")
        preflight = verify_expected_inputs(expected)
        ctx.write_json("expected_inputs_check.json", {
            "schema": "c17-cache-observations-expected-inputs-check-v1",
            "manifest": str(Path(args.expected_inputs).resolve()),
            "n_entries": preflight["n_entries"], "all_ok": True,
            "rows": preflight["rows"]})
    ctx.consume(SCRIPT_PATH, "cached_replay_code", "code_dependency")
    comparator = Comparator()
    branches = {}
    t0 = time.time()
    if args.selftest:
        branches["selftest"] = run_selftest(ctx, analysis_root / "config/CT_C17_NEW28_v1.json")
    if args.selftest_portability:
        branches["portability_selftest"] = run_portability_selftest(ctx)
    if args.branch in ("all", "new28"):
        branches["new28"] = New28(ctx).run(comparator)
    if args.branch in ("all", "windows"):
        branches["windows"] = run_windows(ctx, comparator)
    if args.branch in ("all", "probes"):
        branches["probes"] = run_probes(ctx, comparator)

    # source-unchanged re-check (streaming hashes before/after identical by construction)
    changed = []
    for entry in ctx.consumed:
        digest = sha256_file(entry["abspath"])
        if digest != entry["sha256"]:
            changed.append({"path": entry["abspath"], "before": entry["sha256"],
                            "after": digest})
    source_unchanged = {"n_files": len(ctx.consumed), "changed": changed,
                        "all_unchanged": not changed}
    ctx.write_json("source_unchanged.json", source_unchanged)
    ctx.write_json("consumed_files.json", {
        "schema": "c17-cache-observations-consumed-v1",
        "note": "level: computation_input | comparison_only | pointer_discovery",
        "files": sorted(ctx.consumed, key=lambda e: (e["level"], e["relativepath"])),
    })
    if args.write_expected_inputs:
        manifest_path = Path(args.write_expected_inputs).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "schema": "c17-cache-observations-expected-inputs-v1",
            "producer": "code/c17_cache_observations/replay.py",
            "source_run_dir": str(out_dir),
            "note": ("Enforce with --expected-inputs <this file>: every listed file must exist and "
                     "match; any tamper refuses before computation (exit 3/4)."),
            "files": [{"relativepath": e["relativepath"], "root": e["root"],
                       "sha256": e["sha256"], "role": e["role"], "level": e["level"]}
                      for e in sorted(ctx.consumed, key=lambda e: (e["root"], e["relativepath"]))],
        }, indent=1) + "\n", encoding="utf-8")
    ctx.write_json("comparator_report.json", {
        "schema": "c17-cache-observations-comparator-v1",
        "summary": comparator.summary(),
        "rows": comparator.rows,
    })
    manifest = {
        "schema": SCHEMA,
        "node": "C10o",
        "command": " ".join([sys.executable] + sys.argv),
        "analysis_root": str(analysis_root),
        "out_dir": str(out_dir),
        "branch": args.branch,
        "representative_checks": args.representative,
        "workers": ctx.workers,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__,
                        "blas_threads": 1, "platform": platform.platform()},
        "code_sha256": sha256_file(SCRIPT_PATH),
        "contract_sha256_new28": CONTRACT_SHA256,
        "status": "COMPLETE" if comparator.summary()["n_fail"] == 0 else "COMPLETED_WITH_FAILURES",
        "seconds": round(time.time() - t0, 3),
        "source_unchanged": source_unchanged,
    }
    ctx.write_json("run_manifest.json", manifest)

    unresolved = []
    if args.branch in ("all", "windows"):
        for name, entry in branches["windows"]["models"].items():
            if entry["effective_lengths"].get("saved_reference") == "ABSENT":
                unresolved.append("window %s has no saved effective_lengths.json upstream; "
                                  "lengths recomputed from run_200.json meta only" % name)
    unresolved.append("Historical six-model panel revision/pipeline identity is unrecovered; the "
                      "window branch is a NEW pinned analysis and does not claim historical replay.")
    unresolved.append("Pythia GPU window numbers remain non-CPU-equivalent (41/43 retained); no "
                      "equivalence claim is made by this replay.")
    report = build_report(ctx, {k: (v if isinstance(v, dict) else str(v)) for k, v in branches.items()},
                          comparator, {"unresolved_limits": unresolved,
                                       "source_unchanged": source_unchanged})
    if ctx.report_json:
        Path(ctx.report_json).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
        Path(ctx.report_md).write_text(report_markdown(report), encoding="utf-8")
    if branches:
        pass
    print(json.dumps({"status": manifest["status"], "out_dir": str(out_dir),
                      "comparator": comparator.summary()["n_comparisons"],
                      "failures": comparator.summary()["n_fail"],
                      "seconds": manifest["seconds"]}, indent=1))
    return 0 if comparator.summary()["n_fail"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReplayRefusal as exc:
        print("REFUSED(%s): %s" % (type(exc).__name__, exc), file=sys.stderr)
        sys.exit(exc.exit_code)
