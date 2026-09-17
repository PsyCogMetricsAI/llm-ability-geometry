#!/usr/bin/env python3
"""Deterministic regeneration of the retained G1/H1 Task-C prediction vectors.

Scope: Task C of the frozen `ext_P2_geo_20260914/h1/code/run_h1.py` only
(transfer probes + the transient pair_cache used by the secondary item
bootstrap).  Task A / per-layer Task B are NOT executed.

The numeric routines below are verbatim re-implementations of the frozen
`h1/code/h1_lib.py` formulas (zscore_dims, global_item_folds, kfold_splits,
fit_ridge_primal, pca_fit, pca_transform, spearman, percentile_ci) and the
Task-C section of `run_h1.py`: same seed 20260914, same SeedSequence child
order (0 fold, 2 ab-split, 3 pairs, 4 boot-pair, 5 boot-item), same
ALPHA_GRID=logspace(-4,4,17), K_OUTER=10, K_INNER=5, R_CAP=64, N_PAIRS=240,
INNER_SEED_BASE=90000000, and the same pair order.  Inputs are the saved
per-model code JSONL item_hidden observations and fit_panel/b_loo.npz; no
hidden state is regenerated, no GPU, no new probe design, CPU only.

Outputs (only under --out): pair_cache_v1.npz (the persisted intermediate),
g1_taskc_cache_manifest.json (source/code/seed manifest + per-pair comparison
against the retained h1/transfer_rho.csv).

Usage:
  <venv-cpu-replay>/bin/python code/c17_replay_supplements_b/g1_taskc_regen.py \
      --analysis-root <R> --out <DIR> [--limit-pairs N] [--no-cache]
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.linalg import orthogonal_procrustes
from scipy.stats import rankdata

SEED = 20260914
ALPHA_GRID = np.logspace(-4, 4, 17)
K_OUTER, K_INNER = 10, 5
N_PAIRS = 240
R_CAP = 64
N_BOOT = 2000
INNER_SEED_BASE = 90000000

_IH_RE = re.compile(r'"item_hidden"\s*:\s*\[(.*?)\]', re.S)
_IID_RE = re.compile(r'"item_id"\s*:\s*"([^"]*)"')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------- h1_lib formulas
def zscore_dims(H):
    mu = H.mean(0)
    sd = H.std(0)
    z = np.zeros_like(H)
    nz = sd > 0
    z[:, nz] = (H[:, nz] - mu[nz]) / sd[nz]
    return z


def global_item_folds(n_items_universe, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items_universe)
    chunks = np.array_split(perm, k)
    fold_of = np.empty(n_items_universe, dtype=int)
    for fid, ch in enumerate(chunks):
        fold_of[ch] = fid
    return fold_of


def kfold_splits(idx, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    chunks = np.array_split(perm, k)
    out = []
    for i in range(k):
        te = chunks[i]
        tr = np.concatenate([chunks[j] for j in range(k) if j != i]) if k > 1 else chunks[i]
        if len(te) > 0 and len(tr) > 2:
            out.append((tr, te))
    return out


def fit_ridge_primal(X, y, alpha_grid, inner_k, seed):
    n = len(y)
    idx = np.arange(n)
    inner = kfold_splits(idx, inner_k, seed)
    sse = np.zeros(len(alpha_grid))
    for itr, ite in inner:
        Xtr, Xte = X[itr], X[ite]
        ytr = y[itr]
        xbar = Xtr.mean(0); ybar = ytr.mean()
        Xc = Xtr - xbar; yc = ytr - ybar
        G = Xc.T @ Xc
        scale = np.trace(G) / X.shape[1]
        if scale <= 0:
            continue
        rhs = Xc.T @ yc
        yte = y[ite]
        for ai, av in enumerate(alpha_grid):
            lam = av * scale
            w = sla.solve(G + lam * np.eye(G.shape[0]), rhs, assume_a="pos")
            pred = (Xte - xbar) @ w + ybar
            sse[ai] += np.sum((yte - pred) ** 2)
    av = alpha_grid[int(np.argmin(sse))] if sse.any() else alpha_grid[len(alpha_grid) // 2]
    xbar = X.mean(0); ybar = y.mean()
    Xc = X - xbar; yc = y - ybar
    G = Xc.T @ Xc
    scale = np.trace(G) / X.shape[1]
    lam = av * scale
    w = sla.solve(G + lam * np.eye(G.shape[0]), Xc.T @ yc, assume_a="pos")
    c = ybar - xbar @ w
    return w, c, float(lam)


def pca_fit(X, r):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(r, Vt.shape[0])
    return mu, Vt[:r]


def pca_transform(Xnew, mu, comps):
    return (Xnew - mu) @ comps.T


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = rankdata(a); rb = rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
    if d == 0:
        return float("nan")
    return float(np.sum(ra * rb) / d)


def percentile_ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


# ------------------------------------------------------------- path helpers
def tag_map(project_root):
    inv = project_root / "ext_P2_1_itemmodel_20260913/inventory.csv"
    m = {}
    with inv.open() as fh:
        for row in csv.DictReader(fh):
            if row["domain"] == "code":
                m[row["model_id"]] = row["model"]
    return m


def find_jsonl(project_root, tag):
    for d in (project_root / "Imports/geometry/panel_geom_local_20260607",
              project_root / "Imports/geometry/panel_geom_expansion_20260607"):
        p = d / ("geom_%s.jsonl" % tag)
        if p.exists():
            return p
    return None


def load_bloo(project_root):
    with np.load(project_root / "ext_P2_geo_20260914/fit_panel/b_loo.npz",
                 allow_pickle=True) as z:
        model_ids = [str(x) for x in z["model_ids"]]
        item_ids = [str(x) for x in z["item_ids"]]
        b_loo = z["b_loo"].astype(np.float64)
    return model_ids, item_ids, b_loo


def load_pooled(path):
    iids, vecs = [], []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            iid = _IID_RE.search(line).group(1)
            body = _IH_RE.search(line).group(1)
            iids.append(iid)
            vecs.append(np.fromstring(body, sep=",", dtype=np.float64))
    return iids, np.vstack(vecs)


def _globals(project_root):
    model_ids, item_ids, b_loo = load_bloo(project_root)
    col_of = {iid: j for j, iid in enumerate(item_ids)}
    row_of = {mid: i for i, mid in enumerate(model_ids)}
    n_univ = len(item_ids)
    ch = np.random.SeedSequence(SEED).spawn(6)
    fold_of = global_item_folds(n_univ, K_OUTER, ch[0])
    rng_ab = np.random.default_rng(ch[2])
    perm_ab = rng_ab.permutation(n_univ)
    foldA = set(int(x) for x in perm_ab[:n_univ // 2])
    foldB = set(int(x) for x in perm_ab[n_univ // 2:])
    rng_pairs = np.random.default_rng(ch[3])
    perm = rng_pairs.permutation(len(model_ids))
    pairs = [(int(perm[i]), int(perm[(i + 1) % len(model_ids)])) for i in range(len(model_ids))]
    seen = set(pairs)
    while len(pairs) < N_PAIRS:
        a = int(rng_pairs.integers(len(model_ids))); b = int(rng_pairs.integers(len(model_ids)))
        if a != b and (a, b) not in seen:
            seen.add((a, b)); pairs.append((a, b))
    return dict(model_ids=model_ids, item_ids=item_ids, b_loo=b_loo, col_of=col_of,
                row_of=row_of, n_univ=n_univ, foldA=foldA, foldB=foldB, pairs=pairs,
                tmap=tag_map(project_root))


def _model_cache(project_root, g, mid):
    path = find_jsonl(project_root, g["tmap"][mid])
    if path is None:
        raise FileNotFoundError(mid)
    iids, H = load_pooled(path)
    cols = np.array([g["col_of"][i] for i in iids])
    Htil = zscore_dims(H)
    y = g["b_loo"][g["row_of"][mid], cols]
    rowsA = np.array([i for i, c in enumerate(cols) if int(c) in g["foldA"]])
    rowsB = np.array([i for i, c in enumerate(cols) if int(c) in g["foldB"]])
    colsA, colsB = cols[rowsA], cols[rowsB]
    if len(rowsA) >= 3:
        mu, comps = pca_fit(Htil[rowsA], R_CAP)
        XfA = pca_transform(Htil[rowsA], mu, comps)
        XfB = (pca_transform(Htil[rowsB], mu, comps)
               if len(rowsB) else np.zeros((0, comps.shape[0])))
    else:
        XfA = XfB = None
    return dict(path=str(path), cols=cols, XfA=XfA, colsA=colsA,
                yA=y[rowsA] if len(rowsA) else None, XfB=XfB, colsB=colsB,
                yB=y[rowsB] if len(rowsB) else None,
                idxA={int(c): i for i, c in enumerate(colsA)},
                idxB={int(c): i for i, c in enumerate(colsB)})


def _transfer_pair(g, cache, ai, bi):
        model_ids = g["model_ids"]
        A = cache[model_ids[ai]]; B = cache[model_ids[bi]]
        if A["XfA"] is None or B["XfA"] is None:
            return None
        commonA = sorted(set(int(c) for c in A["colsA"]) & set(int(c) for c in B["colsA"]))
        r = min(R_CAP, len(commonA) - 1, A["XfA"].shape[1], B["XfA"].shape[1])
        if r < 8 or len(commonA) < 10:
            return None
        XA_c = A["XfA"][[A["idxA"][c] for c in commonA], :r]
        XB_c = B["XfA"][[B["idxA"][c] for c in commonA], :r]
        R, _ = orthogonal_procrustes(XB_c, XA_c)
        w, cst, lam = fit_ridge_primal(A["XfA"][:, :r], A["yA"], ALPHA_GRID, K_INNER,
                                       INNER_SEED_BASE)
        if B["XfB"] is None or len(B["yB"]) < 10:
            return None
        yhatB = (B["XfB"][:, :r] @ R) @ w + cst
        rho = spearman(yhatB, B["yB"])
        rho_w = float("nan")
        if A["XfB"] is not None and len(A["yB"]) >= 3:
            yhatA = A["XfB"][:, :r] @ w + cst
            rho_w = spearman(yhatA, A["yB"])
        return dict(rho=rho, rho_w=rho_w, r=r, ncommonA=len(commonA),
                    model_A=model_ids[ai], model_B=model_ids[bi],
                    yhatB=yhatB, yB=B["yB"], colsB=np.array([int(c) for c in B["colsB"]]))


def regenerate_pair_rows(project_root, pair_indices):
    """Bounded independent regeneration path: rebuild only the models needed for
    the requested pair indices and return their Task-C rows (no cache reuse)."""
    g = _globals(project_root)
    needed = set()
    for pi in pair_indices:
        ai, bi = g["pairs"][pi]
        needed.add(g["model_ids"][ai]); needed.add(g["model_ids"][bi])
    cache = {mid: _model_cache(project_root, g, mid) for mid in sorted(needed)}
    rows = []
    for pi in pair_indices:
        ai, bi = g["pairs"][pi]
        res = _transfer_pair(g, cache, ai, bi)
        rows.append({"pair_index": pi, "model_A": g["model_ids"][ai],
                     "model_B": g["model_ids"][bi],
                     "rho": (res["rho"] if res else float("nan")),
                     "rho_within": (res["rho_w"] if res else float("nan")),
                     "r": (res["r"] if res else -1),
                     "ncommonA": (res["ncommonA"] if res else 0),
                     "yhat": (res["yhatB"] if res else np.zeros(0)),
                     "y": (res["yB"] if res else np.zeros(0)),
                     "cols": (res["colsB"] if res else np.zeros(0, dtype=np.int64))})
    return rows


def build_taskc(project_root, limit_pairs=None, progress=True):
    t0 = time.time()
    g = _globals(project_root)
    model_ids = g["model_ids"]
    cache = {}
    used_paths = {}
    for mid in model_ids:
        cache[mid] = _model_cache(project_root, g, mid)
        used_paths[mid] = cache[mid]["path"]
        if progress:
            print("[cache] %s %.1fs" % (mid, time.time() - t0), flush=True)
    pairs = g["pairs"]

    trows = []
    n_use = len(pairs) if limit_pairs is None else min(limit_pairs, len(pairs))
    for pi in range(len(pairs)):
        ai, bi = pairs[pi]
        res = _transfer_pair(g, cache, ai, bi)
        if res is None:
            trows.append({"pair_index": pi, "model_A": model_ids[ai],
                          "model_B": model_ids[bi], "rho": float("nan"),
                          "rho_within": float("nan"), "r": -1, "n_commonA": 0,
                          "yhat": np.zeros(0), "y": np.zeros(0),
                          "cols": np.zeros(0, dtype=np.int64)})
            continue
        trows.append({"pair_index": pi, "model_A": res["model_A"], "model_B": res["model_B"],
                      "rho": res["rho"], "rho_within": res["rho_w"], "r": res["r"],
                      "n_commonA": res["ncommonA"],
                      "yhat": res["yhatB"], "y": res["yB"], "cols": res["colsB"]})
        if pi + 1 >= n_use and limit_pairs is not None:
            break
        if progress and (pi + 1) % 40 == 0:
            print("[taskC] %d/%d pairs %.1fs" % (pi + 1, len(pairs), time.time() - t0),
                  flush=True)
    foldB_arr = np.array(sorted(g["foldB"]))
    return {"model_ids": model_ids, "item_ids": g["item_ids"], "foldB": foldB_arr,
            "n_univ": g["n_univ"], "pairs": trows, "used_paths": used_paths,
            "wall_s": time.time() - t0}


def item_bootstrap_from_rows(rows, foldB_arr, n_univ, n_boot=N_BOOT):
    """Verbatim replay of the run_h1.py secondary item bootstrap from cached rows."""
    rng_bi = np.random.default_rng(np.random.SeedSequence(SEED).spawn(6)[5])
    pair_lut = []
    for row in rows:
        if len(row["yhat"]) == 0:
            continue
        lut = np.full(n_univ, -1, dtype=int)
        lut[row["cols"].astype(int)] = np.arange(len(row["cols"]))
        pair_lut.append((row["yhat"], row["y"], lut))
    boot_item = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draw = rng_bi.choice(foldB_arr, size=len(foldB_arr), replace=True)
        meds = []
        for yhatB, yB, lut in pair_lut:
            take = lut[draw]
            take = take[take >= 0]
            if len(take) < 5:
                continue
            r = spearman(yhatB[take], yB[take])
            if not np.isnan(r):
                meds.append(r)
        if meds:
            boot_item[b] = np.median(meds)
    return {"ci": percentile_ci(boot_item),
            "n_valid_draws": int(np.isfinite(boot_item).sum()),
            "n_pairs_used": len(pair_lut)}




def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    analysis_root = Path(args.analysis_root).resolve()
    project_root = analysis_root.parent
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    taskc = build_taskc(project_root, limit_pairs=args.limit_pairs)
    csv_path = project_root / "ext_P2_geo_20260914/h1/transfer_rho.csv"
    with csv_path.open() as fh:
        ref = list(csv.DictReader(fh))
    diffs, mismatches = [], []
    for row in taskc["pairs"]:
        rr = ref[row["pair_index"]]
        if row["model_A"] != rr["model_A"] or row["model_B"] != rr["model_B"]:
            mismatches.append({"pair": row["pair_index"], "kind": "pair_order",
                               "fresh": [row["model_A"], row["model_B"]],
                               "ref": [rr["model_A"], rr["model_B"]]})
            continue
        r_ref = float(rr["rho_transfer"])
        if np.isnan(row["rho"]) and np.isnan(r_ref):
            continue
        diffs.append(abs(row["rho"] - r_ref))
        if row["r"] != int(float(rr["r_pca"])) or row["n_commonA"] != int(float(rr["n_commonA"])):
            mismatches.append({"pair": row["pair_index"], "kind": "count",
                               "fresh": [row["r"], row["n_commonA"]],
                               "ref": [rr["r_pca"], rr["n_commonA"]]})
    timing = taskc["wall_s"]
    print(json.dumps({"pairs_computed": len(taskc["pairs"]), "max_abs_rho_diff":
                      (max(diffs) if diffs else None), "n_rho_compared": len(diffs),
                      "count_or_order_mismatches": mismatches[:5],
                      "wall_s": round(timing, 2)}))
    if mismatches:
        print("REFUSING to cache: per-pair mismatch vs transfer_rho.csv", file=sys.stderr)
        return 6
    if args.no_cache or args.limit_pairs is not None:
        return 0
    offsets = [0]
    for row in taskc["pairs"]:
        offsets.append(offsets[-1] + len(row["yhat"]))
    np.savez(out / "pair_cache_v1.npz",
             pair_index=np.array([r["pair_index"] for r in taskc["pairs"]], dtype=np.int64),
             model_A=np.array([r["model_A"] for r in taskc["pairs"]]),
             model_B=np.array([r["model_B"] for r in taskc["pairs"]]),
             offset=np.array(offsets, dtype=np.int64),
             yhat=np.concatenate([r["yhat"] for r in taskc["pairs"]]),
             y=np.concatenate([r["y"] for r in taskc["pairs"]]),
             cols=np.concatenate([r["cols"] for r in taskc["pairs"]]).astype(np.int64),
             rho=np.array([r["rho"] for r in taskc["pairs"]], dtype=np.float64),
             rho_within=np.array([r["rho_within"] for r in taskc["pairs"]], dtype=np.float64),
             r=np.array([r["r"] for r in taskc["pairs"]], dtype=np.int64),
             n_commonA=np.array([r["n_commonA"] for r in taskc["pairs"]], dtype=np.int64),
             foldB=taskc["foldB"].astype(np.int64), n_univ=np.int64(taskc["n_univ"]),
             seed=np.int64(SEED), n_boot=np.int64(N_BOOT), r_cap=np.int64(R_CAP))
    h1src = project_root / "ext_P2_geo_20260914/h1"
    manifest = {
        "schema": "c17-g1-taskc-pair-cache-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "code/c17_replay_supplements_b/g1_taskc_regen.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__,
        "seed": SEED, "seed_children": {"fold": 0, "ab_split": 2, "pairs": 3,
                                        "boot_pair": 4, "boot_item": 5},
        "alpha_grid": [float(x) for x in ALPHA_GRID], "k_outer": K_OUTER,
        "k_inner": K_INNER, "r_cap": R_CAP, "n_boot": N_BOOT,
        "inner_seed_base": INNER_SEED_BASE, "n_pairs": len(taskc["pairs"]),
        "n_univ": taskc["n_univ"], "foldB_len": int(len(taskc["foldB"])),
        "cache_file": "pair_cache_v1.npz",
        "cache_sha256": sha256_file(out / "pair_cache_v1.npz"),
        "per_pair_comparator": {
            "path": "ext_P2_geo_20260914/h1/transfer_rho.csv",
            "sha256": sha256_file(csv_path),
            "n_rho_compared": len(diffs),
            "max_abs_rho_diff": (max(diffs) if diffs else None),
            "pair_order_and_counts_exact": not mismatches,
            "tolerance": "<= 1e-9 + 1e-6*abs(ref)",
            "all_within_tolerance": bool(diffs and max(diffs) <= 1e-9 + 1e-6 * max(
                abs(float(r["rho_transfer"])) for r in ref)),
        },
        "source_pins": {
            "h1/run_h1.py": sha256_file(h1src / "code/run_h1.py"),
            "h1/h1_lib.py": sha256_file(h1src / "code/h1_lib.py"),
            "fit_panel/b_loo.npz": sha256_file(
                project_root / "ext_P2_geo_20260914/fit_panel/b_loo.npz"),
            "ext_P2_1_itemmodel_20260913/inventory.csv": sha256_file(
                project_root / "ext_P2_1_itemmodel_20260913/inventory.csv"),
            "jsonl_used": {m: sha256_file(p) for m, p in taskc["used_paths"].items()},
        },
        "scope_note": "Task C only; no Task A, no per-layer Task B, no new probe design, "
                      "no GPU, no hidden-state regeneration. Same-estimator reproduction of "
                      "the retained item-bootstrap intermediate.",
        "wall_s": round(timing, 2),
    }
    (out / "g1_taskc_cache_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"cache": str(out / "pair_cache_v1.npz"),
                      "manifest": str(out / "g1_taskc_cache_manifest.json"),
                      "max_abs_rho_diff": manifest["per_pair_comparator"]["max_abs_rho_diff"],
                      "wall_s": manifest["wall_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
