#!/usr/bin/env python3
"""FAST standalone example: recompute the G4 depth summary from frozen rows.

Runs entirely inside this package: reads examples/depth_profiles/depth_profiles.csv,
recomputes the frozen G4 summary/peaks through the packaged depth aggregator, and
compares against the pinned accepted reference (4-decimal display + numeric
comparator abs<=1e-9+1e-6*abs(ref)).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "code"))
sys.dont_write_bytecode = True

from rfinal_replay.render import depth_aggregate  # noqa: E402


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()
    example = PKG / "examples/depth_profiles"
    ref_summary = Path(args.reference) if args.reference else example / "reference_depth_summary_fresh.csv"
    fresh = depth_aggregate.recompute_summary(rows(example / "depth_profiles.csv"))
    peaks = depth_aggregate.recompute_peaks(fresh)
    ref = rows(ref_summary)
    ref_peaks = rows(example / "reference_depth_peaks_fresh.csv")
    import math
    if len(fresh) != 20 or len(ref) != 20 or len(peaks) != 3 or len(ref_peaks) != 3:
        raise SystemExit(f"FAIL counts rows={len(fresh)}/{len(ref)} peaks={len(peaks)}/{len(ref_peaks)}")
    want_cols = ["relative_depth", "n_models", "median_r2_b", "rho_H2", "median_r2_eps"]
    for r in (fresh[0], ref[0]):
        if [c for c in want_cols if c not in r]:
            raise SystemExit(f"FAIL schema: {want_cols}")
    for row in fresh + ref:
        for col in want_cols:
            v = row[col]
            if v is None or str(v).strip().lower() in ("", "nan", "null"):
                raise SystemExit(f"FAIL null/NaN in {col}")
            if not math.isfinite(float(v)):
                raise SystemExit(f"FAIL non-finite {col}={v}")
    for p_, r_ in zip(peaks, ref_peaks):
        for col in ("metric", "relative_depth", "value", "n_models"):
            if col not in r_ or col not in {"metric"} and col not in p_:
                raise SystemExit(f"FAIL peak schema {col}")
        if p_["metric"] != r_["metric"]:
            raise SystemExit(f"FAIL peak metric {p_} vs {r_}")
    worst = 0.0
    for f, r in zip(fresh, ref):
        for col in ("relative_depth", "n_models", "median_r2_b", "rho_H2", "median_r2_eps"):
            a, b = float(f[col]), float(r[col])
            worst = max(worst, abs(a - b))
            if not (abs(a - b) <= 1e-9 + 1e-6 * abs(b)):
                raise SystemExit(f"FAIL: {col} {a} vs {b}")
    for p, r in zip(peaks, ref_peaks):
        if (p["metric"] != r["metric"]
                or abs(float(p["relative_depth"]) - float(r["relative_depth"])) > 1e-12
                or abs(p["value"] - float(r["value"])) > 1e-9 + 1e-6 * abs(float(r["value"]))
                or int(p["n_models"]) != int(r["n_models"])):
            raise SystemExit(f"FAIL peak {p} vs {r}")
    print(json.dumps({"example": "depth_profiles", "rows": len(fresh), "peaks": len(peaks),
                      "max_abs_diff": worst, "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
