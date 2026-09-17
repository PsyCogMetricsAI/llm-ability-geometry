"""Format adapters: fresh cache-driver outputs -> renderer input schemas.

Every value is taken from the freshly produced cache-replay outputs of the same
run (never from archived numerical summaries).  Adapters only reshape fields;
they never invent or recompute statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

CELL_FIELDS = ("key", "family", "stage", "domain", "metric", "n_models", "n_families",
               "theta_variant", "theta_source", "rel_geo", "rel_th", "n_items",
               "raw_rho", "partial_rho", "ci_controlled", "ci_bare", "p_cond",
               "bh28_cond", "by28_cond", "status", "notes")


def assemble_grid(cell_paths, family="F1", stage="native", expected_cells=28):
    """Assemble a renderer-ready grid document from fresh per-cell files.

    Defensive uniqueness: duplicate keys (e.g. the same file registered twice in
    a run registry) are refused, and the exact expected cell count is required.
    """
    cells, seen = [], {}
    for path in sorted(cell_paths):
        doc = json.loads(Path(path).read_text())
        key = doc.get("key")
        if key in seen:
            raise ValueError(f"duplicate grid cell key {key!r}: {path} duplicates {seen[key]}")
        seen[key] = path
        cells.append({k: doc[k] for k in CELL_FIELDS if k in doc})
    if not cells:
        raise ValueError("no fresh cell files to assemble")
    if expected_cells is not None and len(cells) != expected_cells:
        raise ValueError(f"grid {family}/{stage}: {len(cells)} cells != expected {expected_cells}")
    return {"schema": "c17-fresh-grid-adapter-v1", "family": family, "stage": stage,
            "m": len(cells), "cells": cells,
            "cell_source": "fresh C10o new28 cell files of this run"}


def build_probe_docs(probe_aggregates_path):
    """Map fresh probe aggregates onto the renderer's h1/h2/h3 pointer shape."""
    doc = json.loads(Path(probe_aggregates_path).read_text())
    rows = doc["rows"]

    def value(key):
        return rows[key]["value"]

    def meta(key):
        row = rows[key]
        return {"value": row["value"], "n_models": row.get("n_models"),
                "source": row.get("source"), "rule": row.get("rule")}

    count = rows["residual_threshold_count"]
    value_txt = str(count.get("value"))
    if "/" in value_txt:
        n_both_txt, n_models_txt = value_txt.split("/", 1)
    else:  # value-only form
        n_both_txt, n_models_txt = value_txt, str(count.get("n_models"))
    h1 = {"schema": "c17-fresh-probe-h1-v1",
          "task_A_pooled": {"median_r2": value("difficulty_readout_median_r2"),
                            "provenance": meta("difficulty_readout_median_r2")},
          "task_C_transfer": {"median_transfer_rho": value("aligned_transfer_median_rho"),
                              "provenance": meta("aligned_transfer_median_rho")}}
    h2 = {"schema": "c17-fresh-probe-h2-v1",
          "rho_H2_best_layer": value("gain_slope_best_layer_rho"),
          "rho_H2_pooled": value("gain_slope_pooled_rho"),
          "provenance": {"best_layer": meta("gain_slope_best_layer_rho"),
                         "pooled": meta("gain_slope_pooled_rho")}}
    h3 = {"schema": "c17-fresh-probe-h3-v1",
          "pooled_summary": {"r2_median": value("residual_readout_median_r2"),
                             "provenance": meta("residual_readout_median_r2")},
          "fractions": {"n_both": int(float(n_both_txt)), "n_models": int(float(n_models_txt)),
                        "provenance": count}}
    return {"h1": h1, "h2": h2, "h3": h3}


def write_json(path: Path, obj) -> str:
    import hashlib
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=1) + "\n"
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()
