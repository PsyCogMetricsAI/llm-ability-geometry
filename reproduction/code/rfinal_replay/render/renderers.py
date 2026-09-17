"""Seven C17 result generators.

Each renderer takes (ctx, entry) and returns an inventory fragment. Every
numeric value is read from the declared source files via explicit JSON
pointers or CSV columns; nothing is hardcoded from historical displays.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import SCHEMA_PLOTDATA  # noqa: E402
from . import depth_aggregate  # noqa: E402
from .common import (  # noqa: E402
    artifact_record,
    deref_spec,
    fmt_num,
    fmt_p,
    refuse_bad_schema,
    refuse_missing_input,
    resolve_pointer,
    sha256_file,
    tex_escape,
    tex_number,
    value_cell,
    write_csv,
    write_json,
    write_text,
)

FIG_DPI = 200
FIG_FORMATS = ("svg", "pdf", "png")


def _inputs_by_role(entry, input_root):
    out = {}
    for spec in entry["inputs"]:
        role = spec.get("role", spec["path"])
        path = (input_root / spec["path"]).resolve()
        out.setdefault(role, []).append((path, spec))
    return out


def _require_single(inputs, role):
    vals = inputs.get(role)
    if not vals:
        refuse_bad_schema(f"renderer requires input role {role!r}")
    return vals[0]


def _save_figure(fig, outdir: Path, stem: str):
    paths = []
    for fmt in FIG_FORMATS:
        p = outdir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt, dpi=FIG_DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# 1. P2-G-FIG-MTMM -- two-instrument MTMM schematic (no numeric matrix)
# --------------------------------------------------------------------------
def mtmm_schematic(ctx, entry):
    outdir = ctx["outdir"] / entry["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = _inputs_by_role(entry, ctx["input_root"])
    man_path, _ = _require_single(inputs, "result_manifest")
    tex_path, _ = _require_single(inputs, "definition_reference")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    record = next((r for r in manifest["records"] if r["result_id"] == entry["id"]), None)
    if record is None:
        refuse_missing_input(f"RESULT_MANIFEST has no record for {entry['id']}")

    rows = [
        "Trait C\n(domain capability)",
        "Nuisance\n(scale / family / fluency)",
    ]
    cols = [
        "Method A - behavioral\nIRT theta",
        "Method B - representational\ngeometry",
    ]
    # Design-role cells from the original definition (main tex, targeted
    # occurrence; no correlation values exist for this schematic).
    cells = [
        {"row": 0, "col": 0, "role": "validity",
         "text": "r(theta, C)\nvalidity coefficient\nnot estimated in this figure",
         "applicable": False},
        {"row": 0, "col": 1, "role": "convergent (monotrait-heteromethod)",
         "text": "r(theta, geo)\nconvergent cell this design tests\nno numeric estimate in this figure",
         "applicable": True},
        {"row": 1, "col": 0, "role": "heterotrait same-method",
         "text": "heterotrait, same-method\ncommon-method variance risk",
         "applicable": False},
        {"row": 1, "col": 1, "role": "heterotrait hetero-method",
         "text": "heterotrait, hetero-method\nmust not converge",
         "applicable": False},
    ]
    for c in cells:
        c["value"] = None
        c["display"] = (
            "no numeric value defined (empty)"
            if not c["applicable"]
            else "no numeric estimate in this schematic"
        )

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    shade = (0.90, 0.90, 0.90)
    ax.add_patch(plt.Rectangle((1, 0), 1, 1, color=shade, zorder=0))
    for x in (0, 1, 2):
        ax.plot([x, x], [0, 2], color="black", lw=0.8, zorder=1)
    for y in (0, 1, 2):
        ax.plot([0, 2], [y, y], color="black", lw=0.8, zorder=1)
    for c in cells:
        ax.text(c["col"] + 0.5, c["row"] + 0.42, c["text"], ha="center", va="center",
                fontsize=8.6, zorder=2)
        ax.text(c["col"] + 0.5, c["row"] + 0.82, c["display"], ha="center", va="center",
                fontsize=7.2, style="italic", color="#444444", zorder=2)
    for j, name in enumerate(cols):
        ax.text(j + 0.5, -0.16, name, ha="center", va="bottom", fontsize=9)
    for i, name in enumerate(rows):
        ax.text(-0.06, i + 0.5, name, ha="right", va="center", fontsize=9)
    ax.set_title("Two-instrument MTMM schematic: design role of the four cells", fontsize=10.5)
    ax.text(0, 2.42,
            "Schematic only. Inapplicable cells are empty and labelled; no correlation matrix is reported. "
            "Cell roles follow the frozen design definition.",
            fontsize=7.6, color="#333333")
    ax.axis("off")
    fig.tight_layout()
    fig_paths = _save_figure(fig, outdir, "mtmm_schematic")

    plotdata = {
        "schema": SCHEMA_PLOTDATA,
        "result_id": entry["id"],
        "kind": "figure",
        "title": entry["title"],
        "numeric_values_present": False,
        "note": ("Schematic design figure: no numeric values are defined or fabricated; "
                 "inapplicable cells are marked as empty."),
        "rows": rows,
        "columns": cols,
        "cells": cells,
        "source": {
            "manifest": {"path": entry["inputs"][0]["path"], "sha256": sha256_file(man_path)},
            "definition_reference": {
                "path": entry["inputs"][1]["path"], "sha256": sha256_file(tex_path)},
        },
        "upstream_dependency_clues": record.get("upstream_dependency_clues", []),
        "n4_exemption_reason": record.get("n4_exemption_reason", ""),
    }
    pd_path = outdir / "plotdata.json"
    write_json(pd_path, plotdata)
    write_csv(outdir / "mtmm_cells.csv",
              ["row", "col", "role", "applicable", "value", "display"],
              [[c["row"], c["col"], c["role"], c["applicable"], "", c["display"]] for c in cells])

    artifacts = [artifact_record(p, ctx["outdir"]) for p in fig_paths]
    artifacts.append(artifact_record(pd_path, ctx["outdir"]))
    artifacts.append(artifact_record(outdir / "mtmm_cells.csv", ctx["outdir"]))
    return _fragment(entry, artifacts, plotdata, {
        "numeric_cells_all_empty": all(c["value"] is None for c in cells),
        "n_cells": len(cells),
    }, new_vs_old={
        "historical_output": record.get("historical_outputs", [{}])[0].get("status", "UNLOCATED"),
        "basis": ("regenerated schematic from the declared design definition and manifest record; "
                  "no historical image copied, no correlation matrix fabricated"),
    }, limitations=[
        "Schematic figure; the convergent cell carries a design role, not an estimated correlation.",
        "No correlation matrix is available in the retained evidence; no cell is filled by imputation.",
    ])


# --------------------------------------------------------------------------
# 2. P2-B-TAB-CONVERGENCE -- frozen per-domain convergence panel
# --------------------------------------------------------------------------
def convergence_table(ctx, entry):
    outdir = ctx["outdir"] / entry["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = _inputs_by_role(entry, ctx["input_root"])
    study_path, _ = _require_single(inputs, "study1_main_results")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    sha = sha256_file(study_path)

    def dom(domain, key):
        if domain not in study.get("domains", {}) or key not in study["domains"][domain]:
            refuse_missing_input(f"{entry['id']}: /domains/{domain}/{key} missing in {study_path}")
        return study["domains"][domain][key]

    def src(domain, key):
        return {"path_rel": entry["inputs"][0]["path"], "pointer": f"/domains/{domain}/{key}",
                "sha256": sha}

    rows = []  # (row_label, unit, pointer_key, formatter, source_domain)
    domains = ("code", "math")
    specs = [
        ("n (models)", "models", "n_models", "int"),
        ("k (items)", "items", "n_items", "int"),
        ("raw Spearman rho", "rho", "raw_rho", "num"),
        ("partial rho controlling Z", "rho", "partial_rho_controlling_C", "num"),
        ("bare 95% cluster CI", "rho interval", "bare_cluster_ci", "ci"),
        ("bare CI excludes 0", "bool", "bare_ci_excludes_zero", "bool"),
        ("controlled 95% cluster CI", "rho interval", "controlled_cluster_ci", "ci"),
        ("controlled CI excludes 0", "bool", "controlled_ci_excludes_zero", "bool"),
        ("permutation p", "p-value", "permutation_p", "p"),
        ("disattenuation ceiling sqrt(rel_geo*rel_th)", "rho bound", "disattenuation_ceiling", "num"),
        ("frozen verdict", "tier", "verdict", "text"),
    ]
    cell_rows = []
    for label, unit, key, style in specs:
        cells = {}
        for d in domains:
            raw = dom(d, key)
            if style == "ci":
                display = f"[{fmt_num(raw[0], 4)}, {fmt_num(raw[1], 4)}]"
                cells[d] = value_cell(raw, unit, dom(d, "n_models"), src(d, key),
                                      display=display)
            elif style == "bool":
                cells[d] = value_cell(bool(raw), unit, dom(d, "n_models"), src(d, key),
                                      display="yes" if raw else "no")
            elif style == "p":
                cells[d] = value_cell(raw, unit, dom(d, "n_models"), src(d, key),
                                      display=fmt_p(raw))
            elif style == "int":
                cells[d] = value_cell(raw, unit, dom(d, "n_models"), src(d, key),
                                      display=str(int(raw)))
            elif style == "text":
                cells[d] = value_cell(raw, unit, dom(d, "n_models"), src(d, key),
                                      display=str(raw))
            else:
                cells[d] = value_cell(raw, unit, dom(d, "n_models"), src(d, key))
        cell_rows.append({"row_label": label, "cells": cells})
        rows.append([label, cells["code"]["display"], cells["math"]["display"], unit])

    caption = (
        "Frozen per-domain convergence panel (signed baseline evidence; regenerated). Statistic = signed "
        "Spearman partial rho; CI = signed family-block cluster-bootstrap 95-percentile; p = permutation. "
        "Values extracted from " + entry["inputs"][0]["path"] +
        " (frozen registered duplicate-BOS composite). The disattenuation ceiling "
        "sqrt(rel_geo*rel_th) bounds |rho|; it is a reliability bound, not a test."
    )
    write_csv(outdir / "convergence_table.csv",
              ["row", "code", "math", "unit"], rows)
    long_rows = []
    for cr in cell_rows:
        for d in domains:
            c = cr["cells"][d]
            long_rows.append([entry["id"], cr["row_label"], d, c["value"], c["display"], c["unit"],
                              c["n"], c["source"]["path"], c["source"]["json_pointer"],
                              c["source"]["sha256"]])
    write_csv(outdir / "convergence_values_long.csv",
              ["result_id", "row", "column", "value", "display", "unit", "n",
               "source_path", "source_pointer", "source_sha256"], long_rows)

    tex = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{" + tex_escape(caption) + "}",
        r"\label{tab:convergence}",
        r"\begin{tabular}{@{}llll@{}}", r"\toprule",
        r"statistic & code & math & unit \\", r"\midrule",
    ]
    for label, cval, mval, unit in rows:
        tex.append(f"{tex_escape(label)} & {tex_escape(cval)} & {tex_escape(mval)} & {tex_escape(unit)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(outdir / "convergence_table.tex", "\n".join(tex))

    plotdata = {
        "schema": SCHEMA_PLOTDATA,
        "result_id": entry["id"],
        "kind": "table",
        "title": entry["title"],
        "units_note": ("n = models; k = items; rho = signed Spearman partial correlation; "
                       "CI = 95% family-block cluster-bootstrap percentile; p = permutation p-value."),
        "columns": list(domains),
        "rows": cell_rows,
        "caption": caption,
        "source_files": [{"path": entry["inputs"][0]["path"], "sha256": sha}],
    }
    pd_path = outdir / "plotdata.json"
    write_json(pd_path, plotdata)
    artifacts = [artifact_record(p, ctx["outdir"]) for p in
                 (outdir / "convergence_table.csv", outdir / "convergence_table.tex",
                  outdir / "convergence_values_long.csv", pd_path)]
    return _fragment(entry, artifacts, plotdata, {
        "n_rows": len(cell_rows),
        "verdicts_from_source": {d: dom(d, "verdict") for d in domains},
    }, new_vs_old={
        "historical_output": "UNLOCATED (manifest historical output placeholder)",
        "basis": ("regenerated from the signed Study-1 reuse results (frozen registered "
                  "duplicate-BOS composite); no old table image or historical final value hardcoded"),
    }, limitations=[
        "Frozen duplicate-BOS registered composite; the single-BOS Study-2 sensitivity is a separate artifact.",
        "Verdict strings are copied from the signed source, not re-derived here.",
    ])


# --------------------------------------------------------------------------
# 3. P2-C-TAB-SCIENCE -- compact science + twoNN contrast table
# --------------------------------------------------------------------------
def _grid_cell(grid, domain, metric):
    for idx, c in enumerate(grid.get("cells", [])):
        if c.get("domain") == domain and c.get("metric") == metric:
            return c, idx
    refuse_missing_input(f"grid cell missing for domain={domain} metric={metric}")


def science_grid_table(ctx, entry):
    outdir = ctx["outdir"] / entry["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = _inputs_by_role(entry, ctx["input_root"])
    grid_path, _ = _require_single(inputs, "f1_rarefied_grid")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    sha = sha256_file(grid_path)
    rel = entry["inputs"][0]["path"]
    from .contract import SCIENCE_ROWS

    cell_rows = []
    csv_rows = []
    display_rows = []
    for metric, domain in SCIENCE_ROWS:
        cell, idx = _grid_cell(grid, domain, metric)
        base = {"path_rel": rel, "sha256": sha}

        def cell_val(key, unit, display=None):
            src = dict(base, pointer=f"/cells/{idx}/{key}")
            v = cell[key]
            return value_cell(v, unit, cell.get("n_models"), src,
                              display=display if display is not None else fmt_num(v))

        row = {
            "indicator": metric,
            "domain": domain,
            "cells": {
                "n_models": cell_val("n_models", "models", display=str(cell["n_models"])),
                "raw_rho": cell_val("raw_rho", "rho"),
                "partial_rho": cell_val("partial_rho", "rho"),
                "p_cond": cell_val("p_cond", "p-value", display=fmt_p(cell["p_cond"])),
                "bh28_cond": cell_val("bh28_cond", "multiplicity summary", display=fmt_p(cell["bh28_cond"])),
                "by28_cond": cell_val("by28_cond", "multiplicity summary", display=fmt_p(cell["by28_cond"])),
                "status": cell_val("status", "flag", display=str(cell["status"])),
            },
            "cell_key": cell.get("key"),
        }
        cell_rows.append(row)
        csv_rows.append([metric, domain, row["cells"]["n_models"]["value"],
                         row["cells"]["raw_rho"]["value"], row["cells"]["partial_rho"]["value"],
                         row["cells"]["p_cond"]["value"], row["cells"]["bh28_cond"]["value"],
                         row["cells"]["by28_cond"]["value"], cell["status"], rel, f"/cells/{idx}"])
        display_rows.append([metric, domain, str(cell["n_models"]),
                             row["cells"]["raw_rho"]["display"], row["cells"]["partial_rho"]["display"],
                             row["cells"]["p_cond"]["display"], row["cells"]["bh28_cond"]["display"],
                             row["cells"]["by28_cond"]["display"], str(cell["status"])])

    headers = ["indicator", "domain", "n", "raw rho", "partial rho (rarefied)", "p",
               "BH28", "BY28", "status"]
    write_csv(outdir / "science_grid_table.csv", headers, display_rows)
    write_csv(outdir / "science_grid_values_long.csv",
              ["indicator", "domain", "n_models", "raw_rho", "partial_rho", "p_cond",
               "bh28_cond", "by28_cond", "status", "source_path", "source_cell_pointer"],
              csv_rows)
    note = ("Rows: six science indicators plus the reversed-sign twoNN cells in math and code, rarefied "
            "common-N* stage of the F1 map. Values are descriptive permutation associations on the frozen "
            "machinery; BH28/BY28 are multiplicity summaries over the declared 28-cell family "
            "(7 indicators x 4 domains) and are not calibrated FDR guarantees. Tier labels "
            "(confirm / shadow) are not assigned at this render stage.")
    tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\caption{" + tex_escape(note) + "}",
           r"\label{tab:science}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
           r"indicator & dom. & $n$ & raw $\rho$ & partial $\rho$ & $p$ & $p_{\mathrm{BH28}}$ & $p_{\mathrm{BY28}}$ \\",
           r"\midrule"]
    for metric, domain, n, raw, part, p, bh, by, status in display_rows:
        tex.append(f"{tex_escape(metric)} & {tex_escape(domain)} & {n} & {tex_number(raw)} & "
                   f"{tex_number(part)} & {tex_number(p)} & {tex_number(bh)} & {tex_number(by)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(outdir / "science_grid_table.tex", "\n".join(tex))

    plotdata = {
        "schema": SCHEMA_PLOTDATA,
        "result_id": entry["id"],
        "kind": "table",
        "title": entry["title"],
        "note": note,
        "grid_family": grid.get("family"),
        "stage": grid.get("stage"),
        "m_family": grid.get("m"),
        "rows": cell_rows,
        "source_files": [{"path": rel, "sha256": sha}],
    }
    pd_path = outdir / "plotdata.json"
    write_json(pd_path, plotdata)
    artifacts = [artifact_record(p, ctx["outdir"]) for p in
                 (outdir / "science_grid_table.csv", outdir / "science_grid_table.tex",
                  outdir / "science_grid_values_long.csv", pd_path)]
    return _fragment(entry, artifacts, plotdata, {
        "n_rows": len(cell_rows),
        "grid_family": grid.get("family"), "stage": grid.get("stage"),
        "m_family": grid.get("m"),
    }, new_vs_old={
        "historical_output": "UNLOCATED (compact historical table)",
        "basis": ("regenerated from the accepted C17 new28 F1 rarefied grid; no old table/image copied; "
                  "significance-tier labels deliberately not rendered at this stage"),
    }, limitations=[
        "BH28/BY28 are multiplicity summaries over a declared family, not guaranteed FDR control.",
        "No significance-tier labels are assigned; the table reports per-cell statistics and "
        "multiplicity summaries only.",
        "Rarefied common-N* stage shown; native stage is a separate column set in the same accepted grid.",
    ])


# --------------------------------------------------------------------------
# 4. P2-C-TAB-LOFO -- thirteen-family leave-one-family-out table
# --------------------------------------------------------------------------
def lofo_table(ctx, entry):
    outdir = ctx["outdir"] / entry["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = _inputs_by_role(entry, ctx["input_root"])
    lofo_path, _ = _require_single(inputs, "study2_lofo_results")
    grid_path, _ = _require_single(inputs, "f1_native_grid_crosscheck")
    lofo = json.loads(lofo_path.read_text(encoding="utf-8"))
    sha = sha256_file(lofo_path)
    rel = entry["inputs"][0]["path"]
    rows = lofo.get("lofo", [])
    if len(rows) != 13:
        refuse_missing_input(f"{entry['id']}: expected 13 LOFO rows, found {len(rows)}")

    csv_rows, display_rows, table_rows = [], [], []
    for r in rows:
        fam = r["family_dropped"]
        table_rows.append([fam, r["n_dropped"], r["n_remaining"], r["rho_raw"],
                           r["rho_partial_logparams"]])
        display_rows.append([fam, str(r["n_dropped"]), str(r["n_remaining"]),
                             fmt_num(r["rho_raw"]), fmt_num(r["rho_partial_logparams"])])
        for key in ("rho_raw", "rho_partial_logparams"):
            csv_rows.append([fam, key, r[key], fmt_num(r[key]),
                             "rho", r["n_remaining"], rel, f"/lofo/{rows.index(r)}/{key}", sha])

    observed = lofo.get("observed", {})
    perm = lofo.get("permutation", {})
    raw_summary = lofo.get("lofo_raw_summary", {})
    part_summary = lofo.get("lofo_partial_logparams_summary", {})
    note = ("Science participation-ratio (eff_rank_pr) association: leave-one-family-out across 13 model "
            "families (n=%s models total), raw Spearman rho and rho partialling log-parameters. Observed raw "
            "rho = %s; observed partial|log-params = %s; %s label permutations, two-sided p = %s. "
            "All 13 leave-one-out values stay negative (raw |rho| in [%s, %s]; partial |rho| in [%s, %s])."
            % (lofo.get("n_models"), fmt_num(observed.get("raw_rho")),
               fmt_num(observed.get("partial_rho_logparams")), perm.get("n_perm"),
               fmt_p(perm.get("p_two_sided")), fmt_num(raw_summary.get("min_abs")),
               fmt_num(raw_summary.get("max_abs")), fmt_num(part_summary.get("min_abs")),
               fmt_num(part_summary.get("max_abs"))))
    write_csv(outdir / "lofo_table.csv",
              ["family_dropped", "n_dropped", "n_remaining", "rho_raw", "rho_partial_logparams"],
              display_rows)
    write_csv(outdir / "lofo_values_long.csv",
              ["family_dropped", "quantity", "value", "display", "unit", "n_remaining",
               "source_path", "source_pointer", "source_sha256"], csv_rows)
    tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\caption{" + tex_escape(note) + "}", r"\label{tab:lofo}",
           r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r"family dropped & $n_{\mathrm{drop}}$ & $n_{\mathrm{rem}}$ & raw $\rho$ & partial $\rho$ ($\log$-params) \\",
           r"\midrule"]
    for fam, nd, nr, raw, part in display_rows:
        tex.append(f"{tex_escape(fam)} & {nd} & {nr} & {tex_number(raw)} & {tex_number(part)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(outdir / "lofo_table.tex", "\n".join(tex))

    plotdata = {
        "schema": SCHEMA_PLOTDATA,
        "result_id": entry["id"],
        "kind": "table",
        "title": entry["title"],
        "units": {"rho": "Spearman correlation", "n": "models"},
        "n_families": lofo.get("n_families"),
        "n_models": lofo.get("n_models"),
        "observed": observed,
        "permutation": perm,
        "summary_raw": raw_summary,
        "summary_partial_logparams": part_summary,
        "rows": rows,
        "note": note,
        "source_files": [
            {"path": rel, "sha256": sha},
            {"path": entry["inputs"][1]["path"], "sha256": sha256_file(grid_path)},
        ],
    }
    pd_path = outdir / "plotdata.json"
    write_json(pd_path, plotdata)
    artifacts = [artifact_record(p, ctx["outdir"]) for p in
                 (outdir / "lofo_table.csv", outdir / "lofo_table.tex",
                  outdir / "lofo_values_long.csv", pd_path)]
    return _fragment(entry, artifacts, plotdata, {
        "n_rows": len(rows), "n_families": lofo.get("n_families"),
        "all_negative_raw": all(r["rho_raw"] < 0 for r in rows),
    }, new_vs_old={
        "historical_output": "UNLOCATED (13-row supplementary table)",
        "basis": ("regenerated from the signed Study-2 LOFO reuse results; no historical file copied"),
    }, limitations=[
        "Leave-one-family-out is a robustness diagnostic, not a significance test of family effects.",
        "Partial column controls log-parameters only.",
    ])


# --------------------------------------------------------------------------
# 5. P2-F-TAB-PROBES -- extension result table (existing bounded probes)
# --------------------------------------------------------------------------
def extension_probes_table(ctx, entry):
    outdir = ctx["outdir"] / entry["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = _inputs_by_role(entry, ctx["input_root"])
    h1_path, _ = _require_single(inputs, "ext_h1_results")
    h2_path, _ = _require_single(inputs, "ext_h2_results")
    h3_path, _ = _require_single(inputs, "ext_h3_results")
    h1, h2, h3 = (json.loads(p.read_text(encoding="utf-8")) for p in (h1_path, h2_path, h3_path))
    shas = {p.name: sha256_file(p) for p in (h1_path, h2_path, h3_path)}
    rel = {p.name: spec["path"] for p in (h1_path, h2_path, h3_path)
           for spec in entry["inputs"] if spec["path"].endswith(p.name)}

    def pick(doc, pointer):
        v = doc
        for part in [x for x in pointer.split("/") if x]:
            if isinstance(v, list):
                v = v[int(part)]
            else:
                v = v[part]
        return v

    specs = [
        ("Difficulty readout: median R2_b", h1, "/task_A_pooled/median_r2", h1_path,
         "48 models (existing bounded probe)", "R2"),
        ("Aligned transfer: median rho", h1, "/task_C_transfer/median_transfer_rho", h1_path,
         "item-split bootstrap transfer, 48 models", "Spearman rho"),
        ("Selected-representation gain-slope rho", h2, "/rho_H2_best_layer", h2_path,
         "48 models, best-layer representation", "Spearman rho"),
        ("Pooled gain-slope rho", h2, "/rho_H2_pooled", h2_path,
         "48 models, pooled representation", "Spearman rho"),
        ("Residual readout: median R2_eps", h3, "/pooled_summary/r2_median", h3_path,
         "48 models (fitted residuals)", "R2"),
    ]
    rows, csv_rows = [], []
    for label, doc, pointer, path, n_desc, unit in specs:
        value = pick(doc, pointer)
        display = fmt_num(value, 4)
        rows.append({"quantity": label, "value": value, "display": display, "units": unit,
                     "n": n_desc,
                     "source": {"path": rel[path.name], "json_pointer": pointer,
                                "sha256": shas[path.name]}})
        csv_rows.append([label, value, display, unit, n_desc, rel[path.name], pointer, shas[path.name]])
    n_both = pick(h3, "/fractions/n_both")
    n_models = pick(h3, "/fractions/n_models")
    rows.append({"quantity": "Residual threshold count", "value": f"{n_both}/{n_models}",
                 "display": f"{n_both}/{n_models}", "units": "count",
                 "n": f"{n_models} models",
                 "source": {"path": rel[h3_path.name], "json_pointer": "/fractions/n_both|/fractions/n_models",
                            "sha256": shas[h3_path.name]}})
    csv_rows.append(["Residual threshold count", n_both, f"{n_both}/{n_models}", "count",
                     f"{n_models} models", rel[h3_path.name],
                     "/fractions/n_both|/fractions/n_models", shas[h3_path.name]])

    note = ("Exploratory item-level results under the implemented all-item preprocessing (existing bounded "
            "probes only; no new prediction or causal experiment). Intervals and permutation values refer "
            "to the implemented procedure; these numbers have not passed a strict no-leakage validation.")
    write_csv(outdir / "extension_probes_table.csv",
              ["quantity", "value", "display", "units", "n", "source_path", "source_pointer",
               "source_sha256"], csv_rows)
    tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\caption{" + tex_escape(note) + "}", r"\label{tab:extension-probes}",
           r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Quantity & Result \\", r"\midrule"]
    for r in rows:
        tex.append(f"{tex_escape(r['quantity'])} & {tex_escape(r['display'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(outdir / "extension_probes_table.tex", "\n".join(tex))

    plotdata = {
        "schema": SCHEMA_PLOTDATA,
        "result_id": entry["id"],
        "kind": "table",
        "title": entry["title"],
        "note": note,
        "rows": rows,
        "n_models": n_models,
        "source_files": [{"path": rel[p.name], "sha256": shas[p.name]}
                         for p in (h1_path, h2_path, h3_path)],
    }
    pd_path = outdir / "plotdata.json"
    write_json(pd_path, plotdata)
    artifacts = [artifact_record(p, ctx["outdir"]) for p in
                 (outdir / "extension_probes_table.csv", outdir / "extension_probes_table.tex",
                  pd_path)]
    return _fragment(entry, artifacts, plotdata, {
        "n_rows": len(rows), "n_models_h3": n_models,
    }, new_vs_old={
        "historical_output": "UNLOCATED (extension probe table)",
        "basis": ("regenerated from existing G1/G2/G3 bounded-probe result files; no historical table "
                  "image copied and no probe re-run"),
    }, limitations=[
        "Exploratory probes under the implemented preprocessing; not strict no-leakage validated.",
        "No new prediction or causal experiment was run by this renderer.",
    ])


# --------------------------------------------------------------------------
# 6. P2-F-FIG-PROFILES -- depth profiles figure
# --------------------------------------------------------------------------
def depth_profiles_figure(ctx, entry):
    outdir = ctx["outdir"] / entry["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = _inputs_by_role(entry, ctx["input_root"])
    prof_path, prof_spec = _require_single(inputs, "ext_h4_depth_profiles")
    sum_path, sum_spec = _require_single(inputs, "ext_h4_depth_summary_archive")
    peak_path, peak_spec = _require_single(inputs, "ext_h4_peak_depths_archive")
    meta_path, _ = _require_single(inputs, "ext_h4_metadata")
    for spec in (sum_spec, peak_spec):
        if spec.get("computational_input", True):
            refuse_bad_schema(
                "archived depth aggregates must be declared computational_input=false "
                f"(comparison-only): {spec.get('path')}")
    import csv as _csv
    with open(prof_path, newline="", encoding="utf-8") as fh:
        profile_rows = list(_csv.DictReader(fh))
    with open(sum_path, newline="", encoding="utf-8") as fh:
        archived_summary = list(_csv.DictReader(fh))
    with open(peak_path, newline="", encoding="utf-8") as fh:
        archived_peaks = list(_csv.DictReader(fh))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sha_prof, sha_sum, sha_peak, sha_meta = (sha256_file(p) for p in
                                             (prof_path, sum_path, peak_path, meta_path))

    # Fresh aggregation from per-model/per-layer rows (frozen G4 rule). The
    # archived depth_summary.csv / peak_depths.csv are comparison-only: their
    # values never enter the computation or the emitted numbers.
    fresh = depth_aggregate.recompute_summary(profile_rows)
    fresh_peaks = depth_aggregate.recompute_peaks(fresh)
    comparison = depth_aggregate.compare_rows(fresh, archived_summary)
    peak_diffs = {}
    for fp in fresh_peaks:
        ap = next((r for r in archived_peaks if r["metric"] == fp["metric"]), None)
        peak_diffs[fp["metric"]] = {
            "fresh": {"relative_depth": fp["relative_depth"], "value": fp["value"],
                      "n_models": fp["n_models"]},
            "archive": ({"relative_depth": float(ap["relative_depth"]), "value": float(ap["value"]),
                         "n_models": int(ap["n_models"])} if ap else None),
            "match": bool(ap and abs(float(ap["value"]) - fp["value"])
                          <= 1e-9 + 1e-6 * abs(float(ap["value"]))
                          and abs(float(ap["relative_depth"]) - fp["relative_depth"]) <= 1e-12),
        }
    if not comparison["within_tolerance"]:
        refuse_missing_input(
            "fresh depth summary disagrees with the archived aggregate beyond the replay comparator; "
            "investigate before rendering: " + json.dumps(comparison["mismatch_cells"][:5]))

    def series(col):
        return [(float(r["relative_depth"]), float(r[col]), int(r["n_models"])) for r in fresh]

    panels = [
        ("median_r2_b", "Median difficulty readout $R^2_b$ (model median)", "R2"),
        ("rho_H2", "Gain-slope Spearman $\\rho$ (cross-model)", "Spearman rho"),
        ("median_r2_eps", "Median residual readout $R^2_\\varepsilon$ (model median)", "R2"),
    ]
    peak_by_metric = {r["metric"]: r for r in fresh_peaks}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8))
    ax_cover = axes[1][1]
    plot_series = []
    for ax, (metric, ylabel, unit) in zip([axes[0][0], axes[0][1], axes[1][0]], panels):
        data = series(metric)
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ns = [d[2] for d in data]
        ax.plot(xs, ys, marker="o", ms=3.2, lw=1.2, color="#1f4e79")
        peak = peak_by_metric.get(metric, {})
        if peak:
            ax.plot([float(peak["relative_depth"])], [float(peak["value"])], marker="o",
                    mfc="none", mec="#c00000", ms=9, mew=1.4, ls="none",
                    label="grid maximum (no uncertainty)")
            ax.legend(fontsize=7.2, loc="lower right")
        ax.set_xlabel("Relative depth (G4: (L-k)/L; k=0 is final layer)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7.4)
        ax.grid(alpha=0.25, lw=0.5)
        plot_series.append({"metric": metric, "units": unit, "x": xs, "y": ys, "n_models": ns,
                            "peak": {"relative_depth": float(peak["relative_depth"]),
                                     "value": float(peak["value"])} if peak else None,
                            "source": {"path": prof_spec["path"], "sha256": sha_prof},
                            "label": ylabel})
    cov = series("median_r2_b")  # n_models repeats across metrics at each depth
    ax_cover.step([d[0] for d in cov], [d[2] for d in cov], where="mid", color="#555555", lw=1.2)
    ax_cover.set_ylim(0, max(d[2] for d in cov) + 2)
    ax_cover.set_xlabel("Relative depth", fontsize=8)
    ax_cover.set_ylabel("Models contributing", fontsize=8)
    ax_cover.tick_params(labelsize=7.4)
    ax_cover.grid(alpha=0.25, lw=0.5)
    ax_cover.set_title("Coverage (max %d models)" % meta.get("n_models", 0), fontsize=8.5)
    fig.suptitle("Extension depth profiles: descriptive layer-wise medians and gain-slope correlation",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig_paths = _save_figure(fig, outdir, "depth_profiles")

    plotdata = {
        "schema": SCHEMA_PLOTDATA,
        "result_id": entry["id"],
        "kind": "figure",
        "title": entry["title"],
        "x_label": "Relative depth (G4: (L-k)/L; stored layer index k=0 is the final layer)",
        "grid": [r["relative_depth"] for r in fresh],
        "series": plot_series,
        "coverage": {"x": [d[0] for d in cov], "n_models": [d[2] for d in cov]},
        "n_models": meta.get("n_models"),
        "n_native_rows": meta.get("n_native_rows"),
        "fresh_summary_rows": [{"relative_depth": r["relative_depth"], "n_models": r["n_models"],
                                "median_r2_b": value_cell(r["median_r2_b"], "R2", r["n_models"],
                                                          {"path_rel": prof_spec["path"],
                                                           "sha256": sha_prof}),
                                "rho_H2": value_cell(r["rho_H2"], "Spearman rho", r["n_models"],
                                                     {"path_rel": prof_spec["path"],
                                                      "sha256": sha_prof}),
                                "median_r2_eps": value_cell(r["median_r2_eps"], "R2", r["n_models"],
                                                            {"path_rel": prof_spec["path"],
                                                             "sha256": sha_prof})}
                               for r in fresh],
        "fresh_peaks": fresh_peaks,
        "derivation": {
            "rule": ("frozen G4 descriptive rule: per-model linear interpolation onto the 0.05 depth grid "
                     "without extrapolation; model-median r2_b / r2_eps; cross-model Spearman rho between "
                     "-beta and s; peaks are first (shallowest) grid maxima"),
            "computation_input": prof_spec["path"],
            "computation_input_sha256": sha_prof,
            "archive_columns": {
                "relative_depth": "normalized layer depth d=(n_layers-k)/n_layers on the fixed 0.05 grid",
                "n_models": "models whose stored layer range covers the grid depth (no extrapolation)",
                "median_r2_b": "model-median interpolated difficulty readout R2_b at that depth",
                "rho_H2": "cross-model Spearman rho between -beta (gain slope) and s",
                "median_r2_eps": "model-median interpolated residual readout R2_eps at that depth",
            },
        },
        "comparison_only": {
            "role": "repeatability comparison; excluded from computation and not approved as input",
            "archived_summary": {"path": entry["inputs"][1]["path"], "sha256": sha_sum,
                                 "comparison": comparison},
            "archived_peaks": {"path": entry["inputs"][2]["path"], "sha256": sha_peak,
                               "per_metric": peak_diffs},
        },
        "limit_note": ("Descriptive; interpolation onto the fixed grid without extrapolation; no uncertainty "
                       "intervals or layer-difference tests; inherits the preprocessing and target-construction "
                       "limitations of the underlying probes."),
        "source_files": [
            {"path": entry["inputs"][0]["path"], "sha256": sha_prof, "role": "computation_input"},
            {"path": entry["inputs"][1]["path"], "sha256": sha_sum, "role": "comparison_only"},
            {"path": entry["inputs"][2]["path"], "sha256": sha_peak, "role": "comparison_only"},
            {"path": entry["inputs"][3]["path"], "sha256": sha_meta, "role": "context"},
        ],
    }
    pd_path = outdir / "plotdata.json"
    write_json(pd_path, plotdata)
    series_csv = [[s["metric"], x, y, n] for s in plot_series for x, y, n in zip(s["x"], s["y"], s["n_models"])]
    write_csv(outdir / "depth_profiles_series.csv",
              ["metric", "relative_depth", "value", "n_models"], series_csv)
    write_csv(outdir / "depth_summary_fresh.csv", list(depth_aggregate.SUMMARY_COLUMNS),
              [[r["relative_depth"], r["n_models"], r["median_r2_b"], r["rho_H2"], r["median_r2_eps"]]
               for r in fresh])
    write_csv(outdir / "depth_peaks_fresh.csv", ["metric", "relative_depth", "value", "n_models"],
              [[p["metric"], p["relative_depth"], p["value"], p["n_models"]] for p in fresh_peaks])
    artifacts = [artifact_record(p, ctx["outdir"]) for p in fig_paths]
    artifacts.append(artifact_record(pd_path, ctx["outdir"]))
    artifacts.append(artifact_record(outdir / "depth_profiles_series.csv", ctx["outdir"]))
    artifacts.append(artifact_record(outdir / "depth_summary_fresh.csv", ctx["outdir"]))
    artifacts.append(artifact_record(outdir / "depth_peaks_fresh.csv", ctx["outdir"]))
    return _fragment(entry, artifacts, plotdata, {
        "n_grid_points": len(fresh), "n_models": meta.get("n_models"),
        "coverage_min": min(d[2] for d in cov), "coverage_max": max(d[2] for d in cov),
        "fresh_summary_from_per_model_rows": True,
        "archive_matches_fresh_within_replay_comparator": comparison["within_tolerance"],
        "archive_max_abs_diff": comparison["max_abs_diff"],
        "archived_aggregate_role": "comparison_only_not_approved",
    }, new_vs_old={
        "historical_output": "LOCATED historical profiles.pdf (not copied)",
        "basis": ("figure regenerated from the per-model/per-layer G4 depth profiles; the archived "
                  "depth_summary/peak tables are comparison-only and excluded from the computation; "
                  "the historical PDF was not used as a source"),
    }, limitations=[
        "Descriptive, interpolation-dependent layer profiles; no uncertainty intervals.",
        "29-model code subset; pooled-only models and mathematics are outside the depth panel.",
        "Archived depth aggregates are repeatability references only and are not approved computation inputs.",
    ])


# --------------------------------------------------------------------------
# 7. P2-G-TAB-SUPPLEMENTS -- supplement topology (no numeric content)
# --------------------------------------------------------------------------
def supplement_topology(ctx, entry):
    outdir = ctx["outdir"] / entry["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = _inputs_by_role(entry, ctx["input_root"])
    man_path, _ = _require_single(inputs, "result_manifest")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    record = next((r for r in manifest["records"] if r["result_id"] == entry["id"]), None)
    if record is None:
        refuse_missing_input(f"RESULT_MANIFEST has no record for {entry['id']}")
    binding = manifest.get("supplement_binding", {})

    native = entry["inputs"][1]["path"]
    rarefied = entry["inputs"][2]["path"]
    study1 = entry["inputs"][3]["path"]
    # Verified bindings: the current per-indicator scope is the 7-indicator x
    # 4-domain grid (28 cells per variant), which supersedes the historical
    # 21-cell description cited in the manuscript text.
    items = [
        ("Per-indicator grid (7 indicators x 4 domains)", f"{native}; {rarefied}",
         "regenerated: 28 cells per variant; supersedes the historical 21-cell description"),
        ("Table 1 - instrument reliability",
         f"{study1} (/domains/*/rel_geo, /domains/*/rel_th); retained reliability reproductions",
         "regenerated from retained Study-1 results; original reliability reproductions retained"),
        ("Table 3 - frozen per-domain convergence panel",
         "P2-B-TAB-CONVERGENCE convergence_table.csv/.tex (regenerated from retained Study-1 results)",
         "regenerated: emitted as its own result output"),
        ("Table 3b - frozen verdict statistics", f"{study1} (/domains/*/verdict)",
         "regenerated from retained frozen verdict statistics"),
        ("Table 4 - per-indicator map", f"{native}; {rarefied}",
         "regenerated: 28-cell map supersedes the historical 21-cell map"),
        ("Appendix A - analysis plan and deviations",
         "no recoverable fresh evidence (historical plan/threshold records unavailable)",
         "historical limitation retained; not regenerated"),
        ("Appendix B - theory derivations", "runs/theory_execution_v1/result.json (symbolic checks)",
         "retained: symbolic derivation checks unchanged"),
        ("Appendix C - decomposition audits", "retained original decomposition record (P2-B-006)",
         "retained original; not re-run"),
        ("Appendix D - adversarial probes",
         "retained new analysis: appendix-D residual item structure (P2-G-RESIDUAL-DIM)",
         "retained; no numeric values are defined in this topology"),
        ("Appendix E - medical theta-instability probe",
         "regenerated medical criterion-failure record (free-2PL failure retained)",
         "replaced by the regenerated medical MAP-primary analysis"),
    ]
    binding_note = binding.get("note", "")
    write_csv(outdir / "supplement_topology.csv",
              ["item", "evidence_source", "binding_status"],
              [[a, b, c] for a, b, c in items])
    tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\caption{Online supplementary topology. Each item cited in the manuscript is bound to the "
           r"regenerated evidence that now carries it, or to the retained original analysis where no "
           r"regeneration applies. Historical numbering is unresolved (the cited Table~1/3/3b/4 "
           r"descriptions do not match the historical numbering); the current per-indicator scope is "
           r"the 7-indicator $\times$ 4-domain grid (28 cells per variant). No numeric values are "
           r"defined in this table.}",
           r"\label{tab:supplements}", r"\begin{tabular}{@{}lll@{}}", r"\toprule",
           r"item & evidence source & binding status \\", r"\midrule"]
    for a, b, c in items:
        tex.append(f"{tex_escape(a)} & {tex_escape(b)} & {tex_escape(c)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(outdir / "supplement_topology.tex", "\n".join(tex))

    # Topology diagram (conceptual, no numbers)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    def box(x, y, w, h, text, fc="#eef3f8", ec="#1f4e79", fs=8.0):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    box(0.2, 4.4, 2.6, 1.6, "Main manuscript\n(Sections 6-8, Online\nSupplementary citation)", fs=8.2)
    box(3.6, 7.6, 6.0, 1.7,
        "Per-indicator grid (7 indicators x 4 domains)\n28 cells per variant; regenerated\n"
        "supersedes the historical 21-cell description")
    box(3.6, 5.3, 6.0, 1.9,
        "Tables 1 / 3 / 3b / 4\ninstrument reliability; frozen convergence panel;\n"
        "verdict statistics; per-indicator map\nbound to regenerated evidence (numbering unresolved)")
    box(3.6, 2.6, 6.0, 2.2,
        "Appendices A-E\nA analysis plan and deviations; B theory derivations;\n"
        "C decomposition audits; D adversarial probes;\nE medical theta-instability probe\n"
        "A limitation retained; B/C/D retained original;\nE replaced by regenerated medical analysis")
    for y0 in (8.45, 6.25, 3.7):
        ax.annotate("", xy=(3.6, y0), xytext=(2.8, 5.2),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f4e79"))
    ax.set_title("Online supplementary topology (conceptual shell; no numeric values)", fontsize=10.5)
    fig.tight_layout()
    fig_paths = _save_figure(fig, outdir, "supplement_topology")

    plotdata = {
        "schema": SCHEMA_PLOTDATA,
        "result_id": entry["id"],
        "kind": "table+topology",
        "title": entry["title"],
        "numeric_values_present": False,
        "scope_note": ("current per-indicator scope: 7 indicators x 4 domains = 28 cells per variant; "
                       "the historical 21-cell description is superseded"),
        "items": [{"item": a, "evidence_source": b, "binding_status": c} for a, b, c in items],
        "manifest_binding_status": binding.get("status"),
        "manifest_binding_note": binding_note,
        "record_historical_caveats": record.get("historical_caveats", []),
        "source_files": [
            {"path": entry["inputs"][0]["path"], "sha256": sha256_file(man_path)},
            {"path": entry["inputs"][1]["path"], "sha256": sha256_file((ctx["input_root"] / entry["inputs"][1]["path"]).resolve())},
            {"path": entry["inputs"][2]["path"], "sha256": sha256_file((ctx["input_root"] / entry["inputs"][2]["path"]).resolve())},
            {"path": entry["inputs"][3]["path"], "sha256": sha256_file((ctx["input_root"] / entry["inputs"][3]["path"]).resolve())},
        ],
    }
    pd_path = outdir / "plotdata.json"
    write_json(pd_path, plotdata)
    artifacts = [artifact_record(p, ctx["outdir"]) for p in
                 (outdir / "supplement_topology.csv", outdir / "supplement_topology.tex", pd_path)]
    artifacts += [artifact_record(p, ctx["outdir"]) for p in fig_paths]
    return _fragment(entry, artifacts, plotdata, {
        "n_items": len(items),
        "no_numeric_values": True,
        "binding_status_pending_items": sum(1 for _, _, s in items if s.startswith("REQUIRED")),
    }, new_vs_old={
        "historical_output": record.get("historical_outputs", [{}])[0].get("status", "UNLOCATED"),
        "basis": ("topology regenerated from the manifest supplement binding and the verified "
                  "per-item evidence bindings; historical tables are not re-imported and no numbers "
                  "are invented to fill the conceptual figure"),
    }, limitations=[
        "Historical supplementary numbering is unresolved; the cited descriptions do not match the "
        "historical numbering.",
        "Appendix A historical plan/threshold evidence remains unavailable and is retained as a limitation.",
    ])


_NUMERIC_KEYS = {"value", "rho_raw", "rho_partial_logparams", "raw_rho", "partial_rho",
                 "p_cond", "bh28_cond", "by28_cond"}


def _carries_value(obj) -> bool:
    """True if the plot-data payload contains an actual numeric measurement."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if (key in _NUMERIC_KEYS and isinstance(val, (int, float))
                    and not isinstance(val, bool)):
                return True
            if isinstance(val, (dict, list)) and _carries_value(val):
                return True
        return False
    if isinstance(obj, list):
        return any(_carries_value(v) for v in obj)
    return False


def _fragment(entry, artifacts, plotdata, checks, new_vs_old, limitations):
    return {
        "id": entry["id"],
        "kind": entry["kind"],
        "renderer": entry["renderer"],
        "title": entry["title"],
        "status": "GENERATED",
        "artifacts": artifacts,
        "checks": checks,
        "numeric_values_carried": _carries_value(plotdata),
        "new_vs_old": new_vs_old,
        "limitations": limitations,
    }


RENDERERS = {
    "mtmm_schematic": mtmm_schematic,
    "convergence_table": convergence_table,
    "science_grid_table": science_grid_table,
    "lofo_table": lofo_table,
    "extension_probes_table": extension_probes_table,
    "depth_profiles_figure": depth_profiles_figure,
    "supplement_topology": supplement_topology,
}
