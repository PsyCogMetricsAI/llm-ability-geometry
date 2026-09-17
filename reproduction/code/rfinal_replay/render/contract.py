"""Declared prepare-stage render contract (C10a).

The same schema is used by C11 with fresh result paths swapped in; this module
only binds the *prepare-stage accepted summaries* that root authorized for
C10a (accepted C17 reports plus signed baseline Study1/Study2/LOFO and the
G-extension artifacts). Renderers never hardcode final values: every number is
read from the files declared here, at render time.
"""
from __future__ import annotations

from . import SCHEMA_CONTRACT

MANIFEST = "inputs_manifest/RESULT_MANIFEST.json"
MAIN_TEX = "../Imports/geometry/Paper/hicss_20260610/hicss_submission.tex"
STUDY1_MAIN = "runs/study1_reuse_v1/main/results.json"
NEW28_RAREFIED = "runs/c17_new28_v1/grids/F1-rarefied-28.json"
NEW28_NATIVE = "runs/c17_new28_v1/grids/F1-native-28.json"
LOFO = "runs/study2_reuse_v1/lofo/results.json"
EXT_H1 = "../ext_P2_geo_20260914/h1/results_h1.json"
EXT_H2 = "../ext_P2_geo_20260914/h2/results_h2.json"
EXT_H3 = "../ext_P2_geo_20260914/h3/results_h3.json"
EXT_H4_PROFILES = "../ext_P2_geo_20260914/h4/depth_profiles.csv"
EXT_H4_SUMMARY = "../ext_P2_geo_20260914/h4/depth_summary.csv"
EXT_H4_PEAKS = "../ext_P2_geo_20260914/h4/peak_depths.csv"
EXT_H4_META = "../ext_P2_geo_20260914/h4/G4_METADATA.json"

SCIENCE_ROWS = (
    ("eff_rank_pr", "science"),
    ("rankme", "science"),
    ("stable_rank", "science"),
    ("spectral_alpha", "science"),
    ("vn_entropy", "science"),
    ("isoscore", "science"),
    ("twoNN_id", "math"),
    ("twoNN_id", "code"),
)


def default_contract(input_root: str) -> dict:
    """Prepare-stage contract. Paths are relative to --input-root (R)."""
    return {
        "schema": SCHEMA_CONTRACT,
        "stage": "C10a-prepare",
        "input_root": input_root,
        "stage_note": (
            "Prepare-stage renderers consume accepted C17 summaries and signed baseline "
            "Study1/Study2/LOFO and G-extension artifacts. C11 must rewire the same "
            "contract schema to freshly generated results; no renderer hardcodes final values."
        ),
        "outputs": [
            {
                "id": "P2-G-FIG-MTMM",
                "kind": "figure",
                "renderer": "mtmm_schematic",
                "title": "Two-instrument MTMM schematic (design role of cells)",
                "inputs": [
                    {"role": "result_manifest", "path": MANIFEST},
                    {"role": "definition_reference", "path": MAIN_TEX, "external": True},
                ],
            },
            {
                "id": "P2-B-TAB-CONVERGENCE",
                "kind": "table",
                "renderer": "convergence_table",
                "title": "Frozen per-domain convergence panel",
                "inputs": [
                    {"role": "study1_main_results", "path": STUDY1_MAIN,
                     "pointer": "/domains/code"},
                    {"role": "study1_main_results", "path": STUDY1_MAIN,
                     "pointer": "/domains/math"},
                ],
            },
            {
                "id": "P2-C-TAB-SCIENCE",
                "kind": "table",
                "renderer": "science_grid_table",
                "title": "Compact science and TwoNN grid table (rarefied common-N*)",
                "inputs": [
                    {"role": "f1_rarefied_grid", "path": NEW28_RAREFIED},
                ],
            },
            {
                "id": "P2-C-TAB-LOFO",
                "kind": "table",
                "renderer": "lofo_table",
                "title": "Thirteen-family leave-one-family-out table (science)",
                "inputs": [
                    {"role": "study2_lofo_results", "path": LOFO},
                    {"role": "f1_native_grid_crosscheck", "path": NEW28_NATIVE},
                ],
            },
            {
                "id": "P2-F-TAB-PROBES",
                "kind": "table",
                "renderer": "extension_probes_table",
                "title": "Extension result table (existing bounded probes)",
                "inputs": [
                    {"role": "ext_h1_results", "path": EXT_H1, "external": True},
                    {"role": "ext_h2_results", "path": EXT_H2, "external": True},
                    {"role": "ext_h3_results", "path": EXT_H3, "external": True},
                ],
            },
            {
                "id": "P2-F-FIG-PROFILES",
                "kind": "figure",
                "renderer": "depth_profiles_figure",
                "title": "Extension depth profiles (existing bounded depth analysis)",
                "inputs": [
                    {"role": "ext_h4_depth_profiles", "path": EXT_H4_PROFILES, "external": True},
                    {"role": "ext_h4_depth_summary_archive", "path": EXT_H4_SUMMARY,
                     "external": True, "comparison_only": True, "computational_input": False},
                    {"role": "ext_h4_peak_depths_archive", "path": EXT_H4_PEAKS,
                     "external": True, "comparison_only": True, "computational_input": False},
                    {"role": "ext_h4_metadata", "path": EXT_H4_META, "external": True},
                ],
            },
            {
                "id": "P2-G-TAB-SUPPLEMENTS",
                "kind": "table",
                "renderer": "supplement_topology",
                "title": "Referenced historical tables and Appendix A-E topology",
                "inputs": [
                    {"role": "result_manifest", "path": MANIFEST},
                    {"role": "f1_native_grid_candidate", "path": NEW28_NATIVE},
                    {"role": "f1_rarefied_grid_candidate", "path": NEW28_RAREFIED},
                    {"role": "study1_main_candidate", "path": STUDY1_MAIN},
                ],
            },
        ],
    }


def final_contract(input_root: str) -> dict:
    """C10 final integration contract: same schema, frozen stage, pinned hashes.

    Identical bindings to :func:`default_contract` plus an explicit
    ``stage``/``stage_note`` for the final replay and sha256 pins for every
    declared input (filled by :func:`pin_contract`). The archived depth
    aggregates are marked comparison-only and are excluded from computation.
    """
    contract = default_contract(input_root)
    contract["stage"] = "C10-render-final"
    contract["stage_note"] = (
        "Final integration contract: the seven result outputs are regenerated from the declared "
        "input files under sha256 pin enforcement. The archived depth summary/peaks are "
        "comparison-only references excluded from computation; the depth figures/summaries are "
        "computed afresh from per-model/per-layer rows following the frozen G4 rule."
    )
    return pin_contract(contract, input_root)


def pin_contract(contract: dict, input_root: str) -> dict:
    """Fill/refresh sha256 pins for every declared input path."""
    from pathlib import Path

    from .common import sha256_file

    root = Path(input_root).resolve()
    for entry in contract["outputs"]:
        for spec in entry["inputs"]:
            spec["sha256"] = sha256_file((root / spec["path"]).resolve())
    return contract
