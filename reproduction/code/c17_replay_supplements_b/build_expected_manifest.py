#!/usr/bin/env python3
"""Build the pinned expected-input manifest for the C10v replay (one-off, read-only).

Historical pins are taken from frozen artifacts (fit_meta.json, h1 provenance,
c17_cache_observations/expected_inputs_v1.json, h3/FINALIZE_SHA256.md,
V_RECOVERED16_CONTROLLER.json, VERIFY-G / ledger hash columns).  Files with no
historical hash are pinned at build time from their current bytes and marked
with pin_source = 'pinned_at_C10v_build'.
"""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
P = R.parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = []


def add(root, rel, sha256, role, level, pin_source):
    base = R if root == "R" else P
    entry = {"root": root, "relativepath": rel,
             "sha256": sha256 if sha256 else sha(base / rel),
             "role": role, "level": level}
    if sha256 is None:
        entry["pin_source"] = pin_source
    files.append(entry)


HIST = "historical"
BUILD = "pinned_at_C10v_build"

# --- 48 pooling/G0 per-model JSONL (hash pinned by the frozen reuse binding)
bind = json.loads((R / "config/REUSE_BINDINGS_EXTENSION.json").read_text())
for a in bind["asset_bindings"]:
    rel = str(Path(a["path"]).relative_to(P))
    add("P", rel, a["sha256"], "pooling_and_g0_jsonl:%s" % a["model"],
        "computation_input", HIST)
add("R", "config/REUSE_BINDINGS_EXTENSION.json",
    "517835cc8aa0923a02419d527a10526453c0785095bc13228f0627a0fb90451d",
    "pooling_binding_config", "computation_input", HIST)

# --- G0 panel saved inputs / terminals
g0 = {
    "ext_P2_1_itemmodel_20260913/inventory.csv":
        ("7ab3dbfc32d2021d5a53ab889b2d7a139913a5045870c61c03782a31ca52bfdb", "g0_inventory"),
    "ext_P2_geo_20260914/fit_panel/fit_meta.json":
        ("cd429feb0167a18f970728e904f9ab51575a7ca86fbac4fd5a1af2c98885b203", "g0_fit_meta_terminal"),
    "ext_P2_geo_20260914/fit_panel/theta_s_sigma_panel.csv":
        ("69a059b14c18628bc19700d02fad11280bc1766e88d613368cd82cd79e4160a7", "g0_saved_params_csv"),
    "ext_P2_geo_20260914/fit_panel/b_panel.csv":
        ("52c351c659a93fbfb68fabfa05fbfba8940d909bf0c59a2b360ba9045c745148", "g0_saved_b_csv"),
    "ext_P2_geo_20260914/fit_panel/eps_panel.npz":
        ("36e9b7a88bbcdca85979a4456837a06ac8e6d87a66ac8d266f101f1c0a6f8b94", "g0_saved_eps_observations"),
    "ext_P2_geo_20260914/fit_panel/b_loo.npz":
        ("a6072ab7e6a640f11508b53187844d17b230c6454045a51752b997ba728c4fc2", "g0_saved_loo_observations"),
    "ext_P2_geo_20260914/fit_panel/G0_REPORT.md":
        ("fa1cffd3cb57eb94afcd40067d0545b44f9c94afa50449e6ef03dc5552b63acb", "g0_report_reference"),
}
for rel, (h, role) in g0.items():
    add("P", rel, h, role, "computation_input" if not rel.endswith(".md") else "reference",
        HIST)

# --- G1/G2/G3 saved outputs + terminals
g123 = {
    "ext_P2_geo_20260914/h1/r2_by_model_layer.csv":
        ("beca6ca628b289150a9339c0853aa05194569a47bff547e451323e04c8270185", "g1_r2_by_model_layer", "computation_input"),
    "ext_P2_geo_20260914/h1/transfer_rho.csv":
        ("12c62289267edcadae74215f3717503b2b2236adc03cff159c26f64568c01827", "g1_transfer_rho", "computation_input"),
    "ext_P2_geo_20260914/h1/results_h1.json":
        ("2b39d912673086c2d8b575a4989ed9274e3eb003546f03685dc9ce405f6beee7", "g1_terminal_result", "comparison_only"),
    "ext_P2_geo_20260914/h2/beta_by_model.csv":
        ("b4e08fb914a75333cc9273053518922f989f9dffc43f23d60717e9383d26449f", "g2_beta_by_model", "computation_input"),
    "ext_P2_geo_20260914/h2/results_h2.json":
        ("4f332f010340b14ff1567ea8c7f57641abd91d4ebd735d80ae64fd3be3d50f7e", "g2_terminal_result", "comparison_only"),
    "ext_P2_geo_20260914/h3/r2_eps.csv":
        ("a9a711a45e46f5146b63226a58d0b718b6dbeb3a3f1462508875fb610435cc07", "g3_r2_eps", "computation_input"),
    "ext_P2_geo_20260914/h3/results_h3.json":
        ("4ce5112bf22d1614ddc14173fbae59a6d857593b83bf9b71b39205e59c67b276", "g3_terminal_result", "comparison_only"),
    "ext_P2_geo_20260914/h3/perm_null.npz":
        ("81fdb10e668f2c9d63f4ad58180d5deda231cae148689a05d410fe8afc64c37f", "g3_perm_null", "computation_input"),
    "ext_P2_geo_20260914/h3/eps_loo.npz":
        ("d40236644b018cd87b8b021d8b059c63ca1a4ad728ced35d89d2c0598f540f58", "g3_eps_loo", "computation_input"),
}
for rel, (h, role, level) in g123.items():
    add("P", rel, h, role, level, HIST)

# --- recovered 16-model inputs
p16 = {
    "_gpu2_run/Import_results/activation_retrieval/v2_activation_indicators_20260602/indicators_aggregate.csv":
        ("b2bca2b5e5fabbd81daf3c73a3c1c692b381ad0f309bf0ec19186021e8b5fc3b", "prior16_indicator_aggregate"),
    "_gpu2_run/results/response_matrix_code_execution.csv":
        ("319ecf86084d440e11eb9fc20f55ba5e650eec98ea1d102adb5ba9d044e55210", "prior16_response_matrix"),
    "b200_full_pipeline/results/logprob_matrix_16models_code.csv":
        ("311c97f1c421c89376f414f53238f221b1beb3f3f6111d178cf98d4709a9fee8", "prior16_logprob_matrix"),
    "b200_full_pipeline/activation_indicators/indicators_tier2_peritem.npz":
        ("bf7e1b0e63948d1760cd3a80d90f22c3ef68bbe3e33fc42bf07e696ec6b7afe3", "prior16_peritem_npz"),
}
for rel, (h, role) in p16.items():
    add("P", rel, h, role, "computation_input", HIST)

# --- claim pointer + terminals + independent reviews
extra = {
    "P": {
        "Imports/theory_rescue_20260913/checks/results.json":
            (None, "pooling_ledger_claim_pointer", "comparison_only"),
    },
    "R": {
        "runs/extension_reuse_v1/pooling_result.json":
            (None, "pooling_producer_terminal", "comparison_only"),
        "runs/extension_reuse_v1/pooling_per_group.csv":
            (None, "pooling_per_group_terminal", "comparison_only"),
        "runs/extension_reuse_v1/pooling_run_manifest.json":
            (None, "pooling_run_manifest", "reference"),
        "verifier/V_NUM_POOLING_REUSE.json":
            (None, "pooling_independent_verifier", "comparison_only"),
        "verifier/V_NUM_G_SOURCES_SCOPE.json":
            ("d052f187bd9e856b836ebdb83be8d82daa8bca695e8ed63e9a50fa2cdb3e474c",
             "g_scope_independent_review", "reference"),
        "verifier/V_RECOVERED16_CONTROLLER.json":
            ("10b60751e60e739060a0812b1d73bb01eb5c185bbeca1519b29c759404bd3ce1",
             "prior16_controller_comparator", "comparison_only"),
        "verifier/verify_pooling_reuse.py":
            (None, "formula_reference_pooling", "reference"),
        "verifier/verify_g_sources_scope.py":
            (None, "formula_reference_g_sources_scope", "reference"),
        "verifier/verify_recovered16_controller.py":
            (None, "formula_reference_prior16", "reference"),
        "reports/C17_FINAL_EVIDENCE_LEDGER_v1.json":
            (None, "c17_final_evidence_ledger", "pointer_discovery"),
    },
}
for root, items in extra.items():
    for rel, (h, role, level) in items.items():
        add(root, rel, h, role, level, HIST if h else BUILD)

for rel, h, role in [
        ("ext_P2_geo_20260914/fit_panel/code/fit_panel_als.py", "7ed239fde727a2c5f9c97c8ea466d2f2692d31e0f1ff3500fe3675852ebb2e2e", "formula_reference_fit_panel"),
        ("ext_P2_geo_20260914/h1/code/run_h1.py", "06b7ece9f875240ba293e012d88fab9a373d3b14d1e2a89647745a9a195075fa", "formula_reference_h1_run"),
        ("ext_P2_geo_20260914/h1/code/h1_lib.py", "9021aa1dfd0e24b88c63e05d4b4411280d4b6199d2d4b466852b58098248a2b0", "formula_reference_h1_lib")]:
    add("P", rel, h, role, "reference", HIST)
for rel, role in [
        ("ext_P2_geo_20260914/h2/code/run_h2.py", "formula_reference_h2_run"),
        ("ext_P2_geo_20260914/h2/code/h2_lib.py", "formula_reference_h2_lib"),
        ("ext_P2_geo_20260914/h3/code/run_h3.py", "formula_reference_h3_run"),
        ("ext_P2_geo_20260914/h3/code/h3_lib.py", "formula_reference_h3_lib")]:
    add("P", rel, None, role, "reference", BUILD)

# --- same-estimator Task-C regeneration cache (build-time pins) and driver code pins
for rel, role in [
        ("runs/c17_supplements_b_check_v1/g1_taskc_pair_cache_v1/pair_cache_v1.npz",
         "g1_taskc_pair_cache"),
        ("runs/c17_supplements_b_check_v1/g1_taskc_pair_cache_v1/g1_taskc_cache_manifest.json",
         "g1_taskc_cache_manifest")]:
    add("R", rel, None, role, "computation_input", BUILD)
for rel, role in [
        ("code/c17_replay_supplements_b/replay.py", "driver_code:replay.py"),
        ("code/c17_replay_supplements_b/g1_taskc_regen.py",
         "driver_code:g1_taskc_regen.py"),
        ("code/c17_replay_supplements_b/build_expected_manifest.py",
         "driver_code:build_expected_manifest.py")]:
    add("R", rel, None, role, "code_pin", BUILD)

out = {"schema": "c17-replay-supplements-b-expected-inputs-v1",
       "producer": "code/c17_replay_supplements_b/build_expected_manifest.py",
       "note": "Every declared input is hash-pinned. Historical hashes come from frozen "
               "artifacts; pin_source='pinned_at_C10v_build' marks files with no historical "
               "hash column (still tamper-evident for the replay run).",
       "files": files}
dest = Path(__file__).resolve().parent / "expected_manifest_v1.json"
dest.write_text(json.dumps(out, indent=1) + "\n")
print("wrote %s (%d entries; %d historical pins, %d build pins)" % (
    dest, len(files),
    sum(1 for f in files if f.get("pin_source") != BUILD),
    sum(1 for f in files if f.get("pin_source") == BUILD)))
