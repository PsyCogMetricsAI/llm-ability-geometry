"""CLI entry: render the seven C17 result outputs from declared sources.

Usage:
  python -m render.run_render --input-root R --outdir RUN_DIR [--contract C.json]

Exit codes: 0 ok; 3 missing/invalid required input; 4 bad contract schema;
5 renderer failure. Nothing is written before all declared inputs preflight.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from render import CANONICAL_IDS, EXIT_MISSING_INPUT, EXIT_RENDER_FAILURE, SCHEMA_CONTRACT, SCHEMA_INVENTORY
    from render.common import (RenderRefusal, load_json, refuse_bad_schema, refuse_missing_input,
                               resolve_input, resolve_pointer, sha256_file, write_json)
    from render.contract import default_contract
    from render.renderers import RENDERERS
else:
    from . import CANONICAL_IDS, EXIT_MISSING_INPUT, EXIT_RENDER_FAILURE, SCHEMA_CONTRACT, SCHEMA_INVENTORY
    from .common import (RenderRefusal, load_json, refuse_bad_schema, refuse_missing_input,
                         resolve_input, resolve_pointer, sha256_file, write_json)
    from .contract import default_contract
    from .renderers import RENDERERS

REQUIRED_ROLES = {
    "mtmm_schematic": {"result_manifest", "definition_reference"},
    "convergence_table": {"study1_main_results"},
    "science_grid_table": {"f1_rarefied_grid"},
    "lofo_table": {"study2_lofo_results", "f1_native_grid_crosscheck"},
    "extension_probes_table": {"ext_h1_results", "ext_h2_results", "ext_h3_results"},
    "depth_profiles_figure": {"ext_h4_depth_profiles", "ext_h4_depth_summary_archive",
                              "ext_h4_peak_depths_archive", "ext_h4_metadata"},
    "supplement_topology": {"result_manifest", "f1_native_grid_candidate",
                            "f1_rarefied_grid_candidate", "study1_main_candidate"},
}


def validate_contract(contract, input_root: Path):
    if not isinstance(contract, dict):
        refuse_bad_schema("contract must be a JSON object")
    if contract.get("schema") != SCHEMA_CONTRACT:
        refuse_bad_schema(f"unsupported contract schema: {contract.get('schema')!r}")
    outputs = contract.get("outputs")
    if not isinstance(outputs, list):
        refuse_bad_schema("contract.outputs must be a list")
    ids = [o.get("id") for o in outputs]
    if sorted(ids) != sorted(CANONICAL_IDS):
        missing = sorted(set(CANONICAL_IDS) - set(ids))
        extra = sorted(set(ids) - set(CANONICAL_IDS))
        refuse_bad_schema(f"contract must declare exactly the 7 canonical outputs; missing={missing} extra={extra}")
    if len(ids) != len(set(ids)):
        refuse_bad_schema("contract declares duplicate output ids")
    for o in outputs:
        for key in ("id", "kind", "renderer", "inputs"):
            if key not in o:
                refuse_bad_schema(f"output entry missing key {key!r}: {o.get('id')!r}")
        if o["renderer"] not in RENDERERS:
            refuse_bad_schema(f"unknown renderer {o['renderer']!r} for {o['id']}")
        if not isinstance(o["inputs"], list) or not o["inputs"]:
            refuse_bad_schema(f"output {o['id']} must declare a non-empty inputs list")
        roles = {s.get("role") for s in o["inputs"] if isinstance(s, dict)}
        missing_roles = REQUIRED_ROLES[o["renderer"]] - roles
        if missing_roles:
            refuse_bad_schema(f"output {o['id']} missing input roles: {sorted(missing_roles)}")


def preflight_inputs(contract, input_root: Path):
    records = []
    for o in contract["outputs"]:
        for spec in o["inputs"]:
            if not isinstance(spec, dict) or "path" not in spec:
                refuse_bad_schema(f"bad input spec in {o['id']}: {spec!r}")
            path = resolve_input(input_root, spec["path"], must_exist=True,
                                 allow_external=bool(spec.get("external")))
            digest = sha256_file(path)
            declared = spec.get("sha256")
            if declared is not None and digest != declared:
                # Hash pin enforcement: a declared sha256 is verified before any
                # output writes; a mismatch is an invalid-required-input refusal.
                refuse_missing_input(
                    f"declared input hash mismatch for {spec['path']}: "
                    f"actual {digest} != declared {declared}")
            rec = {"result_id": o["id"], "role": spec.get("role", ""),
                   "path_rel": spec["path"], "path_abs": str(path),
                   "sha256": digest, "declared_sha256": declared,
                   "sha256_pin_enforced": declared is not None,
                   "computational_input": bool(spec.get("computational_input", True))}
            if spec.get("comparison_only"):
                rec["comparison_only"] = True
                rec["computational_input"] = False
            if spec.get("pointer") is not None:
                resolve_pointer(load_json(path), spec["pointer"])
                rec["pointer"] = spec["pointer"]
            records.append(rec)
    return records


def load_final_config_contract(config_path: Path):
    """Load the seven-output render contract embedded in the final replay config.

    The embedded contract is hash-bound: ``render_contract_sha256`` must equal
    the canonical-JSON sha256 of ``render_contract``.
    """
    cfg = load_json(config_path)
    if cfg.get("schema") not in ("paper2-final-replay-config-v1",
                                 "paper2-rfinal-replay-draft-v1",
                                 "paper2-rfinal-replay-draft-v2"):
        refuse_bad_schema(f"unsupported final config schema: {cfg.get('schema')!r}")
    contract = cfg.get("render_contract")
    if not isinstance(contract, dict):
        refuse_bad_schema("final config has no render_contract object")
    import hashlib
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if cfg.get("render_contract_sha256") != digest:
        refuse_bad_schema(
            f"embedded render_contract hash mismatch: {digest} != {cfg.get('render_contract_sha256')}")
    return contract, str(config_path.resolve()), sha256_file(config_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--contract", default=None,
                    help="optional contract JSON; default is the prepare-stage contract")
    ap.add_argument("--final-config", default=None,
                    help="optional final replay config carrying an embedded render_contract")
    args = ap.parse_args(argv)

    input_root = Path(args.input_root).resolve()
    outdir = Path(args.outdir).resolve()
    started = time.time()
    try:
        if args.contract and args.final_config:
            refuse_bad_schema("--contract and --final-config are mutually exclusive")
        if args.final_config:
            contract, contract_path, contract_sha = load_final_config_contract(Path(args.final_config))
        elif args.contract:
            contract = load_json(Path(args.contract))
            contract_path = str(Path(args.contract).resolve())
            contract_sha = sha256_file(Path(args.contract))
        else:
            contract = default_contract(str(input_root))
            contract_path = "inline-default"
            contract_sha = None
        validate_contract(contract, input_root)
        records = preflight_inputs(contract, input_root)  # all-or-nothing, no writes before this
    except RenderRefusal as exc:
        sys.stderr.write(f"REFUSED[{exc.code}]: {exc.message}\n")
        return exc.code
    except FileNotFoundError as exc:
        sys.stderr.write(f"REFUSED[{EXIT_MISSING_INPUT}]: {exc}\n")
        return EXIT_MISSING_INPUT

    outdir.mkdir(parents=True, exist_ok=True)
    ctx = {"input_root": input_root, "outdir": outdir}
    items = []
    try:
        for entry in contract["outputs"]:
            items.append(RENDERERS[entry["renderer"]](ctx, entry))
    except RenderRefusal as exc:
        sys.stderr.write(f"REFUSED[{exc.code}] during render: {exc.message}\n")
        return exc.code
    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        traceback.print_exc()
        sys.stderr.write(f"RENDER_FAILURE: {exc}\n")
        return EXIT_RENDER_FAILURE

    try:
        import matplotlib
        import numpy
        env = {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
               "numpy": numpy.__version__, "platform": platform.platform(),
               "threads": {"OPENBLAS_NUM_THREADS": __import__("os").environ.get("OPENBLAS_NUM_THREADS"),
                           "OMP_NUM_THREADS": __import__("os").environ.get("OMP_NUM_THREADS"),
                           "MKL_NUM_THREADS": __import__("os").environ.get("MKL_NUM_THREADS")}}
    except Exception:
        env = {"python": sys.version.split()[0]}

    inventory = {
        "schema": SCHEMA_INVENTORY,
        "stage": contract.get("stage", "C10a-prepare"),
        "result_count": len(items),
        "expected_result_count": len(CANONICAL_IDS),
        "all_seven_present": sorted(i["id"] for i in items) == sorted(CANONICAL_IDS),
        "items": items,
    }
    write_json(outdir / "OUTPUT_INVENTORY.json", inventory)

    code_dir = Path(__file__).resolve().parent
    code_files = sorted(p for p in code_dir.glob("*.py"))
    provenance = {
        "schema": "c17-render-provenance-v1",
        "stage": contract.get("stage", "C10a-prepare"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "input_root": str(input_root),
        "outdir": str(outdir),
        "contract": {"path": contract_path, "sha256": contract_sha},
        "code_hashes": {p.name: sha256_file(p) for p in code_files},
        "environment": env,
        "declared_inputs": records,
        "note": ("Every emitted numeric value is read from the declared sources at render time via "
                 "explicit JSON pointers/CSV columns; no old final values or historical images are copied. "
                 "C11 must re-run this entrypoint with a contract pointing at freshly generated results."),
    }
    write_json(outdir / "PROVENANCE.json", provenance)
    summary = {"items": len(items), "seconds": round(time.time() - started, 2),
               "outdir": str(outdir)}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
