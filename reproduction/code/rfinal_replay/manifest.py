"""Run-manifest writer: input hashes, code hashes, environment, seeds,
tolerances and output hashes for one clean replay run.

The manifest is stdlib-only and performs no analysis. It never writes outside the
run's fresh output root.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import guards

MANIFEST_SCHEMA = "paper2-rfinal-run-manifest-v1"

REQUIRED_MANIFEST_KEYS = (
    "schema",
    "run_id",
    "created_utc",
    "mode",
    "project_root",
    "analysis_root",
    "config",
    "output_root",
    "guards",
    "inputs",
    "code",
    "environment",
    "seeds",
    "tolerances",
    "stages",
    "outputs",
    "status",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path: Path, value: object) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")
    os.replace(temp, path)
    return guards.sha256_file(path)


def environment_snapshot(executable: str | None = None) -> dict:
    """Record the interpreter and the frozen package pins without importing them.

    Uses ``importlib.metadata`` in a subprocess (no torch import, no CUDA, no GPU).
    Falls back to the recorded lock values only if the probe cannot run; the
    fallback source is stated explicitly in the manifest.
    """
    executable = executable or sys.executable
    packages = [
        "numpy",
        "scipy",
        "girth",
        "sympy",
        "torch",
        "transformers",
        "huggingface-hub",
        "safetensors",
        "tokenizers",
        "accelerate",
    ]
    program = (
        "import importlib.metadata as m, json, sys\n"
        f"names={packages!r}\n"
        "out={}\n"
        "for name in names:\n"
        "    try: out[name]=m.version(name)\n"
        "    except Exception: out[name]=None\n"
        "print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable, 'packages': out}))\n"
    )
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [executable, "-B", "-c", program],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip()[:400])
        snapshot = json.loads(completed.stdout)
        snapshot["source"] = f"live importlib.metadata probe of {executable}"
        return snapshot
    except Exception as exc:  # pragma: no cover - fallback path
        return {
            "python": platform.python_version(),
            "executable": executable,
            "packages": {},
            "source": f"fallback: probe failed ({type(exc).__name__}: {exc})",
        }


def code_hashes(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        path = Path(path)
        rows.append(
            {
                "path": path.name,
                "sha256": guards.sha256_file(path) if path.is_file() else None,
                "exists": path.is_file(),
            }
        )
    return rows


def build_manifest(
    *,
    run_id: str,
    mode: str,
    project_root: Path,
    analysis_root: Path,
    config_path: Path,
    config: dict,
    output_root: Path,
    empty_output_check: dict,
    inputs: list[dict],
    code: list[dict],
    environment: dict,
    stages: list[dict],
    outputs: list[dict],
    status: str,
    extra_guards: dict | None = None,
) -> dict:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_utc": utc_now(),
        "mode": mode,
        "project_root": str(project_root),
        "analysis_root": str(analysis_root),
        "config": {"path": str(config_path), "sha256": guards.sha256_file(config_path)},
        "output_root": str(output_root),
        "guards": {
            "empty_output_assertion": empty_output_check,
            "forbidden_computation_input_patterns": config["forbidden_computation_input_patterns"],
            "execution_authorization": config["execution_authorization"],
            "signed_tree_write_rule": config["root_contract"].get("signed_tree_write_rule", ""),
        },
        "inputs": inputs,
        "code": code,
        "environment": environment,
        "seeds": config["seeds"],
        "tolerances": config["tolerances"],
        "stages": stages,
        "outputs": outputs,
        "status": status,
    }
    if extra_guards:
        manifest["guards"].update(extra_guards)
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise guards.ReplayRefusal(f"manifest is missing required keys: {missing}")
    return manifest
