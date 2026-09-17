"""Sandboxed, path-safe execution of byte-identical frozen consumers.

The frozen consumers in ``code/`` resolve their root from ``Path(__file__)`` and
therefore write next to the signed sources. To keep R6 inside a fresh output
directory, this runner:

1. byte-copies the declared consumer into ``<out>/sandbox/<stage>/code/`` and
   verifies the copy hash equals the declared hash;
2. stages each declared input at its declared analysis-root-relative path inside
   ``<out>/sandbox/<stage>/`` (hardlink when possible, else a byte copy) and
   re-verifies the staged hash;
3. runs the copy with ``cwd=<sandbox>`` and ``PYTHONDONTWRITEBYTECODE=1`` so all
   ``__file__``-relative reads/writes stay inside the fresh output root;
4. returns the sandbox root plus a receipt with input/output hashes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import guards, manifest


def stage_bytes(src: Path, dst: Path) -> dict:
    """Byte-copy ``src`` to ``dst`` and verify the bytes.

    This legacy runner (frozen-consumer stages) always copies: a hardlink would
    share its inode with the signed tree, so a consumer that resolves
    ``Path(__file__)`` inside the sandbox could rewrite the original through the
    link.  Copies keep every write inside the sandbox by construction.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    strategy = "copy"
    shutil.copyfile(src, dst)
    source_sha = guards.sha256_file(src)
    staged_sha = guards.sha256_file(dst)
    if source_sha != staged_sha:
        raise guards.ReplayRefusal(f"staged bytes differ from source: {src} -> {dst}")
    return {"src": str(src), "staged": str(dst), "sha256": staged_sha, "strategy": strategy}


def run_stage(
    *,
    stage: dict,
    out_root: Path,
    project_root: Path,
    analysis_root: Path,
    consumer_code: Path,
    input_paths: list[Path],
) -> dict:
    """Execute one ``frozen_consumer_sandbox`` stage. Returns a receipt dict."""
    sandbox = Path(out_root) / "sandbox" / stage["id"]
    code_dir = sandbox / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    code_dst = code_dir / Path(consumer_code).name
    code_stage = stage_bytes(consumer_code, code_dst)

    staged_inputs = []
    for src in input_paths:
        try:
            rel = Path(src).resolve().relative_to(Path(analysis_root).resolve())
        except ValueError as exc:
            raise guards.ReplayRefusal(
                f"stage {stage['id']}: input outside --analysis-root cannot be staged: {src}"
            ) from exc
        staged_inputs.append(stage_bytes(Path(src), sandbox / rel))

    logs = Path(out_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{stage['id']}.stdout.log"
    stderr_path = logs / f"{stage['id']}.stderr.log"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONHASHSEED", "0")
    command = [sys.executable, "-B", str(code_dst), *stage.get("argv", [])]
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        completed = subprocess.run(command, cwd=str(sandbox), env=env, stdout=out, stderr=err)
    receipt = {
        "stage": stage["id"],
        "command": command,
        "cwd": str(sandbox),
        "exit_code": completed.returncode,
        "consumer": code_stage,
        "staged_inputs": staged_inputs,
        "stdout": str(stdout_path),
        "stdout_sha256": guards.sha256_file(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": guards.sha256_file(stderr_path),
    }
    if completed.returncode != 0:
        manifest.write_json_atomic(logs / f"{stage['id']}.failure.json", receipt)
        raise guards.ReplayRefusal(
            f"stage {stage['id']} failed with exit code {completed.returncode}; "
            f"receipt kept at {logs / (stage['id'] + '.failure.json')}"
        )
    return receipt
