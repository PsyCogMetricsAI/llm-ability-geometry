"""Isolated replay environment: real venv creation, dependency provenance, verification.

C17/C8 requires an *actually built* isolated environment, not a requirements
file claiming reconstruction.  This module creates a fresh venv with the pinned
CPython build, installs the declared CPU replay pins, records every wheel
origin + sha256 from the pip install report, proves the interpreter prefix is
inside the analysis tree, proves the installed packages resolve inside the new
venv (no symlink farm back into the system environment) and writes the lock +
receipt into ``R/recovered/c17_replay_env/``.

The GPU state-extraction environment is declared separately and deliberately
not built here: the accepted window routes replay saved caches, and the pinned
GPU stack (torch/transformers/...) is a different, much larger environment
whose rebuild is recorded as an explicit open gap instead of being implied by a
symlinked system install.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import guards, manifest

ENV_SCHEMA = "rfinal-replay-environment-v1"
CPU_REPLAY_PINS = {
    "numpy": "2.5.0",
    "scipy": "1.18.1",
    "girth": "0.8.0",
    "sympy": "1.14.0",
}
GPU_STATE_PINS = {
    "torch": "2.12.1",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "accelerate": "1.15.0",
}


def env_root(analysis_root: Path) -> Path:
    return Path(analysis_root) / "recovered" / "c17_replay_env"


def cpu_replay_python(analysis_root: Path) -> Path:
    return env_root(analysis_root) / "venv-cpu-replay" / "bin" / "python3"


def _run(command: list[str], *, timeout: int = 1800, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=cwd)


def _installed_tree_audit(venv: Path) -> dict:
    site = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    symlinks: list[dict] = []
    files = 0
    total_bytes = 0
    for base, _dirs, names in os.walk(site):
        for name in names:
            path = Path(base) / name
            files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
            if path.is_symlink():
                symlinks.append({"path": str(path), "target": os.readlink(path)})
    outside = [row for row in symlinks if not row["target"].startswith("/") and ".." in row["target"]]
    return {
        "site_packages": str(site),
        "files": files,
        "bytes": total_bytes,
        "symlinks": symlinks[:50],
        "n_symlinks": len(symlinks),
        "n_symlinks_escaping_venv": len(outside),
    }


def build_cpu_replay_env(
    *,
    analysis_root: Path,
    base_python: Path,
    pins: dict[str, str] | None = None,
    force_new: bool = False,
    pip_index: str = "https://pypi.org/simple",
) -> dict:
    analysis_root = Path(analysis_root)
    pins = dict(pins or CPU_REPLAY_PINS)
    root = env_root(analysis_root)
    root.mkdir(parents=True, exist_ok=True)
    venv = root / "venv-cpu-replay"
    if venv.exists() and any(venv.iterdir()) and not force_new:
        raise guards.ReplayRefusal(
            f"cpu replay environment already exists and is non-empty: {venv} "
            "(refusing in-place rebuild; remove or choose a new directory)"
        )
    if not Path(base_python).is_file():
        raise guards.ReplayRefusal(f"base interpreter not found: {base_python}")
    base_receipt = {
        "path": str(base_python),
        "sha256": guards.sha256_file(base_python),
        "version": _run([str(base_python), "--version"]).stdout.strip(),
    }
    started = manifest.utc_now()
    created = _run([str(base_python), "-m", "venv", "--copies", str(venv)], timeout=900)
    if created.returncode != 0:
        raise guards.ReplayRefusal(f"venv creation failed: {created.stderr.strip()[-500:]}")
    python = venv / "bin" / "python3"
    report_path = root / "pip-install-report-cpu-replay.json"
    requirements = [f"{name}=={version}" for name, version in sorted(pins.items())]
    install = _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--index-url",
            pip_index,
            "--report",
            str(report_path),
            *requirements,
        ],
        timeout=3600,
    )
    if install.returncode != 0:
        raise guards.ReplayRefusal(
            f"pip install failed (exit {install.returncode}): {install.stderr.strip()[-800:]}"
        )
    report = json.loads(report_path.read_text())
    artifacts = []
    resolved_pins = {}
    for item in report.get("install", []):
        meta = item.get("metadata", {})
        name = meta.get("name")
        version = meta.get("version")
        download = item.get("download_info", {})
        url = download.get("url", "")
        archive = download.get("archive_info", {}) or {}
        hashes = archive.get("hashes", {}) or {}
        sha256 = hashes.get("sha256") or (archive.get("hash", "").split("=", 1)[-1] if archive.get("hash") else None)
        requested = item.get("requested", False)
        artifacts.append(
            {
                "name": name,
                "version": version,
                "requested": requested,
                "url": url,
                "sha256": sha256,
                "wheel": url.split("/")[-1],
            }
        )
        if requested:
            resolved_pins[name] = version
    problems = []
    for name, version in pins.items():
        if resolved_pins.get(name) != version:
            problems.append(f"pin {name}=={version} resolved to {resolved_pins.get(name)!r}")
    tree = _installed_tree_audit(venv)
    if tree["n_symlinks_escaping_venv"]:
        problems.append(f"{tree['n_symlinks_escaping_venv']} symlinks escape the venv")
    if not str(venv.resolve()).startswith(str(analysis_root.resolve())):
        problems.append("venv is outside the analysis tree")
    from . import adapters

    probe = adapters.verify_isolated_environment(
        executable=python, expected_pins=pins, analysis_root=analysis_root
    )
    if probe.get("status") != "PASS":
        problems.append(f"import/version probe failed: {probe.get('problems') or probe.get('error')}")
    site_packages_leak = [
        row for row in probe.get("packages", {}).values()
        if row.get("source_package") and not str(row["source_package"]).startswith(str(venv))
    ]
    if site_packages_leak:
        problems.append(f"packages resolved outside the venv: {site_packages_leak}")
    lock = {
        "schema": ENV_SCHEMA,
        "kind": "cpu-replay",
        "created_utc": manifest.utc_now(),
        "base_interpreter": base_receipt,
        "venv": str(venv),
        "interpreter": str(python),
        "pins": pins,
        "resolved_pins": resolved_pins,
        "artifacts": artifacts,
        "pip_index": pip_index,
        "install_seconds": None,
        "notes": [
            "isolated venv created with --copies; installed packages live only in the venv",
            "recorded wheel urls and sha256 come from the pip install report (public index)",
            "this environment covers the CPU branch producers (numpy/scipy/girth/sympy); it is not the GPU state-extraction stack",
        ],
        "problems": problems,
        "status": "BUILT_AND_VERIFIED" if not problems else "BUILT_WITH_PROBLEMS",
    }
    lock_path = root / "cpu-replay-lock.json"
    lock["sha256_path"] = manifest.write_json_atomic(lock_path, lock)
    engine_copy = Path(__file__).resolve().parent / "environment" / "cpu-replay-lock.json"
    engine_copy.parent.mkdir(parents=True, exist_ok=True)
    engine_copy.write_text(json.dumps(lock, indent=2) + "\n")
    receipt = {
        "schema": "rfinal-env-build-receipt-v1",
        "started_utc": started,
        "ended_utc": manifest.utc_now(),
        "lock": str(lock_path),
        "lock_sha256": lock["sha256_path"],
        "tree_audit": tree,
        "import_probe": probe,
        "venv_create": {"exit_code": created.returncode},
        "pip_install": {"exit_code": install.returncode, "command": install.args, "report": str(report_path)},
        "status": lock["status"],
        "problems": problems,
    }
    manifest.write_json_atomic(root / "ENV_BUILD_RECEIPT.json", receipt)
    return {"lock": lock, "receipt": receipt, "python": str(python)}


def verify_cpu_replay_env(*, analysis_root: Path, pins: dict[str, str] | None = None) -> dict:
    analysis_root = Path(analysis_root)
    pins = dict(pins or CPU_REPLAY_PINS)
    root = env_root(analysis_root)
    lock_path = root / "cpu-replay-lock.json"
    if not lock_path.is_file():
        return {"status": "MISSING", "detail": f"no lock at {lock_path}; run the env builder first"}
    lock = json.loads(lock_path.read_text())
    python = Path(lock["interpreter"])
    if not python.is_file():
        return {"status": "FAILED", "detail": f"recorded interpreter missing: {python}"}
    from . import adapters

    probe = adapters.verify_isolated_environment(executable=python, expected_pins=pins, analysis_root=analysis_root)
    ok = (
        probe.get("status") == "PASS"
        and lock.get("status") == "BUILT_AND_VERIFIED"
        and not lock.get("problems")
        and guards.sha256_file(lock_path) == lock.get("sha256_path", guards.sha256_file(lock_path))
    )
    return {
        "status": "PASS" if ok else "FAILED",
        "lock": str(lock_path),
        "lock_sha256": guards.sha256_file(lock_path),
        "interpreter": str(python),
        "import_probe": probe,
        "pins": pins,
    }


def gpu_state_environment_declaration(analysis_root: Path) -> dict:
    """The GPU state-extraction stack is declared, hash-pinned and NOT built in C8."""
    existing = Path(analysis_root) / "recovered" / "window_offload_env"
    return {
        "kind": "gpu-state-extraction",
        "status": "DECLARED_NOT_BUILT_IN_C8",
        "pins": GPU_STATE_PINS,
        "reason": (
            "the accepted window results replay saved caches; the GPU stack is a separate, much larger "
            "environment (torch/transformers builds incl. CUDA identity) and C8 must not claim it is rebuilt "
            "by pointing at the system site-packages"
        ),
        "existing_offload_module": {
            "path": str(existing),
            "present": existing.is_dir(),
            "role": "accepted accelerate 1.15.0 offload environment used by saved window routes (declared input)",
        },
        "recipe": [
            "create a venv from the pinned CPython 3.12.11 build",
            "install the declared GPU pins from hash-recorded wheels in the order tokenizers, safetensors, "
            "huggingface-hub, transformers, accelerate, torch",
            "record wheel urls + sha256 and the CUDA/driver identity of the target host",
            "verify that every imported package resolves inside the new venv (not the host environment)",
        ],
        "required_by": "state-extraction routes only (not the CPU cache-replay path exercised in C17/C8)",
    }
