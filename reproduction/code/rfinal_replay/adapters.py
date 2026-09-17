"""Path-safe adapters for the external branch producers (C17/C8).

Each declared ``external_producer`` stage runs inside a fresh per-stage sandbox
under the run output root:

    <out>/sandbox/<stage>/
        code/...        byte-identical (or declaratively migrated) producer code
        config/...      staged allowlisted inputs at their analysis-root paths
        runs/...        accepted caches, and fresh outputs of earlier stages
        Imports/...     staged project-root inputs (mirror of P for P=R.parent)

The producer is executed by :mod:`rfinal_replay.audit_runner`, which pins
BLAS/OpenMP threads to 1, caps worker pools, refuses any read outside the
sandbox + interpreter environment and any write outside the sandbox, and logs
every access.  Declared outputs are then copied to
``<out>/<collect_to>/`` and registered so later stages can consume them as
fresh inputs of the same run.

Nothing here reads or writes the signed tree: original files are only opened
for hashing via the allowlist verifier.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import guards, manifest
from . import transform as transform_mod

ADAPTER_SCHEMA = "rfinal-external-producer-adapter-v1"
RECEIPT_SCHEMA = "rfinal-external-producer-receipt-v1"
DEFAULT_RESOURCE_POLICY = {"max_workers": 2, "blas_threads": 1, "cap_pools": True}


def adapter_spec(stage: dict) -> dict:
    spec = stage.get("adapter")
    if not isinstance(spec, dict):
        raise guards.ReplayRefusal(f"stage {stage['id']}: external producer lacks an adapter spec")
    if spec.get("schema") != ADAPTER_SCHEMA:
        raise guards.ReplayRefusal(
            f"stage {stage['id']}: adapter schema must be {ADAPTER_SCHEMA!r}, got {spec.get('schema')!r}"
        )
    for key in ("code_path", "consumer_code_ref"):
        if not spec.get(key):
            raise guards.ReplayRefusal(f"stage {stage['id']}: adapter spec lacks {key!r}")
    if not spec.get("outputs") and spec.get("execution_mode", "script") != "import":
        raise guards.ReplayRefusal(f"stage {stage['id']}: adapter spec lacks 'outputs'")
    for output in spec["outputs"]:
        _assert_relative_safe(output["path"], f"stage {stage['id']} adapter output")
    for fresh in spec.get("fresh_inputs", []):
        _assert_relative_safe(fresh["path"], f"stage {stage['id']} adapter fresh input")
    return spec


def _assert_relative_safe(rel: str, context: str) -> None:
    if not rel or rel.startswith("/") or os.path.isabs(rel):
        raise guards.ReplayRefusal(f"{context}: path must be sandbox-relative: {rel!r}")
    normalized = os.path.normpath(rel)
    if normalized.startswith("..") or normalized in (".", ""):
        raise guards.ReplayRefusal(f"{context}: path escapes the sandbox: {rel!r}")


def _sandbox_path(sandbox: Path, rel: str, context: str) -> Path:
    _assert_relative_safe(rel, context)
    path = (sandbox / rel).resolve()
    if not str(path).startswith(str(sandbox.resolve()) + os.sep):
        raise guards.ReplayRefusal(f"{context}: resolved outside sandbox: {rel!r}")
    return path


def stage_bytes(src: Path, dst: Path, *, strategy: str = "copy") -> dict:
    """Stage ``src`` at ``dst`` and verify the bytes.

    The default is a byte copy with hash verification.  Hardlinks are never used
    for code or for files that any declared migration rewrites, because a
    transform on a hardlinked file would silently rewrite the signed original.
    Hardlinks are opt-in per *data* entry (``staging: hardlink``) for very large
    read-only inputs, and the engine re-verifies every hardlinked entry after the
    producer exits (see ``run_external_stage``).
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    applied = "copy"
    if strategy == "hardlink":
        try:
            os.link(src, dst)
            applied = "hardlink"
        except OSError:
            applied = "copy"
    if applied == "copy":
        shutil.copyfile(src, dst)
    source_sha = guards.sha256_file(src)
    staged_sha = guards.sha256_file(dst)
    if source_sha != staged_sha:
        raise guards.ReplayRefusal(f"staged bytes differ from source: {src} -> {dst}")
    return {"src": str(src), "staged": str(dst), "sha256": staged_sha, "strategy": applied}


def _entry_sandbox_rel(entry: dict, stage_id: str) -> str:
    rel = entry.get("sandbox_path")
    if rel:
        _assert_relative_safe(rel, f"stage {stage_id} input {entry['id']}")
        return rel
    if entry["root"] in ("analysis", "project"):
        _assert_relative_safe(entry["path"], f"stage {stage_id} input {entry['id']}")
        return entry["path"]
    raise guards.ReplayRefusal(
        f"stage {stage_id}: absolute-root whitelist entry {entry['id']} needs an explicit sandbox_path"
    )


def layout_rel(rel: str, root_kind: str, spec: dict, stage_id: str) -> str:
    """Sandbox-relative destination honouring the optional project-mirror layout.

    Default layout: the sandbox *is* the analysis root (P == sandbox for the
    producers that resolve ``P = R.parent``).  With ``layout.project_mirror`` the
    sandbox is a mirror of the project root instead, so project-root inputs stay
    at their relative path and analysis-root inputs move under
    ``<analysis_mount>/`` -- required for producers that embed the absolute
    project root and derive ``R = P / "<analysis dir name>"``.
    """
    layout = spec.get("layout") or {}
    if layout.get("project_mirror"):
        if root_kind == "project":
            prefix = ""
        elif root_kind == "analysis":
            mount = layout.get("analysis_mount")
            if not mount:
                raise guards.ReplayRefusal(f"stage {stage_id}: project_mirror layout needs analysis_mount")
            prefix = mount
        else:
            raise guards.ReplayRefusal(f"stage {stage_id}: unsupported root kind {root_kind!r}")
    else:
        prefix = (spec.get("sandbox_prefix") or "").strip("/")
    combined = f"{prefix}/{rel}" if prefix else rel
    _assert_relative_safe(combined, f"stage {stage_id} layout path")
    return combined


def expand_stage_inputs(stage: dict, entries: dict[str, dict]) -> list[str]:
    spec = adapter_spec(stage)
    resolved: list[str] = []
    for entry_id in spec.get("input_refs", []):
        if entry_id not in entries:
            raise guards.ReplayRefusal(f"stage {stage['id']}: unknown adapter input_ref {entry_id}")
        resolved.append(entry_id)
    for pattern in spec.get("input_globs", []):
        matches = sorted(entry_id for entry_id in entries if fnmatch.fnmatchcase(entry_id, pattern))
        if not matches:
            raise guards.ReplayRefusal(f"stage {stage['id']}: adapter input glob matches nothing: {pattern}")
        resolved.extend(matches)
    seen: set[str] = set()
    unique: list[str] = []
    for entry_id in resolved:
        if entry_id not in seen:
            seen.add(entry_id)
            unique.append(entry_id)
    if not unique and not spec.get("fresh_inputs"):
        raise guards.ReplayRefusal(f"stage {stage['id']}: adapter declares no inputs")
    return unique


def _expand_tokens(text: str, placeholders: dict) -> str:
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


def safe_name(stage_id: str) -> str:
    """File-name-safe rendering of a stage id (stage ids may contain '/')."""
    return stage_id.replace("/", "_").replace(" ", "_")


def _parse_access_log(path: Path) -> dict:
    counts: dict[str, int] = {}
    refused: list[dict] = []
    outside_roots: dict[str, int] = {}
    rows = 0
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            event = record.get("event", "?")
            counts[event] = counts.get(event, 0) + 1
            if event.startswith("refused"):
                refused.append(record)
    return {"rows": rows, "event_counts": counts, "refused": refused, "outside_roots": outside_roots}


def run_external_stage(
    *,
    stage: dict,
    out_root: Path,
    project_root: Path,
    analysis_root: Path,
    allow: guards.InputAllowlist,
    fresh_registry: dict[str, list[dict]],
    entries: dict[str, dict],
    env_executable: Path,
    access_log_dir: Path | None = None,
) -> dict:
    """Run one declared external producer inside its sandbox. Returns a receipt."""
    stage_id = stage["id"]
    spec = adapter_spec(stage)
    sandbox = Path(out_root) / "sandbox" / stage_id
    if sandbox.exists() and any(sandbox.iterdir()):
        raise guards.ReplayRefusal(f"stage {stage_id}: sandbox already populated (no in-place resume): {sandbox}")
    sandbox.mkdir(parents=True, exist_ok=True)
    layout = spec.get("layout") or {}
    mount_rel = layout.get("analysis_mount") if layout.get("project_mirror") else None
    if layout.get("project_mirror") and not mount_rel:
        raise guards.ReplayRefusal(f"stage {stage_id}: project_mirror layout needs analysis_mount")
    # pre-create the standard analysis-root directories so frozen producers can
    # create their own run/report directories without needing the signed tree
    analysis_root_dirs = ("runs", "reports", "config", "verifier", "figures", "tables",
                          "inputs_manifest", "recovered", "sources", "code")
    for name in analysis_root_dirs:
        (sandbox / name).mkdir(parents=True, exist_ok=True)
        if mount_rel:
            (sandbox / mount_rel / name).mkdir(parents=True, exist_ok=True)

    # 1. Byte-copy producer code (+ helper modules) and verify hashes.
    code_stages = []
    for code_ref in [spec["consumer_code_ref"], *spec.get("extra_code_refs", [])]:
        if code_ref not in entries:
            raise guards.ReplayRefusal(f"stage {stage_id}: unknown code ref {code_ref}")
        entry = entries[code_ref]
        src = allow.resolve(entry)
        allow.verify(entry, force=True)
        override = (spec.get("code_paths") or {}).get(code_ref)
        if override:
            rel = layout_rel(override, "analysis", spec, stage_id)
        else:
            rel = layout_rel(entry.get("sandbox_path") or entry["path"], entry["root"], spec, stage_id)
        code_stages.append({**stage_bytes(src, _sandbox_path(sandbox, rel, f"stage {stage_id} code")), "ref": code_ref, "path": rel})
    target_rel = layout_rel(spec["code_path"], "analysis", spec, stage_id)

    # 2. Stage allowlisted inputs at their declared sandbox-relative paths.
    input_entry_ids = expand_stage_inputs(stage, entries)
    verified_inputs = allow.verify_many(input_entry_ids, force=True)
    staged_inputs = []
    for entry_id in input_entry_ids:
        entry = entries[entry_id]
        rel = layout_rel(_entry_sandbox_rel(entry, stage_id), entry["root"], spec, stage_id)
        staged_inputs.append(
            {
                "id": entry_id,
                "level": entry["level"],
                "sandbox_path": rel,
                **stage_bytes(
                    allow.resolve(entry),
                    _sandbox_path(sandbox, rel, f"stage {stage_id} input"),
                    strategy=entry.get("staging", "copy"),
                ),
            }
        )

    # 3. Stage fresh outputs of earlier stages declared by this adapter.
    fresh_staged = []
    for fresh in spec.get("fresh_inputs", []):
        source_stage = fresh["stage"]
        rel = fresh["path"]
        dest_rel = fresh.get("as", rel)
        _assert_relative_safe(dest_rel, f"stage {stage_id} fresh input destination")
        record = None
        for row in fresh_registry.get(source_stage, []):
            if row["path"] == rel:
                record = row
                break
        if record is None:
            raise guards.ReplayRefusal(
                f"stage {stage_id}: declared fresh input {rel} from {source_stage} is unavailable "
                "(upstream stage did not run or failed); refusing to fall back to old terminal values"
            )
        src = Path(record["sandbox_path"])
        if not src.is_file():
            raise guards.ReplayRefusal(f"stage {stage_id}: fresh input vanished: {src}")
        if guards.sha256_file(src) != record["sha256"]:
            raise guards.ReplayRefusal(f"stage {stage_id}: fresh input changed mid-run: {src}")
        fresh_staged.append(
            {
                "source_stage": source_stage,
                "path": rel,
                "sandbox_path": dest_rel,
                "source_sha256": record["sha256"],
                **stage_bytes(src, _sandbox_path(sandbox, dest_rel, f"stage {stage_id} fresh input")),
            }
        )

    # 4. Apply declared migrations to staged copies only.  ``to`` placeholders
    #    keep the config independent of the run directory:
    #      $SANDBOX          the sandbox root (project mirror for P=R.parent stages)
    #      $ANALYSIS_MOUNT   the sandbox path that plays the role of the analysis root
    layout = spec.get("layout") or {}
    mount_path = sandbox / layout["analysis_mount"] if layout.get("project_mirror") else sandbox
    placeholders = {"$SANDBOX": str(sandbox), "$ANALYSIS_MOUNT": str(mount_path)}
    expanded_transforms = []
    for item in spec.get("transforms", []):
        item = dict(item)
        if item.get("to"):
            for token, value in placeholders.items():
                item["to"] = item["to"].replace(token, value)
        elif not item.get("rules"):
            item["to"] = str(mount_path)
        if item.get("rules"):
            item["rules"] = [
                {**rule, "to": _expand_tokens(str(rule.get("to", "")), placeholders)}
                for rule in item["rules"]
            ]
        item["target"] = layout_rel(item["target"], "analysis", spec, stage_id)
        expanded_transforms.append(item)
    fresh_paths = {item.get("sandbox_path", item["path"]): item["source_sha256"] for item in fresh_staged}
    from_prefixes = [str(project_root), str(analysis_root)]
    transform_receipts = transform_mod.apply_transforms(
        sandbox=sandbox,
        specs=expanded_transforms,
        from_prefixes=from_prefixes,
        to_prefix=str(mount_path),
        fresh_paths=fresh_paths,
    )

    # 5. Execute under the audit wrapper with the isolated interpreter.
    engine_dir = Path(out_root) / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    runner_src = Path(__file__).resolve().parent / "audit_runner.py"
    runner_dst = engine_dir / "audit_runner.py"
    stage_bytes(runner_src, runner_dst)
    logs = Path(out_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    file_tag = safe_name(stage_id)
    access_log = (access_log_dir or logs) / f"{file_tag}.access.jsonl"
    summary_path = logs / f"{file_tag}.sandbox_summary.json"
    stdout_path = logs / f"{file_tag}.stdout.log"
    stderr_path = logs / f"{file_tag}.stderr.log"
    policy = {**DEFAULT_RESOURCE_POLICY, **(spec.get("resource_policy") or {})}
    mode = spec.get("execution_mode", "script")
    command = [
        str(env_executable),
        "-B",
        str(runner_dst),
        "--sandbox",
        str(sandbox),
        "--target",
        target_rel,
        "--mode",
        mode,
        "--argv",
        json.dumps(spec.get("argv", [])),
        "--access-log",
        str(access_log),
        "--summary",
        str(summary_path),
        "--max-workers",
        str(policy["max_workers"]),
        "--blas-threads",
        str(policy["blas_threads"]),
    ]
    if policy.get("cap_pools"):
        command.append("--cap-pools")
    if layout.get("project_mirror"):
        command += ["--extra-sys-path", str(sandbox / mount_rel / "code"), "--extra-sys-path", str(sandbox / mount_rel)]
    for root in spec.get("extra_read_roots", []) or []:
        command += ["--extra-read-root", str(root)]
    if mode == "fixture":
        fixture_rel = spec.get("fixture_path")
        if not fixture_rel:
            raise guards.ReplayRefusal(f"stage {stage_id}: fixture mode without fixture_path")
        fixture_src = Path(__file__).resolve().parent / fixture_rel
        fixture_dst = _sandbox_path(sandbox, f"__engine__/{fixture_src.name}", f"stage {stage_id} fixture")
        stage_bytes(fixture_src, fixture_dst)
        command += ["--fixture", str(fixture_dst.relative_to(sandbox))]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    timeout = int(spec.get("timeout_seconds", 3600))
    started = manifest.utc_now()
    try:
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            completed = subprocess.run(
                command, cwd=str(sandbox), env=env, stdout=out, stderr=err, timeout=timeout
            )
        exit_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        exit_code = -9
        timed_out = True
    sandbox_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    access = _parse_access_log(access_log)

    # Hardlinked inputs share their inode with the signed tree: verify that the
    # producer did not rewrite them, otherwise the run fails and the shared
    # inode is reported instead of being silently accepted.
    hardlink_checks = []
    for item in staged_inputs:
        if item.get("strategy") != "hardlink":
            continue
        actual = guards.sha256_file(Path(item["staged"]))
        hardlink_checks.append(
            {
                "id": item["id"],
                "sandbox_path": item["sandbox_path"],
                "expected_sha256": item["sha256"],
                "observed_sha256": actual,
                "unchanged": actual == item["sha256"],
            }
        )
        if actual != item["sha256"]:
            raise guards.ReplayRefusal(
                f"stage {stage_id}: hardlinked input was modified during the run "
                f"({item['id']}: {item['sha256']} -> {actual}); the signed original may share this inode"
            )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage": stage_id,
        "node": stage.get("node"),
        "started_utc": started,
        "ended_utc": manifest.utc_now(),
        "command": command,
        "cwd": str(sandbox),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "status": "COMPLETE" if exit_code == 0 and sandbox_summary.get("status") == "COMPLETED" else "FAILED",
        "code": code_stages,
        "target": target_rel,
        "verified_inputs": verified_inputs,
        "staged_inputs": staged_inputs,
        "fresh_inputs": fresh_staged,
        "transforms": transform_receipts,
        "resource_policy": policy,
        "hardlinked_input_checks": hardlink_checks,
        "environment_executable": str(env_executable),
        "sandbox_summary": sandbox_summary,
        "access_log": {"path": str(access_log), "sha256": guards.sha256_file(access_log) if access_log.is_file() else None, **access},
        "stdout": str(stdout_path),
        "stdout_sha256": guards.sha256_file(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": guards.sha256_file(stderr_path),
        "outputs": [],
    }
    if access["refused"]:
        receipt["status"] = "FAILED"
        receipt["refusal_reason"] = "sandbox access refusal(s) recorded; see access log"
    if exit_code != 0 or receipt["status"] != "COMPLETE":
        manifest.write_json_atomic(logs / f"{file_tag}.failure.json", receipt)
        raise guards.ReplayRefusal(
            f"stage {stage_id} failed (exit={exit_code}, timeout={timed_out}); receipt kept at "
            f"{logs / (file_tag + '.failure.json')}"
        )

    # 6. Verify declared outputs stayed inside the sandbox and collect them.
    outputs = collect_declared_outputs(stage_id=stage_id, sandbox=sandbox, out_root=out_root,
                                       spec=spec, fresh_registry=fresh_registry)
    receipt["outputs"] = outputs
    receipt["fresh_registry_entries"] = len(fresh_registry.get(stage_id, []))
    manifest.write_json_atomic(Path(out_root) / "receipts" / f"{safe_name(stage_id)}.json", receipt)
    return receipt


def collect_declared_outputs(*, stage_id: str, sandbox: Path, out_root: Path, spec: dict,
                             fresh_registry: dict[str, list[dict]] | None = None) -> list[dict]:
    """Collect declared file/directory outputs; safe, hash-verified, run-root bound."""
    fresh_registry = fresh_registry if fresh_registry is not None else {}
    collect_root = Path(out_root) / spec.get("collect_to", f"branches/{stage_id}")
    outputs = []
    sandbox_root = sandbox.resolve()
    out_root_resolved = Path(out_root).resolve()
    for output in spec["outputs"]:
        rel = output["path"].rstrip("/") if output["path"] != "/" else output["path"]
        produced = _sandbox_path(sandbox, rel, f"stage {stage_id} output")
        candidates = []
        if produced.is_symlink():
            raise guards.ReplayRefusal(f"stage {stage_id}: declared output is a symlink: {rel}")
        if produced.is_dir():
            for src in sorted(produced.rglob("*")):
                if src.is_symlink():
                    raise guards.ReplayRefusal(f"stage {stage_id}: symlink inside declared output dir: {src}")
                if src.is_file():
                    candidates.append(src)
            if not candidates:
                raise guards.ReplayRefusal(f"stage {stage_id}: declared output dir is empty: {rel}")
        elif produced.is_file():
            candidates.append(produced)
        else:
            raise guards.ReplayRefusal(f"stage {stage_id}: declared output missing inside sandbox: {rel}")
        for src in candidates:
            if not str(src.resolve()).startswith(str(sandbox_root) + os.sep):
                raise guards.ReplayRefusal(f"stage {stage_id}: produced file escapes the sandbox: {src}")
            sub = str(src.relative_to(produced)) if produced.is_dir() else ""
            full_rel = f"{rel}/{sub}".replace("//", "/") if sub else rel
            _assert_relative_safe(full_rel, f"stage {stage_id} collected output")
            target = collect_root / full_rel
            if not str(target.resolve()).startswith(str(out_root_resolved) + os.sep):
                raise guards.ReplayRefusal(f"stage {stage_id}: collected output escapes the run root: {full_rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = guards.sha256_file(src)
            shutil.copyfile(src, target)
            if guards.sha256_file(target) != digest:
                raise guards.ReplayRefusal(f"stage {stage_id}: collected output hash mismatch: {full_rel}")
            row = {
                "path": full_rel,
                "role": output.get("role", ""),
                "result_ids": output.get("result_ids", []),
                "sha256": digest,
                "sandbox_path": str(src),
                "collected_path": str(target),
                "bytes": src.stat().st_size,
            }
            outputs.append(row)
            # canonical registry record: the actual sandbox file, not a relative alias
            fresh_registry.setdefault(stage_id, []).append(
                {"path": str(src), "sha256": digest, "sandbox_path": str(src),
                 "collected_path": str(target), "declared_path": full_rel}
            )
    return outputs

def verify_isolated_environment(
    *,
    executable: Path,
    expected_pins: dict[str, str],
    analysis_root: Path,
    packages: tuple[str, ...] = ("numpy", "scipy", "sympy", "girth"),
) -> dict:
    """Import versions + interpreter/file provenance inside the isolated env."""
    program = (
        "import importlib, importlib.metadata as m, json, sys, pathlib\n"
        f"pins={list(packages)!r}\n"
        "out={'python':sys.version.split()[0],'executable':sys.executable,'prefix':sys.prefix,'packages':{}}\n"
        "for name in pins:\n"
        "    mod=importlib.import_module(name)\n"
        "    try: version=m.version(name)\n"
        "    except Exception: version=getattr(mod,'__version__',None)\n"
        "    out['packages'][name]={'version':version,'file':getattr(mod,'__file__',None),"
        "'source_package':str(pathlib.Path(getattr(mod,'__file__','')).resolve().parent)}\n"
        "print(json.dumps(out))\n"
    )
    completed = subprocess.run(
        [str(executable), "-B", "-c", program], capture_output=True, text=True, timeout=300
    )
    if completed.returncode != 0:
        return {
            "status": "FAILED",
            "error": completed.stderr.strip()[-800:],
            "executable": str(executable),
        }
    probe = json.loads(completed.stdout)
    problems = []
    prefix = str(Path(probe["prefix"]).resolve())
    if not prefix.startswith(str(Path(analysis_root).resolve())):
        problems.append(f"interpreter prefix {prefix} is outside the analysis tree (not an isolated replay environment)")
    for name, expected in expected_pins.items():
        got = (probe["packages"].get(name) or {}).get("version")
        if got != expected:
            problems.append(f"{name}: expected {expected}, got {got}")
    for name, row in probe["packages"].items():
        source = str(row.get("source_package") or "")
        if source and not source.startswith(prefix):
            problems.append(f"{name}: module resolves outside the env prefix: {source}")
    probe["status"] = "PASS" if not problems else "FAILED"
    probe["problems"] = problems
    return probe
