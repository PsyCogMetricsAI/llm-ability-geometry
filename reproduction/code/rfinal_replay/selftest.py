#!/usr/bin/env python3
"""Executor self-tests for the RFINAL clean-replay launcher.

Scope: tiny fixtures, temporary directories only, no analysis, no GPU, no writes
into the signed tree and no run output under R/runs. These are executor
self-checks; they are not independent verification.

Checks:
  T1 output-directory assertion (non-empty refused, empty accepted, missing accepted)
  T2 allowlist refusal (unlisted path refused, hash mismatch refused, listed path accepted)
  T3 forbidden-pattern enforcement in code and in config load
  T4 --dry-run prints the resolved plan and writes nothing outside its temp dir
  T5 launcher refuses a prepopulated --out
  T6 manifest contains the required keys (inputs/code/environment/seeds/tolerances/outputs)
  T7 seven downstream nodes each carry a named, non-exempt generating step
  T8 draft tolerances are left for root and gate execution
  T9 no run output directory and no leftover temp dirs
  T10 all 18 external producers carry complete, resource-capped adapter specs
  T11 adapter staging refusals (hash mismatch, missing fresh dependency, unsafe output path)
  T12 audit-hook enforcement (out-of-sandbox write refused, out-of-sandbox read refused)
  T13 forbidden computation inputs stay refused; declared exceptions block replay but not checks
  T14 producer-registration interface (template, validation, non-self-signing)
  T15 transform path boundary: a sibling sandbox_neighbor is never read or written
  T16 hardlinked transform target is copied to a private inode before any write
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rfinal_replay import adapters, guards, manifest, registry  # type: ignore
else:  # pragma: no cover
    from . import adapters, guards, manifest, registry

FIXTURE_PREFIX = "rfinal-selftest-"
NODES = (
    "P2-G-FIG-MTMM",
    "P2-B-TAB-CONVERGENCE",
    "P2-C-TAB-SCIENCE",
    "P2-C-TAB-LOFO",
    "P2-F-TAB-PROBES",
    "P2-F-FIG-PROFILES",
    "P2-G-TAB-SUPPLEMENTS",
)


def _minimal_config(entries: list[dict]) -> dict:
    return {
        "schema": "paper2-rfinal-replay-draft-v1",
        "root_contract": {"output_root_default": "runs/does-not-exist"},
        "execution_authorization": {"status": "NOT_AUTHORIZED", "required_status_for_run": "AUTHORIZED"},
        "empty_output_assertion": {},
        "forbidden_computation_input_patterns": {
            "basename_patterns": list(guards.FORBIDDEN_INPUT_BASENAME_PATTERNS),
            "path_patterns": list(guards.FORBIDDEN_INPUT_PATH_PATTERNS),
        },
        "input_whitelist": {"entries": entries},
        "environment_lock": {},
        "seeds": {},
        "tolerances": {"float_atol": "TO_BE_FROZEN_BY_ROOT_BEFORE_R6"},
        "failure_rules": {},
        "comparison_rules": {},
        "regeneration_plan": {
            "stages": [],
            "downstream_nodes": [
                {"result_id": node, "generating_step": f"step::{node}", "exempt": False} for node in NODES
            ],
        },
    }


def _run_cli(args: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    launcher = Path(__file__).resolve().parent / "launcher.py"
    return subprocess.run(
        [sys.executable, "-B", str(launcher), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def main(argv: list[str] | None = None) -> int:
    project_root = Path(os.environ.get("RFINAL_PROJECT_ROOT", str(Path.cwd()))).resolve()
    analysis_root = Path(
        os.environ.get("RFINAL_ANALYSIS_ROOT", str(project_root / "repro_paper2_20260915"))
    ).resolve()
    config_path = analysis_root / "config" / "RFINAL_REPLAY_DRAFT.json"
    c17_config_path = analysis_root / "config" / "CT_C17_REPLAY_FINAL_v1.relocated.json"
    if c17_config_path.is_file() and not os.environ.get("RFINAL_SELFTEST_LEGACY_CONFIG"):
        config_path = c17_config_path
    checks: list[dict] = []
    preexisting_temp = {path.name for path in Path(tempfile.gettempdir()).glob(FIXTURE_PREFIX + "*")}

    def check(name: str, fn) -> None:
        started = time.monotonic()
        try:
            detail = fn()
            checks.append({"name": name, "pass": True, "detail": detail, "seconds": round(time.monotonic() - started, 4)})
        except BaseException as exc:  # noqa: BLE001 - report any failure as a failed check
            checks.append(
                {
                    "name": name,
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.monotonic() - started, 4),
                }
            )

    def t1_output_assertion():
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            populated = tmpdir / "populated"
            populated.mkdir()
            (populated / "leftover.json").write_text("{}\n")
            refused = False
            try:
                guards.assert_empty_output_dir(populated)
            except guards.ReplayRefusal:
                refused = True
            assert refused, "non-empty output directory was not refused"
            empty = tmpdir / "empty"
            empty.mkdir()
            result = guards.assert_empty_output_dir(empty)
            assert result["pre_existing_empty"] is True, result
            missing = tmpdir / "missing"
            result = guards.assert_empty_output_dir(missing)
            assert result["exists"] is False, result
            assert not missing.exists(), "assertion must not create the output directory"
            return {"refused_non_empty": True, "accepted_empty": True, "accepted_missing": True}

    def t2_allowlist_refusal():
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            allowed = tmpdir / "allowed.json"
            allowed.write_text('{"tiny": true}\n')
            entries = [
                {
                    "id": "fixture_allowed",
                    "level": "L1_FROZEN_DERIVED_INPUT",
                    "root": "absolute",
                    "path": str(allowed),
                    "sha256": guards.sha256_file(allowed),
                    "role": "fixture",
                }
            ]
            allow = guards.InputAllowlist({"input_whitelist": {"entries": entries}}, tmpdir, tmpdir)
            assert allow.verify_many(["fixture_allowed"]) == {"fixture_allowed": entries[0]["sha256"]}
            unlisted = tmpdir / "unlisted.json"
            unlisted.write_text("{}\n")
            refused = False
            try:
                allow.entry_for_path(unlisted)
            except guards.ReplayRefusal:
                refused = True
            assert refused, "non-allowlisted input was not refused"
            tampered = tmpdir / "tampered.json"
            tampered.write_text('{"tiny": false}\n')
            entries_bad = [dict(entries[0], id="fixture_tampered", path=str(tampered))]
            allow_bad = guards.InputAllowlist({"input_whitelist": {"entries": entries_bad}}, tmpdir, tmpdir)
            refused = False
            try:
                allow_bad.verify_many(["fixture_tampered"])
            except guards.ReplayRefusal:
                refused = True
            assert refused, "hash-mismatched input was not refused"
            return {"listed_verified": True, "unlisted_refused": True, "hash_mismatch_refused": True}

    def t3_forbidden_patterns():
        forbidden = [
            "runs/window_smol_new_v1/metrics.json",
            "runs/window_qwen32_full200_offload_v1/report.json",
            "reports/EV2_FINAL_EVIDENCE_LEDGER_v2.json",
            "reports/EV2_WINDOW_RESULTS_SUMMARY_v1.json",
            "runs/x/run_final.json",
            "a/b/SOME_SUMMARY.md",
        ]
        allowed = [
            "runs/window_smol_new_v1/run_200.json",
            "runs/window_smol_new_v1/item_10.npz",
            "code/window_cloud_metrics.py",
            "reports/WINDOW_PYTHIA_TOKEN_COST.json",
        ]
        for path in forbidden:
            assert guards.is_forbidden_input_path(path), f"not flagged as forbidden: {path}"
        for path in allowed:
            assert not guards.is_forbidden_input_path(path), f"wrongly flagged as forbidden: {path}"
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            bad_entry = {
                "id": "forbidden_entry",
                "level": "L2_ACCEPTED_WINDOW_CACHE",
                "root": "analysis",
                "path": "runs/window_smol_new_v1/report.json",
                "sha256": "0" * 64,
                "role": "fixture",
            }
            refused = False
            try:
                guards.validate_config(_minimal_config([bad_entry]))
            except guards.ReplayRefusal:
                refused = True
            assert refused, "config with a forbidden whitelist entry was not refused at load"
            bad_stage_config = _minimal_config([bad_entry])
            bad_stage_config["input_whitelist"]["entries"] = []
            bad_stage_config["regeneration_plan"]["stages"] = [
                {"id": "s", "kind": "input_audit", "input_refs": ["missing_entry"]}
            ]
            refused = False
            try:
                guards.validate_config(bad_stage_config)
            except guards.ReplayRefusal:
                refused = True
            assert refused, "config with an unknown stage input_ref was not refused"
        return {"forbidden_paths_flagged": len(forbidden), "clean_paths_allowed": len(allowed), "config_load_refuses": True}

    def t4_dry_run_writes_nothing():
        # The tree is shared with other authorized agents, so the dry-run check
        # only asserts what belongs to this run: the output directory is not
        # created, the engine's own package is untouched, and the watched roots
        # gain no *engine* artifacts.  Concurrent scientific writes are recorded
        # separately by the launcher, not failed here.
        watched = [analysis_root / "code" / "rfinal_replay"]
        before = guards.snapshot_tree(watched)
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            completed = _run_cli(
                [
                    "--project-root",
                    str(project_root),
                    "--analysis-root",
                    str(analysis_root),
                    "--config",
                    str(config_path),
                    "--out",
                    str(out_dir),
                    "--dry-run",
                    "--verify-inputs",
                    "none",
                    "--json",
                ],
                cwd=tmpdir,
            )
            assert completed.returncode == 0, f"dry run failed: {completed.returncode}: {completed.stderr[-800:]}"
            payload = json.loads(completed.stdout)
            assert payload["plan"], "dry-run plan is empty"
            assert payload["output_check"]["exists"] is False, payload["output_check"]
            assert not out_dir.exists(), "dry run created the output directory"
            assert any("metrics" in row["id"] for row in payload["plan"]), "plan lacks window metrics stages"
        after = guards.snapshot_tree(watched)
        changes = guards.diff_snapshots(before, after, ignore_additions=False)
        assert changes == [], changes
        return {"plan_stages": len(payload["plan"]), "out_dir_created": False, "engine_package_unchanged": True}

    def t5_refuses_prepopulated_out():
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            out_dir.mkdir()
            (out_dir / "already.txt").write_text("x\n")
            completed = _run_cli(
                [
                    "--project-root",
                    str(project_root),
                    "--analysis-root",
                    str(analysis_root),
                    "--config",
                    str(config_path),
                    "--out",
                    str(out_dir),
                    "--dry-run",
                    "--verify-inputs",
                    "none",
                ],
                cwd=tmpdir,
            )
            assert completed.returncode == 2, f"expected refusal exit code 2, got {completed.returncode}"
            assert "REFUSED" in completed.stderr and "non-empty" in completed.stderr, completed.stderr[-500:]
            assert (out_dir / "already.txt").read_text() == "x\n", "refusal modified the prepopulated directory"
        return {"exit_code": 2, "message": "refused non-empty output directory"}

    def t6_manifest_keys():
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            config_file = tmpdir / "cfg.json"
            config_file.write_text(json.dumps(_minimal_config([])))
            built = manifest.build_manifest(
                run_id="fixture-run",
                mode="dry-run",
                project_root=tmpdir,
                analysis_root=tmpdir,
                config_path=config_file,
                config=_minimal_config([]),
                output_root=tmpdir / "out",
                empty_output_check={"exists": False, "empty": True, "pre_existing_empty": False},
                inputs=[{"id": "x", "sha256": "0" * 64}],
                code=[{"path": "fixture.py", "sha256": "0" * 64, "exists": True}],
                environment={"python": "fixture", "packages": {}},
                stages=[{"id": "s", "status": "SKIPPED"}],
                outputs=[],
                status="FIXTURE",
            )
            missing = [key for key in manifest.REQUIRED_MANIFEST_KEYS if key not in built]
            assert missing == [], missing
            digest = manifest.write_json_atomic(tmpdir / "deep" / "manifest.json", built)
            assert len(digest) == 64
            assert json.loads((tmpdir / "deep" / "manifest.json").read_text())["schema"] == manifest.MANIFEST_SCHEMA
        return {"required_keys": len(manifest.REQUIRED_MANIFEST_KEYS), "atomic_write": True}

    def t7_seven_nodes():
        config = guards.load_config(config_path)
        nodes = config["regeneration_plan"]["downstream_nodes"]
        ids = [node["result_id"] for node in nodes]
        assert sorted(ids) == sorted(NODES), ids
        for node in nodes:
            assert node.get("generating_step"), node
            assert node.get("exempt") is False, node
        return {"nodes": ids, "exempt": 0}

    def t8_tolerances_placeholder():
        config = guards.load_config(config_path)
        tolerance_values = config["tolerances"]
        placeholders = [
            key
            for key, value in tolerance_values.items()
            if isinstance(value, str) and value.startswith("TO_BE_FROZEN_BY_ROOT_BEFORE_R6")
        ]
        assert placeholders, "draft must leave the numeric tolerances for root"
        refused = False
        try:
            guards.assert_tolerances_frozen(config)
        except guards.ReplayRefusal:
            refused = True
        assert refused, "execution must refuse while tolerances are placeholders"
        refused = False
        try:
            guards.assert_authorized(config)
        except guards.ReplayRefusal:
            refused = True
        assert refused, "execution must refuse while the draft is not authorized"
        return {"placeholders": placeholders, "execution_gated": True}

    def t9_no_run_output_or_leftover():
        config = guards.load_config(config_path)
        run_dir = analysis_root / "runs" / config["root_contract"]["output_root_default"].split("/", 1)[-1]
        assert not run_dir.exists(), f"run output directory must not exist in R2: {run_dir}"
        leftover = {path.name for path in Path(tempfile.gettempdir()).glob(FIXTURE_PREFIX + "*")} - preexisting_temp
        assert leftover == set(), f"temporary fixture dirs left behind: {sorted(leftover)}"
        return {"run_output_absent": str(run_dir), "leftover_temp": 0}

    def t10_adapter_specs():
        config = guards.load_config(config_path)
        stages = [s for s in config["regeneration_plan"]["stages"] if s.get("kind") == "external_producer"]
        assert len(stages) == 18, f"expected 18 external producers, found {len(stages)}"
        for stage in stages:
            spec = adapters.adapter_spec(stage)
            assert spec["consumer_code_ref"], stage["id"]
            assert spec["outputs"], stage["id"]
            policy = dict(adapters.DEFAULT_RESOURCE_POLICY, **(spec.get("resource_policy") or {}))
            assert policy["max_workers"] <= 2, (stage["id"], policy)
            assert policy["blas_threads"] == 1, (stage["id"], policy)
            assert spec.get("coverage_expectation"), stage["id"]
            for output in spec["outputs"]:
                adapters._assert_relative_safe(output["path"], stage["id"])
            for entry_id in spec.get("input_refs", []):
                assert entry_id in {e["id"] for e in config["input_whitelist"]["entries"]}, (stage["id"], entry_id)
        covered = sorted(s["id"] for s in stages)
        return {"external_producers": len(stages), "first": covered[0], "last": covered[-1]}

    def t11_adapter_refusals():
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            good = tmpdir / "input.json"
            good.write_text('{"x": 1}\n')
            config = _minimal_config([])
            config["input_whitelist"]["entries"] = [
                {
                    "id": "L1:code/fixture.py",
                    "level": "L1_FROZEN_DERIVED_INPUT",
                    "root": "absolute",
                    "path": str(_fixture("fixture_ok_writer.py")),
                    "sandbox_path": "code/fixture.py",
                    "sha256": guards.sha256_file(_fixture("fixture_ok_writer.py")),
                    "role": "fixture code",
                },
                {
                    "id": "L1:input.json",
                    "level": "L1_FROZEN_DERIVED_INPUT",
                    "root": "absolute",
                    "path": str(good),
                    "sha256": guards.sha256_file(good),
                    "role": "fixture input",
                },
            ]
            config["regeneration_plan"]["stages"] = []
            stage = {
                "id": "S20_fixture",
                "kind": "external_producer",
                "executable": True,
                "adapter": {
                    "schema": adapters.ADAPTER_SCHEMA,
                    "consumer_code_ref": "L1:code/fixture.py",
                    "code_path": "code/fixture.py",
                    "input_refs": ["L1:input.json"],
                    "outputs": [{"path": "runs/fixture_ok/output.json", "role": "fixture output"}],
                    "coverage_expectation": "fixture",
                },
            }
            allow = guards.InputAllowlist(config, project_root, analysis_root)
            # (a) hash mismatch refused
            tampered = dict(config["input_whitelist"]["entries"][1])
            tampered["sha256"] = "0" * 64
            allow_bad = guards.InputAllowlist(
                {"input_whitelist": {"entries": [config["input_whitelist"]["entries"][0], tampered]}},
                project_root,
                analysis_root,
            )
            refused = False
            try:
                allow_bad.verify_many(["L1:input.json"])
            except guards.ReplayRefusal:
                refused = True
            assert refused, "hash mismatch not refused at adapter staging"
            # (b) missing fresh dependency refused (no fallback to old terminal values)
            stage_missing = {
                **stage,
                "adapter": {**stage["adapter"], "input_refs": [], "fresh_inputs": [{"stage": "S20_absent", "path": "runs/x/summary.json"}]},
            }
            refused = False
            try:
                adapters.run_external_stage(
                    stage=stage_missing,
                    out_root=tmpdir / "out_missing",
                    project_root=project_root,
                    analysis_root=analysis_root,
                    allow=allow,
                    fresh_registry={},
                    entries={e["id"]: e for e in config["input_whitelist"]["entries"]},
                    env_executable=Path(sys.executable),
                )
            except guards.ReplayRefusal as exc:
                refused = "fresh input" in str(exc) or "unavailable" in str(exc)
            assert refused, "missing fresh upstream input was not refused"
            # (c) unsafe declared output refused at validation
            bad_output_stage = {
                **stage,
                "adapter": {**stage["adapter"], "outputs": [{"path": "../escape.json", "role": "bad"}]},
            }
            refused = False
            try:
                adapters.adapter_spec(bad_output_stage)
            except guards.ReplayRefusal:
                refused = True
            assert refused, "sandbox-escaping output path was not refused"
        return {"hash_mismatch": True, "missing_fresh_dependency": True, "escaping_output": True}

    def _fixture(name: str) -> Path:
        return Path(__file__).resolve().parent / "tools" / "fixtures" / name

    def _run_fixture(name: str, tmpdir: Path, argv: list[str]) -> tuple[dict, Path]:
        sandbox = tmpdir / f"sandbox_{name}"
        (sandbox / "code").mkdir(parents=True)
        fixture_src = _fixture(name)
        shutil.copyfile(fixture_src, sandbox / "code" / fixture_src.name)
        log = tmpdir / f"{name}.access.jsonl"
        summary = tmpdir / f"{name}.summary.json"
        runner = Path(__file__).resolve().parent / "audit_runner.py"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(runner),
                "--sandbox",
                str(sandbox),
                "--target",
                f"code/{fixture_src.name}",
                "--mode",
                "script",
                "--argv",
                json.dumps(argv),
                "--access-log",
                str(log),
                "--summary",
                str(summary),
                "--max-workers",
                "2",
                "--blas-threads",
                "1",
                "--cap-pools",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        payload = json.loads(summary.read_text())
        payload["process"] = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        return payload, log

    def t12_audit_hook_enforcement():
        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            escape = tmpdir / "escape.txt"
            payload, log = _run_fixture("fixture_write_outside.py", tmpdir, [str(escape)])
            assert payload["status"] == "FAILED", payload
            assert payload["access_counts"]["refused"] >= 1, payload["access_counts"]
            assert not escape.exists(), "out-of-sandbox write reached the filesystem"
            refused_events = [json.loads(line) for line in log.read_text().splitlines() if "refused" in line]
            assert refused_events and refused_events[0]["event"] == "refused_write", refused_events[:2]
            outside = tmpdir / "outside_secret.txt"
            outside.write_text("secret\n")
            payload2, log2 = _run_fixture("fixture_read_outside.py", tmpdir, [str(outside)])
            assert payload2["status"] == "FAILED", payload2
            events2 = [json.loads(line) for line in log2.read_text().splitlines() if "refused" in line]
            assert events2 and events2[0]["event"] == "refused_read", events2[:2]
            payload3, _ = _run_fixture("fixture_ok_writer.py", tmpdir, ["runs/fixture_ok/output.json"])
            assert payload3["status"] == "COMPLETED", payload3
            return {
                "write_refused": True,
                "read_refused": True,
                "legal_fixture_completed": True,
                "access_counts": payload3["access_counts"],
            }

    def t13_forbidden_inputs_and_exceptions():
        config = guards.load_config(config_path)
        entries = config["input_whitelist"]["entries"]
        exceptions = config["forbidden_computation_input_patterns"].get("exceptions", [])
        assert exceptions, "C17 draft must declare its forbidden-pattern exceptions explicitly"
        for exception in exceptions:
            entry = next((e for e in entries if e["id"] == exception["id"]), None)
            assert entry is not None, exception
            assert guards.forbidden_pattern_for(entry["path"]), exception
            assert exception.get("approved_by") in (None, ""), "executor must not self-approve exceptions"
        pending = guards.unapproved_pattern_exceptions(config)
        refused = False
        try:
            guards.assert_pattern_exceptions_approved(config)
        except guards.ReplayRefusal:
            refused = True
        assert refused, "replay must refuse while a forbidden-pattern exception is unapproved"
        # a forbidden input without an exception still cannot be whitelisted at load
        bad = _minimal_config(
            [
                {
                    "id": "L2:runs/window_smol_new_v1/metrics.json",
                    "level": "L2_ACCEPTED_WINDOW_CACHE",
                    "root": "analysis",
                    "path": "runs/window_smol_new_v1/metrics.json",
                    "sha256": "0" * 64,
                    "role": "forbidden",
                }
            ]
        )
        refused = False
        try:
            guards.validate_config(bad)
        except guards.ReplayRefusal:
            refused = True
        assert refused, "metrics.json as computation input was not refused"
        return {"exceptions": len(exceptions), "replay_gated": True, "plain_forbidden_refused": True}

    def t14_registration_interface():
        template = registry.registration_template("C2")
        assert template["schema"] == registry.REGISTRATION_SCHEMA
        placeholders = registry.validate_registration(template)
        assert placeholders, "unfilled template must be refused"
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            script = tmpdir / "producer.py"
            script.write_text("print('producer')\n")
            data = tmpdir / "input.json"
            data.write_text("{}\n")
            registration = {
                "schema": registry.REGISTRATION_SCHEMA,
                "node": "C2",
                "stage_id": "S30_C2_new28_impl",
                "title": "new-28 implementation replay",
                "result_ids": ["C17-NEW28-*"],
                "producer": {
                    "code_root": str(tmpdir),
                    "entrypoint": str(script),
                    "sha256": guards.sha256_file(script),
                    "extra_code": [],
                    "argv": [],
                    "execution_mode": "script",
                },
                "inputs": [
                    {
                        "id": "L1:fixture_input.json",
                        "root": "absolute",
                        "path": str(data),
                        "sha256": guards.sha256_file(data),
                        "level": "L1_FROZEN_DERIVED_INPUT",
                        "role": "fixture",
                    }
                ],
                "fresh_inputs": [],
                "transforms": [],
                "outputs": [{"path": "runs/fixture/output.json", "role": "fixture output", "result_ids": []}],
                "collect_to": "branches/S30_C2_new28_impl",
                "timeout_seconds": 60,
                "resource_policy": {"max_workers": 2, "blas_threads": 1, "cap_pools": True},
                "engineering_checks": {
                    "planned_level": "STAGED_IMPORT_LEVEL",
                    "fixture": None,
                    "refusals": ["non_empty_output", "hash_mismatch"],
                    "integrates_with_full_chain": True,
                },
                "evidence": {"status": "NOT_RUN", "signed_by": None},
                "notes": "fixture registration",
            }
            problems = registry.validate_registration(registration)
            assert problems == [], problems
            self_signed = json.loads(json.dumps(registration))
            self_signed["evidence"] = {"status": "NOT_RUN", "signed_by": "codex_1"}
            assert registry.validate_registration(self_signed), "self-signing must be refused"
            config = guards.load_config(config_path)
            updated = registry.plan_registration(config, registration)
            stage = updated["regeneration_plan"]["stages"][-1]
            assert stage["id"] == registration["stage_id"], stage
            assert stage["executable"] is False, "a fresh registration is a declaration, not an executable pass"
        return {"template_refused_until_filled": True, "self_signing_refused": True, "declaration_not_executable": True}

    def t15_transform_path_boundary():
        from rfinal_replay import transform as transform_mod

        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            sandbox = tmpdir / "sandbox"
            (sandbox / "code").mkdir(parents=True)
            neighbor = tmpdir / "sandbox_neighbor"
            (neighbor / "code").mkdir(parents=True)
            inside = sandbox / "code" / "target.py"
            inside.write_text("P = '/original/root'\n")
            outside = neighbor / "code" / "target.py"
            outside.write_text("P = '/original/root'\n")
            outside_before = guards.sha256_file(outside)
            escaped = False
            try:
                transform_mod.text_prefix_rewrite(
                    sandbox=sandbox,
                    target="../sandbox_neighbor/code/target.py",
                    from_prefix="/original/root",
                    to_prefix=str(sandbox),
                    label="boundary fixture",
                )
            except guards.ReplayRefusal:
                escaped = True
            assert escaped, "sibling-prefix path was not refused"
            assert guards.sha256_file(outside) == outside_before, "sibling file was modified before refusal"
            assert outside.read_text() == "P = '/original/root'\n", "sibling bytes changed"
            # a genuine in-sandbox target still works
            receipt = transform_mod.text_prefix_rewrite(
                sandbox=sandbox,
                target="code/target.py",
                from_prefix="/original/root",
                to_prefix="/sandbox/root",
                label="in-sandbox fixture",
            )
            assert receipt["replacements"] == 1
            return {"sibling_refused_before_write": True, "sibling_bytes_unchanged": True}

    def t16_hardlink_transform_private_inode():
        from rfinal_replay import transform as transform_mod

        with tempfile.TemporaryDirectory(prefix=FIXTURE_PREFIX) as tmp:
            tmpdir = Path(tmp)
            sandbox = tmpdir / "sandbox"
            (sandbox / "config").mkdir(parents=True)
            (sandbox / "data.json").write_text("{}\n")
            original = tmpdir / "signed_config.json"
            declared_hash = guards.sha256_file(sandbox / "data.json")
            original.write_text('{"path": "/original/root/data.json", "sha256": "' + declared_hash + '"}\n')
            before = guards.sha256_file(original)
            linked = sandbox / "config" / "signed_config.json"
            os.link(original, linked)
            assert linked.stat().st_nlink == 2
            receipt = transform_mod.json_root_migration(
                sandbox=sandbox,
                target="config/signed_config.json",
                from_prefixes=["/original/root"],
                to_prefix=str(sandbox),
                fresh_paths={},
                label="hardlink fixture",
            )
            assert receipt["private_inode"]["strategy"] == "copied_before_write", receipt["private_inode"]
            assert guards.sha256_file(original) == before, "hardlinked original changed"
            assert linked.stat().st_nlink == 1, "migration target still shares its inode"
            data = json.loads(linked.read_text())
            assert data["path"] == str(sandbox) + "/data.json", data
            return {
                "original_bytes_unchanged": True,
                "private_inode": True,
                "migration_applied_to_copy": True,
            }

    check("T1_output_directory_assertion", t1_output_assertion)
    check("T2_allowlist_refusal", t2_allowlist_refusal)
    check("T3_forbidden_pattern_enforcement", t3_forbidden_patterns)
    check("T4_dry_run_prints_plan_writes_nothing", t4_dry_run_writes_nothing)
    check("T5_refuse_prepopulated_output_dir", t5_refuses_prepopulated_out)
    check("T6_manifest_required_keys", t6_manifest_keys)
    check("T7_seven_downstream_nodes_named", t7_seven_nodes)
    check("T8_tolerances_left_for_root_and_execution_gated", t8_tolerances_placeholder)
    check("T9_no_run_output_no_leftover_temp", t9_no_run_output_or_leftover)
    check("T10_external_producer_adapter_specs", t10_adapter_specs)
    check("T11_adapter_staging_refusals", t11_adapter_refusals)
    check("T12_audit_hook_sandbox_enforcement", t12_audit_hook_enforcement)
    check("T13_forbidden_inputs_and_exceptions", t13_forbidden_inputs_and_exceptions)
    check("T14_registration_interface", t14_registration_interface)
    check("T15_transform_path_boundary", t15_transform_path_boundary)
    check("T16_hardlink_transform_private_inode", t16_hardlink_transform_private_inode)

    summary = {
        "schema": "paper2-rfinal-replay-selftest-v1",
        "producer": "codex_1 CLI:paper2:R2",
        "created_utc": manifest.utc_now(),
        "mode": "EXECUTOR_SELF_CHECK_NOT_INDEPENDENT_VERIFICATION",
        "no_analysis": True,
        "no_gpu": True,
        "project_root": str(project_root),
        "analysis_root": str(analysis_root),
        "config": str(config_path),
        "checks": checks,
        "n_checks": len(checks),
        "n_pass": sum(1 for row in checks if row["pass"]),
        "all_pass": all(row["pass"] for row in checks),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
