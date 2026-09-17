#!/usr/bin/env python3
"""Single entrypoint for the C17 reproduction program (paper2).

    python3 code/rfinal_replay/entrypoint.py plan      --json
    python3 code/rfinal_replay/entrypoint.py audit     [--level lazy|full]
    python3 code/rfinal_replay/entrypoint.py run       --out DIR [--mode replay|engine-check] ...
    python3 code/rfinal_replay/entrypoint.py env       build|verify
    python3 code/rfinal_replay/entrypoint.py coverage  --run DIR
    python3 code/rfinal_replay/entrypoint.py register  --registration FILE [--emit]

All commands resolve every path from the explicit ``--project-root`` /
``--analysis-root`` arguments or their documented defaults; ``__file__`` is
never used to resolve an output path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rfinal_replay import adapters, environment, guards, manifest, registry, stages  # type: ignore
    from rfinal_replay import launcher  # type: ignore
else:  # pragma: no cover
    from . import adapters, environment, guards, launcher, manifest, registry, stages

DEFAULT_PROJECT = Path.cwd()
DEFAULT_CONFIG_NAME = "CT_C17_REPLAY_FINAL_v1.relocated.json"
LEGACY_CONFIG_NAME = "RFINAL_REPLAY_DRAFT.json"


def _roots(argv: list[str]) -> tuple[Path, Path]:
    project = DEFAULT_PROJECT
    analysis = project / "repro_paper2_20260915"
    if "--project-root" in argv:
        project = Path(argv[argv.index("--project-root") + 1]).resolve()
    if "--analysis-root" in argv:
        analysis = Path(argv[argv.index("--analysis-root") + 1]).resolve()
    if "--analysis-root" not in argv:
        analysis = project / "repro_paper2_20260915"
    return project, analysis


def _config_path(analysis: Path, argv: list[str]) -> Path:
    if "--config" in argv:
        return Path(argv[argv.index("--config") + 1]).resolve()
    preferred = analysis / "config" / DEFAULT_CONFIG_NAME
    if preferred.is_file():
        return preferred
    return analysis / "config" / LEGACY_CONFIG_NAME


def cmd_plan(argv: list[str]) -> int:
    project, analysis = _roots(argv)
    config_path = _config_path(analysis, argv)
    config = guards.load_config(config_path)
    out_root = (analysis / config["root_contract"]["output_root_default"]).resolve()
    plan = stages.build_plan(config, project, analysis, out_root)
    allow = guards.InputAllowlist(config, project, analysis)
    payload = {
        "mode": "plan",
        "config": str(config_path),
        "config_schema": config["schema"],
        "status": config["status"],
        "authorization": config["execution_authorization"],
        "tolerances": config["tolerances"],
        "whitelist_counts_by_level": allow.counts_by_level(),
        "plan_summary": stages.plan_summary(plan),
        "plan": plan,
        "wrote": [],
        "note": "plan is read-only; no directory was created",
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_audit(argv: list[str]) -> int:
    project, analysis = _roots(argv)
    config_path = _config_path(analysis, argv)
    config = guards.load_config(config_path)
    allow = guards.InputAllowlist(config, project, analysis)
    level = argv[argv.index("--level") + 1] if "--level" in argv else "lazy"
    verified: dict[str, str] = {}
    skipped: list[dict] = []
    for entry in allow.entries.values():
        path = allow.resolve(entry)
        if level == "lazy" and not entry.get("consumed_by"):
            skipped.append({"id": entry["id"], "reason": "not consumed by any declared stage"})
            continue
        size = path.stat().st_size if path.is_file() else 0
        if entry.get("hash_source") and size > guards.MAX_INLINE_HASH_BYTES:
            skipped.append({"id": entry["id"], "reason": "pinned-manifest digest (large)", "bytes": size})
            continue
        verified[entry["id"]] = allow.verify(entry, force=True)
    print(
        json.dumps(
            {
                "mode": "audit",
                "config": str(config_path),
                "level": level,
                "n_verified": len(verified),
                "n_skipped": len(skipped),
                "skipped": skipped,
                "verified_sha256": verified,
            },
            indent=2,
        )
    )
    return 0


def cmd_run(argv: list[str]) -> int:
    forwarded = list(argv)
    if "--config" not in forwarded:
        _, analysis = _roots(argv)
        forwarded += ["--config", str(_config_path(analysis, argv))]
    return launcher.main(forwarded)


def cmd_env(argv: list[str]) -> int:
    project, analysis = _roots(argv)
    sub = argv[0] if argv else "verify"
    config_path = _config_path(analysis, argv)
    pins = None
    if config_path.is_file():
        config = guards.load_config(config_path)
        pins = config["environment_lock"].get("cpu_replay", {}).get("pins")
    if sub == "build":
        base = Path(argv[argv.index("--base-python") + 1]) if "--base-python" in argv else Path(
            sys.executable
        )
        result = environment.build_cpu_replay_env(analysis_root=analysis, base_python=base, pins=pins)
        print(json.dumps({"status": result["lock"]["status"], "venv": result["lock"]["venv"],
                          "lock": result["receipt"]["lock"], "problems": result["lock"]["problems"]}, indent=2))
        return 0 if result["lock"]["status"] == "BUILT_AND_VERIFIED" else 1
    if sub == "verify":
        result = environment.verify_cpu_replay_env(analysis_root=analysis, pins=pins)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    raise guards.ReplayRefusal(f"unknown env subcommand: {sub!r} (build|verify)")


def cmd_coverage(argv: list[str]) -> int:
    _, analysis = _roots(argv)
    run_dir = Path(argv[argv.index("--run") + 1]).resolve() if "--run" in argv else None
    if run_dir is None:
        raise guards.ReplayRefusal("--run DIR is required")
    rows = []
    receipts_dir = run_dir / "receipts"
    if receipts_dir.is_dir():
        for path in sorted(receipts_dir.glob("*.json")):
            receipt = json.loads(path.read_text())
            rows.append(
                {
                    "stage": receipt.get("stage"),
                    "status": receipt.get("status"),
                    "exit_code": receipt.get("exit_code"),
                    "outputs": [item["path"] for item in receipt.get("outputs", [])],
                    "refused_accesses": len((receipt.get("access_log") or {}).get("refused", [])),
                    "resource_policy": receipt.get("resource_policy"),
                }
            )
    manifest_path = run_dir / "run_manifest.json"
    summary = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    print(
        json.dumps(
            {
                "run": str(run_dir),
                "run_status": summary.get("status"),
                "execution_mode": summary.get("execution_mode"),
                "stages_recorded": len(rows),
                "stages": rows,
            },
            indent=2,
        )
    )
    return 0


def cmd_register(argv: list[str]) -> int:
    _, analysis = _roots(argv)
    if "--registration" not in argv:
        raise guards.ReplayRefusal("--registration FILE is required")
    reg_path = Path(argv[argv.index("--registration") + 1]).resolve()
    registration = json.loads(reg_path.read_text())
    problems = registry.validate_registration(registration)
    payload = {
        "mode": "register-check",
        "registration": str(reg_path),
        "schema": registration.get("schema"),
        "node": registration.get("node"),
        "problems": problems,
        "status": "VALID_DECLARATION_PENDING_CHECKS" if not problems else "REFUSED",
        "note": "registration never signs; the engine records engineering coverage and an independent reviewer accepts the node",
    }
    if "--emit" in argv and not problems:
        config_path = _config_path(analysis, argv)
        config = guards.load_config(config_path)
        payload["planned_stage"] = registry.plan_registration(config, registration)["regeneration_plan"]["stages"][-1]
    print(json.dumps(payload, indent=2, default=str))
    return 0 if not problems else 2


COMMANDS = {
    "plan": cmd_plan,
    "audit": cmd_audit,
    "run": cmd_run,
    "env": cmd_env,
    "coverage": cmd_coverage,
    "register": cmd_register,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command = argv[0]
    if command == "selftest":
        from . import selftest  # pragma: no cover

        raise SystemExit(selftest.main(argv[1:]))
    if command not in COMMANDS:
        raise guards.ReplayRefusal(f"unknown command {command!r}; expected one of {sorted(COMMANDS)} or selftest")
    return COMMANDS[command](argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except guards.ReplayRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        raise SystemExit(2)
