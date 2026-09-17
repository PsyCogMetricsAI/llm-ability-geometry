"""Producer registration interface for C2/C4/C6/C7 (and later nodes).

The C17 plan expects four later scientific nodes to register their producers
with this replay engine:

    C2  new-28 implementation        -> R/code/new28_rebuild_v1/
    C4  panel calibration             -> R/code/c17_panel_calibration/
    C6  geometry validation           -> R/code/c17_geometry_validation/
    C7  window evidence checks        -> R/code/c17_window_checks/

Registration is a *declaration*, not a pass: a registered producer stays
``PENDING`` until the engine has actually run its declared engineering checks
inside a fresh sandbox and an independent reviewer has accepted the scientific
node.  The validator refuses placeholders, refuses any attempt to mark a
registration verified from inside the executor, and refuses registrations whose
inputs are not hash-bound.
"""

from __future__ import annotations

import copy
import json

from . import adapters, guards

REGISTRATION_SCHEMA = "rfinal-producer-registration-v1"
REGISTRATION_NODES = ("C2", "C4", "C6", "C7")
COVERAGE_LEVELS = (
    "NONE",
    "STAGED_IMPORT_LEVEL",
    "CORE_FIXTURE_EXECUTED",
    "PILOT_EXECUTED",
    "FULL_EXECUTED",
)
EVIDENCE_STATUSES = ("NOT_RUN", "ENGINEERING_CHECKS_RECORDED_PENDING_REVIEW")

TEMPLATE = {
    "schema": REGISTRATION_SCHEMA,
    "node": "<C2|C4|C6|C7>",
    "stage_id": "S30_<node>_<name>",
    "title": "<what this producer regenerates>",
    "result_ids": ["<retained result ids>"],
    "producer": {
        "code_root": "<analysis-root-relative directory>",
        "entrypoint": "<analysis-root-relative script>",
        "sha256": "<64 hex chars>",
        "extra_code": [{"path": "<analysis-root-relative helper>", "sha256": "<64 hex chars>"}],
        "argv": [],
        "execution_mode": "script",
    },
    "inputs": [
        {
            "id": "<whitelist id or new id>",
            "root": "analysis",
            "path": "<relative path>",
            "sha256": "<64 hex chars>",
            "level": "<L0_ORIGINAL_OBSERVATION|L1_FROZEN_DERIVED_INPUT|L2_ACCEPTED_WINDOW_CACHE>",
            "role": "<why it is a computation input>",
        }
    ],
    "fresh_inputs": [{"stage": "<earlier stage id>", "path": "<sandbox-relative path>"}],
    "transforms": [],
    "outputs": [{"path": "<sandbox-relative output>", "role": "<role>", "result_ids": []}],
    "collect_to": "branches/<stage id>",
    "timeout_seconds": 3600,
    "resource_policy": {"max_workers": 2, "blas_threads": 1, "cap_pools": True},
    "engineering_checks": {
        "planned_level": "<NONE|STAGED_IMPORT_LEVEL|CORE_FIXTURE_EXECUTED|PILOT_EXECUTED|FULL_EXECUTED>",
        "fixture": "<analysis-root-relative fixture or null>",
        "refusals": ["non_empty_output", "hash_mismatch", "forbidden_input", "out_of_bounds_output", "dependency_failure"],
        "integrates_with_full_chain": True,
    },
    "evidence": {"status": "NOT_RUN", "signed_by": None},
    "notes": "<known limits of this registration>",
}


def validate_registration(registration: dict) -> list[str]:
    problems: list[str] = []
    if registration.get("schema") != REGISTRATION_SCHEMA:
        problems.append(f"schema must be {REGISTRATION_SCHEMA!r}")
    if registration.get("node") not in REGISTRATION_NODES:
        problems.append(f"node must be one of {REGISTRATION_NODES}")
    stage_id = str(registration.get("stage_id", ""))
    if not stage_id.strip():
        problems.append("stage_id is required")
    producer = registration.get("producer") or {}
    for key in ("code_root", "entrypoint", "sha256"):
        if not producer.get(key):
            problems.append(f"producer.{key} is required")
    if not guards.SHA256_RE.match(str(producer.get("sha256", ""))):
        problems.append("producer.sha256 must be 64 lowercase hex chars")
    for helper in producer.get("extra_code", []) or []:
        if not guards.SHA256_RE.match(str(helper.get("sha256", ""))):
            problems.append(f"extra_code sha256 invalid: {helper.get('path')}")
    level = (registration.get("inputs") or [{}])[0].get("level")
    if not registration.get("inputs"):
        problems.append("at least one hash-bound input is required")
    for item in registration.get("inputs", []) or []:
        if item.get("level") not in guards.ALLOWED_LEVELS:
            problems.append(f"input level invalid: {item.get('id')}")
        if not guards.SHA256_RE.match(str(item.get("sha256", ""))):
            problems.append(f"input sha256 invalid: {item.get('id')}")
        hit = guards.forbidden_pattern_for(str(item.get("path", "")))
        if hit:
            problems.append(f"input matches forbidden computation-input pattern {hit!r}: {item.get('path')}")
    if not registration.get("outputs"):
        problems.append("at least one declared output is required")
    for output in registration.get("outputs", []) or []:
        try:
            adapters._assert_relative_safe(str(output.get("path", "")), f"registration {stage_id} output")
        except guards.ReplayRefusal as exc:
            problems.append(str(exc))
    checks = registration.get("engineering_checks") or {}
    if checks.get("planned_level") not in COVERAGE_LEVELS:
        problems.append(f"engineering_checks.planned_level must be one of {COVERAGE_LEVELS}")
    if not checks.get("refusals"):
        problems.append("engineering_checks.refusals must list the refusal behaviours to exercise")
    evidence = registration.get("evidence") or {}
    if evidence.get("status") not in EVIDENCE_STATUSES:
        problems.append(f"evidence.status must be one of {EVIDENCE_STATUSES}: the executor never self-signs")
    if evidence.get("signed_by") not in (None, ""):
        problems.append("evidence.signed_by must stay empty until an independent reviewer signs")
    placeholder_markers = ("<", ">", "TODO", "TBD", "placeholder")
    blob = str(registration)
    for marker in placeholder_markers:
        if marker in blob:
            problems.append(f"registration still contains placeholder marker {marker!r}")
    if stage_id and not stage_id.startswith("S30_"):
        problems.append("replay-engine registrations use the S30_ namespace so they cannot collide with frozen stages")
    return problems


def registration_template(node: str) -> dict:
    if node not in REGISTRATION_NODES:
        raise guards.ReplayRefusal(f"unknown registration node: {node!r}")
    template = copy.deepcopy(TEMPLATE)
    template["node"] = node
    template["stage_id"] = f"S30_{node}_<name>"
    return template


def plan_registration(config: dict, registration: dict) -> dict:
    """Validate a registration and fold it into a copy of the replay config."""
    problems = validate_registration(registration)
    if problems:
        raise guards.ReplayRefusal("registration refused: " + "; ".join(problems))
    updated = json.loads(json.dumps(config))
    ids = {entry["id"] for entry in updated["input_whitelist"]["entries"]}
    for item in registration.get("inputs", []):
        if item["id"] not in ids:
            entry = {key: value for key, value in item.items() if key != "consumed_by"}
            entry.setdefault("root", "analysis")
            entry["consumed_by"] = [registration["stage_id"]]
            updated["input_whitelist"]["entries"].append(entry)
            ids.add(item["id"])
    stage_ids = {stage["id"] for stage in updated["regeneration_plan"]["stages"]}
    if registration["stage_id"] in stage_ids:
        raise guards.ReplayRefusal(f"registration stage id already exists: {registration['stage_id']}")
    producer = registration["producer"]
    code_path = producer["entrypoint"]
    code_id = f"L1:{code_path}"
    if code_id not in ids:
        updated["input_whitelist"]["entries"].append(
            {
                "id": code_id,
                "level": "L1_FROZEN_DERIVED_INPUT",
                "root": "analysis",
                "path": code_path,
                "sha256": producer["sha256"],
                "role": f"registered producer code for {registration['stage_id']}",
                "consumed_by": [registration["stage_id"]],
            }
        )
        ids.add(code_id)
    extra_refs = []
    for helper in producer.get("extra_code", []) or []:
        helper_id = f"L1:{helper['path']}"
        if helper_id not in ids:
            updated["input_whitelist"]["entries"].append(
                {
                    "id": helper_id,
                    "level": "L1_FROZEN_DERIVED_INPUT",
                    "root": "analysis",
                    "path": helper["path"],
                    "sha256": helper["sha256"],
                    "role": f"registered producer helper for {registration['stage_id']}",
                    "consumed_by": [registration["stage_id"]],
                }
            )
            ids.add(helper_id)
        extra_refs.append(helper_id)
    stage = {
        "id": registration["stage_id"],
        "kind": "external_producer",
        "title": registration["title"],
        "node": registration["node"],
        "result_ids": registration.get("result_ids", []),
        "status": "REGISTERED_PENDING_ENGINEERING_CHECKS",
        "executable": False,
        "blocked_reason": "registered by a later node; executable once the adapter spec is generated and checks pass",
        "adapter": {
            "schema": adapters.ADAPTER_SCHEMA,
            "consumer_code_ref": code_id,
            "code_path": code_path,
            "extra_code_refs": extra_refs,
            "argv": producer.get("argv", []),
            "execution_mode": producer.get("execution_mode", "script"),
            "input_refs": [item["id"] for item in registration.get("inputs", [])],
            "fresh_inputs": registration.get("fresh_inputs", []),
            "transforms": registration.get("transforms", []),
            "outputs": registration.get("outputs", []),
            "collect_to": registration.get("collect_to", f"branches/{registration['stage_id']}"),
            "timeout_seconds": registration.get("timeout_seconds", 3600),
            "resource_policy": registration.get("resource_policy", dict(adapters.DEFAULT_RESOURCE_POLICY)),
        },
        "registration_checks": registration.get("engineering_checks", {}),
    }
    updated["regeneration_plan"]["stages"].append(stage)
    updated["regeneration_plan"]["registered_pending"] = updated["regeneration_plan"].get("registered_pending", 0) + 1
    return updated
