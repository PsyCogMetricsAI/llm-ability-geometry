"""Path, allowlist and forbidden-input guards for the RFINAL clean replay.

Design rules (see config/RFINAL_REPLAY_DRAFT.json for the same rules in config form):

* Every input and output path is resolved from the two explicit CLI roots
  (``--project-root``, ``--analysis-root``). ``__file__`` is never used to resolve
  an output path, so a copy of this package cannot write into the signed tree.
* A run starts only into an output directory that does not exist or is empty.
  There is no in-place resume: a crashed run is re-run into a new empty directory.
* Only inputs listed in the config whitelist (path + sha256 + level + role) may be
  read as computation input. Everything else is refused.
* Files whose name matches the forbidden patterns (``*_final*``, ``*_summary*``,
  ``metrics.json``, ``report.json``) are never computation input, even if someone
  later adds them to a whitelist: the config load refuses such a config.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath


class ReplayRefusal(RuntimeError):
    """Raised whenever a guard refuses to start or refuses an input."""


# Forbidden computation-input patterns. These are printed and enforced by the
# launcher and re-listed in config/RFINAL_REPLAY_DRAFT.json. Matching is
# case-insensitive and applies to the basename and to every path suffix, so
# ``runs/<run>/metrics.json`` and ``reports/EV2_..._SUMMARY_v1.json`` are refused.
FORBIDDEN_INPUT_BASENAME_PATTERNS = ("*_final*", "*_summary*", "metrics.json", "report.json")
FORBIDDEN_INPUT_PATH_PATTERNS = ("*_final*", "*_summary*", "metrics.json", "report.json")

ALLOWED_LEVELS = (
    "L0_ORIGINAL_OBSERVATION",
    "L1_FROZEN_DERIVED_INPUT",
    "L2_ACCEPTED_WINDOW_CACHE",
    "L3_PINNED_MODEL_SNAPSHOT",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_INLINE_HASH_BYTES = 1 << 30  # 1 GiB: larger entries must carry a pinned-manifest digest


def sha256_file(path: os.PathLike[str] | str, chunk: int = 1 << 23) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _posix(path: os.PathLike[str] | str) -> str:
    return PurePosixPath(str(path).replace("\\", "/")).as_posix()


def forbidden_pattern_for(path: os.PathLike[str] | str) -> str | None:
    """Return the first forbidden pattern matching ``path``, else ``None``."""
    text = _posix(path).lower()
    candidates = [text]
    parts = [part for part in text.split("/") if part]
    candidates.extend(parts)
    candidates.extend("/".join(parts[index:]) for index in range(len(parts)))
    for candidate in candidates:
        for pattern in FORBIDDEN_INPUT_PATH_PATTERNS:
            if fnmatch.fnmatchcase(candidate, pattern):
                return pattern
    return None


def is_forbidden_input_path(path: os.PathLike[str] | str) -> bool:
    return forbidden_pattern_for(path) is not None


def resolve_root(value: str, name: str) -> Path:
    if not value:
        raise ReplayRefusal(f"missing --{name}")
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if not root.is_dir():
        raise ReplayRefusal(f"--{name} is not an existing directory: {root}")
    return root


def load_config(path: os.PathLike[str] | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ReplayRefusal(f"config not found: {path}")
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ReplayRefusal(f"config is not valid JSON: {path}: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate the structural contract that the launcher relies on."""
    problems: list[str] = []
    schema = config.get("schema")
    if schema not in ("paper2-rfinal-replay-draft-v1", "paper2-rfinal-replay-draft-v2"):
        problems.append(f"unexpected schema: {config.get('schema')!r}")
    for key in (
        "root_contract",
        "execution_authorization",
        "empty_output_assertion",
        "forbidden_computation_input_patterns",
        "input_whitelist",
        "environment_lock",
        "seeds",
        "tolerances",
        "failure_rules",
        "comparison_rules",
        "regeneration_plan",
    ):
        if key not in config:
            problems.append(f"missing config section: {key}")
    if problems:
        raise ReplayRefusal("config validation failed: " + "; ".join(problems))

    listed = tuple(config["forbidden_computation_input_patterns"].get("path_patterns", ()))
    if set(listed) != set(FORBIDDEN_INPUT_PATH_PATTERNS):
        problems.append(
            "config forbidden path patterns must equal the patterns enforced in code: "
            f"{sorted(FORBIDDEN_INPUT_PATH_PATTERNS)} != {sorted(listed)}"
        )

    entries = config["input_whitelist"].get("entries", [])
    seen_ids: set[str] = set()
    for entry in entries:
        entry_id = entry.get("id")
        if not entry_id or entry_id in seen_ids:
            problems.append(f"duplicate or missing whitelist id: {entry_id!r}")
        seen_ids.add(str(entry_id))
        level = entry.get("level")
        if level not in ALLOWED_LEVELS:
            problems.append(f"{entry_id}: unknown level {level!r}")
        digest = entry.get("sha256", "")
        if not SHA256_RE.match(str(digest)):
            problems.append(f"{entry_id}: sha256 must be 64 lowercase hex chars")
        if entry.get("root") not in ("project", "analysis", "absolute"):
            problems.append(f"{entry_id}: root must be project|analysis|absolute")
        path = str(entry.get("path", ""))
        if level == "L3_PINNED_MODEL_SNAPSHOT" and entry.get("usable_by_default"):
            problems.append(f"{entry_id}: pinned weights are never used by the default cache replay")
        if _posix(path).startswith("runs/window_") and level != "L2_ACCEPTED_WINDOW_CACHE":
            problems.append(
                f"{entry_id}: accepted window caches (runs/window_*) must be declared at L2_ACCEPTED_WINDOW_CACHE"
            )

    # Declared exceptions to the forbidden computation-input patterns.  An
    # exception is data, not an approval: replay refuses while ``approved_by``
    # is empty, and engine-check runs record every unapproved exception.
    exception_ids: set[str] = set()
    exceptions = config["forbidden_computation_input_patterns"].get("exceptions", [])
    entry_by_id = {entry.get("id"): entry for entry in entries}
    for exception in exceptions:
        entry_id = exception.get("id")
        if entry_id not in entry_by_id:
            problems.append(f"pattern exception for unknown whitelist id: {entry_id!r}")
            continue
        entry = entry_by_id[entry_id]
        hit = forbidden_pattern_for(entry["path"])
        if hit is None:
            problems.append(f"pattern exception {entry_id} does not actually match a forbidden pattern")
            continue
        if exception.get("pattern") != hit:
            problems.append(
                f"pattern exception {entry_id} declares pattern {exception.get('pattern')!r} "
                f"but the path matches {hit!r}"
            )
        if not exception.get("reason"):
            problems.append(f"pattern exception {entry_id} must state a reason")
        if not entry.get("consumed_by"):
            problems.append(f"pattern exception {entry_id} is not consumed by any declared stage")
        exception_ids.add(entry_id)
    for entry in entries:
        entry_id = entry.get("id")
        if entry_id in exception_ids:
            continue
        hit = forbidden_pattern_for(entry["path"])
        if hit:
            problems.append(f"{entry_id}: whitelist entry matches forbidden pattern {hit!r}: {entry['path']}")

    for stage in config["regeneration_plan"].get("stages", []):
        if not stage.get("id") or not stage.get("kind"):
            problems.append(f"stage without id/kind: {stage!r}")
        for entry_id in stage.get("input_refs", []):
            if entry_id not in seen_ids:
                problems.append(f"stage {stage.get('id')}: unknown input_ref {entry_id}")
        if stage.get("kind") == "external_producer":
            adapter = stage.get("adapter")
            if not isinstance(adapter, dict):
                if stage.get("executable"):
                    problems.append(
                        f"stage {stage.get('id')}: executable external_producer must carry a full adapter spec "
                        "(a registered stage without an adapter is PENDING, not executable)"
                    )
                continue
            try:
                from . import adapters as adapters_module

                adapters_module.adapter_spec(stage)
            except ReplayRefusal as exc:
                problems.append(str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                problems.append(f"stage {stage.get('id')}: adapter spec could not be validated: {exc}")
            if stage.get("executable") and not adapter.get("engineering_checks_recorded"):
                if not adapter.get("coverage_expectation"):
                    problems.append(
                        f"stage {stage.get('id')}: executable external producer must declare "
                        "adapter.coverage_expectation (how its engineering coverage is verified)"
                    )
    nodes = config["regeneration_plan"].get("downstream_nodes", [])
    if len(nodes) != 7:
        problems.append(f"regeneration plan must name all seven downstream nodes, found {len(nodes)}")
    for node in nodes:
        if node.get("exempt"):
            problems.append(f"downstream node {node.get('result_id')} must not be exempt")
        if not node.get("generating_step"):
            problems.append(f"downstream node {node.get('result_id')} lacks a named generating step")

    if problems:
        raise ReplayRefusal("config validation failed: " + "; ".join(problems))


def assert_tolerances_frozen(config: dict) -> None:
    """Execution refuses while root has not frozen the numeric tolerances."""
    tolerances = config["tolerances"]
    placeholders = {
        key: value
        for key, value in tolerances.items()
        if isinstance(value, str) and value.startswith("TO_BE_FROZEN_BY_ROOT_BEFORE_R6")
    }
    if placeholders:
        raise ReplayRefusal(
            "numeric tolerances are not frozen by root yet (R2 draft): "
            + ", ".join(sorted(placeholders))
        )


def unapproved_pattern_exceptions(config: dict) -> list[dict]:
    return [
        exception
        for exception in config["forbidden_computation_input_patterns"].get("exceptions", [])
        if not exception.get("approved_by")
    ]


def assert_pattern_exceptions_approved(config: dict) -> None:
    """Frozen replay refuses while a forbidden-pattern exception lacks root approval."""
    pending = unapproved_pattern_exceptions(config)
    if pending:
        raise ReplayRefusal(
            "forbidden-pattern exceptions await root approval: "
            + "; ".join(f"{row['id']} ({row['pattern']}): {row['reason']}" for row in pending)
        )


def assert_authorized(config: dict) -> None:
    auth = config["execution_authorization"]
    if auth.get("status") != auth.get("required_status_for_run"):
        raise ReplayRefusal(
            "execution is not authorized for this draft: "
            f"{auth.get('status')!r} != {auth.get('required_status_for_run')!r}"
        )


def assert_empty_output_dir(out_dir: Path) -> dict:
    """Refuse to start when the target output directory exists and is non-empty."""
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ReplayRefusal(f"output path exists and is not a directory: {out_dir}")
        entries = sorted(child.name for child in out_dir.iterdir())
        if entries:
            preview = ", ".join(entries[:5])
            raise ReplayRefusal(
                f"output directory exists and is non-empty ({len(entries)} entries: {preview}); "
                "in-place resume is not supported, choose a new empty directory"
            )
        return {"exists": True, "empty": True, "pre_existing_empty": True}
    return {"exists": False, "empty": True, "pre_existing_empty": False}


class InputAllowlist:
    """Resolves and verifies computation inputs against the config whitelist."""

    def __init__(self, config: dict, project_root: Path, analysis_root: Path):
        self.entries = {entry["id"]: entry for entry in config["input_whitelist"]["entries"]}
        self.pattern_exceptions = {
            exception["id"]: exception
            for exception in config.get("forbidden_computation_input_patterns", {}).get("exceptions", [])
        }
        self.by_path: dict[str, dict] = {}
        self.project_root = project_root
        self.analysis_root = analysis_root
        for entry in self.entries.values():
            self.by_path[self._key(self.resolve(entry))] = entry
        self.verified: dict[str, str] = {}

    def _key(self, path: Path) -> str:
        return str(path)

    def resolve(self, entry: dict) -> Path:
        root_kind = entry["root"]
        if root_kind == "absolute":
            return Path(entry["path"]).expanduser().resolve()
        root = self.project_root if root_kind == "project" else self.analysis_root
        candidate = (root / entry["path"]).resolve()
        if root not in candidate.parents and candidate != root:
            raise ReplayRefusal(f"input escapes declared root: {entry['path']}")
        return candidate

    def entry_for_path(self, path: os.PathLike[str] | str) -> dict:
        key = str(Path(path).expanduser().resolve())
        if key not in self.by_path:
            display = key
            for root in (self.project_root, self.analysis_root):
                try:
                    display = str(Path(key).relative_to(root))
                    break
                except ValueError:
                    continue
            raise ReplayRefusal(
                "refused non-allowlisted input (not in config whitelist): " + display
            )
        return self.by_path[key]

    def verify(self, entry: dict, *, force: bool = True, allow_large: bool = False) -> str:
        path = self.resolve(entry)
        hit = forbidden_pattern_for(path)
        if hit and entry["id"] not in self.pattern_exceptions:
            raise ReplayRefusal(f"refused forbidden computation input ({hit}): {path}")
        if not path.is_file():
            raise ReplayRefusal(f"refused missing input: {entry['id']}: {path}")
        size = entry.get("bytes") or path.stat().st_size
        if entry.get("hash_source") and not allow_large and size > MAX_INLINE_HASH_BYTES:
            raise ReplayRefusal(
                f"{entry['id']}: {size} bytes exceed the inline-hash limit; this entry carries a pinned-manifest digest "
                f"({entry['hash_source']}); pass allow_large/hash_source trust explicitly to override"
            )
        if force or entry["id"] not in self.verified:
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise ReplayRefusal(
                    f"input hash mismatch for {entry['id']}: {digest} != {entry['sha256']} ({path})"
                )
            self.verified[entry["id"]] = digest
        return self.verified[entry["id"]]

    def verify_path(self, path: os.PathLike[str] | str) -> tuple[dict, str]:
        entry = self.entry_for_path(path)
        return entry, self.verify(entry)

    def verify_many(self, entry_ids: list[str], *, force: bool = True, allow_large: bool = False) -> dict[str, str]:
        result: dict[str, str] = {}
        for entry_id in entry_ids:
            if entry_id not in self.entries:
                raise ReplayRefusal(f"refused unknown whitelist id: {entry_id}")
            result[entry_id] = self.verify(self.entries[entry_id], force=force, allow_large=allow_large)
        return result

    def counts_by_level(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries.values():
            counts[entry["level"]] = counts.get(entry["level"], 0) + 1
        return counts


def snapshot_tree(paths: list[Path]) -> dict:
    """Per-file (size, mtime) snapshot: cheap and stable under concurrent readers.

    Directory mtimes are deliberately ignored: other authorized agents work in
    the same tree, and a directory mtime changes whenever any file is touched.
    """
    snapshot: dict[str, dict] = {}
    for root in paths:
        root = Path(root)
        if not root.exists():
            snapshot[str(root)] = {"exists": False, "entries": {}}
            continue
        rows: dict[str, dict] = {}
        if root.is_dir():
            for base, _dirs, names in os.walk(root):
                for name in names:
                    candidate = Path(base) / name
                    try:
                        stat = candidate.stat()
                    except OSError:
                        continue
                    rows[str(candidate.relative_to(root))] = {
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
        snapshot[str(root)] = {"exists": True, "entries": rows}
    return snapshot


def classify_snapshot_diff(before: dict, after: dict) -> dict:
    """Split tree changes into modified / removed / added relative entries."""
    modified: list[str] = []
    removed: list[str] = []
    added: list[str] = []
    for root in sorted(set(before) | set(after)):
        left = before.get(root, {}).get("entries", {})
        right = after.get(root, {}).get("entries", {})
        left_exists = before.get(root, {}).get("exists", False)
        right_exists = after.get(root, {}).get("exists", False)
        if left_exists != right_exists:
            modified.append(f"{root}: existence {left_exists} -> {right_exists}")
            continue
        for rel in sorted(set(left) & set(right)):
            if left[rel] != right[rel]:
                modified.append(f"{root}/{rel}: {left[rel]} -> {right[rel]}")
        removed.extend(f"{root}/{rel}" for rel in sorted(set(left) - set(right)))
        added.extend(f"{root}/{rel}" for rel in sorted(set(right) - set(left)))
    return {"modified": modified, "removed": removed, "added": added}


def diff_snapshots(before: dict, after: dict, *, ignore_additions: bool = False) -> list[str]:
    diff = classify_snapshot_diff(before, after)
    changes = list(diff["modified"]) + list(diff["removed"])
    if not ignore_additions:
        changes += [f"added: {row}" for row in diff["added"]]
    return changes
