"""Declarative, auditable source/input migrations for the clean replay.

The frozen producers resolve their root from ``Path(__file__)`` or embed
absolute roots from the original run.  The clean replay must not modify the
signed sources, so each migration is declared in the replay config and applied
to a *staged copy* inside the fresh run sandbox.  Every migration returns a
receipt with the file hashes before/after and the exact substitutions, and the
engine refuses any migration whose declared expectations are not met exactly.

Supported operations
-------------------
``text_prefix_rewrite``
    Rewrite every occurrence of ``from`` to ``to`` in a staged text file.
    ``expect_min_replacements``/``expect_max_replacements`` bound the edit so a
    silent no-op or an over-broad substitution is refused.

``json_root_migration``
    Walk a staged JSON file.  Every string that starts with the declared
    ``from`` prefixes is rewritten to the sandbox root (ordered, longest prefix
    first).  Every mapping that carries both ``path`` and ``sha256`` is
    re-hashed after the rewrite and its digest updated only when the staged
    bytes differ from the declared one.  A digest change is accepted only when
    the rewritten path is a declared fresh input of the same run; otherwise the
    migration is refused.  This is the documented RP-HASH-LINEAGE behaviour:
    original reference hashes stay visible as ``previous_sha256`` while the
    fresh binding is recorded separately.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import guards

SCHEMA = "rfinal-transform-receipt-v1"


def resolve_within_sandbox(sandbox: Path, target: str | Path, context: str) -> Path:
    """Resolve *before* any read or write and enforce a real path-component boundary.

    A plain string prefix test would accept ``<sandbox>_neighbor/file.py`` and the
    write would already have happened by the time a later check raised.  This
    resolves the candidate and requires it to be relative to the sandbox root.
    """
    sandbox = Path(sandbox).resolve()
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = sandbox / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise guards.ReplayRefusal(f"{context}: cannot resolve {target!r}: {exc}") from exc
    if resolved != sandbox and sandbox not in resolved.parents:
        raise guards.ReplayRefusal(f"{context}: resolved path escapes the sandbox: {target!r}")
    return resolved


def ensure_private_inode(path: Path, sandbox: Path, context: str) -> dict:
    """Guarantee the file about to be rewritten is a private copy inside the sandbox.

    Staged inputs may be hardlinks to signed originals (opt-in for very large
    read-only data).  Writing through such a link would rewrite the signed tree,
    so any hardlinked/aliased file is replaced by a byte-identical private copy
    *before* the first write.  The decision is made on the inode link count, not
    on a file extension.
    """
    path = Path(path)
    stat = path.stat()
    if stat.st_nlink == 1:
        return {"private_inode": True, "strategy": "already_private", "nlink": stat.st_nlink}
    private = path.with_name(path.name + ".engine-private")
    if private.exists():
        private.unlink()
    data = path.read_bytes()
    before = guards.sha256_file(path)
    with private.open("wb") as handle:
        handle.write(data)
    private.chmod(0o644)
    os.replace(private, path)
    after = guards.sha256_file(path)
    if after != before:
        raise guards.ReplayRefusal(f"{context}: private copy changed the bytes of {path.name}")
    if Path(path).stat().st_nlink != 1:
        raise guards.ReplayRefusal(f"{context}: could not obtain a private inode for {path.name}")
    return {
        "private_inode": True,
        "strategy": "copied_before_write",
        "sha256_before": before,
        "sha256_after": after,
    }


def _relpath_from_sandbox(path: Path, sandbox: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(sandbox).resolve()))


def text_prefix_rewrite(
    *,
    sandbox: Path,
    target: str,
    from_prefix: str,
    to_prefix: str,
    expect_min_replacements: int = 1,
    expect_max_replacements: int | None = None,
    label: str = "",
) -> dict:
    sandbox = Path(sandbox)
    path = resolve_within_sandbox(sandbox, target, f"transform {label or target}")
    if not path.is_file():
        raise guards.ReplayRefusal(f"transform target missing: {target}")
    before_sha = guards.sha256_file(path)
    private = ensure_private_inode(path, sandbox, f"transform {label or target}")
    text = path.read_text()
    count = text.count(from_prefix)
    if count < expect_min_replacements:
        raise guards.ReplayRefusal(
            f"transform {label or target}: expected >= {expect_min_replacements} occurrences of "
            f"{from_prefix!r}, found {count}"
        )
    if expect_max_replacements is not None and count > expect_max_replacements:
        raise guards.ReplayRefusal(
            f"transform {label or target}: expected <= {expect_max_replacements} occurrences of "
            f"{from_prefix!r}, found {count}"
        )
    updated = text.replace(from_prefix, to_prefix)
    path.write_text(updated)
    after_sha = guards.sha256_file(path)
    return {
        "schema": SCHEMA,
        "kind": "text_prefix_rewrite",
        "label": label,
        "target": _relpath_from_sandbox(path, sandbox),
        "from_prefix": from_prefix,
        "to_prefix": to_prefix,
        "replacements": count,
        "sha256_before": before_sha,
        "sha256_after": after_sha,
        "private_inode": private,
    }


def _normalise_rules(rules: list[dict]) -> list[tuple[str, str]]:
    """Longest ``from`` first so R-prefixed values are not shadowed by P."""
    pairs = [(str(rule["from"]), str(rule["to"])) for rule in rules if rule.get("from") is not None]
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def _migrate_string(value: str, rules: list[tuple[str, str]]) -> tuple[str, str | None]:
    for prefix, replacement in rules:
        if value.startswith(prefix):
            return replacement + value[len(prefix):], prefix
    return value, None


def json_root_migration(
    *,
    sandbox: Path,
    target: str,
    from_prefixes: list[str] | None = None,
    to_prefix: str | None = None,
    fresh_paths: dict[str, str] | None = None,
    label: str = "",
    rules: list[dict] | None = None,
) -> dict:
    """Rewrite original-root strings inside a staged JSON input.

    ``fresh_paths`` maps sandbox-relative paths (as they appear after the
    rewrite) to the sha256 of the freshly produced file they must now bind to.
    """
    sandbox = Path(sandbox)
    path = resolve_within_sandbox(sandbox, target, f"transform {label or target}")
    if not path.is_file():
        raise guards.ReplayRefusal(f"transform target missing: {target}")
    if rules is None:
        if not from_prefixes or to_prefix is None:
            raise guards.ReplayRefusal(f"transform {label or target}: needs rules or from_prefixes+to_prefix")
        rules = [{"from": prefix, "to": to_prefix} for prefix in from_prefixes]
    rule_pairs = _normalise_rules(rules)
    to_prefix = rule_pairs[0][1] if to_prefix is None else to_prefix
    fresh_paths = {str(key): value for key, value in (fresh_paths or {}).items()}
    before_sha = guards.sha256_file(path)
    private = ensure_private_inode(path, sandbox, f"transform {label or target}")
    document = json.loads(path.read_text())
    rewrites: list[dict] = []
    rebinds: list[dict] = []

    def walk(node, trail: str):
        if isinstance(node, dict):
            migrated = {}
            for key, value in node.items():
                if isinstance(value, str):
                    new_value, hit = _migrate_string(value, rule_pairs)
                    if hit is not None:
                        migrated[key] = new_value
                        rewrites.append({"trail": f"{trail}/{key}", "from": value, "to": new_value})
                    else:
                        migrated[key] = value
                else:
                    migrated[key] = walk(value, f"{trail}/{key}")
            if "path" in migrated and "sha256" in migrated and isinstance(migrated["path"], str):
                candidate = migrated["path"]
                if Path(candidate).is_absolute() or candidate.startswith(to_prefix):
                    resolved = resolve_within_sandbox(
                        sandbox, candidate, f"transform {label or target} referenced path"
                    )
                    rel = str(resolved.relative_to(Path(sandbox).resolve()))
                    if not resolved.is_file():
                        raise guards.ReplayRefusal(
                            f"transform {label or target}: rewritten path missing in sandbox: {rel}"
                        )
                    actual = guards.sha256_file(resolved)
                    declared = str(migrated["sha256"])
                    if actual != declared:
                        if rel not in fresh_paths:
                            raise guards.ReplayRefusal(
                                f"transform {label or target}: digest change for non-fresh path {rel} "
                                f"({declared} -> {actual}); refusing undeclared input substitution"
                            )
                        if fresh_paths[rel] != actual:
                            raise guards.ReplayRefusal(
                                f"transform {label or target}: fresh registry hash mismatch for {rel}"
                            )
                        rebinds.append(
                            {
                                "trail": f"{trail}/sha256",
                                "path": rel,
                                "previous_sha256": declared,
                                "fresh_sha256": actual,
                            }
                        )
                        migrated["sha256"] = actual
            return migrated
        if isinstance(node, list):
            return [walk(item, f"{trail}/{index}") for index, item in enumerate(node)]
        if isinstance(node, str):
            new_value, hit = _migrate_string(node, rule_pairs)
            if hit is not None:
                rewrites.append({"trail": trail, "from": node, "to": new_value})
                return new_value
            return node
        return node

    migrated_document = walk(document, "")
    path.write_text(json.dumps(migrated_document, indent=2) + "\n")
    after_sha = guards.sha256_file(path)
    return {
        "schema": SCHEMA,
        "kind": "json_root_migration",
        "label": label,
        "target": _relpath_from_sandbox(path, sandbox),
        "rules": [{"from": a, "to": b} for a, b in rule_pairs],
        "to_prefix": to_prefix,
        "rewrites": rewrites,
        "rebinds": rebinds,
        "n_rewrites": len(rewrites),
        "n_rebinds": len(rebinds),
        "sha256_before": before_sha,
        "sha256_after": after_sha,
        "private_inode": private,
    }


def apply_transforms(
    *,
    sandbox: Path,
    specs: list[dict],
    from_prefixes: list[str],
    to_prefix: str,
    fresh_paths: dict[str, str],
) -> list[dict]:
    receipts: list[dict] = []
    for spec in specs:
        kind = spec.get("kind")
        if kind == "text_prefix_rewrite":
            receipts.append(
                text_prefix_rewrite(
                    sandbox=sandbox,
                    target=spec["target"],
                    from_prefix=spec["from"],
                    to_prefix=spec.get("to", to_prefix),
                    expect_min_replacements=int(spec.get("expect_min_replacements", 1)),
                    expect_max_replacements=spec.get("expect_max_replacements"),
                    label=spec.get("label", ""),
                )
            )
        elif kind == "json_root_migration":
            receipts.append(
                json_root_migration(
                    sandbox=sandbox,
                    target=spec["target"],
                    from_prefixes=spec.get("from_prefixes", from_prefixes),
                    to_prefix=spec.get("to", to_prefix),
                    fresh_paths=fresh_paths,
                    label=spec.get("label", ""),
                    rules=spec.get("rules"),
                )
            )
        else:
            raise guards.ReplayRefusal(f"unsupported transform kind: {kind!r}")
    return receipts


def sandbox_relative_paths(sandbox: Path, root: Path) -> list[str]:
    """Deterministic, sorted file list of a staged tree (engine-side helper)."""
    files = []
    for base, _dirs, names in os.walk(root):
        for name in sorted(names):
            files.append(str((Path(base) / name).relative_to(sandbox)))
    return sorted(files)
