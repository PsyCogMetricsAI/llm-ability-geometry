"""Shared helpers: env pinning, hash-declared sources, comparator, JSON io."""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # importing shared aggregators must not write source trees

# CPU-only, single-threaded BLAS: must be pinned before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
    os.environ[_var] = "1"

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

BRANCHES = ("legacy", "panel", "geometry")

# Frozen comparator: abs(new-ref) <= 1e-9 + 1e-6*abs(ref); counts/ids/masks exact.
TOL_BASE = 1e-9
TOL_REL = 1e-6


class SourceError(RuntimeError):
    """A declared cache source is missing, undeclared or hash-mismatched."""


class OutputError(RuntimeError):
    """The requested output directory is not a fresh empty directory."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel_to_root(path, root) -> str:
    return Path(os.path.relpath(str(Path(path).resolve()), str(Path(root).resolve()))).as_posix()


class PathPolicy:
    """Root-parameterised read boundary.

    Declared paths are relative to the selected analysis root; ``../`` entries
    must stay inside the selected project root (the analysis root's parent,
    which hosts the shared ``Imports`` tree).  Absolute paths, paths escaping the
    project, sibling-prefix lookalikes and symlinks that leave the project are
    refused, so there is no hidden fallback to the original tree.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.project = self.root.parent
        self.allowed = (self.root, self.project)

    def _inside(self, resolved: Path) -> bool:
        for base in self.allowed:
            try:
                resolved.relative_to(base)
                return True
            except ValueError:
                continue
        return False

    def resolve_declared(self, rel: str) -> Path:
        if os.path.isabs(rel):
            raise SourceError(f"declared path must be relative to the analysis root: {rel}")
        p = Path(rel)
        if any(part == ".." for part in p.parts) and len(p.parts) > 0:
            # only the single '../<dir>' form is allowed (project-level inputs)
            ups = [part for part in p.parts if part == ".."]
            if len(ups) > 1 or p.parts[0] != "..":
                raise SourceError(f"declared path escapes the selected project root: {rel}")
        resolved = (self.root / rel).resolve()
        if not self._inside(resolved):
            raise SourceError(
                f"declared path resolves outside the selected tree (root={self.root}, "
                f"project={self.project}): {rel} -> {resolved}")
        return resolved

    def check_write_target(self, path: Path, allowed_write: Path) -> bool:
        try:
            Path(path).resolve().relative_to(Path(allowed_write).resolve())
            return True
        except ValueError:
            return False


def relocate_original_path(orig: str, root: Path) -> Path | None:
    """Map a frozen-contract absolute path onto the selected tree.

    Only the two declared anchors are accepted: the analysis-root anchor
    (``/<root.name>/``) and the shared geometry anchor
    (``/Imports/geometry/`` -> project-level Imports).  Paths outside these
    anchors (plans/evidence trees, sibling lookalikes) return None: they are
    never read, and no fallback to the original source tree is performed.
    """
    root = Path(root).resolve()
    s = Path(orig).as_posix()
    anchors = ((f"/{root.name}/", root),
               ("/Imports/geometry/", root.parent / "Imports" / "geometry"))
    for anchor, base in anchors:
        idx = s.find(anchor)
        if idx >= 0:
            suffix = s[idx + len(anchor):]
            if not suffix:
                return None
            return (base / suffix)
    return None


class PathAudit:
    """Dynamic read/write audit: records every open() event outside the allowed
    roots so a relocated run can prove it never fell back to the original tree."""

    def __init__(self, code_dir: Path, allowed_read: tuple, allowed_write: Path,
                 original_project: Path | None = None):
        self.code_dir = Path(code_dir).resolve()
        self.allowed_read = tuple(Path(p).resolve() for p in allowed_read)
        self.allowed_write = Path(allowed_write).resolve()
        self.original_project = Path(original_project).resolve() if original_project else None
        self.runtime_prefixes = tuple(
            Path(p).resolve() for p in {sys.prefix, sys.base_prefix,
                                        os.path.dirname(os.__file__)}
            if p and Path(p).exists())
        self.scientific = 0
        self.runtime = 0
        self.wrapper_code = 0
        self.writes_outside_out = []
        self.reads_outside_selected = []
        self.original_tree_reads = []
        self._installed = False

    @staticmethod
    def _under(path: Path, bases) -> bool:
        for base in bases:
            try:
                path.relative_to(base)
                return True
            except ValueError:
                continue
        return False

    def _hook(self, event, args):
        try:
            if event != "open":
                return
            raw, mode = args[0], args[1]
            if not isinstance(raw, (str, bytes)):
                return
            path = Path(os.fsdecode(raw))
            try:
                resolved = path.resolve()
            except OSError:
                return
            writing = bool(mode) and any(c in str(mode) for c in "wax+")
            if writing:
                if not self._under(resolved, (self.allowed_write,)):
                    self.writes_outside_out.append(str(resolved))
                return
            if self._under(resolved, self.allowed_read):
                self.scientific += 1
            elif self._under(resolved, (self.code_dir,)):
                self.wrapper_code += 1
            elif self._under(resolved, self.runtime_prefixes):
                self.runtime += 1
            else:
                self.reads_outside_selected.append(str(resolved))
                if self.original_project is not None and \
                        self._under(resolved, (self.original_project,)) and \
                        not self._under(resolved, (self.code_dir,)):
                    self.original_tree_reads.append(str(resolved))
        except Exception:  # audit hooks must never raise
            return

    def install(self):
        if not self._installed:
            sys.addaudithook(self._hook)
            self._installed = True

    def report(self) -> dict:
        return {
            "allowed_read_roots": [str(p) for p in self.allowed_read],
            "allowed_write_root": str(self.allowed_write),
            "wrapper_code_dir": str(self.code_dir),
            "counts": {"selected_tree_reads": self.scientific,
                       "wrapper_code_reads": self.wrapper_code,
                       "runtime_reads": self.runtime,
                       "reads_outside_selected_tree": len(self.reads_outside_selected),
                       "original_tree_reads": len(self.original_tree_reads),
                       "writes_outside_out_dir": len(self.writes_outside_out)},
            "reads_outside_selected_tree": self.reads_outside_selected[:50],
            "original_tree_reads": self.original_tree_reads[:50],
            "writes_outside_out_dir": self.writes_outside_out[:50],
            "note": "audit hook records open() events; wrapper reads are additionally "
                    "root-bound and sha256-verified in Sources",
        }


def sanitize(obj):
    """JSON-safe copy: numpy scalars -> python, NaN/Inf -> None."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj


def write_json(path, obj) -> str:
    text = json.dumps(sanitize(obj), indent=1, allow_nan=False, sort_keys=False)
    Path(path).write_text(text + "\n", encoding="utf-8")
    return hashlib.sha256((text + "\n").encode()).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def as_float_or_none(v):
    """None for null/NaN; float otherwise (mirrors the accepted null convention)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class Sources:
    """Hash-declared source cache.

    Every consumed file must appear in the declared manifest with a sha256 that
    matches the bytes on disk.  Undeclared, missing or mismatched files raise
    ``SourceError`` before any statistic is computed.
    """

    def __init__(self, root: Path, entries: list[dict], policy: PathPolicy | None = None):
        self.root = Path(root)
        self.policy = policy or PathPolicy(self.root)
        self.declared = {}
        for e in entries:
            rel = e["path"]
            if rel in self.declared:
                raise SourceError(f"duplicate manifest entry: {rel}")
            # boundary check at load time: the manifest itself must be clean
            self.policy.resolve_declared(rel)
            self.declared[rel] = e
        self.used: dict[str, dict] = {}
        self.checked = 0

    def resolve(self, rel: str) -> Path:
        return self.policy.resolve_declared(rel)

    def require(self, rel: str, role: str | None = None) -> Path:
        entry = self.declared.get(rel)
        if entry is None:
            raise SourceError(
                f"source not declared in the hash manifest: {rel} "
                "(fail closed; regenerate the manifest only from verified caches)")
        path = self.resolve(rel)
        if not path.exists():
            raise SourceError(f"declared source missing on disk: {rel} -> {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise SourceError(
                f"sha256 mismatch for {rel}: declared {entry['sha256']} found {actual}")
        self.checked += 1
        used = dict(entry)
        if role:
            used["role"] = role
        used["path"] = rel
        used["resolved_path"] = str(path)
        used["resolved_inside_project"] = self.policy._inside(path)
        self.used[rel] = used
        return path

    def note(self, rel: str, path, role=None, level=None, sha256=None):
        """Record a comparator / cross-check file without hashing it twice."""
        entry = dict(self.declared.get(rel) or {})
        entry.update({"path": rel, "sha256": sha256 or sha256_file(path)})
        if role:
            entry["role"] = role
        if level:
            entry["level"] = level
        self.used[rel] = entry

    def used_entries(self) -> list[dict]:
        return [self.used[k] for k in sorted(self.used)]


def load_declared_manifest(path: Path) -> list[dict]:
    if not Path(path).exists():
        raise SourceError(f"declared source manifest not found: {path}")
    data = load_json(path)
    entries = data["sources"] if isinstance(data, dict) else data
    for e in entries:
        for key in ("path", "sha256", "role", "level"):
            if key not in e:
                raise SourceError(f"declared manifest entry missing '{key}': {e}")
    return entries


def _json_equal(a, b) -> bool:
    return json.dumps(sanitize(a), sort_keys=True) == json.dumps(sanitize(b), sort_keys=True)


class Comparator:
    """Collects numerical comparisons with exact JSON pointers to both sides."""

    def __init__(self, branch: str):
        self.branch = branch
        self.entries: list[dict] = []
        self.skipped: list[dict] = []

    def float(self, name, new, ref, *, new_ptr, ref_ptr, ref_path, ref_sha256):
        self._add(name, "float", new, ref, new_ptr, ref_ptr, ref_path, ref_sha256)

    def exact(self, name, new, ref, *, new_ptr, ref_ptr, ref_path, ref_sha256):
        self._add(name, "exact", new, ref, new_ptr, ref_ptr, ref_path, ref_sha256)

    def tuple_float(self, name, new, ref, *, new_ptr, ref_ptr, ref_path, ref_sha256):
        self._add(name, "float_list", new, ref, new_ptr, ref_ptr, ref_path, ref_sha256)

    def skip(self, name, ref_ptr, ref_path, reason):
        self.skipped.append({"name": name, "ref_pointer": ref_ptr,
                             "ref_source": ref_path, "reason": reason})

    def _add(self, name, kind, new, ref, new_ptr, ref_ptr, ref_path, ref_sha256):
        a = sanitize(new)
        b = sanitize(ref)
        entry = {
            "name": name, "kind": kind,
            "new_pointer": new_ptr, "ref_pointer": ref_ptr,
            "ref_source": ref_path, "ref_sha256": ref_sha256,
            "tolerance": f"{TOL_BASE}+{TOL_REL}*abs(ref)" if kind != "exact" else "exact",
            "value_new": a, "value_ref": b,
        }
        if kind == "float":
            if a is None and b is None:
                ok, diff = True, 0.0
            elif a is None or b is None:
                ok, diff = False, None
            else:
                diff = abs(a - b)
                ok = diff <= TOL_BASE + TOL_REL * abs(b)
            entry["abs_diff"] = diff
            entry["pass"] = ok
        elif kind == "float_list":
            if a is None and b is None:
                ok = True
            elif a is None or b is None or len(a) != len(b):
                ok = False
            else:
                diffs = [abs(x - y) if (x is not None and y is not None) else None
                         for x, y in zip(a, b)]
                ok = all(d is not None and d <= TOL_BASE + TOL_REL * abs(y)
                         for d, y in zip(diffs, b))
                entry["abs_diff"] = diffs
            entry["pass"] = bool(ok)
        else:
            entry["pass"] = _json_equal(a, b)
        self.entries.append(entry)

    def dump(self) -> dict:
        failures = [e for e in self.entries if not e["pass"]]
        return {
            "branch": self.branch,
            "comparator": "abs(new-ref) <= 1e-9 + 1e-6*abs(ref); counts/ids/null-masks exact",
            "n_compared": len(self.entries),
            "n_pass": len(self.entries) - len(failures),
            "n_fail": len(failures),
            "failures": failures,
            "skipped": self.skipped,
            "entries": self.entries,
        }


def finite_moments(values) -> dict:
    """Accepted finite-conditional moment block (same rules as the frozen reports)."""
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    out = {"n_total": int(arr.size), "n_finite": n,
           "n_invalid": int(arr.size - n),
           "failure_rate": (float(arr.size - n) / arr.size) if arr.size else None}
    if n == 1:
        out.update({"mean": float(finite[0]), "sd": None, "mcse_mean": None,
                    "ci95_low": None, "ci95_high": None,
                    "ci95_reason": "insufficient_finite"})
    elif n > 1:
        mean = float(finite.mean())
        sd = float(finite.std(ddof=1))
        mcse = sd / math.sqrt(n)
        out.update({"mean": mean, "sd": sd, "mcse_mean": mcse,
                    "ci95_low": mean - 1.959963984540054 * mcse,
                    "ci95_high": mean + 1.959963984540054 * mcse,
                    "ci95_reason": None})
    else:
        out.update({"mean": None, "sd": None, "mcse_mean": None,
                    "ci95_low": None, "ci95_high": None,
                    "ci95_reason": "no_finite_values"})
    return out


def wall_clock():
    return time.perf_counter()
