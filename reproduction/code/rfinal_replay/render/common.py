"""Shared helpers for the C17 renderers."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

from . import EXIT_BAD_SCHEMA, EXIT_MISSING_INPUT


class RenderRefusal(Exception):
    """Explicit, non-silent refusal with a contract exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def refuse_bad_schema(message: str) -> None:
    raise RenderRefusal(EXIT_BAD_SCHEMA, message)


def refuse_missing_input(message: str) -> None:
    raise RenderRefusal(EXIT_MISSING_INPUT, message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_input(input_root: Path, rel_path: str, must_exist: bool = True,
                  allow_external: bool = False) -> Path:
    if not rel_path or os.path.isabs(rel_path):
        refuse_bad_schema(f"declared input path must be relative: {rel_path!r}")
    path = (input_root / rel_path).resolve()
    root = input_root.resolve()
    if not str(path).startswith(str(root) + os.sep) and path != root and not allow_external:
        refuse_bad_schema(f"declared input escapes the input root: {rel_path!r}")
    if must_exist and not path.is_file():
        refuse_missing_input(f"required input missing: {rel_path}")
    return path


def resolve_pointer(doc, pointer: str):
    """Resolve a JSON pointer like /a/b/0 (no ~ escapes needed by our contracts)."""
    if pointer in ("", "/"):
        return doc
    parts = [p for p in pointer.split("/") if p != ""]
    cur = doc
    seen = []
    for part in parts:
        seen.append(part)
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                refuse_missing_input(f"pointer {pointer}: '{part}' is not a list index")
            if idx >= len(cur) or idx < 0:
                refuse_missing_input(f"pointer {pointer}: index out of range at {'/'.join(seen)}")
            cur = cur[idx]
        elif isinstance(cur, dict):
            if part not in cur:
                refuse_missing_input(f"pointer {pointer}: key missing at {'/'.join(seen)}")
            cur = cur[part]
        else:
            refuse_missing_input(f"pointer {pointer}: cannot descend into scalar at {'/'.join(seen)}")
    return cur


def deref_spec(input_root: Path, spec: dict):
    """Resolve one declared input spec -> (file_path, value_or_None, record)."""
    if not isinstance(spec, dict) or "path" not in spec:
        refuse_bad_schema(f"input spec must be an object with 'path': {spec!r}")
    path = resolve_input(input_root, spec["path"], allow_external=bool(spec.get("external")))
    record = {
        "role": spec.get("role", ""),
        "path_rel": spec["path"],
        "path_abs": str(path),
        "sha256": sha256_file(path),
        "declared_sha256": spec.get("sha256"),
    }
    if spec.get("sha256") and record["sha256"] != spec["sha256"]:
        refuse_missing_input(
            f"declared input hash mismatch for {spec['path']}: {record['sha256']} != {spec['sha256']}"
        )
    value = None
    if spec.get("pointer") is not None:
        value = resolve_pointer(load_json(path), spec["pointer"])
        record["pointer"] = spec["pointer"]
    return path, value, record


def read_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        refuse_missing_input(f"CSV input is empty: {path}")
    header, data = rows[0], rows[1:]
    return header, data


def fmt_num(value, decimals: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return str(value)
    if value != 0 and abs(value) < 10 ** (-decimals):
        return f"{value:.{decimals}g}"
    return f"{value:.{decimals}f}"


def fmt_p(value) -> str:
    if value is None:
        return "--"
    if value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}".replace("e-0", "e-")


def tex_escape(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def tex_number(value, decimals: int = 4) -> str:
    s = fmt_num(value, decimals)
    if s.startswith("-"):
        return f"$-{s[1:]}$"
    return f"${s}$" if any(c.isdigit() for c in s) else s


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n")


def write_csv(path: Path, header, rows) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    write_text(path, buf.getvalue())


def artifact_record(path: Path, outdir: Path) -> dict:
    rel = str(path.relative_to(outdir))
    return {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def value_cell(value, unit, n, source: dict, display: str | None = None) -> dict:
    """Emit one plot-data cell with explicit unit/n/source provenance."""
    return {
        "value": value,
        "display": display if display is not None else fmt_num(value),
        "unit": unit,
        "n": n,
        "source": {
            "path": source.get("path_rel"),
            "json_pointer": source.get("pointer"),
            "sha256": source.get("sha256"),
        },
    }
