#!/usr/bin/env python3
"""Sandbox execution wrapper for external producers (engine-owned, stdlib only).

Runs a byte-identical (or declaratively migrated) producer inside a staged
sandbox and *enforces* the read/write contract at the interpreter level:

* every ``open``/``os.open`` event is checked: writes must stay inside the
  sandbox, reads must stay inside the sandbox or the interpreter environment;
* directory listings, renames, links, deletions, chmod/utime are checked the
  same way (``os.listdir``/``os.scandir``/``os.rename``/``os.remove``/...);
* every access is appended to an access log (JSON lines) through a pre-opened
  fd, so logging cannot recurse into the audit hook;
* BLAS/OpenMP threads are pinned to 1 and process/thread pools are capped at
  the declared ``--max-workers`` (2 by default). The cap changes scheduling
  only: every declared producer derives its results from per-job seeds, not
  from pool size, and the adapter records the cap in the receipt.

Modes:
  ``script``   run the target with ``runpy`` as ``__main__`` (real execution);
  ``import``   import the target as a module (dependency/import-level check);
  ``fixture``  run an engine-supplied fixture file that imports the target and
               calls its pure functions on tiny legal inputs.

This wrapper never imports third-party packages itself, so it can run under any
interpreter in the isolated replay environment.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time

ACCESS_SCHEMA = "rfinal-sandbox-access-log-v1"

_LOG_FD = -1


def _emit(record: dict) -> None:
    if _LOG_FD < 0:
        return
    try:
        os.write(_LOG_FD, (json.dumps(record, sort_keys=True) + "\n").encode())
    except OSError:
        pass


def _norm(path) -> str | None:
    try:
        raw = os.fspath(path)
    except TypeError:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "surrogateescape")
    if not raw or raw.startswith("\0"):
        return None
    return os.path.abspath(raw)


def _under(child: str, root: str) -> bool:
    if child == root:
        return True
    root = root.rstrip(os.sep) + os.sep
    return child.startswith(root)


class SandboxGuard:
    """Interpreter-level read/write enforcement with an append-only access log."""

    def __init__(self, sandbox: str, extra_read_roots: list[str], log_path: str, summary_path: str):
        global _LOG_FD
        self.sandbox = os.path.abspath(sandbox)
        roots = [self.sandbox]
        roots += [os.path.abspath(root) for root in extra_read_roots if root]
        for candidate in (
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            sys.base_exec_prefix,
            os.path.dirname(os.path.abspath(sys.executable)),
            "/usr/lib",
            "/usr/local/lib",
            "/usr/share/zoneinfo",
            "/lib",
            "/etc",
            "/proc",
            "/sys",
            "/dev",
        ):
            if candidate:
                roots.append(os.path.abspath(candidate))
        self.read_roots = tuple(dict.fromkeys(roots))
        self.write_roots = (self.sandbox,)
        # engine-owned files that the wrapper itself must be able to write
        self.engine_write_files = {os.path.abspath(log_path), os.path.abspath(summary_path)}
        _LOG_FD = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self.counts = {"read_inside": 0, "write_inside": 0, "refused": 0}
        self._busy = False
        self.started = time.time()

    def _record(self, kind: str, path: str, mode: str) -> None:
        _emit(
            {
                "schema": ACCESS_SCHEMA,
                "utc_offset_s": round(time.time() - self.started, 6),
                "pid": os.getpid(),
                "event": kind,
                "path": path,
                "mode": mode,
            }
        )

    def _record_refusal(self, path: str, mode: str, source_event: str, args: tuple, writing: bool) -> None:
        _emit(
            {
                "schema": ACCESS_SCHEMA,
                "utc_offset_s": round(time.time() - self.started, 6),
                "pid": os.getpid(),
                "event": "refused_write" if writing else "refused_read",
                "source_event": source_event,
                "path": path,
                "mode": mode,
                "args_preview": [repr(a)[:80] for a in args[:3]],
            }
        )

    def check(self, event: str, args: tuple) -> None:
        if self._busy:  # re-entrancy guard; never recurse through our own logging
            return
        if event == "open":
            path = _norm(args[0]) if args else None
            if path is None:
                return
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if mode is None and isinstance(flags, int):
                if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                    mode = "w"
                else:
                    mode = "r"
            writing = any(flag in str(mode) for flag in ("w", "a", "x", "+"))
            self._enforce(path, "open", str(mode), writing, args)
        elif event in (
            "os.mkdir",
            "os.rmdir",
            "os.remove",
            "os.rename",
            "os.replace",
            "os.link",
            "os.symlink",
            "os.truncate",
            "os.chmod",
            "os.chown",
            "os.utime",
            "shutil.copyfile",
            "shutil.copymode",
            "shutil.copystat",
            "shutil.move",
            "shutil.rmtree",
            "tempfile.mkstemp",
        ):
            for value in args:
                path = _norm(value)
                if path is not None:
                    self._enforce(path, event, "write", True, args)
        elif event in ("os.listdir", "os.scandir"):
            path = _norm(args[0]) if args else None
            if path is not None:
                self._enforce(path, event, "list", False, args)

    BENIGN_WRITE_DEVICES = ("/dev/null",)

    def _enforce(self, path: str, event: str, mode: str, writing: bool, args: tuple = ()) -> None:
        self._busy = True
        try:
            if writing:
                if path in self.BENIGN_WRITE_DEVICES:
                    self.counts["write_inside"] += 1
                    self._record("allow_device_write", path, mode)
                    return
                allowed = any(_under(path, root) for root in self.write_roots) or path in self.engine_write_files
                if allowed:
                    self.counts["write_inside"] += 1
                else:
                    self.counts["refused"] += 1
                    self._record_refusal(path, mode, event, args, writing)
                    raise PermissionError(
                        f"rfinal sandbox refusal: write outside sandbox refused ({event}): {path}"
                    )
            else:
                allowed = any(_under(path, root) for root in self.read_roots)
                if allowed:
                    self.counts["read_inside"] += 1
                else:
                    self.counts["refused"] += 1
                    self._record_refusal(path, mode, event, args, writing)
                    raise PermissionError(
                        f"rfinal sandbox refusal: non-allowlisted read refused ({event}): {path}"
                    )
            self._record(event, path, mode)
        finally:
            self._busy = False


def cap_pools(max_workers: int) -> dict:
    """Cap process/thread pools without changing per-job numerics."""
    import concurrent.futures as futures
    import multiprocessing

    applied: dict = {"cap": max_workers, "pool_calls": []}

    for name in ("ProcessPoolExecutor", "ThreadPoolExecutor"):
        original = getattr(futures, name)

        class Capped(original):  # type: ignore[misc, valid-type]
            def __init__(self, max_workers=None, *a, **kw):
                wanted = max_workers
                if wanted is None or wanted > applied["cap"]:
                    wanted = applied["cap"]
                applied.setdefault("pool_calls", []).append({"pool": name, "requested": max_workers, "applied": wanted})
                super().__init__(max_workers=wanted, *a, **kw)

        Capped.__name__ = name
        setattr(futures, name, Capped)
    applied["multiprocessing_cpu_count"] = multiprocessing.cpu_count()
    return applied


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rfinal sandbox audit runner")
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--target", required=True, help="target path relative to the sandbox")
    parser.add_argument("--mode", choices=("script", "import", "fixture"), default="script")
    parser.add_argument("--argv", default="[]", help="JSON list of producer argv")
    parser.add_argument("--fixture", default=None, help="fixture path (mode=fixture)")
    parser.add_argument("--access-log", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--extra-read-root", action="append", default=[])
    parser.add_argument("--extra-sys-path", action="append", default=[])
    parser.add_argument("--cap-pools", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sandbox = os.path.abspath(args.sandbox)
    os.chdir(sandbox)
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(args.blas_threads)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "0")
    os.environ["RFINAL_SANDBOX_ROOT"] = sandbox

    pools = cap_pools(args.max_workers) if args.cap_pools else {"cap": None}
    guard = SandboxGuard(sandbox, args.extra_read_root, args.access_log, args.summary)
    sys.addaudithook(guard.check)

    target = os.path.join(sandbox, args.target)
    producer_argv = json.loads(args.argv)
    runner_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if p and os.path.abspath(p) != runner_dir]
    # Import semantics: the producer behaves as if launched from the sandbox's code/
    # directory, so ``import <helper>`` resolves to the staged byte-identical copies.
    code_dir = os.path.join(sandbox, "code")
    for entry in [*[os.path.abspath(item) for item in args.extra_sys_path], code_dir, sandbox]:
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    status = "COMPLETED"
    error = None
    exit_code = 0
    try:
        if args.mode == "script":
            sys.argv = [target, *producer_argv]
            runpy.run_path(target, run_name="__main__")
        elif args.mode == "import":
            import importlib.util

            name = "rfinal_import_check"
            spec = importlib.util.spec_from_file_location(name, target)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load module from {target}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            print(json.dumps({"imported": target, "module_file": module.__file__}))
        else:
            if not args.fixture:
                raise SystemExit("fixture mode requires --fixture")
            sys.argv = [os.path.join(sandbox, args.fixture), *producer_argv]
            runpy.run_path(os.path.join(sandbox, args.fixture), run_name="__main__")
    except SystemExit as exc:  # producer called sys.exit / raised SystemExit
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code not in (0, None):
            status, error, exit_code = "FAILED", f"SystemExit({exc.code!r})", int(code)
    except BaseException as exc:  # noqa: BLE001 - report every failure to the engine
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
        exit_code = 1

    summary = {
        "schema": "rfinal-sandbox-run-summary-v1",
        "mode": args.mode,
        "target": args.target,
        "sandbox": sandbox,
        "status": status,
        "error": error,
        "access_counts": guard.counts,
        "pool_policy": pools,
        "blas_threads": args.blas_threads,
        "argv": producer_argv,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "seconds": round(time.time() - guard.started, 6),
    }
    summary_dir = os.path.dirname(os.path.abspath(args.summary))
    if summary_dir and not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)
    with open(args.summary, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: summary[key] for key in ("status", "error", "access_counts", "seconds")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
