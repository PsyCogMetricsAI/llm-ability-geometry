#!/usr/bin/env bash
# Cached replay with explicit packaged roots and pinned interpreters (no original environment).
# Usage: scripts/run_cached_replay.sh <DEST> <RENDER_PY> <OUT_DIR> [extra entrypoint args...]
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:?materialized dest}"; RENDER_PY="${2:?renderer python}"; OUT="${3:?out dir}"; shift 3
PROJECT_ROOT="$DEST/project"; ANALYSIS_ROOT="$PROJECT_ROOT/repro_paper2_20260915"
NUMERIC_PY="$ANALYSIS_ROOT/recovered/c17_replay_env/venv-cpu-replay/bin/python"  # fixed derived path
export PKG_DIR="$PKG"
[ -x "$RENDER_PY" ] || { echo "missing renderer python: $RENDER_PY" >&2; exit 2; }
[ -x "$NUMERIC_PY" ] || { echo "missing numeric python: $NUMERIC_PY (run scripts/rebuild_env.sh)" >&2; exit 2; }
"$RENDER_PY" - <<'PY'
import sys, numpy, matplotlib, json
lock=json.load(open(__import__('os').environ["PKG_DIR"]+"/config/env/render-python-lock.json"))
want=lock["runtime"]
assert sys.version.split()[0]==want["python"] and matplotlib.__version__==want["matplotlib"] and numpy.__version__==want["numpy"], (sys.version, matplotlib.__version__, numpy.__version__)
print("RENDER_ENV_OK")
PY
"$NUMERIC_PY" - <<'PY'
import sys
from importlib import metadata
want = {"numpy": "2.5.0", "scipy": "1.18.1", "girth": "0.8.0", "sympy": "1.14.0"}
assert sys.version.split()[0] == "3.12.11", sys.version
for name, version in want.items():
    got = metadata.version(name)
    assert got == version, (name, got, version)
print("NUMERIC_ENV_OK")
PY
export RFINAL_RENDER_PYTHON="$RENDER_PY"
export PKG_DIR="$PKG"
echo "PROJECT_ROOT=$PROJECT_ROOT"; echo "ANALYSIS_ROOT=$ANALYSIS_ROOT"
echo "RENDER_PY=$RENDER_PY"; echo "NUMERIC_PY=$NUMERIC_PY"
"$NUMERIC_PY" "$PKG/code/rfinal_replay/entrypoint.py" plan --config "$PKG/config/CT_C17_REPLAY_FINAL_v1.relocated.json" \
  --project-root "$PROJECT_ROOT" --analysis-root "$ANALYSIS_ROOT" --json >/dev/null
"$NUMERIC_PY" "$PKG/code/rfinal_replay/entrypoint.py" run --mode replay --execute --require-complete \
  --project-root "$PROJECT_ROOT" --analysis-root "$ANALYSIS_ROOT" \
  --config "$PKG/config/CT_C17_REPLAY_FINAL_v1.relocated.json" --out "$OUT" "$@"
