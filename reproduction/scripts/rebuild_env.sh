#!/usr/bin/env bash
# Build BOTH pinned venvs inside the materialized analysis mirror at the paths the engine expects.
# Usage: scripts/rebuild_env.sh <DEST> <BASE_PYTHON_3_12>
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:?materialized dest (contains project/repro_paper2_20260915)}"
BASE="${2:?python3.12 binary}"
ANALYSIS="$DEST/project/repro_paper2_20260915"
NUMERIC="$ANALYSIS/recovered/c17_replay_env/venv-cpu-replay"
RENDER="$ANALYSIS/recovered/c17_replay_env/venv-cpu-render"
"$BASE" -c 'import sys;assert sys.version.split()[0]=="3.12.11", sys.version'
for dir in "$NUMERIC" "$RENDER"; do [ -x "$dir/bin/python" ] || "$BASE" -m venv "$dir"; done
"$NUMERIC/bin/pip" install --quiet -r "$PKG/config/env/numeric-requirements.txt"
"$RENDER/bin/pip" install --quiet -r "$PKG/config/env/render-requirements.txt"
"$NUMERIC/bin/python" - <<'PY'
import sys
from importlib import metadata
want = {"numpy": "2.5.0", "scipy": "1.18.1", "girth": "0.8.0", "sympy": "1.14.0"}
assert sys.version.split()[0] == "3.12.11", sys.version
for name, version in want.items():
    got = metadata.version(name)
    assert got == version, (name, got, version)
print("NUMERIC_OK")
PY
"$RENDER/bin/python" - <<'PY'
import sys
from importlib import metadata
want = {"matplotlib": "3.11.0", "numpy": "2.5.0"}
assert sys.version.split()[0] == "3.12.11", sys.version
for name, version in want.items():
    got = metadata.version(name)
    assert got == version, (name, got, version)
print("RENDER_OK")
PY
echo "PROJECT_ROOT=$DEST/project"; echo "ANALYSIS_ROOT=$ANALYSIS"
echo "NUMERIC_PY=$NUMERIC/bin/python"; echo "RENDER_PY=$RENDER/bin/python"
