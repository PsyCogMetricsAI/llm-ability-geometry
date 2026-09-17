#!/usr/bin/env bash
# Negative test: a truncated reference must fail the example (real interpreter required).
set -euo pipefail
PKG="$(cd "$(dirname "$0")" && pwd)"
TMP="$1"; PY="$2"; shift 2
test -x "$PY" || { echo "NEGATIVE_TEST_ERROR: python not executable: $PY"; exit 2; }
mkdir -p "$TMP"
head -5 "$PKG/examples/depth_profiles/reference_depth_summary_fresh.csv" > "$TMP/truncated_reference.csv"
set +e
OUT="$("$PY" "$PKG/examples/run_depth_example.py" --reference "$TMP/truncated_reference.csv" 2>&1)"
RC=$?
set -e
if [ "$RC" -ne 0 ] && printf '%s' "$OUT" | grep -q "FAIL counts"; then
  echo "NEGATIVE_TEST_PASS: truncated reference refused ($OUT)"; exit 0
fi
echo "NEGATIVE_TEST_FAILED rc=$RC out=$OUT"; exit 1
