#!/usr/bin/env bash
# run_t3.sh — Orchestrate T3 reproduction pipeline
#
# Usage (from repo/reader/):
#   bash run_t3.sh
#
# Environment requirements:
#   R_LIBS must point to an R library containing mirt 1.47+ (R 4.3+)
#   PYTHON must point to a Python with girth 0.8.0, numpy, scipy
#
# All reads are from inputs/ under this directory; fails if any read
# escapes the repo/ tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
READER_DIR="$SCRIPT_DIR"
REPO_DIR="$(cd "$READER_DIR/.." && pwd)"

INPUT_DIR="$READER_DIR/inputs"
OUTPUT_DIR="$READER_DIR/outputs"

# Resolve tool paths from environment or defaults
PYTHON="${PYTHON:-python3}"
R_LIBS="${R_LIBS:-}"

echo "============================================================"
echo "T3 Reproduction Pipeline"
echo "============================================================"
echo "Reader dir:  $READER_DIR"
echo "Input dir:   $INPUT_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Python:      $PYTHON"
echo "R_LIBS:      $R_LIBS"
echo "============================================================"

# Verify inputs exist
for f in response_matrix_code.csv response_matrix_math.csv \
         response_matrix_science.npz geometry_composite_50.json \
         covariates_50.json science_features_native.json \
         theta_reference_panel51.json theta_reference_science.json; do
    if [ ! -f "$INPUT_DIR/$f" ]; then
        echo "FATAL: Missing input: $INPUT_DIR/$f"
        exit 1
    fi
done

# Clean output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Step 1: Re-fit theta for code and math (R + mirt) ──────────────
echo ""
echo "STEP 1: Re-fitting code/math theta via R mirt..."
THETA_CM_DIR="$OUTPUT_DIR/theta_codemath"
R_LIBS="$R_LIBS" Rscript "$READER_DIR/scripts/fit_theta_codemath.R" \
    "$INPUT_DIR" "$THETA_CM_DIR"

if [ ! -f "$THETA_CM_DIR/theta_2pl_code.csv" ] || \
   [ ! -f "$THETA_CM_DIR/theta_2pl_math.csv" ]; then
    echo "FATAL: R script did not produce theta CSVs"
    exit 1
fi
echo "  -> code/math theta written to $THETA_CM_DIR"

# ── Step 2: Re-fit theta for science (Python + girth) ──────────────
echo ""
echo "STEP 2: Re-fitting science theta via girth..."
THETA_SCI_DIR="$OUTPUT_DIR/theta_science"
"$PYTHON" "$READER_DIR/scripts/fit_theta_science.py" \
    "$INPUT_DIR" "$THETA_SCI_DIR"

if [ ! -f "$THETA_SCI_DIR/theta_science.json" ]; then
    echo "FATAL: Science theta script did not produce output"
    exit 1
fi
echo "  -> science theta written to $THETA_SCI_DIR"

# ── Step 3: Compute associations and compare to targets ────────────
echo ""
echo "STEP 3: Computing associations and comparing to paper targets..."
ASSOC_DIR="$OUTPUT_DIR/associations"
"$PYTHON" "$READER_DIR/scripts/compute_associations.py" \
    "$INPUT_DIR" "$THETA_CM_DIR" "$THETA_SCI_DIR" "$ASSOC_DIR"

echo ""
echo "============================================================"
echo "T3 pipeline complete. All outputs in: $OUTPUT_DIR"
echo "============================================================"
