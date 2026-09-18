#!/usr/bin/env python3
"""Re-fit 2PL IRT theta for the science domain using girth.

Usage:
    python scripts/fit_theta_science.py inputs/ outputs/theta_science/

Reads:  inputs/response_matrix_science.npz
        (keys: matrix [448 items x 43 models], fit_mask [448 bool])
Writes: outputs/theta_science/theta_science.json
        (theta list [43], discrimination/difficulty arrays, fit summary)
"""
import json
import os
import sys

import numpy as np
from girth import twopl_mml, ability_eap


def main():
    if len(sys.argv) < 3:
        print("Usage: python fit_theta_science.py <input_dir> <output_dir>",
              file=sys.stderr)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(input_dir, "response_matrix_science.npz")
    data = np.load(npz_path)
    matrix = data["matrix"]       # (448, 43) items x models
    fit_mask = data["fit_mask"]    # (448,) bool

    print(f"[science] matrix shape: {matrix.shape}, fit_mask sum: {fit_mask.sum()}")

    # Subset to fit_mask items
    X = matrix[fit_mask, :]  # (447, 43)
    print(f"[science] fitting {X.shape[0]} items x {X.shape[1]} models")

    # girth expects [items x participants], X is already that shape
    result = twopl_mml(X)
    discrimination = result["Discrimination"]
    difficulty = result["Difficulty"]
    theta = ability_eap(X, difficulty, discrimination)

    print(f"[science] theta range: [{theta.min():.6f}, {theta.max():.6f}]")

    result = {
        "theta": theta.tolist(),
        "n_items_fit": int(fit_mask.sum()),
        "n_models": int(X.shape[1]),
        "discrimination_range": [float(discrimination.min()),
                                 float(discrimination.max())],
        "difficulty_range": [float(difficulty.min()),
                             float(difficulty.max())],
    }

    out_path = os.path.join(output_dir, "theta_science.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[science] wrote {out_path}")
    print("THETA-SCIENCE-DONE")


if __name__ == "__main__":
    main()
