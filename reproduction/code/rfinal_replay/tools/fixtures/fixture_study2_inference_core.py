"""Fixture: exercise the study2 inference core (rho/bootstrap/permute) on a tiny panel."""

import json
from pathlib import Path

import numpy as np

import study2_reuse_inference as producer


def main() -> int:
    rng = np.random.default_rng(20260611)
    n = 26
    x = rng.normal(size=n)
    y = -0.6 * x + rng.normal(size=n)
    controls = rng.normal(size=(n, 4))
    families = np.arange(n) % 13
    draws = np.array([rng.choice(13, size=13, replace=True) for _ in range(24)])
    permutation = np.array([rng.permutation(n) for _ in range(24)])
    controlled, _bootstrap_draws = producer.bootstrap(x, y, controls, families, draws)
    permuted, _permutation_draws = producer.permute(x, y, controls, permutation)
    clipped = producer.bh([0.01, 0.2, 0.5])
    summary = {
        "fixture": "fixture_study2_inference_core",
        "raw_rho": producer.rho(x, y),
        "controlled_rho": controlled["rho"],
        "permutation_p": permuted["p"],
        "bh": clipped,
        "cwd": str(Path.cwd()),
    }
    out = Path("runs/study2_reuse_v1/inference_fixture")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
