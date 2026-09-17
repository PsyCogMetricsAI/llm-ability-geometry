"""Fixture: exercise the study1 numerical core (rho/bootstrap/permute) on a tiny panel."""

import json
from pathlib import Path

import numpy as np

import study1_reuse_analyze as producer


def main() -> int:
    rng = np.random.default_rng(20260605)
    n = 26
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=n)
    controls = rng.normal(size=(n, 4))
    families = np.arange(n) % 13
    draws = np.array([rng.choice(13, size=13, replace=True) for _ in range(24)])
    statistic, values = producer.bootstrap(x, y, controls, families, draws)
    permutation, pvalues = producer.permute(x, y, controls, np.array([rng.permutation(n) for _ in range(24)]))
    summary = {
        "fixture": "fixture_study1_core",
        "rho": producer.rho(x, y),
        "partial_rho": producer.rho(x, y, controls),
        "bootstrap_rho": statistic["rho"],
        "bootstrap_values": int(np.asarray(values).shape[0]),
        "permutation_p": permutation["p"],
        "permutation_values": int(np.asarray(pvalues).shape[0]),
        "sandbox_root": str(Path.cwd()),
    }
    out = Path("runs/study1_reuse_v1/fixture")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
