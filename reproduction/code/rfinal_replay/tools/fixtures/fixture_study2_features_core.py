"""Fixture: exercise study2 feature extraction on a tiny legal cloud.

The indicator library is loaded through the migrated contract path, so this also
verifies that the absolute-path migration resolved inside the sandbox.
"""

import json
from pathlib import Path

import numpy as np

import study2_reuse_features as producer


def main() -> int:
    rng = np.random.default_rng(20260610)
    cloud = rng.normal(size=(60, 5))
    module = producer.library()
    values = producer.feature(module, cloud)
    summary = {
        "fixture": "fixture_study2_features_core",
        "metrics": dict(zip(producer.NAMES, values)),
        "library_file": producer.library().__name__,
        "contract": str(producer.CONTRACT),
        "cwd": str(Path.cwd()),
    }
    out = Path("runs/study2_reuse_v1/features_fixture")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
