"""Fixture: exercise new_medical_associations.stats() on a tiny synthetic matrix."""

import json
from pathlib import Path

import numpy as np

import new_medical_associations as producer


def main() -> int:
    rng = np.random.default_rng(20260919)
    n, nfeat = 24, 1
    # columns: [total rank input, MAP EAP, feature..., C1..C4, fit accuracy]
    matrix = np.column_stack(
        [
            rng.integers(0, 5, size=n).astype(float),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=(n, 5)),
        ]
    )
    values, statuses = producer.stats(matrix, nfeat)
    summary = {
        "fixture": "fixture_medical_associations_stats",
        "values_shape": list(values.shape),
        "status_shape": list(statuses.shape),
        "finite_values": int(np.isfinite(values).sum()),
        "status_codes": sorted({int(v) for v in np.unique(statuses)}),
        "status_legend": producer.STATUS,
        "cwd": str(Path.cwd()),
    }
    (Path("runs/new_medical_associations_fixture") / "fixture_summary.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (Path("runs/new_medical_associations_fixture") / "fixture_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
