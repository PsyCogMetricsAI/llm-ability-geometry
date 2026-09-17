"""Fixture: exercise new_medical_repair.fit() on a tiny synthetic response matrix.

Legal lightweight path: the numerical routine, its objective, quadrature, gradient
check and on-disk outputs are the frozen ones; only the input size is reduced.
"""

import json
import sys
from pathlib import Path

import numpy as np

import new_medical_repair as producer


def main() -> int:
    label = "fixture_smoke_2pl"
    rng = np.random.default_rng(20260917)
    y = (rng.random((3, 12)) < 0.5).astype(float)
    producer.OUT.mkdir(parents=True, exist_ok=True)
    report = producer.fit(y, "2pl", 1.0, label)
    assert (producer.OUT / f"{label}.npz").is_file(), "fit did not write its arrays"
    assert (producer.OUT / f"{label}.json").is_file(), "fit did not write its report"
    payload = json.loads((producer.OUT / f"{label}.json").read_text())
    summary = {
        "fixture": "fixture_medical_fit",
        "label": label,
        "shape": list(y.shape),
        "numeric_screen_pass": payload["numeric_screen_pass"],
        "gradient_check_maxabs": payload["gradient_check_maxabs"],
        "logML": payload["logML"],
        "cwd": str(Path.cwd()),
    }
    (producer.OUT / "fixture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
