"""Fixture: exercise new_medical_repair_refined.fit() on a tiny synthetic matrix."""

import json
from pathlib import Path

import numpy as np

import new_medical_repair_refined as producer


def main() -> int:
    label = "fixture_smoke_map"
    rng = np.random.default_rng(20260918)
    y = (rng.random((3, 12)) < 0.5).astype(float)
    producer.OUT.mkdir(parents=True, exist_ok=True)
    report = producer.fit(y, "map", 1.0, label)
    assert (producer.OUT / f"{label}.npz").is_file()
    payload = json.loads((producer.OUT / f"{label}.json").read_text())
    summary = {
        "fixture": "fixture_medical_refined_fit",
        "label": label,
        "shape": list(y.shape),
        "numeric_screen_pass": payload["numeric_screen_pass"],
        "all_finite_flag": payload["finite_theta"],
        "cwd": str(Path.cwd()),
    }
    (producer.OUT / "fixture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
