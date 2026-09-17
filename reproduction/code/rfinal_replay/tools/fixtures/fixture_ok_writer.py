"""Fixture: legal producer that writes only inside the sandbox."""

import json
import sys
from pathlib import Path

payload = {"argv": sys.argv[1:], "cwd": str(Path.cwd())}
target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/fixture_ok/output.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2) + "\n")
print("FIXTURE_OK")
