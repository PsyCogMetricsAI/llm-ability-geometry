"""Fixture: attempt a write outside the sandbox (must be refused by audit_runner)."""

import sys

target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rfinal_fixture_escape.txt"
with open(target, "w") as handle:
    handle.write("this write must be refused\n")
print("UNREACHABLE_WRITE_SUCCEEDED")
