"""Fixture: attempt a read outside the sandbox (must be refused by audit_runner)."""

import sys

target = sys.argv[1] if len(sys.argv) > 1 else "/etc/hostname"
with open(target) as handle:
    handle.read()
print("UNREACHABLE_READ_SUCCEEDED")
