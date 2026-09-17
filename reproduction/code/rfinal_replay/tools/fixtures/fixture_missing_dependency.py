"""Fixture: import a package that is not installed (dependency-failure refusal)."""

import rfinal_nonexistent_dependency_xyz  # noqa: F401

print("UNREACHABLE_IMPORT_SUCCEEDED")
