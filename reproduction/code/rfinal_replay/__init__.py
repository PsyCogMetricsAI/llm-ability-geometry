"""RFINAL clean-replay package (paper2 round repro_paper2_20260915).

Preparation-time package for the later authorized clean replay (R6). It contains a
path-safe launcher, input allowlist guards, a run-manifest writer, a declared
regeneration plan and executor self-tests. It performs no analysis and no GPU work.
"""

__all__ = ["guards", "manifest", "stages", "sandbox_runner", "launcher"]

SCHEMA = "paper2-rfinal-replay-draft-v1"
PACKAGE = "code/rfinal_replay"
