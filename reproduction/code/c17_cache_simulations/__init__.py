"""Cached-simulation replay for retained AN-B, C4 and C6 evidence (C10s).

Reads hash-declared per-replicate / per-draw cache files only; never reruns the
5000-unit C4 sweep or the 7900-draw C6 sweep and never writes outside the
caller-supplied output directory.
"""
