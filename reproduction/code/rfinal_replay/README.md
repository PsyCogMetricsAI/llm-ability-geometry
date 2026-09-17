# Scientific replay engine

Use the [package guide](../../README_RUN.md) and [repository reproduction guide](../../../docs/REPRODUCING.md). Pass explicit project and analysis roots and the public relocated configuration to `entrypoint.py`. Its `plan` command is read-only; `run --execute` requires exact scientific input hashes, frozen tolerances and an empty output directory.

Inputs are allowlisted and digest checked. Producers run in isolated output sandboxes. The engine verifies source-tree immutability, collects fresh outputs and validates the scientific comparison and claim map. Internal preparation histories are not distributed.
