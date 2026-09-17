# Cached scientific replay

The public package contains replay modules, pinned configuration, scientific comparison CSVs, source manifests and a standalone depth example. Internal run records, review receipts and interpreter caches are excluded. File integrity is recorded by the repository's `RELEASE_MANIFEST.json`.

See [the reproduction guide](../docs/REPRODUCING.md) for the complete commands and exact Python/package pins. The full cached replay needs separately supplied scientific source trees; no download URL is currently supplied.

From this directory:

```bash
python3 -B examples/run_depth_example.py
python3 -B code/rfinal_replay/tools/materialize_sources.py --dest <DEST> --dry-run
python3 -B code/rfinal_replay/tools/materialize_sources.py --dest <DEST> --plan-check
```

The runnable config is `config/CT_C17_REPLAY_FINAL_v1.relocated.json`. It retains all ten active stages, including result comparison and the 147-record claim-map check (139 mandatory records). Source hashes, tolerances, statistical parameters and scientific limitations remain enforced. Execution requires explicit `--execute`, exact inputs and fresh output directories.
