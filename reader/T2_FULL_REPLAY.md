# T2 (optional) — Full cached replay of the entire analysis graph

The **T3** pipeline (`reader/run_t3.sh`) reproduces the paper's *main* tables from small shipped inputs and is the recommended route. **T2** additionally re-runs the *entire* accepted analysis graph (all supplementary tables/figures), but it needs a large cached-input bundle that is **not** contained in this repository.

## What T2 needs that this repo does not ship
- The frozen cached inputs for the 10-stage replay: **3,094 external source files, ≈ 4.03 GiB** (response/observation records, saved per-draw bootstrap/permutation arrays, geometry caches, saved fit outputs). See `reproduction/large_data/MATERIALIZATION_MANIFEST.json` for the manifest and per-file SHA-256 digests.
- The two pinned virtual environments (Python 3.12.11: a numeric env — numpy 2.5.0 / scipy 1.18.1 / girth 0.8.0 / sympy 1.14.0 — and a renderer env — matplotlib 3.11.0 / numpy 2.5.0). See `reproduction/config/env/`.

## Status of the large bundle
This bundle is **not currently published**, and no public download URL is provided. Reasons and prerequisites for a future public bundle:
- **Redistribution rights** for the underlying external benchmark items and raw model outputs are **not established** (see `docs/DATA_AVAILABILITY.md`). T3 avoids this because it ships only 0/1 response matrices and our own derived vectors; a full-bundle release must clear item/output redistribution first.
- A public T2 release would require: a **versioned, downloadable archive** (e.g. Zenodo with a DOI, or a Hugging Face dataset), its **layout + SHA-256 checksums**, redistribution clearance, and a check that runs the graph **from that published archive**.

## For a holder of the exact source trees
If you already possess the matching source trees, the replay is driven by the packaged tools (unchanged):
```bash
cd reproduction
python3 code/rfinal_replay/tools/materialize_sources.py --source-project <P> --source-analysis <R> --dest <D>
python3 code/rfinal_replay/tools/materialize_sources.py --dest <D> --verify-only     # must be 0 missing / 0 mismatch
bash scripts/rebuild_env.sh <D> <python3.12.11>
bash scripts/run_cached_replay.sh <D> <D>/project/repro_paper2_20260915/recovered/c17_replay_env/venv-cpu-render/bin/python <OUT>
```
The materializer is a local copy+verify tool, **not** a downloader; it will refuse hash mismatches.

## What T2 adds over T3
T2 regenerates every accepted table/figure (convergence, LOFO, the rarefied science grid, depth profiles, MTMM, supplements) and runs the engine's own cell-by-cell validation. It still begins from saved intermediates (θ, per-draw arrays, geometry caches), i.e. it is not a from-model-weights run.
