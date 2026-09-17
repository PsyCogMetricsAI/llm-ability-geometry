# Reproducing and inspecting the results

Clone the [public repository](https://github.com/PsyCogMetricsAI/llm-ability-geometry) and enter its root directory:

```bash
git clone https://github.com/PsyCogMetricsAI/llm-ability-geometry.git
cd llm-ability-geometry
```

Run the initial commands below from this directory. The levels are separate: checking files, exporting accepted results, recomputing an ancillary example, and replaying the accepted analysis graph establish different things.

## 1. Verify the published files

```bash
python3 scripts/verify_release.py
```

This checks the selected release files against `RELEASE_MANIFEST.json`. It verifies package integrity; it does not independently validate scientific conclusions. Use Python 3 for this standard-library reader tool.

## 2. Reconstruct the current paper's table displays

```bash
python3 scripts/export_paper_tables.py --output outputs/paper_tables
```

The exporter reads accepted JSON files in `results/sources/` and formats the paper's tables. It uses the native science indicators and the matched accuracy diagnostics. It does not run IRT estimation, calculate new correlations, or generate new bootstrap intervals. See [EVIDENCE_MAP.md](EVIDENCE_MAP.md) for the result-level mapping.

The corrected code/math accuracy-conditioned intervals are accepted earlier diagnostics retained in the paper; they should not be described as newly recomputed C17 intervals. Medical alternative-estimator results are separate from the three main panels.

## 3. Run the standalone smoke example

```bash
python3 reproduction/examples/run_depth_example.py
```

This requires no external source bundle. It recomputes a saved G4 depth-profile summary and peaks through the archived aggregator and compares them with the pinned reference. Expected output reports `status: PASS`, 20 rows and 3 peaks, with maximum difference at floating-point noise level. It writes its result to standard output.

This is an ancillary calculation and comparator check, not reproduction of the three primary geometry–ability comparisons.

## Full cached replay for holders of the exact source trees

**Availability prerequisite:** the required cached scientific inputs are not included in this repository, and no public download URL is currently provided. This procedure is usable only if you already have the exact source trees matching the materialization manifest. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). The materializer copies and verifies local files; it does not acquire them from the internet.

This public distribution retains the scientific replay graph, source digests and comparison data. Internal execution records and private machine paths have been removed. `RELEASE_MANIFEST.json` describes the distributed files. The manuscript and supplement are supplied separately.

The replay commands use:

- A Unix-style shell and virtual-environment paths; native Windows execution has not been checked.
- A separately supplied **Python 3.12.11** interpreter with virtual-environment support.
- Both locked environments from `reproduction/config/env/`: renderer and numeric replay.
- Existing project and analysis source roots containing the exact manifest-listed inputs.
- Separate destination and output directories with sufficient space. Full replay runtime and peak memory have not been measured for this reader release.

From `reproduction/`, first inspect the plan. Replace `/path/to/...` values with your own real local paths; they are not download locations.

```bash
cd reproduction
python3 code/rfinal_replay/tools/materialize_sources.py \
  --dest /path/to/replay-destination --dry-run
```

Then follow the archived commands with your existing source project and analysis roots:

```bash
python3 code/rfinal_replay/tools/materialize_sources.py \
  --source-project /path/to/existing/project-root \
  --source-analysis /path/to/existing/project-root/repro_paper2_20260915 \
  --dest /path/to/replay-destination
python3 code/rfinal_replay/tools/materialize_sources.py \
  --dest /path/to/replay-destination --verify-only
bash scripts/rebuild_env.sh \
  /path/to/replay-destination /path/to/python-3.12.11
bash scripts/run_cached_replay.sh \
  /path/to/replay-destination \
  /path/to/replay-destination/project/repro_paper2_20260915/recovered/c17_replay_env/venv-cpu-render/bin/python \
  /path/to/replay-output
```

The destination uses `project/` as its project root and `project/repro_paper2_20260915/` as its analysis root. The materializer overlays packaged code/configuration/comparison files and verifies declared source digests. The environment scripts enforce their exact pins. A mismatch should be investigated against the specified input or environment, rather than bypassed by changing the scientific configuration.

The archived graph contains analyses beyond the current manuscript. Its final rendering is not a substitute for the paper-specific source mapping, especially for native versus rarefied science features.

## What the replay claim covers

The accepted work completed actual calculations, underwent independent checkpoint verification, and continued after repairs to downstream integration code. Analyses begin from different saved starting points. This is not a single uninterrupted end-to-end regeneration from original model inference, and it does not restore every historical experiment.

The archive preserves limitations and failed checks in [`reproduction/LIMITATIONS.md`](../reproduction/LIMITATIONS.md). The manuscript's supplement explains their scientific relevance. No new model inference is needed for the reader checks above.
