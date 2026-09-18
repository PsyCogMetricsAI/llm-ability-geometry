# Data and artifact availability

The code and reader materials are publicly available at [PsyCogMetricsAI/llm-ability-geometry](https://github.com/PsyCogMetricsAI/llm-ability-geometry). The approximately 4.03 GiB source bundle required for full cached replay is not publicly distributed with this release.

## Included in this repository
- **End-to-end (T3) inputs** under `reader/inputs/` (< 0.5 MB total): the 0/1 IRT response matrices (code/math/science), our derived per-model geometry summaries (composite; native science indicators), model covariates, and reference θ. These are 0/1 matrices or our own derived vectors — **not** raw benchmark item text or raw model outputs — so they avoid the redistribution concerns that block the large bundle. They let a reader re-fit θ and recompute Tables 1–2 (see [REPRODUCING.md](REPRODUCING.md) §0).


- Small accepted result JSON files used by the table exporter under `results/sources/`, mapped in [EVIDENCE_MAP.md](EVIDENCE_MAP.md).
- Reader-facing verification and table-export scripts.
- Selected C17 delivery materials under `reproduction/`, including code, configurations, comparison references, evidence records, a source materialization manifest, and the standalone depth example.

This is a curated code distribution, not the intact accepted archive. Generated display artifacts and internal execution records are omitted; `RELEASE_MANIFEST.json` describes the distributed files. The manuscript and supplement are supplied separately by the authors; neither their PDFs nor their LaTeX sources are included.

The saved result summaries support inspection and table reconstruction. They are not the full underlying response matrices, activation data, or bootstrap inputs.

## Required for full cached replay, but not bundled

The local input inventory resolved **3,094 unique source files totaling 4,325,608,929 bytes (4.0285 GiB)**. These include response/item records, activation arrays, derived caches, saved results, and other manifest-bound historical inputs. The inventory established local existence and size; it was not a new hash verification of the entire 4 GiB collection.

The materialization manifest is under [`reproduction/large_data/`](../reproduction/large_data/). It specifies the inputs and hashes. The archive's materializer checks and copies files from an existing project root and analysis root into the replay layout. **It is not a downloader.** No public retrieval URL for that exact source bundle is currently supplied, and independent acquisition instructions sufficient to recreate those exact cached files are not available here.

Consequently, a public clone supports the documented inspection, export and smoke-example levels, but not full cached replay by itself. The full replay instructions are conditional on already possessing the matching source trees. See [REPRODUCING.md](REPRODUCING.md).

## Model weights and historical materials

Model weights and credentials are not part of this release. Replaying the documented cached analyses has a different starting point from rerunning original model inference.

Some required archived inputs reference older manuscript files, logs and cached artifacts because the frozen replay configuration addresses them. They are historical inputs, not reading versions of the current manuscript or supplement. Those documents are supplied separately by the authors.

## Redistribution and license status

No new license is assigned by this release documentation. A public code license has not been specified here. The input inventory did not establish blanket redistribution rights for the external benchmark items, model outputs or other source payloads; it also did not determine that any particular file is prohibited from distribution. Availability of saved summaries should not be interpreted as a license for every underlying dataset or model.

A stronger public full-replay release would require a versioned, accessible source bundle, its layout and checksums, appropriate redistribution clearance, and a check using that distributed bundle. Those conditions are not represented as complete in this reader release.
