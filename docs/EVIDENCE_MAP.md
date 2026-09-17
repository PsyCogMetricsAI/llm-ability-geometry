# From the paper to results and code

All links below resolve within this repository. `results/SOURCE_MAP.json` records
hashes of the publicly distributed saved summaries. Private machine paths and internal review records have been removed.

## Current paper tables

| Paper object | Included evidence | Fields / interpretation |
|---|---|---|
| Main Table 1, code/math raw, adjusted, theta–accuracy | [study1_baseline.json](../results/sources/study1_baseline.json) | `domains/code` and `domains/math`: `raw_rho`, `partial_rho_controlling_C`, `controlled_cluster_ci`, `theta_accuracy_rho` |
| Main Table 1, code/math accuracy-conditioned | [study1_accuracy_v2.json](../results/sources/study1_accuracy_v2.json) | `domains/*/q5_rho_g_t__a_C`: corrected `rho` and `ci`; retained separately accepted earlier diagnostics |
| Main Table 1 science PR and Table 2 native indicators | [native science files](../results/sources/science/) | `F1_MAP__native__science__*.json`: `raw_rho`, `partial_rho`, `ci_controlled`; actual estimator is science girth EAP despite the MAP filename family |
| Main Table 1 science accuracy and the six accuracy diagnostics | [science_accuracy.json](../results/sources/science_accuracy.json) | science `cells`: `spearman_theta_accuracy` and `acc_plus_C`; no native TwoNN accuracy-conditioned test |
| Supplement S2 medical table | [medical_associations.json](../results/sources/medical_associations.json) | `rows` with panel `A`, feature `native_eff_rank_pr`, score `MAP`/`Rasch`, modes `raw`/`partial4C`/`partial4C_plus_fitaccuracy`; `point` and `CI95` |

Run `python3 scripts/export_paper_tables.py --output outputs/paper_tables` from
the repository root. It creates three CSVs and three Markdown tables, preserving
the paper's three-decimal display and undefined medical result. This **exports
accepted estimates**; it is not refitting or statistical reanalysis.

The medical CIs come from the separate descriptive association report, not the
larger 28-cell grid. The current science table uses **native** clouds, whereas
some results in the accepted replay archive use rarefied clouds. Keep these versions separate. Generated display artifacts from that archive are omitted from this code distribution.

## Supporting evidence included here

| Statement | Included source |
|---|---|
| Calibration of family-bootstrap intervals | [panel_calibration.json](../results/sources/panel_calibration.json) |
| Model-based EAP / geometry reliability, drop-logprob and cross-domain comparisons | [study1_baseline.json](../results/sources/study1_baseline.json) |
| Separate fitted-half item reliability | [split_item_reliability.json](../results/sources/split_item_reliability.json), claim `P2-A-007` |
| Separate 31-model disjoint-item protocol | [disjoint_item_protocol.json](../results/sources/disjoint_item_protocol.json) |
| Science family omission with fixed ability | [science_family_omission.json](../results/sources/science_family_omission.json) |
| Science cloud-size adjustment | [science_accuracy.json](../results/sources/science_accuracy.json), `sign_diagnostics` |
| Science rarefied PR sensitivity | [rarefied science PR](../results/sources/science/F1_MAP__rarefied__science__eff_rank_pr.json) |
| Original medical fit failure | [medical_original_fit.json](../results/sources/medical_original_fit.json) |
| New six-model fixed-version windows | [window_sensitivity.json](../results/sources/window_sensitivity.json) |
| Retained strict CPU/GPU failures and historical limits | [archived limitations](../reproduction/LIMITATIONS.md); the supplement is supplied separately by the authors |

## Code navigation

[analysis_sources/README.md](../analysis_sources/README.md) identifies the exact
mirt/girth, corrected accuracy, regularized 2PL/Rasch, disjoint-item and window
implementations. These source references retain their scientific calculations and relative dependencies; they are provided for inspection rather than advertised as
standalone scripts. The runnable cached replay entrypoints are in
[REPRODUCING.md](REPRODUCING.md).

The archived replay drivers under `reproduction/code/` compute from differing
saved inputs. Their input manifests, configurations, comparisons are
kept together. The accepted archive covers more analyses than the narrowed current
paper; ancillary outputs are not additional claims of the main manuscript. This
curated distribution omits generated display artifacts and internal execution records.

