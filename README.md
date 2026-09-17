# Two Instruments, One Capability

**A Convergent-Validity Audit of Behavioral IRT Ability and Representation Geometry in LLMs**

Companion code, saved result sources, and reader documentation for the HICSS Submission 12370 manuscript. The manuscript and supplement are supplied separately by the authors; their PDFs and LaTeX sources are not distributed in this code repository.

Canonical repository: **[PsyCogMetricsAI/llm-ability-geometry](https://github.com/PsyCogMetricsAI/llm-ability-geometry)**. To cite the accompanying manuscript, use the metadata in [CITATION.cff](CITATION.cff).

## The research question

Does the geometry of an LLM's hidden states correspond to ability estimated from its answers? If it does, how does that relationship depend on the task domain, geometric indicator, and adjustment for model characteristics, and does it contain information beyond raw accuracy?

This matters for researchers and evaluation practitioners using internal representations to interpret capability. A geometric signal associated with performance may help describe model differences. Its usefulness as an ability measure depends on what it adds to behavioral scores and on which comparison conditions support the association.

## Where this study fits

IRT research estimates ability from item-level response patterns. Representation research relates spectral structure and intrinsic dimension to model performance. Other studies already compare behavioral and internal representations directly. This paper addresses a specific remaining measurement question at the **model level**: how global geometric summaries relate to IRT-derived ability, and what remains after accounting for recorded covariates and accuracy. See the paper's Literature Review for the cited studies and their respective comparison levels.

## What we did

Code and mathematics each include 50 models from 13 families; exploratory science includes 43 models from 13 families. Code/math use a saved geometric composite; science compares individual geometric indicators. For each domain we compare:

1. Raw rank association between geometry and IRT ability.
2. Association after adjustment for recorded model covariates.
3. Association after also conditioning on raw accuracy.

Family-block intervals and sensitivity checks describe uncertainty and dependence on the comparison setting. Medical alternative estimators and a new six-model window analysis are supplementary sensitivities.

## Findings and contribution

- **Adjustment changes the picture.** The positive mathematics association strengthens after adjustment; the negative code association weakens.
- **Correspondence depends on the indicator and domain.** Several science spectral indicators are associated with ability. Participation ratio retains its negative direction when individual families are omitted, holding saved ability estimates fixed.
- **Accuracy clarifies the interpretation.** Ability and accuracy produce closely overlapping rankings. The examined accuracy-conditioned intervals include zero.

The contribution is an empirical account of **where geometry–ability correspondence appears, how the comparison changes it, and what the accuracy comparison permits us to infer**. The evidence supports geometry as a domain-dependent internal correlate of measured performance, without establishing a stable accuracy-independent ability scale across domains and indicators. It is complementary evidence for interpreting observed model differences, not a validated replacement for behavioral evaluation.

Important scope conditions include nonconverged code/math IRT fits, differing science estimation/extraction and missing-response conventions, and only 13 families. See [the reading guide](docs/PAPER_GUIDE.md#interpretation-and-scope) and the paper's limitations for their implications.

## Start here

| To… | Open or run |
|---|---|
| Follow the argument section by section | [Paper guide](docs/PAPER_GUIDE.md) |
| Trace the current tables to accepted results | [Evidence map](docs/EVIDENCE_MAP.md) |
| Verify files and export table displays | [Reproduction instructions](docs/REPRODUCING.md) |
| Understand input availability and reuse scope | [Data availability](docs/DATA_AVAILABILITY.md) |

From the repository root:

```bash
python3 scripts/verify_release.py
python3 scripts/export_paper_tables.py --output outputs/paper_tables
python3 reproduction/examples/run_depth_example.py
```

These are distinct checks. The verifier checks release files against their manifest. The exporter reconstructs table displays from accepted saved JSON results; it does **not** refit models or recompute intervals. The depth example recomputes an ancillary summary, not the paper's main associations.

### Full cached replay requires additional inputs

This repository is a curated code distribution of the accepted replay materials under [`reproduction/`](reproduction/README_RUN.md), not an intact copy of the accepted archive. Generated displays, private machine paths, internal review/run records and interpreter caches are omitted. `RELEASE_MANIFEST.json` describes the current distribution. Its full cached replay requires **3,094 unique external source files (4,325,608,929 bytes, approximately 4.03 GiB)**. Those payloads are not bundled, and no public download URL is currently supplied. Cloning this repository alone therefore does **not** enable full cached replay. Readers who already possess the exact source trees can follow [the documented conditional replay route](docs/REPRODUCING.md#full-cached-replay-for-holders-of-the-exact-source-trees).

Replay begins from specified saved data and intermediate results, with different starting points across analyses. Original model inference and restoration of every historical experiment are outside this release's reproduction claim.
