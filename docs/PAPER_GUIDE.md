# Reading the paper

The manuscript and supplement are supplied separately by the authors and are not included in this code repository. This guide explains their argument and the role of the supporting checks; the [evidence map](EVIDENCE_MAP.md) identifies the included saved results behind the tables.

## 1. Introduction: what would an internal ability signal add?

The motivating question is whether hidden-state geometry corresponds to ability inferred from model answers. Accuracy summarizes correct responses. IRT estimates ability from item-response patterns, accounting for item characteristics. Geometry describes properties of hidden representations. Comparing these quantities can clarify whether an internal signal adds an interpretation specific to ability or primarily tracks observed performance.

The empirical contributions are the science spectral associations, the contrasting code/math patterns, and the close overlap with accuracy that bounds an ability-specific interpretation. Existing work already connects behavioral and internal evidence.

## 2. Literature Review and Theory: define the comparison

The literature review moves from behavioral psychometrics to representation geometry and existing behavioral–internal comparisons. The distinction here is the variables and analysis level: model-level IRT scores versus global geometric summaries.

Theory introduces a common-capability hypothesis as a question to investigate. Its most consequential point is the accuracy boundary: when ability and accuracy rank models almost identically, a geometry–ability correlation alone gives little basis for attributing information specifically to the IRT estimate. That motivates the accuracy-conditioned test.

## 3. Methodology: three questions on the same model panel

For domain `d`, the quantities are ability `theta_d`, geometric indicator `G_d,j`, accuracy `A_d`, and recorded model covariates `Z`.

| Comparison | What it asks |
|---|---|
| Raw association | Do geometry and ability vary together across models? |
| Association conditional on `Z` | How does the relationship change with recorded model characteristics accounted for? |
| Association conditional on `A_d` and `Z` | Is there additional association with the ability ranking beyond accuracy? |

These are observational rank comparisons. Code/math use a geometric composite; science uses individual indicators. Adjustment includes size, representation width, a numeric family code, and mean log-probability. The numeric family code is not categorical family fixed effects. Family-block resampling separately addresses family dependence.

## 4. Results: three empirical findings

The first finding is that geometry and IRT ability correspond in some settings. The code/math composite is associated with ability before adjustment, while several spectral indicators are associated with ability in exploratory science. Table 1 compares the domain-specific estimates; Table 2 reports all seven science indicators.

The second finding is that the correspondence changes with the measurement and evaluation setting. The composite's positive association in mathematics strengthens after covariate adjustment, while its negative association in code weakens. In science, five spectral summaries have adjusted intervals excluding zero, while RankMe and TwoNN do not. These contrasts combine differences in domains, indicators, extraction and panels. Family-omission and covariate-specification checks provide supporting diagnostics, with detailed sensitivity ranges in Supplement S3.

The final finding concerns information beyond accuracy. Ability and accuracy rankings closely overlap, and the examined accuracy-conditioned intervals include zero. These comparisons cover the code/math composite and six science spectral diagnostics; they do not include a native TwoNN accuracy-conditioned test. The alternative medical estimators are reported separately in Supplement S2.

## 5. Discussion: interpretation and scope

### Interpretation and scope

The results support geometry as a domain-dependent internal correlate of measured performance. An accuracy-independent relationship with the IRT ability estimate remains unestablished. The zero-inclusive intervals leave the magnitude and direction of additional association uncertain.

The most important conditions are:

- Code/math IRT fits returned finite scores but did not meet the convergence criterion. Reliability estimates do not resolve this fitting issue.
- Science is exploratory, with different estimation and extraction settings. Its accuracy uses available responses, whereas its IRT matrix codes missing responses as incorrect.
- There are 13 families. Bootstrap intervals condition on saved ability scores and geometry rather than refitting and re-extracting them.
- Science indicators overlap in information, and their individual intervals are not independent replications or family-wide multiplicity correction.
- Domain comparisons also change measurements and panels. Covariate adjustment describes conditional association, not causal mechanisms.

The practical role supported here is complementary description of model performance within specified evaluation settings. Broader measurement or predictive uses require validation in additional settings.

## 6. Supplement: supporting checks and provenance

The supplement contains the Rasch ranking conditions, calibration and reliability results, medical estimator alternatives, detailed family-omission and other sensitivities, window checks, and analysis provenance. Medical regularized 2PL results are descriptive alternative-estimator evidence; Rasch's accuracy-conditioned statistic is undefined when residual ability ranks have no variation. Undefined is not zero.

The six-model fixed-version window analysis is new sensitivity evidence, not recovery of the original historical experiment. The strict Pythia CPU/GPU test failed in 41 of 43 comparisons at the specified tolerance. The provenance note identifies differing replay starting points and retained corrected diagnostics. These details define the documented reproducibility scope without replacing the main scientific argument.

## Navigating the code alongside the paper

Start with [EVIDENCE_MAP.md](EVIDENCE_MAP.md), then export the table displays using [REPRODUCING.md](REPRODUCING.md). The accepted replay archive covered analyses beyond the narrowed current manuscript. Its generated display artifacts are omitted from this code distribution. In particular, rarefied science results must not be substituted for the manuscript's native indicators.
