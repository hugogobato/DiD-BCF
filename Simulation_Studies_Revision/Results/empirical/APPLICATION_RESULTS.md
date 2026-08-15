# The `mpdta` application under the corrected estimator

Refit of `Empirical_Study/DiD_BCF_GATE_CATE_Empirical_Study.ipynb` under the
structured DiD-BCF that the simulation grid uses. Reproduce with
`scripts/run_empirical_structured.py`, or read
`Empirical_Study/DiD_BCF_GATE_CATE_Empirical_Study_structured.ipynb`, which is
the same analysis with commentary.

Why refit at all: the published specification's estimate is not reproducible
across MCMC seeds (`ATT_SEED_INSTABILITY.md`). The structured model removes the
flat likelihood direction that causes it, and the same three seeds then agree to
four decimal places.

## Headline

`structured_rfx_unit`, seed 0, 291 treated observations.

| group | n | effect | 95% CI | `p_bayes` |
|---|---|---|---|---|
| All treated | 291 | **−4.67%** | [−7.26%, −2.06%] | 0.001 |
| Small population | 72 | −4.39% | [−8.53%, −0.66%] | 0.020 |
| Medium population | 119 | −5.63% | [−8.77%, −2.47%] | 0.001 |
| High population | 100 | −3.70% | [−6.99%, **+0.19%**] | 0.059 |

Against the published **−14.33%**. The direction survives, the magnitude does
not: −14.33% falls far outside [−7.26%, −2.06%]. Any claim in the paper that
leans on the size of the effect needs rewriting rather than re-citing.

Percentages transform the draws, `mean(exp(tau) - 1)`, rather than
exponentiating the posterior mean; those are different numbers.

## The unit random intercepts are load-bearing

Same data, same seed, same sampler budget, without them:

| group | `structured` | `structured_rfx_unit` |
|---|---|---|
| All treated | −4.51% [−9.72, +0.91] | −4.67% [−7.26, −2.06] |
| Small | −4.80% [−12.68, +2.75] | −4.39% [−8.53, −0.66] |
| Medium | −5.60% [−11.81, +0.75] | −5.63% [−8.77, −2.47] |
| High | −2.90% [−9.63, +5.26] | −3.70% [−6.99, +0.19] |

The point estimates barely move; the intervals roughly halve. Under `structured`
alone **every** interval spans zero and nothing is detected. Absorbing
county-level unobserved heterogeneity is what makes the effect estimable here,
which is the expected behaviour for a county panel and the reason
`structured_rfx_unit` is the specification to report.

## There is no detectable heterogeneity by county population

This is the part that changes how the application should be described.

The original notebook reports three group means (−0.039 / −0.056 / −0.037) in a
KDE legend with no uncertainty attached, which invites reading medium-population
counties as the worst affected. Overlapping credible intervals are not a test of
difference either. The posterior of the *difference* is, and it comes from the
same draws:

| contrast (`structured_rfx_unit`, log points) | difference | 95% CI | `p_bayes` |
|---|---|---|---|
| Medium − Small | −0.0129 | [−0.0499, +0.0291] | 0.453 |
| High − Small | +0.0072 | [−0.0375, +0.0636] | 0.863 |
| High − Medium | +0.0201 | [−0.0101, +0.0683] | 0.332 |

All three are indistinguishable from zero, and `structured` agrees (p = 0.65,
0.72, 0.49). **The apparent ordering across terciles is not supported.** The
defensible statement is that the effect is negative throughout and the data
cannot resolve heterogeneity by county size at this sample.

Two further cautions for the write-up. `High population` sits at `p_bayes` =
0.059 with an interval that just crosses zero, *before* any correction for
looking at three groups, so it should not be described as a detected effect. And
the tercile GATEs rest on 72 to 119 treated observations each.

## What the KDE figure shows, and what it does not

`fig_mpdta_gate_kde.pdf` has two panels.

The left panel is the original figure: a KDE of `exp(mean_over_draws(tau_i)) - 1`
across treated counties. Each county contributes one number, the posterior
*mean* of its own effect, so the curve's spread is cross-county heterogeneity in
the fitted surface. That is a real quantity and worth plotting. It is not
uncertainty about the estimate, and it is roughly an order of magnitude
narrower, which is why the published figure reads as far more precise than the
estimate is.

The right panel is the estimate's uncertainty: 95% credible intervals for the
ATT and each tercile. A reader takes the headline number from that panel, so it
has to be present.

## Files

```
Results/empirical/mpdta_structured_gate.csv        ATT and per-tercile GATE, both specs
Results/empirical/mpdta_structured_contrasts.csv   between-group differences
Results/empirical/mpdta_structured_catt.csv        per-county posterior mean CATT
Results/empirical/mpdta_structured_seeds.csv       seed-stability check
Results/empirical/fig_mpdta_gate_kde.pdf/.png      the two-panel figure
```
