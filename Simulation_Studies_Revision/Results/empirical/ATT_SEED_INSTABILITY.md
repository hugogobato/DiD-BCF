# The application's headline ATT is not stable across MCMC seeds

Found while building the pre-trend diagnostic's empirical demonstration
(Workstream F4). Not part of that task, but it bears directly on the revision.

> **The application is the odd one out.** Every DiD-BCF results file in this
> suite records `spec = structured` (131 of 131 `summaries_*.csv` under
> `Results/`, `Benchmark_Results/` and `DiD_BCF/`), because the notebooks pass
> `spec="structured"` explicitly -- the engine default in `did_bcf.py` is
> `"published"`, so the argument is load-bearing. So the simulation grid is
> already run on the corrected estimator, and the `mpdta` application is the
> only place the published specification still appears. That is a gap in the
> revision, not a design choice. Numbers under the structured estimator are in
> `mpdta_structured_seeds.csv`.

## What was run

`Empirical_Study/DiD_BCF_GATE_CATE_Empirical_Study.ipynb`'s specification,
transcribed exactly into `scripts/run_pretrend_empirical.py::paper_att`:
`mpdta`, `mu` splitting on `(treat, lpop, year, first.treat)`, `tau` on
`(lpop, year)`, `Z = treat * 1[first.treat <= year]`, stochtree's internal
propensity, `num_gfr=50, num_mcmc=500, keep_every=5, num_chains=3`. The only
thing varied is `random_seed`.

## Result

The notebook's own statistic — collapse the posterior to a per-observation mean,
then `exp(.) - 1`, then average over the 291 treated observations:

| seed | notebook statistic | ATT, log points | posterior SD of the ATT |
|---|---|---|---|
| 0 | **+0.105** | +0.099 | 0.388 |
| 1 | **−0.075** | −0.079 | 0.273 |
| 2 | **−0.045** | −0.046 | 0.289 |
| *published* | *−0.143* | | |

The point estimate **changes sign across seeds**, ranging +10.5% to −7.5% in
three draws. The published −14.33% is one realisation from that spread, not a
stable feature of the data.

## Two separate problems

**1. The reported precision is not posterior uncertainty.** The notebook plots a
KDE of `exp(tau_hat.mean(axis=1)) - 1` across treated observations. That is the
spread of the *fitted CATT surface* — cross-sectional heterogeneity in the
posterior mean — and its SD is 0.014--0.025. The **posterior** SD of the ATT is
0.27--0.39, an order of magnitude larger, with a 95% credible interval of
roughly [−0.83, +0.84] in log points. The figure looks precise because it is
displaying the wrong quantity. `p_bayes` for the ATT is 0.38--0.44 at every
seed: the effect is nowhere near distinguishable from zero under this model.

**2. The seed-to-seed swing is too large to be Monte-Carlo error.** With
posterior SD ≈ 0.3 and 300 retained draws, the Monte-Carlo standard error on the
posterior mean would be ≈ 0.02 if the chains mixed. The observed swing is ≈ 0.18,
roughly nine times that. So the chains are not exploring one well-identified
posterior; they are landing in different places.

That is the signature of the flat likelihood direction documented in
`Results/identification_note.tex`: with `D_it` a deterministic function of
covariates already in `mu`'s split set, `(mu + c·tau·D, (1-c)·tau)` has the same
likelihood for every `c`, and only the prior breaks the tie. On simulated data
with a large true effect (`tau_0 = 3`) the prior pins it down well enough to
look stable. On `mpdta`, where the signal is weak relative to the noise, it does
not — and the estimate becomes a function of where the sampler happened to go.
The note already records that the *published* specification has this problem;
this is that defect surfacing in the application rather than in a simulation.

## The repaired specification does not have this problem

The same three-seed test on the **pre-trend diagnostic** fit — same dataset, same
sampler settings, same seeds, but with the note's route-1 and route-2 repairs
applied (group indicators out of `mu`'s split set, `propensity_covariate="none"`,
unit-level random intercepts):

| | seed 0 | seed 1 | seed 2 | range | posterior SD |
|---|---|---|---|---|---|
| diagnostic slope | −0.0058 | −0.0051 | −0.0069 | **0.0018** | 0.0070 |
| `Delta(k=-2)` | +0.0217 | +0.0192 | +0.0222 | 0.0030 | |
| `Delta(k=-3)` | +0.0209 | +0.0176 | +0.0219 | 0.0043 | |
| *published ATT spec, for contrast* | +0.105 | −0.075 | −0.045 | **0.180** | 0.31 |

Read on the scale of each quantity's own posterior spread — the comparison that
does not depend on assuming an effective sample size — the seed range is **26% of
a posterior SD for the diagnostic against 58% for the published ATT**. So mixing
in the repaired fit is better but not perfect, and that should be said rather
than glossed.

What differs is the thing that actually matters: **the verdict is invariant.**
All three seeds give a slope near −0.006 with `p_bayes` between 0.16 and 0.25 —
"no violation detected" every time. The published ATT changes sign across seeds,
so its conclusion is not invariant at all. A diagnostic whose *decision* is
reproducible is usable; a point estimate whose *sign* is not, is not.

This is the note's own argument confirmed on real data rather than in
simulation: closing the flat direction is what makes the estimate reproducible,
and `mpdta` is a sharper test of it than the DGPs are, because a weak real-data
signal is exactly the regime where the prior stops substituting for
identification.

### Why this does not contradict `rfx_unit`'s poor showing on the ATT

The diagnostic fit *is* the `rfx_unit` route, and the table below shows that
route carrying a +0.21 level bias on the ATT. Those are consistent, because the
diagnostic never reports a level. Its reported quantity is
`Delta(k) = E[tau(X,k) - tau(X,-1) | treated]`, a **within-fit difference across
event time**, and any constant that the unit intercepts and `tau` trade between
themselves cancels out of it. The failure mode that sinks `rfx_unit` for a level
estimand is exactly the one `Delta(k)` is constructed to be invariant to, and
the `rfx_unit` bias in the table is a constant (0.19 to 0.23 across a factor of
eight in `N`), not a drift.

That is a design argument, not evidence. The evidence is the `PT_*` simulation
grid (size under `PT_hold` and `PT_conditional`, power under the eight violation
scenarios), which has not been run yet.

## The application refitted under the structured estimator

Six fits: `structured` and `structured_rfx_unit`, three seeds each, same data and
same sampler budget as the published run. `scripts/`-adjacent driver and raw
numbers in `mpdta_structured_seeds.csv`.

| specification | ATT (log pts) | seed range | as % of a posterior SD | 95% CI | `p_bayes` |
|---|---|---|---|---|---|
| published | +0.105 / −0.075 / −0.045 | **0.180** | **58%** | [−0.83, +0.84] | 0.38--0.44 |
| `structured` | −0.0462 | 0.00068 | 2.4% | [−0.101, +0.010] | 0.10 |
| `structured_rfx_unit` | −0.0480 | **0.00013** | **0.9%** | [−0.075, −0.020] | 0.001 |

Read as percentage effects on employment, `structured_rfx_unit` gives
**−4.7%, 95% CI [−7.2%, −2.0%]**, against the published **−14.33%**.

Three things follow.

*The estimate becomes reproducible.* Under `structured_rfx_unit` three seeds
give −0.047941, −0.047912, −0.048041: agreement to four decimal places, under
1% of a posterior SD. The published specification moves 58% of a posterior SD
and changes sign. This is the identification note's argument demonstrated on
real data, and `mpdta` is a sharper test of it than any DGP in the suite, because
a weak real-data signal is exactly the regime where the prior stops substituting
for identification.

*The effect becomes distinguishable from zero, and only with unit intercepts.*
`structured` alone leaves a credible interval that spans zero. Adding county
random intercepts roughly halves the posterior SD (0.028 to 0.014) and moves the
interval off zero. That is the expected consequence of absorbing county-level
unobserved heterogeneity, and it is why `structured_rfx_unit` rather than plain
`structured` is the right specification for this application.

*The published magnitude does not survive.* −4.7% is about a third of −14.33%,
and the published figure sits far outside the corrected interval. The direction
holds; the size does not. Any claim in the paper that leans on the magnitude
needs rewriting, not just re-citing.

## What to do about it

Reviewer 1 asks about inference validity and Reviewer 2 about variance and
coverage, so this will be looked at. Options, roughly in order of preference:

1. **Refit the application under `structured_rfx_unit`.** DONE -- see the
   section above; the numbers are in `mpdta_structured_seeds.csv`. The choice of
   route was made from the suite's own comparison, not intuition:
   `Results/identification/routes_summary_B2_sweep_lin_1_full.csv`, plain
   estimand, averaged over estimands, by `N`:

   | spec | bias @ N=200 | bias @ N=1600 | cover95 @ N=1600 |
   |---|---|---|---|
   | `published` | −2.33 | −3.12 | 0.00 |
   | `rfx_unit` | +0.19 | +0.23 | **0.00** |
   | `rfx` | −0.02 | +0.02 | 0.93 |
   | `structured` | −0.01 | +0.02 | 1.00 |
   | `structured_rfx_unit` | −0.01 | +0.02 | 0.93 |

   `rfx_unit` (route 1 plus unit intercepts, unrestricted `mu`) carries a level
   bias of about +0.21 that **does not shrink in `N`**, and its coverage goes to
   zero. It is not a usable repair for a level estimand. `structured_rfx_unit`
   adds the two-way restriction on `mu` and is clean, while keeping the unit
   intercepts that a county-level DiD application wants. Refit under that and
   re-check seed stability at three seeds, as here.
2. **Report the posterior interval for the ATT**, not only the CATT-surface KDE,
   and say plainly that the effect is not distinguishable from zero if it is not.
   Relabel the existing figure as what it is: heterogeneity in the fitted effect
   across counties.
3. Raise `num_mcmc` / chains and report a convergence diagnostic. This is worth
   doing regardless but will not fix (2), and if the direction is genuinely flat
   it will not fix (1) either.

## Reproducing

```bash
python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from run_pretrend_empirical import paper_att, DEFAULT_DATA
bp = dict(num_gfr=50, num_mcmc=500, keep_every=5, num_chains=3)
for seed in (0, 1, 2):
    print(seed); print(paper_att(DEFAULT_DATA, bp, seed=seed).to_string(index=False))
PY
```

About 10 minutes per seed.
