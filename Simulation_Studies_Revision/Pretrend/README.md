# Workstream PT — the conditional-parallel-trends diagnostic

**Reviewer 3.2.3.** Every DiD estimator assumes parallel trends, and DiD-BCF's
main specification cannot test its own assumption: it is fitted as
`y = mu(X, t) + tau(X) Z` with `Z = D_it`, which is zero on every pre-treatment
row, so those rows carry no likelihood information about `tau`. A `tau_hat`
evaluated before adoption is not a placebo estimate, it is the post-treatment
estimate re-read at a different time value, and no post-processing of that fit
recovers a pre-trend coefficient.

The diagnostic is therefore a **separate, unconstrained fit** in which the
treatment indicator is the *ever-treated* indicator `W_i = 1[G_i != inf]`, on in
every period, so `tau(X, k)` is estimable at `k < 0`. What it reports is

```
Delta(k) = E_{i treated}[ tau(X_i, k) - tau(X_i, -1) ],    k < -1
```

which is exactly zero under conditional parallel trends and equals
`differential_slope * (k + 1)` under a violation. See the module docstring of
`did_bcf_revision/pretrend.py` for the three specification choices (no group
indicator in `mu`'s split set, `propensity_covariate="none"`, unit random
intercepts) that are load-bearing rather than cosmetic.

These notebooks measure the diagnostic's **operating characteristics**: its size
where the assumption holds, its power where it does not, and — the part that
justifies the section — the two designs where it and a conventional event study
give *opposite* answers.

---

## What each scenario is for

All ten are `B1_baseline`'s canonical DGP (N = 200, 4 pre-periods, 4 post,
single cohort, selection on an unobserved unit effect) with exactly one change.
`kappa`, `delta`, `lambda` are per-period slopes in outcome units.

### The two nulls — these calibrate everything else

| scenario | change | what it measures |
|---|---|---|
| `PT_hold` | none | **Size.** Conditional PT holds. Every rejection here is a false positive. |
| `PT_conditional` | `selection="observable"`, `conf_strength=0`, `trend_heterogeneity=0.6` | **Discrimination, direction 1.** Conditional PT *holds*; unconditional PT *fails*, because `X4` drives both assignment and slope. |

`PT_conditional` is the case where a marginal event study is actively
misleading. The estimator's assumption is satisfied and its ATT is unbiased, so
a covariate-conditional diagnostic should stay quiet while a TWFE event study —
which conditions on nothing — should flag. In the first 50 replications TWFE
rejected at 0.84 and the conditional diagnostic at 0.14, with ATT bias 0.05 and
0.96 coverage. The point is that the event study's alarm is *false*, and that is
measured on the same replications rather than asserted.

### The power curve — a homogeneous violation

| scenario | change | conditional PTA |
|---|---|---|
| `PT_violation_g{05,10,20,40}` | `Y(0) += delta * 1[ever treated] * t`, `delta in {0.05, 0.1, 0.2, 0.4}` | violated |

The textbook differential path: the treated group is simply on a different
trajectory. `delta` is readable directly in outcome units per period, which is
what makes this the axis for a detection-power curve. Both the diagnostic and
the TWFE comparator can see this one — it is the fair fight, and the scenario
that answers "is the Bayesian diagnostic actually more powerful, or just
different?"

These are also the **null for the subgroup contrast**: the violation is constant
in `X`, so every subgroup moves together and `PRE_SUBC` is identically zero.
Its rejection rate across this whole grid is a size, not a power.

### The heterogeneous violation — where only the conditional diagnostic works

| scenario | change | conditional PTA |
|---|---|---|
| `PT_violation_het{10,20,40}` | `Y(0) += kappa * (2*X1 - 1) * 1[ever treated] * t`, `kappa in {0.1, 0.2, 0.4}` | violated, but **zero on average** |

**Discrimination, direction 2**, and the reason this workstream is more than a
horse race. Treated units with `X1 = 1` trend up at `+kappa`, those with
`X1 = 0` trend down at `-kappa`. `X1` is Bernoulli(0.5) and, under selection on
unobservables, balanced across arms — so the treated-minus-control *average*
differential slope is zero to Monte-Carlo error (verified at 0.0002 / 0.0005 /
0.0009 for the three `kappa`, each within one MCSE of zero).

An event study can only report that average. It is therefore powerless here **by
construction, at every violation magnitude** — not by bad luck, and not
fixable with more data. So is DiD-BCF's own aggregate `Delta(k)`. The only thing
that sees it is the contrast between subgroup pre-trends, `PRE_SUBC`
(`X1 = 1` minus `X1 = 0`, true value `2*kappa`), which is an object no
group-level placebo regression can form at all.

Note that conditional PT is genuinely violated for *both* subgroups here. What
cancels is the aggregate ATT bias, not the assumption — so this also measures
whether an estimator that happens to be unbiased by accident still gets flagged.

### The unobservable-channel check

| scenario | change | conditional PTA |
|---|---|---|
| `PT_violation_a20` | `Y(0) += lambda * a_i * t`, `lambda = 0.2` | violated, **unremovably** |

One cell, not a grid, and it is here to close a specific objection. In `g*` and
`het*` the violation is a function of observed quantities (the treated
indicator, `X1`). A sceptical reader can say: *your diagnostic conditions on
`X`, so of course it detects violations that live in `X` — show me one it cannot
adjust its way out of.* Here the differential slope is driven by `a_i`, the
unobserved unit effect that also drives selection. No adjustment on observables
can remove it, the ATT bias is real and uncorrectable, and the diagnostic still
has to detect it.

It is deliberately **not** a magnitude grid. At `conf_strength = 1` the
treated-minus-control gap in `a_i` is close to 1, so `lambda` and `delta` are
numerically the same knob: over 50 replications the `a` and `g` families
produced pre-trend slopes of 0.110/0.108, 0.202/0.199 and 0.397/0.391,
per-replication correlations of 0.97–0.99, and ATT biases of 0.269/0.267.
Running both as full grids traced one curve twice, so `a{10,40}` were retired to
`_retired/`. The single retained point makes the mechanism argument at a fifth
of the cost.

---

## What comes out

Rows land in the shared summary schema, so `metrics.compute_metrics` aggregates
them unchanged. `reject05` uses `p_bayes < 0.025`, a two-sided 5% test, applied
identically to the Bayesian and frequentist rows.

| `estimand_type` | `estimand_id` | what it is |
|---|---|---|
| `PRE` | `k=-2, -3, -4` | `Delta(k)`, zero under conditional PT |
| `PRE` | `slope` | the implied differential pre-trend slope — the headline scalar, directly comparable to the DGP's `delta` |
| `PRE` | `any`, `any_bonf` | per-replication decision rules: smallest per-`k` tail, raw and Bonferroni-scaled |
| `PRE` | `joint` | TWFE only: the joint Wald pre-trends test |
| `PRE_SUB` | `X1=0_k=-2`, … | `Delta(k)` within covariate subgroups — levels, reported for interpretation |
| `PRE_SUBC` | `X1_any`, `X1_slope`, `X2_any`, … | **the conditional test**: contrasts *between* subgroups |
| `ATT`, `ES`, `GATT`, `CATT` | | `WITH_ATT=True` only (degree 1): the violation's *cost*, measured on the same replications that flag it |

`method` is `pretrend` (the diagnostic), `twfe_es` (the standard-practice
comparator, free — no MCMC), `plain` / `corrected` (the constrained DiD-BCF
fit, when `WITH_ATT`), and the five benchmark pre-trend tests below.

**Why `PRE_SUBC` and not `PRE_SUB`.** A subgroup's own `Delta(k)` is not a
decision rule: under a homogeneous violation all four subgroups move together
and reporting each separately is four looks at one fact. The *difference* is
zero under any violation constant in `X`, so it isolates exactly the
heterogeneous part — the part an event study cannot represent.

---

## The benchmark pre-trend tests

Every estimator in the suite is given its **own** pre-trend test, run on the
same seeded panels, reported in the same schema, against the same realised truth
(`pt_slope`, which now travels with the exported R panels the way `CATE` does).
So `compute_metrics` scores all seven methods on identical definitions, and the
question "is the Bayesian diagnostic actually better at pre-testing?" gets an
answer rather than an assertion.

**All seven target the same estimand**, `Delta(k) = ` the differential
treated-minus-control change from `k = -1` to `k`, so the per-`k` rows, the
`slope`, and the decision rules are directly comparable.

| `method` | its pre-trend test | native? |
|---|---|---|
| `did_dr` | `att_gt(base_period="universal")`: `ATT(g,t)` at `t < g-1` **is** `Delta(k)`, doubly-robust and conditional on `X` | **native** — `did` computes these already and reports its own `Wpval` |
| `doubleml` | the same, with DoubleML's ATTE plugged in as `est_method` (random-forest nuisances) | **native** (it runs through `att_gt`) |
| `did2s` | the pre-period coefficients of the documented event study, recentred on `k = -1` | **standard practice** — the package's own plot shows them |
| `synthdid` | the gap `plot(tau.hat, overlay=1)` draws, referenced to `k = -1` | **package-documented picture**, inference constructed |
| `wang` | the grf-DiD recipe with a pre-treatment period substituted for the post period | **constructed** (the recipe is not a package either) |

Two of these deserve the detail, because a reviewer will ask.

**`synthdid`.** The vignette's "Checking for pre-treatment parallel trends"
section says to run `plot(tau.hat, overlay = 1)` and eyeball how parallel the
trajectories are. Reading `synthdid:::synthdid_plot`, the gap it draws is
`[treated average](t) - [omega-weighted control average](t)` up to a constant,
and referencing that to `k = -1` cancels the overlay constant exactly — so these
rows are the package's own object with its intercept choice differenced out, not
an invention. What the package does **not** provide is inference: `se.method`
sizes the error bar on the scalar effect estimate and there is no statistic on
the pre-period gaps. The standard errors here are a delete-one-unit jackknife
following the package's own convention (`synthdid:::jackknife_se`: `omega` and
`lambda` held fixed, `omega` renormalised after the drop), applied to the whole
`Delta` vector so `slope` and `joint` get the cross-`k` covariance instead of an
independence assumption. Note `se.method='placebo'` **cannot run on this design
at all** — the panels are ~54% treated and synthdid errors with "must have more
controls than treated units".

**`wang`.** grf has no DiD module and no pre-trend test; `wang_grf.R` is itself a
recipe. The placebo is that identical recipe with an earlier period substituted,
so no estimator logic changes — but the `slope` (one forest on the per-unit
slope statistic, so grf returns the slope and its SE directly) and the subgroup
contrast (two `average_treatment_effect(..., subset=)` calls) are constructed,
and `wang_pretrend.R` says so in its header.

For balance: **DiD-BCF's own diagnostic is no more native.** It needs a
different model entirely (`Z` = ever-treated, an extra unconstrained fit); the
grf and synthdid placebos need only a different period.

### Two measured properties that belong in the write-up, not a footnote

* **`did2s` is structurally attenuated.** It imputes `Y(0)` from the untreated
  observations, which include the treated units' own pre-treatment rows, so the
  periods under test are inside the first-stage estimation sample and their
  residuals are in-sample. On `PT_violation_g40` seed 0 it returns
  `Delta(k) = -0.469 / -0.309 / -0.071` against a truth of `-1.2 / -0.8 / -0.4`,
  an implied slope of 0.150 against 0.400. Borusyak–Jaravel–Spiess prescribe
  re-fitting the first stage with the tested periods excluded; that is **not
  implementable on this design** — excluding `k = -4,-3,-2` leaves each treated
  unit one first-stage row, fixest drops them as fixed-effect singletons and
  `did2s` errors out (verified). The attenuated test is what the estimator
  affords here, and it is reported as such.
* **`synthdid` looks for a trend it has already fitted away.** `omega` is chosen
  to match the treated group's pre-treatment path, so the diagnostic is
  conservative by construction (`-0.676` against a truth of `-1.2`). Worse, the
  fitted time weights on these panels are `lambda = 0, 0.128, 0.414, 0.458`: the
  **earliest pre-period gets zero weight**, which is exactly where a linear
  violation is largest. That is a property of the picture the vignette asks the
  reader to inspect.

### Who can form the subgroup contrast

`PRE_SUBC` is the row that carries the `PT_violation_het*` argument, and the
honest position is that **it is not unique to DiD-BCF**. A practitioner with a
binary, pre-specified moderator can split the sample and difference the two
event studies, so `did_dr`, `did2s` and `synthdid` all carry a sample-split
contrast here (disjoint unit sets, so the variances add), and `wang` gets it from
grf's own `subset=` on a single forest — its strongest form, since the forest
learns the heterogeneity on the full sample. `doubleml` is the exception: the
contrast needs four extra fits per replication and it is by far the most
expensive estimator, so it reports the aggregate rows only.

What remains genuinely DiD-BCF's is that the moderator does **not** have to be
pre-specified. That is a narrower claim than "an object no group-level placebo
regression can form at all", which the earlier draft of this README made and
which the measurements do not support.

---

## Running these

80 notebooks: 10 scenarios × 2 linearity degrees × 4 replication blocks of 50.
Replications are seeded by index, so blocks concatenate into exactly the
undivided 200-replication run and `aggregate_all.py` de-duplicates. Upload one
notebook to Colab and *Run all*; it writes and downloads
`summaries_pretrend_<scenario>_lin_<d>_reps<a>-<b>.csv`.

`lin_1` carries `WITH_ATT=True` (100 fits/block, ~1.7 h at `JOBS=2`); `lin_3`
does not (50 fits/block, ~50 min). Degree 2 was dropped: the TWFE comparator is
*exactly* invariant to the degree (it adjusts for no covariates, so the
transforms of `X` never enter its design) and the diagnostic moved by less than
one MCSE.

To regenerate: `python scripts/scaffold_suite.py --force` writes the base
notebooks, then `python scripts/split_pretrend_notebooks.py` cuts them into
blocks. Blocks carrying execution outputs are skipped rather than overwritten.
`python scripts/check_pretrend_dgps.py` asserts the DGP properties above with no
MCMC at all, and is the fast way to confirm nothing broke.

## `_retired/`

Cells dropped in revision, kept rather than deleted: the `linearity_degree = 2`
notebooks, `PT_violation_a{10,40}`, and in `superseded_reps0-50/` the summary
CSVs from the first 50 replications. Those results predate two changes and
should not be pooled with new runs — `PRE_SUBC` did not exist, and the TWFE
pre-trend slope SE summed only the diagonal of the coefficient covariance.
Event-study coefficients share the omitted reference period, so their
covariances are positive and dropping them *understates* `Var(w'beta)`: measured
size at nominal 5% was 0.145, against 0.045 with the full covariance. The old
`twfe_es` `slope` rows overstate the comparator.
