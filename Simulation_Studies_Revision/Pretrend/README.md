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
comparator, free — no MCMC), and `plain` / `corrected` (the constrained DiD-BCF
fit, when `WITH_ATT`).

**Why `PRE_SUBC` and not `PRE_SUB`.** A subgroup's own `Delta(k)` is not a
decision rule: under a homogeneous violation all four subgroups move together
and reporting each separately is four looks at one fact. The *difference* is
zero under any violation constant in `X`, so it isolates exactly the
heterogeneous part — the part an event study cannot represent.

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
