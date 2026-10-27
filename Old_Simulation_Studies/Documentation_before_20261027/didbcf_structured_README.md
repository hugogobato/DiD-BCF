# didbcf_structured

Bayesian causal forests for difference-in-differences with a **two-way
restricted** prognostic function, so that the prognostic and treatment-effect
forests are separately identified.

Built on `stochtree`'s documented low-level sampler API. It is not a fork of
`stochtree`: `BCFModel` is itself Python-level orchestration over `Dataset`,
`Residual`, `ForestSampler`, `ForestContainer`, `GlobalVarianceModel`,
`LeafVarianceModel` and `RandomEffectsModel`, and this package uses the same
primitives to run a three-forest Gibbs sampler instead of a two-forest one. No
C++ change and no recompilation are needed, so it tracks upstream `stochtree`
releases for free.

## The problem it solves

DiD-BCF as published fits

```
Y_it = mu(D_i, t, X_it) + tau(X_it, k_it) * D_it + eps_it
```

with `mu` an unrestricted forest over `(D_i, t, X)`. Since

```
D_it = 1{D_i = 1} * 1{t >= g}
```

is a function of `(D_i, t)`, the entire treatment term lies inside `mu`'s own
domain. For any `c` in `[0, 1]` the pair

```
(mu + c * tau * D_it,   (1 - c) * tau)
```

produces identical fitted values, residuals and likelihood. The data contain no
information about `c`; only the priors break the tie, and they break it further
toward `mu` as the sample grows, so the bias **increases** with `N`. In tree
terms the reconstruction is trivial: one split on the group and one on time
isolates the treated post-treatment cells exactly.

Conditional parallel trends is exactly the missing restriction. It says the
untreated outcome has no group-by-time interaction, equivalently that the
baseline is additively separable,

```
mu(g, t, x) = a(g, x) + b(t, x),
```

which is Lemma `lem:twoway` of `DiD_BCF_Theory/DiD_BCF_theory.tex`. Cohort
membership may shift the *level* of the baseline, in a covariate-dependent way,
but not its *trend*. Under that restriction the flat direction closes: `a` has
no time argument and `b` has no cohort argument, so their sum cannot represent
`1{D_i = 1} 1{t >= g}`, and `c` is pinned at zero.

## The model

```
Y_it = a(G_i, X_i) + b(t, X_i) + (tau_0 + tau(X_i, k_it)) * D_it + u_r(i) + eps_it
eps_it ~ N(0, sigma^2)
```

| block | forest | splits on | leaf model |
|---|---|---|---|
| `a` | 125 trees, `alpha=0.95, beta=2` | cohort label, covariates | constant |
| `b` | 125 trees, `alpha=0.95, beta=2` | calendar time, covariates | constant |
| `tau` | 100 trees, `alpha=0.25, beta=3` | effect modifiers, event time, cohort | regression on basis `D_it` |

`tau_0` is a global treatment-effect level with a normal prior (`stochtree`'s
`sample_intercept`), so the shrinkage prior on the treatment forest does not
pull the average effect toward zero. `u_r(i)` is an optional additive,
time-invariant random intercept at unit or cohort level.

The restriction is enforced by *construction of the design matrices*, not by
zeroing split weights: the `a` forest is handed a matrix that has no time column
in it at all, and the `b` forest one with no cohort column. `design.py` builds
the three blocks; `model.py` samples them jointly against one shared residual.

One caveat, and it is the familiar one. The flat direction is closed in the
`(g, t)` coordinates unconditionally, but the covariates remain in both blocks,
so if `X` *determined* cohort membership then `b(t, X)` could reproduce
`tau(X) 1{t >= g(X)}` through `X` alone. That is exactly a failure of overlap
(Assumption `ass:overlap`: `P(G = g | X)` bounded away from 0 and 1), which DiD
needs anyway. Routes 1 and 2 depend on the same condition and, unlike this one,
also fail whenever selection is merely *sharp* rather than degenerate.

The default tree counts split `stochtree`'s prognostic budget (250) between the
two baseline blocks and leave the treatment forest's 100 unchanged, so total
model capacity matches the published estimator and a route comparison is not
confounded by capacity.

## Usage

```python
from didbcf_structured import StructuredDiDBCF, structured_estimands

model = StructuredDiDBCF(rfx="unit").sample(
    panel,                       # long panel: unit_id, time, cohort, D, Y, X1..X5
    covariates=("X1", "X2", "X3", "X4", "X5"),
    effect_modifiers=("X1", "X2"),
    num_gfr=50, num_mcmc=500, keep_every=5, num_chains=3, seed=0,
)

model.tau_draws        # (n_obs, n_draws) treatment-effect surface, tau_0 + tau(X, k)
model.mu_draws         # (n_obs, n_draws) fitted control-arm surface, a + b (+ rfx)
model.sigma2_draws     # (n_draws,)
model.check_identification()
estimands = structured_estimands(model)   # GATT(g,t), event study, ATT, CATT
```

Column names are configurable through `PanelColumns`. The cohort label is
derived from the `cohort` column (first treated period, `inf` for never-treated)
and recoded to `0` for never-treated and `1..K` in adoption order, so the same
code serves single-adoption and staggered designs.

## Identification checks that come with it

`did_contrast_draws()` evaluates the difference-in-differences contrast of the
fitted regression function at the observed design points: for each treated
post-treatment row `(i, t)` of cohort `g`,

```
f(g, t, X_it) - f(g, g-1, X_i,g-1) - f(inf, t, X_it) + f(inf, g-1, X_i,g-1),
```

with the never-treated cells obtained by re-evaluating the fitted level block at
the same rows with the cohort label set to never-treated. This is the
*identified* object of equation `eq:contrast`: a functional of the fitted
conditional mean alone, independent of how the fit is split between baseline and
effect. `check_identification()` reports the largest gap between it and the
treatment forest.

That check is **not** an algebraic identity, and what it tests is the assumption
to watch. The trend block always drops out of the contrast. The level block
drops out only when the group gap `a(g, .) - a(inf, .)` is evaluated at the same
covariate value in the post and the reference period — that is, only when the
covariates are **time-invariant** (Assumption `ass:sutva` (iii)). With
covariates drawn per unit-*period*, as in the original DiD-BCF simulation study,
`a(g, X_it)` moves with `t` and the baseline carries group-by-time variation
again; the test suite includes exactly that case as a check that must fail. Run
`check_identification()` whenever the panel construction changes; it costs four
predictions of the level block.

## Gauge

`a` and `b` are identified only up to `(a + c(x), b - c(x))`, which leaves
`mu = a + b` and therefore `tau` untouched. `level_trend_draws()` imposes the
theory's normalisation `b(t_1, .) = 0` after the fact, so `a_tilde` reads as the
baseline level at the first period and `b_tilde` as the common path away from
it. Nothing about `tau` depends on this choice.

## Sampler

Grow-from-root warm start, then thinned MCMC, mirroring `BCFModel`: outcome
standardised, `sigma^2` sampled from an inverse gamma, prognostic leaf scales
sampled, treatment leaf scale fixed by default. One Gibbs sweep is

1. `a` forest, 2. `b` forest, 3. `tau_0`, 4. `tau` forest, 5. random effects,
6. `sigma^2`, 7. leaf scales.

Chains are run independently from separate warm starts and concatenated, rather
than branching from a shared grow-from-root run.

## Relationship to the other repairs

`Simulation_Studies_Revision/Results/identification_note.tex` sets out three
routes. Routes 1 (drop the group indicator from `mu` and carry the group
difference as a random intercept) and 2 (drop it and pass the propensity score
instead) are argument changes to the same `stochtree` call and live in
`did_bcf_revision.did_bcf.SPECS`. Both are mitigations: a tree can still split
on a covariate or on the propensity and then on time, approximating the
interaction wherever selection is sharp in the observables. This package is
the corrected DiD-BCF implementation, which removes the flat direction
structurally rather than merely discouraging it.
