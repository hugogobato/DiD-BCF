# DiD-BCF revision simulations (Workstreams B1, B2, D, PT)

New simulation suite for the JBES revision, built on top of
`../Simulation_Studies/` but redesigned to address the reviewer comments in
`../../REVISION_PLAN.md`. It implements:

* **B1 — canonical DiD**: selection on *unobserved, time-invariant* heterogeneity
  (a unit fixed effect `alpha_i` correlated with treatment), persistent
  covariates, AR(1) serially correlated errors, and covariate-dependent trends
  (so parallel trends holds only *conditional* on covariates).
* **B2 — decomposed metrics + sample-size sweep**: bias, variance, RMSE, **MAE,
  MAPE** (for parity with the paper's tables), coverage (90/95%), interval
  length, the **calibration ratio** `avg_post_sd / emp_sd`, size/power and their
  **Monte-Carlo standard errors** — plus the paper's **CATT-surface RMSE/MAE/MAPE**
  (over the individual treated observations, mean ± SD across runs) with a
  *pointwise* CATT coverage, and an `N ∈ {200, 400, 800, 1600}` sweep (anchored
  at the base size 200) exhibiting bias→0, variance→0 and √N stabilisation.
* **D — staggered adoption** with treatment effects that vary by **both
  event-time and cohort**, the **Goodman-Bacon decomposition** of TWFE, and two
  distinct sweeps: `--ramp-sweep` over the *strength* of the dynamics, and
  `--design-sweep` over the **weight** already-treated comparisons carry. The
  distinction matters — the Goodman-Bacon weights depend on the adoption design
  alone, so `dynamic_ramp` leaves them exactly constant. The `D_ramp_nt*`
  scenarios move that weight from 0.077 to 0.250 by shrinking the never-treated
  pool with the effect DGP held fixed, and **every estimator in the suite** is
  plotted against it.
* **PT — the pre-trend / conditional-parallel-trends diagnostic** (Reviewer
  3.2.3): the unconstrained specification fitted as a *separate diagnostic
  model* so that `tau(X, k)` exists before adoption, with size, power against
  two families of violation, and the discrimination case where conditional PT
  holds but unconditional PT fails. Demonstrated on simulation and on the
  `mpdta` empirical application.

> Running both of these is documented step by step, with the Colab notebook
> assignment and wall-time budget, in
> [`RUN_PLAN_pretrend_and_ramp.md`](RUN_PLAN_pretrend_and_ramp.md).

The panel matches the original study: **N = 200 units, 4 pre + 4 post periods**
(the staggered DGP has 3 treated cohorts adopting at the first three post
periods, plus a never-treated group). Each scenario is run at every
**`linearity_degree ∈ {1, 2, 3}`**.

**Every evaluation metric is reported for both the original (plain) DiD-BCF and
the proposed posterior correction** (Algorithm 1 of `../../DiD_BCF_Theory/
DiD_BCF_theory.tex`), so the two can be compared directly. See
[*On the posterior correction*](#on-the-posterior-correction) below.

### Benchmark models

Alongside DiD-BCF, the suite includes the same benchmark estimators as the
original `../Simulation_Studies/`, adapted to the revision DGPs:

* **TWFE / OLS** (`TWFE/` notebooks, `did_bcf_revision/twfe_runner.py`) — plain
  two-way fixed-effects event study + static ATT with cluster-robust SEs, in pure
  numpy/pandas; reported in the **same metric tables** as DiD-BCF.
* **R benchmarks** (`R_code/<scenario>_datasets/`): Callaway–Sant'Anna
  doubly-robust (`did`), Gardner two-stage (`did2s`), DoubleML DR-DiD, and
  synthetic DiD (`synthdid`), each reading the CSV panels written by the
  matching `DGPs/data_creation_<scenario>.py`.

---

## Workflow (matches the split: DiD-BCF on Colab, the rest on your PC)

```
1. (Colab, slow)   DiD_BCF/DiD_BCF_<scen>_lin_<d>.ipynb -> Results/summaries_<scen>_lin_<d>.csv
1b.(Colab, slow)   Pretrend/Pretrend_<PT>_lin_<d>.ipynb -> Results/summaries_pretrend_<PT>_lin_<d>.csv
2. (PC, fast)      TWFE/OLS_<scen>_lin_<d>.ipynb        -> Results/summaries_twfe_<scen>_lin_<d>.csv
3. (PC, fast)      DGPs/data_creation_<scen>.py         -> R_code/<scen>_datasets/.../iteration_*.csv
4. (R, fast)       R_code/<scen>_datasets/*.R           -> summaries_<method>_<scen>_lin_<d>.csv
                   (or scripts/run_r_benchmarks.py, which does 3+4 in a temp dir)
5. (PC, fast)      scripts/aggregate_metrics.py         -> Results/metrics_*.csv / .xlsx
5b.(PC, fast)      scripts/aggregate_r_metrics.py       -> Results/metrics_r_*.csv
6. (PC, parallel)  scripts/run_goodman_bacon.py         -> Results/goodman_bacon_*.csv
7. (PC, fast)      scripts/make_figures.py              -> Results/B2_sweep_*.png
8. (PC, fast)      Results/analysis/aggregate_all.py    -> Results/aggregated/*.csv
9. (PC, fast)      Results/analysis/make_*_analysis.py  -> Results/{tables,figures}/
```

Step 1 (the BCF MCMC fits) is the only expensive part. **There is one notebook
per model (DiD-BCF, OLS), per scenario, per `linearity_degree`** — mirroring the
original `Simulation_Studies/` layout — so run only the cells you need, never a
monolith. The remaining steps read the saved per-replication summaries (and the
exported CSVs) and are cheap; the dataset-export, both runner CLIs and the
Goodman-Bacon analysis accept `--jobs N` for replication-level parallelism.

### Quick start

```bash
# 1. fit DiD-BCF (plain + corrected) — on Colab via the per-scenario/per-linearity
#    notebooks in DiD_BCF/, or headless with the equivalent CLI:
python scripts/run_did_bcf.py --experiment B2_sweep --reps 200 --jobs 4      # all 3 linearity degrees
python scripts/run_did_bcf.py --experiment B1_null --linearity-degree 1 --reps 300
python scripts/run_did_bcf.py --all --reps 200                                # every scenario

# 2. TWFE / OLS benchmark (pure numpy/pandas, runs on your PC)
python scripts/run_twfe.py --all --reps 200 --jobs 8

# 3. export the exact CSV panels the R benchmarks read
python DGPs/data_creation_D_staggered.py --jobs 8
#    then, from inside each scenario folder, run the R estimators:
#    cd R_code/D_staggered_datasets && Rscript did_dr_new.R && Rscript did2s.R ...

# 4. decomposed metrics, all methods (plain / corrected / twfe) side by side
python scripts/aggregate_metrics.py
#    bring the R benchmarks (CS, did2s, DoubleML, synthdid) into the same
#    bias/variance/coverage/RMSE-MAE-MAPE comparison:
python scripts/aggregate_r_metrics.py

# 5. Goodman-Bacon + TWFE vs truth (Workstream D)
python scripts/run_goodman_bacon.py --experiment D_staggered --reps 500 --jobs 8
#    --ramp-sweep varies how WRONG the already-treated comparisons are;
#    --design-sweep varies how much WEIGHT they carry (the one that moves it):
python scripts/run_goodman_bacon.py --experiment D_staggered --design-sweep --reps 200 --jobs 3

# 5b. the pre-trend / conditional-PTA diagnostic (Workstream PT)
python scripts/run_pretrend.py --experiment PT_hold --reps 200 --jobs 2 --with-att
python scripts/run_pretrend.py --all --linearity-degree 1 --with-att --jobs 2
python scripts/run_pretrend_empirical.py --with-att     # on mpdta

# 6. sample-size-sweep figures
python scripts/make_figures.py --setting B2_sweep

# 7. the two new analyses (after Results/analysis/aggregate_all.py)
python Results/analysis/make_ramp_analysis.py       # error vs already-treated weight
python Results/analysis/make_pretrend_analysis.py   # size / power / discrimination
```

The notebooks are thin and **self-bootstrapping**: upload a single one to Colab
and *Run all* — the setup cell `git clone`s `https://github.com/hugogobato/DiD-BCF`
when the engine isn't already present (and reuses the local checkout when you run
it inside the repo), then makes one `run_named("<scen>", linearity_degree=<d>,
...)` (DiD-BCF) or `run_twfe_named("<scen>", linearity_degree=<d>, ...)` (OLS)
call. The fitting logic lives once in `did_bcf_revision/{runner,twfe_runner}.py`,
so the 54 notebooks cannot drift apart — regenerate them all from
`scripts/scaffold_suite.py`.

---

## Layout

Mirrors `../Simulation_Studies/`: per-scenario data-creation scripts under
`DGPs/`, **one model notebook per scenario per `linearity_degree`** under
`DiD_BCF/` and `TWFE/`, per-scenario R benchmarks under `R_code/`, and a
`Results/` sink — with a shared engine package so nothing is duplicated.

The **estimation scenarios** are `B1_baseline`, `B1_strong_confounder`,
`B1_serial_corr`, `B1_selection_obs`, `B1_selection_both`, `B1_null`,
`B2_sweep`, `B2_sweep_serial`, `D_staggered`, `D_contamination` and the six
`D_ramp_nt{40,30,25,20,10,05}` designs. Each becomes 1 data script, 3 DiD-BCF
notebooks, 3 OLS notebooks, and one `R_code/<scenario>_datasets/` folder of 4 R
estimators.

The **10 diagnostic scenarios** (workstream `PT`) are `PT_hold`,
`PT_conditional`, `PT_violation_g{05,10,20,40}`, `PT_violation_het{10,20,40}`
and `PT_violation_a20`. They have their own driver and no benchmark comparison,
so each becomes 2 notebooks under `Pretrend/` (one per retained linearity
degree) and nothing else.

```
Simulation_Studies_Revision/
├── README.md
├── did_bcf_revision/              # importable engine (the source of truth)
│   ├── dgps.py                    # canonical (B1) + staggered (D) DGPs, true_estimands()
│   ├── pretrend.py                # PT: unconstrained diagnostic fit + Delta(k)
│   ├── pretrend_runner.py         # PT Monte-Carlo driver (Pretrend/ notebooks + CLI)
│   ├── did_bcf.py                 # stochtree BCF fit + PLAIN estimand summaries; SPECS (routes 1-2)
│   ├── structured.py              # corrected DiD-BCF adapter -> ../../didbcf_structured/
│   ├── posterior_correction.py    # Algorithm 1 (DR post-processing) -> CORRECTED summaries
│   ├── runner.py                  # DiD-BCF Monte-Carlo driver (used by DiD_BCF/ notebooks + CLI)
│   ├── twfe.py                    # static + event-study TWFE with cluster-robust SEs
│   ├── twfe_runner.py             # TWFE Monte-Carlo driver (used by TWFE/ notebooks + CLI)
│   ├── metrics.py                 # bias/var/RMSE/coverage/length/size-power (B2)
│   ├── goodman_bacon.py           # Goodman-Bacon (2021) decomposition
│   ├── exports.py                 # tidy frame -> R-benchmark CSV column layout
│   └── config.py                  # the scenario grid (N=200, linearity 1/3)
├── DGPs/                          # one data-creation script per scenario (-> R CSV panels)
│   ├── data_creation_B1_baseline.py ... data_creation_D_contamination.py   (9)
├── DiD_BCF/                       # DiD-BCF: one notebook per scenario × linearity (32; Colab)
│   ├── DiD_BCF_B1_baseline_lin_1.ipynb ... DiD_BCF_D_contamination_lin_3.ipynb
├── TWFE/                          # OLS benchmark: one notebook per scenario × linearity (32; PC)
│   ├── OLS_B1_baseline_lin_1.ipynb ... OLS_D_contamination_lin_3.ipynb
├── Pretrend/                      # PT diagnostic: one notebook per PT scenario × linearity (20)
│   ├── Pretrend_PT_hold_lin_1.ipynb ... Pretrend_PT_violation_a20_lin_3.ipynb
│   ├── *_reps{0-50,...}.ipynb     # Colab-sized blocks (split_pretrend_notebooks.py)
│   └── _retired/                  # cells dropped in revision: lin_2, a10, a40
├── R_code/                        # R benchmarks, one folder per scenario (9 × 4 scripts)
│   └── <scenario>_datasets/{did_dr_new.R, did2s.R, DoubleML_did.R, synthdid.R}
├── scripts/                       # the cheap / parallel local steps
│   ├── run_did_bcf.py             # headless equivalent of the DiD_BCF/ notebooks (--spec)
│   ├── run_pretrend.py            # headless equivalent of the Pretrend/ notebooks (PT)
│   ├── check_pretrend_dgps.py     # numerical checks on the PT grid (no MCMC)
│   ├── split_pretrend_notebooks.py   # Pretrend/*.ipynb -> Colab-sized rep blocks
│   ├── run_pretrend_empirical.py  # the diagnostic on the mpdta application
│   ├── run_identification_routes.py  # the identification acceptance test, all routes
│   ├── run_twfe.py                # headless equivalent of the TWFE/ notebooks
│   ├── run_r_benchmarks.py        # all four R estimators, data generated on the fly
│   ├── aggregate_metrics.py       # decomposed metrics, all methods (B2)
│   ├── run_goodman_bacon.py       # Goodman-Bacon; --ramp-sweep and --design-sweep (D)
│   ├── make_figures.py            # sample-size-sweep figures
│   ├── scaffold_suite.py          # regenerates DGPs/, DiD_BCF/, TWFE/, Pretrend/, R_code/
│   └── scaffold_colab_benchmarks.py  # regenerates DoubleML_Colab/, CFFE_Wang_Colab/
└── Results/                       # all outputs land here
    ├── analysis/style.py                    # the shared figure design system
    ├── analysis/make_ramp_analysis.py       # D: every estimator vs already-treated weight
    ├── analysis/make_pretrend_analysis.py   # PT: size, power, discrimination
    └── empirical/                           # the mpdta diagnostic output
```

> **`scaffold_suite.py` does not template the R scripts.** It used to, and its
> copies had gone stale: a scaffold run silently replaced 3222 lines of the
> current schema-emitting R scripts with older versions. They are now copied
> from the checked-in reference scenario for the matching DGP (`B1_baseline` for
> canonical, `D_staggered` for staggered) with only the scenario name
> substituted, and the reference scenario is skipped so it can never overwrite
> itself.

### Estimand schema (shared everywhere)

`dgps.true_estimands`, `did_bcf.plain_estimands` and
`posterior_correction.corrected_estimands` all emit the same estimand set:

| `estimand_type` | meaning |
|---|---|
| `GATT` | cohort × calendar-time cell, `g=<g>_t=<t>`, for `t ≥ g` |
| `ES`   | event-study, `k=<k>` (averaged over cohorts), `k ≥ 0` |
| `ATT`  | overall average over all treated post observations |
| `CATT` | one **per-replication** surface row (`estimand_id='surface'`) carrying the within-rep RMSE/MAE/MAPE over the *individual* treated observations + pointwise CATT coverage/length |

Per-replication summary rows carry `post_mean, sd, q025, q05, q95, q975,
p_bayes` for each `method ∈ {plain, corrected, twfe}`, plus the `true` value and
a `linearity_degree` column, so `metrics.compute_metrics` aggregates every model
and every linearity degree in one table. (Plain TWFE reports only `ES` and `ATT`
— pooling cohorts into clean `GATT(g,t)` is exactly the contamination it suffers
from, so `GATT` is left to DiD-BCF and the Callaway–Sant'Anna R benchmark.)

The `CATT`/`surface` rows instead carry `surf_rmse, surf_mae, surf_mape,
surf_cover90, surf_cover95, surf_len90, surf_len95, surf_n` (the point-estimate
columns are left blank). `metrics.compute_metrics` **excludes** them — they are
within-replication error metrics, not a sampling distribution — and
`metrics.surface_metrics` aggregates them into the paper's `mean ± SD`
RMSE/MAE/MAPE tables. For TWFE the surface estimate is the event-study
coefficient `ATT(k)` broadcast to every treated observation; for the posterior
correction it is the cell-level `GATT(g,t)` broadcast to its members — exactly
how the GATT-only R benchmarks are scored, so all methods land in one
`metrics_surface.csv` / `metrics_r_surface.csv` comparison.

---

## DGP specifications

Both DGPs share the same building blocks. For unit $i$ in period
$t \in \{0,\dots,T-1\}$:

**Persistent, unit-level covariates** (drawn once per unit, hence time-invariant
— this is the change from the original suite, which redrew covariates every
unit-period):

$$
X_{1i}\sim\mathrm{Bernoulli}(0.5),\quad
X_{2i},X_{3i},X_{4i}\sim N(0,1),\quad
X_{5i}\sim \mathrm{Unif}(-1,1).
$$

**Unobserved, time-invariant unit effect** (the canonical-DiD ingredient; never
passed to any estimator):

$$
\alpha_i = \sigma_\alpha\, a_i,\qquad a_i\sim N(0,1),
$$

with $\sigma_\alpha=$ `alpha_sd`.

**Prognostic covariate level**, linear in the latent covariates at every degree:

$$
f(X_i)=-0.75X_{1i}+0.5X_{2i}-0.5X_{3i}-1.3X_{4i}+1.8X_{5i}.
$$

**Common time effect** $\gamma_t=\beta_{\text{time}}\,t$, with $\beta_0=-0.5$,
$\beta_{\text{time}}=0.2$.

**Covariate-dependent trend** (makes parallel trends hold only *conditional* on
$X$): $s(X_i)=\rho_{\text{tr}}\,(0.7X_{4i}+\sqrt{0.51}\,X_{3i})$ with
$\rho_{\text{tr}}=$ `trend_heterogeneity`, so $\mathrm{Var}(s)=\rho_{\text{tr}}^2$.
$X_4$ enters the assignment utility and $X_3$ does not, so the treated and
control groups genuinely have different average slopes.

> **Note.** Under `selection="unobservable"` the covariates are balanced across
> arms, so this trend produces no bias and no adjustment is needed. It violates
> conditional parallel trends only in `B1_selection_obs` and
> `B1_selection_both`.

**Linearity degree.** The DGP above never varies with $d$. What varies is the
covariate parameterisation the estimators are handed, and (at $d=3$) the shape
of $\tau$:

| $d$ | what the estimator observes | $\tau(X_i)$ |
|---|---|---|
| 1 | the latent $X_1,\dots,X_5$; every nuisance function is linear in them | $\tau_0+1.5X_1+0.75X_2$ |
| 2 | $X_3,X_4,X_5$ replaced by Kang–Schafer transforms $\big((X_3X_4/5+0.6)^3,\ e^{X_4/2},\ (X_3+X_5+2)^2\big)$, standardised | unchanged |
| 3 | as $d=2$ | $\tau_0+1.5X_1+0.75\big[(X_2^2-1)/\sqrt2+(2X_1-1)X_2\big]/\sqrt2$ |

At $d=2$ the outcome and every true estimand are **bit-identical** to $d=1$, so
any difference in estimator performance is misspecification alone. $X_1,X_2$
are never transformed, keeping "non-linear nuisance" and "non-linear effect" as
separate axes, and $\mathrm{Var}(\tau)$ is held constant so the CATT-surface
metrics stay comparable. Asserted by
`Results/_staging/check_linearity_axis.py`.

**Errors** $\varepsilon_{it}$ are AR(1) within unit (iid when `ar1_rho` $=0$),
initialised at the stationary distribution so $\mathrm{Var}(\varepsilon_{it})=\sigma^2$ for all $t$:

$$
\varepsilon_{i0}\sim N(0,\sigma^2),\qquad
\varepsilon_{it}=\rho\,\varepsilon_{i,t-1}+\nu_{it},\quad
\nu_{it}\sim N\!\big(0,\sigma^2(1-\rho^2)\big),
$$

with $\rho=$ `ar1_rho`, $\sigma=$ `epsilon_scale`.

The **untreated potential outcome** is, in both DGPs,

$$
Y_{it}(0)=\beta_0+\alpha_i+\gamma_t+f(X_i)+s(X_i)\,t+\varepsilon_{it},
$$

and the observed outcome is $Y_{it}=Y_{it}(0)+\text{CATT}_{it}$, where the
realised effect $\text{CATT}_{it}$ and the treatment-timing $G_i$ differ by DGP.

### Canonical DiD DGP (B1)

A single adoption period $g_0=$ `num_pre_periods`. Treatment is assigned by a
latent index combining an **observed** and an **unobserved** driver,

$$
V_i=\underbrace{0.8X_{1i}+0.6X_{4i}}_{\text{observed}}\cdot\mathbb{1}[\text{sel}\in\{\text{obs,both}\}]
+\underbrace{c\,a_i}_{\text{unobserved}}\cdot\mathbb{1}[\text{sel}\in\{\text{unobs,both}\}],
$$

with $c=$ `conf_strength` and `sel` $=$ `selection`. After centring $V_i$ at its
$(1-\bar p)$-quantile (so the treated share targets $\bar p=$
`treated_share_target`),

$$
G_i=\begin{cases} g_0 & \text{if } u_i<\mathrm{sigmoid}(1.5V_i+\eta_i),\ \eta_i\sim N(0,0.5^2),\\ \infty & \text{otherwise,}\end{cases}
\qquad u_i\sim\mathrm{Unif}(0,1).
$$

The **treatment-effect function** is

$$
\tau(X_i)=
\begin{cases}
\tau_0, & \text{`homogeneous`},\\
\tau_0+1.5X_{1i}+0.75\tanh(X_{2i}), & \text{`heterogeneous`},
\end{cases}
$$

with base $\tau_0=$ `base_effect`, and the realised effect is
$\text{CATT}_{it}=\tau(X_i)\,D_{it}$, where $D_{it}=\mathbb{1}[G_i\ne\infty,\ t\ge G_i]$.
Because $\alpha_i$ is a pure level shift it is differenced out by any DiD
contrast, but DiD-BCF models *levels* and absorbs only a group intercept — so
$\sigma_\alpha,c>0$ is the setting that stresses it (and that the posterior
correction, which differences $Y$, is meant to handle).

### Staggered DGP (D)

$K$ cohorts adopt at $g_j=$ `num_pre_periods` $+$ `cohort_offsets`$_j$, plus a
never-treated group. Cohort membership is a Gumbel-max (multinomial-logit)
choice in which the unobserved driver also shifts *timing*:

$$
\text{driver}_i = 0.6X_{1i}+0.4X_{4i}\ (\text{obs}) \;\text{and/or}\; c\,a_i\ (\text{unobs}),
$$

$$
U_{i0}=\log \pi_{\infty}+\epsilon_{i0},\qquad
U_{ij}=\log \pi_j+0.8\,\text{driver}_i+0.5\Big(1-\tfrac{j-1}{K-1}\Big)\text{driver}_i+\epsilon_{ij},
$$

with $\epsilon_{ij}\sim\mathrm{Gumbel}(0,1)$, shares $\pi_j=$ `cohort_shares`,
$\pi_\infty=1-\sum_j\pi_j$; unit $i$ joins $\arg\max_j U_{ij}$. The effect varies
by **cohort and event-time** $k=t-G_i$:

$$
\text{CATT}_{it}=m_{g}\,\big(1+r\,k\big)\,\tau(X_i)\,\mathbb{1}[k\ge 0],
$$

where $m_g=$ `cohort_multipliers` (per cohort), $r=$ `dynamic_ramp` (effects grow
with exposure), and $\tau(X_i)$ is as above. This cohort × event-time
heterogeneity is exactly what makes already-treated units invalid controls and
contaminates TWFE (quantified by the Goodman-Bacon decomposition).

### True estimands

For any DGP the reported targets are averages of the realised $\text{CATT}_{it}$
over treated post observations: $\text{GATT}(g,t)=\mathbb{E}[\text{CATT}_{it}\mid G_i=g,\ t\ge g]$,
the event-study $\text{ATT}(k)$ averaging over cohorts at fixed $k$, and the
overall $\text{ATT}$ — computed exactly by `dgps.true_estimands`.

---

## What each DGP knob does

`config.py` fixes the suite, but the generators in `dgps.py` are fully
parameterised. The canonical-DiD knobs (and their reviewer target):

| knob | effect | addresses |
|---|---|---|
| `alpha_sd` | SD of the unobserved unit effect `alpha_i` | R1.4 / R3.1.1 |
| `conf_strength` | corr(`alpha_i`, treatment): selection on unobservables | R1.4 |
| `selection` | `unobservable` / `observable` / `both` | R1.4 |
| `ar1_rho` | within-unit serial correlation of errors | R3.1.3 |
| `trend_heterogeneity` | covariate-dependent trend → conditional PTA | R1.4 |
| `effect_type` | `homogeneous` / `heterogeneous` CATT | — |
| `base_effect=0` | sharp null for size/coverage | R1.3 / R2.2 |
| `group_trend` | `delta · 1[ever treated] · t`: **violates** conditional PTA | R3.2.3 |
| `alpha_trend` | `lambda · a_i · t`: violation routed through the *unobserved* confounder | R3.2.3 |

Staggered adds `cohort_offsets`, `cohort_shares`, `cohort_multipliers` and
`dynamic_ramp` (effect grows with exposure) — the cohort × event-time
heterogeneity that breaks TWFE (R3.1.2).

> `group_trend` and `alpha_trend` default to `0.0`, so every scenario that
> predates them is bit-identical. Unlike `trend_heterogeneity`, which is a
> function of the observed covariates and is therefore *removable* by
> conditioning, these two are genuine violations of conditional parallel trends
> — which is what makes them the right axis for the diagnostic's power curve.
> The per-unit violation slope is stored as `pt_slope`, alongside `alpha`, for
> diagnostics only, and is never passed to an estimator.

> The unobserved `alpha` column is included in the data frame **for diagnostics
> only** and is never placed in the estimator's design matrix.

---

## Mathematical Formulations per DGP Scenario

Each of the 9 scenarios is defined by overriding specific parameters in [config.py](file:///home/hugo_souto/Stuff/Research/DiD-BCF/DiD-BCF/Simulation_Studies_Revision/did_bcf_revision/config.py) which are passed to the generators in [dgps.py](file:///home/hugo_souto/Stuff/Research/DiD-BCF/DiD-BCF/Simulation_Studies_Revision/did_bcf_revision/dgps.py).

Below is the complete mathematical description for each of the 9 scenarios.

### 1. `B1_baseline`
* **Type**: Canonical DiD (B1)
* **Objective**: Standard canonical DiD setup with moderate confounding on unobservables, linear covariate trends, and independent errors.
* **Sample Size**: $N = 200$, $T = 8$ (4 pre-treatment periods $t \in \{0, 1, 2, 3\}$, 4 post-treatment periods $t \in \{4, 5, 6, 7\}$).
* **Potential Outcome under Control**:
  $$
  Y_{it}(0) = \beta_0 + \alpha_i + \gamma_t + f(X_i) + s(X_i) \cdot t + \varepsilon_{it}
  $$
  where:
  - $\beta_0 = -0.5$
  - $\alpha_i \sim N(0, 1.0)$ ($\sigma_\alpha = 1.0$) is the unobserved unit fixed effect.
  - $\gamma_t = 0.2 \cdot t$ (for linearity degrees $d \in \{1, 2\}$, and $\gamma_t = 0.2 \cdot t^2$ for $d = 3$).
  - $f(X_i)$ is the prognostic covariate function defined for linearity degrees $d \in \{1, 2, 3\}$.
  - $s(X_i) = 0.3 \cdot X_{3i}$ (trend heterogeneity $\rho_{\text{tr}} = 0.3$).
  - $\varepsilon_{it} \sim N(0, 1.0)$ iid ($\text{AR}(1)\ \rho = 0.0$, $\sigma = 1.0$).
* **Selection / Treatment Assignment**:
  - The driver utility is based purely on unobserved heterogeneity:
    $$
    V_i = c \cdot a_i = 1.0 \cdot a_i = a_i
    $$
    where $a_i = \alpha_i / \sigma_\alpha \sim N(0, 1.0)$ ($c = 1.0$).
  - The centered utility is $U_i = V_i - q_{0.5}(V)$, where $q_{0.5}(V)$ is the empirical median of $V$ (targeting treatment share $\bar{p} = 0.5$).
  - Unit $i$ adopts treatment at $G_i = 4$ if $u_i < \text{sigmoid}(1.5 U_i + \eta_i)$, where $\eta_i \sim N(0, 0.5^2)$ and $u_i \sim \text{Unif}(0, 1)$; otherwise $G_i = \infty$.
* **Treatment Effect / Observed Outcome**:
  - Unit treatment effect: $\tau(X_i) = 3.0 + 1.5 X_{1i} + 0.75 \tanh(X_{2i})$ (since $\tau_0 = 3.0$, type is `heterogeneous`).
  - Realised treatment effect: $\text{CATT}_{it} = \tau(X_i) \cdot \mathbb{1}[G_i = 4 \text{ and } t \ge 4]$.
  - Observed outcome: $Y_{it} = Y_{it}(0) + \text{CATT}_{it}$.

### 2. `B1_strong_confounder`
* **Type**: Canonical DiD (B1)
* **Objective**: Test robustness under high-variance unit effects and strong selection on unobservables.
* **Sample Size**: $N = 200$, $T = 8$.
* **Potential Outcome under Control**:
  - Identical to `B1_baseline`, except the unobserved unit fixed effect has higher variance:
    $$
    \alpha_i \sim N(0, 4.0) \quad (\sigma_\alpha = 2.0)
    $$
* **Selection / Treatment Assignment**:
  - Utility driver has stronger selection on unobservables ($c = 1.5$):
    $$
    V_i = 1.5 \cdot a_i
    $$
    where $a_i = \alpha_i / 2.0 \sim N(0, 1.0)$.
  - Centering utility $U_i$ and assignment probability are identical to `B1_baseline`.
* **Treatment Effect**: Identical to `B1_baseline`.

### 3. `B1_serial_corr`
* **Type**: Canonical DiD (B1)
* **Objective**: Evaluate estimator coverage and precision in the presence of serially correlated errors.
* **Sample Size**: $N = 200$, $T = 8$.
* **Potential Outcome under Control**:
  - Identical to `B1_baseline`, except the errors $\varepsilon_{it}$ follow an $\text{AR}(1)$ process within units:
    $$
    \begin{aligned}
    \varepsilon_{i0} &\sim N(0, 1.0) \\
    \varepsilon_{it} &= 0.6 \cdot \varepsilon_{i,t-1} + \nu_{it}, \quad \nu_{it} \sim N(0, 0.64) \quad \text{for } t \ge 1
    \end{aligned}
    $$
    where $\rho = 0.6$ and $\text{Var}(\nu_{it}) = \sigma^2(1 - \rho^2) = 1.0 \times (1 - 0.36) = 0.64$, preserving the marginal variance $\text{Var}(\varepsilon_{it}) = 1.0$ at all $t$.
* **Selection & Treatment Effect**: Identical to `B1_baseline`.

### 4. `B1_selection_obs`
* **Type**: Canonical DiD (B1)
* **Objective**: Continuity check where selection is purely driven by observed covariates.
* **Sample Size**: $N = 200$, $T = 8$.
* **Potential Outcome under Control**: Identical to `B1_baseline` ($\alpha_i \sim N(0, 1.0)$ is independent of treatment).
* **Selection / Treatment Assignment**:
  - Selection is on observables only ($c = 0.0$, selection `observable`):
    $$
    V_i = 0.8 X_{1i} + 0.6 X_{4i}
    $$
  - Centering utility $U_i$ and assignment probability are identical to `B1_baseline`.
* **Treatment Effect**: Identical to `B1_baseline`.

### 5. `B1_null`
* **Type**: Canonical DiD (B1)
* **Objective**: Check size/coverage behavior under a sharp null hypothesis of zero treatment effect.
* **Sample Size**: $N = 200$ (with default replications increased to 200), $T = 8$.
* **Potential Outcome under Control**: Identical to `B1_baseline`.
* **Selection / Treatment Assignment**: Identical to `B1_baseline`.
* **Treatment Effect**:
  - Treatment effect is identically zero ($\tau_0 = 0.0$, type `homogeneous`):
    $$
    \tau(X_i) = 0.0 \implies \text{CATT}_{it} = 0.0
    $$
  - Observed outcome: $Y_{it} = Y_{it}(0)$.

### 6. `B2_sweep`
* **Type**: Canonical DiD (B2)
* **Objective**: Verify asymptotic behavior (consistency and $\sqrt{N}$-stabilization) by sweeping sample size $N$.
* **Sample Size**: $N \in \{200, 400, 800, 1600\}$, $T = 8$.
* **Mathematical Formulations**: Identical to `B1_baseline` for each respective sample size $N$.

### 7. `B2_sweep_serial`
* **Type**: Canonical DiD (B2)
* **Objective**: Verify asymptotic behavior under serial correlation by sweeping sample size $N$.
* **Sample Size**: $N \in \{200, 400, 800, 1600\}$, $T = 8$.
* **Mathematical Formulations**: Identical to `B1_serial_corr` for each respective sample size $N$.

### 8. `D_staggered`
* **Type**: Staggered adoption (D)
* **Objective**: Evaluate performance with treatment effects that vary by both cohort ($g$) and event-time ($k$).
* **Sample Size**: $N = 200$, $T = 8$ (earliest adoption $g_1 = 4$).
* **Potential Outcome under Control**:
  - Same as `B1_baseline` with $\alpha_i \sim N(0, 1.0)$ and iid errors $\varepsilon_{it} \sim N(0, 1.0)$.
* **Selection / Cohort Assignment**:
  - Selection driver: $driver_i = 1.0 \cdot a_i = a_i$, where $a_i = \alpha_i / 1.0 \sim N(0, 1.0)$ ($c = 1.0$).
  - Units are assigned to one of three treated cohorts ($j \in \{1, 2, 3\}$ with adoption periods $g_1 = 4, g_2 = 5, g_3 = 6$) or never-treated ($j = 0$).
  - Latent utilities for assignment choices:
    $$
    \begin{aligned}
    U_{i0} &= \log(0.25) + \epsilon_{i0} \\
    U_{i1} &= \log(0.25) + 0.8 a_i + 0.5 a_i + \epsilon_{i1} = \log(0.25) + 1.3 a_i + \epsilon_{i1} \\
    U_{i2} &= \log(0.25) + 0.8 a_i + 0.25 a_i + \epsilon_{i2} = \log(0.25) + 1.05 a_i + \epsilon_{i2} \\
    U_{i3} &= \log(0.25) + 0.8 a_i + 0.0 a_i + \epsilon_{i3} = \log(0.25) + 0.8 a_i + \epsilon_{i3}
    \end{aligned}
    $$
    where $\epsilon_{ij} \sim \text{Gumbel}(0, 1)$ are independent.
  - Cohort assignment: $C_i = \arg\max_{j \in \{0, 1, 2, 3\}} U_{ij}$.
  - Adoption time:
    $$
    G_i = \begin{cases}
    4 & \text{if } C_i = 1 \\
    5 & \text{if } C_i = 2 \\
    6 & \text{if } C_i = 3 \\
    \infty & \text{if } C_i = 0
    \end{cases}
    $$
* **Treatment Effect / Observed Outcome**:
  - Unit base treatment effect: $\tau(X_i) = 2.0 + 1.5 X_{1i} + 0.75 \tanh(X_{2i})$ (since $\tau_0 = 2.0$, type is `heterogeneous`).
  - Realised treatment effect for event-time $k = t - G_i \ge 0$:
    $$
    \text{CATT}_{it} = m_{G_i} \cdot (1 + 0.4 \cdot k) \cdot \tau(X_i)
    $$
    where cohort multipliers are $m_{g_1} = 1.0$, $m_{g_2} = 1.5$, $m_{g_3} = 2.0$, and the dynamic ramp is $r = 0.4$.
  - Observed outcome: $Y_{it} = Y_{it}(0) + \text{CATT}_{it} \cdot \mathbb{1}[t \ge G_i]$.

### 9. `D_contamination`
* **Type**: Staggered adoption (D)
* **Objective**: Intensify cohort and event-time heterogeneity to analyze TWFE estimation failure (via Goodman-Bacon decomposition).
* **Sample Size**: $N = 200$, $T = 8$.
* **Mathematical Formulations**:
  - Identical to `D_staggered`, except the cohort multipliers and dynamic ramp are stronger:
    - Cohort multipliers: $m_{g_1} = 1.0, m_{g_2} = 2.0, m_{g_3} = 3.0$.
    - Dynamic ramp: $r = 0.8$.
  - Realised treatment effect for event-time $k = t - G_i \ge 0$:
    $$
    \text{CATT}_{it} = m_{G_i} \cdot (1 + 0.8 \cdot k) \cdot \tau(X_i)
    $$
  - All other formulas are identical to `D_staggered`.

### 10. `D_ramp_nt40` … `D_ramp_nt05` — the already-treated-weight ramp
* **Type**: Staggered adoption (D)
* **Objective**: Move the **Goodman-Bacon weight on already-treated
  comparisons** while holding the effect DGP fixed, so every estimator's error
  can be plotted against it (Reviewer 3.1.2).
* **Mathematical Formulations**: identical to `D_staggered` ($m = (1, 1.5, 2)$,
  $r = 0.4$) except for the cohort shares. With never-treated share
  $\pi_\infty \in \{0.40, 0.30, 0.25, 0.20, 0.10, 0.05\}$, each treated cohort
  receives $\pi_j = (1 - \pi_\infty)/3$.
* **Why this knob and not `dynamic_ramp`.** The Goodman-Bacon weights are
  determined by the adoption design alone — group sizes $n_g$ and the fraction
  of the panel $\bar D_g$ each group spends treated. `dynamic_ramp` changes the
  2×2 comparison *estimates*, never their weights, so the earlier sweep reported
  $w_{\text{already-treated}} = 0.1158$ at every point. Shrinking the clean
  control pool moves it, measured over 200 replications:

  | $\pi_\infty$ | 0.40 | 0.30 | 0.25 | 0.20 | 0.10 | 0.05 |
  |---|---|---|---|---|---|---|
  | $w_{\text{already-treated}}$ | 0.077 | 0.100 | 0.116 | 0.136 | 0.196 | 0.250 |
  | TWFE bias | −0.26 | −0.39 | −0.47 | −0.57 | −0.89 | −1.18 |

  `D_ramp_nt25` has exactly `D_staggered`'s design and reproduces its 0.116 /
  −0.468, which is the consistency check on the family.

### 11. `PT_hold`, `PT_conditional`, `PT_violation_g*`, `PT_violation_a*`
* **Type**: Canonical DiD (workstream PT)
* **Objective**: Operating characteristics of the pre-trend diagnostic — size,
  power, and discrimination between conditional and unconditional PT.
* **Mathematical Formulations**: all are `B1_baseline` with one change.

  | scenario | change | conditional PTA |
  |---|---|---|
  | `PT_hold` | none | holds |
  | `PT_conditional` | `selection="observable"`, $c = 0$, $\rho_{\text{tr}} = 0.6$ | holds (but *unconditional* PT fails: $X_4$ drives both assignment and slope) |
  | `PT_violation_g{05,10,20,40}` | $Y_{it}(0) \mathrel{+}= \delta \cdot \mathbb{1}[G_i \ne \infty] \cdot t$, $\delta \in \{0.05, 0.1, 0.2, 0.4\}$ | **violated** |
  | `PT_violation_het{10,20,40}` | $Y_{it}(0) \mathrel{+}= \kappa (2 X_{1i} - 1) \mathbb{1}[G_i \ne \infty] \cdot t$, $\kappa \in \{0.1, 0.2, 0.4\}$ | **violated**, but *zero on average* |
  | `PT_violation_a20` | $Y_{it}(0) \mathrel{+}= \lambda \cdot a_i \cdot t$, $\lambda = 0.2$ | **violated**, and unremovably so — $a_i$ is unobserved |

  `PT_conditional` is the first discriminating case: the estimator's assumption
  is satisfied, so a covariate-conditional diagnostic should stay quiet, while a
  marginal event study (which does not condition) should flag. Reported
  detection rates for both make that a measurement rather than a claim.

  `PT_violation_het*` is the second, and runs the other way. $X_1$ is
  Bernoulli(0.5) and, under selection on unobservables, balanced across arms, so
  the treated-minus-control *average* differential slope is zero: the marginal
  event study is powerless here by construction, at every $\kappa$, and so is
  the aggregate $\Delta(k)$. Only the contrast between subgroup pre-trends
  (`PRE_SUBC`: $X_1 = 1$ minus $X_1 = 0$, true value $2\kappa$) can see it, and
  that contrast is an object no group-level placebo regression can form. Because
  the contrast is identically zero whenever the violation is constant in $X$,
  its rejection rate under `PT_hold` *and* under the whole `PT_violation_g*`
  grid is a size, which is what makes its rejection rate here a power.

  `PT_violation_a20` is a single mechanism check rather than a magnitude grid.
  At $c = 1$ the treated-minus-control gap in $a_i$ is close to 1, so $\lambda$
  and $\delta$ are numerically the same knob: measured over 50 replications the
  $a$ and $g$ families produced pre-trend slopes of 0.110/0.108, 0.202/0.199 and
  0.397/0.391, per-replication correlations of 0.97–0.99, and ATT biases of
  0.269/0.267. Running both as full grids traced one curve twice.

---

## Identification specifications (the three repairs)

`Results/identification_note.tex` shows that with an unrestricted prognostic
forest over `(D_i, t, X)` the pair `(mu + c·tau·D, (1−c)·tau)` has the same
likelihood for every `c ∈ [0,1]`, so `mu` and `tau` are not separately
identified and only the priors break the tie — more decisively the more data
there are, which is why the bias *grows* in `N`. The note proposes three
repairs; all three are implemented and selectable with `--spec`.

| `--spec` | route | what changes |
|---|---|---|
| `published` | — | the submitted specification, as the revision runs it |
| `published_constant_pi` | — | the same, with the original notebooks' `pi = 0.5` |
| `rfx` | 1 | group indicators out of `mu`; additive **group** random intercept |
| `rfx_unit` | 1 | same, with **unit**-level random intercepts |
| `propensity` | 2 | group indicators out of `mu`; unit-level `P(ever treated \| X)` in |
| `rfx_propensity` | 1+2 | both, which the note observes compose |
| `structured` | corrected | `mu(g,t,x) = a(g,x) + b(t,x)` fitted as two separate forests |
| `structured_rfx_unit` | corrected | corrected DiD-BCF plus unit-level random intercepts |

The first two repairs are argument changes to the same `stochtree` call and live
in `did_bcf_revision.did_bcf.SPECS`. The corrected DiD-BCF is a different model
with three forests sharing one residual, and lives in the standalone
`../didbcf_structured/` package, wired in through
`did_bcf_revision/structured.py`. All of them return
the same `FitResult`, so `plain_estimands`, `corrected_estimands` and the whole
metrics layer work on any of them unchanged.

> **A code-level defect found while testing this.** `BCFModel.sample` defaults
> to `propensity_covariate="prognostic"`, and `did_bcf.py` passes no propensity,
> so `stochtree` fits an *internal* BART model of `Z = D_it` on `X` — which
> contains `time` and `treatment_group`, of which `D_it` is a deterministic
> function — and appends the fitted `pi_hat` to `mu`'s split set with non-zero
> weight **regardless of `keep_vars`**. Measured on `B1_baseline`,
> `corr(pi_hat, D_it) = 0.97` at `N = 200` and `0.98` at `N = 800`, with the
> arms perfectly separated, so one split on `pi_hat > 0.5` reproduces `D_it`
> exactly. The note's flat direction needs two splits; this needs one, on a
> covariate purpose-built to be a perfect proxy for the treatment. The original
> notebooks passed the constant `pi_train = 0.5` and so never had this channel,
> which is why the published results are unaffected.
> `published_constant_pi` isolates it.

### The acceptance test

The note's own criterion — *on `B2_sweep`, the bias of both plain and corrected
DiD-BCF must decrease in `N`* — is implemented as a script that runs it for
every specification at once, and also reports the retention `tau_hat / ATT` on
treated post-treatment rows (Table 1 of the note):

```bash
python scripts/run_identification_routes.py --reps 50 --jobs 4
python scripts/run_identification_routes.py --reps 5 --n 200 400 \
    --specs published rfx structured --jobs 4      # quick look
```

It writes `Results/identification/routes_{raw,summary,acceptance}_*.csv`. It
uses a lighter sampler than production by default (`--num-gfr 25 --num-mcmc 200
--keep-every 2 --num-chains 2`), because it measures a bias that is a property
of the model rather than of the number of draws.

For a full production run under one repair, `run_did_bcf.py` takes the same
flag and appends it to the output filename, so route runs never overwrite the
published-specification ones:

```bash
python scripts/run_did_bcf.py --experiment B2_sweep --spec structured --jobs 4
```

Per-replication summaries carry a `spec` column and `metrics.GROUP_KEYS`
includes it, so `aggregate_metrics.py` reports each specification as its own
row instead of pooling it with the published run. Summary files written before
this existed have no such column and are labelled `published`, so their numbers
are unchanged.

---

## On the posterior correction

`posterior_correction.py` implements Algorithm 1 of the theory note faithfully:
per cohort-time cell it forms the **augmented (efficient-influence-function)**
draw `theta^s`, then subtracts the **posterior bias correction** `b_hat^s`,
using the BCF prognostic draws `m^s(x) = mu^s(x,t) − mu^s(x,g−1)`, a pilot
propensity, and Bayesian-bootstrap weights.

It is reported alongside plain DiD-BCF precisely because its status is worth
checking empirically. Two observations:

1. **Algebraically** the `m^s` dependence in the augmentation and in the
   correction largely cancels, so `check_theta^s` is, to first order, a
   **Bayesian bootstrap of the doubly robust DiD estimator** (Sant'Anna–Zhao
   style) with the BCF posterior mean as the outcome-regression pilot. That is a
   legitimate, known construction — but note its posterior *spread* comes from
   the bootstrap over the influence function, **not** from the BCF posterior of
   `tau`. In effect, "DiD-BCF + correction" is closer to DR-DiD-with-Bayesian-
   bootstrap than to a Bayesian update of the BCF treatment posterior.
2. Because the correction **differences the outcome** (`ΔY = Y_t − Y_{g−1}`), it
   removes the unit effect `alpha_i`, whereas plain DiD-BCF models *levels* and
   absorbs only a group intercept. So in the B1 settings we expect the
   correction to help most exactly where `alpha_i` (or AR(1) errors) makes plain
   DiD-BCF intervals over-confident — and to be redundant (Corollary on the
   Donsker regime) when the prognostic surface is smooth. The simulations are
   designed to show **where each holds**, which is the honest way to settle
   whether the correction is doing something real or is an ad-hoc add-on.

Implementation choices worth knowing:

* The pilot propensity defaults to cross-fitted logistic regression
  (`--propensity rf`, `--n-splits` to change); it is **separate** from any
  propensity BCF estimates internally.
* For panel data the cell construction uses **one differenced observation per
  unit**, so the unit-level Bayesian bootstrap here *is* the cluster bootstrap
  of the theory note's clustered-data remark.
* `m^s(x)` is read directly from `mu_hat_train` at the unit's period-`t` and
  period-`(g−1)` rows; this is exact because the B1/D covariates are
  time-invariant. For a DGP with time-varying covariates, predict the prognostic
  at constructed rows via `BCFModel.predict(..., terms="mu")` instead.

---

## Reproducibility / environment

* DiD-BCF needs `stochtree` (`pip install stochtree`); the metrics / TWFE /
  Goodman-Bacon layers need only `numpy`, `pandas`, `scikit-learn`,
  `matplotlib`, `openpyxl` (and `joblib`, `tqdm` for parallelism/progress).
* `did_bcf.py` imports `stochtree` lazily, so the whole package imports — and
  steps 2–4 run — on a machine without it.
* Every DGP is seeded by the replication index, so results are reproducible and
  the same panels feed both the in-memory notebooks and the exported CSVs that
  the R benchmarks read.
* `config.DEFAULT_REPS = 100`; raise `--reps` to 200–500 for the coverage/size
  settings, as the revision plan recommends.
* The R benchmarks additionally need R with `did`, `did2s`, `DoubleML`
  (+ `mlr3`, `mlr3learners`, `ranger`), `synthdid`, and `openxlsx`.

### Not included (out of scope for B1/B2/D)

CATT-capable competitor benchmarks (Workstream C: CFFE, grf-DiD) are not part of
this suite. The standard DiD benchmarks **are** included (TWFE/OLS in `TWFE/`;
Callaway–Sant'Anna, `did2s`, DoubleML and `synthdid` in `R_code/`), and the
`DGPs/data_creation_*.py` scripts export the panels in a tidy CSV layout, so the
Workstream-C methods can be added on identical data when that workstream is
tackled.
