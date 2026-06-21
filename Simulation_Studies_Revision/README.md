# DiD-BCF revision simulations (Workstreams B1, B2, D)

New simulation suite for the JBES revision, built on top of
`../Simulation_Studies/` but redesigned to address the reviewer comments in
`../../REVISION_PLAN.md`. It implements:

* **B1 — canonical DiD**: selection on *unobserved, time-invariant* heterogeneity
  (a unit fixed effect `alpha_i` correlated with treatment), persistent
  covariates, AR(1) serially correlated errors, and covariate-dependent trends
  (so parallel trends holds only *conditional* on covariates).
* **B2 — decomposed metrics + sample-size sweep**: bias, variance, RMSE,
  coverage (90/95%), interval length, size/power — plus an `N ∈ {200, 400, 800,
  1600}` sweep (anchored at the base size 200) exhibiting bias→0, variance→0 and
  √N stabilisation.
* **D — staggered adoption** with treatment effects that vary by **both
  event-time and cohort**, the **Goodman-Bacon decomposition** of TWFE, and a
  contamination sweep tracing how TWFE degrades as the weight on
  already-treated comparisons grows.

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
2. (PC, fast)      TWFE/OLS_<scen>_lin_<d>.ipynb        -> Results/summaries_twfe_<scen>_lin_<d>.csv
3. (PC, fast)      DGPs/data_creation_<scen>.py         -> R_code/<scen>_datasets/.../iteration_*.csv
4. (R, fast)       R_code/<scen>_datasets/*.R           -> *_GATE_and_PValues_*.xlsx
5. (PC, fast)      scripts/aggregate_metrics.py         -> Results/metrics_*.csv / .xlsx
6. (PC, parallel)  scripts/run_goodman_bacon.py         -> Results/goodman_bacon_*.csv
7. (PC, fast)      scripts/make_figures.py              -> Results/B2_sweep_*.png
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

# 5. Goodman-Bacon + TWFE vs truth (Workstream D)
python scripts/run_goodman_bacon.py --experiment D_staggered --reps 500 --jobs 8
python scripts/run_goodman_bacon.py --experiment D_contamination --ramp-sweep --jobs 8

# 6. sample-size-sweep figures
python scripts/make_figures.py --setting B2_sweep
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

The **9 scenarios** are `B1_baseline`, `B1_strong_confounder`, `B1_serial_corr`,
`B1_selection_obs`, `B1_null`, `B2_sweep`, `B2_sweep_serial`, `D_staggered`,
`D_contamination`. Each becomes 1 data script, 3 DiD-BCF notebooks, 3 OLS
notebooks, and one `R_code/<scenario>_datasets/` folder of 4 R estimators.

```
Simulation_Studies_Revision/
├── README.md
├── did_bcf_revision/              # importable engine (the source of truth)
│   ├── dgps.py                    # canonical (B1) + staggered (D) DGPs, true_estimands()
│   ├── did_bcf.py                 # stochtree BCF fit + PLAIN estimand summaries
│   ├── posterior_correction.py    # Algorithm 1 (DR post-processing) -> CORRECTED summaries
│   ├── runner.py                  # DiD-BCF Monte-Carlo driver (used by DiD_BCF/ notebooks + CLI)
│   ├── twfe.py                    # static + event-study TWFE with cluster-robust SEs
│   ├── twfe_runner.py             # TWFE Monte-Carlo driver (used by TWFE/ notebooks + CLI)
│   ├── metrics.py                 # bias/var/RMSE/coverage/length/size-power (B2)
│   ├── goodman_bacon.py           # Goodman-Bacon (2021) decomposition
│   ├── exports.py                 # tidy frame -> R-benchmark CSV column layout
│   └── config.py                  # the scenario grid (N=200, linearity 1/2/3)
├── DGPs/                          # one data-creation script per scenario (-> R CSV panels)
│   ├── data_creation_B1_baseline.py ... data_creation_D_contamination.py   (9)
├── DiD_BCF/                       # DiD-BCF: one notebook per scenario × linearity (27; Colab)
│   ├── DiD_BCF_B1_baseline_lin_1.ipynb ... DiD_BCF_D_contamination_lin_3.ipynb
├── TWFE/                          # OLS benchmark: one notebook per scenario × linearity (27; PC)
│   ├── OLS_B1_baseline_lin_1.ipynb ... OLS_D_contamination_lin_3.ipynb
├── R_code/                        # R benchmarks, one folder per scenario (9 × 4 scripts)
│   └── <scenario>_datasets/{did_dr_new.R, did2s.R, DoubleML_did.R, synthdid.R}
├── scripts/                       # the cheap / parallel local steps
│   ├── run_did_bcf.py             # headless equivalent of the DiD_BCF/ notebooks
│   ├── run_twfe.py                # headless equivalent of the TWFE/ notebooks
│   ├── aggregate_metrics.py       # decomposed metrics, all methods (B2)
│   ├── run_goodman_bacon.py       # Goodman-Bacon + TWFE-vs-truth (D)
│   ├── make_figures.py            # sample-size-sweep figures
│   └── scaffold_suite.py          # regenerates DGPs/, DiD_BCF/, TWFE/, R_code/ from templates
└── Results/                       # all outputs land here
```

### Estimand schema (shared everywhere)

`dgps.true_estimands`, `did_bcf.plain_estimands` and
`posterior_correction.corrected_estimands` all emit the same estimand set:

| `estimand_type` | meaning |
|---|---|
| `GATT` | cohort × calendar-time cell, `g=<g>_t=<t>`, for `t ≥ g` |
| `ES`   | event-study, `k=<k>` (averaged over cohorts), `k ≥ 0` |
| `ATT`  | overall average over all treated post observations |

Per-replication summary rows carry `post_mean, sd, q025, q05, q95, q975,
p_bayes` for each `method ∈ {plain, corrected, twfe}`, plus the `true` value and
a `linearity_degree` column, so `metrics.compute_metrics` aggregates every model
and every linearity degree in one table. (Plain TWFE reports only `ES` and `ATT`
— pooling cohorts into clean `GATT(g,t)` is exactly the contamination it suffers
from, so `GATT` is left to DiD-BCF and the Callaway–Sant'Anna R benchmark.)

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

**Prognostic covariate level** $f(X_i)$, by `linearity_degree` $d$:

$$
f(X_i)=
\begin{cases}
-0.75X_{1i}+0.5X_{2i}-0.5X_{3i}-1.3X_{4i}+1.8X_{5i}, & d=1,\\[2pt]
-0.75X_{1i}^2+0.5\,e^{X_{2i}/2}-0.5X_{3i}-1.3X_{4i}+1.8X_{5i}, & d=2,\\[2pt]
-0.75X_{1i}+0.5|X_{2i}|+0.8\sin(2X_{3i})-1.3\sqrt{|X_{4i}|}+1.8X_{5i}^2, & d\ge 3.
\end{cases}
$$

**Common time effect** $\gamma_t=\beta_{\text{time}}\,t$ for $d<3$ and
$\gamma_t=\beta_{\text{time}}\,t^2$ for $d\ge 3$, with $\beta_0=-0.5$,
$\beta_{\text{time}}=0.2$.

**Covariate-dependent trend** (makes parallel trends hold only *conditional* on
$X$): $s(X_i)=\rho_{\text{tr}}\,X_{3i}$ with $\rho_{\text{tr}}=$
`trend_heterogeneity`.

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

Staggered adds `cohort_offsets`, `cohort_shares`, `cohort_multipliers` and
`dynamic_ramp` (effect grows with exposure) — the cohort × event-time
heterogeneity that breaks TWFE (R3.1.2).

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
