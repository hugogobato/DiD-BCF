# Run plan — pre-trend diagnostic (F4) and the Goodman-Bacon ramp (D)

Two remaining revision items, both of which needed new experiments rather than
new prose. This is what was built, why the previous version did not answer the
question, and exactly what to run where.

---

## 1. What was wrong, and what replaces it

### Item 3 — the pre-trend diagnostic was proposed but never demonstrated

The manuscript proposes the diagnostic; nothing in the suite ran it. It also
could not be bolted onto the existing fits: the estimation model uses
`Z = D_it`, which is zero on every pre-treatment row, so those rows carry **no
likelihood information about `tau`** and `tau_hat` at a pre-treatment row is
just the post-treatment estimate re-evaluated at a different time value. (That
is the documented cause of the `pretrend_recenter` defect, which annihilated the
estimate rather than recentring it.)

**What is new.** `did_bcf_revision/pretrend.py` fits the *unconstrained*
specification as a **separate diagnostic model** — `Z = 1[G_i != inf]`, on in
every period, so `tau(X, k)` is estimable before adoption — and reports

```
Delta(k) = E[ tau(X, k) - tau(X, -1) | ever treated ],     k < -1
```

which is exactly zero under conditional parallel trends and equals
`differential_slope * (k + 1)` under a violation. Three specification choices
are load-bearing and documented in that module: the group indicators leave
`mu`'s split set (otherwise `mu` spans `tau(X,k) W_i` and the diagnostic is
unidentified), stochtree's internal propensity is switched off (it is a
near-perfect proxy for the group, which re-opens the same channel), and
unit-level random intercepts absorb the level gap that conditional PT *permits*.

**New DGP knobs** (`dgps.py`, both default `0.0`, so every pre-existing scenario
is bit-identical — verified by hashing `B1_baseline`, `B1_selection_obs`,
`D_staggered`, `D_contamination`):

| knob | violation |
|---|---|
| `group_trend` (delta) | `delta * 1[ever treated] * t` — the textbook differential path; readable directly in outcome units per period |
| `alpha_trend` (lambda) | `lambda * a_i * t` — the unobserved confounder drives the *slope* too, so no adjustment on observables can remove it |

**New scenarios** (`config.py`, workstream `PT`): `PT_hold` (size),
`PT_conditional` (conditional PT holds, *unconditional* PT fails — the
discrimination case), `PT_violation_g05/g10/g20/g40` (power curve) and
`PT_violation_a10/a20/a40`.

Sanity check at production sampler settings, 3 replications per cell — the
pattern the full runs should sharpen:

| scenario | true slope | diagnostic slope | diagnostic flags | TWFE ES flags |
|---|---|---|---|---|
| `PT_hold` | 0 | 0.015 | 0/3 | 0/3 |
| `PT_conditional` | 0 | 0.052 | **0/3** | **2/3** |
| `PT_violation_g10` | 0.10 | 0.092 | 0/3 | 0/3 |
| `PT_violation_a20` | 0.19 | 0.177 | 3/3 | 1/3 |

Two things to note. The diagnostic recovers the violation slope nearly
unbiasedly while the unconditional TWFE event study is attenuated (0.056 against
a true 0.10), and it stays quiet in `PT_conditional` where the assumption the
estimator actually needs is satisfied. Both are reasons to prefer it, and both
are measured rather than asserted.

Also: the **slope** rule (one statistic for the differential trend) is much more
powerful than the any-`k` Bonferroni rule. Both are reported; the slope rule is
the headline.

### Item 4 — the Goodman-Bacon ramp did not vary what it claimed to

`goodman_bacon_ramp_sweep_*.csv` swept `dynamic_ramp` and reported
`w_already_treated = 0.1157720184` at **every** point. That is not a rounding
artefact: the Goodman-Bacon weights are a function of the adoption design alone
(cohort sizes and how long each cohort is treated), so `dynamic_ramp` changes
how *inconsistent* the bad comparisons are and never how much weight they carry.
The sweep therefore could not support a claim about "as already-treated
comparison weight increases", and it reported TWFE only.

**What is new.** `--design-sweep` shrinks the never-treated pool with the effect
DGP held fixed, which is the knob that does move the weight — measured, not
assumed:

| never-treated share | 0.40 | 0.30 | 0.25 | 0.20 | 0.10 | 0.05 |
|---|---|---|---|---|---|---|
| weight on already-treated comparisons | 0.077 | 0.100 | 0.116 | 0.136 | 0.196 | 0.250 |
| TWFE bias | −0.26 | −0.39 | −0.47 | −0.57 | −0.89 | −1.18 |

(The 0.25 column reproduces `D_staggered`'s existing 0.116 / −0.468 exactly,
which is the consistency check that the new designs are the same family.)

Six matching scenarios `D_ramp_nt40 … D_ramp_nt05` run **every estimator in the
suite** on these designs, and `make_ramp_analysis.py` plots each one's error
against the *measured* weight. Because estimators report different estimands
natively (overall ATT, cohort-time cells, event-study coefficients), the figure
is drawn in three internally consistent panels rather than forcing one number.

Expect the honest answer to have two parts: TWFE's **bias** explodes, while the
robust estimators stay unbiased but pay in **variance** as the clean control
pool shrinks. The second panel row (95% coverage) is there to show that.

---

## 2. Run it

### The local / Colab split for the R benchmarks

All four R estimators install and run locally, but they are not equally cheap.
DoubleML fits random-forest nuisances and was still running at **67 minutes** on
the first of six ramp designs, against roughly two minutes for the other three
combined. So it, and grf-DiD (which additionally wants its own R toolchain), go
to Colab; the three fast estimators stay local:

```bash
# local: Callaway--Sant'Anna, Gardner, synthdid across all six designs (~15 min)
python scripts/run_r_benchmarks.py \
    --scenarios D_ramp_nt40 D_ramp_nt30 D_ramp_nt25 D_ramp_nt20 D_ramp_nt10 D_ramp_nt05 \
    --scripts did_dr_new did2s synthdid --reps 200 --batch-size 1

# Colab: 6 + 6 notebooks, generated from the checked-in reference notebooks
python scripts/scaffold_colab_benchmarks.py --scenarios D_ramp_nt40 ... D_ramp_nt05
#   -> DoubleML_Colab/DoubleML_D_ramp_nt<xx>.ipynb      (6)
#   -> CFFE_Wang_Colab/CFFE_Wang_D_ramp_nt<xx>.ipynb    (6)
```

grf-DiD is worth having on the ramp for a second reason: it reports cohort-time
cells, so it populates the middle panel of `fig_ramp_methods.pdf`, which
otherwise depends entirely on Callaway--Sant'Anna and DoubleML.

`scaffold_colab_benchmarks.py` retargets a checked-in reference notebook
(`B1_baseline` for canonical, `D_staggered` for staggered) rather than carrying
its own copy — the same fix applied to the R scripts, for the same reason. The
reference notebook is skipped, so a run cannot damage its own source. Verified:
the generated notebooks differ from the reference only in the scenario name, the
scenario `note`, and `REPS`.

### Already done locally

```
scripts/run_goodman_bacon.py --experiment D_staggered --design-sweep --reps 200
scripts/run_twfe.py --experiment D_ramp_nt{40,30,25,20,10,05} --reps 200
```

### Before any Colab notebook: push the engine

**Every** notebook here (`Pretrend/`, `DiD_BCF/`, `DoubleML_Colab/`,
`CFFE_Wang_Colab/`) starts by `git clone`-ing
`https://github.com/hugogobato/DiD-BCF.git` and importing the engine from the
clone. None of the new work is on that remote yet, so a notebook run now fails at
the first `get_experiment` call with

```
KeyError: No experiment named 'D_ramp_nt05'.   (or 'PT_hold')
```

Commit and push at least:

| path | needed by |
|---|---|
| `did_bcf_revision/config.py` | every new notebook (the scenarios themselves) |
| `did_bcf_revision/dgps.py` | `Pretrend/` (the `group_trend` / `alpha_trend` knobs) |
| `did_bcf_revision/pretrend.py`, `pretrend_runner.py` | `Pretrend/` |
| `did_bcf_revision/runner.py` | `DiD_BCF/` (the `rep_start` / `rep_end` arguments) |
| `R_code/D_ramp_*_datasets/*.R` | `DoubleML_Colab/` (it runs the scenario folder's script) |
| the notebooks themselves | only if you open them from the repo rather than uploading |

### Colab — 23 notebooks for DiD-BCF, plus 12 for the two heavy benchmarks

One fit is ~100 s locally at `N = 200`; budget ~2 min on a Colab CPU VM. **Run
the first three replications and extrapolate before committing to a notebook.**
`JOBS = 2` needs about 1 GB per worker.

**Item 3 — `Pretrend/Pretrend_<scenario>_lin_<d>.ipynb` (13 notebooks)**

| notebook | `WITH_ATT` | fits | ~wall time at `JOBS=2` |
|---|---|---|---|
| `PT_hold_lin_1` | True | 400 | 6.7 h |
| `PT_conditional_lin_1` | True | 400 | 6.7 h |
| `PT_violation_g05/g10/g20/g40_lin_1` | True | 400 each | 6.7 h each |
| `PT_violation_a10/a20/a40_lin_1` | False | 200 each | 3.3 h each |
| `PT_hold/PT_conditional/PT_violation_g20/PT_violation_a20_lin_2` | False | 200 each | 3.3 h each |

`WITH_ATT` is preset per degree in the generated notebooks. If a session looks
like it will not finish, set `REP_START, REP_END = 0, 100` in one copy and
`100, 200` in another — replications are seeded by index, so the two parts
concatenate into exactly the undivided run and `aggregate_all.py` de-duplicates.

**Item 4 — `DiD_BCF/DiD_BCF_D_ramp_nt<xx>_lin_<d>.ipynb` (10 notebooks)**

`REPS` is already 200 in these notebooks: the `D_ramp_*` scenarios carry that
count in `config.py` and the notebooks now take it from there, so the DiD-BCF
runs match the TWFE and R-benchmark runs replication for replication. (A ramp
point where methods differ in replication count would confound the comparison
with Monte-Carlo noise, which is exactly what this figure is measuring.)

**Set `JOBS = 2`** in these — unlike the `Pretrend/` notebooks, the DiD-BCF
template still defaults to `JOBS = 1`, which doubles the times below.

| notebooks | ~wall time at `JOBS=2` |
|---|---|
| `nt40, nt30, nt25, nt20, nt10, nt05` at `lin_1` | 3.3 h each |
| `nt40, nt05` at `lin_2` and `lin_3` (the anchors) | 3.3 h each |

If budget allows, the remaining `lin_2` / `lin_3` cells (8 more notebooks) fill
in the degree axis; the contamination result is a property of the design, so
degree 1 carries the argument.

### Empirical application (local, ~25 min; already run)

```bash
python scripts/run_pretrend_empirical.py --with-att
```

Runs the diagnostic on `mpdta` (500 counties, 2003–2007, cohorts 2004/2006/2007)
with `mu` on `(lpop, year)` and `tau` on `(lpop, k)`, and writes
`Results/empirical/pretrend_mpdta.csv` plus a two-panel figure: `Delta(k)` with
credible intervals against the TWFE event-study placebo, and the same within
`lpop` terciles. Available pre-treatment event times are `k = -4, -3, -2`
(cohort 2004 has only 2003, the reference, so it contributes to none of them).

**Result.** `Delta(k) = 0.006 / 0.021 / 0.022` log employment at
`k = -4 / -3 / -2`, with an implied differential slope of `-0.006`
(`p_bayes = 0.248`); the any-`k` Bonferroni rule gives `0.072`. Both `lpop`
terciles behave the same way. So the diagnostic finds **no evidence against
conditional parallel trends** in the application, agreeing with the TWFE
event-study placebo (slope `-0.006`, `p = 0.143`) — which is the outcome that
supports the paper's application rather than undermining it. Note `k = -2` is
individually marginal (`p_bayes = 0.024`) and only survives as a non-rejection
after the multiplicity correction; worth stating plainly rather than rounding
away.

> **`--with-att` uncovered a separate problem — read
> [`Results/empirical/ATT_SEED_INSTABILITY.md`](Results/empirical/ATT_SEED_INSTABILITY.md).**
> Two things happened here. First, routing the constrained fit through
> `did_bcf.fit_did_bcf` was wrong: that builds the *simulation* design matrix,
> dropping `first.treat` from `mu`'s split set, and under staggered adoption a
> prognostic forest that cannot see the cohort pushes cohort-level differences
> into `tau` (ATT came out +0.269). `paper_att()` now transcribes the
> application notebook exactly. Second, and more seriously, **that corrected
> specification still does not reproduce the published −0.143 — and cannot,
> because the estimate is not stable.** Varying only the MCMC seed gives
> +0.105 / −0.075 / −0.045 on the notebook's own statistic: the point estimate
> changes sign. The posterior SD of the ATT is 0.27--0.39 (the notebook's KDE
> shows cross-county spread of the fitted surface, ~0.02, not posterior
> uncertainty), and the seed swing is ~9x the implied Monte-Carlo SE, which
> points at the flat likelihood direction of `identification_note.tex` rather
> than at sampling noise. This needs resolving before resubmission; it does not
> affect the pre-trend diagnostic above, which is a separate fit.

### Then, locally

```bash
python Results/analysis/aggregate_all.py            # picks up the new files
python Results/analysis/make_pretrend_analysis.py   # item 3 table + figure
python Results/analysis/make_ramp_analysis.py       # item 4 table + figure
```

Outputs: `Results/tables/tab_pretrend.tex`, `tab_ramp_methods.tex`;
`Results/figures/fig_pretrend.pdf`, `fig_ramp_methods.pdf`;
`Results/aggregated/pretrend_all.csv`, `ramp_methods.csv`.

`make_ramp_analysis.py` prints which ramp designs still lack estimator runs, so
it is safe to run early to check progress.

---

## 3. Incidental fixes made while doing this

* **`scripts/run_r_benchmarks.py` copied nothing back.** Its filter required an
  output filename to start with the script name or contain
  GATE/Metrics/Estimates/output; the scripts write
  `summaries_<method>_<scenario>_lin_<d>.csv`, which matches none of those. The
  runner reported success and produced no summaries. It now collects every
  result file once all four scripts for a scenario have finished. Its scenario
  list is also derived from `config.py` instead of hard-coded.
* **`scripts/scaffold_suite.py` reverted the R scripts.** It carried its own
  copies of the four R templates, which had since been revised to emit the
  DiD-BCF summary schema; a scaffold run replaced 3222 lines of the current
  scripts with the older ones. The embedded templates are gone: scripts are now
  copied from the checked-in reference scenario for the matching DGP
  (`B1_baseline` / `D_staggered`) with only the scenario name substituted, and
  the reference is skipped so it can never overwrite itself. Verified: a
  scaffold run now leaves every tracked R script byte-identical.
* **`.gitignore` did not cover new scenarios.** It listed generated R panels
  scenario by scenario, so the `D_ramp_*` families' ~730 MB of CSVs would have
  been committed. Replaced with scenario-agnostic patterns.
* **`--reps` was silently ignored** for scenarios carrying a replication floor
  (`max(reps, 200)`), so a 4-replication smoke test ran 200.
* **Figure style extracted** to `Results/analysis/style.py` and imported by
  `make_analysis.py`, so the new figures share the report's palette and legend
  instead of duplicating it. The existing report rebuilds unchanged.
