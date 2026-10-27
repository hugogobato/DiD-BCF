# Final paper results: visualization handoff

Use the completed 2026-10-26 results below. Paths are relative to the Git repository root, the directory containing this file. The manuscript lives separately at `../JBES_Submission/` in the author's workspace; a public checkout need not contain that directory to read or plot these CSVs.

## Start here

| Purpose | Directory or file |
|---|---|
| Current simulation results, including benchmark comparisons | [`Simulation_Studies_Revision/Results/aggregated_repaired_20261026/`](Simulation_Studies_Revision/Results/aggregated_repaired_20261026/) |
| Current minimum-wage application | [`Simulation_Studies_Revision/Results/empirical_repaired_20261014/`](Simulation_Studies_Revision/Results/empirical_repaired_20261014/) |
| Existing paper figure code | [`Simulation_Studies_Revision/Results/analysis/`](Simulation_Studies_Revision/Results/analysis/) |
| Existing vector figures and LaTeX tables | [`Simulation_Studies_Revision/Results/figures/`](Simulation_Studies_Revision/Results/figures/), [`tables/`](Simulation_Studies_Revision/Results/tables/) |
| Source, completeness and numerical audits | [Main campaign analysis](Simulation_Studies_Revision/DiD_BCF/Paper_Rerun_20261011/analysis_returned_20261014/), [non-main campaign analysis](Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014/analysis_returned_20261026/) |

The simulation CSVs are small enough to use directly. No ZIP extraction, sampling, native compilation or posterior-array export is needed for the proposed figures.

## Simulation file dictionary

The following files are all in `Results/aggregated_repaired_20261026/` under `Simulation_Studies_Revision/`.

| File | Rows | Plotting use |
|---|---:|---|
| `main_scalar_metrics.csv` | 447 | Final DiD-BCF main-study ATT, cohort-time GATT and event-time ES metrics, aggregated over replications. Contains only the final estimator. |
| `main_surface_metrics.csv` | 48 | Final treated CATT surface accuracy and pointwise interval metrics. One row per main simulation cell. |
| `metrics_all.csv` | 6,604 | Final DiD-BCF scalar metrics alongside retained benchmark and historical comparator rows. Filter method, origin, setting, degree, sample size and target explicitly. |
| `surface_all.csv` | 592 | CATT surface comparisons. Some retained comparator cells do not have a final DiD-BCF counterpart. |
| `main_per_replication.csv` | 52,500 | Posterior summaries and realized truth for each main fit and scalar target, plus per-fit surface summaries. Useful for error distributions and Monte Carlo uncertainty. |
| `nonmain_per_replication.csv` | 172,378 | Diagnostic, pretrend-impact, prior-sensitivity and application summaries. Keep `phase`, `spec` and prior rate separate. |
| `nonmain_scalar_metrics.csv` | 888 | Separate aggregated diagnostic, pretrend-impact and sensitivity metrics; rate-1 sensitivity anchors are supplied separately in `prior_sensitivity.csv`. |
| `nonmain_surface_metrics.csv` | 32 | Pretrend-impact and prior-sensitivity CATT surface metrics. |
| `pretrend_summary.csv` | 180 | Size and power of slope, subgroup-contrast and any-lead diagnostics; includes retained comparators. |
| `pretrend_profiles.csv` | 120 | Average lead estimates versus truth. `empirical_sd_error` is the across-replication standard deviation of estimation error. |
| `prior_sensitivity.csv` | 162 | Noise-prior rates 0.1, 1 and 10, with rate-1 anchors from the completed main study. |
| `prior_sensitivity_paired.csv` | 24 | Paired RMSE ratios and bias differences against rate 1, with fixed-seed 5,000-resample bootstrap intervals. |
| `sqrtn_all.csv` | 1,026 | Summaries of scaled scalar estimation error for sample-size figures. Do not interpret a stable curve alone as proof of a central limit theorem. |
| `coverage_grid.csv` | 712 | Method/cell availability and replication counts, not posterior interval coverage rates. |
| `provenance.json` | JSON | Source identities, counts and checksums for this overlay. |

## Select the estimator and target correctly

In these result files, `method="plain"` is the final structured DiD-BCF with the repaired native engine and proper variance prior. The name is a retained storage label; display it as **DiD-BCF**. Main scalar rows have `result_origin="repaired_main_20261014"`; pretrend-impact scalar rows have `result_origin="repaired_nonmain_ptimpact_20261026"`. `main_scalar_metrics.csv` is the simplest source if no benchmark comparison is needed.

`method="pretrend"` is the final, separately fitted diagnostic model, with `result_origin="repaired_nonmain_pretrend_20261026"` in `metrics_all.csv`. It is not the treatment-effect estimator. `plain_historical` and `pretrend_historical` are superseded implementations and should appear only in explicitly labeled historical comparisons. `corrected` in the central overlay is an earlier posterior-correction comparator; it does not mean the repaired main sampler.

Other method labels are `twfe` (static TWFE), `twfe_es` (TWFE event-study placebo), `did_dr` (Callaway and Sant'Anna), `did2s` (Gardner), `doubleml` (DoubleML DR-DiD), `synthdid` (synthetic DiD), and `wang` (grf-DiD). Retained benchmark rows have `result_origin="historical"` because these methods were not rerun with the repaired DiD-BCF engine. Their stored sample sizes and replication counts must remain visible.

The manuscript also uses a historical **reference-fold** comparator. Its selected scalar results are assembled by `make_jbes_sim_tables.py` and `make_jbes_figures.py` from `Theory_Calibration/results/aggregated_correction/` and `aggregated_completion/`. Those scripts select `task_estimator="reference_fold_convolution_rf"`, `method="reference_fold_convolution"`, and `point_summary="mean"`. For a figure that reproduces the manuscript's reference-fold comparison, use those selections, not the central overlay's generic `corrected` rows. The corresponding diagnostic is already labeled `reference_fold_pretrend` in the final pretrend summary/profile files.

The scalar target is identified by both `estimand_type` and `estimand_id`: `ATT/ATT` is the overall treated average, `GATT/g=4_t=4` is a cohort-time treated average, and `ES/k=0` is an event-time treated average. Do not relabel these empirical ATT/GATT/CATT metrics as population ATE/GATE/CATE without specifying the additional target/transport assumptions. CATT surface metrics average over treated observations; they are in the surface files rather than scalar ATT rows.

The main final grid has 48 cells and 5,100 fits: 100 replications per cell, except 200 in each of the three sharp-null cells. Final main sample sizes are 50, 100, 200, 400 and 800. Historical benchmark rows also include other cells, including N=1600, for which no final main DiD-BCF fit exists. Never extend the final curve using an older DiD-BCF row.

## Metric and uncertainty conventions

`bias` is the mean of estimate minus each replication's realized truth. `rmse`, `mae`, `emp_sd`, `avg_post_sd`, `sd_ratio`, `cover90`, `cover95`, `len90`, `len95`, and `reject05` have their literal meanings; `sd_ratio` is average posterior SD divided by empirical estimate SD. `mape` can be missing or unstable for zero or near-zero effects. Missing values are not zeros.

In final scalar rows, `mcse_bias` is based on the standard deviation of replication-level estimation errors, divided by the square root of their count. `legacy_mcse_bias_estimate_sd` retains the older estimate-SD convention for comparison. The error SD and empirical estimate SD need not coincide when realized truth changes across panels. `cover95_count`, `cover95_n` and `cover95_wilson95_lower/upper` provide the counts and Wilson intervals for coverage. Use stored intervals where available. A single posterior credible interval, Monte Carlo uncertainty across simulation replications, and a bootstrap interval for an RMSE ratio are different quantities.

The surface columns `surf_rmse_mean`, `surf_cover95_mean` and `surf_len95_mean` first evaluate each fit's treated-observation surface and then average across fits. Their `_sd` columns measure across-fit dispersion. They are not posterior SDs. The `*_per_replication.csv` files contain per-fit `surf_rmse`, `surf_cover95` and related columns if a different uncertainty summary is needed.

For final pretrend rows, `slope_detect` and `subgroup_detect` are rejection rates using the stored smaller posterior sign tail, with rejection at `p_bayes < 0.025`. `p_bayes` is not a conventional two-sided p-value. `slope_n`, `slope_count` and Wilson limits describe the overall slope rate. `subgroup_detect` refers to the X1 high-minus-low slope contrast. `anyk_detect` for final DiD-BCF is the Bonferroni any-lead rule (`PRE/any_bonf`); the per-replication `PRE/any` target supplies the unadjusted rule. Tail rejection and percentile interval exclusion can differ when finite draws tie at zero. Historical TWFE uses its recorded event-study decision convention, not this posterior-tail rule.

For `pretrend_profiles.csv`, the mean estimate is `mean_true + bias`. A bar of `empirical_sd_error` shows replication dispersion, not a posterior interval or a standard error of the plotted mean. Diagnostic n=200 in each final cell. The historical TWFE degree-3 `PT_violation_het20` cell has n=150; label that count if used.

## Application files and filters

These files are in `Results/empirical_repaired_20261014/`.

| File | Use |
|---|---|
| `application_primary_estimates.csv` | Eight primary group rows at rate 1 and seed bundle 0: four with unit random intercepts and four without. |
| `application_gates_rep0.csv` | Group posterior means and intervals for all three prior rates, both model specifications, bundle 0. |
| `application_gates_rep1.csv`, `application_gates_rep2.csv` | Same-panel computational repetitions. Plot separately if showing stability. |
| `application_contrasts_rep0.csv` | Pairwise population-group differences with posterior SD and intervals, all rates and both specifications. |
| `application_catt_rep0.csv` | County-year fitted-effect summaries, all three rates and both specifications. Filter before plotting. |
| `application_pretrend_diagnostics.csv` | Diagnostic posterior summaries for all nine rate/bundle combinations. |
| `application_bundle_rep*.json` | Additional compact seed-specific summaries and provenance. |

The primary specification is `spec="structured_rfx_unit"`, `sigma2_rate=1`, `rep=0`; the no-unit-intercept comparison is `spec="structured"`. The primary overall estimate is -4.724160%, with 95% posterior interval [-7.329008%, -2.032059%]. The primary treated surface has 291 county-year observations from 191 counties. For CATT plots, select the primary spec and prior rate; confirm 291 distinct unit/time rows after filtering. Do not combine all 1,746 rows in `application_catt_rep0.csv` into one surface.

The `_pct` group means summarize the draw-wise percent transformation. They are not obtained by transforming `att_log` after averaging. County-year `catt_pct` follows the stored transformation of the posterior-mean log effect. Use the existing percent columns instead of imposing a different transformation. Group contrast intervals are supplied on the log scale. All three primary population-group difference intervals include zero, so a descriptive CATT distribution plot should not assert a supported population-size gradient.

Seed bundles 0, 1 and 2 refit the same application panel. Do not treat them as independent datasets, average interval endpoints, or concatenate their summaries into a newly pooled posterior. The compact application CATT CSV does not contain pointwise posterior draws or pointwise intervals, so it cannot support new CATT credible bands.

## Minimal loading example

Run from the repository root:

```python
from pathlib import Path
import pandas as pd

res = Path("Simulation_Studies_Revision/Results")
sim = res / "aggregated_repaired_20261026"
app = res / "empirical_repaired_20261014"

main = pd.read_csv(sim / "main_scalar_metrics.csv")
att = main.query("estimand_type == 'ATT' and estimand_id == 'ATT'")
iid_sweep = att.query("setting == 'B2_sweep'").sort_values(
    ["linearity_degree", "N"]
)

comparison = pd.read_csv(sim / "metrics_all.csv")
comparison = comparison[
    comparison["method"].isin(["plain", "twfe", "did_dr", "did2s",
                                "doubleml", "synthdid", "wang"])
    & comparison["estimand_type"].eq("ATT")
    & comparison["estimand_id"].eq("ATT")
    & comparison["setting"].eq("B2_sweep")
    & comparison["N"].isin(iid_sweep["N"].unique())
]

pretrend = pd.read_csv(sim / "pretrend_summary.csv")
pretrend = pretrend[pretrend["method"].isin(["pretrend", "twfe_es"])]
prior = pd.read_csv(sim / "prior_sensitivity.csv")
prior_att = prior.query("estimand_type == 'ATT' and estimand_id == 'ATT'")

primary = pd.read_csv(app / "application_primary_estimates.csv")
primary = primary.query("spec == 'structured_rfx_unit'")
catt = pd.read_csv(app / "application_catt_rep0.csv")
catt = catt.query("spec == 'structured_rfx_unit' and sigma2_rate == 1")
assert len(catt) == 291
```

Keep source CSVs unchanged. Save proposed figures and their plotting code in a new review directory; add them to the manuscript only after choosing which existing figures they replace or supplement. Use vector PDF for the paper and report method selection, target, sample size, replication count and uncertainty convention in each caption. Existing outputs and source hashes allow numerical checks without rerunning experiments.
