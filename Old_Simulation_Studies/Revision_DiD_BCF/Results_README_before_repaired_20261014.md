# Revision simulation results — Workstreams B, C, D

Everything here is generated from the per-replication summaries produced by the
suite. Nothing is hand-edited.

```
Results/
├── revision_simulation_report.tex   the report (compile with pdflatex, twice)
├── revision_simulation_report.pdf   compiled, 23 pp.
├── identification_note.tex          why mu and tau are not separately
├── identification_note.pdf          identified, whether the original paper
│                                    has the same problem, and the two design
│                                    repairs made on 2026-07-23 (6 pp.)
├── analysis/
│   ├── aggregate_all.py             collect every summary -> aggregated/
│   └── make_analysis.py             aggregated/ -> tables/ + figures/
├── aggregated/
│   ├── metrics_all.csv              bias / SD / RMSE / MAE / MAPE / coverage /
│   │                                length / size / MCSE, per (scenario, d, N,
│   │                                estimand, method)
│   ├── surface_all.csv              CATT-surface RMSE/MAE/MAPE + pointwise
│   │                                coverage, mean ± SD across replications
│   ├── sqrtn_all.csv                sqrt(N)*(estimate - truth), per N
│   ├── coverage_grid.csv            replications available per cell
│   └── all_summaries.csv.gz         the full tidy input (226k rows)
├── tables/                          booktabs/longtable .tex, \input by report
├── figures/                         vector .pdf, \includegraphics by report
├── summaries_twfe_*.csv             TWFE benchmark (run locally for this report)
├── goodman_bacon_*.csv              Workstream D decomposition + ramp sweep
├── twfe_event_study_*.csv           TWFE event study vs truth
└── _staging/                        unpacked Colab archives, run logs, and the
                                     two verification scripts of Section 8
```

## Rebuilding

```bash
python3 Results/analysis/aggregate_all.py     # collect + score
python3 Results/analysis/make_analysis.py     # tables + figures
cd Results && pdflatex revision_simulation_report.tex && pdflatex revision_simulation_report.tex
```

`aggregate_all.py` reads the old-generation `DiD_BCF` summaries from the local
legacy tree (`Old_Simulation_Studies/_moved_from_repo/Simulation_Studies_Revision/DiD_BCF_old_generation/`,
gitignored), plus `R_code/*_datasets/**`, `Results/summaries_twfe_*.csv` and
`Results/_staging/**`, so new fits are picked up automatically as soon as their
summaries land in any of those places. The committed `aggregated/` files remain
the published copy of the old-generation runs; rebuilding them on a machine
without the legacy summaries omits those rows.

## Status of the three defects (2026-07-23)

Defects 1 and 3 are **fixed in `did_bcf_revision/`**; defect 2 is diagnosed but
the estimator change is not yet made. See `identification_note.pdf`. Every
estimator must be re-run: the panels change at `linearity_degree >= 2`.

| | what changed |
|---|---|
| 1. recentring | `plain_estimands(pretrend_recenter=...)` now defaults to `False`; CLI flag inverted to `--pretrend-recenter` |
| 2. mu/tau identification | not yet fixed. Recommended: drop the group indicator from `PROGNOSTIC_COLS` and carry it as `rfx_group_ids_train`. **The published results are unaffected** — the same spec on the original DGP gives `tau_hat/ATT` = 1.028/1.009/1.008 at N=200/800/1600 (`_staging/leakage_original_paper.csv`); the revision's unobserved unit effect is what activates it |
| 3. linearity axis | `d=2` now swaps the nuisance covariates for their Kang–Schafer transforms (outcome and truth unchanged, adjustment made hard); `d=3` additionally makes `tau(X)` quadratic + non-additive. New scenario `B1_selection_both`. Assert with `_staging/check_linearity_axis.py` |

Section 8 of the report documents the original diagnosis with its evidence:

1. **`pretrend_recenter=True`** (the default the whole suite ran with)
   annihilates plain DiD-BCF — it subtracts the treatment forest from itself,
   because `Z = D_it` is zero on every pre-treatment row so the forest carries
   no time variation there. Verified: on one `B1_baseline` replication
   `tau_hat` is 3.655 post and 3.669 at `g-1` against a true ATT of 3.849; the
   reported ATT is 3.655 with the flag off and −0.014 with it on.
2. **The posterior correction's bias grows with N** (−0.64 → −2.12 across the
   sweep). `D_it` is a deterministic function of `treatment_group` and `time`,
   both of which are in the prognostic design, so `mu` can span `tau(X)Z` and
   absorbs a growing share of the effect as N grows. Measured directly
   (`_staging/diag_leakage.py`): the treatment forest retains 0.95/0.50 of the
   true ATT at N=200, 0.47/0.36 at N=800, 0.49 at N=1600, matching the
   attenuation implied by the sweep bias (0.83, 0.67, 0.50, 0.43).
3. **The linearity-degree axis is vacuous.** `d` enters the outcome only as a
   unit-level constant plus a time-level constant, so the two-way demeaned
   outcome is identical across `d` to 6e-15. Callaway–Sant'Anna, Gardner and
   synthdid return bit-identical estimates at d=1,2,3. This is a regression
   against the original study, where covariates were drawn per unit-period and
   the axis did bite (`_staging/diag_linearity_vs_original.py` runs both DGPs
   side by side: demeaned-Y moves by 47.8 across `d` in the original, by 2e-15
   in the revision).

## Note on the pre-existing `Benchmark_Results/`

Left untouched but stale: it predates several benchmark runs, and its
file-name-based grouping drops the DiD-BCF summaries and every N above 200 for
the sweep scenarios. Use `aggregated/` instead.
