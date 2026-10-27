# Final paper experiments

The two final DiD-BCF campaign packages are [`DiD_BCF/Paper_Rerun_20261011/`](DiD_BCF/Paper_Rerun_20261011/README.md) for the main simulations and [`DiD_BCF/Nonmain_Rerun_20261014/`](DiD_BCF/Nonmain_Rerun_20261014/README.md) for pretrend diagnostics, effects under pretrend violations, prior sensitivity and the application. All 14,327 planned jobs are complete. Use these packages to reproduce sampling, and the [Results workflow](Results/README.md) to reproduce publication assets from saved metrics.

For plotting, pass another agent the repository's [visualization handoff](../RESULTS_VISUALIZATION_HANDOFF.md). It identifies exact input files, current/historical labels, target filters and uncertainty conventions. For fitting a new panel, use the [model guide](../MODEL_USAGE.md).

## Study map

| Study | Final inputs |
|---|---|
| Canonical designs: baseline, null, selection and confounding | Main campaign; current main scalar and surface metrics |
| IID and serial-error sample-size sweeps | Main campaign; N=50,100,200,400,800, degree=1,2,3 |
| Staggered adoption and contamination | Main campaign; cohort-time and event-time effects |
| Pretrend diagnostics and post-treatment impact | Non-main campaign; 200 replications in each of 20 cells |
| Noise-prior sensitivity | Non-main rates 0.1 and 10 with rate-1 main anchors; 100 replications per cell/rate |
| Minimum-wage application | Non-main campaign; 27 fits including separately prespecified model, prior and computational-bundle settings |

Main results consist of 5,100 fits across 48 cells. The 9,227 non-main fits include 4,000 diagnostic, 4,000 pretrend-impact, 1,200 simulation-sensitivity and 27 application fits. Each campaign's manifest records exact settings, priors, seeds, assignments, source hashes and reuse evidence. No diagnostic or performance filtering was applied to the final completed collection.

## Supporting code and benchmarks

`DGPs/` and `did_bcf_revision/` contain shared DGP, metric and benchmark utilities. The sealed source copies within each final campaign, together with the repaired native engine, are authoritative for the final DiD-BCF computation. `TWFE/`, `R_code/`, `DoubleML_Colab/` and `CFFE_Wang_Colab/` preserve benchmark replication workflows; `Benchmark_Results/` contains saved benchmark metrics. `Theory_Calibration/` retains source and result inputs for separately labeled correction, reference-fold and information comparisons used by the paper.

Superseded DiD-BCF pretrend notebooks, initial proper-prior reruns, unpatched runner entry points and development reconciliation reports are under [`../Old_Simulation_Studies/`](../Old_Simulation_Studies/README.md). Their archived documentation may assume older paths or contain claims superseded by the final analysis. Historical result generations remain distinct from the repaired final estimator.

The `Proper_Prior_Rerun` compatibility link is for old paths. New users should start from `DiD_BCF/`, `Results/` or the top-level model guide.
