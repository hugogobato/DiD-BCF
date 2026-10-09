# Replicated paper-metric comparison, version 2026-10-09

The user requested 30 replications per simulation cell, comparing the repaired engine at the earlier proper-prior sampling budget and B3, with the saved proper-prior campaign as a reference. This is a preliminary performance study. It does not require every internal parameter to pass the previous diagnostic-only precision gate, and it does not relabel that gate as passed. It does not authorize a full production rerun or declare a publication-final sampling budget.

## Frozen cases and replications

Use replication indices 0 through 29 in each of the following nine cells. The cells were discussed before launching this new study. The previous N=800 serial-correlation anchor at replication 82 remains preserved but is outside this fixed 30-replication set. Previously observed problematic runs are never used to remove a cell or replace a replication.

| Setting | N | Linearity degree |
| --- | ---: | ---: |
| B1_baseline | 200 | 1 |
| B1_baseline | 200 | 3 |
| B1_null | 200 | 1 |
| B1_selection_obs | 200 | 1 |
| B1_serial_corr | 200 | 1 |
| D_staggered | 200 | 1 |
| B2_sweep | 800 | 1 |
| B2_sweep_serial | 100 | 3 |
| B2_sweep_serial | 800 | 2 |

These are 270 datasets and 540 newly sampled fits. They cover selected representative and difficult settings, not the full manuscript's simulation universe. Application, prior-sensitivity and pretrend-test phases are outside this study. Application estimates have already been compared in the overnight report.

## Three comparison arms

All arms use the raw total-effect estimand, the original DGP/job IDs, outcome standardization, forest architecture and proper IG(2,1) noise prior. New sampling uses only the pinned isolated repaired source. Numeric features and fixed equal eligible-feature weights are unchanged.

| Arm | Engine | GFR | Ordinary burn-in | Retained draws per chain | Thinning | Chains |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Saved proper-prior reference | Unpatched | 50 | 0 | 500 | 5 | 3 |
| Repaired earlier budget | Repaired | 50 | 0 | 500 | 5 | 3 |
| Repaired B3 | Repaired | 50 | 2,500 | 2,000 | 5 | 4 |

The earlier-budget arm has a new name because the previous B0 label meant archived results only. There are 1,890 new independent chain units, with seeds `1000 * replication + chain_index`. Each fit pools all of its complete chains in index order. No historical and repaired draws are pooled, and no incomplete chain is appended as if it were a continuation. New fits start fresh even where an overnight dataset overlaps, so this study has a single manifest and recovery identity.

The saved reference was run in its recorded Colab environment, including its recorded interpreter. The new arms use the matching local pinned Python dependencies and Python 3.12.3. The repaired-arm comparison isolates the sampling-budget change within the same local engine/environment. Comparison with the saved campaign also changes the engine and interpreter; it is not an isolated native-kernel experiment. Preparation must verify all 270 regenerated panel hashes against the archived reference's verified panel hashes before sampling.

## Evaluation and uncertainty

Use the existing paper metric definitions for ATT, every cohort/time GATT and event-time mean, and the treated-row conditional-effect surface. Report bias, RMSE, MAE, MAPE where truth is nonzero, empirical SD, posterior SD, 90%/95% interval coverage and length, and size/power. Report heterogeneous-effect surface RMSE/MAE/MAPE, pointwise coverage and interval length averaged across dataset replications. These are CATT surfaces; this study does not establish transported population CATE without the additional identification assumptions.

Scalar bias MCSE is additionally computed from the SD of estimate minus replication-specific truth. The inherited estimate-SD-based MCSE remains separately labeled. Report paired differences and paired RMSE ratios, with 5,000 dataset-level bootstrap resamples and a fixed reporting seed. Resample replication indices jointly across arms within each cell and target; never treat cohort/time targets or treated observations as independent simulation replications. If a historical optional metric was not stored, leave it missing rather than imputing it.

Thirty replications support a preliminary comparison. Coverage changes by 1/30 with one replication, and nominal 95% coverage has a binomial standard error of approximately four percentage points at this size. Report counts, Monte Carlo uncertainty and paired differences. No formal equivalence margin is specified, so nonsignificant differences must not be called proof of equivalence. No favorable pilot metric automatically releases a full campaign or changes the budget.

## Diagnostics, failures and preservation

Preserve chain agreement, effective sample sizes, mean and endpoint MCSEs, noise/leaf-scale diagnostics, and traces as supporting evidence. Publish the frequency and severity of flags alongside the paper metrics. No R-hat/ESS filter is used to select replications or posterior draws. A numerical failure is retried only as a fresh attempt with the same recorded seed and budget, with failed attempts retained. Unresolved missing/failing fits prevent a complete paired report; they are never silently scored as successful coverage or omitted from the denominator.

A RAM-reserved local process pool uses at most six workers and one numeric/OpenMP thread per worker. It adapts concurrency to accessible CPU affinity and available RAM, reserves system memory, and checks free disk. Completed chains are checked and checkpointed individually. A local 72-hour deadline preserves completed work on interruption. Collection and the paired report run automatically after all requested units complete. The collector's timing inventory is redirected into this new namespace so the frozen recovery inventory is not overwritten.

The original sources, manifests, archived experiments, previous pilot and its failed full-state gate remain untouched. This study and its reports have a separate namespace. Self-contained Colab inputs remain individual notebooks; notebooks download one output ZIP per shard with the standard non-Colab fallback.
