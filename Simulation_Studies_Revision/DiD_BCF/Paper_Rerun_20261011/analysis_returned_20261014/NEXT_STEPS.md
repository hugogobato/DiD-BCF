# Remaining experiment and manuscript work, 2026-10-14

The main rerun is complete: all 5,100 main jobs pass compact-output verification. Do not repeat the 60 main notebooks. Preserve the 64 returned ZIPs, compact reused evidence and collected metrics. The reader-only adapter `collect_returned.py` handles four numeric-looking hexadecimal job IDs without changing the frozen campaign helpers or any returned result bytes.

Matched point accuracy is close to historical raw across the full grid. ATT RMSE changes span -4.32% to +4.88%, with median +0.50%; mean treated conditional-effect surface RMSE changes span -0.51% to +3.57%, with median +1.50%. The original raw comparator differs in prior, kernel, computational budget and environment, so these comparisons are not a controlled repair-only effect or a formal empirical-equivalence test. All 5,100 repaired jobs enter the full metrics, while the historical paired comparison uses 4,800 matched datasets. The 300 additional repaired null replications remain in the full metrics.

The practical recommendation is to retain the earlier repaired budget for the remaining performance-focused experiment campaign: GFR 50, additional burn-in 0, 500 retained draws, thinning 5 and three chains. Its six non-main ATT anchors differ from B3 by at most 0.132 B3 posterior standard deviations, and none of the 48 reported pretrend target/decision rejections changes. Single-dataset anchors do not establish repeated-sample equivalence. One subgroup pretrend interval changes substantially in width, and the prior full-state convergence gate remains not passed. Preserve these details in supporting evidence and define the next version's policy explicitly.

## Remaining original scope

| Phase | Original jobs | Completed earlier-budget anchors | Remaining if anchors reused |
| --- | ---: | ---: | ---: |
| Pretrend diagnostic | 4,000 | 1 | 3,999 |
| Pretrend impact | 4,000 | 0 | 4,000 |
| Simulation prior sensitivity | 1,200 | 2 | 1,198 |
| Application | 9 | 3 | 6 |
| Application prior sensitivity | 18 | 2 | 16 |
| Total | 9,227 | 8 | 9,219 |

The next notebook package should retain all original jobs and seeds under a new immutable repaired-source manifest, explicitly reuse compatible completed anchors, and leave the completed main package unchanged. Keep CPU/RAM limits, complete-fit resume, optional Drive checkpoints, one compact downloadable ZIP and no posterior arrays. Use measured phase-specific costs from these returns and the earlier pilot to size independent shards for approximately eight hours, with at most 48 concurrent notebooks. Pretrend diagnostic and effect-impact jobs are separate evidence and both are needed to retain the original paper scope.

## Manuscript work

Use the repaired main scalar and surface tables for the final empirical update. The main 95% ATT coverage ranges from 0.75 to 1.00 (median 0.915); serial-correlation N=800 coverage is 0.79, 0.75 and 0.76 for degrees 1, 2 and 3. Report these results directly. Similar RMSE does not establish calibrated uncertainty, and longer MCMC has not been shown to cure this undercoverage. A different inferential procedure would be a separately versioned method change requiring its own evaluation; it is not needed merely to describe the existing estimator accurately.

After non-main returns, regenerate the affected simulation and application tables, figures, captions and numerical statements using final metrics. Preserve competitor results and explain which computation/prior their rows use. Report sampling budgets, source pins, seed repetitions, coverage limitations and supporting diagnostics. The existing exact-posterior consistency theory does not imply finite-run MCMC accuracy or nominal raw-interval coverage; retain its stated assumptions and scope.

Compile and check the full manuscript and supplement after updating all empirical assets. Submission readiness also requires final references, internal consistency and manuscript review. This analysis launches no new experiments and does not overwrite historical manuscript results.

## Reproduce collection and analysis

From the parent campaign directory, using the project's virtual environment:

```sh
python analysis_returned_20261014/validate_returns.py
python analysis_returned_20261014/collect_returned.py Colab/didbcf_paper_*.zip --out Results/collected_20261014 --require-main
python analysis_returned_20261014/report_results.py
python analysis_returned_20261014/plot_results.py
```

The validation records archive hashes, controls and the complete assignment partition. The collector validates every compact fit, output checksum and chain provenance. The report checks historical scalar truths and surface sizes and independently recomputes every repaired ATT cell's bias, RMSE, error-based Monte Carlo standard error and coverage count. Original posterior arrays are unnecessary for these paper metrics.
