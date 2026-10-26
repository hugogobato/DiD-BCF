# Returned main paper rerun, 2026-10-14

All 60 main archives and four budget-check archives are complete. Checksums, job and assignment identities, source pins, variance/forest priors, seeds, engine specification and native/eligible-feature controls pass. There are 4,836 returned new main fits plus 264 verified reused main fits: 5,100/5,100 main fits, with eight additional non-main checks. No failed or missing assigned fits remain. Every fit enters evaluation, without diagnostic or performance filtering.

The repaired earlier budget retains similar point-estimation accuracy to historical raw in this grid: matched ATT RMSE changes range from -4.32% to +4.88% (median +0.50%), and treated CATT-surface changes from -0.51% to +3.57% (median +1.50%). No main replacement fits are needed for completeness. Interval undercoverage remains material: ATT 95% inclusion is 75/100, 76/100 and 79/100 at N=800 in the serial sweep for degrees 2, 3 and 1. The manuscript must report this limitation.

The main grid has 48 cells: 100 replications per cell except the three null cells, which have 200 each. Historical raw has matching summaries for 4,800 datasets; the 300 extra null replications (indices 100–199) enter full repaired metrics but not paired comparisons. Scalar truths and treated surface row counts match on every paired target. Historical panel bytes are unavailable in the scalar summaries; matching summaries is not a byte-level historical panel verification.

## Matched changes from historical raw

| estimand_type | target_cells | min_rmse_change_percent | median_rmse_change_percent | max_rmse_change_percent |
| --- | --- | --- | --- | --- |
| ATT | 48 | -4.3176 | 0.5049 | 4.8763 |
| CATT | 48 | -0.5119 | 1.5048 | 3.5651 |
| ES | 192 | -5.3401 | -0.1218 | 5.8988 |
| GATT | 207 | -5.3401 | -0.1790 | 3.5077 |

Changes are computed within each target/cell on exactly matched datasets. Surface RMSE is the mean per-dataset treated CATT-surface RMSE. Paired dataset bootstrap intervals (5,000 resamples) for every ratio are in paired_metric_contrasts.csv. Historical raw differs in kernel, prior, computation and environment; these comparisons do not isolate the native repair or prove empirical equivalence.

## Full repaired ATT metrics

| setting | N | linearity_degree | n_reps | bias | mcse_bias | rmse | mae | cover95_count | cover95_n | cover95 | len95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1_baseline | 200 | 1 | 100 | 0.0036 | 0.0100 | 0.0992 | 0.0769 | 95.0000 | 100.0000 | 0.9500 | 0.4251 |
| B1_baseline | 200 | 2 | 100 | -0.0055 | 0.0116 | 0.1156 | 0.0899 | 92.0000 | 100.0000 | 0.9200 | 0.4383 |
| B1_baseline | 200 | 3 | 100 | -0.0102 | 0.0115 | 0.1151 | 0.0881 | 93.0000 | 100.0000 | 0.9300 | 0.4434 |
| B1_null | 200 | 1 | 200 | 0.0025 | 0.0073 | 0.1031 | 0.0809 | 192.0000 | 200.0000 | 0.9600 | 0.4225 |
| B1_null | 200 | 2 | 200 | 0.0011 | 0.0084 | 0.1185 | 0.0925 | 186.0000 | 200.0000 | 0.9300 | 0.4340 |
| B1_null | 200 | 3 | 200 | 0.0011 | 0.0084 | 0.1185 | 0.0925 | 186.0000 | 200.0000 | 0.9300 | 0.4340 |
| B1_selection_obs | 200 | 1 | 100 | 0.0531 | 0.0110 | 0.1220 | 0.1005 | 95.0000 | 100.0000 | 0.9500 | 0.4663 |
| B1_selection_obs | 200 | 2 | 100 | 0.0492 | 0.0139 | 0.1463 | 0.1210 | 90.0000 | 100.0000 | 0.9000 | 0.4794 |
| B1_selection_obs | 200 | 3 | 100 | 0.0389 | 0.0141 | 0.1454 | 0.1185 | 89.0000 | 100.0000 | 0.8900 | 0.4931 |
| B1_serial_corr | 200 | 1 | 100 | 0.0068 | 0.0129 | 0.1282 | 0.0995 | 85.0000 | 100.0000 | 0.8500 | 0.3631 |
| B1_serial_corr | 200 | 2 | 100 | 0.0019 | 0.0145 | 0.1444 | 0.1131 | 82.0000 | 100.0000 | 0.8200 | 0.3758 |
| B1_serial_corr | 200 | 3 | 100 | -0.0118 | 0.0141 | 0.1413 | 0.1094 | 82.0000 | 100.0000 | 0.8200 | 0.3800 |
| B1_strong_confounder | 200 | 1 | 100 | -0.0015 | 0.0099 | 0.0988 | 0.0800 | 98.0000 | 100.0000 | 0.9800 | 0.4263 |
| B1_strong_confounder | 200 | 2 | 100 | -0.0073 | 0.0107 | 0.1069 | 0.0846 | 95.0000 | 100.0000 | 0.9500 | 0.4379 |
| B1_strong_confounder | 200 | 3 | 100 | -0.0134 | 0.0108 | 0.1084 | 0.0842 | 94.0000 | 100.0000 | 0.9400 | 0.4418 |
| B2_sweep | 50 | 1 | 100 | -0.0071 | 0.0237 | 0.2360 | 0.1853 | 94.0000 | 100.0000 | 0.9400 | 0.9210 |
| B2_sweep | 50 | 2 | 100 | -0.0107 | 0.0266 | 0.2649 | 0.2018 | 91.0000 | 100.0000 | 0.9100 | 0.9406 |
| B2_sweep | 50 | 3 | 100 | -0.0174 | 0.0275 | 0.2746 | 0.2109 | 92.0000 | 100.0000 | 0.9200 | 0.9616 |
| B2_sweep | 100 | 1 | 100 | 0.0150 | 0.0155 | 0.1549 | 0.1209 | 94.0000 | 100.0000 | 0.9400 | 0.6183 |
| B2_sweep | 100 | 2 | 100 | 0.0121 | 0.0186 | 0.1855 | 0.1528 | 91.0000 | 100.0000 | 0.9100 | 0.6401 |
| B2_sweep | 100 | 3 | 100 | -0.0026 | 0.0193 | 0.1917 | 0.1550 | 89.0000 | 100.0000 | 0.8900 | 0.6535 |
| B2_sweep | 200 | 1 | 100 | 0.0036 | 0.0100 | 0.0992 | 0.0769 | 95.0000 | 100.0000 | 0.9500 | 0.4251 |
| B2_sweep | 200 | 2 | 100 | -0.0055 | 0.0116 | 0.1156 | 0.0899 | 92.0000 | 100.0000 | 0.9200 | 0.4383 |
| B2_sweep | 200 | 3 | 100 | -0.0102 | 0.0115 | 0.1151 | 0.0881 | 93.0000 | 100.0000 | 0.9300 | 0.4434 |
| B2_sweep | 400 | 1 | 100 | 0.0014 | 0.0062 | 0.0622 | 0.0505 | 100.0000 | 100.0000 | 1.0000 | 0.2958 |
| B2_sweep | 400 | 2 | 100 | 0.0095 | 0.0075 | 0.0755 | 0.0608 | 97.0000 | 100.0000 | 0.9700 | 0.3069 |
| B2_sweep | 400 | 3 | 100 | 0.0051 | 0.0077 | 0.0772 | 0.0627 | 96.0000 | 100.0000 | 0.9600 | 0.3088 |
| B2_sweep | 800 | 1 | 100 | -0.0015 | 0.0052 | 0.0513 | 0.0410 | 96.0000 | 100.0000 | 0.9600 | 0.2097 |
| B2_sweep | 800 | 2 | 100 | -0.0048 | 0.0058 | 0.0577 | 0.0462 | 95.0000 | 100.0000 | 0.9500 | 0.2184 |
| B2_sweep | 800 | 3 | 100 | -0.0063 | 0.0058 | 0.0578 | 0.0459 | 94.0000 | 100.0000 | 0.9400 | 0.2170 |
| B2_sweep_serial | 50 | 1 | 100 | -0.0398 | 0.0287 | 0.2884 | 0.2273 | 82.0000 | 100.0000 | 0.8200 | 0.7945 |
| B2_sweep_serial | 50 | 2 | 100 | -0.0213 | 0.0304 | 0.3028 | 0.2385 | 81.0000 | 100.0000 | 0.8100 | 0.8258 |
| B2_sweep_serial | 50 | 3 | 100 | -0.0347 | 0.0314 | 0.3140 | 0.2446 | 84.0000 | 100.0000 | 0.8400 | 0.8437 |
| B2_sweep_serial | 100 | 1 | 100 | 0.0073 | 0.0194 | 0.1930 | 0.1494 | 81.0000 | 100.0000 | 0.8100 | 0.5311 |
| B2_sweep_serial | 100 | 2 | 100 | -0.0047 | 0.0220 | 0.2186 | 0.1797 | 79.0000 | 100.0000 | 0.7900 | 0.5583 |
| B2_sweep_serial | 100 | 3 | 100 | -0.0213 | 0.0216 | 0.2157 | 0.1791 | 79.0000 | 100.0000 | 0.7900 | 0.5718 |
| B2_sweep_serial | 200 | 1 | 100 | 0.0068 | 0.0129 | 0.1282 | 0.0995 | 85.0000 | 100.0000 | 0.8500 | 0.3631 |
| B2_sweep_serial | 200 | 2 | 100 | 0.0019 | 0.0145 | 0.1444 | 0.1131 | 82.0000 | 100.0000 | 0.8200 | 0.3758 |
| B2_sweep_serial | 200 | 3 | 100 | -0.0118 | 0.0141 | 0.1413 | 0.1094 | 82.0000 | 100.0000 | 0.8200 | 0.3800 |
| B2_sweep_serial | 400 | 1 | 100 | 0.0058 | 0.0087 | 0.0869 | 0.0699 | 84.0000 | 100.0000 | 0.8400 | 0.2479 |
| B2_sweep_serial | 400 | 2 | 100 | 0.0140 | 0.0094 | 0.0946 | 0.0766 | 80.0000 | 100.0000 | 0.8000 | 0.2614 |
| B2_sweep_serial | 400 | 3 | 100 | 0.0095 | 0.0097 | 0.0967 | 0.0790 | 81.0000 | 100.0000 | 0.8100 | 0.2626 |
| B2_sweep_serial | 800 | 1 | 100 | -0.0025 | 0.0068 | 0.0675 | 0.0543 | 79.0000 | 100.0000 | 0.7900 | 0.1752 |
| B2_sweep_serial | 800 | 2 | 100 | -0.0082 | 0.0072 | 0.0726 | 0.0572 | 75.0000 | 100.0000 | 0.7500 | 0.1846 |
| B2_sweep_serial | 800 | 3 | 100 | -0.0095 | 0.0071 | 0.0709 | 0.0566 | 76.0000 | 100.0000 | 0.7600 | 0.1847 |
| D_staggered | 200 | 1 | 100 | -0.0326 | 0.0108 | 0.1124 | 0.0897 | 96.0000 | 100.0000 | 0.9600 | 0.4410 |
| D_staggered | 200 | 2 | 100 | -0.0404 | 0.0114 | 0.1199 | 0.0944 | 94.0000 | 100.0000 | 0.9400 | 0.4555 |
| D_staggered | 200 | 3 | 100 | -0.0752 | 0.0127 | 0.1467 | 0.1207 | 90.0000 | 100.0000 | 0.9000 | 0.4671 |

All GATT and event-study targets, size/power, 90% intervals, Wilson intervals and conditional-effect surface metrics are saved in the accompanying CSVs. Undefined null-truth MAPE is retained as NaN. These surfaces concern treated observations, not transported population CATE.

## Non-main earlier versus B3 checks

| phase_earlier | spec_earlier | sigma2_rate_earlier | post_mean_earlier | post_mean_B3 | difference_over_B3_sd | width95_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| application | structured | 1.0000 | -0.0479 | -0.0500 | 0.1313 | 1.1282 |
| application | structured_rfx_unit | 1.0000 | -0.0485 | -0.0488 | 0.0212 | 0.9589 |
| sensitivity | structured | 10.0000 | 3.9062 | 3.9238 | 0.0634 | 1.0350 |
| sensitivity | structured | 0.1000 | 3.9383 | 3.9582 | 0.0859 | 0.9591 |
| application_sensitivity | structured_rfx_unit | 10.0000 | -0.0479 | -0.0475 | 0.0183 | 0.9870 |
| application_sensitivity | structured_rfx_unit | 0.1000 | -0.0503 | -0.0504 | 0.0095 | 1.0228 |

The two pretrend anchors contain 48 reported targets/decisions; 0 change rejection at the stored one-tail p-value cutoff 0.025. These are single-dataset checks, not repeated-sample validation of power or coverage. Application truth is unavailable; no application error or coverage is invented. Detailed endpoints, pretrend decisions, county effects, group effects and contrasts are retained.

Across the six ATT anchors, the earlier/B3 posterior mean difference is at most 0.132 B3 posterior standard deviations. Earlier/B3 95% interval-width ratios range from 0.959 to 1.128. A subgroup pretrend interval has a larger width difference (ratio about 0.50), despite unchanged rejection; aggregate ATT stability does not imply every internal target is equally stable. These checks support recommending the earlier budget for the remaining performance-focused campaign, with all diagnostic flags retained. They do not establish a formal budget-equivalence or MCMC-convergence result.

## Supporting diagnostics and runtime

| phase | family | sequences | median_rhat | max_rhat | undefined_rhat | fraction_rhat_over_105 | median_bulk_ess |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main | ATT | 5100 | 1.0201 | 1.3597 | 0 | 0.1992 | 263.4514 |
| main | ES | 20400 | 1.0145 | 1.3103 | 0 | 0.1183 | 395.2100 |
| main | GATT | 21900 | 1.0147 | 1.3103 | 0 | 0.1166 | 384.2087 |
| main | sigma2 | 5100 | 1.0521 | 1.6266 | 0 | 0.5120 | 41.8962 |

The earlier budget remains GFR50, burn-in0, retained500, thin5, three chains. The old full-state convergence gate remains not passed. Performance comparisons and exact-posterior asymptotics do not certify finite-MCMC convergence or nominal interval coverage.

Main notebook elapsed hours: minimum 3.91, median 6.30, maximum 7.91. All 64 returned archives total 55.25 MiB. Outputs contain metrics/diagnostics/provenance and no posterior arrays. The collector reader fix preserves numeric-looking hexadecimal IDs as strings; the sealed helpers and original CSV/ZIP bytes are unchanged.

## Remaining paper work

The main rerun needs no replacement fits. The full original non-main universe still requires a separately frozen budget/allocation decision: 4,000 pretrend diagnostic jobs, 4,000 pretrend-impact jobs, 1,200 simulation sensitivity jobs, nine application jobs and 18 application-sensitivity jobs. Reusing the eight compatible earlier-budget anchors would leave 9,219 jobs (3,999 diagnostic, 4,000 pretrend-impact, 1,198 simulation sensitivity, six application and 16 application-sensitivity). No full non-main campaign or new fits were launched by this analysis. Final manuscript empirical tables/figures/text and the submission checks remain pending. See NEXT_STEPS.md and the standalone main_comparison.pdf figure.
