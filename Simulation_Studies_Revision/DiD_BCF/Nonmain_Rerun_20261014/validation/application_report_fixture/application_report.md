# Empirical application analysis

**Partial validation only:** 5/27 application and prior-sensitivity jobs are present. This report is not a complete application result and is not for manuscript inference.

The collection status is 8/9227 total non-main jobs. Application completeness is reported separately because this report uses only the 27 empirical application and prior-sensitivity jobs.

The application uses the 500-county MPDTA panel. The application estimand tables contain 291 treated county-year CATT rows and 191 distinct treated counties per estimation fit. The treated county-year group counts are 72 small-population, 119 medium-population, and 100 high-population rows.

## Prespecified estimates

The primary fit is `structured_rfx_unit` at prior rate 1, replication 0. `structured` at the same rate and replication is the no-random-effects comparison. Log-point and percentage means and interval bounds below are copied from the collector's exported fields. The accompanying CSV retains their original precision.

| Fit | Group | Treated county-years | ATT (log) | 95% interval (log) | ATT (%) | 95% interval (%) | p_bayes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Primary: unit random effects | All treated | 291 | -0.04850161581800385 | -0.076114682571630551 to -0.020529890581556139 | -4.724160131386304 | -7.3290077012061694 to -2.0320587150520875 | 0.0013333333333333333 |
| Primary: unit random effects | Small population | 72 | -0.045420812860075384 | -0.087980937171449405 to -0.0071063598427250161 | -4.4204554170670134 | -8.4221666117697218 to -0.7081169373783669 | 0.02 |
| Primary: unit random effects | Medium population | 119 | -0.059412118065369841 | -0.095133796315434671 to -0.025279245497454798 | -5.7538540299404213 | -9.0748728140570556 to -2.4962400850382149 | 0 |
| Primary: unit random effects | High population | 100 | -0.037736296273346806 | -0.074660598469196274 to 0.0057494208907476396 | -3.6832167448410043 | -7.1941582670985387 to 0.57659805319416613 | 0.070666666666666669 |
| Comparison: no random effects | All treated | 291 | -0.047928289391065679 | -0.08318442996221273 to -0.013935483819199951 | -4.6648853664656285 | -7.9818577455540662 to -1.3838834437406411 | 0.0093333333333333341 |
| Comparison: no random effects | Small population | 72 | -0.044233826354297512 | -0.094442706846901592 to 0.0054101368901735862 | -4.2968119206643127 | -9.0120136980991195 to 0.54247981085671082 | 0.074666666666666673 |
| Comparison: no random effects | Medium population | 119 | -0.060535774502181366 | -0.10194257089190975 to -0.019167890690046374 | -5.8537393553969572 | -9.6918586660221244 to -1.8985354809763488 | 0.0040000000000000001 |
| Comparison: no random effects | High population | 100 | -0.035585395495311083 | -0.081352689647843354 to 0.014746986219907271 | -3.4669882689714431 | -7.8131499374582303 to 1.4856259511004222 | 0.15066666666666667 |

## Prior-rate sensitivity and computational repeats

The table keeps the all-treated estimate for each rate, specification, and replication separate. Replications 1 and 2 reuse the same observed panel and are computational repeats, not independent samples. We do not pool posterior quantiles or average confidence or credible intervals across them.

| Prior rate | Specification | Replication | ATT (log) | 95% interval (log) | ATT (%) | 95% interval (%) | p_bayes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | structured_rfx_unit | 0 | -0.050269534801965658 | -0.077893825458243096 to -0.023570756143415975 | -4.8935970733819127 | -7.493736056856914 to -2.3295135645574394 | 0 |
| 1.0 | structured | 0 | -0.047928289391065679 | -0.08318442996221273 to -0.013935483819199951 | -4.6648853664656285 | -7.9818577455540662 to -1.3838834437406411 | 0.0093333333333333341 |
| 1.0 | structured_rfx_unit | 0 | -0.04850161581800385 | -0.076114682571630551 to -0.020529890581556139 | -4.724160131386304 | -7.3290077012061694 to -2.0320587150520875 | 0.0013333333333333333 |
| 10.0 | structured_rfx_unit | 0 | -0.04788850144899668 | -0.0890446505850261 to -0.008116098721976224 | -4.6566153675052 | -8.5195273903677986 to -0.808325211491423 | 0.022666666666666668 |

The seed-specific CSVs contain all three treated groups, all 291 county-year CATT rows, and all three group contrasts for each available fit. They preserve the exported numeric strings without recomputing intervals.

The frozen CATT export defines `catt_pct` as `100 * (exp(catt_log) - 1)`, where `catt_log` is the posterior mean log effect. Its distribution plot therefore shows transformed posterior means. The GATE/ATT percentage summaries instead transform the group-average effect draw by draw before taking the posterior mean. These are the original export conventions and are not interchangeable.

## Application pretrend decisions

The exported `application_pretrend` rows are retained in `application_pretrend_diagnostics.csv`, including event-time and subgroup summaries, 95% bounds, and the pipeline's `p_bayes` decision rows (`any`, `any_bonf`, `lpop_any`, and `lpop_any_bonf`). The table below displays those original decision values; no new decision rule is applied here.

| Prior rate | Replication | Estimand type | Decision row | p_bayes |
| --- | --- | --- | --- | --- |
| 1.0 | 0 | PRE | any | 0.045999999999999999 |
| 1.0 | 0 | PRE | any_bonf | 0.13800000000000001 |
| 1.0 | 0 | PRE_SUBC | lpop_any | 0.30733333333333335 |
| 1.0 | 0 | PRE_SUBC | lpop_any_bonf | 0.92200000000000004 |

## Reproducibility counts

| Phase | Expected jobs | Present jobs |
| --- | --- | --- |
| application | 9 | 3 |
| application_sensitivity | 18 | 2 |

Estimation fits present: 4 of 18. Application pretrend fits present: 1 of 9. Primary/comparison CATT rows: 582. Pretrend rows retained: 18; decision rows shown: 4.

## Files

`application_primary_estimates.csv` contains the prespecified primary and comparison group estimates. `application_gates_rep0.csv` through `application_gates_rep2.csv`, the corresponding `application_catt_rep*.csv` and `application_contrasts_rep*.csv` files retain seed-specific estimates. `application_pretrend_diagnostics.csv` contains the pretrend estimates and original decision rows.

## Figures

The vector PDF figures use exported posterior means and interval bounds. The CATT histograms describe 291 treated county-year estimates; counties may recur across years.

- [Primary and structured comparison](application_primary_intervals.pdf)
- [Prior-rate sensitivity by computational replication](application_prior_sensitivity.pdf)
- [County-year CATT percentage distribution](application_catt_distribution.pdf)
