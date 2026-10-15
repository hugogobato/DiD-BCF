# Empirical application analysis

**Application results complete:** all 27/27 application and prior-sensitivity jobs are present. This completeness statement covers the application jobs only.

The collection status is 30/9227 total non-main jobs. Application completeness is reported separately because this report uses only the 27 empirical application and prior-sensitivity jobs.

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
| 0.1 | structured | 0 | -0.04786675453651467 | -0.083122292749025897 to -0.012536687042794386 | -4.6587117991871709 | -7.9761398169855173 to -1.2458430150546373 | 0.012 |
| 0.1 | structured | 1 | -0.048526487058070507 | -0.079910748726124706 to -0.014257893075087617 | -4.7234653477053037 | -7.6801260626729313 to -1.415673067588219 | 0.0066666666666666671 |
| 0.1 | structured | 2 | -0.046471400989389337 | -0.078647582940204047 to -0.01354685952012089 | -4.5276771858508207 | -7.5634370733009266 to -1.3455513766621463 | 0.0040000000000000001 |
| 0.1 | structured_rfx_unit | 0 | -0.050269534801965658 | -0.077893825458243096 to -0.023570756143415975 | -4.8935970733819127 | -7.493736056856914 to -2.3295135645574394 | 0 |
| 0.1 | structured_rfx_unit | 1 | -0.049568244382967377 | -0.07700920756594469 to -0.022897095934768141 | -4.8267431975677795 | -7.4118671546141162 to -2.2636946769929889 | 0 |
| 0.1 | structured_rfx_unit | 2 | -0.04977196881328512 | -0.076908683140163622 to -0.021314696740502428 | -4.8465522316441207 | -7.4025593179010691 to -2.1089143964010182 | 0 |
| 1.0 | structured | 0 | -0.047928289391065679 | -0.08318442996221273 to -0.013935483819199951 | -4.6648853664656285 | -7.9818577455540662 to -1.3838834437406411 | 0.0093333333333333341 |
| 1.0 | structured | 1 | -0.049002005762593996 | -0.086763713777067028 to -0.010273973094762117 | -4.7651073213140336 | -8.3106280606674243 to -1.0221376113953684 | 0.013333333333333334 |
| 1.0 | structured | 2 | -0.047724545920532456 | -0.086820035972310944 to -0.010303997958453895 | -4.6432923235904298 | -8.3157920619502672 to -1.025109363608645 | 0.012 |
| 1.0 | structured_rfx_unit | 0 | -0.04850161581800385 | -0.076114682571630551 to -0.020529890581556139 | -4.724160131386304 | -7.3290077012061694 to -2.0320587150520875 | 0.0013333333333333333 |
| 1.0 | structured_rfx_unit | 1 | -0.049165944216387925 | -0.078424326749546691 to -0.020927630819032865 | -4.7869633281894579 | -7.5427977345401072 to -2.0710167591794226 | 0.0026666666666666666 |
| 1.0 | structured_rfx_unit | 2 | -0.048837012302789848 | -0.077725932429974581 to -0.02015314202829693 | -4.7562913402944034 | -7.4782035962102533 to -1.9951424812750251 | 0 |
| 10.0 | structured | 0 | -0.046039058226651129 | -0.099560976238174337 to 0.012382639872954935 | -4.4629331412714919 | -9.4765249624146222 to 1.2459622177969365 | 0.10000000000000001 |
| 10.0 | structured | 1 | -0.047119277887794651 | -0.10146299422290646 to 0.0091154268116359766 | -4.5663950468470791 | -9.6485386016014747 to 0.915709883780913 | 0.090666666666666673 |
| 10.0 | structured | 2 | -0.04813962811748311 | -0.10117480933058264 to 0.0061322313315181577 | -4.6641653252433937 | -9.622496923198165 to 0.61510719540667225 | 0.084000000000000005 |
| 10.0 | structured_rfx_unit | 0 | -0.04788850144899668 | -0.0890446505850261 to -0.008116098721976224 | -4.6566153675052 | -8.5195273903677986 to -0.808325211491423 | 0.022666666666666668 |
| 10.0 | structured_rfx_unit | 1 | -0.047877412994209181 | -0.086834990215063065 to -0.0063489551849750733 | -4.6545348025957498 | -8.317163119600723 to -0.63288431549736179 | 0.022666666666666668 |
| 10.0 | structured_rfx_unit | 2 | -0.047937644580091365 | -0.089796127342164953 to -0.0092205202146201024 | -4.6613839872757072 | -8.5882470154315822 to -0.91781415691786217 | 0.012 |

The seed-specific CSVs contain all three treated groups, all 291 county-year CATT rows, and all three group contrasts for each available fit. They preserve the exported numeric strings without recomputing intervals.

The frozen CATT export defines `catt_pct` as `100 * (exp(catt_log) - 1)`, where `catt_log` is the posterior mean log effect. Its distribution plot therefore shows transformed posterior means. The GATE/ATT percentage summaries instead transform the group-average effect draw by draw before taking the posterior mean. These are the original export conventions and are not interchangeable.

## Application pretrend decisions

The exported `application_pretrend` rows are retained in `application_pretrend_diagnostics.csv`, including event-time and subgroup summaries, 95% bounds, and the pipeline's `p_bayes` decision rows (`any`, `any_bonf`, `lpop_any`, and `lpop_any_bonf`). The table below displays those original decision values; no new decision rule is applied here.

| Prior rate | Replication | Estimand type | Decision row | p_bayes |
| --- | --- | --- | --- | --- |
| 0.1 | 0 | PRE | any | 0.021333333333333333 |
| 0.1 | 0 | PRE | any_bonf | 0.064000000000000001 |
| 0.1 | 0 | PRE_SUBC | lpop_any | 0.094 |
| 0.1 | 0 | PRE_SUBC | lpop_any_bonf | 0.28200000000000003 |
| 0.1 | 1 | PRE | any | 0.035999999999999997 |
| 0.1 | 1 | PRE | any_bonf | 0.10799999999999998 |
| 0.1 | 1 | PRE_SUBC | lpop_any | 0.14733333333333334 |
| 0.1 | 1 | PRE_SUBC | lpop_any_bonf | 0.44200000000000006 |
| 0.1 | 2 | PRE | any | 0.025999999999999999 |
| 0.1 | 2 | PRE | any_bonf | 0.078 |
| 0.1 | 2 | PRE_SUBC | lpop_any | 0.14000000000000001 |
| 0.1 | 2 | PRE_SUBC | lpop_any_bonf | 0.42000000000000004 |
| 1.0 | 0 | PRE | any | 0.045999999999999999 |
| 1.0 | 0 | PRE | any_bonf | 0.13800000000000001 |
| 1.0 | 0 | PRE_SUBC | lpop_any | 0.30733333333333335 |
| 1.0 | 0 | PRE_SUBC | lpop_any_bonf | 0.92200000000000004 |
| 1.0 | 1 | PRE | any | 0.048666666666666664 |
| 1.0 | 1 | PRE | any_bonf | 0.14599999999999999 |
| 1.0 | 1 | PRE_SUBC | lpop_any | 0.28133333333333332 |
| 1.0 | 1 | PRE_SUBC | lpop_any_bonf | 0.84399999999999997 |
| 1.0 | 2 | PRE | any | 0.032666666666666663 |
| 1.0 | 2 | PRE | any_bonf | 0.09799999999999999 |
| 1.0 | 2 | PRE_SUBC | lpop_any | 0.24533333333333332 |
| 1.0 | 2 | PRE_SUBC | lpop_any_bonf | 0.73599999999999999 |
| 10.0 | 0 | PRE | any | 0.092666666666666661 |
| 10.0 | 0 | PRE | any_bonf | 0.27799999999999997 |
| 10.0 | 0 | PRE_SUBC | lpop_any | 0.26933333333333331 |
| 10.0 | 0 | PRE_SUBC | lpop_any_bonf | 0.80799999999999994 |
| 10.0 | 1 | PRE | any | 0.089999999999999997 |
| 10.0 | 1 | PRE | any_bonf | 0.27000000000000002 |
| 10.0 | 1 | PRE_SUBC | lpop_any | 0.24533333333333332 |
| 10.0 | 1 | PRE_SUBC | lpop_any_bonf | 0.73599999999999999 |
| 10.0 | 2 | PRE | any | 0.099333333333333329 |
| 10.0 | 2 | PRE | any_bonf | 0.29799999999999999 |
| 10.0 | 2 | PRE_SUBC | lpop_any | 0.26600000000000001 |
| 10.0 | 2 | PRE_SUBC | lpop_any_bonf | 0.79800000000000004 |

## Reproducibility counts

| Phase | Expected jobs | Present jobs |
| --- | --- | --- |
| application | 9 | 9 |
| application_sensitivity | 18 | 18 |

Estimation fits present: 18 of 18. Application pretrend fits present: 9 of 9. Primary/comparison CATT rows: 582. Pretrend rows retained: 162; decision rows shown: 36.

## Files

`application_primary_estimates.csv` contains the prespecified primary and comparison group estimates. `application_gates_rep0.csv` through `application_gates_rep2.csv`, the corresponding `application_catt_rep*.csv` and `application_contrasts_rep*.csv` files retain seed-specific estimates. `application_pretrend_diagnostics.csv` contains the pretrend estimates and original decision rows.

## Figures

The vector PDF figures use exported posterior means and interval bounds. The CATT histograms describe 291 treated county-year estimates; counties may recur across years.

- [Primary and structured comparison](application_primary_intervals.pdf)
- [Prior-rate sensitivity by computational replication](application_prior_sensitivity.pdf)
- [County-year CATT percentage distribution](application_catt_distribution.pdf)
