# Application manuscript integration plan

`integrate_application.py` is a guarded downstream updater for the repaired
MPDTA compact collection. It reads only the collected CSV/JSON metrics. It
does not sample, fit, access posterior arrays, or change the historical
corrected-estimator outputs.

## Gate and reproducibility

The script defaults to dry run. It checks the frozen package and source hashes,
the campaign manifest hash, the expected phase counts (9 `application` and 18
`application_sensitivity` jobs), status completion counts, absence of missing
application IDs, and all 27 corresponding fit-record hashes in collection
provenance. It then calls the existing `analysis/application_report.py` into a
temporary directory to validate the compact output schemas, four group rows,
291 county-year CATT rows and 191 distinct counties per fit, three contrasts,
the 3 by 2 by 3 estimator grid, and nine application-pretrend jobs. It refuses
to integrate insufficient rows or a partial collection.

Run the default check with:

```sh
rtk python3 analysis/integrate_application.py
```

No manuscript or `Results` files are written in dry-run mode. After reviewing
the planned outputs and counts, `--apply` is the only mode that stages the
versioned results, snapshots the existing paper targets under
`JBES_Submission/audit/application_update_20261014/before`, and installs the
updated manuscript tables and figure.

```sh
rtk python3 analysis/integrate_application.py --apply
```

The default input is `Results/application_collected`. A missing, stale,
incomplete, or provenance-inconsistent collection blocks both modes before
any manuscript or results write.

## Produced application evidence

The versioned directory `Results/empirical_repaired_20261014` retains the
primary rate-1 bundle-0 rows and compact gate, CATT, and contrast data for each
of bundles 0, 1, and 2. CSV numbers remain the exported strings; JSON sidecars
include validated job IDs, fit-record hashes, and source manifest identity.
Bundles 1 and 2 are labeled as same-panel computational repeats. Their
posterior draws, quantiles, and intervals are not pooled or averaged.

The supplementary prior-sensitivity table reports all 18 all-treated rows for
rates 0.1, 1, and 10, both specifications, and all three computational
bundles. The script computes the range of unit-random-effects point estimates
over all nine settings and checks every exported interval for a zero crossing.
It does not assert that all cells agree; the generated prose reports the
observed zero-crossing cells from the source.

The empirical pretrend table has nine rows, one per rate and computational
bundle. It carries the slope mean and central interval, the original
one-sided `p_bayes`, and the pipeline's raw-any and Bonferroni-any decision
probabilities. Decisions use the prespecified `p_bayes < 0.025` rule. The
script keeps these one-sided pretrend probabilities distinct from the
application ATT/GATE summaries.

The replacement vector figure preserves a two-panel layout. Its left panel
uses `scipy.stats.gaussian_kde` on the primary fit's exported `catt_pct`, which
is the transform of the posterior-mean log CATT, and labels it as
cross-sectional dispersion rather than posterior uncertainty. It plots the
291 treated county-year rows from 191 distinct counties, allowing a county to
recur across years. Its right panel uses the exact exported percentage means
and 95 percent bounds for the primary unit-random-effects fit and the
structured no-random-effects comparison. Display rounding is confined to the
figure and manuscript tables; the accompanying CSV/JSON files retain the
exported values.

## Manuscript edits staged by `--apply`

The application status statements in the abstract, methods, application
section, and conclusion are updated while simulation pretrend reruns remain
identified as pending. The primary application table becomes a paired
unit-random-effects/no-random-effects table using the rate-1 bundle-0 compact
exports. The prose replaces the old seed-agreement claim, no-unit interval
claim, raw group contrasts, empirical pretrend slope, and county-only CATT
description. The raw estimate repeated in the historical reduced-data
correction paragraph is refreshed from the new primary fit; all corrected
estimator numerical claims are preserved. The discussion explicitly retains
the corrected estimator as a historical comparison.

The supplement's blanket application-pending statement is updated, and it
receives separate application sensitivity and empirical pretrend tables under
`sim_tables/`. Its simulation pretrend tables and figure are not replaced by
the empirical diagnostic.

Every manuscript replacement requires exactly one current source anchor.
Before writing, all existing manuscript targets are copied byte-for-byte to
the dated audit snapshot directory. The paper figure and TeX inputs are also
written to the versioned results directory for reproducibility. The updater
does not touch theory source files or bibliography data.

