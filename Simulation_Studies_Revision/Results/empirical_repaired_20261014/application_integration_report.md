# Repaired-engine MPDTA application integration

Complete compact application collection: 27/27 jobs, comprising 9 main-prior jobs and 18 prior-sensitivity jobs. The wider simulation campaign may remain incomplete.

The rate-1 computational bundle 0 unit-random-effects fit is the prespecified primary estimate. Bundle 0 without random effects is a comparison. Bundles 1 and 2 reuse the same observed panel and remain separate computational repeats; posterior intervals are not pooled.

The primary CATT distribution uses `catt_pct = 100 * (exp(catt_log) - 1)`, where `catt_log` is the posterior mean log effect. It shows cross-sectional dispersion, not posterior uncertainty. The 291 treated county-year rows come from 191 distinct counties.

The pretrend table reports the exported one-sided posterior-tail probabilities separately from gate `p_bayes` summaries. The prespecified diagnostic cutoff is 0.025 and includes the Bonferroni-any decision.

The full-precision source summaries and validation provenance are in `application_integration_source.json`; bundle-level compact data are in the three `application_bundle_rep*.json` files and corresponding CSVs.
