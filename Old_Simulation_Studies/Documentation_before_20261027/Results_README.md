# Simulation results and paper assets

**Final update (2026-10-26):** all 14,327 planned jobs are complete. The current combined main/non-main simulation overlay is `aggregated_repaired_20261026/`. It adds all 4,000 diagnostic, 4,000 pretrend-impact and 1,200 simulation sensitivity fits while preserving the completed main values. All 27 application jobs remain in `empirical_repaired_20261014/`. No planned reruns remain.

After the main-overlay commands below, run:

```sh
python Results/analysis/update_repaired_nonmain.py
python Results/analysis/publish_nonmain_assets.py
```

These commands validate the complete non-main collection, retain each prior rate separately, and generate the final pretrend/sensitivity tables and figures. Historical benchmark inputs remain unchanged. The main simulation adapters preserve the new non-main and application table fragments. The dated descriptions below explain the preceding main update.

The current repaired main results are in `aggregated_repaired_20261014/`. They cover all 5,100 main fits across 48 cells, with 100 replications per cell except 200 in each of the three sharp-null cells. The main sampling budget is 50 GFR sweeps, no extra burn-in, 500 retained draws, thinning five and three chains, with the proper IG(2,1) variance prior and repaired native tree transition.

`aggregated/` remains the immutable historical result set. It is needed for the original-raw comparison and benchmark evidence. The versioned current overlay replaces only the repaired main DiD-BCF cells, preserves historical benchmark rows, and labels any historical non-main raw rows `plain_historical`. Historical corrected/reduced-data comparisons are separate from the repaired posterior. The repaired simulation pretrend and sensitivity campaigns are complete in the 2026-10-26 overlay; the returned eight non-main checks are supporting budget evidence.

`empirical_repaired_20261014/` contains the completed repaired-engine application inputs, all prior-rate and computational-bundle summaries, two supplementary table fragments, and the updated application figure. All 27 jobs are validated, comprising 22 new local fits and five verified reuses. `empirical/` remains the historical application input. The [current application report](../DiD_BCF/Nonmain_Rerun_20261014/analysis/application_report.md) and [non-main Colab guide](../DiD_BCF/Nonmain_Rerun_20261014/Colab/OPEN_IN_COLAB.md) document the completed local branch and completed three simulation waves.

The current loose notebooks, returned ZIPs, compact summaries and validation records live in [`../DiD_BCF/Paper_Rerun_20261011/`](../DiD_BCF/Paper_Rerun_20261011/README.md). No posterior arrays are needed to regenerate the reported paper metrics. Older experiment workflows are under `../../Old_Simulation_Studies/Revision_DiD_BCF/`; the old proper-prior paths have documented compatibility redirects.

From `Simulation_Studies_Revision/`, regenerate the repaired main overlay and publication assets with:

```sh
python Results/analysis/update_repaired_main.py
python Results/analysis/make_jbes_sim_tables.py
python Results/analysis/make_jbes_figures.py
```

The overlay builder validates the sealed campaign and its completed-fit status, copies the verified scalar/surface metrics, rebuilds sqrt(N) error summaries from all repaired replications, and records source/output checksums. Historical `aggregated/` is not overwritten. The table and figure adapters retain existing benchmark and reference-fold data, update the repaired main rows, and write the affected assets to `JBES_Submission/`. Pretrend tables and figures remain historical and are regenerated only with `--include-historical`.

`Results/figures/` contains updated main simulation figures mirrored into `JBES_Submission/sim_figures/`; the main small-panel figure is also mirrored into its existing image location. `JBES_Submission/sim_tables/` contains the updated numerical table fragments. The paired main comparison and all supporting diagnostics are in the campaign's `analysis_returned_20261014/` folder. The manuscript evidence and integration checks are recorded in `JBES_Submission/audit/results_update_20261014/` at the project root.

The legacy `aggregate_all.py` and `make_analysis.py` commands rebuild historical-generation reports. They are not the repaired main update commands above. The earlier Results README is preserved under `Old_Simulation_Studies/Revision_DiD_BCF/Results_README_before_repaired_20261014.md`.
