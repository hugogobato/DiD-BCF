# Final paper results and publication assets

The complete simulation overlay is [`aggregated_repaired_20261026/`](aggregated_repaired_20261026/). The complete application is [`empirical_repaired_20261014/`](empirical_repaired_20261014/). All 14,327 planned final jobs are complete. For exact file descriptions, method labels, filtering examples and uncertainty conventions, use the [visualization handoff](../../RESULTS_VISUALIZATION_HANDOFF.md).

| Directory | Role |
|---|---|
| `aggregated_repaired_20261026/` | Current combined main/non-main simulation metrics and retained benchmark comparisons. Start here for new analysis. |
| `empirical_repaired_20261014/` | Current application group, county-year, contrast, diagnostic and prior/bundle summaries. |
| `tables/`, `figures/` | Current publication LaTeX fragments and vector figures mirrored from the manuscript integration. |
| `analysis/` | Builders for current overlays, table/figure adapters and shared plotting helpers. |
| `aggregated_repaired_20261014/` | Validated main-only intermediate dependency of the final combined builder. Its main values are preserved in the current overlay. |
| `aggregated/`, `empirical/` | Historical-generation inputs, retained for benchmarks and explicit historical comparisons. These are not the current proposed-method results. |

The older main-only overlay remains because `update_repaired_nonmain.py` consumes it and verifies unchanged main results. It is a required build input, not a competing final result release. Similarly, historical benchmark and reference-fold inputs remain because the paper still compares those estimators.

## Read results without sampling

Load the current CSVs directly with pandas. `main_scalar_metrics.csv` and `main_surface_metrics.csv` isolate final main DiD-BCF. `metrics_all.csv` and `surface_all.csv` add benchmark/historical rows. `pretrend_summary.csv`, `pretrend_profiles.csv`, `prior_sensitivity.csv` and `prior_sensitivity_paired.csv` support the non-main figures. The visualization handoff explains the additional reference-fold selection used by existing manuscript adapters.

Main and non-main replication summaries are included. No posterior effect arrays, native compilation or ZIP extraction are needed for the reported paper metrics. The complete non-main scalar diagnostic table is committed losslessly as `DiD_BCF/Nonmain_Rerun_20261014/Results/collected_20261026/diagnostics.csv.gz`; pandas reads that gzip directly. Its storage record preserves the uncompressed checksum.

## Regenerate existing simulation tables and figures

From `Simulation_Studies_Revision/`, run in this order:

```sh
python Results/analysis/update_repaired_main.py
python Results/analysis/make_jbes_sim_tables.py
python Results/analysis/make_jbes_figures.py
python Results/analysis/update_repaired_nonmain.py
python Results/analysis/publish_nonmain_assets.py
```

The first builder validates the sealed 5,100-fit main collection and preserves benchmark inputs. The final combined builder validates all 9,227 non-main jobs, merges the repaired diagnostic and pretrend-impact results, retains each prior rate separately, and records hashes. The final asset publisher writes pretrend/sensitivity tables and figures. The main adapters preserve those non-main and application table fragments on subsequent runs. The historical `aggregate_all.py` rebuilds historical-generation inputs and is not part of this current workflow.

Publication scripts locate `JBES_Submission/` beside the Git checkout in the author's workspace and mirror generated assets into this Results directory. They may create that sibling asset directory in a standalone checkout. These commands do not run MCMC or regenerate application fits.

The [application reporting and integration scripts](../DiD_BCF/Nonmain_Rerun_20261014/analysis/) retain their original input validation and dry-run integration support. The current application outputs already contain all 27 validated jobs; primary bundle 0 and computational repetitions remain separate. To create new application visualizations, load the current application CSVs rather than reapplying old manuscript text replacements.

## Evidence and historical scope

The final campaign packages contain returned compact ZIPs, source pins, manifests, assignments, reuse records, collection status and independent numerical audit reports. [Main report](../DiD_BCF/Paper_Rerun_20261011/analysis_returned_20261014/) and [non-main report](../DiD_BCF/Nonmain_Rerun_20261014/analysis_returned_20261026/) document validation. The [application report](../DiD_BCF/Nonmain_Rerun_20261014/analysis/application_report.md) documents its separate local branch.

Development history is preserved visibly under [Old_Simulation_Studies](../../Old_Simulation_Studies/README.md). Historical method outputs are not relabeled as the repaired posterior. Finite-chain diagnostics and interval-coverage limitations remain part of the paper's evidence.
