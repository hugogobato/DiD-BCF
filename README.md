# DiD-BCF

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This is the Github Repository for the paper "Forests for Differences: Robust Causal Inference Beyond Parametric DiD" (https://doi.org/10.48550/arXiv.2505.09706).

## Paper abstract

This paper introduces the Difference-in-Differences Bayesian Causal Forest (DiD-BCF), a novel non-parametric model addressing key challenges in DiD estimation, such as staggered adoption and heterogeneous treatment effects. DiD-BCF provides a unified framework for estimating Average (ATE), Group-Average (GATE), and Conditional Average Treatment Effects (CATE). A core innovation, its Parallel Trends Assumption (PTA)-based reparameterization, enhances estimation accuracy and stability in complex panel data settings. Extensive simulations demonstrate DiD-BCF's superior performance over established benchmarks, particularly under non-linearity, selection biases, and effect heterogeneity. Applied to U.S. minimum wage policy, the model uncovers significant conditional treatment effect heterogeneity related to county population, insights obscured by traditional methods. DiD-BCF offers a robust and versatile tool for more nuanced causal inference in modern DiD applications.

## Completed paper campaign (2026-10-26)

All 14,327 planned jobs are complete: 5,100 main fits and 9,227 non-main fits. The current repaired campaign packages are `Simulation_Studies_Revision/DiD_BCF/Paper_Rerun_20261011/` and `Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014/`. Current publication inputs are in `Simulation_Studies_Revision/Results/aggregated_repaired_20261026/`; completed empirical inputs are in `Results/empirical_repaired_20261014/`. See the [Results workflow](Simulation_Studies_Revision/Results/README.md). No original notebook assignment remains to rerun.

## Repository layout

The repository keeps the current repaired paper rerun beside the revision code and preserves older DiD-BCF experiments under `Old_Simulation_Studies/`. Historical experiment code and results remain ordinary publishable files. Only the separately documented `.local_experiment_archive/` is excluded from Git.

### `Simulation_Studies_Revision/`

Final simulation suite for the paper. Its own `README.md` documents the full workflow.

| path | contents |
|---|---|
| `DiD_BCF/Paper_Rerun_20261011/` | Current repaired paper campaign, including uploadable notebooks, sealed source snapshots, verified reuse, and collected results. See its README and Colab upload guide |
| `Pretrend/` | pre-trend diagnostic notebooks and per-replication summaries |
| `TWFE/` | TWFE/OLS benchmark notebooks |
| `R_code/` | Callaway-Sant'Anna (`did`), Gardner (`did2s`), DoubleML and synthetic DiD benchmark scripts with dataset-generation scaffolds; generated panels are local-only and gitignored |
| `DoubleML_Colab/`, `CFFE_Wang_Colab/` | Colab notebooks and result archives for the DoubleML and grf-DiD (Wang) benchmarks |
| `Benchmark_Results/` | decomposed benchmark metric CSVs |
| `DGPs/` | scenario data-generating processes |
| `did_bcf_revision/` | revision library (DGPs, runners, metrics, posterior correction, pre-trend diagnostics) |
| `scripts/` | dataset generation and aggregation runners |
| `Results/` | published result archive: `aggregated/` (tidy `all_summaries.csv.gz`, metrics, CATT surfaces, sqrt(N) stabilization), `tables/` and `figures/` (the LaTeX fragments and vector PDFs used by the paper), `empirical/` (mpdta empirical outputs), plus `summaries_twfe_*.csv`, `goodman_bacon_*.csv` and `twfe_event_study_*.csv` |
| `Theory_Calibration/` | theory-aligned correction package: `src/`, `configs/`, `scripts/`, `tests/`, `results/` and `reconciliation/`. Earlier correction and completion notebook workflows are preserved under `Old_Simulation_Studies/Revision_DiD_BCF/` |

### Other top-level folders

- `didbcf_structured/`: the structured DiD-BCF production package used by the experiments.
- `Empirical_Study/`: U.S. minimum wage (`mpdta`) empirical notebooks and figures.
- `Old_Simulation_Studies/`: publishable historical DiD-BCF experiments and outputs, including older simulation studies and archived revision correction workflows.

## Reproducing the paper results

The final metrics, returned ZIPs, source pins, and validation records are committed. Posterior effect arrays are not needed to regenerate the paper metrics. The immutable historical benchmark inputs remain in `Simulation_Studies_Revision/Results/aggregated/`; the current repaired overlays are separate.

From `Simulation_Studies_Revision/`, regenerate the simulation overlays and publication assets in this order:

```sh
python Results/analysis/update_repaired_main.py
python Results/analysis/make_jbes_sim_tables.py
python Results/analysis/make_jbes_figures.py
python Results/analysis/update_repaired_nonmain.py
python Results/analysis/publish_nonmain_assets.py
```

The publication scripts write paper assets when the `JBES_Submission/` sources are beside this repository. The non-main builder requires the complete collected results and validates phase counts, source hashes, targets and replication counts. Application data and seed-specific summaries are in `Results/empirical_repaired_20261014/`; the application reporting code is in the non-main campaign's `analysis/` directory. The old `aggregate_all.py` and `make_analysis.py` rebuild historical-generation results and are not the final overlay commands.

Earlier correction and completion workflows remain under `Old_Simulation_Studies/Revision_DiD_BCF/`. Historical reduced-data estimator results are distinct from the repaired posterior. The covariance, finite-chain and interval-coverage limitations remain documented in the manuscript and result reports.

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2505.09706,
  doi = {10.48550/ARXIV.2505.09706},
  url = {https://arxiv.org/abs/2505.09706},
  author = {Souto,  Hugo Gobato and Neto,  Francisco Louzada},
  keywords = {Methodology (stat.ME),  Machine Learning (cs.LG),  Machine Learning (stat.ML),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Forests for Differences: Robust Causal Inference Beyond Parametric DiD},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
