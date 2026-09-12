# DiD-BCF

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This is the Github Repository for the paper "Forests for Differences: Robust Causal Inference Beyond Parametric DiD" (https://doi.org/10.48550/arXiv.2505.09706).

## Paper abstract

This paper introduces the Difference-in-Differences Bayesian Causal Forest (DiD-BCF), a novel non-parametric model addressing key challenges in DiD estimation, such as staggered adoption and heterogeneous treatment effects. DiD-BCF provides a unified framework for estimating Average (ATE), Group-Average (GATE), and Conditional Average Treatment Effects (CATE). A core innovation, its Parallel Trends Assumption (PTA)-based reparameterization, enhances estimation accuracy and stability in complex panel data settings. Extensive simulations demonstrate DiD-BCF's superior performance over established benchmarks, particularly under non-linearity, selection biases, and effect heterogeneity. Applied to U.S. minimum wage policy, the model uncovers significant conditional treatment effect heterogeneity related to county population, insights obscured by traditional methods. DiD-BCF offers a robust and versatile tool for more nuanced causal inference in modern DiD applications.

## Repository layout

The tracked tree contains only the final paper experiments and results. Legacy and pre-revision artifacts are kept locally under `Old_Simulation_Studies/`, which is gitignored and not part of the published repository.

### `Simulation_Studies_Revision/`

Final simulation suite for the paper. Its own `README.md` documents the full workflow.

| path | contents |
|---|---|
| `DiD_BCF/` | DiD-BCF run notebooks: `DiD_BCF/Theory_Calibration/` (correction-audit, information-ablation and pilot notebooks) and `DiD_BCF/Correction_Completion/` (strong-confounder, selection, staggered, d=3, sample-size and pre-trend completion shards). The old-generation per-scenario notebooks and summaries moved to the local legacy tree |
| `Pretrend/` | pre-trend diagnostic notebooks and per-replication summaries |
| `TWFE/` | TWFE/OLS benchmark notebooks |
| `R_code/` | Callaway-Sant'Anna (`did`), Gardner (`did2s`), DoubleML and synthetic DiD benchmark scripts with dataset-generation scaffolds; generated panels are local-only and gitignored |
| `DoubleML_Colab/`, `CFFE_Wang_Colab/` | Colab notebooks and result archives for the DoubleML and grf-DiD (Wang) benchmarks |
| `Benchmark_Results/` | decomposed benchmark metric CSVs |
| `DGPs/` | scenario data-generating processes |
| `did_bcf_revision/` | revision library (DGPs, runners, metrics, posterior correction, pre-trend diagnostics) |
| `scripts/` | dataset generation and aggregation runners |
| `Results/` | published result archive: `aggregated/` (tidy `all_summaries.csv.gz`, metrics, CATT surfaces, sqrt(N) stabilization), `tables/` and `figures/` (the LaTeX fragments and vector PDFs used by the paper), `empirical/` (mpdta empirical outputs), plus `summaries_twfe_*.csv`, `goodman_bacon_*.csv` and `twfe_event_study_*.csv` |
| `Theory_Calibration/` | theory-aligned correction package: `src/`, `configs/`, `scripts/`, `tests/`, `results/` (shard archives plus `aggregated_correction/`, `aggregated_information/` and `aggregated_completion/`) and `reconciliation/` (reconciliation note plus LaTeX tables). Its uploadable notebooks live under `DiD_BCF/` |

### Other top-level folders

- `didbcf_structured/`: the structured DiD-BCF production package used by the experiments.
- `Empirical_Study/`: U.S. minimum wage (`mpdta`) empirical notebooks and figures.
- `Old_Simulation_Studies/`: local-only legacy artifacts (the original simulation studies and material moved out of the published tree); gitignored.

## Reproducing the paper results

All final outputs are committed, so the paper numbers can be inspected directly. To rebuild them from the raw per-replication summaries:

1. `python3 Simulation_Studies_Revision/Results/analysis/aggregate_all.py` collects every per-replication summary into `Simulation_Studies_Revision/Results/aggregated/`.
2. `python3 Simulation_Studies_Revision/Results/analysis/make_analysis.py` regenerates the LaTeX fragments in `Results/tables/` and the vector PDFs in `Results/figures/` from the aggregated files.
3. The correction experiments are aggregated and reconciled with the published tables via `python Simulation_Studies_Revision/Theory_Calibration/scripts/make_reconciliation_tables.py`; the resulting fragments live in `Theory_Calibration/reconciliation/tables/`. When the JBES submission sources are available next to this repository, `python Simulation_Studies_Revision/Theory_Calibration/scripts/verify_jbes_corrected_numbers.py` checks every corrected-estimator number against them.

The expensive BCF fits run in the Colab notebooks under `Simulation_Studies_Revision/DiD_BCF/Theory_Calibration/`, `Simulation_Studies_Revision/DiD_BCF/Correction_Completion/` and `Simulation_Studies_Revision/Pretrend/`. Benchmark summaries from `DoubleML_Colab/` and `CFFE_Wang_Colab/` are unpacked locally into `Simulation_Studies_Revision/Results/_staging/` before step 1; that staging directory is not tracked. R benchmark summaries are regenerated locally from `R_code/`. See `Simulation_Studies_Revision/README.md` and `Simulation_Studies_Revision/Theory_Calibration/README.md` for the full assignment and wall-time guidance, including how to run and aggregate the completion family.

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
