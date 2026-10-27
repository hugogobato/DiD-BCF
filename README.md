# DiD-BCF

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Replication code and results for **Forests for Differences: Robust Causal Inference Beyond Parametric DiD**, by Hugo Gobato Souto and Francisco Louzada Neto. [Paper/preprint](https://doi.org/10.48550/arXiv.2505.09706).

DiD-BCF estimates heterogeneous treatment effects in panel difference-in-differences designs using Bayesian forests. The final implementation separates the baseline into cohort/covariate levels and calendar-time/covariate trends, with a treatment-effect forest and optional unit random intercepts. The paper evaluates treated overall, cohort-time and conditional effects, pretrend diagnostics, prior sensitivity, and a minimum-wage application. Its application finds a negative overall employment effect; the primary population-group contrast intervals include zero.

## Find what you need

| Task | Start here |
|---|---|
| Fit the proposed model to a new panel | [Model setup and Python usage](MODEL_USAGE.md), [new-panel example](examples/fit_panel.py), [model library](didbcf_structured/) |
| Read or visualize the final results | [Result file dictionary and visualization handoff](RESULTS_VISUALIZATION_HANDOFF.md) |
| Main simulation replication | [Final main campaign](Simulation_Studies_Revision/DiD_BCF/Paper_Rerun_20261011/README.md) |
| Pretrend, sensitivity and application replication | [Final non-main campaign](Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014/README.md) |
| Regenerate tables and figures from saved results | [Results workflow](Simulation_Studies_Revision/Results/README.md) |
| Inspect development history and unsuccessful/intermediate experiments | [Archive guide](Old_Simulation_Studies/README.md) |

All **14,327 planned final jobs** are complete: 5,100 main fits and 9,227 diagnostic, pretrend-impact, sensitivity and application fits. The saved metrics can be analyzed without rerunning experiments or exporting posterior effect arrays.

## Current paper results

The current combined simulation results are in [`Simulation_Studies_Revision/Results/aggregated_repaired_20261026/`](Simulation_Studies_Revision/Results/aggregated_repaired_20261026/). Current application results are in [`Simulation_Studies_Revision/Results/empirical_repaired_20261014/`](Simulation_Studies_Revision/Results/empirical_repaired_20261014/). Ready-made vector figures and table fragments are in [`Results/figures/`](Simulation_Studies_Revision/Results/figures/) and [`Results/tables/`](Simulation_Studies_Revision/Results/tables/).

Storage labels preserve earlier interfaces: `plain` identifies the final main DiD-BCF in the current overlay; `plain_historical`, `pretrend_historical` and earlier posterior-correction results are explicitly separate. See the handoff before combining methods or target types. Original benchmark and reference-fold comparator inputs remain available because the final paper uses them.

## Code and replication layout

| Path | Role |
|---|---|
| `didbcf_structured/` | Model, design matrices, priors, posterior estimands and model tests. Use the verified repaired engine described in `MODEL_USAGE.md`. |
| `examples/`, `requirements-paper.txt` | New-panel example and the exact final campaign dependency pins. |
| `Simulation_Studies_Revision/DiD_BCF/` | Two final campaign packages: loose Colab notebooks, helpers, frozen model/native source, assignments, seeds, verified reuse, returned results and validation. |
| `Simulation_Studies_Revision/Results/` | Current result overlays, retained comparator inputs, publication code, tables and figures. |
| `Simulation_Studies_Revision/DGPs/`, `did_bcf_revision/`, `R_code/`, `TWFE/`, `DoubleML_Colab/`, `CFFE_Wang_Colab/`, `Benchmark_Results/` | Shared scenario and metric code, benchmark replication workflows and saved benchmark summaries. The frozen campaign copies define the exact final DiD-BCF runs. |
| `Simulation_Studies_Revision/Theory_Calibration/` | Source and saved inputs for the paper's explicitly labeled reference-fold/correction and information comparisons. These are separate estimators, not the final native-engine repair. |
| `Empirical_Study/` | Application dataset and links to the final application workflow. |
| `Old_Simulation_Studies/` | Visible historical studies, superseded notebooks, initial reruns, development notes and migration inventories. |

The dated final package names are immutable experiment identifiers. Their source snapshots retain original internal paths so that source checksums and the existing notebooks remain reproducible. Older development entry points are archived rather than mixed into the main workflow.

## Reproducibility

The experiment notebooks build a pinned, repaired native engine in isolation and verify its controls. Installing an arbitrary upstream `stochtree` wheel does not reproduce the paper. Follow [MODEL_USAGE.md](MODEL_USAGE.md) or the final campaign guides for sampling. For figures and tables, only Python analysis dependencies and the committed metrics are needed.

The current [Results workflow](Simulation_Studies_Revision/Results/README.md) gives the regeneration commands. In the author's workspace the manuscript directory `JBES_Submission/` is beside this repository; publication scripts write assets there and mirror them into `Results/`. The public result CSVs remain usable independently of the manuscript workspace.

The theoretical results concern the exact posterior under the manuscript's stated identification and approximation assumptions. The finite sampling budget, serial-correlation coverage limitations and diagnostic evidence remain reported separately; the repository does not certify convergence of every finite MCMC run.

## Citation

```bibtex
@misc{souto2025forests,
  title = {Forests for Differences: Robust Causal Inference Beyond Parametric DiD},
  author = {Souto, Hugo Gobato and Neto, Francisco Louzada},
  year = {2025},
  doi = {10.48550/arXiv.2505.09706},
  url = {https://arxiv.org/abs/2505.09706}
}
```
