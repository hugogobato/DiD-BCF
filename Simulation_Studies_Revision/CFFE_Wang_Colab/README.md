# grf-DiD (Wang 2022) — Colab benchmark

A **CATT-capable** competitor for the revision (R1.5 / R3.1.1), run on the *same
seeded panels* as DiD-BCF and written to the DiD-BCF summary schema, so it is
scored by the engine's own `did_bcf_revision.metrics.compute_metrics` and
`surface_metrics` — side by side with DiD-BCF and the other benchmarks.

One self-contained notebook per scenario (mirrors `DoubleML_Colab/`). Upload a
single notebook to Colab and **Run all**: it installs R + `grf`, clones the
engine, regenerates the seeded panels, runs the method, and prints the decomposed
+ CATT-surface metric tables.

> **Folder/file names still read `CFFE_Wang_*` for continuity with the earlier
> plan; the notebooks now run grf-DiD only.** (Easy to rename on request.)

| Notebook | Scenario |
|---|---|
| `CFFE_Wang_B1_baseline.ipynb` … `_B1_strong_confounder.ipynb` | B1 canonical-DiD settings (5) |
| `CFFE_Wang_B2_sweep_N{200,400,800,1600}.ipynb` | sample-size sweep (4) |
| `CFFE_Wang_B2_sweep_serial_N{200,400,800,1600}.ipynb` | sweep with AR(1) errors (4) |
| `CFFE_Wang_D_contamination.ipynb`, `_D_staggered.ipynb` | staggered adoption (2) |

`wang_grf.R` is the canonical method script; each notebook also embeds it in a
`%%writefile` cell, so the notebooks run on Colab **without** anything being
committed first.

## The method

**grf-DiD (Wang 2022)** — `method="wang"`. First-difference the outcome against
the last clean pre-period, then a standard `grf::causal_forest` per (cohort *g*,
post period *t*) with never-treated controls. Per-obs `tau_hat(X)` with grf's
variance estimate → a genuine CATT surface + `average_treatment_effect`
GATT(g,t). Fast (seconds/rep) and dynamics-aware.

## Why CFFE is NOT used

R1.5 / R3.1.1 named *both* CFFE and a grf-based causal-forest DiD as acceptable
CATT-capable benchmarks. We satisfy the requirement with the **grf-DiD (Wang
2022)** alone, because **neither available CFFE implementation is usable**:

1. **The original CFFE R package does not build.** A fresh clone of
   `github.com/MACKattenberg/cffe` ships only `r-package/` (modified R + Rcpp
   **bindings**) and `simulation-dp/`. The modified grf C++ **core** the FE method
   needs — `fe_trainer`, the FE-aware `Data` (`set_individual_fe_index`, …), the
   FE splitting rule, all called from `bindings/CausalForestFEBindings.cpp` —
   lives in a `core/` directory that was **never committed**, and cannot be
   reconstructed from upstream grf-labs/grf (which has no `fe_trainer`).
   `grf/src/*` are broken symlinks committed as plain-text stubs (git mode
   `100644`), and the repo's own committed build `log` already fails with
   `Makevars:1: *** missing separator. Stop.` — the author's build never
   succeeded.

2. **The Python `causalfe` library is not a faithful re-implementation.** It
   captures the CFFE *idea* (node-level two-way FE residualisation + honest
   cluster-by-unit causal trees with τ-heterogeneity splits) but diverges from
   grf and from Kattenberg's `causal_fe_forest` on what matters: it **omits grf's
   local-centering / R-learner orthogonalisation** (so it carries a level bias),
   uses **exact-greedy splits** rather than grf's gradient-based splitting, and
   reports a **heuristic** variance (`B/(B-2)` inflation) instead of grf's
   infinitesimal jackknife. A single pooled forest also yields a time-invariant
   `tau(X)`. Reporting it as "CFFE" would not survive a referee.

Accordingly we run the grf-DiD only, and the response letter documents the
CFFE-repo build failure (it is the same "install issue" noted previously).

## Outputs

Per scenario × linearity degree, in the scenario data folder:

```
summaries_wang_<scenario>_lin_<d>.csv
```

in the exact schema of `DiD_BCF/summaries_<scenario>_lin_<d>.csv`. The offline
aggregator `benchmark_metrics.py` (KNOWN_METHODS includes `wang`) folds them into
`Benchmark_Results/` alongside every other method.
