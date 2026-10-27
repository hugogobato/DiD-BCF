# Using the paper's DiD-BCF implementation

The estimator is [`didbcf_structured.StructuredDiDBCF`](didbcf_structured/model.py), with an additive level/trend baseline, a treatment-effect forest, and an optional unit random intercept. The five top-level Python implementation modules match the model snapshot used by the final campaigns. Exact replication uses the source snapshots and manifests in [`Paper_Rerun_20261011/`](Simulation_Studies_Revision/DiD_BCF/Paper_Rerun_20261011/) and [`Nonmain_Rerun_20261014/`](Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014/).

The Python model orchestrates a pinned **repaired native stochtree engine**. A generic `pip install stochtree` or the old package's loose dependency range is insufficient to reproduce the paper. The final notebooks build isolated engine copies, verify source archives and patches, and check the actual loaded binary. They do not install a global stochtree package. The native patch, build manifest and control code are retained under each final campaign's `source_workspace/JBES_Submission/recovery_repaired_20261008/engine/`.

## Easiest reproduction

Use the committed loose Colab notebooks and their launch links: [main campaign](Simulation_Studies_Revision/DiD_BCF/Paper_Rerun_20261011/README.md) and [pretrend/sensitivity/application campaign](Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014/README.md). They include pinned dependencies, the native build, exact assignments, seeds, checkpointing and compact result export. All original assignments are already complete, so reading or plotting the paper results requires no rerun.

## Fit a new long panel locally

Use Python 3.12 through 3.14 on Linux, with a C++ build toolchain. Run the commands below from the repository root in a separate virtual environment. The requirements are copied from the final campaign's pinned dependency versions.

```sh
python -m pip install -r requirements-paper.txt
```

Prepare one isolated native build. The paths below are the frozen source layout, whose historical names are retained to preserve hashes. Building and running the tiny controls is separate from fitting the panel.

```sh
python Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014/helpers/runtime.py prepare \
  --assignment Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014/assignments/local_application.json \
  --work /tmp/didbcf-native-work \
  --cache /tmp/didbcf-native-cache \
  --out /tmp/didbcf-native-setup
```

The preparation command uses an existing sealed assignment to validate the environment; it does not run that assignment's fits. It writes `/tmp/didbcf-native-work/setup_complete.json`. An existing matching native build may be supplied with `--existing-build /absolute/path/to/build`.

Then use the [new-panel example](examples/fit_panel.py):

```sh
python examples/fit_panel.py \
  --panel /absolute/path/to/panel.csv \
  --setup /tmp/didbcf-native-work/setup_complete.json \
  --covariates X1 X2 X3 X4 X5 \
  --effect-modifiers X1 X2 \
  --rfx none --seed 0 \
  --out /absolute/path/to/didbcf-output
```

The example verifies the frozen package, dependency versions and native controls, activates the frozen model, fits one panel and writes `estimands.csv` plus provenance. It uses the paper's prior and production budget by default; command-line flags can change the sampling budget for a new analysis. Such changes do not reproduce an original paper fit. Exact replication also requires its DGP, data, seed, model specification and aggregation rules.

The input has one row per unit and time, with numeric columns `unit_id`, `time`, `cohort`, `D`, `Y` and the selected covariates. `cohort` is the first treated period and is `inf` for never-treated units. `D` is the unit-period treatment indicator. Optional `event_time` is otherwise computed from time minus cohort. Unit identifiers must be representable as int32 if unit random effects are requested. Covariates used for the paper's identification argument are time invariant; supplying a panel does not establish the causal assumptions.

The primary simulations use `rfx="none"`; the primary application uses `rfx="unit"` and `lpop` as both covariate and effect modifier. The primary standardized variance prior is IG(2,1). Default model capacity is 125 level trees, 125 trend trees, and 100 effect trees, with cohort-specific effects enabled. The default retained sample is 500 draws per chain across three chains, following 50 GFR sweeps, with thinning five and no extra burn-in. Chains have seeds `1000 * seed + chain_index`.

## Python interface after verified activation

The example is also a readable template for importing the frozen model. After activation:

```python
from didbcf_structured import StructuredDiDBCF, structured_estimands

model = StructuredDiDBCF(rfx="none").sample(
    panel,
    covariates=["X1", "X2", "X3", "X4", "X5"],
    effect_modifiers=["X1", "X2"],
    effect_by_cohort=True,
    num_gfr=50, num_burnin=0, num_mcmc=500,
    keep_every=5, num_chains=3, seed=0,
)
summary = structured_estimands(model)
```

`model.tau_draws` stores treatment-effect draws by panel row; `mu_draws` and `sigma2_draws` expose baseline and variance draws. `structured_estimands` computes treated overall ATT, cohort-time GATT and event-time ES summaries by averaging within each posterior draw. With a simulated `CATT` truth column it also evaluates the treated CATT surface. Those treated averages do not automatically become population ATE/GATE/CATE targets.

Finite simulation coverage and MCMC mixing are assessed separately from the paper's exact-posterior consistency results. The stored draws alone do not establish a finite-run convergence theorem. For plotting completed results, use the [visualization handoff](RESULTS_VISUALIZATION_HANDOFF.md).
