# DiD-BCF model library

This is the Python model used by the final paper campaigns. The five implementation modules match their frozen campaign copies. For installation, repaired-engine activation and a runnable new-panel example, follow [MODEL_USAGE.md](../MODEL_USAGE.md).

The sampler fits

```text
Y_it = a(G_i, X_i) + b(t, X_i) + [tau_0 + tau(G_i, X_i, k_it)] D_it
       + optional unit random intercept + noise.
```

The level forest receives cohort and covariates, the trend forest receives calendar time and covariates, and the effect forest receives effect modifiers, event time and, by default, cohort. Splitting these design matrices enforces the additive baseline restriction. The causal interpretation still requires the paper's identification assumptions, including suitable time-invariant covariates and overlap.

| Module | Purpose |
|---|---|
| `model.py` | `StructuredDiDBCF`, Gibbs sampling, prediction draws and contrast helpers |
| `design.py` | `PanelColumns`, design matrices and cohort/event-time encoding |
| `priors.py` | Forest and global prior settings, including the proper noise prior |
| `estimands.py` | Draw-wise treated averages, posterior summaries and simulated surface metrics |
| `tests/` | Model/design and proper-prior checks |

The top-level defaults are the paper's 125 level trees, 125 trend trees, 100 effect trees and standardized IG(2,1) noise-variance prior. Simulations use no unit random intercept; the primary application includes one. The production budget is 50 GFR sweeps, no additional burn-in, 500 retained draws per chain, thinning five and three chains.

**Use the frozen repaired native engine.** The Python library calls stochtree's low-level sampler API, but the final computation includes a native source patch and binary controls. Installing an arbitrary upstream wheel, including one allowed by the older `pyproject.toml` range, does not reproduce the final paper. `MODEL_USAGE.md` describes the verified setup; the committed Colab notebooks perform it automatically.

After verified activation, the public interface is:

```python
from didbcf_structured import StructuredDiDBCF, structured_estimands

model = StructuredDiDBCF(rfx="none").sample(
    panel, covariates=["X1", "X2", "X3", "X4", "X5"],
    effect_modifiers=["X1", "X2"], effect_by_cohort=True,
    num_gfr=50, num_burnin=0, num_mcmc=500,
    keep_every=5, num_chains=3, seed=0,
)
summary = structured_estimands(model)
```

`tau_draws`, `mu_draws`, `sigma2_draws`, `did_contrast_draws()` and `check_identification()` expose draws and design checks. `structured_estimands` averages within posterior draws for ATT, cohort-time GATT and event-time ES. A simulated `CATT` truth column adds treated-surface evaluation. A design check does not establish parallel trends from data or certify finite-chain convergence.

Earlier model documentation is preserved in [Old_Simulation_Studies/Documentation_before_20261027/](../Old_Simulation_Studies/Documentation_before_20261027/). For exact experiment replication, use the final campaign source snapshots and manifests rather than changing the top-level library defaults.
