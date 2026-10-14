"""DiD-BCF: the structured, two-way-restricted estimator.

Thin adapter over the :mod:`didbcf_structured` package, which lives one level up
(next to ``stochtree-main``) because it is a general estimator rather than part
of this simulation study.  It returns the same
:class:`did_bcf_revision.did_bcf.FitResult` as the stochtree-based specs, so
:func:`did_bcf_revision.did_bcf.plain_estimands`,
:func:`did_bcf_revision.posterior_correction.corrected_estimands` and the whole
metrics layer work on a structured fit without modification.

The mapping from ``FitResult`` to the structured model is exact:

``mu_draws``
    ``a(g, X) + b(t, X)`` (plus the random intercept when one is used) -- the
    fitted control-arm surface.  The posterior correction consumes this only
    through the long difference ``mu(i, t) - mu(i, g-1)``, in which ``a`` and any
    time-invariant random effect cancel, leaving ``b(t, X_i) - b(g-1, X_i)``:
    exactly the model's estimate of the untreated outcome change, which is what
    the correction's nuisance ``m^s`` is defined to be.  Under the unrestricted
    specification that same difference also contains whatever share of the
    treatment effect leaked into ``mu``, which is why the correction was
    faithfully propagating the leak rather than causing it.

``tau_draws``
    ``tau_0 + tau(X, k)``, the treatment-effect surface.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from .did_bcf import DEFAULT_BCF_PARAMS, FitResult, _row_index_map

# The package sits at DiD-BCF/didbcf_structured; this file at
# DiD-BCF/Simulation_Studies_Revision/did_bcf_revision.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STRUCTURED_SPECS = {
    "structured": "none",
    "structured_rfx": "group",
    "structured_rfx_unit": "unit",
}

# The two the revision actually uses.  `structured` is what the whole simulation
# grid ran under; `structured_rfx_unit` adds unit-level random intercepts and is
# the specification for panel applications with unobserved unit heterogeneity.
# `structured_rfx` (group-level intercepts) is a middle case kept only because
# the routes table reports it; it is not a production choice.
PRODUCTION_SPECS = ("structured", "structured_rfx_unit")

__all__ = ["STRUCTURED_SPECS", "PRODUCTION_SPECS", "is_structured",
           "fit_structured"]


def is_structured(spec: str) -> bool:
    return spec in STRUCTURED_SPECS


def _import_package():
    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)
    from didbcf_structured import StructuredDiDBCF
    return StructuredDiDBCF


def fit_structured(df: pd.DataFrame, bcf_params: dict | None = None,
                   seed: int | None = None,
                   spec: str = "structured",
                   effect_by_cohort: bool = True) -> FitResult:
    """Fit the structured model and return it in the suite's ``FitResult`` form.

    ``bcf_params`` uses the same keys as the stochtree specs (``num_gfr``,
    ``num_mcmc``, ``keep_every``, ``num_chains``), so a route comparison holds
    the sampling budget fixed across all of them.
    """
    StructuredDiDBCF = _import_package()
    from didbcf_structured import GlobalPrior

    if spec not in STRUCTURED_SPECS:
        raise KeyError(f"Unknown structured spec {spec!r}. "
                       f"Available: {sorted(STRUCTURED_SPECS)}")
    p = {**DEFAULT_BCF_PARAMS, **(bcf_params or {})}
    prior = GlobalPrior(sigma2_shape=p.get("sigma2_shape", 2.0),
                        sigma2_rate=p.get("sigma2_rate", 1.0))
    p.update(sigma2_shape=prior.sigma2_shape, sigma2_rate=prior.sigma2_rate,
             method_version="0.2.0-proper-ig", effect_by_cohort=effect_by_cohort)
    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)

    model = StructuredDiDBCF(rfx=STRUCTURED_SPECS[spec], global_prior=prior).sample(
        df, num_gfr=p["num_gfr"], num_mcmc=p["num_mcmc"],
        num_burnin=p.get("num_burnin", 0), effect_by_cohort=effect_by_cohort,
        keep_every=p["keep_every"], num_chains=p["num_chains"], seed=seed)

    fit = FitResult(df=model.df,
                    mu_draws=np.asarray(model.mu_draws, dtype=float),
                    tau_draws=np.asarray(model.tau_draws, dtype=float),
                    row_of=_row_index_map(model.df),
                    design_cols=list(model.design.level_cols),
                    bcf_params=p, spec=spec)
    fit.structured_model = model          # keep for the identification check
    return fit
