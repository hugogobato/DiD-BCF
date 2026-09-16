"""Structured DiD-BCF: Bayesian causal forests with an identified DiD baseline.

The published DiD-BCF specification gives the prognostic function an
unrestricted forest over ``(D_i, t, X)``.  Because the post-treatment indicator
is itself a function of those arguments, the treatment-effect term lies inside
the prognostic function's own domain and the two are not separately identified:
the likelihood is exactly flat along ``(mu + c tau D, (1-c) tau)``.  Only the
priors break the tie, and they break it more decisively the more data there
are, so the resulting bias grows with the sample size.

Conditional parallel trends is the restriction that closes the gap.  It is
equivalent to the baseline being additively separable in cohort and time,

    mu(g, t, x) = a(g, x) + b(t, x),

and this package fits that model directly: three forests (``a`` on cohort and
covariates, ``b`` on time and covariates, ``tau`` on the effect modifiers and
event time with basis ``D_it``) sampled jointly on ``stochtree``'s low-level
sampler API.

    from didbcf_structured import StructuredDiDBCF, structured_estimands

    model = StructuredDiDBCF(rfx="unit").sample(panel, seed=0)
    print(model.check_identification())
    estimands = structured_estimands(model)

See ``README.md`` for the model, the sampler and the assumptions it needs.
"""

from .design import (DEFAULT_COVARIATES, DEFAULT_EFFECT_MODIFIERS,
                     DesignMatrices, PanelColumns, build_design)
from .estimands import structured_estimands, summarise_draws
from .model import StructuredDiDBCF
from .priors import (EFFECT_PRIOR, GLOBAL_PRIOR, LEVEL_PRIOR, TREND_PRIOR,
                     ForestPrior, GlobalPrior)

__version__ = "0.2.0"

__all__ = [
    "StructuredDiDBCF",
    "structured_estimands", "summarise_draws",
    "build_design", "DesignMatrices", "PanelColumns",
    "DEFAULT_COVARIATES", "DEFAULT_EFFECT_MODIFIERS",
    "ForestPrior", "GlobalPrior",
    "LEVEL_PRIOR", "TREND_PRIOR", "EFFECT_PRIOR", "GLOBAL_PRIOR",
]
