"""DiD-BCF revision simulations: canonical-DiD DGPs (B1), decomposed metrics
(B2) and staggered / Goodman-Bacon analysis (D), with every metric reported for
**both** plain DiD-BCF and the proposed posterior correction (Algorithm 1 of the
theory note).

See ``README.md`` for the intended workflow (DiD-BCF fitting on Colab, metrics
and Goodman-Bacon on a parallelised local machine).
"""

from . import dgps, metrics, twfe, goodman_bacon, config, exports  # noqa: F401

# did_bcf, posterior_correction and the runners import stochtree / sklearn /
# joblib lazily, so they are safe to expose; importing them does not require
# stochtree until a fit actually runs.  The TWFE runner is pure numpy/pandas.
from . import did_bcf, posterior_correction, runner, twfe_runner  # noqa: F401

__all__ = [
    "dgps", "did_bcf", "posterior_correction", "runner", "twfe_runner",
    "metrics", "twfe", "goodman_bacon", "config", "exports",
]
