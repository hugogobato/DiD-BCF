"""Prior specification for the three forests of the structured model.

The calibration follows ``stochtree``'s ``BCFModel`` so that a structured fit is
comparable with the published DiD-BCF fit rather than differing from it in the
priors as well as in the restriction.  On the standardised outcome, where
``var(resid) = 1``, stochtree calibrates

    prognostic forest   sigma2_leaf = 2 / m,    IG(3, 1 / m)
    treatment forest    sigma2_leaf = 0.5 / m,  IG(3, 1 / (2m))

and gives the prognostic forest the permissive ``(alpha, beta) = (0.95, 2)``
tree prior against the treatment forest's aggressive ``(0.25, 3)`` -- the
asymmetric regularisation of Hahn et al. (2020).

The structured model splits the prognostic function into two blocks, so the
default tree counts split stochtree's prognostic budget in half (125 + 125
against its 250) and leave the treatment forest's 100 unchanged.  Total model
capacity is therefore the same as the published estimator's.

Note what the asymmetric prior is *for* in each model.  In the unrestricted
specification it is the only thing that decides how much of the effect lands in
``tau``, because the likelihood is flat in that direction; here the likelihood
identifies the split and the prior does the ordinary job of regularising a
nonparametric fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ForestPrior", "GlobalPrior", "LEVEL_PRIOR", "TREND_PRIOR",
           "EFFECT_PRIOR", "GLOBAL_PRIOR"]


@dataclass
class ForestPrior:
    """Tree, leaf and leaf-scale prior for one forest."""
    num_trees: int
    alpha: float = 0.95
    beta: float = 2.0
    min_samples_leaf: int = 5
    max_depth: int = 10
    cutpoint_grid_size: int = 100
    sample_leaf_scale: bool = True
    # Leaf-scale prior, expressed per unit of residual variance; `calibrate`
    # turns these into absolute numbers.  `None` means "use stochtree's rule".
    leaf_scale_init: float | None = None
    leaf_scale_shape: float = 3.0
    leaf_scale_rate: float | None = None
    # "prognostic" -> stochtree's mu rule; "effect" -> its tau rule.
    calibration: str = "prognostic"

    def calibrate(self, resid_var: float) -> tuple:
        """Return ``(leaf_scale_init, shape, rate)`` for this residual variance."""
        m = float(self.num_trees)
        if self.calibration == "effect":
            init = 0.5 * resid_var / m
            rate = resid_var / (2.0 * m)
        else:
            init = 2.0 * resid_var / m
            rate = resid_var / m
        if self.leaf_scale_init is not None:
            init = float(self.leaf_scale_init)
        if self.leaf_scale_rate is not None:
            rate = float(self.leaf_scale_rate)
        return float(init), float(self.leaf_scale_shape), float(rate)

    def leaf_scale_matrix(self, resid_var: float) -> np.ndarray:
        init, _, _ = self.calibrate(resid_var)
        return np.array([[init]], order="C")


@dataclass
class GlobalPrior:
    """Prior on the error variance and on the global treatment-effect intercept.

    ``shape = rate = 0`` is stochtree's default improper ``IG`` on ``sigma^2``.
    ``sample_intercept`` reproduces the ``tau_0`` term of ``BCFModel``: a global
    treatment-effect level with a normal prior, so the shrinkage prior on the
    treatment forest does not pull the *average* effect toward zero.  ``tau_0``
    multiplies ``D_it`` and so is a group-by-time term; it is identified in the
    structured model for exactly the reason ``tau`` is.
    """
    sigma2_shape: float = 0.0
    sigma2_rate: float = 0.0
    sample_intercept: bool = True
    intercept_prior_var: float | None = None    # defaults to var(resid)


LEVEL_PRIOR = ForestPrior(num_trees=125, alpha=0.95, beta=2.0, max_depth=10,
                          sample_leaf_scale=True, calibration="prognostic")
TREND_PRIOR = ForestPrior(num_trees=125, alpha=0.95, beta=2.0, max_depth=10,
                          sample_leaf_scale=True, calibration="prognostic")
EFFECT_PRIOR = ForestPrior(num_trees=100, alpha=0.25, beta=3.0, max_depth=5,
                           sample_leaf_scale=False, calibration="effect")
GLOBAL_PRIOR = GlobalPrior()
