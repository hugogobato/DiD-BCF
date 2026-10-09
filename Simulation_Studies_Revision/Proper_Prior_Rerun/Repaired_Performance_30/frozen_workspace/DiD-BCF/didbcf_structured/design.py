"""Turn a long panel into the three design matrices the structured model needs.

The structured DiD-BCF model of ``DiD_BCF_Theory/DiD_BCF_theory.tex``,
equations (2.2)--(2.3), is

    Y_it = a(G_i, X_i) + b(t, X_i) + tau(X_i, k_it) * D_it + eps_it,

so the sampler needs three *separate* covariate blocks, one per forest:

``level``   the split set of ``a``: a cohort label and the covariates.  No time.
``trend``   the split set of ``b``: calendar time and the covariates.  No cohort.
``effect``  the split set of ``tau``: the effect modifiers and event time.

Keeping them in three matrices rather than one shared matrix with per-forest
``keep_vars`` is what makes the restriction structural: the ``a`` forest has no
time column to split on and the ``b`` forest has no cohort column, so no
combination of the two can produce ``1{G_i != inf} 1{t >= g}``.  This is the
whole content of the fix -- see :mod:`didbcf_structured.model`.

The cohort label is derived from the ``cohort`` column (first treated period,
``inf`` for never-treated) and recoded to ``0`` for never-treated and ``1..K``
for the adoption dates in increasing order, so the same code serves the
single-adoption and staggered designs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["PanelColumns", "DesignMatrices", "build_design",
           "DEFAULT_COVARIATES", "DEFAULT_EFFECT_MODIFIERS"]

DEFAULT_COVARIATES = ("X1", "X2", "X3", "X4", "X5")
DEFAULT_EFFECT_MODIFIERS = ("X1", "X2")

# Event time of a never-treated unit is undefined.  Those rows carry basis
# ``D_it = 0``, so the treatment forest's leaf sufficient statistics -- which
# are weighted by the basis -- receive nothing from them and the value is
# irrelevant to the likelihood.  It still has to be a number, and -1 puts those
# rows with the treated units' last pre-treatment period, which are equally
# inert.
NEVER_TREATED_EVENT_TIME = -1.0


@dataclass(frozen=True)
class PanelColumns:
    """Names of the structural columns of the long panel."""
    unit: str = "unit_id"
    time: str = "time"
    cohort: str = "cohort"          # first treated period; inf if never treated
    treated: str = "D"              # D_it, the post-treatment indicator
    outcome: str = "Y"
    event_time: str = "event_time"  # t - cohort; NaN if never treated


@dataclass
class DesignMatrices:
    """The three covariate blocks plus the treatment basis and the outcome."""
    level: np.ndarray               # (n, p_a)  cohort label + covariates
    trend: np.ndarray               # (n, p_b)  time + covariates
    effect: np.ndarray              # (n, p_t)  modifiers + event time
    Z: np.ndarray                   # (n,)      D_it, the treatment basis
    y: np.ndarray                   # (n,)
    level_cols: list = field(default_factory=list)
    trend_cols: list = field(default_factory=list)
    effect_cols: list = field(default_factory=list)
    cohort_code: np.ndarray = None  # (n,) 0 = never treated, 1..K = adoption order
    time_values: np.ndarray = None   # (n,) calendar time, as passed to `trend`
    cols: PanelColumns = field(default_factory=PanelColumns)

    @property
    def n(self) -> int:
        return self.level.shape[0]

    def trend_at(self, time_value) -> np.ndarray:
        """``trend`` with every row's time replaced by ``time_value``.

        Used for the gauge normalisation ``b(t_1, .) = 0`` and for the
        difference-in-differences contrast, both of which need ``b`` evaluated
        at a period other than the row's own.  ``time_value`` may be a scalar or
        a length-``n`` array (the contrast needs ``g - 1``, which varies by row).
        """
        out = self.trend.copy()
        out[:, self.trend_cols.index(self.cols.time)] = time_value
        return out

    def level_at(self, cohort_value) -> np.ndarray:
        """``level`` with every row's cohort label replaced by ``cohort_value``."""
        out = self.level.copy()
        out[:, 0] = cohort_value
        return out


def cohort_codes(cohort: np.ndarray) -> np.ndarray:
    """Recode adoption dates to ``0`` (never treated) and ``1..K`` in time order."""
    cohort = np.asarray(cohort, dtype=float)
    adoptions = np.unique(cohort[np.isfinite(cohort)])
    code = np.zeros(len(cohort), dtype=float)
    for j, g in enumerate(adoptions, start=1):
        code[cohort == g] = float(j)
    return code


def build_design(df: pd.DataFrame,
                 covariates=DEFAULT_COVARIATES,
                 effect_modifiers=DEFAULT_EFFECT_MODIFIERS,
                 cols: PanelColumns = PanelColumns(),
                 effect_by_cohort: bool = True) -> DesignMatrices:
    """Build the three design matrices from a long panel.

    Parameters
    ----------
    df : long panel, one row per unit-period, sorted or not.
    covariates : columns entering both prognostic blocks.
    effect_modifiers : columns the treatment-effect surface may split on.
    effect_by_cohort : whether the treatment forest may split on the cohort
        label.  Needed whenever effects differ across cohorts (the staggered
        designs); harmless when they do not.  It cannot reintroduce the
        identification failure because the treatment forest only ever acts on
        rows with ``D_it = 1``.
    """
    covariates = list(covariates)
    effect_modifiers = list(effect_modifiers)
    missing = [c for c in covariates + effect_modifiers if c not in df.columns]
    if missing:
        raise KeyError(f"Panel is missing covariate columns: {missing}")

    code = cohort_codes(df[cols.cohort].to_numpy())
    time = df[cols.time].to_numpy(dtype=float)
    if cols.event_time in df.columns:
        k = df[cols.event_time].to_numpy(dtype=float)
    else:
        k = time - df[cols.cohort].to_numpy(dtype=float)
    k = np.where(np.isfinite(k), k, NEVER_TREATED_EVENT_TIME)

    level_cols = ["cohort_code"] + covariates
    trend_cols = [cols.time] + covariates
    effect_cols = list(effect_modifiers) + [cols.event_time]
    if effect_by_cohort:
        effect_cols = effect_cols + ["cohort_code"]

    Xc = df[covariates].to_numpy(dtype=float)
    level = np.column_stack([code, Xc])
    trend = np.column_stack([time, Xc])
    effect_blocks = [df[effect_modifiers].to_numpy(dtype=float), k[:, None]]
    if effect_by_cohort:
        effect_blocks.append(code[:, None])
    effect = np.column_stack(effect_blocks)

    return DesignMatrices(
        level=np.ascontiguousarray(level, dtype=float),
        trend=np.ascontiguousarray(trend, dtype=float),
        effect=np.ascontiguousarray(effect, dtype=float),
        Z=np.array(df[cols.treated].to_numpy(dtype=float), dtype=float,
                   order="C", copy=True),
        y=df[cols.outcome].to_numpy(dtype=float),
        level_cols=level_cols, trend_cols=trend_cols, effect_cols=effect_cols,
        cohort_code=code, time_values=time, cols=cols,
    )
