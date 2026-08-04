"""The pre-trend / conditional-parallel-trends diagnostic (Reviewer 3.2.3).

Why the main specification cannot test its own assumption
---------------------------------------------------------
DiD-BCF is fitted as ``y = mu(X, t) + tau(X) Z`` with ``Z = D_it``.  ``D_it`` is
zero on every pre-treatment row, so those rows carry **no likelihood
information about tau**: the treatment forest has no gradient with which to
split on time before adoption.  ``tau_hat`` evaluated at a pre-treatment row is
not an estimate of a pre-treatment effect, it is the post-treatment estimate
re-evaluated at a different time value, which is why subtracting it annihilates
the estimate rather than recentring it (``Results/identification_note.tex``,
Section on the pre-trend recentring).  No amount of post-processing of the
constrained fit recovers a placebo coefficient.

The diagnostic
--------------
Fit, **as a separate diagnostic model**, the *unconstrained* specification of
Equation (1) of the manuscript,

    y_it = mu(X_i, t) + tau(X_i, k_it) * W_i,     W_i = 1[G_i != inf],

in which ``W_i`` is on in *every* period, so ``tau(X, k)`` is estimable at
``k < 0``.  Writing the DGP as ``Y(0) = a_i + gamma_t + f(X) + s(X) t + eps``
and letting ``Da(X) = E[a_i | X, W=1] - E[a_i | X, W=0]``, the fitted surface at
a pre-treatment event time is

    tau(X, k) = Da(X) + [differential slope](X) * t,

so the **level** ``Da(X)`` (which conditional PT permits: equal trends, unequal
levels) is a nuisance and the object of interest is the *change* across
pre-periods.  The reported diagnostic is therefore

    Delta(k) = E_{i treated}[ tau(X_i, k) - tau(X_i, -1) ],      k < -1,

which is **exactly zero under conditional parallel trends** and equals
``differential_slope * (k + 1)`` under a violation.  Reported per draw, so it
carries a credible interval, and summarised by the implied differential
**pre-trend slope** (the least-squares slope of ``Delta(k)`` on ``k + 1``),
which is the single most powerful summary when the violation is a linear
differential trend and is directly comparable to the DGP's ``group_trend``.

Three specification choices matter and are not cosmetic
------------------------------------------------------
1. **The group indicators must leave mu's split set.**  If ``mu`` may split on
   ``eventually_treated``/``treatment_group`` it spans ``tau(X, k) W_i``
   exactly, and the diagnostic is unidentified for the same reason documented
   in ``identification_note.tex``.  ``mu`` here sees only ``(X, t)``.
2. **stochtree's internal propensity must be switched off.**  Its default
   ``propensity_covariate="prognostic"`` regresses ``Z`` on ``X`` and appends
   the fit to ``mu``'s split set, which for ``Z = W_i`` is a near-perfect proxy
   for the group and re-opens the channel closed in (1).
3. **Unit-level random intercepts are on by default.**  They absorb ``a_i`` and
   hence the nuisance level ``Da(X)``, leaving ``tau`` to explain only the
   *time-varying* part of the treated-control gap.  Being time-invariant they
   cannot represent a trend, so they cannot mask the violation being tested.

What it does *not* claim
------------------------
The diagnostic conditions on the observed covariates, so it tests **conditional**
parallel trends: it is silent, by construction, about violations that are
orthogonal to the pre-period (an anticipation-free level jump exactly at
adoption remains untestable, as in any event study).  Its power also depends on
overlap: where ``X`` predicts group membership strongly, ``mu(X, t)`` can itself
mimic a group-by-time term and absorb part of the violation.  Both are
quantified by the ``PT_*`` scenarios rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .did_bcf import DEFAULT_BCF_PARAMS, _summarise

__all__ = ["fit_pretrend", "pretrend_estimands", "twfe_pretrend",
           "true_pretrend", "PretrendFit", "PRETREND_COVARIATE_COLS"]

# mu sees the covariates and calendar time -- and, deliberately, no group.
PRETREND_PROGNOSTIC_COLS = ["X1", "X2", "X3", "X4", "X5", "time"]
# tau is indexed by the effect modifiers and by *event* time, so that it can
# vary before adoption.  This is the only place ``k_diag`` is used.
PRETREND_TREATMENT_COLS = ["X1", "X2", "k_diag"]
PRETREND_COVARIATE_COLS = ("X1", "X2", "X3", "X4", "X5")

# Subgroups reported for the *conditional* version of the diagnostic.
_SUBGROUPS = {
    "X1=0": lambda d: d["X1"] <= 0.5,
    "X1=1": lambda d: d["X1"] > 0.5,
    "X2=low": lambda d: d["X2"] <= d["X2"].quantile(1 / 3),
    "X2=high": lambda d: d["X2"] >= d["X2"].quantile(2 / 3),
}


@dataclass
class PretrendFit:
    """One unconstrained (diagnostic) fit."""
    df: pd.DataFrame
    tau_draws: np.ndarray            # (n_obs, S)
    row_of: dict                     # (unit_id, time) -> row index
    ref_k: int = -1
    bcf_params: dict = field(default_factory=dict)
    spec: str = "pretrend"

    @property
    def n_draws(self) -> int:
        return self.tau_draws.shape[1]


def _add_k_diag(df: pd.DataFrame) -> pd.DataFrame:
    """Event time as a finite design column.

    Never-treated rows have ``Z = W_i = 0`` and so contribute nothing to the
    treatment forest's likelihood; their ``k_diag`` value is arbitrary and is
    set to 0 purely so the design matrix is finite.
    """
    out = df.copy()
    k = out["event_time"].to_numpy(dtype=float)
    out["k_diag"] = np.where(np.isfinite(k), k, 0.0)
    return out


def fit_pretrend(df: pd.DataFrame, bcf_params: dict | None = None,
                 seed: int | None = None, rfx: str = "unit",
                 prognostic_cols: "list | None" = None,
                 treatment_cols: "list | None" = None,
                 outcome: str = "Y") -> PretrendFit:
    """Fit the unconstrained diagnostic model (see the module docstring).

    Parameters
    ----------
    df : panel from :mod:`did_bcf_revision.dgps` (or any frame with
        ``unit_id, time, event_time, eventually_treated`` plus the covariates
        and the outcome).
    rfx : ``"unit"`` (default), ``"none"``.  Unit-level random intercepts absorb
        the time-invariant level gap that conditional PT permits.
    prognostic_cols, treatment_cols : split sets of ``mu`` and ``tau``.  Default
        to the simulation suite's columns; the empirical application passes its
        own (``lpop``).  ``treatment_cols`` must contain ``"k_diag"`` -- without
        an event-time axis ``tau`` cannot vary before adoption and the
        diagnostic is empty.  Neither list may contain a group indicator: see
        point (1) of the module docstring.
    """
    from stochtree import BCFModel

    p = {**DEFAULT_BCF_PARAMS, **(bcf_params or {})}
    prognostic_cols = list(prognostic_cols or PRETREND_PROGNOSTIC_COLS)
    treatment_cols = list(treatment_cols or PRETREND_TREATMENT_COLS)
    if "k_diag" not in treatment_cols:
        raise ValueError("treatment_cols must include 'k_diag': the diagnostic "
                         "reads tau at pre-treatment event times.")
    df = _add_k_diag(df.sort_values(["unit_id", "time"]).reset_index(drop=True))

    design_cols = prognostic_cols + [c for c in treatment_cols
                                     if c not in prognostic_cols]
    X = df[design_cols].to_numpy(dtype=float)
    # Z is the EVER-treated indicator, on in every period -- this is what makes
    # tau estimable before adoption.
    Z = df["eventually_treated"].to_numpy(dtype=float)
    y = df[outcome].to_numpy(dtype=float)
    prog_idx = np.array([design_cols.index(c) for c in prognostic_cols])
    treat_idx = np.array([design_cols.index(c) for c in treatment_cols])

    general_params = {"keep_every": p["keep_every"], "num_chains": p["num_chains"],
                      "propensity_covariate": "none"}
    if seed is not None:
        general_params["random_seed"] = int(seed)

    kwargs: dict = {}
    if rfx == "unit":
        kwargs["rfx_group_ids_train"] = df["unit_id"].to_numpy(dtype=np.int32)
        kwargs["rfx_basis_train"] = np.ones((len(df), 1))

    model = BCFModel()
    model.sample(X_train=X, Z_train=Z, y_train=y,
                 num_gfr=p["num_gfr"], num_mcmc=p["num_mcmc"],
                 general_params=general_params,
                 prognostic_forest_params={"keep_vars": prog_idx},
                 treatment_effect_forest_params={"keep_vars": treat_idx},
                 **kwargs)

    tau = np.asarray(model.tau_hat_train, dtype=float)
    if tau.ndim == 1:
        tau = tau[:, None]
    row_of = {(int(u), int(t)): i
              for i, (u, t) in enumerate(zip(df["unit_id"], df["time"]))}
    return PretrendFit(df=df, tau_draws=tau, row_of=row_of, bcf_params=p)


# --------------------------------------------------------------------------- #
# Diagnostic estimands
# --------------------------------------------------------------------------- #
def _delta_draws(fit: PretrendFit, rows: np.ndarray, ref_rows: np.ndarray) -> np.ndarray:
    """``mean_i [ tau_s(i, k) - tau_s(i, ref) ]`` -> one value per draw."""
    return np.mean(fit.tau_draws[rows, :] - fit.tau_draws[ref_rows, :], axis=0)


def _pre_rows(fit: PretrendFit):
    """Treated pre-treatment rows and, for each, the same unit's reference row."""
    df = fit.df
    pre = df[(df["eventually_treated"] == 1) & (df["event_time"] < 0)]
    ref_rows, keep = [], []
    for idx, u, g in zip(pre.index, pre["unit_id"], pre["cohort"]):
        r = fit.row_of.get((int(u), int(g) + fit.ref_k))
        if r is not None:
            ref_rows.append(r)
            keep.append(idx)
    return pre.loc[keep], np.asarray(ref_rows, dtype=np.int64)


def quantile_subgroups(col: str, n: int = 3) -> dict:
    """``{label: predicate}`` splitting ``col`` at its outer ``1/n`` quantiles.

    Used by the empirical application, whose covariate is not ``X1``/``X2``.
    """
    return {
        f"{col}=low": lambda d, c=col, n=n: d[c] <= d[c].quantile(1 / n),
        f"{col}=high": lambda d, c=col, n=n: d[c] >= d[c].quantile(1 - 1 / n),
    }


def pretrend_estimands(fit: PretrendFit,
                       subgroups: "bool | dict" = True) -> pd.DataFrame:
    """Tidy posterior summaries of the pre-trend diagnostic.

    Emitted rows (``method='pretrend'``)

    ``estimand_type='PRE'``     one per pre-treatment event time ``k < ref``:
                                ``Delta(k)``, zero under conditional PT.
    ``estimand_type='PRE'``, ``estimand_id='slope'``
                                the differential pre-trend slope implied by
                                ``Delta(k)`` -- the headline scalar, directly
                                comparable to the DGP's ``group_trend``.
    ``estimand_type='PRE'``, ``estimand_id='any'`` / ``'any_bonf'``
                                the per-replication decision rules: smallest
                                per-``k`` posterior tail, raw and
                                Bonferroni-scaled by the number of pre-periods,
                                so ``metrics.compute_metrics`` reports the
                                any-``k`` detection rate directly.
    ``estimand_type='PRE_SUB'`` the same ``Delta(k)`` within covariate
                                subgroups -- the *conditional* version, which a
                                marginal event study cannot produce.
    """
    pre, ref_rows = _pre_rows(fit)
    if pre.empty:
        return pd.DataFrame()

    records = []
    ks = sorted(int(k) for k in pre["event_time"].unique() if int(k) != fit.ref_k)

    per_k = {}
    for k in ks:
        m = (pre["event_time"] == k).to_numpy()
        draws = _delta_draws(fit, pre.index.to_numpy()[m], ref_rows[m])
        per_k[k] = draws
        rec = {"estimand_type": "PRE", "estimand_id": f"k={k}",
               "g": np.nan, "t": np.nan, "k": k, "method": "pretrend"}
        rec.update(_summarise(draws))
        records.append(rec)

    if per_k:
        # Differential pre-trend slope: least squares of Delta(k) on (k - ref),
        # through the origin (Delta(ref) = 0 by construction), per draw.
        w = np.array([k - fit.ref_k for k in ks], dtype=float)
        stack = np.vstack([per_k[k] for k in ks])           # (n_k, S)
        slope = (w @ stack) / float(w @ w)
        rec = {"estimand_type": "PRE", "estimand_id": "slope",
               "g": np.nan, "t": np.nan, "k": np.nan, "method": "pretrend"}
        rec.update(_summarise(slope))
        records.append(rec)

        # Per-replication decision rules, expressed as a posterior tail so the
        # existing metrics layer turns them into detection rates.
        tails = [min(float(np.mean(d > 0)), float(np.mean(d < 0))) for d in stack]
        p_min = float(min(tails))
        for eid, p in (("any", p_min), ("any_bonf", min(1.0, len(ks) * p_min))):
            records.append({"estimand_type": "PRE", "estimand_id": eid,
                            "g": np.nan, "t": np.nan, "k": np.nan,
                            "method": "pretrend", "post_mean": np.nan,
                            "sd": np.nan, "q025": np.nan, "q05": np.nan,
                            "q95": np.nan, "q975": np.nan, "p_bayes": p})

    if subgroups is not False:
        groups = _SUBGROUPS if subgroups is True else subgroups
        for name, sel in groups.items():
            m_sub = sel(pre).to_numpy()
            for k in ks:
                m = m_sub & (pre["event_time"] == k).to_numpy()
                if m.sum() < 5:
                    continue
                draws = _delta_draws(fit, pre.index.to_numpy()[m], ref_rows[m])
                rec = {"estimand_type": "PRE_SUB", "estimand_id": f"{name}_k={k}",
                       "g": np.nan, "t": np.nan, "k": k, "method": "pretrend"}
                rec.update(_summarise(draws))
                records.append(rec)

    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Truth, and the standard-practice comparator
# --------------------------------------------------------------------------- #
def true_pretrend(df: pd.DataFrame, ref_k: int = -1) -> pd.DataFrame:
    """True values of the diagnostic's estimands, from the DGP's violation slope.

    ``pt_slope`` is the per-unit differential slope written by the generators
    (zero unless ``alpha_trend`` or ``group_trend`` is non-zero).  The
    treated-minus-control difference in its mean is the differential trend the
    diagnostic is meant to recover, so ``Delta(k) = diff * (k - ref)`` and the
    reported slope estimand has true value ``diff``.
    """
    if "pt_slope" not in df.columns:
        return pd.DataFrame()
    units = df.drop_duplicates("unit_id")
    ever = units["eventually_treated"] == 1
    if ever.all() or (~ever).all():
        return pd.DataFrame()
    diff = float(units.loc[ever, "pt_slope"].mean()
                 - units.loc[~ever, "pt_slope"].mean())

    pre = df[(df["eventually_treated"] == 1) & (df["event_time"] < 0)]
    ks = sorted(int(k) for k in pre["event_time"].unique() if int(k) != ref_k)
    rows = [{"estimand_type": "PRE", "estimand_id": f"k={k}", "true": diff * (k - ref_k)}
            for k in ks]
    rows.append({"estimand_type": "PRE", "estimand_id": "slope", "true": diff})
    # The decision-rule rows have no point estimate; their "truth" is that the
    # assumption holds (diff == 0), which is what makes reject05 a size when
    # diff == 0 and a detection rate otherwise.
    rows += [{"estimand_type": "PRE", "estimand_id": eid, "true": diff}
             for eid in ("any", "any_bonf")]
    for name in _SUBGROUPS:
        rows += [{"estimand_type": "PRE_SUB", "estimand_id": f"{name}_k={k}",
                  "true": np.nan} for k in ks]
    return pd.DataFrame.from_records(rows)


def twfe_pretrend(df: pd.DataFrame, ref_k: int = -1) -> pd.DataFrame:
    """The standard-practice comparator: TWFE event-study placebo coefficients.

    Same schema as :func:`pretrend_estimands` with ``method='twfe_es'``, so the
    two land in one table.  This is what applied work reports, and it does
    **not** condition on covariates -- which is precisely why it flags a
    violation in designs where parallel trends holds *conditionally* (see the
    ``PT_conditional`` scenario).
    """
    from scipy.stats import norm

    from .twfe import twfe_event_study_se

    es = twfe_event_study_se(df, ref=ref_k)
    pre = es[(es["k"] < 0) & (es["k"] != ref_k) & es["se"].notna()]
    if pre.empty:
        return pd.DataFrame()

    z95, z90 = 1.959964, 1.644854
    records, tails = [], []
    for k, coef, se in zip(pre["k"], pre["coef"], pre["se"]):
        tail = float(norm.cdf(-abs(coef / se))) if se > 0 else np.nan
        tails.append(tail)
        records.append({"estimand_type": "PRE", "estimand_id": f"k={int(k)}",
                        "g": np.nan, "t": np.nan, "k": int(k), "method": "twfe_es",
                        "post_mean": float(coef), "sd": float(se),
                        "q025": coef - z95 * se, "q05": coef - z90 * se,
                        "q95": coef + z90 * se, "q975": coef + z95 * se,
                        "p_bayes": tail})

    w = (pre["k"].to_numpy(dtype=float) - ref_k)
    coefs = pre["coef"].to_numpy(dtype=float)
    ses = pre["se"].to_numpy(dtype=float)
    slope = float(w @ coefs / (w @ w))
    # Conservative slope SE: treats the pre-period coefficients as independent,
    # which is why it is reported as a comparator and not as the test itself.
    slope_se = float(np.sqrt(np.sum((w * ses) ** 2)) / (w @ w))
    records.append({"estimand_type": "PRE", "estimand_id": "slope",
                    "g": np.nan, "t": np.nan, "k": np.nan, "method": "twfe_es",
                    "post_mean": slope, "sd": slope_se,
                    "q025": slope - z95 * slope_se, "q05": slope - z90 * slope_se,
                    "q95": slope + z90 * slope_se, "q975": slope + z95 * slope_se,
                    "p_bayes": float(norm.cdf(-abs(slope / slope_se)))
                               if slope_se > 0 else np.nan})
    p_min = float(np.nanmin(tails))
    for eid, p in (("any", p_min), ("any_bonf", min(1.0, len(tails) * p_min))):
        records.append({"estimand_type": "PRE", "estimand_id": eid,
                        "g": np.nan, "t": np.nan, "k": np.nan, "method": "twfe_es",
                        "post_mean": np.nan, "sd": np.nan, "q025": np.nan,
                        "q05": np.nan, "q95": np.nan, "q975": np.nan, "p_bayes": p})
    return pd.DataFrame.from_records(records)
