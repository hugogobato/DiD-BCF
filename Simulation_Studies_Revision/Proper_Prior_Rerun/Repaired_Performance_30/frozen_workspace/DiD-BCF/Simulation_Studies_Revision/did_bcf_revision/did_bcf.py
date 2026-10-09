"""Fit DiD-BCF (stochtree BCF) and extract the *plain* posterior estimands.

This wraps ``stochtree.BCFModel`` exactly as in the original simulation
notebooks (same prognostic/treatment forest split, GFR warm-start -> MCMC) but
exposes a clean, reusable interface:

* :func:`fit_did_bcf`    -> a :class:`FitResult` holding the posterior draws of
  the prognostic surface (``mu_draws``) and the treatment-effect surface
  (``tau_draws``), each ``(n_obs, S)``, plus the bookkeeping the posterior
  correction needs.
* :func:`plain_estimands` -> tidy posterior summaries for the *uncorrected*
  DiD-BCF GATT(g,t), event-study ATT(k) and overall ATT.

The "plain" estimand is the treatment forest itself, evaluated on the treated
post-treatment rows:

    CATT_i(t) = tau_hat(i, t)                                (draw by draw)

An earlier version of this module defaulted to a "pre-trend recentring" that
reported ``tau_hat(i, t) - tau_hat(i, g-1)`` instead, on the reasoning that
under conditional parallel trends the subtracted term is ~0.  That reasoning
does not apply to this parameterisation.  The model is ``y = mu(X) +
tau(X) Z`` with ``Z = D_it``, so ``Z`` is zero on every pre-treatment row and
those rows carry **no likelihood information about tau at all**: the forest has
no gradient with which to split on time before adoption, so ``tau_hat(i, g-1)``
is not an estimate of a pre-treatment effect, it is the post-treatment estimate
re-evaluated at a different time value.  Subtracting it annihilates the
estimate (measured: 3.655 post against 3.669 at ``g-1``, true ATT 3.849).
``pretrend_recenter`` is retained for reproducing the earlier runs but defaults
to ``False`` and should stay there.

A pre-trend diagnostic requires the *unconstrained* specification, in which
``Z = 1[G_i != inf]`` so that ``tau(X, k)`` is estimable for ``k < 0``; it
cannot be recovered by subtraction inside the constrained fit.

``stochtree`` is imported lazily so this module can be imported on a machine
without it (e.g. to run the metrics layer); only :func:`fit_did_bcf` needs it.

--------------------------------------------------------------------------------
Identification specifications (``spec=``)
--------------------------------------------------------------------------------
``Results/identification_note.tex`` shows that with an unrestricted prognostic
forest over ``(D_i, t, X)`` the pair ``(mu + c tau D, (1-c) tau)`` has the same
likelihood for every ``c``, so ``mu`` and ``tau`` are not separately identified
and only the priors break the tie.  :data:`SPECS` collects the repairs; every
one of them is a different set of arguments to the *same* ``stochtree`` call, so
they can be compared on one line of the runner.

``published``
    The submitted specification: the ever-treated indicator and calendar time
    both inside ``mu``.  Kept as the reproduction target.

    Note what this actually does in ``stochtree``.  ``BCFModel.sample`` defaults
    to ``propensity_covariate="prognostic"``, and no ``pi_train`` is passed
    here, so stochtree fits an *internal* BART model of ``Z = D_it`` on ``X``
    -- which contains ``time`` and ``treatment_group``, of which ``D_it`` is a
    deterministic function -- and appends the fitted ``pi_hat`` to ``mu``'s
    split set with non-zero weight *regardless of* ``keep_vars``.  Measured on
    ``B1_baseline``: ``corr(pi_hat, D_it) = 0.97`` at ``N = 200`` and ``0.98``
    at ``N = 800``, with the two groups perfectly separated, so ``1{pi_hat >
    0.5}`` reproduces ``D_it`` exactly.  The flat direction of the note needs
    two splits (group, then time); this needs one, on a covariate purpose-built
    to be a perfect proxy for the treatment.  This channel is *not* in the
    original notebooks, which passed the constant ``pi_train = 0.5``.

``published_constant_pi``
    The original notebooks' call: identical to ``published`` but with the
    uninformative ``pi_train = 0.5``.  Isolates how much of the leak is the
    internal propensity and how much is the group-by-time flat direction.

``rfx`` (route 1 of the note, Section 3)
    Drop the group indicators from ``mu``'s split set and carry the static
    group difference as an additive group-level random intercept, giving
    ``mu(X, t) + a_G``: the forest sees time but not the group, the random
    effect sees the group but not time.  The propensity is switched off.

``rfx_unit``
    Same, but the random intercept is at the *unit* level.  Time-invariant, so
    it still cannot represent ``D_it``, and it additionally absorbs the
    unobserved ``alpha_i`` that the canonical DGP puts in the residual.  With
    ``N`` groups instead of two, its variance component is far better
    identified than the group-level version.

``propensity`` (route 2 of the note)
    The Hahn et al. (2020) prescription: drop the group indicators from ``mu``
    and pass ``pi_hat(X_i) = P(G_i != inf | X_i)`` in their place.  Estimated
    at the *unit* level from the covariates alone, so -- unlike stochtree's
    internal model -- it is a propensity for *ever* being treated and is
    constant within unit, which is what makes it a summary of selection rather
    than a copy of ``D_it``.  A mitigation, not a guarantee: a tree can still
    split on ``pi_hat > c`` and then on ``time``.

``rfx_propensity``
    Routes 1 and 2 together, which the note observes compose.

The corrected DiD-BCF (impose the two-way restriction structurally) changes the
*model*, not its arguments, and lives in :mod:`did_bcf_revision.structured`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .seeds import stochtree_seed

# Fixed design-matrix column order shared with the posterior-correction module.
PROGNOSTIC_COLS = ["eventually_treated", "X1", "X2", "X3", "X4", "X5",
                   "time", "treatment_group"]
TREATMENT_COLS = ["X1", "X2", "time", "treatment_group"]  # effect modifiers + axes

# The group indicators appear twice in the published design matrix; both have to
# go for a spec that removes the group from the prognostic forest.
GROUP_COLS = ("eventually_treated", "treatment_group")
COVARIATE_COLS = ("X1", "X2", "X3", "X4", "X5")

DEFAULT_BCF_PARAMS = dict(
    num_gfr=50,
    num_mcmc=500,
    keep_every=5,
    num_chains=3,
)


@dataclass(frozen=True)
class Spec:
    """One identification specification: a set of arguments to ``BCFModel``."""
    name: str
    prognostic_cols: tuple      # split set of the mu forest
    treatment_cols: tuple       # split set of the tau forest
    propensity: str             # "internal" | "constant" | "unit" | "none"
    rfx: str                    # "none" | "group" | "unit"
    note: str = ""


_ALL_PROG = tuple(PROGNOSTIC_COLS)
_NO_GROUP_PROG = tuple(c for c in PROGNOSTIC_COLS if c not in GROUP_COLS)
_TREAT = tuple(TREATMENT_COLS)

SPECS: dict[str, Spec] = {
    s.name: s for s in (
        Spec("published", _ALL_PROG, _TREAT, "internal", "none",
             "submitted specification; stochtree's internal propensity is a "
             "perfect proxy for D_it and enters mu"),
        Spec("published_constant_pi", _ALL_PROG, _TREAT, "constant", "none",
             "original notebooks: pi_train = 0.5, so the only leak channel is "
             "the group-by-time flat direction"),
        Spec("rfx", _NO_GROUP_PROG, _TREAT, "none", "group",
             "route 1: group out of mu, additive group random intercept"),
        Spec("rfx_unit", _NO_GROUP_PROG, _TREAT, "none", "unit",
             "route 1 with unit-level random intercepts, which also absorb "
             "the unobserved alpha_i"),
        Spec("propensity", _NO_GROUP_PROG, _TREAT, "unit", "none",
             "route 2: group out of mu, unit-level P(ever treated | X) in"),
        Spec("rfx_propensity", _NO_GROUP_PROG, _TREAT, "unit", "group",
             "routes 1 and 2 together"),
    )
}
# Every specification in SPECS above is a *stochtree* fit with an unrestricted
# prognostic forest, so every one of them carries the flat likelihood direction
# that `Results/identification_note.tex` documents.  None is a usable estimator.
# They are kept for one reason only: the note's central claim is that the
# published specification is inconsistent, and its evidence is the route
# comparison in `scripts/run_identification_routes.py`, which has to be able to
# *run* the broken specifications to show they are broken.  A reviewer who asks
# "show me" needs that to still work.
#
# Nothing in the production path may use them.  `runner.fit_any` refuses unless
# the caller passes `allow_legacy=True`, which only the routes script does.
LEGACY_SPECS = tuple(SPECS)

# The estimator.  `structured` is the two-way-restricted model of
# `didbcf_structured`; `structured_rfx_unit` adds unit-level random intercepts.
# The whole simulation grid is run under `structured`.
DEFAULT_SPEC = "structured"


def get_spec(spec: "str | Spec") -> Spec:
    if isinstance(spec, Spec):
        return spec
    try:
        return SPECS[spec]
    except KeyError:
        raise KeyError(f"Unknown spec {spec!r}. Available: {sorted(SPECS)}") from None


@dataclass
class FitResult:
    """Everything downstream code needs from one DiD-BCF fit."""
    df: pd.DataFrame
    mu_draws: np.ndarray            # (n_obs, S) prognostic / control-arm draws
    tau_draws: np.ndarray           # (n_obs, S) treatment-effect draws
    row_of: dict                    # (unit_id, time) -> row index in df
    design_cols: list = field(default_factory=lambda: list(PROGNOSTIC_COLS))
    bcf_params: dict = field(default_factory=dict)
    spec: str = DEFAULT_SPEC

    @property
    def n_draws(self) -> int:
        return self.tau_draws.shape[1]


def _build_design(df: pd.DataFrame, spec: Spec):
    """Return (X, Z, y, design_cols, prognostic_keep_idx, treatment_keep_idx).

    The design matrix is the ordered union of the two split sets, in
    :data:`PROGNOSTIC_COLS` order; ``keep_vars`` then zeroes the split weight of
    every column a given forest may not use.
    """
    design_cols = [c for c in PROGNOSTIC_COLS
                   if c in spec.prognostic_cols or c in spec.treatment_cols]
    X = df[design_cols].to_numpy(dtype=float)
    Z = df["D"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    prog_idx = np.array([design_cols.index(c) for c in spec.prognostic_cols])
    treat_idx = np.array([design_cols.index(c) for c in spec.treatment_cols])
    return X, Z, y, design_cols, prog_idx, treat_idx


def _row_index_map(df: pd.DataFrame) -> dict:
    return {(int(u), int(t)): i
            for i, (u, t) in enumerate(zip(df["unit_id"], df["time"]))}


def unit_propensity(df: pd.DataFrame, seed: int | None = None) -> np.ndarray:
    """``pi_hat(X_i) = P(G_i != inf | X_i)``, broadcast to every panel row.

    Fitted at the *unit* level (one row per unit) from the observed covariates
    only.  Neither ``time`` nor any group indicator is an input, so the result
    is a genuine summary of selection into ever-treated status and is constant
    within unit -- in contrast to stochtree's internal model, which regresses
    ``D_it`` on a design matrix that determines it.

    Uses BART so the estimate survives the Kang--Schafer covariate transforms at
    ``linearity_degree >= 2``; falls back to a logistic regression if
    ``stochtree`` is unavailable.
    """
    units = df.drop_duplicates("unit_id").sort_values("unit_id")
    Xu = units[list(COVARIATE_COLS)].to_numpy(dtype=float)
    du = units["eventually_treated"].to_numpy(dtype=float)

    if du.min() == du.max():                       # degenerate draw
        pi_unit = np.full(len(du), float(du.mean()))
    else:
        try:
            from stochtree import BARTModel
            m = BARTModel()
            m.sample(X_train=Xu, y_train=du, num_gfr=10, num_burnin=0,
                     num_mcmc=10,
                     general_params={"random_seed": stochtree_seed(seed)} if seed is not None else {})
            pi_unit = np.asarray(m.predict(X=Xu, terms="y_hat", type="mean"),
                                 dtype=float)
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            pi_unit = LogisticRegression(max_iter=1000).fit(Xu, du).predict_proba(Xu)[:, 1]
    pi_unit = np.clip(pi_unit, 1e-3, 1 - 1e-3)

    lookup = dict(zip(units["unit_id"].astype(int), pi_unit))
    return df["unit_id"].astype(int).map(lookup).to_numpy(dtype=float)


def fit_did_bcf(df: pd.DataFrame, bcf_params: dict | None = None,
                seed: int | None = None,
                spec: "str | Spec" = DEFAULT_SPEC) -> FitResult:
    """Fit DiD-BCF on a single replication's panel.

    Parameters
    ----------
    df : panel produced by :mod:`did_bcf_revision.dgps` (must contain the
        columns in :data:`PROGNOSTIC_COLS` plus ``D`` and ``Y``).
    bcf_params : overrides for :data:`DEFAULT_BCF_PARAMS`.
    seed : optional ``random_seed`` forwarded to ``general_params``.
    spec : identification specification, a key of :data:`SPECS` (see the module
        docstring).  Defaults to the published one.
    """
    from stochtree import BCFModel  # lazy: only needed when actually fitting

    spec = get_spec(spec)
    p = {**DEFAULT_BCF_PARAMS, **(bcf_params or {})}
    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
    X, Z, y, design_cols, prog_idx, treat_idx = _build_design(df, spec)

    general_params = {"keep_every": p["keep_every"], "num_chains": p["num_chains"]}
    if seed is not None:
        general_params["random_seed"] = stochtree_seed(seed)

    kwargs: dict = {}
    if spec.propensity == "constant":
        kwargs["propensity_train"] = np.full(len(df), 0.5)
    elif spec.propensity == "unit":
        kwargs["propensity_train"] = unit_propensity(df, seed=seed)
    elif spec.propensity == "none":
        general_params["propensity_covariate"] = "none"
    # "internal": leave stochtree's default in place (see the module docstring).

    rfx_ids = rfx_basis = None
    if spec.rfx != "none":
        key = "treatment_group" if spec.rfx == "group" else "unit_id"
        rfx_ids = df[key].to_numpy(dtype=np.int32)
        rfx_basis = np.ones((len(df), 1))
        kwargs["rfx_group_ids_train"] = rfx_ids
        kwargs["rfx_basis_train"] = rfx_basis

    model = BCFModel()
    model.sample(
        X_train=X, Z_train=Z, y_train=y,
        num_gfr=p["num_gfr"], num_mcmc=p["num_mcmc"],
        general_params=general_params,
        prognostic_forest_params={"keep_vars": prog_idx},
        treatment_effect_forest_params={"keep_vars": treat_idx},
        **kwargs,
    )

    mu_draws = np.asarray(model.mu_hat_train, dtype=float)
    tau_draws = np.asarray(model.tau_hat_train, dtype=float)
    # stochtree squeezes trailing singleton dims; enforce (n_obs, S).
    if mu_draws.ndim == 1:
        mu_draws = mu_draws[:, None]
    if tau_draws.ndim == 1:
        tau_draws = tau_draws[:, None]

    if spec.rfx != "none":
        # stochtree adds the random effect into y_hat but not into mu_hat.  The
        # correction's nuisance m^s is a *difference* over time, in which a
        # time-invariant intercept cancels, so this only affects readability of
        # mu_draws -- but the control-arm surface should be the whole of it.
        mu_draws = mu_draws + np.asarray(
            model.rfx_container.predict(rfx_ids, rfx_basis), dtype=float) * model.y_std

    return FitResult(df=df, mu_draws=mu_draws, tau_draws=tau_draws,
                     row_of=_row_index_map(df), design_cols=design_cols,
                     bcf_params=p, spec=spec.name)


# --------------------------------------------------------------------------- #
# Plain (uncorrected) estimands
# --------------------------------------------------------------------------- #
def _catt_draws(fit: FitResult, pretrend_recenter: bool) -> np.ndarray:
    """Per-observation CATT posterior draws (reparameterised DiD contrast).

    Returns an ``(n_obs, S)`` array; non-treated-post rows are left as NaN
    because they do not enter any reported estimand.
    """
    df = fit.df
    tau = fit.tau_draws
    out = np.full_like(tau, np.nan)
    treated_post = df.index[(df["D"] == 1)].to_numpy()

    if not pretrend_recenter:
        out[treated_post] = tau[treated_post]
        return out

    # Subtract each treated unit's last pre-treatment (g-1) tau draw.
    ref_idx = np.empty(len(df), dtype=np.int64)
    ref_idx.fill(-1)
    for u, g in df.loc[df["D"] == 1, ["unit_id", "cohort"]].drop_duplicates().itertuples(index=False):
        ref = fit.row_of.get((int(u), int(g) - 1))
        if ref is None:
            continue
        rows_u = df.index[(df["unit_id"] == u) & (df["D"] == 1)].to_numpy()
        ref_idx[rows_u] = ref
    valid = treated_post[ref_idx[treated_post] >= 0]
    out[valid] = tau[valid] - tau[ref_idx[valid]]
    return out


def _summarise(draws: np.ndarray) -> dict:
    """Posterior summary of a 1-D array of draws of an averaged estimand."""
    above = float(np.mean(draws > 0))
    below = float(np.mean(draws < 0))
    return {
        "post_mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "q025": float(np.quantile(draws, 0.025)),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
        "q975": float(np.quantile(draws, 0.975)),
        "p_bayes": min(above, below),
    }


def plain_estimands(fit: FitResult, pretrend_recenter: bool = False) -> pd.DataFrame:
    """Tidy posterior summaries of the uncorrected DiD-BCF estimands.

    One row per estimand with ``method='plain'`` and the columns produced by
    :func:`_summarise`.  Mirrors the estimand set of
    :func:`did_bcf_revision.dgps.true_estimands`.
    """
    df = fit.df
    catt = _catt_draws(fit, pretrend_recenter)          # (n_obs, S)
    treated_post = df[df["D"] == 1].copy()
    records = []

    def add(estimand_type, estimand_id, g, t, k, rows):
        rows = np.asarray(rows)
        rows = rows[~np.isnan(catt[rows, 0])]
        if rows.size == 0:
            return
        draws = np.nanmean(catt[rows, :], axis=0)        # average -> S draws
        rec = {"estimand_type": estimand_type, "estimand_id": estimand_id,
               "g": g, "t": t, "k": k, "method": "plain"}
        rec.update(_summarise(draws))
        records.append(rec)

    for (g, t), grp in treated_post.groupby(["cohort", "time"]):
        add("GATT", f"g={g:g}_t={int(t)}", float(g), int(t), int(t - g),
            grp.index.to_numpy())
    for k, grp in treated_post.groupby("event_time"):
        add("ES", f"k={int(k)}", np.nan, np.nan, int(k), grp.index.to_numpy())
    add("ATT", "ATT", np.nan, np.nan, np.nan, treated_post.index.to_numpy())

    # CATT surface: per-observation heterogeneous effect.  This is the paper's
    # headline RMSE/MAE/MAPE (over individual treated obs) plus a *pointwise*
    # CATT coverage -- the evidence that DiD-BCF recovers the heterogeneous
    # effect that GATT-only methods cannot.
    surf = _surface_record(df, catt, "plain")
    if surf is not None:
        records.append(surf)

    return pd.DataFrame.from_records(records)


def _surface_record(df: pd.DataFrame, catt: np.ndarray, method: str) -> dict | None:
    """One ``estimand_type='CATT'`` row: within-rep CATT-surface error metrics.

    ``catt`` is the ``(n_obs, S)`` per-observation posterior draw array (NaN off
    the treated-post rows).  Compares the per-observation posterior mean and
    pointwise credible bounds to the true individual ``CATT``.

    Returns ``None`` when the frame carries no true ``CATT`` column, which is
    the case for a real panel: the surface metrics are error measurements
    against a known truth and simply do not exist outside a simulation.  The
    averaged estimands are unaffected, so :func:`plain_estimands` runs on the
    empirical application as it does on the DGPs.
    """
    from .metrics import surface_summary

    if "CATT" not in df.columns:
        return None
    rows = df.index[(df["D"] == 1)].to_numpy()
    rows = rows[~np.isnan(catt[rows, 0])]
    if rows.size == 0:
        return None
    obs = catt[rows, :]                                   # (n_treated, S)
    est = np.nanmean(obs, axis=1)
    lo90 = np.nanquantile(obs, 0.05, axis=1); hi90 = np.nanquantile(obs, 0.95, axis=1)
    lo95 = np.nanquantile(obs, 0.025, axis=1); hi95 = np.nanquantile(obs, 0.975, axis=1)
    true = df.loc[rows, "CATT"].to_numpy(dtype=float)
    rec = {"estimand_type": "CATT", "estimand_id": "surface",
           "g": np.nan, "t": np.nan, "k": np.nan, "method": method}
    rec.update(surface_summary(true, est, lo90, hi90, lo95, hi95))
    return rec
