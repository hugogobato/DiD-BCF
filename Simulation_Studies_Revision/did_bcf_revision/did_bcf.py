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

The "plain" estimand is the reparameterised DiD contrast used throughout the
paper: for a treated unit ``i`` of cohort ``g``,

    CATT_i(t) = tau_hat(i, t) - tau_hat(i, g-1)              (draw by draw)

i.e. the treatment-effect forest evaluated at calendar time ``t`` minus its
value at the last pre-treatment period ``g-1``.  Under conditional parallel
trends the subtracted term is ~0, so this both (a) implements the
reparameterisation's differencing and (b) leaves ``tau_hat(i, k)`` for ``k<0``
available as a pre-trends diagnostic.  Set ``pretrend_recenter=False`` to report
the raw forest instead.

``stochtree`` is imported lazily so this module can be imported on a machine
without it (e.g. to run the metrics layer); only :func:`fit_did_bcf` needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Fixed design-matrix column order shared with the posterior-correction module.
PROGNOSTIC_COLS = ["eventually_treated", "X1", "X2", "X3", "X4", "X5",
                   "time", "treatment_group"]
TREATMENT_COLS = ["X1", "X2", "time", "treatment_group"]  # effect modifiers + axes

DEFAULT_BCF_PARAMS = dict(
    num_gfr=50,
    num_mcmc=500,
    keep_every=5,
    num_chains=3,
)


@dataclass
class FitResult:
    """Everything downstream code needs from one DiD-BCF fit."""
    df: pd.DataFrame
    mu_draws: np.ndarray            # (n_obs, S) prognostic / control-arm draws
    tau_draws: np.ndarray           # (n_obs, S) treatment-effect draws
    row_of: dict                    # (unit_id, time) -> row index in df
    design_cols: list = field(default_factory=lambda: list(PROGNOSTIC_COLS))
    bcf_params: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return self.tau_draws.shape[1]


def _build_design(df: pd.DataFrame):
    """Return (X, Z, y, prognostic_keep_idx, treatment_keep_idx)."""
    X = df[PROGNOSTIC_COLS].to_numpy(dtype=float)
    Z = df["D"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    prog_idx = np.arange(len(PROGNOSTIC_COLS))
    treat_idx = np.array([PROGNOSTIC_COLS.index(c) for c in TREATMENT_COLS])
    return X, Z, y, prog_idx, treat_idx


def _row_index_map(df: pd.DataFrame) -> dict:
    return {(int(u), int(t)): i
            for i, (u, t) in enumerate(zip(df["unit_id"], df["time"]))}


def fit_did_bcf(df: pd.DataFrame, bcf_params: dict | None = None,
                seed: int | None = None) -> FitResult:
    """Fit DiD-BCF on a single replication's panel.

    Parameters
    ----------
    df : panel produced by :mod:`did_bcf_revision.dgps` (must contain the
        columns in :data:`PROGNOSTIC_COLS` plus ``D`` and ``Y``).
    bcf_params : overrides for :data:`DEFAULT_BCF_PARAMS`.
    seed : optional ``random_seed`` forwarded to ``general_params``.
    """
    from stochtree import BCFModel  # lazy: only needed when actually fitting

    p = {**DEFAULT_BCF_PARAMS, **(bcf_params or {})}
    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
    X, Z, y, prog_idx, treat_idx = _build_design(df)

    general_params = {"keep_every": p["keep_every"], "num_chains": p["num_chains"]}
    if seed is not None:
        general_params["random_seed"] = int(seed)

    model = BCFModel()
    model.sample(
        X_train=X, Z_train=Z, y_train=y,
        num_gfr=p["num_gfr"], num_mcmc=p["num_mcmc"],
        general_params=general_params,
        prognostic_forest_params={"keep_vars": prog_idx},
        treatment_effect_forest_params={"keep_vars": treat_idx},
    )

    mu_draws = np.asarray(model.mu_hat_train, dtype=float)
    tau_draws = np.asarray(model.tau_hat_train, dtype=float)
    # stochtree squeezes trailing singleton dims; enforce (n_obs, S).
    if mu_draws.ndim == 1:
        mu_draws = mu_draws[:, None]
    if tau_draws.ndim == 1:
        tau_draws = tau_draws[:, None]

    return FitResult(df=df, mu_draws=mu_draws, tau_draws=tau_draws,
                     row_of=_row_index_map(df), bcf_params=p)


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


def plain_estimands(fit: FitResult, pretrend_recenter: bool = True) -> pd.DataFrame:
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

    return pd.DataFrame.from_records(records)
