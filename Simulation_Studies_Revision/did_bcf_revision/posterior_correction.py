"""The proposed *posterior correction* for DiD-BCF (Algorithm 1 of the theory).

This implements the "doubly robust post-processing of DiD-BCF output" described
in ``DiD_BCF_Theory/DiD_BCF_theory.tex`` (Section ``sec:alg``, Algorithm 1).  It
takes the *existing* DiD-BCF MCMC draws (no refit, no prior change) and, per
cohort-time cell ``(g, t)``, produces a corrected posterior for ``GATT(g, t)``.

--------------------------------------------------------------------------------
What the algorithm does (faithful to the .tex)
--------------------------------------------------------------------------------
Restrict to the cell sample ``S_g = {i : G_i in {g, infinity}}`` and define

    delta_i  = 1{G_i = g}                        (treated cohort vs never-treated)
    DeltaY_i = Y_{i,t} - Y_{i,g-1}               (long difference)
    m^s(X_i) = mu^s(X_i, t) - mu^s(X_i, g-1)     (control-arm change, draw s)

with pilot propensity ``pi_hat(x) ~ P(delta=1 | x)`` on S_g, ``barpi_hat =
mean(delta)`` and Riesz representer ``gamma_hat = (delta - pi_hat) /
((1 - pi_hat) * barpi_hat)``.  For each MCMC draw ``s`` and Bayesian-bootstrap
weights ``W^s_i`` (Exp(1), normalised over S_g):

    theta^s   = [ sum_i W^s_i (delta_i - (1-delta_i) pi_hat_i/(1-pi_hat_i))
                              (DeltaY_i - m^s(X_i)) ]
                / [ sum_i W^s_i delta_i ]                              (step b)

    b_hat^s   = (1/n_g) sum_i gamma_hat_i ( mbar(X_i) - m^s(X_i) )     (step c)
    check_theta^s = theta^s - b_hat^s

where ``mbar = mean_s m^s``.  The corrected credible interval is the empirical
quantile interval of ``{check_theta^s}``.

--------------------------------------------------------------------------------
A note on whether this is "correct" (the reason both versions are reported)
--------------------------------------------------------------------------------
Step (b) augments the plain plug-in with the efficient-influence-function term,
making the functional Neyman-orthogonal to the prognostic fit; step (c) removes
the empirical-process bias term that survives orthogonalisation outside Donsker
regimes.  Algebraically the ``m^s`` dependence in (b) and (c) nearly cancels, so
``check_theta^s`` is, to first order, a **Bayesian bootstrap of the doubly
robust DiD estimator** (Sant'Anna-Zhao style) with the BCF posterior mean as the
outcome-regression pilot -- a legitimate construction, but one whose posterior
*spread* is driven by the bootstrap over the influence function rather than by
the BCF posterior of ``tau``.  Whether that delivers better finite-sample
coverage than plain DiD-BCF is exactly what these simulations are built to
adjudicate, which is why every metric is reported for **both** methods.

For panel data with within-unit dependence the cell construction already uses
**one differenced observation per unit**, so the unit-level Bayesian bootstrap
here *is* the cluster bootstrap of Remark ``rem:bvm-cluster``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .did_bcf import FitResult

__all__ = ["bayesian_bootstrap_weights", "corrected_estimands"]

COVARIATE_COLS = ["X1", "X2", "X3", "X4", "X5"]
_PI_CLIP = 1e-3


# --------------------------------------------------------------------------- #
# Pilot propensity
# --------------------------------------------------------------------------- #
def _fit_propensity(X: np.ndarray, delta: np.ndarray, method: str,
                    n_splits: int, seed: int) -> np.ndarray:
    """Pilot estimate of P(delta=1 | X), optionally K-fold cross-fitted."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    def _new_model():
        if method == "rf":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                          random_state=seed, n_jobs=1)
        return LogisticRegression(max_iter=1000)

    # Degenerate label -> fall back to the marginal share.
    if delta.min() == delta.max():
        return np.full(len(delta), float(delta.mean()))

    if n_splits and n_splits > 1 and delta.sum() >= n_splits and (1 - delta).sum() >= n_splits:
        pi = np.empty(len(delta))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, delta):
            mdl = _new_model().fit(X[tr], delta[tr])
            pi[te] = mdl.predict_proba(X[te])[:, 1]
        return np.clip(pi, _PI_CLIP, 1 - _PI_CLIP)

    mdl = _new_model().fit(X, delta)
    return np.clip(mdl.predict_proba(X)[:, 1], _PI_CLIP, 1 - _PI_CLIP)


# --------------------------------------------------------------------------- #
# Bayesian bootstrap weights (common across cells, per Corollary on aggregation)
# --------------------------------------------------------------------------- #
def bayesian_bootstrap_weights(unit_ids: np.ndarray, n_draws: int,
                               seed: int = 0) -> dict:
    """One Exp(1) weight per *unit* per draw, shared across cells.

    Returns ``{unit_id: weight_vector_of_length_S}`` (un-normalised; each cell
    renormalises over its own ``S_g``).  Using a common draw across cells makes
    aggregated estimands (ATT, event-study) coherent (Corollary ``cor:agg``).
    """
    rng = np.random.default_rng(seed)
    units = np.unique(unit_ids)
    e = rng.exponential(1.0, size=(len(units), n_draws))
    return {int(u): e[j] for j, u in enumerate(units)}


# --------------------------------------------------------------------------- #
# Per-cell corrected draws
# --------------------------------------------------------------------------- #
def _cell_corrected_draws(fit: FitResult, g: float, t: int,
                          bb_weights: dict, propensity_method: str,
                          n_splits: int, seed: int) -> np.ndarray | None:
    """Return the ``S`` corrected draws ``check_theta^s`` for cell (g, t)."""
    df = fit.df
    S = fit.n_draws
    g_int = int(g)

    units = df.loc[(df["cohort"] == g) | (np.isinf(df["cohort"])),
                   "unit_id"].unique()
    rows_t, rows_ref, delta, dY, keep_units = [], [], [], [], []
    for u in units:
        r_t = fit.row_of.get((int(u), int(t)))
        r_ref = fit.row_of.get((int(u), g_int - 1))
        if r_t is None or r_ref is None:
            continue
        rows_t.append(r_t)
        rows_ref.append(r_ref)
        coh = df.at[r_t, "cohort"]
        delta.append(1.0 if coh == g else 0.0)
        dY.append(float(df.at[r_t, "Y"] - df.at[r_ref, "Y"]))
        keep_units.append(int(u))

    rows_t = np.asarray(rows_t)
    rows_ref = np.asarray(rows_ref)
    delta = np.asarray(delta)
    dY = np.asarray(dY)
    n_g = len(keep_units)
    if n_g < 5 or delta.sum() < 2 or (1 - delta).sum() < 2:
        return None

    # m^s(X_i) = mu^s(i, t) - mu^s(i, g-1)         -> (n_g, S)
    M = fit.mu_draws[rows_t, :] - fit.mu_draws[rows_ref, :]
    mbar = M.mean(axis=1)                                   # (n_g,)

    # Pilot propensity and Riesz representer.
    Xcov = df.loc[rows_t, COVARIATE_COLS].to_numpy(dtype=float)
    pi = _fit_propensity(Xcov, delta.astype(int), propensity_method,
                         n_splits, seed)
    barpi = float(delta.mean())
    gamma = (delta - pi) / ((1.0 - pi) * barpi)             # (n_g,)
    aug_w = delta - (1.0 - delta) * pi / (1.0 - pi)         # (n_g,)

    # Bayesian-bootstrap weights restricted + renormalised to this cell.
    W = np.vstack([bb_weights[u] for u in keep_units])      # (n_g, S)
    W = W / W.sum(axis=0, keepdims=True)

    resid = dY[:, None] - M                                  # (n_g, S)
    numer = np.einsum("i,is,is->s", aug_w, W, resid)        # (S,)
    denom = np.einsum("i,is->s", delta, W)                  # (S,)
    theta = numer / denom                                    # step (b)

    b_hat = (gamma @ (mbar[:, None] - M)) / n_g             # step (c), (S,)
    return theta - b_hat                                     # check_theta^s


def _summarise_draws(draws: np.ndarray, method: str) -> dict:
    above = float(np.mean(draws > 0))
    below = float(np.mean(draws < 0))
    return {
        "method": method,
        "post_mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "q025": float(np.quantile(draws, 0.025)),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
        "q975": float(np.quantile(draws, 0.975)),
        "p_bayes": min(above, below),
    }


def corrected_estimands(fit: FitResult, propensity_method: str = "logit",
                        n_splits: int = 2, seed: int = 0) -> pd.DataFrame:
    """Posterior-corrected GATT(g,t), event-study ATT(k) and overall ATT.

    Aggregated estimands reuse the per-cell corrected draws with treated-count
    weights and the common Bayesian-bootstrap weights, per Corollary
    ``cor:agg``.  Returns the same tidy schema as
    :func:`did_bcf_revision.did_bcf.plain_estimands`, with ``method='corrected'``.
    """
    df = fit.df
    bb = bayesian_bootstrap_weights(df["unit_id"].to_numpy(), fit.n_draws, seed)

    treated_post = df[df["D"] == 1]
    cells = treated_post[["cohort", "time"]].drop_duplicates()

    cell_draws: dict[tuple, np.ndarray] = {}
    cell_weight: dict[tuple, int] = {}
    records = []
    for g, t in cells.itertuples(index=False):
        draws = _cell_corrected_draws(fit, float(g), int(t), bb,
                                      propensity_method, n_splits, seed)
        if draws is None:
            continue
        cell_draws[(float(g), int(t))] = draws
        cell_weight[(float(g), int(t))] = int(
            treated_post[(treated_post["cohort"] == g) &
                         (treated_post["time"] == t)]["unit_id"].nunique())
        rec = {"estimand_type": "GATT", "estimand_id": f"g={g:g}_t={int(t)}",
               "g": float(g), "t": int(t), "k": int(t - g)}
        rec.update(_summarise_draws(draws, "corrected"))
        records.append(rec)

    def _weighted(keys):
        keys = [k for k in keys if k in cell_draws]
        if not keys:
            return None
        w = np.array([cell_weight[k] for k in keys], dtype=float)
        w /= w.sum()
        stacked = np.vstack([cell_draws[k] for k in keys])     # (len, S)
        return w @ stacked                                     # (S,)

    # Event-study ATT(k)
    for k in sorted({int(t - g) for (g, t) in cell_draws}):
        draws = _weighted([key for key in cell_draws if int(key[1] - key[0]) == k])
        if draws is None:
            continue
        rec = {"estimand_type": "ES", "estimand_id": f"k={k}",
               "g": np.nan, "t": np.nan, "k": k}
        rec.update(_summarise_draws(draws, "corrected"))
        records.append(rec)

    # Overall ATT
    draws = _weighted(list(cell_draws.keys()))
    if draws is not None:
        rec = {"estimand_type": "ATT", "estimand_id": "ATT",
               "g": np.nan, "t": np.nan, "k": np.nan}
        rec.update(_summarise_draws(draws, "corrected"))
        records.append(rec)

    # CATT surface: the correction is a cell-level (GATT) estimator, so -- like
    # the GATT-only R benchmarks -- its corrected estimate and interval are
    # broadcast to each treated observation in the cell and compared to the true
    # individual CATT.  Puts the correction in the same surface table as plain
    # DiD-BCF / TWFE.
    surf = _corrected_surface_record(treated_post, cell_draws)
    if surf is not None:
        records.append(surf)

    return pd.DataFrame.from_records(records)


def _corrected_surface_record(treated_post: pd.DataFrame,
                              cell_draws: dict) -> dict | None:
    """One ``estimand_type='CATT'`` row for the posterior correction."""
    from .metrics import surface_summary

    est, lo90, hi90, lo95, hi95, true = [], [], [], [], [], []
    for (g, t), draws in cell_draws.items():
        members = treated_post[(treated_post["cohort"] == g) &
                               (treated_post["time"] == t)]
        n = len(members)
        if n == 0:
            continue
        est += [float(np.mean(draws))] * n
        lo90 += [float(np.quantile(draws, 0.05))] * n
        hi90 += [float(np.quantile(draws, 0.95))] * n
        lo95 += [float(np.quantile(draws, 0.025))] * n
        hi95 += [float(np.quantile(draws, 0.975))] * n
        true += members["CATT"].astype(float).tolist()
    if not est:
        return None
    rec = {"estimand_type": "CATT", "estimand_id": "surface",
           "g": np.nan, "t": np.nan, "k": np.nan, "method": "corrected"}
    rec.update(surface_summary(true, est, lo90, hi90, lo95, hi95))
    return rec
