"""Posterior summaries of the structured fit: GATT(g,t), event study, ATT, CATT.

The schema matches ``did_bcf_revision.did_bcf.plain_estimands`` so that the
structured estimator drops into the revision's metrics and aggregation layer
without changes: one row per estimand, with ``estimand_type`` in
``{GATT, ES, ATT, CATT}`` and posterior mean, sd and quantiles.

Each averaged estimand is formed draw by draw -- average the per-observation
effect over the relevant treated post-treatment rows *within* a draw, then
summarise the resulting posterior sample -- so the credible interval reflects
the posterior of the average rather than an average of intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["structured_estimands", "summarise_draws"]


def summarise_draws(draws: np.ndarray) -> dict:
    """Posterior summary of a 1-D sample of an averaged estimand."""
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


def structured_estimands(model, method: str = "structured",
                         use_contrast: bool = False) -> pd.DataFrame:
    """Tidy posterior summaries from a fitted :class:`StructuredDiDBCF`.

    Parameters
    ----------
    model : a fitted ``StructuredDiDBCF``.
    method : value written into the ``method`` column.
    use_contrast : report the DiD contrast of the fitted regression function
        instead of the treatment forest.  In the structured model the two agree
        exactly whenever the covariates are time-invariant, so this is a check
        rather than a different estimator; see
        :meth:`StructuredDiDBCF.check_identification`.
    """
    df = model.df
    cols = model.design.cols
    effect = (model.did_contrast_draws() if use_contrast else model.tau_draws)

    # The contrast is undefined on treated units whose g-1 period is missing.
    usable = np.isfinite(effect[:, 0])
    treated_post = df.index[(df[cols.treated] == 1) & usable].to_numpy()
    records = []

    def add(estimand_type, estimand_id, g, t, k, rows):
        rows = np.intersect1d(np.asarray(rows), treated_post)
        if rows.size == 0:
            return
        draws = effect[rows, :].mean(axis=0)
        rec = {"estimand_type": estimand_type, "estimand_id": estimand_id,
               "g": g, "t": t, "k": k, "method": method}
        rec.update(summarise_draws(draws))
        records.append(rec)

    post = df.loc[treated_post]
    for (g, t), grp in post.groupby([cols.cohort, cols.time]):
        add("GATT", f"g={g:g}_t={int(t)}", float(g), int(t), int(t - g),
            grp.index.to_numpy())
    for k, grp in post.groupby(cols.event_time):
        add("ES", f"k={int(k)}", np.nan, np.nan, int(k), grp.index.to_numpy())
    add("ATT", "ATT", np.nan, np.nan, np.nan, treated_post)

    surf = _surface_record(df, effect, treated_post, method)
    if surf is not None:
        records.append(surf)
    return pd.DataFrame.from_records(records)


def _surface_record(df: pd.DataFrame, effect: np.ndarray,
                    rows: np.ndarray, method: str) -> dict | None:
    """One ``estimand_type='CATT'`` row of within-replication surface errors.

    Reported only when the panel carries the true individual ``CATT``, i.e. in
    simulation.  Uses ``did_bcf_revision.metrics.surface_summary`` when that
    package is importable so the columns line up with the rest of the suite, and
    falls back to a self-contained computation otherwise.
    """
    if rows.size == 0 or "CATT" not in df.columns:
        return None
    obs = effect[rows, :]
    est = obs.mean(axis=1)
    lo90, hi90 = np.quantile(obs, 0.05, axis=1), np.quantile(obs, 0.95, axis=1)
    lo95, hi95 = np.quantile(obs, 0.025, axis=1), np.quantile(obs, 0.975, axis=1)
    true = df.loc[rows, "CATT"].to_numpy(dtype=float)

    rec = {"estimand_type": "CATT", "estimand_id": "surface",
           "g": np.nan, "t": np.nan, "k": np.nan, "method": method}
    try:
        from did_bcf_revision.metrics import surface_summary
        rec.update(surface_summary(true, est, lo90, hi90, lo95, hi95))
    except ImportError:
        err = est - true
        nz = np.abs(true) > 1e-8
        rec.update({
            "post_mean": float(np.mean(est)),
            "surf_rmse": float(np.sqrt(np.mean(err ** 2))),
            "surf_mae": float(np.mean(np.abs(err))),
            "surf_mape": (float(np.mean(np.abs(err[nz] / true[nz])))
                          if nz.any() else np.nan),
            "surf_n": int(err.size),
            "surf_cover90": float(np.mean((lo90 <= true) & (true <= hi90))),
            "surf_cover95": float(np.mean((lo95 <= true) & (true <= hi95))),
            "surf_len90": float(np.mean(hi90 - lo90)),
            "surf_len95": float(np.mean(hi95 - lo95)),
        })
    return rec
