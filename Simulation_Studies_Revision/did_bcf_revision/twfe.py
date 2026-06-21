"""Two-way fixed-effects (TWFE) estimators used as the comparison in Workstream D.

These are deliberately plain TWFE -- the estimator Goodman-Bacon (2021) shows is
contaminated by "already-treated as control" comparisons under dynamic,
cohort-varying effects.  They are the foil against which DiD-BCF (and the
posterior correction) are compared.

Fixed effects are partialled out by exact two-way demeaning (valid for balanced
panels) and the treatment terms estimated by OLS on the residualised design
(Frisch-Waugh-Lovell), so nothing blows up with many units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["twfe_att", "twfe_event_study",
           "twfe_att_se", "twfe_event_study_se"]


def _two_way_demean(df: pd.DataFrame, cols, unit="unit_id", time="time"):
    """Return ``cols`` with unit and time means removed and grand mean added.

    Exact for balanced panels (one observation per unit-time).
    """
    out = {}
    g_unit = df.groupby(unit)
    g_time = df.groupby(time)
    for c in cols:
        x = df[c].astype(float)
        x = x - g_unit[c].transform("mean") - g_time[c].transform("mean") + x.mean()
        out[c] = x.to_numpy()
    return out


def twfe_att(df: pd.DataFrame, outcome="Y", treat="D",
             unit="unit_id", time="time") -> float:
    """Static TWFE coefficient on the treatment indicator ``D``."""
    dm = _two_way_demean(df, [outcome, treat], unit, time)
    y, d = dm[outcome], dm[treat]
    denom = float(d @ d)
    if denom <= 0:
        return np.nan
    return float((d @ y) / denom)


def _ols_cluster(Xd: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS of ``y`` on ``Xd`` (no intercept) with cluster-robust (CR1) SEs.

    ``Xd`` is assumed already two-way-demeaned (fixed effects partialled out), so
    the unit / time intercepts do not appear as columns; ``clusters`` gives the
    unit id of each row.  Returns ``(beta, se)``.
    """
    n, k = Xd.shape
    XtX = Xd.T @ Xd
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xd.T @ y)
    resid = y - Xd @ beta

    meat = np.zeros((k, k))
    uniq = np.unique(clusters)
    for c in uniq:
        m = clusters == c
        Xc = Xd[m]
        uc = resid[m]
        s = Xc.T @ uc
        meat += np.outer(s, s)

    G = uniq.size
    # CR1 small-sample correction (absorbed FE -> use demeaned column count k).
    dof = max(n - k, 1)
    corr = (G / max(G - 1, 1)) * ((n - 1) / dof)
    V = corr * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    return beta, se


def twfe_att_se(df: pd.DataFrame, outcome="Y", treat="D",
                unit="unit_id", time="time"):
    """Static TWFE ATT on ``D`` with cluster-robust (by unit) SE.

    Returns ``(coef, se)``.
    """
    dm = _two_way_demean(df, [outcome, treat], unit, time)
    y = dm[outcome]
    Xd = dm[treat][:, None]
    beta, se = _ols_cluster(Xd, y, df[unit].to_numpy())
    return float(beta[0]), float(se[0])


def twfe_event_study_se(df: pd.DataFrame, outcome="Y", unit="unit_id",
                        time="time", event="event_time", ref=-1,
                        k_min=None, k_max=None) -> pd.DataFrame:
    """TWFE event-study coefficients with cluster-robust (by unit) SEs.

    Like :func:`twfe_event_study` but returns a tidy frame ``[k, coef, se]``
    (the omitted reference ``k=ref`` is returned with coef/se = 0/NaN).
    """
    d = df.copy()
    k = d[event].to_numpy(dtype=float)
    finite = np.isfinite(k)
    ks = np.unique(k[finite]).astype(int)
    if k_min is not None:
        ks = ks[ks >= k_min]
    if k_max is not None:
        ks = ks[ks <= k_max]
    ks = [int(v) for v in ks if int(v) != ref]
    if not ks:
        return pd.DataFrame({"k": [], "coef": [], "se": []})

    dummy_cols = []
    for kk in ks:
        col = f"_es_{kk}"
        d[col] = ((k == kk) & finite).astype(float)
        dummy_cols.append(col)

    dm = _two_way_demean(d, [outcome] + dummy_cols, unit, time)
    y = dm[outcome]
    Xd = np.column_stack([dm[c] for c in dummy_cols])
    beta, se = _ols_cluster(Xd, y, d[unit].to_numpy())
    rows = [{"k": ref, "coef": 0.0, "se": np.nan}]
    rows += [{"k": kk, "coef": float(b), "se": float(s)}
             for kk, b, s in zip(ks, beta, se)]
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def twfe_event_study(df: pd.DataFrame, outcome="Y", unit="unit_id",
                     time="time", event="event_time", ref=-1,
                     k_min=None, k_max=None) -> pd.DataFrame:
    """Plain TWFE event-study coefficients (relative-time dummies).

    Never-treated units (``event_time`` NaN) contribute all-zero event dummies,
    serving as clean controls.  The reference event time ``ref`` is omitted.
    Returns a tidy frame ``[k, coef]``.
    """
    d = df.copy()
    k = d[event].to_numpy(dtype=float)
    finite = np.isfinite(k)
    ks = np.unique(k[finite]).astype(int)
    if k_min is not None:
        ks = ks[ks >= k_min]
    if k_max is not None:
        ks = ks[ks <= k_max]
    ks = [int(v) for v in ks if int(v) != ref]

    dummy_cols = []
    for kk in ks:
        col = f"_es_{kk}"
        d[col] = ((k == kk) & finite).astype(float)
        dummy_cols.append(col)

    dm = _two_way_demean(d, [outcome] + dummy_cols, unit, time)
    y = dm[outcome]
    Xd = np.column_stack([dm[c] for c in dummy_cols]) if dummy_cols else np.empty((len(y), 0))
    if Xd.shape[1] == 0:
        return pd.DataFrame({"k": [], "coef": []})
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    rows = [{"k": ref, "coef": 0.0}] + [{"k": kk, "coef": float(b)}
                                        for kk, b in zip(ks, beta)]
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
