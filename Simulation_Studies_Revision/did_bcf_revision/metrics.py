"""Decomposed Monte-Carlo metrics (Workstream B2).

The original suite reports only RMSE/MAE/MAPE, which mix bias and variance and
say nothing about coverage or sqrt(N) behaviour.  This module takes the tidy
per-replication summaries written by :mod:`scripts.run_did_bcf` and produces,
per ``(dgp, setting, N, estimand, method)``:

* empirical **bias** and absolute bias,
* Monte-Carlo **SD** of the point estimate (the "variance" piece) and its square,
* **RMSE**, and -- for parity with the paper's tables -- **MAE** and **MAPE**,
* average reported posterior SD and the **calibration ratio** ``avg_post_sd /
  emp_sd`` (a value < 1 flags over-confident, > 1 under-confident intervals),
* **credible-interval coverage** at 90% and 95%,
* average **interval length** at 90% and 95%,
* **rejection rate** of the posterior-probability test at 5% and 10% -- which is
  *size* when the truth is 0 and *power* otherwise,
* **Monte-Carlo standard errors** for bias, coverage and the rejection rates, so
  the precision of the size/coverage estimates is reported alongside them (R2).

Everything is computed for **both** ``method='plain'`` and
``method='corrected'`` (and ``method='twfe'``) so the estimators can be evaluated
side by side.

The **CATT surface** rows (``estimand_type='CATT'``, one per replication, written
by the emitters) carry the within-replication RMSE/MAE/MAPE over the individual
treated observations -- the paper's headline metric and the evidence that
DiD-BCF recovers the *heterogeneous* effect that GATT-only methods cannot.  They
are aggregated separately by :func:`surface_metrics` (mean +/- SD across runs),
not mixed into the scalar-estimand decomposition.

The per-replication input schema (one row per rep x estimand x method)::

    dgp, setting, linearity_degree, N, rep, estimand_type, estimand_id, g, t, k,
    true, method, post_mean, sd, q025, q05, q95, q975, p_bayes
    [+ surf_rmse, surf_mae, surf_mape, surf_cover90, surf_cover95,
       surf_len90, surf_len95, surf_n   on estimand_type='CATT' rows]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["compute_metrics", "plain_vs_corrected", "sqrt_n_summary",
           "surface_summary", "surface_metrics"]

# ``spec`` is the identification specification the fit used ("published" and the
# repairs of ``Results/identification_note.tex``).  It is a grouping key wherever
# it is present, so a route run and a published run never pool into one row;
# summaries written before it existed simply lack the column and are grouped as
# before.  ``scripts/aggregate_metrics.py`` fills it with "published" so a mixed
# set of files still lines up.
SPEC_KEY = "spec"
GROUP_KEYS = ["dgp", "setting", "linearity_degree", "N", SPEC_KEY,
              "estimand_type", "estimand_id", "method"]
SURFACE_GROUP_KEYS = ["dgp", "setting", "linearity_degree", "N", SPEC_KEY, "method"]
SURFACE_TYPE = "CATT"
SURFACE_COLS = ["surf_rmse", "surf_mae", "surf_mape",
                "surf_cover90", "surf_cover95", "surf_len90", "surf_len95"]
_ZERO_TOL = 1e-8
# Per-replication *decision rules*: a test, not a point estimate.  Their emitters
# leave the estimate and interval columns missing by design, so they must be
# routed away from the point-estimate path (which would drop every replication
# and report ``n_reps = 0``).
#
# This was an exact-match set of two names, which missed every rule the pre-trend
# module actually emits under a prefix -- ``joint`` (the TWFE Wald pre-test) and
# the subgroup contrasts ``X1_any`` / ``X1_any_bonf`` / ``X2_any`` /
# ``X2_any_bonf``, the row that carries the whole heterogeneous-violation
# argument.  All of them silently aggregated to ``n_reps = 0, role = undefined``.
# Matching on the *suffix* covers those, and the benchmark pre-trend scripts that
# emit the same family, without catching any other id in the suite (``ATT``,
# ``g=4_t=4``, ``k=0``, ``slope``, ``X1_slope``, ``X1_k=-2``, ``surface``).
DECISION_ESTIMANDS = frozenset(("any", "any_bonf", "joint"))
_DECISION_SUFFIXES = tuple("_" + s for s in DECISION_ESTIMANDS)


def is_decision_estimand(estimand_id: str) -> bool:
    """True for the per-replication test rows (no point estimate, only a tail)."""
    return (estimand_id in DECISION_ESTIMANDS
            or estimand_id.endswith(_DECISION_SUFFIXES))


# --------------------------------------------------------------------------- #
# Within-replication CATT-surface summary (shared by all emitters)
# --------------------------------------------------------------------------- #
def surface_summary(true, est, lo90=None, hi90=None, lo95=None, hi95=None) -> dict:
    """RMSE/MAE/MAPE (+ optional pointwise coverage/length) over a CATT surface.

    Computed *within one replication* over the individual treated observations,
    exactly matching the paper's RMSE/MAE/MAPE definitions
    (``sqrt(mean((tau_hat - tau)^2))`` etc.).  ``est`` is the per-observation
    point estimate (posterior mean / coefficient); the optional ``lo*/hi*`` are
    the per-observation credible/confidence bounds, enabling a *pointwise* CATT
    coverage -- a calibration check for the heterogeneous effect that the
    averaged-estimand coverage cannot provide.
    """
    true = np.asarray(true, dtype=float)
    est = np.asarray(est, dtype=float)
    err = est - true
    out = {
        "surf_rmse": float(np.sqrt(np.mean(err ** 2))),
        "surf_mae": float(np.mean(np.abs(err))),
        "surf_n": int(err.size),
    }
    nz = np.abs(true) > _ZERO_TOL
    out["surf_mape"] = (float(np.mean(np.abs(err[nz] / true[nz])))
                        if nz.any() else np.nan)
    if lo95 is not None and hi95 is not None:
        lo95 = np.asarray(lo95, dtype=float); hi95 = np.asarray(hi95, dtype=float)
        out["surf_cover95"] = float(np.mean((lo95 <= true) & (true <= hi95)))
        out["surf_len95"] = float(np.mean(hi95 - lo95))
    if lo90 is not None and hi90 is not None:
        lo90 = np.asarray(lo90, dtype=float); hi90 = np.asarray(hi90, dtype=float)
        out["surf_cover90"] = float(np.mean((lo90 <= true) & (true <= hi90)))
        out["surf_len90"] = float(np.mean(hi90 - lo90))
    return out


# --------------------------------------------------------------------------- #
# Decomposed metrics for the scalar (averaged) estimands
# --------------------------------------------------------------------------- #
def _empty_metrics(n_attempted: int) -> dict:
    """Return the common shape for a group with no usable replications."""
    return {"n_reps": 0, "n_reps_attempted": n_attempted,
            "retention": 0.0, "role": "undefined"}


def _decision_metrics_for_group(g: pd.DataFrame) -> pd.Series:
    """Aggregate an any-k decision rule using its posterior tail only.

    ``any`` and ``any_bonf`` are per-replication tests, not point estimates.
    Their emitters therefore leave the point-estimate and interval columns
    missing by design.  They must not pass through the ordinary point-estimate
    filter, which would drop every replication before computing rejection
    rates.
    """
    n_attempted = len(g)
    finite = (np.isfinite(g["p_bayes"].to_numpy(dtype=float)) &
              np.isfinite(g["true"].to_numpy(dtype=float)))
    g = g[finite]
    if g.empty:
        return pd.Series(_empty_metrics(n_attempted))

    truth = g["true"].to_numpy(dtype=float)
    p_bayes = g["p_bayes"].to_numpy(dtype=float)
    m = len(g)
    reject05 = float(np.mean(p_bayes < 0.025))
    reject10 = float(np.mean(p_bayes < 0.05))
    nz = np.abs(truth) > _ZERO_TOL
    return pd.Series({
        "n_reps": m,
        "n_reps_attempted": n_attempted,
        "retention": m / n_attempted if n_attempted else np.nan,
        "mean_true": float(np.mean(truth)),
        # Decision rules have no point estimate, interval, or calibration
        # statistic.  Keep the columns present, but do not invent values.
        "bias": np.nan,
        "abs_bias": np.nan,
        "emp_sd": np.nan,
        "variance": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "avg_post_sd": np.nan,
        "sd_ratio": np.nan,
        "cover90": np.nan,
        "cover95": np.nan,
        "len90": np.nan,
        "len95": np.nan,
        "reject05": reject05,
        "reject10": reject10,
        "mcse_bias": np.nan,
        "mcse_cover90": np.nan,
        "mcse_cover95": np.nan,
        "mcse_reject05": float(np.sqrt(reject05 * (1 - reject05) / m)),
        "mcse_reject10": float(np.sqrt(reject10 * (1 - reject10) / m)),
        "role": "size" if not nz.any() else "power",
    })


def _metrics_for_group(g: pd.DataFrame) -> pd.Series:
    estimand_id = str(g["estimand_id"].iloc[0]) if "estimand_id" in g else ""
    if is_decision_estimand(estimand_id):
        return _decision_metrics_for_group(g)

    # An estimator can fail on a replication and still emit a row: Callaway--
    # Sant'Anna does this at the far end of the ramp, where a 5% never-treated
    # share leaves it almost no clean comparison group. Those rows must be
    # dropped *before* anything is computed, and dropped from every statistic
    # alike. Left in, they corrupt the table two different ways at once: the
    # mean-based columns (bias, rmse, mae, emp_sd) propagate the NaN and go
    # blank, while the coverage columns compare `NaN <= true`, get False, and
    # silently score the failure as a miss -- biasing coverage downward.
    n_attempted = len(g)
    finite = (np.isfinite(g["post_mean"].to_numpy(dtype=float)) &
              np.isfinite(g["true"].to_numpy(dtype=float)))
    g = g[finite]
    if g.empty:
        return pd.Series(_empty_metrics(n_attempted))

    true = g["true"].to_numpy(dtype=float)
    est = g["post_mean"].to_numpy(dtype=float)
    err = est - true
    m = len(g)
    emp_sd = float(np.std(est, ddof=1)) if m > 1 else np.nan
    avg_post_sd = float(np.mean(g["sd"]))
    cover90 = float(np.mean((g["q05"] <= g["true"]) & (g["true"] <= g["q95"])))
    cover95 = float(np.mean((g["q025"] <= g["true"]) & (g["true"] <= g["q975"])))
    reject05 = float(np.mean(g["p_bayes"] < 0.025))   # two-sided 5% test
    reject10 = float(np.mean(g["p_bayes"] < 0.05))    # two-sided 10% test
    nz = np.abs(true) > _ZERO_TOL
    mape = float(np.mean(np.abs(err[nz] / true[nz]))) if nz.any() else np.nan
    role = "size" if not nz.any() else "power"
    return pd.Series({
        "n_reps": m,
        # Every metric above is computed on these m replications. When
        # retention < 1 the row describes a *selected* subsample -- the runs in
        # which the estimator returned an estimate -- and is not comparable with
        # a row from an estimator that ran everywhere.
        "n_reps_attempted": n_attempted,
        "retention": m / n_attempted if n_attempted else np.nan,
        "mean_true": float(np.mean(true)),
        "bias": float(np.mean(err)),
        "abs_bias": float(np.abs(np.mean(err))),
        "emp_sd": emp_sd,
        "variance": emp_sd ** 2 if np.isfinite(emp_sd) else np.nan,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": mape,
        "avg_post_sd": avg_post_sd,
        "sd_ratio": (avg_post_sd / emp_sd) if (np.isfinite(emp_sd) and emp_sd > 0)
                    else np.nan,
        "cover90": cover90,
        "cover95": cover95,
        "len90": float(np.mean(g["q95"] - g["q05"])),
        "len95": float(np.mean(g["q975"] - g["q025"])),
        "reject05": reject05,
        "reject10": reject10,
        # Monte-Carlo standard errors of the headline summaries (R2 scrutiny).
        "mcse_bias": (emp_sd / np.sqrt(m)) if np.isfinite(emp_sd) else np.nan,
        "mcse_cover90": float(np.sqrt(cover90 * (1 - cover90) / m)),
        "mcse_cover95": float(np.sqrt(cover95 * (1 - cover95) / m)),
        "mcse_reject05": float(np.sqrt(reject05 * (1 - reject05) / m)),
        "mcse_reject10": float(np.sqrt(reject10 * (1 - reject10) / m)),
        "role": role,
    })


def compute_metrics(summaries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-rep summaries into the decomposed-metrics table.

    The CATT-surface rows (``estimand_type='CATT'``) are excluded here and
    handled by :func:`surface_metrics`; this function covers the scalar averaged
    estimands (GATT/ES/ATT) whose sampling distribution the decomposition
    describes.

    Implemented with an explicit groupby loop so it works identically across
    pandas versions (the ``apply(..., include_groups=...)`` kwarg only exists in
    pandas >= 2.2).
    """
    if "estimand_type" in summaries.columns:
        summaries = summaries[summaries["estimand_type"] != SURFACE_TYPE]
    keys = [k for k in GROUP_KEYS if k in summaries.columns]
    records = []
    for key_vals, g in summaries.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        rec = dict(zip(keys, key_vals))
        rec.update(_metrics_for_group(g).to_dict())
        records.append(rec)
    return pd.DataFrame.from_records(records)


def surface_metrics(summaries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-replication CATT-surface rows (mean +/- SD across runs).

    Reproduces the paper's RMSE/MAE/MAPE tables (reported as ``mean +/- SD``
    over the Monte-Carlo runs) and adds the average pointwise CATT coverage and
    interval length.  One row per ``(dgp, setting, linearity_degree, N,
    method)``.  Methods that cannot estimate individual CATT (e.g. plain TWFE
    falls back to broadcasting its event-study coefficient) are still reported,
    so the GATT-only competitors and DiD-BCF sit in one comparison table.
    """
    if "estimand_type" not in summaries.columns:
        return pd.DataFrame()
    sub = summaries[summaries["estimand_type"] == SURFACE_TYPE].copy()
    if sub.empty:
        return pd.DataFrame()
    keys = [k for k in SURFACE_GROUP_KEYS if k in sub.columns]
    cols = [c for c in SURFACE_COLS if c in sub.columns]
    records = []
    for key_vals, g in sub.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        rec = dict(zip(keys, key_vals))
        rec["n_reps"] = len(g)
        if "surf_n" in g.columns:
            rec["avg_n_treated_obs"] = float(np.nanmean(g["surf_n"]))
        for c in cols:
            vals = g[c].to_numpy(dtype=float)
            rec[f"{c}_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
            rec[f"{c}_sd"] = (float(np.nanstd(vals, ddof=1))
                             if np.sum(~np.isnan(vals)) > 1 else np.nan)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def plain_vs_corrected(metrics: pd.DataFrame,
                       columns=("bias", "emp_sd", "rmse", "mae",
                                "cover90", "cover95", "len95",
                                "sd_ratio", "reject05")) -> pd.DataFrame:
    """Pivot the metrics table to put plain and corrected side by side."""
    idx = [k for k in ["dgp", "setting", "linearity_degree", "N", SPEC_KEY,
                       "estimand_type", "estimand_id", "role"]
           if k in metrics.columns]
    columns = [c for c in columns if c in metrics.columns]
    wide = metrics.pivot_table(index=idx, columns="method",
                               values=list(columns), aggfunc="first")
    wide.columns = [f"{m}_{meth}" for m, meth in wide.columns]
    return wide.reset_index()


def sqrt_n_summary(summaries: pd.DataFrame, estimand_type: str = "ATT",
                   estimand_id: str = "ATT") -> pd.DataFrame:
    """``sqrt(N) * (estimate - true)`` per (setting, N, method).

    Its **mean -> 0** and **SD stabilising** across N is the finite-sample
    evidence for the sqrt(N) / Bernstein-von Mises claim (Theorem 3 / the BvM
    theorems).  Returns mean, SD and the implied scaled RMSE.
    """
    sub = summaries[(summaries["estimand_type"] == estimand_type) &
                    (summaries["estimand_id"] == estimand_id)].copy()
    sub["scaled_err"] = np.sqrt(sub["N"].astype(float)) * (sub["post_mean"] - sub["true"])
    keys = [k for k in SURFACE_GROUP_KEYS if k in sub.columns]
    return (sub.groupby(keys, dropna=False)["scaled_err"]
            .agg(mean="mean", sd="std", rmse=lambda x: np.sqrt(np.mean(x ** 2)))
            .reset_index())
