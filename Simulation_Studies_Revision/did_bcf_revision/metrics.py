"""Decomposed Monte-Carlo metrics (Workstream B2).

The original suite reports only RMSE/MAE/MAPE, which mix bias and variance and
say nothing about coverage or sqrt(N) behaviour.  This module takes the tidy
per-replication summaries written by :mod:`scripts.run_did_bcf` and produces,
per ``(dgp, setting, N, estimand, method)``:

* empirical **bias** and absolute bias,
* Monte-Carlo **SD** of the point estimate (the "variance" piece),
* **RMSE**,
* average reported posterior SD (for comparison with the MC SD),
* **credible-interval coverage** at 90% and 95%,
* average **interval length** at 90% and 95%,
* **rejection rate** of the posterior-probability test at 5% and 10% -- which is
  *size* when the truth is 0 and *power* otherwise.

Everything is computed for **both** ``method='plain'`` and
``method='corrected'`` so the posterior correction can be evaluated directly.

The per-replication input schema (one row per rep x estimand x method)::

    dgp, setting, linearity_degree, N, rep, estimand_type, estimand_id, g, t, k,
    true, method, post_mean, sd, q025, q05, q95, q975, p_bayes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["compute_metrics", "plain_vs_corrected", "sqrt_n_summary"]

GROUP_KEYS = ["dgp", "setting", "linearity_degree", "N",
              "estimand_type", "estimand_id", "method"]
_ZERO_TOL = 1e-8


def _metrics_for_group(g: pd.DataFrame) -> pd.Series:
    true = g["true"].to_numpy(dtype=float)
    est = g["post_mean"].to_numpy(dtype=float)
    err = est - true
    cover90 = np.mean((g["q05"] <= g["true"]) & (g["true"] <= g["q95"]))
    cover95 = np.mean((g["q025"] <= g["true"]) & (g["true"] <= g["q975"]))
    role = "size" if np.all(np.abs(true) < _ZERO_TOL) else "power"
    return pd.Series({
        "n_reps": len(g),
        "mean_true": float(np.mean(true)),
        "bias": float(np.mean(err)),
        "abs_bias": float(np.abs(np.mean(err))),
        "emp_sd": float(np.std(est, ddof=1)) if len(g) > 1 else np.nan,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "avg_post_sd": float(np.mean(g["sd"])),
        "cover90": float(cover90),
        "cover95": float(cover95),
        "len90": float(np.mean(g["q95"] - g["q05"])),
        "len95": float(np.mean(g["q975"] - g["q025"])),
        "reject05": float(np.mean(g["p_bayes"] < 0.025)),
        "reject10": float(np.mean(g["p_bayes"] < 0.05)),
        "role": role,
    })


def compute_metrics(summaries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-rep summaries into the decomposed-metrics table.

    Implemented with an explicit groupby loop so it works identically across
    pandas versions (the ``apply(..., include_groups=...)`` kwarg only exists in
    pandas >= 2.2).
    """
    keys = [k for k in GROUP_KEYS if k in summaries.columns]
    records = []
    for key_vals, g in summaries.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        rec = dict(zip(keys, key_vals))
        rec.update(_metrics_for_group(g).to_dict())
        records.append(rec)
    return pd.DataFrame.from_records(records)


def plain_vs_corrected(metrics: pd.DataFrame,
                       columns=("bias", "emp_sd", "rmse",
                                "cover90", "cover95", "len95",
                                "reject05")) -> pd.DataFrame:
    """Pivot the metrics table to put plain and corrected side by side."""
    idx = [k for k in ["dgp", "setting", "linearity_degree", "N",
                       "estimand_type", "estimand_id", "role"]
           if k in metrics.columns]
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
    keys = [k for k in ["dgp", "setting", "linearity_degree", "N", "method"]
            if k in sub.columns]
    return (sub.groupby(keys, dropna=False)["scaled_err"]
            .agg(mean="mean", sd="std", rmse=lambda x: np.sqrt(np.mean(x ** 2)))
            .reset_index())
