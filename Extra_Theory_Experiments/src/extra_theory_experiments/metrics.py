"""Replication metrics with truth-varying errors.

The empirical SD used for calibration is SD(est - true), not SD(est).  Raw
SD(est) is retained under a distinct name.  Truth joins always include
estimand_type and estimand_id; a GATT truth is never broadcast to CATT rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def replication_errors(estimates, truth):
    est = np.asarray(estimates, float)
    true = np.asarray(truth, float)
    if est.shape != true.shape:
        raise ValueError("estimate and truth arrays must have the same shape")
    return est - true


def scalar_metrics(estimates, truth, posterior_sd=None, q05=None, q95=None,
                    q025=None, q975=None) -> dict[str, float]:
    """Compute metrics from replication-specific errors."""
    err = replication_errors(estimates, truth)
    keep = np.isfinite(err)
    err = err[keep]
    est = np.asarray(estimates, float)[keep]
    true = np.asarray(truth, float)[keep]
    n = len(err)
    if n == 0:
        return {"n_reps": 0}
    err_sd = float(np.std(err, ddof=1)) if n > 1 else np.nan
    raw_sd = float(np.std(est, ddof=1)) if n > 1 else np.nan
    out = {
        "n_reps": int(n),
        "mean_true": float(np.mean(true)),
        "bias": float(np.mean(err)),
        "abs_bias": float(abs(np.mean(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "empirical_sd_error": err_sd,
        "emp_sd": err_sd,
        "raw_sd_est": raw_sd,
        "raw_variance_est": raw_sd ** 2 if np.isfinite(raw_sd) else np.nan,
        "bias_mcse": err_sd / np.sqrt(n) if np.isfinite(err_sd) else np.nan,
        "mcse_bias": err_sd / np.sqrt(n) if np.isfinite(err_sd) else np.nan,
    }
    if posterior_sd is not None:
        psd = np.asarray(posterior_sd, float)[keep]
        out["avg_post_sd"] = float(np.nanmean(psd))
        out["sd_ratio_error"] = (out["avg_post_sd"] / err_sd
                                 if np.isfinite(err_sd) and err_sd > 0 else np.nan)
        out["sd_ratio"] = out["sd_ratio_error"]
    for level, lo, hi in ((.90, q05, q95), (.95, q025, q975)):
        if lo is not None and hi is not None:
            lo, hi = np.asarray(lo, float)[keep], np.asarray(hi, float)[keep]
            covered = (lo <= true) & (true <= hi)
            tag = "90" if level == .90 else "95"
            out[f"cover{tag}"] = float(np.mean(covered))
            out[f"mcse_cover{tag}"] = float(np.sqrt(
                out[f"cover{tag}"] * (1 - out[f"cover{tag}"]) / n))
            out[f"len{tag}"] = float(np.nanmean(hi - lo))
    return out


def attach_replication_truth(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join truth by replication and exact estimand identity, never by CATT shape."""
    key = ["rep", "estimand_type", "estimand_id"]
    missing = [x for x in key if x not in estimates or x not in truth]
    if missing:
        raise ValueError(f"truth join requires columns {missing}")
    if "estimand_type" in estimates and (estimates["estimand_type"] == "CATT").any():
        # Surface truth may be supplied only for surface rows with exact id.
        bad = estimates.loc[estimates["estimand_type"] == "CATT", "estimand_id"].astype(str)
        if not np.all(bad.eq("surface")):
            raise ValueError("CATT rows require estimand_id='surface'; no scalar truth broadcast")
    return estimates.merge(truth[key + ["true"]], on=key, how="left",
                           validate="many_to_one")


def summarise_replications(summaries: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    """Aggregate per-replication rows for separate mean and median summaries."""
    if summaries.empty:
        return pd.DataFrame()
    work = summaries.copy()
    if "estimand_type" in work:
        work = work[work["estimand_type"] != "CATT"].copy()
    if group_cols is None:
        group_cols = [c for c in (
            "family", "design", "degree", "N", "task_estimator", "estimator",
            "method", "estimand_type", "estimand_id"
        ) if c in work]
    if not group_cols:
        group_cols = ["estimand_id"] if "estimand_id" in work else []
    records = []
    grouped = work.groupby(group_cols, dropna=False) if group_cols else [((), work)]
    for vals, grp in grouped:
        if not isinstance(vals, tuple):
            vals = (vals,)
        for point_summary, estimate_col in (("mean", "post_mean"),
                                            ("median", "post_median")):
            if estimate_col not in grp:
                continue
            rec = dict(zip(group_cols, vals))
            rec["point_summary"] = point_summary
            kwargs = {"estimates": grp[estimate_col], "truth": grp["true"]}
            for name in ("sd", "q05", "q95", "q025", "q975"):
                if name in grp:
                    kwargs[name if name != "sd" else "posterior_sd"] = grp[name]
            rec.update(scalar_metrics(**kwargs))
            records.append(rec)
    return pd.DataFrame.from_records(records)


def assert_no_catt_broadcast(summaries: pd.DataFrame) -> None:
    """Fail if a scalar estimator was copied into an individual CATT surface."""
    if "estimand_type" not in summaries or "estimand_id" not in summaries:
        return
    catt = summaries[summaries["estimand_type"] == "CATT"]
    if catt.empty:
        return
    if not np.all(catt["estimand_id"].astype(str).eq("surface")):
        raise AssertionError("CATT surface rows must carry exact surface identity")
    if "truth_source" in catt and np.any(catt["truth_source"].astype(str).str.contains("GATT")):
        raise AssertionError("GATT truth may not be broadcast to CATT")
