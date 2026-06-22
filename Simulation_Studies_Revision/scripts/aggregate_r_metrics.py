#!/usr/bin/env python3
"""Bring the R benchmarks into the *decomposed*-metric table (B2).

The R estimators (``R_code/<scen>_datasets/*.R``) already save per-iteration
RMSE/MAE/MAPE (the "Metrics" sheet) and the per-cell ``estimate / se / true``
(the "Estimates" sheet) -- but nothing computes their **coverage, interval
length, bias, variance or Monte-Carlo SEs**, so they cannot be compared with
DiD-BCF on those axes (a key Reviewer-2 ask).

This script reads each ``*_GATE_and_PValues_linearity_degree=<d>.xlsx`` (or the
``*_Estimates_/_Metrics_*.csv`` fallback), maps the per-cell estimates onto the
same per-replication schema the DiD-BCF/TWFE pipeline uses, and reuses
:mod:`did_bcf_revision.metrics` -- so the metric definitions live in exactly one
place.  CI bounds for the R methods are the normal pivot ``estimate +/- z*se``
(the same convention the R ``sig`` flag uses), which is what those frequentist
estimators report.

Writes:

* ``Results/metrics_r_long.csv``    -- decomposed metrics for the R benchmarks
                                       (one row per dgp/setting/N/estimand/method);
* ``Results/metrics_r_surface.csv`` -- their RMSE/MAE/MAPE surface table
                                       (mean +/- SD across runs), comparable to
                                       ``Results/metrics_surface.csv``.

Run after the R scripts have produced their outputs.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision import metrics as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "Results")
RCODE_DIR = os.path.join(ROOT, "R_code")

_Z90, _Z95 = 1.6448536269514722, 1.959963984540054
# Map an R output file prefix to the method label used in the tables.
_METHOD_OF = {"did_dr": "did_cs", "did2s": "did2s",
              "DoubleML_did": "doubleml", "synthdid": "synthdid"}
_LIN_RE = re.compile(r"linearity_degree=(\d+)")


def _normal_pivot(estimate: np.ndarray, se: np.ndarray) -> pd.DataFrame:
    """estimate/se -> the post_mean/sd/quantile/p_bayes summary columns."""
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    safe = np.where(np.isfinite(se) & (se > 0), se, np.nan)
    z = np.abs(estimate / safe)
    p_bayes = 0.5 * np.array([math.erfc(zz / math.sqrt(2.0)) if np.isfinite(zz)
                              else 0.0 for zz in z])
    return pd.DataFrame({
        "post_mean": estimate, "sd": se,
        "q025": estimate - _Z95 * safe, "q05": estimate - _Z90 * safe,
        "q95": estimate + _Z90 * safe, "q975": estimate + _Z95 * safe,
        "p_bayes": p_bayes,
    })


def _estimand_ids(est: pd.DataFrame) -> pd.DataFrame:
    """Attach estimand_type / estimand_id columns from the R Estimates layout."""
    cols = set(est.columns)
    if {"group", "t"} <= cols:                       # did_cs / doubleml -> GATT
        et = "GATT"
        eid = ["g=%g_t=%d" % (float(g), int(t))
               for g, t in zip(est["group"], est["t"])]
        k = est["k"] if "k" in cols else (est["t"] - est["group"])
        return est.assign(estimand_type=et, estimand_id=eid, k=k)
    if "k" in cols:                                  # did2s -> event study
        return est.assign(estimand_type="ES",
                          estimand_id=["k=%d" % int(k) for k in est["k"]])
    # synthdid -> overall ATT
    return est.assign(estimand_type="ATT", estimand_id="ATT", k=np.nan)


def _read_pair(folder: str, method_prefix: str, d: int):
    """Return (estimates_df, metrics_df) for one method x linearity, or (None, None)."""
    xlsx = os.path.join(folder, f"{method_prefix}_GATE_and_PValues_linearity_degree={d}.xlsx")
    est, met = None, None
    if os.path.exists(xlsx):
        try:
            est = pd.read_excel(xlsx, sheet_name="Estimates", engine="openpyxl")
            met = pd.read_excel(xlsx, sheet_name="Metrics", engine="openpyxl")
        except Exception as e:
            print(f"  ! could not read {xlsx}: {e}")
    else:
        ecsv = os.path.join(folder, f"{method_prefix}_Estimates_linearity_degree={d}.csv")
        mcsv = os.path.join(folder, f"{method_prefix}_Metrics_linearity_degree={d}.csv")
        if os.path.exists(ecsv):
            est = pd.read_csv(ecsv)
        if os.path.exists(mcsv):
            met = pd.read_csv(mcsv)
    return est, met


def collect(rcode_dir: str = RCODE_DIR):
    """Walk R_code/<scen>_datasets/ and assemble one tidy summaries frame."""
    summ_rows = []
    for scen_dir in sorted(glob.glob(os.path.join(rcode_dir, "*_datasets"))):
        scen = os.path.basename(scen_dir).replace("_datasets", "")
        try:
            exp = cfg.get_experiment(scen)
            dgp, base_N = exp.dgp, int(exp.n_values[0])
        except Exception:
            dgp, base_N = "unknown", np.nan
        for prefix, method in _METHOD_OF.items():
            for d in cfg.LINEARITY_DEGREES:
                est, met = _read_pair(scen_dir, prefix, d)
                if est is not None and len(est):
                    est = _estimand_ids(est.copy())
                    piv = _normal_pivot(est["estimate"].to_numpy(),
                                        est["se"].to_numpy())
                    block = pd.concat([est.reset_index(drop=True), piv], axis=1)
                    block = block.rename(columns={"iteration": "rep"})
                    block["true"] = est["true"].to_numpy(dtype=float)
                    block = block.assign(dgp=dgp, setting=scen, linearity_degree=d,
                                         N=base_N, method=method)
                    keep = ["dgp", "setting", "linearity_degree", "N", "rep",
                            "estimand_type", "estimand_id", "k", "true", "method",
                            "post_mean", "sd", "q025", "q05", "q95", "q975", "p_bayes"]
                    summ_rows.append(block[[c for c in keep if c in block.columns]])
                if met is not None and len(met):
                    # The R "Metrics" sheet already holds the per-iteration
                    # surface RMSE/MAE/MAPE -> emit as CATT-surface rows.
                    surf = pd.DataFrame({
                        "dgp": dgp, "setting": scen, "linearity_degree": d,
                        "N": base_N, "rep": met.get("iteration", range(len(met))),
                        "estimand_type": "CATT", "estimand_id": "surface",
                        "method": method,
                        "surf_rmse": met.get("RMSE"), "surf_mae": met.get("MAE"),
                        "surf_mape": met.get("MAPE"),
                    })
                    summ_rows.append(surf)
    if not summ_rows:
        return pd.DataFrame()
    return pd.concat(summ_rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rcode", default=RCODE_DIR)
    ap.add_argument("--results", default=RESULTS_DIR)
    args = ap.parse_args()

    summaries = collect(args.rcode)
    if summaries.empty:
        sys.exit("No R benchmark outputs found. Run the R scripts in "
                 "R_code/<scenario>_datasets/ first.")
    print(f"Assembled {len(summaries)} R benchmark rows "
          f"({summaries['method'].nunique()} methods).")

    metrics_long = M.compute_metrics(summaries)
    surface = M.surface_metrics(summaries)

    os.makedirs(args.results, exist_ok=True)
    p_long = os.path.join(args.results, "metrics_r_long.csv")
    p_surf = os.path.join(args.results, "metrics_r_surface.csv")
    metrics_long.to_csv(p_long, index=False)
    surface.to_csv(p_surf, index=False)
    print(f"Wrote:\n  {p_long}\n  {p_surf}")

    if not surface.empty:
        scols = [c for c in ["setting", "N", "method", "surf_rmse_mean",
                             "surf_mae_mean", "surf_mape_mean"]
                 if c in surface.columns]
        print("\nR benchmark CATT-surface metrics:")
        print(surface[scols].to_string(index=False))


if __name__ == "__main__":
    main()
