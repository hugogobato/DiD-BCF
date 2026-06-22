#!/usr/bin/env python3
"""Aggregate per-replication summaries into decomposed-metric tables (B2).

Reads ``Results/summaries_*.csv`` (written by ``run_did_bcf.py``) and writes:

* ``Results/metrics_long.csv``       -- bias / variance / RMSE / MAE / MAPE /
                                        coverage / interval length / calibration
                                        ratio / size-power (+ Monte-Carlo SEs),
                                        one row per (dgp, setting, N, estimand,
                                        method);
* ``Results/metrics_plain_vs_corrected.csv`` -- the same with plain and corrected
                                        columns side by side;
* ``Results/metrics_surface.csv``    -- the paper's CATT-surface RMSE/MAE/MAPE
                                        (mean +/- SD across runs) + pointwise
                                        CATT coverage, one row per (dgp, setting,
                                        N, method);
* ``Results/sqrt_n_ATT.csv``         -- sqrt(N) * (ATT_hat - ATT) mean/SD by N,
                                        the BvM / sqrt(N) evidence;
* ``Results/metrics_summary.xlsx``   -- all of the above as sheets.

This part is cheap; run it locally.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")


def load_summaries(results_dir: str, pattern: str = "summaries_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No summary files matching {pattern} in {results_dir}. "
            "Run scripts/run_did_bcf.py first.")
    frames = [pd.read_csv(f) for f in files]
    print(f"Loaded {len(files)} summary file(s), "
          f"{sum(len(x) for x in frames)} rows.")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--pattern", default="summaries_*.csv")
    args = ap.parse_args()

    summaries = load_summaries(args.results, args.pattern)

    metrics_long = M.compute_metrics(summaries)
    wide = M.plain_vs_corrected(metrics_long)
    surface = M.surface_metrics(summaries)
    sqrtn = M.sqrt_n_summary(summaries, "ATT", "ATT")

    p_long = os.path.join(args.results, "metrics_long.csv")
    p_wide = os.path.join(args.results, "metrics_plain_vs_corrected.csv")
    p_surf = os.path.join(args.results, "metrics_surface.csv")
    p_sqrt = os.path.join(args.results, "sqrt_n_ATT.csv")
    metrics_long.to_csv(p_long, index=False)
    wide.to_csv(p_wide, index=False)
    surface.to_csv(p_surf, index=False)
    sqrtn.to_csv(p_sqrt, index=False)

    try:
        xlsx = os.path.join(args.results, "metrics_summary.xlsx")
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            metrics_long.to_excel(w, sheet_name="metrics_long", index=False)
            wide.to_excel(w, sheet_name="plain_vs_corrected", index=False)
            surface.to_excel(w, sheet_name="surface_metrics", index=False)
            sqrtn.to_excel(w, sheet_name="sqrt_n_ATT", index=False)
        print(f"Wrote {xlsx}")
    except Exception as e:  # openpyxl missing etc. -- CSVs are still written
        print(f"(skipped xlsx: {e})")

    print(f"Wrote:\n  {p_long}\n  {p_wide}\n  {p_surf}\n  {p_sqrt}")

    if not surface.empty:
        scols = ["setting", "N", "method", "surf_rmse_mean", "surf_mae_mean",
                 "surf_mape_mean", "surf_cover95_mean"]
        scols = [c for c in scols if c in surface.columns]
        print("\nCATT-surface metrics (paper RMSE/MAE/MAPE + pointwise coverage):")
        print(surface[scols].to_string(index=False))

    # Console preview: the headline ATT coverage comparison.
    att = metrics_long[metrics_long["estimand_type"] == "ATT"]
    cols = ["setting", "N", "method", "bias", "emp_sd", "rmse",
            "cover95", "len95", "reject05", "role"]
    cols = [c for c in cols if c in att.columns]
    if not att.empty:
        print("\nATT metrics (plain vs corrected):")
        print(att[cols].to_string(index=False))


if __name__ == "__main__":
    main()
