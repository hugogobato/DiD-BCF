#!/usr/bin/env python3
"""Collect every per-replication summary in the revision suite into one tidy
frame, then emit the decomposed-metric / CATT-surface / sqrt(N) tables used by
the Workstream B, C and D analysis.

Sources
-------
DiD_BCF_old_generation/summaries_*.csv      plain + corrected DiD-BCF (old
                                            generation; moved out of the tracked
                                            tree to the local legacy folder, see
                                            the note below)
Pretrend/summaries_pretrend_*.csv           DiD-BCF pre-trend diagnostic + TWFE
                                            event-study placebo, 200 reps, at
                                            linearity degrees 1 and 3
Results/summaries_twfe_*.csv                TWFE / OLS
R_code/*_datasets/**/summaries_*.csv        did_dr, did2s, synthdid
Results/_staging/**/summaries_*.csv         wang (grf-DiD), doubleml  [unzipped Colab runs]

The committed ``Results/aggregated/`` files remain the published copy of the
old-generation runs.  Rebuilding them requires the legacy summaries that sit on
disk under ``Old_Simulation_Studies/_moved_from_repo/Simulation_Studies_Revision/DiD_BCF_old_generation/``
(that folder is intentionally outside the tracked tree); without it the rebuild
simply omits the old-generation rows.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from did_bcf_revision.metrics import compute_metrics, surface_metrics, sqrt_n_summary  # noqa: E402

OUT = os.path.join(ROOT, "Results", "aggregated")
# The old-generation DiD-BCF notebooks and summaries were moved out of the
# tracked tree; they stay on disk under the repo-level legacy folder
# (Old_Simulation_Studies/, gitignored).
LEGACY_OLD_GENERATION = os.path.join(
    os.path.dirname(ROOT), "Old_Simulation_Studies", "_moved_from_repo",
    "Simulation_Studies_Revision", "DiD_BCF_old_generation")

PATTERNS = [
    os.path.join(LEGACY_OLD_GENERATION, "summaries_*.csv"),
    os.path.join(ROOT, "Results", "summaries_twfe_*.csv"),
    os.path.join(ROOT, "Results", "summaries_pretrend_*.csv"),
    # The PT_* diagnostic runs were split into rep blocks and land in Pretrend/;
    # the three files in Results/ are an early 24-rep subset of the same seeds
    # (verified identical where they overlap), so the drop_duplicates below
    # keeps whichever is read first without changing any value.  _retired/ holds
    # superseded blocks and is deliberately not globbed.
    os.path.join(ROOT, "Pretrend", "summaries_pretrend_*.csv"),
    os.path.join(ROOT, "R_code", "*_datasets", "**", "summaries_*.csv"),
    os.path.join(ROOT, "Results", "_staging", "**", "summaries_*.csv"),
]

KEEP = ["dgp", "setting", "linearity_degree", "N", "rep", "estimand_type",
        "estimand_id", "g", "t", "k", "method", "post_mean", "sd", "q025",
        "q05", "q95", "q975", "p_bayes", "surf_rmse", "surf_mae", "surf_n",
        "surf_mape", "surf_cover95", "surf_len95", "surf_cover90",
        "surf_len90", "true"]


def load_all() -> pd.DataFrame:
    files = []
    for p in PATTERNS:
        files += glob.glob(p, recursive=True)
    files = sorted(set(files))
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as exc:                       # pragma: no cover
            print("  !! unreadable", f, exc)
            continue
        missing = [c for c in ("setting", "method", "estimand_type") if c not in d]
        if missing:
            print("  ?? skip (missing cols)", os.path.relpath(f, ROOT), missing)
            continue
        for c in KEEP:
            if c not in d:
                d[c] = np.nan
        frames.append(d[KEEP])
    print(f"loaded {len(frames)} summary files")
    all_ = pd.concat(frames, ignore_index=True)
    # The B2 sweep was fitted in parts; identical rows can appear twice.
    all_ = all_.drop_duplicates(
        subset=["setting", "linearity_degree", "N", "rep", "estimand_type",
                "estimand_id", "method"])
    return all_


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load_all()
    print("rows:", len(df))
    print(df.groupby(["method"]).size().to_string())

    cov = (df.groupby(["setting", "N", "linearity_degree", "method"])["rep"]
             .nunique().reset_index(name="n_reps"))
    cov.to_csv(os.path.join(OUT, "coverage_grid.csv"), index=False)

    met = compute_metrics(df)

    # `compute_metrics` can only see replications that produced a row, so its
    # `retention` misses the *hard* failures where an estimator emitted nothing
    # at all. Recover the intended replication count per design as the largest
    # attempt count any method managed there -- a method that ran everywhere
    # defines the target -- and rescale. Callaway--Sant'Anna at D_ramp_nt05
    # reads 0.99 on the within-rows measure and 0.92 on this one, because 15 of
    # its 200 replications returned no rows whatsoever.
    if "n_reps_attempted" in met.columns:
        expected = (met.groupby(["dgp", "setting", "linearity_degree", "N"])
                       ["n_reps_attempted"].transform("max"))
        met["n_reps_expected"] = expected
        met["retention"] = met["n_reps"] / expected
    met.to_csv(os.path.join(OUT, "metrics_all.csv"), index=False)
    surf = surface_metrics(df)
    surf.to_csv(os.path.join(OUT, "surface_all.csv"), index=False)

    sq = []
    for et, eid in (("ATT", "ATT"), ("GATT", "g=4_t=4"), ("GATT", "g=4_t=7")):
        sub = df[(df.estimand_type == et) & (df.estimand_id == eid)]
        if sub.empty:
            continue
        s = sqrt_n_summary(sub, et, eid)
        s["estimand_type"], s["estimand_id"] = et, eid
        sq.append(s)
    pd.concat(sq, ignore_index=True).to_csv(
        os.path.join(OUT, "sqrtn_all.csv"), index=False)

    df.to_csv(os.path.join(OUT, "all_summaries.csv.gz"), index=False,
              compression="gzip")
    print("metrics:", met.shape, "surface:", surf.shape)
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
