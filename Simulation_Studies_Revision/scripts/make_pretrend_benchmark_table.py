"""Pre-trend workstream: DiD-BCF against the benchmark pre-tests, one table set.

Pools every PT summary file in the tree -- the Bayesian diagnostic
(`Pretrend/summaries_pretrend_*`), the TWFE event study (`twfe_es`, written by
the same notebooks) and the benchmark scripts run under
`R_code/PT_*_datasets/` -- and scores them all through the engine's own
`compute_metrics`, so every method is judged on the same estimand with the same
decision rule.

    python3 scripts/make_pretrend_benchmark_table.py [-o Pretrend/BENCHMARK_TABLES.txt]

Estimands (all of them target Delta(k), the differential treated-minus-control
change from k = -1 to k):

    slope        least-squares slope of Delta(k) through the origin
    any_bonf     min-p over the pre-periods, Bonferroni-scaled
    joint        Wald test of Delta(k) = 0 for all k          (no Bayesian analogue)
    X1_*         the same, for the CONTRAST between the two X_1 subgroups

`reject05` is size where `mean_true` is 0 and detection power otherwise.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from did_bcf_revision.metrics import compute_metrics  # noqa: E402

# Display order: the Bayesian diagnostic first, then the benchmarks, then TWFE.
METHODS = ["pretrend", "did_dr", "doubleml", "did2s", "synthdid", "wang", "twfe_es"]
SCENARIOS = ["PT_hold", "PT_conditional", "PT_violation_g05", "PT_violation_g10",
             "PT_violation_g20", "PT_violation_g40", "PT_violation_het10",
             "PT_violation_het20", "PT_violation_het40", "PT_violation_a20"]
PANELS = [
    ("slope", "reject05", "AGGREGATE slope test -- rejection rate"),
    ("any_bonf", "reject05", "AGGREGATE any-k Bonferroni -- rejection rate"),
    ("joint", "reject05", "AGGREGATE joint Wald -- rejection rate"),
    ("slope", "bias", "AGGREGATE slope -- bias"),
    ("slope", "cover95", "AGGREGATE slope -- 95% coverage"),
    ("slope", "n_reps", "AGGREGATE slope -- replications scored"),
    ("X1_slope", "reject05", "SUBGROUP X1 contrast slope -- rejection rate"),
    ("X1_any_bonf", "reject05", "SUBGROUP X1 any-k Bonferroni -- rejection rate"),
    ("X1_slope", "bias", "SUBGROUP X1 contrast slope -- bias"),
    ("X1_slope", "cover95", "SUBGROUP X1 contrast slope -- 95% coverage"),
]


def load() -> pd.DataFrame:
    paths = (sorted(glob.glob(f"{ROOT}/R_code/PT_*_datasets/summaries_*_lin_*.csv"))
             + sorted(glob.glob(f"{ROOT}/Pretrend/summaries_pretrend_*_lin_*.csv"))
             + sorted(glob.glob(f"{ROOT}/CFFE_Wang_Colab/PT_*_datasets/"
                                "summaries_wang_*_lin_*.csv")))
    if not paths:
        raise SystemExit("no PT summary files found -- run the suite first")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True), len(paths)


def panel(pre: pd.DataFrame, eid: str, value: str, degree: int) -> pd.DataFrame | None:
    t = pre[(pre["estimand_id"] == eid) & (pre["linearity_degree"] == degree)]
    if t.empty:
        return None
    p = t.pivot_table(index="setting", columns="method", values=value)
    p = p.reindex([s for s in SCENARIOS if s in p.index])
    p = p[[m for m in METHODS if m in p.columns]]
    if value != "n_reps":
        truth = t.groupby("setting")["mean_true"].max().reindex(p.index)
        p.insert(0, "true", truth)
        return p.round(3)
    return p.astype("Int64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=f"{ROOT}/Pretrend/BENCHMARK_TABLES.txt")
    args = ap.parse_args()

    summ, n_files = load()
    met = compute_metrics(summ)
    met.to_csv(f"{ROOT}/Pretrend/benchmark_metrics_all.csv", index=False)
    pre = met[met["estimand_type"].isin(["PRE", "PRE_SUBC"])].copy()

    lines = ["Pre-trend workstream: DiD-BCF vs the benchmark pre-tests",
             "=" * 88,
             f"pooled {n_files} summary files | methods: "
             f"{', '.join(sorted(summ['method'].unique()))}",
             "reject05 = size where true == 0, detection power otherwise", ""]
    for degree in sorted(pre["linearity_degree"].dropna().unique().astype(int)):
        lines += ["", "=" * 88, f"LINEARITY DEGREE {degree}", "=" * 88]
        for eid, value, label in PANELS:
            p = panel(pre, eid, value, degree)
            if p is None:
                continue
            lines += ["", f"--- {label} ---", p.to_string()]
    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
