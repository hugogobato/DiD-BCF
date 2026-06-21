#!/usr/bin/env python3
"""Workstream D: Goodman-Bacon decomposition + TWFE-vs-truth under staggered,
dynamic, cohort-varying effects.

This is pure Python (no BCF), so it runs locally and is parallelised over
replications.  It demonstrates the "bad comparisons" (already-treated units used
as controls) and quantifies how TWFE degrades as their weight grows, providing
the foil for the DiD-BCF results from ``run_did_bcf.py`` (experiments
``D_staggered`` / ``D_contamination``).

Outputs (in ``Results/``):
* ``goodman_bacon_components_<setting>.csv`` -- replication-averaged 2x2
  components: weight and DD estimate per comparison type;
* ``goodman_bacon_summary.csv`` -- per setting: TWFE estimate, true ATT, TWFE
  bias, and the **weight on already-treated comparisons**;
* ``twfe_event_study_<setting>.csv`` -- TWFE event-study coefficients vs truth.

A ``--ramp-sweep`` mode sweeps the dynamic-effect strength to trace TWFE bias and
the already-treated weight as contamination increases.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision.dgps import generate_staggered_did, true_estimands
from did_bcf_revision.goodman_bacon import bacon_summary
from did_bcf_revision.twfe import twfe_event_study

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")


def _one_rep(dgp_params: dict, N: int, rep: int) -> dict:
    df = generate_staggered_did(seed=int(rep), **{**dgp_params, "n_units": int(N)})
    summ = bacon_summary(df)
    truth = true_estimands(df)
    att_true = float(truth.loc[truth["estimand_type"] == "ATT", "true"].iloc[0])
    comp = summ["components"][["type", "weight", "dd"]].copy()
    comp["rep"] = rep
    es = twfe_event_study(df)
    es = es.merge(
        truth[truth["estimand_type"] == "ES"][["k", "true"]], on="k", how="left")
    es["rep"] = rep
    return {
        "scalar": {
            "rep": rep, "N": N,
            "twfe": summ["twfe"], "att_true": att_true,
            "twfe_bias": summ["twfe"] - att_true,
            "w_treated_vs_untreated": summ["weight_treated_vs_untreated"],
            "w_earlier_vs_later": summ["weight_earlier_vs_later"],
            "w_already_treated": summ["weight_already_treated"],
        },
        "components": comp,
        "event_study": es,
    }


def _run(dgp_params: dict, N: int, reps: int, jobs: int) -> dict:
    if jobs and jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_one_rep)(dgp_params, N, r) for r in range(reps))
    else:
        try:
            from tqdm import tqdm
            it = tqdm(range(reps), desc="GB", unit="rep")
        except Exception:
            it = range(reps)
        res = [_one_rep(dgp_params, N, r) for r in it]

    scalars = pd.DataFrame([r["scalar"] for r in res])
    components = (pd.concat([r["components"] for r in res], ignore_index=True)
                 .groupby("type", as_index=False)[["weight", "dd"]].mean())
    components["contribution"] = components["weight"] * components["dd"]
    event = (pd.concat([r["event_study"] for r in res], ignore_index=True)
             .groupby("k", as_index=False)
             .agg(coef=("coef", "mean"), true=("true", "mean")))
    event["twfe_bias"] = event["coef"] - event["true"]
    return {"scalars": scalars, "components": components, "event": event}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="D_staggered",
                    help="staggered experiment name from config")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--ramp-sweep", action="store_true",
                    help="sweep dynamic_ramp to trace TWFE bias vs contamination")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    exp = cfg.get_experiment(args.experiment)
    if exp.dgp != "staggered":
        ap.error(f"{args.experiment} is not a staggered experiment")
    N = exp.n_values[0]

    if args.ramp_sweep:
        rows = []
        for ramp in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            params = {**exp.dgp_params, "dynamic_ramp": ramp}
            out = _run(params, N, args.reps, args.jobs)
            s = out["scalars"][["twfe", "att_true", "twfe_bias",
                                 "w_already_treated"]].mean()
            rows.append({"dynamic_ramp": ramp, **s.to_dict()})
            print(f"ramp={ramp:.1f}: TWFE bias={s['twfe_bias']:+.3f}, "
                  f"already-treated weight={s['w_already_treated']:.3f}")
        path = os.path.join(args.out, "goodman_bacon_ramp_sweep.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Wrote {path}")
        return

    out = _run(exp.dgp_params, N, args.reps, args.jobs)
    s = out["scalars"].mean(numeric_only=True)
    summary = pd.DataFrame([{
        "setting": exp.name, "N": N, "reps": args.reps,
        "twfe": s["twfe"], "att_true": s["att_true"], "twfe_bias": s["twfe_bias"],
        "weight_treated_vs_untreated": s["w_treated_vs_untreated"],
        "weight_earlier_vs_later": s["w_earlier_vs_later"],
        "weight_already_treated": s["w_already_treated"],
    }])

    out["components"].to_csv(
        os.path.join(args.out, f"goodman_bacon_components_{exp.name}.csv"), index=False)
    out["event"].to_csv(
        os.path.join(args.out, f"twfe_event_study_{exp.name}.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "goodman_bacon_summary.csv"), index=False)

    print("\nComponent weights (replication mean):")
    print(out["components"].to_string(index=False))
    print("\nHeadline:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
