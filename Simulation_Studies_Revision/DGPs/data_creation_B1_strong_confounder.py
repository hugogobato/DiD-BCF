#!/usr/bin/env python3
"""DGP: B1_strong_confounder  (workstream B1, canonical DiD (selection on unobservables)).

large Var(alpha) and strong corr(alpha, treatment)

Writes CSV replications in the **R-benchmark column layout** for every
``linearity_degree`` to
``R_code/B1_strong_confounder_datasets/linearity_degree=<d>/iteration_<rep>.csv`` -- exactly
the files the R estimators in ``R_code/B1_strong_confounder_datasets/`` read.  The DiD-BCF and
OLS notebooks regenerate identical (seeded) data in-memory, so this step is only
needed for the R benchmarks (or any external tool).

Panel: N=200, 4 pre + 4 post periods (override with the engine if needed).

Examples
--------
    python DGPs/data_creation_B1_strong_confounder.py --reps 100 --jobs 8
    python DGPs/data_creation_B1_strong_confounder.py --all-N        # sweep scenarios only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision.config import get_experiment, LINEARITY_DEGREES
from did_bcf_revision.dgps import generate_canonical_did, generate_staggered_did
from did_bcf_revision.exports import to_r_frame

SCENARIO = "B1_strong_confounder"
GEN = {"canonical": generate_canonical_did,
       "staggered": generate_staggered_did}["canonical"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "R_code", SCENARIO + "_datasets")


def _write_one(params, N, d, rep, out_dir):
    df = GEN(seed=int(rep), **{**params, "n_units": int(N), "linearity_degree": int(d)})
    to_r_frame(df).to_csv(os.path.join(out_dir, "iteration_%d.csv" % rep), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=None, help="default: scenario reps")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--all-N", action="store_true",
                    help="also write the full N sweep under N=<N>/ (sweep scenarios)")
    args = ap.parse_args()

    exp = get_experiment(SCENARIO)
    reps = args.reps if args.reps is not None else exp.reps
    base_N = exp.n_values[0]

    tasks = []
    for d in LINEARITY_DEGREES:
        base_dir = os.path.join(OUT_ROOT, "linearity_degree=%d" % d)
        os.makedirs(base_dir, exist_ok=True)
        for rep in range(reps):
            tasks.append((exp.dgp_params, base_N, d, rep, base_dir))
        if args.all_N and len(exp.n_values) > 1:
            for N in exp.n_values:
                nd = os.path.join(OUT_ROOT, "N=%d" % N, "linearity_degree=%d" % d)
                os.makedirs(nd, exist_ok=True)
                for rep in range(reps):
                    tasks.append((exp.dgp_params, N, d, rep, nd))

    print("[%s] writing %d CSVs -> %s (jobs=%d)" % (SCENARIO, len(tasks), OUT_ROOT, args.jobs))
    if args.jobs and args.jobs > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(_write_one)(*t) for t in tasks)
    else:
        for t in tasks:
            _write_one(*t)
    print("Done.")


if __name__ == "__main__":
    main()
