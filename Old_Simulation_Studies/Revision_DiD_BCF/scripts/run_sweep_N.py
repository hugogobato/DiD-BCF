#!/usr/bin/env python3
"""Run every benchmark estimator at one sample size of the B2 sweep, locally.

``run_r_benchmarks.py`` runs the R estimators at a scenario's **base** panel
(``config.BASE_N``) only -- the ATT scripts glob ``linearity_degree=*`` relative
to their working directory, so a sweep size is reached by running them with
``R_code/<scen>_datasets/N=<N>/`` as the CWD instead.  That is exactly what the
per-N Colab notebooks in ``DoubleML_Colab/`` and ``CFFE_Wang_Colab/`` do, one N
per notebook, because DoubleML and grf are slow at the top of the sweep.

At the *bottom* of the sweep (N = 50, 100) they are not slow, so there is no
reason to spend a Colab session on them.  This script does the whole per-N
column locally: it generates the seeded panels, runs all five estimators
concurrently as single-threaded processes, and files the results exactly where
the Colab route would have left them, so ``Results/analysis/aggregate_all.py``
picks them up without knowing which route produced them:

===============  ==========================================================
did_dr, did2s,   ``R_code/<scen>_datasets/N=<N>/summaries_<m>_<scen>_lin_<d>.csv``
synthdid
doubleml         ``Results/_staging/DoubleML_Colab/<scen>_N<N>/`` (N-tagged
                 filenames) + ``DoubleML_Colab/DoubleML_<scen>_N<N>_results.zip``
wang (grf-DiD)   ``Results/_staging/CFFE_Wang_Colab/<scen>_N<N>/`` +
                 ``CFFE_Wang_Colab/Wang_<scen>_results_N<N>.zip``
===============  ==========================================================

The panels are seeded by replication index, identically to the DiD-BCF notebooks
(``seed=rep``), so the comparison stays apples-to-apples.

Examples
--------
    python scripts/run_sweep_N.py --N 50 100
    python scripts/run_sweep_N.py --N 50 --scripts did_dr_new synthdid --jobs 4
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from did_bcf_revision.config import get_experiment, degrees_for  # noqa: E402
from did_bcf_revision.dgps import (generate_canonical_did,  # noqa: E402
                                   generate_staggered_did)
from did_bcf_revision.exports import to_r_frame  # noqa: E402

R_CODE = os.path.join(ROOT, "R_code")
STAGING = os.path.join(ROOT, "Results", "_staging")
GEN = {"canonical": generate_canonical_did, "staggered": generate_staggered_did}

SCENARIOS = ("B2_sweep", "B2_sweep_serial")
SCRIPTS = ("did_dr_new", "did2s", "synthdid", "DoubleML_did", "wang_grf")
# Everything runs single-threaded so the pool size is the real core count.
# att_gt already takes cores=1; mlr3learners defaults ranger to num.threads=1;
# grf is the only one that grabs every core, hence GRF_THREADS.
ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "GRF_THREADS": "1"}

# Where a script's outputs belong once it has finished, and under which name.
# `None` means "leave it in the N= folder" (the three fast R estimators).
STAGE = {
    "DoubleML_did": ("DoubleML_Colab", "doubleml", "output_DoubleML_did.txt"),
    "wang_grf": ("CFFE_Wang_Colab", "wang", "output_wang.txt"),
}


def generate_panels(scen: str, N: int, reps: int, jobs: int) -> str:
    """Write the seeded replications for (scen, N); return the N= folder."""
    exp = get_experiment(scen)
    gen = GEN[exp.dgp]
    out = os.path.join(R_CODE, f"{scen}_datasets", f"N={N}")
    tasks = []
    for d in degrees_for(exp.workstream):
        lin_dir = os.path.join(out, f"linearity_degree={d}")
        os.makedirs(lin_dir, exist_ok=True)
        for rep in range(reps):
            tasks.append((d, rep, lin_dir))

    def _one(d, rep, lin_dir):
        df = gen(seed=int(rep), **{**exp.dgp_params, "n_units": int(N),
                                   "linearity_degree": int(d)})
        to_r_frame(df).to_csv(os.path.join(lin_dir, f"iteration_{rep}.csv"),
                              index=False)

    if jobs > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(*t) for t in tasks)
    else:
        for t in tasks:
            _one(*t)
    print(f"[{scen} N={N}] {len(tasks)} panels -> {os.path.relpath(out, ROOT)}")
    return out


def run_one(scen: str, N: int, script: str, folder: str, reps: int,
            trees: int) -> tuple:
    """Run one estimator with the N= folder as CWD (see module docstring)."""
    exp = get_experiment(scen)
    if script == "wang_grf":
        # Lives with the notebooks, not per-scenario; takes its target as args.
        cmd = ["Rscript", os.path.join(ROOT, "CFFE_Wang_Colab", "wang_grf.R"),
               exp.dgp, scen, str(trees), str(reps)]
    else:
        cmd = ["Rscript", os.path.join("..", f"{script}.R")]
        if not os.path.exists(os.path.join(R_CODE, f"{scen}_datasets", f"{script}.R")):
            return (scen, N, script, "missing", 0.0, "")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=folder, capture_output=True, text=True,
                         env={**os.environ, **ENV})
    dt = time.time() - t0
    status = "ok" if res.returncode == 0 else f"exit {res.returncode}"
    tail = (res.stderr or res.stdout or "")[-1200:]
    print(f"[{'DONE' if res.returncode == 0 else 'FAIL'}] {scen} N={N} "
          f"{script} ({dt/60:.1f} min){'' if res.returncode == 0 else ': ' + tail}")
    return (scen, N, script, status, dt, tail)


def stage(scen: str, N: int, script: str, folder: str) -> None:
    """Move a Colab-route estimator's outputs to staging and zip them."""
    family, method, logname = STAGE[script]
    dest = os.path.join(STAGING, family, f"{scen}_N{N}")
    os.makedirs(dest, exist_ok=True)
    moved = []
    for src in sorted(glob.glob(os.path.join(
            folder, f"summaries_{method}_{scen}_lin_*.csv"))):
        base = os.path.basename(src)
        # DoubleML tags the filename with N (the CSVs already carry an N column;
        # the tag is what keeps the sweep's five archives apart on download).
        if script == "DoubleML_did":
            base = base.replace(f"_{scen}_lin_", f"_{scen}_N{N}_lin_")
        shutil.move(src, os.path.join(dest, base))
        moved.append(base)
    log = os.path.join(folder, logname)
    if os.path.exists(log):
        shutil.move(log, os.path.join(dest, logname))
        moved.append(logname)
    if not moved:
        print(f"  !! {scen} N={N} {script}: nothing to stage")
        return

    zname = (f"DoubleML_{scen}_N{N}_results.zip" if script == "DoubleML_did"
             else f"Wang_{scen}_results_N{N}.zip")
    zpath = os.path.join(ROOT, family, zname)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for base in moved:
            zf.write(os.path.join(dest, base), base)
    print(f"  -> {os.path.relpath(dest, ROOT)}  ({len(moved)} files) "
          f"+ {os.path.relpath(zpath, ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                    choices=list(SCENARIOS))
    ap.add_argument("--scripts", nargs="+", default=list(SCRIPTS),
                    choices=list(SCRIPTS))
    ap.add_argument("--reps", type=int, default=None, help="default: scenario reps")
    ap.add_argument("--trees", type=int, default=2000,
                    help="grf::causal_forest trees for wang_grf.R")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                    help="concurrent single-threaded R processes")
    ap.add_argument("--skip-data", action="store_true",
                    help="reuse the panels already on disk")
    args = ap.parse_args()

    folders = {}
    for scen in args.scenarios:
        reps = args.reps or get_experiment(scen).reps
        for N in args.N:
            f = os.path.join(R_CODE, f"{scen}_datasets", f"N={N}")
            folders[(scen, N)] = (generate_panels(scen, N, reps, args.jobs)
                                  if not args.skip_data else f)

    reps_of = {s: args.reps or get_experiment(s).reps for s in args.scenarios}
    tasks = [(scen, N, sc) for scen in args.scenarios for N in args.N
             for sc in args.scripts]
    print(f"\n{len(tasks)} estimator runs, {args.jobs} at a time "
          f"(single-threaded each)\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, scen, N, sc, folders[(scen, N)],
                          reps_of[scen], args.trees): (scen, N, sc)
                for scen, N, sc in tasks}
        for fut in as_completed(futs):
            results.append(fut.result())

    print("\nStaging Colab-route outputs...")
    for scen, N, script, status, _, _ in results:
        if status == "ok" and script in STAGE:
            stage(scen, N, script, folders[(scen, N)])

    print("\n=== summary ===")
    for scen, N, script, status, dt, _ in sorted(results):
        print(f"  {scen:16s} N={N:<5d} {script:14s} {status:8s} {dt/60:6.1f} min")
    bad = [r for r in results if r[3] != "ok"]
    if bad:
        sys.exit(f"{len(bad)} run(s) failed")


if __name__ == "__main__":
    main()
