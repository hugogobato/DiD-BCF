#!/usr/bin/env python3
"""Headless CLI for the TWFE / OLS benchmark -- the same engine the per-DGP
notebooks in ``TWFE/`` use.  Pure numpy/pandas (no stochtree), so it runs on a
laptop; parallelise replications with ``--jobs``.

Examples
--------
    python scripts/run_twfe.py --experiment D_staggered --reps 200 --jobs 8
    python scripts/run_twfe.py --all
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision.twfe_runner import run_twfe_experiment, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="scenario name (see config.all_experiments)")
    ap.add_argument("--all", action="store_true", help="run every scenario")
    ap.add_argument("--reps", type=int, default=cfg.DEFAULT_REPS)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (reps)")
    ap.add_argument("--linearity-degree", type=int, default=None,
                    help="single linearity degree; default runs all of "
                         f"{list(cfg.LINEARITY_DEGREES)}")
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args()

    if args.all:
        experiments = cfg.all_experiments(reps=args.reps)
    elif args.experiment:
        experiments = [cfg.get_experiment(args.experiment, reps=args.reps)]
    else:
        ap.error("pass --experiment <name> or --all")

    degrees = ([args.linearity_degree] if args.linearity_degree is not None
               else list(cfg.LINEARITY_DEGREES))
    for exp in experiments:
        for d in degrees:
            run_twfe_experiment(exp, linearity_degree=d, jobs=args.jobs,
                                out_dir=args.out)


if __name__ == "__main__":
    main()
