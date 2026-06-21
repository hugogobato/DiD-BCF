#!/usr/bin/env python3
"""Headless CLI to fit DiD-BCF for one experiment (or all) -- the same engine the
per-DGP notebooks in ``DiD_BCF/`` use.

The per-DGP notebooks are the intended entry point for Colab; this script is for
running on a server / from the command line.

Examples
--------
    python scripts/run_did_bcf.py --experiment B2_sweep --reps 200 --jobs 4
    python scripts/run_did_bcf.py --all
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision.runner import run_experiment, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="experiment name (see config.all_experiments)")
    ap.add_argument("--all", action="store_true", help="run every experiment")
    ap.add_argument("--reps", type=int, default=cfg.DEFAULT_REPS)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (reps)")
    ap.add_argument("--linearity-degree", type=int, default=None,
                    help="single linearity degree; default runs all of "
                         f"{list(cfg.LINEARITY_DEGREES)}")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--propensity", default="logit", choices=["logit", "rf"])
    ap.add_argument("--n-splits", type=int, default=2)
    ap.add_argument("--no-pretrend-recenter", action="store_true")
    ap.add_argument("--num-gfr", type=int, default=50)
    ap.add_argument("--num-mcmc", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=5)
    ap.add_argument("--num-chains", type=int, default=3)
    args = ap.parse_args()

    bcf_params = dict(num_gfr=args.num_gfr, num_mcmc=args.num_mcmc,
                      keep_every=args.keep_every, num_chains=args.num_chains)

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
            run_experiment(exp, bcf_params=bcf_params, prop_method=args.propensity,
                           n_splits=args.n_splits,
                           pretrend_recenter=not args.no_pretrend_recenter,
                           jobs=args.jobs, linearity_degree=d, out_dir=args.out)


if __name__ == "__main__":
    main()
