#!/usr/bin/env python3
"""Headless CLI to fit DiD-BCF for one experiment (or all) -- the same engine the
per-DGP notebooks in ``DiD_BCF/`` use.

The per-DGP notebooks are the intended entry point for Colab; this script is for
running on a server / from the command line.

``--spec`` selects the estimator, and only the two correct ones are offered:
``structured`` (the default, and what the whole simulation grid was run under)
and ``structured_rfx_unit`` (the same model plus unit-level random intercepts).
The specifications with the identification failure documented in
``Results/identification_note.tex`` -- ``published`` and the partial repairs --
are deliberately not selectable here; only
``scripts/run_identification_routes.py`` can run them, because demonstrating
their failure is that script's purpose.

A non-default spec is appended to the output filename, so the two never
overwrite each other.

Examples
--------
    python scripts/run_did_bcf.py --experiment B2_sweep --reps 200 --jobs 4
    python scripts/run_did_bcf.py --experiment B2_sweep --spec structured_rfx_unit
    python scripts/run_did_bcf.py --all
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision.did_bcf import DEFAULT_SPEC
from did_bcf_revision.runner import run_experiment, RESULTS_DIR
from did_bcf_revision.structured import PRODUCTION_SPECS


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
    ap.add_argument("--rep-start", type=int, default=0,
                    help="first replication (inclusive); with --rep-end, splits "
                         "a long run across Colab sessions")
    ap.add_argument("--rep-end", type=int, default=None,
                    help="last replication (exclusive); default --reps")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--propensity", default="logit", choices=["logit", "rf"])
    ap.add_argument("--n-splits", type=int, default=2)
    ap.add_argument("--pretrend-recenter", action="store_true",
                    help="report tau_hat(i,t) - tau_hat(i,g-1) instead of the "
                         "raw forest.  Reproduces the first revision run only; "
                         "it annihilates the estimate (see report, Sec. 8.1).")
    ap.add_argument("--num-gfr", type=int, default=50)
    ap.add_argument("--num-mcmc", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=5)
    ap.add_argument("--num-chains", type=int, default=3)
    ap.add_argument("--spec", default=DEFAULT_SPEC,
                    choices=list(PRODUCTION_SPECS),
                    help="estimator: structured (default) or structured_rfx_unit. "
                         "The specifications with the known identification "
                         "failure are not selectable; see "
                         "scripts/run_identification_routes.py")
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
                           pretrend_recenter=args.pretrend_recenter,
                           jobs=args.jobs, linearity_degree=d, out_dir=args.out,
                           spec=args.spec, rep_start=args.rep_start,
                           rep_end=args.rep_end)


if __name__ == "__main__":
    main()
