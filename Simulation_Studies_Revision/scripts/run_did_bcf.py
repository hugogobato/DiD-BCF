#!/usr/bin/env python3
"""Headless CLI to fit DiD-BCF for one experiment (or all) -- the same engine the
per-DGP notebooks in ``DiD_BCF/`` use.

The per-DGP notebooks are the intended entry point for Colab; this script is for
running on a server / from the command line.

``--spec`` selects the identification specification: ``published`` (the default,
the submitted model) or one of the repairs of
``Results/identification_note.tex``.  A non-default spec is appended to the
output filename, so route comparisons never overwrite the published runs.  To
compare routes systematically, use ``scripts/run_identification_routes.py``,
which also runs the acceptance test.

Examples
--------
    python scripts/run_did_bcf.py --experiment B2_sweep --reps 200 --jobs 4
    python scripts/run_did_bcf.py --experiment B2_sweep --spec structured --jobs 4
    python scripts/run_did_bcf.py --all
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision.did_bcf import DEFAULT_SPEC, SPECS
from did_bcf_revision.runner import run_experiment, RESULTS_DIR
from did_bcf_revision.structured import STRUCTURED_SPECS


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
    ap.add_argument("--pretrend-recenter", action="store_true",
                    help="report tau_hat(i,t) - tau_hat(i,g-1) instead of the "
                         "raw forest.  Reproduces the first revision run only; "
                         "it annihilates the estimate (see report, Sec. 8.1).")
    ap.add_argument("--num-gfr", type=int, default=50)
    ap.add_argument("--num-mcmc", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=5)
    ap.add_argument("--num-chains", type=int, default=3)
    ap.add_argument("--spec", default=DEFAULT_SPEC,
                    choices=list(SPECS) + list(STRUCTURED_SPECS),
                    help="identification specification; see the module docstring "
                         "of did_bcf_revision.did_bcf")
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
                           spec=args.spec)


if __name__ == "__main__":
    main()
