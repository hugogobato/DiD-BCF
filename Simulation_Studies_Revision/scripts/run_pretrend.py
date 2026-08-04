#!/usr/bin/env python3
"""Workstream F4: the pre-trend / conditional-parallel-trends diagnostic.

Fits the *unconstrained* specification (``Z = 1[G_i != inf]``, active in every
period) as a diagnostic model alongside the constrained one used for estimation,
and reports ``Delta(k) = E[tau(X, k) - tau(X, -1)]`` for ``k < -1`` -- zero under
conditional parallel trends, ``differential_slope * (k + 1)`` under a violation.
See the module docstring of ``did_bcf_revision/pretrend.py`` for why the
constrained fit cannot produce this by post-processing.

The ``PT_*`` scenarios in ``config.py`` give the diagnostic's operating
characteristics:

``PT_hold``            conditional PT holds -> false-positive rate (size).
``PT_conditional``     conditional PT holds, unconditional PT fails -> the
                       covariate-conditional diagnostic should stay quiet while
                       the marginal TWFE event study flags.
``PT_violation_g*``    differential group slope 0.05 ... 0.40 -> power curve.
``PT_violation_a*``    the violation runs through the unobserved confounder.

Each replication also runs the TWFE event-study placebo (free) as the
standard-practice comparator; ``--with-att`` adds the ordinary constrained
DiD-BCF fit so the ATT bias the diagnostic is warning about is measured on the
same replications (doubles the MCMC cost).

Examples
--------
    python scripts/run_pretrend.py --experiment PT_hold --reps 200 --jobs 2
    python scripts/run_pretrend.py --all --linearity-degree 1 --with-att --jobs 2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision.pretrend_runner import run_experiment, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="PT_* experiment name")
    ap.add_argument("--all", action="store_true",
                    help="run every workstream-PT experiment")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--linearity-degree", type=int, default=1,
                    help="single degree (default 1); pass -1 for all of "
                         f"{list(cfg.LINEARITY_DEGREES)}")
    ap.add_argument("--rfx", default="unit", choices=["unit", "none"],
                    help="unit-level random intercepts absorb the level gap that "
                         "conditional PT permits (default), or omit them")
    ap.add_argument("--with-att", action="store_true",
                    help="also fit the constrained model, to measure the ATT "
                         "bias caused by the violation (doubles the cost)")
    ap.add_argument("--rep-start", type=int, default=0,
                    help="first replication (inclusive); with --rep-end, splits "
                         "a long run across Colab sessions")
    ap.add_argument("--rep-end", type=int, default=None,
                    help="last replication (exclusive); default --reps")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--num-gfr", type=int, default=50)
    ap.add_argument("--num-mcmc", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=5)
    ap.add_argument("--num-chains", type=int, default=3)
    args = ap.parse_args()

    bcf_params = dict(num_gfr=args.num_gfr, num_mcmc=args.num_mcmc,
                      keep_every=args.keep_every, num_chains=args.num_chains)

    if args.all:
        experiments = [e for e in cfg.all_experiments(reps=args.reps)
                       if e.workstream == "PT"]
    elif args.experiment:
        experiments = [cfg.get_experiment(args.experiment, reps=args.reps)]
    else:
        ap.error("pass --experiment <name> or --all")
    # The PT scenarios carry a 200-replication floor (detection rates need it),
    # which would silently override a smaller --reps used for a smoke test.
    for e in experiments:
        e.reps = int(args.reps)

    degrees = (list(cfg.LINEARITY_DEGREES) if args.linearity_degree == -1
               else [args.linearity_degree])
    for exp in experiments:
        for d in degrees:
            run_experiment(exp, bcf_params=bcf_params, rfx=args.rfx,
                           with_att=args.with_att, jobs=args.jobs,
                           linearity_degree=d, out_dir=args.out,
                           rep_start=args.rep_start, rep_end=args.rep_end)


if __name__ == "__main__":
    main()
