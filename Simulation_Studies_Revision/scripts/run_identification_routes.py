#!/usr/bin/env python3
"""The acceptance test for the identification repairs of ``identification_note.tex``.

The note's own criterion, which it recommends becoming a permanent part of the
suite: *on* ``B2_sweep``\\ *, the bias of both plain and corrected DiD-BCF must
decrease in* ``N``.  An estimator whose bias increases in ``N`` is not noisy, it
is inconsistent, and that single check would have caught the problem before
publication.

This script runs that test for every repair at once.  For each specification and
each panel size it reports the bias of the plain and the posterior-corrected
ATT, the 95% coverage, and the *retention* ``tau_hat / ATT`` on treated
post-treatment rows -- the direct measurement of how much of the true effect the
treatment forest still holds, which is Table 1 of the note.

Specifications (see ``did_bcf_revision.did_bcf.SPECS`` and
``did_bcf_revision.structured``)::

    published              the submitted specification, as the revision runs it
    published_constant_pi  the same, with the original notebooks' pi = 0.5
    rfx                    route 1: group out of mu, group random intercept
    rfx_unit               route 1 with unit-level random intercepts
    propensity             route 2: group out of mu, P(ever treated | X) in
    rfx_propensity         routes 1 and 2 together
    structured             route 3: mu(g,t,x) = a(g,x) + b(t,x)
    structured_rfx_unit    route 3 plus unit-level random intercepts

Examples
--------
    # quick look: 3 specs, few reps, small panels
    python scripts/run_identification_routes.py --reps 5 --n 200 400 \\
        --specs published rfx structured --jobs 4

    # the full acceptance test
    python scripts/run_identification_routes.py --reps 50 --jobs 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision import config as cfg
from did_bcf_revision.did_bcf import SPECS, plain_estimands
from did_bcf_revision.dgps import generate_canonical_did, generate_staggered_did, true_estimands
from did_bcf_revision.posterior_correction import corrected_estimands
from did_bcf_revision.runner import RESULTS_DIR, fit_any
from did_bcf_revision.structured import STRUCTURED_SPECS, is_structured

ALL_SPECS = list(SPECS) + list(STRUCTURED_SPECS)
DEFAULT_SPECS = ["published", "published_constant_pi", "rfx", "rfx_unit",
                 "propensity", "rfx_propensity", "structured",
                 "structured_rfx_unit"]
# Lighter than the production sampler: the acceptance test needs many fits
# across specifications and panel sizes, and it measures a bias that is a
# property of the model rather than of the number of draws.
ACCEPTANCE_BCF_PARAMS = dict(num_gfr=25, num_mcmc=200, keep_every=2, num_chains=2)
_GENERATORS = {"canonical": generate_canonical_did, "staggered": generate_staggered_did}


def one_rep(spec: str, dgp: str, params: dict, N: int, rep: int,
            bcf_params: dict, prop_method: str, n_splits: int,
            check_identification: bool = False) -> pd.DataFrame:
    """Fit one replication under one specification; return two tidy rows."""
    df = _GENERATORS[dgp](seed=int(rep), **{**params, "n_units": int(N)})
    truth = true_estimands(df)
    att_true = float(truth.query("estimand_type == 'ATT'")["true"].iloc[0])

    t0 = time.time()
    fit = fit_any(df, bcf_params=bcf_params, seed=int(rep), spec=spec)
    fit_seconds = time.time() - t0

    # Retention: the share of the true effect the treatment forest still holds,
    # measured directly on treated post-treatment rows (Table 1 of the note).
    treated_post = fit.df["D"].to_numpy() == 1
    retention = float(fit.tau_draws[treated_post].mean()) / att_true

    id_gap = np.nan
    if check_identification and hasattr(fit, "structured_model"):
        id_gap = fit.structured_model.check_identification()["max_abs_tau_minus_contrast"]

    plain = plain_estimands(fit)
    corrected = corrected_estimands(fit, propensity_method=prop_method,
                                    n_splits=n_splits, seed=int(rep))
    surf = plain.query("estimand_type == 'CATT'")
    surf_rmse = float(surf["surf_rmse"].iloc[0]) if len(surf) else np.nan

    rows = []
    for method, table in (("plain", plain), ("corrected", corrected)):
        att = table.query("estimand_type == 'ATT'")
        if not len(att):
            continue
        att = att.iloc[0]
        rows.append({
            "spec": spec, "dgp": dgp, "N": int(N), "rep": int(rep),
            "method": method, "true_att": att_true,
            "est": float(att["post_mean"]), "post_sd": float(att["sd"]),
            "q025": float(att["q025"]), "q975": float(att["q975"]),
            "covered95": int(att["q025"] <= att_true <= att["q975"]),
            "retention": retention, "surf_rmse": surf_rmse,
            "id_gap": id_gap, "fit_seconds": fit_seconds,
        })
    return pd.DataFrame.from_records(rows)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """Bias, RMSE, coverage and retention per (spec, N, method)."""
    def agg(g: pd.DataFrame) -> pd.Series:
        err = g["est"] - g["true_att"]
        return pd.Series({
            "reps": len(g),
            "bias": err.mean(),
            "abs_bias": abs(err.mean()),
            "emp_sd": g["est"].std(ddof=1),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "coverage95": g["covered95"].mean(),
            "retention": g["retention"].mean(),
            "surf_rmse": g["surf_rmse"].mean(),
            "mean_fit_s": g["fit_seconds"].mean(),
        })
    out = raw.groupby(["spec", "N", "method"], as_index=False).apply(
        agg, include_groups=False)
    return out.sort_values(["spec", "method", "N"]).reset_index(drop=True)


def acceptance(summary: pd.DataFrame) -> pd.DataFrame:
    """Does |bias| decrease in N?  One verdict per (spec, method).

    Undefined -- reported as ``None`` rather than ``False`` -- when the run
    covers a single panel size, since the test is a statement about the
    sequence.
    """
    records = []
    for (spec, method), g in summary.groupby(["spec", "method"]):
        g = g.sort_values("N")
        ab = g["abs_bias"].to_numpy()
        single = len(ab) < 2
        records.append({
            "spec": spec, "method": method, "n_panel_sizes": len(ab),
            "abs_bias_first": ab[0], "abs_bias_last": ab[-1],
            "monotone_decreasing": None if single else bool(np.all(np.diff(ab) <= 1e-12)),
            "passes": None if single else bool(ab[-1] < ab[0]),
        })
    return pd.DataFrame.from_records(records).sort_values(["method", "spec"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", nargs="+", default=DEFAULT_SPECS,
                    choices=ALL_SPECS, metavar="SPEC")
    ap.add_argument("--experiment", default="B2_sweep",
                    help="scenario supplying the DGP parameters")
    ap.add_argument("--n", nargs="+", type=int, default=list(cfg.N_SWEEP))
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--linearity-degree", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--propensity", default="logit", choices=["logit", "rf"])
    ap.add_argument("--n-splits", type=int, default=2)
    ap.add_argument("--num-gfr", type=int, default=ACCEPTANCE_BCF_PARAMS["num_gfr"])
    ap.add_argument("--num-mcmc", type=int, default=ACCEPTANCE_BCF_PARAMS["num_mcmc"])
    ap.add_argument("--keep-every", type=int, default=ACCEPTANCE_BCF_PARAMS["keep_every"])
    ap.add_argument("--num-chains", type=int, default=ACCEPTANCE_BCF_PARAMS["num_chains"])
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "identification"))
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    args = ap.parse_args()

    exp = cfg.get_experiment(args.experiment)
    params = dict(exp.dgp_params)
    params["linearity_degree"] = int(args.linearity_degree)
    bcf_params = dict(num_gfr=args.num_gfr, num_mcmc=args.num_mcmc,
                      keep_every=args.keep_every, num_chains=args.num_chains)

    tasks = [(spec, N, rep)
             for spec in args.specs for N in args.n for rep in range(args.reps)]
    print(f"[identification] {args.experiment} (d={args.linearity_degree}) | "
          f"{len(args.specs)} specs x {len(args.n)} panel sizes x {args.reps} reps "
          f"= {len(tasks)} fits | jobs={args.jobs}")
    print(f"[identification] sampler: {bcf_params}")

    def _one(spec, N, rep):
        return one_rep(spec, exp.dgp, params, N, rep, bcf_params,
                       args.propensity, args.n_splits,
                       check_identification=(rep == 0 and is_structured(spec)))

    if args.jobs and args.jobs > 1:
        from joblib import Parallel, delayed
        frames = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(_one)(s, N, r) for s, N, r in tasks)
    else:
        frames = []
        for s, N, r in tasks:
            frames.append(_one(s, N, r))
            print(f"  {s} N={N} rep={r} done", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    summary = summarise(raw)
    verdict = acceptance(summary)

    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    stem = f"{args.experiment}_lin_{args.linearity_degree}{tag}"
    raw.to_csv(os.path.join(args.out, f"routes_raw_{stem}.csv"), index=False)
    summary.to_csv(os.path.join(args.out, f"routes_summary_{stem}.csv"), index=False)
    verdict.to_csv(os.path.join(args.out, f"routes_acceptance_{stem}.csv"), index=False)

    pd.set_option("display.width", 200, "display.max_rows", 400)
    print("\n=== bias / coverage / retention by (spec, N) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("\n=== acceptance test: does |bias| fall in N? ===")
    print(verdict.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    if raw["id_gap"].notna().any():
        print("\n=== structured-model identification check "
              "(max |tau - DiD contrast of the fit|) ===")
        print(raw.loc[raw["id_gap"].notna(), ["spec", "N", "id_gap"]]
              .drop_duplicates().to_string(index=False))
    print(f"\nwrote {args.out}/routes_*_{stem}.csv")


if __name__ == "__main__":
    main()
