"""Monte-Carlo driver: fit DiD-BCF over the replications of one experiment and
write the per-replication summaries (plain + posterior-corrected).

This is the shared engine called by every per-DGP notebook in ``DiD_BCF/`` and
by the headless CLI ``scripts/run_did_bcf.py`` -- so the fitting logic lives in
exactly one place.
"""

from __future__ import annotations

import os

import pandas as pd

from .dgps import generate_canonical_did, generate_staggered_did, true_estimands
from .did_bcf import fit_did_bcf, plain_estimands
from .posterior_correction import corrected_estimands
from . import config as cfg

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
_GENERATORS = {"canonical": generate_canonical_did, "staggered": generate_staggered_did}


def process_rep(dgp: str, dgp_params: dict, N: int, rep: int, setting: str,
                bcf_params: dict | None = None, prop_method: str = "logit",
                n_splits: int = 2, pretrend_recenter: bool = True) -> pd.DataFrame:
    """Run one replication; return tidy summary rows for both methods."""
    gen = _GENERATORS[dgp]
    df = gen(seed=int(rep), **{**dgp_params, "n_units": int(N)})

    fit = fit_did_bcf(df, bcf_params=bcf_params, seed=int(rep))
    plain = plain_estimands(fit, pretrend_recenter=pretrend_recenter)
    corrected = corrected_estimands(fit, propensity_method=prop_method,
                                    n_splits=n_splits, seed=int(rep))
    truth = true_estimands(df)[["estimand_type", "estimand_id", "true"]]

    out = pd.concat([plain, corrected], ignore_index=True)
    out = out.merge(truth, on=["estimand_type", "estimand_id"], how="left")
    out.insert(0, "dgp", dgp)
    out.insert(1, "setting", setting)
    out.insert(2, "linearity_degree", int(dgp_params.get("linearity_degree", 1)))
    out.insert(3, "N", int(N))
    out.insert(4, "rep", int(rep))
    return out


def run_experiment(exp: "cfg.Experiment", bcf_params: dict | None = None,
                   prop_method: str = "logit", n_splits: int = 2,
                   pretrend_recenter: bool = True, jobs: int = 1,
                   linearity_degree: int | None = None,
                   out_dir: str | None = None, save: bool = True) -> pd.DataFrame:
    """Run every (N, rep) of ``exp`` and (optionally) save the summaries.

    Returns the concatenated summary DataFrame (plain + corrected DiD-BCF).
    When ``linearity_degree`` is given it overrides the scenario default and is
    appended to the output filename; writes
    ``<out_dir>/summaries_<exp.name>[_lin_<d>].csv`` when ``save`` is true.
    """
    out_dir = out_dir or RESULTS_DIR
    params = dict(exp.dgp_params)
    suffix = ""
    if linearity_degree is not None:
        params["linearity_degree"] = int(linearity_degree)
        suffix = f"_lin_{int(linearity_degree)}"

    tasks = [(N, rep) for N in exp.n_values for rep in range(exp.reps)]
    print(f"[{exp.name}{suffix}] {exp.dgp} DGP | N={exp.n_values} | reps={exp.reps} "
          f"| {len(tasks)} fits | jobs={jobs}")

    def _one(N, rep):
        return process_rep(exp.dgp, params, N, rep, exp.name,
                           bcf_params, prop_method, n_splits, pretrend_recenter)

    if jobs and jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_one)(N, rep) for N, rep in tasks)
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(tasks, desc=exp.name, unit="fit")
        except Exception:
            iterator = tasks
        rows = [_one(N, rep) for N, rep in iterator]

    result = pd.concat(rows, ignore_index=True)
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"summaries_{exp.name}{suffix}.csv")
        result.to_csv(path, index=False)
        print(f"[{exp.name}{suffix}] wrote {len(result)} rows -> {path}")
    return result


def run_named(name: str, reps: int | None = None, **kwargs) -> pd.DataFrame:
    """Convenience for notebooks: run a single experiment by name.

    ``run_named("B1_baseline", reps=200, jobs=2)``.  ``reps`` overrides the
    experiment default; remaining kwargs are forwarded to :func:`run_experiment`.
    """
    exp = cfg.get_experiment(name)
    if reps is not None:
        exp.reps = int(reps)
    return run_experiment(exp, **kwargs)
