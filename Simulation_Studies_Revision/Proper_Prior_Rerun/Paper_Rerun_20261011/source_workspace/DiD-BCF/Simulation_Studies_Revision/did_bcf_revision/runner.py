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
from .did_bcf import DEFAULT_SPEC, fit_did_bcf, plain_estimands
from .posterior_correction import corrected_estimands
from .structured import PRODUCTION_SPECS, fit_structured, is_structured
from . import config as cfg

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
_GENERATORS = {"canonical": generate_canonical_did, "staggered": generate_staggered_did}


def fit_any(df, bcf_params: dict | None = None, seed: int | None = None,
            spec: str = DEFAULT_SPEC, allow_legacy: bool = False):
    """Fit whichever estimator ``spec`` names.

    Only the structured specifications are estimators.  The stochtree-based ones
    (``published``, ``rfx``, ``propensity``, ...) all leave the prognostic forest
    unrestricted and so all carry the flat likelihood direction of
    ``Results/identification_note.tex``; they exist only so the routes script can
    demonstrate that, and reaching them requires ``allow_legacy=True``.  This
    gate exists because the default used to be ``published``, and a template that
    simply omitted ``spec=`` therefore ran the broken estimator silently.
    """
    if is_structured(spec):
        return fit_structured(df, bcf_params=bcf_params, seed=seed, spec=spec)
    if not allow_legacy:
        raise ValueError(
            f"{spec!r} is a legacy specification with a known identification "
            f"failure (see Results/identification_note.tex) and is not usable "
            f"for results. Use one of {list(PRODUCTION_SPECS)}. Only "
            f"scripts/run_identification_routes.py may run it, via "
            f"allow_legacy=True.")
    return fit_did_bcf(df, bcf_params=bcf_params, seed=seed, spec=spec)


def process_rep(dgp: str, dgp_params: dict, N: int, rep: int, setting: str,
                bcf_params: dict | None = None, prop_method: str = "logit",
                n_splits: int = 2, pretrend_recenter: bool = False,
                spec: str = DEFAULT_SPEC) -> pd.DataFrame:
    """Run one replication; return tidy summary rows for both methods."""
    gen = _GENERATORS[dgp]
    df = gen(seed=int(rep), **{**dgp_params, "n_units": int(N)})

    fit = fit_any(df, bcf_params=bcf_params, seed=int(rep), spec=spec)
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
    out.insert(5, "spec", spec)
    return out


def run_experiment(exp: "cfg.Experiment", bcf_params: dict | None = None,
                   prop_method: str = "logit", n_splits: int = 2,
                   pretrend_recenter: bool = False, jobs: int = 1,
                   linearity_degree: int | None = None,
                   out_dir: str | None = None, save: bool = True,
                   spec: str = DEFAULT_SPEC,
                   rep_start: int = 0, rep_end: int | None = None) -> pd.DataFrame:
    """Run ``(N, rep)`` for ``rep in [rep_start, rep_end)`` and save the summaries.

    Returns the concatenated summary DataFrame (plain + corrected DiD-BCF).
    When ``linearity_degree`` is given it overrides the scenario default and is
    appended to the output filename; writes
    ``<out_dir>/summaries_<exp.name>[_lin_<d>][_<spec>][_reps<a>-<b>].csv`` when
    ``save`` is true.  A non-default ``spec`` is appended so route comparisons do
    not overwrite the published-specification runs.

    ``rep_start``/``rep_end`` split a long run across Colab sessions (capped at
    roughly 8 hours).  Replications are seeded by their index, so the parts
    concatenate into exactly the whole; ``aggregate_all.py`` reads every matching
    file and de-duplicates on ``rep``.
    """
    out_dir = out_dir or RESULTS_DIR
    if save and os.path.commonpath([os.path.abspath(out_dir), os.path.abspath(RESULTS_DIR)]) == os.path.abspath(RESULTS_DIR):
        raise ValueError("Historical Results/ is frozen after the variance-prior repair. "
                         "Use Proper_Prior_Rerun/campaign.py for versioned raw-only results.")
    params = dict(exp.dgp_params)
    suffix = ""
    if linearity_degree is not None:
        params["linearity_degree"] = int(linearity_degree)
        suffix = f"_lin_{int(linearity_degree)}"
    if spec != DEFAULT_SPEC:
        suffix += f"_{spec}"

    rep_end = int(exp.reps if rep_end is None else rep_end)
    rep_start = int(rep_start)
    if (rep_start, rep_end) != (0, exp.reps):
        suffix += f"_reps{rep_start}-{rep_end}"
    tasks = [(N, rep) for N in exp.n_values for rep in range(rep_start, rep_end)]
    print(f"[{exp.name}{suffix}] {exp.dgp} DGP | N={exp.n_values} | "
          f"reps={rep_start}..{rep_end} | spec={spec} | {len(tasks)} fits "
          f"| jobs={jobs}")

    def _one(N, rep):
        return process_rep(exp.dgp, params, N, rep, exp.name,
                           bcf_params, prop_method, n_splits, pretrend_recenter,
                           spec)

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
