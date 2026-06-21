"""Monte-Carlo driver for the **TWFE / OLS benchmark** (the foil in every
scenario, and the contaminated estimator in Workstream D).

Mirrors :mod:`did_bcf_revision.runner` but for plain two-way fixed-effects OLS.
It emits per-replication summaries in the *same* tidy schema as DiD-BCF
(``method='twfe'``) so :mod:`did_bcf_revision.metrics` aggregates DiD-BCF, the
posterior correction and TWFE side by side.

For each replication it reports

* the **event-study** path ``ATT(k)`` for ``k >= 0`` (relative-time dummies,
  never-treated as clean controls), and
* the overall static **ATT** (coefficient on the current-treatment indicator),

both with cluster-robust (by unit) standard errors.  Plain TWFE does not deliver
clean cohort-specific ``GATT(g,t)`` -- pooling cohorts is precisely the
Goodman-Bacon contamination -- so GATT is intentionally left to DiD-BCF and the
Callaway-Sant'Anna R benchmark.

This module is **pure numpy/pandas** (no stochtree), so the OLS notebooks and
``scripts/run_twfe.py`` run anywhere.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from .dgps import generate_canonical_did, generate_staggered_did, true_estimands
from .twfe import twfe_att_se, twfe_event_study_se
from . import config as cfg

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
_GENERATORS = {"canonical": generate_canonical_did, "staggered": generate_staggered_did}

# Two-sided normal critical values for the 90% / 95% intervals.
_Z90, _Z95 = 1.6448536269514722, 1.959963984540054


def _normal_summary(coef: float, se: float) -> dict:
    """Map an OLS (coef, se) to the DiD-BCF summary schema via a normal pivot."""
    if not np.isfinite(se) or se <= 0:
        z = np.inf if coef != 0 else 0.0
        p = 0.0 if coef != 0 else 0.5
        return {"post_mean": coef, "sd": se, "q025": coef, "q05": coef,
                "q95": coef, "q975": coef, "p_bayes": p}
    z = abs(coef / se)
    p_bayes = 0.5 * math.erfc(z / math.sqrt(2.0))   # one-tail prob (matches BCF)
    return {
        "post_mean": float(coef), "sd": float(se),
        "q025": float(coef - _Z95 * se), "q05": float(coef - _Z90 * se),
        "q95": float(coef + _Z90 * se), "q975": float(coef + _Z95 * se),
        "p_bayes": float(p_bayes),
    }


def process_rep(dgp: str, dgp_params: dict, N: int, rep: int, setting: str) -> pd.DataFrame:
    """Run one replication of the TWFE benchmark; return tidy summary rows."""
    gen = _GENERATORS[dgp]
    df = gen(seed=int(rep), **{**dgp_params, "n_units": int(N)})

    records = []
    # Event-study ATT(k), k >= 0.
    es = twfe_event_study_se(df, k_min=0)
    for _, r in es.iterrows():
        k = int(r["k"])
        if k < 0:
            continue
        rec = {"estimand_type": "ES", "estimand_id": f"k={k}",
               "g": np.nan, "t": np.nan, "k": k, "method": "twfe"}
        rec.update(_normal_summary(float(r["coef"]), float(r["se"])))
        records.append(rec)

    # Overall static ATT.
    coef, se = twfe_att_se(df)
    rec = {"estimand_type": "ATT", "estimand_id": "ATT",
           "g": np.nan, "t": np.nan, "k": np.nan, "method": "twfe"}
    rec.update(_normal_summary(coef, se))
    records.append(rec)

    out = pd.DataFrame.from_records(records)
    truth = true_estimands(df)[["estimand_type", "estimand_id", "true"]]
    out = out.merge(truth, on=["estimand_type", "estimand_id"], how="left")
    out.insert(0, "dgp", dgp)
    out.insert(1, "setting", setting)
    out.insert(2, "linearity_degree", int(dgp_params.get("linearity_degree", 1)))
    out.insert(3, "N", int(N))
    out.insert(4, "rep", int(rep))
    return out


def run_twfe_experiment(exp: "cfg.Experiment", linearity_degree: int | None = None,
                        jobs: int = 1, out_dir: str | None = None,
                        save: bool = True) -> pd.DataFrame:
    """Run the TWFE benchmark over every (N, rep) of ``exp``.

    When ``linearity_degree`` is given it overrides the scenario default and is
    appended to the output filename (``summaries_twfe_<name>_lin_<d>.csv``).
    """
    out_dir = out_dir or RESULTS_DIR
    params = dict(exp.dgp_params)
    suffix = ""
    if linearity_degree is not None:
        params["linearity_degree"] = int(linearity_degree)
        suffix = f"_lin_{int(linearity_degree)}"

    tasks = [(N, rep) for N in exp.n_values for rep in range(exp.reps)]
    print(f"[twfe:{exp.name}{suffix}] {exp.dgp} DGP | N={exp.n_values} "
          f"| reps={exp.reps} | {len(tasks)} fits | jobs={jobs}")

    def _one(N, rep):
        return process_rep(exp.dgp, params, N, rep, exp.name)

    if jobs and jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_one)(N, rep) for N, rep in tasks)
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(tasks, desc=f"twfe:{exp.name}{suffix}", unit="fit")
        except Exception:
            iterator = tasks
        rows = [_one(N, rep) for N, rep in iterator]

    result = pd.concat(rows, ignore_index=True)
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"summaries_twfe_{exp.name}{suffix}.csv")
        result.to_csv(path, index=False)
        print(f"[twfe:{exp.name}{suffix}] wrote {len(result)} rows -> {path}")
    return result


def run_twfe_named(name: str, linearity_degree: int | None = None,
                   reps: int | None = None, **kwargs) -> pd.DataFrame:
    """Convenience for the OLS notebooks: run the TWFE benchmark by scenario name.

    ``run_twfe_named("D_staggered", linearity_degree=1, reps=100)``.
    """
    exp = cfg.get_experiment(name)
    if reps is not None:
        exp.reps = int(reps)
    return run_twfe_experiment(exp, linearity_degree=linearity_degree, **kwargs)
