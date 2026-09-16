"""Monte-Carlo driver for the pre-trend / conditional-PTA diagnostic.

One replication produces, in the *same* schema as every other summary file in
the suite (so :mod:`did_bcf_revision.metrics` aggregates it unchanged):

* ``method='pretrend'``  -- the unconstrained diagnostic fit of
  :mod:`did_bcf_revision.pretrend`;
* ``method='twfe_es'``   -- the standard-practice TWFE event-study placebo, the
  comparator applied work actually reports (free: no MCMC);
* ``method='plain'`` / ``'corrected'`` -- optionally (``with_att=True``) the
  ordinary constrained DiD-BCF fit, so the *consequence* of the violation (ATT
  bias) is measured on the same replications that the diagnostic flags.  This
  doubles the cost per replication, which is why it is a switch.

Because ``true_pretrend`` puts the realised differential slope in the ``true``
column, ``compute_metrics`` reports ``reject05`` on the ``PRE`` rows as the
diagnostic's **size** when the violation is zero and its **detection rate**
otherwise, with the Monte-Carlo standard errors it already computes.
"""

from __future__ import annotations

import os

import pandas as pd

from .dgps import generate_canonical_did, generate_staggered_did, true_estimands
from .did_bcf import plain_estimands
from .pretrend import fit_pretrend, pretrend_estimands, true_pretrend, twfe_pretrend
from .posterior_correction import corrected_estimands
from .runner import fit_any
from . import config as cfg

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
_GENERATORS = {"canonical": generate_canonical_did, "staggered": generate_staggered_did}


def process_rep(dgp: str, dgp_params: dict, N: int, rep: int, setting: str,
                bcf_params: dict | None = None, rfx: str = "unit",
                with_att: bool = False) -> pd.DataFrame:
    """Run one replication of the diagnostic; return tidy summary rows."""
    df = _GENERATORS[dgp](seed=int(rep), **{**dgp_params, "n_units": int(N)})

    fit = fit_pretrend(df, bcf_params=bcf_params, seed=int(rep), rfx=rfx)
    parts = [pretrend_estimands(fit), twfe_pretrend(df)]
    truth = true_pretrend(df)

    if with_att:
        # Use the shared production dispatcher.  Calling fit_did_bcf directly
        # would resolve DEFAULT_SPEC through the legacy registry, which does
        # not contain the structured estimator used by the revision.
        cfit = fit_any(df, bcf_params=bcf_params, seed=int(rep),
                       spec="structured")
        parts += [plain_estimands(cfit),
                  corrected_estimands(cfit, seed=int(rep))]
        truth = pd.concat(
            [truth, true_estimands(df)[["estimand_type", "estimand_id", "true"]]],
            ignore_index=True)

    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if truth.empty:
        # Degenerate draw (no treated or no controls): keep the row shape so the
        # replication still concatenates, with an unknown truth.
        out["true"] = float("nan")
    else:
        out = out.merge(truth, on=["estimand_type", "estimand_id"], how="left")
    out.insert(0, "dgp", dgp)
    out.insert(1, "setting", setting)
    out.insert(2, "linearity_degree", int(dgp_params.get("linearity_degree", 1)))
    out.insert(3, "N", int(N))
    out.insert(4, "rep", int(rep))
    out.insert(5, "spec", f"pretrend_{rfx}")
    return out


def run_experiment(exp: "cfg.Experiment", bcf_params: dict | None = None,
                   rfx: str = "unit", with_att: bool = False, jobs: int = 1,
                   linearity_degree: int | None = None,
                   out_dir: str | None = None, save: bool = True,
                   rep_start: int = 0, rep_end: int | None = None) -> pd.DataFrame:
    """Run replications ``[rep_start, rep_end)`` of ``exp`` and save the summaries.

    Writes ``<out_dir>/summaries_pretrend_<exp.name>[_lin_<d>][_reps<a>-<b>].csv``.

    ``rep_start``/``rep_end`` exist because a Colab session is capped at roughly
    8 hours: one 200-replication ``--with-att`` cell is close to that budget, so
    the work can be split across notebooks and the parts concatenated later
    (``aggregate_all.py`` reads every matching file and de-duplicates on
    ``rep``).  Replications are seeded by their index, so a split run is
    bit-identical to the whole.
    """
    out_dir = out_dir or RESULTS_DIR
    if save and os.path.commonpath([os.path.abspath(out_dir), os.path.abspath(RESULTS_DIR)]) == os.path.abspath(RESULTS_DIR):
        raise ValueError("Historical Results/ is frozen after the variance-prior repair. "
                         "Use Proper_Prior_Rerun/campaign.py for versioned diagnostic results.")
    params = dict(exp.dgp_params)
    suffix = ""
    if linearity_degree is not None:
        params["linearity_degree"] = int(linearity_degree)
        suffix = f"_lin_{int(linearity_degree)}"

    rep_end = int(exp.reps if rep_end is None else rep_end)
    rep_start = int(rep_start)
    if (rep_start, rep_end) != (0, exp.reps):
        suffix += f"_reps{rep_start}-{rep_end}"
    tasks = [(N, rep) for N in exp.n_values for rep in range(rep_start, rep_end)]
    n_fits = len(tasks) * (2 if with_att else 1)
    print(f"[pretrend {exp.name}{suffix}] {exp.dgp} DGP | N={exp.n_values} | "
          f"reps={rep_start}..{rep_end} | rfx={rfx} | with_att={with_att} | "
          f"{n_fits} BCF fits | jobs={jobs}")

    def _one(N, rep):
        return process_rep(exp.dgp, params, N, rep, exp.name,
                           bcf_params, rfx, with_att)

    if jobs and jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_one)(N, rep) for N, rep in tasks)
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(tasks, desc=f"pretrend {exp.name}", unit="rep")
        except Exception:
            iterator = tasks
        rows = [_one(N, rep) for N, rep in iterator]

    result = pd.concat(rows, ignore_index=True)
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"summaries_pretrend_{exp.name}{suffix}.csv")
        result.to_csv(path, index=False)
        print(f"[pretrend {exp.name}{suffix}] wrote {len(result)} rows -> {path}")
    return result


def run_named(name: str, reps: int | None = None, **kwargs) -> pd.DataFrame:
    """Convenience for notebooks: ``run_named("PT_violation_g20", reps=200)``."""
    exp = cfg.get_experiment(name)
    if reps is not None:
        exp.reps = int(reps)
    return run_experiment(exp, **kwargs)
