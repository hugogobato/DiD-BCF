#!/usr/bin/env python3
"""Regenerate the per-scenario simulation tree from templates.

The revision mirrors the original ``Simulation_Studies/`` layout: **one file per
model, per DGP (scenario), per linearity_degree**.  Rather than hand-maintain
~90 near-identical files, this script emits them from a single source of truth
(:func:`did_bcf_revision.config.all_experiments`):

* ``DGPs/data_creation_<scenario>.py``                  (one per scenario)
* ``DiD_BCF/DiD_BCF_<scenario>_lin_<d>.ipynb``          (scenario x linearity)
* ``TWFE/OLS_<scenario>_lin_<d>.ipynb``                 (scenario x linearity)
* ``R_code/<scenario>_datasets/{did_dr_new,did2s,DoubleML_did,synthdid}.R``

Run from anywhere::

    python scripts/scaffold_suite.py

It is idempotent (overwrites the generated files); edit the templates here and
re-run to propagate a change across the whole suite.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from did_bcf_revision import config as cfg

LIN = cfg.LINEARITY_DEGREES
DGP_HUMAN = {"canonical": "canonical DiD (selection on unobservables)",
             "staggered": "staggered adoption (cohort x event-time effects)"}


def sub(template: str, **kw) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace(f"__{k}__", str(v))
    return out


# --------------------------------------------------------------------------- #
# Notebook helpers
# --------------------------------------------------------------------------- #
def _cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


def _notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
DATA_CREATION_TMPL = '''#!/usr/bin/env python3
"""DGP: __SCEN__  (workstream __WS__, __DGPHUMAN__).

__NOTE__

Writes CSV replications in the **R-benchmark column layout** for every
``linearity_degree`` to
``R_code/__SCEN___datasets/linearity_degree=<d>/iteration_<rep>.csv`` -- exactly
the files the R estimators in ``R_code/__SCEN___datasets/`` read.  The DiD-BCF and
OLS notebooks regenerate identical (seeded) data in-memory, so this step is only
needed for the R benchmarks (or any external tool).

Panel: N=200, 4 pre + 4 post periods (override with the engine if needed).

Examples
--------
    python DGPs/data_creation___SCEN__.py --reps 100 --jobs 8
    python DGPs/data_creation___SCEN__.py --all-N        # sweep scenarios only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision.config import get_experiment, LINEARITY_DEGREES
from did_bcf_revision.dgps import generate_canonical_did, generate_staggered_did
from did_bcf_revision.exports import to_r_frame

SCENARIO = "__SCEN__"
GEN = {"canonical": generate_canonical_did,
       "staggered": generate_staggered_did}["__DGP__"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "R_code", SCENARIO + "_datasets")


def _write_one(params, N, d, rep, out_dir):
    df = GEN(seed=int(rep), **{**params, "n_units": int(N), "linearity_degree": int(d)})
    to_r_frame(df).to_csv(os.path.join(out_dir, "iteration_%d.csv" % rep), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=None, help="default: scenario reps")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--all-N", action="store_true",
                    help="also write the full N sweep under N=<N>/ (sweep scenarios)")
    args = ap.parse_args()

    exp = get_experiment(SCENARIO)
    reps = args.reps if args.reps is not None else exp.reps
    base_N = exp.n_values[0]

    tasks = []
    for d in LINEARITY_DEGREES:
        base_dir = os.path.join(OUT_ROOT, "linearity_degree=%d" % d)
        os.makedirs(base_dir, exist_ok=True)
        for rep in range(reps):
            tasks.append((exp.dgp_params, base_N, d, rep, base_dir))
        if args.all_N and len(exp.n_values) > 1:
            for N in exp.n_values:
                nd = os.path.join(OUT_ROOT, "N=%d" % N, "linearity_degree=%d" % d)
                os.makedirs(nd, exist_ok=True)
                for rep in range(reps):
                    tasks.append((exp.dgp_params, N, d, rep, nd))

    print("[%s] writing %d CSVs -> %s (jobs=%d)" % (SCENARIO, len(tasks), OUT_ROOT, args.jobs))
    if args.jobs and args.jobs > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(_write_one)(*t) for t in tasks)
    else:
        for t in tasks:
            _write_one(*t)
    print("Done.")


if __name__ == "__main__":
    main()
'''

# Self-bootstrap snippet: makes a notebook runnable after uploading ONLY itself
# (it fetches the engine from GitHub when it is not already next to the notebook).
BOOTSTRAP = """import os, sys

# --- Locate the DiD-BCF engine ------------------------------------------------
# So you can upload just THIS notebook to Colab and Run all. Resolution order:
#   1. `did_bcf_revision` already importable;
#   2. running inside a repo checkout (the parent folder holds the package);
#   3. otherwise clone https://github.com/hugogobato/DiD-BCF and use it.
REPO_URL = "https://github.com/hugogobato/DiD-BCF.git"
ENGINE_SUBDIR = os.path.join("DiD-BCF", "Simulation_Studies_Revision")

def _locate_root():
    try:
        import did_bcf_revision  # noqa: F401
        return os.path.dirname(os.path.dirname(did_bcf_revision.__file__))
    except Exception:
        pass
    parent = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if os.path.isdir(os.path.join(parent, "did_bcf_revision")):
        return parent
    if not os.path.isdir("DiD-BCF"):
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    return os.path.abspath(ENGINE_SUBDIR)

ROOT = _locate_root()
sys.path.insert(0, ROOT)
print("Using DiD-BCF engine at:", ROOT)
"""

# ---- DiD-BCF notebook cell sources --------------------------------------- #
BCF_MD = """# DiD-BCF — __SCEN__ (linearity_degree=__LIN__)

**Workstream __WS__ · __DGPHUMAN__**

__NOTE__

Fits DiD-BCF on the `__SCEN__` scenario at `linearity_degree=__LIN__` and reports
metrics for **both** the plain DiD-BCF posterior and the proposed **posterior
correction** (Algorithm 1 of the theory note), so the correction can be judged
directly. Panel: N=200, 4 pre + 4 post periods.

> **Colab:** upload just this notebook and *Run all* — the first cell installs the
> dependencies and the second clones the engine automatically."""

BCF_PIP = """# Colab: install the DiD-BCF dependencies (stochtree provides the BCF sampler).
%pip install -q stochtree scikit-learn joblib tqdm pandas numpy"""

BCF_SETUP = BOOTSTRAP + """
from did_bcf_revision.runner import run_named
from did_bcf_revision.metrics import (compute_metrics, plain_vs_corrected,
                                      surface_metrics)"""

BCF_RUN = """REPS = __REPS__      # this scenario's configured count (config.py)
JOBS = 1        # parallel reps (keep 1 on a single-core/GPU Colab)

# A Colab session is capped at roughly 8 hours and one fit is ~2 min. To split a
# long run, set these to (0, 100) here and (100, 200) in a second copy of this
# notebook: replications are seeded by index, so the parts concatenate into
# exactly the undivided run and are written to separate files.
REP_START, REP_END = 0, REPS

bcf_params = dict(num_gfr=50, num_mcmc=500, keep_every=5, num_chains=3)

summaries = run_named(
    "__SCEN__",
    linearity_degree=__LIN__,
    reps=REPS,
    jobs=JOBS,
    bcf_params=bcf_params,
    prop_method="logit",   # pilot propensity for the posterior correction
    n_splits=2,            # cross-fitting folds for the correction
    spec="structured",     # the whole grid runs the corrected estimator; the
                           # engine default is "published", so this is required
    rep_start=REP_START, rep_end=REP_END,
    save=False,
)
summaries.head()

out_csv = "summaries___SCEN___lin___LIN__.csv"
summaries.to_csv(out_csv, index=False)
print("wrote", out_csv, "| rows:", len(summaries))

output_file = out_csv
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)"""

BCF_METRICS = """# Decomposed metrics: bias, MC SD/variance, RMSE, MAE, MAPE, coverage 90/95,
# interval length, calibration ratio (avg_post_sd/emp_sd), size/power and their
# Monte-Carlo SEs -- for plain AND corrected DiD-BCF.
metrics = compute_metrics(summaries)
plain_vs_corrected(metrics)"""

BCF_SURFACE_MD = """## CATT-surface metrics (the paper's headline RMSE/MAE/MAPE)

Within-replication RMSE/MAE/MAPE over the *individual* treated observations
(mean +/- SD across runs) plus the *pointwise* CATT coverage -- the evidence
that DiD-BCF recovers the heterogeneous effect that GATT-only methods cannot."""

BCF_SURFACE = """surface_metrics(summaries)"""

BCF_GB_MD = """## Goodman-Bacon decomposition (TWFE contamination)

How much of a naive TWFE estimate on this DGP comes from the
"already-treated as control" comparisons that bias it."""

BCF_GB = """from did_bcf_revision.dgps import generate_staggered_did
from did_bcf_revision.goodman_bacon import bacon_summary

df0 = generate_staggered_did(seed=0, linearity_degree=__LIN__)
bacon_summary(df0)"""

# ---- OLS / TWFE notebook cell sources ------------------------------------ #
OLS_MD = """# TWFE / OLS benchmark — __SCEN__ (linearity_degree=__LIN__)

**Workstream __WS__ · __DGPHUMAN__**

Plain two-way fixed-effects OLS benchmark for the `__SCEN__` scenario at
`linearity_degree=__LIN__`: the event-study path ATT(k) (k ≥ 0) and the overall
static ATT, each with cluster-robust (by unit) standard errors. This is the foil
DiD-BCF is compared against -- and, under staggered dynamic effects, the
estimator Goodman-Bacon (2021) shows is contaminated. Pure numpy/pandas (no
stochtree), so it runs on a laptop with parallelisation.

> **Colab:** upload just this notebook and *Run all* — the setup cell clones the
> engine automatically (no extra installs needed)."""

OLS_SETUP = BOOTSTRAP + """
from did_bcf_revision.twfe_runner import run_twfe_named
from did_bcf_revision.metrics import compute_metrics, surface_metrics"""

OLS_RUN = """REPS = 100
JOBS = 1        # raise to parallelise replications on your PC

summaries = run_twfe_named("__SCEN__", linearity_degree=__LIN__, reps=REPS, jobs=JOBS)
summaries.head()"""

OLS_METRICS = """# Decomposed metrics (incl. MAE/MAPE, calibration ratio, MC SEs) and the
# CATT-surface RMSE/MAE/MAPE (TWFE's event-study coef broadcast to each obs --
# where heterogeneity-blind TWFE pays its price).
display(compute_metrics(summaries))
surface_metrics(summaries)"""


# ---- Pre-trend diagnostic notebook cell sources (workstream PT) ---------- #
PT_MD = """# Pre-trend diagnostic — __SCEN__ (linearity_degree=__LIN__)

**Workstream PT · conditional-parallel-trends diagnostic (Reviewer 3.2.3)**

__NOTE__

The estimation model sets `Z = D_it`, which is zero on every pre-treatment row,
so those rows carry **no information about tau** and no post-processing of that
fit can produce a placebo coefficient. This notebook therefore fits the
*unconstrained* specification (`Z = 1[G_i != inf]`, on in every period) as a
**separate diagnostic model**, and reports

$$\\\\Delta(k) = E\\\\big[\\\\tau(X, k) - \\\\tau(X, -1)\\\\;\\\\big|\\\\;\\\\text{treated}\\\\big],
\\\\qquad k < -1,$$

which is exactly zero under conditional parallel trends. The TWFE event-study
placebo is run on the same replications, for free, as the standard-practice
comparator.

> **Colab:** upload just this notebook and *Run all*."""

PT_PIP = BCF_PIP

PT_SETUP = BOOTSTRAP + """
from did_bcf_revision.pretrend_runner import run_named
from did_bcf_revision.metrics import compute_metrics"""

PT_RUN = """REPS = __REPS__      # detection rates need replications; 200 gives MCSE <= 0.035
JOBS = 2        # ~1 GB RAM per worker; drop to 1 if the VM is memory-starved

# WITH_ATT also fits the constrained estimation model on the same replications,
# so the ATT bias the violation causes is measured alongside its detection.
# It doubles the MCMC cost -- worth it at degree 1, skip it at 2 and 3.
WITH_ATT = __WITHATT__

# A Colab session is capped at roughly 8 hours. One fit is ~2 min, so a full
# 200-replication WITH_ATT run is ~7 h at JOBS=2 -- close enough to the cap that
# it is worth splitting. Set these to (0, 100) here and (100, 200) in a second
# copy of this notebook; replications are seeded by index, so the two parts
# concatenate into exactly the undivided run and land in separate files.
REP_START, REP_END = 0, REPS

summaries = run_named(
    "__SCEN__",
    linearity_degree=__LIN__,
    reps=REPS,
    jobs=JOBS,
    with_att=WITH_ATT,
    rfx="unit",     # unit intercepts absorb the level gap conditional PT allows
    bcf_params=dict(num_gfr=50, num_mcmc=500, keep_every=5, num_chains=3),
    rep_start=REP_START, rep_end=REP_END,
)
summaries.head()"""

PT_METRICS = """# `reject05` on the PRE rows is the diagnostic's **size** when the true
# differential slope is 0 and its **detection rate** otherwise; `any_bonf` is the
# per-replication decision rule (any pre-period significant, Bonferroni-scaled).
metrics = compute_metrics(summaries)
pre = metrics[metrics.estimand_type == "PRE"]
pre[["method", "estimand_id", "mean_true", "bias", "cover95",
     "reject05", "mcse_reject05", "role"]].sort_values(["estimand_id", "method"])"""

PT_SUB_MD = """## Conditional check

`Delta(k)` within covariate subgroups. A marginal event study cannot produce
this without pre-specifying the interactions; here it comes out of the same fit,
and it is the version that matches what the estimator actually assumes."""

PT_SUB = """sub = metrics[metrics.estimand_type == "PRE_SUB"]
sub[["estimand_id", "mean_true", "bias", "emp_sd", "cover95", "reject05"]]"""


# ---- R benchmark templates (token: @@SCEN@@) ----------------------------- #
# ---- R benchmark scripts ------------------------------------------------- #
# NOT templated here.  These scripts emit the DiD-BCF summary schema and have
# been revised since this scaffold was written; embedding a second copy meant a
# scaffold run silently reverted them (measured once: 3222 lines of the
# schema-emitting versions replaced by the older templates).  They are instead
# copied from the checked-in reference scenario for the matching DGP, with only
# the two scenario-identifying lines substituted -- which is an exact
# transformation, since the per-scenario scripts differ in nothing else.
R_FILES = ["did_dr_new.R", "did2s.R", "DoubleML_did.R", "synthdid.R"]
R_REFERENCE = {"canonical": "B1_baseline", "staggered": "D_staggered"}


def r_script(script: str, scen: str, dgp: str) -> str:
    """The reference scenario's script, retargeted at ``scen``.

    The reference is picked by DGP, so ``DGP <- "..."`` never needs changing and
    the scenario name is the only variable.  It appears only as a quoted string
    (the header comment and the ``SETTING <- "..."`` assignment), and the two
    use different phrasings across the four scripts, so every quoted occurrence
    is replaced rather than a fixed pair of sentences.
    """
    ref = R_REFERENCE[dgp]
    path = os.path.join(ROOT, "R_code", f"{ref}_datasets", script)
    with open(path) as fh:
        src = fh.read()
    if f'SETTING <- "{ref}"' not in src:
        raise RuntimeError(f"{ref}/{script} has no SETTING <- \"{ref}\" line; "
                           "the reference script changed shape.")
    return src.replace(f'"{ref}"', f'"{scen}"')



def _is_tracked(path):
    """True if git tracks `path`.

    A tracked notebook is one the user committed, which for this repo means it
    was run and its result kept.  Regenerating it silently replaces the
    specification that produced the committed results -- this is exactly how the
    `spec="structured"` argument got dropped from the whole B1/D grid -- so the
    scaffold refuses to touch tracked files without `--force`.
    """
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                           cwd=ROOT, capture_output=True)
        return r.returncode == 0
    except Exception:
        return True          # no git, or git failed: assume precious


def _has_been_run(path):
    """True if `path` is a notebook carrying evidence of execution.

    Belt and braces for files git does not track. Executed notebooks keep
    `outputs` and a non-null `execution_count`; a notebook whose outputs were
    stripped before committing still carries Colab's per-cell `id`/`outputId`
    metadata, so check that too.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            nb = json.load(fh)
    except Exception:
        return True          # unreadable: refuse to touch it
    for cell in nb.get("cells", []):
        if cell.get("outputs") or cell.get("execution_count") is not None:
            return True
        meta = cell.get("metadata", {})
        if "outputId" in meta or "colab" in meta:
            return True
    return False


def write_notebook(path, cells, force=False):
    """Write a notebook unless it is committed or holds a completed run."""
    if not force:
        if _is_tracked(path):
            print(f"  SKIP (tracked by git): {os.path.relpath(path, ROOT)}")
            return False
        if _has_been_run(path):
            print(f"  SKIP (already run): {os.path.relpath(path, ROOT)}")
            return False
    with open(path, "w") as f:
        json.dump(_notebook(cells), f, indent=1)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="regenerate notebooks even if they carry execution "
                         "outputs (this DESTROYS the record of that run)")
    args = ap.parse_args()
    force = args.force

    exps = cfg.all_experiments()
    dgps_dir = os.path.join(ROOT, "DGPs")
    bcf_dir = os.path.join(ROOT, "DiD_BCF")
    ols_dir = os.path.join(ROOT, "TWFE")
    rcode_dir = os.path.join(ROOT, "R_code")
    pt_dir = os.path.join(ROOT, "Pretrend")
    for dd in (dgps_dir, bcf_dir, ols_dir, rcode_dir, pt_dir):
        os.makedirs(dd, exist_ok=True)

    n_data = n_bcf = n_ols = n_r = n_pt = 0
    for e in exps:
        ctx = dict(SCEN=e.name, WS=e.workstream, DGP=e.dgp,
                   DGPHUMAN=DGP_HUMAN[e.dgp], NOTE=e.note, REPS=e.reps)

        # Workstream PT has its own driver (the unconstrained diagnostic fit)
        # and no benchmark comparison, so it gets one notebook family and none
        # of the estimation-model scaffolding.
        if e.workstream == "PT":
            for d in LIN:
                # The constrained refit that measures the violation's cost is
                # only worth its doubled MCMC bill at the headline degree.
                c = dict(ctx, LIN=d, WITHATT="True" if d == 1 else "False")
                cells = [_cell("markdown", sub(PT_MD, **c)),
                         _cell("code", sub(PT_PIP, **c)),
                         _cell("code", sub(PT_SETUP, **c)),
                         _cell("code", sub(PT_RUN, **c)),
                         _cell("code", sub(PT_METRICS, **c)),
                         _cell("markdown", sub(PT_SUB_MD, **c)),
                         _cell("code", sub(PT_SUB, **c))]
                n_pt += write_notebook(
                    os.path.join(pt_dir, f"Pretrend_{e.name}_lin_{d}.ipynb"),
                    cells, force)
            continue

        # 1) data-creation script
        with open(os.path.join(dgps_dir, f"data_creation_{e.name}.py"), "w") as f:
            f.write(sub(DATA_CREATION_TMPL, **ctx))
        n_data += 1

        # 2) notebooks (one per linearity degree)
        for d in LIN:
            c = dict(ctx, LIN=d)
            bcf_cells = [
                _cell("markdown", sub(BCF_MD, **c)),
                _cell("code", sub(BCF_PIP, **c)),
                _cell("code", sub(BCF_SETUP, **c)),
                _cell("code", sub(BCF_RUN, **c)),
                _cell("code", sub(BCF_METRICS, **c)),
                _cell("markdown", sub(BCF_SURFACE_MD, **c)),
                _cell("code", sub(BCF_SURFACE, **c)),
            ]
            if e.dgp == "staggered":
                bcf_cells += [_cell("markdown", sub(BCF_GB_MD, **c)),
                              _cell("code", sub(BCF_GB, **c))]
            n_bcf += write_notebook(
                os.path.join(bcf_dir, f"DiD_BCF_{e.name}_lin_{d}.ipynb"),
                bcf_cells, force)

            ols_cells = [
                _cell("markdown", sub(OLS_MD, **c)),
                _cell("code", sub(OLS_SETUP, **c)),
                _cell("code", sub(OLS_RUN, **c)),
                _cell("code", sub(OLS_METRICS, **c)),
            ]
            n_ols += write_notebook(
                os.path.join(ols_dir, f"OLS_{e.name}_lin_{d}.ipynb"),
                ols_cells, force)

        # 3) R benchmark scripts, retargeted from the reference scenario.
        #    The reference scenario is its own source of truth and is skipped,
        #    so re-running this can never rewrite the scripts it copies from.
        sdir = os.path.join(rcode_dir, f"{e.name}_datasets")
        os.makedirs(sdir, exist_ok=True)
        if e.name in R_REFERENCE.values():
            continue
        for fname in R_FILES:
            with open(os.path.join(sdir, fname), "w") as f:
                f.write(r_script(fname, e.name, e.dgp))
            n_r += 1

    print(f"scenarios: {len(exps)}")
    print(f"data_creation scripts: {n_data}")
    print(f"DiD_BCF notebooks:     {n_bcf}")
    print(f"OLS notebooks:         {n_ols}")
    print(f"Pretrend notebooks:    {n_pt}")
    print(f"R scripts:             {n_r}")


if __name__ == "__main__":
    main()
