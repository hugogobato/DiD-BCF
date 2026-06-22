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

import json
import os
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

BCF_RUN = """REPS = 100      # replications (lower for a quick smoke test)
JOBS = 1        # parallel reps (keep 1 on a single-core/GPU Colab)

bcf_params = dict(num_gfr=50, num_mcmc=500, keep_every=5, num_chains=3)

summaries = run_named(
    "__SCEN__",
    linearity_degree=__LIN__,
    reps=REPS,
    jobs=JOBS,
    bcf_params=bcf_params,
    prop_method="logit",   # pilot propensity for the posterior correction
    n_splits=2,            # cross-fitting folds for the correction
)
summaries.head()"""

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


# ---- R benchmark templates (token: @@SCEN@@) ----------------------------- #
R_HEADER = r'''# %s benchmark for scenario "@@SCEN@@".
# Run from inside R_code/@@SCEN@@_datasets/  (the folder holding the
# linearity_degree=1/2/3 sub-folders written by DGPs/data_creation_@@SCEN@@.py).
# Estimands are detected from the data, so this works for both the single-cohort
# (canonical) and 3-cohort (staggered) scenarios.
'''

R_DID_DR = R_HEADER % "Callaway & Sant'Anna doubly-robust (R `did`)" + r'''
library(did)
library(progress)
sink("output_did_dr.txt")

lin_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")
options(warn = -1)

for (lin in lin_folders) {
  files <- list.files(lin, pattern = "^iteration_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) { cat("No files in", lin, "- skipping\n"); next }
  niter <- length(files)
  RMSE <- numeric(niter); MAE <- numeric(niter); MAPE <- rep(NA_real_, niter)
  est_rows <- list()
  pb <- progress_bar$new(total = niter, format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick()
    d <- read.csv(files[ii])
    d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
    out <- tryCatch(
      att_gt(yname = "Y", tname = "time", idname = "unit_id",
             gname = "first_treat_period", xformla = ~ X_1+X_2+X_3+X_4+X_5,
             data = d, est_method = "dr", control_group = "nevertreated",
             print_details = FALSE, pl = FALSE, cores = 1),
      error = function(e) NULL)
    if (is.null(out)) next
    groups <- sort(unique(out$group[out$group > 0]))
    errs <- c(); aerr <- c(); ape <- c()
    for (g in groups) {
      post_t <- sort(unique(out$t[out$group == g & out$t >= g]))
      for (t in post_t) {
        sel <- which(out$group == g & out$t == t)
        if (length(sel) == 0) next
        est <- out$att[sel]; se <- out$se[sel]
        truth <- mean(d$CATE[d$first_treat_period == g & d$time == t], na.rm = TRUE)
        sig <- as.integer(abs(est / se) > 1.96)
        errs <- c(errs, est - truth); aerr <- c(aerr, abs(est - truth))
        if (truth != 0) ape <- c(ape, abs((est - truth) / truth))
        est_rows[[length(est_rows) + 1]] <- data.frame(
          iteration = ii - 1, group = g, t = t, k = t - g,
          estimate = est, se = se, true = truth, sig = sig)
      }
    }
    if (length(errs)) { RMSE[ii] <- sqrt(mean(errs^2)); MAE[ii] <- mean(aerr) }
    if (length(ape)) MAPE[ii] <- mean(ape)
  }
  cat(sprintf("\n%s : mean RMSE=%.4f (sd %.4f)  MAE=%.4f  MAPE=%.4f\n",
              lin, mean(RMSE), sd(RMSE), mean(MAE), mean(MAPE, na.rm = TRUE)))
  metrics_df <- data.frame(iteration = 0:(niter - 1), RMSE = RMSE, MAE = MAE, MAPE = MAPE)
  est_df <- if (length(est_rows)) do.call(rbind, est_rows) else data.frame()
  fn <- paste0("did_dr_GATE_and_PValues_", lin, ".xlsx")
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Metrics");  openxlsx::writeData(wb, "Metrics", metrics_df)
    openxlsx::addWorksheet(wb, "Estimates"); openxlsx::writeData(wb, "Estimates", est_df)
    openxlsx::saveWorkbook(wb, fn, overwrite = TRUE); cat("wrote", fn, "\n")
  } else {
    write.csv(metrics_df, paste0("did_dr_Metrics_", lin, ".csv"), row.names = FALSE)
    write.csv(est_df,     paste0("did_dr_Estimates_", lin, ".csv"), row.names = FALSE)
  }
}
sink()
'''

R_DID2S = R_HEADER % "Gardner two-stage (R `did2s`) event study" + r'''
library(did2s)
library(progress)
sink("output_did2s.txt")

lin_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")
options(warn = -1)

for (lin in lin_folders) {
  files <- list.files(lin, pattern = "^iteration_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) { cat("No files in", lin, "- skipping\n"); next }
  niter <- length(files)
  RMSE <- numeric(niter); MAE <- numeric(niter); MAPE <- rep(NA_real_, niter)
  est_rows <- list()
  pb <- progress_bar$new(total = niter, format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick()
    d <- read.csv(files[ii])
    d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
    out <- tryCatch(
      event_study(yname = "Y", tname = "time", idname = "unit_id",
                  gname = "first_treat_period", xformla = ~ X_1+X_2+X_3+X_4+X_5,
                  data = d, estimator = "did2s"),
      error = function(e) NULL)
    if (is.null(out)) next
    sub <- out[out$estimator == "did2s", ]
    sub$k <- suppressWarnings(as.integer(as.character(sub$term)))
    errs <- c(); aerr <- c(); ape <- c()
    for (kk in sort(unique(sub$k[is.finite(sub$k) & sub$k >= 0]))) {
      row <- sub[sub$k == kk, ]
      est <- row$estimate[1]; se <- row$std.error[1]
      truth <- mean(d$CATE[d$D == 1 & d$event_time == kk], na.rm = TRUE)
      sig <- as.integer(abs(est / se) > 1.96)
      errs <- c(errs, est - truth); aerr <- c(aerr, abs(est - truth))
      if (truth != 0) ape <- c(ape, abs((est - truth) / truth))
      est_rows[[length(est_rows) + 1]] <- data.frame(
        iteration = ii - 1, k = kk, estimate = est, se = se, true = truth, sig = sig)
    }
    if (length(errs)) { RMSE[ii] <- sqrt(mean(errs^2)); MAE[ii] <- mean(aerr) }
    if (length(ape)) MAPE[ii] <- mean(ape)
  }
  cat(sprintf("\n%s : mean RMSE=%.4f (sd %.4f)  MAE=%.4f  MAPE=%.4f\n",
              lin, mean(RMSE), sd(RMSE), mean(MAE), mean(MAPE, na.rm = TRUE)))
  metrics_df <- data.frame(iteration = 0:(niter - 1), RMSE = RMSE, MAE = MAE, MAPE = MAPE)
  est_df <- if (length(est_rows)) do.call(rbind, est_rows) else data.frame()
  fn <- paste0("did2s_GATE_and_PValues_", lin, ".xlsx")
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Metrics");  openxlsx::writeData(wb, "Metrics", metrics_df)
    openxlsx::addWorksheet(wb, "Estimates"); openxlsx::writeData(wb, "Estimates", est_df)
    openxlsx::saveWorkbook(wb, fn, overwrite = TRUE); cat("wrote", fn, "\n")
  } else {
    write.csv(metrics_df, paste0("did2s_Metrics_", lin, ".csv"), row.names = FALSE)
    write.csv(est_df,     paste0("did2s_Estimates_", lin, ".csv"), row.names = FALSE)
  }
}
sink()
'''

R_DOUBLEML = R_HEADER % "DoubleML doubly-robust DiD (random-forest nuisances)" + r'''
library(did)
library(progress)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(lgr)
lgr::get_logger("mlr3")$set_threshold("fatal")

# Plug DoubleML's ATTE estimator into att_gt as a custom est_method (as Chang 2020).
doubleml_did_rf <- function(y1, y0, D, covariates,
                            ml_g = lrn("regr.ranger", num.trees = 500),
                            ml_m = lrn("classif.ranger", num.trees = 500),
                            n_folds = 5, n_rep = 1, ...) {
  delta_y <- y1 - y0
  dml_data <- DoubleML::double_ml_data_from_matrix(X = covariates, y = delta_y, d = D)
  dml_obj <- DoubleML::DoubleMLIRM$new(dml_data, ml_g = ml_g, ml_m = ml_m,
                                       score = "ATTE", n_folds = n_folds)
  dml_obj$fit()
  list(ATT = dml_obj$coef[1], att.inf.func = dml_obj$psi[, 1, 1])
}

sink("output_DoubleML_did.txt")
lin_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")
options(warn = -1)

for (lin in lin_folders) {
  files <- list.files(lin, pattern = "^iteration_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) { cat("No files in", lin, "- skipping\n"); next }
  niter <- length(files)
  RMSE <- numeric(niter); MAE <- numeric(niter); MAPE <- rep(NA_real_, niter)
  est_rows <- list()
  pb <- progress_bar$new(total = niter, format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick()
    d <- read.csv(files[ii])
    d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
    out <- tryCatch(
      att_gt(yname = "Y", tname = "time", idname = "unit_id",
             gname = "first_treat_period", xformla = ~ X_1+X_2+X_3+X_4+X_5,
             data = d, est_method = doubleml_did_rf),
      error = function(e) NULL)
    if (is.null(out)) next
    groups <- sort(unique(out$group[out$group > 0]))
    errs <- c(); aerr <- c(); ape <- c()
    for (g in groups) {
      post_t <- sort(unique(out$t[out$group == g & out$t >= g]))
      for (t in post_t) {
        sel <- which(out$group == g & out$t == t)
        if (length(sel) == 0) next
        est <- out$att[sel]; se <- out$se[sel]
        truth <- mean(d$CATE[d$first_treat_period == g & d$time == t], na.rm = TRUE)
        sig <- as.integer(abs(est / se) > 1.96)
        errs <- c(errs, est - truth); aerr <- c(aerr, abs(est - truth))
        if (truth != 0) ape <- c(ape, abs((est - truth) / truth))
        est_rows[[length(est_rows) + 1]] <- data.frame(
          iteration = ii - 1, group = g, t = t, k = t - g,
          estimate = est, se = se, true = truth, sig = sig)
      }
    }
    if (length(errs)) { RMSE[ii] <- sqrt(mean(errs^2)); MAE[ii] <- mean(aerr) }
    if (length(ape)) MAPE[ii] <- mean(ape)
  }
  cat(sprintf("\n%s : mean RMSE=%.4f (sd %.4f)  MAE=%.4f  MAPE=%.4f\n",
              lin, mean(RMSE), sd(RMSE), mean(MAE), mean(MAPE, na.rm = TRUE)))
  metrics_df <- data.frame(iteration = 0:(niter - 1), RMSE = RMSE, MAE = MAE, MAPE = MAPE)
  est_df <- if (length(est_rows)) do.call(rbind, est_rows) else data.frame()
  fn <- paste0("DoubleML_did_GATE_and_PValues_", lin, ".xlsx")
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Metrics");  openxlsx::writeData(wb, "Metrics", metrics_df)
    openxlsx::addWorksheet(wb, "Estimates"); openxlsx::writeData(wb, "Estimates", est_df)
    openxlsx::saveWorkbook(wb, fn, overwrite = TRUE); cat("wrote", fn, "\n")
  } else {
    write.csv(metrics_df, paste0("DoubleML_did_Metrics_", lin, ".csv"), row.names = FALSE)
    write.csv(est_df,     paste0("DoubleML_did_Estimates_", lin, ".csv"), row.names = FALSE)
  }
}
sink()
'''

R_SYNTHDID = R_HEADER % "Synthetic DiD (overall ATT; covariate-residualised)" + r'''
library(synthdid)
library(progress)
sink("output_synthdid.txt")

# synthdid requires a single (block) adoption time, so -- as in the original
# study -- eventually-treated units are treated as switching on at the earliest
# post period; the estimand is the overall ATT.
lin_folders <- c("linearity_degree=1", "linearity_degree=2", "linearity_degree=3")
options(warn = -1)

for (lin in lin_folders) {
  files <- list.files(lin, pattern = "^iteration_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) { cat("No files in", lin, "- skipping\n"); next }
  niter <- length(files)
  RMSE <- numeric(niter); MAE <- numeric(niter); MAPE <- rep(NA_real_, niter)
  est_rows <- list()
  pb <- progress_bar$new(total = niter, format = paste0(lin, " [:bar] :current/:total"))
  for (ii in seq_along(files)) {
    pb$tick()
    d <- read.csv(files[ii])
    d$first_treat_period[!is.finite(d$first_treat_period)] <- 0
    ols <- lm(Y ~ X_1+X_2+X_3+X_4+X_5, data = d)
    d$res_Y <- d$Y - predict(ols, newdata = d)
    d$treated <- d$eventually_treated * d$post_treatment
    panel <- data.frame(unit_id = d$unit_id, time = d$time,
                        res_Y = d$res_Y, treated = d$treated)
    est <- NA_real_; se <- NA_real_
    try({
      setup <- panel.matrices(panel)
      tau <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
      est <- as.numeric(tau)
      se <- tryCatch(sqrt(vcov(tau, method = "placebo")[1, 1]), error = function(e) NA_real_)
    }, silent = TRUE)
    truth <- mean(d$CATE[d$D == 1], na.rm = TRUE)
    sig <- as.integer(is.finite(se) & abs(est / se) > 1.96)
    if (is.finite(est)) {
      RMSE[ii] <- abs(est - truth); MAE[ii] <- abs(est - truth)
      if (truth != 0) MAPE[ii] <- abs((est - truth) / truth)
    }
    est_rows[[length(est_rows) + 1]] <- data.frame(
      iteration = ii - 1, estimand = "ATT", estimate = est, se = se,
      true = truth, sig = sig)
  }
  cat(sprintf("\n%s : mean |err|=%.4f (sd %.4f)  MAPE=%.4f\n",
              lin, mean(RMSE), sd(RMSE), mean(MAPE, na.rm = TRUE)))
  metrics_df <- data.frame(iteration = 0:(niter - 1), RMSE = RMSE, MAE = MAE, MAPE = MAPE)
  est_df <- if (length(est_rows)) do.call(rbind, est_rows) else data.frame()
  fn <- paste0("synthdid_GATE_and_PValues_", lin, ".xlsx")
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Metrics");  openxlsx::writeData(wb, "Metrics", metrics_df)
    openxlsx::addWorksheet(wb, "Estimates"); openxlsx::writeData(wb, "Estimates", est_df)
    openxlsx::saveWorkbook(wb, fn, overwrite = TRUE); cat("wrote", fn, "\n")
  } else {
    write.csv(metrics_df, paste0("synthdid_Metrics_", lin, ".csv"), row.names = FALSE)
    write.csv(est_df,     paste0("synthdid_Estimates_", lin, ".csv"), row.names = FALSE)
  }
}
sink()
'''

R_FILES = {"did_dr_new.R": R_DID_DR, "did2s.R": R_DID2S,
           "DoubleML_did.R": R_DOUBLEML, "synthdid.R": R_SYNTHDID}


def main():
    exps = cfg.all_experiments()
    dgps_dir = os.path.join(ROOT, "DGPs")
    bcf_dir = os.path.join(ROOT, "DiD_BCF")
    ols_dir = os.path.join(ROOT, "TWFE")
    rcode_dir = os.path.join(ROOT, "R_code")
    for dd in (dgps_dir, bcf_dir, ols_dir, rcode_dir):
        os.makedirs(dd, exist_ok=True)

    n_data = n_bcf = n_ols = n_r = 0
    for e in exps:
        ctx = dict(SCEN=e.name, WS=e.workstream, DGP=e.dgp,
                   DGPHUMAN=DGP_HUMAN[e.dgp], NOTE=e.note)

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
            with open(os.path.join(bcf_dir, f"DiD_BCF_{e.name}_lin_{d}.ipynb"), "w") as f:
                json.dump(_notebook(bcf_cells), f, indent=1)
            n_bcf += 1

            ols_cells = [
                _cell("markdown", sub(OLS_MD, **c)),
                _cell("code", sub(OLS_SETUP, **c)),
                _cell("code", sub(OLS_RUN, **c)),
                _cell("code", sub(OLS_METRICS, **c)),
            ]
            with open(os.path.join(ols_dir, f"OLS_{e.name}_lin_{d}.ipynb"), "w") as f:
                json.dump(_notebook(ols_cells), f, indent=1)
            n_ols += 1

        # 3) R benchmark scripts
        sdir = os.path.join(rcode_dir, f"{e.name}_datasets")
        os.makedirs(sdir, exist_ok=True)
        for fname, tmpl in R_FILES.items():
            with open(os.path.join(sdir, fname), "w") as f:
                f.write(tmpl.replace("@@SCEN@@", e.name))
            n_r += 1

    print(f"scenarios: {len(exps)}")
    print(f"data_creation scripts: {n_data}")
    print(f"DiD_BCF notebooks:     {n_bcf}")
    print(f"OLS notebooks:         {n_ols}")
    print(f"R scripts:             {n_r}")


if __name__ == "__main__":
    main()
