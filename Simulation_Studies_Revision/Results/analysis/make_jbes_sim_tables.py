#!/usr/bin/env python3
"""Regenerate the JBES submission sim_tables fragments from the archived runs.

Reads (read-only):
  Results/aggregated/metrics_all.csv, surface_all.csv      published JBES run
  Theory_Calibration/results/aggregated_correction/        correction_audit ETE
  Theory_Calibration/results/aggregated_completion/        completion ETE
  Results/goodman_bacon_*.csv                              TWFE decomposition
Writes: JBES_Submission/sim_tables/*.tex (15 files)

A "DiD-BCF (corrected)" row resolves to the reference-fold rerun wherever one
exists (completion first, then correction_audit).  The published same-sample
value is kept with the label "corrected, earlier" only where no rerun exists
(the two surface tables).  Every other row reproduces the published
aggregation, so the diff against the previous fragments shows only intended
changes (new corrected rows, dropped earlier rows, updated coverage notes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]          # Simulation_Studies_Revision
ETE = REPO / "Theory_Calibration"
JBES = REPO.parent.parent / "JBES_Submission"
TAB = JBES / "sim_tables"

PUB_MET = pd.read_csv(REPO / "Results" / "aggregated" / "metrics_all.csv")
PUB_SURF = pd.read_csv(REPO / "Results" / "aggregated" / "surface_all.csv")


def _read_ete(directory: str, name: str) -> pd.DataFrame:
    return pd.read_csv(ETE / "results" / directory / name,
                       keep_default_na=False, na_values=[])


CORR_MET = _read_ete("aggregated_correction", "metrics_mean_median.csv")
COMP_MET = _read_ete("aggregated_completion", "metrics_mean_median.csv")
CORR_REP = pd.read_csv(ETE / "results" / "aggregated_correction"
                       / "combined_per_replication.csv",
                       engine="python", keep_default_na=False, na_values=[],
                       on_bad_lines="skip")
COMP_REP = pd.read_csv(ETE / "results" / "aggregated_completion"
                       / "combined_per_replication.csv",
                       engine="python", keep_default_na=False, na_values=[],
                       on_bad_lines="skip")

REF_EST = "reference_fold_convolution_rf"
REF_METHOD = "reference_fold_convolution"

# Published setting -> ETE design names holding the same DGP (sweep designs
# share the baseline/serial DGP at the listed sizes).
SETTING_DESIGNS = {
    "B1_baseline": ["baseline"],
    "B1_serial_corr": ["serial"],
    "B1_null": ["null"],
    "B1_strong_confounder": ["strong_confounder"],
    "B1_selection_obs": ["selection_obs"],
    "B2_sweep": ["baseline", "baseline_sweep", "baseline_sweep_d3"],
    "B2_sweep_serial": ["serial", "serial_sweep", "serial_sweep_d3"],
    "D_staggered": ["staggered"],
}


def f3(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:0.3f}"


def sd_err_of(bias, rmse) -> float:
    try:
        return float(np.sqrt(max(float(rmse) ** 2 - float(bias) ** 2, 0.0)))
    except (TypeError, ValueError):
        return float("nan")


def raw_ratio(avg_post_sd, raw_sd) -> float:
    try:
        a, r = float(avg_post_sd), float(raw_sd)
    except (TypeError, ValueError):
        return float("nan")
    return a / r if np.isfinite(r) and r > 0 else float("nan")


def pub_row(setting, method, degree, N, etype, eid):
    sub = PUB_MET[(PUB_MET["setting"] == setting)
                  & (PUB_MET["method"] == method)
                  & (PUB_MET["linearity_degree"] == int(degree))
                  & (PUB_MET["N"] == int(N))
                  & (PUB_MET["estimand_type"] == etype)
                  & (PUB_MET["estimand_id"] == eid)]
    if sub.empty:
        return None
    assert len(sub) == 1, (setting, method, degree, N, etype, eid)
    return sub.iloc[0].to_dict()


def ref_row(setting, degree, N, etype, eid, task_est=REF_EST,
            method=REF_METHOD, summary="mean"):
    """Reference-fold metrics row, completion first, then correction_audit."""
    for frame in (COMP_MET, CORR_MET):
        sub = frame[(frame["design"].isin(SETTING_DESIGNS[setting]))
                    & (frame["degree"] == int(degree)) & (frame["N"] == int(N))
                    & (frame["task_estimator"] == task_est)
                    & (frame["method"] == method)
                    & (frame["point_summary"] == summary)
                    & (frame["estimand_type"] == etype)
                    & (frame["estimand_id"] == eid)]
        if len(sub) == 1:
            return sub.iloc[0].to_dict()
        assert len(sub) == 0, (setting, degree, N, etype, eid)
    return None


def ref_size(designs, degree, N, method, etype, eid,
             task_est=REF_EST):
    """Two-sided 5% rejection rate from per-replication rows.

    Completion first, then the correction_audit archive (whose null design only
    covers d=1,2); both store the one-tail minimum in ``p_bayes_tail_min``.
    The estimator filter matters: several audit variants share one method.
    """
    for perrep in (COMP_REP, CORR_REP):
        sub = perrep[(perrep["design"].isin(designs))
                     & (perrep["degree"] == int(degree))
                     & (perrep["N"] == int(N))
                     & (perrep["task_estimator"] == task_est)
                     & (perrep["method"] == method)
                     & (perrep["estimand_type"] == etype)
                     & (perrep["estimand_id"] == eid)]
        p = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
        p = p[p.notna()]
        if len(p):
            r = float((p < 0.025).mean())
            return r, len(p)
    return float("nan"), 0


def mcse_rate(r, n) -> float:
    try:
        return float(np.sqrt(float(r) * (1.0 - float(r)) / int(n)))
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Scaffolding: longtable wrapper shared by the regenerated fragments.
# --------------------------------------------------------------------------- #
def longtable(path, colspec, caption, label, header, body, notes):
    ncols = len(colspec.replace("|", ""))
    lines = [r"% Auto-generated by Results/analysis/make_jbes_sim_tables.py -- do not edit by hand.",
             r"{\footnotesize",
             rf"\begin{{longtable}}{{{colspec}}}",
             rf"\caption{{{caption}}}\label{{{label}}}\\",
             r"\toprule", header, r"\midrule", r"\endfirsthead",
             (r"\multicolumn{" + str(ncols) + r"}{l}{\emph{\footnotesize Table \ref{"
              + label + r"} continued from the previous page.}}\\"),
             r"\toprule", header, r"\midrule", r"\endhead",
             r"\midrule",
             (r"\multicolumn{" + str(ncols) + r"}{r}{\emph{\footnotesize continued on the next page}}\\"),
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    lines += body
    lines += [r"\end{longtable}"]
    if notes:
        lines += [r"\noindent " + notes + r"\par\vspace{6pt}"]
    lines += ["}"]
    Path(path).write_text("\n".join(lines) + "\n")
    print("  ->", Path(path).name)


B1_HEAD = (r"Estimator & $d$ & Bias & SD(err) & RMSE & Cov.90 & Cov.95 & Len.95 & "
           r"$\overline{\mathrm{sd}}/\mathrm{SD}$ \\")
B1_NOTE = (r"\emph{Note.} $N=200$, 100 replications ($200$ for the null $d=3$ rerun), "
           r"$d$ is the linearity degree. Bias and RMSE are Monte-Carlo moments of "
           r"the estimation error and SD(err) is its standard deviation, "
           r"$\sqrt{\text{RMSE}^2-\text{Bias}^2}$ (the target itself varies by "
           r"replication, so the raw dispersion of the estimate is not the error "
           r"SD); Cov.90/95 are credible/confidence-interval coverage rates; Len.95 "
           r"the average 95\% interval length; $\overline{\mathrm{sd}}/\mathrm{SD}$ "
           r"the calibration ratio of the average reported posterior/standard "
           r"error to the raw Monte-Carlo SD of the estimates (1 is perfect "
           r"calibration, $<1$ over-confident). Rows labeled ``earlier'' come from "
           r"the same-sample implementation superseded by the reference-fold runs "
           r"and are not produced by the current corrected estimator; entries "
           r"labeled ``corrected'' are the fixed-$K$ reference-fold convolution "
           r"with random-forest pilots (Algorithm~2), which now covers every "
           r"degree and design shown.")
B1_BLOCKS = ["B1_baseline", "B1_strong_confounder", "B1_serial_corr",
             "B1_selection_obs"]
B1_SCEN = {"B1_baseline": "B1\\_baseline",
           "B1_strong_confounder": "B1\\_strong\\_confounder",
           "B1_serial_corr": "B1\\_serial\\_corr",
           "B1_selection_obs": "B1\\_selection\\_obs"}


def tab_b1(etype, eid, fname, label, caption, methods):
    body = []
    for s in B1_BLOCKS:
        body.append(r"\multicolumn{9}{l}{\textit{" + B1_SCEN[s] + r"}} \\")
        for d in (1, 2, 3):
            rows = []
            for meth, pub_method in methods:
                if meth == "DiD-BCF (corrected)":
                    m = ref_row(s, d, 200, etype, eid)
                    if m is not None:
                        lab = "DiD-BCF (corrected)"
                        # ETE rows report the aggregation's own error SD
                        # (ddof=1); the published rows below use the
                        # sqrt(RMSE^2-bias^2) recipe, whose raw-spread column
                        # differs from it.
                        cells = [m["bias"], m["emp_sd"], m["rmse"],
                                 m["cover90"], m["cover95"], m["len95"],
                                 raw_ratio(m["avg_post_sd"], m["raw_sd_est"])]
                    else:
                        m = pub_row(s, "corrected", d, 200, etype, eid)
                        assert m is not None, (s, d)
                        lab = "DiD-BCF (corrected, earlier)"
                        cells = [m["bias"],
                                 sd_err_of(m["bias"], m["rmse"]), m["rmse"],
                                 m["cover90"], m["cover95"], m["len95"],
                                 m["sd_ratio"]]
                else:
                    m = pub_row(s, pub_method, d, 200, etype, eid)
                    assert m is not None, (s, pub_method, d)
                    lab, cells = meth, [m["bias"], sd_err_of(m["bias"], m["rmse"]),
                                        m["rmse"], m["cover90"], m["cover95"],
                                        m["len95"], m["sd_ratio"]]
                rows.append((lab, d, cells))
            for lab, dd, cells in rows:
                body.append(f"\\quad {lab} & {dd} & " +
                            " & ".join(f3(c) for c in cells[:7]) + r" \\")
        body.append(r"\addlinespace")
    longtable(TAB / fname, "lr" + "r" * 7, caption, label, B1_HEAD, body,
              B1_NOTE)


ATT_METHODS = [("DiD-BCF (plain)", "plain"), ("DiD-BCF (corrected)", "corrected"),
               ("TWFE", "twfe"), ("Synthetic DiD", "synthdid")]
GATT_METHODS = [("DiD-BCF (plain)", "plain"), ("DiD-BCF (corrected)", "corrected"),
                ("Callaway--Sant'Anna", "did_dr"),
                ("DoubleML DR-DiD", "doubleml"), ("grf-DiD (Wang)", "wang")]


def tab_null():
    etypes = [("ATT", "ATT"), ("ES", "k=0"), ("GATT", "g=4_t=4")]
    method_sets = {"ATT": [("DiD-BCF (plain)", "plain"),
                           ("DiD-BCF (corrected)", "corrected"),
                           ("TWFE", "twfe"), ("Synthetic DiD", "synthdid")],
                   "ES": [("DiD-BCF (plain)", "plain"),
                          ("DiD-BCF (corrected)", "corrected"),
                          ("TWFE", "twfe"), ("Gardner (did2s)", "did2s")],
                   "GATT": [("DiD-BCF (plain)", "plain"),
                            ("DiD-BCF (corrected)", "corrected"),
                            ("Callaway--Sant'Anna", "did_dr"),
                            ("DoubleML DR-DiD", "doubleml"),
                            ("grf-DiD (Wang)", "wang")]}
    body = []
    for et, eid in etypes:
        eid_tex = eid.replace("_", r"\_").replace("=", "$=$")
        body.append(r"\multicolumn{8}{l}{\textit{" + et + ", " + eid_tex + r"}} \\")
        for d in (1, 2, 3):
            for lab, pub_method in method_sets[et]:
                if lab == "DiD-BCF (corrected)":
                    m = ref_row("B1_null", d, 200, et, eid)
                    assert m is not None, ("null", d, et)
                    r, n = ref_size(["null"], d, 200,
                                    "reference_fold_convolution", et, eid)
                    cells = [d, m["bias"], m["emp_sd"],
                             m["cover90"], m["cover95"], r, mcse_rate(r, n)]
                else:
                    m = pub_row("B1_null", pub_method, d, 200, et, eid)
                    assert m is not None, ("null", pub_method, d, et)
                    cells = [d, m["bias"], sd_err_of(m["bias"], m["rmse"]),
                             m["cover90"], m["cover95"], m["reject05"],
                             m["mcse_reject05"]]
                body.append("\\quad " + lab + " & " + str(int(cells[0])) +
                            " & " + " & ".join(f3(c) for c in cells[1:]) + r" \\")
        body.append(r"\addlinespace")
    head = (r"Estimator & $d$ & Bias & SD(err) & Cov.90 & Cov.95 & "
            r"Size (5\%) & MCSE \\")
    notes = (r"\emph{Note.} $N=200$; 100 replications for the model-based "
             r"estimators and 200 for TWFE (the corrected $d=3$ rerun also uses "
             r"200 replications). A nominal 5\% test should reject in $0.05$ of "
             r"replications; MCSE is the Monte-Carlo standard error of that "
             r"rejection rate. Corrected entries at every degree are the fixed-$K$ "
             r"reference-fold convolution with random-forest pilots (Algorithm~2); "
             r"at $d=3$ it rejects at $0.10$--$0.13$ across ATT, event-study, and "
             r"cohort-time cells (about two to three Monte-Carlo standard errors "
             r"above nominal), against $0.03$--$0.05$ for the raw fit. The same "
             r"reference-fold runs reject at $0.04$--$0.09$ across degrees and "
             r"estimands at $N=800$.")
    longtable(TAB / "tab_null.tex", "lr" + "r" * 6,
              r"Behaviour under the sharp null $\tau\equiv 0$ (scenario "
              r"\texttt{B1\_null}): bias, dispersion, coverage and the empirical size "
              r"of the two-sided 5\% test.", "tab:null", head, body, notes)


SWEEP_NS = [50, 100, 200, 400, 800]
# DoubleML and grf-DiD were also run at N=1600; DiD-BCF stops at N=800.
SWEEP_NS_BENCH = [50, 100, 200, 400, 800, 1600]
SWEEP_METHODS = [("DiD-BCF (plain)", "plain"),
                 ("DiD-BCF (corrected)", "corrected"),
                 ("Callaway--Sant'Anna", "did_dr"),
                 ("DoubleML DR-DiD", "doubleml"), ("grf-DiD (Wang)", "wang")]


def tab_sweep():
    body = []
    for s, it in (("B2_sweep", r"\textit{B2\_sweep}"),
                  ("B2_sweep_serial", r"\textit{B2\_sweep\_serial}")):
        body.append(r"\multicolumn{8}{l}{" + it + r"} \\")
        for lab, pub_method in SWEEP_METHODS:
            ns = (SWEEP_NS_BENCH if lab in ("DoubleML DR-DiD",
                                            "grf-DiD (Wang)") else SWEEP_NS)
            for N in ns:
                if lab == "DiD-BCF (corrected)":
                    m = ref_row(s, 1, N, "GATT", "g=4_t=4")
                    assert m is not None, (s, N)
                    cells = [N, m["bias"], m["emp_sd"],
                             m["rmse"], m["cover95"], m["len95"],
                             raw_ratio(m["avg_post_sd"], m["raw_sd_est"])]
                else:
                    m = pub_row(s, pub_method, 1, N, "GATT", "g=4_t=4")
                    assert m is not None, (s, pub_method, N)
                    cells = [N, m["bias"], sd_err_of(m["bias"], m["rmse"]),
                             m["rmse"], m["cover95"], m["len95"], m["sd_ratio"]]
                body.append("\\quad " + lab + " & " + str(int(cells[0])) +
                            " & " + " & ".join(f3(c) for c in cells[1:]) + r" \\")
            body.append(r"\addlinespace[2pt]")
        body.append(r"\addlinespace")
    head = (r"Estimator & $N$ & Bias & SD(err) & RMSE & Cov.95 & Len.95 & "
            r"$\overline{\mathrm{sd}}/\mathrm{SD}$ \\")
    notes = (r"\emph{Note.} 100 replications per cell. A consistent estimator "
             r"has $|\text{Bias}|\to 0$ and $\text{SD}\propto N^{-1/2}$; the "
             r"$\sqrt{N}$-scaled diagnostics are plotted in Figure~\ref{fig:sqrtn}. "
             r"Entries labeled ``corrected'' are the fixed-$K$ reference-fold "
             r"convolution with random-forest pilots (Algorithm~2), reported at "
             r"every size.")
    longtable(TAB / "tab_sweep.tex", "lr" + "r" * 6,
              r"Sample-size sweep for $\mathrm{GATT}(g{=}4,t{=}4)$ at linearity degree "
              r"$d=1$: the estimand every method reports, so all estimators are "
              r"directly comparable across $N$.", "tab:sweep", head, body, notes)


def tab_sweep_deg():
    body = []
    for s, it in (("B2_sweep", r"\textit{B2\_sweep}"),
                  ("B2_sweep_serial", r"\textit{B2\_sweep\_serial}")):
        body.append(r"\multicolumn{9}{l}{" + it + r"} \\")
        for lab, pub_method in SWEEP_METHODS:
            ns = (SWEEP_NS_BENCH if lab in ("DoubleML DR-DiD",
                                            "grf-DiD (Wang)") else SWEEP_NS)
            for N in ns:
                for d in (2, 3):
                    if lab == "DiD-BCF (corrected)":
                        m = ref_row(s, d, N, "GATT", "g=4_t=4")
                        assert m is not None, (s, d, N)
                        cells = [N, d, m["bias"], m["emp_sd"], m["rmse"],
                                 m["cover95"], m["len95"],
                                 raw_ratio(m["avg_post_sd"], m["raw_sd_est"])]
                    else:
                        m = pub_row(s, pub_method, d, N, "GATT", "g=4_t=4")
                        assert m is not None, (s, pub_method, d, N)
                        cells = [N, d, m["bias"],
                                 sd_err_of(m["bias"], m["rmse"]), m["rmse"],
                                 m["cover95"], m["len95"], m["sd_ratio"]]
                    body.append("\\quad " + lab + " & " +
                                str(int(cells[0])) + " & " + str(int(cells[1])) +
                                " & " + " & ".join(f3(c) for c in cells[2:])
                                + r" \\")
            body.append(r"\addlinespace[2pt]")
        body.append(r"\addlinespace")
    head = (r"Estimator & $N$ & $d$ & Bias & SD(err) & RMSE & Cov.95 & Len.95 & "
            r"$\overline{\mathrm{sd}}/\mathrm{SD}$ \\")
    notes = (r"\emph{Note.} 100 replications per cell; the $d=1$ counterpart is "
             r"Table \ref{tab:sweep}. TWFE and Synthetic DiD use no covariates, "
             r"so their entries are invariant to $d$ by construction. DoubleML "
             r"DR-DiD is available at all 5 panel sizes in both sweeps. Entries "
             r"labeled ``corrected'' are the fixed-$K$ reference-fold convolution "
             r"with random-forest pilots (Algorithm~2) at every size and degree.")
    longtable(TAB / "tab_sweep_deg.tex", "lrr" + "r" * 6,
              r"Sample-size sweep for $\mathrm{GATT}(g{=}4,t{=}4)$ at the nonlinear "
              r"degrees $d=2$ and $d=3$.", "tab:sweepdeg", head, body, notes)


def tab_smalln():
    NS = [50, 100, 200]
    body = []
    for s, it in (("B2_sweep", r"\textit{B2 ordinary}"),
                  ("B2_sweep_serial", r"\textit{B2 serial}")):
        body.append(r"\multicolumn{14}{l}{" + it + r"} \\")
        for target, et, eid in ((r"$\gatt(4,4)$", "GATT", "g=4_t=4"),
                                (r"$\att$", "ATT", "ATT")):
            rows = []
            order = (["DiD-BCF (plain)", "DiD-BCF (ref. fold)"]
                     + (["Callaway--Sant'Anna", "DoubleML DR-DiD",
                         "grf-DiD (Wang)"] if et == "GATT"
                        else ["TWFE", "Synthetic DiD"]))
            for lab in order:
                # (value, decimals, suffix) triples per N column group.
                cells = []
                for N in NS:
                    if lab == "DiD-BCF (plain)":
                        m = pub_row(s, "plain", 1, N, et, eid)
                        assert m is not None, (s, N, et)
                        star = (r"$^{\dagger}$"
                                if float(m["retention"]) < 0.999 else "")
                        cells += [(m["bias"], 3, ""), (m["rmse"], 3, ""),
                                  (m["cover95"], 2, star),
                                  (m["sd_ratio"], 2, "")]
                    elif lab == "DiD-BCF (ref. fold)":
                        m = ref_row(s, 1, N, et, eid)
                        assert m is not None, (s, N, et)
                        cells += [(m["bias"], 3, ""), (m["rmse"], 3, ""),
                                  (m["cover95"], 2, ""),
                                  (raw_ratio(m["avg_post_sd"],
                                             m["raw_sd_est"]), 2, "")]
                    else:
                        bench = {"Callaway--Sant'Anna": "did_dr",
                                 "DoubleML DR-DiD": "doubleml",
                                 "grf-DiD (Wang)": "wang", "TWFE": "twfe",
                                 "Synthetic DiD": "synthdid"}[lab]
                        m = pub_row(s, bench, 1, N, et, eid)
                        if m is None:
                            cells += [(np.nan, 3, "")] * 4
                        else:
                            star = (r"$^{\dagger}$"
                                    if float(m["retention"]) < 0.999 else "")
                            cells += [(m["bias"], 3, ""), (m["rmse"], 3, ""),
                                      (m["cover95"], 2, star),
                                      (m["sd_ratio"], 2, "")]
                out = [("--" if pd.isna(v) else f"{float(v):.{n}f}{sfx}")
                       for v, n, sfx in cells]
                rows.append((lab, out))
            for i, (lab, out) in enumerate(rows):
                first = target if i == 0 else " "
                body.append(" & ".join([first, lab] + out) + r" \\")
            body.append(r"\addlinespace[2pt]")
        body.append(r"\addlinespace")
    _write_smalln_table(body)


def _write_smalln_table(body):
    lines = [r"% Auto-generated by Results/analysis/make_jbes_sim_tables.py.",
             r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{2.6pt}",
             r"\caption{The small-panel end of the B2 sweeps at linearity degree "
             r"$d=1$. Bias, RMSE, 95\% coverage and the calibration ratio "
             r"$\overline{\mathrm{sd}}/\mathrm{SD}$ for the cohort-time target "
             r"$\mathrm{GATT}(4,4)$ and the pooled $\mathrm{ATT}$.}",
             r"\label{tab:smalln}",
             r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
             ("  &   & " + r"\multicolumn{4}{c}{$N=50$}" + " & "
              + r"\multicolumn{4}{c}{$N=100$}" + " & "
              + r"\multicolumn{4}{c}{$N=200$}" + r" \\"),
             (r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} "
              r"\cmidrule(lr){11-14}"),
             ("Target & Estimator & Bias & RMSE & Cov. & "
              r"$\overline{\mathrm{sd}}/\mathrm{SD}$ & Bias & RMSE & Cov. & "
              r"$\overline{\mathrm{sd}}/\mathrm{SD}$ & Bias & RMSE & Cov. & "
              r"$\overline{\mathrm{sd}}/\mathrm{SD}$ \\"),
             r"\midrule"]
    lines += body
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize \emph{Note.} 100 replications per "
              r"cell. $^{\dagger}$computed on the replications in which the "
              r"estimator returned an estimate. TWFE was not run on the "
              r"sweeps below $N=200$. Gardner reports event-study coefficients "
              r"only and synthetic DiD a pooled ATT only, so neither appears "
              r"against every target. The ``ref.\ fold'' rows are the fixed-$K$ "
              r"reference-fold convolution with random-forest pilots "
              r"(Algorithm~2, the current corrected estimator).",
              r"\end{table}"]
    (TAB / "tab_smalln.tex").write_text("\n".join(lines) + "\n")
    print("  -> tab_smalln.tex")


def tab_surface():
    blocks = ["B1_baseline", "B1_strong_confounder", "B1_serial_corr",
              "B1_selection_obs", "D_staggered"]
    scen = {s: s.replace("_", r"\_") for s in blocks}
    order = [("DiD-BCF (plain)", "plain"), ("DiD-BCF (corrected, earlier)", "corrected"),
             ("TWFE", "twfe"), ("Callaway--Sant'Anna", "did_dr"),
             ("Gardner (did2s)", "did2s"), ("DoubleML DR-DiD", "doubleml"),
             ("Synthetic DiD", "synthdid"), ("grf-DiD (Wang)", "wang")]
    body = []
    for s in blocks:
        body.append(r"\multicolumn{7}{l}{\textit{" + scen[s] + r"}} \\")
        for d in (1, 2, 3):
            for lab, meth in order:
                m = pub_row_surface(s, meth, d)
                assert m is not None, (s, meth, d)
                body.append(
                    f"\\quad {lab} & {d} & "
                    f"{f3(m['surf_rmse_mean'])} ({f3(m['surf_rmse_sd'])}) & "
                    f"{f3(m['surf_mae_mean'])} & {f3(m['surf_mape_mean'])} & "
                    f"{f3(m['surf_cover95_mean'])} & {f3(m['surf_len95_mean'])} \\\\")
        body.append(r"\addlinespace")
    # The archived fragment carries one extra trailing \addlinespace.
    body.append(r"\addlinespace")
    head = r"Estimator & $d$ & RMSE (SD) & MAE & MAPE & Cov.95 & Len.95 \\"
    longtable(TAB / "tab_surface.tex", "lr" + "r" * 5,
              r"CATT-surface accuracy: error of the estimated individual treatment "
              r"effect over the treated observations, averaged across replications "
              r"(Monte-Carlo SD of the within-replication RMSE in parentheses).",
              "tab:surface", head, body,
              r"\emph{Note.} $N=200$, 100 replications. Estimators that produce "
              r"only cell-level effects have their $\mathrm{GATT}(g,t)$ (or "
              r"event-study, or overall ATT) estimate broadcast to every treated "
              r"observation in the cell, which is the natural best case for a "
              r"homogeneous-effect estimator. Cov.95 is \emph{pointwise} coverage "
              r"of the individual effect. Corrected surface entries come from the "
              r"earlier same-sample post-processing implementation and are not "
              r"produced by the reference-fold convolution, which was not rerun "
              r"for individual-surface summaries.")


def pub_row_surface(setting, method, degree):
    sub = PUB_SURF[(PUB_SURF["setting"] == setting)
                   & (PUB_SURF["method"] == method)
                   & (PUB_SURF["linearity_degree"] == int(degree))
                   & (PUB_SURF["N"] == 200)]
    if sub.empty:
        return None
    assert len(sub) == 1, (setting, method, degree)
    return sub.iloc[0].to_dict()


def tab_smalln_surface():
    NS = [50, 100, 200]
    order = [("DiD-BCF (plain)", "plain"),
             ("DiD-BCF (corrected, earlier)", "corrected"),
             ("TWFE", "twfe"), ("Callaway--Sant'Anna", "did_dr"),
             ("Gardner (did2s)", "did2s"), ("DoubleML DR-DiD", "doubleml"),
             ("Synthetic DiD", "synthdid"), ("grf-DiD (Wang)", "wang")]
    lines = [r"% Auto-generated by Results/analysis/make_jbes_sim_tables.py.",
             r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Individual-level CATT surface at the small end of the "
             r"B2 sweeps, $d=1$. Within-replication RMSE and MAPE over the "
             r"treated observations, and pointwise 95\% coverage, averaged "
             r"across replications.}",
             r"\label{tab:smallnsurface}",
             r"\begin{tabular}{ll" + "rrr" * len(NS) + "}", r"\toprule",
             " & ".join([" ", " "] + [r"\multicolumn{3}{c}{$N=%d$}" % n
                                      for n in NS]) + r" \\",
             " ".join(r"\cmidrule(lr){%d-%d}" % (3 + 3 * i, 5 + 3 * i)
                      for i in range(len(NS))),
             " & ".join(["Design", "Estimator"] +
                        ["RMSE", "MAPE", "Cov."] * len(NS)) + r" \\",
             r"\midrule"]
    for setting, it in (("B2_sweep", "B2 ordinary"),
                        ("B2_sweep_serial", "B2 serial")):
        blk = PUB_SURF[(PUB_SURF["linearity_degree"] == 1)
                       & PUB_SURF["N"].isin(NS)
                       & (PUB_SURF["setting"] == setting)]
        first = True
        for lab, meth in order:
            g = blk[blk["method"] == meth]
            if g.empty:
                continue
            cells = []
            for N in NS:
                r = g[g["N"] == N]
                if r.empty:
                    cells += ["--"] * 3
                    continue
                r = r.iloc[0]
                cells += [f3(r["surf_rmse_mean"]), f3(r["surf_mape_mean"]),
                          ("--" if pd.isna(r["surf_cover95_mean"])
                           else f"{r['surf_cover95_mean']:.2f}")]
            lines.append(" & ".join([it if first else " ", lab] + cells)
                         + r" \\")
            first = False
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize \emph{Note.} Every estimator other "
              r"than DiD-BCF reports a single number per cohort-time cell, so "
              r"its surface is that number broadcast to the individuals in the "
              r"cell; the comparison measures what is lost by not modelling "
              r"individual heterogeneity. Corrected surface entries come from "
              r"the earlier same-sample post-processing implementation and are "
              r"not produced by the reference-fold convolution, which was not "
              r"rerun for individual-surface summaries.",
              r"\end{table}"]
    (TAB / "tab_smalln_surface.tex").write_text("\n".join(lines) + "\n")
    print("  -> tab_smalln_surface.tex")


def tab_bacon():
    comp = pd.read_csv(REPO / "Results" / "goodman_bacon_components_D_staggered.csv")
    summ = pd.read_csv(REPO / "Results" / "goodman_bacon_summary_D_staggered.csv")
    lines = [r"% Auto-generated by Results/analysis/make_jbes_sim_tables.py.",
             r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\caption{Goodman-Bacon (2021) decomposition of the TWFE estimate "
             r"under staggered adoption with cohort- and event-time-varying "
             r"effects, averaged over 500 replications.}",
             r"\label{tab:bacon}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Comparison type & Weight & $\widehat{DD}$ & Contribution \\",
             r"\midrule",
             r"\multicolumn{4}{l}{\textit{D\_staggered}} \\"]
    for _, r in comp.iterrows():
        lines.append(f"\\quad {r['type'].replace('_', ' ')} & {f3(r['weight'])} & "
                     f"{f3(r['dd'])} & {f3(r['contribution'])} \\\\")
    srow = summ.iloc[0]
    lines.append(r"\quad \textit{TWFE estimate} & \multicolumn{3}{r}{"
                 f"{f3(srow['twfe'])}" r"} \\")
    lines.append(r"\quad \textit{true ATT} & \multicolumn{3}{r}{"
                 f"{f3(srow['att_true'])}" r"} \\")
    lines.append(r"\quad \textit{TWFE bias} & \multicolumn{3}{r}{"
                 f"{f3(srow['twfe_bias'])}" r"} \\")
    lines.append(r"\quad \textit{weight on already-treated comparisons} & "
                 r"\multicolumn{3}{r}{" f"{f3(srow['weight_already_treated'])}" r"} \\")
    lines += [r"\addlinespace", r"\addlinespace", r"\bottomrule", r"\end{tabular}",
              r"\begin{minipage}{\linewidth}\vspace{2pt}\footnotesize",
              r"\emph{Note.} ``Earlier vs.\ later'' and ``later vs.\ earlier'' are "
              r"the two-by-two comparisons that use already-treated units as "
              r"controls; their combined weight is reported in the last line of "
              r"each block. Weights are normalised over all two-by-two "
              r"comparisons including those against the never-treated group.",
              r"\end{minipage}", r"\end{table}"]
    (TAB / "tab_bacon.tex").write_text("\n".join(lines) + "\n")
    print("  -> tab_bacon.tex")


def tab_staggered_es():
    body = []
    body.append(r"\multicolumn{7}{l}{\textit{D\_staggered}} \\")
    for meth, pub_method in [("DiD-BCF (plain)", "plain"),
                             ("DiD-BCF (corrected)", "corrected"),
                             ("TWFE", "twfe"), ("Gardner (did2s)", "did2s")]:
        for k in (0, 1, 2, 3):
            if meth == "DiD-BCF (corrected)":
                m = ref_row("D_staggered", 1, 200, "ES", f"k={k}")
                assert m is not None, k
                cells = [k, m["mean_true"], m["bias"],
                         m["emp_sd"], m["rmse"], m["cover95"]]
            else:
                m = pub_row("D_staggered", pub_method, 1, 200, "ES", f"k={k}")
                assert m is not None, (pub_method, k)
                cells = [k, m["mean_true"], m["bias"],
                         sd_err_of(m["bias"], m["rmse"]), m["rmse"], m["cover95"]]
            body.append("\\quad " + meth + " & " +
                        " & ".join([str(cells[0])] + [f3(c) for c in cells[1:]])
                        + r" \\")
        body.append(r"\addlinespace[2pt]")
    body.append(r"\addlinespace")
    body.append(r"\addlinespace")
    head = (r"Estimator & $k$ & True ATT($k$) & Bias & SD(err) & RMSE & Cov.95 \\")
    notes = (r"\emph{Note.} The true effect grows with exposure $k$ and differs "
             r"across cohorts, so ATT($k$) is not constant. Callaway--Sant'Anna, "
             r"DoubleML and grf-DiD report cohort-by-time rather than event-time "
             r"effects; they are summarised over cells in Table "
             r"\ref{tab:staggered_gatt} and given cell by cell, ordered by event "
             r"time, in Table \ref{tab:staggeredcells}. The nonlinear degrees are "
             r"in Table \ref{tab:staggereddeg}. The corrected entries are the "
             r"fixed-$K$ reference-fold convolution with random-forest pilots "
             r"(Algorithm~2).")
    longtable(TAB / "tab_staggered.tex", "lr" + "r" * 5,
              r"Event-study estimands under staggered adoption with cohort- and "
              r"event-time-varying effects ($d=1$, $N=200$, 100 replications).",
              "tab:staggered", head, body, notes)


def tab_staggered_gatt():
    order = [("DiD-BCF (plain)", "plain"), ("DiD-BCF (corrected)", "corrected"),
             ("Callaway--Sant'Anna", "did_dr"),
             ("DoubleML DR-DiD", "doubleml"), ("grf-DiD (Wang)", "wang")]
    body = [r"\multicolumn{8}{l}{\textit{D\_staggered}} \\"]
    for d in (1, 2, 3):
        agg_rows = {}
        for lab, pub_method in order:
            if lab == "DiD-BCF (corrected)":
                cells = [(f"g={g}_t={t}")
                         for g in (4, 5, 6) for t in range(g, 8)]
                vals = []
                for eid in cells:
                    m = ref_row("D_staggered", d, 200, "GATT", eid)
                    assert m is not None, (d, eid)
                    vals.append(m)
            else:
                sub = PUB_MET[(PUB_MET["setting"] == "D_staggered")
                              & (PUB_MET["method"] == pub_method)
                              & (PUB_MET["linearity_degree"] == d)
                              & (PUB_MET["estimand_type"] == "GATT")]
                vals = [r.to_dict() for _, r in sub.iterrows()]
                assert len(vals) == 9, (pub_method, d, len(vals))
            n = len(vals)
            mean_true = float(np.mean([v["mean_true"] for v in vals]))
            bias = float(np.mean([v["bias"] for v in vals]))
            if lab == "DiD-BCF (corrected)":
                emp = float(np.mean([v["emp_sd"] for v in vals]))
            else:
                emp = float(np.mean([sd_err_of(v["bias"], v["rmse"])
                                     for v in vals]))
            rmse = float(np.mean([v["rmse"] for v in vals]))
            cov = float(np.mean([v["cover95"] for v in vals]))
            body.append(f"\\quad {lab} & {d} & {n} & {f3(mean_true)} & "
                        f"{f3(bias)} & {f3(emp)} & {f3(rmse)} & {f3(cov)} \\\\")
        body.append(r"\addlinespace[2pt]")
    body += [r"\addlinespace", r"\addlinespace"]
    head = [r"Estimator & $d$ & Cells & True & Bias & SD(err) & RMSE & Cov.95 \\"]
    longtable(TAB / "tab_staggered_gatt.tex", "lrr" + "r" * 5,
              r"Cohort-by-time estimands $\mathrm{GATT}(g,t)$ under staggered "
              r"adoption, averaged over the estimated cells ($N=200$, all three "
              r"linearity degrees).", "tab:staggered_gatt", head[0], body,
              r"\emph{Note.} ``Cells'' is the number of $(g,t)$ cells each "
              r"estimator reports; all quantities are unweighted averages across "
              r"those cells, which mixes cohorts and event times. The cell-by-cell "
              r"version at $d=1$ is Table \ref{tab:staggeredcells}. The corrected "
              r"entries are the fixed-$K$ reference-fold convolution with "
              r"random-forest pilots (Algorithm~2).")


def tab_staggered_cells():
    order = [("DiD-BCF (plain)", "plain"), ("DiD-BCF (corrected)", "corrected"),
             ("Callaway--Sant'Anna", "did_dr"),
             ("DoubleML DR-DiD", "doubleml"), ("grf-DiD (Wang)", "wang")]
    body = [r"\multicolumn{8}{l}{\textit{D\_staggered}} \\"]
    cells = [((4, 4), 0), ((5, 5), 0), ((6, 6), 0),
             ((4, 5), 1), ((5, 6), 1), ((6, 7), 1),
             ((4, 6), 2), ((5, 7), 2), ((4, 7), 3)]
    for (g, t), k in cells:
        body.append(r"\multicolumn{8}{l}{\quad $k=" + str(k) + r"$, cohort $g="
                    + str(g) + r"$} \\")
        for lab, pub_method in order:
            if lab == "DiD-BCF (corrected)":
                m = ref_row("D_staggered", 1, 200, "GATT", f"g={g}_t={t}")
                assert m is not None, (g, t)
                cells_out = [g, t, k, m["mean_true"], m["bias"], m["rmse"],
                             m["cover95"]]
            else:
                m = pub_row("D_staggered", pub_method, 1, 200, "GATT",
                            f"g={g}_t={t}")
                assert m is not None, (pub_method, g, t)
                cells_out = [g, t, k, m["mean_true"], m["bias"], m["rmse"],
                             m["cover95"]]
            body.append("\\quad\\quad " + lab + " & " +
                        " & ".join([str(cells_out[0]), str(cells_out[1]),
                                    str(cells_out[2])]
                                   + [f3(c) for c in cells_out[3:]]) + r" \\")
        body.append(r"\addlinespace[2pt]")
    body += [r"\addlinespace", r"\addlinespace"]
    head = [r"Estimator & $g$ & $t$ & $k$ & True & Bias & RMSE & Cov.95 \\"]
    longtable(TAB / "tab_staggered_cells.tex", "lrrr" + "r" * 4,
              r"Cell-by-cell cohort-time estimands $\mathrm{GATT}(g,t)$ under staggered "
              r"adoption, grouped by event time $k=t-g$ ($d=1$, $N=200$, 100 "
              r"replications).", "tab:staggeredcells", head[0], body,
              r"\emph{Note.} Every estimator here reports the same $(g,t)$ cell, "
              r"so the comparison is exact and requires no aggregation. TWFE and "
              r"Gardner are absent because they report only pooled or event-time "
              r"coefficients; see Table \ref{tab:staggered}. Reading down a fixed "
              r"$k$ block shows how each estimator degrades as the cohort adopts "
              r"later and its control group shrinks. The corrected entries are "
              r"the fixed-$K$ reference-fold convolution with random-forest "
              r"pilots (Algorithm~2).")


def tab_staggered_deg():
    body = [r"\multicolumn{8}{l}{\textit{D\_staggered}} \\"]
    for meth, pub_method in [("DiD-BCF (plain)", "plain"),
                             ("DiD-BCF (corrected)", "corrected"),
                             ("TWFE", "twfe"), ("Gardner (did2s)", "did2s")]:
        for d in (2, 3):
            for k in (0, 1, 2, 3):
                if meth == "DiD-BCF (corrected)":
                    m = ref_row("D_staggered", d, 200, "ES", f"k={k}")
                    assert m is not None, (d, k)
                    cells = [d, k, m["mean_true"], m["bias"],
                             m["emp_sd"], m["rmse"],
                             m["cover95"]]
                else:
                    m = pub_row("D_staggered", pub_method, d, 200, "ES",
                                f"k={k}")
                    assert m is not None, (pub_method, d, k)
                    cells = [d, k, m["mean_true"], m["bias"],
                             sd_err_of(m["bias"], m["rmse"]), m["rmse"],
                             m["cover95"]]
                body.append("\\quad " + meth + " & " +
                            " & ".join([str(cells[0]), str(cells[1])]
                                       + [f3(c) for c in cells[2:]]) + r" \\")
        body.append(r"\addlinespace[2pt]")
    head = (r"Estimator & $d$ & $k$ & True ATT($k$) & Bias & SD(err) & RMSE & "
            r"Cov.95 \\")
    longtable(TAB / "tab_staggered_deg.tex", "lrr" + "r" * 5,
              r"Event-study estimands under staggered adoption at the nonlinear degrees "
              r"$d=2$ and $d=3$ ($N=200$, 100 replications).",
              "tab:staggereddeg", head, body,
              r"\emph{Note.} The $d=1$ counterpart is Table \ref{tab:staggered}. "
              r"TWFE uses no covariates, so its entries are invariant to $d$ by "
              r"construction; at $d=3$ the treatment-effect surface itself becomes "
              r"quadratic, so the true ATT($k$) path changes as well. The corrected "
              r"entries are the fixed-$K$ reference-fold convolution with "
              r"random-forest pilots (Algorithm~2).")


PT_ORDER = ["PT_hold", "PT_conditional", "PT_violation_g05", "PT_violation_g10",
            "PT_violation_g20", "PT_violation_g40", "PT_violation_a20",
            "PT_violation_het10", "PT_violation_het20", "PT_violation_het40"]
PT_VIOLATION = {"PT_hold": "none (PTA holds)",
                "PT_conditional": "unconditional only",
                "PT_violation_g05": "group slope 0.05",
                "PT_violation_g10": "group slope 0.10",
                "PT_violation_g20": "group slope 0.20",
                "PT_violation_g40": "group slope 0.40",
                "PT_violation_a20": r"via $\alpha_i$",
                "PT_violation_het10": "heterogeneous, cancels",
                "PT_violation_het20": "heterogeneous, cancels",
                "PT_violation_het40": "heterogeneous, cancels"}


def pt_detect(design, degree, method, etype, eid):
    sub = COMP_REP[(COMP_REP["design"] == design)
                   & (COMP_REP["degree"] == int(degree))
                   & (COMP_REP["method"] == method)
                   & (COMP_REP["estimand_type"] == etype)
                   & (COMP_REP["estimand_id"] == eid)]
    p = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
    p = p[p.notna()]
    if len(p) == 0:
        return float("nan"), 0
    return float((p < 0.025).mean()), len(p)


def pt_anyk(design, degree, method):
    sub = COMP_REP[(COMP_REP["design"] == design)
                   & (COMP_REP["degree"] == int(degree))
                   & (COMP_REP["method"] == method)
                   & (COMP_REP["estimand_type"] == "PRE")
                   & (COMP_REP["estimand_id"].str.startswith("k="))]
    if sub.empty:
        return float("nan"), 0
    vals = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
    sub = sub.assign(_p=vals.values).dropna(subset=["_p"])
    if sub.empty:
        return float("nan"), 0
    per_rep = sub.groupby("rep")["_p"].min() * 3.0
    per_rep = per_rep.clip(upper=1.0)
    return float((per_rep < 0.025).mean()), int(len(per_rep))


def pt_metric(design, degree, method):
    sub = COMP_MET[(COMP_MET["design"] == design)
                   & (COMP_MET["degree"] == int(degree))
                   & (COMP_MET["method"] == method)
                   & (COMP_MET["point_summary"] == "mean")
                   & (COMP_MET["estimand_type"] == "PRE")
                   & (COMP_MET["estimand_id"] == "slope")]
    assert len(sub) == 1, (design, degree, method)
    return sub.iloc[0].to_dict()


def pt_twfe(design, degree):
    m = pub_row(design, "twfe_es", degree, 200, "PRE", "slope")
    assert m is not None, (design, degree)
    return m["reject05"]


def tab_pretrend(degree, fname):
    body = []
    for design in PT_ORDER:
        for ver, meth in (("raw", "pretrend"),
                          ("ref. fold", "reference_fold_pretrend")):
            sdet, _ = pt_detect(design, degree, meth, "PRE", "slope")
            cdet, _ = pt_detect(design, degree, meth, "PRE_SUBC", "X1_slope")
            adet, _ = pt_anyk(design, degree, meth)
            sm = pt_metric(design, degree, meth)
            body.append(" & ".join([
                design.replace("_", r"\_") if ver == "raw" else " ",
                ver, PT_VIOLATION[design], f3(sm["mean_true"]),
                f3(sdet), f3(cdet), f3(adet),
                f3(sm["bias"]), f3(sm["cover95"]),
                f3(pt_twfe(design, degree)) if ver == "raw" else " "])
                + r" \\")
        body.append(r"\addlinespace[2pt]")
    lines = [r"% Auto-generated by Results/analysis/make_jbes_sim_tables.py.",
             r"\begin{tabular}{lllrrrrrrrr}",
             r"\toprule",
             (r"Scenario & Diagnostic & Violation & True & "
              r"\multicolumn{3}{c}{Detection} & "
              r"\multicolumn{2}{c}{Slope recovery} & TWFE \\"),
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             (r" & & & slope & Slope & Contrast & Any-$k$ & bias & "
              r"cover.\ 95\% & ES \\"),
             r"\midrule"]
    lines += body
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / fname).write_text("\n".join(lines) + "\n")
    print("  ->", fname)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    tab_b1("ATT", "ATT", "tab_b1_att.tex", "tab:b1att",
           "Decomposed Monte-Carlo metrics for the overall ATT on the "
           "canonical-DiD scenarios.", ATT_METHODS)
    tab_b1("GATT", "g=4_t=4", "tab_b1_gatt.tex", "tab:b1gatt",
           "Decomposed Monte-Carlo metrics for $\\mathrm{GATT}(g{=}4,t{=}4)$ on "
           "the canonical-DiD scenarios -- the cell every estimator reports.",
           GATT_METHODS)
    tab_null()
    tab_sweep()
    tab_sweep_deg()
    tab_smalln()
    tab_smalln_surface()
    tab_surface()
    tab_bacon()
    tab_staggered_es()
    tab_staggered_gatt()
    tab_staggered_cells()
    tab_staggered_deg()
    tab_pretrend(1, "tab_pretrend.tex")
    tab_pretrend(3, "tab_pretrend_d3.tex")


if __name__ == "__main__":
    main()
