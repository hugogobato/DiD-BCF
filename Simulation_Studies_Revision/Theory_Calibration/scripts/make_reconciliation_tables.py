#!/usr/bin/env python3
"""Reconcile the Theory_Calibration experiment results with the JBES run.

Reads the shard archives under ``results/`` (Theory_Calibration runs),
the revision simulation summaries under
``Simulation_Studies_Revision/Results/aggregated/all_summaries.csv.gz`` (the
published JBES run), aggregates both with identical metric definitions, writes
LaTeX table fragments, and prints a digest of every number quoted in
``reconciliation/theory_reconciliation.tex``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ETE = Path(__file__).resolve().parents[1]
REPO = ETE.parent.parent
sys.path.insert(0, str(ETE / "src"))
sys.path.insert(0, str(ETE / "scripts"))

from aggregate_archives import aggregate_archives  # noqa: E402
from extra_theory_experiments.metrics import scalar_metrics  # noqa: E402

OUT = ETE / "reconciliation"
TAB = OUT / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

CORR_AGG = ETE / "results" / "aggregated_correction"
INFO_AGG = ETE / "results" / "aggregated_information"


def f3(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:0.3f}"


def fi(x) -> str:
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return "--"


def mcse_rate(r, n) -> float:
    try:
        return float(np.sqrt(float(r) * (1.0 - float(r)) / int(n)))
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


# ---------------------------------------------------------------------------
# 1. Aggregate the new runs.
# ---------------------------------------------------------------------------
corr_archives = sorted((ETE / "results").glob("extra_theory_correction_audit_shard_*.zip"))
info_archives = sorted((ETE / "results").glob("extra_theory_information_ablation_shard_*.zip"))
_, corr_summary, corr_metrics = aggregate_archives(
    [str(p) for p in corr_archives], output_dir=CORR_AGG,
    family="correction_audit", n_shards=48)
_, info_summary, info_metrics_df = aggregate_archives(
    [str(p) for p in info_archives], output_dir=INFO_AGG,
    family="information_ablation", n_shards=48)

# ---------------------------------------------------------------------------
# 2. Published JBES per-replication summaries.
# ---------------------------------------------------------------------------
OLD_PATH = (REPO / "Simulation_Studies_Revision" / "Results" / "aggregated"
            / "all_summaries.csv.gz")
with gzip.open(OLD_PATH, "rt") as handle:
    old = pd.read_csv(handle)


def old_metrics(setting, method, N=200, degree=1,
                estimand_type="GATT", estimand_id="g=4_t=4") -> dict:
    sub = old[(old["setting"] == setting) & (old["method"] == method)
              & (old["N"] == N) & (old["linearity_degree"] == degree)
              & (old["estimand_type"] == estimand_type)
              & (old["estimand_id"] == estimand_id)]
    if sub.empty:
        return {}
    return scalar_metrics(sub["post_mean"], sub["true"], posterior_sd=sub["sd"],
                          q05=sub["q05"], q95=sub["q95"],
                          q025=sub["q025"], q975=sub["q975"])


def old_size(setting, method, N=200, degree=1,
             estimand_type="GATT", estimand_id="g=4_t=4") -> tuple[float, int]:
    sub = old[(old["setting"] == setting) & (old["method"] == method)
              & (old["N"] == N) & (old["linearity_degree"] == degree)
              & (old["estimand_type"] == estimand_type)
              & (old["estimand_id"] == estimand_id)]
    if sub.empty:
        return np.nan, 0
    p = pd.to_numeric(sub["p_bayes"], errors="coerce")
    return float(np.mean(p < 0.025)), int(len(p))  # two-sided 5% test


def new_metrics(metrics, design, degree, N, estimator, summary="mean",
                estimand_type="GATT", estimand_id="g=4_t=4") -> dict:
    sub = metrics[(metrics["design"] == design) & (metrics["degree"] == degree)
                  & (metrics["N"] == N) & (metrics["task_estimator"] == estimator)
                  & (metrics["point_summary"] == summary)
                  & (metrics["estimand_type"] == estimand_type)
                  & (metrics["estimand_id"] == estimand_id)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def new_size(design, degree, N, estimator, estimand_type="GATT",
             estimand_id="g=4_t=4") -> tuple[float, int]:
    sub = corr_summary[(corr_summary["design"] == design)
                       & (corr_summary["degree"] == degree)
                       & (corr_summary["N"] == N)
                       & (corr_summary["task_estimator"] == estimator)
                       & (corr_summary["estimand_type"] == estimand_type)
                       & (corr_summary["estimand_id"] == estimand_id)]
    if sub.empty:
        return np.nan, 0
    p = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
    return float(np.mean(p < 0.025)), int(p.notna().sum())


def info_metrics(design, degree, N, estimator, summary="mean",
                 estimand_type="GATT", estimand_id="g=4_t=4") -> dict:
    return new_metrics(info_metrics_df, design, degree, N, estimator,
                       summary=summary, estimand_type=estimand_type,
                       estimand_id=estimand_id)


# ---------------------------------------------------------------------------
# 3. Table fragments.
# ---------------------------------------------------------------------------
BASELINE_OLD = [("JBES plain (structured)", "plain"),
                ("JBES corrected (same-sample)", "corrected")]
BASELINE_NEW = [("New raw structured", "raw_structured"),
                ("New current hybrid, logit", "current_hybrid_logit"),
                ("New current hybrid, logit clip .05", "current_hybrid_logit_clip05"),
                ("New current hybrid, RF", "current_hybrid_rf"),
                ("New reference fold, logit", "reference_fold_convolution_logit"),
                ("New reference fold, RF", "reference_fold_convolution_rf"),
                ("New reference fold, intercept", "reference_fold_convolution_intercept")]

ORACLE_NEW = [("New raw structured", "raw_structured"),
              ("New current hybrid, logit", "current_hybrid_logit"),
              ("New current hybrid, RF", "current_hybrid_rf"),
              ("New reference fold, logit", "reference_fold_convolution_logit"),
              ("New reference fold, RF", "reference_fold_convolution_rf"),
              ("New oracle current hybrid", "oracle_current_hybrid"),
              ("New oracle reference fold", "oracle_reference_fold_convolution")]

HEADER = (r"\toprule" "\n"
          r"Estimator & Bias & SD(err) & RMSE & Cov.95 & Len.95 & "
          r"$\overline{\mathrm{sd}}/\mathrm{SD}$ & $n$ \\" "\n"
          r"\midrule" "\n")
OLD_SETTINGS = {"baseline": "B1_baseline", "serial": "B1_serial_corr"}


def reproduction_table(design: str) -> str:
    """Old JBES vs new estimators, GATT(4,4), N=200, d=1 and d=2."""
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrr}",
             r"\caption{$\mathrm{GATT}(g{=}4,t{=}4)$, $N=200$, "
             + ("baseline" if design == "baseline" else "AR(1) serial")
             + r" design: published JBES run (100 replications) against the "
             r"Extra\_Theory calibration run (available replications shown). "
             r"Metrics use replication-specific truths; here $\overline{\mathrm{sd}}/\mathrm{SD}$ "
             r"is average reported posterior SD over the Monte-Carlo error SD "
             r"$\mathrm{SD}(\widehat\theta-\theta_0)$, which is the common scale "
             r"used throughout this note (the published tables divided by the raw "
             r"spread of estimates instead).}\\label{tab:repro_"
             + design + "}\\\\", HEADER,
             r"\endfirsthead", HEADER,
             r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for degree in (1, 2):
        lines.append(r"\multicolumn{8}{l}{\textit{degree $d=" + str(degree) + r"$}} \\")
        for label, method in BASELINE_OLD:
            m = old_metrics(OLD_SETTINGS[design], method, degree=degree)
            if m:
                lines.append(
                    f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                    f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
                    f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")
        for label, est in BASELINE_NEW:
            m = new_metrics(corr_metrics, design, degree, 200, est)
            if m:
                lines.append(
                    f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                    f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
                    f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines) + "\n"


def sweep_table() -> str:
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrr}",
             r"\caption{$\mathrm{GATT}(g{=}4,t{=}4)$, $N=800$: the published "
             r"$B2$ sweep against the new calibration runs.}\\label{tab:repro800}\\\\", HEADER,
             r"\endfirsthead", HEADER, r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for design, setting in (("baseline", "B2_sweep"), ("serial", "B2_sweep_serial")):
        for degree in (1, 2):
            lines.append(r"\multicolumn{8}{l}{\textit{" + design
                         + r", degree $d=" + str(degree) + r"$}} \\")
            for label, method in BASELINE_OLD:
                m = old_metrics(setting, method, N=800, degree=degree)
                if m:
                    lines.append(
                        f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                        f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
                        f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")
            for label, est in BASELINE_NEW:
                m = new_metrics(corr_metrics, design, degree, 800, est)
                if m:
                    lines.append(
                        f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                        f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
                        f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines) + "\n"


def oracle_table() -> str:
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrr}",
             r"\caption{\texttt{oracle\_canonical} design (known bounded "
             r"$\pi(X)\in(0.001,0.999)$, exact $m_0$; population truth $=3$): "
             r"the new estimators only.}\\label{tab:oracle}\\\\", HEADER,
             r"\endfirsthead", HEADER, r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for N in (200, 800):
        for degree in (1, 2):
            lines.append(r"\multicolumn{8}{l}{\textit{$N=" + str(N)
                         + r"$, degree $d=" + str(degree) + r"$}} \\")
            for label, est in ORACLE_NEW:
                m = new_metrics(corr_metrics, "oracle_canonical", degree, N, est)
                if m:
                    lines.append(
                        f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                        f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
                        f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines) + "\n"


NULL_ROWS = [("JBES plain (structured)", "old_plain", "plain"),
             ("JBES corrected (same-sample)", "old_corrected", "corrected"),
             ("New raw structured", "new", "raw_structured"),
             ("New current hybrid, logit", "new", "current_hybrid_logit"),
             ("New current hybrid, RF", "new", "current_hybrid_rf"),
             ("New reference fold, logit", "new", "reference_fold_convolution_logit"),
             ("New reference fold, RF", "new", "reference_fold_convolution_rf")]


def null_table() -> str:
    lines = [r"{\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{lrrrrrrrr}",
             r"\caption{Sharp null ($\tau\equiv 0$): rejection rate of the "
             r"two-sided 5\% posterior test. Published JBES run: 100 "
             r"replications; new run: available replications.}\\label{tab:null}\\\\",
             r"\toprule",
             r"Estimator & \multicolumn{4}{c}{$\mathrm{GATT}(4,4)$} & "
             r"\multicolumn{4}{c}{ATT} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r" & $d1,N200$ & $d2,N200$ & $d1,N800$ & $d2,N800$ & "
             r"$d1,N200$ & $d2,N200$ & $d1,N800$ & $d2,N800$ \\",
             r"\midrule", r"\endfirsthead", r"\toprule",
             r"Estimator & \multicolumn{4}{c}{$\mathrm{GATT}(4,4)$} & "
             r"\multicolumn{4}{c}{ATT} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r" & $d1,N200$ & $d2,N200$ & $d1,N800$ & $d2,N800$ & "
             r"$d1,N200$ & $d2,N200$ & $d1,N800$ & $d2,N800$ \\",
             r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    cells = []
    for label, src, est in NULL_ROWS:
        row = [label]
        for typ, ident in (("GATT", "g=4_t=4"), ("ATT", "ATT")):
            for N, degree in ((200, 1), (200, 2), (800, 1), (800, 2)):
                if src == "old_plain":
                    p, _ = old_size("B1_null", "plain", N=N, degree=degree,
                                    estimand_type=typ, estimand_id=ident)
                elif src == "old_corrected":
                    p, _ = old_size("B1_null", "corrected", N=N, degree=degree,
                                    estimand_type=typ, estimand_id=ident)
                else:
                    p, _ = new_size("null", degree, N, est,
                                    estimand_type=typ, estimand_id=ident)
                row.append(f3(p))
        cells.append(r" & ".join(row) + r" \\")
    lines += cells
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines) + "\n"


def info_table(estimand_type: str, ident: str, label: str, slug: str) -> str:
    rows = []
    for design, degrees in (("baseline", (1, 2)), ("serial", (1, 2)),
                            ("staggered", (1, 3))):
        for degree in degrees:
            for N in (200, 800):
                for variant in ("full_panel_raw", "reduced_cell", "pooled_full_panel"):
                    m = info_metrics(design, degree, N, variant,
                                     estimand_type=estimand_type,
                                     estimand_id=ident)
                    if m:
                        rows.append((design, degree, N, variant, m))
    lines = [r"{\footnotesize", r"\begin{longtable}{llrrrrrr}",
             r"\caption{" + label + r" in the information-ablation family "
             r"(100 replications per cell). ``reduced'' uses only the two "
             r"groups and periods of the cell; ``pooled'' fits one effect "
             r"surface with \texttt{effect\_by\_cohort=False}.}\\label{tab:" + slug + r"}\\",
             r"\toprule",
             r"Design & $d$ & $N$ & Variant & Bias & RMSE & Cov.95 & Len.95 \\",
             r"\midrule", r"\endfirsthead", r"\toprule",
             r"Design & $d$ & $N$ & Variant & Bias & RMSE & Cov.95 & Len.95 \\",
             r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    pretty = {"full_panel_raw": "full panel", "reduced_cell": "reduced cell",
              "pooled_full_panel": "pooled"}
    for design, degree, N, variant, m in rows:
        lines.append(f"{design} & {degree} & {N} & {pretty[variant]} & "
                     f"{f3(m['bias'])} & {f3(m['rmse'])} & {f3(m['cover95'])} & "
                     f"{f3(m['len95'])} \\\\")
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines) + "\n"


def att_table() -> str:
    """Old JBES vs new estimators, pooled ATT, N=200, d=1 and d=2."""
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrr}",
             r"\caption{Pooled ATT, $N=200$: published JBES run against the new "
             r"calibration runs. The ratio column is recomputed on the common "
             r"error-SD scale.}\\label{tab:repro_att}\\\\", HEADER,
             r"\endfirsthead", HEADER, r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for design in ("baseline", "serial"):
        for degree in (1, 2):
            lines.append(r"\multicolumn{8}{l}{\textit{" + design
                         + r", degree $d=" + str(degree) + r"$}} \\")
            for label, method in BASELINE_OLD:
                m = old_metrics(OLD_SETTINGS[design], method, degree=degree,
                                estimand_type="ATT", estimand_id="ATT")
                if m:
                    lines.append(
                        f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                        f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
                        f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")
            for label, est in BASELINE_NEW:
                m = new_metrics(corr_metrics, design, degree, 200, est,
                                estimand_type="ATT", estimand_id="ATT")
                if m:
                    lines.append(
                        f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                        f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
                        f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines) + "\n"


BENCH_LABELS = {"did_dr": "Callaway--Sant'Anna", "did2s": "Gardner (DiD2s)",
                "doubleml": "DoubleML DR-DiD", "wang": "grf-DiD (Wang)",
                "twfe": "TWFE", "synthdid": "Synthetic DiD"}
BENCH_GATT = ["did_dr", "doubleml", "wang"]
BENCH_ATT = ["twfe", "synthdid"]
BENCH_BLOCKS_200 = [("baseline, $N=200$", "baseline", "B1_baseline", 200, (1, 2)),
                    ("serial, $N=200$", "serial", "B1_serial_corr", 200, (1, 2))]
BENCH_BLOCKS_800 = [("baseline, $N=800$", "baseline", "B2_sweep", 800, (1, 2)),
                    ("serial, $N=800$", "serial", "B2_sweep_serial", 800, (1, 2))]
BENCH_NEW = [("DiD-BCF raw (new)", "raw_structured"),
             ("DiD-BCF same-sample, logit (new)", "current_hybrid_logit"),
             ("DiD-BCF same-sample, logit clip .05 (new)",
              "current_hybrid_logit_clip05"),
             ("DiD-BCF reference fold, RF (new)",
              "reference_fold_convolution_rf"),
             ("DiD-BCF reference fold, logit (new)",
              "reference_fold_convolution_logit")]


def _row(label, m):
    return (f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
            f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} \\\\")


def benchmark_table(typ, ident, slug, blocks, bench_methods) -> str:
    title = (r"$\mathrm{GATT}(g{=}4,t{=}4)$" if typ == "GATT" else "Pooled ATT")
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrr}",
             r"\caption{" + title + r": the new DiD-BCF calibration estimators "
             r"against the published JBES benchmarks (100 replications per "
             r"cell, replication-specific truths).}\\label{tab:" + slug + r"}\\",
             r"\toprule",
             r"Estimator & Bias & SD(err) & RMSE & Cov.95 & Len.95 \\",
             r"\midrule", r"\endfirsthead", r"\toprule",
             r"Estimator & Bias & SD(err) & RMSE & Cov.95 & Len.95 \\",
             r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for label, design, setting, N, degrees in blocks:
        for degree in degrees:
            lines.append(r"\multicolumn{6}{l}{\textit{" + label
                         + r", degree $d=" + str(degree) + r"$}} \\")
            for lbl, method in BASELINE_OLD:
                m = old_metrics(setting, method, N=N, degree=degree,
                                estimand_type=typ, estimand_id=ident)
                if m:
                    lines.append(_row(lbl, m))
            for lbl, est in BENCH_NEW:
                m = new_metrics(corr_metrics, design, degree, N, est,
                                estimand_type=typ, estimand_id=ident)
                if m:
                    lines.append(_row(lbl, m))
            for method in bench_methods:
                m = old_metrics(setting, method, N=N, degree=degree,
                                estimand_type=typ, estimand_id=ident)
                if m:
                    lines.append(_row(BENCH_LABELS[method], m))
            lines.append(r"\addlinespace")
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines) + "\n"


def _fix_tex(text: str) -> str:
    """Undo over-escaped backslashes introduced by raw caption strings."""
    return (text.replace("}\\\\label", "}\\label")
                .replace("}\\\\\\\\", "}\\\\"))


TAB.joinpath("tab_repro_baseline.tex").write_text(_fix_tex(reproduction_table("baseline")))
TAB.joinpath("tab_repro_serial.tex").write_text(_fix_tex(reproduction_table("serial")))
TAB.joinpath("tab_repro_att.tex").write_text(_fix_tex(att_table()))
TAB.joinpath("tab_repro_sweep.tex").write_text(_fix_tex(sweep_table()))
TAB.joinpath("tab_oracle.tex").write_text(_fix_tex(oracle_table()))
TAB.joinpath("tab_null.tex").write_text(_fix_tex(null_table()))
TAB.joinpath("tab_info_gatt.tex").write_text(_fix_tex(
    info_table("GATT", "g=4_t=4", r"$\mathrm{GATT}(g{=}4,t{=}4)$", "infogatt")))
TAB.joinpath("tab_info_att.tex").write_text(_fix_tex(
    info_table("ATT", "ATT", r"ATT", "infoatt")))
TAB.joinpath("tab_bench_gatt.tex").write_text(_fix_tex(
    benchmark_table("GATT", "g=4_t=4", "benchgatt", BENCH_BLOCKS_200, BENCH_GATT)))
TAB.joinpath("tab_bench_gatt_800.tex").write_text(_fix_tex(
    benchmark_table("GATT", "g=4_t=4", "benchgatt800", BENCH_BLOCKS_800, BENCH_GATT)))
TAB.joinpath("tab_bench_att.tex").write_text(_fix_tex(
    benchmark_table("ATT", "ATT", "benchatt", BENCH_BLOCKS_200, BENCH_ATT)))

# ---------------------------------------------------------------------------
# 4. Digest.
# ---------------------------------------------------------------------------
digest: dict[str, object] = {}


def store(tag, **values):
    digest[tag] = {k: (None if v is None or not np.isfinite(v) else round(float(v), 6))
                   for k, v in values.items()}


for design in ("baseline", "serial"):
    for degree in (1, 2):
        for label, method in BASELINE_OLD:
            m = old_metrics(OLD_SETTINGS[design], method, degree=degree)
            if m:
                store(f"old_{design}_d{degree}_{method}",
                      bias=m["bias"], sd=m["emp_sd"], rmse=m["rmse"],
                      cov95=m["cover95"], len95=m["len95"], ratio=m["sd_ratio"],
                      n=m["n_reps"])
        for label, est in BASELINE_NEW:
            m = new_metrics(corr_metrics, design, degree, 200, est)
            if m:
                store(f"new_{design}_d{degree}_{est}",
                      bias=m["bias"], sd=m["emp_sd"], rmse=m["rmse"],
                      cov95=m["cover95"], len95=m["len95"], ratio=m["sd_ratio"],
                      n=m["n_reps"])

for design, setting in (("baseline", "B1_baseline"), ("serial", "B1_serial_corr")):
    for degree in (1, 2):
        for method in ("did_dr", "doubleml", "wang", "twfe", "synthdid"):
            for typ, ident in (("GATT", "g=4_t=4"), ("ATT", "ATT")):
                m = old_metrics(setting, method, degree=degree,
                                estimand_type=typ, estimand_id=ident)
                if m:
                    store(f"bench_{design}_d{degree}_{method}_{typ.lower()}",
                          bias=m["bias"], rmse=m["rmse"], cov95=m["cover95"],
                          len95=m["len95"], n=m["n_reps"])

# outlier diagnostics on the nonlinear-degree baseline cell
cell = corr_summary[(corr_summary["design"] == "baseline")
                    & (corr_summary["degree"] == 2)
                    & (corr_summary["N"] == 200)
                    & (corr_summary["estimand_type"] == "GATT")
                    & (corr_summary["estimand_id"] == "g=4_t=4")]
for est in ("current_hybrid_logit", "current_hybrid_logit_clip05",
            "reference_fold_convolution_logit", "reference_fold_convolution_rf"):
    sub = cell[cell["task_estimator"] == est].copy()
    if sub.empty:
        continue
    err = pd.to_numeric(sub["post_mean"], errors="coerce") - pd.to_numeric(sub["true"], errors="coerce")
    odds = pd.to_numeric(sub["max_control_odds"], errors="coerce")
    ess = pd.to_numeric(sub["effective_sample_size"], errors="coerce")
    trimmed = err[np.abs(err) <= 2]
    store(f"outliers_baseline_d2_{est}",
          n=len(err), max_abs_err=float(np.abs(err).max()),
          n_gt2=int((np.abs(err) > 2).sum()),
          rmse_all=float(np.sqrt(np.mean(err ** 2))),
          rmse_trimmed=float(np.sqrt(np.mean(trimmed ** 2))),
          max_odds=float(odds.max()), min_ess=float(ess.min()),
          median_odds=float(odds.median()))

# same tail diagnostic on the published JBES corrected run
old_d2 = old[(old["setting"] == "B1_baseline") & (old["method"] == "corrected")
             & (old["N"] == 200) & (old["linearity_degree"] == 2)
             & (old["estimand_type"] == "GATT")
             & (old["estimand_id"] == "g=4_t=4")]
if not old_d2.empty:
    oerr = old_d2["post_mean"].to_numpy(float) - old_d2["true"].to_numpy(float)
    store("outliers_baseline_d2_old_corrected",
          n=len(oerr), max_abs_err=float(np.abs(oerr).max()),
          n_gt2=int((np.abs(oerr) > 2).sum()),
          rmse_all=float(np.sqrt(np.mean(oerr ** 2))),
          rmse_trimmed=float(np.sqrt(np.mean(oerr[np.abs(oerr) <= 2] ** 2))))

for design, degrees in (("baseline", (1, 2)), ("serial", (1, 2)), ("staggered", (1, 3))):
    for degree in degrees:
        for N in (200, 800):
            for variant in ("full_panel_raw", "reduced_cell", "pooled_full_panel"):
                m = info_metrics(design, degree, N, variant)
                if m:
                    store(f"info_{design}_d{degree}_N{N}_{variant}",
                          bias=m["bias"], rmse=m["rmse"], cov95=m["cover95"],
                          len95=m["len95"], n=m["n_reps"])

# ---------------------------------------------------------------------------
# 5. Completion family: the cells that still used the same-sample correction.
# ---------------------------------------------------------------------------
COMP_AGG = ETE / "results" / "aggregated_completion"
comp_metrics = pd.read_csv(COMP_AGG / "metrics_mean_median.csv",
                           keep_default_na=False, na_values=[])
comp_summary = pd.read_csv(COMP_AGG / "combined_per_replication.csv",
                           engine="python", keep_default_na=False,
                           na_values=[], on_bad_lines="skip")

COMP_SETTING = {
    "strong_confounder": "B1_strong_confounder",
    "selection_obs": "B1_selection_obs",
    "selection_both": "B1_selection_both",
    "baseline": "B1_baseline",
    "serial": "B1_serial_corr",
    "null": "B1_null",
    "baseline_sweep": "B2_sweep",
    "baseline_sweep_d3": "B2_sweep",
    "serial_sweep": "B2_sweep_serial",
    "serial_sweep_d3": "B2_sweep_serial",
    "staggered": "D_staggered",
}
COMP_PRETTY = {
    "strong_confounder": "strong confounder",
    "selection_obs": "selection on observables",
    "selection_both": "selection on both",
    "baseline": "baseline",
    "serial": "serial correlation",
    "null": "sharp null",
    "baseline_sweep": "baseline sweep",
    "baseline_sweep_d3": "baseline sweep",
    "serial_sweep": "serial sweep",
    "serial_sweep_d3": "serial sweep",
    "staggered": "staggered",
}
# (task_estimator, method) pairs in the completion archive.
COMP_RAW = ("raw_structured", "raw_structured")
COMP_REF = ("reference_fold_convolution_rf", "reference_fold_convolution")
COMP_RAW_PT = ("raw_structured", "pretrend")
COMP_REF_PT = ("reference_fold_convolution_rf", "reference_fold_pretrend")


def comp_cell(design, degree, N, estimator, method, etype, eid,
              summary="mean"):
    sub = comp_metrics[(comp_metrics["design"] == design)
                       & (comp_metrics["degree"] == int(degree))
                       & (comp_metrics["N"] == int(N))
                       & (comp_metrics["task_estimator"] == estimator)
                       & (comp_metrics["method"] == method)
                       & (comp_metrics["point_summary"] == summary)
                       & (comp_metrics["estimand_type"] == etype)
                       & (comp_metrics["estimand_id"] == eid)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def old_cell_metrics(setting, method, N, degree, etype, eid):
    """Published-run metrics recomputed from the archived per-rep summaries."""
    sub = old[(old["setting"] == setting) & (old["method"] == method)
              & (old["N"] == int(N))
              & (old["linearity_degree"] == int(degree))
              & (old["estimand_type"] == etype)
              & (old["estimand_id"] == eid)]
    if sub.empty:
        return {}
    return scalar_metrics(sub["post_mean"], sub["true"],
                          posterior_sd=sub["sd"], q05=sub["q05"],
                          q95=sub["q95"], q025=sub["q025"], q975=sub["q975"])


def _comp_row(label, m):
    return (f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
            f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(m['len95'])} & "
            f"{f3(m['sd_ratio'])} & {fi(m['n_reps'])} \\\\")


COMP_HEAD = (r"Estimator & Bias & SD(err) & RMSE & Cov.95 & Len.95 & "
             r"$\overline{\mathrm{sd}}/\mathrm{SD}$ & $n$ \\")


def _comp_longtable(name, caption, label, body):
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrr}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule", COMP_HEAD, r"\midrule",
             r"\endfirsthead", COMP_HEAD, r"\midrule",
             r"\endhead", r"\bottomrule", r"\endlastfoot"]
    lines += body
    lines += [r"\end{longtable}", "}"]
    (TAB / name).write_text("\n".join(lines) + "\n")
    print("  ->", name)


def completion_b1_table():
    """ATT at N=200 for every newly rerun canonical design and degree."""
    body = []
    for design, degrees in (("strong_confounder", (1, 2, 3)),
                            ("selection_obs", (1, 2, 3)),
                            ("selection_both", (1, 2, 3)),
                            ("baseline", (3,)), ("serial", (3,)),
                            ("null", (3,))):
        body.append(r"\multicolumn{8}{l}{\textit{" + design.replace("_", r"\_")
                    + r"}} \\")
        setting = COMP_SETTING[design]
        for degree in degrees:
            body.append(r"\multicolumn{8}{l}{\textit{degree $d=" + str(degree)
                        + r"$}} \\")
            for label, (est, meth) in (
                    ("New raw structured", COMP_RAW),
                    ("New reference fold, RF", COMP_REF),
                    ("Published corrected (same-sample)", (None, "corrected"))):
                if est is None:
                    m = old_cell_metrics(setting, meth, 200, degree,
                                         "ATT", "ATT")
                else:
                    m = comp_cell(design, degree, 200, est, meth,
                                  "ATT", "ATT")
                if m:
                    body.append(_comp_row(label, m))
            body.append(r"\addlinespace")
    _comp_longtable(
        "tab_completion_b1.tex",
        r"Pooled ATT, $N=200$: the newly rerun canonical cells (strong "
        r"confounder, selection designs, and degree $d=3$) with the published "
        r"same-sample correction alongside for contrast.",
        "tab:completion_b1", body)


def completion_sweep_table():
    """GATT(4,4) sweeps: the N=50,100,400 and d=3 cells the audit skipped."""
    body = []
    for design, degrees, Ns in (("baseline_sweep", (1, 2), (50, 100, 400)),
                                ("baseline_sweep_d3", (3,),
                                 (50, 100, 400, 800)),
                                ("serial_sweep", (1, 2), (50, 100, 400)),
                                ("serial_sweep_d3", (3,),
                                 (50, 100, 400, 800))):
        body.append(r"\multicolumn{8}{l}{\textit{" + design.replace("_", r"\_")
                    + r"}} \\")
        setting = COMP_SETTING[design]
        for degree in degrees:
            for N in Ns:
                body.append(r"\multicolumn{8}{l}{\textit{$N=" + str(N)
                            + r"$, degree $d=" + str(degree) + r"$}} \\")
                for label, (est, meth) in (
                        ("New raw structured", COMP_RAW),
                        ("New reference fold, RF", COMP_REF),
                        ("Published corrected (same-sample)",
                         (None, "corrected"))):
                    if est is None:
                        m = old_cell_metrics(setting, meth, N, degree,
                                             "GATT", "g=4_t=4")
                    else:
                        m = comp_cell(design, degree, N, est, meth,
                                      "GATT", "g=4_t=4")
                    if m:
                        body.append(_comp_row(label, m))
                body.append(r"\addlinespace")
    _comp_longtable(
        "tab_completion_sweep.tex",
        r"$\mathrm{GATT}(g{=}4,t{=}4)$ sweeps: every cell the correction-audit "
        r"family skipped, with the published same-sample correction alongside "
        r"for contrast.",
        "tab:completion_sweep", body)


def completion_staggered_table():
    """Staggered ES/ATT/GATT cells at all three degrees."""
    body = []
    for degree in (1, 2, 3):
        body.append(r"\multicolumn{8}{l}{\textit{degree $d=" + str(degree)
                    + r"$}} \\")
        for etype, eid in ([("ES", f"k={k}") for k in (0, 1, 2, 3)]
                           + [("ATT", "ATT")]):
            body.append(r"\multicolumn{8}{l}{\textit{" + etype + ", "
                        + eid.replace("_", r"\_").replace("=", "$=$")
                        + r"}} \\")
            for label, (est, meth) in (
                    ("New raw structured", COMP_RAW),
                    ("New reference fold, RF", COMP_REF),
                    ("Published corrected (same-sample)", (None, "corrected"))):
                if est is None:
                    m = old_cell_metrics("D_staggered", meth, 200, degree,
                                         etype, eid)
                else:
                    m = comp_cell("staggered", degree, 200, est, meth,
                                  etype, eid)
                if m:
                    body.append(_comp_row(label, m))
            body.append(r"\addlinespace")
    _comp_longtable(
        "tab_completion_staggered.tex",
        r"Staggered event-study and pooled ATT cells at all three linearity "
        r"degrees ($N=200$), with the published same-sample correction "
        r"alongside for contrast.",
        "tab:completion_staggered", body)


def _comp_detect(design, degree, method, etype, eid):
    sub = comp_summary[(comp_summary["design"] == design)
                       & (comp_summary["degree"] == int(degree))
                       & (comp_summary["method"] == method)
                       & (comp_summary["estimand_type"] == etype)
                       & (comp_summary["estimand_id"] == eid)]
    p = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
    p = p[p.notna()]
    if len(p) == 0:
        return float("nan"), 0
    return float((p < 0.025).mean()), len(p)


def _comp_anyk(design, degree, method):
    sub = comp_summary[(comp_summary["design"] == design)
                       & (comp_summary["degree"] == int(degree))
                       & (comp_summary["method"] == method)
                       & (comp_summary["estimand_type"] == "PRE")
                       & (comp_summary["estimand_id"].str.startswith("k="))]
    if sub.empty:
        return float("nan"), 0
    vals = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
    sub = sub.assign(_p=vals.values).dropna(subset=["_p"])
    if sub.empty:
        return float("nan"), 0
    per_rep = (sub.groupby("rep")["_p"].min() * 3.0).clip(upper=1.0)
    return float((per_rep < 0.025).mean()), int(len(per_rep))


PT_COMP_ORDER = ["PT_hold", "PT_conditional", "PT_violation_g05",
                 "PT_violation_g10", "PT_violation_g20", "PT_violation_g40",
                 "PT_violation_a20", "PT_violation_het10", "PT_violation_het20",
                 "PT_violation_het40"]


def completion_pt_table(degree, name, label):
    body = []
    for design in PT_COMP_ORDER:
        body.append(r"\multicolumn{8}{l}{\textit{" + design.replace("_", r"\_")
                    + r"}} \\")
        for dlabel, (est, meth) in (("New raw diagnostic", COMP_RAW_PT),
                                    ("New reference-fold diagnostic",
                                     COMP_REF_PT)):
            m = comp_cell(design, degree, 200, est, meth, "PRE", "slope")
            if not m:
                continue
            sdet, _ = _comp_detect(design, degree, meth, "PRE", "slope")
            cdet, _ = _comp_detect(design, degree, meth, "PRE_SUBC",
                                   "X1_slope")
            adet, _ = _comp_anyk(design, degree, meth)
            body.append(
                f"\\quad {dlabel} & {f3(sdet)} & {f3(cdet)} & {f3(adet)} & "
                f"{f3(m['bias'])} & {f3(m['rmse'])} & {f3(m['cover95'])} & "
                f"{fi(m['n_reps'])} \\\\")
        body.append(r"\addlinespace")
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrr}",
             r"\caption{Pre-trend diagnostic operating characteristics at "
             r"degree $d=" + str(degree) + r"$: slope-rule, $X_1$-contrast, and "
             r"any-$k$ detection rates with slope bias, RMSE, and coverage. "
             r"The fold-pooled diagnostic is an exploratory aggregation, not a "
             r"theorem-covered estimator.}\label{" + label + r"}\\",
             r"\toprule",
             (r"Diagnostic & Slope & Contrast & Any-$k$ & Bias & RMSE & "
              r"Cov.95 & $n$ \\"),
             r"\midrule", r"\endfirsthead",
             (r"Diagnostic & Slope & Contrast & Any-$k$ & Bias & RMSE & "
              r"Cov.95 & $n$ \\"),
             r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    lines += body
    lines += [r"\end{longtable}", "}"]
    (TAB / name).write_text("\n".join(lines) + "\n")
    print("  ->", name)


def completion_null_table():
    body = []
    for etype, eid in (("ATT", "ATT"), ("ES", "k=0"), ("GATT", "g=4_t=4")):
        body.append(r"\multicolumn{8}{l}{\textit{" + etype + ", "
                    + eid.replace("_", r"\_").replace("=", "$=$") + r"}} \\")
        for label, (est, meth) in (("New raw structured", COMP_RAW),
                                   ("New reference fold, RF", COMP_REF)):
            m = comp_cell("null", 3, 200, est, meth, etype, eid)
            if not m:
                continue
            r, n = _comp_detect("null", 3, meth, etype, eid)
            body.append(
                f"\\quad {label} & {f3(m['bias'])} & {f3(m['emp_sd'])} & "
                f"{f3(m['rmse'])} & {f3(m['cover95'])} & {f3(r)} & "
                f"{f3(mcse_rate(r, n))} & {fi(m['n_reps'])} \\\\")
        body.append(r"\addlinespace")
    lines = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrr}",
             r"\caption{Sharp-null $d=3$ rerun ($N=200$, 200 replications): "
             r"bias, dispersion, coverage, and the two-sided 5\% rejection "
             r"rate with its Monte-Carlo standard error.}\label{"
             r"tab:completion_null}\\",
             r"\toprule",
             (r"Estimator & Bias & SD(err) & RMSE & Cov.95 & Size (5\%) & "
              r"MCSE & $n$ \\"),
             r"\midrule", r"\endfirsthead",
             (r"Estimator & Bias & SD(err) & RMSE & Cov.95 & Size (5\%) & "
              r"MCSE & $n$ \\"),
             r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    lines += body
    lines += [r"\end{longtable}", "}"]
    (TAB / "tab_completion_null.tex").write_text("\n".join(lines) + "\n")
    print("  -> tab_completion_null.tex")


completion_b1_table()
completion_sweep_table()
completion_staggered_table()
completion_pt_table(1, "tab_completion_pt.tex", "tab:completion_pt")
completion_pt_table(3, "tab_completion_pt_d3.tex", "tab:completion_pt_d3")
completion_null_table()

for design, degrees, N, etype, eid in (
        ("strong_confounder", (1, 2, 3), 200, "ATT", "ATT"),
        ("selection_obs", (1, 2, 3), 200, "ATT", "ATT"),
        ("selection_both", (1, 2, 3), 200, "ATT", "ATT"),
        ("baseline", (3,), 200, "ATT", "ATT"),
        ("serial", (3,), 200, "ATT", "ATT"),
        ("staggered", (1, 2, 3), 200, "ATT", "ATT"),
        ("null", (3,), 200, "ATT", "ATT")):
    for degree in degrees:
        for est, meth in (COMP_RAW, COMP_REF):
            m = comp_cell(design, degree, N, est, meth, etype, eid)
            if m:
                store(f"completion_{design}_d{degree}_{est}",
                      bias=m["bias"], sd=m["emp_sd"], rmse=m["rmse"],
                      cov95=m["cover95"], len95=m["len95"],
                      ratio=m["sd_ratio"], n=m["n_reps"])
for design, degrees, N in (("baseline_sweep", (1, 2), 50),
                           ("baseline_sweep", (1, 2), 100),
                           ("baseline_sweep", (1, 2), 400),
                           ("serial_sweep", (1, 2), 50),
                           ("serial_sweep", (1, 2), 100),
                           ("serial_sweep", (1, 2), 400),
                           ("baseline_sweep_d3", (3,), 50),
                           ("baseline_sweep_d3", (3,), 100),
                           ("baseline_sweep_d3", (3,), 400),
                           ("baseline_sweep_d3", (3,), 800),
                           ("serial_sweep_d3", (3,), 50),
                           ("serial_sweep_d3", (3,), 100),
                           ("serial_sweep_d3", (3,), 400),
                           ("serial_sweep_d3", (3,), 800)):
    for degree in degrees:
        m = comp_cell(design, degree, N, *COMP_REF, "GATT", "g=4_t=4")
        if m:
            tag = design.replace("_d3", "")
            store(f"completion_{tag}_d{degree}_N{N}_reference",
                  bias=m["bias"], rmse=m["rmse"], cov95=m["cover95"],
                  len95=m["len95"], ratio=m["sd_ratio"], n=m["n_reps"])
for design in PT_COMP_ORDER:
    for degree in (1, 3):
        for est, meth in (COMP_RAW_PT, COMP_REF_PT):
            m = comp_cell(design, degree, 200, est, meth, "PRE", "slope")
            if m:
                sdet, _ = _comp_detect(design, degree, meth, "PRE", "slope")
                store(f"completion_{design}_d{degree}_{est}_slope",
                      detect=sdet, bias=m["bias"], rmse=m["rmse"],
                      cov95=m["cover95"], n=m["n_reps"])

(OUT / "analysis_digest.json").write_text(json.dumps(digest, indent=2) + "\n")
print(json.dumps({k: digest[k] for k in sorted(digest) if "outliers" in k}, indent=2))
print("wrote", OUT / "analysis_digest.json")
print("wrote tables to", TAB)
