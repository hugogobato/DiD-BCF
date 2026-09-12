#!/usr/bin/env python3
"""Reconcile the Extra_Theory_Experiments calibration results with the JBES run.

Reads the shard archives under ``results/`` (Extra_Theory_Experiments runs),
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
REPO = ETE.parent
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

(OUT / "analysis_digest.json").write_text(json.dumps(digest, indent=2) + "\n")
print(json.dumps({k: digest[k] for k in sorted(digest) if "outliers" in k}, indent=2))
print("wrote", OUT / "analysis_digest.json")
print("wrote tables to", TAB)
