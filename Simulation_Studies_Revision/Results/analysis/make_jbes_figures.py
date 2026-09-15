#!/usr/bin/env python3
"""Regenerate the JBES submission simulation figures from the archived runs.

The established figure code (Results/analysis/make_analysis.py,
make_smalln_analysis.py) reads the published aggregation only, so its
DiD-BCF corrected curves still show the same-sample implementation.  This step
builds combined frames (published rows for every non-corrected method plus the
reference-fold reruns relabeled as "corrected"), reuses the established figure
functions unchanged, and writes Results/figures/*.pdf, mirrored to
JBES_Submission/sim_figures/ (plus images/fig_smalln.pdf).

Cells never rerun with the reference fold (the D_contamination design and the
CATT-surface summaries) keep the earlier values under an explicit
"corrected, earlier" legend entry.  fig_event_study and fig_ramp have no
corrected curves and are left untouched.
"""
from __future__ import annotations

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.dirname(HERE)
ROOT = os.path.dirname(RES)
AGG = os.path.join(RES, "aggregated")
FIG = os.path.join(RES, "figures")
JBES = os.path.join(ROOT, "..", "..", "JBES_Submission")
SIMFIG = os.path.join(JBES, "sim_figures")
IMAGES = os.path.join(JBES, "images")

import make_analysis as ma  # noqa: E402
import make_smalln_analysis as ms  # noqa: E402
from style import COLOR, MARKER, PLAINLABEL  # noqa: E402

# The fold-pooled pre-trend diagnostic gets the corrected hue with its own
# marker so it reads as a corrected-family curve in the PT figure.
COLOR["reference_fold_pretrend"] = COLOR["corrected"]
MARKER["reference_fold_pretrend"] = "D"

PUB_MET = pd.read_csv(os.path.join(AGG, "metrics_all.csv"))
PUB_SURF = pd.read_csv(os.path.join(AGG, "surface_all.csv"))
PUB_SQN = pd.read_csv(os.path.join(AGG, "sqrtn_all.csv"))

ETE_RES = os.path.join(RES, "..", "Theory_Calibration", "results")


def _read_ete(directory, name):
    return pd.read_csv(os.path.join(ETE_RES, directory, name),
                       keep_default_na=False, na_values=[])


CORR_MET = _read_ete("aggregated_correction", "metrics_mean_median.csv")
COMP_MET = _read_ete("aggregated_completion", "metrics_mean_median.csv")
CORR_REP = pd.read_csv(os.path.join(ETE_RES, "aggregated_correction",
                                    "combined_per_replication.csv"),
                       engine="python", keep_default_na=False, na_values=[],
                       on_bad_lines="skip")
COMP_REP = pd.read_csv(os.path.join(ETE_RES, "aggregated_completion",
                                    "combined_per_replication.csv"),
                       engine="python", keep_default_na=False, na_values=[],
                       on_bad_lines="skip")

REF_EST = "reference_fold_convolution_rf"
REF_METHOD = "reference_fold_convolution"

SETTING_DESIGNS = {
    "B1_baseline": ["baseline"],
    "B1_serial_corr": ["serial"],
    "B1_null": ["null"],
    "B1_strong_confounder": ["strong_confounder"],
    "B1_selection_obs": ["selection_obs"],
    "B2_sweep": ["baseline", "baseline_sweep", "baseline_sweep_d3"],
    "B2_sweep_serial": ["serial", "serial_sweep", "serial_sweep_d3"],
    "D_staggered": ["staggered"],
    "D_contamination": [],
}


def _ref_frame():
    """Reference-fold metrics rows in the published-table schema."""
    use = [c for c in
           ("design", "degree", "N", "task_estimator", "method",
            "point_summary", "estimand_type", "estimand_id", "n_reps",
            "mean_true", "bias", "abs_bias", "rmse", "mae", "cover90",
            "cover95", "len90", "len95", "avg_post_sd", "raw_sd_est")
           if c in CORR_MET.columns and c in COMP_MET.columns]
    ref = pd.concat([CORR_MET[use], COMP_MET[use]], ignore_index=True)
    ref = ref[(ref["task_estimator"] == REF_EST)
              & (ref["method"] == REF_METHOD)
              & (ref["point_summary"] == "mean")].copy()
    rows = []
    for setting, designs in SETTING_DESIGNS.items():
        if not designs:
            continue
        sub = ref[ref["design"].isin(designs)]
        for _, r in sub.iterrows():
            rows.append({
                "setting": setting,
                "linearity_degree": int(r["degree"]),
                "N": int(r["N"]),
                "estimand_type": r["estimand_type"],
                "estimand_id": r["estimand_id"],
                "method": "corrected",
                "n_reps": r["n_reps"],
                "mean_true": r["mean_true"],
                "bias": r["bias"],
                "abs_bias": r["abs_bias"],
                "rmse": r["rmse"],
                "cover90": r["cover90"],
                "cover95": r["cover95"],
                "len95": r["len95"],
                "avg_post_sd": r["avg_post_sd"],
                "emp_sd": r["raw_sd_est"],
                "sd_ratio": (r["avg_post_sd"] / r["raw_sd_est"]
                             if r["raw_sd_est"] else np.nan),
            })
    return pd.DataFrame.from_records(rows)


def _combined_met():
    ref = _ref_frame()
    keys = ["setting", "linearity_degree", "N", "estimand_type",
            "estimand_id"]
    have = set(map(tuple, ref[keys].astype(str).values.tolist()))
    pub = PUB_MET.copy()
    # Published same-sample corrected rows survive only where no rerun exists
    # (the contamination design); everywhere else the rerun replaces them.
    keep_corr = []
    for _, r in pub[pub["method"] == "corrected"].iterrows():
        key = tuple(str(r[k]) for k in
                    ["setting", "linearity_degree", "N", "estimand_type",
                     "estimand_id"])
        keep_corr.append(key not in have)
    pub_corr = pub[pub["method"] == "corrected"][keep_corr].copy()
    pub_corr["method"] = "corrected_earlier"
    pub_rest = pub[pub["method"] != "corrected"].copy()
    out = pd.concat([pub_rest, pub_corr, ref], ignore_index=True)
    out["sd_err"] = np.sqrt(np.clip(out["rmse"] ** 2 - out["bias"] ** 2,
                                    0, None))
    return out


def _combined_sqn():
    sqn = PUB_SQN[PUB_SQN["method"] != "corrected"].copy()
    perrep = pd.concat([CORR_REP, COMP_REP], ignore_index=True)
    perrep = perrep[(perrep["task_estimator"] == REF_EST)
                    & (perrep["method"] == REF_METHOD)].copy()
    perrep["setting"] = perrep["design"].map(
        {d: s for s, ds in SETTING_DESIGNS.items() for d in ds})
    perrep = perrep[perrep["setting"].isin(["B2_sweep", "B2_sweep_serial"])
                    & (perrep["estimand_type"] == "ATT")
                    & (perrep["estimand_id"] == "ATT")]
    perrep = perrep.assign(
        _est=pd.to_numeric(perrep["post_mean"], errors="coerce"),
        _tru=pd.to_numeric(perrep["true"], errors="coerce")).dropna(
            subset=["_est", "_tru"])
    perrep["scaled"] = (np.sqrt(perrep["N"].astype(float))
                        * (perrep["_est"] - perrep["_tru"]))
    grp = (perrep.groupby(["setting", "degree", "N"], dropna=False)["scaled"]
           .agg(mean="mean", sd="std",
                rmse=lambda x: np.sqrt(np.mean(x ** 2))).reset_index())
    for _, r in grp.iterrows():
        sqn = pd.concat([sqn, pd.DataFrame([{
            "setting": r["setting"], "linearity_degree": int(r["degree"]),
            "N": int(r["N"]), "method": "corrected", "mean": r["mean"],
            "sd": r["sd"], "rmse": r["rmse"], "estimand_type": "ATT",
            "estimand_id": "ATT"}])], ignore_index=True)
    return sqn


def _combined_surf():
    surf = PUB_SURF.copy()
    surf.loc[surf["method"] == "corrected", "method"] = "corrected_earlier"
    return surf


EARLIER_STYLE = {"color": COLOR["corrected"], "marker": "s",
                 "linestyle": ":", "label": "DiD-BCF (corrected, earlier)"}


def _register_styles():
    ma.METHOD_ORDER = ["plain", "corrected", "corrected_earlier", "twfe",
                       "did_dr", "did2s", "doubleml", "synthdid", "wang"]
    ma.COLOR["corrected_earlier"] = COLOR["corrected"]
    ma.MARKER["corrected_earlier"] = "s"
    ma.DASH["corrected_earlier"] = ":"
    ma.LABEL["corrected_earlier"] = "DiD-BCF (corrected, earlier)"
    ma.PLAINLABEL["corrected_earlier"] = "DiD-BCF (corrected, earlier)"
    ma.TEXLABEL["corrected_earlier"] = "DiD-BCF (corrected, earlier)"


PT_ORDER = ["PT_hold", "PT_conditional", "PT_violation_g05", "PT_violation_g10",
            "PT_violation_g20", "PT_violation_g40", "PT_violation_a20"]
PT_SHOW = ["PT_hold", "PT_conditional", "PT_violation_g10", "PT_violation_g40"]
DIAG, TWFE_ES = "pretrend", "twfe_es"
NOMINAL = 0.05


def _pt_detect(design, degree, method, etype="PRE", eid="slope"):
    sub = COMP_REP[(COMP_REP["design"] == design)
                   & (COMP_REP["degree"] == int(degree))
                   & (COMP_REP["method"] == method)
                   & (COMP_REP["estimand_type"] == etype)
                   & (COMP_REP["estimand_id"] == eid)]
    p = pd.to_numeric(sub["p_bayes_tail_min"], errors="coerce")
    p = p[p.notna()]
    return float((p < 0.025).mean()) if len(p) else float("nan")


def _pt_metric(design, degree, method):
    sub = COMP_MET[(COMP_MET["design"] == design)
                   & (COMP_MET["degree"] == int(degree))
                   & (COMP_MET["method"] == method)
                   & (COMP_MET["point_summary"] == "mean")
                   & (COMP_MET["estimand_type"] == "PRE")
                   & (COMP_MET["estimand_id"] == "slope")]
    return sub.iloc[0].to_dict()


def _pt_twfe(design, degree):
    sub = PUB_MET[(PUB_MET["setting"] == design) & (PUB_MET["method"] == TWFE_ES)
                  & (PUB_MET["linearity_degree"] == int(degree))
                  & (PUB_MET["estimand_type"] == "PRE")
                  & (PUB_MET["estimand_id"] == "slope")]
    return float(sub.iloc[0]["reject05"])


def _pt_kprofile(design, degree, method):
    sub = COMP_MET[(COMP_MET["design"] == design)
                   & (COMP_MET["degree"] == int(degree))
                   & (COMP_MET["method"] == method)
                   & (COMP_MET["point_summary"] == "mean")
                   & (COMP_MET["estimand_type"] == "PRE")
                   & (COMP_MET["estimand_id"].str.startswith("k="))].copy()
    sub["kk"] = sub["estimand_id"].str.slice(2).astype(int)
    return sub.sort_values("kk")


def fig_pretrend(degree, path):
    """Detection curves and Delta(k) profiles for raw vs fold diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.5))
    ax = axes[0]
    hold = {"true_slope": 0.0,
            "raw": _pt_detect("PT_hold", degree, "pretrend"),
            "ref": _pt_detect("PT_hold", degree, "reference_fold_pretrend"),
            "twfe": _pt_twfe("PT_hold", degree)}
    for fam, ls in (("group", "-"), ("alpha", "-.")):
        designs = ([f"PT_violation_g{v:02d}" for v in (5, 10, 20, 40)]
                   if fam == "group" else ["PT_violation_a20"])
        xs = [0.0] + [{"PT_violation_g05": 0.05, "PT_violation_g10": 0.10,
                        "PT_violation_g20": 0.20, "PT_violation_g40": 0.40,
                        "PT_violation_a20": 0.207}[d] for d in designs]
        raw = [hold["raw"]] + [_pt_detect(d, degree, "pretrend") for d in designs]
        ref = [hold["ref"]] + [_pt_detect(d, degree, "reference_fold_pretrend")
                               for d in designs]
        tw = [hold["twfe"]] + [_pt_twfe(d, degree) for d in designs]
        for ys, meth, tag in ((raw, "pretrend", "raw diagnostic"),
                              (ref, "reference_fold_pretrend",
                               "ref. fold diagnostic"),
                              (tw, "twfe_es", "TWFE placebo")):
            ax.plot(xs, ys, color=COLOR[meth], linestyle=ls, lw=1.5, ms=4.5,
                    marker=MARKER[meth],
                    label=tag + ("" if fam == "group" else ", via $\\alpha_i$"))
    ax.axhline(NOMINAL, color="#9a9a95", lw=0.8, ls=":", zorder=1)
    ax.annotate("nominal 5%", (0.0, NOMINAL), fontsize=7,
                va="bottom", ha="left", color="#52514e")
    ax.set_xlabel("true differential pre-trend slope")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) Size at 0, power beyond", fontsize=9.5, loc="left")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.axhline(0, color="#9a9a95", lw=0.8, zorder=1)
    cmap = plt.get_cmap("viridis")
    for i, design in enumerate([d for d in PT_SHOW]):
        c = cmap(i / max(len(PT_SHOW) - 1, 1) * 0.85)
        for meth, ls, tag in (("pretrend", "-", "raw"),
                              ("reference_fold_pretrend", "--", "ref. fold")):
            g = _pt_kprofile(design, degree, meth)
            if g.empty:
                continue
            ax.plot(g["kk"], g["mean_true"], color=c, lw=2.6, alpha=0.28,
                    zorder=2)
            ax.errorbar(g["kk"], g["mean_true"] + g["bias"],
                        yerr=g["empirical_sd_error"], color=c, marker="o",
                        ms=4, lw=1.4, capsize=2.5, linestyle=ls,
                        label=f"{design.replace('PT_', '')} ({tag})", zorder=3)
    ax.set_xlabel("event time $k$ (reference $k=-1$)")
    ax.set_ylabel(r"$\Delta(k)$")
    ax.set_title(r"(b) Estimated vs true $\Delta(k)$", fontsize=9.5, loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  ->", os.path.relpath(path, ROOT))


REGENERATED = ["fig_sweep.pdf", "fig_sqrtn.pdf", "fig_calibration.pdf",
               "fig_surface.pdf", "fig_surface_sweep.pdf",
               "fig_es_methods.pdf", "fig_es_bias.pdf", "fig_smalln.pdf",
               "fig_pretrend.pdf", "fig_pretrend_d3.pdf"]


def main():
    os.makedirs(FIG, exist_ok=True)
    _register_styles()
    met = _combined_met()
    ma.met = met
    ma.surf = _combined_surf()
    ma.sqn = _combined_sqn()
    ma.fig_sweep()
    ma.fig_sqrtn()
    ma.fig_calibration()
    ma.fig_surface()
    ma.fig_surface_sweep()
    ma.fig_es_methods()
    ma.fig_es_bias()
    ms.make_figure(met, _combined_surf(), os.path.join(FIG, "fig_smalln.pdf"))
    fig_pretrend(1, os.path.join(FIG, "fig_pretrend.pdf"))
    fig_pretrend(3, os.path.join(FIG, "fig_pretrend_d3.pdf"))
    for name in REGENERATED:
        shutil.copy(os.path.join(FIG, name), os.path.join(SIMFIG, name))
        print("  -> sim_figures/" + name)
    shutil.copy(os.path.join(FIG, "fig_smalln.pdf"),
                os.path.join(IMAGES, "fig_smalln.pdf"))
    print("  -> images/fig_smalln.pdf")


if __name__ == "__main__":
    main()
