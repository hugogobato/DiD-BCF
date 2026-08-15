#!/usr/bin/env python3
"""Workstream F4, Reviewer 3.2.3: operating characteristics of the pre-trend
diagnostic.

The manuscript proposes the diagnostic but never demonstrates it.  This turns
the ``PT_*`` Monte-Carlo runs into the three things a reviewer will ask for:

1. **Size.**  ``PT_hold`` -- conditional parallel trends holds, so every
   rejection is a false positive.
2. **Power.**  ``PT_violation_g*`` (a differential group slope, 0.05 to 0.40 per
   period) and ``PT_violation_a*`` (the violation routed through the
   *unobserved* confounder), against the standard-practice TWFE event-study
   placebo run on the same replications.
3. **Discrimination.**  ``PT_conditional`` -- conditional PT holds while
   *unconditional* PT fails, because assignment depends on a covariate that also
   shifts trends.  The estimator's assumption is satisfied here, so a useful
   diagnostic should stay quiet; a marginal event study, which does not
   condition, should not.

With ``--with-att`` runs present it also plots the *consequence*: the ATT bias
the violation induces in plain and corrected DiD-BCF, so detection is tied to
the error it is warning about.

Inputs
------
``Results/aggregated/metrics_all.csv``   (``Results/analysis/aggregate_all.py``,
which now globs ``Results/summaries_pretrend_*.csv``)

Outputs
-------
``Results/figures/fig_pretrend.pdf``   power curve, Delta(k) profiles, ATT bias
``Results/tables/tab_pretrend.tex``    size / power / discrimination, per scenario
``Results/aggregated/pretrend_all.csv`` the tidy join, for reuse
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.dirname(HERE)
ROOT = os.path.dirname(RES)
AGG = os.path.join(RES, "aggregated")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")

import matplotlib.pyplot as plt                                  # noqa: E402
from style import COLOR, MARKER, PLAINLABEL                      # noqa: E402

DIAG, COMP = "pretrend", "twfe_es"
NOMINAL = 0.05

# Two decision rules, reported side by side because they behave very differently.
# ``slope`` tests the single differential-trend coefficient implied by Delta(k);
# ``any_bonf`` flags if *any* pre-period Delta(k) is individually significant
# after a Bonferroni scaling.  When the violation is a linear differential trend
# -- which is what a pre-trend test is usually worried about -- the slope rule
# concentrates the evidence into one statistic and the any-k rule spends power on
# a multiplicity correction, so the slope rule is the headline and the any-k rule
# is reported as the conservative alternative.
RULES = {"slope": "slope", "anyk": "any_bonf"}
HEADLINE = "slope"


def load(degree: int = 1) -> pd.DataFrame:
    path = os.path.join(AGG, "metrics_all.csv")
    if not os.path.exists(path):
        sys.exit("Missing " + os.path.relpath(path, ROOT) +
                 "\nRun: python Results/analysis/aggregate_all.py")
    m = pd.read_csv(path)
    m = m[m["setting"].astype(str).str.startswith("PT_") &
          (m["linearity_degree"] == degree)]
    if m.empty:
        sys.exit(f"No PT_* results at linearity degree {degree}. Run "
                 "scripts/run_pretrend.py first.")
    return m


def summarise(m: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario: realised violation, detection rates, slope recovery."""
    rows = []
    for setting, g in m.groupby("setting"):
        pre = g[g["estimand_type"] == "PRE"]
        rec = {"setting": setting}
        sl = pre[(pre["estimand_id"] == "slope") & (pre["method"] == DIAG)]
        rec["true_slope"] = float(sl["mean_true"].iloc[0]) if not sl.empty else np.nan
        rec["n_reps"] = int(sl["n_reps"].iloc[0]) if not sl.empty else 0
        rec["slope_bias"] = float(sl["bias"].iloc[0]) if not sl.empty else np.nan
        rec["slope_cover95"] = float(sl["cover95"].iloc[0]) if not sl.empty else np.nan
        for rule, eid in RULES.items():
            for meth, tag in ((DIAG, "diag"), (COMP, "twfe")):
                r = pre[(pre["estimand_id"] == eid) & (pre["method"] == meth)]
                rec[f"{tag}_detect_{rule}"] = (float(r["reject05"].iloc[0])
                                               if not r.empty else np.nan)
                rec[f"{tag}_mcse_{rule}"] = (float(r["mcse_reject05"].iloc[0])
                                             if not r.empty else np.nan)
        for tag in ("diag", "twfe"):
            rec[f"{tag}_detect"] = rec[f"{tag}_detect_{HEADLINE}"]
        att = g[(g["estimand_type"] == "ATT") & (g["estimand_id"] == "ATT")]
        for meth in ("plain", "corrected"):
            a = att[att["method"] == meth]
            rec[f"att_bias_{meth}"] = float(a["bias"].iloc[0]) if not a.empty else np.nan
        rec["family"] = ("hold" if setting == "PT_hold" else
                         "conditional" if setting == "PT_conditional" else
                         "group_trend" if "_g" in setting else "alpha_trend")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["family", "true_slope"])


def make_figure(s: pd.DataFrame, m: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    # --- (a) detection rate against the realised violation ------------------ #
    ax = axes[0]
    hold = s[s["family"] == "hold"]
    for fam, ls in (("group_trend", "-"), ("alpha_trend", "-.")):
        f = pd.concat([hold, s[s["family"] == fam]]).sort_values("true_slope")
        if f.empty:
            continue
        for meth, tag in ((DIAG, "diag"), (COMP, "twfe")):
            ax.plot(f["true_slope"], f[f"{tag}_detect"], color=COLOR[meth],
                    marker=MARKER[meth], linestyle=ls, lw=1.5, ms=4.5,
                    label=f"{PLAINLABEL[meth]}"
                          + ("" if fam == "group_trend" else ", via $\\alpha_i$"))
    ax.axhline(NOMINAL, color="#9a9a95", lw=0.8, ls=":", zorder=1)
    ax.annotate("nominal 5%", (ax.get_xlim()[0], NOMINAL), fontsize=7,
                va="bottom", ha="left", color="#52514e")
    ax.set_xlabel("true differential pre-trend slope")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) Size at 0, power beyond", fontsize=9.5, loc="left")
    ax.legend(fontsize=7)

    # --- (b) the Delta(k) profile the diagnostic reports -------------------- #
    ax = axes[1]
    ax.axhline(0, color="#9a9a95", lw=0.8, zorder=1)
    show = ["PT_hold", "PT_conditional", "PT_violation_g10", "PT_violation_g40"]
    cmap = plt.get_cmap("viridis")
    ks: set = set()
    for i, setting in enumerate([x for x in show if x in set(m["setting"])]):
        g = m[(m["setting"] == setting) & (m["method"] == DIAG) &
              (m["estimand_type"] == "PRE") &
              (m["estimand_id"].str.startswith("k="))].copy()
        if g.empty:
            continue
        g["kk"] = g["estimand_id"].str.slice(2).astype(int)
        g = g.sort_values("kk")
        ks |= set(g["kk"])
        c = cmap(i / max(len(show) - 1, 1) * 0.85)
        # The pale wide line is the truth; the marked line with error bars is
        # the diagnostic's Monte-Carlo mean +/- its across-replication SD.
        ax.plot(g["kk"], g["mean_true"], color=c, lw=2.6, alpha=0.28, zorder=2)
        ax.errorbar(g["kk"], g["mean_true"] + g["bias"], yerr=g["emp_sd"],
                    color=c, marker="o", ms=4, lw=1.4, capsize=2.5,
                    label=setting.replace("PT_", ""), zorder=3)
    if ks:
        ax.set_xticks(sorted(ks | {-1}))
    ax.set_xlabel("event time $k$ (reference $k=-1$)")
    ax.set_ylabel(r"$\Delta(k)$")
    ax.set_title(r"(b) Estimated vs true $\Delta(k)$", fontsize=9.5, loc="left")
    ax.legend(fontsize=7)

    # --- (c) the consequence the diagnostic is warning about ---------------- #
    ax = axes[2]
    ax.axhline(0, color="#9a9a95", lw=0.8, zorder=1)
    f = pd.concat([hold, s[s["family"] == "group_trend"]]).sort_values("true_slope")
    plotted = False
    for meth in ("plain", "corrected"):
        col = f"att_bias_{meth}"
        if col in f and f[col].notna().any():
            ax.plot(f["true_slope"], f[col], color=COLOR[meth],
                    marker=MARKER[meth], lw=1.5, ms=4.5,
                    label=PLAINLABEL[meth])
            plotted = True
    if plotted:
        ax.legend(fontsize=7)
        ax.set_ylabel("ATT bias")
    else:
        ax.text(0.5, 0.5, "run with --with-att\nto populate", ha="center",
                va="center", transform=ax.transAxes, fontsize=8, color="#52514e")
    ax.set_xlabel("true differential pre-trend slope")
    ax.set_title("(c) ATT bias caused by the violation", fontsize=9.5, loc="left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {os.path.relpath(path, ROOT)}")


def _f(x, n=3):
    return "--" if pd.isna(x) else f"{x:.{n}f}"


def make_table(s: pd.DataFrame, path: str) -> None:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Scenario & Violation & True & "
             r"\multicolumn{2}{c}{Detection, slope rule} & "
             r"\multicolumn{1}{c}{Any-$k$} & "
             r"\multicolumn{2}{c}{Slope recovery} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}",
             r" & & slope & Diagnostic & TWFE ES & Diagnostic "
             r"& bias & cover.\ 95\% \\",
             r"\midrule"]
    fam_label = {"hold": "none (PTA holds)", "conditional": "unconditional only",
                 "group_trend": "group slope", "alpha_trend": r"via $\alpha_i$"}
    for _, r in s.iterrows():
        lines.append(" & ".join([
            r["setting"].replace("_", r"\_"), fam_label[r["family"]],
            _f(r["true_slope"]), _f(r["diag_detect_slope"]),
            _f(r["twfe_detect_slope"]), _f(r["diag_detect_anyk"]),
            _f(r["slope_bias"]), _f(r["slope_cover95"], 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  -> {os.path.relpath(path, ROOT)}")


def main() -> None:
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    degree = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    m = load(degree)
    s = summarise(m)
    s.to_csv(os.path.join(AGG, "pretrend_all.csv"), index=False)
    print(f"  -> {os.path.relpath(os.path.join(AGG, 'pretrend_all.csv'), ROOT)}")
    make_figure(s, m, os.path.join(FIG, "fig_pretrend.pdf"))
    make_table(s, os.path.join(TAB, "tab_pretrend.tex"))
    print()
    cols = ["setting", "family", "true_slope", "n_reps", "diag_detect_slope",
            "twfe_detect_slope", "diag_detect_anyk", "slope_bias", "slope_cover95"]
    print(s[cols].to_string(index=False, float_format=lambda v: f"{v: .3f}"))


if __name__ == "__main__":
    main()
