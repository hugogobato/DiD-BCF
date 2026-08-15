#!/usr/bin/env python3
"""Workstream D, Reviewer 3.1.2: every estimator's error against the weight on
already-treated comparisons.

What this fixes
---------------
The original ramp swept ``dynamic_ramp``, which changes how *inconsistent* the
already-treated comparisons are but leaves their Goodman-Bacon **weight** exactly
constant (0.116 in both D designs) -- the weight is a function of the adoption
design alone.  It therefore could not support a claim about what happens "as
already-treated comparison weight increases", and it reported only TWFE.

Here the x-axis is the *measured* weight on ``Later_vs_Earlier`` comparisons,
moved from 0.077 to 0.250 by shrinking the never-treated pool
(``config.NEVER_TREATED_SHARES``) with the effect DGP held fixed, and every
estimator in the suite is plotted on it.

Inputs
------
``Results/goodman_bacon_design_sweep_D_staggered.csv``  the weights per design
    (``scripts/run_goodman_bacon.py --design-sweep``)
``Results/aggregated/metrics_all.csv``                  every estimator's
    decomposed metrics (``Results/analysis/aggregate_all.py``)

Because estimators report different estimands natively -- an overall ATT
(DiD-BCF, TWFE, synthdid), cohort-time cells (Callaway--Sant'Anna, DoubleML,
grf-DiD) or event-study coefficients (Gardner) -- the comparison is drawn in
three internally consistent panels rather than forcing one number.  Cell-based
methods are summarised by the mean of their per-cell errors, the same convention
the report's event-study figures use.

Outputs
-------
``Results/figures/fig_ramp_methods.pdf``  error vs already-treated weight
``Results/tables/tab_ramp_methods.tex``   the same numbers, per weight
``Results/aggregated/ramp_methods.csv``   the tidy join, for reuse
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

import matplotlib.pyplot as plt                                    # noqa: E402
from style import COLOR, MARKER, DASH, PLAINLABEL, TEXLABEL, METHOD_ORDER  # noqa: E402

# One panel per natively reported estimand, so no method is scored on a target
# it does not estimate.
PANELS = [
    ("ATT", "Overall ATT", lambda d: d["estimand_type"] == "ATT"),
    ("GATT", r"Cohort-time cells $\mathrm{GATT}(g,t)$, mean over cells",
     lambda d: d["estimand_type"] == "GATT"),
    ("ES", r"Event study $\mathrm{ATT}(k)$, mean over $k$",
     lambda d: d["estimand_type"] == "ES"),
]


def load() -> "tuple[pd.DataFrame, pd.DataFrame]":
    wpath = os.path.join(RES, "goodman_bacon_design_sweep_D_staggered.csv")
    if not os.path.exists(wpath):
        sys.exit("Missing " + os.path.relpath(wpath, ROOT) + "\nRun: python "
                 "scripts/run_goodman_bacon.py --experiment D_staggered "
                 "--design-sweep --reps 200 --jobs 3")
    weights = pd.read_csv(wpath)

    mpath = os.path.join(AGG, "metrics_all.csv")
    if not os.path.exists(mpath):
        sys.exit("Missing " + os.path.relpath(mpath, ROOT) +
                 "\nRun: python Results/analysis/aggregate_all.py")
    metrics = pd.read_csv(mpath)
    return weights, metrics


def collapse(metrics: pd.DataFrame, weights: pd.DataFrame,
             degree: int) -> pd.DataFrame:
    """Mean per-cell error per (setting, method, estimand_type), joined to weight."""
    m = metrics[metrics["setting"].isin(weights["setting"]) &
                (metrics["linearity_degree"] == degree)].copy()
    if m.empty:
        return m
    m["rel_abs_bias"] = np.abs(m["bias"]) / np.abs(m["mean_true"])
    agg = (m.groupby(["setting", "method", "estimand_type"], as_index=False)
             .agg(n_cells=("estimand_id", "nunique"),
                  n_reps=("n_reps", "min"),
                  retention=("retention", "min"),
                  mean_true=("mean_true", "mean"),
                  bias=("bias", "mean"),
                  abs_bias=("bias", lambda s: float(np.mean(np.abs(s)))),
                  rel_abs_bias=("rel_abs_bias", "mean"),
                  rmse=("rmse", "mean"),
                  cover95=("cover95", "mean"),
                  mcse_bias=("mcse_bias", "mean")))
    out = agg.merge(weights[["setting", "never_treated_share",
                             "w_already_treated", "w_earlier_vs_later",
                             "w_treated_vs_untreated"]],
                    on="setting", how="left")
    out["linearity_degree"] = degree

    # `retention` comes from the metrics layer, which computes every statistic
    # on the replications where the estimator returned a finite estimate and
    # records what fraction that was. Carrying it here keeps a selected
    # subsample visible instead of averaged away: Callaway--Sant'Anna's
    # estimability degrades monotonically along this axis (0.995 down to 0.85),
    # which is itself an answer to the reviewer's question.
    return out.sort_values(["estimand_type", "method", "w_already_treated"])


def make_figure(tidy: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, len(PANELS), figsize=(11.4, 6.0), sharex=True)
    for j, (etype, title, _) in enumerate(PANELS):
        sub = tidy[tidy["estimand_type"] == etype]
        for row, (col, ylab) in enumerate(
                [("abs_bias", "mean $|$bias$|$"), ("cover95", "95% coverage")]):
            ax = axes[row, j]
            if row == 0:
                ax.set_title(title, fontsize=9.5, loc="left")
            drawn = 0
            for meth in METHOD_ORDER:
                g = sub[sub["method"] == meth].sort_values("w_already_treated")
                if g.empty:
                    continue
                ax.plot(g["w_already_treated"], g[col],
                        color=COLOR[meth], marker=MARKER[meth],
                        linestyle=DASH[meth], lw=1.5, ms=4.5,
                        label=PLAINLABEL[meth])
                # Ring any point computed on a selected subsample, so a value
                # that looks well-behaved cannot be read as one that is.
                inc = g[g["retention"] < 0.99]
                if not inc.empty:
                    ax.scatter(inc["w_already_treated"], inc[col],
                               s=110, facecolors="none",
                               edgecolors=COLOR[meth], linewidths=1.4,
                               zorder=5)
                drawn += 1
            if row == 1:
                ax.axhline(0.95, color="#9a9a95", lw=0.8, ls=":", zorder=1)
                ax.set_ylim(-0.03, 1.03)
                if j == len(PANELS) // 2:      # label the axis once, centred
                    ax.set_xlabel("Goodman--Bacon weight on already-treated "
                                  "comparisons")
            if j == 0:
                ax.set_ylabel(ylab)
            if not drawn:
                # An empty panel means "no estimator reporting this estimand has
                # been run yet", which is very different from "no effect".
                ax.text(0.5, 0.5, "no runs yet", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8.5, color="#8a8a85")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    if (tidy["retention"] < 0.99).any():
        ring = plt.Line2D([], [], marker="o", linestyle="none",
                          markerfacecolor="none", markeredgecolor="#4a4a45",
                          markersize=9, markeredgewidth=1.3)
        handles.append(ring)
        labels.append("estimator failed on some replications")
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {os.path.relpath(path, ROOT)}")


def _f(x, n=3):
    return "--" if pd.isna(x) else f"{x:.{n}f}"


def make_table(tidy: pd.DataFrame, path: str) -> None:
    ws = sorted(tidy["w_already_treated"].dropna().unique())
    lines = [r"\begin{tabular}{ll" + "r" * len(ws) + "}", r"\toprule",
             r"& & \multicolumn{%d}{c}{Weight on already-treated comparisons} \\"
             % len(ws),
             r"\cmidrule(lr){3-%d}" % (2 + len(ws)),
             "Estimand & Estimator & " + " & ".join(f"{w:.3f}" for w in ws)
             + r" \\", r"\midrule"]
    for etype, _title, _ in PANELS:
        sub = tidy[tidy["estimand_type"] == etype]
        if sub.empty:
            continue
        first = True
        for meth in METHOD_ORDER:
            g = sub[sub["method"] == meth]
            if g.empty:
                continue
            vals = {round(float(w), 6): v for w, v in
                    zip(g["w_already_treated"], g["abs_bias"])}
            keep = {round(float(w), 6): r for w, r in
                    zip(g["w_already_treated"], g["retention"])}
            cell = []
            for w in ws:
                k = round(float(w), 6)
                txt = _f(vals.get(k, np.nan))
                # A dagger marks a figure computed on the replications where
                # the estimator returned an estimate at all.
                if txt != "--" and keep.get(k, 1.0) < 0.99:
                    txt += r"$^{\dagger}$"
                cell.append(txt)
            lines.append(("%s & %s & %s \\\\" %
                          (etype if first else "", TEXLABEL[meth],
                           " & ".join(cell))))
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    inc = tidy[tidy["retention"] < 0.99]
    if not inc.empty:
        worst = inc.sort_values("retention").iloc[0]
        lines.append(
            r"\par\smallskip\footnotesize $^{\dagger}$Computed on the "
            r"replications in which the estimator returned an estimate: "
            "%s retains %.0f\\%% of replications at this design, so the figure "
            "is a selected subsample and is not comparable with estimators that "
            "ran everywhere."
            % (TEXLABEL[worst["method"]], 100 * worst["retention"]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  -> {os.path.relpath(path, ROOT)}")


def main() -> None:
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    weights, metrics = load()
    have = sorted(set(metrics["setting"]) & set(weights["setting"]))
    missing = sorted(set(weights["setting"]) - set(metrics["setting"]))
    print(f"ramp designs with estimator runs: {have or 'NONE'}")
    if missing:
        print(f"still missing (run the D_ramp_* experiments): {missing}")
    if not have:
        sys.exit("No D_ramp_* estimator results yet -- nothing to plot.")

    frames = [collapse(metrics, weights, d) for d in (1, 2, 3)]
    tidy = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    tidy.to_csv(os.path.join(AGG, "ramp_methods.csv"), index=False)
    print(f"  -> {os.path.relpath(os.path.join(AGG, 'ramp_methods.csv'), ROOT)}")

    d1 = tidy[tidy["linearity_degree"] == 1]
    if d1.empty:
        d1 = tidy
    make_figure(d1, os.path.join(FIG, "fig_ramp_methods.pdf"))
    make_table(d1, os.path.join(TAB, "tab_ramp_methods.tex"))

    print("\nHeadline (degree 1, overall ATT):")
    show = d1[d1["estimand_type"] == "ATT"].pivot_table(
        index="method", columns="w_already_treated", values="abs_bias")
    print(show.round(3).to_string())

    inc = tidy[tidy["retention"] < 0.99]
    if not inc.empty:
        print("\nEstimability along the ramp -- these points are computed on a "
              "SELECTED subsample:")
        print(inc[["setting", "method", "estimand_type", "linearity_degree",
                   "w_already_treated", "n_reps", "retention"]]
              .sort_values(["method", "w_already_treated", "linearity_degree"])
              .to_string(index=False))


if __name__ == "__main__":
    main()
