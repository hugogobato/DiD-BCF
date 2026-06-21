#!/usr/bin/env python3
"""Figures for the sample-size sweep (Workstream B2): bias -> 0, variance -> 0,
sqrt(N) stabilisation and coverage, overlaying plain vs corrected DiD-BCF.

Consumes ``Results/metrics_long.csv`` and ``Results/sqrt_n_ATT.csv`` (produced by
``aggregate_metrics.py``).  Writes PNGs to ``Results/``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
_STYLES = {"plain": dict(marker="o", linestyle="-"),
           "corrected": dict(marker="s", linestyle="--")}


def _plot_metric(ax, df, ycol, title, ylabel, hline=None, logy=False):
    for method, g in df.groupby("method"):
        g = g.sort_values("N")
        ax.plot(g["N"], g[ycol], label=method, **_STYLES.get(method, {}))
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle=":", linewidth=1,
                   label=f"nominal {hline:g}")
    if logy:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("N (units)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--setting", default="B2_sweep",
                    help="which sample-size-sweep setting to plot")
    ap.add_argument("--linearity-degree", type=int, default=1,
                    help="which linearity degree to plot (1/2/3)")
    args = ap.parse_args()

    metrics_path = os.path.join(args.results, "metrics_long.csv")
    sqrt_path = os.path.join(args.results, "sqrt_n_ATT.csv")
    if not os.path.exists(metrics_path):
        sys.exit(f"Missing {metrics_path}; run aggregate_metrics.py first.")

    d = int(args.linearity_degree)
    metrics = pd.read_csv(metrics_path)
    att = metrics[(metrics["estimand_type"] == "ATT") &
                  (metrics["setting"] == args.setting)].copy()
    if "linearity_degree" in att.columns:
        att = att[att["linearity_degree"] == d]
    if att.empty:
        sys.exit(f"No ATT rows for setting {args.setting!r} "
                 f"(linearity_degree={d}) in {metrics_path}.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _plot_metric(axes[0, 0], att, "abs_bias", "Absolute bias", "|bias|")
    _plot_metric(axes[0, 1], att, "emp_sd", "Monte-Carlo SD (variance piece)",
                 "SD of estimate", logy=True)
    _plot_metric(axes[0, 2], att, "rmse", "RMSE", "RMSE", logy=True)
    _plot_metric(axes[1, 0], att, "cover95", "95% CI coverage", "coverage",
                 hline=0.95)
    _plot_metric(axes[1, 1], att, "len95", "95% CI length", "length", logy=True)

    if os.path.exists(sqrt_path):
        sq = pd.read_csv(sqrt_path)
        sq = sq[sq["setting"] == args.setting]
        if "linearity_degree" in sq.columns:
            sq = sq[sq["linearity_degree"] == d]
        ax = axes[1, 2]
        for method, g in sq.groupby("method"):
            g = g.sort_values("N")
            ax.plot(g["N"], g["sd"], label=f"{method} (sd)", **_STYLES.get(method, {}))
        ax.set_xscale("log")
        ax.set_xlabel("N (units)")
        ax.set_ylabel(r"SD of $\sqrt{N}(\hat{ATT}-ATT)$")
        ax.set_title(r"$\sqrt{N}$ stabilisation")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 2].axis("off")

    fig.suptitle(f"DiD-BCF sample-size sweep ({args.setting}, linearity={d}): "
                 "plain vs posterior-corrected", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.results, f"sweep_{args.setting}_lin_{d}.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
