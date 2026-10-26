"""Standalone comparison figure from verified metrics; no fitted models."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/didbcf-paper-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    comparison = pd.read_csv(HERE / "paired_metric_contrasts.csv")
    att = comparison.loc[comparison.estimand_type.eq("ATT")].sort_values(["setting","N","linearity_degree"])
    surfaces = comparison.loc[comparison.estimand_type.eq("CATT")]
    surfaces = att[["dgp","setting","N","linearity_degree"]].merge(surfaces, validate="one_to_one")
    y = np.arange(len(att))
    fig, axes = plt.subplots(1, 3, figsize=(14, 15), sharey=True,
                             gridspec_kw={"width_ratios": [1,1,1.05]})
    for axis, frame, title, color in zip(axes[:2], [att,surfaces],
            ["ATT RMSE change", "Treated CATT surface RMSE change"], ["#2366a3","#23764b"]):
        axis.hlines(y, 100*(frame.rmse_ratio_lo95-1), 100*(frame.rmse_ratio_hi95-1), color=color, lw=1)
        axis.scatter(100*(frame.rmse_ratio-1), y, s=15, color=color, zorder=3)
        axis.axvline(0, color="black", lw=.8, ls="--")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Change from historical raw (%)\nPaired bootstrap 95% intervals")
        axis.grid(axis="y", alpha=.15)
    axes[0].set_yticks(y, [f"{r.setting}  N={int(r.N)} d={int(r.linearity_degree)}" for r in att.itertuples()],fontsize=7)
    axes[0].invert_yaxis()
    axes[2].scatter(att.coverage95_original, y, marker="o", facecolors="none", edgecolors="#888888", s=20, label="Historical raw")
    axes[2].scatter(att.coverage95_repaired, y, marker="x", color="#2366a3", s=23, label="Repaired earlier")
    axes[2].axvline(.95, color="black", ls="--", lw=.8)
    axes[2].set_xlim(min(att.coverage95_original.min(),att.coverage95_repaired.min())-.025,1.015)
    axes[2].set_title("ATT 95% interval coverage",fontsize=10)
    axes[2].set_xlabel("Observed coverage on matched datasets\nDashed line: nominal 0.95")
    axes[2].grid(axis="y",alpha=.15)
    axes[2].legend(fontsize=8,loc="lower left")
    fig.suptitle("Repaired DiD-BCF main rerun versus historical raw\n48 cells, 100 paired datasets per cell",fontsize=13)
    fig.text(.5,.008,"All 5,100 repaired fits enter full metrics. This paired comparison uses 4,800 historical matches; 300 additional null replications are separate.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.96))
    for suffix in ("png","pdf"):
        fig.savefig(HERE/f"main_comparison.{suffix}",dpi=180,bbox_inches="tight")
    print("Wrote main_comparison.png and main_comparison.pdf")


if __name__ == "__main__":
    main()
