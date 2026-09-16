#!/usr/bin/env python3
"""The `mpdta` application under the corrected estimator, with GATE by population.

This is the analysis of ``Empirical_Study/DiD_BCF_GATE_CATE_Empirical_Study.ipynb``
redone under the estimator the simulation grid actually uses.  It keeps that
notebook's structure -- overall effect, terciles of ``lpop``, the KDE figure --
so the two are directly comparable, and fixes two things the original could not
report.

**What the original figure shows.**  It plots a KDE of
``exp(mean_over_draws(tau_i)) - 1`` across treated counties.  Each county
contributes one number, the posterior *mean* of its own effect, so the spread of
that curve is cross-county heterogeneity in the fitted surface.  It is not
posterior uncertainty, and it is roughly an order of magnitude narrower, which
is why the published figure looks far more precise than the estimate is.  Both
quantities are drawn here, side by side and labelled for what they are.

**Which specification.**  ``structured_rfx_unit`` is the headline: the two-way
restriction identifies the effect surface (see ``Results/identification_note.tex``)
and the unit-level random intercepts absorb county-level unobserved
heterogeneity, which is what halves the posterior SD and moves the interval off
zero.  ``structured`` is run alongside so the contribution of the intercepts is
visible rather than asserted.

Outputs (all under ``Results/empirical/``)::

    mpdta_structured_gate.csv        ATT and per-tercile GATE, both specs
    mpdta_structured_catt.csv        per-county posterior mean CATT, both specs
    fig_mpdta_gate_kde.pdf/.png      the two-panel KDE figure

Usage::

    python scripts/run_empirical_structured.py                 # both specs
    python scripts/run_empirical_structured.py --spec structured_rfx_unit
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(ROOT))          # for didbcf_structured

from didbcf_structured import StructuredDiDBCF                   # noqa: E402
from run_pretrend_empirical import DEFAULT_DATA, load_mpdta      # noqa: E402

OUT = os.path.join(ROOT, "Results", "empirical")
BCF = dict(num_gfr=50, num_mcmc=500, keep_every=5, num_chains=3)
RFX = {"structured": "none", "structured_rfx_unit": "unit"}

# mpdta's only covariate is log county population; it is both the prognostic
# covariate and the effect modifier, as in the original notebook.
COVARIATES = ["lpop"]
EFFECT_MODIFIERS = ["lpop"]

GROUPS = {1: "Small population", 2: "Medium population", 3: "High population"}
GROUP_COLOR = {1: "#3b6ea5", 2: "#4a8c5f", 3: "#b5533c"}


def fit(df: pd.DataFrame, spec: str, seed: int) -> StructuredDiDBCF:
    return StructuredDiDBCF(rfx=RFX[spec]).sample(
        df, covariates=COVARIATES, effect_modifiers=EFFECT_MODIFIERS,
        seed=seed, **BCF)


def _summarise(draws: np.ndarray, label: str, spec: str, n: int) -> dict:
    """Posterior summary of a scalar built from the tau draws."""
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(
        spec=spec, group=label, n_obs=n,
        att_log=float(draws.mean()), post_sd=float(draws.std(ddof=1)),
        lo95_log=float(lo), hi95_log=float(hi),
        # The notebook reports effects as percentages; exp(.)-1 of the posterior
        # mean is not the posterior mean of exp(.)-1, so transform the draws.
        att_pct=float(np.mean(np.exp(draws) - 1) * 100),
        lo95_pct=float((np.exp(lo) - 1) * 100),
        hi95_pct=float((np.exp(hi) - 1) * 100),
        p_bayes=float(2 * min((draws > 0).mean(), (draws < 0).mean())),
    )


def analyse(model: StructuredDiDBCF, spec: str):
    d = model.df.reset_index(drop=True)
    tau = np.asarray(model.tau_draws, dtype=float)          # (n_obs, n_draws)

    # Terciles over the whole panel, as the original notebook does, then
    # restricted to treated observations.
    d["lpop_group"] = pd.qcut(d["lpop"], q=3, labels=[1, 2, 3]).astype(int)
    treated = d["D"].to_numpy() == 1

    rows = [_summarise(tau[treated, :].mean(axis=0), "All treated", spec,
                       int(treated.sum()))]
    for g, name in GROUPS.items():
        m = treated & (d["lpop_group"].to_numpy() == g)
        if m.sum():
            rows.append(_summarise(tau[m, :].mean(axis=0), name, spec, int(m.sum())))

    catt = d.loc[treated, ["unit_id", "time", "lpop", "lpop_group"]].copy()
    catt["spec"] = spec
    catt["catt_log"] = tau[treated, :].mean(axis=1)
    catt["catt_pct"] = (np.exp(catt["catt_log"]) - 1) * 100

    # Overlapping credible intervals are not a test of whether two groups
    # differ. The posterior of the *difference* is, and it is available from the
    # same draws, so compute it rather than inviting the reader to eyeball the
    # intervals.
    contrasts = []
    gd = {g: tau[treated & (d["lpop_group"].to_numpy() == g), :].mean(axis=0)
          for g in GROUPS if (treated & (d["lpop_group"].to_numpy() == g)).any()}
    for a, b in ((1, 2), (1, 3), (2, 3)):
        if a in gd and b in gd:
            diff = gd[b] - gd[a]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            contrasts.append(dict(
                spec=spec, contrast=f"{GROUPS[b]} - {GROUPS[a]}",
                diff_log=float(diff.mean()), post_sd=float(diff.std(ddof=1)),
                lo95_log=float(lo), hi95_log=float(hi),
                p_bayes=float(2 * min((diff > 0).mean(), (diff < 0).mean()))))
    return pd.DataFrame(rows), catt, pd.DataFrame(contrasts)


def make_figure(catt: pd.DataFrame, gate: pd.DataFrame, spec: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    c = catt[catt["spec"] == spec]
    g = gate[gate["spec"] == spec].set_index("group")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: the original notebook's figure -- spread of the per-county posterior
    # mean. Same quantity, same construction, honestly labelled.
    ax = axes[0]
    for grp, name in GROUPS.items():
        v = c.loc[c["lpop_group"] == grp, "catt_pct"] / 100
        if v.empty:
            continue
        sns.kdeplot(v, ax=ax, fill=True, color=GROUP_COLOR[grp], alpha=0.35,
                    lw=1.4, label=f"{name} (mean: {v.mean():.3f})")
    ax.axvline(0, color="#8a8a85", lw=0.8, ls=":")
    ax.set_title("Across-county heterogeneity in the fitted effect\n"
                 "(spread of the per-county posterior mean)", fontsize=10, loc="left")
    ax.set_xlabel("County-level CATT, $\\exp(\\hat\\tau_i)-1$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8.5)

    # Right: what the estimate's uncertainty actually is.
    ax = axes[1]
    order = [n for n in ["All treated"] + list(GROUPS.values()) if n in g.index]
    inv = {v: k for k, v in GROUPS.items()}
    lo = float(g.loc[order, "lo95_pct"].min())
    hi = float(g.loc[order, "hi95_pct"].max())
    pad = max(0.6, 0.06 * (hi - lo))
    # Reserve a fixed column on the right for the annotations so they are all
    # left-aligned at the same x and cannot collide with the intervals, the zero
    # line, or the axis edge.
    label_x = hi + pad
    ax.set_xlim(lo - pad, label_x + 0.62 * (hi - lo))

    for i, name in enumerate(order):
        r = g.loc[name]
        col = "#4a4a45" if name == "All treated" else GROUP_COLOR[inv[name]]
        ax.plot([r["lo95_pct"], r["hi95_pct"]], [i, i], color=col, lw=2.4,
                solid_capstyle="round", zorder=2)
        ax.plot([r["att_pct"]], [i], "o", color=col, ms=7, zorder=3)
        ax.text(label_x, i,
                f"{r['att_pct']:+.1f}%  [{r['lo95_pct']:+.1f}, {r['hi95_pct']:+.1f}]",
                va="center", ha="left", fontsize=8.5, color=col,
                fontfamily="DejaVu Sans Mono")
    ax.axvline(0, color="#b03a2e", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(len(order) - 0.5, -0.7)
    ax.set_title("Posterior uncertainty in the group effect\n"
                 "(95% credible intervals)", fontsize=10, loc="left")
    ax.set_xlabel("Effect on employment (%)")
    # The reserved label column is not data; keep it off the tick axis.
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.suptitle(f"mpdta under {spec}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=150)
    print(f"  -> {os.path.relpath(path, ROOT)}.pdf / .png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", nargs="+", default=list(RFX), choices=list(RFX))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = load_mpdta(args.data).sort_values(["unit_id", "time"]).reset_index(drop=True)
    print(f"mpdta: {len(df)} rows, {df.unit_id.nunique()} counties, "
          f"{int(df.D.sum())} treated observations")

    gates, catts, cons = [], [], []
    for spec in args.spec:
        print(f"\nfitting {spec} (seed {args.seed}) ...", flush=True)
        model = fit(df, spec, args.seed)
        gate, catt, con = analyse(model, spec)
        gates.append(gate); catts.append(catt); cons.append(con)
        print(gate[["group", "n_obs", "att_pct", "lo95_pct", "hi95_pct",
                    "p_bayes"]].round(3).to_string(index=False))
        print("\n  between-group contrasts (log points):")
        print(con[["contrast", "diff_log", "lo95_log", "hi95_log", "p_bayes"]]
              .round(4).to_string(index=False))

    gate = pd.concat(gates, ignore_index=True)
    catt = pd.concat(catts, ignore_index=True)
    con = pd.concat(cons, ignore_index=True)
    gate.to_csv(os.path.join(OUT, "mpdta_structured_gate.csv"), index=False)
    catt.to_csv(os.path.join(OUT, "mpdta_structured_catt.csv"), index=False)
    con.to_csv(os.path.join(OUT, "mpdta_structured_contrasts.csv"), index=False)
    print(f"\n  -> Results/empirical/mpdta_structured_gate.csv")
    print(f"  -> Results/empirical/mpdta_structured_catt.csv")
    print(f"  -> Results/empirical/mpdta_structured_contrasts.csv")

    headline = "structured_rfx_unit" if "structured_rfx_unit" in args.spec else args.spec[0]
    make_figure(catt, gate, headline, os.path.join(OUT, "fig_mpdta_gate_kde"))


if __name__ == "__main__":
    raise SystemExit("Historical application writer is frozen after the prior repair. "
                     "Use Proper_Prior_Rerun/campaign.py run --phase application.")
