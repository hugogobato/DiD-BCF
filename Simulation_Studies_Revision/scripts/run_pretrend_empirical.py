#!/usr/bin/env python3
"""Workstream F4 on the empirical application: pre-trend diagnostic for `mpdta`.

Runs the diagnostic of ``did_bcf_revision/pretrend.py`` on the county-level
minimum-wage / teen-employment panel used in the paper's application
(``DiD-BCF/Empirical_Study/mpdta.csv``; Callaway and Sant'Anna, 2021): 500
counties, 2003--2007, adoption cohorts 2004, 2006 and 2007 plus a never-treated
group, outcome ``lemp``, covariate ``lpop``.

It reports, for the pre-treatment event times the panel supports,

    Delta(k) = E[ tau(lpop, k) - tau(lpop, -1) | ever treated ],

zero under conditional parallel trends; the implied differential pre-trend
slope; the same quantities within ``lpop`` terciles (the *conditional* check,
which is the one the estimator's assumption actually requires); and the TWFE
event-study placebo coefficients as the standard-practice comparator.

The diagnostic is a **separate fit** from the estimation model: ``Z`` is the
ever-treated indicator so that ``tau`` is estimable before adoption, ``mu`` may
not split on any group indicator, and stochtree's internal propensity is off.
``--with-att`` additionally refits the paper's constrained specification so the
headline ATT is printed next to the diagnostic that qualifies it.

Which pre-periods are available: cohort 2004 has only 2003 (the reference), so
it contributes to no Delta(k); cohort 2006 contributes at k = -3, -2; cohort
2007 at k = -4, -3, -2.

Examples
--------
    python scripts/run_pretrend_empirical.py
    python scripts/run_pretrend_empirical.py --with-att
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

from did_bcf_revision.pretrend import (fit_pretrend, pretrend_estimands,
                                       quantile_subgroups, twfe_pretrend)

DEFAULT_DATA = os.path.abspath(
    os.path.join(ROOT, "..", "Empirical_Study", "mpdta.csv"))
OUT_DIR = os.path.join(ROOT, "Results", "empirical")


def load_mpdta(path: str) -> pd.DataFrame:
    """`mpdta.csv` in the suite's panel conventions (cohort = inf if never)."""
    raw = pd.read_csv(path)
    g = raw["first.treat"].to_numpy(dtype=float)
    cohort = np.where(g > 0, g, np.inf)
    ever = (g > 0).astype(int)
    year = raw["year"].to_numpy(dtype=float)
    return pd.DataFrame({
        "unit_id": raw["countyreal"].astype(int),
        "time": year.astype(int),
        "cohort": cohort,
        "eventually_treated": ever,
        "event_time": np.where(np.isfinite(cohort), year - cohort, np.nan),
        "D": ((np.isfinite(cohort)) & (year >= cohort)).astype(int),
        "lpop": raw["lpop"].to_numpy(dtype=float),
        "Y": raw["lemp"].to_numpy(dtype=float),
    }).sort_values(["unit_id", "time"]).reset_index(drop=True)


def paper_att(data_path: str, bcf_params: dict, seed: int = 0) -> pd.DataFrame:
    """The application's headline ATT, under the *paper's* specification.

    Deliberately not routed through :func:`did_bcf_revision.did_bcf.fit_did_bcf`:
    that builds a fixed simulation design matrix, which for this dataset means
    dropping ``first.treat`` from the prognostic split set and padding X2..X5
    with zeros.  Under staggered adoption, a prognostic forest that cannot see
    the cohort pushes cohort-level level differences into ``tau`` -- measured
    here, it flipped the sign of the reported effect (+0.269 against the
    published -0.143).  This reproduces
    ``Empirical_Study/DiD_BCF_GATE_CATE_Empirical_Study.ipynb`` exactly:
    ``mu`` splits on ``(treat, lpop, year, first.treat)``, ``tau`` on
    ``(lpop, year)``, and no propensity is passed, so stochtree's internal one is
    used as in the published run.

    Reported on both scales: the log-point ATT and the paper's
    ``exp(tau) - 1`` percentage effect.
    """
    from stochtree import BCFModel

    raw = pd.read_csv(data_path)
    D = (raw["treat"] * (raw["first.treat"] <= raw["year"])).to_numpy(dtype=float)
    X = raw[["treat", "lpop", "year", "first.treat"]].to_numpy(dtype=float)
    y = raw["lemp"].to_numpy(dtype=float)

    model = BCFModel()
    model.sample(X_train=X, Z_train=D, y_train=y,
                 num_gfr=bcf_params["num_gfr"], num_mcmc=bcf_params["num_mcmc"],
                 general_params={"keep_every": bcf_params["keep_every"],
                                 "num_chains": bcf_params["num_chains"],
                                 "random_seed": int(seed)},
                 prognostic_forest_params={"keep_vars": np.array([0, 1, 2, 3])},
                 treatment_effect_forest_params={"keep_vars": np.array([1, 2])})

    tau = np.asarray(model.tau_hat_train, dtype=float)
    if tau.ndim == 1:
        tau = tau[:, None]
    treated = D == 1
    draws = tau[treated, :].mean(axis=0)              # one ATT per draw
    rows = []
    for name, v in (("ATT (log points)", draws),
                    ("ATT (exp(tau)-1)", np.exp(draws) - 1.0)):
        rows.append({"estimand_id": name, "post_mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)),
                     "q025": float(np.quantile(v, 0.025)),
                     "q975": float(np.quantile(v, 0.975)),
                     "p_bayes": float(min(np.mean(v > 0), np.mean(v < 0))),
                     "n_treated_obs": int(treated.sum())})

    # The application notebook's exact statistic: collapse the posterior to a
    # per-observation mean FIRST, then transform, then average over treated
    # observations.  It is a different functional of the same posterior from the
    # two above (Jensen), so it is reported separately rather than conflated.
    per_obs = np.exp(tau[treated, :].mean(axis=1)) - 1.0
    rows.append({"estimand_id": "ATT (notebook: mean of exp(E[tau_i])-1)",
                 "post_mean": float(per_obs.mean()), "sd": float(per_obs.std(ddof=1)),
                 "q025": float(np.quantile(per_obs, 0.025)),
                 "q975": float(np.quantile(per_obs, 0.975)),
                 "p_bayes": np.nan, "n_treated_obs": int(treated.sum())})
    out = pd.DataFrame(rows)
    out.insert(0, "dataset", "mpdta")
    out.insert(1, "source", "did_bcf_constrained_paper_spec")
    return out


def _fmt(df: pd.DataFrame) -> str:
    cols = [c for c in ("estimand_id", "k", "post_mean", "sd", "q025", "q975",
                        "p_bayes") if c in df.columns]
    return df[cols].to_string(index=False, float_format=lambda v: f"{v: .4f}")


def make_figure(diag: pd.DataFrame, twfe: pd.DataFrame, sub: pd.DataFrame,
                path: str) -> None:
    sys.path.insert(0, os.path.join(ROOT, "Results", "analysis"))
    import matplotlib.pyplot as plt
    from style import COLOR, MARKER, PLAINLABEL      # shared report design system

    per_k = diag[diag["estimand_id"].str.startswith("k=")].sort_values("k")
    tw_k = twfe[twfe["estimand_id"].str.startswith("k=")].sort_values("k")

    # The reference period is zero by construction; drawing it makes the anchor
    # of every other point explicit rather than implied by the axis label.
    ref = pd.DataFrame({"k": [-1.0], "post_mean": [0.0], "q025": [0.0],
                        "q975": [0.0]})
    per_k = pd.concat([per_k, ref], ignore_index=True).sort_values("k")
    tw_k = pd.concat([tw_k, ref], ignore_index=True).sort_values("k")
    ks = sorted(int(v) for v in per_k["k"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.axhline(0, color="#9a9a95", lw=0.8, zorder=1)
    ax.errorbar(per_k["k"], per_k["post_mean"],
                yerr=[per_k["post_mean"] - per_k["q025"],
                      per_k["q975"] - per_k["post_mean"]],
                color=COLOR["pretrend"], marker=MARKER["pretrend"], capsize=3,
                lw=1.4, label=PLAINLABEL["pretrend"], zorder=3)
    ax.errorbar(tw_k["k"] + 0.08, tw_k["post_mean"],
                yerr=[tw_k["post_mean"] - tw_k["q025"],
                      tw_k["q975"] - tw_k["post_mean"]],
                color=COLOR["twfe_es"], marker=MARKER["twfe_es"], capsize=3,
                lw=1.4, linestyle="--", label=PLAINLABEL["twfe_es"], zorder=2)
    ax.set_xticks(ks)
    ax.set_xlabel("event time $k$ (reference $k=-1$)")
    ax.set_ylabel(r"$\Delta(k)$   (log employment)")
    ax.set_title("Pre-treatment placebo effects", fontsize=9.5, loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0, color="#9a9a95", lw=0.8, zorder=1)
    if not sub.empty:
        sub = sub.copy()
        sub["group"] = sub["estimand_id"].str.split("_k=").str[0]
        for i, (name, grp) in enumerate(sub.groupby("group")):
            grp = pd.concat([grp, ref], ignore_index=True).sort_values("k")
            ax.errorbar(grp["k"] + 0.06 * (i - 0.5), grp["post_mean"],
                        yerr=[grp["post_mean"] - grp["q025"],
                              grp["q975"] - grp["post_mean"]],
                        marker="os"[i % 2], capsize=3, lw=1.3, label=name,
                        color=["#2a78d6", "#eb6834"][i % 2], zorder=3)
        ax.legend(fontsize=8, title="subgroup", title_fontsize=8)
    ax.set_xticks(ks)
    ax.set_xlabel("event time $k$ (reference $k=-1$)")
    ax.set_ylabel(r"$\Delta(k)$")
    ax.set_title("Conditional check, by county population", fontsize=9.5, loc="left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rfx", default="unit", choices=["unit", "none"])
    ap.add_argument("--with-att", action="store_true",
                    help="also refit the constrained model for the headline ATT")
    ap.add_argument("--num-gfr", type=int, default=50)
    ap.add_argument("--num-mcmc", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=5)
    ap.add_argument("--num-chains", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_mpdta(args.data)
    print(f"mpdta: {df['unit_id'].nunique()} counties x "
          f"{df['time'].nunique()} years; cohorts "
          f"{sorted(c for c in df['cohort'].unique() if np.isfinite(c))}, "
          f"{int((df.drop_duplicates('unit_id')['eventually_treated'] == 0).sum())} "
          f"never-treated")

    bcf_params = dict(num_gfr=args.num_gfr, num_mcmc=args.num_mcmc,
                      keep_every=args.keep_every, num_chains=args.num_chains)
    fit = fit_pretrend(df, bcf_params=bcf_params, seed=args.seed, rfx=args.rfx,
                       prognostic_cols=["lpop", "time"],
                       treatment_cols=["lpop", "k_diag"])
    est = pretrend_estimands(fit, subgroups=quantile_subgroups("lpop"))
    tw = twfe_pretrend(df)

    diag = est[est["estimand_type"] == "PRE"]
    sub = est[est["estimand_type"] == "PRE_SUB"]
    print("\nDiD-BCF pre-trend diagnostic (Delta(k); zero under conditional PT)")
    print(_fmt(diag))
    print("\nBy lpop tercile (the conditional check)")
    print(_fmt(sub) if not sub.empty else "  (none)")
    print("\nTWFE event-study placebo (standard practice; does not condition on lpop)")
    print(_fmt(tw))

    out = pd.concat([est.assign(source="did_bcf_pretrend"),
                     tw.assign(source="twfe_event_study")], ignore_index=True)
    out.insert(0, "dataset", "mpdta")
    path = os.path.join(args.out, "pretrend_mpdta.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")
    make_figure(diag, tw, sub, os.path.join(args.out, "fig_pretrend_empirical.pdf"))

    if args.with_att:
        att = paper_att(args.data, bcf_params, seed=args.seed)
        print("\nConstrained (estimation) model, overall ATT "
              "(the paper's application specification)")
        print(att.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        att.to_csv(os.path.join(args.out, "att_mpdta.csv"), index=False)


if __name__ == "__main__":
    main()
