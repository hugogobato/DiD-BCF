#!/usr/bin/env python3
"""Generate ``Empirical_Study/DiD_BCF_GATE_CATE_Empirical_Study_structured.ipynb``.

The application redone under the estimator the simulation grid uses, keeping the
original notebook's structure (overall effect, ``lpop`` terciles, KDE figure) so
the two read side by side.  Generated rather than hand-written so it stays in
step with ``scripts/run_empirical_structured.py``, which computes the same
numbers headlessly.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(os.path.dirname(ROOT), "Empirical_Study",
                    "DiD_BCF_GATE_CATE_Empirical_Study_structured.ipynb")

CELLS = [
    ("markdown", """# DiD-BCF on `mpdta` -- corrected estimator

The application of `DiD_BCF_GATE_CATE_Empirical_Study.ipynb`, refitted under the
**structured** DiD-BCF that the whole simulation grid uses.

**Why refit.** The original notebook fits an unrestricted prognostic forest over
`(treat, lpop, year, first.treat)`. Since `D_it = 1{treat} * 1{year >= first.treat}`
is a function of columns already in that split set, the pair
`(mu + c*tau*D, (1-c)*tau)` has the same likelihood for every `c` in `[0, 1]`, so
only the prior separates `mu` from `tau` (`Results/identification_note.tex`).
On simulated data with a large true effect the prior pins it down well enough to
look stable. On `mpdta` it does not: varying only `random_seed`, the published
statistic came out **+0.105 / -0.075 / -0.045** across three seeds -- it changes
sign. The structured model removes the flat direction by construction, and the
same three seeds then agree to four decimal places.

**What is different from the original figure.** The published KDE plots
`exp(mean_over_draws(tau_i)) - 1` across treated counties: one number per county,
each the posterior *mean* of that county's effect. Its spread is cross-county
heterogeneity in the fitted surface, not uncertainty about the estimate, and it
is about an order of magnitude narrower than the posterior. Both are drawn
below, side by side, each labelled for what it is."""),

    ("code", """# Colab: the sampler, plus this repo for the structured model and the data.
%pip install -q stochtree seaborn"""),

    ("code", '''import os, sys, subprocess

REPO_URL = "https://github.com/hugogobato/DiD-BCF.git"

def _locate_repo():
    """Return the DiD-BCF checkout root, cloning it on Colab if needed."""
    here = os.path.abspath(os.getcwd())
    for cand in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if os.path.isdir(os.path.join(cand, "didbcf_structured")):
            return cand
    if not os.path.isdir("DiD-BCF"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    return os.path.abspath("DiD-BCF")

REPO = _locate_repo()
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "Simulation_Studies_Revision"))
sys.path.insert(0, os.path.join(REPO, "Simulation_Studies_Revision", "scripts"))
print("repo:", REPO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from didbcf_structured import StructuredDiDBCF
from run_pretrend_empirical import load_mpdta
from run_empirical_structured import (BCF, COVARIATES, EFFECT_MODIFIERS,
                                      GROUPS, GROUP_COLOR, RFX,
                                      analyse, fit, make_figure)'''),

    ("code", '''DATA = os.path.join(REPO, "Empirical_Study", "mpdta.csv")
df = load_mpdta(DATA).sort_values(["unit_id", "time"]).reset_index(drop=True)
print(f"{len(df)} rows | {df.unit_id.nunique()} counties | "
      f"cohorts {sorted(set(df.cohort))} | {int(df.D.sum())} treated observations")
df.head()'''),

    ("markdown", """## Fit

`structured_rfx_unit` is the headline specification: the two-way restriction
`mu(g,t,x) = a(g,x) + b(t,x)` identifies the effect surface, and the unit-level
random intercepts absorb county-level unobserved heterogeneity. `structured`
without the intercepts is fitted alongside, so their contribution is visible
rather than asserted. Sampler settings are the original's: 50 GFR, 500 MCMC,
`keep_every=5`, 3 chains."""),

    ("code", '''SEED = 0
models, gates, catts, cons = {}, [], [], []
for spec in ("structured", "structured_rfx_unit"):
    print("fitting", spec, "...", flush=True)
    models[spec] = fit(df, spec, SEED)
    g, c, k = analyse(models[spec], spec)
    gates.append(g); catts.append(c); cons.append(k)

gate = pd.concat(gates, ignore_index=True)
catt = pd.concat(catts, ignore_index=True)
con = pd.concat(cons, ignore_index=True)
gate[["spec", "group", "n_obs", "att_pct", "lo95_pct", "hi95_pct", "p_bayes"]].round(3)'''),

    ("markdown", """## Group average treatment effects by county population

Terciles of `lpop` over the whole panel, then restricted to treated
observations, exactly as the original notebook constructs them. The difference
is that each group effect now carries a posterior interval: the group mean of a
fitted surface is a point on a curve, and without an interval there is no way to
tell a real difference between terciles from sampling noise.

Percentages are computed by transforming the draws, `mean(exp(tau) - 1)`, not by
exponentiating the posterior mean -- those are not the same number."""),

    ("code", '''HEAD = "structured_rfx_unit"
g = gate[gate.spec == HEAD].set_index("group")
print(f"--- {HEAD} ---")
for name in ["All treated"] + list(GROUPS.values()):
    if name in g.index:
        r = g.loc[name]
        star = " *" if r.p_bayes < 0.05 else ""
        print(f"  {name:20s} n={int(r.n_obs):4d}  {r.att_pct:+6.2f}%  "
              f"95% CI [{r.lo95_pct:+6.2f}%, {r.hi95_pct:+6.2f}%]  "
              f"p_bayes={r.p_bayes:.4f}{star}")'''),

    ("markdown", """### Do the terciles actually differ?

Overlapping credible intervals are not a test of whether two groups differ, and
reading a difference off three group means with no intervals -- which is what the
original notebook's legend invites -- is weaker still. The posterior of the
*difference* between two groups comes from the same draws, so report that."""),

    ("code", '''print(f"--- {HEAD}: between-group contrasts (log points) ---")
k = con[con.spec == HEAD]
for _, r in k.iterrows():
    verdict = "differ" if r.p_bayes < 0.05 else "not distinguishable"
    print(f"  {r.contrast:38s} {r.diff_log:+.4f}  "
          f"95% CI [{r.lo95_log:+.4f}, {r.hi95_log:+.4f}]  "
          f"p_bayes={r.p_bayes:.3f}  -> {verdict}")'''),

    ("markdown", """## The figure

Left is the original notebook's plot: the distribution across counties of each
county's own posterior-mean effect. It answers "how much does the fitted effect
vary between counties?"

Right is the estimate's uncertainty: 95% credible intervals for the overall ATT
and for each tercile. It answers "how well do we know the effect?" These are
different questions and the second one is the one a reader takes the headline
number from."""),

    ("code", '''OUT = os.path.join(os.getcwd(), "fig_mpdta_gate_kde")
make_figure(catt, gate, HEAD, OUT)
from IPython.display import Image, display
display(Image(OUT + ".png"))'''),

    ("markdown", """## Seed stability

The check the original specification fails. Refit at three seeds and compare the
spread of the point estimate with its own posterior SD -- a comparison that does
not require assuming an effective sample size."""),

    ("code", '''rows = []
for spec in ("structured", "structured_rfx_unit"):
    for seed in (0, 1, 2):
        m = models[spec] if seed == SEED else fit(df, spec, seed)
        tau = np.asarray(m.tau_draws, dtype=float)
        draws = tau[m.df["D"].to_numpy() == 1, :].mean(axis=0)
        rows.append(dict(spec=spec, seed=seed, att=float(draws.mean()),
                         post_sd=float(draws.std(ddof=1))))
seeds = pd.DataFrame(rows)

print("seed range as a share of the posterior SD "
      "(published specification, for contrast: 58%)\\n")
for spec, s in seeds.groupby("spec", sort=False):
    rng = s.att.max() - s.att.min()
    print(f"  {spec:20s} {[f'{a:+.5f}' for a in s.att]}  "
          f"range {rng:.5f} = {100 * rng / s.post_sd.mean():.1f}% of a posterior SD")
seeds.round(5)'''),

    ("code", '''# Save the tables next to the notebook, and download them on Colab.
for name, obj in (("mpdta_structured_gate.csv", gate),
                  ("mpdta_structured_catt.csv", catt),
                  ("mpdta_structured_contrasts.csv", con),
                  ("mpdta_structured_seeds.csv", seeds)):
    obj.to_csv(name, index=False)
    print("wrote", name)

for output_file in ("mpdta_structured_gate.csv", "mpdta_structured_catt.csv",
                    "mpdta_structured_seeds.csv", "fig_mpdta_gate_kde.pdf"):
    try:
        from google.colab import files
        files.download(output_file)
        print("Downloaded:", output_file)
    except Exception as e:
        print("(Not on Colab / download skipped):", e)'''),

    ("markdown", """## Summary

| specification | ATT | seed range | share of a posterior SD | 95% CI |
|---|---|---|---|---|
| published (original notebook) | +0.105 / -0.075 / -0.045 | 0.180 | 58% | [-0.83, +0.84] |
| `structured` | -0.0462 | 0.00068 | 2.4% | [-0.101, +0.010] |
| `structured_rfx_unit` | -0.0480 | 0.00013 | 0.9% | [-0.075, -0.020] |

The corrected application gives roughly **-4.7% on employment, 95% CI
[-7.2%, -2.0%]**, against the published **-14.33%**. The direction survives; the
magnitude does not, and -14.33% falls well outside the corrected interval. Any
statement in the paper that leans on the size of the effect needs rewriting.

Note also that the interval clears zero only once the unit random intercepts are
in: `structured` alone gives [-0.101, +0.010]. Absorbing county-level
heterogeneity roughly halves the posterior SD, from 0.028 to 0.014."""),
]


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s,
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", DEST)


if __name__ == "__main__":
    main()
