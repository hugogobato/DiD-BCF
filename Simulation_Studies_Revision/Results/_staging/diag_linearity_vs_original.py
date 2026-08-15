"""Does linearity_degree survive the two-way within transformation?

Compares the ORIGINAL suite's DGP with the REVISION suite's DGP.
"""
import os, sys
import numpy as np, pandas as pd

REV = "/home/hugo_souto/Stuff/Research/DiD-BCF/DiD-BCF/Simulation_Studies_Revision"
ORIG = "/home/hugo_souto/Stuff/Research/DiD-BCF/DiD-BCF/Simulation_Studies/Original/DGPs"


def within(df, ycol="Y", unit="unit_id", time="time"):
    y = df[ycol].values.astype(float)
    u = df.groupby(unit)[ycol].transform("mean").values
    t = df.groupby(time)[ycol].transform("mean").values
    return y - u - t + y.mean()


def report(tag, frames):
    base = frames[1]
    print("\n=== %s ===" % tag)
    print("  raw Y            : range across d = %.4f"
          % max(np.abs(frames[d]["Y"].values - base["Y"].values).max() for d in (2, 3)))
    for d in (2, 3):
        w0, wd = within(base), within(frames[d])
        print("  within-Y  d=1 vs d=%d : max|diff| = %.3e" % (d, np.abs(wd - w0).max()))
    # naive 2x2 DiD on the balanced panel, for reference
    for d in (1, 2, 3):
        f = frames[d]
        g = f.groupby([f["_treat"], f["_post"]])["Y"].mean()
        did = (g[(1, 1)] - g[(1, 0)]) - (g[(0, 1)] - g[(0, 0)])
        print("  2x2 DiD  d=%d = %.6f" % (d, did))


# ---------------------------------------------------------------- revision
sys.path.insert(0, REV)
from did_bcf_revision.config import get_experiment
from did_bcf_revision.dgps import generate_canonical_did

exp = get_experiment("B1_baseline")
rev = {}
for d in (1, 2, 3):
    df = generate_canonical_did(seed=0, **{**exp.dgp_params, "n_units": 200,
                                           "linearity_degree": d})
    df["_treat"] = df["eventually_treated"]
    df["_post"] = (df["time"] >= 4).astype(int)
    rev[d] = df
report("REVISION  B1_baseline (covariates drawn once per unit)", rev)

# ---------------------------------------------------------------- original
sys.path.insert(0, ORIG)
from data_creation_ATE import generate_did_data

orig = {}
for d in (1, 2, 3):
    df = generate_did_data(n_units=200, linearity_degree=d, seed=42)
    df["_treat"] = df["treated_group"]
    df["_post"] = df["post_treatment"]
    orig[d] = df
report("ORIGINAL  data_creation_ATE (covariates drawn per unit-period)", orig)

# is X time-varying?
print("\nX within-unit SD (mean over units):")
print("  revision X2 : %.4f" % rev[1].groupby("unit_id")["X2"].std().mean())
print("  original X_2: %.4f" % orig[1].groupby("unit_id")["X_2"].std().mean())
