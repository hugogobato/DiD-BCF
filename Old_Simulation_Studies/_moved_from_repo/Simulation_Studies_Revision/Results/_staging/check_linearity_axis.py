"""Acceptance test for the repaired linearity-degree axis.

Design: the DGP is fixed in latent covariates; ``d`` changes what the estimator
observes (d=2 hands it Kang--Schafer transforms) and, at d=3, the shape of
tau(X).  The checks are

  (a) the observed covariates must differ across d -- before this fix nothing
      an estimator could see changed at all, the two-way demeaned outcome
      being identical across d to 6e-15;
  (b) the outcome and the true estimands must be *identical* at d=1 and d=2,
      so any difference in estimator performance is misspecification and
      nothing else;
  (c) tau(X) must be identical at d=1,2 and differ at d=3, with Var(tau)
      preserved so the CATT-surface metrics stay comparable;
  (d) linear adjustment in the observed covariates must remove the
      conditional-parallel-trends violation at d=1 and fail to at d>=2.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from did_bcf_revision.config import get_experiment          # noqa: E402
from did_bcf_revision.dgps import generate_canonical_did    # noqa: E402

XCOLS = ["X1", "X2", "X3", "X4", "X5"]


def within(df):
    y = df["Y"].values.astype(float)
    return (y - df.groupby("unit_id")["Y"].transform("mean").values
            - df.groupby("time")["Y"].transform("mean").values + y.mean())


def pta_left_after_linear_adjustment(f):
    """DiD on the *untreated* potential outcome after linear covariate control.

    Y(0) is reconstructed from the frame as Y - CATT, so anything non-zero here
    is a violation of conditional parallel trends that linear adjustment in the
    observed covariates could not remove.
    """
    f = f.copy()
    f["Y0"] = f["Y"] - f["CATT"]
    pre = f[f["time"] == 3].set_index("unit_id")
    post = f[f["time"] >= 4].groupby("unit_id").agg(
        {"Y0": "mean", "eventually_treated": "first", **{c: "first" for c in XCOLS}})
    dY0 = (post["Y0"] - pre["Y0"]).values
    tr = post["eventually_treated"].values.astype(float)
    X = post[XCOLS].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(tr)), X, tr])
    return float(np.linalg.lstsq(A, dY0, rcond=None)[0][-1])


exp = get_experiment("B1_baseline")
# selection="both" so treatment depends on the covariates and the trend
# violation is real; under pure selection-on-unobservables X is balanced and no
# adjustment (right or wrong) is needed.
params = {**exp.dgp_params, "n_units": 40000, "selection": "both"}
frames = {d: generate_canonical_did(seed=0, **{**params, "linearity_degree": d})
          for d in (1, 2, 3)}

rows = []
for d, f in frames.items():
    u = f.drop_duplicates("unit_id")
    u1 = frames[1].drop_duplicates("unit_id")
    rows.append(dict(
        d=d,
        obs_X_diff_vs_d1=float(np.abs(u[XCOLS].to_numpy()
                                      - u1[XCOLS].to_numpy()).max()),
        within_Y_diff_vs_d1=float(np.abs(within(f) - within(frames[1])).max()),
        true_att=float(f.loc[f["D"] == 1, "CATT"].mean()),
        sd_tau=float(u["tau_true"].std()),
        tau_diff_vs_d1=float(np.abs(u["tau_true"].values - u1["tau_true"].values).max()),
        pta_left_linear=pta_left_after_linear_adjustment(f),
    ))

out = pd.DataFrame(rows).set_index("d")
pd.set_option("display.width", 160)
print(out.round(4).to_string())

checks = {
    "observed X changes at d=2": out.loc[2, "obs_X_diff_vs_d1"] > 1e-3,
    "outcome unchanged at d=2": out.loc[2, "within_Y_diff_vs_d1"] < 1e-10,
    "outcome changes at d=3": out.loc[3, "within_Y_diff_vs_d1"] > 1e-3,
    "true ATT identical at d=1,2": abs(out.loc[1, "true_att"] - out.loc[2, "true_att"]) < 1e-10,
    "tau identical at d=1,2": out.loc[2, "tau_diff_vs_d1"] < 1e-12,
    "tau differs at d=3": out.loc[3, "tau_diff_vs_d1"] > 1e-3,
    "Var(tau) preserved": abs(out["sd_tau"].max() - out["sd_tau"].min()) < 0.05,
    "linear adjustment works at d=1": abs(out.loc[1, "pta_left_linear"]) < 0.01,
    "linear adjustment fails at d=2": abs(out.loc[2, "pta_left_linear"]) > 0.03,
}
for k, v in checks.items():
    print(f"  [{'ok' if v else 'FAIL'}] {k}")
print("\nACCEPTANCE:", "PASS" if all(checks.values()) else "FAIL")
