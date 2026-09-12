"""Is the mu/tau identification failure present in the ORIGINAL paper's setup?

Reproduces the published DiD-BCF specification exactly as it appears in
``Simulation_Studies/Original/DiD_BCF/DiD_BCF_ATE_lin_1.ipynb``:

    X            = [treated_group, X_1..X_7, time]
    prognostic   keep_vars = every column   (so treated_group AND time are in mu)
    treatment    keep_vars = [time]
    pi_train     = 0.5 (constant)
    Z            = D_it = treated_group * post_treatment

and runs it on the original DGP at several panel sizes.  If mu can span
tau(X)Z, the share of the true effect retained by tau_hat should fall as N
grows.  The published study only ever ran N=200.
"""
import os
import sys

import numpy as np

ORIG = ("/home/hugo_souto/Stuff/Research/DiD-BCF/DiD-BCF/"
        "Simulation_Studies/Original/DGPs")
sys.path.insert(0, ORIG)
os.chdir("/tmp")                      # the module writes ./data on import
from data_creation_ATE import generate_did_data  # noqa: E402

from stochtree import BCFModel       # noqa: E402

NUM_X = 5
N_PRE, N_POST = 4, 4
TRUE_ATE = 3.0                        # linearity_degree in {1, 2} -> beta = 3


def one_fit(n_units, seed):
    d = generate_did_data(n_units=n_units, linearity_degree=1,
                          num_pre_periods=N_PRE, num_post_periods=N_POST,
                          pre_trend_bias_delta=0, num_x_covariates=NUM_X,
                          epsilon_scale=1, seed=seed)
    d["D"] = d["post_treatment"] * d["treated_group"]
    xcols = ["treated_group"] + [f"X_{i}" for i in range(1, NUM_X + 3)] + ["time"]
    X = d[xcols].to_numpy(dtype=float)
    Z = d["D"].to_numpy(dtype=float)
    y = d["Y"].to_numpy(dtype=float)
    t_idx = len(xcols) - 1                      # index of "time"

    m = BCFModel()
    m.sample(X_train=X, Z_train=Z, y_train=y,
             propensity_train=np.full(len(d), 0.5),
             num_gfr=50, num_mcmc=500,
             general_params={"keep_every": 5, "num_chains": 3},
             prognostic_forest_params={"keep_vars": np.arange(len(xcols))},
             treatment_effect_forest_params={"keep_vars": np.array([t_idx])})

    tau = np.asarray(m.tau_hat_train, dtype=float)
    if tau.ndim == 1:
        tau = tau[:, None]
    post = (d["D"] == 1).to_numpy()
    pre_treated = ((d["treated_group"] == 1) & (d["post_treatment"] == 0)).to_numpy()
    return dict(N=n_units, rep=seed, true_ATE=TRUE_ATE,
                tauhat_post=float(tau[post].mean()),
                tauhat_pre_treated=float(tau[pre_treated].mean()),
                share_recovered=float(tau[post].mean() / TRUE_ATE))


DEST = ("/home/hugo_souto/Stuff/Research/DiD-BCF/DiD-BCF/"
        "Simulation_Studies_Revision/Results/_staging/"
        "leakage_original_paper.csv")


def _done(dest):
    """(N, rep) pairs already on disk, so an interrupted run can resume."""
    if not os.path.exists(dest):
        return set()
    import pandas as pd
    d = pd.read_csv(dest)
    return {(int(a), int(b)) for a, b in zip(d["N"], d["rep"])}


if __name__ == "__main__":
    import gc

    import pandas as pd

    done = _done(DEST)
    # 3 replications at the published panel size, 2 at the larger ones: the
    # question is the trend in N, and the large fits are the expensive ones.
    plan = [(200, r) for r in range(3)] + [(800, r) for r in range(2)] \
        + [(1600, r) for r in range(2)]
    for n, rep in plan:
        if (n, rep) in done:
            print("skip (already done):", n, rep, flush=True)
            continue
        r = one_fit(n, rep)
        print(r, flush=True)
        # append immediately so a kill mid-run never loses completed fits
        pd.DataFrame([r]).to_csv(DEST, mode="a", index=False,
                                 header=not os.path.exists(DEST))
        gc.collect()

    df = pd.read_csv(DEST).drop_duplicates(subset=["N", "rep"], keep="last")
    print(df.groupby("N")["share_recovered"].agg(["mean", "count"]).round(3)
          .to_string(), flush=True)
