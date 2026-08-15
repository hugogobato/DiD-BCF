"""Is stochtree silently handing the prognostic forest a copy of D_it?

``did_bcf_revision.did_bcf.fit_did_bcf`` (spec ``published``) calls
``BCFModel.sample`` without a propensity.  stochtree's ``propensity_covariate``
defaults to ``"prognostic"``, so when none is supplied it fits an *internal*
BART model of ``Z = D_it`` on ``X`` -- and ``X`` contains both ``time`` and
``treatment_group``, of which ``D_it`` is a deterministic function.  The fitted
``pi_hat`` is then appended to the mu forest's split set with non-zero weight
**regardless of** ``keep_vars``.

If that is right, ``pi_hat`` separates the arms perfectly and a single split on
``pi_hat > 0.5`` reconstructs ``D_it`` exactly -- a shorter path into ``mu``
than the two-split group-by-time route of ``identification_note.tex``.  The
original notebooks passed ``pi_train = 0.5`` (constant, no information), so the
channel is absent there; ``diag_leakage.py`` measures the consequence and this
script measures the mechanism.

Run from ``Simulation_Studies_Revision/``:  python Results/_staging/diag_internal_propensity.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from stochtree import BCFModel

from did_bcf_revision import config as cfg
from did_bcf_revision.dgps import generate_canonical_did, true_estimands
from did_bcf_revision.did_bcf import PROGNOSTIC_COLS, TREATMENT_COLS, fit_did_bcf

p = dict(cfg.get_experiment("B1_baseline").dgp_params)
p["linearity_degree"] = 1
BCF = dict(num_gfr=20, num_mcmc=100, keep_every=2, num_chains=1)

rows = []
for N in (200, 800):
    df = generate_canonical_did(seed=0, **{**p, "n_units": N})
    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
    X = df[PROGNOSTIC_COLS].to_numpy(dtype=float)
    Z = df["D"].to_numpy(dtype=float)

    # --- the mechanism: what does the internal propensity model learn? ------ #
    m = BCFModel()
    m.sample(X_train=X, Z_train=Z, y_train=df["Y"].to_numpy(dtype=float),
             num_gfr=BCF["num_gfr"], num_mcmc=BCF["num_mcmc"],
             general_params={"random_seed": 0, "num_chains": 1},
             prognostic_forest_params={"keep_vars": np.arange(len(PROGNOSTIC_COLS))},
             treatment_effect_forest_params={
                 "keep_vars": np.array([PROGNOSTIC_COLS.index(c) for c in TREATMENT_COLS])})
    pi = np.asarray(m.bart_propensity_model.predict(X=X, terms="y_hat", type="mean"))

    # --- the consequence: retention with and without that channel ----------- #
    att = float(true_estimands(df).query("estimand_type == 'ATT'")["true"].iloc[0])
    retention = {}
    for spec in ("published", "published_constant_pi"):
        fit = fit_did_bcf(df, bcf_params=BCF, seed=0, spec=spec)
        post = fit.df["D"].to_numpy() == 1
        retention[spec] = float(fit.tau_draws[post].mean()) / att

    rec = {
        "N": N,
        "internal_propensity_used": bool(m.internal_propensity_model),
        "corr_pi_D": float(np.corrcoef(pi, Z)[0, 1]),
        "mean_pi_treated": float(pi[Z == 1].mean()),
        "mean_pi_control": float(pi[Z == 0].mean()),
        "perfectly_separated": bool(pi[Z == 1].min() > pi[Z == 0].max()),
        "accuracy_of_threshold_rule": float(np.mean((pi > 0.5) == (Z == 1))),
        "retention_published": retention["published"],
        "retention_constant_pi": retention["published_constant_pi"],
    }
    rows.append(rec)
    print(rec, flush=True)

pd.DataFrame(rows).to_csv("Results/_staging/diag_internal_propensity.csv", index=False)
print("wrote Results/_staging/diag_internal_propensity.csv")
