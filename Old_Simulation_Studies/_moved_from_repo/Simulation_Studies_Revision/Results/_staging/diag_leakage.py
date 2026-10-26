"""Does the BCF prognostic forest absorb the treatment effect as N grows?

D_it is a deterministic function of (treatment_group, time), both of which are in
the prognostic design matrix, so mu(X) can represent tau(X)*Z exactly: the two
forests are not separately identified and only the priors break the tie.  If the
prognostic forest wins increasingly as N grows, tau_hat shrinks below the truth.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from did_bcf_revision import config as cfg
from did_bcf_revision.dgps import generate_canonical_did, true_estimands
from did_bcf_revision.did_bcf import fit_did_bcf

e = cfg.get_experiment('B1_baseline')
rows = []
for N in (200, 800, 1600):
    for rep in (0, 1):
        p = dict(e.dgp_params); p['linearity_degree'] = 1
        df = generate_canonical_did(seed=rep, **{**p, 'n_units': N})
        fit = fit_did_bcf(df, seed=rep)
        d = fit.df.copy(); d['tauhat'] = fit.tau_draws.mean(1)
        post = d[(d.D == 1)]
        att = float(true_estimands(df).query("estimand_type=='ATT'").true.iloc[0])
        r = {'N': N, 'rep': rep, 'true_ATT': att,
             'tauhat_post': float(post.tauhat.mean()),
             'tauhat_pre_treated': float(d[(d.treatment_group == 1) & (d.time == 3)].tauhat.mean()),
             'share_recovered': float(post.tauhat.mean()) / att}
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv('Results/_staging/diag_leakage.csv', index=False)
