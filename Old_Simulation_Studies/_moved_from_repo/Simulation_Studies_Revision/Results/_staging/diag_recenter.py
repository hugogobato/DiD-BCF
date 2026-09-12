import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from did_bcf_revision import config as cfg
from did_bcf_revision.dgps import generate_canonical_did, true_estimands
from did_bcf_revision.did_bcf import fit_did_bcf, plain_estimands

rows=[]
e=cfg.get_experiment('B1_baseline')
for rep in range(5):
    p=dict(e.dgp_params); p['linearity_degree']=1
    df=generate_canonical_did(seed=rep, **{**p,'n_units':200})
    fit=fit_did_bcf(df, seed=rep)
    tau=fit.tau_draws.mean(1)
    d=fit.df.copy(); d['tauhat']=tau
    tr=d[d.treatment_group==1]
    pre=tr[tr.time==3].tauhat.mean(); post=tr[tr.time>=4].tauhat.mean()
    att=float(true_estimands(df).query("estimand_type=='ATT'").true.iloc[0])
    r={'rep':rep,'true_ATT':att,'tauhat_pre_g-1':pre,'tauhat_post':post}
    for rc in (True,False):
        pe=plain_estimands(fit,pretrend_recenter=rc)
        r[f'plainATT_recenter={rc}']=float(pe.query("estimand_type=='ATT'").post_mean.iloc[0])
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv('Results/_staging/diag_recenter.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
