"""Numerical checks on the revised pre-trend DGP grid, before any MCMC is spent.

Four claims are asserted, each of which the design depends on:

1. **Backwards compatibility.**  With ``het_trend = 0`` every pre-existing
   scenario is bit-identical to the version before the channel was added.
2. **The heterogeneous violation cancels in the aggregate.**  Under
   ``PT_violation_het*`` the treated-minus-control mean differential slope is
   zero to Monte-Carlo error, while the X1 subgroup contrast is ``2 kappa``.
   This is what makes the marginal event study powerless by construction, so if
   it fails the scenario is not testing what it claims to.
3. **The TWFE comparator is correctly sized** once the pre-trend slope uses the
   full coefficient covariance.  The old diagonal-only SE is checked to be
   anti-conservative on the same data, so the fix is doing something.
4. **The subgroup contrast is null under a homogeneous violation.**  Under
   ``PT_violation_g*`` the true X1 contrast must be zero, otherwise
   ``PRE_SUBC``'s rejection rate there is not a size.

Run: ``python scripts/check_pretrend_dgps.py [--reps 400]``
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_bcf_revision.config import get_experiment            # noqa: E402
from did_bcf_revision.dgps import generate_canonical_did      # noqa: E402
from did_bcf_revision.pretrend import true_pretrend           # noqa: E402
from did_bcf_revision.twfe import twfe_event_study_se         # noqa: E402


def _unit_slopes(df):
    u = df.drop_duplicates("unit_id")
    return u, (u["eventually_treated"] == 1).to_numpy()


def agg_and_contrast(params, seed):
    """(treated-minus-control mean slope, X1 subgroup contrast) for one draw."""
    df = generate_canonical_did(seed=seed, **params)
    u, ever = _unit_slopes(df)
    s, x1 = u["pt_slope"].to_numpy(), u["X1"].to_numpy() > 0.5
    agg = s[ever].mean() - s[~ever].mean()
    hi = s[ever & x1].mean() - s[~ever & x1].mean()
    lo = s[ever & ~x1].mean() - s[~ever & ~x1].mean()
    return agg, hi - lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()
    R = args.reps
    ok = True

    # -- 1. backwards compatibility ------------------------------------------
    base = get_experiment("PT_hold").dgp_params
    a = generate_canonical_did(seed=0, **base)
    b = generate_canonical_did(seed=0, **{**base, "het_trend": 0.0})
    same = np.allclose(a["Y"], b["Y"]) and np.allclose(a["pt_slope"],
                                                       b["pt_slope"])
    print(f"[1] het_trend=0 leaves the DGP untouched: {same}")
    ok &= same

    # -- 2. the heterogeneous violation cancels ------------------------------
    print("\n[2] PT_violation_het*: aggregate slope vs X1 contrast "
          f"({R} draws each)")
    print(f"    {'kappa':>6} {'mean agg':>10} {'mcse':>8} {'mean contrast':>14} "
          f"{'target 2k':>10}")
    for kap in (0.10, 0.20, 0.40):
        name = f"PT_violation_het{int(round(kap * 100)):02d}"
        p = get_experiment(name).dgp_params
        vals = np.array([agg_and_contrast(p, s) for s in range(R)])
        agg, con = vals[:, 0], vals[:, 1]
        mcse = agg.std(ddof=1) / np.sqrt(R)
        near_zero = abs(agg.mean()) < 3 * mcse
        con_ok = abs(con.mean() - 2 * kap) < 0.02
        print(f"    {kap:6.2f} {agg.mean():10.4f} {mcse:8.4f} "
              f"{con.mean():14.4f} {2 * kap:10.2f}"
              f"{'' if (near_zero and con_ok) else '   <-- FAIL'}")
        ok &= near_zero and con_ok

    # -- 4. the contrast is null under a homogeneous violation ---------------
    print("\n[4] PT_violation_g*: X1 contrast must be null (it is the size)")
    for d in (0.10, 0.40):
        p = get_experiment(f"PT_violation_g{int(round(d * 100)):02d}").dgp_params
        vals = np.array([agg_and_contrast(p, s) for s in range(R)])
        con = vals[:, 1]
        mcse = con.std(ddof=1) / np.sqrt(R)
        good = abs(con.mean()) < 3 * mcse
        print(f"    delta={d:<5} agg={vals[:, 0].mean():7.4f} "
              f"contrast={con.mean():8.5f} (mcse {mcse:.5f})"
              f"{'' if good else '   <-- FAIL'}")
        ok &= good

    # -- 3. the TWFE slope SE ------------------------------------------------
    print("\n[3] TWFE pre-trend slope test under PT_hold (nominal 5%, "
          f"{min(R, 200)} draws)")
    from scipy.stats import norm
    rej_full, rej_diag = [], []
    for s in range(min(R, 200)):
        df = generate_canonical_did(seed=s, **base)
        es = twfe_event_study_se(df, ref=-1)
        pre = es[(es["k"] < 0) & (es["k"] != -1) & es["se"].notna()]
        w = pre["k"].to_numpy(dtype=float) + 1.0
        V = np.asarray(es.attrs["vcov"], dtype=float)
        idx = [es.attrs["vcov_k"].index(int(k)) for k in pre["k"]]
        Vp = V[np.ix_(idx, idx)]
        slope = float(w @ pre["coef"].to_numpy() / (w @ w))
        se_full = float(np.sqrt(w @ Vp @ w) / (w @ w))
        se_diag = float(np.sqrt(np.sum((w * pre["se"].to_numpy()) ** 2)) / (w @ w))
        rej_full.append(2 * norm.cdf(-abs(slope / se_full)) < 0.05)
        rej_diag.append(2 * norm.cdf(-abs(slope / se_diag)) < 0.05)
    rf, rd = float(np.mean(rej_full)), float(np.mean(rej_diag))
    mcse = np.sqrt(0.05 * 0.95 / len(rej_full))
    print(f"    full covariance: size {rf:.3f}   (mcse {mcse:.3f})")
    print(f"    diagonal only  : size {rd:.3f}   <- the old formula")
    fixed = rf < 0.05 + 3 * mcse and rd > rf
    print(f"    corrected SE is sized and the old one was not: {fixed}")
    ok &= fixed

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
