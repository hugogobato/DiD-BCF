"""Goodman-Bacon (2021) decomposition of the static TWFE DiD estimator.

For a balanced panel with binary, absorbing, staggered treatment, the TWFE
coefficient on ``D`` is a weighted average of all possible 2x2
difference-in-differences estimators (Goodman-Bacon, 2021, *JoE*).  This module
computes those components and their weights, isolating the **"later vs earlier"
comparisons that use already-treated units as controls** -- the bad comparisons
whose weight, under dynamic / cohort-varying effects, drives TWFE bias.

Comparison types returned
-------------------------
* ``Treated_vs_Untreated``   timing group k vs the never-treated group.
* ``Earlier_vs_Later``       earlier group k (switches on) vs not-yet-treated l.
* ``Later_vs_Earlier``       later group l (switches on) vs **already-treated** k
                             -- the contaminated comparisons.

Weights follow Goodman-Bacon (2021), eq. (9)-(12); they sum to 1 and the
weighted sum of the 2x2 estimates reconstructs the TWFE coefficient (checked
against :func:`did_bcf_revision.twfe.twfe_att` in :func:`bacon_summary`).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .twfe import twfe_att

__all__ = ["goodman_bacon_decomposition", "bacon_summary"]


def _cell_mean(piv: pd.DataFrame, units, periods) -> float:
    """Mean outcome over the given units x periods block of a unit-by-time pivot."""
    block = piv.loc[units, periods]
    return float(np.nanmean(block.to_numpy()))


def goodman_bacon_decomposition(df: pd.DataFrame, outcome="Y", unit="unit_id",
                                time="time", cohort="cohort") -> pd.DataFrame:
    """Return the 2x2 components, their weights and DD estimates.

    Assumes a balanced panel (every unit observed in every period) with an
    absorbing binary treatment encoded by ``cohort`` (first treated period, or
    ``inf`` for never-treated).
    """
    d = df[[unit, time, cohort, outcome]].copy()
    piv = d.pivot(index=unit, columns=time, values=outcome).sort_index()
    periods = list(piv.columns)
    T = len(periods)

    # Map each unit to its cohort; never-treated -> inf.
    unit_cohort = d.groupby(unit)[cohort].first()
    timing_groups = sorted(c for c in unit_cohort.unique() if np.isfinite(c))
    never_units = unit_cohort.index[~np.isfinite(unit_cohort.values)].tolist()

    n_total = len(unit_cohort)
    n = {g: int((unit_cohort == g).sum()) for g in timing_groups}
    n_U = len(never_units)
    units_of = {g: unit_cohort.index[unit_cohort == g].tolist() for g in timing_groups}

    # Share of units and share of *time* each timing group spends treated.
    nshare = {g: n[g] / n_total for g in timing_groups}
    nshare_U = n_U / n_total
    Dbar = {g: sum(1 for t in periods if t >= g) / T for g in timing_groups}

    records = []

    # --- (1) timing group k vs never-treated ------------------------------ #
    for g in timing_groups:
        if n_U == 0:
            continue
        pre = [t for t in periods if t < g]
        post = [t for t in periods if t >= g]
        if not pre or not post:
            continue
        dd = ((_cell_mean(piv, units_of[g], post) - _cell_mean(piv, units_of[g], pre))
              - (_cell_mean(piv, never_units, post) - _cell_mean(piv, never_units, pre)))
        n_kU = nshare[g] / (nshare[g] + nshare_U)
        weight = ((nshare[g] + nshare_U) ** 2
                  * n_kU * (1 - n_kU) * Dbar[g] * (1 - Dbar[g]))
        records.append({"type": "Treated_vs_Untreated", "treat_group": g,
                        "control_group": np.inf, "weight": weight, "dd": dd})

    # --- (2)/(3) pairs of timing groups ----------------------------------- #
    for i, k in enumerate(timing_groups):
        for l in timing_groups[i + 1:]:           # t*_k < t*_l
            pre_k = [t for t in periods if t < k]
            mid = [t for t in periods if k <= t < l]
            post_l = [t for t in periods if t >= l]
            n_kl = nshare[k] / (nshare[k] + nshare[l])

            # (2) earlier k treated, later l as not-yet-treated control
            if pre_k and mid:
                dd_k = ((_cell_mean(piv, units_of[k], mid) - _cell_mean(piv, units_of[k], pre_k))
                        - (_cell_mean(piv, units_of[l], mid) - _cell_mean(piv, units_of[l], pre_k)))
                w_k = (((nshare[k] + nshare[l]) * (1 - Dbar[l])) ** 2
                       * n_kl * (1 - n_kl)
                       * ((Dbar[k] - Dbar[l]) / (1 - Dbar[l]))
                       * ((1 - Dbar[k]) / (1 - Dbar[l])))
                records.append({"type": "Earlier_vs_Later", "treat_group": k,
                                "control_group": l, "weight": w_k, "dd": dd_k})

            # (3) later l treated, earlier k as ALREADY-TREATED control
            if mid and post_l:
                dd_l = ((_cell_mean(piv, units_of[l], post_l) - _cell_mean(piv, units_of[l], mid))
                        - (_cell_mean(piv, units_of[k], post_l) - _cell_mean(piv, units_of[k], mid)))
                w_l = (((nshare[k] + nshare[l]) * Dbar[k]) ** 2
                       * n_kl * (1 - n_kl)
                       * (Dbar[l] / Dbar[k])
                       * ((Dbar[k] - Dbar[l]) / Dbar[k]))
                records.append({"type": "Later_vs_Earlier", "treat_group": l,
                                "control_group": k, "weight": w_l, "dd": dd_l})

    comp = pd.DataFrame.from_records(records)
    if comp.empty:
        return comp
    comp["weight"] = comp["weight"] / comp["weight"].sum()      # normalise to 1
    comp["contribution"] = comp["weight"] * comp["dd"]
    return comp.sort_values(["type", "treat_group", "control_group"]).reset_index(drop=True)


def bacon_summary(df: pd.DataFrame, outcome="Y", unit="unit_id",
                  time="time", cohort="cohort", check_tol=1e-6) -> dict:
    """Headline numbers plus a self-consistency check against the TWFE coef.

    Returns the TWFE estimate, the Bacon-reconstructed estimate, the total
    weight on each comparison type, and -- the quantity of interest for
    Workstream D -- the **weight on already-treated ("Later_vs_Earlier")
    comparisons**.
    """
    comp = goodman_bacon_decomposition(df, outcome, unit, time, cohort)
    twfe = twfe_att(df, outcome=outcome, unit=unit, time=time)
    if comp.empty:
        return {"twfe": twfe, "bacon_total": np.nan, "components": comp,
                "weight_already_treated": np.nan}
    bacon_total = float(comp["contribution"].sum())
    by_type = comp.groupby("type")["weight"].sum().to_dict()
    if np.isfinite(twfe) and abs(twfe - bacon_total) > max(check_tol, 1e-4 * abs(twfe)):
        warnings.warn(
            f"Bacon reconstruction {bacon_total:.6f} != TWFE {twfe:.6f}; "
            "check that the panel is balanced and treatment is absorbing.")
    return {
        "twfe": twfe,
        "bacon_total": bacon_total,
        "weight_treated_vs_untreated": by_type.get("Treated_vs_Untreated", 0.0),
        "weight_earlier_vs_later": by_type.get("Earlier_vs_Later", 0.0),
        "weight_already_treated": by_type.get("Later_vs_Earlier", 0.0),
        "components": comp,
    }
