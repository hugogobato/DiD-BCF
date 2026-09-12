"""Exploratory fold-corrected version of the pre-trend diagnostic.

The production diagnostic (:mod:`did_bcf_revision.pretrend`) fits the
unconstrained specification ``y = mu(X, t) + tau(X, k) W`` on the full panel and
reports ``Delta(k) = E_treated[tau(X_i, k) - tau(X_i, -1)]`` for ``k < -1``.

This module mirrors that object under the fixed-K reference architecture used by
:mod:`extra_theory_experiments.reference`:

1. original units are partitioned once into ``K`` folds, and all rows of a unit
   follow the unit;
2. one unconstrained diagnostic model is fitted per fold on that fold's units
   only, so fold posteriors are independent;
3. for every estimand the fold-level draws are pooled with fixed unit-count
   weights, and the slope uses the same least-squares combination on
   ``k - ref_k`` as the published raw slope rule.

The raw arm (:func:`raw_pretrend_estimands`) reproduces the published
:func:`did_bcf_revision.pretrend.pretrend_estimands` point summaries.  It is a
local schema-consistent copy: the only difference is that summaries carry
``post_median`` (and the ETE tail conventions) in addition to the published
columns.  The fold arm is a diagnostic aggregation only.  It is NOT covered by
the reference-fold convolution theorem: there is no Algorithm-2
efficient-influence-function correction here, the pooling weights are
descriptive unit counts rather than selected cell sizes, and the finite fold
draws are approximations.  It must never be labelled theorem-validated.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .correction import summarise_draws
from .manifest import deterministic_seed
from .reference import attach_unit_folds, check_unit_fold_separation

# These subgroup definitions and the contrast pairing mirror the production
# diagnostic exactly (``did_bcf_revision.pretrend``); they are repeated here
# because this package deliberately keeps a local copy of the production
# algebra it adapts.
_SUBGROUPS = {
    "X1=0": lambda d: d["X1"] <= 0.5,
    "X1=1": lambda d: d["X1"] > 0.5,
    "X2=low": lambda d: d["X2"] <= d["X2"].quantile(1 / 3),
    "X2=high": lambda d: d["X2"] >= d["X2"].quantile(2 / 3),
}
_CONTRAST_SUFFIXES = (("1", "0"), ("high", "low"))


def _derive_contrasts(labels) -> tuple:
    by_col: dict[str, dict[str, str]] = {}
    for lab in labels:
        if "=" not in lab:
            continue
        col, level = lab.split("=", 1)
        by_col.setdefault(col, {})[level] = lab
    out = []
    for col, levels in by_col.items():
        for hi, lo in _CONTRAST_SUFFIXES:
            if hi in levels and lo in levels:
                out.append((col, levels[hi], levels[lo]))
                break
    return tuple(out)


def _pre_rows_and_refs(fit, ref_k: int = -1):
    """Rows ``(unit, k<0)`` and each unit's row at ``cohort + ref_k``.

    This is the local equivalent of ``did_bcf_revision.pretrend._pre_rows``.
    """
    df = fit.df
    ref_k = int(getattr(fit, "ref_k", ref_k))
    pre = df[(df["eventually_treated"] == 1) & (df["event_time"] < 0)]
    ref_rows, keep = [], []
    for idx, unit, cohort in zip(pre.index, pre["unit_id"], pre["cohort"]):
        row = fit.row_of.get((int(unit), int(cohort) + ref_k))
        if row is not None:
            ref_rows.append(row)
            keep.append(idx)
    return pre.loc[keep], np.asarray(ref_rows, dtype=int)


def _subgroup_units(pre: pd.DataFrame, subgroups) -> dict[str, set[int]]:
    groups: Mapping[str, Any] = _SUBGROUPS if subgroups is True else (subgroups or {})
    out: dict[str, set[int]] = {}
    for name, sel in groups.items():
        try:
            mask = sel(pre).to_numpy()
        except Exception:  # predicate not applicable to this panel
            continue
        units = set(pre.loc[mask, "unit_id"].astype(int))
        if len(units) >= 5:
            out[name] = units
    return out


def _pretrend_records(*, pre: pd.DataFrame, ks: Sequence[int], ref_k: int,
                      groups: Mapping[str, set[int]], method: str,
                      delta) -> pd.DataFrame:
    """Build the estimation rows shared by both arms.

    ``delta(unit_set_or_None, k)`` returns one posterior draw vector of
    ``Delta(k)`` over the requested units, or ``None`` when no unit is
    available.  Only point-estimate rows are emitted (``PRE``, ``PRE`` slope,
    ``PRE_SUB``, ``PRE_SUBC``); the production ``any``/``any_bonf``/``joint``
    decision rows are a metrics-layer construct and are recovered from the
    stored posterior intervals instead.
    """
    records: list[dict[str, Any]] = []
    per_k: dict[int, np.ndarray] = {}
    for k in ks:
        draws = delta(None, k)
        if draws is None:
            continue
        per_k[int(k)] = draws
        rec = summarise_draws(draws, method=method, estimand_type="PRE",
                              estimand_id=f"k={int(k)}", k=int(k))
        rec["g"], rec["t"] = np.nan, np.nan
        records.append(rec)

    if per_k:
        cs = sorted(per_k)
        w = np.asarray([k - ref_k for k in cs], dtype=float)
        slope = (w @ np.vstack([per_k[k] for k in cs])) / float(w @ w)
        rec = summarise_draws(slope, method=method, estimand_type="PRE",
                              estimand_id="slope", k=np.nan)
        rec["g"], rec["t"] = np.nan, np.nan
        records.append(rec)

    sub_draws: dict[tuple[str, int], np.ndarray] = {}
    for name, units in groups.items():
        for k in ks:
            draws = delta(units, k)
            if draws is None:
                continue
            sub_draws[(name, int(k))] = draws
            rec = summarise_draws(draws, method=method, estimand_type="PRE_SUB",
                                  estimand_id=f"{name}_k={int(k)}", k=int(k))
            rec["g"], rec["t"] = np.nan, np.nan
            records.append(rec)

    for label, hi, lo in _derive_contrasts(groups):
        per_k_c = {k: sub_draws[(hi, k)] - sub_draws[(lo, k)]
                   for k in ks if (hi, k) in sub_draws and (lo, k) in sub_draws}
        if not per_k_c:
            continue
        cks = sorted(per_k_c)
        for k in cks:
            rec = summarise_draws(per_k_c[k], method=method,
                                  estimand_type="PRE_SUBC",
                                  estimand_id=f"{label}_k={int(k)}", k=int(k))
            rec["g"], rec["t"] = np.nan, np.nan
            records.append(rec)
        w = np.asarray([k - ref_k for k in cks], dtype=float)
        stack = np.vstack([per_k_c[k] for k in cks])
        rec = summarise_draws((w @ stack) / float(w @ w), method=method,
                              estimand_type="PRE_SUBC",
                              estimand_id=f"{label}_slope", k=np.nan)
        rec["g"], rec["t"] = np.nan, np.nan
        records.append(rec)
    return pd.DataFrame.from_records(records)


def raw_pretrend_estimands(fit, subgroups: "bool | dict" = True,
                           method: str = "pretrend") -> pd.DataFrame:
    """Schema-consistent copy of the published raw diagnostic point summaries.

    The point estimates are computed from exactly the same draws and masks as
    :func:`did_bcf_revision.pretrend.pretrend_estimands`; they therefore
    reproduce the published raw diagnostic.  The ``any``/``any_bonf`` decision
    rows are intentionally not duplicated here.
    """
    pre, refs = _pre_rows_and_refs(fit, ref_k=-1)
    if pre.empty:
        return pd.DataFrame()
    ks = sorted({int(k) for k in pre["event_time"].unique() if int(k) != -1})
    groups = _subgroup_units(pre, subgroups)

    def delta(units, k):
        mask = (pre["event_time"] == int(k)).to_numpy()
        if units is not None:
            mask = mask & pre["unit_id"].astype(int).isin(units).to_numpy()
        if not mask.any():
            return None
        rows = pre.index.to_numpy()[mask]
        return np.mean(fit.tau_draws[rows] - fit.tau_draws[refs[mask]], axis=0)

    return _pretrend_records(pre=pre, ks=ks, ref_k=-1, groups=groups,
                             method=method, delta=delta)


def _default_pretrend_fit(panel, *, bcf_params=None, seed=0, **kwargs):
    try:
        from Simulation_Studies_Revision.did_bcf_revision.pretrend import fit_pretrend
    except ImportError:
        from did_bcf_revision.pretrend import fit_pretrend
    return fit_pretrend(panel, bcf_params=bcf_params, seed=int(seed))


def reference_fold_pretrend(
    df: pd.DataFrame,
    *,
    K: int = 2,
    fold_seed: int = 0,
    posterior_seed: int = 0,
    bcf_params: dict | None = None,
    fit_factory=None,
    subgroups: "bool | dict" = True,
    fold_col: str = "fold",
    method: str = "reference_fold_pretrend",
) -> pd.DataFrame:
    """Fold-fit the unconstrained diagnostic and pool the lead draws.

    Only a single adoption cohort is supported (the PT designs are canonical).
    Every fold fit uses that fold's units only; the pooled ``Delta(k)`` is the
    unit-count weighted average of the fold-level mean lead contrasts, which is
    the same linear object the published raw diagnostic reports.
    """
    work = attach_unit_folds(df, K=K, seed=fold_seed, column=fold_col)
    check_unit_fold_separation(work, column=fold_col)
    treated = work.loc[work["eventually_treated"] == 1, ["unit_id", "cohort"]] \
        .drop_duplicates()
    cohorts = sorted({float(x) for x in treated["cohort"]})
    if len(cohorts) != 1:
        raise ValueError(
            "reference_fold_pretrend supports exactly one adoption cohort; "
            f"found {cohorts}")
    g = cohorts[0]
    fit_factory = fit_factory or _default_pretrend_fit
    fold_fits: dict[int, Any] = {}
    fold_pre: dict[int, pd.DataFrame] = {}
    fold_refs: dict[int, np.ndarray] = {}
    for fold in range(int(K)):
        fold_units = set(int(u) for u in work.loc[
            work[fold_col] == int(fold), "unit_id"].unique())
        panel = work[work["unit_id"].astype(int).isin(fold_units)].copy()
        fit = fit_factory(
            panel, bcf_params=bcf_params,
            seed=deterministic_seed("pretrend", g, fold, base_seed=posterior_seed))
        expected = set(int(u) for u in fit.df["unit_id"].unique())
        if expected != fold_units:
            raise AssertionError("diagnostic fold fit includes non-fold units")
        pre, refs = _pre_rows_and_refs(fit, ref_k=-1)
        if pre.empty:
            raise ValueError(f"diagnostic fold {fold} has no usable pre rows")
        fold_fits[fold], fold_pre[fold], fold_refs[fold] = fit, pre, refs

    pooled = pd.concat([fold_pre[f] for f in sorted(fold_pre)], ignore_index=True)
    pooled = pooled.drop_duplicates(subset=["unit_id", "event_time"])
    ks = sorted({int(k) for k in pooled["event_time"].unique() if int(k) != -1})
    groups = _subgroup_units(pooled, subgroups)

    def delta(units, k):
        numerator, denominator = None, 0
        for fold in sorted(fold_fits):
            pre, refs = fold_pre[fold], fold_refs[fold]
            mask = (pre["event_time"] == int(k)).to_numpy()
            if units is not None:
                mask = mask & pre["unit_id"].astype(int).isin(units).to_numpy()
            if not mask.any():
                continue
            rows = pre.index.to_numpy()[mask]
            draws = np.mean(fold_fits[fold].tau_draws[rows] -
                            fold_fits[fold].tau_draws[refs[mask]], axis=0)
            weight = int(mask.sum())
            numerator = draws * weight if numerator is None else numerator + draws * weight
            denominator += weight
        if numerator is None or denominator == 0:
            return None
        return numerator / float(denominator)

    return _pretrend_records(pre=pooled, ks=ks, ref_k=-1, groups=groups,
                             method=method, delta=delta)
