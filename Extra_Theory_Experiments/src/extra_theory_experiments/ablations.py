"""Information ablations for full-panel versus reduced-cell fits.

Only raw posterior contrasts are reported here.  No full-panel EIF is invented;
multiplier intervals are an explicitly unsupported exploratory option and are
disabled by default.
"""
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def reduced_two_group_two_period_panel(df, g: float, t: int):
    """Select the target cohort/never-treated units and exactly two periods."""
    ref = int(g) - 1
    out = df[
        ((df["cohort"] == float(g)) | np.isinf(df["cohort"])) &
        (df["time"].isin([ref, int(t)]))
    ].copy().sort_values(["unit_id", "time"]).reset_index(drop=True)
    if out["unit_id"].nunique() < 4 or out["time"].nunique() != 2:
        raise ValueError("reduced cell fit needs at least four units and two periods")
    return out


def _production_fit(df, *, bcf_params=None, seed=0, spec="structured",
                    effect_by_cohort=True):
    try:
        from didbcf_structured import StructuredDiDBCF
        from Simulation_Studies_Revision.did_bcf_revision.did_bcf import (
            DEFAULT_BCF_PARAMS, FitResult, _row_index_map)
    except ImportError:
        from didbcf_structured import StructuredDiDBCF
        from did_bcf_revision.did_bcf import DEFAULT_BCF_PARAMS, FitResult, _row_index_map
    p = {**DEFAULT_BCF_PARAMS, **(bcf_params or {})}
    rfx = {"structured": "none", "structured_rfx": "group",
           "structured_rfx_unit": "unit"}.get(spec, "none")
    model = StructuredDiDBCF(rfx=rfx).sample(
        df, num_gfr=p["num_gfr"], num_mcmc=p["num_mcmc"],
        keep_every=p["keep_every"], num_chains=p["num_chains"], seed=int(seed),
        effect_by_cohort=bool(effect_by_cohort))
    return FitResult(df=model.df, mu_draws=np.asarray(model.mu_draws, float),
                     tau_draws=np.asarray(model.tau_draws, float),
                     row_of=_row_index_map(model.df),
                     design_cols=list(model.design.level_cols), bcf_params=p,
                     spec=spec)


def fit_panel_variant(
    df,
    *,
    variant: str = "full_panel_raw",
    g: float | None = None,
    t: int | None = None,
    fit_factory: Callable[..., Any] | None = None,
    bcf_params: dict | None = None,
    seed: int = 0,
    spec: str = "structured",
    effect_by_cohort: bool = True,
    pooling: bool = False,
    multiplier_interval: bool = False,
):
    """Fit one information variant and return (fit, used_panel, metadata)."""
    if multiplier_interval:
        raise NotImplementedError(
            "exploratory center-preserving multiplier intervals are disabled; "
            "their target/formula are not documented or theorem-covered")
    variant = str(variant)
    if variant == "full_panel_raw":
        used = df.copy()
    elif variant in {"reduced_cell", "two_group_two_period"}:
        if g is None or t is None:
            raise ValueError("reduced_cell requires target g and t")
        used = reduced_two_group_two_period_panel(df, g, t)
    elif variant in {"pooled_full_panel", "full_panel_pooling"}:
        used = df.copy()
        pooling = True
    else:
        raise ValueError("unknown panel variant: full_panel_raw, reduced_cell, pooled_full_panel")
    factory = fit_factory or _production_fit
    fit_effect = False if variant in {"pooled_full_panel", "full_panel_pooling"} \
        else bool(effect_by_cohort)
    fit = factory(used, bcf_params=bcf_params, seed=int(seed), spec=spec,
                  effect_by_cohort=fit_effect)
    metadata = {
        "variant": variant, "panel_rows": int(len(used)),
        "panel_units": int(used["unit_id"].nunique()),
        "pooling": bool(pooling or variant in {"pooled_full_panel", "full_panel_pooling"}),
        "effect_by_cohort": bool(fit_effect),
        "interval_status": "not_requested",
        "scientific_label": "exploratory information ablation",
    }
    return fit, used, metadata


def raw_structured_tau(fit):
    """Extract raw structured posterior treatment-effect draws only."""
    return np.asarray(fit.tau_draws, dtype=float)


def _raw_scalar_records(fit, method):
    """Local mean/median summaries of raw tau draws, without CATT broadcasting."""
    from .correction import summarise_draws
    df = fit.df
    tau = np.asarray(fit.tau_draws, dtype=float)
    if tau.ndim == 1:
        tau = tau[:, None]
    treated = df[df["D"] == 1]
    rows = []
    for (g, t), grp in treated.groupby(["cohort", "time"]):
        d = tau[grp.index.to_numpy(int)].mean(axis=0)
        rows.append(summarise_draws(d, method=method, estimand_type="GATT",
                                    estimand_id=f"g={g:g}_t={int(t)}",
                                    g=float(g), t=int(t), k=int(t-g)))
    for k, grp in treated.groupby("event_time"):
        d = tau[grp.index.to_numpy(int)].mean(axis=0)
        rows.append(summarise_draws(d, method=method, estimand_type="ES",
                                    estimand_id=f"k={int(k)}", k=int(k)))
    if len(treated):
        rows.append(summarise_draws(tau[treated.index.to_numpy(int)].mean(axis=0),
                                    method=method, estimand_type="ATT",
                                    estimand_id="ATT"))
    import pandas as pd
    return pd.DataFrame.from_records(rows)


def ablation_estimands(fit, *, method: str = "full_panel_raw",
                       target_g: float | None = None,
                       target_t: int | None = None):
    """Use local raw tau summaries, retaining only scalar estimands."""
    out = _raw_scalar_records(fit, method)
    if method in {"reduced_cell", "two_group_two_period"}:
        if target_g is None or target_t is None:
            raise ValueError("reduced-cell summaries require target g and t")
        target_id = f"g={float(target_g):g}_t={int(target_t)}"
        out = out[(out["estimand_type"] == "GATT") &
                  (out["estimand_id"] == target_id)].copy()
    out["method"] = method
    return out


def run_information_ablation(
    df,
    *,
    variants=("full_panel_raw", "reduced_cell", "pooled_full_panel"),
    bcf_params=None,
    seed=0,
    fit_factory=None,
    spec="structured",
    effect_by_cohort=True,
    pooling=False,
    multiplier_interval=False,
):
    """Run variants, with reduced fits for every treated cell."""
    records, fits, metadata = [], {}, []
    cells = df.loc[df["D"] == 1, ["cohort", "time"]].drop_duplicates()
    for variant in variants:
        if variant in {"reduced_cell", "two_group_two_period"}:
            for g, t in cells.itertuples(index=False):
                fit, used, meta = fit_panel_variant(
                    df, variant=variant, g=float(g), t=int(t),
                    fit_factory=fit_factory, bcf_params=bcf_params,
                    seed=int(seed), spec=spec,
                    effect_by_cohort=effect_by_cohort, pooling=pooling,
                    multiplier_interval=multiplier_interval)
                fits[(variant, float(g), int(t))] = fit
                tab = ablation_estimands(fit, method=variant,
                                         target_g=float(g), target_t=int(t))
                tab["target_g"], tab["target_t"] = float(g), int(t)
                records.append(tab)
                metadata.append({**meta, "target_g": float(g), "target_t": int(t)})
        else:
            fit, used, meta = fit_panel_variant(
                df, variant=variant, fit_factory=fit_factory,
                bcf_params=bcf_params, seed=int(seed), spec=spec,
                effect_by_cohort=effect_by_cohort, pooling=pooling,
                multiplier_interval=multiplier_interval)
            fits[(variant, None, None)] = fit
            records.append(ablation_estimands(fit, method=variant))
            metadata.append(meta)
    import pandas as pd
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return result, fits, metadata
