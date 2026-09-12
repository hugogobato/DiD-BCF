"""Fixed-K, unit-separated reference correction.

Original units are assigned to one fold, every panel row follows its unit, posterior
BCF fits use only the evaluation fold's two groups, nuisance pilots use only
complementary units, and independent fold draws are convolved with selected
two-group cell-size weights N_{g,k}.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping
import numpy as np
import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .correction import (
    CorrectionOutput, OracleUnavailableError, algorithm2_draws,
    bayesian_bootstrap_weights, fit_propensity, summarise_draws,
)
from .manifest import deterministic_seed


def assign_unit_folds(unit_ids: np.ndarray, K: int = 2, seed: int = 0) -> dict[int, int]:
    """Deterministically map each original unit to exactly one of K folds."""
    if int(K) < 2:
        raise ValueError("the fixed-K reference requires K >= 2")
    units = np.unique(np.asarray(unit_ids, dtype=int))
    keys = np.asarray([
        deterministic_seed("fold", int(u), base_seed=int(seed)) for u in units
    ])
    order = np.argsort(keys, kind="mergesort")
    return {int(units[j]): int(rank % int(K)) for rank, j in enumerate(order)}


def attach_unit_folds(df, K: int = 2, seed: int = 0, column: str = "fold"):
    """Return a copy with a fold column, checking all rows of a unit agree."""
    out = df.copy()
    folds = assign_unit_folds(out["unit_id"].to_numpy(), K=K, seed=seed)
    out[column] = out["unit_id"].astype(int).map(folds).astype(int)
    check_unit_fold_separation(out, column=column)
    return out


def check_unit_fold_separation(df, column: str = "fold") -> None:
    bad = df.groupby("unit_id", sort=False)[column].nunique()
    if np.any(bad.to_numpy() != 1):
        raise AssertionError("all panel rows of a unit must remain in one fold")


def convolve_fold_draws(
    fold_draws: Mapping[int, np.ndarray] | list[np.ndarray],
    fold_cell_counts: Mapping[int, int] | list[int] | None = None,
    *,
    expected_folds: int | None = None,
    fold_treated_counts: Mapping[int, int] | list[int] | None = None,
) -> np.ndarray:
    """Exact selected two-group cell-size weighted convolution of fold draws.

    ``fold_cell_counts`` are N_{g,k}=|I_k intersect S_g|, the selected
    two-group cell sizes.  ``fold_treated_counts`` remains a keyword alias for
    older callers, but its name is misleading and is deprecated.  When
    ``expected_folds`` is supplied, every fold 0,...,K-1 must be present;
    partial convolutions are rejected rather than silently renormalized.
    """
    if fold_cell_counts is None:
        fold_cell_counts = fold_treated_counts
    elif fold_treated_counts is not None:
        raise TypeError("pass fold_cell_counts, not both count argument names")
    if fold_cell_counts is None:
        raise ValueError("selected fold cell counts are required")
    if isinstance(fold_draws, Mapping):
        keys = list(fold_draws)
        draws = [np.asarray(fold_draws[k], dtype=float).reshape(-1) for k in keys]
        if expected_folds is not None:
            expected = set(range(int(expected_folds)))
            if set(keys) != expected:
                raise ValueError(
                    f"expected exactly {int(expected_folds)} valid folds "
                    f"{sorted(expected)}, got {sorted(keys)}")
        try:
            counts = [float(fold_cell_counts[k]) for k in keys]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("fold draws and selected cell counts are misaligned") from exc
    else:
        draws = [np.asarray(x, dtype=float).reshape(-1) for x in fold_draws]
        if expected_folds is not None and len(draws) != int(expected_folds):
            raise ValueError(
                f"expected exactly {int(expected_folds)} valid folds, "
                f"got {len(draws)}")
        try:
            counts = [float(x) for x in fold_cell_counts]
        except TypeError as exc:
            raise ValueError("fold draws and selected cell counts are misaligned") from exc
    if not draws or len(draws) != len(counts):
        raise ValueError("fold draws/selected cell counts must be non-empty and aligned")
    if any(len(x) != len(draws[0]) for x in draws):
        raise ValueError("fold posterior vectors must have equal length")
    if any(c < 0 for c in counts) or not np.sum(counts) > 0:
        raise ValueError("selected fold cell counts must be nonnegative and nonzero")
    if any(not np.all(np.isfinite(x)) for x in draws):
        raise ValueError("fold draws must be finite")
    w = np.asarray(counts, dtype=float)
    w /= w.sum()
    return np.tensordot(w, np.vstack(draws), axes=(0, 0))


def _cell_rows(df, g: float, t: int, fold: int, fold_col: str):
    sample = df[((df["cohort"] == g) | np.isinf(df["cohort"])) &
                (df[fold_col] == int(fold))]
    units = sample["unit_id"].drop_duplicates().astype(int).to_numpy()
    rows_t, rows_ref, delta, dy, keep = [], [], [], [], []
    index = {(int(u), int(tm)): i for i, (u, tm) in enumerate(
        zip(df["unit_id"].astype(int), df["time"].astype(int)))}
    for u in units:
        rt, rr = index.get((int(u), int(t))), index.get((int(u), int(g) - 1))
        if rt is None or rr is None:
            continue
        rows_t.append(rt)
        rows_ref.append(rr)
        delta.append(float(df.iloc[rt]["cohort"] == g))
        dy.append(float(df.iloc[rt]["Y"] - df.iloc[rr]["Y"]))
        keep.append(int(u))
    return (np.asarray(rows_t, int), np.asarray(rows_ref, int),
            np.asarray(delta), np.asarray(dy), np.asarray(keep, int))


def _fit_offfold_propensity(X_eval, delta_eval, X_train, delta_train, *,
                            method, seed, clip, oracle_pi=None):
    method = str(method).lower()
    if method == "oracle":
        p = fit_propensity(X_eval, delta_eval, method="oracle", clip=clip,
                           oracle_pi=oracle_pi)
        p.fold_sizes = [{"fold": 0, "train": int(len(X_train)),
                         "evaluation": int(len(X_eval))}]
        return p
    if method == "intercept":
        if len(delta_train) == 0:
            raise ValueError("off-fold intercept pilot has no complementary units")
        pred = np.full(len(delta_eval), float(np.mean(delta_train)))
        p = fit_propensity(X_eval, delta_eval, method="oracle", clip=clip,
                           oracle_pi=pred)
        p.method = "intercept"
        p.fold_sizes = [{"fold": 0, "train": int(len(X_train)),
                         "evaluation": int(len(X_eval))}]
        return p
    if len(delta_train) < 2 or len(np.unique(delta_train)) < 2:
        raise ValueError("off-fold propensity pilot needs both cell groups in training data")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    if method == "rf":
        mdl = RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                     random_state=int(seed), n_jobs=1)
    elif method == "logit":
        mdl = LogisticRegression(max_iter=1000, random_state=int(seed))
    else:
        raise ValueError("propensity method must be logit, rf, intercept, or oracle")
    mdl.fit(np.asarray(X_train, float), np.asarray(delta_train, int))
    pred = mdl.predict_proba(np.asarray(X_eval, float))[:, 1]
    p = fit_propensity(X_eval, delta_eval, method="oracle", clip=clip,
                       oracle_pi=pred)
    p.method = method
    p.fold_sizes = [{"fold": 0, "train": int(len(X_train)),
                     "evaluation": int(len(X_eval))}]
    return p


def _outcome_pilot(df, units, g: float, t: int, eval_units, *, method="rf"):
    """Fit m0 on complementary cell units and predict evaluation units."""
    # m0 is a control-arm nuisance.  Never-treated complement units only are
    # admissible training observations; target-cohort outcomes are treated at t.
    control_units = set(int(x) for x in units
                        if np.isinf(df.loc[df["unit_id"] == int(x), "cohort"].iloc[0]))
    train_units = control_units - set(int(x) for x in eval_units)
    if not train_units:
        raise ValueError("outcome pilot has no complementary units")
    index = {(int(u), int(tm)): i for i, (u, tm) in enumerate(
        zip(df["unit_id"].astype(int), df["time"].astype(int)))}
    cols = [c for c in ("X1", "X2", "X3", "X4", "X5") if c in df]
    train_X, train_y = [], []
    for u in sorted(train_units):
        rt, rr = index.get((u, int(t))), index.get((u, int(g) - 1))
        if rt is None or rr is None:
            continue
        train_X.append(df.iloc[rt][cols].to_numpy(float))
        train_y.append(float(df.iloc[rt]["Y"] - df.iloc[rr]["Y"]))
    if len(train_y) < 2:
        raise ValueError("outcome pilot has too few complementary observations")
    eval_X = np.vstack([
        df.iloc[index[(int(u), int(t))]][cols].to_numpy(float) for u in eval_units
    ])
    method = str(method).lower()
    if method == "intercept":
        return np.full(len(eval_units), float(np.mean(train_y)))
    if method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        mdl = RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                                    random_state=0, n_jobs=1)
    elif method == "ridge":
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=1.0)
    else:
        raise ValueError("outcome method must be ridge, rf, or intercept")
    mdl.fit(np.asarray(train_X), np.asarray(train_y))
    return np.asarray(mdl.predict(eval_X), float)


def oracle_nuisance(df, g: float, t: int, rows_t: np.ndarray, delta: np.ndarray):
    """Obtain exact nuisance columns, or fail loudly.

    The production DGPs do not expose exact assignment probabilities because
    treatment includes logistic-normal assignment noise.  A custom DGP adapter
    may provide m0_oracle and pi_oracle (and optionally barpi_oracle).
    """
    if "m0_oracle" not in df.columns or "pi_oracle" not in df.columns:
        raise OracleUnavailableError(
            "exact oracle nuisance unavailable: expected m0_oracle and pi_oracle columns")
    m0 = np.asarray(df.iloc[rows_t]["m0_oracle"], float)
    pi = np.asarray(df.iloc[rows_t]["pi_oracle"], float)
    if len(m0) != len(delta) or len(pi) != len(delta):
        raise OracleUnavailableError("oracle nuisance columns are not cell-aligned")
    if "barpi_oracle" not in df.columns and "barpi_oracle" not in df.attrs:
        raise OracleUnavailableError(
            "exact oracle bar-pi unavailable: expected barpi_oracle column or attribute")
    bar = (float(np.asarray(df.iloc[rows_t]["barpi_oracle"], float).mean())
           if "barpi_oracle" in df.columns else float(df.attrs["barpi_oracle"]))
    if np.any(~np.isfinite(m0)) or np.any(~np.isfinite(pi)) or not 0 < bar <= 1:
        raise OracleUnavailableError("oracle nuisance values must be finite and valid")
    return m0, pi, bar


def _default_fit(eval_df, *, bcf_params=None, seed=0, spec="structured"):
    try:
        from Simulation_Studies_Revision.did_bcf_revision.structured import fit_structured
    except ImportError:
        from did_bcf_revision.structured import fit_structured
    return fit_structured(eval_df, bcf_params=bcf_params, seed=int(seed), spec=spec)


def reference_fold_convolution(
    df,
    *,
    K=2,
    fold_seed=0,
    posterior_seed=0,
    propensity_method="logit",
    outcome_method="rf",
    clip=1e-3,
    bcf_params=None,
    fit_factory: Callable[..., Any] | None = None,
    fit_cache: dict | None = None,
    oracle=False,
    fold_col="fold",
) -> CorrectionOutput:
    """Run the theorem-reference architecture with exact fold convolution.

    The posterior median is the theorem-aligned point summary; the posterior
    mean is retained as a diagnostic.  Fixed sampler draws are finite-sample
    approximations and this function never labels them theorem-validated.
    """
    if oracle and propensity_method != "oracle":
        raise ValueError("oracle=True requires propensity_method='oracle'")
    work = attach_unit_folds(df, K=K, seed=fold_seed, column=fold_col)
    check_unit_fold_separation(work, column=fold_col)
    fit_factory = fit_factory or _default_fit
    fit_cache = fit_cache if fit_cache is not None else {}
    cells = work.loc[work["D"] == 1, ["cohort", "time"]].drop_duplicates()
    all_draws: dict[tuple[float, int], np.ndarray] = {}
    # Cache one posterior per (cohort, fold), then reuse its draw index across
    # every calendar time.  BB multipliers are global to a fold and restricted
    # to each cell below, preserving cross-cell coupling for shared controls.
    # External cache permits propensity/clipping variants to reuse the exact
    # same posterior fits within one replication bundle.
    global_bb: dict[int, dict[int, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "method": "reference_fold_convolution", "K": int(K),
        "posterior_point_summary": "median (mean retained as diagnostic)",
        "fixed_draw_warning": True, "cells": {},
    }
    for g, t in cells.itertuples(index=False):
        g, t = float(g), int(t)
        fold_draws: dict[int, np.ndarray] = {}
        fold_cell_counts: dict[int, int] = {}
        cell_diag: dict[str, Any] = {}
        cell_units = work.loc[
            (work["cohort"] == g) | np.isinf(work["cohort"]), "unit_id"
        ].drop_duplicates().astype(int).to_numpy()
        for fold in range(int(K)):
            rows_t, rows_ref, delta, dy, eval_units = _cell_rows(
                work, g, t, fold, fold_col)
            if len(eval_units) < 4 or delta.sum() < 1 or (1 - delta).sum() < 1:
                continue
            eval_ids = set(int(x) for x in eval_units)
            complement_ids = set(int(x) for x in cell_units) - eval_ids
            if eval_ids & complement_ids:
                raise AssertionError("evaluation and complementary unit sets overlap")
            eval_panel = work[work["unit_id"].astype(int).isin(eval_ids)].copy()
            comp_panel = work[work["unit_id"].astype(int).isin(complement_ids)].copy()
            fold_seed_i = deterministic_seed("posterior", g, fold,
                                              base_seed=posterior_seed)
            cache_key = (g, int(fold),
                         json.dumps(bcf_params or {}, sort_keys=True, default=str))
            if cache_key not in fit_cache:
                fit_cache[cache_key] = fit_factory(
                    eval_panel, bcf_params=bcf_params, seed=fold_seed_i)
            fit = fit_cache[cache_key]
            expected = set(int(x) for x in eval_units)
            if set(int(x) for x in fit.df["unit_id"].unique()) != expected:
                raise AssertionError("posterior fit includes non-evaluation units")
            rt_fit = np.asarray([fit.row_of[(int(u), int(t))]
                                 for u in eval_units], int)
            rr_fit = np.asarray([fit.row_of[(int(u), int(g) - 1)]
                                 for u in eval_units], int)
            cols = [c for c in ("X1", "X2", "X3", "X4", "X5") if c in work]
            X_eval = work.loc[rows_t, cols].to_numpy(float)
            train_rows = comp_panel[
                (comp_panel["cohort"] == g) | np.isinf(comp_panel["cohort"])]
            train_u = train_rows.drop_duplicates("unit_id").sort_values("unit_id")
            X_train = train_u[cols].to_numpy(float)
            delta_train = (train_u["cohort"].to_numpy(float) == g).astype(int)
            if oracle:
                m0_eval, oracle_vec, barpi = oracle_nuisance(
                    work, g, t, rows_t, delta)
                M0 = np.asarray(fit.mu_draws[rt_fit] - fit.mu_draws[rr_fit], float)
                p = _fit_offfold_propensity(
                    X_eval, delta, X_train, delta_train, method="oracle",
                    seed=fold_seed_i, clip=clip, oracle_pi=oracle_vec)
            else:
                m_pilot = _outcome_pilot(
                    work, cell_units, g, t, eval_units, method=outcome_method)
                M0 = np.asarray(fit.mu_draws[rt_fit] - fit.mu_draws[rr_fit], float)
                # Keep the posterior draws unchanged in step (b).  The
                # complementary outcome pilot enters only step (c).
                M = M0
                p = _fit_offfold_propensity(
                    X_eval, delta, X_train, delta_train, method=propensity_method,
                    seed=fold_seed_i, clip=clip)
                barpi = float(delta_train.mean())
            if fold not in global_bb:
                fold_units = work.loc[work[fold_col] == int(fold), "unit_id"] \
                    .drop_duplicates().astype(int).to_numpy()
                global_bb[fold] = bayesian_bootstrap_weights(
                    fold_units, int(fit.n_draws),
                    seed=deterministic_seed("bb", fold, base_seed=posterior_seed))
            elif len(next(iter(global_bb[fold].values()))) != int(fit.n_draws):
                raise ValueError("all cached fold fits must have the same draw count")
            bb = {int(u): global_bb[fold][int(u)] for u in eval_units}
            W = np.vstack([bb[int(u)] for u in eval_units])
            draws = algorithm2_draws(delta, dy, M0, p.pi, barpi, W,
                                     m_pilot=(m0_eval if oracle else m_pilot))
            fold_draws[fold] = draws
            # N_{g,k}=|I_k intersection S_g|, the selected two-group cell
            # size, not the treated count.  This is the exact convolution
            # weight required by the cell target.
            fold_cell_counts[fold] = int(len(eval_units))
            cell_diag[str(fold)] = {
                **p.diagnostics, "fold_sizes": p.fold_sizes,
                "n_eval_units": int(len(eval_units)),
                "n_complement_units": int(len(complement_ids)),
                "posterior_draws": int(len(draws)),
                "posterior_cache_key": f"g={g:g}_fold={fold}",
                "global_fold_bb": True,
            }
        if len(fold_draws) != int(K):
            missing = sorted(set(range(int(K))) - set(fold_draws))
            raise ValueError(
                f"reference_fold_convolution cell g={g:g}, t={t} has "
                f"{len(fold_draws)} valid folds; expected exactly K={int(K)} "
                f"(missing folds {missing})")
        draws = convolve_fold_draws(
            fold_draws, fold_cell_counts, expected_folds=int(K))
        all_draws[(g, t)] = draws
        diagnostics["cells"][f"g={g:g}_t={t}"] = cell_diag
        records.append(summarise_draws(
            draws, method="reference_fold_convolution", estimand_type="GATT",
            estimand_id=f"g={g:g}_t={t}", g=g, t=t, k=t - g))
    def weighted(keys):
        keys = [x for x in keys if x in all_draws]
        if not keys:
            return None
        treated = work[work["D"] == 1]
        w = np.asarray([
            np.sum((treated["cohort"] == g) & (treated["time"] == t))
            for g, t in keys
        ], float)
        w /= w.sum()
        return w @ np.vstack([all_draws[k] for k in keys])
    for k in sorted({int(t - g) for g, t in all_draws}):
        d = weighted([(g, t) for g, t in all_draws if int(t - g) == k])
        if d is not None:
            records.append(summarise_draws(
                d, method="reference_fold_convolution",
                estimand_type="ES", estimand_id=f"k={k}", k=k))
    d = weighted(list(all_draws))
    if d is not None:
        records.append(summarise_draws(
            d, method="reference_fold_convolution",
            estimand_type="ATT", estimand_id="ATT"))
    return CorrectionOutput("reference_fold_convolution", all_draws, records,
                            diagnostics)
