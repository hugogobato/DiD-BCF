"""Local posterior-correction adapter.

This is intentionally a copy/adaptation of the production cell algebra.  The
production implementation remains untouched.  The same-sample construction is
called ``current_hybrid`` in every output because cross-fitted pilots do not
make the posterior fit itself independent of the evaluation sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


class OracleUnavailableError(ValueError):
    """Raised when an exact nuisance is not exposed by the DGP/adapter."""


@dataclass
class PropensityResult:
    pi: np.ndarray
    method: str
    clip: float
    fold_sizes: list[dict[str, int]] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __array__(self, dtype=None):
        return np.asarray(self.pi, dtype=dtype)


@dataclass
class CorrectionOutput:
    method: str
    cell_draws: dict[tuple[float, int], np.ndarray]
    records: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_frame(self):
        import pandas as pd
        import json
        rows = []
        for rec in self.records:
            out = dict(rec)
            key = out.get("estimand_id", "")
            diag = self.diagnostics.get("cells", {}).get(key, {})
            # Reference diagnostics have one dictionary per fold; preserve the
            # full JSON and expose conservative scalar extrema for filtering.
            nested = list(diag.values()) if diag and all(
                isinstance(v, dict) for v in diag.values()) else [diag]
            odds = [v.get("max_control_odds") for v in nested
                    if isinstance(v, dict) and v.get("max_control_odds") is not None]
            ess = [v.get("effective_sample_size", v.get("control_ess")) for v in nested
                   if isinstance(v, dict) and v.get("effective_sample_size", v.get("control_ess")) is not None]
            mxw = [v.get("max_normalized_weight") for v in nested
                   if isinstance(v, dict) and v.get("max_normalized_weight") is not None]
            out["max_control_odds"] = float(np.nanmax(odds)) if odds else np.nan
            out["effective_sample_size"] = float(np.nanmin(ess)) if ess else np.nan
            out["max_normalized_weight"] = float(np.nanmax(mxw)) if mxw else np.nan
            out["fold_sizes"] = json.dumps(diag, sort_keys=True, default=str) if diag else "{}"
            rows.append(out)
        return pd.DataFrame.from_records(rows)


def bayesian_bootstrap_weights(unit_ids: np.ndarray, n_draws: int,
                               seed: int = 0) -> dict[int, np.ndarray]:
    """Independent Exp(1) unit weights, reused across cells in one fit."""
    rng = np.random.default_rng(int(seed))
    units = np.unique(np.asarray(unit_ids, dtype=int))
    draws = rng.exponential(1.0, size=(len(units), int(n_draws)))
    return {int(unit): draws[j] for j, unit in enumerate(units)}


def _new_propensity_model(method: str, seed: int):
    if method == "logit":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=int(seed))
    if method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, random_state=int(seed), n_jobs=1
        )
    raise ValueError("propensity method must be one of 'logit', 'rf', 'intercept', 'oracle'")


def _fold_splits(delta: np.ndarray, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedKFold
    if n_splits <= 1 or delta.sum() < n_splits or (len(delta) - delta.sum()) < n_splits:
        return [(np.arange(len(delta)), np.arange(len(delta)))]
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    return list(skf.split(np.zeros((len(delta), 1)), delta.astype(int)))


def fit_propensity(
    X: np.ndarray,
    delta: np.ndarray,
    *,
    method: str = "logit",
    n_splits: int = 2,
    seed: int = 0,
    clip: float = 1e-3,
    oracle_pi: np.ndarray | None = None,
) -> PropensityResult:
    """Fit a cell propensity with explicit clipping and fold diagnostics.

    ``intercept`` estimates the cell share, ``logit`` and ``rf`` use
    cross-fitting, and ``oracle`` requires a caller-supplied exact vector.  A
    missing oracle vector raises rather than silently falling back to a pilot.
    The reported ESS and odds are computed after clipping, so raw extremes are
    also retained in ``raw_max_control_odds``.
    """
    X = np.asarray(X, dtype=float)
    delta = np.asarray(delta, dtype=int).reshape(-1)
    if X.ndim != 2 or len(X) != len(delta) or len(delta) == 0:
        raise ValueError("X and delta must have the same non-zero row count")
    if not 0 < float(clip) < 0.5:
        raise ValueError("clip must lie in (0, .5)")
    method = str(method).lower()
    fold_sizes: list[dict[str, int]] = []
    if method == "intercept":
        pi_raw = np.full(len(delta), float(delta.mean()))
        fold_sizes = [{"fold": 0, "train": len(delta), "evaluation": len(delta)}]
    elif method == "oracle":
        if oracle_pi is None:
            raise OracleUnavailableError(
                "oracle propensity requested without an exact oracle_pi vector")
        pi_raw = np.asarray(oracle_pi, dtype=float).reshape(-1)
        if len(pi_raw) != len(delta) or not np.all(np.isfinite(pi_raw)):
            raise OracleUnavailableError("oracle_pi must be finite and cell-aligned")
        fold_sizes = [{"fold": 0, "train": 0, "evaluation": len(delta)}]
    else:
        if delta.min() == delta.max():
            pi_raw = np.full(len(delta), float(delta.mean()))
            fold_sizes = [{"fold": 0, "train": len(delta), "evaluation": len(delta)}]
        else:
            pi_raw = np.empty(len(delta), dtype=float)
            for fold, (train, test) in enumerate(_fold_splits(delta, int(n_splits), int(seed))):
                # The one-fold fallback is deliberately in-sample, matching
                # the production adapter when a cell is too small for K folds.
                mdl = _new_propensity_model(method, int(seed) + fold)
                mdl.fit(X[train], delta[train])
                pi_raw[test] = mdl.predict_proba(X[test])[:, 1]
                fold_sizes.append({"fold": int(fold), "train": int(len(train)),
                                   "evaluation": int(len(test))})
    if not np.all(np.isfinite(pi_raw)):
        raise ValueError("propensity pilot returned a non-finite probability")
    raw_odds = pi_raw[delta == 0] / np.maximum(1.0 - pi_raw[delta == 0], 1e-15)
    pi = np.clip(pi_raw, float(clip), 1.0 - float(clip))
    odds = pi[delta == 0] / (1.0 - pi[delta == 0])
    if len(odds):
        # The inverse-odds multiplier in Algorithm 2 is pi/(1-pi).  Report
        # its normalized-control ESS, not an ESS for 1/(1-pi).
        control_w = odds.copy()
        control_w /= control_w.sum()
        ess = float(1.0 / np.sum(control_w ** 2))
    else:
        ess = float("nan")
    return PropensityResult(
        pi=pi, method=method, clip=float(clip), fold_sizes=fold_sizes,
        diagnostics={
            "bar_pi": float(delta.mean()),
            "raw_max_control_odds": float(np.max(raw_odds)) if len(raw_odds) else np.nan,
            "max_control_odds": float(np.max(odds)) if len(odds) else np.nan,
            "control_ess": ess,
            "n_clipped": int(np.sum((pi_raw < clip) | (pi_raw > 1.0 - clip))),
        },
    )


def algorithm2_draws(
    delta: np.ndarray,
    delta_y: np.ndarray,
    m_draws: np.ndarray,
    pi: np.ndarray,
    bar_pi: float,
    weights: np.ndarray,
    m_pilot: np.ndarray | None = None,
) -> np.ndarray:
    """Exact sign/scaling of Algorithm 2, including the empirical bias term.

    Inputs have ``n`` rows and ``S`` posterior/BB draws.  ``weights`` may be
    unnormalised Exp(1) draws; normalization is done within this cell.  The
    returned vector is ``theta^s - b_hat^s``.
    """
    delta = np.asarray(delta, dtype=float).reshape(-1)
    delta_y = np.asarray(delta_y, dtype=float).reshape(-1)
    M = np.asarray(m_draws, dtype=float)
    pi = np.asarray(pi, dtype=float).reshape(-1)
    W = np.asarray(weights, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    if W.ndim == 1:
        W = W[:, None]
    n, S = M.shape
    if any(len(v) != n for v in (delta, delta_y, pi)) or W.shape != (n, S):
        raise ValueError("Algorithm-2 arrays must be cell-aligned with shape (n,S)")
    if not 0 < float(bar_pi) <= 1 or np.any((pi <= 0) | (pi >= 1)):
        raise ValueError("bar_pi must be in (0,1] and pi strictly inside (0,1)")
    W = W / W.sum(axis=0, keepdims=True)
    aug_w = delta - (1.0 - delta) * pi / (1.0 - pi)
    gamma = (delta - pi) / ((1.0 - pi) * float(bar_pi))
    residual = delta_y[:, None] - M
    numerator = np.einsum("i,is,is->s", aug_w, W, residual)
    denominator = np.einsum("i,is->s", delta, W)
    if np.any(denominator <= 0):
        raise ValueError("a Bayesian-bootstrap draw has no treated mass")
    theta = numerator / denominator
    # The same-sample construction uses the posterior mean.  The fixed-fold
    # reference supplies an independently fitted complementary-unit pilot.
    pilot = M.mean(axis=1) if m_pilot is None else np.asarray(m_pilot, float).reshape(-1)
    if len(pilot) != n or not np.all(np.isfinite(pilot)):
        raise ValueError("m_pilot must be a finite vector aligned to cell rows")
    b_hat = (gamma @ (pilot[:, None] - M)) / n
    return theta - b_hat


def summarise_draws(draws: np.ndarray, *, method: str, estimand_type: str,
                    estimand_id: str, g=np.nan, t=np.nan, k=np.nan) -> dict[str, Any]:
    """One consistent two-sided Bayesian p-value convention."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 1 or len(draws) < 1 or not np.all(np.isfinite(draws)):
        raise ValueError("draws must be a finite non-empty vector")
    tail_min = float(min(np.mean(draws >= 0), np.mean(draws <= 0)))
    return {
        "estimand_type": estimand_type, "estimand_id": estimand_id,
        "g": g, "t": t, "k": k, "method": method,
        "post_mean": float(np.mean(draws)), "post_median": float(np.median(draws)),
        "sd": float(np.std(draws, ddof=1)) if len(draws) > 1 else np.nan,
        "q025": float(np.quantile(draws, .025)), "q05": float(np.quantile(draws, .05)),
        "q95": float(np.quantile(draws, .95)), "q975": float(np.quantile(draws, .975)),
        # Store both forms so the two-sided convention cannot be confused with
        # the one-tail minimum used by the legacy production tables.
        "p_bayes_tail_min": tail_min,
        "p_bayes_two_sided": float(min(1.0, 2.0 * tail_min)),
        "p_bayes": float(min(1.0, 2.0 * tail_min)),
    }


def _cell_arrays(fit, g: float, t: int):
    df = fit.df
    units = df.loc[(df["cohort"] == g) | np.isinf(df["cohort"]), "unit_id"].unique()
    rows_t, rows_ref, delta, dy, keep = [], [], [], [], []
    for u in units:
        rt = fit.row_of.get((int(u), int(t)))
        rr = fit.row_of.get((int(u), int(g) - 1))
        if rt is None or rr is None:
            continue
        rows_t.append(rt); rows_ref.append(rr)
        delta.append(float(df.at[rt, "cohort"] == g))
        dy.append(float(df.at[rt, "Y"] - df.at[rr, "Y"]))
        keep.append(int(u))
    return (np.asarray(rows_t, int), np.asarray(rows_ref, int), np.asarray(delta),
            np.asarray(dy), keep)


def _cell_diagnostics(pi_result: PropensityResult, W: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    Wn = W / W.sum(axis=0, keepdims=True)
    control = delta == 0
    if np.any(control):
        odds = pi_result.pi[control] / (1.0 - pi_result.pi[control])
        c = Wn[control] * odds[:, None]
        c = c / c.sum(axis=0, keepdims=True)
        ess = 1.0 / np.sum(c ** 2, axis=0)
        max_norm = np.max(c, axis=0)
    else:
        ess = np.full(W.shape[1], np.nan); max_norm = np.full(W.shape[1], np.nan)
    return {**pi_result.diagnostics,
            "effective_sample_size": float(np.nanmin(ess)),
            "max_normalized_weight": float(np.nanmax(max_norm)),
            "combined_weight_definition": "normalize(W * pi/(1-pi)) over controls",
            "fold_sizes": pi_result.fold_sizes}


def current_hybrid_correction(
    fit,
    *,
    propensity_method: str = "logit",
    clip: float = 1e-3,
    n_splits: int = 2,
    seed: int = 0,
    oracle_pi_by_cell: Mapping[tuple[float, int], np.ndarray] | None = None,
    oracle_m0_by_cell: Mapping[tuple[float, int], np.ndarray] | None = None,
    oracle_barpi_by_cell: Mapping[tuple[float, int], float] | None = None,
) -> CorrectionOutput:
    """Apply the current same-sample correction and retain cell diagnostics."""
    df = fit.df
    S = int(fit.n_draws)
    bb = bayesian_bootstrap_weights(df["unit_id"].to_numpy(), S, int(seed))
    treated = df[df["D"] == 1]
    cell_draws: dict[tuple[float, int], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"method": "current_hybrid", "cells": {}}
    for g, t in treated[["cohort", "time"]].drop_duplicates().itertuples(index=False):
        g, t = float(g), int(t)
        rows_t, rows_ref, delta, dy, units = _cell_arrays(fit, g, t)
        if len(units) < 5 or delta.sum() < 2 or (1 - delta).sum() < 2:
            continue
        X = df.loc[rows_t, [c for c in ("X1", "X2", "X3", "X4", "X5") if c in df]].to_numpy(float)
        oracle = None if oracle_pi_by_cell is None else oracle_pi_by_cell.get((g, t))
        p = fit_propensity(X, delta, method=propensity_method, n_splits=n_splits,
                           seed=int(seed) + t, clip=clip, oracle_pi=oracle)
        M = np.asarray(fit.mu_draws[rows_t] - fit.mu_draws[rows_ref], dtype=float)
        W = np.vstack([bb[u] for u in units])
        m_pilot = None if oracle_m0_by_cell is None else oracle_m0_by_cell.get((g, t))
        barpi = (float(delta.mean()) if oracle_barpi_by_cell is None
                 else float(oracle_barpi_by_cell[(g, t)]))
        draws = algorithm2_draws(delta, dy, M, p.pi, barpi, W,
                                 m_pilot=m_pilot)
        cell_draws[(g, t)] = draws
        diagnostics["cells"][f"g={g:g}_t={t}"] = _cell_diagnostics(p, W, delta)
        n_treated = int(np.sum(delta))
        records.append(summarise_draws(draws, method="current_hybrid",
                                       estimand_type="GATT", estimand_id=f"g={g:g}_t={t}",
                                       g=g, t=t, k=t - g))
    def weighted(keys):
        keys = [x for x in keys if x in cell_draws]
        if not keys: return None
        w = np.asarray([np.sum((treated["cohort"] == g) & (treated["time"] == t))
                        for g, t in keys], float)
        w /= w.sum()
        return w @ np.vstack([cell_draws[k] for k in keys])
    for k in sorted({int(t - g) for g, t in cell_draws}):
        d = weighted([(g, t) for g, t in cell_draws if int(t - g) == k])
        if d is not None:
            records.append(summarise_draws(d, method="current_hybrid", estimand_type="ES",
                                           estimand_id=f"k={k}", k=k))
    d = weighted(list(cell_draws))
    if d is not None:
        records.append(summarise_draws(d, method="current_hybrid", estimand_type="ATT",
                                       estimand_id="ATT"))
    return CorrectionOutput("current_hybrid", cell_draws, records, diagnostics)


# Explicit aliases used by notebooks and tests.
correct_current_hybrid = current_hybrid_correction
