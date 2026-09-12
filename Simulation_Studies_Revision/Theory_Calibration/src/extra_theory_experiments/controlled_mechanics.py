"""Controlled special-case mechanics diagnostic.

This module deliberately bypasses BCF training.  It fixes every prognostic
posterior draw at the exact ``m0_oracle`` from ``oracle_canonical`` and compares
the full-sample Algorithm-1 Bayesian-bootstrap law with the K=2 fold
convolution law.  It is a mechanics/algebra diagnostic, not a theorem proof.
"""
from __future__ import annotations

from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd

from .correction import algorithm2_draws, bayesian_bootstrap_weights
from .manifest import deterministic_seed
from .metrics import scalar_metrics
from .reference import attach_unit_folds, convolve_fold_draws
from .oracle_dgp import generate_oracle_canonical_did


METHODS = ("full_algorithm1_bb", "reference_fold_convolution")
SCENARIOS = (("signal", 3.0), ("null", 0.0))


def _cell_vectors(df, g: float, t: int, units):
    """Return aligned cell arrays for a selected set of original units."""
    index = {(int(u), int(tm)): i for i, (u, tm) in enumerate(
        zip(df["unit_id"].astype(int), df["time"].astype(int)))}
    rows_t, rows_ref, delta, dy, keep = [], [], [], [], []
    for unit in sorted(set(int(x) for x in units)):
        rt = index.get((unit, int(t)))
        rr = index.get((unit, int(g) - 1))
        if rt is None or rr is None:
            continue
        rows_t.append(rt)
        rows_ref.append(rr)
        delta.append(float(df.iloc[rt]["cohort"] == float(g)))
        dy.append(float(df.iloc[rt]["Y"] - df.iloc[rr]["Y"]))
        keep.append(unit)
    if not keep:
        raise ValueError("controlled mechanics cell has no complete units")
    rows_t = np.asarray(rows_t, dtype=int)
    return {
        "rows_t": rows_t,
        "delta": np.asarray(delta, dtype=float),
        "dy": np.asarray(dy, dtype=float),
        "m0": df.iloc[rows_t]["m0_oracle"].to_numpy(float),
        "pi": df.iloc[rows_t]["pi_oracle"].to_numpy(float),
        "barpi": float(df.iloc[rows_t]["barpi_oracle"].iloc[0]),
        "units": np.asarray(keep, dtype=int),
    }


def _exact_algorithm1_draws(cell, bb, n_draws: int):
    """Algorithm 2 arrays with M fixed exactly at the oracle m0."""
    weights = np.vstack([bb[int(unit)] for unit in cell["units"]])
    m0 = np.asarray(cell["m0"], dtype=float)
    M = np.repeat(m0[:, None], int(n_draws), axis=1)
    return algorithm2_draws(
        cell["delta"], cell["dy"], M, cell["pi"], cell["barpi"], weights,
        m_pilot=m0)


def full_algorithm1_bb(df, *, g: float, t: int, n_draws: int, seed: int):
    """Full-sample Algorithm-1 BB law with common original-unit multipliers."""
    all_units = df["unit_id"].drop_duplicates().astype(int).to_numpy()
    cell_units = df.loc[
        (df["cohort"] == float(g)) | np.isinf(df["cohort"]), "unit_id"
    ].drop_duplicates().astype(int).to_numpy()
    cell = _cell_vectors(df, g, t, cell_units)
    bb = bayesian_bootstrap_weights(
        all_units, int(n_draws), seed=deterministic_seed("bb", seed))
    return _exact_algorithm1_draws(cell, bb, int(n_draws))


def reference_fold_mechanics(df, *, g: float, t: int, n_draws: int,
                             fold_seed: int, posterior_seed: int, K: int = 2):
    """K-fold exact-m0 law with global fold BB maps and exact cell weights."""
    work = attach_unit_folds(df, K=int(K), seed=int(fold_seed))
    cell_units = work.loc[
        (work["cohort"] == float(g)) | np.isinf(work["cohort"]), "unit_id"
    ].drop_duplicates().astype(int).to_numpy()
    fold_draws, fold_cell_counts = {}, {}
    for fold in range(int(K)):
        fold_units = work.loc[work["fold"] == fold, "unit_id"] \
            .drop_duplicates().astype(int).to_numpy()
        eval_units = np.intersect1d(cell_units, fold_units)
        cell = _cell_vectors(work, g, t, eval_units)
        if len(cell["units"]) < 4 or cell["delta"].sum() < 1 \
                or (1.0 - cell["delta"]).sum() < 1:
            continue
        bb = bayesian_bootstrap_weights(
            fold_units, int(n_draws),
            seed=deterministic_seed("bb", fold, base_seed=int(posterior_seed)))
        fold_draws[fold] = _exact_algorithm1_draws(cell, bb, int(n_draws))
        # N_{g,k}=|I_k intersection S_g|, the selected two-group cell size.
        fold_cell_counts[fold] = int(len(cell["units"]))
    if len(fold_draws) != int(K):
        missing = sorted(set(range(int(K))) - set(fold_draws))
        raise ValueError(
            f"controlled mechanics cell g={g:g}, t={t} has "
            f"{len(fold_draws)} valid folds; expected exactly K={int(K)} "
            f"(missing folds {missing})")
    return convolve_fold_draws(
        fold_draws, fold_cell_counts, expected_folds=int(K))


def _summary(draws, *, scenario, base_effect, method, estimand_type,
             estimand_id, rep, seed):
    draws = np.asarray(draws, dtype=float)
    q025, q975 = np.quantile(draws, [.025, .975])
    tail_min = float(min(np.mean(draws >= 0), np.mean(draws <= 0)))
    return {
        "scenario": scenario, "base_effect": float(base_effect),
        "method": method, "estimand_type": estimand_type,
        "estimand_id": estimand_id, "rep": int(rep), "seed": int(seed),
        "post_mean": float(np.mean(draws)),
        "post_median": float(np.median(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "q025": float(q025), "q975": float(q975),
        "true": float(base_effect),
        "cover95_rep": bool(q025 <= base_effect <= q975),
        "interval_length95": float(q975 - q025),
        "p_bayes_two_sided": float(min(1.0, 2.0 * tail_min)),
        "null_reject_rep": bool(
            base_effect == 0.0 and not (q025 <= 0.0 <= q975)),
        "posterior_m_fixed_exact_m0": True,
    }


def _aggregate(per_rep: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = ["scenario", "base_effect", "method", "estimand_type",
                  "estimand_id"]
    for keys, grp in per_rep.groupby(group_cols, dropna=False):
        scenario, base_effect, method, estimand_type, estimand_id = keys
        for point_summary, estimate_col in (("mean", "post_mean"),
                                            ("median", "post_median")):
            out = scalar_metrics(grp[estimate_col], grp["true"],
                                 q025=grp["q025"], q975=grp["q975"])
            out.update({"scenario": scenario, "base_effect": base_effect,
                        "method": method, "estimand_type": estimand_type,
                        "estimand_id": estimand_id,
                        "point_summary": point_summary,
                        "mean_interval_length95": float(grp["interval_length95"].mean()),
                        "null_rejection_rate": (
                            float(grp["null_reject_rep"].mean())
                            if float(base_effect) == 0.0 else np.nan),
                        "exact_m0_draws": True})
            records.append(out)
    return pd.DataFrame.from_records(records)


def run_controlled_mechanics(*, out_dir: str | Path, reps: int = 100,
                             n_draws: int = 300, n_units: int = 200,
                             base_seed: int = 20260829, K: int = 2):
    """Run the controlled signal/null mechanics audit and zip its artifacts."""
    if int(reps) < 1 or int(n_draws) < 2 or int(n_units) < 8:
        raise ValueError("reps>=1, n_draws>=2, and n_units>=8 are required")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for scenario, base_effect in SCENARIOS:
        for rep in range(int(reps)):
            seed = deterministic_seed("controlled", scenario, rep,
                                      base_seed=int(base_seed))
            df = generate_oracle_canonical_did(
                seed=seed, n_units=int(n_units), linearity_degree=2,
                base_effect=float(base_effect), effect_type="homogeneous")
            cells = list(df.loc[df["D"] == 1, ["cohort", "time"]]
                          .drop_duplicates().itertuples(index=False, name=None))
            cell_draws = {method: {} for method in METHODS}
            for g, t in cells:
                cell_draws["full_algorithm1_bb"][(float(g), int(t))] = \
                    full_algorithm1_bb(
                        df, g=float(g), t=int(t), n_draws=int(n_draws), seed=seed)
                cell_draws["reference_fold_convolution"][(float(g), int(t))] = \
                    reference_fold_mechanics(
                        df, g=float(g), t=int(t), n_draws=int(n_draws),
                        fold_seed=seed, posterior_seed=seed, K=int(K))
            treated = df[df["D"] == 1]
            cell_weights = {
                (float(g), int(t)): int(len(grp))
                for (g, t), grp in treated.groupby(["cohort", "time"])
            }
            for method in METHODS:
                for (g, t), draws in cell_draws[method].items():
                    rows.append(_summary(
                        draws, scenario=scenario, base_effect=base_effect,
                        method=method, estimand_type="GATT",
                        estimand_id=f"g={g:g}_t={t}", rep=rep, seed=seed))
                keys = list(cell_draws[method])
                weights = np.asarray([cell_weights[k] for k in keys], float)
                weights /= weights.sum()
                att_draws = weights @ np.vstack([cell_draws[method][k] for k in keys])
                rows.append(_summary(
                    att_draws, scenario=scenario, base_effect=base_effect,
                    method=method, estimand_type="ATT", estimand_id="ATT",
                    rep=rep, seed=seed))
            manifest.append({
                "scenario": scenario, "base_effect": base_effect,
                "rep": rep, "seed": seed, "n_units": int(n_units),
                "n_draws": int(n_draws), "K": int(K),
                "posterior_m": "exact m0_oracle repeated for every draw",
                "bb_coupling": "original-unit multipliers shared across cells",
            })
    per_rep = pd.DataFrame.from_records(rows)
    metrics = _aggregate(per_rep)
    manifest_frame = pd.DataFrame.from_records(manifest)
    per_rep_path = out / "controlled_mechanics_per_replication.csv"
    metrics_path = out / "controlled_mechanics_metrics.csv"
    manifest_path = out / "controlled_mechanics_manifest.csv"
    per_rep.to_csv(per_rep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    manifest_frame.to_csv(manifest_path, index=False)
    provenance = {
        "diagnostic": "controlled_special_case_mechanics",
        "theorem_claim": False,
        "posterior_m": "exact m0_oracle, fixed across posterior draws",
        "methods": list(METHODS), "scenarios": dict(SCENARIOS),
        "reps": int(reps), "n_draws": int(n_draws), "n_units": int(n_units),
        "K": int(K), "base_seed": int(base_seed),
        "warnings": [
            "special-case algebra/BB/fold diagnostic; not a theorem proof",
            "finite Monte Carlo reps and finite BB draws are reported",
        ],
    }
    provenance_path = out / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    readme_path = out / "README_run.txt"
    readme_path.write_text(
        "Controlled special-case mechanics diagnostic; not a theorem proof.\n"
        f"reps={int(reps)}, draws={int(n_draws)}, units={int(n_units)}, K={int(K)}\n",
        encoding="utf-8")
    archive_path = out.parent / f"{out.name}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(out))
    return per_rep, metrics, archive_path


__all__ = [
    "METHODS", "SCENARIOS", "full_algorithm1_bb", "reference_fold_mechanics",
    "run_controlled_mechanics",
]
