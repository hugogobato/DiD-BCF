"""Task runner, smoke adapter, checkpoint/resume, and provenance."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .correction import current_hybrid_correction
from .manifest import ExperimentTask, deterministic_seed, manifest_frame
from .reference import reference_fold_convolution
from .ablations import run_information_ablation


def _dgp(task: ExperimentTask):
    if task.design == "oracle_canonical":
        from .oracle_dgp import generate_oracle_canonical_did
        return generate_oracle_canonical_did(
            seed=task.seed, n_units=task.N, linearity_degree=task.degree)
    try:
        from Simulation_Studies_Revision.did_bcf_revision.dgps import (
            generate_canonical_did, generate_staggered_did)
    except ImportError:
        from did_bcf_revision.dgps import generate_canonical_did, generate_staggered_did
    if task.design == "staggered":
        return generate_staggered_did(seed=task.seed, n_units=task.N,
                                      linearity_degree=task.degree)
    kwargs = {"n_units": task.N, "linearity_degree": task.degree}
    if task.design == "serial":
        kwargs["ar1_rho"] = .6
    elif task.design == "null":
        kwargs.update(base_effect=0.0, effect_type="homogeneous")
    return generate_canonical_did(seed=task.seed, **kwargs)


class _SmokeFit:
    """Tiny deterministic fit with production-compatible attributes."""
    def __init__(self, df, seed=0, draws=24):
        self.df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
        self.row_of = {(int(u), int(t)): i for i, (u, t) in enumerate(
            zip(self.df.unit_id, self.df.time))}
        rng = np.random.default_rng(int(seed))
        n = len(self.df)
        S = int(draws)
        truth = self.df["CATT"].to_numpy(float) if "CATT" in self.df else np.zeros(n)
        self.tau_draws = truth[:, None] + rng.normal(0, .05, size=(n, S))
        self.mu_draws = (0.2 * self.df["time"].to_numpy(float))[:, None] + rng.normal(
            0, .03, size=(n, S))
        self.n_draws = S
        self.spec = "smoke"
        self.bcf_params = {"smoke": True}


def _production_fit(df, *, bcf_params=None, seed=0, spec="structured",
                    effect_by_cohort=True):
    try:
        from didbcf_structured import StructuredDiDBCF
        from Simulation_Studies_Revision.did_bcf_revision.did_bcf import (
            DEFAULT_BCF_PARAMS, FitResult, _row_index_map)
    except ImportError:
        from didbcf_structured import StructuredDiDBCF
        from did_bcf_revision.did_bcf import DEFAULT_BCF_PARAMS, FitResult, _row_index_map
    rfx = {"structured": "none", "structured_rfx": "group",
           "structured_rfx_unit": "unit"}.get(spec, "none")
    p = {**DEFAULT_BCF_PARAMS, **(bcf_params or {})}
    model = StructuredDiDBCF(rfx=rfx).sample(
        df, num_gfr=p["num_gfr"], num_mcmc=p["num_mcmc"],
        keep_every=p["keep_every"], num_chains=p["num_chains"], seed=int(seed),
        effect_by_cohort=bool(effect_by_cohort))
    return FitResult(df=model.df, mu_draws=np.asarray(model.mu_draws, float),
                     tau_draws=np.asarray(model.tau_draws, float),
                     row_of=_row_index_map(model.df),
                     design_cols=list(model.design.level_cols), bcf_params=p,
                     spec=spec)


def _smoke_plain(fit, method="raw_structured"):
    df, tau = fit.df, np.asarray(fit.tau_draws)
    treated = df["D"].to_numpy(int) == 1
    recs = []
    for (g, t), grp in df[treated].groupby(["cohort", "time"]):
        rows = grp.index.to_numpy(int)
        d = tau[rows].mean(axis=0)
        recs.append(_summary(d, method, "GATT", f"g={g:g}_t={int(t)}",
                             g=float(g), t=int(t), k=int(t-g)))
    if recs:
        all_rows = df.index[treated].to_numpy(int)
        recs.append(_summary(tau[all_rows].mean(axis=0), method, "ATT", "ATT"))
    return pd.DataFrame.from_records(recs)


def _summary(draws, method, typ, ident, g=np.nan, t=np.nan, k=np.nan):
    d = np.asarray(draws, float)
    tail_min = float(min(np.mean(d >= 0), np.mean(d <= 0)))
    return {"estimand_type": typ, "estimand_id": ident, "g": g, "t": t, "k": k,
            "method": method, "post_mean": float(d.mean()),
            "post_median": float(np.median(d)), "sd": float(d.std(ddof=1)),
            "q025": float(np.quantile(d, .025)), "q05": float(np.quantile(d, .05)),
            "q95": float(np.quantile(d, .95)), "q975": float(np.quantile(d, .975)),
            "p_bayes_tail_min": tail_min,
            "p_bayes_two_sided": float(min(1.0, 2.0 * tail_min)),
            "p_bayes": float(min(1.0, 2.0 * tail_min))}


def _decorate(frame, task: ExperimentTask):
    out = frame.copy()
    for key, val in {
        "family": task.family, "design": task.design, "degree": task.degree,
        "N": task.N, "rep": task.rep, "seed": task.seed,
        "task_estimator": task.estimator,
    }.items():
        out[key] = val
    return out


def run_task(task: ExperimentTask, *, bcf_params=None, smoke=False, K=2,
             clip=1e-3, outcome_method="rf", fit_cache=None,
             reference_fit_cache=None) -> pd.DataFrame:
    """Run one manifest task.  Smoke mode avoids every stochtree fit."""
    if task.estimator.startswith("oracle") and not task.oracle_available:
        return pd.DataFrame([{
            **task.as_dict(), "status": "unavailable",
            "reason": "exact m0/pi/barpi are not exposed by this DGP",
        }])
    df = _dgp(task)
    def _smoke_factory(panel, *, seed=0, **kwargs):
        return _SmokeFit(panel, seed=seed)
    base_factory = _smoke_factory if smoke else _production_fit
    fit_cache = fit_cache if fit_cache is not None else {}
    reference_fit_cache = reference_fit_cache if reference_fit_cache is not None else {}
    def factory(panel, *, seed=0, **kwargs):
        # One full-panel posterior is paired across raw/current variants.
        key = ("full", tuple(panel["unit_id"].astype(int).unique()),
               tuple(panel["time"].astype(int).unique()),
               bool(kwargs.get("effect_by_cohort", True)),
               str(kwargs.get("spec", "structured")), int(seed),
               json.dumps(kwargs.get("bcf_params") or {}, sort_keys=True, default=str))
        if key not in fit_cache:
            fit_cache[key] = base_factory(panel, seed=seed, **kwargs)
        return fit_cache[key]
    if task.estimator == "raw_structured":
        fit = factory(df, bcf_params=bcf_params, seed=task.seed)
        from .ablations import _raw_scalar_records
        result = _raw_scalar_records(fit, "raw_structured")
    elif task.estimator == "oracle_current_hybrid":
        from .correction import _cell_arrays
        from .reference import oracle_nuisance
        fit = factory(df, bcf_params=bcf_params, seed=task.seed)
        pi_map, m_map, bar_map = {}, {}, {}
        for g, t in df.loc[df["D"] == 1, ["cohort", "time"]].drop_duplicates().itertuples(index=False):
            key = (float(g), int(t))
            rows_t, _, delta, _, _ = _cell_arrays(fit, *key)
            m, p, b = oracle_nuisance(df, *key, rows_t, delta)
            pi_map[key], m_map[key], bar_map[key] = p, m, b
        result = current_hybrid_correction(
            fit, propensity_method="oracle", clip=clip, n_splits=1,
            seed=task.seed, oracle_pi_by_cell=pi_map,
            oracle_m0_by_cell=m_map, oracle_barpi_by_cell=bar_map).to_frame()
    elif task.estimator.startswith("current_hybrid"):
        fit = factory(df, bcf_params=bcf_params, seed=task.seed)
        current_specs = {
            "current_hybrid_intercept": ("intercept", clip),
            "current_hybrid_logit": ("logit", clip),
            "current_hybrid_rf": ("rf", clip),
            "current_hybrid_logit_clip01": ("logit", .01),
            "current_hybrid_logit_clip05": ("logit", .05),
        }
        if task.estimator not in current_specs:
            raise ValueError(f"unknown current-hybrid variant {task.estimator!r}")
        prop, c = current_specs[task.estimator]
        result = current_hybrid_correction(
            fit, propensity_method=prop, clip=c, n_splits=2,
            seed=task.seed).to_frame()
    elif task.estimator == "oracle_reference_fold_convolution":
        result = reference_fold_convolution(
            df, K=K, fold_seed=task.seed,
            posterior_seed=deterministic_seed("posterior", task.seed),
            propensity_method="oracle", outcome_method=outcome_method,
            clip=clip, bcf_params=bcf_params, fit_factory=factory,
            fit_cache=reference_fit_cache, oracle=True).to_frame()
    elif task.estimator.startswith("reference_fold_convolution"):
        reference_specs = {
            "reference_fold_convolution_intercept": ("intercept", clip),
            "reference_fold_convolution_logit": ("logit", clip),
            "reference_fold_convolution_rf": ("rf", clip),
        }
        if task.estimator not in reference_specs:
            raise ValueError(f"unknown reference variant {task.estimator!r}")
        prop, ref_clip = reference_specs[task.estimator]
        result = reference_fold_convolution(
            df, K=K, fold_seed=task.seed,
            posterior_seed=deterministic_seed("posterior", task.seed),
            propensity_method=prop, outcome_method=outcome_method,
            clip=ref_clip, bcf_params=bcf_params, fit_factory=factory,
            fit_cache=reference_fit_cache).to_frame()
    elif task.estimator in {"full_panel_raw", "reduced_cell", "pooled_full_panel"}:
        result, _, _ = run_information_ablation(
            df, variants=(task.estimator,), bcf_params=bcf_params,
            seed=task.seed, fit_factory=factory)
    else:
        raise ValueError(f"unknown estimator task {task.estimator!r}")
    if result.empty:
        return pd.DataFrame()
    try:
        from Simulation_Studies_Revision.did_bcf_revision.dgps import true_estimands
    except ImportError:
        from did_bcf_revision.dgps import true_estimands
    truth = true_estimands(df)
    truth_columns = ["estimand_type", "estimand_id", "true"]
    if task.design == "oracle_canonical":
        if "gatt_population_oracle" not in df.columns:
            raise AssertionError("oracle_canonical must expose gatt_population_oracle")
        population_truth = np.asarray(df["gatt_population_oracle"], dtype=float)
        if (not np.all(np.isfinite(population_truth)) or
                not np.allclose(population_truth, population_truth[0])):
            raise AssertionError("gatt_population_oracle must be one finite scalar")
        expected_truth = float(population_truth[0])
        if not np.allclose(truth["true"].to_numpy(float), expected_truth):
            raise AssertionError(
                "oracle_canonical realized truth drifted from gatt_population_oracle")
        truth = truth.copy()
        truth["truth_source"] = "gatt_population_oracle"
        truth_columns.append("truth_source")
    # Exact estimand-type/id join prevents a GATT value being copied to CATT.
    result = result.merge(truth[truth_columns],
                          on=["estimand_type", "estimand_id"], how="left")
    return _decorate(result, task)


def checkpoint_name(task: ExperimentTask) -> str:
    return (f"{task.family}__{task.design}__d{task.degree}__N{task.N}"
            f"__r{task.rep}__{task.estimator}.csv")


def save_checkpoint(frame: pd.DataFrame, task: ExperimentTask, out_dir: str | Path):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / checkpoint_name(task)
    frame.to_csv(path, index=False)
    return path


def run_tasks(tasks, *, out_dir="results", bcf_params=None, smoke=False,
              K=2, clip=1e-3, outcome_method="rf", resume=True):
    """Run/resume task rows, caching paired fits inside each replication bundle."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    bundles = {}
    for task in tasks:
        bundles.setdefault((task.family, task.design, task.degree, task.N, task.rep), []).append(task)
    for bundle_tasks in bundles.values():
        full_fit_cache, reference_fit_cache = {}, {}
        for task in bundle_tasks:
            path = out / checkpoint_name(task)
            if resume and path.exists():
                try:
                    rows.append(pd.read_csv(path)); continue
                except Exception:
                    path.unlink(missing_ok=True)
            frame = run_task(task, bcf_params=bcf_params, smoke=smoke, K=K,
                             clip=clip, outcome_method=outcome_method,
                             fit_cache=full_fit_cache,
                             reference_fit_cache=reference_fit_cache)
            save_checkpoint(frame, task, out)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def git_commit(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def write_provenance(path: str | Path, *, tasks, repo_root=".", config=None,
                     bcf_params=None, smoke=False):
    payload = {
        "package": "extra_theory_experiments",
        "repository_commit": git_commit(repo_root),
        "config": config or {},
        "bcf_params": bcf_params or {},
        "smoke": bool(smoke),
        "tasks": len(tasks),
        "task_seed_hashes": [t.as_dict() for t in tasks],
        "warnings": [
            "simulations motivate but do not prove theory",
            "reference_fold_convolution uses finite posterior/BB draws",
            "clipping is an exploratory stabilization and is not theoretically neutral",
        ],
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
