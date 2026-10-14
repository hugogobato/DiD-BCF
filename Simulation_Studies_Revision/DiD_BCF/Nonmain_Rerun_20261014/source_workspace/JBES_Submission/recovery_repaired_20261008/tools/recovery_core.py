"""On-notebook execution core: chain-split sampling, tracing, unit artifacts.

The structured estimator (``StructuredDiDBCF``) is sampled one chain at a time
so that each chain is an independent recovery unit.  A subclass redirects the
chain seed explicitly through named parameters (never positional private-argument
patching) and preserves the original seed map ``1000 * rep + chain_index``.

The instrumented path records per-sweep scalar traces.  It is a copy of
``StructuredDiDBCF._sample_chain`` with recording calls inserted between the
same stochtree calls; ``tests_recovery.py::test_trace_leaves_kernel_unchanged``
asserts the main draws are bit-identical with tracing on and off.

Pretrend diagnostic jobs use stochtree's ``BCFModel`` wrapper, whose chains are
initialised from different grow-from-root ensembles and whose per-chain state
is not serialised by the wrapper.  Those jobs are therefore atomic recovery
units: an interrupted job is rerun from its recorded seed and is never appended
to.  This limitation is recorded in every pretrend completion record.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from recovery_common import (CAMPAIGN, PROJECT, SIM, TOOLS,
                             sha256_bytes, sha256_file, write_json_atomic)

for _path in (PROJECT, SIM, SIM / "scripts", TOOLS, CAMPAIGN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

__all__ = ["ChainModel", "load_panel", "panel_hash", "execute_unit",
           "structured_scalar_draws"]

TRACE_COLUMNS = ("sigma2", "tau_0", "leaf_scale_level", "leaf_scale_trend",
                 "leaf_scale_effect", "leaves_level", "leaves_trend",
                 "leaves_effect", "effect_mean", "level_mean", "trend_mean")


class ChainModel:
    """Factory for a single-chain structured fit with an explicit seed."""

    @staticmethod
    def build(target_chain_seed: int, trace: bool = False, **kwargs):
        from didbcf_structured import StructuredDiDBCF
        from didbcf_structured.model import _Block, _stochtree_seed

        class _Chain(StructuredDiDBCF):
            def __init__(self, *args, **inner):
                self._target_chain_seed = int(target_chain_seed)
                self._trace = bool(trace)
                self._stochtree_seed_fn = staticmethod(_stochtree_seed)
                self._block_cls = _Block
                super().__init__(*args, **inner)

            def _sample_chain(self, dm, resid0, rfx_ids, seed, num_gfr,
                              num_burnin, num_mcmc, keep_every):
                from recovery_validation import numeric_design
                for matrix in (dm.level, dm.trend, dm.effect):
                    numeric_design(matrix)
                if not self._trace:
                    return super()._sample_chain(
                        dm, resid0, rfx_ids, seed=self._target_chain_seed,
                        num_gfr=num_gfr, num_burnin=num_burnin,
                        num_mcmc=num_mcmc, keep_every=keep_every)
                return self._traced_chain(dm, resid0, rfx_ids, num_gfr,
                                          num_burnin, num_mcmc, keep_every)

            # ------------------------------------------------------------- #
            # Instrumented copy of StructuredDiDBCF._sample_chain
            # ------------------------------------------------------------- #
            def _traced_chain(self, dm, resid0, rfx_ids, num_gfr,
                              num_burnin, num_mcmc, keep_every):
                from stochtree import (GlobalModelConfig, GlobalVarianceModel,
                                       RNG, Residual)
                _Block = self._block_cls

                n = dm.n
                resid_var = float(np.var(resid0))
                rng = RNG(_stochtree_seed(self._target_chain_seed))
                np_rng = np.random.default_rng(self._target_chain_seed)

                sigma2 = resid_var
                global_config = GlobalModelConfig(global_error_variance=sigma2)
                residual = Residual(resid0.copy())

                level = _Block(self.level_prior, dm.level, None, n,
                               global_config, resid_var)
                trend = _Block(self.trend_prior, dm.trend, None, n,
                               global_config, resid_var)
                effect = _Block(self.effect_prior, dm.effect, dm.Z, n,
                                global_config, resid_var)
                level.initialise(residual, float(np.mean(resid0)))
                trend.initialise(residual, 0.0)
                effect.initialise(residual, 0.0)

                rfx = self._init_rfx(rfx_ids, n) if rfx_ids is not None else None

                global_var_model = GlobalVarianceModel()
                tau_0 = 0.0
                tau_0_prior_var = (self.global_prior.intercept_prior_var
                                   if self.global_prior.intercept_prior_var
                                   is not None else resid_var)
                ztz = float(dm.Z @ dm.Z)

                plan = _trace_plan(self.df, dm)
                stages, scalars, groups = [], [], []
                sigma2_draws, tau0_draws = [], []
                total_sweeps = num_gfr + num_burnin + num_mcmc * keep_every

                def record(stage):
                    stages.append(stage)
                    effect_cached = np.asarray(
                        effect.sampler.get_cached_forest_predictions()
                    ).reshape(-1)
                    level_cached = np.asarray(
                        level.sampler.get_cached_forest_predictions()
                    ).reshape(-1)
                    trend_cached = np.asarray(
                        trend.sampler.get_cached_forest_predictions()
                    ).reshape(-1)
                    scalars.append((
                        sigma2, tau_0,
                        float(level.leaf_scale[0, 0]),
                        float(trend.leaf_scale[0, 0]),
                        float(effect.leaf_scale[0, 0]),
                        float(level.active.num_forest_leaves()),
                        float(trend.active.num_forest_leaves()),
                        float(effect.active.num_forest_leaves()),
                        float(effect_cached.mean()),
                        float(level_cached.mean()),
                        float(trend_cached.mean()),
                    ))
                    groups.append([float(effect_cached[rows].mean())
                                   for rows in plan["groups"]])

                def sweep(gfr: bool, keep: bool, stage: int):
                    nonlocal sigma2, tau_0
                    level.step(residual, rng, global_config, keep, gfr)
                    trend.step(residual, rng, global_config, keep, gfr)
                    if self.global_prior.sample_intercept and ztz > 0:
                        partial = (np.squeeze(residual.get_residual())
                                   + tau_0 * dm.Z)
                        v_post = 1.0 / (ztz / sigma2 + 1.0 / tau_0_prior_var)
                        m_post = v_post * float(dm.Z @ partial) / sigma2
                        tau_0_new = float(np_rng.normal(m_post, np.sqrt(v_post)))
                        residual.add_vector(-(tau_0_new - tau_0) * dm.Z)
                        tau_0 = tau_0_new
                    effect.step(residual, rng, global_config, keep, gfr)
                    if rfx is not None:
                        rfx["model"].sample(rfx["dataset"], residual,
                                            rfx["tracker"], rfx["container"],
                                            keep, sigma2, rng)
                    sigma2 = float(global_var_model.sample_one_iteration(
                        residual, rng, self.global_prior.sigma2_shape,
                        self.global_prior.sigma2_rate))
                    global_config.update_global_error_variance(sigma2)
                    level.step_leaf_scale(rng)
                    trend.step_leaf_scale(rng)
                    effect.step_leaf_scale(rng)
                    if keep:
                        sigma2_draws.append(sigma2)
                        tau0_draws.append(tau_0)
                    record(stage)

                for _ in range(num_gfr):
                    sweep(True, False, 0)
                for i in range(num_burnin + num_mcmc * keep_every):
                    past_burnin = i >= num_burnin
                    keep = past_burnin and (
                        (i - num_burnin + 1) % keep_every == 0)
                    sweep(False, keep, 2 if past_burnin else 1)
                trace_data = {
                    "sweep": np.arange(total_sweeps, dtype=np.int64),
                    "stage": np.asarray(stages, dtype=np.int8),
                    "scalar_names": np.asarray(list(TRACE_COLUMNS), dtype=str),
                    "scalars": np.asarray(scalars, dtype=float),
                    "group_names": np.asarray(plan["names"], dtype=str),
                    "group_means": np.asarray(groups, dtype=float),
                    "num_gfr": np.int64(num_gfr),
                    "num_burnin": np.int64(num_burnin),
                    "keep_every": np.int64(keep_every),
                    "num_mcmc": np.int64(num_mcmc),
                    "chain_seed": np.int64(self._target_chain_seed),
                    "y_std": np.float64(self.y_std),
                }
                return {"level": level, "trend": trend, "effect": effect,
                        "rfx": rfx,
                        "sigma2": np.asarray(sigma2_draws, dtype=float),
                        "tau_0": np.asarray(tau0_draws, dtype=float),
                        "trace": trace_data}

        return _Chain(**kwargs)


def _trace_plan(df, dm) -> dict:
    """Row positions of the reported treatment-effect groups, from model.df."""
    post = df[df[dm.cols.treated] == 1]
    names = ["ATT"]
    groups = [post.index.to_numpy(dtype=int)]
    for (g, t), rows in post.groupby([dm.cols.cohort, dm.cols.time]):
        names.append(f"GATT_g={g:g}_t={t}")
        groups.append(rows.index.to_numpy(dtype=int))
    for k, rows in post.groupby(dm.cols.event_time):
        names.append(f"ES_k={k:g}")
        groups.append(rows.index.to_numpy(dtype=int))
    return {"names": names, "groups": groups}


def structured_scalar_draws(model) -> dict:
    """Per-chain scalar draws, identical keys and grouping to campaign.py."""
    from campaign import scalar_draws as _campaign_scalar_draws
    return {key: np.asarray(value, dtype=float)
            for key, value in _campaign_scalar_draws(model).items()}


def load_panel(job: dict):
    """Deterministic panel for one statistical job (DGP or mpdta)."""
    if job["kind"].startswith("application"):
        from run_pretrend_empirical import DEFAULT_DATA, load_mpdta
        return load_mpdta(DEFAULT_DATA).sort_values(
            ["unit_id", "time"]).reset_index(drop=True)
    from did_bcf_revision.dgps import (generate_canonical_did,
                                       generate_staggered_did)
    gen = {"canonical": generate_canonical_did,
           "staggered": generate_staggered_did}[job["dgp"]]
    params = dict(job["dgp_params"])
    return gen(seed=job["rep"], **params).sort_values(["unit_id", "time"]).reset_index(drop=True)


def panel_hash(df) -> str:
    import pandas as pd
    return sha256_bytes(pd.util.hash_pandas_object(df, index=True).values.tobytes())


def _artifact_checksums(folder: Path) -> dict:
    return {path.name: sha256_file(path) for path in sorted(folder.iterdir())
            if path.is_file() and path.name not in
            ("started.json", "complete.json", "started.json.tmp",
             "complete.json.tmp")}


def _save_npz_atomic(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def execute_unit(unit: dict, identity: dict, workdir: Path) -> dict:
    """Run one recovery unit and publish its completion record last."""
    started = time.monotonic()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(workdir / "started.json", identity)

    job = unit["job"]
    bdg = dict(unit["budget"])
    df = load_panel(job)
    from recovery_engine import activate, fingerprint
    active_engine = fingerprint(activate())
    numeric = df.select_dtypes(include=["number"]).to_numpy(dtype=float)
    # Infinite cohort sentinels are not engine inputs; validate actual design in each path.
    data_hash = panel_hash(df)

    if job["kind"] in ("pretrend", "application_pretrend"):
        info = _run_pretrend(unit, bdg, df, workdir, identity)
    else:
        info = _run_structured(unit, bdg, df, workdir, identity)

    arrays = info.pop("arrays")
    from recovery_validation import validate_arrays
    validate_arrays(arrays, unit)
    _save_npz_atomic(workdir / "draws.npz", arrays)
    trace = info.pop("trace")
    if trace is not None:
        _save_npz_atomic(workdir / "trace.npz", trace)
    if info.pop("panel"):
        df.to_csv(workdir / "panel.csv", index=False)
    info["engine"] = active_engine
    info["panel_hash"] = data_hash
    info["seconds"] = time.monotonic() - started
    info["output_bytes"] = sum(p.stat().st_size for p in workdir.iterdir() if p.is_file() and p.name != "started.json")
    info["outputs"] = _artifact_checksums(workdir)
    record = {**identity, **info, "completed": True, "artifact": workdir.name}
    write_json_atomic(workdir / "complete.json", record)
    return record


def _run_structured(unit: dict, bdg: dict, df, workdir: Path,
                    identity: dict) -> dict:
    import resource
    job = unit["job"]
    chain_index = int(unit["chain_index"])
    chain_seed = int(unit["chain_seed"])
    trace = bool(unit.get("trace", False))

    from didbcf_structured import GlobalPrior
    rfx = "unit" if job.get("spec", "").endswith("unit") else "none"
    kwargs = {}
    if job["kind"] == "application":
        kwargs = dict(covariates=["lpop"], effect_modifiers=["lpop"])
    model = ChainModel.build(
        target_chain_seed=chain_seed, trace=trace,
        rfx=rfx, global_prior=GlobalPrior(job["sigma2_shape"],
                                          job["sigma2_rate"]),
    ).sample(df, effect_by_cohort=True, seed=job["rep"],
             num_gfr=bdg["num_gfr"], num_burnin=bdg["num_burnin"],
             num_mcmc=bdg["num_mcmc"], keep_every=bdg["keep_every"],
             num_chains=1, **kwargs)

    arrays = structured_scalar_draws(model)
    arrays["tau"] = np.asarray(model.tau_draws, dtype=float)
    rfx_state = model._chains[0].get("rfx")
    if rfx_state is not None:
        try:
            components = rfx_state["container"].extract_parameter_samples()
            sigma = np.asarray(components["sigma_samples"], dtype=float)
            if sigma.ndim == 1:
                sigma = sigma[None, :]
            arrays["rfx_sigma"] = sigma
        except Exception as error:
            raise RuntimeError("Required random-effect variance draws unavailable") from error
    trace_payload = model._chains[0].get("trace") if trace else None
    if trace_payload is not None:
        positions = np.flatnonzero(trace_payload["stage"] == 2)[bdg["keep_every"] - 1::bdg["keep_every"]]
        names = list(trace_payload["scalar_names"])
        for name in ("leaf_scale_level", "leaf_scale_trend", "tau_0"):
            arrays[name] = trace_payload["scalars"][positions, names.index(name)][None, :]
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError("Non-finite sampled output: " + key)
    info = {
        "kind": job["kind"],
        "chain_index": chain_index,
        "chain_seed": chain_seed,
        "sampler": bdg,
        "actual_sampler": {**bdg, "sigma2_shape": job["sigma2_shape"],
                           "sigma2_rate": job["sigma2_rate"]},
        "prior": {**model.prior_metadata, "rfx": rfx,
                  "level_prior": __import__("dataclasses").asdict(model.level_prior),
                  "trend_prior": __import__("dataclasses").asdict(model.trend_prior),
                  "effect_prior": __import__("dataclasses").asdict(model.effect_prior)},
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024,
        "arrays": arrays,
        "trace": trace_payload,
        "panel": job["kind"] == "application",
        "unit_mode": "independent_chain",
    }
    return info


def _run_pretrend(unit: dict, bdg: dict, df, workdir: Path,
                  identity: dict) -> dict:
    import resource
    import stochtree
    from did_bcf_revision.pretrend import fit_pretrend
    job = unit["job"]

    empirical = job["kind"] == "application_pretrend"
    extra = dict(prognostic_cols=["lpop", "time"],
                 treatment_cols=["lpop", "k_diag"]) if empirical else {}
    params = {**bdg, "sigma2_shape": job["sigma2_shape"],
              "sigma2_rate": job["sigma2_rate"]}

    # Capture the BCFModel instance so its retained diagnostic series can be
    # archived.  The fit itself is the unmodified frozen fit_pretrend; only the
    # class binding is wrapped for the duration of the call.
    original = stochtree.BCFModel
    captured = []

    class _Capture(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

        def sample(self, *args, **kw):
            from recovery_weight_controls import numeric_support_guard
            from recovery_bcf_spec import actual_parameters
            self._recovery_parameters = actual_parameters(kw)
            X = np.asarray(kw["X_train"], dtype=float)
            for name in ("prognostic_forest_params", "treatment_effect_forest_params"):
                keep = np.asarray(kw[name]["keep_vars"], dtype=int)
                weights = np.zeros(X.shape[1])
                weights[keep] = 1. / len(keep)
                numeric_support_guard(X, weights, num_features_subsample=X.shape[1])
            return super().sample(*args, **kw)

    stochtree.BCFModel = _Capture
    try:
        fit = fit_pretrend(df, bcf_params=params, seed=job["rep"], rfx="unit",
                           **extra)
    finally:
        stochtree.BCFModel = original

    arrays = {"tau": np.asarray(fit.tau_draws, dtype=float)}
    model = captured[0] if captured else None
    series = {}
    if model is not None:
        for name in ("global_var_samples", "leaf_scale_mu_samples",
                     "leaf_scale_tau_samples", "tau_0_samples"):
            value = getattr(model, name, None)
            if value is None:
                continue
            series[name] = np.asarray(value, dtype=float)
    if model is None:
        raise RuntimeError("Frozen pretrend BCFModel recording hook was not invoked")
    if getattr(model, "rfx_container", None) is None:
        raise RuntimeError("Required pretrend random-effect container unavailable")
    rfx = model.rfx_container.extract_parameter_samples()
    series["rfx_sigma"] = np.asarray(rfx["sigma_samples"], dtype=float)
    arrays.update(series)
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError("Non-finite sampled output: " + key)
    info = {
        "kind": job["kind"],
        "chain_index": None,
        "chain_seed": None,
        "sampler": bdg,
        "actual_sampler": params,
        "prior": {
            "variance_prior_scale": "standardized_outcome",
            "sigma2_shape": job["sigma2_shape"],
            "sigma2_rate": job["sigma2_rate"],
            "effect_by_cohort": None,
            "rfx": "unit",
            "estimator": "stochtree.BCFModel",
            "bcf_parameters": model._recovery_parameters,
            "bcf_default_spec": __import__("recovery_bcf_spec").defaults(),
        },
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024,
        "arrays": arrays,
        "trace": None,
        "panel": True,
        "unit_mode": "atomic_job_bcfmodel",
        "continuation_limitation": (
            "stochtree.BCFModel does not expose per-chain state; this job is "
            "atomic and an interrupted attempt is rerun from its recorded seed, "
            "never appended to"),
        "bcfmodel_series_shapes": {k: list(v.shape)
                                   for k, v in series.items()},
    }
    return info
