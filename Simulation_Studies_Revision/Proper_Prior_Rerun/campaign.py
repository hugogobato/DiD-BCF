#!/usr/bin/env python3
"""Prepare or execute isolated, raw-only proper-prior reruns.

No sampling occurs under `prepare`. Each job is checkpointed independently;
existing completed jobs are verified, never silently overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PROJECT = REV.parent
for path in (PROJECT, REV, REV / "scripts"):
    sys.path.insert(0, str(path))

VERSION = "0.2.0-proper-ig"
DEFAULT_OUT = REV / "Reruns" / "proper_ig_v1"
DEPENDENCIES = ("stochtree", "numpy", "pandas", "scipy", "scikit-learn")
SAMPLER = dict(num_gfr=50, num_burnin=0, num_mcmc=500,
               keep_every=5, num_chains=3)
MAIN_SETTINGS = {"B1_baseline", "B1_strong_confounder", "B1_serial_corr",
                 "B1_selection_obs", "B1_null", "B2_sweep", "B2_sweep_serial",
                 "D_staggered"}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def source_files():
    files = []
    for folder in (PROJECT / "didbcf_structured", REV / "did_bcf_revision", HERE):
        files.extend(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)
    files += [REV / "scripts" / name for name in
              ("run_pretrend_empirical.py", "run_empirical_structured.py")]
    files.append(PROJECT / "Empirical_Study" / "mpdta.csv")
    return sorted(set(files))


def source_hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files()}


def environment():
    return {name: importlib.metadata.version(name) for name in DEPENDENCIES}


def prepare(path, shards=48):
    from did_bcf_revision.config import all_experiments, degrees_for
    jobs = []
    phase_counts = Counter()
    def add(kind, phase, setting, rep, spec="structured", rate=1., **extra):
        job = dict(kind=kind, phase=phase, setting=setting, rep=rep, spec=spec,
                   sigma2_shape=2., sigma2_rate=rate, **extra)
        job["id"] = digest(job)[:20]
        job["shard"] = phase_counts[phase] % shards
        phase_counts[phase] += 1
        jobs.append(job)
    exps = all_experiments(reps=100)
    for exp in exps:
        if exp.name not in MAIN_SETTINGS and exp.workstream != "PT":
            continue
        for degree in degrees_for(exp.workstream):
            for n in exp.n_values:
                for rep in range(exp.reps):
                    extra = dict(dgp=exp.dgp, dgp_params={**exp.dgp_params,
                                 "linearity_degree": degree, "n_units": n},
                                 N=n, linearity_degree=degree)
                    if exp.workstream == "PT":
                        add("pretrend", "diagnostic", exp.name, rep, spec="pretrend_unit", **extra)
                        add("effect", "pt_impact", exp.name, rep, **extra)
                    else:
                        add("effect", "main", exp.name, rep, **extra)
    # Pre-specified scale sensitivity, not selected after seeing results.
    for exp in exps:
        if exp.name not in {"B2_sweep", "B2_sweep_serial"}:
            continue
        for rate in (.1, 10.):
            for n in (50, 200, 800):
                for rep in range(100):
                    add("effect", "sensitivity", exp.name, rep, rate=rate,
                        dgp=exp.dgp, dgp_params={**exp.dgp_params,
                        "linearity_degree": 1, "n_units": n}, N=n, linearity_degree=1)
    for rate in (1., .1, 10.):
        for seed in range(3):
            phase = "application" if rate == 1 else "application_sensitivity"
            for spec in ("structured", "structured_rfx_unit"):
                add("application", phase, "mpdta", seed, spec, rate)
            add("application_pretrend", phase, "mpdta", seed,
                "pretrend_unit", rate)
    manifest = dict(method_version=VERSION, sampler=SAMPLER,
                    effect_by_cohort=True, outcome_standardization="population_sd_ddof0",
                    source_hashes=source_hashes(), dependencies=environment(),
                    python=platform.python_version(), shards=shards,
                    phases=dict(Counter(j["phase"] for j in jobs)), jobs=jobs)
    manifest["manifest_hash"] = digest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    print(json.dumps({"manifest": str(path), "jobs": len(jobs),
                      "phases": manifest["phases"], "hash": manifest["manifest_hash"]}, indent=2))


def verify_manifest(path):
    manifest = json.loads(path.read_text())
    expected = manifest.pop("manifest_hash")
    if digest(manifest) != expected:
        raise ValueError("Manifest was altered; prepare a new campaign")
    manifest["manifest_hash"] = expected
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("Source/data changed since preparation; prepare a new manifest")
    if manifest["dependencies"] != environment():
        raise ValueError("Dependency versions differ from the manifest")
    return manifest


def scalar_draws(model):
    import numpy as np
    df, tau = model.df, model.tau_draws
    post = df[df.D == 1]
    draws = {"ATT": tau[post.index].mean(axis=0), "sigma2": model.sigma2_draws}
    for (g, t), rows in post.groupby(["cohort", "time"]):
        draws[f"GATT_g={g:g}_t={t}"] = tau[rows.index].mean(axis=0)
    for k, rows in post.groupby("event_time"):
        draws[f"ES_k={k:g}"] = tau[rows.index].mean(axis=0)
    n = len(model._chains)
    return {key: np.asarray(value).reshape(n, -1) for key, value in draws.items()}


def execute(job, manifest, root, smoke=False):
    import numpy as np
    import pandas as pd
    from didbcf_structured import StructuredDiDBCF, GlobalPrior
    from did_bcf_revision.dgps import generate_canonical_did, generate_staggered_did, true_estimands
    from did_bcf_revision.structured import fit_structured
    from did_bcf_revision.did_bcf import plain_estimands
    from did_bcf_revision.pretrend import fit_pretrend, pretrend_estimands, true_pretrend, quantile_subgroups
    from run_pretrend_empirical import load_mpdta, DEFAULT_DATA
    from run_empirical_structured import analyse

    root = root.resolve()
    historical = (REV / "Results").resolve()
    if root == historical or historical in root.parents:
        raise ValueError("Refusing to write repaired results under historical Results/")
    folder = root / ("smoke" if smoke else "production") / job["id"]
    identity = dict(job=job, manifest_hash=manifest["manifest_hash"], smoke=smoke)
    done = folder / "complete.json"
    if done.exists():
        record = json.loads(done.read_text())
        if record["identity"] != identity:
            raise ValueError(f"Conflicting existing job: {folder}")
        for name, checksum in record["outputs"].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Corrupt/incomplete output: {folder/name}")
        return "verified_existing"
    # A partial job is never overwritten; recovery requires a new output root.
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "started.json").write_text(json.dumps(identity, indent=2))
    start = time.monotonic()
    p = {**manifest["sampler"], "sigma2_shape": job["sigma2_shape"],
         "sigma2_rate": job["sigma2_rate"]}
    if smoke:
        p.update(num_gfr=3, num_mcmc=4, num_chains=2, keep_every=1)
    if job["kind"].startswith("application"):
        df = load_mpdta(DEFAULT_DATA)
    else:
        gen = {"canonical": generate_canonical_did, "staggered": generate_staggered_did}[job["dgp"]]
        params = dict(job["dgp_params"])
        if smoke:
            params["n_units"] = 50
        df = gen(seed=job["rep"], **params)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    arrays = {}
    if job["kind"] in ("pretrend", "application_pretrend"):
        empirical = job["kind"] == "application_pretrend"
        extra = dict(prognostic_cols=["lpop", "time"],
                     treatment_cols=["lpop", "k_diag"]) if empirical else {}
        fit = fit_pretrend(df, bcf_params=p, seed=job["rep"], rfx="unit", **extra)
        summaries = pretrend_estimands(fit, subgroups=quantile_subgroups("lpop") if empirical else True)
        truth = true_pretrend(df)
        if not truth.empty:
            summaries = summaries.merge(truth, on=["estimand_type", "estimand_id"], how="left")
        # Full diagnostic draws permit reconstruction of joint/subgroup statistics.
        arrays["tau"] = fit.tau_draws
        fit.df.to_csv(folder / "panel.csv", index=False)
        prior_metadata = dict(variance_prior_scale="standardized_outcome",
                              sigma2_shape=p["sigma2_shape"], sigma2_rate=p["sigma2_rate"])
    elif job["kind"] == "application":
        model = StructuredDiDBCF(rfx="unit" if job["spec"].endswith("unit") else "none",
                    global_prior=GlobalPrior(p["sigma2_shape"], p["sigma2_rate"])).sample(
                    df, covariates=["lpop"], effect_modifiers=["lpop"],
                    effect_by_cohort=True, seed=job["rep"],
                    **{key: p[key] for key in SAMPLER})
        summaries, catt, contrasts = analyse(model, job["spec"])
        catt.to_csv(folder / "catt.csv", index=False)
        contrasts.to_csv(folder / "contrasts.csv", index=False)
        arrays = {**scalar_draws(model), "tau": model.tau_draws}
        model.df.to_csv(folder / "panel.csv", index=False)
        prior_metadata = model.prior_metadata
    else:
        fit = fit_structured(df, bcf_params=p, seed=job["rep"], spec=job["spec"], effect_by_cohort=True)
        summaries = plain_estimands(fit).merge(true_estimands(df)[["estimand_type", "estimand_id", "true"]],
                         on=["estimand_type", "estimand_id"], how="left")
        arrays = scalar_draws(fit.structured_model)
        prior_metadata = fit.structured_model.prior_metadata
    for key in ("setting", "rep", "spec", "sigma2_shape", "sigma2_rate", "N", "linearity_degree", "dgp"):
        if key in job:
            summaries[key] = job[key]
    summaries["method_version"] = VERSION
    summaries["smoke_only"] = smoke
    summaries["method_label"] = "DiD-BCF diagnostic" if "pretrend" in job["kind"] else "DiD-BCF"
    summaries.to_csv(folder / "summaries.csv", index=False)
    np.savez_compressed(folder / "draws.npz", **arrays)
    outputs = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
               for f in folder.iterdir() if f.is_file()}
    metadata = dict(identity=identity, method_version=VERSION, prior=prior_metadata,
                    actual_sampler=p, actual_units=int(df.unit_id.nunique()),
                    panel_hash=data_hash, seconds=time.monotonic()-start,
                    dependencies=environment(), python=platform.python_version(), outputs=outputs)
    with done.open("x") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
    return metadata["seconds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "run", "smoke"])
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--shards", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--phase", default="main")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--hours", type=float, default=8.)
    args = ap.parse_args()
    if args.action == "prepare":
        if args.shards < 1:
            ap.error("--shards must be positive")
        prepare(args.manifest, args.shards)
        return
    manifest = verify_manifest(args.manifest)
    if not 0 <= args.shard < manifest["shards"]:
        ap.error("--shard is out of range")
    if args.action == "smoke":
        jobs = []
        for kind in ("effect", "pretrend", "application", "application_pretrend"):
            jobs.append(next(j for j in manifest["jobs"] if j["kind"] == kind))
        jobs.append(next(j for j in manifest["jobs"] if j["kind"] == "application" and j["spec"].endswith("unit")))
    else:
        phases = set(args.phase.split(","))
        if not phases <= set(manifest["phases"]):
            ap.error(f"Unknown phase; choose {list(manifest['phases'])}")
        jobs = [j for j in manifest["jobs"] if j["shard"] == args.shard and j["phase"] in phases]
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    deadline = time.monotonic() + args.hours*3600
    for job in jobs:
        if time.monotonic() >= deadline:
            print("Time budget reached; completed jobs are checkpointed.")
            break
        print(job["id"], execute(job, manifest, args.out, smoke=args.action == "smoke"), flush=True)


if __name__ == "__main__":
    main()
