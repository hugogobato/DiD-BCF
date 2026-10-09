#!/usr/bin/env python3
"""E4 fresh collector: verify returned recovery runs and assemble diagnostics.

Verifies the new manifest, source/environment bundle, every run identity,
completion status, output checksums, shapes, seeds and prior fields; rejects
duplicate conflicting copies and smoke artifacts; assembles all chains of a job
before computing multi-chain diagnostics; recomputes finite-sample truths from
the regenerated panel; and writes a machine-readable status file with expected,
complete, missing, failed, duplicate and conflict counts.

Inputs are returned zips and/or directories produced by the recovery
notebooks.  Old analyses are never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from recovery_common import (INVENTORY, REPO, digest, sha256_bytes,
                             sha256_file, write_json_atomic, zip_member_safe)
from recovery_diagnostics import summarise
from recovery_plan import ATOMIC_KINDS, make_units
from recovery_validation import validate_arrays

EXPECTED_OUTPUTS = {"started.json", "draws.npz", "complete.json"}
OPTIONAL_OUTPUTS = {"trace.npz", "panel.csv"}
STRATA = ["phase", "budget_key", "sigma2_shape", "sigma2_rate"]


class _LazyPayload:
    """Keep archive locations, never full effect surfaces, in collected records."""

    def __init__(self, source, folder="", names=None, engine=None):
        self.source = Path(source).resolve()
        self.folder = folder
        self.names = names
        self.engine = engine

    def _member(self, member):
        if not zip_member_safe(member) or "/" in member or "\\" in member:
            raise ValueError("Unsafe output artifact name")
        return f"{self.folder}/{member}" if self.folder else member

    def __call__(self, member):
        name = self._member(member)
        if self.names is not None:
            if name not in self.names:
                return None
            with zipfile.ZipFile(self.source) as archive:
                return archive.read(name)
        path = self.source / name
        return path.read_bytes() if path.is_file() else None

    def checksum(self, member):
        name = self._member(member)
        digest_value = hashlib.sha256()
        if self.names is not None:
            if name not in self.names:
                return None
            with zipfile.ZipFile(self.source) as archive, archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest_value.update(chunk)
        else:
            path = self.source / name
            if not path.is_file():
                return None
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest_value.update(chunk)
        return digest_value.hexdigest()


def verify_engine_archive(source, manifest):
    """Verify one archived repaired build, its files and both loaded controls."""
    from recovery_engine import specification
    expected = manifest.get("engine_specification", specification())
    source = Path(source).resolve()
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        if archive is not None:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(not zip_member_safe(n) for n in names):
                raise ValueError("Unsafe or duplicate paths in returned archive")
            names = set(names)

        def payload(name):
            if not zip_member_safe(name):
                raise ValueError("Unsafe engine artifact path")
            if archive is not None:
                if name not in names:
                    raise ValueError("Missing engine artifact: " + name)
                return archive.read(name)
            path = (source / name).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError("Missing/unsafe engine artifact: " + name)
            return path.read_bytes()

        setup = json.loads(payload("_engine/setup_complete.json"))
        if setup.get("status") != "PASS":
            raise ValueError("Archived repaired setup did not pass")
        engine = setup["engine"]
        if engine.get("specification") != expected or expected.get("variant") != "repaired":
            raise ValueError("Archived engine specification differs from repaired manifest")
        build_payload = payload("_engine/build_complete.json")
        build = json.loads(build_payload)
        for key in ("variant", "build_manifest_sha256", "patch_sha256", "source_hashes", "archives"):
            if build.get(key) != expected.get(key):
                raise ValueError("Archived engine build differs: " + key)
        if (engine.get("build_record_sha256") != sha256_bytes(build_payload)
                or engine.get("build_identity") != build.get("identity")
                or engine.get("site_files") != build.get("site_files")
                or engine.get("python") != build.get("python", "").split()[0]):
            raise ValueError("Archived setup/build identity mismatch")
        extensions = []
        for rel, expected_sha in build["site_files"].items():
            if not zip_member_safe(rel):
                raise ValueError("Unsafe native site member")
            name = "_engine/site/" + rel
            actual = hashlib.sha256()
            if archive is not None:
                if name not in names:
                    raise ValueError("Missing native site member: " + rel)
                with archive.open(name) as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        actual.update(chunk)
            else:
                path = (source / name).resolve()
                if not path.is_relative_to(source) or not path.is_file():
                    raise ValueError("Missing/unsafe native site member: " + rel)
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        actual.update(chunk)
            if actual.hexdigest() != expected_sha:
                raise ValueError("Native site checksum mismatch: " + rel)
            if Path(rel).name.startswith("stochtree_cpp") and Path(rel).suffix == ".so":
                extensions.append(expected_sha)
        if extensions != [engine.get("extension_sha256")]:
            raise ValueError("Archived loaded extension differs from compiled site")
        native_payload = payload("_engine/repaired_controls.json")
        native = json.loads(native_payload)
        probes = native.get("native_binary_probes", {})
        loaded = probes.get("loaded_native_binary", {})
        binary = loaded.get("extension_sha256", loaded.get("sha256"))
        if (native.get("deterministic_scope_gate") != "PASS"
                or probes.get("native_scope_gate") != "PASS"
                or binary != engine["extension_sha256"]
                or engine.get("controls_sha256") != sha256_bytes(native_payload)):
            raise ValueError("Archived native controls failed or belong to another binary")
        weights_payload = payload("_engine/weight_controls.json")
        weights = json.loads(weights_payload)
        if (weights.get("status") != "PASS"
                or weights.get("loaded_engine", {}).get("binary_sha256") != engine["extension_sha256"]
                or engine.get("weight_controls_sha256") != sha256_bytes(weights_payload)):
            raise ValueError("Archived feature-support controls failed or binary differs")
        return __import__("recovery_engine").fingerprint(engine)
    finally:
        if archive is not None:
            archive.close()


class _PretrendShim:
    """Minimal PretrendFit-compatible object for pretrend_estimands."""

    def __init__(self, df, tau, ref_k=-1):
        self.df = df
        self.tau_draws = np.asarray(tau, dtype=float)
        self.ref_k = ref_k
        self.row_of = {(int(u), int(t)): i
                       for i, (u, t) in enumerate(zip(df["unit_id"],
                                                      df["time"]))}


class _ModelShim:
    def __init__(self, df, tau):
        self.df = df
        self.tau_draws = np.asarray(tau, dtype=float)


def iter_complete_records(sources, manifest=None):
    """Yield lazy per-run loaders after one engine verification per input."""
    for source in sources:
        source = Path(source).resolve()
        engine = verify_engine_archive(source, manifest or {})
        if source.is_file() and source.suffix == ".zip":
            with zipfile.ZipFile(source) as archive:
                names = frozenset(archive.namelist())
                for name in sorted(n for n in names if n.endswith("/complete.json")):
                    folder = name.rsplit("/", 1)[0]
                    record = json.loads(archive.read(name))
                    yield str(source), name, record, _LazyPayload(source, folder, names, engine)
        elif source.is_dir():
            for path in sorted(source.rglob("complete.json")):
                if "_engine" in path.relative_to(source).parts:
                    continue
                folder = str(path.parent.relative_to(source))
                yield str(source), str(path), json.loads(path.read_text()), _LazyPayload(source, folder, engine=engine)
        else:
            raise FileNotFoundError(source)


def _panel_for_entry(entry):
    from recovery_core import load_panel, panel_hash
    fresh = load_panel(entry["unit"]["job"]).reset_index(drop=True)
    if panel_hash(fresh) != entry["record"].get("panel_hash"):
        raise ValueError("Panel hash differs from frozen job regeneration")
    stored = _load_panel(entry)
    if stored is not None:
        if list(stored.columns) != list(fresh.columns) or stored.shape != fresh.shape:
            raise ValueError("Archived panel columns/rows differ from frozen job")
        # CSV float serialization can change the final representable bit.
        # All categorical/integer/string columns must agree exactly.
        for key in fresh.columns:
            if pd.api.types.is_numeric_dtype(fresh[key]):
                if not np.allclose(stored[key].to_numpy(dtype=float), fresh[key].to_numpy(dtype=float),
                                   rtol=1e-13, atol=1e-13, equal_nan=True):
                    raise ValueError("Archived panel values differ: " + key)
            elif not stored[key].equals(fresh[key]):
                raise ValueError("Archived panel values differ: " + key)
    return fresh


def validate_record(record, expected_units, manifest, loader):
    run_id = record.get("run_id")
    if run_id not in expected_units:
        return None, [f"unknown run id {run_id!r}"]
    unit = expected_units[run_id]
    job = unit["job"]
    problems = []
    fields = {"stat_job_id": unit["stat_job_id"], "parent_job_id": job["id"],
              "budget_key": unit["budget_key"], "chain_index": unit["chain_index"],
              "chain_seed": unit["chain_seed"], "parent_manifest": manifest["parent_manifest"],
              "budget": unit["budget"], "sampler": unit["budget"],
              "trace": bool(unit["trace"]), "recovery_version": manifest["recovery_version"]}
    fields.update({k: job[k] for k in ("phase", "kind", "setting", "rep", "spec")})
    for field, expected in fields.items():
        if record.get(field) != expected:
            problems.append(field + " mismatch")
    actual = {**unit["budget"], "sigma2_shape": job["sigma2_shape"], "sigma2_rate": job["sigma2_rate"]}
    if record.get("actual_sampler") != actual:
        problems.append("actual sampler mismatch")
    if record.get("source_hashes_sha256") != digest(manifest["source_hashes"]):
        problems.append("source bundle mismatch")
    if record.get("dependencies") != manifest["dependencies"]:
        problems.append("dependency mismatch")
    engine = record.get("engine")
    if engine != getattr(loader, "engine", None):
        problems.append("record engine differs from verified archive build")
    if not engine or engine.get("specification") != manifest.get("engine_specification"):
        problems.append("manifest engine specification mismatch")
    if record.get("smoke") or "smoke" in str(record.get("purpose", "")):
        problems.append("smoke artifact")
    outputs = record.get("outputs", {})
    required = {"draws.npz"}
    if unit["trace"]:
        required.add("trace.npz")
    if unit["chain_index"] is None or job["kind"].startswith("application"):
        required.add("panel.csv")
    if not required.issubset(outputs):
        problems.append("missing checksummed mandatory outputs: " + repr(sorted(required - set(outputs))))
    if set(outputs) - EXPECTED_OUTPUTS - OPTIONAL_OUTPUTS:
        problems.append("unexpected output names")
    for name, checksum in outputs.items():
        try:
            actual_sha = loader.checksum(name) if hasattr(loader, "checksum") else (
                sha256_bytes(loader(name)) if loader(name) is not None else None)
            if actual_sha != checksum:
                problems.append("missing/checksum artifact: " + name)
        except Exception as error:
            problems.append("unsafe/unreadable artifact: " + repr(error))
    prior = record.get("prior", {})
    global_prior = prior.get("global_prior", prior)
    for key in ("sigma2_shape", "sigma2_rate"):
        if global_prior.get(key) != job[key]:
            problems.append("prior " + key + " mismatch")
    expected_rfx = "unit" if unit["chain_index"] is None or job.get("spec", "").endswith("unit") else "none"
    if prior.get("rfx") != expected_rfx:
        problems.append("random-effect specification mismatch")
    if unit["chain_index"] is None:
        if prior.get("bcf_default_spec") != manifest["model_spec"]["diagnostic_defaults"]:
            problems.append("Full diagnostic forest/global prior defaults mismatch")
        parameters = prior.get("bcf_parameters", {})
        general = parameters.get("general_params_default", {})
        if general.get("sigma2_global_shape") != job["sigma2_shape"] or general.get("sigma2_global_scale") != job["sigma2_rate"] or general.get("propensity_covariate") != "none":
            problems.append("Diagnostic explicit proper-prior/input overrides mismatch")
        if prior.get("estimator") != "stochtree.BCFModel" or prior.get("effect_by_cohort") is not None:
            problems.append("diagnostic model identity mismatch")
    else:
        if prior.get("effect_by_cohort") is not True:
            problems.append("structured cohort specification mismatch")
        frozen = manifest["model_spec"]
        for key in ("level", "trend", "effect"):
            if prior.get(key + "_prior") != frozen["forest_priors"][key]:
                problems.append("forest prior mismatch: " + key)
        expected_global = {**frozen["global_prior"], "sigma2_shape": job["sigma2_shape"], "sigma2_rate": job["sigma2_rate"]}
        if prior.get("global_prior") != expected_global:
            problems.append("full global-prior mismatch")
    try:
        entry = {"loader": loader, "unit": unit, "record": record}
        panel = _panel_for_entry(entry)
        arrays = _load_draws(entry)
        validate_arrays(arrays, unit)
        if arrays["tau"].shape[0] != len(panel):
            raise ValueError("Full effect surface does not match regenerated panel rows")
        if expected_rfx == "unit" and "rfx_sigma" not in arrays:
            raise ValueError("Required random-effect variance series absent")
    except Exception as error:
        problems.append("array/panel verification: " + repr(error))
    return unit, problems


def gather(sources, manifest):
    expected_units = {u["run_id"]: u for u in make_units(manifest)}
    records, problems, duplicates, smoke = {}, [], [], []
    for source in sources:
        try:
            iterator = iter_complete_records([source], manifest)
            for origin, member, record, loader in iterator:
                unit, issues = validate_record(record, expected_units, manifest, loader)
                if unit is None or issues:
                    item = {"source": origin, "member": member, "run_id": record.get("run_id"), "issues": issues}
                    (smoke if any("smoke" in i for i in issues) else problems).append(item)
                    continue
                run_id = unit["run_id"]
                if run_id in records:
                    same = record == records[run_id]["record"]
                    item = {"run_id": run_id, "first": records[run_id]["source"], "second": origin,
                            "identical": same}
                    (duplicates if same else problems).append(item)
                    continue
                records[run_id] = {"unit": unit, "record": record, "source": origin,
                                   "member": member, "loader": loader}
        except Exception as error:
            problems.append({"source": str(source), "issues": ["archive engine/integrity: " + repr(error)]})
    expected_pairs = {}
    for run_id, unit in expected_units.items():
        expected_pairs.setdefault((unit["stat_job_id"], unit["budget_key"]), set()).add(run_id)
    complete_pairs = {pair for pair, ids in expected_pairs.items() if ids.issubset(records)}
    status = {"manifest_hash": manifest["manifest_hash"], "expected_units": len(expected_units),
              "complete_units": len(records), "missing_units": len(expected_units) - len(records),
              "conflicting_units": len(problems), "duplicate_units": len(duplicates),
              "smoke_rejected": len(smoke), "expected_job_budget_pairs": len(expected_pairs),
              "complete_job_budget_pairs": len(complete_pairs),
              "started_job_budget_pairs": len({(e["unit"]["stat_job_id"],e["unit"]["budget_key"]) for e in records.values()}),
              "incomplete_job_budget_pairs": [{"job_id": pair[0], "budget_key": pair[1],
                    "missing_run_ids": sorted(ids - set(records))}
                    for pair, ids in sorted(expected_pairs.items()) if pair not in complete_pairs],
              "problems": problems, "duplicates": duplicates, "smoke": smoke}
    return expected_units, records, status


def _load_draws(entry):
    payload = entry["loader"]("draws.npz")
    with np.load(io.BytesIO(payload), allow_pickle=False) as draws:
        return {key: draws[key].copy() for key in draws.files}


def _load_panel(entry):
    payload = entry["loader"]("panel.csv")
    return pd.read_csv(io.BytesIO(payload)) if payload else None


def _parse_key(key):
    if key == "ATT":
        return "ATT", "ATT"
    if key.startswith("GATT_"):
        return "GATT", key[len("GATT_"):]
    if key.startswith("ES_"):
        return "ES", key[len("ES_"):]
    return None, key


def _per_rep_row(job, key, summary, draws, budget_key):
    estimand_type, estimand_id = _parse_key(key)
    return {
        "job_id": job["id"], "method": "plain",
        "estimand_type": estimand_type, "estimand_id": estimand_id,
        "post_mean": summary["post_mean"], "sd": summary["post_sd"],
        "q025": summary["q025"], "q05": summary["q05"],
        "q95": summary["q95"], "q975": summary["q975"],
        "p_bayes": float(min((draws > 0).mean(), (draws < 0).mean())),
        "dgp": job.get("dgp", "empirical"), "setting": job["setting"], "N": job.get("N", 500),
        "linearity_degree": job.get("linearity_degree"), "spec": job["spec"],
        "rep": job["rep"], "sigma2_shape": job["sigma2_shape"],
        "sigma2_rate": job["sigma2_rate"], "phase": job["phase"],
        "budget_key": budget_key,
    }


def _truth_for(job, panel):
    from did_bcf_revision.dgps import true_estimands
    truth = true_estimands(panel)
    return {(row.estimand_type, row.estimand_id): float(row.true)
            for row in truth.itertuples()}


def _metadata(job, budget_key):
    return {"job_id": job["id"], "budget_key": budget_key,
            "phase": job["phase"], "kind": job["kind"],
            "dgp": job.get("dgp", "empirical"), "setting": job["setting"],
            "N": job.get("N", 500), "linearity_degree": job.get("linearity_degree"),
            "rep": job["rep"], "spec": job["spec"],
            "sigma2_shape": job["sigma2_shape"], "sigma2_rate": job["sigma2_rate"]}


def _surface_heads(panel, tau):
    post = panel[panel["D"] == 1]
    heads = {"ATT": tau[post.index].mean(axis=0)}
    for (g, t), rows in post.groupby(["cohort", "time"]):
        heads[f"GATT_g={g:g}_t={t}"] = tau[rows.index].mean(axis=0)
    for k, rows in post.groupby("event_time"):
        heads[f"ES_k={k:g}"] = tau[rows.index].mean(axis=0)
    return heads


def _structured_entries(job, records, budget_key):
    entries = [e for e in records.values() if e["unit"]["stat_job_id"] == job["id"]
               and e["unit"]["budget_key"] == budget_key and e["unit"]["chain_index"] is not None]
    if not entries:
        return []
    expected = set(range(entries[0]["unit"]["budget"]["num_chains"]))
    if {e["unit"]["chain_index"] for e in entries} != expected:
        return []
    return sorted(entries, key=lambda e: e["unit"]["chain_index"])


def _trace_arrays(entry):
    payload = entry["loader"]("trace.npz")
    if payload is None:
        return {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as trace:
        sweep, stage = trace["sweep"], trace["stage"]
        bdg = entry["unit"]["budget"]
        retained = (stage == 2) & ((sweep - bdg["num_gfr"] - bdg["num_burnin"] + 1) % bdg["keep_every"] == 0)
        if retained.sum() != bdg["num_mcmc"]:
            raise ValueError("Trace retained count differs from sampler budget")
        names = trace["scalar_names"].tolist()
        return {"trace_" + str(name): trace["scalars"][retained, i].copy()
                for i, name in enumerate(names) if str(name) != "sigma2"}


@contextmanager
def _pooled_structured(job, records, budget_key):
    """One disk-backed pooled surface; only one decompressed chain in memory."""
    entries = _structured_entries(job, records, budget_key)
    if not entries:
        yield None
        return
    panel = _panel_for_entry(entries[0])
    n_mcmc = entries[0]["unit"]["budget"]["num_mcmc"]
    with tempfile.TemporaryDirectory(prefix="didbcf-collect-") as scratch:
        tau = np.memmap(Path(scratch) / "pooled.dat", dtype=np.float64, mode="w+",
                        shape=(len(panel), len(entries) * n_mcmc))
        scalar = {}
        keys = None
        for c, entry in enumerate(entries):
            if _panel_for_entry(entry).shape != panel.shape:
                raise ValueError("Chain panels disagree")
            arrays = _load_draws(entry)
            validate_arrays(arrays, entry["unit"])
            if arrays["tau"].shape != (len(panel), n_mcmc):
                raise ValueError("Chain full surface dimensions disagree")
            for key, values in _surface_heads(panel, arrays["tau"]).items():
                if key not in arrays or not np.allclose(values, arrays[key].reshape(-1), rtol=1e-12, atol=1e-12):
                    raise ValueError("Stored scalar does not match effect surface: " + key)
            current = set(arrays) - {"tau"}
            if keys is not None and current != keys:
                raise ValueError("Chains archive different scalar target sets")
            keys = current
            tau[:, c * n_mcmc:(c + 1) * n_mcmc] = arrays["tau"]
            for key in current:
                scalar.setdefault(key, []).append(arrays[key].reshape(-1).copy())
            if entry["unit"]["trace"]:
                for key, values in _trace_arrays(entry).items():
                    scalar.setdefault(key, []).append(values)
            del arrays
        tau.flush()
        yield panel, tau, scalar, entries
        del tau


def _surface_row_chunked(panel, tau, chunk_rows=256):
    from did_bcf_revision.metrics import surface_summary
    if "CATT" not in panel:
        return None
    rows = panel.index[panel["D"] == 1].to_numpy(dtype=int)
    if not rows.size:
        raise ValueError("Simulation panel has no treated post rows")
    est, lo90, hi90, lo95, hi95 = [np.empty(rows.size) for _ in range(5)]
    for start in range(0, rows.size, chunk_rows):
        end = min(rows.size, start + chunk_rows)
        block = np.asarray(tau[rows[start:end], :])
        est[start:end] = block.mean(axis=1)
        quantiles = np.quantile(block, [.05, .95, .025, .975], axis=1)
        lo90[start:end], hi90[start:end], lo95[start:end], hi95[start:end] = quantiles
    row = {"estimand_type": "CATT", "estimand_id": "surface", "method": "plain",
           "g": np.nan, "t": np.nan, "k": np.nan}
    row.update(surface_summary(panel.loc[rows, "CATT"].to_numpy(dtype=float), est,
                               lo90, hi90, lo95, hi95))
    return row


def scalar_rows_for_job(job, records, budget_key, verify_panel=True):
    """Complete chains only; exact pooled CATT intervals and scalar diagnostics."""
    with _pooled_structured(job, records, budget_key) as pooled:
        if pooled is None:
            return [], []
        panel, tau, arrays, entries = pooled
        truth = {} if job["kind"].startswith("application") else _truth_for(job, panel)
        rows, diagnostics = [], []
        meta = _metadata(job, budget_key)
        for key, values in sorted(arrays.items()):
            if len(values) != len(entries):
                raise ValueError("Incomplete scalar chain set: " + key)
            stacked = np.vstack(values)
            summary = summarise(stacked)
            estimand_type, estimand_id = _parse_key(key)
            if estimand_type is not None:
                row = _per_rep_row(job, key, summary, stacked, budget_key)
                row["true"] = truth.get((estimand_type, estimand_id), np.nan)
                rows.append(row)
            diagnostics.append({**meta, "parameter": key, **summary,
                                "chain_seeds": [e["unit"]["chain_seed"] for e in entries],
                                "chain_set_complete": True})
        surface = _surface_row_chunked(panel, tau)
        if surface is not None:
            rows.append({**surface, **meta})
        return rows, diagnostics


def _pretrend_chain_diagnostics(panel, tau, n_chains, n_mcmc):
    """Per-chain Delta(k) and slope draws, computed from the frozen formula."""
    pre = panel[(panel["eventually_treated"] == 1) & (panel["event_time"] < 0)]
    row_of = {(int(u), int(t)): i for i, (u, t) in
              enumerate(zip(panel["unit_id"], panel["time"]))}
    per_k = {}
    for k in sorted(int(x) for x in pre["event_time"].unique() if int(x) != -1):
        block = pre[pre["event_time"] == k]
        rows, refs = [], []
        for idx, unit_id, cohort in zip(block.index, block["unit_id"],
                                        block["cohort"]):
            ref = row_of.get((int(unit_id), int(cohort) - 1))
            if ref is not None:
                rows.append(idx)
                refs.append(ref)
        if not rows:
            continue
        rows_arr = np.asarray(rows, dtype=int)
        refs_arr = np.asarray(refs, dtype=int)
        per_k[k] = np.stack([
            np.mean(tau[rows_arr, c * n_mcmc:(c + 1) * n_mcmc]
                    - tau[refs_arr, c * n_mcmc:(c + 1) * n_mcmc], axis=0)
            for c in range(n_chains)])
    out = {}
    if per_k:
        ks = sorted(per_k)
        weights = np.asarray([k + 1 for k in ks], dtype=float)
        stack = np.stack([per_k[k] for k in ks])          # (n_k, K, n_mcmc)
        slope = np.tensordot(weights, stack, axes=(0, 0)) / float(weights @ weights)
        for k in ks:
            out[f"PRE_k={k}"] = per_k[k]
        out["PRE_slope"] = slope
    return out


def pretrend_draws(fit, subgroups):
    import did_bcf_revision.pretrend as module
    original = module._summarise
    saved = []
    def capture(values):
        saved.append(np.asarray(values).copy())
        return original(values)
    module._summarise = capture
    try:
        pooled = module.pretrend_estimands(fit, subgroups=subgroups)
    finally:
        module._summarise = original
    scalar_rows = pooled[pooled["post_mean"].notna()]
    if len(scalar_rows) != len(saved):
        raise ValueError("Pretrend summary/draw recording order changed")
    targets = {str(row.estimand_type) + "_" + str(row.estimand_id): values
               for row, values in zip(scalar_rows.itertuples(), saved)}
    return pooled, targets


def pretrend_rows_for_job(job, records, budget_key):
    from did_bcf_revision.pretrend import (pretrend_estimands,
                                           quantile_subgroups, true_pretrend)
    from recovery_core import load_panel as regenerate_panel

    entries = [entry for entry in records.values()
               if entry["unit"]["stat_job_id"] == job["id"]
               and entry["unit"]["budget_key"] == budget_key
               and entry["unit"]["chain_index"] is None]
    if not entries:
        return pd.DataFrame(), []
    entry = entries[0]
    draws = _load_draws(entry)
    tau = draws["tau"]
    panel = _load_panel(entry)
    if panel is None:
        panel = regenerate_panel(job)
    n_chains = entry["unit"]["budget"]["num_chains"]
    n_mcmc = entry["unit"]["budget"]["num_mcmc"]
    if tau.shape[1] != n_chains * n_mcmc:
        raise ValueError(f"tau shape {tau.shape} != {n_chains} x {n_mcmc}")
    empirical = job["kind"] == "application_pretrend"
    subgroups = quantile_subgroups("lpop") if empirical else True
    pooled, target_draws = pretrend_draws(_PretrendShim(panel, tau), subgroups)
    if pooled.empty:
        return pooled, []
    truth = true_pretrend(panel)
    if not truth.empty:
        pooled = pooled.merge(truth, on=["estimand_type", "estimand_id"],
                              how="left")
    else:
        pooled["true"] = np.nan
    pooled["job_id"] = job["id"]
    pooled["budget_key"] = budget_key
    pooled["method"] = "pretrend"
    pooled["dgp"] = job.get("dgp", "mpdta")
    pooled["setting"] = job["setting"]
    pooled["N"] = job.get("N", 500)
    pooled["linearity_degree"] = job.get("linearity_degree", 0)
    pooled["rep"] = job["rep"]
    pooled["spec"] = "pretrend_unit"
    pooled["phase"] = job["phase"]
    pooled["sigma2_shape"] = job["sigma2_shape"]
    pooled["sigma2_rate"] = job["sigma2_rate"]
    diagnostics = []
    for name, value in target_draws.items():
        value = value.reshape(n_chains, n_mcmc)
        diagnostics.append({"job_id": job["id"], "budget_key": budget_key,
                            "parameter": name, **summarise(value)})
    series = {}
    for name in ("global_var_samples", "leaf_scale_mu_samples",
                 "leaf_scale_tau_samples", "rfx_sigma", "tau_0_samples"):
        value = draws.get(name)
        if value is None:
            continue
        flat = np.asarray(value).reshape(-1)
        if flat.size == n_chains * n_mcmc:
            series[name] = flat.reshape(n_chains, n_mcmc)
        else:
            series[name] = flat.reshape(1, -1)
    for name, value in series.items():
        diagnostics.append({"job_id": job["id"], "budget_key": budget_key,
                            "parameter": f"bcf_{name}", **summarise(value)})
    return pooled, diagnostics


def application_outputs(job, records, budget_key):
    from run_empirical_structured import analyse
    from recovery_core import load_panel as regenerate_panel

    entries = [entry for entry in records.values()
               if entry["unit"]["stat_job_id"] == job["id"]
               and entry["unit"]["budget_key"] == budget_key
               and entry["unit"]["chain_index"] is not None]
    if not entries:
        return None
    chains = {entry["unit"]["chain_index"]: entry for entry in entries}
    taus, panel = [], None
    for index in sorted(chains):
        draws = _load_draws(chains[index])
        taus.append(draws["tau"])
        if panel is None:
            panel = _load_panel(chains[index])
    if panel is None:
        panel = regenerate_panel(job)
    gate, catt, contrasts = analyse(_ModelShim(panel, np.hstack(taus)),
                                    job["spec"])
    return {"gate": gate, "catt": catt, "contrasts": contrasts}


def trace_summaries(records):
    """Validate pilot traces against retained draws and summarise drift."""
    rows = []
    for run_id, entry in records.items():
        payload = entry["loader"]("trace.npz")
        if payload is None or entry["unit"]["chain_index"] is None:
            continue
        with np.load(io.BytesIO(payload), allow_pickle=False) as trace:
            scalar_names = [str(x) for x in trace["scalar_names"]]
            group_names = [str(x) for x in trace["group_names"]]
            scalars = trace["scalars"]
            group_means = trace["group_means"]
            stages = trace["stage"]
            sweep = trace["sweep"]
            num_burnin = int(trace["num_burnin"])
            keep_every = int(trace["keep_every"])
            y_std = float(trace["y_std"])
            tau0 = scalars[:, scalar_names.index("tau_0")]
            if "ATT" in group_names:
                att_trace = (group_means[:, group_names.index("ATT")]
                             + tau0) * y_std
            else:
                att_trace = np.full(sweep.size, np.nan)
            sigma2_trace = scalars[:, scalar_names.index("sigma2")]
            retained = ((stages == 2)
                        & ((sweep - num_burnin + 1) % keep_every == 0))
            draws = _load_draws(entry)
            matches = np.nan
            if "ATT" in draws:
                matches = bool(np.allclose(att_trace[retained],
                                           draws["ATT"][0], rtol=0,
                                           atol=1e-12))
            quarter = max(1, sweep.size // 4)
            rows.append({
                "run_id": run_id, "trace_matches_draws": matches,
                "n_sweeps": int(sweep.size),
                "n_retained_positions": int(retained.sum()),
                "att_burnin_first_quarter": float(att_trace[:quarter].mean()),
                "att_burnin_last_quarter": float(
                    att_trace[stages == 1][-quarter:].mean()
                    if (stages == 1).any() else np.nan),
                "att_retained_first_quarter": float(
                    att_trace[retained][:quarter].mean()),
                "att_retained_last_quarter": float(
                    att_trace[retained][-quarter:].mean()),
                "sigma2_burnin_mean": float(
                    sigma2_trace[stages == 1].mean()
                    if (stages == 1).any() else np.nan),
                "sigma2_retained_mean": float(sigma2_trace[stages == 2].mean()),
                "sigma2_first_quarter": float(sigma2_trace[:quarter].mean()),
                "sigma2_last_quarter": float(
                    sigma2_trace[-quarter:].mean()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(body) != manifest.get("manifest_hash"):
        raise SystemExit("Manifest hash mismatch; refusing to collect")
    drift = [rel for rel in manifest["source_hashes"]
             if manifest["source_hashes"][rel] != sha256_file(REPO / rel)]
    if drift:
        raise SystemExit(f"Frozen source drift: {drift}")

    expected_units, records, status = gather(args.incoming, manifest)
    if status["conflicting_units"] or status["duplicate_units"]:
        write_json_atomic(args.out / "status.json", status)
        print(json.dumps(status, indent=2))
        raise SystemExit("Conflicts or duplicate units present; inspect status.json")

    unit_rows = []
    for run_id, entry in sorted(records.items()):
        record, unit = entry["record"], entry["unit"]
        unit_rows.append({
            "run_id": run_id, "stat_job_id": unit["stat_job_id"],
            "recovery_version": record["recovery_version"],
            "phase": unit["job"]["phase"], "kind": unit["job"]["kind"],
            "setting": unit["job"]["setting"], "N": unit["job"].get("N", 500),
            "linearity_degree": unit["job"].get("linearity_degree", 0),
            "rep": unit["job"]["rep"], "spec": unit["job"]["spec"],
            "budget_key": unit["budget_key"],
            "chain_index": unit["chain_index"], "chain_seed": unit["chain_seed"],
            "seconds": record.get("seconds"),
            "peak_rss_mb": record.get("peak_rss_mb"),
            "output_bytes": record.get("output_bytes"),
            "sigma2_shape": unit["job"]["sigma2_shape"], "sigma2_rate": unit["job"]["sigma2_rate"],
            "source": entry["source"],
            "python": (record.get("runtime") or {}).get("python")})
    units = pd.DataFrame(unit_rows)
    units.to_csv(args.out / "collected_runs.csv", index=False)
    if not units.empty and units.seconds.notna().any():
        units[units.seconds.notna()].to_csv(
            INVENTORY / "observed_seconds.csv", index=False)

    budget_keys = manifest.get("budget_keys") or list(manifest["budgets"])
    per_rep, diagnostics, app_gate = [], [], []
    processed = set()
    for job in manifest["jobs"]:
        for budget_key in budget_keys:
            if job["kind"] in ATOMIC_KINDS:
                pooled, diag = pretrend_rows_for_job(job, records, budget_key)
                if not pooled.empty:
                    per_rep.append(pooled)
                    diagnostics.extend(diag)
                    processed.add((job["id"], budget_key))
            else:
                rows, diag = scalar_rows_for_job(job, records, budget_key)
                if rows:
                    per_rep.append(pd.DataFrame(rows))
                    diagnostics.extend(diag)
                    processed.add((job["id"], budget_key))
                if job["kind"] == "application" and rows:
                    app = application_outputs(job, records, budget_key)
                    if app is not None:
                        gate = app["gate"].copy()
                        gate["budget_key"] = budget_key
                        gate["rep"] = job["rep"]
                        app_gate.append(gate)

    frame = (pd.concat(per_rep, ignore_index=True) if per_rep
             else pd.DataFrame())
    if not frame.empty:
        frame.to_csv(args.out / "per_replication.csv", index=False)
    diag = pd.DataFrame(diagnostics)
    if not diag.empty:
        diag.to_csv(args.out / "chain_diagnostics.csv", index=False)
    traces = trace_summaries(records)
    if not traces.empty:
        traces.to_csv(args.out / "trace_diagnostics.csv", index=False)
    if app_gate:
        pd.concat(app_gate, ignore_index=True).to_csv(
            args.out / "application_gate.csv", index=False)

    if not frame.empty:
        from did_bcf_revision.metrics import compute_metrics, surface_metrics
        metric_frames, surface_frames = [], []
        for strata, group in frame.groupby(STRATA, dropna=False):
            scope = dict(zip(STRATA, strata))
            scalar = group[group["true"].notna()] if "true" in group else group.iloc[:0]
            if not scalar.empty:
                metrics = compute_metrics(scalar)
                for key, value in scope.items(): metrics[key] = value
                if "rmse" in metrics:
                    metrics["error_sd_population"] = np.sqrt(np.maximum(0, metrics.rmse**2 - metrics.bias**2))
                    metrics["post_sd_to_error_sd"] = metrics.avg_post_sd / metrics.error_sd_population
                metric_frames.append(metrics)
            surfaces = surface_metrics(group)
            if not surfaces.empty:
                for key, value in scope.items(): surfaces[key] = value
                surface_frames.append(surfaces)
        if metric_frames: pd.concat(metric_frames, ignore_index=True).to_csv(args.out / "scalar_metrics.csv", index=False)
        if surface_frames: pd.concat(surface_frames, ignore_index=True).to_csv(args.out / "surface_metrics.csv", index=False)

    missing = sorted(set(expected_units) - set(records))
    status.update({
        "processed_job_budget_pairs": len(processed),
        "missing_run_ids": missing,
        "analysis_frames": {
            "per_replication_rows": int(len(frame)),
            "chain_diagnostic_rows": int(len(diag)),
            "trace_rows": int(len(traces)),
            "application_gate_rows": int(sum(len(g) for g in app_gate))},
        "note": ("Old analyses remain separate. Assembly used verified chains "
                 "only; no smoke or conflicting artifact entered the "
                 "diagnostics."),
    })
    write_json_atomic(args.out / "status.json", status)
    write_json_atomic(args.out / "integrity_report.json", status)
    print(json.dumps({k: v for k, v in status.items()
                      if k not in ("problems", "duplicates", "smoke")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
