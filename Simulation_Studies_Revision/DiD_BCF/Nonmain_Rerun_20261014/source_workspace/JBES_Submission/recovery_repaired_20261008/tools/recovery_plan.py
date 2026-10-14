#!/usr/bin/env python3
"""E0/E1/E2/E3 planning: manifests, pilot selection, assignments, production.

Subcommands
-----------
``pilot``       build the prespecified diagnostic-pilot manifest (no fits);
``missing``     build the immutable missing-phase manifest (no budget assigned);
``template``    build the accepted-budget production manifest template;
``assign``      shard a prepared manifest's run units into notebook assignments;
``production``  freeze an accepted budget into a new production manifest and
                assign it (run only after the E2 gate decision).

Nothing here samples.  Duration estimates are scheduling inputs, never
scientific evidence; they are regenerated from measured runs when available.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from collections import Counter
from pathlib import Path

import pandas as pd

from recovery_common import (ACCEPTANCE, ASSIGNMENTS, BUDGETS, DEPENDENCIES,
                             GATED_BUDGET_KEYS, INVENTORY, MANIFESTS,
                             ORIGINAL_MANIFEST_HASH, ORIGINAL_SAMPLER,
                             PARENT_MANIFEST, PILOT, PILOT_BUDGET_KEYS,
                             RECOVERY_VERSION, REPO, assignment_hash,
                             bundle_hashes, digest, environment, model_spec,
                             run_id_for, sha256_file, sweeps_per_chain,
                             write_json_atomic)
from recovery_inventory import load_original_manifest

ATOMIC_KINDS = ("pretrend", "application_pretrend")
EFFECT_KINDS = ("effect", "application")
KIND_FACTORS = {"effect": 1.0, "application": 4.0, "pretrend": 4.0,
                "application_pretrend": 8.0}
DEFAULT_OLD_FIT_SECONDS = 400.0

OUTPUT_SCHEMA = {
    "structured_chain": {
        "paths": ["started.json", "draws.npz", "trace.npz (pilot only)",
                  "panel.csv (application only)", "complete.json"],
        "scalar_draw_keys": ["ATT", "sigma2", "GATT_g=<g>_t=<t>", "ES_k=<k>"],
        "scalar_shapes": {"per chain": [1, "num_mcmc"]},
        "tau_key": "every structured job: full (n_obs, num_mcmc) effect surface",
        "completion": "complete.json published last, with output checksums",
    },
    "pretrend_atomic": {
        "paths": ["started.json", "draws.npz", "panel.csv", "complete.json"],
        "keys": ["tau", "global_var_samples", "leaf_scale_mu_samples",
                 "tau_0_samples when exposed by stochtree.BCFModel"],
        "tau_shape": ["n_obs", "num_chains * num_mcmc"],
        "note": ("atomic recovery unit: BCFModel per-chain state is not "
                 "serialised, so an interrupted job restarts from its seed and "
                 "is never appended to"),
    },
    "assembly": ("the collector concatenates chains in chain_index order, "
                 "computes multi-chain diagnostics on the assembled draws and "
                 "recomputes finite-sample truths from the regenerated panel"),
}


# --------------------------------------------------------------------------- #
# Selection of the prespecified pilot
# --------------------------------------------------------------------------- #
ANCHORS = {
    "ae95bf48eea71d8a38f2": "anchor: ordinary baseline (N=200, degree 1, rep 0)",
    "6f6f4cde84c83516c33c": "anchor: difficult ATT (serial AR(1), N=100, degree 3, rep 9)",
    "bd2541dd8b654930a2fe": "anchor: large-N variance (serial AR(1), N=800, degree 2, rep 82)",
}
STRATA_RULES = [
    dict(label="large N", setting="B2_sweep", N=800, linearity_degree=1, rep=0,
         sigma2_rate=1.0, kind="effect"),
    dict(label="serial correlation", setting="B1_serial_corr", N=200,
         linearity_degree=1, rep=0, sigma2_rate=1.0, kind="effect"),
    dict(label="observable selection", setting="B1_selection_obs", N=200,
         linearity_degree=1, rep=0, sigma2_rate=1.0, kind="effect"),
    dict(label="sharp null", setting="B1_null", N=200, linearity_degree=1,
         rep=0, sigma2_rate=1.0, kind="effect"),
    dict(label="staggered adoption", setting="D_staggered", N=200,
         linearity_degree=1, rep=0, sigma2_rate=1.0, kind="effect"),
    dict(label="degree 3 baseline", setting="B1_baseline", N=200,
         linearity_degree=3, rep=0, sigma2_rate=1.0, kind="effect"),
    dict(label="pretrend unit-effects diagnostic", setting="PT_hold", N=200,
         linearity_degree=1, rep=0, kind="pretrend", phase="diagnostic",
         sigma2_rate=1.0),
    dict(label="application raw/no-unit-effects", setting="mpdta", rep=0,
         kind="application", spec="structured", sigma2_rate=1.0),
    dict(label="application raw/unit-effects", setting="mpdta", rep=0,
         kind="application", spec="structured_rfx_unit", sigma2_rate=1.0),
]
STRATA_RULES.extend([
    dict(label="application pretrend unit-effects", setting="mpdta", rep=0, kind="application_pretrend", spec="pretrend_unit", sigma2_rate=1.),
    *[dict(label=f"simulation prior-rate {rate}", phase="sensitivity", setting="B2_sweep", N=50, linearity_degree=1, rep=0, kind="effect", sigma2_rate=rate) for rate in (.1,10.)],
    *[dict(label=f"application unit prior-rate {rate}", phase="application_sensitivity", setting="mpdta", rep=0, kind="application", spec="structured_rfx_unit", sigma2_rate=rate) for rate in (.1,10.)],
])
LOCAL_RECOVERY_UNIT = dict(
    stat_job_id="bd2541dd8b654930a2fe",
    budget_key="B1_3chain_local_recovery",
    chain_index=2,
    purpose=("recover the missing third chain (seed 82002) of the local N=800 "
             "retry in a new attempt namespace; the local chain_0/chain_1 files "
             "are validated, never appended to by filename"))


def measured_table() -> pd.DataFrame:
    path = INVENTORY / "job_seconds.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["job_id", "setting", "N", "linearity_degree",
                                 "rep", "seconds"])


def reference_seconds(job: dict, table: pd.DataFrame) -> float:
    """Conservative full-job (three-chain, old budget) time estimate."""
    if len(table):
        exact = table[table.job_id == job["id"]]
        if len(exact):
            return float(exact.seconds.iloc[0])
        n_units = job.get("N")
        degree = job.get("linearity_degree")
        if n_units is not None and degree is not None:
            cell = table[(table.setting == job["setting"])
                         & (table.N == n_units)
                         & (table.linearity_degree == degree)]
            if len(cell):
                return float(cell.seconds.median())
            size = table[(table.N == n_units)
                         & (table.linearity_degree == degree)]
            if len(size):
                return float(size.seconds.median())
        return float(table.seconds.median())
    return DEFAULT_OLD_FIT_SECONDS


def find_job(jobs: list, **criteria):
    matches = [job for job in jobs
               if all(job.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one job for {criteria}, found {len(matches)}")
    return matches[0]


def pilot_selection(manifest: dict) -> list:
    jobs = {job["id"]: job for job in manifest["jobs"]}
    selected = []
    for jid, note in ANCHORS.items():
        job = jobs.get(jid)
        if job is None:
            raise ValueError(f"Anchor {jid} absent from the original manifest")
        selected.append((job, note))
    for rule in STRATA_RULES:
        criteria = {key: value for key, value in rule.items()
                    if key != "label"}
        job = find_job(manifest["jobs"], **criteria)
        selected.append((job, f"stratum: {rule['label']} "
                              "(explicit seed rule rep=0)"))
    ids = [job["id"] for job, _ in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("Pilot selection contains duplicate statistical jobs")
    return selected


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #
def _manifest_body(purpose: str, jobs: list, budgets: dict, budget_keys,
                   status: str, trace: bool, notes: list) -> dict:
    body = {
        "recovery_version": RECOVERY_VERSION,
        "status": status,
        "purpose": purpose,
        "parent_manifest": ORIGINAL_MANIFEST_HASH,
        "method_version": "0.2.0-proper-ig",
        "model_spec": model_spec(),
        "budgets": budgets,
        "budget_keys": budget_keys,
        "trace": trace,
        "engine_specification": __import__("recovery_engine").specification(),
        "source_hashes": bundle_hashes(),
        "dependencies": __import__("recovery_common").pin_dependencies(),
        "python_recorded": platform.python_version(),
        "output_schema": OUTPUT_SCHEMA,
        "acceptance_targets": ACCEPTANCE,
        "notes": notes,
        "jobs": jobs,
    }
    return body


def build_pilot() -> dict:
    original = load_original_manifest()
    table = measured_table()
    selected = pilot_selection(original)

    def annotate(job, note):
        return {**job, "recovery": {
            "budget_keys": list(PILOT_BUDGET_KEYS),
            "trace": True,
            "reference_old_fit_seconds": reference_seconds(job, table),
            "selection": note,
            "diagnostic_only": True,
        }}

    jobs = [annotate(job, note) for job, note in selected]
    budgets = {key: BUDGETS[key] for key in PILOT_BUDGET_KEYS}

    body = _manifest_body(
        "diagnostic_pilot", jobs, budgets, list(PILOT_BUDGET_KEYS),
        "prepared_not_run", True,
        ["Diagnostic sample, not representative performance evidence.",
         "B1/B2 vary warmup with retained draws fixed; B1/B3 vary retained "
         "draws with warmup fixed; B4 is gated until B1/B2/B3 are reviewed.",
         "Historical unpatched partial chains remain separate; every repaired pilot chain starts from its recorded seed.",
         "Prior, forest architecture, features, DGP and seeds are frozen; only "
         "the repaired engine, sampling budget and instrumentation differ from the archive."])
    body["special_units"] = []
    body["manifest_hash"] = digest(body)
    write_json_atomic(MANIFESTS / "manifest_pilot.json", body)
    frame = pd.DataFrame([{**{k: job.get(k) for k in
                              ("id", "kind", "phase", "setting", "rep", "spec",
                               "N", "linearity_degree", "sigma2_rate")},
                           "reference_old_fit_seconds": job["recovery"][
                               "reference_old_fit_seconds"],
                           "selection": job["recovery"]["selection"]}
                          for job in jobs])
    frame.to_csv(MANIFESTS / "pilot_selection.csv", index=False)
    return body


def build_missing() -> dict:
    original = load_original_manifest()
    table = measured_table()
    missing = list(original["jobs"])
    jobs = []
    for job in missing:
        jobs.append({**job, "recovery": {
            "status": "awaiting_E2_budget",
            "reference_old_fit_seconds": reference_seconds(job, table),
            "selection": "repaired replacement main" if job["phase"] == "main" else "missing proper-prior phase",
            "diagnostic_only": False,
        }})
    body = _manifest_body(
        "missing_phases", jobs, {}, [], "frozen_job_universe_pending_E2",
        False,
        ["Immutable repaired job universe including replacement main, diagnostic, pt_impact, application, "
         "sensitivity and application_sensitivity.",
         "No sampler budget is assigned until the E2 gate decision; the "
         "questionable historical budget must not schedule these phases.",
         "Duration estimates are unmeasured proxies from matched main cells "
         "and are scheduling inputs only."])
    body["phase_counts"] = dict(Counter(job["phase"] for job in jobs))
    body["manifest_hash"] = digest(body)
    write_json_atomic(MANIFESTS / "manifest_missing_phases.json", body)
    return body


def build_template(pilot_manifest: dict, missing_manifest: dict) -> dict:
    body = {
        "recovery_version": RECOVERY_VERSION,
        "status": "TEMPLATE_NOT_SCHEDULED",
        "purpose": "accepted_budget_production",
        "parent_manifest": ORIGINAL_MANIFEST_HASH,
        "missing_phase_manifest_hash": missing_manifest["manifest_hash"],
        "pilot_manifest_hash": pilot_manifest["manifest_hash"],
        "model_spec": model_spec(),
        "required_fill_in": {
            "budget_key": "selected from the pilot ladder by the frozen "
                          "acceptance rules, never by treatment truth, coverage "
                          "or favourable point estimates",
            "sampler_budget": "(num_gfr, num_burnin, num_mcmc, keep_every, "
                              "num_chains) accepted at the E2 gate",
            "accepted_source_hashes": "hashes of the exact source run in the "
                                      "pilot, re-frozen after acceptance",
            "accepted_dependencies": "pinned dependency versions recorded by "
                                     "the accepted pilot notebooks",
            "python_recorded": "actual Colab interpreter version",
            "gate_evidence": "path/hash of the documented gate decision",
        },
        "candidate_budgets": {key: BUDGETS[key] for key in
                              (*PILOT_BUDGET_KEYS, *GATED_BUDGET_KEYS)},
        "output_schema": OUTPUT_SCHEMA,
        "acceptance_targets": ACCEPTANCE,
    }
    body["manifest_hash"] = None
    write_json_atomic(MANIFESTS / "manifest_production_template.json", body)
    return body


# --------------------------------------------------------------------------- #
# Unit construction and sharding
# --------------------------------------------------------------------------- #
def _observed_overrides() -> dict:
    path = INVENTORY / "observed_seconds.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "recovery_version" not in frame:
        return {}
    frame = frame[frame.recovery_version == RECOVERY_VERSION]
    return {"by_run": {row.run_id: float(row.seconds) for row in frame.itertuples()}, "rows": frame.to_dict("records")}


def observed_matches(job, bdg, observed):
    # Match sampled strata and the accepted budget, rather than run IDs (which
    # deliberately differ between instrumented pilot and production).
    keys = ("kind", "N", "linearity_degree", "spec", "sigma2_shape", "sigma2_rate")
    candidates = []
    for row in observed.get("rows", []):
        if row.get("budget_key") not in BUDGETS or BUDGETS[row["budget_key"]] != bdg:
            continue
        if all(row.get(k) == job.get(k, 500 if k == "N" else 0 if k == "linearity_degree" else None) for k in keys):
            candidates.append(row)
    return candidates


def estimate_unit_seconds(job: dict, bdg: dict, atomic: bool,
                          observed: dict, unit_run_id: str) -> float:
    if unit_run_id in observed.get("by_run", {}):
        return observed["by_run"][unit_run_id]
    matches = observed_matches(job, bdg, observed)
    if matches:
        return 1.5 * max(float(row["seconds"]) for row in matches)
    reference = float(job.get("recovery", {}).get(
        "reference_old_fit_seconds", DEFAULT_OLD_FIT_SECONDS))
    ratio = sweeps_per_chain(bdg) / sweeps_per_chain(ORIGINAL_SAMPLER)
    factor = KIND_FACTORS.get(job["kind"], 1.0)
    if atomic:
        return reference * ratio * factor * bdg["num_chains"] / ORIGINAL_SAMPLER["num_chains"]
    return reference / ORIGINAL_SAMPLER["num_chains"] * ratio * factor


def make_units(manifest: dict, trace_override=None,
               budget_keys=None) -> list:
    units = []
    observed = _observed_overrides()
    keys = budget_keys or manifest.get("budget_keys", [])
    for job in manifest["jobs"]:
        job_keys = keys or job.get("recovery", {}).get("budget_keys", [])
        for key in job_keys:
            bdg = manifest["budgets"][key]
            atomic = job["kind"] in ATOMIC_KINDS
            trace = manifest["trace"] if trace_override is None else trace_override
            trace = bool(trace) and not atomic
            chains = [None] if atomic else list(range(bdg["num_chains"]))
            for chain in chains:
                seed = None if chain is None else int(job["rep"]) * 1000 + chain
                run_id = run_id_for(job["id"], bdg, chain, trace,
                                    manifest["manifest_hash"])
                units.append({
                    "run_id": run_id,
                    "stat_job_id": job["id"],
                    "budget_key": key,
                    "budget": bdg,
                    "chain_index": chain,
                    "chain_seed": seed,
                    "trace": bool(trace),
                    "est_seconds": estimate_unit_seconds(
                        job, bdg, atomic, observed, run_id),
                    "job": job,
                    "measured_peak_rss_mb": max((float(r["peak_rss_mb"]) for r in observed_matches(job, bdg, observed)), default=None),
                    "measured_output_bytes": max((int(r["output_bytes"]) for r in observed_matches(job, bdg, observed)), default=None),
                })
    for special in manifest.get("special_units", []):
        job = next(j for j in manifest["jobs"]
                   if j["id"] == special["stat_job_id"])
        bdg = manifest["budgets"][special["budget_key"]]
        chain = special["chain_index"]
        seed = int(job["rep"]) * 1000 + chain
        trace = True
        run_id = run_id_for(job["id"], bdg, chain, trace,
                            manifest["manifest_hash"])
        units.append({
            "run_id": run_id,
            "stat_job_id": job["id"],
            "budget_key": special["budget_key"],
            "budget": bdg,
            "chain_index": chain,
            "chain_seed": seed,
            "trace": trace,
            "est_seconds": estimate_unit_seconds(
                job, bdg, False, observed, run_id),
            "job": job,
            "purpose": special["purpose"],
        })
    if len({unit["run_id"] for unit in units}) != len(units):
        raise ValueError("Duplicate run IDs in unit construction")
    return units


def shard_units(units: list, target_hours: float) -> list:
    """Greedy longest-processing-time bin packing into notebook-sized shards."""
    capacity = target_hours * 3600
    ordered = sorted(units, key=lambda unit: -unit["est_seconds"])
    if capacity <= 0:
        raise ValueError("Shard capacity must be positive")
    count = math.ceil(sum(u["est_seconds"] for u in ordered) / capacity)
    bins: list = [{"load": 0.0, "units": []} for _ in range(count)]
    for unit in ordered:
        fits = [i for i in range(len(bins)) if bins[i]["load"] + unit["est_seconds"] <= capacity]
        if fits:
            index = min(fits, key=lambda i: bins[i]["load"])
        else:
            bins.append({"load": 0.0, "units": []})
            index = len(bins) - 1
        bins[index]["units"].append(unit)
        bins[index]["load"] += unit["est_seconds"]
    return [{"load": bucket["load"], "units": bucket["units"],
             "shard": number} for number, bucket in enumerate(b for b in bins if b["units"])]


PHASE_ORDER = ("application", "application_sensitivity", "diagnostic",
               "pt_impact", "sensitivity", "main")


def write_assignments(manifest: dict, purpose: str, target_hours: float,
                      wave_size: int, out_dir: Path,
                      phase_order=None) -> list:
    units = make_units(manifest)
    if phase_order:
        grouped = []
        for phase in phase_order:
            grouped.extend(shard_units(
                [u for u in units if u["job"]["phase"] == phase],
                target_hours))
        leftover = [u for u in units if u["job"]["phase"] not in phase_order]
        grouped.extend(shard_units(leftover, target_hours))
        shards = grouped
    else:
        shards = shard_units(units, target_hours)
    wave_count = math.ceil(len(shards) / wave_size) if shards else 0
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = []
    written = []
    for global_shard, shard in enumerate(shards):
        shard["shard"] = global_shard
        wave = shard["shard"] // wave_size
        shard_in_wave = shard["shard"] % wave_size
        body = {
            "recovery_version": RECOVERY_VERSION,
            "purpose": purpose,
            "parent_manifest": manifest.get("parent_manifest",
                                            ORIGINAL_MANIFEST_HASH),
            "manifest_hash": manifest["manifest_hash"],
            "model_spec": manifest["model_spec"],
            "budgets": manifest["budgets"],
            "dependencies": manifest["dependencies"],
            "engine_specification": manifest["engine_specification"],
            "python_recorded": manifest.get("python_recorded"),
            "required_python": manifest.get("accepted_python"),
            "source_hashes": manifest["source_hashes"],
            "source_hashes_sha256": digest(manifest["source_hashes"]),
            "output_schema": manifest.get("output_schema"),
            "wave": wave,
            "wave_count": wave_count,
            "shard": shard_in_wave,
            "shard_count": min(wave_size, len(shards) - wave * wave_size),
            "estimated_seconds": shard["load"],
            "measured_output_bytes_sum": sum(u.get("measured_output_bytes") or 0 for u in shard["units"]),
            "resource_estimate_note": "Pilot measurements when strata match; historical timing proxies elsewhere. Inspect free disk before each wave.",
            "units": shard["units"],
        }
        body["assignment_hash"] = assignment_hash(body)
        folder = out_dir / f"wave_{wave:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"shard_{shard_in_wave:02d}.json"
        write_json_atomic(path, body)
        try:
            display_path = str(path.relative_to(REPO))
        except ValueError:
            display_path = str(path)
        written.append({"wave": wave, "shard": shard_in_wave,
                        "units": len(shard["units"]),
                        "estimated_hours": round(shard["load"] / 3600, 3),
                        "path": display_path,
                        "assignment_hash": body["assignment_hash"]})
        for unit in shard["units"]:
            schedule.append({
                "wave": wave, "shard": shard_in_wave,
                "notebook": f"didbcf_recovery_{purpose}_w{wave:02d}_s{shard_in_wave:02d}.ipynb",
                "run_id": unit["run_id"], "stat_job_id": unit["stat_job_id"],
                "phase": unit["job"]["phase"], "kind": unit["job"]["kind"],
                "setting": unit["job"]["setting"],
                "N": unit["job"].get("N"),
                "linearity_degree": unit["job"].get("linearity_degree"),
                "rep": unit["job"]["rep"], "spec": unit["job"]["spec"],
                "budget_key": unit["budget_key"],
                "chain_index": unit["chain_index"],
                "chain_seed": unit["chain_seed"],
                "trace": unit["trace"],
                "est_seconds": unit["est_seconds"],
            })
    actual = [json.loads(path.read_text()) for path in out_dir.glob("wave_*/shard_*.json")]
    ids = [u["run_id"] for body in actual for u in body["units"]]
    if any(body["manifest_hash"] != manifest["manifest_hash"] for body in actual) or len(ids) != len(set(ids)) or set(ids) != {u["run_id"] for u in units}:
        raise ValueError("Stale, duplicate, or omitted assignment units; use a clean versioned assignment directory")
    frame = pd.DataFrame(schedule)
    frame.to_csv(out_dir / "scheduling.csv", index=False)
    summary = pd.DataFrame(written)
    summary.to_csv(out_dir / "shard_summary.csv", index=False)
    return written


def build_production(budget_key: str, accepted_python: str,
                     accepted_dependencies: dict, gate_evidence: str) -> dict:
    missing = json.loads((MANIFESTS / "manifest_missing_phases.json").read_text())
    if budget_key not in BUDGETS:
        raise KeyError(f"Unknown budget key {budget_key}")
    from recovery_accept import validate_decision
    decision = validate_decision(Path(gate_evidence), budget_key, accepted_python)
    jobs = [job for job in missing["jobs"]]
    body = _manifest_body(
        "production_repaired_all_phases", jobs, {budget_key: BUDGETS[budget_key]},
        [budget_key], "frozen_accepted_budget_pending_user_run", False,
        ["Accepted budget frozen after the E2 gate decision.",
         f"Gate evidence: {gate_evidence}",
         "Same source hashes and model specification as the accepted pilot; "
         "only the sampling budget and run namespace change.",
         "Run the application batch first; waves are ordered accordingly."])
    body["gate_decision"] = decision
    body["accepted_python"] = accepted_python
    body["accepted_dependencies"] = accepted_dependencies
    body["missing_phase_manifest_hash"] = missing["manifest_hash"]
    body["manifest_hash"] = digest(body)
    write_json_atomic(MANIFESTS / "manifest_production.json", body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pilot")
    sub.add_parser("missing")
    sub.add_parser("template")
    assign = sub.add_parser("assign")
    assign.add_argument("--manifest", type=Path, required=True)
    assign.add_argument("--out", type=Path, default=ASSIGNMENTS)
    assign.add_argument("--purpose", default=None)
    assign.add_argument("--target-hours", type=float, default=6.5)
    assign.add_argument("--wave-size", type=int, default=48)
    production = sub.add_parser("production")
    production.add_argument("--budget", required=True)
    production.add_argument("--accepted-python", required=True)
    production.add_argument("--gate-evidence", required=True)
    production.add_argument("--out", type=Path, default=ASSIGNMENTS / "production")
    production.add_argument("--target-hours", type=float, default=7.0)
    production.add_argument("--wave-size", type=int, default=48)
    args = parser.parse_args()

    if args.command == "pilot":
        body = build_pilot()
        print(json.dumps({"manifest_hash": body["manifest_hash"],
                          "jobs": len(body["jobs"]),
                          "budget_keys": body["budget_keys"]}, indent=2))
        return 0
    if args.command == "missing":
        body = build_missing()
        print(json.dumps({"manifest_hash": body["manifest_hash"],
                          "jobs": len(body["jobs"]),
                          "phases": body["phase_counts"]}, indent=2))
        return 0
    if args.command == "template":
        pilot = json.loads((MANIFESTS / "manifest_pilot.json").read_text())
        missing = json.loads(
            (MANIFESTS / "manifest_missing_phases.json").read_text())
        build_template(pilot, missing)
        print("Wrote manifest_production_template.json")
        return 0
    if args.command == "assign":
        manifest = json.loads(args.manifest.read_text())
        purpose = args.purpose or manifest["purpose"]
        written = write_assignments(manifest, purpose, args.target_hours,
                                    args.wave_size, args.out)
        print(json.dumps(written, indent=2))
        return 0
    if args.command == "production":
        accepted = __import__("recovery_common").pin_dependencies()
        body = build_production(args.budget, args.accepted_python,
                                accepted, args.gate_evidence)
        written = write_assignments(body, "production", args.target_hours,
                                    args.wave_size, args.out,
                                    phase_order=PHASE_ORDER)
        print(json.dumps({"manifest_hash": body["manifest_hash"],
                          "shards": len(written),
                          "wave_count": max(w["wave"] for w in written) + 1
                          if written else 0}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
