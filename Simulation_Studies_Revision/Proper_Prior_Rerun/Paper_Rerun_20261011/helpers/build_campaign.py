"""Freeze the main paper campaign from the independently verified comparison.

This preparation script performs no sampling. Input is the local intermediate
archive, which need not be published. Generated final compact reuse records
retain original chain identities and archive hashes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import zipfile

from common import PACKAGE, WORKSPACE, BUDGET_KEY, atomic, digest, read, sha

BRIDGE_IDS = ["33f6de1d7aa9626b216a", "5e6cf4b70cbd92bbd07a", "2fb3f12d0a581d8b98e4",
              "3d71a7146144e2479b5c", "f5f99eef5ea464cd63a6", "71da8f440a5f3366927e",
              "d97121845302457a09d4", "51454a5a9d0f152740d3"]

def freeze(old):
    import pandas as pd
    from paper_metrics import _write_csv, _diagnostics_frame
    old = Path(old).resolve()
    analysis = old / "analysis_returned_20261011"
    verification = read(analysis / "ANALYSIS_VERIFICATION.json")
    original = read(old / "frozen_workspace/DiD-BCF/Simulation_Studies_Revision/Proper_Prior_Rerun/manifest.json")
    comparison = read(old / "frozen_workspace/JBES_Submission/performance_comparison_20261009/manifest.json")
    inventory = read(analysis / "next_scope_inventory.json")
    reuse_ids = sorted(inventory["reusable_earlier_job_ids"])
    assert len(reuse_ids) == 264
    assert digest({k:v for k,v in original.items() if k != "manifest_hash"}) == original["manifest_hash"]
    main = [j for j in original["jobs"] if j["phase"] == "main"]
    bridge = [j for j in original["jobs"] if j["id"] in BRIDGE_IDS]
    assert len(main) == 5100 and len(bridge) == 8
    sources = {name:value for name,value in comparison["source_hashes"].items()
               if "/E0_inventory/" not in name}
    for name, checksum in sources.items():
        source = old / "frozen_workspace" / name
        assert sha(source) == checksum, name
        target = WORKSPACE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "schema": "didbcf-paper-metrics-1", "recovery_version": comparison["recovery_version"],
        "purpose": "main_paper_rerun_and_eight_nonmain_budget_checks",
        "parent_manifest": original["manifest_hash"], "method_version": comparison["method_version"],
        "model_spec": comparison["model_spec"], "engine_specification": comparison["engine_specification"],
        "source_hashes": sources, "dependencies": comparison["dependencies"],
        "budgets": {BUDGET_KEY: comparison["budgets"][BUDGET_KEY]}, "budget_keys": [BUDGET_KEY],
        "trace": True, "jobs": main + bridge, "reused_main_job_ids": reuse_ids,
        "bridge_job_ids": BRIDGE_IDS,
        "reuse_source_hashes_sha256": digest(comparison["source_hashes"]),
        "decision_policy": {
            "policy": "user_authorized_performance_focused_budget_selection",
            "evidence": "paired paper metrics on 264 datasets at earlier and B3 budgets",
            "comparison_manifest_hash": comparison["manifest_hash"],
            "strict_full_state_gate_passed": False,
            "note": "This policy does not certify MCMC convergence or nominal frequentist interval coverage. No fit is excluded by diagnostics or performance.",
        },
        "export": {"posterior_arrays": False, "sweep_traces": False, "native_binaries": False,
                   "completion_unit": "complete fit with all three chains; atomic pretrend fit",
                   "files": "CSV paper summaries, scalar diagnostics and JSON provenance",
                   "aggregation": "same frozen pooled-quantile and treated CATT definitions as comparison"},
        "remaining_phases": {k:v for k,v in inventory["original_job_counts"].items() if k != "main"},
    }
    manifest["manifest_hash"] = digest(manifest)
    atomic(PACKAGE / "manifest.json", manifest)
    original_by_id = {j["id"]:j for j in main}
    records = {}
    archive_hashes = {}
    for entry in read(analysis / "archive_selection.json"):
        if not entry["selected"]:
            continue
        archive = old / "Colab" / entry["name"]
        archive_hashes[entry["name"]] = entry["sha256"]
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if name.startswith("runs/") and name.endswith("/complete.json"):
                    import json
                    rec = json.loads(zipped.read(name))
                    if rec["budget_key"] == BUDGET_KEY:
                        records[rec["run_id"]] = {"record":rec, "archive":entry["name"],
                                                  "archive_sha256":entry["sha256"], "member":name}
    for path in (old / "local_completed_outputs/runs").glob("*/attempt_*/complete.json"):
        rec = read(path)
        if rec["budget_key"] == BUDGET_KEY:
            records[rec["run_id"]] = {"record":rec, "local_record_sha256":sha(path),
                                      "member":str(path.relative_to(old))}
    per_path, diag_path = analysis / "collection/per_replication.csv", analysis / "collection/chain_diagnostics.csv"
    per = pd.read_csv(per_path, float_precision="round_trip")
    diag = pd.read_csv(diag_path, float_precision="round_trip")
    runs = pd.read_csv(analysis / "collection/collected_runs.csv", float_precision="round_trip")
    if len(records) != 792:
        raise ValueError("Expected exactly 792 independently verified earlier chains")
    for job_id in reuse_ids:
        job = original_by_id[job_id]
        chain_entries = sorted([e for e in records.values() if e["record"]["stat_job_id"] == job_id],
                               key=lambda e:e["record"]["chain_index"])
        assert [e["record"]["chain_index"] for e in chain_entries] == [0,1,2]
        assert len({e["record"]["panel_hash"] for e in chain_entries}) == 1
        for entry in chain_entries:
            rec = entry["record"]
            assert rec["engine"]["specification"] == manifest["engine_specification"]
            assert rec["budget"] == manifest["budgets"][BUDGET_KEY]
            assert rec["chain_seed"] == 1000 * job["rep"] + rec["chain_index"]
        folder = PACKAGE / "reuse/fits" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        frame = per.loc[(per.job_id == job_id) & (per.budget_key == BUDGET_KEY)].copy()
        diagnostics = _diagnostics_frame(diag.loc[(diag.job_id == job_id) & (diag.budget_key == BUDGET_KEY)])
        diagnostics = diagnostics.drop(columns=[c for c in ("chain_seeds",) if c in diagnostics])
        assert len(frame) and "ATT" in frame.estimand_type.values and "CATT" in frame.estimand_type.values
        _write_csv(frame, folder / "per_replication.csv")
        _write_csv(diagnostics, folder / "diagnostics.csv")
        atomic(folder / "chain_provenance.json", {"origin":"verified_comparison", "chains":chain_entries,
               "input_metrics_sha256":sha(per_path), "input_diagnostics_sha256":sha(diag_path),
               "verification_sha256":sha(analysis / "ANALYSIS_VERIFICATION.json")})
        complete = {
            "manifest_hash": manifest["manifest_hash"], "job_id": job_id, "job_hash":digest(job),
            "budget_key":BUDGET_KEY, "chain_seeds":[1000 * job["rep"] + i for i in range(3)],
            "engine_specification": manifest["engine_specification"], "origin":"reused_verified_earlier_fit",
            "per_replication_rows":len(frame), "diagnostic_rows":len(diagnostics),
            "outputs":{name:sha(folder / name) for name in ("per_replication.csv", "diagnostics.csv", "chain_provenance.json")},
        }
        complete["record_hash"] = digest(complete)
        atomic(folder / "complete.json", complete)
    timing = runs.loc[runs.budget_key == BUDGET_KEY].groupby("stat_job_id").seconds.sum()
    metadata = {j["id"]:j for j in comparison["jobs"]}
    measured = pd.DataFrame([{**metadata[j], "seconds":s} for j,s in timing.items()])
    size_seconds = measured.groupby("N").seconds.quantile(.75).to_dict()
    def estimate(job):
        n = job.get("N", 500)
        if job["kind"] in ("pretrend", "application_pretrend"):
            return 3600.0 if job["kind"] == "pretrend" else 5400.0
        if job["kind"] == "application":
            return 2400.0 if job["spec"] == "structured" else 3600.0
        if n in size_seconds:
            seconds = size_seconds[n]
        else:
            seconds = size_seconds[200] * (n / 200) ** .95
        # Profile interpolation, degree/prior uncertainty and postprocessing.
        return float(seconds * 1.20 * (1.08 if job["setting"] == "B1_strong_confounder" else 1.0))
    remaining = [j for j in main if j["id"] not in reuse_ids]
    assert len(remaining) == 4836
    total = sum(estimate(j) for j in remaining)
    target_load = (8 * 3600 - 1800) * 2  # two workers, 30 minutes setup reserve
    count = math.ceil(total / target_load)
    bins, loads = [[] for _ in range(count)], [0.0] * count
    for job in sorted(remaining, key=lambda j:(-estimate(j),j["id"])):
        index = min(range(count), key=lambda i:(loads[i],len(bins[i]),i))
        bins[index].append(job["id"])
        loads[index] += estimate(job)
    assigned_all = []
    schedule = []
    for index, (ids, load) in enumerate(zip(bins, loads)):
        wave, shard = index // 48 + 1, index % 48
        name = f"main_wave_{wave:02d}_shard_{shard:02d}"
        assigned = {"name":name,"phase":"main","manifest_hash":manifest["manifest_hash"],
                    "job_ids":ids,"estimated_fit_seconds":{j:estimate(original_by_id[j]) for j in ids},
                    "target_hours":8, "nominal_workers":2,"estimated_wall_hours":load / 7200 + .5,
                    "hard_runtime_hours":9.5, "setup_reserve_hours":.5}
        assigned["assignment_hash"] = digest(assigned)
        atomic(PACKAGE / "assignments" / f"{name}.json", assigned)
        assigned_all += ids
        schedule.append({"notebook":name,"phase":"main","fits":len(ids),"wave":wave,
                         "estimated_hours_two_workers":assigned["estimated_wall_hours"]})
    # Four independent checks, two complete jobs apiece, deliberately smaller
    # than the production shards because their budgets are still being checked.
    for index in range(4):
        ids = BRIDGE_IDS[2*index:2*index+2]
        by_id = {j["id"]:j for j in bridge}
        name = f"bridge_checks_{index:02d}"
        estimates = {j:estimate(by_id[j]) for j in ids}
        assigned = {"name":name,"phase":"bridge_checks", "manifest_hash":manifest["manifest_hash"],
                    "job_ids":ids,"estimated_fit_seconds":estimates,"target_hours":8,
                    "nominal_workers":2,"estimated_wall_hours":max(estimates.values()) / 3600 + .5,
                    "hard_runtime_hours":9.5,"setup_reserve_hours":.5}
        assigned["assignment_hash"] = digest(assigned)
        atomic(PACKAGE / "assignments" / f"{name}.json", assigned)
        schedule.append({"notebook":name,"phase":"bridge_checks","fits":2,"wave":0,
                         "estimated_hours_two_workers":assigned["estimated_wall_hours"]})
    assert len(assigned_all) == len(set(assigned_all)) == 4836
    assert set(assigned_all) | set(reuse_ids) == set(original_by_id)
    _write_csv(pd.DataFrame(schedule), PACKAGE / "scheduling.csv")
    atomic(PACKAGE / "preparation.json", {"main_fits":5100,"reused":264,"new_main_fits":4836,
           "main_notebooks":count,"bridge_notebooks":4,"waves":math.ceil(count / 48),
           "measured_full_fit_seconds_q75_by_N":{str(k):v for k,v in size_seconds.items()},
           "total_estimated_worker_hours_including_margin":total/3600,
           "load_min_hours":min(loads)/7200+.5,"load_max_hours":max(loads)/7200+.5,
           "timing_note":"Planning estimates from previous Colab worker elapsed times; hardware and memory variation can increase wall time. One-worker execution takes roughly twice as long.",
           "input_analysis_verification_sha256":sha(analysis / "ANALYSIS_VERIFICATION.json")})
    print(read(PACKAGE / "preparation.json"))

def seal():
    manifest = read(PACKAGE / "manifest.json")
    files = {}
    for base in ("helpers", "assignments", "reuse", "bridge_reference"):
        for path in sorted((PACKAGE / base).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".csv"):
                files[str(path.relative_to(PACKAGE))] = sha(path)
    for name in ("scheduling.csv", "preparation.json"):
        files[name] = sha(PACKAGE / name)
    package = {"schema":"didbcf-paper-package-1","manifest_hash":manifest["manifest_hash"],"files":files}
    package["package_hash"] = digest(package)
    atomic(PACKAGE / "package.json", package)
    print("Sealed", package["package_hash"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-from", type=Path)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.freeze_from:
        freeze(args.freeze_from)
    if args.seal:
        seal()
