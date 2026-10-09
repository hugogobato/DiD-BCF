"""Preserve finished local chains and assign only remaining work to 48 Colabs.

Preparation does not sample. Requires the original stopped study alongside
the nested Git repository. Run with its pinned scientific Python environment.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from portable_runtime import PACKAGE, WORKSPACE, atomic, canonical, sha


def copy_file(source, dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and sha(source) != sha(dest):
        raise ValueError("Refusing to overwrite changed frozen bytes: " + str(dest))
    if not dest.exists():
        shutil.copy2(source,dest)


def main():
    research = PACKAGE.parents[2].parent
    original = research / "JBES_Submission/performance_comparison_20261009"
    local = original / "local_run"
    manifest = json.loads((original / "manifest.json").read_text())
    import hashlib
    if hashlib.sha256(canonical({k:v for k,v in manifest.items() if k!="manifest_hash"}).encode()).hexdigest() != manifest["manifest_hash"]:
        raise ValueError("Original statistical manifest changed")
    with (local / "controller.lock").open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = json.loads((local / "status.json").read_text())
        if stopped["status"] != "STOPPED_FOR_COLAB_TRANSFER" or stopped["active_workers"]:
            raise ValueError("Local run must be stopped before assigning its remaining units")
    setup = json.loads((local / "setup/setup_complete.json").read_text())
    for var,field in (("DIDBCF_REPAIRED_BUILD","build"),("DIDBCF_REPAIRED_CONTROLS","controls"),("DIDBCF_WEIGHT_CONTROLS","weight_controls")):
        os.environ[var] = setup[field]
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MPLCONFIGDIR","/tmp/didbcf-transfer-mpl")
    sys.path.insert(0,str(research / "JBES_Submission/recovery_repaired_20261008/tools"))
    from recovery_common import assignment_hash, digest, verify_assignment
    from recovery_runner import verify_completed
    from recovery_collect import verify_engine_archive
    units, owners = {}, {}
    for path in sorted((original / "assignments").glob("wave_*/shard_*.json")):
        assignment = verify_assignment(json.loads(path.read_text()))
        for unit in assignment["units"]:
            key = unit["run_id"]
            if key in units:
                raise ValueError("Original study assignments overlap")
            units[key],owners[key] = unit,assignment
    if len(units) != 1890:
        raise ValueError("Expected all 1,890 study units")
    archive = PACKAGE / "local_completed_outputs"
    completed = []
    for path in sorted((local / "outputs").glob("runs/*/attempt_*/complete.json")):
        key = path.parent.parent.name
        if key not in units or key in completed:
            raise ValueError("Unexpected or duplicate local completion")
        if not verify_completed(path.parent,units[key],owners[key]):
            raise ValueError("Local completion failed validation")
        for artifact in path.parent.iterdir():
            if artifact.is_file():
                copy_file(artifact,archive / artifact.relative_to(local / "outputs"))
        completed.append(key)
    if len(completed) != 9 or set(completed) != {k for k,v in stopped["units"].items() if v["state"]=="COMPLETE"}:
        raise ValueError("Expected exactly the nine completed local chains")
    # Byte-for-byte original engine, raw arrays and provenance remain results.
    for folder in ("_engine","_bundle"):
        for path in (local / "outputs" / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                copy_file(path,archive / path.relative_to(local / "outputs"))
    copy_file(local / "status.json",archive / "local_stop_status.json")
    verify_engine_archive(archive,manifest)
    frozen = {**manifest["source_hashes"],**manifest["performance_sources"],
              **manifest["comparison"]["reference_sources"],**manifest["comparison"]["reference_subsets"]}
    for rel,expected in frozen.items():
        if sha(research / rel) != expected:
            raise ValueError("Original frozen file changed: " + rel)
        copy_file(research / rel,WORKSPACE / rel)
    copy_file(original / "manifest.json",WORKSPACE / "JBES_Submission/performance_comparison_20261009/manifest.json")
    groups = {}
    remaining = {key:u for key,u in units.items() if key not in completed}
    for unit in remaining.values():
        groups.setdefault(unit["stat_job_id"],[]).append(unit)
    bins = [dict(units=[],jobs=[],seconds=0.) for _ in range(48)]
    for job_id,group in sorted(groups.items(),key=lambda item:(-sum(u["est_seconds"] for u in item[1]),item[0])):
        index = min(range(48),key=lambda i:(bins[i]["seconds"],i))
        bins[index]["units"].extend(sorted(group,key=lambda u:(u["budget_key"],u["chain_index"])))
        bins[index]["jobs"].append(job_id)
        bins[index]["seconds"] += sum(u["est_seconds"] for u in group)
    assert all(b["units"] for b in bins)
    assert len(groups)==270 and sum(len(b["jobs"]) for b in bins)==270
    scheduled = [u["run_id"] for b in bins for u in b["units"]]
    assert len(scheduled)==1881 and set(scheduled)==set(remaining) and len(set(scheduled))==1881
    paths, rows, summary = {}, [], []
    template = next(iter(owners.values()))
    for index,bin in enumerate(bins):
        assignment = {k:v for k,v in template.items() if k not in ("units","assignment_hash","estimated_seconds","measured_output_bytes_sum")}
        assignment.update(purpose="replicated_performance_colab_transfer",wave=0,wave_count=1,shard=index,shard_count=48,
                          estimated_seconds=bin["seconds"],units=bin["units"],
                          measured_output_bytes_sum=sum(u.get("measured_output_bytes") or 0 for u in bin["units"]),
                          resource_estimate_note="Same timing proxies as original study; 48 balanced dataset bundles, at most two memory-bounded single-thread sampling workers")
        assignment["assignment_hash"] = assignment_hash(assignment)
        verify_assignment(assignment)
        rel = f"assignments/wave_00/shard_{index:02d}.json"
        atomic(PACKAGE / rel,assignment)
        paths[rel] = assignment["assignment_hash"]
        summary.append(dict(shard=index,notebook=f"Colab/remaining_{index:02d}.ipynb",datasets=len(bin["jobs"]),
                            remaining_chains=len(bin["units"]),estimated_cpu_hours=bin["seconds"]/3600,
                            ideal_hours_at_two_workers=bin["seconds"]/7200,estimated_output_MiB=assignment["measured_output_bytes_sum"]/2**20))
        for u in bin["units"]:
            rows.append(dict(shard=index,run_id=u["run_id"],stat_job_id=u["stat_job_id"],setting=u["job"]["setting"],
                             N=u["job"]["N"],degree=u["job"]["linearity_degree"],rep=u["job"]["rep"],
                             budget=u["budget_key"],chain=u["chain_index"],seed=u["chain_seed"],estimated_seconds=u["est_seconds"]))
    for name,data in (("scheduling.csv",rows),("shard_summary.csv",summary)):
        with (PACKAGE / name).open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    transfer = dict(version="1.0-colab-transfer-20261009",statistical_manifest_hash=manifest["manifest_hash"],
                    original_expected_units=1890,local_completed_run_ids=sorted(completed),remaining_run_ids=sorted(remaining),
                    assignments=paths,workers_per_notebook_max=2,notebook_count=48,session_hours_including_setup=9.5,
                    sampling_export_reserve_minutes=45,
                    helper_hashes={str(p.relative_to(PACKAGE)):sha(p) for p in sorted((PACKAGE / "helpers").glob("*.py"))},
                    local_completed_files={str(p.relative_to(PACKAGE)):sha(p) for p in sorted(archive.rglob("*")) if p.is_file()},
                    statistical_changes="none: preserve original model, proper prior, both budgets, trace setting, jobs, run IDs and chain seeds",
                    scheduling_changes="only remaining units; 48 balanced shards, keeping each dataset's remaining arms and chains in the same shard",
                    reference_environment_note="Saved unpatched proper-prior fits used Python 3.13.15; local repaired fits used Python 3.12.3. All runs record platform, Python, dependencies and verified engine. Cross-platform paths need not be identical.",
                    production_release=False)
    transfer["transfer_hash"] = digest(transfer)
    atomic(PACKAGE / "transfer.json",transfer)
    report = dict(status="PASS",local_run_stopped=True,controller_lock_released=True,local_verified_units=len(completed),
                  remaining_units=len(remaining),notebooks=len(bins),unique_dataset_groups=len(groups),
                  preserved_frozen_files=len(frozen),unchanged_statistical_manifest=manifest["manifest_hash"],
                  no_new_sampling=True,notebook_estimated_cpu_hours_min=min(b["seconds"] for b in bins)/3600,
                  notebook_estimated_cpu_hours_max=max(b["seconds"] for b in bins)/3600,
                  total_remaining_estimated_cpu_hours=sum(b["seconds"] for b in bins)/3600,
                  transfer_hash=transfer["transfer_hash"])
    atomic(PACKAGE / "validation/preparation.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
