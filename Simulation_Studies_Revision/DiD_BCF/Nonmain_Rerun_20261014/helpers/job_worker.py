"""Sample one complete fit and atomically publish its compact paper summaries."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from common import (PACKAGE, TOOLS, BUDGET_KEY, atomic, bootstrap, digest, read,
                    sha, verify_fit)

def run(job_id, assignment_path, setup, work, out):
    bootstrap(setup)
    from recovery_engine import activate, fingerprint
    from recovery_core import panel_hash, load_panel
    from recovery_common import runtime_versions
    from recovery_plan import make_units
    from recovery_runner import _identity_core, verify_completed
    from paper_metrics import summarize_job
    manifest = read(PACKAGE / "manifest.json")
    assigned = read(assignment_path)
    job = next(j for j in manifest["jobs"] if j["id"] == job_id)
    if job_id not in assigned["job_ids"]:
        raise ValueError("Unassigned fit")
    compact = out / "fits" / job_id
    if verify_fit(compact, manifest):
        return
    engine = fingerprint(activate())
    if engine["specification"] != manifest["engine_specification"]:
        raise ValueError("Repaired engine differs from frozen model")
    units = make_units({**manifest,"jobs":[job]})
    identity_assignment = {
        "parent_manifest":manifest["parent_manifest"],"assignment_hash":assigned["assignment_hash"],
        "recovery_version":manifest["recovery_version"],"purpose":manifest["purpose"],
        "source_hashes_sha256":digest(manifest["source_hashes"]),"dependencies":manifest["dependencies"],
    }
    expected_panel_hash = panel_hash(load_panel(job))
    records = {}
    started = time.monotonic()
    for unit in units:
        home = work / "raw" / job_id / unit["run_id"]
        home.mkdir(parents=True, exist_ok=True)
        attempts = sorted(home.glob("attempt_*"))
        completed = [folder for folder in attempts if verify_completed(folder, unit, identity_assignment)]
        if len(completed) > 1:
            raise ValueError("Duplicate complete chain attempts")
        if completed:
            folder = completed[0]
        else:
            folder = home / f"attempt_{len(attempts):03d}"
            folder.mkdir()
            identity = _identity_core(unit, identity_assignment, len(attempts))
            identity.update(paper_manifest_hash=manifest["manifest_hash"],
                            package_hash=read(PACKAGE / "package.json")["package_hash"])
            request = home / "request.json"
            atomic(request, {"unit":unit,"identity":identity,"workdir":str(folder.resolve())})
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(TOOLS)
            with (home / f"attempt_{len(attempts):03d}.log").open("w") as log:
                subprocess.run([sys.executable,str(TOOLS / "recovery_worker.py"),str(request.resolve())],
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not verify_completed(folder, unit, identity_assignment):
                raise ValueError("Worker did not publish a valid chain")
        record = read(folder / "complete.json")
        if record["panel_hash"] != expected_panel_hash:
            raise ValueError("Wrong regenerated panel")
        def loader(name, source=folder):
            path = source / name
            return path.read_bytes() if path.exists() else None
        records[unit["run_id"]] = {"unit":unit,"record":record,"loader":loader,"source":str(folder)}
    compact.mkdir(parents=True, exist_ok=True)
    summaries = summarize_job(job, BUDGET_KEY, records, compact)
    if not summaries["per_replication_rows"]:
        raise ValueError("No paper summaries from completed fit")
    provenance = {
        "origin":"new_repaired_fit", "panel_hash":expected_panel_hash,
        "package_hash":read(PACKAGE / "package.json")["package_hash"],
        "assignment_hash":assigned["assignment_hash"],"runtime":runtime_versions(),
        "engine":engine,"chains":[records[u["run_id"]]["record"] for u in units],
        "elapsed_seconds":time.monotonic()-started,
        "local_arrays_exported":False,
    }
    atomic(compact / "chain_provenance.json", provenance)
    summaries["outputs"]["chain_provenance.json"] = sha(compact / "chain_provenance.json")
    completion = {"manifest_hash":manifest["manifest_hash"],"job_id":job_id,"job_hash":digest(job),
                  "budget_key":BUDGET_KEY,"chain_seeds":[u["chain_seed"] for u in units],
                  "engine_specification":manifest["engine_specification"],"origin":"new_repaired_fit",
                  **summaries}
    completion["record_hash"] = digest(completion)
    atomic(compact / "complete.json", completion)
    verify_fit(compact, manifest)
    # Successful raw fits are disposable runtime scratch. Nothing outside
    # this job-specific scratch tree is removed, and failed attempts remain.
    import shutil
    shutil.rmtree(work / "raw" / job_id)
    print("Completed compact fit", job_id, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    for name in ("assignment", "setup", "work", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.job,args.assignment,args.setup,args.work.resolve(),args.out.resolve())
