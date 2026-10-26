"""Bounded two-worker Colab execution and metrics-only checkpoints."""
from __future__ import annotations
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from common import (PACKAGE, TOOLS, atomic, assignment, bootstrap, read,
                    restore_metrics, verify_fit, verify_package)

GIB = 2**30
STOP = False
ACTIVE = {}

def available_memory():
    data = {line.split(":",1)[0]:line.split(":",1)[1] for line in Path("/proc/meminfo").read_text().splitlines()}
    available = int(data["MemAvailable"].split()[0]) * 1024
    maximum, used = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if maximum.exists() and used.exists() and maximum.read_text().strip().isdecimal():
        available = min(available, int(maximum.read_text())-int(used.read_text()))
    return max(0, available)

def cpu_allowance():
    count = len(os.sched_getaffinity(0))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        values = quota.read_text().split()
        if values[0] != "max":
            count = min(count, max(1,int(values[0])//int(values[1])))
    return count

def prepare(args):
    _, manifest = verify_package()
    assigned = assignment(args.assignment, manifest)
    actual = {name:importlib.metadata.version(name) for name in manifest["dependencies"]}
    if actual != manifest["dependencies"]:
        raise ValueError("Dependency versions differ from frozen campaign")
    bootstrap()
    if args.resume:
        restore_metrics(args.resume, args.out, manifest, assigned)
    args.work.mkdir(parents=True, exist_ok=True)
    import cmake
    os.environ["PATH"] = str(cmake.CMAKE_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")
    toolchain = args.work.resolve() / "active_python.cmake"
    marker = "=="
    while "]" + marker + "]" in sys.executable:
        marker += "="
    literal = "[" + marker + "[" + sys.executable + "]" + marker + "]"
    toolchain.write_text("".join(f'set({name} {literal} CACHE FILEPATH "Active Python" FORCE)\n'
                         for name in ("Python_EXECUTABLE", "Python3_EXECUTABLE", "PYTHON_EXECUTABLE")))
    os.environ["CMAKE_TOOLCHAIN_FILE"] = str(toolchain)
    command = [sys.executable,str(TOOLS / "recovery_setup.py"),"--work",str(args.work),"--cache",str(args.cache)]
    if args.existing_build:
        command += ["--existing-build",str(args.existing_build)]
    remaining = max(1,args.deadline-time.time()) if args.deadline else 3600
    subprocess.run(command, check=True, timeout=min(3600,remaining))
    bootstrap(args.work / "setup_complete.json")
    from recovery_engine import activate, fingerprint
    engine = fingerprint(activate())
    if engine["specification"] != manifest["engine_specification"]:
        raise ValueError("Native setup differs from frozen repaired engine")
    bundle = args.out / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for source, name in ((args.assignment,"assignment.json"),(PACKAGE / "manifest.json","manifest.json"),
                         (PACKAGE / "package.json","package.json")):
        shutil.copy2(source,bundle / name)
    commit = subprocess.check_output(["git","-C",str(PACKAGE.parents[2]),"rev-parse","HEAD"],text=True).strip()
    atomic(bundle / "setup_provenance.json", {"engine":engine,"dependencies":actual,"source_git_commit":commit,
           "python":sys.version,"global_stochtree_installed":False,
           "toolchain":toolchain.read_text(),"native_controls":"PASS","eligible_feature_controls":"PASS"})
    print("Native engine and source controls passed.", flush=True)

def completed(out, assigned, manifest):
    result = {}
    for job_id in assigned["job_ids"]:
        record = verify_fit(out / "fits" / job_id, manifest)
        if record:
            result[job_id] = record
    return result

def snapshot(out, assigned, manifest, zip_path, status, drive_dir=None):
    complete = completed(out, assigned, manifest)
    state = {**status,"completed":len(complete),"expected":len(assigned["job_ids"]),
             "missing_job_ids":sorted(set(assigned["job_ids"])-set(complete)),
             "assignment_hash":assigned["assignment_hash"],"manifest_hash":manifest["manifest_hash"],
             "snapshot_epoch":time.time()}
    atomic(out / "bundle/status.json", state)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted((out / "bundle").glob("*.json")):
            archive.write(path,str(path.relative_to(out)))
        for job_id, record in sorted(complete.items()):
            folder = out / "fits" / job_id
            for name in [*record["outputs"],"complete.json"]:
                archive.write(folder / name,str((folder / name).relative_to(out)))
    temporary.replace(zip_path)
    if drive_dir:
        drive_dir.mkdir(parents=True, exist_ok=True)
        destination = drive_dir / zip_path.name
        temporary_drive = destination.with_name(destination.name + ".tmp")
        shutil.copy2(zip_path,temporary_drive)
        temporary_drive.replace(destination)
    print(f"Checkpoint: {len(complete)}/{len(assigned['job_ids'])} fits; {zip_path.stat().st_size/2**20:.2f} MiB",flush=True)
    return state

def stop_handler(signum, frame):
    global STOP
    STOP = True

def shutdown():
    for item in ACTIVE.values():
        if item["process"].poll() is None:
            os.killpg(item["process"].pid,signal.SIGTERM)
    for item in ACTIVE.values():
        try:
            item["process"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(item["process"].pid,signal.SIGKILL)
            item["process"].wait()
        item["log"].close()
    ACTIVE.clear()

def run(args):
    global STOP
    _, manifest = verify_package()
    assigned = assignment(args.assignment, manifest)
    bootstrap(args.setup)
    from recovery_engine import activate
    activate()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    ready = completed(args.out, assigned, manifest)
    queue = [j for j in assigned["job_ids"] if j not in ready]
    failures, deferred = {}, []
    concurrency = min(2,args.workers,cpu_allowance())
    if available_memory() < 7 * GIB:
        concurrency = 1
    if available_memory() < 3 * GIB:
        raise RuntimeError("Less than 3 GiB available; free runtime memory before sampling")
    deadline = args.deadline or time.time()+9*3600
    start, next_snapshot = time.time(), time.time()
    calibration = 1.0
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    def state():
        return {"failed":failures,"deferred":deferred,"active":list(ACTIVE),
                "workers":concurrency,"elapsed_hours":(time.time()-start)/3600,
                "deadline_epoch":deadline,"calibration_factor":calibration}
    try:
        while (queue or ACTIVE) and not STOP:
            if time.time() >= deadline-120:
                deferred.extend(queue)
                queue.clear()
                break
            for job_id,item in list(ACTIVE.items()):
                exit_code = item["process"].poll()
                if exit_code is None:
                    continue
                item["log"].close()
                elapsed = time.time()-item["started"]
                record = verify_fit(args.out / "fits" / job_id,manifest)
                if exit_code or not record:
                    failures[job_id] = {"exit_code":exit_code,"elapsed_seconds":elapsed,
                                        "log":str(item["log_path"]),
                                        "error_tail":item["log_path"].read_text(errors="replace")[-12000:],
                                        "rerun":"same notebook with metrics checkpoint"}
                else:
                    estimate = assigned["estimated_fit_seconds"][job_id]
                    calibration = max(calibration,elapsed/max(estimate,1))
                    print(f"Fit {job_id} done in {elapsed/60:.1f} minutes",flush=True)
                del ACTIVE[job_id]
            while queue and len(ACTIVE) < concurrency and available_memory() >= 3*GIB:
                job_id = queue[0]
                predicted = assigned["estimated_fit_seconds"][job_id] * calibration
                if time.time()+max(predicted*1.3,900)+600 >= deadline:
                    deferred.extend(queue)
                    queue.clear()
                    break
                queue.pop(0)
                log_path = args.work / f"job_{job_id}.log"
                log = log_path.open("w")
                environment = os.environ.copy()
                environment["DIDBCF_WORKER_MEMORY_BYTES"] = str(6*GIB)
                command = [sys.executable,str(PACKAGE / "helpers/job_worker.py"),"--job",job_id,
                           "--assignment",str(args.assignment),"--setup",str(args.setup),
                           "--work",str(args.work),"--out",str(args.out)]
                process = subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                ACTIVE[job_id] = {"process":process,"log":log,"log_path":log_path,"started":time.time()}
            if time.time() >= next_snapshot:
                snapshot(args.out,assigned,manifest,args.zip,state(),args.drive_dir)
                next_snapshot = time.time()+1200
            time.sleep(5)
    finally:
        shutdown()
        final = snapshot(args.out,assigned,manifest,args.zip,{**state(),"stopped":STOP},args.drive_dir)
    return 0 if final["completed"] == final["expected"] else 2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    for name in ("assignment","work","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("cache","setup","resume","existing-build","zip","drive-dir"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--deadline",type=float)
    parser.add_argument("--workers",type=int,default=2,choices=(1,2))
    args = parser.parse_args()
    for name in ("assignment","work","out","cache","setup","resume","existing_build","zip","drive_dir"):
        value = getattr(args,name)
        if value:
            setattr(args,name,value.resolve())
    if args.command == "prepare":
        prepare(args)
        return 0
    return run(args)

if __name__ == "__main__":
    sys.exit(main())
