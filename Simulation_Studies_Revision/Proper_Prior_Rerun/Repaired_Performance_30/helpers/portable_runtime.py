"""Pinned Git-clone execution of the remaining paired performance chains.

The statistical worker is copied byte-for-byte from the validated source
packet. This adapter changes locations, sharding, concurrency and recovery only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE / "frozen_workspace"
PERFORMANCE = WORKSPACE / "JBES_Submission/performance_comparison_20261009"
TOOLS = WORKSPACE / "JBES_Submission/recovery_repaired_20261008/tools"
GIB = 2**30
STOP = False
ACTIVE = {}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_package():
    transfer = json.loads((PACKAGE / "transfer.json").read_text())
    if hashlib.sha256(canonical({k:v for k,v in transfer.items() if k != "transfer_hash"}).encode()).hexdigest() != transfer["transfer_hash"]:
        raise ValueError("Transfer manifest changed")
    manifest = json.loads((PERFORMANCE / "manifest.json").read_text())
    if hashlib.sha256(canonical({k:v for k,v in manifest.items() if k != "manifest_hash"}).encode()).hexdigest() != manifest["manifest_hash"]:
        raise ValueError("Statistical manifest changed")
    if manifest["manifest_hash"] != transfer["statistical_manifest_hash"]:
        raise ValueError("Transfer belongs to a different study")
    for name, expected in transfer["helper_hashes"].items():
        if sha(PACKAGE / name) != expected:
            raise ValueError("Portable helper changed: " + name)
    for name, expected in {**manifest["source_hashes"], **manifest["performance_sources"],
                           **manifest["comparison"]["reference_sources"], **manifest["comparison"]["reference_subsets"]}.items():
        if sha(WORKSPACE / name) != expected:
            raise ValueError("Frozen model/report/reference source changed: " + name)
    for name, expected in transfer["local_completed_files"].items():
        if sha(PACKAGE / name) != expected:
            raise ValueError("Preserved local result changed: " + name)
    remaining = []
    for name, expected in transfer["assignments"].items():
        assignment = json.loads((PACKAGE / name).read_text())
        actual = hashlib.sha256(canonical({k:v for k,v in assignment.items() if k != "assignment_hash"}).encode()).hexdigest()
        if actual != expected or assignment["assignment_hash"] != expected:
            raise ValueError("Assignment changed: " + name)
        if assignment["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("Wrong statistical manifest: " + name)
        remaining.extend(u["run_id"] for u in assignment["units"])
    if (len(remaining) != len(set(remaining)) or sorted(remaining) != transfer["remaining_run_ids"]
            or set(remaining) & set(transfer["local_completed_run_ids"])):
        raise ValueError("Remaining work overlaps or has missing/duplicate units")
    return transfer, manifest


def bootstrap(setup_path=None):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "didbcf-portable-mpl"))
    if setup_path:
        setup = json.loads(Path(setup_path).read_text())
        for var, field in (("DIDBCF_REPAIRED_BUILD", "build"), ("DIDBCF_REPAIRED_CONTROLS", "controls"), ("DIDBCF_WEIGHT_CONTROLS", "weight_controls")):
            os.environ[var] = setup[field]
    sys.path.insert(0, str(TOOLS))
    return setup if setup_path else None


def available_memory():
    info = {line.split(":",1)[0]: line.split(":",1)[1] for line in Path("/proc/meminfo").read_text().splitlines()}
    capacity = int(info["MemAvailable"].split()[0])*1024
    limit, used = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if limit.exists() and used.exists() and limit.read_text().strip().isdecimal():
        capacity = min(capacity, int(limit.read_text())-int(used.read_text()))
    return capacity


def cpu_allowance():
    count = len(os.sched_getaffinity(0))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        values = quota.read_text().split()
        if values[0] != "max":
            count = min(count, max(1,int(values[0])//int(values[1])))
    return count


def source_commit():
    return subprocess.check_output(["git", "-C", str(PACKAGE.parents[2]), "rev-parse", "HEAD"], text=True).strip()


def write_bundle(out, assignment_path):
    transfer, manifest = verify_package()
    bundle = Path(out) / "_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(assignment_path, bundle / "assignment.json")
    shutil.copy2(PERFORMANCE / "manifest.json", bundle / "manifest.json")
    shutil.copy2(PACKAGE / "transfer.json", bundle / "transfer.json")
    with zipfile.ZipFile(bundle / "source_bundle.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for rel in manifest["source_hashes"]:
            archive.write(WORKSPACE / rel, rel)
        archive.write(assignment_path, "recovery_bundle/assignment.json")
        archive.write(PERFORMANCE / "manifest.json", "recovery_bundle/manifest.json")
        for path in sorted((PACKAGE / "helpers").glob("*.py")):
            archive.write(path, "portable_helpers/" + path.name)
    atomic(bundle / "git_source.json", dict(commit=source_commit(), transfer_hash=transfer["transfer_hash"],
           statistical_manifest_hash=manifest["manifest_hash"], helper_sha256=sha(__file__)))


def prepare_engine(args):
    """Restore outputs safely, then build or reuse this shard's verified engine."""
    _, manifest = verify_package()
    actual = {name:importlib.metadata.version(name) for name in manifest["dependencies"]}
    if actual != manifest["dependencies"]:
        raise ValueError("Dependency versions differ from the frozen study")
    # The pip-installed CMake executable may be absent from PATH in an
    # unactivated venv, or shadowed by a different system CMake on Colab.
    # Select the executable belonging to the verified package explicitly.
    import cmake
    cmake_bin = Path(cmake.CMAKE_BIN_DIR)
    if not (cmake_bin / "cmake").is_file():
        raise ValueError("Verified CMake package has no executable")
    os.environ["PATH"] = str(cmake_bin) + os.pathsep + os.environ.get("PATH","")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        with zipfile.ZipFile(args.resume) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate resume archive members")
            for entry in archive.infolist():
                target = (out / entry.filename).resolve()
                if (not target.is_relative_to(out) or entry.filename.startswith("/")
                        or ".." in Path(entry.filename).parts or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("Unsafe resume archive path")
                if entry.is_dir():
                    continue
                if target.exists():
                    with archive.open(entry) as stream:
                        incoming_sha = hashlib.file_digest(stream,"sha256").hexdigest()
                    if sha(target) != incoming_sha:
                        raise ValueError("Resume would overwrite different bytes: " + str(target))
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(entry) as source, target.open("wb") as dest:
                        shutil.copyfileobj(source,dest)
        saved = json.loads((out / "_bundle/assignment.json").read_text())
        expected = json.loads(args.assignment.read_text())
        if saved["assignment_hash"] != expected["assignment_hash"]:
            raise ValueError("Resume archive belongs to a different notebook")
    # Setup uses a fresh process, so the notebook never imports a stale native
    # module after pip changes. A resumed engine must match Python exactly.
    bootstrap()
    from recovery_collect import verify_engine_archive
    existing = out / "_engine"
    old_fingerprint = verify_engine_archive(out,manifest) if existing.exists() else None
    if old_fingerprint:
        import platform
        if old_fingerprint["python"] != platform.python_version():
            raise ValueError("Resume with the same Python version as the saved shard; no cross-binary continuation")
    command = [sys.executable,str(TOOLS / "recovery_setup.py"),"--work",str(args.work),"--cache",str(args.cache)]
    if old_fingerprint:
        command += ["--existing-build",str(existing)]
    subprocess.run(command,check=True)
    setup_path = args.work / "setup_complete.json"
    bootstrap(setup_path)
    from recovery_engine import activate, fingerprint
    active = fingerprint(activate())
    if old_fingerprint:
        if active != old_fingerprint:
            raise ValueError("Resume binary changed after native controls")
    else:
        setup = json.loads(setup_path.read_text())
        home = Path(setup["build"])
        build = json.loads((home / "build_complete.json").read_text())
        existing.mkdir()
        for rel in build["site_files"]:
            target = existing / "site" / rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(home / "site" / rel,target)
        for name,source in (("build_complete.json",home / "build_complete.json"),
                            ("setup_complete.json",setup_path),
                            ("repaired_controls.json",Path(setup["controls"])),
                            ("weight_controls.json",Path(setup["weight_controls"]))):
            shutil.copy2(source,existing / name)
        if verify_engine_archive(out,manifest) != active:
            raise ValueError("Archived engine differs from the loaded build")
    write_bundle(out,args.assignment)
    print(json.dumps(dict(status="PASS",setup=str(setup_path),engine=active),indent=2))
    return 0


def stop_handler(signum, frame):
    global STOP
    STOP = True


def shutdown():
    for entry in ACTIVE.values():
        if entry["process"].poll() is None:
            os.killpg(entry["process"].pid, signal.SIGTERM)
    for entry in ACTIVE.values():
        try:
            entry["process"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(entry["process"].pid, signal.SIGKILL)
            entry["process"].wait()
        entry["log"].close()
    ACTIVE.clear()


def run_assignment(args):
    transfer, manifest = verify_package()
    bootstrap(args.setup)
    from recovery_common import environment, verify_assignment, digest
    from recovery_engine import activate
    from recovery_runner import _attempts, _identity_core, unit_status, verify_completed
    assignment = verify_assignment(json.loads(args.assignment.read_text()))
    if assignment["manifest_hash"] != manifest["manifest_hash"] or assignment["dependencies"] != environment():
        raise ValueError("Wrong study or Python dependency versions")
    if assignment["assignment_hash"] != transfer["assignments"][str(args.assignment.relative_to(PACKAGE))]:
        raise ValueError("Assignment is not in this transfer")
    if activate()["specification"] != manifest["engine_specification"]:
        raise ValueError("Wrong repaired engine")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from recovery_collect import verify_engine_archive
    from recovery_engine import fingerprint
    if verify_engine_archive(out, manifest) != fingerprint(activate()):
        raise ValueError("Output archive engine differs from the loaded engine")
    pool = available_memory()-2*GIB
    workers = min(2, args.workers, cpu_allowance(), int(pool//(3*GIB)))
    if workers < 1:
        raise ValueError("Insufficient RAM for one safely bounded worker")
    resource_plan = dict(workers=workers, memory_pool_GiB=pool/GIB, worker_ceiling_GiB=3,
                         numeric_threads_per_worker=1, accessible_cpus=cpu_allowance())
    if args.dry_run:
        print(json.dumps(dict(status="DRY_RUN_ONLY_NO_SAMPLING", units=len(assignment["units"]), **resource_plan), indent=2))
        return 0
    write_bundle(out, args.assignment)
    deadline = time.monotonic()+args.hours*3600-args.reserve_min*60
    if deadline <= time.monotonic():
        raise ValueError("Sampling window does not exceed export reserve")
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    queue = sorted(assignment["units"], key=lambda u: (-u["est_seconds"], u["stat_job_id"], u["chain_index"]))
    state, retries, deferred = {}, {}, []
    status = dict(assignment_hash=assignment["assignment_hash"], manifest_hash=manifest["manifest_hash"],
                  transfer_hash=transfer["transfer_hash"], source_commit=source_commit(), resources=resource_plan,
                  started_utc=datetime.now(timezone.utc).isoformat(), units=state, events=[])
    print(json.dumps(dict(units=len(queue), **resource_plan), indent=2), flush=True)
    try:
        while queue or ACTIVE:
            if STOP:
                raise KeyboardInterrupt("Stopped; complete chains survive")
            for key, entry in list(ACTIVE.items()):
                code = entry["process"].poll()
                timed_out = time.monotonic() >= deadline
                if code is None and not timed_out:
                    continue
                if code is None:
                    os.killpg(entry["process"].pid, signal.SIGTERM)
                    try: entry["process"].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(entry["process"].pid, signal.SIGKILL)
                        entry["process"].wait()
                    code = entry["process"].returncode
                entry["log"].close()
                del ACTIVE[key]
                unit = entry["unit"]
                try:
                    if code != 0 or not verify_completed(entry["folder"], unit, assignment):
                        raise RuntimeError(f"Worker incomplete or exit {code}; see {entry['log_path']}")
                    record = json.loads((entry["folder"] / "complete.json").read_text())
                    state[key] = dict(status="complete", attempt=entry["attempt"], seconds=record["seconds"], peak_rss_mb=record["peak_rss_mb"])
                    print(f"COMPLETED {key} in {record['seconds']:.1f}s", flush=True)
                except Exception as error:
                    text = entry["log_path"].read_text(errors="replace").lower()
                    memory_error = any(word in text for word in ("memoryerror", "bad_alloc", "cannot allocate memory", "unable to allocate"))
                    if memory_error and entry["memory"] < 6*GIB and not timed_out:
                        retries[key] = 6*GIB
                        queue.insert(0, unit)
                        state[key] = dict(status="memory_retry", attempt=entry["attempt"], error=repr(error))
                    elif timed_out:
                        deferred.append(key)
                        state[key] = dict(status="interrupted_deadline", attempt=entry["attempt"], error=repr(error))
                    else:
                        state[key] = dict(status="failed", attempt=entry["attempt"], error=repr(error))
            if time.monotonic() >= deadline:
                deferred.extend(u["run_id"] for u in queue)
                for unit in queue: state[unit["run_id"]] = dict(status="deferred_deadline")
                queue.clear()
            while queue and len(ACTIVE) < workers and available_memory() >= 2*GIB:
                charged = sum(e["memory"] for e in ACTIVE.values())
                index = next((i for i,u in enumerate(queue) if charged+retries.get(u["run_id"],3*GIB) <= pool),None)
                if index is None:
                    break
                unit = queue.pop(index)
                key = unit["run_id"]
                existing = unit_status(out,unit,assignment)
                if existing["status"] == "complete":
                    state[key] = dict(status="complete", skipped_verified=True, attempt=existing["attempt"])
                    continue
                if unit["est_seconds"] > deadline-time.monotonic()-300:
                    deferred.append(key)
                    state[key] = dict(status="deferred_estimate")
                    continue
                if shutil.disk_usage(out).free < 2*GIB:
                    deferred.append(key)
                    state[key] = dict(status="deferred_disk")
                    continue
                attempt = len(_attempts(out / "runs" / key))
                folder = out / "runs" / key / f"attempt_{attempt:03d}"
                request = out / "_requests" / f"{key}_{attempt}.json"
                identity = _identity_core(unit,assignment,attempt)
                identity.update(source_git_commit=status["source_commit"], portable_transfer_hash=transfer["transfer_hash"])
                atomic(request,dict(unit=unit, identity=identity, workdir=str(folder)))
                log_path = out / "_logs" / f"{key}_{attempt}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("w")
                memory = retries.get(key,3*GIB)
                process = subprocess.Popen([sys.executable,str(TOOLS / "recovery_worker.py"),str(request)],
                           stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                           env=dict(os.environ,DIDBCF_WORKER_MEMORY_BYTES=str(memory)))
                ACTIVE[key] = dict(unit=unit, process=process, log=handle, log_path=log_path, folder=folder, memory=memory, attempt=attempt)
                state[key] = dict(status="running", attempt=attempt, pid=process.pid)
                print(f"START {key} {unit['job']['setting']} N={unit['job']['N']} degree={unit['job']['linearity_degree']} rep={unit['job']['rep']} {unit['budget_key']} chain={unit['chain_index']}",flush=True)
            if queue and not ACTIVE and min(retries.get(u["run_id"],3*GIB) for u in queue) > pool:
                deferred.extend(u["run_id"] for u in queue)
                for unit in queue: state[unit["run_id"]] = dict(status="deferred_memory")
                queue.clear()
            status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            status["summary"] = dict(completed=sum(v["status"]=="complete" for v in state.values()),
                    expected=len(assignment["units"]), deferred=len(set(deferred)), failed=[k for k,v in state.items() if v["status"]=="failed"],
                    active=len(ACTIVE), remaining=len(queue))
            atomic(out / "run_status.json",status)
            if queue or ACTIVE: time.sleep(2)
    except KeyboardInterrupt:
        deferred.extend(u["run_id"] for u in queue)
        for unit in queue: state[unit["run_id"]] = dict(status="deferred_user_stop")
        for key, entry in ACTIVE.items():
            deferred.append(key)
            state[key] = dict(status="interrupted_user_stop",attempt=entry["attempt"])
        queue.clear()
    finally:
        shutdown()
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        status["summary"] = dict(completed=sum(v["status"]=="complete" for v in state.values()),
                    expected=len(assignment["units"]), deferred=len(set(deferred)), failed=[k for k,v in state.items() if v["status"]=="failed"],
                    active=0, remaining=len(queue))
        atomic(out / "run_status.json",status)
    print(json.dumps(status["summary"],indent=2),flush=True)
    return 1 if status["summary"]["failed"] else 2 if status["summary"]["deferred"] else 0


def collect(args):
    verify_package()
    bootstrap()
    import recovery_collect as collector
    collector.INVENTORY = args.out.resolve() / "timing_inventory"
    collector.INVENTORY.mkdir(parents=True,exist_ok=True)
    sources = [PACKAGE / "local_completed_outputs"]
    for path in args.incoming:
        if path.is_dir() and not (path / "_engine").is_dir():
            sources.extend(sorted(path.glob("*.zip")))
        else: sources.append(path)
    sources = list(dict.fromkeys(str(p.resolve()) for p in sources))
    sys.argv = ["recovery_collect.py","--manifest",str(PERFORMANCE / "manifest.json"),"--incoming",*sources,"--out",str(args.out / "collection")]
    result = collector.main()
    status = json.loads((args.out / "collection/status.json").read_text())
    if status["complete_units"] != 1890 or status["complete_job_budget_pairs"] != 540:
        print("Collection is partial; return or rerun the remaining assignments before a full report.")
        return 2
    import importlib.util
    spec = importlib.util.spec_from_file_location("paired_performance_report",PERFORMANCE / "report_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_report(args.out.resolve())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    sub.add_parser("verify")
    setup=sub.add_parser("setup")
    for name in ("work","cache","out","assignment"):
        setup.add_argument("--"+name,type=Path,required=True)
    setup.add_argument("--resume",type=Path)
    bundle=sub.add_parser("bundle");bundle.add_argument("--out",type=Path,required=True);bundle.add_argument("--assignment",type=Path,required=True)
    run=sub.add_parser("run")
    run.add_argument("--assignment",type=Path,required=True);run.add_argument("--out",type=Path,required=True)
    run.add_argument("--setup",type=Path,required=True);run.add_argument("--workers",type=int,default=2)
    run.add_argument("--hours",type=float,default=9.5);run.add_argument("--reserve-min",type=float,default=45.)
    run.add_argument("--dry-run",action="store_true")
    coll=sub.add_parser("collect");coll.add_argument("--incoming",type=Path,nargs="+",required=True);coll.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="verify":
        transfer,manifest=verify_package()
        print(json.dumps(dict(status="PASS",transfer_hash=transfer["transfer_hash"],statistical_manifest_hash=manifest["manifest_hash"],
             local_completed=len(transfer["local_completed_run_ids"]),remaining=len(transfer["remaining_run_ids"]),notebooks=len(transfer["assignments"])),indent=2));return 0
    if args.action=="bundle":write_bundle(args.out,args.assignment);return 0
    if args.action=="setup":return prepare_engine(args)
    if args.action=="run":return run_assignment(args)
    return collect(args)


if __name__=="__main__":
    raise SystemExit(main())
