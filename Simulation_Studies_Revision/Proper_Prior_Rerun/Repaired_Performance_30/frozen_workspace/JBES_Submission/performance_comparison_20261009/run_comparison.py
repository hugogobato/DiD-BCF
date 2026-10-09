"""Restartable RAM-reserved local execution and automatic paired reporting."""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECOVERY = REPO / "JBES_Submission/recovery_repaired_20261008"
TOOLS = RECOVERY / "tools"
GIB = 2**30
STOP = False
ACTIVE = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def available_memory():
    info = {line.split(":", 1)[0]: line.split(":", 1)[1] for line in Path("/proc/meminfo").read_text().splitlines()}
    result = int(info["MemAvailable"].split()[0]) * 1024
    limit, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if limit.exists() and current.exists() and limit.read_text().strip().isdigit():
        result = min(result, int(limit.read_text()) - int(current.read_text()))
    return result


def stop_handler(signum, frame):
    global STOP
    STOP = True


def shutdown():
    for entry in ACTIVE.values():
        process = entry["process"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for entry in ACTIVE.values():
        try:
            entry["process"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(entry["process"].pid, signal.SIGKILL)
            entry["process"].wait()
        entry["log"].close()
    ACTIVE.clear()


def configure_environment(run):
    os.environ["MPLCONFIGDIR"] = str(run / "matplotlib_cache")
    os.environ["PYTHONUNBUFFERED"] = "1"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    setup = json.loads((run / "setup/setup_complete.json").read_text())
    for var, key in (("DIDBCF_REPAIRED_BUILD", "build"), ("DIDBCF_REPAIRED_CONTROLS", "controls"), ("DIDBCF_WEIGHT_CONTROLS", "weight_controls")):
        os.environ[var] = setup[key]
    sys.path.insert(0, str(TOOLS))
    return setup


def archive_environment(out, setup, assignments, run):
    from recovery_collect import verify_engine_archive
    from recovery_engine import activate, fingerprint
    engine = out / "_engine"
    if not engine.exists():
        build_home = Path(setup["build"])
        build = json.loads((build_home / "build_complete.json").read_text())
        for name in build["site_files"]:
            target = engine / "site" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(build_home / "site" / name, target)
        for source, name in ((build_home / "build_complete.json", "build_complete.json"),
                             (run / "setup/setup_complete.json", "setup_complete.json"),
                             (Path(setup["controls"]), "repaired_controls.json"),
                             (Path(setup["weight_controls"]), "weight_controls.json")):
            shutil.copy2(source, engine / name)
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert verify_engine_archive(out, manifest) == fingerprint(activate())
    bundle = out / "_bundle"
    bundle.mkdir(exist_ok=True)
    shutil.copy2(HERE / "manifest.json", bundle / "manifest.json")
    shutil.copy2(HERE / "PROTOCOL.md", bundle / "PROTOCOL.md")
    for path, assignment in assignments:
        target = bundle / "assignments" / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    nb = json.loads(next((HERE / "upload_to_colab").glob("*.ipynb")).read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and any(isinstance(t, ast.Name) and t.id == "BUNDLE_B64" for t in node.targets):
                    (bundle / "source_bundle.zip").write_bytes(base64.b64decode(node.value.value))
    shutil.copy2(run / "execution_manifest.json", bundle / "local_execution.json")


def collect(run):
    configure_environment(run)
    import recovery_collect as collector
    collector.INVENTORY = run / "collection_timing_inventory"
    collector.INVENTORY.mkdir(exist_ok=True)
    sys.argv = ["recovery_collect.py", "--manifest", str(HERE / "manifest.json"), "--incoming", str(run / "outputs"), "--out", str(run / "collection")]
    return collector.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "local_run")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hours", type=float, default=72.)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.collect:
        return collect(run)
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLCONFIGDIR"] = str(run / "matplotlib_cache")
    sys.path.insert(0, str(TOOLS))
    from recovery_common import (digest, environment, sha256_file, verify_assignment, write_json_atomic)
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert digest({k:v for k,v in manifest.items() if k != "manifest_hash"}) == manifest["manifest_hash"]
    for name, expected in {**manifest["source_hashes"], **manifest["performance_sources"]}.items():
        if sha256_file(REPO / name) != expected:
            raise ValueError("Frozen source drift: " + name)
    if environment() != manifest["dependencies"]:
        raise ValueError("Use the matching project .venv; dependency drift is refused")
    required_free = 2*json.loads((HERE / "preparation_summary.json").read_text())["expected_output_GiB"]*GIB + 5*GIB
    if shutil.disk_usage(REPO).free < required_free:
        raise ValueError("Insufficient disk for full output and result archive")
    controller_log = (run / "controller.log").open("a", buffering=1)
    def say(message):
        print(message, flush=True)
        controller_log.write(f"{utc()} {message}\n")
    snapshot = dict(status="RUNNING", phase="SETUP", started_utc=utc(), controller_pid=os.getpid(), manifest_hash=manifest["manifest_hash"],
                    expected_units=manifest["comparison"]["expected_chain_units"], completed_units=0, failed_units=0, active_workers=[],
                    full_production_started=False, previous_full_state_gate_passed=False)
    atomic(run / "status.json", snapshot)
    def logged(command, name):
        say("Starting " + name)
        with (run / (name + ".log")).open("w") as handle:
            result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {run / (name + '.log')}")
        say("Completed " + name)
    try:
        previous = json.loads((RECOVERY / "local_runs/20261008_overnight_v1/setup/setup_complete.json").read_text())
        logged([sys.executable, str(TOOLS / "recovery_setup.py"), "--work", str(run / "setup"), "--cache", str(run / "cache"), "--existing-build", previous["build"]], "setup")
        setup = configure_environment(run)
        from recovery_engine import activate
        activate()
        from recovery_runner import _attempts, _identity_core, unit_status, verify_completed
        assignments = []
        for path in sorted((HERE / "assignments").glob("wave_*/shard_*.json")):
            assignment = verify_assignment(json.loads(path.read_text()))
            assert assignment["manifest_hash"] == manifest["manifest_hash"]
            assignments.append((path, assignment))
        units = [(unit, assignment) for path, assignment in assignments for unit in assignment["units"]]
        assert len(units) == manifest["comparison"]["expected_chain_units"] == len({u["run_id"] for u,a in units})
        out = run / "outputs"
        out.mkdir(exist_ok=True)
        memory_pool = available_memory() - 2*GIB
        workers = min(args.workers, max(1, len(os.sched_getaffinity(0))-1), int(memory_pool // (3*GIB)))
        if workers < 1:
            raise ValueError("Insufficient RAM for one safely bounded worker")
        atomic(run / "resource_plan.json", dict(accessible_cpus=len(os.sched_getaffinity(0)), workers=workers,
               memory_pool_GiB=memory_pool/GIB, initial_worker_limit_GiB=3, numeric_threads_per_worker=1, hard_deadline_hours=args.hours))
        atomic(run / "execution_manifest.json", dict(started_utc=utc(), manifest_hash=manifest["manifest_hash"],
               performance_sources=manifest["performance_sources"], executable=sys.executable, runtime_python=sys.version,
               source_notebooks=json.loads((HERE / "upload_to_colab/notebook_index.json").read_text()),
               engine=setup["engine"], adaptation="Same frozen notebook units; local shared memory-reserved pool and 72-hour deadline", production_campaign=False))
        archive_environment(out, setup, assignments, run)
        queue = sorted(units, key=lambda pair: (pair[0]["job"]["rep"], pair[0]["job"]["recovery"]["cell_index"],
                       pair[0]["budget_key"] != "earlier_budget_repaired", pair[0]["chain_index"]))
        state, retry_limits = {}, {}
        say(f"Sampling {len(queue)} units with {workers} workers, one numeric thread each")
        while queue or ACTIVE:
            if STOP or time.monotonic()-started >= args.hours*3600:
                raise RuntimeError("Stopped or reached deadline; completed chains remain available for restart")
            for key, entry in list(ACTIVE.items()):
                code = entry["process"].poll()
                if code is None:
                    continue
                entry["log"].close()
                del ACTIVE[key]
                unit, assignment = entry["unit"], entry["assignment"]
                try:
                    if code != 0:
                        raise RuntimeError(f"Worker exit code {code}; see {entry['log_path']}")
                    if not verify_completed(entry["folder"], unit, assignment):
                        raise ValueError("Unverified completion")
                    record = json.loads((entry["folder"] / "complete.json").read_text())
                    state[key] = dict(state="COMPLETE", attempt=entry["attempt"], seconds=record["seconds"], peak_rss_mb=record["peak_rss_mb"])
                    say(f"COMPLETE {key} {unit['job']['setting']} N={unit['job']['N']} rep={unit['job']['rep']} {unit['budget_key']} chain={unit['chain_index']} in {record['seconds']:.1f}s")
                except Exception as error:
                    tail = entry["log_path"].read_text(errors="replace")[-16000:].lower()
                    memory_error = any(term in tail for term in ("memoryerror", "bad_alloc", "cannot allocate memory", "unable to allocate"))
                    if memory_error and entry["memory"] < 6*GIB:
                        retry_limits[key] = 6*GIB
                        queue.insert(0, (unit, assignment))
                        state[key] = dict(state="RETRY_MEMORY", error=repr(error), attempt=entry["attempt"])
                        say(f"MEMORY RETRY {key}: fresh same-seed attempt, 6 GiB reservation")
                    else:
                        state[key] = dict(state="FAILED", error=repr(error), attempt=entry["attempt"], log=str(entry["log_path"]))
                        say(f"FAILED {key}: {error!r}")
            while queue and len(ACTIVE) < workers and available_memory() >= 2*GIB:
                reserved = sum(entry["memory"] for entry in ACTIVE.values())
                index = next((i for i,(u,a) in enumerate(queue) if reserved + retry_limits.get(u["run_id"],3*GIB) <= memory_pool), None)
                if index is None:
                    break
                unit, assignment = queue.pop(index)
                key = unit["run_id"]
                existing = unit_status(out, unit, assignment)
                if existing["status"] == "complete":
                    state[key] = dict(state="COMPLETE", resumed_verified=True, attempt=existing["attempt"])
                    continue
                attempt = len(_attempts(out / "runs" / key))
                folder = out / "runs" / key / f"attempt_{attempt:03d}"
                request = out / "_requests" / f"{key}_{attempt}.json"
                write_json_atomic(request, dict(unit=unit, identity=_identity_core(unit, assignment, attempt), workdir=str(folder)))
                log_path = out / "_logs" / f"{key}_{attempt}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("w")
                memory = retry_limits.get(key, 3*GIB)
                process = subprocess.Popen([sys.executable, str(TOOLS / "recovery_worker.py"), str(request)],
                          cwd=REPO, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                          start_new_session=True, env=dict(os.environ, DIDBCF_WORKER_MEMORY_BYTES=str(memory)))
                ACTIVE[key] = dict(process=process, log=handle, log_path=log_path, memory=memory, started=time.monotonic(),
                                   unit=unit, assignment=assignment, folder=folder, attempt=attempt)
                state[key] = dict(state="RUNNING", attempt=attempt, pid=process.pid)
                say(f"START {key} {unit['job']['setting']} N={unit['job']['N']} rep={unit['job']['rep']} {unit['budget_key']} chain={unit['chain_index']}")
            if queue and not ACTIVE and min(retry_limits.get(u["run_id"],3*GIB) for u,a in queue) > memory_pool:
                raise RuntimeError("Remaining unit exceeds safe memory pool")
            completed = sum(value["state"] == "COMPLETE" for value in state.values())
            snapshot.update(phase="SAMPLING", updated_utc=utc(), completed_units=completed,
                failed_units=sum(value["state"] == "FAILED" for value in state.values()), worker_cap=workers,
                elapsed_hours=(time.monotonic()-started)/3600,
                active_workers=[dict(run_id=key, pid=e["process"].pid, job_id=e["unit"]["stat_job_id"],
                                    budget_key=e["unit"]["budget_key"], elapsed_seconds=time.monotonic()-e["started"], reserved_GiB=e["memory"]/GIB)
                                for key,e in ACTIVE.items()], units=state)
            atomic(run / "status.json", snapshot)
            (run / "LOCAL_PROGRESS.md").write_text(f"# Paired performance comparison\n\nUpdated {utc()}.\n\n"
                f"Completed {completed}/{len(units)} independent chains; active {len(ACTIVE)}; failed {snapshot['failed_units']}. "
                f"Elapsed {snapshot['elapsed_hours']:.2f} hours.\n\nBoth repaired budgets use all 30 replications per selected cell. "
                "Diagnostics do not screen replications. Automatic collection and paired reporting follow sampling.\n")
            if queue or ACTIVE:
                time.sleep(5)
        if any(value["state"] == "FAILED" for value in state.values()):
            raise RuntimeError("Unresolved failures; no complete paired performance claim is generated")
        snapshot.update(phase="COLLECTING", active_workers=[])
        atomic(run / "status.json", snapshot)
        logged([sys.executable, str(Path(__file__)), "--run-dir", str(run), "--collect"], "collection")
        status = json.loads((run / "collection/status.json").read_text())
        assert status["complete_units"] == status["expected_units"] == 1890 and status["complete_job_budget_pairs"] == 540
        snapshot.update(phase="REPORTING")
        atomic(run / "status.json", snapshot)
        logged([sys.executable, str(HERE / "report_comparison.py"), "--run-dir", str(run)], "report")
        snapshot.update(phase="ARCHIVING")
        atomic(run / "status.json", snapshot)
        with zipfile.ZipFile(run / "paired_performance_results.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for folder in (out, run / "collection", run / "report", run / "setup"):
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts:
                        archive.write(path, str(path.relative_to(run)))
            for name in ("manifest.json", "PROTOCOL.md", "reference_per_replication.csv", "paired_panel_verification.csv"):
                archive.write(HERE / name, "plan/" + name)
        snapshot.update(status="COMPLETED", phase="PAIRED_PERFORMANCE_READY", completed_utc=utc(),
                        elapsed_hours=(time.monotonic()-started)/3600, report=str(run / "report/PERFORMANCE_REPORT.md"))
        atomic(run / "status.json", snapshot)
        (run / "LOCAL_PROGRESS.md").write_text("# Comparison completed\n\nAll 1,890 chains and 540 repaired fits are verified. "
            "Read [PERFORMANCE_REPORT.md](report/PERFORMANCE_REPORT.md). The saved reference covers the same 270 datasets. "
            "No full production campaign was started.\n")
        say("COMPLETED: paired paper-metric report ready")
        return 0
    except BaseException as error:
        shutdown()
        snapshot.update(status="FAILED_OR_INCOMPLETE", error=repr(error), updated_utc=utc(), active_workers=[])
        atomic(run / "status.json", snapshot)
        traceback.print_exc()
        controller_log.write(traceback.format_exc())
        return 1
    finally:
        controller_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
