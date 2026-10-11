#!/usr/bin/env python3
"""Bounded, restartable recovery runner for one Colab notebook assignment.

Checkpoint contract
-------------------
* Structured estimator runs are independent chains under
  ``OUT/runs/<run_id>/attempt_NNN/``; a completed chain carries its seed,
  budget, run/job IDs, source/environment hashes, draw shapes, diagnostic
  inputs, output checksums and a completion record published last.
* Pretrend diagnostic jobs are atomic (stochtree ``BCFModel`` state is not
  serialisable per chain); the completion record says so explicitly.
* A verified completed attempt is never overwritten.  An incomplete attempt is
  retained and the unit restarts in a new attempt directory from its recorded
  seed; restarted draws are never appended as a continuation.
* The controller refuses to start a unit whose conservative duration estimate
  does not fit in the remaining sampling window (``--hours`` minus the
  checkpoint/export reserve).  Units that do not fit are deferred to another
  session and reported in ``run_status.json``.

Usage
-----
    python recovery_runner.py run --assignment A.json --out OUT --hours 8
    python recovery_runner.py status --assignment A.json --out OUT
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from recovery_common import (DEPENDENCIES, ORIGINAL_PINS, REPO,
                             assignment_hash, environment,
                             resource_report, runtime_versions,
                             set_single_thread_environment, sha256_file,
                             verify_assignment, write_json_atomic)

set_single_thread_environment()

import numpy as np  # noqa: E402  (after the thread environment is fixed)

from recovery_core import execute_unit  # noqa: E402


RUNS = "runs"
ATTEMPT = "attempt_{:03d}"


def _run_dir(out: Path, run_id: str) -> Path:
    return Path(out) / RUNS / run_id


def _attempts(folder: Path) -> list:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_dir() and p.name.startswith("attempt_"))


def _load_record(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _identity_core(unit: dict, assignment: dict, attempt: int) -> dict:
    job = unit["job"]
    return {
        "run_id": unit["run_id"],
        "stat_job_id": unit["stat_job_id"],
        "parent_job_id": job["id"],
        "phase": job["phase"],
        "setting": job["setting"],
        "rep": job["rep"],
        "spec": job["spec"],
        "kind": job["kind"],
        "budget_key": unit["budget_key"],
        "budget": unit["budget"],
        "chain_index": unit["chain_index"],
        "chain_seed": unit["chain_seed"],
        "trace": bool(unit.get("trace", False)),
        "attempt": attempt,
        "parent_manifest": assignment["parent_manifest"],
        "assignment_hash": assignment["assignment_hash"],
        "recovery_version": assignment["recovery_version"],
        "purpose": assignment["purpose"],
        "source_hashes_sha256": assignment["source_hashes_sha256"],
        "dependencies": assignment["dependencies"],
        "runtime": runtime_versions(),
        "engine": __import__("recovery_engine").fingerprint(__import__("recovery_engine").activate()),
        "continuation": ("independent_chain" if unit["chain_index"] is not None
                         else "atomic_job_no_per_chain_state"),
    }


def verify_completed(folder: Path, unit: dict, assignment: dict) -> bool:
    """Verify a completed attempt: identity, checksums, shapes, prior."""
    record = _load_record(folder / "complete.json")
    if record and record.get("engine") != __import__("recovery_engine").fingerprint(__import__("recovery_engine").activate()):
        raise ValueError("Completed chain belongs to a different binary or native-control record")
    if not record:
        return False
    if record.get("run_id") != unit["run_id"]:
        raise ValueError(f"Conflicting completed run in {folder}")
    expected = {
        "budget_key": unit["budget_key"],
        "chain_index": unit["chain_index"],
        "chain_seed": unit["chain_seed"],
        "parent_manifest": assignment["parent_manifest"],
        "source_hashes_sha256": assignment["source_hashes_sha256"],
        "assignment_hash": assignment["assignment_hash"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Conflicting completed run field {key} in {folder}")
    for name, checksum in record["outputs"].items():
        path = folder / name
        if not path.exists() or sha256_file(path) != checksum:
            raise ValueError(f"Corrupt or missing output {path}")
    atomic = unit["chain_index"] is None
    with np.load(folder / "draws.npz", allow_pickle=False) as draws:
        from recovery_validation import validate_arrays
        validate_arrays({key: draws[key] for key in draws.files}, unit)
        keys = set(draws.files)
        if atomic:
            if "tau" not in keys or draws["tau"].ndim != 2:
                raise ValueError(f"Invalid pretrend draws in {folder}")
        else:
            expected = (1, unit["budget"]["num_mcmc"])
            for key in keys:
                if key == "tau":
                    continue
                if draws[key].shape != expected:
                    raise ValueError(
                        f"Invalid chain draw shape {key} {draws[key].shape} "
                        f"expected {expected} in {folder}")
            if not np.isfinite(draws["ATT"]).all():
                raise ValueError(f"Non-finite ATT draws in {folder}")
            if not (draws.get("sigma2", np.array([1.0])) > 0).all():
                raise ValueError(f"Nonpositive variance draws in {folder}")
    prior = record["prior"]
    shape = prior.get("sigma2_shape",
                      prior.get("global_prior", {}).get("sigma2_shape"))
    rate = prior.get("sigma2_rate",
                     prior.get("global_prior", {}).get("sigma2_rate"))
    if shape != unit["job"]["sigma2_shape"]:
        raise ValueError(f"Prior shape mismatch in {folder}")
    if rate != unit["job"]["sigma2_rate"]:
        raise ValueError(f"Prior rate mismatch in {folder}")
    if unit["trace"] and not (folder / "trace.npz").exists():
        raise ValueError(f"Missing trace artifact in {folder}")
    return True


def unit_status(out: Path, unit: dict, assignment: dict) -> dict:
    folder = _run_dir(out, unit["run_id"])
    attempts = _attempts(folder)
    for attempt in reversed(attempts):
        if (attempt / "complete.json").exists():
            if verify_completed(attempt, unit, assignment):
                return {"status": "complete", "attempt": attempt.name,
                        "folder": str(attempt)}
            raise ValueError(f"Unverifiable completion in {attempt}")
    return {"status": "incomplete" if attempts else "pending",
            "attempts": [p.name for p in attempts]}


def run(args) -> int:
    assignment = verify_assignment(json.loads(Path(args.assignment).read_text()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    window = args.hours * 3600 - args.reserve_min * 60
    if window <= 0:
        raise SystemExit("--hours must exceed --reserve-min")

    # Complete bundle/environment verification before any sampling.
    for rel, expected in assignment["source_hashes"].items():
        path = REPO / rel
        if not path.exists():
            raise SystemExit(f"Missing bundle file {rel}")
        if sha256_file(path) != expected:
            raise SystemExit(f"Source hash mismatch for {rel}; refusing to run")
    from recovery_engine import activate
    active = activate()
    if active["specification"] != assignment["engine_specification"]:
        raise SystemExit("Engine specification differs from assignment")
    installed = environment()
    if installed != assignment["dependencies"]:
        raise SystemExit(
            "Dependency versions differ from the assignment; refusing to run. "
            f"expected {assignment['dependencies']}, installed {installed}")
    required = assignment.get("required_python")
    actual_python = runtime_versions()["python"]
    if required and actual_python != required:
        raise SystemExit(
            f"Production assignments require the accepted interpreter "
            f"{required}; this runtime is {actual_python}. Prepare a new "
            "environment record and equivalence check instead of substituting.")

    units = list(assignment["units"])
    if args.unit:
        wanted = set(args.unit.split(","))
        units = [u for u in units if u["run_id"] in wanted]
    order = sorted(range(len(units)),
                   key=lambda i: units[i]["est_seconds"], reverse=True)

    print(json.dumps({"purpose": assignment["purpose"],
                      "units": len(units), "window_seconds": window,
                      "reserve_minutes": args.reserve_min,
                      "manifest_hash": assignment.get("manifest_hash")},
                     indent=2), flush=True)
    print(json.dumps(resource_report(), indent=2), flush=True)

    status = {"purpose": assignment["purpose"],
              "assignment_hash": assignment["assignment_hash"],
              "started_unix": time.time(), "units": {}, "events": []}
    completed, deferred, failed, conflicts, skipped = [], [], [], [], []
    for index in order:
        unit = units[index]
        try:
            state = unit_status(out, unit, assignment)
        except ValueError as error:
            conflicts.append(unit["run_id"])
            status["units"][unit["run_id"]] = {
                "status": "conflict", "error": repr(error)}
            status["events"].append(
                {"time": time.time(), "event": "conflict",
                 "run_id": unit["run_id"], "error": repr(error)})
            write_json_atomic(out / "run_status.json", status)
            print(f"CONFLICT {unit['run_id']}: {error!r}", flush=True)
            continue
        if state["status"] == "complete":
            skipped.append(unit["run_id"])
            status["units"][unit["run_id"]] = state
            continue
        if args.dry_run:
            deferred.append(unit["run_id"])
            status["units"][unit["run_id"]] = {**state, "dry_run": True}
            continue
        remaining = window - (time.monotonic() - started)
        estimate = float(unit["est_seconds"])
        if estimate > remaining - args.margin_min * 60:
            deferred.append(unit["run_id"])
            status["units"][unit["run_id"]] = {
                **state, "deferred": True, "est_seconds": estimate,
                "remaining_seconds": max(remaining, 0.0)}
            status["events"].append(
                {"time": time.time(), "event": "deferred",
                 "run_id": unit["run_id"], "est_seconds": estimate})
            write_json_atomic(out / "run_status.json", status)
            continue
        memory = resource_report()
        available_kib = int(str(memory.get("mem_available_kib", "0 kB")).split()[0])
        memory_bytes = min(9 * 1024**3, available_kib * 1024 - 3 * 1024**3) if available_kib else 4 * 1024**3
        if memory_bytes < 2 * 1024**3:
            deferred.append(unit["run_id"])
            status["units"][unit["run_id"]] = {**state, "deferred": True, "reason": "RAM reserve: less than 2 GiB worker allowance"}
            write_json_atomic(out / "run_status.json", status)
            continue
        attempts = _attempts(_run_dir(out, unit["run_id"]))
        attempt = len(attempts)
        identity = _identity_core(unit, assignment, attempt)
        workdir = _run_dir(out, unit["run_id"]) / ATTEMPT.format(attempt)
        status["events"].append({"time": time.time(), "event": "start",
                                 "run_id": unit["run_id"], "attempt": attempt,
                                 "est_seconds": estimate})
        write_json_atomic(out / "run_status.json", status)
        try:
            import subprocess, signal
            request_path = out / "_requests" / (unit["run_id"] + "_" + str(attempt) + ".json")
            write_json_atomic(request_path, {"unit": unit, "identity": identity, "workdir": str(workdir.resolve())})
            log_path = out / "_logs" / (unit["run_id"] + "_" + str(attempt) + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            worker_environment = dict(os.environ, DIDBCF_WORKER_MEMORY_BYTES=str(memory_bytes))
            with log_path.open("w") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("recovery_worker.py")), str(request_path.resolve())], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=worker_environment)
                try:
                    return_code = process.wait(timeout=max(1., remaining - args.margin_min * 60))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError("Hard sampling deadline interrupted this unit; retain attempt and restart from seed")
            if return_code != 0:
                raise RuntimeError("Worker failed; inspect " + str(log_path))
            if not verify_completed(workdir, unit, assignment):
                raise RuntimeError("Worker completion failed verification")
            record = json.loads((workdir / "complete.json").read_text())
        except Exception as error:  # keep partial work and report the failure
            failed.append(unit["run_id"])
            status["units"][unit["run_id"]] = {
                **state, "failed": True, "error": repr(error),
                "attempt": attempt}
            status["events"].append(
                {"time": time.time(), "event": "failed",
                 "run_id": unit["run_id"], "error": repr(error)})
            write_json_atomic(out / "run_status.json", status)
            print(f"FAILED {unit['run_id']}: {error!r}", flush=True)
            continue
        status["units"][unit["run_id"]] = {
            "status": "complete", "attempt": attempt,
            "seconds": record["seconds"], "peak_rss_mb": record["peak_rss_mb"]}
        completed.append(unit["run_id"])
        status["events"].append({"time": time.time(), "event": "complete",
                                 "run_id": unit["run_id"],
                                 "seconds": record["seconds"]})
        write_json_atomic(out / "run_status.json", status)
        print(f"COMPLETED {unit['run_id']} in {record['seconds']:.1f}s; "
              f"peak RSS {record['peak_rss_mb']:.0f} MiB", flush=True)

    status["summary"] = {
        "completed_now": len(completed), "skipped_verified": len(skipped),
        "deferred": len(deferred),
        "failed": failed, "conflicts": conflicts,
        "elapsed_seconds": time.monotonic() - started,
        "finished_unix": time.time()}
    write_json_atomic(out / "run_status.json", status)
    print(json.dumps(status["summary"], indent=2), flush=True)
    if failed or conflicts:
        return 1
    if deferred:
        return 2
    return 0


def status(args) -> int:
    assignment = verify_assignment(json.loads(Path(args.assignment).read_text()))
    out = Path(args.out)
    rows = []
    for unit in assignment["units"]:
        try:
            state = unit_status(out, unit, assignment)
        except ValueError as error:
            state = {"status": "conflict", "error": repr(error)}
        rows.append({"run_id": unit["run_id"], "stat_job_id": unit["stat_job_id"],
                     "budget_key": unit["budget_key"],
                     "chain": unit["chain_index"], **state})
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--assignment", required=True, type=Path)
    common.add_argument("--out", required=True, type=Path)
    run_p = sub.add_parser("run", parents=[common])
    run_p.add_argument("--hours", type=float, default=8.0)
    run_p.add_argument("--reserve-min", type=float, default=45.0)
    run_p.add_argument("--margin-min", type=float, default=10.0)
    run_p.add_argument("--unit", default=None,
                       help="comma-separated run IDs to execute")
    run_p.add_argument("--dry-run", action="store_true")
    status_p = sub.add_parser("status", parents=[common])
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main())
