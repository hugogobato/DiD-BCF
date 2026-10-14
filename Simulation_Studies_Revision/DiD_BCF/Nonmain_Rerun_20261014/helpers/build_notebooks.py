"""Generate loose, assignment-specific Colab notebooks for the non-main rerun."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from common import PACKAGE, read


RELATIVE = "Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014"
ASSIGNMENT_NAME = re.compile(r"^nonmain_wave_(?P<wave>\d+)_shard_(?P<shard>\d+)$")


def notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"provenance": []},
        },
        "cells": [
            {
                "cell_type": kind,
                "id": f"cell-{i:02d}",
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **(
                    {"outputs": [], "execution_count": None}
                    if kind == "code"
                    else {}
                ),
            }
            for i, (kind, source) in enumerate(cells)
        ],
    }


def write(path, cells):
    for kind, source in cells:
        if kind == "code":
            ast.parse(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n")


def setup_cell(commit):
    return f'''import json, os, sys, subprocess, time
from pathlib import Path
assert (3, 12) <= sys.version_info[:2] <= (3, 14), "Use Python 3.12 to 3.14"
SESSION_STARTED = time.time()
DEADLINE = SESSION_STARTED + 9.5 * 3600
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
SOURCE_COMMIT = {commit!r}
REPO = Path("/content/unique_nonmain") if Path("/content").exists() else Path.cwd() / "unique_nonmain"
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    "https://github.com/hugogobato/DiD-BCF.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "fetch", "origin", SOURCE_COMMIT], check=True)
subprocess.run(["git", "-C", str(REPO), "sparse-checkout", "set", {RELATIVE!r}], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", SOURCE_COMMIT], check=True)
actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert actual == SOURCE_COMMIT, (actual, SOURCE_COMMIT)
PACKAGE = REPO / {RELATIVE!r}
manifest = json.loads((PACKAGE / "manifest.json").read_text())
requirements = [name + "==" + version for name, version in manifest["dependencies"].items()]
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                *requirements], check=True)
HELPERS = PACKAGE / "helpers"
print("Pinned non-main sources and dependencies installed:", SOURCE_COMMIT)
'''


def _phase_counts(manifest):
    return Counter(job["phase"] for job in manifest["jobs"])


def _read_assignments():
    assignments = []
    for path in sorted((PACKAGE / "assignments").glob("*.json")):
        value = read(path)
        if value.get("execution") not in {"colab", "local"}:
            raise ValueError(f"Assignment {path.name} must declare execution as colab or local")
        assignments.append((path, value))
    if not assignments:
        raise ValueError("No non-main assignments found")
    return assignments


def _format_phase_counts(counts):
    return ", ".join(f"{phase}: {counts[phase]:,}" for phase in sorted(counts)) or "none"


def build(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full immutable Git commit is required")

    preparation = read(PACKAGE / "preparation.json")
    manifest = read(PACKAGE / "manifest.json")
    jobs = {job["id"]: job for job in manifest["jobs"]}
    assignment_rows = _read_assignments()
    output = PACKAGE / "Colab"
    generated = []
    execution_counts = Counter()
    execution_phases = defaultdict(Counter)
    wave_counts = Counter()
    assigned_ids = set()

    for path, assigned in assignment_rows:
        name = assigned["name"]
        ids = assigned["job_ids"]
        missing = sorted(set(ids) - set(jobs))
        if missing:
            raise ValueError(f"Assignment {name} contains unknown jobs: {missing[:3]}")
        overlap = assigned_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Jobs appear in more than one assignment: {sorted(overlap)[:3]}")
        assigned_ids.update(ids)
        execution = assigned["execution"]
        execution_counts[execution] += len(ids)
        for job_id in ids:
            execution_phases[execution][jobs[job_id]["phase"]] += 1

        if execution != "colab":
            continue

        match = ASSIGNMENT_NAME.fullmatch(name)
        if not match:
            raise ValueError(f"Colab assignment has an invalid non-main name: {name}")
        wave = int(match.group("wave"))
        wave_counts[wave] += 1
        target = output / f"wave_{wave:02d}" / f"{name}.ipynb"
        phases = Counter(jobs[job_id]["phase"] for job_id in ids)
        phase_text = _format_phase_counts(phases)
        fit_count = len(ids)
        estimated_hours = assigned.get("estimated_wall_hours")
        runtime_text = (
            f"{estimated_hours:.1f} hours"
            if isinstance(estimated_hours, (int, float))
            else "about 8 hours"
        )
        target_hours = assigned.get("target_hours", preparation.get("target_hours", 8))

        intro = f'''# DiD-BCF non-main rerun: {name}

This independent notebook runs {fit_count:,} complete fits from these frozen study phases: {phase_text}. It uses the campaign's earlier budget: GFR 50, no additional burn-in, 500 retained draws per chain, thinning 5, and three chains. The campaign's panels, priors, and seed schedule remain pinned to the manifest.

The estimated runtime is {runtime_text} with up to two single-threaded CPU workers. Each worker is admitted only when the runtime has enough free memory; execution falls to one worker when memory or CPU limits require it. The notebook stops within 9.5 hours, including setup. Python 3.12 to 3.14 and a Colab runtime with about 12 GB RAM are appropriate. The scheduling target for this assignment is {target_hours} hours; actual time varies by runtime.

This notebook exports paper metrics, compact diagnostics, and provenance in one ZIP. Posterior arrays, source archives, and native build files stay on the runtime disk. Verified complete fits can be resumed from a matching assignment ZIP or Drive checkpoint. An interrupted fit restarts from its original seed. Results are collected without filtering fits by diagnostics or performance.
'''
        config = f'''# Optional Drive checkpoints are copied every 20 minutes.
USE_DRIVE = False
RESUME_ZIP = ""  # Set to this assignment's earlier metrics ZIP if resuming.
WORKERS = 2
NAME = {name!r}
WORK = Path("/content") / (NAME + "_work") if Path("/content").exists() else Path.cwd() / (NAME + "_work")
OUT = WORK / "nonmain_output"
OUTPUT_FILE = WORK / ("didbcf_nonmain_" + NAME + ".zip")
ASSIGNMENT = PACKAGE / "assignments" / (NAME + ".json")
DRIVE_DIR = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/DiD_BCF_Nonmain_Rerun_20261014")
    previous = DRIVE_DIR / OUTPUT_FILE.name
    if not RESUME_ZIP and previous.exists():
        RESUME_ZIP = str(previous)
# To restore a downloaded metrics ZIP, upload it in Colab's Files pane and set
# RESUME_ZIP to its /content/... path before running the preparation cell.
print("Fits:", {fit_count}, "Phases:", {phase_text!r}, "Output:", OUTPUT_FILE)
'''
        prepare = '''command = [sys.executable, str(HELPERS / "runtime.py"), "prepare",
           "--assignment", str(ASSIGNMENT), "--work", str(WORK), "--out", str(OUT),
           "--cache", str(WORK / "source_cache"), "--deadline", str(DEADLINE)]
if RESUME_ZIP:
    command += ["--resume", RESUME_ZIP]
subprocess.run(command, check=True)
'''
        execute = '''command = [sys.executable, str(HELPERS / "runtime.py"), "run",
           "--assignment", str(ASSIGNMENT), "--work", str(WORK), "--out", str(OUT),
           "--setup", str(WORK / "setup_complete.json"), "--zip", str(OUTPUT_FILE),
           "--workers", str(WORKERS), "--deadline", str(DEADLINE)]
if DRIVE_DIR:
    command += ["--drive-dir", str(DRIVE_DIR)]
result = subprocess.run(command)
if result.returncode not in (0, 2):
    raise RuntimeError("Execution failed; inspect job logs and preserve the latest metrics checkpoint")
status = json.loads((OUT / "bundle" / "status.json").read_text())
print("Completed:", status["completed"], "/", status["expected"])
if status["missing_job_ids"]:
    print("Resume this same assignment with its metrics ZIP to complete",
          len(status["missing_job_ids"]), "remaining fits.")
'''
        download = '''output_file = str(OUTPUT_FILE)
assert Path(output_file).is_file(), "No metrics archive was created"
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''
        write(
            target,
            [
                ("markdown", intro),
                ("code", setup_cell(commit)),
                ("code", config),
                ("code", prepare),
                ("code", execute),
                ("code", download),
            ],
        )
        generated.append((target, assigned, phases))

    reused_count = len(manifest.get("reuse_registry", {}))
    collector_intro = f'''# Collect DiD-BCF non-main metrics

Upload the metrics ZIPs from this non-main campaign, or add their paths in `INPUT_PATHS`. The collector validates assignments, fit identities, checksums, and the {reused_count} registered reused fit(s). It reports missing jobs by phase and saves partial tables when phases remain incomplete. It does not read or include the separate main rerun's results. Empirical application ZIPs produced by the local assignment can be supplied here as well.
'''
    collect_code = '''from google.colab import files
uploaded = files.upload()
UPLOADED_PATHS = [str(Path(name).resolve()) for name in uploaded if name.lower().endswith(".zip")]
INPUT_PATHS = []  # Optional paths such as "/content/didbcf_nonmain_local_application.zip".
INPUTS = list(dict.fromkeys([*UPLOADED_PATHS, *map(str, INPUT_PATHS)]))
if not INPUTS:
    raise ValueError("Upload at least one didbcf_nonmain_*.zip or add a ZIP path to INPUT_PATHS")
RESULTS = Path("/content/didbcf_nonmain_tables")
subprocess.run([sys.executable, str(HELPERS / "collect.py"), *INPUTS,
                "--out", str(RESULTS)], check=True)
status = json.loads((RESULTS / "status.json").read_text())
print("Phase expected:", status["phase_expected"])
print("Phase completed:", status["phase_completed"])
print("Missing fits:", status["missing_total"])
import shutil
output_file = shutil.make_archive("/content/didbcf_nonmain_tables", "zip", RESULTS)
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''
    write(
        output / "collect_nonmain_metrics.ipynb",
        [("markdown", collector_intro), ("code", setup_cell(commit)), ("code", collect_code)],
    )

    all_phase_counts = _phase_counts(manifest)
    scheduled_phase_counts = Counter(preparation.get("phase_expected", all_phase_counts))
    if scheduled_phase_counts != all_phase_counts:
        raise ValueError(
            "Preparation phase counts do not match the frozen manifest: "
            f"{dict(scheduled_phase_counts)} != {dict(all_phase_counts)}"
        )
    colab_counts = execution_phases["colab"]
    local_counts = execution_phases["local"]
    wave_summary = ", ".join(
        f"wave_{wave:02d}: {count} notebooks"
        for wave, count in sorted(wave_counts.items())
    ) or "no Colab waves"
    max_concurrent = preparation.get("max_concurrent_notebooks", 48)
    schedule_summary = preparation.get(
        "note",
        preparation.get("timing_note", "Assignments carry individual runtime estimates."),
    )
    guide = f'''# Non-main Colab upload guide

Upload each generated `.ipynb` directly into a CPU Colab runtime and choose **Run all**. The notebooks clone and verify immutable helper commit `{commit}` in the dedicated `unique_nonmain` folder. The package pins Python dependencies and the repaired engine. No global stochtree package is installed.

This guide is generated from the frozen assignments and manifest. It contains {sum(1 for _, assigned in assignment_rows if assigned["execution"] == "colab")} Colab notebooks covering {execution_counts["colab"]:,} assigned fits across {wave_summary}. Run up to {max_concurrent} notebooks concurrently within a wave. Each notebook uses at most two single-threaded workers, with memory-based admission reducing that number when needed. The package carries {reused_count} verified reused fit(s), which do not create Colab jobs. The local assignments contain {execution_counts["local"]:,} fits ({_format_phase_counts(local_counts)}); those jobs do not have Colab notebooks and their returned metrics ZIP can be included in the collector.

The manifest contains {_format_phase_counts(all_phase_counts)} fits by study phase. The assignments schedule {_format_phase_counts(colab_counts)} on Colab and {_format_phase_counts(local_counts)} locally. This describes the campaign workload, not completion status. The collector reports collected and missing fits separately for every phase.

Each notebook targets its scheduled wall time, usually about 8 hours, and stops within 9.5 hours including setup. Enable `USE_DRIVE` before running to copy a metrics-only ZIP checkpoint to Drive every 20 minutes. To resume, run the same notebook and provide its ZIP through `RESUME_ZIP`; Drive checkpoints are detected automatically when enabled. Only verified complete fits resume. An interrupted fit restarts from its original seed. Native arrays and build products stay on the runtime disk, and each notebook downloads one metrics ZIP.

Upload the simulation and local application ZIPs to `collect_nonmain_metrics.ipynb`. To provide a ZIP already present in the Colab runtime, add its path to `INPUT_PATHS`. The collector validates the non-main campaign, reports missing jobs by phase, and downloads one ZIP of collected tables. Partial collections remain available with explicit missing-fit counts.

{schedule_summary}

| Wave and assignment | Study phases | Fits | Estimated hours with two workers |
| --- | --- | ---: | ---: |
'''
    for target, assigned, phases in generated:
        relative = target.relative_to(output).as_posix()
        estimated = assigned.get("estimated_wall_hours")
        hours = f"{estimated:.1f}" if isinstance(estimated, (int, float)) else "not estimated"
        guide += (
            f"| [{assigned['name']}]({relative}) | {_format_phase_counts(phases)} "
            f"| {len(assigned['job_ids']):,} | {hours} |\n"
        )
    (output / "UPLOAD_GUIDE.md").write_text(guide)
    print(
        "Generated",
        len(generated),
        "Colab run notebooks and one collector; local assignments were skipped.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    build(args.source_commit)
