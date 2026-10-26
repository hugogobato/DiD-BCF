#!/usr/bin/env python3
"""Generate self-contained, hash-pinned, restartable Colab notebooks.

Each notebook embeds the frozen source bundle, its shard assignment and the
bundle hash manifest, then runs :mod:`recovery_runner` in a bounded sampling
window with a background partial-checkpoint archive written outside the output
directory.  The final archive is also written outside the output directory and
downloaded with the standard Colab fallback.

Required cells, in order: purpose/configuration; pinned setup and source
extraction; bundle/environment verification; runtime CPU/RAM/disk inspection;
safe resume import and checkpoint verification; explicit job/chain assignment;
bounded execution; diagnostics and integrity summary; archive and automatic
download.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

from recovery_common import (NOTEBOOKS, RECOVERY, RECOVERY_VERSION, REPO,
                             digest, sha256_bytes, sha256_file,
                             bundle_paths, write_json_atomic)

PURPOSE_TITLES = {
    "diagnostic_pilot": "DiD-BCF recovery diagnostic pilot",
    "production": "DiD-BCF recovery production phase",
}


def cell(kind, text, cell_id):
    out = {"cell_type": kind, "metadata": {}, "id": cell_id,
           "source": text.splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def build_bundle(assignment: dict, assignment_path: Path,
                 manifest_path: Path, manifest: dict) -> str:
    buffer = io.BytesIO()
    files = {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_paths():
            rel = str(path.relative_to(REPO))
            archive.write(path, rel)
            files[rel] = sha256_file(path)
        archive.writestr("recovery_bundle/assignment.json",
                         json.dumps(assignment, indent=2))
        archive.writestr("recovery_bundle/manifest.json",
                         json.dumps(manifest, indent=2))
        bundle_manifest = {
            "recovery_version": RECOVERY_VERSION,
            "files": files,
            "assignment_sha256": sha256_file(assignment_path),
            "manifest_sha256": sha256_file(manifest_path),
            "assignment_hash": assignment["assignment_hash"],
            "manifest_hash": manifest["manifest_hash"],
        }
        archive.writestr("recovery_bundle/bundle_manifest.json",
                         json.dumps(bundle_manifest, indent=2))
    return base64.b64encode(buffer.getvalue()).decode()


CONFIG_TEMPLATE = '''# ======================= CONFIGURATION =======================
# Sampling window: the controller refuses to start a unit whose conservative
# estimate does not fit in HOURS minus RESERVE_MIN. A hard deadline terminates
# an unfinished chain; completed work is checkpointed and a background
# thread refreshes a partial archive outside the output directory.
HOURS = @@HOURS@@
SETUP_ATTEMPT = "setup01"
# Local fallback uses a task directory; Colab uses /content.
import os
from pathlib import Path
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
WORKSPACE = Path("/content") if Path("/content").is_dir() else Path.cwd() / "didbcf_repaired_workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)
RESERVE_MIN = 45.0
CHECKPOINT_EVERY_MIN = 20.0

# Optional path to a previously downloaded output zip (e.g. an earlier wave
# returned by this same notebook).  Set to None for a clean start.
RESUME_ZIP = None

# Optional Google Drive directory for partial-checkpoint copies.  Set to None
# to keep everything local; /content is lost on a forced runtime disconnect.
DRIVE_CHECKPOINT_DIR = None

LABEL = "@@LABEL@@"
# ==============================================================
import platform
import sys

RECORDED_PYTHON = "@@RECORDED_PYTHON@@"
print("Python", platform.python_version(), "| recorded for this campaign:",
      RECORDED_PYTHON)
if not (3, 12) <= sys.version_info[:2] <= (3, 14):
    raise RuntimeError("Use Python 3.12 through 3.14 for this pinned package")
if platform.python_version() != RECORDED_PYTHON:
    print("NOTE: interpreter differs from the recorded environment. Exact "
          "cross-interpreter reproduction of MCMC paths is not guaranteed; "
          "the collector records the environment of every run.")
'''

INSTALL_TEMPLATE = '''# Pinned dependencies from the frozen assignment.  Fail closed on mismatch;
# do not substitute newer packages without a documented equivalence check.
import importlib.metadata
import subprocess
import sys

PINS = @@PINS@@
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *PINS])
problems = {}
for spec in PINS:
    name, _, version = spec.partition("==")
    installed = importlib.metadata.version(name)
    if installed != version:
        problems[name] = (installed, version)
if problems:
    raise SystemExit(
        "Installed versions differ from the pinned assignment: " + repr(problems)
        + ". Do not continue: prepare a documented new environment and an "
        "equivalence check instead of an unrecorded fallback.")
print("Pinned dependencies verified:", PINS)
'''

EXTRACT_TEMPLATE = '''# Extract and verify the embedded frozen bundle.  Every file is checked
# against the embedded bundle manifest; the assignment and manifest hashes are
# re-derived.  This cell must pass before any sampling.
import base64
import hashlib
import io
import json
import pathlib
import shutil
import sys
import zipfile

BUNDLE_B64 = "@@BUNDLE_B64@@"
BUNDLE_ID = hashlib.sha256(BUNDLE_B64.encode()).hexdigest()[:12]
ROOT = WORKSPACE / ("didbcf_recovery_src_" + BUNDLE_ID)
BUNDLE_MANIFEST_SHA256 = "@@BUNDLE_MANIFEST_SHA256@@"
EMBEDDED_ZIP_SHA256 = "@@ZIP_SHA@@"

def _extract():
    payload = base64.b64decode(BUNDLE_B64)
    assert hashlib.sha256(payload).hexdigest() == EMBEDDED_ZIP_SHA256, "Embedded source archive changed"
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            name = info.filename
            parts = pathlib.PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts:
                raise ValueError("Unsafe bundle member " + name)
        archive.extractall(ROOT)

_extract()
assert hashlib.sha256((ROOT / "recovery_bundle/bundle_manifest.json").read_bytes()).hexdigest() == BUNDLE_MANIFEST_SHA256
bundle_meta = json.loads((ROOT / "recovery_bundle/bundle_manifest.json").read_text())
assert bundle_meta["assignment_hash"] == "@@ASSIGNMENT_HASH@@"
for rel, expected in bundle_meta["files"].items():
    path = ROOT / rel
    assert path.exists(), "Missing bundle file " + rel
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, "Bundle hash mismatch " + rel
tools = ROOT / "JBES_Submission/recovery_repaired_20261008/tools"
if str(tools) not in sys.path:
    sys.path.insert(0, str(tools))
SETUP_WORK = WORKSPACE / ("didbcf_repaired_setup_" + LABEL + "_" + SETUP_ATTEMPT)
try:
    subprocess.run([sys.executable, str(tools / "recovery_setup.py"), "--work", str(SETUP_WORK), "--cache", str(WORKSPACE / "didbcf_pinned_archives")], check=True)
except Exception:
    if SETUP_WORK.exists():
        failed_zip = shutil.make_archive(str(WORKSPACE / (LABEL + "_setup_failure")), "zip", str(SETUP_WORK))
        try:
            from google.colab import files
            files.download(failed_zip)
            print("Downloaded:", failed_zip)
        except Exception as e:
            print("(Not on Colab / download skipped):", e)
    raise
setup = json.loads((SETUP_WORK / "setup_complete.json").read_text())
os.environ["DIDBCF_REPAIRED_BUILD"] = setup["build"]
os.environ["DIDBCF_REPAIRED_CONTROLS"] = setup["controls"]
os.environ["DIDBCF_WEIGHT_CONTROLS"] = setup["weight_controls"]
import recovery_common as rc
assignment = rc.verify_assignment(
    json.loads((ROOT / "recovery_bundle/assignment.json").read_text()))
manifest = json.loads((ROOT / "recovery_bundle/manifest.json").read_text())
assert rc.digest({k: v for k, v in manifest.items()
                  if k != "manifest_hash"}) == manifest["manifest_hash"]
for rel, expected in assignment["source_hashes"].items():
    actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    assert actual == expected, "Assignment source mismatch " + rel
assert assignment["dependencies"] == rc.environment()
print("Bundle, assignment, manifest, sources and dependencies verified.")
print("Manifest:", manifest["manifest_hash"])
print("Assignment:", assignment["assignment_hash"])
'''

RESUME_TEMPLATE = '''# Safe resume import and runtime inspection.  An existing output file is
# accepted only when its bytes match the incoming archive; conflicting files
# abort instead of being overwritten.
import importlib
import json
import pathlib
import sys

rc = importlib.import_module("recovery_common")
print(json.dumps(rc.resource_report(), indent=2))

OUT = WORKSPACE / "didbcf_repaired_runs" / LABEL
OUT.mkdir(parents=True, exist_ok=True)
BUNDLE_DIR = OUT / "_bundle"
BUNDLE_DIR.mkdir(exist_ok=True)
(BUNDLE_DIR / "source_bundle.zip").write_bytes(base64.b64decode(BUNDLE_B64))
(BUNDLE_DIR / "assignment.json").write_bytes(
    (ROOT / "recovery_bundle/assignment.json").read_bytes())
(BUNDLE_DIR / "manifest.json").write_bytes(
    (ROOT / "recovery_bundle/manifest.json").read_bytes())
print("Embedded source bundle saved into the output for provenance:",
      BUNDLE_DIR / "source_bundle.zip")
if RESUME_ZIP:
    summary = rc.extract_zip_verified(RESUME_ZIP, OUT, verified=True)
    print("Resume import:", {k: len(v) for k, v in summary.items()})
    print("Verified completed chains are skipped; incomplete attempts are kept "
          "and restarted in a new attempt directory.")
else:
    print("Clean start: RESUME_ZIP is None.")

ENGINE_DIR = OUT / "_engine"
from recovery_engine import activate, fingerprint
if ENGINE_DIR.exists():
    from recovery_collect import verify_engine_archive
    assert verify_engine_archive(OUT, manifest) == fingerprint(activate()), "Resume requires the same verified binary and Python pins"
else:
    ENGINE_DIR.mkdir()
    home = pathlib.Path(setup["build"])
    build = json.loads((home / "build_complete.json").read_text())
    for rel in build["site_files"]:
        target = ENGINE_DIR / "site" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(home / "site" / rel, target)
    shutil.copy2(home / "build_complete.json", ENGINE_DIR / "build_complete.json")
    for name, source in (("setup_complete.json", SETUP_WORK / "setup_complete.json"), ("repaired_controls.json", setup["controls"]), ("weight_controls.json", setup["weight_controls"])):
        shutil.copy2(source, ENGINE_DIR / name)

DRIVE_PATH = pathlib.Path(DRIVE_CHECKPOINT_DIR) if DRIVE_CHECKPOINT_DIR else None
if DRIVE_PATH is not None:
    try:
        from google.colab import drive
        if not pathlib.Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        DRIVE_PATH.mkdir(parents=True, exist_ok=True)
        print("Drive checkpoint directory:", DRIVE_PATH)
    except Exception as error:
        print("Drive unavailable; continuing with local checkpoints only:", error)
        DRIVE_PATH = None
'''

ASSIGNMENT_TEMPLATE = '''# Explicit job/chain assignment for this notebook.
import json

rows = []
for unit in assignment["units"]:
    rows.append({
        "run_id": unit["run_id"], "job": unit["stat_job_id"],
        "phase": unit["job"]["phase"], "setting": unit["job"]["setting"],
        "N": unit["job"].get("N", 500), "degree": unit["job"].get("linearity_degree"),
        "rep": unit["job"]["rep"], "budget": unit["budget_key"],
        "chain": unit["chain_index"], "seed": unit["chain_seed"],
        "trace": unit["trace"],
        "est_minutes": round(unit["est_seconds"] / 60, 1)})
print("Purpose:", assignment["purpose"], "| wave", assignment["wave"],
      "shard", assignment["shard"], "of", assignment["shard_count"])
print("Units:", len(rows), "| estimated total hours:",
      round(assignment["estimated_seconds"] / 3600, 2))
print(json.dumps(rows, indent=1))
'''

RUN_TEMPLATE = '''# Bounded execution with a background partial-checkpoint archive outside OUT.
# The controller defers units that do not fit the window; it cannot interrupt a
# sampler mid-chain.  Hard-stop deadline and checkpoint policy are printed.
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import zipfile

RUNNER = ROOT / "JBES_Submission/recovery_repaired_20261008/tools/recovery_runner.py"
PARTIAL = WORKSPACE / f"didbcf_recovery_{LABEL}_partial.zip"
session_start = time.time()
hard_deadline = session_start + HOURS * 3600 - RESERVE_MIN * 60
print("Session start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_start)))
print("Hard stop (no new units):",
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hard_deadline)))
print("Partial checkpoint archive (outside OUT):", PARTIAL)
print("Background checkpoint interval:", CHECKPOINT_EVERY_MIN, "minutes")

zip_lock = threading.Lock()

def _refresh_partial():
    tmp = PARTIAL.with_name(PARTIAL.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_dir():
                continue
            if any(part.startswith("attempt_") for part in path.parts) and path.name not in (
                    "complete.json", "started.json"):
                if not (path.parent / "complete.json").exists():
                    continue
            archive.write(path, str(path.relative_to(OUT)))
    os.replace(tmp, PARTIAL)
    if DRIVE_PATH is not None:
        shutil.copy2(PARTIAL, DRIVE_PATH / PARTIAL.name)

def refresh_partial():
    with zip_lock:
        _refresh_partial()

stop = threading.Event()

def checkpoint_loop():
    while not stop.wait(CHECKPOINT_EVERY_MIN * 60):
        try:
            refresh_partial()
            print("Partial checkpoint refreshed:", PARTIAL, flush=True)
        except Exception as error:
            print("Checkpoint refresh failed (sampling continues):", error,
                  flush=True)

checkpoint_thread = threading.Thread(target=checkpoint_loop, daemon=True)
checkpoint_thread.start()
try:
    returncode = subprocess.call(
        [sys.executable, str(RUNNER), "run",
         "--assignment", str(ROOT / "recovery_bundle/assignment.json"),
         "--out", str(OUT), "--hours", str(HOURS),
         "--reserve-min", str(RESERVE_MIN)])
finally:
    stop.set()
    checkpoint_thread.join(timeout=30)
    refresh_partial()
print("Runner return code:", returncode)
print("Return code 0 = all units complete; 2 = some units deferred to a later "
      "session (rerun with RESUME_ZIP or a fresh notebook); 1 = failures or "
      "conflicts need inspection before rerunning.")
'''

FINAL_TEMPLATE = '''# Diagnostics and integrity summary, then one archive outside OUT and the
# automatic Colab download fallback.
import json
import pathlib
import shutil

status_path = OUT / "run_status.json"
summary = json.loads(status_path.read_text())["summary"] if status_path.exists() else {}
completed = sum(1 for p in OUT.glob("runs/*/attempt_*/complete.json"))
incomplete = sum(1 for p in OUT.glob("runs/*/attempt_*") if not p.name.startswith("."))
print("Completed run packages:", completed)
print("Status summary:", json.dumps(summary, indent=2))
for path in sorted(OUT.glob("runs/*/attempt_*/complete.json")):
    record = json.loads(path.read_text())
    print(" ", record["run_id"], record.get("kind"), record.get("chain_index"),
          f"{record.get('seconds', float('nan')):.1f}s",
          f"{record.get('peak_rss_mb', float('nan')):.0f}MiB")

archive_stem = str(WORKSPACE / f"didbcf_recovery_{LABEL}")
output_file = shutil.make_archive(archive_stem, "zip", str(OUT))
print("Output archive (outside OUT):", output_file)
print("Size (MiB):", round(pathlib.Path(output_file).stat().st_size / 2**20, 2))
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
    print("Output archive:", output_file)
print("Rerun this notebook with RESUME_ZIP set to this archive to continue "
      "deferred units; verified completed chains are skipped, incomplete "
      "attempts are retained and restarted in a new attempt directory.")
'''


def build_notebook(assignment: dict, assignment_path: Path, manifest: dict,
                   manifest_path: Path, purpose: str, hours: float,
                   include_protocol: bool) -> dict:
    label = (f"repaired_{manifest['manifest_hash'][:12]}_{'pilot' if purpose == 'diagnostic_pilot' else purpose}"
             f"_w{assignment['wave']:02d}_s{assignment['shard']:02d}")
    bundle_b64 = build_bundle(assignment, assignment_path, manifest_path,
                              manifest)
    import hashlib
    payload = base64.b64decode(bundle_b64)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        bundle_manifest_sha = hashlib.sha256(archive.read("recovery_bundle/bundle_manifest.json")).hexdigest()
    pins = " ".join(f"{name}=={version}" for name, version
                    in assignment["dependencies"].items())
    context = {
        "@@HOURS@@": str(hours), "@@LABEL@@": label,
        "@@RECORDED_PYTHON@@": str(assignment.get("python_recorded")),
        "@@PINS@@": repr(pins.split()),
        "@@BUNDLE_B64@@": bundle_b64,
        "@@BUNDLE_MANIFEST_SHA256@@": bundle_manifest_sha,
        "@@ZIP_SHA@@": hashlib.sha256(payload).hexdigest(),
        "@@ASSIGNMENT_HASH@@": assignment["assignment_hash"],
    }

    def render(template):
        for key, value in context.items():
            template = template.replace(key, value)
        return template

    unit_phase = (assignment["units"][0]["job"]["phase"]
                  if assignment["units"] else "n/a")
    readme_text = f"""# {PURPOSE_TITLES.get(purpose, purpose)}

This notebook is one independent repaired-source Colab worker. It compiles an isolated stochtree engine, verifies source archives and native controls, and runs one chain worker at a time with one numeric thread. No global stochtree wheel is installed. It embeds its own frozen source
bundle, manifest and explicit job/chain assignment; it does not need any other
notebook's files or an editable online checkout.

* phase: `{unit_phase}` | purpose: `{assignment['purpose']}`
* wave {assignment['wave']} / shard {assignment['shard']} of {assignment['shard_count']}
* units: {len(assignment['units'])} | estimated sampling: {round(assignment['estimated_seconds'] / 3600, 2)} hours
* parent manifest: `{manifest['parent_manifest']}`
* manifest: `{manifest['manifest_hash']}`
* assignment: `{assignment['assignment_hash']}`

Run the cells top to bottom. Completed chains are checkpointed atomically;
deferred units can be run later by setting `RESUME_ZIP`. A partial checkpoint
archive is refreshed every 20 minutes outside the output directory, but a
forced runtime disconnect cannot be caught: stay connected and download the
final archive.
"""
    cells = [
        cell("markdown", readme_text, "purpose"),
        cell("code", render(CONFIG_TEMPLATE), "configuration"),
        cell("code", render(INSTALL_TEMPLATE), "pinned-setup"),
        cell("code", render(EXTRACT_TEMPLATE), "bundle-verify"),
        cell("code", render(RESUME_TEMPLATE), "resume-and-resources"),
        cell("code", render(ASSIGNMENT_TEMPLATE), "assignment"),
        cell("code", render(RUN_TEMPLATE), "bounded-execution"),
        cell("code", render(FINAL_TEMPLATE), "summary-archive-download"),
    ]
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "cells": cells,
    }
    return notebook, label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True,
                        help="directory containing wave_*/shard_*.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=NOTEBOOKS)
    parser.add_argument("--zip-name", default=None)
    parser.add_argument("--zip", action="store_true", help="optional notebook archive; individual .ipynb files are always primary")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--extra", type=Path, nargs="*", default=None,
                        help="extra documents to include in the notebook zip")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    assignment_paths = sorted(args.assignments.glob("wave_*/shard_*.json"))
    if not assignment_paths:
        raise SystemExit(f"No assignments under {args.assignments}")
    args.out.mkdir(parents=True, exist_ok=True)
    notebooks = []
    for path in assignment_paths:
        assignment = json.loads(path.read_text())
        purpose = assignment["purpose"]
        notebook, label = build_notebook(assignment, path, manifest,
                                         args.manifest, purpose, args.hours,
                                         include_protocol=False)
        prefix = "00_setup_smoke" if purpose == "smoke_setup" else "01_budget_pilot" if purpose == "diagnostic_pilot" else "03_production"
        target = args.out / f"{prefix}_w{assignment['wave']:02d}_s{assignment['shard']:02d}.ipynb"
        target.write_text(json.dumps(notebook, indent=1))
        notebooks.append(target)
    readme = args.out / "NOTEBOOK_SET_README.md"
    readme.write_text(
        f"# Recovery notebook set\n\n{len(notebooks)} independent notebooks "
        f"for purpose `{manifest['purpose']}`. Open one per Colab account; "
        "each embeds the same frozen bundle and its own assignment.\n\n"
        "Run top to bottom, stay connected, and keep the downloaded zip for "
        "collection.\n")
    import hashlib
    index = [{"notebook": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes":target.stat().st_size} for target in notebooks]
    (args.out / "notebook_index.json").write_text(json.dumps(index, indent=2) + "\n")
    if args.zip:
        zip_name = args.zip_name or f"didbcf_{manifest['purpose']}_notebooks"
        archive = args.out.parent / f"{zip_name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for target in notebooks: z.write(target, target.name)
            z.write(readme, readme.name)
    print(json.dumps({"notebooks":len(notebooks), "folder":str(args.out), "index":str(args.out / "notebook_index.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
