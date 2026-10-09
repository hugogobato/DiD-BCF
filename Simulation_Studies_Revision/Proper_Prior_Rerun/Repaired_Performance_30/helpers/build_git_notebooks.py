"""Generate 48 loose Colab notebooks pinned to a committed helper revision."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from portable_runtime import PACKAGE, atomic, sha

RELATIVE_PACKAGE = "Simulation_Studies_Revision/Proper_Prior_Rerun/Repaired_Performance_30"
REPOSITORY = "https://github.com/hugogobato/DiD-BCF.git"


def cell(kind,source,key):
    out = dict(cell_type=kind,id=key,metadata={},source=source.splitlines(keepends=True))
    if kind=="code":
        ast.parse(source)
        out.update(execution_count=None,outputs=[])
    return out


CONFIG = '''# CPU runtime; no GPU needed. Run all cells in order.
import os, sys, time, platform, json, subprocess, shutil, signal, zipfile
from pathlib import Path
SESSION_STARTED = time.monotonic()  # Includes cloning, pip and compilation.
SESSION_HOURS = 9.5
RESERVE_MIN = 45.0
MAX_WORKERS = 2  # Automatically reduced for CPU quota and available RAM.
CHECKPOINT_EVERY_MIN = 20.0
RESUME_ZIP = None  # Set to the previous result ZIP from THIS notebook only.
DRIVE_CHECKPOINT_DIR = None  # Optional: "/content/drive/MyDrive/didbcf_30"
SOURCE_COMMIT = "@@COMMIT@@"
EXPECTED_ASSIGNMENT_HASH = "@@ASSIGNMENT_HASH@@"
LABEL = "didbcf_repaired_30_remaining_@@SHARD@@"
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
if not (3,12) <= sys.version_info[:2] <= (3,14):
    raise RuntimeError("Use Python 3.12 through 3.14 for the pinned numerical dependencies")
WORK = Path("/content") if Path("/content").is_dir() else Path.cwd() / "didbcf_colab_transfer"
WORK.mkdir(parents=True,exist_ok=True)
ROOT = WORK / ("didbcf_source_" + SOURCE_COMMIT[:12])
OUT = WORK / "didbcf_outputs" / LABEL
SETUP_WORK = WORK / (LABEL + "_setup_" + str(time.time_ns()))
OUT.mkdir(parents=True,exist_ok=True)
print("Python",platform.python_version(),"| source",SOURCE_COMMIT,"| notebook",LABEL)
print("Total session window",SESSION_HOURS,"hours, including setup; export reserve",RESERVE_MIN,"minutes")
'''

CLONE = '''# Only the frozen portable package is checked out. Every notebook is independent.
REPOSITORY = "@@REPOSITORY@@"
RELATIVE_PACKAGE = "@@PACKAGE@@"
if not (ROOT / ".git").exists():
    if ROOT.exists():
        raise RuntimeError("Source directory exists without Git metadata; use a fresh directory")
    subprocess.run(["git","clone","--filter=blob:none","--no-checkout",REPOSITORY,str(ROOT)],check=True,timeout=600)
subprocess.run(["git","-C",str(ROOT),"sparse-checkout","init","--cone"],check=True,timeout=120)
subprocess.run(["git","-C",str(ROOT),"sparse-checkout","set",RELATIVE_PACKAGE],check=True,timeout=120)
subprocess.run(["git","-C",str(ROOT),"checkout","--detach",SOURCE_COMMIT],check=True,timeout=600)
actual_commit = subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
assert actual_commit == SOURCE_COMMIT, "Git source revision differs from notebook pin"
PACKAGE = ROOT / RELATIVE_PACKAGE
HELPER = PACKAGE / "helpers/portable_runtime.py"
ASSIGNMENT = PACKAGE / "assignments/wave_00/shard_@@SHARD@@.json"
assignment = json.loads(ASSIGNMENT.read_text())
assert assignment["assignment_hash"] == EXPECTED_ASSIGNMENT_HASH
subprocess.run([sys.executable,str(HELPER),"verify"],check=True)
'''

INSTALL = '''# Install the exact Python dependencies; the repaired stochtree engine is built
# separately into an isolated site directory, rather than installed globally.
import importlib.metadata
PINS = @@PINS@@
remaining = SESSION_HOURS*3600 - (time.monotonic()-SESSION_STARTED) - RESERVE_MIN*60
if remaining <= 300:
    raise RuntimeError("Too little time remains for setup; start a new session")
subprocess.run([sys.executable,"-m","pip","install","--quiet",*PINS],check=True,timeout=remaining)
for pin in PINS:
    name,version = pin.split("==")
    assert importlib.metadata.version(name)==version, "Dependency pin mismatch: " + name
print("Verified dependency versions",PINS)
# Scientific imports and sampling run in fresh Python processes, so the notebook
# kernel cannot retain an old NumPy or native stochtree module after installation.
'''

DRIVE = '''# Optional durable checkpoints. Local /content files disappear on runtime reset.
DRIVE_PATH = Path(DRIVE_CHECKPOINT_DIR) if DRIVE_CHECKPOINT_DIR else None
if DRIVE_PATH is not None:
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        DRIVE_PATH.mkdir(parents=True,exist_ok=True)
        print("Checkpoint copies:",DRIVE_PATH)
    except Exception as e:
        print("Drive checkpoint copies unavailable:",e)
        DRIVE_PATH = None
'''

SETUP = '''# Build with one compilation worker; verify native and eligible-feature controls.
# Resume restores only identical files and reuses that shard's verified binary.
command = [sys.executable,str(HELPER),"setup","--work",str(SETUP_WORK),
           "--cache",str(WORK / "didbcf_pinned_archives"),"--out",str(OUT),
           "--assignment",str(ASSIGNMENT)]
if RESUME_ZIP:
    command += ["--resume",str(RESUME_ZIP)]
remaining = SESSION_HOURS*3600 - (time.monotonic()-SESSION_STARTED) - RESERVE_MIN*60
if remaining <= 300:
    raise RuntimeError("Too little time remains for compilation; start a new session")
process = subprocess.Popen(command,start_new_session=True)
try:
    returncode = process.wait(timeout=remaining)
    if returncode:
        raise RuntimeError("Engine setup failed; inspect its logs")
except BaseException:
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    failure = WORK / (LABEL + "_setup_failure")
    failure.mkdir(exist_ok=True)
    for folder in (SETUP_WORK,OUT):
        if folder.exists():
            shutil.copytree(folder,failure / folder.name,dirs_exist_ok=True)
    output_file = shutil.make_archive(str(WORK / (LABEL + "_setup_failure")),"zip",str(failure))
    try:
        from google.colab import files
        files.download(output_file)
        print("Downloaded:",output_file)
    except Exception as e:
        print("(Not on Colab / download skipped):",e)
    raise
SETUP_JSON = SETUP_WORK / "setup_complete.json"
print("Verified isolated engine. Completed chains survive a same-notebook resume.")
'''

ASSIGNMENTS = '''# This assignment excludes the nine completed local chains.
print("Remaining independent chains:",len(assignment["units"]))
print("Estimated CPU hours:",round(assignment["estimated_seconds"]/3600,2))
print("Up to two workers with one numerical thread each; RAM/CPU admission is automatic.")
for u in assignment["units"]:
    job=u["job"]
    print(u["run_id"],job["setting"],"N",job["N"],"degree",job["linearity_degree"],
          "rep",job["rep"],u["budget_key"],"chain",u["chain_index"],"seed",u["chain_seed"])
'''

RUN_EXPORT = '''# Sampling, background checkpointing, status and one automatic ZIP download.
import threading
PARTIAL = WORK / (LABEL + "_partial.zip")
zip_lock = threading.Lock()
stop_checkpoint = threading.Event()

def refresh_checkpoint():
    with zip_lock:
        temporary = PARTIAL.with_name(PARTIAL.name + ".tmp")
        with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in sorted(OUT.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                # A completion is published last. Only those chains contribute draws.
                if any(part.startswith("attempt_") for part in path.parts):
                    if not (path.parent / "complete.json").exists() and path.name not in ("started.json",):
                        continue
                archive.write(path,str(path.relative_to(OUT)))
        os.replace(temporary,PARTIAL)
        if DRIVE_PATH is not None:
            try:
                drive_tmp = DRIVE_PATH / (PARTIAL.name + ".tmp")
                shutil.copy2(PARTIAL,drive_tmp)
                os.replace(drive_tmp,DRIVE_PATH / PARTIAL.name)
            except Exception as e:
                print("Drive copy failed; local checkpoint retained:",e,flush=True)

def checkpoint_loop():
    while not stop_checkpoint.wait(CHECKPOINT_EVERY_MIN*60):
        try:
            refresh_checkpoint()
            print("Checkpoint updated:",PARTIAL,flush=True)
        except Exception as e:
            print("Checkpoint refresh failed:",e,flush=True)

thread = threading.Thread(target=checkpoint_loop,daemon=True)
process = None
returncode = None
refresh_checkpoint()
thread.start()
try:
    hours_left = SESSION_HOURS - (time.monotonic()-SESSION_STARTED)/3600
    if hours_left <= RESERVE_MIN/60:
        raise RuntimeError("Sampling window exhausted during setup; download and resume in a new session")
    process = subprocess.Popen([sys.executable,str(HELPER),"run","--assignment",str(ASSIGNMENT),
        "--out",str(OUT),"--setup",str(SETUP_JSON),"--workers",str(MAX_WORKERS),
        "--hours",str(hours_left),"--reserve-min",str(RESERVE_MIN)],start_new_session=True)
    returncode = process.wait()
except KeyboardInterrupt:
    print("Stopping workers and exporting completed chains.")
finally:
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    stop_checkpoint.set()
    thread.join()
    refresh_checkpoint()
    output_file = str(WORK / (LABEL + ".zip"))
    shutil.copy2(PARTIAL,output_file)
    if DRIVE_PATH is not None:
        try:
            shutil.copy2(output_file,DRIVE_PATH / Path(output_file).name)
        except Exception as e:
            print("Drive copy failed; final ZIP available for download:",e)
    status_path = OUT / "run_status.json"
    if status_path.exists():
        print("Status:",json.dumps(json.loads(status_path.read_text()).get("summary",{}),indent=2))
    print("Runner code:",returncode,"| 0 complete, 2 deferred, 1 inspect failures")
    print("Result ZIP:",output_file,"| MiB",round(Path(output_file).stat().st_size/2**20,1))
    try:
        from google.colab import files
        files.download(output_file)
        print("Downloaded:",output_file)
    except Exception as e:
        print("(Not on Colab / download skipped):",e)
    print("Keep the final ZIP from each notebook. To resume, set RESUME_ZIP to this same notebook's ZIP.")
    print("Collect only the newest final ZIP per notebook; partial snapshots contain repeated completions.")
'''


def build(commit):
    from portable_runtime import verify_package
    transfer,manifest = verify_package()
    index = []
    for rel,assignment_hash in sorted(transfer["assignments"].items()):
        assignment = json.loads((PACKAGE / rel).read_text())
        shard = f"{assignment['shard']:02d}"
        context = {"@@COMMIT@@":commit,"@@ASSIGNMENT_HASH@@":assignment_hash,"@@SHARD@@":shard,
                   "@@REPOSITORY@@":REPOSITORY,"@@PACKAGE@@":RELATIVE_PACKAGE,
                   "@@PINS@@":repr([f"{k}=={v}" for k,v in manifest["dependencies"].items()])}
        def render(template):
            for key,value in context.items():
                template = template.replace(key,value)
            if "@@" in template:
                raise ValueError("Unexpanded notebook placeholder")
            return template
        text = f"""# Repaired DiD-BCF: 30-replication comparison, notebook {shard}/47

Upload this loose notebook to a **CPU** Colab runtime and run all cells. Run each of the 48 numbered notebooks once, across your accounts. This study compares both repaired budgets with saved proper-prior results in nine fixed cells, with 30 replications per cell. Its scope is 540 repaired fits, not the full paper campaign.

Nine completed local chains are already preserved. This notebook runs **{len(assignment['units'])} remaining independent chains** and downloads one result ZIP. Estimated sampling is approximately 3–5 hours with two workers or 6–8 hours with one, depending on Colab speed. The 9.5-hour window includes setup and reserves 45 minutes for export. All completed draws are retained; diagnostics do not filter replications.

The helpers, worker, exact source archives, dependencies and assignment are pinned to Git revision `{commit}` and verified before sampling. Compilation uses one worker; sampling admits up to two workers according to CPU quota and available RAM. The repaired stochtree engine is built into an isolated directory. No global stochtree installation or GPU is needed.

For a fresh run leave `RESUME_ZIP=None`. Optional Google Drive checkpoint copies are configured in the first cell. A forced runtime reset can lose files in `/content`, so keep downloaded ZIPs or enable Drive copies. To resume, use the previous ZIP from this same notebook and the same Python version. An interrupted chain restarts from its original seed; a completed chain is verified and skipped. Collect the newest final ZIP from each notebook, together with the nine local chains included by the collector automatically.
"""
        templates = [("code",CONFIG,"configuration"),("code",CLONE,"pinned-source"),("code",INSTALL,"dependencies"),
                     ("code",DRIVE,"drive-checkpoints"),("code",SETUP,"isolated-engine"),
                     ("code",ASSIGNMENTS,"assignment"),("code",RUN_EXPORT,"run-and-download")]
        notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
                      language_info=dict(name="python",version="3.12"),colab=dict(name=f"remaining_{shard}.ipynb",provenance=[]),
                      didbcf_transfer=dict(source_commit=commit,transfer_hash=transfer["transfer_hash"],assignment_hash=assignment_hash)),
                      cells=[cell("markdown",text,"purpose")]+[cell(kind,render(source),key) for kind,source,key in templates])
        path = PACKAGE / f"Colab/remaining_{shard}.ipynb"
        atomic(path,notebook)
        index.append(dict(notebook=str(path.relative_to(PACKAGE)),sha256=sha(path),shard=assignment["shard"],
                          source_commit=commit,assignment_hash=assignment_hash,units=len(assignment["units"])))
    if len(index)!=48 or sum(row["units"] for row in index)!=1881:
        raise ValueError("Notebook assignment coverage changed")
    atomic(PACKAGE / "Colab/index.json",dict(source_commit=commit,transfer_hash=transfer["transfer_hash"],notebooks=index))
    print(json.dumps(dict(status="PASS",notebooks=len(index),units=sum(r["units"] for r in index),
                         source_commit=commit,all_code_cells_parse=True),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit",required=True)
    args=parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}",args.source_commit):
        raise ValueError("Pin a full 40-character Git commit")
    build(args.source_commit)
