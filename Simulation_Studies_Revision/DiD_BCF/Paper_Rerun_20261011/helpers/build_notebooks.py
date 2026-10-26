"""Generate loose independent Colab notebooks pinned to a committed package."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
from common import PACKAGE, read

RELATIVE = "Simulation_Studies_Revision/Proper_Prior_Rerun/Paper_Rerun_20261011"

def notebook(cells):
    return {"nbformat":4,"nbformat_minor":5,"metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"},"colab":{"provenance":[]}},
        "cells":[{"cell_type":kind,"id":f"cell-{i:02d}","metadata":{},"source":source.splitlines(keepends=True),
                  **({"outputs":[],"execution_count":None} if kind == "code" else {})}
                 for i,(kind,source) in enumerate(cells)]}

def write(path,cells):
    for kind,source in cells:
        if kind == "code":
            ast.parse(source)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook(cells),indent=1)+"\n")

def setup_cell(commit):
    return f'''import os, sys, subprocess, time
from pathlib import Path
assert (3, 12) <= sys.version_info[:2] <= (3, 14), "Use Python 3.12 to 3.14"
SESSION_STARTED = time.time()
DEADLINE = SESSION_STARTED + 9.5 * 3600
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
SOURCE_COMMIT = {commit!r}
REPO = Path("/content/didbcf_paper_repo") if Path("/content").exists() else Path.cwd() / "didbcf_paper_repo"
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/hugogobato/DiD-BCF.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "fetch", "origin", SOURCE_COMMIT], check=True)
subprocess.run(["git", "-C", str(REPO), "sparse-checkout", "set", {RELATIVE!r}], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", SOURCE_COMMIT], check=True)
actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert actual == SOURCE_COMMIT
PACKAGE = REPO / {RELATIVE!r}
import json
manifest = json.loads((PACKAGE / "manifest.json").read_text())
requirements = [name + "==" + version for name, version in manifest["dependencies"].items()]
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *requirements], check=True)
HELPERS = PACKAGE / "helpers"
print("Pinned sources and dependencies installed:", SOURCE_COMMIT)
'''

def build(commit):
    if not re.fullmatch("[0-9a-f]{40}",commit):
        raise ValueError("A full immutable Git commit is required")
    schedule = read(PACKAGE / "preparation.json")
    manifest = read(PACKAGE / "manifest.json")
    output = PACKAGE / "Colab"
    written = []
    for path in sorted((PACKAGE / "assignments").glob("*.json")):
        assigned = read(path)
        name = assigned["name"]
        wave = "Bridge_Checks" if name.startswith("bridge") else name.split("_shard")[0].replace("main_", "")
        target = output / wave / f"{name}.ipynb"
        intro = f'''# DiD-BCF paper rerun: {name}

This independent notebook runs {len(assigned['job_ids'])} complete fits using the pinned repaired engine, GFR 50, burn-in 0, 500 retained draws per chain, thinning 5, and three chains. It preserves the original panels, priors and seed schedule.

The estimated runtime is {assigned['estimated_wall_hours']:.1f} hours with two CPU workers, including setup. Runtime varies across Colab machines. The session stops before 9.5 hours; rerun this same notebook with its checkpoint if any fits remain. CPU runtime, Python 3.12 to 3.14, and approximately 12 GB RAM are appropriate. The controller uses one worker when RAM or CPU quotas require it.

Download contains paper evaluation summaries, small scalar diagnostics and provenance. Posterior arrays, sweep traces and native binaries stay on the runtime disk. Completed fits are recoverable; an interrupted fit restarts from its original seeds. No fits are excluded based on diagnostics or performance.

The earlier budget was selected for paper performance. It did not pass the previous full-state diagnostic gate and does not certify nominal interval coverage.
'''
        config = f'''# Optional: preserve checkpoints in your own Drive folder.
USE_DRIVE = False
RESUME_ZIP = ""  # Path to this notebook's previously downloaded metrics ZIP.
WORKERS = 2
NAME = {name!r}
WORK = Path("/content") / (NAME + "_work") if Path("/content").exists() else Path.cwd() / (NAME + "_work")
OUT = WORK / "paper_output"
OUTPUT_FILE = WORK / ("didbcf_paper_" + NAME + ".zip")
ASSIGNMENT = PACKAGE / "assignments" / (NAME + ".json")
DRIVE_DIR = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/DiD_BCF_Paper_Rerun_20261011")
    previous = DRIVE_DIR / OUTPUT_FILE.name
    if not RESUME_ZIP and previous.exists():
        RESUME_ZIP = str(previous)
# To restore a local downloaded ZIP, upload it in Colab's Files pane and set
# RESUME_ZIP to its /content/... path before running the next cell.
print("Fits:", {len(assigned['job_ids'])}, "Output:", OUTPUT_FILE)
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
    raise RuntimeError("Execution failed; inspect the job logs and preserve any metrics checkpoint")
status = json.loads((OUT / "bundle/status.json").read_text())
print("Completed:", status["completed"], "/", status["expected"])
if status["missing_job_ids"]:
    print("Resume this same notebook with its ZIP to complete", len(status["missing_job_ids"]), "remaining fits.")
'''
        download = '''output_file = str(OUTPUT_FILE)
assert Path(output_file).is_file(), "No completed metrics archive was created"
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''
        write(target,[("markdown",intro),("code",setup_cell(commit)),("code",config),
                      ("code",prepare),("code",execute),("code",download)])
        written.append(str(target.relative_to(PACKAGE)))
    collector_intro = '''# Collect DiD-BCF paper metrics

Upload the small `didbcf_paper_*.zip` result files when available. This notebook automatically includes the 264 verified earlier fits, validates identities and checksums, reports missing jobs, and exports all tables in one ZIP. Partial collections are clearly marked. This collection covers the main rerun and eight budget checks; the remaining pretrend, application and sensitivity campaign follows those checks.
'''
    collect_code = '''from google.colab import files
uploaded = files.upload()
INPUTS = [Path(name).resolve() for name in uploaded if name.endswith(".zip")]
RESULTS = Path("/content/didbcf_paper_tables")
subprocess.run([sys.executable, str(HELPERS / "collect.py"), *map(str, INPUTS), "--out", str(RESULTS)], check=True)
status = json.loads((RESULTS / "status.json").read_text())
print("Main fits:", status["main_completed"], "/", status["main_expected"])
print("Bridge checks:", status["bridge_completed"], "/", status["bridge_expected"])
import shutil
output_file = shutil.make_archive("/content/didbcf_paper_tables", "zip", RESULTS)
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''
    write(output / "collect_paper_metrics.ipynb",[("markdown",collector_intro),("code",setup_cell(commit)),("code",collect_code)])
    written.append("Colab/collect_paper_metrics.ipynb")
    guide = f'''# Upload guide

These are loose notebooks. Upload each `.ipynb` directly to a CPU Colab runtime and select **Run all**. They clone immutable helper commit `{commit}` and compile an isolated repaired engine. No global stochtree package is installed.

Run the 48 notebooks in `wave_01/`, followed by the 12 in `wave_02/`. Each main notebook targets about eight hours with two workers. Run the four `Bridge_Checks/` notebooks as well; these cover eight non-main anchors and compare against saved B3 summaries. They are shorter and can run alongside the main campaign.

Each notebook downloads one small metrics ZIP. Move returned ZIPs to `../Results/` for local collection. Enable `USE_DRIVE` before execution to persist a metrics checkpoint every 20 minutes. For recovery, run the same notebook again, upload its ZIP through the Files pane and set `RESUME_ZIP`; Drive checkpoints are detected automatically when enabled. If the runtime ends abruptly without Drive or a downloaded checkpoint, its local files are lost. A resumed notebook rebuilds its engine and skips verified complete fits. An interrupted fit restarts from its original seeds.

Collect with `collect_paper_metrics.ipynb`, or locally:

```text
python helpers/collect.py Results/didbcf_paper_*.zip --out Results/collected --require-main
```

The collector includes the 264 reused main fits. A complete main collection has 5,100 fits. Diagnostics do not select or exclude results. The remaining full paper phases still need budget decisions after the eight checks. Existing manuscript numbers are unchanged until final results are available.

| Notebook | Estimated hours with two workers |
| --- | ---: |
'''
    for relative in written[:-1]:
        assigned = read(PACKAGE / "assignments" / (Path(relative).stem + ".json"))
        guide += f"| [{Path(relative).stem}]({relative.removeprefix('Colab/')}) | {assigned['estimated_wall_hours']:.1f} |\n"
    (output / "UPLOAD_GUIDE.md").write_text(guide)
    print("Generated",len(written),"loose notebooks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit",required=True)
    args = parser.parse_args()
    build(args.source_commit)
