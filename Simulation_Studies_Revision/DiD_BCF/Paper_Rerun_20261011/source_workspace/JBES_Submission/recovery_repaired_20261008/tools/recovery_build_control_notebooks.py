"""Create individual collection and accepted-budget preparation notebooks."""
from __future__ import annotations
import json
from pathlib import Path
from recovery_common import RECOVERY
from recovery_build_notebooks import build_notebook, cell

UPLOAD = '''# Pilot: select the six returned result ZIPs. Production: select a
# frozen manifest_production.json and returned result ZIPs (or use Drive paths).
# INCOMING_PATHS may also point to files already mounted from Google Drive.
INCOMING_PATHS = []
PRODUCTION_MANIFEST = None
ANALYSIS_ATTEMPT = "analysis01"
ANALYSIS = WORKSPACE / ("didbcf_repaired_@@MODE@@_" + ANALYSIS_ATTEMPT)
INCOMING = ANALYSIS / "incoming"
INCOMING.mkdir(parents=True, exist_ok=True)
if not INCOMING_PATHS:
    try:
        from google.colab import files
        uploaded = files.upload()
        for name, payload in uploaded.items():
            target = INCOMING / Path(name).name
            if target.exists() and target.read_bytes() != payload:
                raise ValueError("Conflicting uploaded file: " + name)
            target.write_bytes(payload)
            if target.suffix == ".zip":
                INCOMING_PATHS.append(str(target))
            elif target.name == "manifest_production.json":
                PRODUCTION_MANIFEST = str(target)
        del uploaded
    except ImportError:
        print("Local runtime: set INCOMING_PATHS to the returned ZIP paths.")
if not INCOMING_PATHS:
    raise ValueError("Supply returned result ZIPs before collection")
ANALYSIS.mkdir(parents=True, exist_ok=True)
'''

COLLECT = '''# Strict archive, binary, source, prior, seed and draw verification.
MODE = "@@MODE@@"
if MODE == "pilot":
    manifest_file = ANALYSIS / "manifest_pilot.json"
    shutil.copy2(ROOT / "recovery_bundle/manifest.json", manifest_file)
else:
    if not PRODUCTION_MANIFEST:
        raise ValueError("Supply the accepted manifest_production.json")
    manifest_file = ANALYSIS / "manifest_production.json"
    shutil.copy2(PRODUCTION_MANIFEST, manifest_file)
collection = ANALYSIS / "collection"
subprocess.run([sys.executable, str(tools / "recovery_collect.py"),
                "--manifest", str(manifest_file), "--incoming", *map(str, INCOMING_PATHS),
                "--out", str(collection)], check=True)
status = json.loads((collection / "status.json").read_text())
print(json.dumps({k:v for k,v in status.items() if k not in ("problems", "duplicates", "smoke", "missing_run_ids", "incomplete_job_budget_pairs")}, indent=2))
if MODE == "pilot":
    report = ANALYSIS / "report"
    subprocess.run([sys.executable, str(tools / "recovery_report_pilot.py"),
                    "--manifest", str(manifest_file), "--collection", str(collection),
                    "--incoming", *map(str, INCOMING_PATHS), "--out", str(report)], check=True)
    from IPython.display import Markdown, display
    display(Markdown((report / "PILOT_ACCEPTANCE.md").read_text()))
# Keep the frozen source bundle with the analysis handoff.
(ANALYSIS / "source_bundle.zip").write_bytes(base64.b64decode(BUNDLE_B64))
'''

GATE = '''# Optional second pass after reading the report and all trace figures.
# Leaving ACCEPTED_BUDGET=None exports the report without scheduling production.
# Select by Monte Carlo precision and stability, never by true effects/coverage.
ACCEPTED_BUDGET = None
TRACE_REVIEW_NOTE = ""
# Individual production notebooks are copied into this Drive folder. This avoids
# packaging notebook inputs as ZIPs or hitting Colab's multiple-download cap.
PRODUCTION_DRIVE_DIR = None
if ACCEPTED_BUDGET is None:
    print("Pilot report prepared. Production awaits an accepted budget and written trace review.")
else:
    from recovery_accept import record_decision
    note_path = ANALYSIS / "trace_review.md"
    note_path.write_text(TRACE_REVIEW_NOTE)
    decision_path = ANALYSIS / "accepted_budget.json"
    decision = record_decision(manifest_file, collection, report, ACCEPTED_BUDGET,
                               note_path, decision_path)
    # Recreate the pinned planning inputs from this exact source bundle.
    from recovery_common import MANIFESTS
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_file, MANIFESTS / "manifest_pilot.json")
    from recovery_plan import build_missing
    build_missing()
    subprocess.run([sys.executable, str(tools / "prepare_campaign.py"), "production",
                    "--decision", str(decision_path), "--setup-record",
                    str(SETUP_WORK / "setup_complete.json")], check=True)
    production_home = tools.parent / "production/notebooks"
    outputs = sorted(production_home.glob("*.ipynb"))
    print("Prepared individual production notebooks:", len(outputs), "in", production_home)
    shutil.copy2(MANIFESTS / "manifest_production.json", ANALYSIS / "manifest_production.json")
    shutil.copy2(tools.parent / "assignments/production/shard_summary.csv", ANALYSIS / "production_shard_summary.csv")
    shutil.copy2(production_home / "notebook_index.json", ANALYSIS / "production_notebook_index.json")
    if PRODUCTION_DRIVE_DIR:
        try:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
        except ImportError:
            pass
        destination = Path(PRODUCTION_DRIVE_DIR)
        destination.mkdir(parents=True, exist_ok=True)
        for output in outputs:
            target = destination / output.name
            if target.exists() and target.read_bytes() != output.read_bytes():
                raise ValueError("Use a new Drive folder for a different production release")
            shutil.copy2(output, target)
        shutil.copy2(ANALYSIS / "manifest_production.json", destination / "manifest_production.json")
        print("Individual notebooks saved:", destination)
    elif len(outputs) < 5:
        for output in outputs:
            try:
                from google.colab import files
                files.download(str(output))
                print("Downloaded:", output)
            except Exception as e:
                print("(Not on Colab / download skipped):", e)
    else:
        print("Set PRODUCTION_DRIVE_DIR and rerun this cell to save the individual .ipynb files. You can also return the analysis archive for local notebook preparation.")
'''

EXPORT = '''# Export generated analysis files as one result archive. The individual
# production notebook inputs stay in their folder and are not put in this ZIP.
result_file = WORKSPACE / ("didbcf_repaired_@@MODE@@_analysis_" + ANALYSIS_ATTEMPT + ".zip")
with zipfile.ZipFile(result_file, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(ANALYSIS.rglob("*")):
        if path.is_file() and "incoming" not in path.relative_to(ANALYSIS).parts:
            archive.write(path, str(path.relative_to(ANALYSIS)))
print("Analysis result:", result_file)
try:
    from google.colab import files
    files.download(str(result_file))
    print("Downloaded:", result_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''


def main():
    manifest_path = RECOVERY / "manifests/manifest_pilot.json"
    assignment_path = RECOVERY / "assignments/pilot/wave_00/shard_00.json"
    manifest = json.loads(manifest_path.read_text())
    assignment = json.loads(assignment_path.read_text())
    out = RECOVERY / "upload_to_colab"
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("pilot", "production"):
        base, _ = build_notebook(assignment, assignment_path, manifest, manifest_path,
                                "collection_" + mode, 8., include_protocol=False)
        intro = ("Collect all six pilot result ZIPs, inspect diagnostics and traces, and optionally freeze a production budget. Production notebooks are generated individually after acceptance and can be saved to a Drive folder."
                 if mode == "pilot" else "Collect the returned production ZIPs under the accepted manifest. For a large campaign, point INCOMING_PATHS to mounted Drive files instead of uploading all bytes at once. Collect a subset to inspect progress, then rerun on a complete disjoint set for final metrics.")
        cells = [cell("markdown", "# DiD-BCF repaired " + mode + " collection\n\n" + intro + "\n\nThis notebook does not launch statistical fits. Run the setup cells, then supply result paths. Use a new ANALYSIS_ATTEMPT for a different input set.", "purpose")]
        cells += base["cells"][1:4]
        cells += [cell("code", UPLOAD.replace("@@MODE@@", mode), "input-results"),
                  cell("code", COLLECT.replace("@@MODE@@", mode), "verified-collection")]
        if mode == "pilot":
            cells.append(cell("code", GATE, "accepted-budget-preparation"))
        cells.append(cell("code", EXPORT.replace("@@MODE@@", mode), "download-analysis"))
        base["cells"] = cells
        name = "02_collect_pilot_prepare_production.ipynb" if mode == "pilot" else "04_collect_production.ipynb"
        (out / name).write_text(json.dumps(base, indent=1))
    print(json.dumps({"control_notebooks":2, "folder":str(out)}))

if __name__ == "__main__":
    main()
