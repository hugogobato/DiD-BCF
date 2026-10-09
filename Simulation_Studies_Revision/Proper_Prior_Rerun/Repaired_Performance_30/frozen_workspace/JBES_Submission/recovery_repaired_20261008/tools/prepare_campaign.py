"""Prepare repaired pilot/smoke notebooks or production from accepted evidence."""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "production"))
    p.add_argument("--setup-record", type=Path)
    p.add_argument("--decision", type=Path)
    args = p.parse_args()
    if args.setup_record:
        setup = json.loads(args.setup_record.read_text())
        if setup["status"] != "PASS":
            raise ValueError("Setup did not pass")
        for env, key in (("DIDBCF_REPAIRED_BUILD", "build"), ("DIDBCF_REPAIRED_CONTROLS", "controls"), ("DIDBCF_WEIGHT_CONTROLS", "weight_controls")):
            os.environ[env] = setup[key]
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/didbcf-repaired-mpl")
    import recovery_common as rc
    import recovery_plan as plan
    from recovery_inventory import load_original_manifest

    def notebooks(assignments, manifest, target, zip_name, hours=8.):
        subprocess.run([sys.executable, str(ROOT / "tools/recovery_build_notebooks.py"),
                        "--assignments", str(assignments), "--manifest", str(manifest),
                        "--out", str(target), "--zip-name", zip_name, "--hours", str(hours),
                        "--extra", str(ROOT / "pilot/pilot_protocol.md")], check=True)

    if args.command == "prepare":
        pilot = plan.build_pilot()
        universe = plan.build_missing()
        plan.build_template(pilot, universe)
        assignments = ROOT / "assignments/pilot"
        written = plan.write_assignments(pilot, "diagnostic_pilot", 6.5, 48, assignments)
        notebooks(assignments, ROOT / "manifests/manifest_pilot.json", ROOT / "pilot/notebooks", "didbcf_repaired_pilot_notebooks")
        original = load_original_manifest()
        choices = [plan.find_job(original["jobs"], id="ae95bf48eea71d8a38f2"),
                   plan.find_job(original["jobs"], phase="application", kind="application", spec="structured_rfx_unit", rep=0),
                   plan.find_job(original["jobs"], phase="application", kind="application_pretrend", rep=0)]
        choices = [{**job, "recovery":{"reference_old_fit_seconds": plan.reference_seconds(job, plan.measured_table()), "diagnostic_only":True}} for job in choices]
        bdg = rc.budget(2, 2, 4, 1, 2, label="SMOKE ONLY, no statistical interpretation")
        smoke = plan._manifest_body("smoke_setup", choices, {"S0_smoke":bdg}, ["S0_smoke"], "SMOKE_NOT_PRODUCTION", True,
                                    ["API, recording, source/binary verification and recovery only; not a budget or performance result."])
        smoke["manifest_hash"] = rc.digest(smoke)
        smoke_path = ROOT / "manifests/manifest_smoke.json"
        rc.write_json_atomic(smoke_path, smoke)
        smoke_assignments = ROOT / "assignments/smoke"
        plan.write_assignments(smoke, "smoke_setup", .5, 48, smoke_assignments)
        notebooks(smoke_assignments, smoke_path, ROOT / "smoke/notebooks", "didbcf_repaired_smoke_notebook", 1.)
        subprocess.run([sys.executable, str(ROOT / "tools/recovery_build_control_notebooks.py")], check=True)
        upload = ROOT / "upload_to_colab"
        for source in [*(ROOT / "smoke/notebooks").glob("*.ipynb"), *(ROOT / "pilot/notebooks").glob("*.ipynb")]:
            shutil.copy2(source, upload / source.name)
        notebook_index = [{"notebook":str(path.name), "sha256":rc.sha256_file(path), "bytes":path.stat().st_size} for path in sorted(upload.glob("*.ipynb"))]
        rc.write_json_atomic(upload / "notebook_index.json", notebook_index)
        print(json.dumps({"pilot_cases":len(pilot["jobs"]), "pilot_units":len(plan.make_units(pilot)),
                          "pilot_notebooks":len(written), "production_jobs":len(universe["jobs"]),
                          "production_status":"requires verified accepted-budget decision"}))
    else:
        if not args.decision:
            raise ValueError("--decision must identify the verified accepted-budget JSON")
        decision = json.loads(args.decision.read_text())
        body = plan.build_production(decision["budget_key"], decision["accepted_python"],
                                     decision["accepted_dependencies"], str(args.decision.resolve()))
        assignments = ROOT / "assignments/production"
        written = plan.write_assignments(body, "production", 6.5, 48, assignments, phase_order=plan.PHASE_ORDER)
        expected = {unit["run_id"] for unit in plan.make_units(body)}
        actual = []
        for path in assignments.glob("wave_*/shard_*.json"):
            actual.extend(unit["run_id"] for unit in json.loads(path.read_text())["units"])
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Production assignments omit or duplicate required run units")
        notebooks(assignments, ROOT / "manifests/manifest_production.json", ROOT / "production/notebooks", "didbcf_repaired_production_notebooks")
        print(json.dumps({"production_jobs":len(body["jobs"]), "units":len(actual), "notebooks":len(written)}))

if __name__ == "__main__":
    main()
