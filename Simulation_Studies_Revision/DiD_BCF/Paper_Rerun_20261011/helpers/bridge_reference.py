"""Export the eight already-saved B3 pilot anchors for paired budget checks."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import PACKAGE, atomic, bootstrap, digest, read, sha
from build_campaign import BRIDGE_IDS

def export(pilot_manifest, outputs, setup):
    bootstrap(setup)
    from recovery_engine import activate, fingerprint
    from recovery_plan import make_units
    from recovery_collect import validate_record, _LazyPayload
    from paper_metrics import summarize_job
    engine = fingerprint(activate())
    pilot = read(pilot_manifest)
    budget_key = "B3_warm2500_ret2000"
    units = {u["run_id"]:u for u in make_units(pilot)
             if u["stat_job_id"] in BRIDGE_IDS and u["budget_key"] == budget_key}
    records = {}
    for path in outputs.glob("runs/*/attempt_*/complete.json"):
        record = read(path)
        if record["run_id"] not in units:
            continue
        loader = _LazyPayload(path.parent, engine=engine)
        unit, problems = validate_record(record,units,pilot,loader)
        if problems:
            raise ValueError(problems)
        records[record["run_id"]] = {"unit":unit,"record":record,"loader":loader,"source":str(path)}
    assert set(records) == set(units)
    results = []
    for job_id in BRIDGE_IDS:
        job = next(j for j in pilot["jobs"] if j["id"] == job_id)
        folder = PACKAGE / "bridge_reference" / "fits" / job_id
        summary = summarize_job(job,budget_key,records,folder)
        atomic(folder / "provenance.json", {"job_id":job_id,"budget_key":budget_key,
               "pilot_manifest_hash":pilot["manifest_hash"],"engine":engine,
               "chains":[e["record"] for e in records.values() if e["unit"]["stat_job_id"] == job_id],
               "outputs":summary["outputs"],"role":"single-anchor supporting comparison; not main Monte Carlo evidence"})
        results.append({"job_id":job_id,**summary})
    atomic(PACKAGE / "bridge_reference/index.json", {"budget_key":budget_key,
           "pilot_manifest_sha256":sha(pilot_manifest),"fits":results,"verified_units":len(records)})
    print("Verified and exported eight saved B3 anchors", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot-manifest","outputs","setup"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    export(args.pilot_manifest,args.outputs,args.setup)
