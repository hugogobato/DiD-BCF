"""Record a production budget only from verified complete pilot diagnostics."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from recovery_common import MANIFESTS, bundle_hashes, digest, pin_dependencies, sha256_file, write_json_atomic

def required_parameter(name):
    return (name in {"ATT", "sigma2", "leaf_scale_level", "leaf_scale_trend", "rfx_sigma",
                     "bcf_global_var_samples", "bcf_leaf_scale_mu_samples", "bcf_leaf_scale_tau_samples", "bcf_rfx_sigma"}
            or name.startswith(("GATT_", "ES_", "PRE_", "PRE_SUB_", "PRE_SUBC_")))

def expected_parameters(job):
    from recovery_core import load_panel
    from recovery_collect import _PretrendShim, _surface_heads, pretrend_draws
    from did_bcf_revision.pretrend import quantile_subgroups
    panel = load_panel(job)
    if job["kind"] in ("pretrend", "application_pretrend"):
        groups = quantile_subgroups("lpop") if job["kind"] == "application_pretrend" else True
        _, targets = pretrend_draws(_PretrendShim(panel, np.zeros((len(panel), 2))), groups)
        return set(targets) | {"bcf_global_var_samples", "bcf_leaf_scale_mu_samples", "bcf_rfx_sigma"}
    keys = set(_surface_heads(panel, np.zeros((len(panel), 2))))
    keys.update({"sigma2", "leaf_scale_level", "leaf_scale_trend"})
    if job.get("spec", "").endswith("unit"):
        keys.add("rfx_sigma")
    return keys

def record_decision(manifest_path, collection, report, budget, trace_note, out):
    from recovery_report_pilot import acceptance_table, stability_table
    manifest = json.loads(Path(manifest_path).read_text())
    if digest({k:v for k,v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("Pilot manifest digest mismatch")
    if bundle_hashes() != manifest["source_hashes"]:
        raise ValueError("Pilot source bundle changed; prepare a versioned pilot")
    if budget not in manifest["budget_keys"]:
        raise ValueError("Budget was not run in this pilot")
    collection, report = Path(collection), Path(report)
    status = json.loads((collection / "status.json").read_text())
    if status["complete_units"] != status["expected_units"] or status["conflicting_units"] or status["duplicate_units"] or status["smoke_rejected"]:
        raise ValueError("Pilot collection must be complete and free of conflicts/duplicates/smoke")
    if status.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("Collection belongs to a different pilot manifest")
    meta = json.loads((report / "pilot_report_meta.json").read_text())
    if meta.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("Report belongs to a different pilot manifest")
    for name, expected in meta["files"].items():
        if sha256_file(collection / name) != expected:
            raise ValueError("Report input changed: " + name)
    diagnostics = pd.read_csv(collection / "chain_diagnostics.csv")
    jobs = {job["id"]: job for job in manifest["jobs"]}
    selected = diagnostics[diagnostics.budget_key == budget]
    if selected.duplicated(["job_id", "parameter"]).any():
        raise ValueError("Duplicate diagnostic parameter rows")
    for jid, job in jobs.items():
        for key in manifest["budget_keys"]:
            present_at_budget = set(diagnostics.loc[(diagnostics.job_id == jid) & (diagnostics.budget_key == key), "parameter"])
            if expected_parameters(job) - present_at_budget:
                raise ValueError("Missing required target/budget diagnostic rows: " + jid + " " + key)
        present = set(selected.loc[selected.job_id == jid, "parameter"])
        absent = expected_parameters(job) - present
        if absent:
            raise ValueError("Missing required diagnostic parameters: " + jid + " " + repr(sorted(absent)))
    screened = acceptance_table(selected[selected.parameter.map(required_parameter)], jobs)
    if screened.empty or not screened.pass_all.all():
        raise ValueError("Required pilot precision screens failed; retain failed cases and develop the next pilot")
    stability = stability_table(diagnostics[diagnostics.parameter.map(required_parameter)], jobs)
    pairs = stability[(stability["first"] == budget) | (stability["second"] == budget)]
    expected_pairs = sum(len(expected_parameters(job)) for job in jobs.values()) * (len(manifest["budget_keys"]) - 1)
    if len(pairs) < expected_pairs or pairs.empty or pairs[["mean_stability_flag", "endpoint_stability_flag"]].any().any():
        raise ValueError("Prespecified budget stability flags require resolution before production")
    traces = pd.read_csv(collection / "trace_diagnostics.csv")
    if traces.empty or not traces.trace_matches_draws.all():
        raise ValueError("Instrumented traces differ from retained draws")
    note = Path(trace_note).read_text().strip()
    if len(note) < 40:
        raise ValueError("Record the completed trace review and any limitations in a substantive note")
    runs = pd.read_csv(collection / "collected_runs.csv")
    interpreters = sorted(runs.python.dropna().unique())
    if len(interpreters) != 1:
        raise ValueError("Mixed or absent pilot Python interpreters require a versioned environment decision")
    files = [collection / name for name in ("status.json", "chain_diagnostics.csv", "trace_diagnostics.csv", "collected_runs.csv")]
    files += [report / "PILOT_ACCEPTANCE.md", report / "pilot_report_meta.json", Path(trace_note), Path(manifest_path)]
    evidence = {os.path.relpath(path.resolve(), Path(out).parent.resolve()): sha256_file(path) for path in files}
    body = {"schema": "didbcf-repaired-budget-decision-v1", "status": "ACCEPTED",
            "pilot_manifest_hash": manifest["manifest_hash"], "budget_key": budget,
            "budget": manifest["budgets"][budget], "accepted_python": interpreters[0],
            "accepted_dependencies": pin_dependencies(), "engine_specification": manifest["engine_specification"],
            "source_hashes": manifest["source_hashes"], "required_checks": len(screened),
            "trace_review_note": note, "evidence": evidence,
            "scope": "prespecified finite-computation precision screen, not a mixing or coverage theorem"}
    body["decision_hash"] = digest(body)
    write_json_atomic(out, body)
    return body

def validate_decision(path, budget, python):
    body = json.loads(Path(path).read_text())
    if body.get("status") != "ACCEPTED" or digest({k:v for k,v in body.items() if k != "decision_hash"}) != body.get("decision_hash"):
        raise ValueError("A verified accepted-budget decision is required")
    pilot = json.loads((MANIFESTS / "manifest_pilot.json").read_text())
    if bundle_hashes() != pilot["source_hashes"]:
        raise ValueError("Current bundle differs from the accepted pilot")
    for field in ("source_hashes", "engine_specification"):
        if body.get(field) != pilot.get(field):
            raise ValueError("Accepted pilot differs from current production sources: " + field)
    if body.get("pilot_manifest_hash") != pilot["manifest_hash"] or body.get("budget_key") != budget or body.get("accepted_python") != python:
        raise ValueError("Decision budget/interpreter/pilot identity mismatch")
    if body.get("accepted_dependencies") != pin_dependencies() or body.get("budget") != pilot["budgets"][budget]:
        raise ValueError("Decision pins or sampling budget mismatch")
    if not body.get("evidence"):
        raise ValueError("Accepted decision lacks evidence")
    for name, expected in body["evidence"].items():
        evidence_file = Path(name) if Path(name).is_absolute() else Path(path).resolve().parent / name
        if not evidence_file.is_file() or sha256_file(evidence_file) != expected:
            raise ValueError("Gate evidence missing or changed: " + name)
    return body

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--budget", required=True)
    p.add_argument("--trace-review-note", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    record = record_decision(args.manifest, args.collection, args.report, args.budget, args.trace_review_note, args.out)
    print(json.dumps({"status":record["status"], "decision_hash":record["decision_hash"]}))

if __name__ == "__main__":
    main()
