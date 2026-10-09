"""Freeze the user's 30-per-cell comparison and individual fallback notebooks."""
from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECOVERY = REPO / "JBES_Submission/recovery_repaired_20261008"
REFERENCE = REPO / "DiD-BCF/Simulation_Studies_Revision/Reruns/proper_ig_v1/analysis_returned"
CELLS = [("B1_baseline", 200, 1), ("B1_baseline", 200, 3),
         ("B1_null", 200, 1), ("B1_selection_obs", 200, 1),
         ("B1_serial_corr", 200, 1), ("D_staggered", 200, 1),
         ("B2_sweep", 800, 1), ("B2_sweep_serial", 100, 3),
         ("B2_sweep_serial", 800, 2)]


def main():
    os.environ["MPLCONFIGDIR"] = str(HERE / "matplotlib_cache")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    setup = json.loads((RECOVERY / "local_runs/20261008_overnight_v1/setup/setup_complete.json").read_text())
    for var, field in (("DIDBCF_REPAIRED_BUILD", "build"), ("DIDBCF_REPAIRED_CONTROLS", "controls"), ("DIDBCF_WEIGHT_CONTROLS", "weight_controls")):
        os.environ[var] = setup[field]
    sys.path.insert(0, str(RECOVERY / "tools"))
    from recovery_common import (BUDGETS,
                                 environment, digest, sha256_file, write_json_atomic)
    from recovery_core import load_panel, panel_hash
    from recovery_plan import _manifest_body, write_assignments
    from recovery_inventory import load_original_manifest
    import pandas as pd
    original = load_original_manifest()
    reference_jobs = pd.read_csv(REFERENCE / "verified_jobs.csv").set_index("job_id", verify_integrity=True)
    ref = pd.read_csv(REFERENCE / "per_replication.csv")
    timings = pd.read_csv(RECOVERY / "local_runs/20261008_overnight_v1/pilot_collection/collected_runs.csv")
    keys = ["earlier_budget_repaired", "B3_warm2500_ret2000"]
    budgets = {keys[0]: dict(num_gfr=50, num_burnin=0, num_mcmc=500, keep_every=5, num_chains=3,
                           label="repaired engine at the earlier proper-prior budget"),
               keys[1]: BUDGETS[keys[1]]}
    jobs, panel_checks = [], []
    for index, (setting, n, degree) in enumerate(CELLS):
        for rep in range(30):
            candidates = [j for j in original["jobs"] if j["phase"] == "main" and j["kind"] == "effect"
                          and (j["setting"], j["N"], j["linearity_degree"], j["rep"]) == (setting, n, degree, rep)
                          and j["sigma2_shape"] == 2 and j["sigma2_rate"] == 1]
            if len(candidates) != 1:
                raise ValueError(f"Nonunique requested statistical job: {(setting,n,degree,rep)}")
            job = candidates[0]
            old = reference_jobs.loc[job["id"]]
            panel = load_panel(job)
            observed_hash = panel_hash(panel)
            if observed_hash != old.panel_hash:
                raise ValueError(f"Dataset mismatch with saved proper-prior reference: {job['id']}")
            panel_checks.append(dict(job_id=job["id"], setting=setting, N=n, degree=degree, rep=rep,
                                     panel_hash=observed_hash, reference_python=old.python, archive=old.archive))
            jobs.append({**job, "recovery": dict(budget_keys=keys, trace=True, cell_index=index,
                        diagnostic_only=False, reference_old_fit_seconds=float(old.seconds),
                        selection="Fixed first 30 replication indices in the user-approved nine cells")})
    ids = {j["id"] for j in jobs}
    selected = ref[ref.job_id.isin(ids)].copy()
    if set(selected.job_id) != ids or set(selected.method) != {"plain"}:
        raise ValueError("Incomplete or wrong-method saved reference")
    if selected.duplicated(["job_id", "estimand_type", "estimand_id", "method"]).any():
        raise ValueError("Duplicate reference target identity")
    if not (selected.sigma2_shape.eq(2).all() and selected.sigma2_rate.eq(1).all()):
        raise ValueError("Reference prior mismatch")
    scalar = selected[selected.estimand_type != "CATT"]
    if not scalar[["post_mean", "sd", "q025", "q05", "q95", "q975", "true"]].notna().all().all():
        raise ValueError("Missing required scalar reference metrics")
    assert len(jobs) == 270 and selected[selected.estimand_type == "ATT"].job_id.nunique() == 270
    selected.to_csv(HERE / "reference_per_replication.csv", index=False)
    pd.read_csv(REFERENCE / "chain_diagnostics.csv").query("job_id in @ids").to_csv(HERE / "reference_chain_diagnostics.csv", index=False)
    pd.DataFrame(panel_checks).to_csv(HERE / "paired_panel_verification.csv", index=False)
    sources = [HERE / name for name in ("PROTOCOL.md", "prepare_comparison.py", "run_comparison.py", "report_comparison.py")]
    body = _manifest_body("replicated_paper_metric_comparison", jobs, budgets, keys,
                          "frozen_pending_local_execution", True,
                          ["User requested 30 replications per cell and both repaired budgets, with saved proper-prior reference.",
                           "Diagnostics are supporting evidence; the previous full-state gate is not a prerequisite or marked passed.",
                           "No posterior draw/replication selection and no automatic production release."])
    if environment() != body["dependencies"]:
        raise ValueError("Run preparation in the matching project .venv")
    body["acceptance_targets"] = {**body["acceptance_targets"], "role": "supporting_diagnostics_only"}
    body["comparison"] = dict(replications=list(range(30)), cells=CELLS, new_fits=540,
                              expected_chain_units=1890, bootstrap_resamples=5000, reporting_seed=20261009,
                              equivalence_margin=None, production_release=False,
                              reference_sources={str((REFERENCE / name).relative_to(REPO)): sha256_file(REFERENCE / name)
                                                 for name in ("per_replication.csv", "verified_jobs.csv", "chain_diagnostics.csv", "integrity_report.json")},
                              reference_subsets={str((HERE / name).relative_to(REPO)): sha256_file(HERE / name)
                                                 for name in ("reference_per_replication.csv", "reference_chain_diagnostics.csv", "paired_panel_verification.csv")})
    body["performance_sources"] = {str(p.relative_to(REPO)): sha256_file(p) for p in sources}
    body["manifest_hash"] = digest(body)
    target = HERE / "manifest.json"
    if target.exists() and json.loads(target.read_text())["manifest_hash"] != body["manifest_hash"]:
        raise ValueError("Existing comparison manifest differs; use a new namespace")
    write_json_atomic(target, body)
    # Preparation estimates use measured B3 timings for exactly these cells.
    estimates = []
    for setting, n, degree in CELLS:
        observed = timings[(timings.setting == setting) & timings.N.eq(n) & timings.linearity_degree.eq(degree)
                           & timings.budget_key.eq(keys[1])]
        if observed.empty:
            raise ValueError("Missing measured timing for selected cell")
        b3_seconds = float(observed.seconds.mean())
        estimates.append(dict(setting=setting, N=n, degree=degree, B3_cpu_hours=30*4*b3_seconds/3600,
                              earlier_cpu_hours_estimate=30*3*b3_seconds*(2550/12550)/3600,
                              maximum_measured_RSS_MiB=float(observed.peak_rss_mb.max()),
                              approximate_output_GiB=30*(4+3*.25)*float(observed.output_bytes.mean())/2**30))
    pd.DataFrame(estimates).to_csv(HERE / "runtime_estimates.csv", index=False)
    assignment_summary = write_assignments(body, body["purpose"], 6.5, 48, HERE / "assignments")
    from recovery_build_notebooks import build_notebook
    import nbformat
    notebook_dir = HERE / "upload_to_colab"
    notebook_dir.mkdir(exist_ok=True)
    index = []
    for path in sorted((HERE / "assignments").glob("wave_*/shard_*.json")):
        assignment = json.loads(path.read_text())
        nb, label = build_notebook(assignment, path, body, target, body["purpose"], 8., False)
        nb["cells"][0]["source"].append("\nThis is the user's 30-replication paper-metric comparison. Diagnostic flags do not select draws or replications. This notebook does not release the full production campaign.\n")
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": ast.parse("".join(cell["source"]))
        nbformat.validate(nbformat.from_dict(nb))
        output = notebook_dir / f"performance_w{assignment['wave']:02d}_s{assignment['shard']:02d}.ipynb"
        output.write_text(json.dumps(nb, indent=1) + "\n")
        index.append(dict(notebook=output.name, sha256=sha256_file(output), units=len(assignment["units"])))
    write_json_atomic(notebook_dir / "notebook_index.json", index)
    cpu_hours = sum(e["B3_cpu_hours"] + e["earlier_cpu_hours_estimate"] for e in estimates)
    summary = dict(manifest_hash=body["manifest_hash"], datasets=270, new_fits=540, expected_chain_units=1890,
                   panel_hash_matches=len(panel_checks), estimated_CPU_hours=cpu_hours,
                   ideal_six_worker_hours=cpu_hours/6, expected_output_GiB=sum(e["approximate_output_GiB"] for e in estimates),
                   fallback_notebooks=len(index), dependencies=environment())
    write_json_atomic(HERE / "preparation_summary.json", summary)
    (HERE / "README.md").write_text("# Local paired performance comparison\n\n"
        "The user selected 30 replications per cell, both repaired budgets, and reuse of the saved proper-prior reference. "
        "All 270 panel hashes match the saved reference. See [PROTOCOL.md](PROTOCOL.md).\n\n"
        "Local execution: run `run_comparison.py` using the repository `.venv/bin/python`. "
        "Progress is in `local_run/status.json`; completion automatically verifies outputs and writes `local_run/report/PERFORMANCE_REPORT.md`. "
        "The runner is restartable and retains failed attempts. No full production campaign is launched.\n\n"
        f"Measured timing proxies suggest {cpu_hours/6:.1f} hours at six workers before overhead; allow roughly 1.5 to 2 days.\n\n"
        f"There are {len(index)} individual self-contained fallback notebooks in `upload_to_colab/`. "
        "The local run executes the same assignments in one RAM-reserved pool. Do not run overlapping local and Colab assignments simultaneously. "
        "Notebook inputs are loose `.ipynb` files; each notebook downloads one checkpoint/results ZIP.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
