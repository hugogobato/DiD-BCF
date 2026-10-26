"""Bounded infrastructure/metric checks; never launch production sampling."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np
import pandas as pd
from common import (PACKAGE, BUDGET_KEY, atomic, bootstrap, digest, read, sha,
                    restore_metrics, verify_fit, verify_package)

def validate(old, setup, report):
    package, manifest = verify_package()
    bootstrap(setup)
    from recovery_plan import make_units
    from recovery_engine import activate, fingerprint
    from paper_metrics import aggregate_metrics, _write_csv
    from job_worker import run
    from runtime import snapshot
    from did_bcf_revision.metrics import compute_metrics, surface_metrics
    checks = {}
    assignments = [read(p) for p in (PACKAGE / "assignments").glob("main*.json")]
    ids = [j for a in assignments for j in a["job_ids"]]
    assert len(ids) == len(set(ids)) == 4836
    assert set(ids).isdisjoint(manifest["reused_main_job_ids"])
    assert len(set(ids) | set(manifest["reused_main_job_ids"])) == 5100
    assert all(7.9 < a["estimated_wall_hours"] < 8.1 for a in assignments)
    assert len(assignments) == 60
    checks["exact_assignment_partition"] = True
    for folder in (PACKAGE / "reuse/fits").iterdir():
        assert verify_fit(folder,manifest)
    for path in [PACKAGE / "reuse", PACKAGE / "bridge_reference"]:
        assert not any(p.suffix in (".npz",".npy",".so") for p in path.rglob("*"))
    checks["all_264_reused_fits_and_no_array_exports"] = True
    index = read(PACKAGE / "bridge_reference/index.json")
    assert len(index["fits"]) == 8
    for item in index["fits"]:
        for name,checksum in item["outputs"].items():
            assert sha(PACKAGE / "bridge_reference/fits" / item["job_id"] / name) == checksum
        assert item["per_replication_rows"] > 0
    checks["eight_saved_B3_anchors_exported"] = True
    per = pd.concat([pd.read_csv(p,float_precision="round_trip") for p in
                     (PACKAGE / "reuse/fits").glob("*/per_replication.csv")],ignore_index=True)
    current_scalar, current_surface = aggregate_metrics(per)
    frozen_scalar, frozen_surface = compute_metrics(per), surface_metrics(per)
    shared = [c for c in frozen_scalar if c != "mcse_bias"]
    pd.testing.assert_frame_equal(current_scalar[shared],frozen_scalar[shared],check_exact=True)
    pd.testing.assert_frame_equal(current_surface,frozen_surface,check_exact=True)
    # Independent error/MCSE computation at a varying-truth ATT cell.
    group = per.loc[(per.setting == "B1_baseline") & (per.linearity_degree == 1) & (per.estimand_type == "ATT")]
    result = current_scalar.loc[(current_scalar.setting == "B1_baseline") & (current_scalar.linearity_degree == 1) & (current_scalar.estimand_type == "ATT")].iloc[0]
    error = group.post_mean.to_numpy()-group.true.to_numpy()
    assert result.bias == np.mean(error)
    assert result.rmse == np.sqrt(np.mean(error**2))
    assert result.mcse_bias == np.std(error,ddof=1)/np.sqrt(len(error))
    assert int(result.cover95_n) == len(group)
    assert int(result.cover95_count) == int(((group.q025 <= group.true)&(group.true <= group.q975)).sum())
    null = current_scalar.loc[(current_scalar.setting == "B1_null") & (current_scalar.estimand_type == "ATT")]
    assert null.mape.isna().all()
    # The heterogeneous truth varies across replications; estimate-SD is not
    # the same as error-SD. This catches the legacy bias-MCSE formula.
    assert not np.isclose(result.mcse_bias,result.legacy_mcse_bias_estimate_sd)
    checks["paper_metrics_exact_and_bias_MCSE_from_errors"] = True
    engine = fingerprint(activate())
    baseline_id = "ae95bf48eea71d8a38f2"
    job = next(j for j in manifest["jobs"] if j["id"] == baseline_id)
    units = make_units({**manifest,"jobs":[job]})
    with tempfile.TemporaryDirectory(prefix="didbcf-paper-test-") as temporary:
        root = Path(temporary)
        work, out = root / "work", root / "out"
        assigned = {"name":"fixture","manifest_hash":manifest["manifest_hash"],
                    "job_ids":[baseline_id],"estimated_fit_seconds":{baseline_id:500},"phase":"fixture"}
        assigned["assignment_hash"] = digest(assigned)
        assigned_path = root / "assignment.json"
        atomic(assigned_path,assigned)
        # Copy already-sampled local baseline chains into an explicitly
        # synthetic recovery namespace. Only metadata locations/identities
        # are remapped. Sampling is never called; real draws stay unchanged.
        native_paths = list((old / "local_completed_outputs/runs").glob("*/attempt_*/complete.json"))
        for unit in units:
            candidates = [p for p in native_paths if read(p)["stat_job_id"] == baseline_id
                          and read(p)["budget_key"] == BUDGET_KEY and read(p)["chain_index"] == unit["chain_index"]]
            assert len(candidates) == 1
            source = candidates[0].parent
            folder = work / "raw" / baseline_id / unit["run_id"] / "attempt_000"
            folder.mkdir(parents=True)
            record = read(source / "complete.json")
            assert record["engine"] == engine
            for name in record["outputs"]:
                shutil.copy2(source / name,folder / name)
            record.update(run_id=unit["run_id"],assignment_hash=assigned["assignment_hash"],
                          source_hashes_sha256=digest(manifest["source_hashes"]),
                          validation_fixture=True)
            atomic(folder / "complete.json",record)
        # It should consume complete raw-chain checkpoints and postprocess
        # exactly, without invoking the native sampling subprocess at all.
        import job_worker
        original_run = job_worker.subprocess.run
        def forbidden_sampling(*args,**kwargs):
            raise AssertionError("A validation fixture attempted to sample")
        job_worker.subprocess.run = forbidden_sampling
        try:
            run(baseline_id,assigned_path,setup,work,out)
        finally:
            job_worker.subprocess.run = original_run
        folder = out / "fits" / baseline_id
        new = pd.read_csv(folder / "per_replication.csv",float_precision="round_trip")
        expected = pd.read_csv(PACKAGE / "reuse/fits" / baseline_id / "per_replication.csv",float_precision="round_trip")
        cols = [c for c in new if c in expected and c != "kind"]
        pd.testing.assert_frame_equal(new[cols],expected[cols],check_exact=True,check_dtype=False)
        assert not any(p.suffix in (".npz",".npy") for p in out.rglob("*"))
        assert not (work / "raw" / baseline_id).exists()
        checks["real_pooled_draws_export_exact_no_sampling"] = True
        bundle = out / "bundle"
        atomic(bundle / "assignment.json",assigned)
        atomic(bundle / "manifest.json",manifest)
        atomic(bundle / "package.json",package)
        zip_path = root / "fixture.zip"
        state = snapshot(out,assigned,manifest,zip_path,{"validation_fixture":True})
        assert state["completed"] == 1 and not state["missing_job_ids"]
        restored = root / "restored"
        restore_metrics(zip_path,restored,manifest,assigned)
        assert verify_fit(restored / "fits" / baseline_id,manifest)
        # A complete metrics result must be skipped before native execution.
        run(baseline_id,assigned_path,setup,work,restored)
        bad = restored / "fits" / baseline_id / "per_replication.csv"
        bad.write_text(bad.read_text()+"\ncorrupt")
        try:
            verify_fit(bad.parent,manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt metric was accepted")
        malicious = root / "unsafe.zip"
        with zipfile.ZipFile(malicious,"w") as archive:
            archive.write(assigned_path,"bundle/assignment.json")
            archive.writestr("../escape.txt","bad")
        try:
            restore_metrics(malicious,root / "unsafe",manifest,assigned)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe resume ZIP was accepted")
        checks["compact_resume_corruption_and_traversal_checks"] = True
    notebooks = list((PACKAGE / "Colab").rglob("*.ipynb"))
    code_cells = 0
    for path in notebooks:
        notebook = read(path)
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                code_cells += 1
    if notebooks:
        assert len(notebooks) == 65 and code_cells == 322
        checks["notebook_formats_and_code_syntax"] = True
    else:
        checks["notebook_formats_and_code_syntax"] = "pending generation from immutable source commit"
    atomic(report,{"status":"PASS","checks":checks,"sampling_calls":0,
           "source_engine_fingerprint":engine,"package_hash":package["package_hash"],
           "notebooks":len(notebooks),"code_cells":code_cells,
           "note":"Draw-based fixture contains remapped metadata, is validation only and is never a final experimental result."})
    print("PASS",list(checks),flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-package","setup","report"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    validate(args.old_package,args.setup,args.report)
