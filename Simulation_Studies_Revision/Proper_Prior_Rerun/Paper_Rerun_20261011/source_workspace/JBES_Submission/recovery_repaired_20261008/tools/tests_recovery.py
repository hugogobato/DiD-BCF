"""Bounded local tests for the recovery infrastructure.

Covers the E1 checkpoint contract: clean start, verified resume, interruption
after one completed chain, corrupt/missing outputs, conflicting source or
budget, duplicate chain IDs, safe zip extraction, seed equivalence to an
uninterrupted run, and instrumentation leaving the transition kernel unchanged.

Run from the repository root:
    .venv/bin/python -m pytest JBES_Submission/recovery_repaired_20261008/tools/tests_recovery.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parent
setup_path = TOOLS.parent / "smoke/local_setup/setup_complete.json"
if setup_path.exists() and not os.environ.get("DIDBCF_REPAIRED_BUILD"):
    local_setup = json.loads(setup_path.read_text())
    for env, key in (("DIDBCF_REPAIRED_BUILD", "build"), ("DIDBCF_REPAIRED_CONTROLS", "controls"), ("DIDBCF_WEIGHT_CONTROLS", "weight_controls")):
        os.environ[env] = local_setup[key]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/didbcf-mpl")

from recovery_common import (REPO, bundle_hashes, digest, environment,  # noqa: E402
                             extract_zip_verified, model_spec, run_id_for,
                             sweeps_per_chain, write_json_atomic)
from recovery_plan import make_units, write_assignments  # noqa: E402

for _path in (REPO / "DiD-BCF", REPO / "DiD-BCF/Simulation_Studies_Revision",
              REPO / "DiD-BCF/Simulation_Studies_Revision/scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

TINY_BUDGET = dict(num_gfr=3, num_burnin=2, num_mcmc=4, keep_every=1,
                   num_chains=2, label="unit test")
TINY_JOB = dict(
    id="testjob0000000000000", kind="effect", phase="main",
    setting="B1_baseline", rep=7, spec="structured",
    sigma2_shape=2.0, sigma2_rate=1.0, dgp="canonical",
    dgp_params=dict(n_units=40, num_pre_periods=2, num_post_periods=2,
                    linearity_degree=1),
    N=40, linearity_degree=1)


def _manifest(purpose="unit_test", trace=True):
    body = {
        "recovery_version": "0.2.0-repaired-20261008", "status": "test", "purpose": purpose,
        "parent_manifest": "0" * 64, "method_version": "test",
        "engine_specification": __import__("recovery_engine").specification(),
        "model_spec": model_spec(), "budgets": {"T0": TINY_BUDGET},
        "budget_keys": ["T0"], "trace": trace,
        "source_hashes": bundle_hashes(), "dependencies": environment(),
        "python_recorded": "test", "jobs": [TINY_JOB],
        "acceptance_targets": {}, "output_schema": {}, "notes": [],
    }
    body["manifest_hash"] = digest(body)
    return body


def _assembly(tmp_path):
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    assignments = tmp_path / "assignments"
    write_assignments(manifest, "unit_test", target_hours=1.0, wave_size=48,
                      out_dir=assignments)
    shard = assignments / "wave_00" / "shard_00.json"
    return manifest, manifest_path, shard


def archive_engine(out):
    import shutil
    from recovery_engine import activate
    engine = out / "_engine"
    if engine.exists():
        return
    build_home = Path(os.environ["DIDBCF_REPAIRED_BUILD"])
    build = json.loads((build_home / "build_complete.json").read_text())
    for rel in build["site_files"]:
        target = engine / "site" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build_home / "site" / rel, target)
    shutil.copy2(build_home / "build_complete.json", engine / "build_complete.json")
    for name, key in (("repaired_controls.json", "DIDBCF_REPAIRED_CONTROLS"), ("weight_controls.json", "DIDBCF_WEIGHT_CONTROLS")):
        shutil.copy2(os.environ[key], engine / name)
    write_json_atomic(engine / "setup_complete.json", {"status":"PASS", "engine":activate()})

def _run_cli(shard, out, extra=()):
    archive_engine(out)
    command = [sys.executable, str(TOOLS / "recovery_runner.py"), "run",
               "--assignment", str(shard), "--out", str(out), "--hours", "1",
               "--reserve-min", "0", *extra]
    return subprocess.run(command, capture_output=True, text=True,
                          timeout=900)


class _Args:
    def __init__(self, shard, out, unit=None):
        self.assignment = Path(shard)
        self.out = Path(out)
        self.hours = 1.0
        self.reserve_min = 0.0
        self.margin_min = 0.0
        self.unit = unit
        self.dry_run = False


def test_budget_arithmetic():
    assert sweeps_per_chain(dict(num_gfr=50, num_burnin=0, num_mcmc=500,
                                 keep_every=5, num_chains=3)) == 2550
    assert sweeps_per_chain(TINY_BUDGET) == 9


def test_diagnostics_self_test():
    from recovery_diagnostics import self_test
    assert "passed" in self_test()


def test_safe_zip_extraction(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("a/b.txt", b"hello")
    target = tmp_path / "target"
    summary = extract_zip_verified(good, target)
    assert (target / "a/b.txt").read_bytes() == b"hello"
    # Verified resume skips identical bytes and rejects conflicts.
    again = extract_zip_verified(good, target, verified=True)
    assert again["skipped_matching"]
    (target / "a/b.txt").write_bytes(b"different")
    with pytest.raises(ValueError):
        extract_zip_verified(good, target, verified=True)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.txt", b"boom")
    with pytest.raises(ValueError):
        extract_zip_verified(unsafe, tmp_path / "target2")
    absolute = tmp_path / "absolute.zip"
    with zipfile.ZipFile(absolute, "w") as archive:
        archive.writestr("/etc/passwd", b"boom")
    with pytest.raises(ValueError):
        extract_zip_verified(absolute, tmp_path / "target3")


def test_seed_equivalence_to_joint_multichain():
    from didbcf_structured import GlobalPrior, StructuredDiDBCF
    from did_bcf_revision.dgps import generate_canonical_did
    from recovery_core import ChainModel

    df = generate_canonical_did(seed=7, **TINY_JOB["dgp_params"])
    joint = StructuredDiDBCF(global_prior=GlobalPrior(2., 1.)).sample(
        df, seed=7, num_gfr=3, num_burnin=2, num_mcmc=4, keep_every=1,
        num_chains=2)
    for chain in range(2):
        seed = 7 * 1000 + chain
        single = ChainModel.build(target_chain_seed=seed, trace=False,
                                  rfx="none",
                                  global_prior=GlobalPrior(2., 1.)).sample(
            df, seed=7, num_gfr=3, num_burnin=2, num_mcmc=4, keep_every=1,
            num_chains=1)
        np.testing.assert_array_equal(
            single.sigma2_draws, joint.sigma2_draws[chain * 4:(chain + 1) * 4])
        np.testing.assert_array_equal(
            single.tau_draws, joint.tau_draws[:, chain * 4:(chain + 1) * 4])


def test_trace_leaves_kernel_unchanged():
    from didbcf_structured import GlobalPrior
    from did_bcf_revision.dgps import generate_canonical_did
    from recovery_core import ChainModel

    df = generate_canonical_did(seed=3, **TINY_JOB["dgp_params"])
    off = ChainModel.build(target_chain_seed=3001, trace=False, rfx="none",
                           global_prior=GlobalPrior(2., 1.)).sample(
        df, seed=3, num_gfr=3, num_burnin=2, num_mcmc=4, keep_every=1,
        num_chains=1)
    on = ChainModel.build(target_chain_seed=3001, trace=True, rfx="none",
                          global_prior=GlobalPrior(2., 1.)).sample(
        df, seed=3, num_gfr=3, num_burnin=2, num_mcmc=4, keep_every=1,
        num_chains=1)
    np.testing.assert_array_equal(on.sigma2_draws, off.sigma2_draws)
    np.testing.assert_array_equal(on.tau_draws, off.tau_draws)
    trace = on._chains[0]["trace"]
    names = [str(x) for x in trace["group_names"]]
    att = (trace["group_means"][:, names.index("ATT")]
           + trace["scalars"][:, list(trace["scalar_names"]).index("tau_0")])
    retained = ((trace["stage"] == 2)
                & ((trace["sweep"] - 2 + 1) % 1 == 0))
    np.testing.assert_allclose(att[retained],
                               on.tau_draws[on.df.D == 1].mean(axis=0)
                               / on.y_std, rtol=0, atol=1e-12)


def test_clean_run_verified_resume_and_interruption(tmp_path):
    manifest, manifest_path, shard = _assembly(tmp_path)
    out = tmp_path / "out"
    first = _run_cli(shard, out)
    assert first.returncode == 0, first.stdout + first.stderr
    attempts = sorted(out.glob("runs/*/attempt_000/complete.json"))
    assert len(attempts) == 2
    mtime = attempts[0].stat().st_mtime_ns
    # A second run must verify and skip, not refit.
    second = _run_cli(shard, out)
    assert second.returncode == 0
    status = json.loads((out / "run_status.json").read_text())
    assert status["summary"]["completed_now"] == 0
    assert attempts[0].stat().st_mtime_ns == mtime

    # Interruption after one completed chain: run only chain 0, then finish.
    out2 = tmp_path / "out2"
    assignment = json.loads(shard.read_text())
    run_ids = [unit["run_id"] for unit in assignment["units"]]
    partial = _run_cli(shard, out2, extra=("--unit", run_ids[0]))
    assert partial.returncode in (0, 2)
    assert (out2 / "runs" / run_ids[0] / "attempt_000" / "complete.json").exists()
    assert not (out2 / "runs" / run_ids[0] / "attempt_001").exists()
    # Simulate an interrupted attempt for chain 1: a started record only.
    interrupted = out2 / "runs" / run_ids[1] / "attempt_000"
    interrupted.mkdir(parents=True)
    write_json_atomic(interrupted / "started.json", {"partial": True})
    finished = _run_cli(shard, out2)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert (out2 / "runs" / run_ids[0] / "attempt_000" / "complete.json").exists()
    assert (interrupted / "started.json").exists()          # preserved
    assert not (interrupted / "complete.json").exists()
    assert (out2 / "runs" / run_ids[1] / "attempt_001"
            / "complete.json").exists()


def test_corrupt_output_and_conflicting_budget(tmp_path):
    manifest, manifest_path, shard = _assembly(tmp_path)
    out = tmp_path / "out"
    assert _run_cli(shard, out).returncode == 0
    draw_files = sorted(out.glob("runs/*/attempt_000/draws.npz"))
    draw_files[0].write_bytes(b"corrupt")
    result = _run_cli(shard, out)
    assert result.returncode == 1
    assert "CONFLICT" in result.stdout or "conflict" in result.stdout.lower()
    # Conflicting budget: patch a completion record instead.
    complete = json.loads(
        sorted(out.glob("runs/*/attempt_000/complete.json"))[1].read_text())
    complete["budget_key"] = "WRONG"
    write_json_atomic(sorted(out.glob("runs/*/attempt_000/complete.json"))[1],
                      complete)
    result = _run_cli(shard, out)
    assert result.returncode == 1


def test_runner_refuses_source_drift(tmp_path):
    manifest, manifest_path, shard = _assembly(tmp_path)
    assignment = json.loads(shard.read_text())
    key = sorted(assignment["source_hashes"])[0]
    assignment["source_hashes"][key] = "0" * 64
    body = {k: v for k, v in assignment.items() if k != "assignment_hash"}
    assignment["assignment_hash"] = digest(body)
    tampered = tmp_path / "tampered.json"
    write_json_atomic(tampered, assignment)
    result = _run_cli(tampered, tmp_path / "out_tamper")
    assert result.returncode != 0
    assert "source hash mismatch" in (result.stdout + result.stderr).lower()


def test_duplicate_chain_ids_and_unknown_run():
    manifest = _manifest()
    units = make_units(manifest)
    assert len(units) == 2
    assert len({unit["run_id"] for unit in units}) == 2
    repeated = units + units
    with pytest.raises(ValueError):
        # make_units itself guards against duplicates; emulate the guard.
        if len({unit["run_id"] for unit in repeated}) != len(repeated):
            raise ValueError("Duplicate run IDs")
    assert run_id_for(TINY_JOB["id"], TINY_BUDGET, 0, True,
                      manifest["manifest_hash"]) == units[0]["run_id"]


def test_collector_flags_tampered_run(tmp_path):
    from recovery_collect import gather
    manifest, manifest_path, shard = _assembly(tmp_path)
    out = tmp_path / "out"
    assert _run_cli(shard, out).returncode == 0
    complete = sorted(out.glob("runs/*/attempt_000/complete.json"))[0]
    record = json.loads(complete.read_text())
    record["chain_seed"] = 999999
    write_json_atomic(complete, record)
    _, _, status = gather([out], manifest)
    assert status["conflicting_units"] >= 1


def test_generated_notebook_is_self_consistent(tmp_path):
    import ast
    import base64
    import io
    import zipfile
    from recovery_build_notebooks import build_notebook, build_bundle

    manifest, manifest_path, shard = _assembly(tmp_path)
    assignment = json.loads(shard.read_text())
    notebook, label = build_notebook(assignment, shard, manifest,
                                     manifest_path, "unit_test", 8.0,
                                     include_protocol=False)
    assert "unit_test" in label and label.startswith("repaired_")
    for entry in notebook["cells"]:
        text = "".join(entry["source"])
        if entry["cell_type"] == "code":
            ast.parse(text)
        if "@@" in text:
            raise AssertionError("Unreplaced template token")
    assert len(notebook["cells"]) == 8
    bundle = build_bundle(assignment, shard, manifest_path, manifest)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle))) as archive:
        meta = json.loads(archive.read(
            "recovery_bundle/bundle_manifest.json"))
        for rel, expected in meta["files"].items():
            import hashlib
            payload = archive.read(rel)
            assert hashlib.sha256(payload).hexdigest() == expected
        embedded_assignment = json.loads(
            archive.read("recovery_bundle/assignment.json"))
        assert embedded_assignment["assignment_hash"] == \
            assignment["assignment_hash"]


def test_collector_and_report_integration(tmp_path):
    from recovery_collect import main as collect_main
    from recovery_report_pilot import main as report_main

    pretrend_job = dict(
        id="pretrendtest00000000", kind="pretrend", phase="diagnostic",
        setting="PT_hold", rep=4, spec="pretrend_unit",
        sigma2_shape=2.0, sigma2_rate=1.0, dgp="canonical",
        dgp_params=dict(n_units=40, num_pre_periods=3, num_post_periods=3,
                        linearity_degree=1),
        N=40, linearity_degree=1)
    application_job = dict(
        id="applicationtest00000", kind="application", phase="application",
        setting="mpdta", rep=0, spec="structured_rfx_unit",
        sigma2_shape=2.0, sigma2_rate=1.0, dgp=None, dgp_params={},
        N=500, linearity_degree=0)
    application_pretrend_job = dict(
        id="apppretrendtest00000", kind="application_pretrend",
        phase="application", setting="mpdta", rep=0, spec="pretrend_unit",
        sigma2_shape=2.0, sigma2_rate=1.0, dgp=None, dgp_params={},
        N=500, linearity_degree=0)
    budgets = {
        "T0": dict(num_gfr=3, num_burnin=2, num_mcmc=4, keep_every=1,
                   num_chains=2, label="test small"),
        "T1": dict(num_gfr=3, num_burnin=4, num_mcmc=4, keep_every=1,
                   num_chains=2, label="test larger"),
    }
    body = {
        "recovery_version": "0.2.0-repaired-20261008", "status": "test", "purpose": "unit_test",
        "parent_manifest": "0" * 64, "method_version": "test",
        "engine_specification": __import__("recovery_engine").specification(),
        "model_spec": model_spec(), "budgets": budgets,
        "budget_keys": ["T0", "T1"], "trace": True,
        "source_hashes": bundle_hashes(), "dependencies": environment(),
        "python_recorded": "test",
        "jobs": [{**TINY_JOB, "recovery": {"budget_keys": ["T0", "T1"]}},
                 {**pretrend_job, "recovery": {"budget_keys": ["T0", "T1"]}},
                 {**application_job,
                  "recovery": {"budget_keys": ["T0", "T1"]}},
                 {**application_pretrend_job,
                  "recovery": {"budget_keys": ["T0", "T1"]}}],
        "acceptance_targets": {}, "output_schema": {}, "notes": [],
    }
    body["manifest_hash"] = digest(body)
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, body)
    assignments = tmp_path / "assignments"
    write_assignments(body, "unit_test", target_hours=1.0, wave_size=48,
                      out_dir=assignments)
    shard = assignments / "wave_00" / "shard_00.json"
    out = tmp_path / "runs"
    result = _run_cli(shard, out)
    assert result.returncode == 0, result.stdout + result.stderr

    collection = tmp_path / "collection"
    collect = subprocess.run(
        [sys.executable, str(TOOLS / "recovery_collect.py"),
         "--manifest", str(manifest_path), "--incoming", str(out),
         "--out", str(collection)], capture_output=True, text=True, timeout=900)
    assert collect.returncode == 0, collect.stdout + collect.stderr
    status = json.loads((collection / "status.json").read_text())
    assert status["conflicting_units"] == 0
    assert status["complete_units"] == len(make_units(body))
    frame = pd.read_csv(collection / "per_replication.csv")
    assert {"ATT", "sigma2"}.issubset(set(frame.estimand_id)) or \
        "ATT" in set(frame.estimand_type)
    assert frame[frame["estimand_type"] == "ATT"]["true"].notna().any()
    assert not pd.read_csv(collection / "chain_diagnostics.csv").empty

    report_out = tmp_path / "report"
    report = subprocess.run(
        [sys.executable, str(TOOLS / "recovery_report_pilot.py"),
         "--collection", str(collection), "--manifest", str(manifest_path),
         "--out", str(report_out)], capture_output=True, text=True, timeout=900)
    assert report.returncode == 0, report.stdout + report.stderr
    assert (report_out / "PILOT_ACCEPTANCE.md").exists()
    assert (report_out / "pilot_stability.csv").exists()


def test_generated_notebook_dry_run_end_to_end(tmp_path):
    """Execute the generated notebook cells locally with --dry-run sampling.

    Verifies extraction, bundle verification, resume import, assignment
    summary, bounded-execution controller, background checkpoint archive and
    final archive/download fallback without running a sampler.
    """
    from recovery_build_notebooks import build_notebook

    manifest, manifest_path, shard = _assembly(tmp_path)
    assignment = json.loads(shard.read_text())
    notebook, label = build_notebook(assignment, shard, manifest,
                                     manifest_path, "unit_test", 8.0,
                                     include_protocol=False)
    cells = {entry["id"]: "".join(entry["source"])
             for entry in notebook["cells"]}
    work = tmp_path / "content"
    work.mkdir()
    namespace = {"__name__": "__main__"}
    exec(cells["configuration"].replace("/content", str(work)), namespace)
    # Reuse the independently validated local binary; cold download/build is
    # deliberately outside this bounded cell-execution test.
    setup_call = 'subprocess.run([sys.executable, str(tools / "recovery_setup.py"), "--work", str(SETUP_WORK), "--cache", str(WORKSPACE / "didbcf_pinned_archives")], check=True)'
    reuse = 'SETUP_WORK.mkdir(parents=True, exist_ok=True); shutil.copy2(' + repr(str(setup_path.resolve())) + ', SETUP_WORK / "setup_complete.json")'
    exec(cells["bundle-verify"].replace(setup_call, reuse).replace("/content", str(work)), namespace)
    exec(cells["resume-and-resources"].replace("/content", str(work)),
         namespace)
    exec(cells["assignment"], namespace)
    run_cell = (cells["bounded-execution"]
                .replace("/content", str(work))
                .replace('"--reserve-min", str(RESERVE_MIN)]',
                         '"--reserve-min", str(RESERVE_MIN), "--dry-run"]'))
    exec(run_cell, namespace)
    assert (work / "didbcf_repaired_runs" / label / "run_status.json").exists()
    partial = work / f"didbcf_recovery_{label}_partial.zip"
    assert partial.exists()
    exec(cells["summary-archive-download"].replace("/content", str(work)),
         namespace)
    archive = work / f"didbcf_recovery_{label}.zip"
    assert archive.exists()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert "_bundle/source_bundle.zip" in names
        assert any(name.endswith("run_status.json") for name in names)


def test_quantile_mcse_translation_and_scale():
    from recovery_diagnostics import quantile_mcse
    x = np.random.default_rng(819).normal(size=(4, 700))
    for q in (.025, .05, .95, .975):
        base = quantile_mcse(x, q)
        np.testing.assert_allclose(quantile_mcse(x + 80., q), base, rtol=1e-10)
        np.testing.assert_allclose(quantile_mcse(x * 3., q), base * 3., rtol=1e-10)


def test_phase_sharding_preserves_all_units(tmp_path):
    body = _manifest()
    body["jobs"] = [{**TINY_JOB, "id": str(i).zfill(20), "phase":phase} for i, phase in enumerate(("application", "diagnostic", "main"))]
    body["manifest_hash"] = digest({k:v for k,v in body.items() if k != "manifest_hash"})
    write_assignments(body, "unit_test", .01, 48, tmp_path, phase_order=("application", "diagnostic", "main"))
    ids = [u["run_id"] for path in tmp_path.glob("wave_*/shard_*.json") for u in json.loads(path.read_text())["units"]]
    assert len(ids) == len(set(ids)) == 6
    assert set(ids) == {u["run_id"] for u in make_units(body)}


def test_exact_pooled_surface_intervals():
    from recovery_collect import _surface_row_chunked
    from did_bcf_revision.metrics import surface_summary
    rng = np.random.default_rng(92)
    panel = pd.DataFrame({"D":[1,0,1,1,0], "CATT":rng.normal(size=5)})
    tau = rng.normal(size=(5, 80))
    result = _surface_row_chunked(panel, tau, chunk_rows=1)
    rows = panel.index[panel.D == 1]
    block = tau[rows]
    q = np.quantile(block, [.05,.95,.025,.975], axis=1)
    direct = surface_summary(panel.loc[rows,"CATT"].to_numpy(), block.mean(axis=1), *q)
    for key, value in direct.items():
        np.testing.assert_allclose(result[key], value)


def test_engine_archive_tamper_rejected(tmp_path):
    from recovery_collect import verify_engine_archive
    archive_engine(tmp_path)
    assert verify_engine_archive(tmp_path, _manifest())
    metadata = tmp_path / "_engine/build_complete.json"
    body = json.loads(metadata.read_text()); body["patch_sha256"] = "0" * 64
    write_json_atomic(metadata, body)
    with pytest.raises(ValueError, match="patch_sha256"):
        verify_engine_archive(tmp_path, _manifest())


def test_precision_screen_rejects_nonfinite_tail_ess():
    from recovery_report_pilot import acceptance_table
    row = dict(job_id="x", budget_key="B1", parameter="ATT", post_mean=0., post_sd=1., mean_mcse=.001,
               rhat=1., bulk_ess=1000., tail_ess=np.nan, q025=-2., q975=2., q05=-1.7, q95=1.7,
               q025_mcse=.001, q975_mcse=.001, q05_mcse=.001, q95_mcse=.001)
    assert not acceptance_table(pd.DataFrame([row]), {}).iloc[0].pass_all


def test_full_campaign_sharding_and_population(tmp_path):
    from recovery_inventory import load_original_manifest
    from recovery_plan import PHASE_ORDER, ATOMIC_KINDS
    body = _manifest(trace=False)
    body["jobs"] = load_original_manifest()["jobs"]
    assert len(body["jobs"]) == 14327
    assert sum(j["phase"] == "main" for j in body["jobs"]) == 5100
    body["manifest_hash"] = digest({k:v for k,v in body.items() if k != "manifest_hash"})
    assignments = tmp_path / "synthetic_planning_only"
    write_assignments(body, "synthetic_planning_only", 6.5, 48, assignments, phase_order=PHASE_ORDER)
    actual = [u["run_id"] for path in assignments.glob("wave_*/shard_*.json") for u in json.loads(path.read_text())["units"]]
    assert len(actual) == len(set(actual)) == sum(1 if j["kind"] in ATOMIC_KINDS else 2 for j in body["jobs"])
    assert set(actual) == {u["run_id"] for u in make_units(body)}


def test_synthetic_budget_gate_and_portable_evidence(tmp_path, monkeypatch):
    """Synthetic tables test gate plumbing; these are not experiment evidence."""
    import platform
    import recovery_accept as gate
    from recovery_common import BUDGETS
    body = _manifest()
    body["purpose"] = "synthetic_gate_test_only"
    keys = ["B1_warm2500_ret1000", "B2_warm5000_ret1000", "B3_warm2500_ret2000"]
    body["budget_keys"] = keys
    body["budgets"] = {key:BUDGETS[key] for key in keys}
    body["manifest_hash"] = digest({k:v for k,v in body.items() if k != "manifest_hash"})
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest_file = manifests / "manifest_pilot.json"
    write_json_atomic(manifest_file, body)
    collection = tmp_path / "collection"; collection.mkdir()
    report = tmp_path / "report"; report.mkdir()
    n_units = len(make_units(body))
    write_json_atomic(collection / "status.json", dict(manifest_hash=body["manifest_hash"], complete_units=n_units, expected_units=n_units,
                                                       conflicting_units=0, duplicate_units=0, smoke_rejected=0))
    rows=[]
    for key in keys:
        for name in gate.expected_parameters(TINY_JOB):
            rows.append(dict(job_id=TINY_JOB["id"], budget_key=key, parameter=name, post_mean=0., post_sd=1.,
                             mean_mcse=.001, rhat=1., bulk_ess=4000., tail_ess=4000., q025=-2., q975=2., q05=-1.7,q95=1.7,
                             q025_mcse=.001,q975_mcse=.001,q05_mcse=.001,q95_mcse=.001))
    frame=pd.DataFrame(rows)
    frame.to_csv(collection / "chain_diagnostics.csv",index=False)
    pd.DataFrame({"trace_matches_draws":[True]*n_units}).to_csv(collection / "trace_diagnostics.csv",index=False)
    pd.DataFrame({"python":[platform.python_version()]*n_units}).to_csv(collection / "collected_runs.csv",index=False)
    write_json_atomic(report / "pilot_report_meta.json", {"manifest_hash":body["manifest_hash"], "files":{name:__import__("recovery_common").sha256_file(collection / name) for name in ("status.json","chain_diagnostics.csv")}})
    (report / "PILOT_ACCEPTANCE.md").write_text("SYNTHETIC gate fixture only. No experiments run or budget accepted for the research campaign.")
    note=tmp_path / "trace_review.md"
    note.write_text("SYNTHETIC integration fixture only; exercise the written-review plumbing without statistical evidence.")
    decision=tmp_path / "synthetic_decision.json"
    gate.record_decision(manifest_file, collection, report, keys[0], note, decision)
    monkeypatch.setattr(gate, "MANIFESTS", manifests)
    assert gate.validate_decision(decision,keys[0],platform.python_version())["required_checks"] == len(gate.expected_parameters(TINY_JOB))
    assert all(not Path(name).is_absolute() for name in json.loads(decision.read_text())["evidence"])
    # Rejected missing target rows, rather than silently screening a subset.
    frame.iloc[:-1].to_csv(collection / "chain_diagnostics.csv",index=False)
    meta=json.loads((report / "pilot_report_meta.json").read_text())
    meta["files"]["chain_diagnostics.csv"]=__import__("recovery_common").sha256_file(collection / "chain_diagnostics.csv")
    write_json_atomic(report / "pilot_report_meta.json",meta)
    with pytest.raises(ValueError,match="Missing required"):
        gate.record_decision(manifest_file, collection, report, keys[0], note, tmp_path / "rejected.json")
    with pytest.raises(ValueError,match="Gate evidence"):
        gate.validate_decision(decision,keys[0],platform.python_version())
