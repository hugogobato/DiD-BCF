import json
from itertools import product
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from extra_theory_experiments.correction import (
    OracleUnavailableError, algorithm2_draws, fit_propensity,
)
from extra_theory_experiments.manifest import build_manifest
from extra_theory_experiments.metrics import (
    assert_no_catt_broadcast, scalar_metrics, summarise_replications,
)
from extra_theory_experiments.reference import (
    assign_unit_folds, convolve_fold_draws, oracle_nuisance,
    reference_fold_convolution,
)
from extra_theory_experiments.reference import _outcome_pilot
from extra_theory_experiments.ablations import fit_panel_variant, run_information_ablation
from extra_theory_experiments import runner
from extra_theory_experiments.oracle_dgp import (
    generate_oracle_canonical_did, oracle_assignment_probability,
    oracle_control_long_difference, oracle_control_slope,
)
from extra_theory_experiments.controlled_mechanics import (
    reference_fold_mechanics, run_controlled_mechanics,
)


def test_algorithm2_sign_scaling_and_offfold_pilot():
    delta = np.array([1., 0., 1., 0.])
    dy = np.array([3., 1., 4., 2.])
    M = np.array([[.2, .4], [.3, .1], [.5, .7], [.2, .6]])
    pi = np.array([.4, .2, .6, .3])
    W = np.ones((4, 2))
    pilot = np.array([.1, .2, .3, .4])
    got = algorithm2_draws(delta, dy, M, pi, .5, W, m_pilot=pilot)
    wn = W / W.sum(axis=0, keepdims=True)
    aw = delta - (1 - delta) * pi / (1 - pi)
    expected_theta = np.sum(wn * aw[:, None] * (dy[:, None] - M), axis=0) / np.sum(
        wn * delta[:, None], axis=0)
    gamma = (delta - pi) / ((1 - pi) * .5)
    expected = expected_theta - gamma @ (pilot[:, None] - M) / len(delta)
    np.testing.assert_allclose(got, expected)


def test_fold_size_convolution_uses_all_selected_units():
    got = convolve_fold_draws({0: np.array([0., 2.]), 1: np.array([4., 6.])},
                              {0: 1, 1: 3})
    np.testing.assert_allclose(got, np.array([3., 5.]))


def test_outcome_pilot_uses_only_complementary_controls():
    df = _panel()
    # Treated units have DeltaY=2 and never-treated units DeltaY=1 in _panel.
    # If treated complement units leaked into m0, this intercept would be > 1.
    got = _outcome_pilot(df, np.arange(24), 1., 1,
                         np.array([0, 1, 12, 13]), method="intercept")
    assert np.allclose(got, 1.)


def test_propensity_clip_and_inverse_odds_ess():
    p = fit_propensity(np.zeros((4, 1)), np.array([1, 0, 1, 0]),
                       method="oracle", oracle_pi=np.array([.2, .2, .8, .8]),
                       clip=.1)
    assert p.diagnostics["max_control_odds"] == pytest.approx(4)
    expected_ess = 1 / ((.25 / 4.25) ** 2 + (4 / 4.25) ** 2)
    assert p.diagnostics["control_ess"] == pytest.approx(expected_ess)


def test_unsupported_oracle_rejected():
    df = pd.DataFrame({"unit_id": [0, 0], "time": [0, 1],
                       "cohort": [1., 1.], "Y": [0., 1.]})
    with pytest.raises(OracleUnavailableError):
        oracle_nuisance(df, 1., 1, np.array([1]), np.array([1.]))


def test_oracle_canonical_nuisances_are_exact_and_rows_run():
    with pytest.raises(ValueError, match="non-theorem stress"):
        generate_oracle_canonical_did(seed=4, n_units=80, alpha_sd=1.0)
    with pytest.raises(ValueError, match="heterogeneous.*unsupported"):
        generate_oracle_canonical_did(
            seed=4, n_units=80, effect_type="heterogeneous",
            non_theorem_stress=True)
    df = generate_oracle_canonical_did(seed=4, n_units=80, linearity_degree=2)
    post = df[df["time"] == 4].sort_values("unit_id")
    X = [post.X1.to_numpy(), post.X2.to_numpy(), post.X3.to_numpy(),
         post.X4.to_numpy(), post.X5.to_numpy()]
    transformed_X = [1.0 - X[0], *[-x for x in X[1:]]]
    for degree in (1, 2):
        pi = oracle_assignment_probability(*X, degree)
        pi_transformed = oracle_assignment_probability(*transformed_X, degree)
        np.testing.assert_allclose(pi + pi_transformed, 1.0)
        assert np.min(pi) > 1e-3 and np.max(pi) < 1.0 - 1e-3
        corners = np.asarray(list(product((0.0, 1.0), repeat=5)))
        corners[:, 1:] = 2.0 * corners[:, 1:] - 1.0
        corner_pi = oracle_assignment_probability(
            corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3],
            corners[:, 4], degree)
        assert np.min(corner_pi) > 1e-3 and np.max(corner_pi) < 1.0 - 1e-3
    expected_pi = oracle_assignment_probability(*X, 2)
    expected_slope = oracle_control_slope(
        *X, 2)
    np.testing.assert_allclose(post["pi_oracle"], expected_pi)
    np.testing.assert_allclose(
        post["m0_oracle"], oracle_control_long_difference(expected_slope, 4, 4))
    assert np.allclose(post["barpi_oracle"], 0.5)
    assert np.allclose(df["alpha"], 0.0)
    assert np.allclose(df.loc[df["D"] == 1, "CATT"], 3.0)
    assert np.allclose(df["gatt_population_oracle"], 3.0)
    assert df.attrs["truth"] == {
        "gatt_population_oracle": 3.0,
        "truth_source": "homogeneous_population_effect",
        "homogeneous": True,
    }
    m0, pi, barpi = oracle_nuisance(
        df, 4., 4, post.index.to_numpy(), post["D"].to_numpy())
    np.testing.assert_allclose(m0, post["m0_oracle"])
    np.testing.assert_allclose(pi, post["pi_oracle"])
    assert barpi == pytest.approx(0.5)

    tasks = [task for task in build_manifest("correction_audit", reps=1)
             if task.design == "oracle_canonical" and task.N == 200
             and task.estimator in {"oracle_current_hybrid",
                                    "oracle_reference_fold_convolution"}]
    assert len(tasks) == 4 and all(task.oracle_available for task in tasks)
    for task in tasks:
        out = runner.run_task(task, smoke=True, K=2)
        assert not out.empty
        assert "status" not in out or not np.any(out["status"] == "unavailable")
        assert np.allclose(out["true"].dropna(), 3.0)
        assert set(out["truth_source"]) == {"gatt_population_oracle"}


def test_error_sd_differs_from_raw_est_sd():
    out = scalar_metrics(np.array([1., 4.]), np.array([0., 2.]))
    assert out["empirical_sd_error"] == pytest.approx(np.sqrt(.5))
    assert out["raw_sd_est"] == pytest.approx(np.sqrt(4.5))
    assert out["bias_mcse"] == pytest.approx(np.sqrt(.5 / 2))


def test_controlled_mechanics_smoke_reports_both_laws(tmp_path):
    per_rep, metrics, archive = run_controlled_mechanics(
        out_dir=tmp_path / "mechanics", reps=2, n_draws=20, n_units=40)
    assert set(per_rep["scenario"]) == {"signal", "null"}
    assert set(per_rep["method"]) == {
        "full_algorithm1_bb", "reference_fold_convolution"}
    assert set(per_rep["estimand_type"]) == {"GATT", "ATT"}
    assert per_rep["posterior_m_fixed_exact_m0"].all()
    assert {"bias", "cover95", "mean_interval_length95",
            "null_rejection_rate"}.issubset(metrics.columns)
    assert metrics["estimand_id"].notna().all()
    assert (metrics["scenario"] == "null").any()
    assert archive.exists()
    with zipfile.ZipFile(archive) as handle:
        assert {"controlled_mechanics_per_replication.csv",
                "controlled_mechanics_metrics.csv",
                "controlled_mechanics_manifest.csv", "provenance.json",
                "README_run.txt"}.issubset(handle.namelist())


def test_reference_requires_exactly_k_valid_folds_for_small_n():
    df = _panel(n=8)

    def factory(panel, **kwargs):
        return FakeFit(panel, kwargs["seed"])

    with pytest.raises(ValueError, match="expected exactly K=4"):
        reference_fold_convolution(
            df, K=4, fold_seed=9, posterior_seed=3,
            fit_factory=factory, propensity_method="intercept")

    oracle = generate_oracle_canonical_did(seed=12, n_units=8)
    with pytest.raises(ValueError, match="expected exactly K=4"):
        reference_fold_mechanics(
            oracle, g=4, t=4, n_draws=8, fold_seed=9,
            posterior_seed=3, K=4)


def test_convolution_expected_folds_rejects_partial_mapping():
    with pytest.raises(ValueError, match="expected exactly 2 valid folds"):
        convolve_fold_draws({0: np.array([1., 2.])}, {0: 4}, expected_folds=2)


def test_replication_metrics_keep_mean_and_median_rows():
    frame = pd.DataFrame({
        "estimator": ["x", "x"], "estimand_type": ["ATT", "ATT"],
        "estimand_id": ["ATT", "ATT"], "post_mean": [1., 3.],
        "post_median": [0., 4.], "true": [0., 2.], "sd": [1., 1.],
        "q05": [-1., -1.], "q95": [2., 5.],
        "q025": [-2., -2.], "q975": [3., 6.],
    })
    out = summarise_replications(frame)
    assert set(out["point_summary"]) == {"mean", "median"}


def test_no_catt_broadcasting():
    good = pd.DataFrame({"estimand_type": ["GATT", "CATT"],
                         "estimand_id": ["g=1_t=2", "surface"]})
    assert_no_catt_broadcast(good)
    bad = good.copy()
    bad.loc[1, "estimand_id"] = "g=1_t=2"
    with pytest.raises(AssertionError):
        assert_no_catt_broadcast(bad)


def _panel(n=24):
    rows = []
    for u in range(n):
        cohort = 1. if u < n // 2 else np.inf
        for t in (0, 1, 2):
            rows.append({"unit_id": u, "time": t, "cohort": cohort,
                         "D": int(cohort == 1 and t >= 1),
                         "event_time": (float(t - 1) if cohort == 1 else np.nan),
                         "X1": float(u % 2), "X2": float(u),
                         "X3": float(u % 3), "X4": float(u % 4),
                         "X5": float(u % 5), "Y": float(t + (u < n // 2) * t),
                         "CATT": float((u < n // 2) * t)})
    return pd.DataFrame(rows)


class FakeFit:
    def __init__(self, panel, seed=0):
        self.df = panel.sort_values(["unit_id", "time"]).reset_index(drop=True)
        self.row_of = {(int(u), int(t)): i for i, (u, t) in enumerate(
            zip(self.df.unit_id, self.df.time))}
        self.n_draws = 4
        rng = np.random.default_rng(seed)
        self.mu_draws = (self.df.time.to_numpy(float)[:, None] +
                         rng.normal(size=(len(self.df), self.n_draws)) * .01)
        self.tau_draws = np.zeros((len(self.df), self.n_draws))


def test_reference_strict_eval_units_and_unit_folds():
    df = _panel()
    folds = assign_unit_folds(df.unit_id.to_numpy(), K=2, seed=9)
    assert len(set(folds.values())) == 2
    seen = []

    def factory(panel, **kwargs):
        seen.append(set(panel.unit_id.astype(int)))
        return FakeFit(panel, kwargs["seed"])

    out = reference_fold_convolution(df, K=2, fold_seed=9, posterior_seed=3,
                                     fit_factory=factory, propensity_method="intercept")
    assert out.method == "reference_fold_convolution"
    assert len(seen) == 2 and all(len(x) == len(set(x)) for x in seen)
    assert out.diagnostics["cells"]["g=1_t=1"]
    assert all(v["global_fold_bb"] for v in out.diagnostics["cells"]["g=1_t=1"].values())


def test_run_tasks_reuses_one_full_panel_fit_per_bundle(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_dgp", lambda task: _panel())
    monkeypatch.setattr(runner, "_production_fit",
                        lambda panel, **kwargs: (calls.append(1) or FakeFit(panel, 0)))
    from extra_theory_experiments.manifest import ExperimentTask
    tasks = [ExperimentTask("correction_audit", "baseline", 1, 24, 0, method)
             for method in ("raw_structured", "current_hybrid_logit")]
    runner.run_tasks(tasks, out_dir=tmp_path, resume=False)
    assert len(calls) == 1


def test_fit_cache_separates_effect_by_cohort_values(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_dgp", lambda task: _panel())
    def fake_fit(panel, **kwargs):
        calls.append(bool(kwargs["effect_by_cohort"]))
        return FakeFit(panel, 0)
    monkeypatch.setattr(runner, "_production_fit", fake_fit)
    from extra_theory_experiments.manifest import ExperimentTask
    tasks = [ExperimentTask("information_ablation", "baseline", 1, 24, 0, method)
             for method in ("full_panel_raw", "pooled_full_panel")]
    runner.run_tasks(tasks, out_dir=tmp_path, resume=False)
    assert calls == [True, False]


def test_reference_cache_reuses_fold_fits_across_propensity_variants():
    calls = []
    cache = {}
    def factory(panel, **kwargs):
        calls.append(1)
        return FakeFit(panel, kwargs["seed"])
    df = _panel()
    reference_fold_convolution(df, K=2, fold_seed=4, posterior_seed=5,
                               fit_factory=factory, fit_cache=cache,
                               propensity_method="logit", outcome_method="rf")
    reference_fold_convolution(df, K=2, fold_seed=4, posterior_seed=5,
                               fit_factory=factory, fit_cache=cache,
                               propensity_method="rf", outcome_method="rf")
    assert len(calls) == 2


def test_pooling_forces_effect_by_cohort_false_and_reduced_is_exact_gatt():
    df = _panel()
    seen = []
    def factory(panel, **kwargs):
        seen.append(bool(kwargs["effect_by_cohort"]))
        return FakeFit(panel, 0)
    fit_panel_variant(df, variant="full_panel_raw", fit_factory=factory)
    fit_panel_variant(df, variant="pooled_full_panel", fit_factory=factory)
    assert seen == [True, False]
    out, _, _ = run_information_ablation(
        df, variants=("reduced_cell",), fit_factory=factory)
    assert set(out["estimand_type"]) == {"GATT"}


def test_oracle_requires_exact_barpi():
    df = _panel()
    rows = np.array([0, 1])
    df["m0_oracle"] = 0.
    df["pi_oracle"] = .5
    with pytest.raises(OracleUnavailableError):
        oracle_nuisance(df, 1., 1, rows, np.array([1., 0.]))
    df["barpi_oracle"] = .5
    m, p, b = oracle_nuisance(df, 1., 1, rows, np.array([1., 0.]))
    assert b == .5


def test_deterministic_manifest_sharding():
    a = build_manifest("correction_audit", reps=2, n_shards=4, shard_id=1)
    b = build_manifest("correction_audit", reps=2, n_shards=4, shard_id=1)
    assert [x.as_dict() for x in a] == [x.as_dict() for x in b]
    assert all(x.shard == 1 and x.n_shards == 4 for x in a)
    same_rep = [x for x in build_manifest("correction_audit", reps=1)
                if x.design == "baseline" and x.degree == 1 and x.N == 200]
    assert len({x.seed for x in same_rep}) == 1
    assert len({x.shard for x in same_rep}) == 1
    oracle = [x for x in build_manifest("correction_audit", reps=1)
              if x.estimator.startswith("oracle")]
    assert len(oracle) == 32
    assert sum(x.oracle_available for x in oracle) == 8
    assert {x.design for x in oracle if x.oracle_available} == {"oracle_canonical"}


def test_multiwave_manifest_bundle_allocation_is_complete_and_disjoint():
    n_shards, n_waves = 4, 3
    allocations = {}
    all_tasks = []
    for wave in range(n_waves):
        for shard in range(n_shards):
            tasks = build_manifest(
                "correction_audit", reps=2, n_shards=n_shards,
                shard_id=shard, n_waves=n_waves, wave_id=wave)
            all_tasks.extend(tasks)
            for task in tasks:
                key = (task.design, task.degree, task.N, task.rep)
                allocations.setdefault(key, set()).add((task.wave_id, task.shard))
                assert task.n_waves == n_waves and task.n_shards == n_shards
    full = build_manifest("correction_audit", reps=2, n_shards=1)
    task_keys = lambda rows: {
        (x.design, x.degree, x.N, x.rep, x.estimator) for x in rows
    }
    assert task_keys(all_tasks) == task_keys(full)
    assert allocations and all(len(locations) == 1 for locations in allocations.values())
    repeated = build_manifest(
        "correction_audit", reps=2, n_shards=n_shards, shard_id=2,
        n_waves=n_waves, wave_id=1)
    assert [x.as_dict() for x in repeated] == [
        x.as_dict() for x in build_manifest(
            "correction_audit", reps=2, n_shards=n_shards, shard_id=2,
            n_waves=n_waves, wave_id=1)]


def test_notebooks_are_valid_and_download_one_zip():
    revision = Path(__file__).parents[2]
    root = revision / "DiD_BCF" / "Theory_Calibration"
    completion = revision / "DiD_BCF" / "Correction_Completion"
    books = sorted(root.glob("*.ipynb"))
    assert len(books) == 101
    for path in books:
        obj = json.loads(path.read_text())
        assert obj["nbformat"] == 4
        code = "\n".join("".join(c.get("source", [])) for c in obj["cells"]
                          if c["cell_type"] == "code")
        assert "git" in code and "hugogobato/DiD-BCF.git" in code
        assert "BRANCH = 'main'" in code
        assert "output_file = " in code
        assert "files.download(output_file)" in code
        assert "print(\"Downloaded:\", output_file)" in code
        assert "Not on Colab / download skipped" in code
        compile(code, str(path), "exec")
    compute_code = "\n".join("".join(c.get("source", [])) for c in
                               json.loads((root / "correction_audit_shard_00.ipynb").read_text())["cells"])
    assert "num_gfr': 50" in compute_code and "num_mcmc': 500" in compute_code
    assert "N_SHARDS = 48" in compute_code
    assert "n_shards=N_SHARDS" in compute_code and "bcf_params=BCF_PARAMS" in compute_code
    assert "ETE_N_WAVES" in compute_code and "ETE_WAVE_ID" in compute_code
    assert "wave_id=WAVE_ID" in compute_code and "n_waves=N_WAVES" in compute_code
    assert "_wave_" in compute_code and '"wave_id": WAVE_ID' in compute_code
    assert len(list(root.glob("correction_audit_shard_*.ipynb"))) == 48
    assert len(list(root.glob("information_ablation_shard_*.ipynb"))) == 48
    pilots = {
        "correction_bcf_pilot.ipynb": (
            "correction_audit", "serial", "degree == 2", "N=800",
            ["raw_structured", "current_hybrid_logit",
             "reference_fold_convolution_logit"], "expected cached sampler fits: 3"),
        "information_bcf_pilot.ipynb": (
            "information_ablation", "staggered", "degree == 3", "N=800",
            ["full_panel_raw", "reduced_cell", "pooled_full_panel"],
            "expected cached sampler fits: 11"),
        "oracle_bcf_pilot.ipynb": (
            "correction_audit", "oracle_canonical", "degree == 2", "N=800",
            ["oracle_current_hybrid", "oracle_reference_fold_convolution"],
            "expected cached sampler fits: 3"),
    }
    for name, (family, design, degree, n_value, estimators, fit_note) in pilots.items():
        pilot = json.loads((root / name).read_text())
        pilot_code = "\n".join("".join(c.get("source", [])) for c in pilot["cells"])
        assert "num_gfr': 2" in pilot_code and "SMOKE = False" in pilot_code
        assert f"FAMILY = '{family}'" in pilot_code
        assert f"task.design == '{design}'" in pilot_code
        assert degree in pilot_code and n_value in pilot_code
        task_count = 2 if name == "oracle_bcf_pilot.ipynb" else 3
        assert "task.rep == 0" in pilot_code
        assert f"assert len(tasks) == {task_count}" in pilot_code
        for estimator in estimators:
            assert estimator in pilot_code
        assert fit_note in pilot_code
        if name == "oracle_bcf_pilot.ipynb":
            assert "exact nuisance-column alignment" in pilot_code
    mechanics = json.loads((root / "controlled_mechanics.ipynb").read_text())
    mechanics_code = "\n".join("".join(c.get("source", [])) for c in mechanics["cells"])
    assert "controlled_mechanics" in mechanics_code
    assert "exact m0_oracle" in mechanics_code
    assert "not a theorem proof" in mechanics_code
    assert "ETE_MECH_REPS" in mechanics_code and "ETE_MECH_DRAWS" in mechanics_code

    completion_books = sorted(completion.glob("*.ipynb"))
    assert len(completion_books) == 48
    for path in completion_books:
        obj = json.loads(path.read_text())
        assert obj["nbformat"] == 4
        code = "\n".join("".join(c.get("source", [])) for c in obj["cells"]
                          if c["cell_type"] == "code")
        assert "FAMILY = 'correction_completion'" in code
        assert "N_SHARDS = 48" in code
        assert "n_shards=N_SHARDS" in code and "bcf_params=BCF_PARAMS" in code
        assert "num_gfr': 50" in code and "num_mcmc': 500" in code
        assert "ETE_N_WAVES" in code and "ETE_WAVE_ID" in code
        assert "wave_id=WAVE_ID" in code and "n_waves=N_WAVES" in code
        assert "_wave_" in code and '"wave_id": WAVE_ID' in code
        assert "ETE_REPS" in code
        assert "BRANCH = 'main'" in code
        assert "files.download(output_file)" in code
        compile(code, str(path), "exec")


def test_archive_aggregation_deduplicates_and_reports_metrics(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from aggregate_archives import aggregate_archives
    from extra_theory_experiments.manifest import ExperimentTask, manifest_frame
    task = ExperimentTask("correction_audit", "baseline", 1, 200, 0,
                          "raw_structured", shard=0, n_shards=1,
                          config_hash="abc")
    manifest = manifest_frame([task])
    summary = pd.DataFrame([{**task.as_dict(), "task_estimator": task.estimator,
                             "estimand_type": "ATT", "estimand_id": "ATT",
                             "method": "raw_structured", "post_mean": 1.,
                             "post_median": 1., "sd": 1., "q05": 0.,
                             "q95": 2., "q025": 0., "q975": 2., "true": 0.}])
    archive_path = tmp_path / "shard.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.csv", manifest.to_csv(index=False))
        archive.writestr("summary.csv", summary.to_csv(index=False))
    report, combined, metrics = aggregate_archives(
        [archive_path], output_dir=tmp_path / "out", n_shards=1)
    assert report["missing_shards"] == []
    assert len(combined) == 1 and set(metrics["point_summary"]) == {"mean", "median"}


def test_archive_aggregation_retains_unavailable_only_summary(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from aggregate_archives import aggregate_archives
    from extra_theory_experiments.manifest import ExperimentTask, manifest_frame
    task = ExperimentTask("correction_audit", "baseline", 1, 200, 0,
                          "oracle_current_hybrid", shard=0, n_shards=1,
                          config_hash="abc", oracle_available=False)
    manifest = manifest_frame([task])
    summary = pd.DataFrame([{
        "status": "unavailable",
        "reason": "exact nuisance unavailable",
    }])
    archive_path = tmp_path / "unavailable.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.csv", manifest.to_csv(index=False))
        archive.writestr("summary.csv", summary.to_csv(index=False))
    report, combined, metrics = aggregate_archives(
        [archive_path], output_dir=tmp_path / "out", n_shards=1)
    assert len(combined) == 1 and combined.iloc[0]["status"] == "unavailable"
    assert metrics.empty
    assert report["oracle_unavailable_not_failed"] is True
    assert report["unavailable_rows_without_task_key"] == 1


def test_archive_aggregation_reports_missing_waves(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from aggregate_archives import aggregate_archives
    from extra_theory_experiments.manifest import ExperimentTask, manifest_frame
    task = ExperimentTask("correction_audit", "baseline", 1, 200, 0,
                          "raw_structured", shard=0, n_shards=1,
                          wave_id=0, n_waves=2, config_hash="abc")
    manifest = manifest_frame([task])
    summary = pd.DataFrame([{
        **task.as_dict(), "task_estimator": task.estimator,
        "estimand_type": "ATT", "estimand_id": "ATT",
        "method": "raw_structured", "post_mean": 1.,
        "post_median": 1., "sd": 1., "q05": 0., "q95": 2.,
        "q025": 0., "q975": 2., "true": 0.,
    }])
    archive_path = tmp_path / "wave0.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.csv", manifest.to_csv(index=False))
        archive.writestr("summary.csv", summary.to_csv(index=False))
    report, _, _ = aggregate_archives(
        [archive_path], output_dir=tmp_path / "out", n_shards=1)
    assert report["wave_fields_present"] is True
    assert report["expected_n_waves"] == 2
    assert report["observed_waves"] == [0]
    assert report["missing_waves"] == [1]
    assert report["missing_wave_shards"] == [{"wave_id": 1, "shard": 0}]


def test_legacy_families_keep_seeds_hashes_and_shards():
    from extra_theory_experiments.manifest import _config_hash, _load_config
    assert _config_hash(_load_config("correction_audit")) == "1d4ec075e86f013d"
    assert _config_hash(_load_config("information_ablation")) == "7999df0eb8c5fe6a"
    first = build_manifest("correction_audit", reps=1, n_shards=4, shard_id=1)[0]
    assert first.seed == 4081262023
    assert (first.shard, first.n_shards, first.wave_id, first.n_waves) == (1, 4, 0, 1)
    assert first.dgp_params == {}
    info = build_manifest("information_ablation", reps=1, n_shards=4, shard_id=1)[0]
    assert info.seed == 1597900271 and info.dgp_params == {}


def test_correction_completion_manifest_covers_cells_and_params():
    tasks = build_manifest("correction_completion", n_shards=1)
    assert len(tasks) == 15200
    assert {t.estimator for t in tasks} == {
        "raw_structured", "reference_fold_convolution_rf"}
    assert len({t.design for t in tasks}) == 21
    pt_designs = {"PT_hold", "PT_conditional", "PT_violation_g05",
                  "PT_violation_g10", "PT_violation_g20", "PT_violation_g40",
                  "PT_violation_het10", "PT_violation_het20",
                  "PT_violation_het40", "PT_violation_a20"}
    assert {t.design for t in tasks if t.design.startswith("PT_")} == pt_designs
    assert {t.rep for t in tasks if t.design == "null"} == set(range(200))
    assert {t.rep for t in tasks if t.design == "PT_hold"} == set(range(200))
    assert {t.rep for t in tasks if t.design == "baseline"} == set(range(100))
    cells = {(t.design, t.degree, t.N) for t in tasks}
    for required in (
        ("baseline", 3, 200), ("serial", 3, 200), ("null", 3, 200),
        ("strong_confounder", 1, 200), ("strong_confounder", 2, 200),
        ("strong_confounder", 3, 200), ("selection_obs", 2, 200),
        ("selection_both", 2, 200), ("staggered", 3, 200),
        ("baseline_sweep", 1, 50), ("baseline_sweep", 2, 400),
        ("baseline_sweep_d3", 3, 800), ("serial_sweep", 1, 100),
        ("serial_sweep_d3", 3, 50),
    ):
        assert required in cells
    # No duplication of correction_audit's degree 1-2, N=200/800 sweep cells.
    assert ("baseline_sweep", 1, 200) not in cells
    assert ("baseline_sweep", 2, 800) not in cells
    assert ("serial_sweep", 2, 200) not in cells

    from extra_theory_experiments.runner import _dgp
    strong = next(t for t in tasks if t.design == "strong_confounder" and t.degree == 1)
    params = _dgp(strong).attrs["params"]
    assert params["alpha_sd"] == 2.0 and params["conf_strength"] == 1.5
    pt = next(t for t in tasks if t.design == "PT_violation_g20" and t.degree == 3)
    assert _dgp(pt).attrs["params"]["group_trend"] == 0.2
    sel = next(t for t in tasks if t.design == "selection_obs" and t.degree == 2)
    assert _dgp(sel).attrs["params"]["selection"] == "observable"
    stag = next(t for t in tasks if t.design == "staggered" and t.degree == 1)
    stag_df = _dgp(stag)
    assert stag_df.attrs["dgp"] == "staggered"
    assert stag_df.attrs["params"]["dynamic_ramp"] == 0.4


def _pretrend_panel(n_units=60, n_draws=7, seed=0):
    from did_bcf_revision.pretrend import PretrendFit
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        cohort = 4.0 if unit < n_units // 2 else np.inf
        x = {"X1": float(unit % 2), "X2": float(rng.normal()),
             "X3": float(rng.normal()), "X4": float(rng.normal()),
             "X5": float(rng.uniform(-1, 1))}
        for t in range(8):
            rows.append({"unit_id": unit, "time": t, "cohort": cohort,
                         "D": int(cohort == 4 and t >= 4),
                         "eventually_treated": int(cohort == 4),
                         "event_time": (float(t - 4) if cohort == 4 else np.nan),
                         **x, "Y": float(t)})
    df = pd.DataFrame(rows).sort_values(["unit_id", "time"]).reset_index(drop=True)
    row_of = {(int(u), int(t)): i for i, (u, t) in enumerate(
        zip(df.unit_id, df.time))}
    tau = rng.normal(size=(len(df), n_draws))
    return PretrendFit(df=df, tau_draws=tau, row_of=row_of, ref_k=-1)


def test_raw_pretrend_reproduces_published_point_summaries():
    from did_bcf_revision.pretrend import pretrend_estimands
    from extra_theory_experiments.pretrend_fold import raw_pretrend_estimands

    fit = _pretrend_panel()
    published = pretrend_estimands(fit)
    local = raw_pretrend_estimands(fit)
    assert set(local["estimand_type"]) == {"PRE", "PRE_SUB", "PRE_SUBC"}
    assert not {"any", "any_bonf"} & set(local["estimand_id"])
    joined = published.merge(local, on=["estimand_type", "estimand_id"],
                             suffixes=("_pub", "_loc"))
    assert len(joined) == len(local) > 0
    for col in ("post_mean", "sd", "q025", "q05", "q95", "q975"):
        np.testing.assert_allclose(joined[f"{col}_pub"], joined[f"{col}_loc"],
                                   atol=1e-12)
    np.testing.assert_allclose(joined["p_bayes_pub"],
                               joined["p_bayes_tail_min"], atol=1e-12)


def test_fold_pretrend_runs_smoke_and_pools_fold_draws():
    tasks = build_manifest("correction_completion", reps=1, n_shards=1)
    for estimator, method in (("raw_structured", "pretrend"),
                              ("reference_fold_convolution_rf",
                               "reference_fold_pretrend")):
        task = next(t for t in tasks if t.design == "PT_violation_het20"
                    and t.degree == 1 and t.estimator == estimator)
        out = runner.run_task(task, smoke=True)
        assert not out.empty
        assert set(out["method"]) == {method}
        assert {"PRE", "PRE_SUB", "PRE_SUBC"}.issubset(set(out["estimand_type"]))
        assert out["true"].notna().all()
    het = next(t for t in tasks if t.design == "PT_violation_het20"
               and t.degree == 1 and t.estimator == "raw_structured")
    truth = runner._dgp(het)
    from did_bcf_revision.pretrend import true_pretrend
    tp = true_pretrend(truth)
    slope = tp[(tp.estimand_type == "PRE") & (tp.estimand_id == "slope")]
    contrast = tp[(tp.estimand_type == "PRE_SUBC") &
                  (tp.estimand_id == "X1_slope")]
    # The aggregate differential slope cancels to Monte-Carlo error while the
    # X1 contrast carries the full 2*kappa = 0.4 violation.
    assert abs(float(slope["true"].iloc[0])) < 0.1
    assert np.isclose(float(contrast["true"].iloc[0]), 0.4, atol=1e-12)


def test_completion_smoke_aggregation(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from aggregate_archives import aggregate_archives
    from extra_theory_experiments.manifest import manifest_frame
    from extra_theory_experiments.runner import run_tasks
    tasks = build_manifest("correction_completion", reps=1, n_shards=1)
    wanted = {("baseline", 3, 200), ("strong_confounder", 1, 200),
              ("staggered", 1, 200), ("PT_hold", 1, 200)}
    subset = [t for t in tasks if (t.design, t.degree, t.N) in wanted]
    manifest_path = tmp_path / "manifest.csv"
    manifest_frame(subset).to_csv(manifest_path, index=False)
    summary = run_tasks(subset, out_dir=tmp_path / "checkpoints", smoke=True,
                        resume=False)
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)
    archive_path = tmp_path / "completion.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(manifest_path, arcname="manifest.csv")
        archive.write(summary_path, arcname="summary.csv")
    report, combined, metrics = aggregate_archives(
        [archive_path], output_dir=tmp_path / "aggregated",
        family="correction_completion", n_shards=1)
    assert report["missing_shards"] == []
    assert report["missing_task_bundles"] == []
    assert report["config_hashes"]
    assert not combined.empty and not metrics.empty
    assert set(metrics["point_summary"]) == {"mean", "median"}

