"""Pipeline invariants; no production-size sampling."""
import json
import numpy as np
import pytest
import campaign


def test_manifest_counts_and_unique_ids(tmp_path):
    path = tmp_path / "manifest.json"
    campaign.prepare(path)
    m = campaign.verify_manifest(path)
    assert m["phases"] == dict(main=5100, diagnostic=4000, pt_impact=4000,
                              sensitivity=1200, application=9, application_sensitivity=18)
    assert len({j["id"] for j in m["jobs"]}) == len(m["jobs"])
    assert set(j["shard"] for j in m["jobs"]) == set(range(48))
    for phase in ("main", "diagnostic", "pt_impact", "sensitivity"):
        assert set(j["shard"] for j in m["jobs"] if j["phase"] == phase) == set(range(48))
    assert all(j["sigma2_shape"] > 0 and j["sigma2_rate"] > 0 for j in m["jobs"])
    m["sampler"]["num_mcmc"] = 1
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="altered"):
        campaign.verify_manifest(path)


def test_adapter_forwards_prior_and_cohort_option():
    from did_bcf_revision.dgps import generate_canonical_did
    from did_bcf_revision.structured import fit_structured
    df = generate_canonical_did(seed=3, n_units=50)
    fit = fit_structured(df, bcf_params=dict(num_gfr=2, num_mcmc=3,
                          num_chains=1, keep_every=1, sigma2_rate=10.),
                         effect_by_cohort=False, seed=3)
    model = fit.structured_model
    assert model.prior_metadata["global_prior"]["sigma2_rate"] == 10.
    assert model.prior_metadata["effect_by_cohort"] is False
    assert np.isfinite(model.sigma2_draws).all()
    assert (model.sigma2_draws > 0).all()


def test_historical_writer_guards():
    from did_bcf_revision import runner, pretrend_runner
    for module in (runner, pretrend_runner):
        with pytest.raises(ValueError, match="Historical Results"):
            module.run_experiment(None)


def test_diagnostic_rejects_improper_prior():
    from did_bcf_revision.pretrend import fit_pretrend
    with pytest.raises(ValueError, match="positive and finite"):
        fit_pretrend(None, bcf_params=dict(sigma2_shape=0, sigma2_rate=0))


def test_resume_integrity_and_historical_output_guard(tmp_path):
    # Existing-job handling occurs before sampling and verifies every file.
    job = {"id": "test"}
    manifest = {"manifest_hash": "test-manifest"}
    folder = tmp_path / "production" / "test"
    folder.mkdir(parents=True)
    (folder / "complete.json").write_text(json.dumps(dict(
        identity=dict(job=job, manifest_hash="test-manifest", smoke=False),
        outputs={"summaries.csv": "invalid"})))
    (folder / "summaries.csv").write_text("not a completed result")
    with pytest.raises(ValueError, match="Corrupt"):
        campaign.execute(job, manifest, tmp_path)
    with pytest.raises(ValueError, match="historical"):
        campaign.execute(job, manifest, campaign.REV / "Results")
