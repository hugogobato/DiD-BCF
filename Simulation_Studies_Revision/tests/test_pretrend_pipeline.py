"""Regression tests for the pre-trend diagnostic pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from did_bcf_revision.metrics import compute_metrics
from did_bcf_revision import pretrend_runner


def _decision_rows(estimand_id: str, method: str = "pretrend") -> pd.DataFrame:
    return pd.DataFrame({
        "dgp": "PT_hold",
        "setting": "test",
        "linearity_degree": 1,
        "N": 24,
        "rep": [0, 1, 2],
        "spec": "pretrend_unit",
        "estimand_type": "PRE",
        "estimand_id": estimand_id,
        "method": method,
        "post_mean": np.nan,
        "sd": np.nan,
        "q025": np.nan,
        "q05": np.nan,
        "q95": np.nan,
        "q975": np.nan,
        "p_bayes": [0.01, 0.03, 0.40],
        "true": 0.0,
    })


def test_decision_rule_metrics_use_posterior_tail_without_point_estimate():
    summaries = pd.concat([
        _decision_rows("any"),
        _decision_rows("any_bonf"),
    ], ignore_index=True)

    metrics = compute_metrics(summaries)

    for estimand_id in ("any", "any_bonf"):
        row = metrics.loc[metrics["estimand_id"] == estimand_id].iloc[0]
        assert row["n_reps"] == 3
        assert row["retention"] == 1.0
        assert np.isclose(row["reject05"], 1 / 3)
        assert np.isclose(row["reject10"], 2 / 3)
        assert row["role"] == "size"


def test_pretrend_att_path_dispatches_to_structured_spec(monkeypatch):
    called = {}
    estimate = pd.DataFrame({
        "estimand_type": ["ATT"],
        "estimand_id": ["overall"],
        "post_mean": [1.0],
    })

    monkeypatch.setattr(
        pretrend_runner,
        "_GENERATORS",
        {"canonical": lambda **kwargs: pd.DataFrame({"Y": [1.0]})},
    )
    monkeypatch.setattr(pretrend_runner, "fit_pretrend", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrend_runner, "pretrend_estimands", lambda fit: pd.DataFrame())
    monkeypatch.setattr(pretrend_runner, "twfe_pretrend", lambda df: pd.DataFrame())
    monkeypatch.setattr(pretrend_runner, "true_pretrend", lambda df: pd.DataFrame())
    monkeypatch.setattr(
        pretrend_runner,
        "true_estimands",
        lambda df: pd.DataFrame({
            "estimand_type": ["ATT"],
            "estimand_id": ["overall"],
            "true": [1.0],
        }),
    )

    def fake_fit_any(df, bcf_params=None, seed=None, spec=None):
        called["spec"] = spec
        return object()

    monkeypatch.setattr(pretrend_runner, "fit_any", fake_fit_any)
    monkeypatch.setattr(pretrend_runner, "plain_estimands", lambda fit: estimate)
    monkeypatch.setattr(pretrend_runner, "corrected_estimands", lambda fit, seed: estimate)

    out = pretrend_runner.process_rep(
        "canonical", {}, N=1, rep=0, setting="test", with_att=True)

    assert called["spec"] == "structured"
    assert len(out) == 2
