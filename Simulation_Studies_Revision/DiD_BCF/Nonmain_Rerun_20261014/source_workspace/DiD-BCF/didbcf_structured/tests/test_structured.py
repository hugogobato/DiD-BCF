"""Tests for the structured DiD-BCF model.

The important ones are :func:`test_two_way_restriction_holds` and
:func:`test_time_varying_covariates_break_the_restriction`: together they say
what the restriction is and what it needs.  Everything runs on tiny panels with
a short sampler; these check structure, not statistical performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from didbcf_structured import (StructuredDiDBCF, build_design,
                               structured_estimands)
from didbcf_structured.model import _stochtree_seed

SHORT = dict(num_gfr=5, num_mcmc=20, keep_every=1, num_chains=1, seed=0)


def make_panel(n_units=120, n_pre=3, n_post=3, seed=0, time_varying=False,
               effect=3.0, alpha_sd=1.0, conf_strength=1.0) -> pd.DataFrame:
    """Canonical DiD panel with selection on an unobserved unit effect.

    ``time_varying`` redraws the covariates every period, which is what the
    original DiD-BCF simulation study did and what breaks the two-way
    restriction (see the README).
    """
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    adoption = n_pre

    alpha_raw = rng.normal(size=n_units)
    alpha = alpha_sd * alpha_raw
    utility = conf_strength * alpha_raw
    utility = utility - np.median(utility)
    treated = (rng.uniform(size=n_units) < 1 / (1 + np.exp(-1.5 * utility))).astype(int)
    treated[0], treated[1] = 1, 0                       # guard degenerate draws
    cohort = np.where(treated == 1, float(adoption), np.inf)

    Xu = {f"X{j}": rng.normal(size=n_units) for j in range(1, 6)}
    tau_i = effect + 1.5 * (Xu["X1"] > 0) + 0.75 * Xu["X2"]
    level = -0.75 * Xu["X1"] + 0.5 * Xu["X2"] - 1.3 * Xu["X4"]
    slope = 0.3 * Xu["X3"]

    rows = []
    for t in range(T):
        X = ({f"X{j}": rng.normal(size=n_units) for j in range(1, 6)}
             if time_varying else Xu)
        D = ((cohort != np.inf) & (t >= cohort)).astype(int)
        catt = tau_i * D
        y = (alpha + 0.2 * t + level + slope * t + rng.normal(size=n_units) + catt)
        rows.append(pd.DataFrame({
            "unit_id": np.arange(n_units), "time": t, "cohort": cohort, "D": D,
            "event_time": np.where(cohort == np.inf, np.nan, t - cohort),
            **X, "CATT": catt, "Y": y,
        }))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
def test_design_blocks_exclude_the_forbidden_column():
    dm = build_design(make_panel(n_units=20))
    assert "time" not in dm.level_cols          # a(g, x) has no time argument
    assert "cohort_code" not in dm.trend_cols   # b(t, x) has no cohort argument
    assert dm.level.shape[1] == len(dm.level_cols)
    assert dm.trend.shape[1] == len(dm.trend_cols)


def test_never_treated_event_time_is_finite():
    dm = build_design(make_panel(n_units=20))
    assert np.isfinite(dm.effect).all()


# --------------------------------------------------------------------------- #
# The restriction
# --------------------------------------------------------------------------- #
def test_two_way_restriction_holds():
    """The DiD contrast of the fit equals the treatment forest, exactly."""
    model = StructuredDiDBCF().sample(make_panel(), **SHORT)
    check = model.check_identification()
    assert check["identified"], check
    assert check["rows_checked"] > 0


def test_two_way_restriction_holds_with_random_effects():
    model = StructuredDiDBCF(rfx="unit").sample(make_panel(), **SHORT)
    assert model.check_identification()["identified"]


def test_time_varying_covariates_break_the_restriction():
    """The assumption the restriction rests on, stated as a failing check.

    With covariates redrawn each period, ``a(g, X_it)`` moves with ``t`` and the
    group gap it carries is no longer the same in the post and the reference
    period, so the baseline stops cancelling from the contrast.  The check must
    notice.
    """
    model = StructuredDiDBCF().sample(make_panel(time_varying=True), **SHORT)
    check = model.check_identification()
    assert not check["identified"]
    assert check["max_abs_tau_minus_contrast"] > 1e-6


# --------------------------------------------------------------------------- #
# Sampler output
# --------------------------------------------------------------------------- #
def test_draw_shapes_and_chain_concatenation():
    panel = make_panel(n_units=60)
    model = StructuredDiDBCF().sample(
        panel, num_gfr=3, num_mcmc=10, keep_every=1, num_chains=2, seed=0)
    n_obs = len(panel)
    assert model.tau_draws.shape == (n_obs, 20)
    assert model.mu_draws.shape == (n_obs, 20)
    assert model.sigma2_draws.shape == (20,)
    assert model.tau_0_draws.shape == (20,)
    assert np.isfinite(model.tau_draws).all()
    assert (model.sigma2_draws > 0).all()


@pytest.mark.parametrize("rfx", ["none", "unit"])
def test_three_forests_share_one_residual(rfx):
    """The invariant the whole design rests on: it is one MCMC, not three.

    ``sigma^2`` is drawn inside C++ from the ``Residual`` object all three forest
    samplers write to.  If any block's contribution were not being subtracted
    from that shared residual, the ``sigma^2`` C++ returns would not match the
    residual implied by the blocks' own returned predictions.  Reconstruct
    ``r_s = y - mu_s - tau_s D_it`` per retained draw and compare the proper
    inverse-gamma conditional mean with ``sigma2_s``.
    """
    panel = make_panel(n_units=120)
    model = StructuredDiDBCF(rfx=rfx).sample(
        panel, num_gfr=20, num_mcmc=120, keep_every=1, num_chains=1, seed=0)

    y = model.design.y[:, None]
    Z = model.design.Z[:, None]
    implied = ((y - model.mu_draws - model.tau_draws * Z) ** 2).mean(axis=0)
    a = model.global_prior.sigma2_shape + model.design.n / 2
    implied = (model.global_prior.sigma2_rate * model.y_std**2
               + model.design.n * implied / 2) / (a - 1)
    sampled = model.sigma2_draws

    noise = 1 / np.sqrt(a - 2)
    assert np.abs(sampled - implied).mean() / implied.mean() < 3 * noise
    # Correlation depends on how much the conditional mean varies and has no
    # universal lower bound. The normalized conditional draw has mean one
    # and known variance, regardless of changes in the residual scale.
    ratio = sampled / implied
    assert abs(ratio.mean() - 1) < 4 * noise / np.sqrt(len(ratio))
    assert .5 < ratio.std(ddof=1) / noise < 1.5

    # The check has teeth: drop the trend block and the agreement collapses.
    a_only = model.level_trend_draws()[0]
    bad = ((y - a_only - model.tau_draws * Z) ** 2).mean(axis=0)
    assert np.abs(sampled - bad).mean() / implied.mean() > 0.5


def test_gauge_normalisation_leaves_mu_unchanged():
    model = StructuredDiDBCF().sample(make_panel(n_units=60), **SHORT)
    a_tilde, b_tilde = model.level_trend_draws()
    assert np.allclose(a_tilde + b_tilde, model.mu_draws, atol=1e-8)
    # b(t_1, .) = 0 by construction of the gauge.
    first = model.design.time_values == model.design.time_values.min()
    assert np.allclose(b_tilde[first], 0.0, atol=1e-8)


def test_recovers_the_effect_the_unrestricted_model_leaks():
    """Retention on treated post rows should be near one, not near a half."""
    panel = make_panel(n_units=200, seed=1)
    model = StructuredDiDBCF().sample(
        panel, num_gfr=15, num_mcmc=60, keep_every=1, num_chains=1, seed=1)
    treated = model.design.Z == 1
    att_true = panel.loc[panel["D"] == 1, "CATT"].mean()
    retention = model.tau_draws[treated].mean() / att_true
    assert 0.8 < retention < 1.2, retention


# --------------------------------------------------------------------------- #
# Estimands
# --------------------------------------------------------------------------- #
def test_estimand_table_schema():
    model = StructuredDiDBCF().sample(make_panel(n_units=60), **SHORT)
    table = structured_estimands(model)
    assert set(table["estimand_type"]) == {"GATT", "ES", "ATT", "CATT"}
    att = table.query("estimand_type == 'ATT'").iloc[0]
    assert att["q025"] < att["post_mean"] < att["q975"]
    assert (table.query("estimand_type == 'GATT'")["k"] >= 0).all()


def test_contrast_estimands_match_forest_estimands():
    model = StructuredDiDBCF().sample(make_panel(n_units=60), **SHORT)
    forest = structured_estimands(model).query("estimand_type == 'ATT'")
    contrast = structured_estimands(model, use_contrast=True).query(
        "estimand_type == 'ATT'")
    assert np.isclose(forest["post_mean"].iloc[0], contrast["post_mean"].iloc[0],
                      atol=1e-8)


def test_rejects_unknown_rfx():
    with pytest.raises(ValueError):
        StructuredDiDBCF(rfx="household")


def test_large_seeds_are_normalized_for_stochtree_cpp():
    """Manifest/chain seeds may exceed RngCpp's signed 32-bit constructor."""
    assert _stochtree_seed(None) == -1
    assert _stochtree_seed(-1) == -1
    assert 0 <= _stochtree_seed(3928232464000) < 2 ** 31
    assert _stochtree_seed(2 ** 31) == 0
    model = StructuredDiDBCF().sample(
        make_panel(n_units=30), num_gfr=1, num_mcmc=2, keep_every=1,
        num_chains=2, seed=3928232464)
    assert model.n_draws == 4
