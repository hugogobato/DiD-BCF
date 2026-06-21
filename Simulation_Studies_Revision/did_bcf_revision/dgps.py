"""Data-generating processes for the DiD-BCF revision simulations.

This module implements the new DGPs requested in ``REVISION_PLAN.md``:

* **Workstream B1** -- *canonical* difference-in-differences, i.e. selection on
  **unobserved, time-invariant** heterogeneity.  Relative to the original
  simulation suite (covariates drawn iid per unit-time, selection only on
  observables, iid errors) the canonical DGP adds:

    1. a unit-level **unobserved fixed effect** ``alpha_i`` in ``Y_it(0)`` that
       is correlated with treatment assignment (``conf_strength``);
    2. **persistent covariates** -- drawn once per unit, not per unit-time;
    3. optional **serially correlated errors** (AR(1) within unit, ``ar1_rho``);
    4. **covariate-dependent trends**, so that parallel trends holds only
       *conditional* on the observed covariates -- this is what makes the
       conditioning in DiD-BCF (and the propensity term in the posterior
       correction) actually load.

  The strength of the unobserved confounder (``alpha_sd`` and ``conf_strength``)
  is a knob, so the suite can show where plain DiD-BCF -- which absorbs only a
  *group-level* intercept, not unit fixed effects -- holds up and where the
  differencing implied by the posterior correction helps.

* **Workstream D** -- *staggered adoption* with treatment effects that vary by
  **both event-time ``k`` and cohort ``g``** (the exact case Goodman-Bacon 2021
  shows breaks TWFE), again with an unobserved unit effect.

Every generator returns a tidy long-format ``pandas.DataFrame`` and stores the
*true* individual effect (``CATT``) so that bias / variance / coverage can be
computed downstream.  The unobserved effect ``alpha`` is included in the frame
**for diagnostics only** and must never be passed to an estimator.

The conventions (column names, ``cohort`` = first treated period with
``np.inf`` for never-treated, ``D`` = post-treatment indicator) are shared by
the estimation, metrics and Goodman-Bacon modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "generate_canonical_did",
    "generate_staggered_did",
    "true_estimands",
    "DEFAULT_CANONICAL_PARAMS",
    "DEFAULT_STAGGERED_PARAMS",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ar1_errors(n_units: int, n_periods: int, rho: float, scale: float,
                rng: np.random.Generator) -> np.ndarray:
    """Return an ``(n_units, n_periods)`` matrix of AR(1) errors.

    With ``rho == 0`` this is simply iid ``N(0, scale^2)``.  The process is
    initialised at its stationary distribution so the marginal variance is
    ``scale^2`` at every period.
    """
    eps = np.empty((n_units, n_periods))
    innov_sd = scale * np.sqrt(1.0 - rho ** 2) if abs(rho) < 1 else scale
    eps[:, 0] = rng.normal(0.0, scale, size=n_units)
    for t in range(1, n_periods):
        eps[:, t] = rho * eps[:, t - 1] + rng.normal(0.0, innov_sd, size=n_units)
    return eps


def _covariate_block(n_units: int, rng: np.random.Generator) -> dict:
    """Persistent, unit-level covariates (drawn once per unit).

    * ``X1`` Bernoulli(0.5)              -- binary, effect modifier
    * ``X2`` N(0, 1)                     -- continuous, effect modifier
    * ``X3`` N(0, 1)                     -- continuous, drives covariate trends
    * ``X4`` N(0, 1)                     -- continuous, prognostic only
    * ``X5`` Uniform(-1, 1)              -- continuous, prognostic only
    """
    return {
        "X1": rng.binomial(1, 0.5, size=n_units).astype(float),
        "X2": rng.normal(0.0, 1.0, size=n_units),
        "X3": rng.normal(0.0, 1.0, size=n_units),
        "X4": rng.normal(0.0, 1.0, size=n_units),
        "X5": rng.uniform(-1.0, 1.0, size=n_units),
    }


def _prognostic_levels(Xu: dict, linearity_degree: int) -> np.ndarray:
    """Time-invariant part of E[Y(0)] driven by covariates, f(X_i)."""
    X1, X2, X3, X4, X5 = (Xu["X1"], Xu["X2"], Xu["X3"], Xu["X4"], Xu["X5"])
    if linearity_degree == 1:                       # fully linear
        return -0.75 * X1 + 0.5 * X2 - 0.5 * X3 - 1.3 * X4 + 1.8 * X5
    if linearity_degree == 2:                       # half non-linear
        return (-0.75 * X1 ** 2 + 0.5 * np.exp(X2 / 2.0)
                - 0.5 * X3 - 1.3 * X4 + 1.8 * X5)
    # linearity_degree >= 3: strongly non-linear
    return (-0.75 * X1 + 0.5 * np.abs(X2) + 0.8 * np.sin(2 * X3)
            - 1.3 * np.sqrt(np.abs(X4)) + 1.8 * X5 ** 2)


def _effect_function(Xu: dict, base: float, effect_type: str) -> np.ndarray:
    """Unit-level treatment-effect function tau(X_i).

    ``homogeneous`` -> constant ``base``.
    ``heterogeneous`` -> effect modulated by X1 (binary) and X2 (continuous).
    """
    if effect_type == "homogeneous":
        return np.full_like(Xu["X1"], base, dtype=float)
    modifier = 1.5 * Xu["X1"] + 0.75 * np.tanh(Xu["X2"])
    return base + modifier


# --------------------------------------------------------------------------- #
# Workstream B1 -- canonical (single adoption time) DiD
# --------------------------------------------------------------------------- #
DEFAULT_CANONICAL_PARAMS = dict(
    n_units=200,                   # matches the original study's panel size
    num_pre_periods=4,
    num_post_periods=4,
    linearity_degree=1,
    base_effect=3.0,
    effect_type="heterogeneous",   # or "homogeneous"
    # --- canonical-DiD machinery -------------------------------------------
    alpha_sd=1.0,                  # Var(alpha_i)^(1/2): strength of unit FE
    conf_strength=1.0,             # corr(alpha_i, treatment): selection on unobs.
    selection="unobservable",      # "unobservable" | "observable" | "both"
    trend_heterogeneity=0.3,       # covariate-dependent slope (conditional PTA)
    ar1_rho=0.0,                   # within-unit serial correlation of errors
    epsilon_scale=1.0,
    treated_share_target=0.5,
)


def generate_canonical_did(seed: int = 0, **overrides) -> pd.DataFrame:
    """Canonical DiD panel with selection on an unobserved unit effect.

    Outcome model (potential outcome under control):

        Y_it(0) = alpha_i + gamma_t + f(X_i) + s(X_i) * t + eps_it

    where ``alpha_i`` is the unobserved, time-invariant unit effect (correlated
    with treatment when ``conf_strength > 0``), ``gamma_t`` a common time trend,
    ``f(X_i)`` a covariate level term, ``s(X_i) * t`` a *covariate-dependent*
    trend (so parallel trends holds only conditional on X), and ``eps_it`` iid
    or AR(1).  The realised outcome adds ``tau(X_i) * D_it``.

    Notes
    -----
    * ``alpha_i`` is a pure level shift, so it is differenced out by any DiD
      contrast; DiD-BCF, however, models *levels* and absorbs only a
      group-level intercept, so the within-group part of ``alpha_i`` lands in
      the residual.  Because that residual is unit-persistent (an unmodelled
      random effect, exactly like AR(1) errors), plain DiD-BCF point estimates
      stay roughly unbiased but its credible intervals -- built under
      conditional independence -- become over-confident.  This is the canonical
      setting in which the posterior correction's (cluster) Bayesian bootstrap
      should restore coverage.
    * ``cohort`` is the single adoption period for treated units and ``np.inf``
      for never-treated.
    """
    p = {**DEFAULT_CANONICAL_PARAMS, **overrides}
    rng = np.random.default_rng(seed)

    n = int(p["n_units"])
    n_pre, n_post = int(p["num_pre_periods"]), int(p["num_post_periods"])
    n_periods = n_pre + n_post
    adoption = n_pre  # treated units switch on at the first post period

    Xu = _covariate_block(n, rng)

    # --- unobserved unit effect ------------------------------------------- #
    alpha_raw = rng.normal(0.0, 1.0, size=n)              # standardised draw
    alpha = p["alpha_sd"] * alpha_raw

    # --- treatment assignment (selection) --------------------------------- #
    # Observed driver and unobserved driver of the assignment utility.
    obs_index = 0.8 * Xu["X1"] + 0.6 * Xu["X4"]           # depends on covariates
    unobs_index = p["conf_strength"] * alpha_raw          # selection on unobs.
    if p["selection"] == "observable":
        utility = obs_index
    elif p["selection"] == "unobservable":
        utility = unobs_index
    else:                                                  # "both"
        utility = obs_index + unobs_index
    # Centre the utility so the treated share hits the target on average.
    target = float(p["treated_share_target"])
    utility = utility - np.quantile(utility, 1.0 - target)
    prob = _sigmoid(1.5 * utility + rng.normal(0.0, 0.5, size=n))
    treated = (rng.uniform(size=n) < prob).astype(int)
    # Guard against a degenerate draw with no treated / no controls.
    if treated.sum() == 0:
        treated[rng.integers(0, n)] = 1
    if treated.sum() == n:
        treated[rng.integers(0, n)] = 0

    cohort = np.where(treated == 1, float(adoption), np.inf)

    # --- effect and trend functions --------------------------------------- #
    tau_i = _effect_function(Xu, float(p["base_effect"]), p["effect_type"])
    f_levels = _prognostic_levels(Xu, int(p["linearity_degree"]))
    slope_i = p["trend_heterogeneity"] * Xu["X3"]         # covariate trend

    eps = _ar1_errors(n, n_periods, float(p["ar1_rho"]),
                      float(p["epsilon_scale"]), rng)

    # --- assemble the long panel ------------------------------------------ #
    beta_0, beta_time = -0.5, 0.2
    rows = []
    for t in range(n_periods):
        gamma_t = beta_time * (t ** 2 if int(p["linearity_degree"]) >= 3 else t)
        D = ((cohort != np.inf) & (t >= cohort)).astype(int)
        y0 = (beta_0 + alpha + gamma_t + f_levels + slope_i * t + eps[:, t])
        catt = tau_i * D
        y = y0 + catt
        rows.append(pd.DataFrame({
            "unit_id": np.arange(n),
            "time": t,
            "cohort": cohort,
            "D": D,
            "eventually_treated": treated,
            "event_time": np.where(cohort == np.inf, np.nan, t - cohort),
            **{k: Xu[k] for k in Xu},
            "alpha": alpha,                 # UNOBSERVED -- diagnostics only
            "tau_true": tau_i,
            "CATT": catt,
            "Y": y,
        }))

    df = pd.concat(rows, ignore_index=True)
    df["post"] = (df["time"] >= adoption).astype(int)
    # treatment_group: 0 = never-treated, 1 = the (single) treated cohort.
    df["treatment_group"] = df["eventually_treated"].astype(int)
    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
    df.attrs["dgp"] = "canonical"
    df.attrs["params"] = p
    df.attrs["adoption"] = adoption
    return df


# --------------------------------------------------------------------------- #
# Workstream D -- staggered adoption, cohort x event-time varying effects
# --------------------------------------------------------------------------- #
DEFAULT_STAGGERED_PARAMS = dict(
    n_units=200,                  # matches the original study's panel size
    num_pre_periods=4,            # periods before the EARLIEST adoption
    num_post_periods=4,           # periods after the earliest adoption
    cohort_offsets=(0, 1, 2),     # adoption at earliest+offset for cohorts 1..K
    cohort_shares=(0.25, 0.25, 0.25),  # 3 treated cohorts + never-treated (4 groups)
    linearity_degree=1,
    effect_type="heterogeneous",
    base_effect=2.0,
    # cohort x event-time effect:  tau(g, k, X) = cohort_mult[g] * ramp(k) * tau(X)
    cohort_multipliers=(1.0, 1.5, 2.0),   # later cohorts have larger effects
    dynamic_ramp=0.4,             # effect grows by this per event period
    # --- canonical-DiD machinery -------------------------------------------
    alpha_sd=1.0,
    conf_strength=1.0,
    selection="unobservable",
    trend_heterogeneity=0.3,
    ar1_rho=0.0,
    epsilon_scale=1.0,
)

def generate_staggered_did(seed: int = 0, **overrides) -> pd.DataFrame:
    """Staggered-adoption panel with cohort- *and* event-time-varying effects.

    Treatment effect for a treated unit i of cohort g at event time
    ``k = t - g >= 0``:

        tau(g, k, X_i) = cohort_mult[g] * (1 + dynamic_ramp * k) * tau(X_i)

    so effects (i) differ across cohorts and (ii) grow with exposure -- the
    heterogeneity Goodman-Bacon (2021) shows contaminates the TWFE estimator
    through "already-treated as control" comparisons.  An unobserved unit
    effect ``alpha_i`` (optionally correlated with cohort timing) and
    covariate-dependent trends are included as in :func:`generate_canonical_did`.
    """
    p = {**DEFAULT_STAGGERED_PARAMS, **overrides}
    rng = np.random.default_rng(seed)

    n = int(p["n_units"])
    n_pre, n_post = int(p["num_pre_periods"]), int(p["num_post_periods"])
    n_periods = n_pre + n_post
    earliest = n_pre
    offsets = list(p["cohort_offsets"])
    K = len(offsets)
    adoption_times = [earliest + off for off in offsets]
    if max(adoption_times) >= n_periods:
        raise ValueError("A cohort adopts after the panel ends; increase "
                         "num_post_periods or reduce cohort_offsets.")

    Xu = _covariate_block(n, rng)
    alpha_raw = rng.normal(0.0, 1.0, size=n)
    alpha = p["alpha_sd"] * alpha_raw

    # --- assign each unit to a cohort (1..K) or never-treated (0) ---------- #
    shares = list(p["cohort_shares"])
    if len(shares) != K:
        raise ValueError("cohort_shares must have one entry per cohort offset.")
    never_share = 1.0 - sum(shares)
    if never_share < 0:
        raise ValueError("cohort_shares sum to more than 1.")
    # Cohort utilities: never-treated is the baseline (utility 0); cohorts get a
    # share-based intercept plus a covariate term plus the unobserved driver.
    obs_index = 0.6 * Xu["X1"] + 0.4 * Xu["X4"]
    unobs_index = p["conf_strength"] * alpha_raw
    if p["selection"] == "observable":
        driver = obs_index
    elif p["selection"] == "unobservable":
        driver = unobs_index
    else:
        driver = obs_index + unobs_index

    intercepts = [0.0]                      # never-treated baseline
    log_shares = np.log(np.clip([never_share] + shares, 1e-6, None))
    util = np.zeros((n, K + 1))
    util[:, 0] = log_shares[0] + rng.gumbel(size=n)        # never-treated
    for j in range(K):
        # earlier cohorts get a stronger pull from the unobserved driver, so
        # that timing is correlated with alpha (selection into *when* to treat).
        timing_pull = (1.0 - j / max(K - 1, 1)) * driver
        util[:, j + 1] = (log_shares[j + 1] + 0.8 * driver + 0.5 * timing_pull
                          + rng.gumbel(size=n))
        intercepts.append(log_shares[j + 1])
    choice = util.argmax(axis=1)            # 0 = never, 1..K = cohort index

    cohort = np.full(n, np.inf)
    cohort_idx = np.zeros(n, dtype=int)     # 0 never, else 1..K
    for j in range(K):
        mask = choice == (j + 1)
        cohort[mask] = float(adoption_times[j])
        cohort_idx[mask] = j + 1
    # Guarantee at least one never-treated unit (clean control group).
    if (cohort_idx == 0).sum() == 0:
        cohort[0], cohort_idx[0] = np.inf, 0

    tau_i = _effect_function(Xu, float(p["base_effect"]), p["effect_type"])
    f_levels = _prognostic_levels(Xu, int(p["linearity_degree"]))
    slope_i = p["trend_heterogeneity"] * Xu["X3"]
    eps = _ar1_errors(n, n_periods, float(p["ar1_rho"]),
                      float(p["epsilon_scale"]), rng)
    cohort_mult = list(p["cohort_multipliers"])
    if len(cohort_mult) != K:
        raise ValueError("cohort_multipliers must have one entry per cohort.")
    ramp = float(p["dynamic_ramp"])

    beta_0, beta_time = -0.5, 0.2
    rows = []
    for t in range(n_periods):
        gamma_t = beta_time * (t ** 2 if int(p["linearity_degree"]) >= 3 else t)
        D = ((cohort != np.inf) & (t >= cohort)).astype(int)
        k = np.where(cohort == np.inf, np.nan, t - cohort)
        # cohort- and event-time-varying multiplier on the unit effect
        mult = np.zeros(n)
        for j in range(K):
            in_coh = (cohort_idx == (j + 1))
            mult[in_coh] = cohort_mult[j]
        ramp_factor = np.where((k >= 0) & np.isfinite(k), 1.0 + ramp * k, 0.0)
        catt = mult * ramp_factor * tau_i * D
        y0 = beta_0 + alpha + gamma_t + f_levels + slope_i * t + eps[:, t]
        y = y0 + catt
        rows.append(pd.DataFrame({
            "unit_id": np.arange(n),
            "time": t,
            "cohort": cohort,
            "D": D,
            "eventually_treated": (cohort_idx > 0).astype(int),
            "event_time": k,
            **{c: Xu[c] for c in Xu},
            "alpha": alpha,                 # UNOBSERVED -- diagnostics only
            "tau_true": tau_i,
            "CATT": catt,
            "Y": y,
        }))

    df = pd.concat(rows, ignore_index=True)
    df["post"] = (df["time"] >= earliest).astype(int)
    df["treatment_group"] = cohort_idx[df["unit_id"].values]
    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
    df.attrs["dgp"] = "staggered"
    df.attrs["params"] = p
    df.attrs["adoption_times"] = adoption_times
    return df


# --------------------------------------------------------------------------- #
# True estimands
# --------------------------------------------------------------------------- #
def true_estimands(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame of the *true* values of every reported estimand.

    Estimand types
    --------------
    * ``GATT``   one row per treated (cohort g, calendar time t) cell, t >= g.
    * ``ES``     event-study: one row per event time k >= 0 (averaged over
                 cohorts, treated units only).
    * ``ATT``    a single overall average over all treated post observations.

    The true value of an averaged estimand is the mean of the realised
    individual ``CATT`` over the relevant treated post observations, which is
    the target the (G)ATT estimators are consistent for.
    """
    treated_post = df[(df["D"] == 1)].copy()
    records = []

    # GATT(g, t)
    for (g, t), grp in treated_post.groupby(["cohort", "time"]):
        records.append({
            "estimand_type": "GATT",
            "estimand_id": f"g={g:g}_t={int(t)}",
            "g": float(g), "t": int(t),
            "k": int(t - g),
            "true": float(grp["CATT"].mean()),
            "n_treated": int(grp["unit_id"].nunique()),
        })

    # Event-study ATT(k)
    for k, grp in treated_post.groupby("event_time"):
        records.append({
            "estimand_type": "ES",
            "estimand_id": f"k={int(k)}",
            "g": np.nan, "t": np.nan, "k": int(k),
            "true": float(grp["CATT"].mean()),
            "n_treated": int(grp["unit_id"].nunique()),
        })

    # Overall ATT
    records.append({
        "estimand_type": "ATT",
        "estimand_id": "ATT",
        "g": np.nan, "t": np.nan, "k": np.nan,
        "true": float(treated_post["CATT"].mean()),
        "n_treated": int(treated_post["unit_id"].nunique()),
    })

    return pd.DataFrame.from_records(records)
