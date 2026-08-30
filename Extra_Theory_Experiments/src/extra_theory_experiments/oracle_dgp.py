"""A self-contained canonical DiD design with exact nuisance oracles.

The production canonical DGP deliberately adds logistic-normal assignment noise,
so its conditional treatment probability is not available in closed form.  This
small local design keeps the same long-panel conventions and estimand definitions
but samples treatment directly from a known Bernoulli probability pi(X).  Its
control potential outcome has a time-invariant level and iid mean-zero errors,
so the exact conditional control long difference is available as m0(X, g, t).
The observed-X law is jointly symmetric and bounded, making the population
cell share exactly one half and keeping the known propensity away from the
default clipping boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_ORACLE_CANONICAL_PARAMS = {
    "n_units": 200,
    "num_pre_periods": 4,
    "num_post_periods": 4,
    "linearity_degree": 1,
    "base_effect": 3.0,
    "effect_type": "homogeneous",
    "alpha_sd": 0.0,
    "epsilon_scale": 1.0,
    "non_theorem_stress": False,
}


def _sigmoid(value):
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-value))


def oracle_assignment_probability(X1, X2, X3, X4, X5, degree: int):
    """Exact P(D=1 | observed X) used by the oracle-ready design.

    Here ``z1 = 2 X1 - 1`` and every input is bounded.  Both indices are odd
    under ``T(X) = (1-X1, -X2, -X3, -X4, -X5)``.  Their absolute values are at
    most 2.2 and 2.15, respectively, so probabilities are strictly inside the
    default propensity clipping interval by a wide margin.
    """
    z1 = 2.0 * np.asarray(X1, dtype=float) - 1.0
    if int(degree) == 1:
        index = 0.80 * z1 + 0.50 * X2 + 0.40 * X3 - 0.30 * X4 + 0.20 * X5
    elif int(degree) == 2:
        index = (0.70 * z1 + 0.60 * X2 + 0.40 * z1 * (X2 ** 2)
                 + 0.25 * (X3 ** 3) + 0.20 * X4 * (X5 ** 2))
    else:
        raise ValueError("oracle_canonical supports degrees 1 and 2 only")
    return _sigmoid(index)


def oracle_control_slope(X1, X2, X3, X4, X5, degree: int):
    """Exact conditional mean slope of Y(0) under the oracle design."""
    if int(degree) == 1:
        return 0.20 + 0.25 * X3 + 0.10 * X4
    if int(degree) == 2:
        return 0.20 + 0.25 * X3 + 0.10 * (X4 ** 2 - 1.0)
    raise ValueError("oracle_canonical supports degrees 1 and 2 only")


def oracle_control_long_difference(slope, g: float, t: int):
    """Exact E[Y_t(0)-Y_(g-1)(0) | observed X]."""
    return (0.10 + np.asarray(slope, dtype=float)) * (int(t) - (int(g) - 1))


def generate_oracle_canonical_did(seed: int = 0, **overrides) -> pd.DataFrame:
    """Generate an oracle-ready single-adoption canonical DiD panel.

    Assignment is Bernoulli with the known probability ``pi(X)`` above.  The
    outcome is

        Y_it(0) = -0.5 + alpha_i + 0.1 t + f(X_i) + s(X_i)t + eps_it,

    where the default has no unit-level alpha term and iid Gaussian eps. Thus
    the row errors are iid conditional on observed X, the stored ``m0_oracle``
    at time t is exactly ``(0.1+s(X_i)) * (t-(g-1))`` for the sole adoption
    cohort g, and homogeneous default ``tau=3`` makes every GATT/ES/ATT truth
    exactly 3 in every replication. Setting ``alpha_sd`` nonzero is an explicit
    non-theorem stress option, but heterogeneous effects are rejected
    unconditionally because this adapter does not implement their exact
    population GATT. ``barpi_oracle`` is the population share 1/2 implied by
    joint sign-flip symmetry, not the realized treatment share or a sample mean
    of pi.
    """
    params = {**DEFAULT_ORACLE_CANONICAL_PARAMS, **overrides}
    degree = int(params["linearity_degree"])
    n = int(params["n_units"])
    n_pre = int(params["num_pre_periods"])
    n_post = int(params["num_post_periods"])
    if n < 8 or n_pre < 1 or n_post < 1:
        raise ValueError("oracle_canonical needs at least 8 units and one period on each side")
    if degree not in (1, 2):
        raise ValueError("oracle_canonical supports degrees 1 and 2 only")
    stress = bool(params.get("non_theorem_stress", False))
    if str(params["effect_type"]) != "homogeneous":
        raise ValueError(
            "heterogeneous oracle_canonical effects are unsupported: exact "
            "population GATT is not implemented")
    if float(params["alpha_sd"]) != 0.0 and not stress:
        raise ValueError(
            "nonzero alpha_sd is non-theorem stress; set "
            "non_theorem_stress=True explicitly")
    rng = np.random.default_rng(int(seed))
    X1 = rng.binomial(1, 0.5, size=n).astype(float)
    X2 = rng.uniform(-1.0, 1.0, size=n)
    X3 = rng.uniform(-1.0, 1.0, size=n)
    X4 = rng.uniform(-1.0, 1.0, size=n)
    X5 = rng.uniform(-1.0, 1.0, size=n)
    pi = oracle_assignment_probability(X1, X2, X3, X4, X5, degree)
    eventually_treated = rng.binomial(1, pi).astype(int)
    if eventually_treated.sum() == 0 or eventually_treated.sum() == n:
        raise RuntimeError("oracle_canonical draw has a degenerate treatment arm")
    adoption = n_pre
    cohort = np.where(eventually_treated == 1, float(adoption), np.inf)
    slope = oracle_control_slope(X1, X2, X3, X4, X5, degree)
    levels = -0.75 * X1 + 0.50 * X2 - 0.50 * X3 - 0.30 * X4 + 0.20 * X5
    alpha = rng.normal(0.0, float(params["alpha_sd"]), size=n)
    eps = rng.normal(0.0, float(params["epsilon_scale"]),
                      size=(n, n_pre + n_post))
    tau = np.full(n, float(params["base_effect"]))
    # The joint sign-flip law is symmetric.  Since pi(TX)=1-pi(X), the
    # population cell share is exactly E[pi(X)] = 1/2, independent of this
    # replication's realized sample mean.
    barpi = 0.5
    population_gatt = float(params["base_effect"])
    rows = []
    for t in range(n_pre + n_post):
        D = ((cohort != np.inf) & (t >= cohort)).astype(int)
        y0 = -0.5 + alpha + 0.1 * t + levels + slope * t + eps[:, t]
        catt = tau * D
        m0 = oracle_control_long_difference(slope, adoption, t)
        rows.append(pd.DataFrame({
            "unit_id": np.arange(n),
            "time": t,
            "cohort": cohort,
            "D": D,
            "eventually_treated": eventually_treated,
            "event_time": np.where(cohort == np.inf, np.nan, t - cohort),
            "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5,
            "alpha": alpha,
            "pt_slope": np.zeros(n),
            "tau_true": tau,
            "CATT": catt,
            "gatt_population_oracle": np.full(n, population_gatt),
            "Y": y0 + catt,
            "m0_oracle": m0,
            "pi_oracle": pi,
            "barpi_oracle": np.full(n, barpi),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["post"] = (df["time"] >= adoption).astype(int)
    df["treatment_group"] = df["eventually_treated"].astype(int)
    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
    df.attrs["dgp"] = "oracle_canonical"
    df.attrs["params"] = params
    df.attrs["adoption"] = adoption
    df.attrs["truth"] = {
        "gatt_population_oracle": population_gatt,
        "truth_source": "homogeneous_population_effect",
        "homogeneous": str(params["effect_type"]) == "homogeneous",
    }
    df.attrs["oracle_formula"] = {
        "pi": "sigmoid(.80 z1 + .50 X2 + .40 X3 - .30 X4 + .20 X5) for degree 1; "
              "degree 2 uses sigmoid(.70 z1 + .60 X2 + .40 z1 X2^2 + .25 X3^3 "
              "+ .20 X4 X5^2), with z1=2X1-1",
        "m0": "(.10 + s(X)) * (t - (g - 1))",
        "barpi": "1/2 by joint sign-flip symmetry and expit(-u)=1-expit(u)",
    }
    return df


__all__ = [
    "DEFAULT_ORACLE_CANONICAL_PARAMS",
    "generate_oracle_canonical_did",
    "oracle_assignment_probability",
    "oracle_control_slope",
    "oracle_control_long_difference",
]
