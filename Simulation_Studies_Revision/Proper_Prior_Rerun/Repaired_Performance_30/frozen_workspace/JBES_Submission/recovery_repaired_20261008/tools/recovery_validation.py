"""Fail-closed array and numeric-design checks shared by runner and collector."""
import numpy as np

def numeric_design(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("Native design must be a finite numeric matrix with eligible columns")
    with np.errstate(over="ignore", invalid="ignore"):
        span = X.max(axis=0) - X.min(axis=0)
    if not np.isfinite(span).all():
        raise ValueError("Native design has a nonfinite numeric range")

def validate_arrays(arrays, unit):
    bdg = unit["budget"]
    atomic = unit["chain_index"] is None
    draws = bdg["num_mcmc"] * (bdg["num_chains"] if atomic else 1)
    required = {"tau", "global_var_samples", "leaf_scale_mu_samples", "rfx_sigma"} if atomic else {"tau", "ATT", "sigma2"}
    if not required.issubset(arrays):
        raise ValueError("Missing sampled arrays: " + repr(sorted(required - set(arrays))))
    expected_rows = int(unit["job"].get("dgp_params", {}).get("n_units", unit["job"].get("N", 0)))
    tau = arrays["tau"]
    if tau.ndim != 2 or tau.shape[1] != draws:
        raise ValueError("Effect surface draw count/shape mismatch")
    if expected_rows and not unit["job"]["kind"].startswith("application"):
        params = unit["job"].get("dgp_params", {})
        periods = int(params.get("n_periods", params.get("num_pre_periods", 0) + params.get("num_post_periods", 0)))
        if periods and tau.shape[0] != expected_rows * periods:
            raise ValueError("Effect surface panel row count mismatch")
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite sampled array: " + key)
        if key != "tau" and value.size != draws:
            raise ValueError("Scalar draw count mismatch: " + key)
        if key in {"sigma2", "global_var_samples", "leaf_scale_mu_samples", "leaf_scale_tau_samples", "leaf_scale_level", "leaf_scale_trend", "rfx_sigma"} and (value <= 0).any():
            raise ValueError("Nonpositive variance draws: " + key)
