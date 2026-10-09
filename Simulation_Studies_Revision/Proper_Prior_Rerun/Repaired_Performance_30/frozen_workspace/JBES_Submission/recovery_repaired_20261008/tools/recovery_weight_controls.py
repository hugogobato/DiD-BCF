"""New native checks for fixed equal eligible weights and zero excluded weights.

The frozen repair certificate is preserved. This file adds a narrower support
extension for BCF keep_vars; it makes no statement about forest mixing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

EPSILON = 1e-15
BURN = 100
DRAWS = 20_000
ABS_TOL = 0.015
RESIDUAL_TOL = 1e-10


def numeric_support_guard(covariates, weights, feature_types=None,
                          num_features_subsample=None,
                          initialized_split_indices=()):
    """Reject inputs outside the new fixed-support numeric scope.

    Call before native sampling, using each forest's actual processed matrix
    and actual weights. This does not inspect or certify positive leaf/noise
    variances; those must be checked by the caller separately.
    """
    import numpy as np
    x = np.asarray(covariates, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("A nonempty numeric covariate matrix is required")
    if w.shape != (x.shape[1],) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Finite nonnegative feature weights of matching size required")
    active = np.flatnonzero(w > 0)
    if not active.size or not (w[active] > EPSILON).all():
        raise ValueError("Eligible feature weights must exceed native GFR epsilon")
    if not np.array_equal(w[active], np.repeat(w[active[0]], active.size)):
        raise ValueError("Eligible feature weights must be exactly equal and fixed")
    if feature_types is not None:
        types = np.asarray(feature_types)
        if types.shape != w.shape or not (types == 0).all():
            raise ValueError("Only numeric feature types are in the support extension")
    if num_features_subsample is not None and num_features_subsample != x.shape[1]:
        raise ValueError("Feature subsampling is outside the support extension")
    if not np.isfinite(x).all():
        raise ValueError("All processed covariates must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        spans = x.max(axis=0) - x.min(axis=0)
    if not np.isfinite(spans).all():
        raise ValueError("Numeric covariate spans must be finite and representable")
    if any(int(j) not in set(active.tolist()) for j in initialized_split_indices):
        raise ValueError("Initial trees may only split on eligible features")
    return {"scope": "fixed_equal_eligible_weights_exact_zero_exclusions_v1",
            "eligible_indices": active.tolist(),
            "excluded_indices": np.flatnonzero(w == 0).tolist(),
            "eligible_weight": float(w[active[0]]),
            "num_features": int(x.shape[1]), "epsilon": EPSILON}


def _loaded_identity(expected_site, expected_binary_sha256=None):
    import stochtree
    import stochtree_cpp
    site = Path(expected_site).resolve()
    package = Path(stochtree.__file__).resolve()
    binary = Path(stochtree_cpp.__file__).resolve()
    if not package.is_relative_to(site) or not binary.is_relative_to(site):
        raise ValueError("Controls did not load the requested isolated engine")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if expected_binary_sha256 is not None and digest != expected_binary_sha256:
        raise ValueError("Loaded isolated native binary has the wrong digest")
    return {"package_path": str(package), "binary_path": str(binary),
            "binary_sha256": digest}


def _fixture(eligible, padded, mode="mcmc", tied=False):
    import numpy as np
    from stochtree import (Dataset, Forest, ForestContainer, ForestModelConfig,
                          ForestSampler, GlobalModelConfig, Residual, RNG)
    alpha, beta, leaf_var, noise_var = 0.25, 3.0, 0.005, 0.5
    x_active = np.repeat([1.0, 2.0], 10)
    if tied:
        x_active = np.ones(20)
    y = np.repeat([-1.0, 1.0], 10)
    if padded:
        indices = [1] if eligible == 1 else [1, 3]
        p = 3 if eligible == 1 else 5
        # Strongly varying excluded columns exercise support decisions, not
        # merely harmless constant padding. Zeros occur before/between/after S.
        x = np.column_stack([np.arange(20.0) * (-1 if j % 2 else 1)
                             for j in range(p)])
        w = np.zeros(p)
        for j in indices:
            x[:, j] = x_active
            w[j] = 1.0 / eligible
    else:
        indices = list(range(eligible))
        x = np.column_stack([x_active] * eligible)
        w = np.repeat(1.0 / eligible, eligible)
        p = eligible
    guard = numeric_support_guard(x, w, np.zeros(p), p)
    dataset = Dataset()
    dataset.add_covariates(x)
    dataset.add_basis(np.ones(y.size))
    global_config = GlobalModelConfig(global_error_variance=noise_var)
    config = ForestModelConfig(
        num_trees=1, num_features=p, num_observations=y.size,
        feature_types=np.zeros(p, dtype=int), variable_weights=w,
        leaf_dimension=1, alpha=alpha, beta=beta, min_samples_leaf=5,
        max_depth=5, leaf_model_type=1,
        leaf_model_scale=np.array([[leaf_var]]), cutpoint_grid_size=100,
        num_features_subsample=p)
    sampler = ForestSampler(dataset, global_config, config)
    forest = Forest(1, 1, False, False)
    container = ForestContainer(1, 1, False, False)
    residual = Residual(y.copy())
    sampler.prepare_for_sampler(dataset, residual, forest, 1, np.array([0.0]))
    rng = RNG(202610080 + eligible * 100 + int(padded) * 10 + int(tied))
    retained = 20 if tied or mode == "gfr" else DRAWS
    burn = BURN if mode == "mcmc" and not tied else 0
    visits = {j: 0 for j in indices}
    excluded_splits = 0
    max_residual_error = 0.0
    for k in range(burn + retained):
        sampler.sample_one_iteration(container, forest, dataset, residual,
                                     rng, global_config, config, False,
                                     mode == "gfr")
        if forest.num_nodes(0) > 1:
            chosen = int(forest.node_split_index(0, 0))
            if chosen not in visits:
                excluded_splits += 1
            elif k >= burn:
                visits[chosen] += 1
        predicted = np.asarray(forest.predict(dataset)).reshape(-1)
        actual = np.asarray(residual.get_residual()).reshape(-1)
        max_residual_error = max(max_residual_error,
                                float(np.max(np.abs(actual - (y - predicted)))))
    split_fraction = sum(visits.values()) / retained
    log_bf = (0.5 * math.log1p(20 * leaf_var / noise_var)
              - math.log1p(10 * leaf_var / noise_var)
              + leaf_var * 100 / (noise_var * (noise_var + 10 * leaf_var)))
    odds = alpha * (1 - alpha / 2 ** beta) ** 2 / (1 - alpha) * math.exp(log_bf)
    target = 0.0 if tied else odds / (1 + odds)
    passed = excluded_splits == 0 and max_residual_error <= RESIDUAL_TOL
    if tied:
        passed = passed and sum(visits.values()) == 0
    elif mode == "mcmc":
        passed = (passed and abs(split_fraction - target) <= ABS_TOL
                  and all(abs(count / retained - target / eligible) <= ABS_TOL
                          for count in visits.values()))
    return {"eligible_features": eligible, "padded": padded, "mode": mode,
            "tied_active_columns": tied, "scope_guard": guard,
            "burnin": burn, "retained_iterations": retained,
            "split_fraction": split_fraction, "target_split_probability": target,
            "split_feature_visits": visits, "excluded_feature_splits": excluded_splits,
            "max_residual_error": max_residual_error,
            "absolute_tolerance": ABS_TOL, "pass": bool(passed)}


def run_native_weight_controls(expected_site, expected_binary_sha256=None):
    """Run bounded controls after isolated activation; return PASS/FAIL record."""
    identity = _loaded_identity(expected_site, expected_binary_sha256)
    cases = [_fixture(m, padded) for m in (1, 2) for padded in (False, True)]
    cases += [_fixture(2, True, mode=mode, tied=True)
              for mode in ("mcmc", "gfr")]
    comparisons = []
    for m in (1, 2):
        compact, padded = [c for c in cases if c["eligible_features"] == m
                           and not c["tied_active_columns"]]
        difference = abs(compact["split_fraction"] - padded["split_fraction"])
        comparisons.append({"eligible_features": m,
                            "absolute_split_frequency_difference": difference,
                            "tolerance": 2 * ABS_TOL,
                            "pass": difference <= 2 * ABS_TOL})
    passed = all(c["pass"] for c in cases) and all(c["pass"] for c in comparisons)
    return {"schema": "didbcf-fixed-support-native-controls-v1",
            "status": "PASS" if passed else "FAIL", "loaded_engine": identity,
            "scope": "fixed_equal_positive_eligible_exact_zero_exclusions",
            "cases": cases, "compact_padded_comparisons": comparisons,
            "limitations": "Bounded support and stationary two-state controls only; "
                           "no forest mixing, exact floating-point sampling, "
                           "contraction, causal consistency, or coverage certificate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/didbcf-weight-controls-mpl")
    sys.path.insert(0, str(args.site.resolve()))
    result = run_native_weight_controls(args.site, args.expected_binary_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"],
                      "binary_sha256": result["loaded_engine"]["binary_sha256"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
