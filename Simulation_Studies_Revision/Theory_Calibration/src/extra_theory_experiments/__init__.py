"""Disposable, simulation-only calibration tools for the DiD-BCF theory.

The package deliberately lives outside the production simulation engine.  It
uses the engine's sampler through small adapters, and labels the same-sample
correction ``current_hybrid``.  The fixed-fold construction is named
``reference_fold_convolution`` because it is a theorem-oriented reference
implementation, not a claim that finite Monte-Carlo draws validate a theorem.
"""

from .manifest import ExperimentTask, build_manifest, deterministic_seed
from .correction import (
    algorithm2_draws,
    current_hybrid_correction,
    fit_propensity,
    summarise_draws,
)
from .reference import convolve_fold_draws, reference_fold_convolution
from .metrics import summarise_replications
from .oracle_dgp import generate_oracle_canonical_did

__all__ = [
    "ExperimentTask", "build_manifest", "deterministic_seed",
    "algorithm2_draws", "current_hybrid_correction", "fit_propensity",
    "summarise_draws", "convolve_fold_draws", "reference_fold_convolution",
    "summarise_replications", "generate_oracle_canonical_did",
]
