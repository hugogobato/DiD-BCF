"""Experiment grid for the revision simulations (Workstreams B1, B2, D).

An :class:`Experiment` is one ``(dgp, setting, list of N values)`` cell of the
design.  :func:`all_experiments` returns the full suite; the runner scripts loop
over it.  ``M`` (replications) defaults to 100 but the revision plan recommends
**200-500 for the coverage/size settings**, which need more replications to be
precise -- override with ``--reps`` on the command line.

Design summary
--------------
* **B1** (canonical DiD): vary the unobserved-confounder strength, the error
  structure (iid vs AR(1)), the selection mechanism, and a sharp-null setting
  for size/coverage -- all at a single moderate ``N``.
* **B2** (decomposed metrics + sample-size sweep): the baseline canonical DGP
  across ``N in {200, 400, 800, 1600}`` (anchored at the base size 200) to
  exhibit bias -> 0, variance -> 0 and the sqrt(N) behaviour, separately for
  plain and corrected DiD-BCF.

Every scenario is run at each ``linearity_degree in {1, 2, 3}`` -- one notebook
per model (DiD-BCF, OLS/TWFE), per scenario, per linearity degree, mirroring the
original ``Simulation_Studies/`` layout.
* **D** (staggered, cohort x event-time effects): the headline staggered DGP and
  a "contamination" variant with stronger dynamics (larger weight on
  already-treated comparisons) for the Goodman-Bacon analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dgps import DEFAULT_CANONICAL_PARAMS, DEFAULT_STAGGERED_PARAMS

DEFAULT_REPS = 100
# The base panel size matches the original study (200 units); the B2 sample-size
# sweep is *anchored* at 200 and grows from there to exhibit the asymptotics.
BASE_N = 200
N_SWEEP = (200, 400, 800, 1600)
LINEARITY_DEGREES = (1, 2, 3)


@dataclass
class Experiment:
    name: str                 # unique label, used in output filenames
    workstream: str           # "B1" | "B2" | "D"
    dgp: str                  # "canonical" | "staggered"
    dgp_params: dict          # overrides passed to the generator
    n_values: tuple = (BASE_N,)
    reps: int = DEFAULT_REPS
    note: str = ""


def _canon(**ov) -> dict:
    return {**DEFAULT_CANONICAL_PARAMS, **ov}


def _stag(**ov) -> dict:
    return {**DEFAULT_STAGGERED_PARAMS, **ov}


def all_experiments(reps: int = DEFAULT_REPS) -> list:
    exps: list = []

    # ---- B1: canonical-DiD settings (single N) --------------------------- #
    exps += [
        Experiment("B1_baseline", "B1", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=1.0, ar1_rho=0.0),
                   n_values=(BASE_N,), reps=reps,
                   note="moderate unit FE, selection on unobservables, iid errors"),
        Experiment("B1_strong_confounder", "B1", "canonical",
                   _canon(alpha_sd=2.0, conf_strength=1.5, ar1_rho=0.0),
                   n_values=(BASE_N,), reps=reps,
                   note="large Var(alpha) and strong corr(alpha, treatment)"),
        Experiment("B1_serial_corr", "B1", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=1.0, ar1_rho=0.6),
                   n_values=(BASE_N,), reps=reps,
                   note="AR(1) within-unit serially correlated errors"),
        Experiment("B1_selection_obs", "B1", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=0.0, selection="observable"),
                   n_values=(BASE_N,), reps=reps,
                   note="continuity check: selection on observables only"),
        Experiment("B1_null", "B1", "canonical",
                   _canon(base_effect=0.0, effect_type="homogeneous",
                          alpha_sd=1.0, conf_strength=1.0),
                   n_values=(BASE_N,), reps=max(reps, 200),
                   note="sharp null tau=0: size and coverage under H0"),
    ]

    # ---- B2: sample-size sweep on the baseline canonical DGP -------------- #
    exps += [
        Experiment("B2_sweep", "B2", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=1.0, ar1_rho=0.0),
                   n_values=N_SWEEP, reps=reps,
                   note="bias->0, var->0, sqrt(N) behaviour"),
        Experiment("B2_sweep_serial", "B2", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=1.0, ar1_rho=0.6),
                   n_values=N_SWEEP, reps=reps,
                   note="sample-size sweep under serial correlation"),
    ]

    # ---- D: staggered, cohort x event-time effects ----------------------- #
    exps += [
        Experiment("D_staggered", "D", "staggered",
                   _stag(dynamic_ramp=0.4,
                         cohort_multipliers=(1.0, 1.5, 2.0)),
                   n_values=(BASE_N,), reps=reps,
                   note="dynamic, cohort-varying effects (Goodman-Bacon case)"),
        Experiment("D_contamination", "D", "staggered",
                   _stag(dynamic_ramp=0.8,
                         cohort_multipliers=(1.0, 2.0, 3.0)),
                   n_values=(BASE_N,), reps=reps,
                   note="stronger dynamics -> larger TWFE contamination"),
    ]
    return exps


def get_experiment(name: str, reps: int = DEFAULT_REPS) -> Experiment:
    for e in all_experiments(reps=reps):
        if e.name == name:
            return e
    raise KeyError(f"No experiment named {name!r}. Available: "
                   f"{[e.name for e in all_experiments()]}")


def scenario_names() -> list:
    """Ordered list of scenario (experiment) names -- one 'DGP' per scenario."""
    return [e.name for e in all_experiments()]
