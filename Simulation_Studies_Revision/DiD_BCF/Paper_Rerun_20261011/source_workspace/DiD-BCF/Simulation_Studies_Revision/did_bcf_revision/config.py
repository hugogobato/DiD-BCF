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
  across ``N in {50, 100, 200, 400, 800}`` (bracketing the base size 200 on both
  sides) to exhibit bias -> 0, variance -> 0 and the sqrt(N) behaviour, separately
  for plain and corrected DiD-BCF.

Every scenario is run at each ``linearity_degree in {1, 2, 3}`` -- one notebook
per model (DiD-BCF, OLS/TWFE), per scenario, per linearity degree, mirroring the
original ``Simulation_Studies/`` layout.  The degree changes the covariate
parameterisation handed to the estimators (and, at ``d = 3``, the shape of
``tau``); see the module docstring of :mod:`did_bcf_revision.dgps`.  Note that
under ``selection="unobservable"`` the covariates are balanced across arms, so
the degree loads only through nuisance-fitting variance; it moves *bias* only
in ``B1_selection_obs`` and ``B1_selection_both``.
* **D** (staggered, cohort x event-time effects): the headline staggered DGP and
  a "contamination" variant with stronger dynamics (larger weight on
  already-treated comparisons) for the Goodman-Bacon analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dgps import DEFAULT_CANONICAL_PARAMS, DEFAULT_STAGGERED_PARAMS

DEFAULT_REPS = 100
# The base panel size matches the original study (200 units); every non-sweep
# scenario runs at exactly this N.  The B2 sample-size sweep brackets it on both
# sides -- 50 and 100 below, 400 and 800 above -- so the sqrt(N) plots show the
# small-sample regime as well as the asymptotics.  N_SWEEP[0] is therefore the
# *smallest* sweep size, NOT the base panel: read BASE_N when you mean the base.
BASE_N = 200
N_SWEEP = (50, 100, 200, 400, 800)
LINEARITY_DEGREES = (1, 2, 3)

# Workstream PT drops degree 2, and *only* workstream PT.  Its comparator is a
# TWFE event study, which adjusts for no covariates at all, so the transforms of
# X never enter its design and it is EXACTLY invariant to the degree (measured
# difference 4e-14 across d = 1, 2, 3); the DiD-BCF diagnostic's rejection rates
# moved by less than one Monte-Carlo standard error between d = 2 and d = 3.
# The degree therefore buys nothing there and costs a third of the compute.
#
# This does NOT generalise to B1/B2/D, whose estimators do adjust for the
# covariates: degree 2 is precisely the cell where linear adjustment leaves a
# residual violation of conditional PT, and it moves *bias* in B1_selection_obs
# and B1_selection_both (see the module docstring).  Those workstreams keep all
# three degrees.
PT_LINEARITY_DEGREES = (1, 3)


def degrees_for(workstream: str) -> tuple:
    """Linearity degrees to run for a workstream."""
    return PT_LINEARITY_DEGREES if workstream == "PT" else LINEARITY_DEGREES

# The Goodman-Bacon ramp (Workstream D, Reviewer 3.1.2).  ``dynamic_ramp`` moves
# how *wrong* the already-treated comparisons are but leaves their **weight**
# untouched -- that weight is a function of the adoption design alone.  Shrinking
# the never-treated pool is the knob that actually moves it: measured mean weight
# on "Later_vs_Earlier" comparisons runs 0.076 -> 0.249 across this grid while the
# effect DGP is held fixed, so estimator error can be plotted against it.
NEVER_TREATED_SHARES = (0.40, 0.30, 0.25, 0.20, 0.10, 0.05)

# The pre-trend diagnostic (Workstream F4, Reviewer 3.2.3): violation magnitude
# axes.  ``group_trend`` is a differential slope in outcome units per period;
# ``het_trend`` is a differential slope that is heterogeneous in X1 and cancels
# in the aggregate, which no marginal event study can see.
GROUP_TREND_GRID = (0.05, 0.10, 0.20, 0.40)
HET_TREND_GRID = (0.10, 0.20, 0.40)
# ``alpha_trend`` routes the violation through the *unobserved* confounder.  It
# is kept as a single mechanism check rather than a magnitude grid: at
# ``conf_strength = 1`` the treated-minus-control gap in ``a_i`` is close to 1,
# so ``lambda`` and ``delta`` are numerically the same knob.  Measured on the
# first 50 replications, ``PT_violation_a{10,20,40}`` and the matching
# ``PT_violation_g*`` produced pre-trend slopes of 0.110/0.108, 0.202/0.199 and
# 0.397/0.391, per-replication correlations of 0.97-0.99, and ATT biases of
# 0.269/0.267 -- one curve, run twice.  The remaining setting exists to show
# that the unobserved-confounder channel behaves like the group-trend channel,
# not to trace a second power curve.
ALPHA_TREND_CHECK = 0.20


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
        Experiment("B1_selection_both", "B1", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=1.0, selection="both"),
                   n_values=(BASE_N,), reps=reps,
                   note="unobserved confounder AND covariate-driven assignment: "
                        "the only setting where conditional parallel trends is "
                        "violated, so the linearity degree loads through bias"),
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

    # ---- D-ramp: sweep the *weight* on already-treated comparisons -------- #
    for share in NEVER_TREATED_SHARES:
        per_cohort = (1.0 - share) / 3.0
        exps.append(Experiment(
            f"D_ramp_nt{int(round(share * 100)):02d}", "D", "staggered",
            _stag(dynamic_ramp=0.4, cohort_multipliers=(1.0, 1.5, 2.0),
                  cohort_shares=(per_cohort,) * 3),
            # 200 to match the TWFE and R-benchmark runs on these designs: the
            # ramp compares estimators point by point, so a rep-count mismatch
            # across methods would confound the comparison with Monte-Carlo noise.
            n_values=(BASE_N,), reps=max(reps, 200),
            note=f"never-treated share {share:.2f}: Goodman-Bacon weight on "
                 f"already-treated comparisons rises as the clean control pool "
                 f"shrinks (effect DGP held fixed)"))

    # ---- PT: the pre-trend / conditional-PTA diagnostic ------------------- #
    exps += [
        Experiment("PT_hold", "PT", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=1.0),
                   n_values=(BASE_N,), reps=max(reps, 200),
                   note="conditional PT holds: false-positive rate of the "
                        "diagnostic (this is B1_baseline's DGP)"),
        Experiment("PT_conditional", "PT", "canonical",
                   _canon(alpha_sd=1.0, conf_strength=0.0,
                          selection="observable", trend_heterogeneity=0.6),
                   n_values=(BASE_N,), reps=max(reps, 200),
                   note="conditional PT holds but UNconditional PT fails: the "
                        "covariate-driven trend differs across arms because "
                        "assignment depends on X. A marginal event study should "
                        "flag it; the covariate-conditional diagnostic should not"),
    ]
    for d in GROUP_TREND_GRID:
        exps.append(Experiment(
            f"PT_violation_g{int(round(d * 100)):02d}", "PT", "canonical",
            _canon(alpha_sd=1.0, conf_strength=1.0, group_trend=d),
            n_values=(BASE_N,), reps=max(reps, 200),
            note=f"treated group on a differential path, slope {d:g} per period: "
                 f"detection power against violation magnitude"))
    for kap in HET_TREND_GRID:
        exps.append(Experiment(
            f"PT_violation_het{int(round(kap * 100)):02d}", "PT", "canonical",
            _canon(alpha_sd=1.0, conf_strength=1.0, het_trend=kap),
            n_values=(BASE_N,), reps=max(reps, 200),
            note=f"heterogeneous violation, slope +/-{kap:g} per period by X1, "
                 f"zero on average: the marginal event study is powerless by "
                 f"construction and only the subgroup contrast (PRE_SUBC) can "
                 f"detect it"))
    lam = ALPHA_TREND_CHECK
    exps.append(Experiment(
        f"PT_violation_a{int(round(lam * 100)):02d}", "PT", "canonical",
        _canon(alpha_sd=1.0, conf_strength=1.0, alpha_trend=lam),
        n_values=(BASE_N,), reps=max(reps, 200),
        note=f"the unobserved confounder also drives the slope (lambda={lam:g}): "
             f"violation of conditional PT that no adjustment on observables can "
             f"remove.  Single mechanism check -- see ALPHA_TREND_CHECK for why "
             f"this is not a magnitude grid"))
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
