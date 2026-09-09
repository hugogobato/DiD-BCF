"""Structured DiD-BCF: the two-way-restricted prognostic function.

--------------------------------------------------------------------------------
What this fixes
--------------------------------------------------------------------------------
DiD-BCF as published fits

    Y_it = mu(D_i, t, X_it) + tau(X_it, k_it) D_it + eps_it                  (1)

with ``mu`` an unrestricted forest over ``(D_i, t, X)``.  Because
``D_it = 1{D_i = 1} 1{t >= g}`` is itself a function of ``(D_i, t)``, the whole
of ``tau(X, k) D_it`` lies in the domain of ``mu``, and for any ``c in [0, 1]``

    (mu + c tau D_it,  (1 - c) tau)                                          (2)

gives the same fitted values, the same residuals and the same likelihood.  The
data say nothing about ``c``; only the asymmetric prior breaks the tie, and the
tie is broken more decisively toward ``mu`` the more data there are, so the bias
*grows* with ``N``.  A single tree in ``mu`` that splits once on the group and
once on time reconstructs the treated post-treatment cells exactly.

Conditional parallel trends is precisely the restriction that closes this.  It
says the untreated outcome has no group-by-time interaction, that is,

    mu(D_i, t, X) = a(D_i, X) + b(t, X),                                     (3)

which is Lemma ``lem:twoway`` of ``DiD_BCF_Theory/DiD_BCF_theory.tex``: the
cohort may shift the *level* of the baseline, in a covariate-dependent way, but
not its *trend*.  Under (3) the flat direction (2) disappears, because
``a`` has no time argument and ``b`` has no group argument, so their sum cannot
represent ``1{D_i = 1} 1{t >= g}`` and ``c`` is pinned at zero.  Equivalently,
the difference-in-differences contrast of the fitted regression function

    f(g,t,x) - f(g,g-1,x) - f(inf,t,x) + f(inf,g-1,x)

annihilates ``a + b`` and returns ``tau(x, t-g)`` (equation ``eq:contrast``);
:meth:`StructuredDiDBCF.did_contrast_draws` computes it at the observed design
points and :meth:`StructuredDiDBCF.check_identification` reports the gap, which
is zero exactly when the covariates are time-invariant.

--------------------------------------------------------------------------------
How it is implemented
--------------------------------------------------------------------------------
Three forests share one residual and are sampled in a single Gibbs sweep:

    a : constant-leaf forest on (cohort label, X)     -- no time column
    b : constant-leaf forest on (calendar time, X)    -- no cohort column
    tau : leaf-regression forest on (modifiers, k) with basis D_it

built on ``stochtree``'s low-level sampler API (``Dataset``, ``Residual``,
``ForestSampler``, ``ForestContainer``, ``GlobalVarianceModel``,
``LeafVarianceModel``, ``RandomEffectsModel``), which is the interface
``stochtree`` documents for custom samplers -- the ``BCFModel`` class is itself
Python-level orchestration over the same primitives, so no C++ change is needed.
The restriction is enforced by *construction* of the design matrices, not by
zeroing split weights: the ``a`` forest is handed a matrix with no time column
at all.

The sampler mirrors ``BCFModel``: grow-from-root warm start, then MCMC with
thinning, outcome standardised, ``sigma^2`` and the prognostic leaf scales
sampled, and an optional global treatment-effect intercept ``tau_0``.  Chains
are run independently from separate warm starts and concatenated.

Optional random effects (``rfx="unit"`` or ``"group"``) add a time-invariant
intercept.  Time-invariant is the point: such a term cannot represent ``D_it``,
so it cannot reopen the flat direction, while ``rfx="unit"`` additionally
absorbs the unobserved unit effect that the canonical DiD design puts in the
residual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .design import DesignMatrices, PanelColumns, build_design
from .priors import (EFFECT_PRIOR, GLOBAL_PRIOR, LEVEL_PRIOR, TREND_PRIOR,
                     ForestPrior, GlobalPrior)

__all__ = ["StructuredDiDBCF"]


# ``stochtree_cpp.RngCpp`` is currently bound to a signed 32-bit integer,
# although the Python-level ``stochtree.sampler.RNG`` annotation only says
# ``int``.  Experiment manifests deliberately use a wider deterministic seed
# space, and chain derivation multiplies those seeds by 1,000.  Normalize only
# the seed crossing into the C++ binding; NumPy continues to receive the full
# chain seed, so this compatibility guard does not reduce the Python RNG's
# seed space.
_STOCHTREE_SEED_MODULUS = 2 ** 31


def _stochtree_seed(seed: int | None) -> int:
    """Map a Python seed to the range accepted by ``stochtree_cpp.RngCpp``.

    ``None`` and the stochtree sentinel ``-1`` retain their documented
    nondeterministic behavior.  Other values, including large deterministic
    seeds and negative values, are wrapped into ``[0, 2**31 - 1]``.
    """
    if seed is None:
        return -1
    value = int(seed)
    if value == -1:
        return -1
    return value % _STOCHTREE_SEED_MODULUS


def _forest_config(prior: ForestPrior, n_obs: int, n_features: int,
                   leaf_model_type: int, leaf_scale: np.ndarray):
    from stochtree import ForestModelConfig
    return ForestModelConfig(
        num_trees=prior.num_trees,
        num_features=n_features,
        num_observations=n_obs,
        feature_types=np.zeros(n_features, dtype=int),      # all numeric
        variable_weights=np.repeat(1.0 / n_features, n_features),
        leaf_dimension=1,
        alpha=prior.alpha,
        beta=prior.beta,
        min_samples_leaf=prior.min_samples_leaf,
        max_depth=prior.max_depth,
        leaf_model_type=leaf_model_type,
        leaf_model_scale=leaf_scale,
        cutpoint_grid_size=prior.cutpoint_grid_size,
    )


class _Block:
    """One forest and the objects needed to sample and store it."""

    def __init__(self, prior: ForestPrior, X: np.ndarray, basis, n_obs: int,
                 global_config, resid_var: float):
        from stochtree import (Dataset, Forest, ForestContainer, ForestSampler,
                               LeafVarianceModel)

        self.prior = prior
        self.leaf_constant = basis is None
        self.leaf_model_type = 0 if self.leaf_constant else 1

        self.dataset = Dataset()
        self.dataset.add_covariates(X)
        if basis is not None:
            self.dataset.add_basis(basis)

        self.leaf_scale = prior.leaf_scale_matrix(resid_var)
        self.config = _forest_config(prior, n_obs, X.shape[1],
                                     self.leaf_model_type, self.leaf_scale)
        self.container = ForestContainer(prior.num_trees, 1, self.leaf_constant, False)
        self.active = Forest(prior.num_trees, 1, self.leaf_constant, False)
        self.sampler = ForestSampler(self.dataset, global_config, self.config)
        self.leaf_var_model = LeafVarianceModel() if prior.sample_leaf_scale else None
        _, self.leaf_shape, self.leaf_rate = prior.calibrate(resid_var)

    def initialise(self, residual, init_value: float) -> None:
        self.sampler.prepare_for_sampler(self.dataset, residual, self.active,
                                         self.leaf_model_type,
                                         np.array([init_value]))

    def step(self, residual, rng, global_config, keep: bool, gfr: bool) -> None:
        self.sampler.sample_one_iteration(
            self.container, self.active, self.dataset, residual, rng,
            global_config, self.config, keep, gfr)

    def step_leaf_scale(self, rng) -> float | None:
        if self.leaf_var_model is None:
            return None
        value = self.leaf_var_model.sample_one_iteration(
            self.active, rng, self.leaf_shape, self.leaf_rate)
        self.leaf_scale[0, 0] = value
        self.config.update_leaf_model_scale(self.leaf_scale)
        return float(value)


class StructuredDiDBCF:
    """Bayesian causal forest for DiD with an additively separable baseline.

    Parameters
    ----------
    level_prior, trend_prior, effect_prior, global_prior
        See :mod:`didbcf_structured.priors`.  Defaults reproduce stochtree's
        ``BCFModel`` calibration with the prognostic tree budget split between
        the two baseline blocks.
    rfx : ``"none"`` | ``"unit"`` | ``"group"``
        Optional additive, time-invariant random intercept.

    Examples
    --------
    >>> model = StructuredDiDBCF()                              # doctest: +SKIP
    >>> model.sample(panel, num_gfr=50, num_mcmc=500, seed=0)   # doctest: +SKIP
    >>> model.tau_draws.shape                                   # doctest: +SKIP
    (n_obs, n_draws)
    """

    def __init__(self,
                 level_prior: ForestPrior = None,
                 trend_prior: ForestPrior = None,
                 effect_prior: ForestPrior = None,
                 global_prior: GlobalPrior = None,
                 rfx: str = "none") -> None:
        self.level_prior = level_prior or LEVEL_PRIOR
        self.trend_prior = trend_prior or TREND_PRIOR
        self.effect_prior = effect_prior or EFFECT_PRIOR
        self.global_prior = global_prior or GLOBAL_PRIOR
        if rfx not in ("none", "unit", "group"):
            raise ValueError("rfx must be one of 'none', 'unit', 'group'")
        self.rfx = rfx
        self.sampled = False

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, df: pd.DataFrame,
               covariates=None, effect_modifiers=None,
               cols: PanelColumns = None, effect_by_cohort: bool = True,
               num_gfr: int = 50, num_burnin: int = 0, num_mcmc: int = 500,
               keep_every: int = 5, num_chains: int = 3,
               seed: int | None = None) -> "StructuredDiDBCF":
        """Fit the structured model to one long panel.

        ``num_mcmc`` draws are retained per chain after thinning by
        ``keep_every``, so the total number of retained draws is
        ``num_mcmc * num_chains``.
        """
        from .design import DEFAULT_COVARIATES, DEFAULT_EFFECT_MODIFIERS

        cols = cols or PanelColumns()
        df = df.sort_values([cols.unit, cols.time]).reset_index(drop=True)
        # ``None`` means use the package defaults; an explicit empty list is
        # useful when reproducing a treatment forest that uses only event time
        # and cohort.
        covariates = (list(DEFAULT_COVARIATES)
                      if covariates is None else list(covariates))
        effect_modifiers = (list(DEFAULT_EFFECT_MODIFIERS)
                            if effect_modifiers is None
                            else list(effect_modifiers))
        dm = build_design(df,
                          covariates=covariates,
                          effect_modifiers=effect_modifiers,
                          cols=cols, effect_by_cohort=effect_by_cohort)

        self.df = df
        self.design = dm
        self.y_bar = float(np.mean(dm.y))
        self.y_std = float(np.std(dm.y))
        resid0 = (dm.y - self.y_bar) / self.y_std

        rfx_ids = None
        if self.rfx == "unit":
            rfx_ids = df[cols.unit].to_numpy(dtype=np.int32)
        elif self.rfx == "group":
            rfx_ids = dm.cohort_code.astype(np.int32)

        chains = []
        for chain in range(num_chains):
            chain_seed = None if seed is None else int(seed) * 1000 + chain
            chains.append(self._sample_chain(dm, resid0, rfx_ids, chain_seed,
                                             num_gfr, num_burnin, num_mcmc,
                                             keep_every))
        self._chains = chains
        self._collect(dm, rfx_ids)
        self.sampled = True
        return self

    def _sample_chain(self, dm: DesignMatrices, resid0: np.ndarray,
                      rfx_ids, seed, num_gfr, num_burnin, num_mcmc, keep_every):
        from stochtree import (GlobalModelConfig, GlobalVarianceModel, RNG,
                               Residual)

        n = dm.n
        resid_var = float(np.var(resid0))
        # The C++ stochtree RNG accepts only a signed 32-bit seed.  Keep the
        # full chain seed for NumPy, but normalize the value at this boundary.
        rng = RNG(_stochtree_seed(seed))
        np_rng = np.random.default_rng(seed)

        sigma2 = resid_var
        global_config = GlobalModelConfig(global_error_variance=sigma2)
        residual = Residual(resid0.copy())

        level = _Block(self.level_prior, dm.level, None, n, global_config, resid_var)
        trend = _Block(self.trend_prior, dm.trend, None, n, global_config, resid_var)
        effect = _Block(self.effect_prior, dm.effect, dm.Z, n, global_config, resid_var)

        # Root initialisation: the level block carries the outcome mean, the
        # other two start at zero so the initial fit is the grand mean.
        level.initialise(residual, float(np.mean(resid0)))
        trend.initialise(residual, 0.0)
        effect.initialise(residual, 0.0)

        rfx = self._init_rfx(rfx_ids, n) if rfx_ids is not None else None

        global_var_model = GlobalVarianceModel()
        tau_0 = 0.0
        tau_0_prior_var = (self.global_prior.intercept_prior_var
                           if self.global_prior.intercept_prior_var is not None
                           else resid_var)
        ztz = float(dm.Z @ dm.Z)

        sigma2_draws, tau0_draws = [], []

        def sweep(gfr: bool, keep: bool):
            nonlocal sigma2, tau_0
            level.step(residual, rng, global_config, keep, gfr)
            trend.step(residual, rng, global_config, keep, gfr)

            if self.global_prior.sample_intercept and ztz > 0:
                # Conjugate normal draw for the global effect level, as in
                # BCFModel: partial out tau_0 from the running residual, draw,
                # then push the change back into the residual.
                partial = np.squeeze(residual.get_residual()) + tau_0 * dm.Z
                v_post = 1.0 / (ztz / sigma2 + 1.0 / tau_0_prior_var)
                m_post = v_post * float(dm.Z @ partial) / sigma2
                tau_0_new = float(np_rng.normal(m_post, np.sqrt(v_post)))
                residual.add_vector(-(tau_0_new - tau_0) * dm.Z)
                tau_0 = tau_0_new

            effect.step(residual, rng, global_config, keep, gfr)

            if rfx is not None:
                rfx["model"].sample(rfx["dataset"], residual, rfx["tracker"],
                                    rfx["container"], keep, sigma2, rng)

            sigma2 = float(global_var_model.sample_one_iteration(
                residual, rng, self.global_prior.sigma2_shape,
                self.global_prior.sigma2_rate))
            global_config.update_global_error_variance(sigma2)

            level.step_leaf_scale(rng)
            trend.step_leaf_scale(rng)
            effect.step_leaf_scale(rng)

            if keep:
                sigma2_draws.append(sigma2)
                tau0_draws.append(tau_0)

        for _ in range(num_gfr):
            sweep(gfr=True, keep=False)
        for i in range(num_burnin + num_mcmc * keep_every):
            past_burnin = i >= num_burnin
            keep = past_burnin and ((i - num_burnin + 1) % keep_every == 0)
            sweep(gfr=False, keep=keep)

        return {"level": level, "trend": trend, "effect": effect, "rfx": rfx,
                "sigma2": np.asarray(sigma2_draws, dtype=float),
                "tau_0": np.asarray(tau0_draws, dtype=float)}

    def _init_rfx(self, rfx_ids: np.ndarray, n: int) -> dict:
        from stochtree import (RandomEffectsContainer, RandomEffectsDataset,
                               RandomEffectsModel, RandomEffectsTracker)

        num_groups = int(np.unique(rfx_ids).size)
        basis = np.ones((n, 1))
        dataset = RandomEffectsDataset()
        dataset.add_group_labels(rfx_ids)
        dataset.add_basis(basis)
        tracker = RandomEffectsTracker(rfx_ids)
        model = RandomEffectsModel(1, num_groups)
        model.set_working_parameter(np.array([0.0]))
        model.set_group_parameters(np.zeros((1, num_groups)))
        model.set_working_parameter_covariance(np.identity(1))
        model.set_group_parameter_covariance(np.identity(1))
        model.set_variance_prior_shape(1.0)
        model.set_variance_prior_scale(1.0)
        container = RandomEffectsContainer()
        container.load_new_container(1, num_groups, tracker)
        return {"dataset": dataset, "tracker": tracker, "model": model,
                "container": container, "ids": rfx_ids, "basis": basis}

    # ------------------------------------------------------------------ #
    # Posterior summaries
    # ------------------------------------------------------------------ #
    def _collect(self, dm: DesignMatrices, rfx_ids) -> None:
        """Assemble the per-chain draws into ``(n_obs, n_draws)`` arrays."""
        n = dm.n
        a = np.hstack([_as_2d(c["level"].container.predict(c["level"].dataset), n)
                       for c in self._chains])
        b = np.hstack([_as_2d(c["trend"].container.predict(c["trend"].dataset), n)
                       for c in self._chains])
        tau_raw = np.hstack([
            _as_2d(c["effect"].container.predict_raw(c["effect"].dataset), n)
            for c in self._chains])
        tau_0 = np.concatenate([c["tau_0"] for c in self._chains])

        self.sigma2_draws = np.concatenate([c["sigma2"] for c in self._chains]) * self.y_std ** 2
        self.tau_0_draws = tau_0 * self.y_std
        self.tau_draws = (tau_raw + tau_0[None, :]) * self.y_std
        self.mu_draws = (a + b) * self.y_std + self.y_bar

        if self._chains[0]["rfx"] is not None:
            self.rfx_draws = np.hstack([
                _as_2d(c["rfx"]["container"].predict(c["rfx"]["ids"],
                                                     c["rfx"]["basis"]), n)
                for c in self._chains]) * self.y_std
            self.mu_draws = self.mu_draws + self.rfx_draws
        else:
            self.rfx_draws = None

    @property
    def n_draws(self) -> int:
        return self.tau_draws.shape[1]

    def _predict_block(self, block: str, X: np.ndarray) -> np.ndarray:
        """Predict one baseline block at an arbitrary design matrix."""
        from stochtree import Dataset
        X = np.ascontiguousarray(X, dtype=float)
        ds = Dataset()
        ds.add_covariates(X)
        return np.hstack([_as_2d(c[block].container.predict(ds), X.shape[0])
                          for c in self._chains])

    def level_trend_draws(self) -> tuple:
        """The gauge-normalised baseline components ``(a_tilde, b_tilde)``.

        ``a`` and ``b`` are identified only up to ``(a + c(x), b - c(x))``, which
        leaves ``mu = a + b`` and therefore ``tau`` untouched.  The theory fixes
        the gauge with ``b(t_1, .) = 0`` (Section ``sec:prior``); imposing it
        after the fact costs one extra prediction of the trend block at the
        first period and makes the two components separately readable:
        ``a_tilde`` is the baseline level at ``t_1`` and ``b_tilde`` the common
        path away from it.
        """
        t1 = float(np.min(self.design.time_values))
        b_ref = self._predict_block("trend", self.design.trend_at(t1))
        b = self._predict_block("trend", self.design.trend)
        a = self._predict_block("level", self.design.level)
        a_tilde = (a + b_ref) * self.y_std + self.y_bar
        b_tilde = (b - b_ref) * self.y_std
        return a_tilde, b_tilde

    def contrast_rows(self) -> tuple:
        """Treated post-treatment rows that have a ``(i, g-1)`` partner.

        Returns ``(rows, reference_rows)``: positions of the rows on which the
        difference-in-differences contrast is defined, and the position of each
        one's last pre-treatment observation.
        """
        cols = self.design.cols
        df = self.df
        row_of = {(int(u), int(t)): i for i, (u, t)
                  in enumerate(zip(df[cols.unit], df[cols.time]))}
        rows, refs = [], []
        for r in np.flatnonzero(df[cols.treated].to_numpy() == 1):
            u = int(df.at[r, cols.unit])
            g = df.at[r, cols.cohort]
            ref = row_of.get((u, int(g) - 1))
            if ref is not None:
                rows.append(r)
                refs.append(ref)
        return np.asarray(rows, dtype=int), np.asarray(refs, dtype=int)

    def did_contrast_draws(self) -> np.ndarray:
        """The DiD contrast of the fitted regression function, at observed rows.

        For each treated post-treatment row ``(i, t)`` of cohort ``g``,

            f(g, t, X_it) - f(g, g-1, X_i,g-1)
          - f(inf, t, X_it) + f(inf, g-1, X_i,g-1),

        with the never-treated cells obtained by evaluating the fitted level
        block at the same rows with the cohort label set to never-treated.  This
        is the *identified* object of equation ``eq:contrast``: a functional of
        the fitted conditional mean alone, which does not depend on how the fit
        is split between the baseline and the effect block.

        Note that the four cells are evaluated at each row's **own** covariates,
        not at a common ``x``.  That is what makes the quantity computable from a
        panel, and it is also what gives :meth:`check_identification` its
        content: the trend block cancels outright, and the level block cancels
        only when ``a(g, .) - a(inf, .)`` is the same at ``X_it`` and at
        ``X_i,g-1``, which is automatic for time-invariant covariates and
        generally false otherwise.

        Returns an ``(n_obs, n_draws)`` array, ``NaN`` on rows where the
        contrast is undefined (never-treated units, pre-treatment periods, and
        treated units whose ``g-1`` period is not in the panel).
        """
        rows, refs = self.contrast_rows()
        out = np.full_like(self.tau_draws, np.nan)
        if rows.size == 0:
            return out

        level_t, level_ref = self.design.level[rows], self.design.level[refs]
        a_g_t = self._predict_block("level", level_t)
        a_g_ref = self._predict_block("level", level_ref)
        a_0_t = self._predict_block("level", _with_column(level_t, 0, 0.0))
        a_0_ref = self._predict_block("level", _with_column(level_ref, 0, 0.0))

        # b(t, X_it) and b(g-1, X_i,g-1) enter the treated and the never-treated
        # arm identically, so the trend block drops out of the contrast.
        baseline = (a_g_t - a_g_ref) - (a_0_t - a_0_ref)
        out[rows] = baseline * self.y_std + self.tau_draws[rows]
        return out

    def check_identification(self, tol: float = 1e-8) -> dict:
        """Check that the two-way restriction really holds for this fit.

        The treatment forest is the identified DiD contrast of the fit exactly
        when the baseline part of that contrast vanishes.  The trend block
        always drops out; the level block drops out when the group gap
        ``a(g, .) - a(inf, .)`` is evaluated at the same covariate value in the
        post and the reference period, i.e. when the covariates are
        **time-invariant** (Assumption ``ass:sutva`` (iii)).  With covariates
        drawn per unit-*period*, as in the original DiD-BCF simulation study,
        ``a(g, X_it)`` moves with ``t`` and the baseline can carry group-by-time
        variation again, so the restriction is weaker than it looks.

        This is therefore a real check on the data actually passed in, not an
        algebraic identity, and it should be run whenever the panel construction
        changes.  It is cheap: four predictions of the level block on the
        treated post-treatment rows.
        """
        contrast = self.did_contrast_draws()
        rows, _ = self.contrast_rows()
        if rows.size == 0:
            return {"rows_checked": 0, "max_abs_tau_minus_contrast": 0.0,
                    "identified": True}
        gap = float(np.max(np.abs(contrast[rows] - self.tau_draws[rows])))
        return {"rows_checked": int(rows.size),
                "max_abs_tau_minus_contrast": gap,
                "identified": gap < tol}


def _as_2d(arr, n_rows: int) -> np.ndarray:
    """stochtree squeezes trailing singleton dimensions; restore ``(n, S)``."""
    return np.asarray(arr, dtype=float).reshape(n_rows, -1)


def _with_column(X: np.ndarray, col: int, value) -> np.ndarray:
    out = X.copy()
    out[:, col] = value
    return out
