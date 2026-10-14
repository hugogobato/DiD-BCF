"""Existence repair and native inverse-gamma parameterization checks."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln
from didbcf_structured import GlobalPrior, StructuredDiDBCF


@pytest.mark.parametrize("field", ["sigma2_shape", "sigma2_rate", "intercept_prior_var"])
@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_prior_rejected(field, value):
    with pytest.raises(ValueError, match="strictly positive"):
        GlobalPrior(**{field: value})


def test_default_and_copy():
    prior = GlobalPrior()
    assert (prior.sigma2_shape, prior.sigma2_rate) == (2., 1.)
    model = StructuredDiDBCF(global_prior=prior)
    prior.sigma2_rate = 0
    assert model.global_prior.sigma2_rate == 1
    model.global_prior.sigma2_rate = 0
    with pytest.raises(ValueError, match="strictly positive"):
        model.sample(None)


def test_cpp_inverse_gamma_conditional():
    from stochtree import Residual, RNG, GlobalVarianceModel
    residual = Residual(np.ones(10))
    rng = RNG(981)
    sampler = GlobalVarianceModel()
    # IG(3 + n/2, 2 + SSE/2) = IG(8, 7), mean 1, variance 1/6.
    draws = np.array([sampler.sample_one_iteration(residual, rng, 3., 2.)
                      for _ in range(20000)])
    assert np.isfinite(draws).all() and (draws > 0).all()
    assert abs(draws.mean() - 1) < .025
    assert abs(draws.var() - 1/6) < .025


@pytest.mark.parametrize("b", [.1, 1., 10.])
def test_integrated_likelihood_upper_bound(b):
    # All possible regression means satisfy L <= (2*pi*v)^(-m/2).
    # Numerical integration in log(v) checks the analytic finite bound.
    a, m = 2., 8
    exact_log = (-m/2*np.log(2*np.pi*b) + gammaln(a+m/2)-gammaln(a))
    def integrand(z):
        return np.exp(a*np.log(b)-gammaln(a)-m/2*np.log(2*np.pi)
                      -(a+m/2)*z-b*np.exp(-z)-exact_log)
    integral, err = quad(integrand, -35, 35, epsabs=1e-10)
    assert abs(integral-1) < 1e-8


def test_saturated_design_has_finite_repaired_normalizer():
    # K=I, y=0: the old marginal likelihood stays positive at v=0.
    # With IG(2,1), both tails are integrable even in this saturated case.
    def density_logv(z):
        v = np.exp(z)
        return np.exp(-2*np.log(2*np.pi*(1+v))-2*z-1/v)
    values = [quad(density_logv, -lim, lim, epsabs=1e-11)[0]
              for lim in [12, 24, 36]]
    assert values[-1] > 0 and abs(values[-1]-values[-2]) < 1e-10
