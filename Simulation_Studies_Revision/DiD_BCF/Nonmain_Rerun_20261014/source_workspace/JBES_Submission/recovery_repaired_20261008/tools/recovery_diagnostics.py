"""Pinned diagnostic conventions for the recovery campaign.

Implemented here, with self-tests, so that notebooks and the local collector
share one version:

* rank-normalised and folded split R-hat, the maximum of the two, following
  Vehtari et al. (2021), https://doi.org/10.1214/20-BA1221 (copied from
  ``audit/collect_proper_reruns.py:rank_rhat`` with the same convention);
* Geyer paired-monotone effective sample size, copied from ``audit/mcmc_ess.py``
  (complete positive lag pairs, antithetic ESS capped at the draw count);
* mean MCSE  = posterior SD / sqrt(mean ESS);
* SD MCSE    = SD * sqrt((Pearson kurtosis - 1) / (4 * mean ESS)), the usual
  delta-method expression for the standard deviation;
* quantile MCSE = sqrt(p(1-p) / ESS(indicator(x <= q))) / f_hat(q), with a
  normal-kernel density estimate of the posterior at q.  This replaces the
  independent-draw N of ``posterior::mcse_quantile`` by the autocorrelation-
  adjusted effective sample size of the quantile indicator, which is the
  conservative choice for MCMC output.  It is a project convention, not a
  claim of exact agreement with any software package.

None of these diagnostics certifies convergence; they flag it.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.stats import norm, rankdata

__all__ = ["rank_rhat", "raw_rhat", "ess", "split", "tail_ess",
           "mean_mcse", "sd_mcse", "quantile_mcse", "summarise",
           "batch_mcse", "self_test"]


def split(x):
    n = x.shape[1] // 2
    return np.concatenate((x[:, :n], x[:, -n:]), axis=0)


def _raw(values):
    n = values.shape[1]
    within = values.var(axis=1, ddof=1).mean()
    between = values.mean(axis=1).var(ddof=1)
    if within <= 0:
        return float("inf")
    return float(np.sqrt(((n - 1) / n * within + between) / within))


def raw_rhat(x):
    return _raw(np.asarray(x, dtype=float))


def rank_rhat(x):
    """Maximum of rank-normalised and folded split R-hat."""
    x = np.asarray(x, dtype=float)
    n = x.shape[1] // 2
    s = np.concatenate([x[:, :n], x[:, -n:]], axis=0)

    def rank_normal(a):
        return norm.ppf((rankdata(a).reshape(a.shape) - .375) / (a.size + .25))

    bulk = _raw(rank_normal(s))
    folded = _raw(rank_normal(np.abs(s - np.median(s))))
    return float(max(bulk, folded))


def ess(x):
    """Geyer paired-monotone ESS (audit/mcmc_ess.py convention)."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    length = next_fast_len(2 * n)
    ft = rfft(centered, n=length, axis=1)
    acov = irfft(ft * ft.conjugate(), n=length, axis=1)[:, :n] / n
    within = np.mean(acov[:, 0]) * n / (n - 1)
    total = (n - 1) / n * within + np.var(x.mean(axis=1), ddof=1)
    if total <= 0:
        return float("nan")
    rho = 1 - (within - acov.mean(axis=0)) / total
    rho[0] = 1
    pairs = rho[:2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    pairs = pairs[:stop[0]] if len(stop) else pairs
    if not len(pairs):
        return float(m * n)
    tau = max(1., -1 + 2 * np.minimum.accumulate(pairs).sum())
    return float(m * n / tau)


def bulk_ess(x):
    x = np.asarray(x, dtype=float)
    s = split(x)
    z = norm.ppf((rankdata(s).reshape(s.shape) - .375) / (s.size + .25))
    return ess(z)


def tail_ess(x):
    x = np.asarray(x, dtype=float)
    return float(min(
        ess(split((x <= np.quantile(x, p)).astype(float)))
        for p in (.05, .95)))


def mean_mcse(x):
    x = np.asarray(x, dtype=float)
    value = ess(split(x))
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(value))


def sd_mcse(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[0] * x.shape[1]
    s = x.std(ddof=1)
    if s <= 0:
        return 0.0
    mu = x.mean()
    m4 = np.mean((x - mu) ** 4)
    kurt = m4 / s ** 4                     # Pearson kurtosis, normal = 3
    value = ess(split(x))
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    factor = max(kurt - 1.0, 1e-12) / (4.0 * value)
    return float(s * np.sqrt(factor))


def _density_at(x, q):
    """Normal-kernel density estimate of the pooled draws at quantile q."""
    flat = np.asarray(x, dtype=float).reshape(-1)
    sd = flat.std(ddof=1)
    if sd <= 0:
        return float("nan")
    iqr = np.subtract(*np.percentile(flat, [75, 25]))
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    bandwidth = 0.9 * scale * flat.size ** (-0.2)
    if bandwidth <= 0:
        return float("nan")
    z = (q - flat) / bandwidth
    weights = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return float(weights.mean() / bandwidth)


def quantile_mcse(x, q):
    """MCSE of the posterior q-quantile (project convention, see module doc)."""
    x = np.asarray(x, dtype=float)
    dens = _density_at(x, float(np.quantile(x, q)))
    if not np.isfinite(dens) or dens <= 0:
        return float("nan")
    indicator = (x <= np.quantile(x, q)).astype(float)
    value = ess(split(indicator))
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    return float(np.sqrt(q * (1 - q) / value) / dens)


def batch_mcse(x):
    """Conservative non-overlapping batch-means MCSE (diagnostic only).

    Persistent dependence beyond the batch length can bias this downward; it is
    reported alongside, never instead of, the ESS-based MCSE.
    """
    x = np.asarray(x, dtype=float)
    estimates = []
    for size in (25, 50, 100):
        batches = x.shape[1] // size
        if batches < 4:
            continue
        means = x[:, :batches * size].reshape(x.shape[0], batches, size).mean(axis=2)
        var_mean = means.var(axis=1, ddof=1) / batches
        estimates.append(float(np.sqrt(var_mean.sum() / x.shape[0] ** 2)))
    return max(estimates) if estimates else float("nan")


_QUANTILE_LABELS = {0.025: "q025", 0.05: "q05", 0.5: "q50", 0.95: "q95",
                    0.975: "q975"}


def summarise(x, quantiles=(.025, .05, .5, .95, .975)) -> dict:
    x = np.asarray(x, dtype=float)
    out = {
        "n_chains": int(x.shape[0]),
        "n_draws_per_chain": int(x.shape[1]),
        "post_mean": float(x.mean()),
        "post_sd": float(x.std(ddof=1)),
        "rhat": (rank_rhat(x) if x.shape[1] >= 4 else float("nan")),
        "rhat_second_half": (rank_rhat(x[:, x.shape[1] // 2:])
                             if x.shape[1] // 2 >= 2 else float("nan")),
        "bulk_ess": bulk_ess(x),
        "tail_ess": tail_ess(x),
        "mean_ess": ess(split(x)),
        "mean_mcse": mean_mcse(x),
        "sd_mcse": sd_mcse(x),
        "batch_mcse": batch_mcse(x),
        "chain_mean_range": float(np.ptp(x.mean(axis=1))),
        "second_half_shift_in_sd": float(
            (x[:, x.shape[1] // 2:].mean() - x[:, :x.shape[1] // 2].mean())
            / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("nan"),
    }
    for q in quantiles:
        key = _QUANTILE_LABELS.get(round(float(q), 6), f"q{int(round(1000 * q)):03d}")
        out[key] = float(np.quantile(x, q))
        out[f"{key}_mcse"] = quantile_mcse(x, q)
    return out


def self_test() -> str:
    rng = np.random.default_rng(20260929)
    iid = rng.normal(size=(4, 10000))
    assert rank_rhat(iid) < 1.02
    shifted = iid.copy()
    shifted[0] += 8
    assert rank_rhat(shifted) > 1.1
    values = ess(split(iid))
    assert .8 * iid.size < values <= iid.size
    rho = .9
    ar1 = rng.normal(size=(4, 10000))
    for i in range(1, ar1.shape[1]):
        ar1[:, i] = rho * ar1[:, i - 1] + np.sqrt(1 - rho * rho) * ar1[:, i]
    theoretical = ar1.size * (1 - rho) / (1 + rho)
    assert .65 < ess(split(ar1)) / theoretical < 1.35
    q25 = quantile_mcse(iid, .25)
    sigma = iid.std(ddof=1)
    reference = np.sqrt(.25 / iid.size) / (norm.pdf(0) / sigma)
    assert .6 < q25 / reference < 1.6
    sd_ref = iid.std(ddof=1) / np.sqrt(2 * iid.size)
    assert .5 < sd_mcse(iid) / sd_ref < 2.0
    return "diagnostics self-test passed"


if __name__ == "__main__":
    print(self_test())
