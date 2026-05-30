"""Closed-form quantities for Geometric Brownian Motion.

For dS_t = mu S_t dt + sigma S_t dW_t with S_0 = 1, Ito's lemma gives

    log S_t = (mu - 1/2 sigma^2) t + sigma W_t

so log S_t ~ Normal((mu - sigma^2/2) t, sigma^2 t) and S_t is log-normal.
The functions here expose the resulting moments and quantiles.
"""

from __future__ import annotations

import math


def volatility_drag(sigma: float) -> float:
    """The drag term 1/2 * sigma^2.

    This is the gap between the ensemble (arithmetic) growth rate mu and
    the time-average (geometric) growth rate that a single path realises.
    """
    return 0.5 * sigma * sigma


def arithmetic_mean_return(mu: float) -> float:
    """Continuously-compounded arithmetic mean growth rate.

    E[S_t] = exp(mu * t), so the per-unit-time arithmetic rate is mu.
    """
    return mu


def geometric_mean_return(mu: float, sigma: float) -> float:
    """Almost-sure long-run growth rate of a single path: mu - sigma^2/2."""
    return mu - volatility_drag(sigma)


def ensemble_mean(mu: float, t: float, s0: float = 1.0) -> float:
    """E[S_t] for GBM starting at s0."""
    return s0 * math.exp(mu * t)


def median_terminal(mu: float, sigma: float, t: float, s0: float = 1.0) -> float:
    """Median of S_t. Since log S_t is normal, median = exp(E[log S_t])."""
    return s0 * math.exp((mu - volatility_drag(sigma)) * t)


def mode_terminal(mu: float, sigma: float, t: float, s0: float = 1.0) -> float:
    """Mode of S_t, i.e. the location of the peak of the log-normal pdf."""
    return s0 * math.exp((mu - 3 * volatility_drag(sigma)) * t)


def variance_terminal(mu: float, sigma: float, t: float, s0: float = 1.0) -> float:
    """Var(S_t) for GBM."""
    return s0 * s0 * math.exp(2 * mu * t) * (math.exp(sigma * sigma * t) - 1.0)


def prob_path_below_mean(mu: float, sigma: float, t: float) -> float:
    """Probability that a single realisation S_t falls below its expectation E[S_t].

    P(S_t < E[S_t]) = Phi(sigma sqrt(t) / 2), where Phi is the standard normal CDF.
    The drag means this probability exceeds 1/2 and grows with sigma sqrt(t).
    """
    from math import erf, sqrt

    z = sigma * math.sqrt(t) / 2.0
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))
