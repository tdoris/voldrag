"""Volatility drag explorations.

The headline identity behind this package: for a Geometric Brownian Motion

    dS_t / S_t = mu dt + sigma dW_t

the realised long-run growth rate of a *single path* converges to

    mu - 1/2 * sigma^2

even though the ensemble mean still grows as exp(mu * t). The gap

    1/2 * sigma^2

is "volatility drag" -- the cost the typical path pays for variance.
"""

from voldrag.analytics import (
    arithmetic_mean_return,
    ensemble_mean,
    geometric_mean_return,
    median_terminal,
    volatility_drag,
)
from voldrag.gbm import simulate_gbm, simulate_multiplicative_returns

__all__ = [
    "arithmetic_mean_return",
    "ensemble_mean",
    "geometric_mean_return",
    "median_terminal",
    "simulate_gbm",
    "simulate_multiplicative_returns",
    "volatility_drag",
]
