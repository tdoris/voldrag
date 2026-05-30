"""Simulators for Geometric Brownian Motion and discrete multiplicative games."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GBMPaths:
    """Container for simulated GBM paths.

    Attributes:
        t: shape (n_steps + 1,) time grid starting at 0.
        s: shape (n_paths, n_steps + 1) simulated price levels.
        mu: drift used.
        sigma: diffusion used.
        s0: initial level.
    """

    t: np.ndarray
    s: np.ndarray
    mu: float
    sigma: float
    s0: float

    @property
    def log_s(self) -> np.ndarray:
        return np.log(self.s)

    @property
    def terminal(self) -> np.ndarray:
        return self.s[:, -1]


def simulate_gbm(
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    s0: float = 1.0,
    seed: int | None = None,
) -> GBMPaths:
    """Simulate paths of dS = mu S dt + sigma S dW on [0, t].

    Uses the exact log-Euler update
        log S_{k+1} = log S_k + (mu - sigma^2/2) dt + sigma sqrt(dt) Z
    so each step is exact for GBM, not just a first-order approximation.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    dt = t / n_steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    increments = drift + diffusion * rng.standard_normal((n_paths, n_steps))
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    s = s0 * np.exp(log_paths)
    grid = np.linspace(0.0, t, n_steps + 1)
    return GBMPaths(t=grid, s=s, mu=mu, sigma=sigma, s0=s0)


def simulate_multiplicative_returns(
    up: float,
    down: float,
    p_up: float,
    n_rounds: int,
    n_paths: int,
    s0: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a discrete multiplicative game.

    Each round the wealth is multiplied by `up` with probability `p_up` and
    by `down` otherwise. The canonical "lose-50%-or-gain-60% on a fair coin"
    example -- positive arithmetic expectation, negative geometric growth --
    falls out by setting up=1.6, down=0.5, p_up=0.5.

    Returns an array of shape (n_paths, n_rounds + 1).
    """
    if not (0.0 <= p_up <= 1.0):
        raise ValueError("p_up must be a probability")

    rng = np.random.default_rng(seed)
    draws = rng.random((n_paths, n_rounds)) < p_up
    multipliers = np.where(draws, up, down)
    log_steps = np.log(multipliers)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1
    )
    return s0 * np.exp(log_paths)
