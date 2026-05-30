"""High-level experiments illustrating volatility drag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voldrag.analytics import (
    ensemble_mean,
    geometric_mean_return,
    median_terminal,
    prob_path_below_mean,
    volatility_drag,
)
from voldrag.gbm import simulate_gbm, simulate_multiplicative_returns


@dataclass
class EnsembleVsTypical:
    """Side-by-side summary of ensemble mean vs typical (median) path."""

    mu: float
    sigma: float
    t: float
    n_paths: int
    sample_mean: float
    sample_median: float
    theoretical_mean: float
    theoretical_median: float
    drag: float
    geometric_rate: float
    prob_below_mean: float

    def report(self) -> str:
        lines = [
            f"GBM(mu={self.mu:.4f}, sigma={self.sigma:.4f}) over t={self.t}",
            f"  n_paths               = {self.n_paths}",
            f"  volatility drag       = 1/2 sigma^2 = {self.drag:.6f}",
            f"  arithmetic rate (mu)  = {self.mu:.6f}",
            f"  geometric rate        = mu - 1/2 sigma^2 = {self.geometric_rate:.6f}",
            "",
            f"  theoretical E[S_t]    = {self.theoretical_mean:.6f}",
            f"  sample mean S_t       = {self.sample_mean:.6f}",
            f"  theoretical median    = {self.theoretical_median:.6f}",
            f"  sample median S_t     = {self.sample_median:.6f}",
            "",
            f"  P(S_t < E[S_t])       = {self.prob_below_mean:.4f}",
            "  (a single path falls below the ensemble mean MORE OFTEN than not)",
        ]
        return "\n".join(lines)


def ensemble_vs_typical(
    mu: float = 0.10,
    sigma: float = 0.30,
    t: float = 30.0,
    n_steps: int = 30 * 252,
    n_paths: int = 20_000,
    seed: int | None = 0,
) -> EnsembleVsTypical:
    """Simulate many GBM paths and compare the sample mean to the typical path.

    The defaults pick equity-like numbers (10% annual drift, 30% annual vol,
    30 year horizon) where volatility drag is large enough to be visceral.
    """
    paths = simulate_gbm(mu, sigma, t, n_steps, n_paths, seed=seed)
    terminal = paths.terminal
    return EnsembleVsTypical(
        mu=mu,
        sigma=sigma,
        t=t,
        n_paths=n_paths,
        sample_mean=float(terminal.mean()),
        sample_median=float(np.median(terminal)),
        theoretical_mean=ensemble_mean(mu, t),
        theoretical_median=median_terminal(mu, sigma, t),
        drag=volatility_drag(sigma),
        geometric_rate=geometric_mean_return(mu, sigma),
        prob_below_mean=prob_path_below_mean(mu, sigma, t),
    )


@dataclass
class CoinFlipGame:
    """The +60% / -50% fair-coin game.

    E[multiplier] = 1.05 > 1, so arithmetic expectation says "play".
    E[log multiplier] = 1/2 (log 1.6 + log 0.5) ~= -0.1116 < 0,
    so geometric expectation says "ruin". The simulation confirms the second.
    """

    up: float
    down: float
    p_up: float
    n_rounds: int
    n_paths: int
    arithmetic_expected_multiplier: float
    geometric_expected_log: float
    sample_mean_wealth: float
    sample_median_wealth: float
    fraction_above_start: float

    def report(self) -> str:
        lines = [
            f"Multiplicative coin game: up x{self.up}, down x{self.down}, "
            f"p_up={self.p_up}",
            f"  rounds                 = {self.n_rounds}",
            f"  paths                  = {self.n_paths}",
            f"  E[multiplier]          = {self.arithmetic_expected_multiplier:.6f} "
            "(> 1 means positive arithmetic edge)",
            f"  E[log multiplier]      = {self.geometric_expected_log:.6f} "
            "(< 0 means negative geometric edge)",
            "",
            f"  sample mean wealth     = {self.sample_mean_wealth:.4f}",
            f"  sample median wealth   = {self.sample_median_wealth:.4f}",
            f"  fraction P(wealth > 1) = {self.fraction_above_start:.4f}",
            "  (ensemble is positive, typical player is ruined)",
        ]
        return "\n".join(lines)


def coin_flip_game(
    up: float = 1.6,
    down: float = 0.5,
    p_up: float = 0.5,
    n_rounds: int = 100,
    n_paths: int = 50_000,
    seed: int | None = 0,
) -> CoinFlipGame:
    paths = simulate_multiplicative_returns(
        up=up, down=down, p_up=p_up, n_rounds=n_rounds, n_paths=n_paths, seed=seed
    )
    terminal = paths[:, -1]
    return CoinFlipGame(
        up=up,
        down=down,
        p_up=p_up,
        n_rounds=n_rounds,
        n_paths=n_paths,
        arithmetic_expected_multiplier=p_up * up + (1 - p_up) * down,
        geometric_expected_log=p_up * np.log(up) + (1 - p_up) * np.log(down),
        sample_mean_wealth=float(terminal.mean()),
        sample_median_wealth=float(np.median(terminal)),
        fraction_above_start=float(np.mean(terminal > 1.0)),
    )


def sigma_sweep(
    mu: float = 0.10,
    sigmas: np.ndarray | None = None,
    t: float = 30.0,
    n_steps: int = 30 * 12,
    n_paths: int = 5_000,
    seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """Hold mu fixed, sweep sigma, record sample mean & median of S_t."""
    if sigmas is None:
        sigmas = np.linspace(0.0, 0.6, 13)
    means = np.empty_like(sigmas)
    medians = np.empty_like(sigmas)
    theoretical_means = np.empty_like(sigmas)
    theoretical_medians = np.empty_like(sigmas)

    for i, sigma in enumerate(sigmas):
        paths = simulate_gbm(
            mu, float(sigma), t, n_steps, n_paths, seed=None if seed is None else seed + i
        )
        terminal = paths.terminal
        means[i] = terminal.mean()
        medians[i] = np.median(terminal)
        theoretical_means[i] = ensemble_mean(mu, t)
        theoretical_medians[i] = median_terminal(mu, float(sigma), t)

    return {
        "sigma": sigmas,
        "sample_mean": means,
        "sample_median": medians,
        "theoretical_mean": theoretical_means,
        "theoretical_median": theoretical_medians,
    }
