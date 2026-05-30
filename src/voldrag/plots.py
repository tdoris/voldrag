"""Plotting helpers. Matplotlib only; no global style hacks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from voldrag.analytics import ensemble_mean, median_terminal
from voldrag.gbm import GBMPaths


def plot_path_fan(
    paths: GBMPaths,
    n_to_plot: int = 50,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a handful of paths plus the ensemble mean, median, and quantiles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sample = paths.s[: min(n_to_plot, paths.s.shape[0])]
    ax.plot(paths.t, sample.T, color="C0", alpha=0.15, linewidth=0.8)

    qs = np.quantile(paths.s, quantiles, axis=0)
    for q, row in zip(quantiles, qs, strict=True):
        ax.plot(paths.t, row, label=f"q{int(q * 100)}", linewidth=1.5)

    mean_curve = paths.s.mean(axis=0)
    ax.plot(paths.t, mean_curve, color="black", linewidth=2.0, label="sample mean")

    theoretical_mean = paths.s0 * np.exp(paths.mu * paths.t)
    theoretical_median = paths.s0 * np.exp(
        (paths.mu - 0.5 * paths.sigma**2) * paths.t
    )
    ax.plot(
        paths.t,
        theoretical_mean,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="theory E[S_t] = e^(mu t)",
    )
    ax.plot(
        paths.t,
        theoretical_median,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="theory median = e^((mu - sigma^2/2) t)",
    )

    ax.set_xlabel("t")
    ax.set_ylabel("S_t")
    ax.set_yscale("log")
    ax.set_title(
        f"GBM paths (mu={paths.mu}, sigma={paths.sigma}, "
        f"n_paths={paths.s.shape[0]})"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_terminal_distribution(
    paths: GBMPaths,
    ax: plt.Axes | None = None,
    bins: int = 80,
) -> plt.Axes:
    """Histogram of terminal log-wealth with mean / median lines marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    log_terminal = np.log(paths.terminal)
    ax.hist(log_terminal, bins=bins, density=True, alpha=0.6, color="C0")

    mean_log_theory = (paths.mu - 0.5 * paths.sigma**2) * paths.t[-1]
    ax.axvline(
        mean_log_theory,
        color="green",
        linestyle="--",
        label=f"E[log S_t] = (mu - sigma^2/2) t = {mean_log_theory:.3f}",
    )
    ax.axvline(
        np.log(ensemble_mean(paths.mu, paths.t[-1], paths.s0)),
        color="red",
        linestyle="--",
        label=f"log E[S_t] = mu t = {paths.mu * paths.t[-1]:.3f}",
    )
    ax.axvline(
        log_terminal.mean(),
        color="black",
        linewidth=1.0,
        label=f"sample E[log S_t] = {log_terminal.mean():.3f}",
    )

    ax.set_xlabel("log S_T")
    ax.set_ylabel("density")
    ax.set_title("Terminal log-wealth distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_sigma_sweep(sweep: dict[str, np.ndarray], ax: plt.Axes | None = None) -> plt.Axes:
    """Show how mean and median terminal wealth diverge as sigma grows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sigma = sweep["sigma"]
    ax.plot(sigma, sweep["sample_mean"], "o-", color="red", label="sample mean S_T")
    ax.plot(
        sigma,
        sweep["theoretical_mean"],
        "--",
        color="red",
        alpha=0.5,
        label="theory mean = e^(mu T)",
    )
    ax.plot(sigma, sweep["sample_median"], "o-", color="green", label="sample median S_T")
    ax.plot(
        sigma,
        sweep["theoretical_median"],
        "--",
        color="green",
        alpha=0.5,
        label="theory median = e^((mu - sigma^2/2) T)",
    )
    ax.axhline(1.0, color="grey", linewidth=0.8)
    ax.set_xlabel("sigma")
    ax.set_ylabel("S_T")
    ax.set_yscale("log")
    ax.set_title("Mean stays fixed; median collapses as sigma grows (mu held constant)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
