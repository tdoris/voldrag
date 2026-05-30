"""Plotting helpers. Matplotlib only; no global style hacks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

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


REGIME_LEVELS = (0.0, 0.05, 0.30, 0.80, 1.00, 2.00, 5.00)
REGIME_LABELS = (
    "<5%: invisible",
    "5-30%: tolerable",
    "30-80%: material",
    "80-100%: critical",
    "100-200%: self-defeating",
    ">200%: typical decay",
)
REGIME_COLORS = ("#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027", "#67001f")


def plot_regime_map(
    mu_range: tuple[float, float] = (0.0, 0.30),
    sigma_range: tuple[float, float] = (0.0, 0.85),
    n_grid: int = 400,
    markers: Sequence[tuple] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of `sigma^2 / (2 mu)` over the (mu, sigma) plane.

    The bands match the regime table in the README: <5% invisible,
    5-30% tolerable, 30-80% material, 80-100% critical, >100% self-defeating.
    `markers` is a list of `(label, mu, sigma)` or
    `(label, mu, sigma, (dx, dy))` tuples where `(dx, dy)` is the label
    offset in display points (defaults to (6, 4)).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    mu_grid = np.linspace(*mu_range, n_grid)
    sigma_grid = np.linspace(*sigma_range, n_grid)
    sigma_mesh, mu_mesh = np.meshgrid(sigma_grid, mu_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            mu_mesh > 1e-9, (sigma_mesh**2) / (2.0 * mu_mesh), np.nan
        )

    cmap = ListedColormap(REGIME_COLORS)
    norm = BoundaryNorm(REGIME_LEVELS, cmap.N)
    pcm = ax.pcolormesh(
        sigma_mesh,
        mu_mesh,
        np.clip(ratio, REGIME_LEVELS[0] + 1e-6, REGIME_LEVELS[-1] - 1e-6),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    cbar = plt.colorbar(pcm, ax=ax, ticks=REGIME_LEVELS, label="sigma^2 / (2 mu)")
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in REGIME_LEVELS])

    # Break-even curve sigma = sqrt(2 mu) is the 100% boundary.
    mu_curve = np.linspace(mu_range[0] + 1e-6, mu_range[1], 200)
    sigma_breakeven = np.sqrt(2.0 * mu_curve)
    inside = sigma_breakeven <= sigma_range[1]
    if inside.any():
        ax.plot(
            sigma_breakeven[inside],
            mu_curve[inside],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="break-even: sigma = sqrt(2 mu)",
        )

    if markers:
        for entry in markers:
            if len(entry) == 4:
                label, mu, sigma, offset = entry
            else:
                label, mu, sigma = entry
                offset = (6, 4)
            ax.plot(
                sigma,
                mu,
                marker="o",
                markersize=8,
                color="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=5,
            )
            ax.annotate(
                label,
                (sigma, mu),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=6,
            )

    # Legend swatches for the regime bands.
    band_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in REGIME_COLORS
    ]
    ax.legend(
        band_handles + [plt.Line2D([0], [0], color="black", linestyle="--")],
        list(REGIME_LABELS) + ["break-even sigma=sqrt(2 mu)"],
        loc="upper left",
        fontsize=7,
        framealpha=0.85,
    )

    ax.set_xlim(sigma_range)
    ax.set_ylim(mu_range)
    ax.set_xlabel("sigma (annualized vol)")
    ax.set_ylabel("mu (annualized drift)")
    ax.set_title("Volatility-drag regime map: sigma^2 / (2 mu)")
    return ax


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
