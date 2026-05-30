"""`voldrag-demo` entry point: runs the canonical experiments and saves figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from voldrag.experiments import coin_flip_game, ensemble_vs_typical, sigma_sweep
from voldrag.gbm import simulate_gbm
from voldrag.plots import (
    plot_path_fan,
    plot_regime_map,
    plot_sigma_sweep,
    plot_terminal_distribution,
    save,
)

# Illustrative (mu, sigma) markers for the regime map. Numbers match the
# README's regime table plus the worked levered-strategy example.
REGIME_MARKERS: tuple[tuple, ...] = (
    ("T-bills", 0.04, 0.01, (8, 6)),
    ("IG bonds", 0.05, 0.06, (8, 4)),
    ("Equity index", 0.10, 0.18, (8, 4)),
    ("Large-cap", 0.12, 0.35, (8, 4)),
    ("3x ETF", 0.30, 0.60, (-50, -14)),
    ("BTC", 0.30, 0.70, (8, 4)),
    ("Small-cap", 0.15, 0.80, (-70, 4)),
    ("Strategy 1x", 0.04, 0.0175, (8, -12)),
    ("Strategy 5x", 0.195, 0.0875, (8, 4)),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Volatility-drag demo runner.")
    parser.add_argument("--mu", type=float, default=0.10, help="drift per unit time")
    parser.add_argument("--sigma", type=float, default=0.30, help="volatility per unit time")
    parser.add_argument("--t", type=float, default=30.0, help="horizon")
    parser.add_argument("--n-paths", type=int, default=20_000)
    parser.add_argument("--n-steps", type=int, default=30 * 252)
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("figures"),
        help="directory to write PNGs into",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip matplotlib output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print("=" * 72)
    print("Experiment 1: GBM ensemble mean vs typical path")
    print("=" * 72)
    summary = ensemble_vs_typical(
        mu=args.mu,
        sigma=args.sigma,
        t=args.t,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        seed=args.seed,
    )
    print(summary.report())
    print()

    print("=" * 72)
    print("Experiment 2: discrete +60% / -50% coin-flip game")
    print("=" * 72)
    game = coin_flip_game(seed=args.seed)
    print(game.report())
    print()

    print("=" * 72)
    print("Experiment 3: sigma sweep at fixed mu")
    print("=" * 72)
    sweep = sigma_sweep(mu=args.mu, t=args.t, seed=args.seed)
    for s, mean, median in zip(
        sweep["sigma"], sweep["sample_mean"], sweep["sample_median"], strict=True
    ):
        print(
            f"  sigma={s:0.3f}  sample_mean={mean:10.4f}  sample_median={median:10.4f}"
        )
    print()

    if args.no_plots:
        return 0

    # Plotting: re-use the same path object for the two GBM figures.
    paths = simulate_gbm(
        args.mu, args.sigma, args.t, args.n_steps, args.n_paths, seed=args.seed
    )

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    plot_path_fan(paths, ax=ax1)
    save(fig1, args.figures / "01_path_fan.png")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    plot_terminal_distribution(paths, ax=ax2)
    save(fig2, args.figures / "02_terminal_distribution.png")

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    plot_sigma_sweep(sweep, ax=ax3)
    save(fig3, args.figures / "03_sigma_sweep.png")

    fig4, ax4 = plt.subplots(figsize=(10, 7))
    plot_regime_map(markers=REGIME_MARKERS, ax=ax4)
    save(fig4, args.figures / "04_regime_map.png")

    plt.close("all")
    print(f"Wrote figures to {args.figures.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
