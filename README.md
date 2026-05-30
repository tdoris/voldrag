# voldrag

An exploration of **volatility drag** in stochastic processes: the term
`mu - 1/2 sigma^2` that appears whenever you compound random returns.
This repo is an educational sandbox — code and exposition together.

## The headline result

For Geometric Brownian Motion

```
dS_t / S_t = mu dt + sigma dW_t
```

Ito's lemma gives

```
log S_t = (mu - 1/2 sigma^2) t + sigma W_t
```

so two different "expected returns" live in the same model:

| Quantity                      | Value                | Interpretation                                       |
| ----------------------------- | -------------------- | ---------------------------------------------------- |
| Arithmetic / ensemble rate    | `mu`                 | `E[S_t] = e^(mu t)` -- average across infinite paths |
| Geometric / time-average rate | `mu - 1/2 sigma^2`   | growth rate the *typical* single path realises       |
| **Volatility drag**           | `1/2 sigma^2`        | the gap between the two                              |

The ensemble mean is pulled up by a thin tail of extreme winners. Almost
every individual path grows at the slower geometric rate, and as `t` grows
`P(S_t < E[S_t]) -> 1` even though `E[S_t]` keeps rising.

## Building intuition

Three angles that reinforce each other:

**1. Drag falls out of Taylor-expanding the log.** For small returns `r`,

```
log(1 + r) ~= r - r^2 / 2
```

Compounding sums logs, so over many periods you get
`Sum r_i - Sum r_i^2 / 2 ~= mu t - (sigma^2 / 2) t`. The drag *is* the
second-order term — variance is automatically a quadratic penalty on
compounding. Nothing mysterious past that.

Equivalent gut-feel version: a 50% loss takes a 100% gain to recover.
Multiplication is asymmetric around 1, and `1/2 sigma^2` is the size of
that asymmetry.

**2. The dimensionless ratio that matters is `sigma^2 / (2 mu)`** —
drag as a *fraction* of the drift you'd otherwise earn.

| `sigma^2 / (2 mu)` | Regime           | What happens                                              |
| ------------------ | ---------------- | --------------------------------------------------------- |
| < 5%               | Negligible       | Geometric ~= arithmetic. Plan in arithmetic terms.        |
| 10-30%             | Material         | You forfeit 10-30% of your return. Plan in geometric.     |
| ~50-80%            | Critical         | Median return is a small fraction of mean return.         |
| ~= 100%            | Break-even       | Typical wealth flat forever; ensemble mean still climbs.  |
| > 100%             | Self-defeating   | Typical wealth -> 0 even though E[wealth] -> inf.         |

The break-even volatility for a given drift is `sigma_crit = sqrt(2 mu)`.
For 10% drift that's 45% vol; for 4% drift it's only 28%.

**3. Per-period drag is constant; the gap between ensemble and typical
grows exponentially in `t`.** The ratio
`E[S_t] / median(S_t) = exp((sigma^2 / 2) t)`. Even at modest sigma,
30 years of compounding turns a small per-year drag into a 3-5x gap
between the average outcome and yours.

## Concrete regimes (annualized, illustrative)

| Asset                          | mu   | sigma | Drag (1/2 sigma^2) | sigma^2/(2 mu) | Verdict                              |
| ------------------------------ | ---- | ----- | ------------------ | -------------- | ------------------------------------ |
| T-bills                        | 4%   | 1%    | 0.005%             | 0.1%           | Invisible                            |
| IG bonds                       | 5%   | 6%    | 0.18%              | 4%             | Invisible                            |
| Diversified equity index       | 10%  | 18%   | 1.6%               | 16%            | Material but tolerable               |
| Single large-cap stock         | 12%  | 35%   | 6.1%               | 51%            | Material                             |
| 3x leveraged equity ETF        | 30%  | 60%   | 18%                | 60%            | Dominant — canonical cautionary tale |
| BTC (recent vol regime)        | 30%  | 70%   | 24.5%              | 82%            | Critical                             |
| Single speculative small-cap   | 15%  | 80%   | 32%                | 213%           | Typical path decays                  |

The 3x leveraged ETF row is the most useful one to internalize. It is
sold on arithmetic terms ("3x the index return"); the typical holder
gets roughly the geometric return, which after drag is often *worse
than the unlevered index*. This is not a flaw in the product — it is
`1/2 sigma^2` doing exactly what it says on the tin.

## Rules of thumb

- **Drag (% per year) ~= sigma^2 / 2** with sigma in decimals. So 20%
  vol -> 2%, 40% -> 8%, 60% -> 18%. Quadratic — doubling vol quadruples
  drag.
- **Kelly bet `f* = mu / sigma^2`** maximizes geometric growth. Levering
  past `f*` puts you in the regime where drag beats drift; equivalently,
  `sigma^2 / (2 mu) > 1` once you scale everything by leverage.
- **Why rebalancing helps**: across uncorrelated assets, rebalancing
  harvests the gap between arithmetic and geometric returns (Shannon's
  demon / volatility pumping). The drag is the prize.
- **The frame I find most useful**: arithmetic returns describe a
  parallel-universe ensemble. Geometric returns describe your single
  life. If you only get one path, drag isn't a correction — it's the
  actual number you should plan around.

## How leverage interacts with drag

Leverage `L` scales drift and vol linearly, so drag scales
*quadratically*:

- gross drift -> `L * mu`
- vol         -> `L * sigma`
- drag        -> `L^2 * (1/2) sigma^2`
- after a financing cost `c`: geometric rate = `L * mu - c - L^2 * (1/2) sigma^2`

The growth-optimal (Kelly) leverage solves
`d/dL [L mu - c - L^2 sigma^2 / 2] = 0`, giving
`L* = mu / sigma^2` (ignoring the L-dependence of `c`). For `L > L*`
every additional turn of leverage *reduces* your typical compounded
return: drag has overtaken drift. The key qualitative fact is that
drift is linear in L while drag is quadratic, so however small drag
looks at L=1, there is always a leverage level past which it dominates.

## Tutorial notebook

For a full guided walkthrough of every concept in this repo — derivation,
simulations, regime map, leverage, Kelly, caveats — the executed tutorial
is in [`notebooks/voldrag_tutorial.ipynb`](notebooks/voldrag_tutorial.ipynb).

GitHub's in-browser notebook renderer is currently failing for this repo
("An error occurred"), so a pre-rendered HTML copy is committed alongside:

- **View the rendered tutorial:**
  [htmlpreview.github.io/?...voldrag_tutorial.html](https://htmlpreview.github.io/?https://raw.githubusercontent.com/tdoris/voldrag/master/notebooks/voldrag_tutorial.html)
- Or clone the repo and open the `.ipynb` locally in Jupyter.

## What's in the repo

```
src/voldrag/
  analytics.py     -- closed-form quantities (drag, mean, median, etc.)
  gbm.py           -- exact log-Euler GBM + discrete multiplicative games
  experiments.py   -- ensemble-vs-typical, coin-flip game, sigma sweep
  plots.py         -- path fans, terminal-distribution histograms, sweeps, regime map
  cli.py           -- `voldrag-demo` entry point
scripts/demo.py    -- equivalent script entry point
notebooks/
  voldrag_tutorial.ipynb  -- executed tutorial, viewable on GitHub
  build_tutorial.py       -- regenerates the notebook from inline cell sources
figures/           -- output of the demo runner
```

## Running

```bash
uv sync
uv run voldrag-demo                       # full demo, writes PNGs to figures/
uv run voldrag-demo --no-plots            # console-only
uv run voldrag-demo --mu 0.07 --sigma 0.4 --t 50
```

Quick programmatic use:

```python
from voldrag.experiments import ensemble_vs_typical
print(ensemble_vs_typical(mu=0.10, sigma=0.30, t=30).report())
```

## What the demo produces

1. **GBM ensemble vs typical** -- 20,000 paths over 30 years at mu=10%,
   sigma=30%. The sample mean tracks `e^(mu t)`; the sample median sits
   far below at `e^((mu - sigma^2 / 2) t)`. Roughly 75% of paths finish
   below the ensemble mean.
2. **+60% / -50% fair coin** -- arithmetic expectation per round is 1.05
   (positive edge!), but `E[log multiplier] = -0.1116`, so almost every
   player goes broke. A handful of long up-streaks pull the ensemble
   mean above 1.
3. **Sigma sweep at fixed mu** -- the sample mean stays roughly at
   `e^(mu T)` regardless of sigma, but the sample median falls off a
   cliff as sigma grows. The picture of "volatility eating returns".
4. **Regime map** -- a heatmap of `sigma^2 / (2 mu)` over the
   `(mu, sigma)` plane with the regime bands shaded and reference
   assets (and the worked levered-strategy example) marked. Lets you
   eyeball where any candidate strategy sits relative to break-even
   `sigma = sqrt(2 mu)`.
