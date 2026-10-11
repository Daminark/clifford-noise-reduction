---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# When CliNR pays off

The paper's numerics report where CliNR, with its parameters set by a simple rule, beats the direct implementation: against the number of qubits for random Clifford circuits (Sec. V, Fig. 2), and against the number of qubits and the shape of the circuit (App. D, Fig. 4).
We repeat both under the paper's rates, which checks that this implementation behaves as the paper's does, and then ask the same question under the Forte-like rates of {doc}`families`.
We first describe the experiment, then compare with the paper's two figures and with the scaling of its Theorem 2, and close with the Forte-like rates.

```{code-cell} python
---
tags: [remove-cell]
---
# Loads the results of scripts/payoff.py and computes the figures quoted below.
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_inline.backend_inline import set_matplotlib_formats
from myst_nb import glue

from clinr import bounds, estimators
from clinr.noise_models import NoiseModel

set_matplotlib_formats("svg")
root = Path.cwd().parents[1]
for name in ("regret", "payoff"):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
regret, payoff = sys.modules["regret"], sys.modules["payoff"]
data = payoff.load(root / "results" / "payoff.json.gz")
config = data["config"]
rows = payoff.outcomes(data)
rates = dict((name, noise) for name, noise in config["regimes"])
sizes = data["sizes"]
map_sizes = dict(config["map_sizes"])
alphas = config["alphas"]
families = list(regret.FAMILIES)


def key(*parts):
    return "_".join(str(part) for part in parts).replace("-", "_").replace(".", "_")


def cell(part, regime, budget, family, n, alpha=0.0):
    """Return the outcomes of one cell, in circuit order."""
    return sorted(
        (row for row in rows
         if (row["part"], row["regime"], row["budget"], row["family"], row["n"], row["alpha"])
         == (part, regime, budget, family, n, alpha)),
        key=lambda row: row["circuit"],
    )


def fitted(outcomes):
    return [row for row in outcomes if row["t"] is not None]


def gain(outcomes):
    """Median over the circuits with a t of the direct rate over CliNR's."""
    ratios = [row["direct"] / row["rate"] for row in fitted(outcomes)]
    return float(np.median(ratios)) if ratios else np.nan


def curve(regime, budget, family, name, quantiles=(50,)):
    """Quantiles of a quantity over the circuits of each size, nan where fewer than half have a t."""
    out = []
    for n in sizes[regime]:
        values = np.array([row[name] for row in cell("curves", regime, budget, family, n)], dtype=float)
        enough = np.sum(~np.isnan(values)) >= len(values) / 2
        out.append(np.nanpercentile(values, quantiles) if enough else np.full(len(quantiles), np.nan))
    return np.array(out).T


glue("pilot_draws", config["pilot_draws"], display=False)
glue("pilot_shots", f"{config['pilot_shots']:,}", display=False)
glue("draws", config["draws"], display=False)
glue("shots", f"{config['shots']:,}", display=False)
glue("direct_shots", f"{config['direct_shots']:,}", display=False)
glue("t_max", config["t_max"], display=False)
glue("circuits", config["curve_circuits"], display=False)
for regime in rates:
    glue(key("largest", regime), sizes[regime][-1], display=False)

for regime in rates:
    for budget in config["curve_budgets"]:
        for family in families:
            for n in sizes[regime]:
                outcomes = cell("curves", regime, budget, family, n)
                name = key(regime, int(budget), family, n)
                glue(f"gain_{name}", gain(outcomes), display=False)
                glue(f"fits_{name}", len(fitted(outcomes)), display=False)
for n in sizes["paper"]:
    glue(key("s", n), int(np.median([row["s"] for row in cell("curves", "paper", 4.0, "uniform", n)])), display=False)
for regime in rates:
    for n in sizes[regime]:
        glue(key("direct", regime, n), float(np.median([row["direct"] for row in cell("curves", regime, 4.0, "uniform", n)])), display=False)

# The statistical uncertainty of single estimates.
widths = []
for name, unit in data["units"].items():
    part, regime, n, alpha, c, family = name.split("/")
    for record in unit["estimates"].values():
        rate, low, high = estimators.logical_error_rate(regret.estimates_of(record, int(n), regret.FAMILIES[family]))
        widths.append((high - low) / 2 / rate)
glue("width_median", float(np.median(widths)), display=False)
glue("width_max", float(np.max(widths)), display=False)
for part in ("curves", "maps"):
    counts = [record["errors"] for case, record in data["direct"].items() if case.startswith(part)]
    glue(f"direct_errors_{part}", min(counts), display=False)
few = [record for case, record in data["direct"].items() if record["errors"] < min(
    record["errors"] for case, record in data["direct"].items() if case.startswith("curves"))]
glue("few_rate", max(record["errors"] / record["shots"] for record in few), display=False)
```

## The experiment

The data come from `scripts/payoff.py`, which writes `results/payoff.json.gz` with its settings, the versions that produced it, the circuits, and the counts of every draw.

There are three noise regimes, none with idle noise.
Two are the paper's: {func}`clinr.noise_models.asymmetric_noise` at $10^{-3}$ and at $10^{-4}$, i.e. two-qubit gates fault with probability $p_2$ and every other operation with $p_2/10$, for $p_2 = 10^{-3}$ and $10^{-4}$.
The third is the Forte-like regime of {doc}`families`: $p_1 = 2 \times 10^{-4}$, $p_2 = 4 \times 10^{-3}$, $p_m = 4.8 \times 10^{-3}$ and $p_{\mathrm{prep}} = 2 \times 10^{-4}$.

CliNR's parameters follow the paper's rule (Fig. 2): $r = \lfloor \log_2(s/n) \rfloor$ checks, at least 1 since the paper takes $r \ge 1$, and the smallest $t$, up to {glue:text}`t_max`, whose gate overhead is within a budget of 2 or of 4.
The overhead of each $t$ is judged by a pilot estimate over {glue:text}`pilot_draws` draws of the checks of {glue:text}`pilot_shots` shots each, and the chosen pair is then estimated afresh over {glue:text}`draws` draws of {glue:text}`shots` shots.
The direct implementation is estimated with {glue:text}`direct_shots` shots.
Both families of checks are run on the same circuits.

The experiment has two parts.
The *curves* take {glue:text}`circuits` random Clifford circuits at each size from {func}`clinr.circuits.random_clifford`, as the paper does, at $n = 5, 10, 15, \dots$, in steps of 10 beyond $n = 80$, until the direct implementation's bound on the median circuit, which is nearly exact ({doc}`regret`), reaches $0.9$.
This takes the sizes to $n =$ {glue:text}`largest_paper`, {glue:text}`largest_paper_1e_4` and {glue:text}`largest_forte_like` in the three regimes.
The *maps* take {glue:text}`circuits` circuits of $s = n^\alpha$ gates drawn uniformly from $\{H, S, \mathrm{CX}\}$, as in the paper's App. D, from {func}`clinr.circuits.random_gate_sequence`, for $\alpha = 1.2, 1.3, \dots, 2.0$ and $n = 5, 10, \dots, 55$ at $p_2 = 10^{-4}$ and $n = 5, \dots, 30$ at the Forte-like rates.

Every rate is a logical error rate per run of the circuit.
We report medians over the circuits of a size, and measure CliNR's *gain* on a circuit by the ratio of the direct implementation's rate to CliNR's, so a gain above 1 means CliNR pays off; the gain of a size is the median over the circuits on which the rule finds a $t$.
The 95% interval of a single CliNR estimate has a median half-width of {glue:text}`width_median:.0%` of the rate, and at most {glue:text}`width_max:.0%`, at the largest sizes with restricted checks, where the draws of the checks differ most.
Every direct estimate on the curves counts at least {glue:text}`direct_errors_curves` logical errors; on the maps the smallest circuits count as few as {glue:text}`direct_errors_maps`, at rates of at most {glue:text}`few_rate:.3f`, far below the differences the maps resolve.

## Against the paper's Fig. 2

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-payoff-curves
    caption: |-
      Logical error rate of random Clifford circuits, directly and with CliNR under the paper's rule, for each family of checks and a budget of 4 (dots) or 2 (rings) on the gate overhead; a dot inside a ring is a size where the two budgets give the same choice.
      Lines join the medians over the circuits of each size, drawn where the rule finds a $t$ on at least half of them; bands span the interquartile range at a budget of 4.
---
plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
colors = {"uniform": "#2a78d6", "restricted": "#eb6834"}
gray = "#52514e"
titles = {
    "paper": r"Paper's rates, $p_2 = 10^{-3}$",
    "paper-1e-4": r"Paper's rates, $p_2 = 10^{-4}$",
    "forte-like": "Forte-like rates",
}


def style(ax):
    ax.grid(color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2), layout="constrained")
for ax, regime in zip(axes, titles):
    ns = sizes[regime]
    ax.plot(ns, curve(regime, 4.0, "uniform", "direct")[0], color=gray, linestyle="--", linewidth=1.5, label="direct")
    for family in families:
        low, middle, high = curve(regime, 4.0, family, "rate", (25, 50, 75))
        ax.fill_between(ns, low, high, color=colors[family], alpha=0.15, linewidth=0)
        ax.plot(ns, middle, color=colors[family], linewidth=1.5, marker="o", markersize=3.5, label=f"{family}, budget 4")
        ax.plot(
            ns, curve(regime, 2.0, family, "rate")[0], color=colors[family], linestyle=":", linewidth=1.5,
            marker="o", markersize=6.5, markerfacecolor="none", label=f"{family}, budget 2",
        )
    ax.set_yscale("log")
    ax.set_title(titles[regime])
    ax.set_xlabel("qubits $n$")
    style(ax)
axes[0].set_ylabel("logical error rate")
axes[0].legend(frameon=False, fontsize=7)
plt.show()
```

The paper reports a gain of 2 at $n = 25$ for $p_2 = 10^{-3}$, and of 4 at $n = 60$ for $p_2 = 10^{-4}$.
Its circuits have $s \approx n^2$ gates, while Stim's synthesis gives about $1.5 n^2$ ({doc}`sharpness`), so we compare at matched size: our $n = 20$ has a median of {glue:text}`s_20` gates against the paper's 625, and our $n = 50$ has {glue:text}`s_50` against its 3,600.
At $n = 20$ and $p_2 = 10^{-3}$, the rule fits a budget of 2 with restricted checks, the paper's family, on {glue:text}`fits_paper_2_restricted_20` of the {glue:text}`circuits` circuits, with a gain of {glue:text}`gain_paper_2_restricted_20:.1f`.
At $n = 50$ and $p_2 = 10^{-4}$, the gain is {glue:text}`gain_paper_1e_4_2_restricted_50:.1f` with restricted checks and {glue:text}`gain_paper_1e_4_2_uniform_50:.1f` with uniform ones, at either budget.
The curves thus have the paper's shape, and our gains are larger than the paper's, by factors of about 1.5 and 2.

One plausible cause is idle noise, which the paper's numerics include at rate $p_1$ and ours omit.
In a schedule that runs one operation at a time, the input block idles while each resource is prepared, where no check can see its faults, so idle noise reaches CliNR's output much as it reaches the direct implementation's, and the gain shrinks.

Matching the size also settles why the rule fits a budget of 2 at the paper's $n = 25$ and, with restricted checks, on only {glue:text}`fits_paper_2_restricted_25` of our circuits at $n = 25$, the question left open in {doc}`regret`.
Our $n = 25$ has {glue:text}`s_25` gates, so each chunk is half as large again as the paper's and carries one more check, five rather than four, which lowers its acceptance and raises its cost.

The two budgets coincide wherever the smaller one is met with one chunk, since the rule fixes $r$ and takes the smallest $t$ that fits.
The larger budget changes the choice only where the smaller one needs more chunks, and then buys fewer teleportations: at $p_2 = 10^{-4}$ and $n = 90$, restricted checks gain {glue:text}`gain_paper_1e_4_2_restricted_90:.1f` with two chunks under a budget of 2, and {glue:text}`gain_paper_1e_4_4_restricted_90:.1f` with one under a budget of 4.

Matching sizes calibrates the curves at each rate; taken together, the two rates also test how far each implementation reaches as $p$ falls, which we now turn to.

## The scaling of Theorem 2

```{code-cell} python
---
tags: [remove-cell]
---
# Reach: the size at which the median rate first reaches a threshold, by linear interpolation in n.
def reach(ns, middle, threshold):
    above = np.nonzero(middle >= threshold)[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan
    i = above[0]
    return ns[i - 1] + (threshold - middle[i - 1]) * (ns[i] - ns[i - 1]) / (middle[i] - middle[i - 1])


def table(regime, family, name):
    return np.array([[row[name] for row in cell("curves", regime, 4.0, family, n)] for n in sizes[regime]], dtype=float)


bootstrap = np.random.default_rng(20261010)
for label, family, name, threshold in (
    ("direct", "uniform", "direct", 0.1),
    ("uniform", "uniform", "rate", 0.05),
    ("restricted", "restricted", "rate", 0.05),
    ("restricted", "restricted", "rate", 0.2),
):
    tables = {regime: table(regime, family, name) for regime in ("paper", "paper-1e-4")}
    reaches = {
        regime: reach(np.array(sizes[regime]), np.median(tables[regime], axis=1), threshold)
        for regime in tables
    }
    ratios = []
    for _ in range(1000):
        picks = bootstrap.integers(0, config["curve_circuits"], config["curve_circuits"])
        resampled = [
            reach(np.array(sizes[regime]), np.median(tables[regime][:, picks], axis=1), threshold)
            for regime in ("paper", "paper-1e-4")
        ]
        ratios.append(resampled[1] / resampled[0])
    name = key("reach", label, threshold)
    glue(f"{name}_3", float(reaches["paper"]), display=False)
    glue(f"{name}_4", float(reaches["paper-1e-4"]), display=False)
    glue(f"{name}_ratio", float(reaches["paper-1e-4"] / reaches["paper"]), display=False)
    low, high = np.nanpercentile(ratios, [2.5, 97.5])
    glue(f"{name}_low", float(low), display=False)
    glue(f"{name}_high", float(high), display=False)

for family in families:
    per_np = []
    for regime in ("paper", "paper-1e-4"):
        middle, t = curve(regime, 4.0, family, "rate")[0], curve(regime, 4.0, family, "t")[0]
        per_np.extend(middle[t == 1] / (np.array(sizes[regime])[t == 1] * rates[regime]["p2"]))
    glue(f"per_np_low_{family}", float(min(per_np)), display=False)
    glue(f"per_np_high_{family}", float(max(per_np)), display=False)
```

Theorem 2 and its Corollary 2 compare how far the two implementations reach as $p$ falls: for circuits of $s \lesssim n^2$ gates, the direct implementation keeps a vanishing logical error rate up to $n = O(p^{-1/2})$ qubits, and CliNR up to $n = O(p^{-2/3})$.
Dividing the rates by 10 should therefore stretch the direct implementation's reach by $\sqrt{10} \approx 3.2$ and CliNR's by $10^{2/3} \approx 4.6$.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-payoff-scaling
    caption: |-
      Left: the direct implementation's logical error rate against $n \sqrt{p_2}$, under the paper's two rates.
      Right: CliNR's logical error rate at a budget of 4, divided by $n p_2$, against $n p_2$, for each family of checks; filled markers where the median $t$ is 1, open where it is larger.
      Lines join the medians over the circuits of each size; bands span the interquartile range.
---
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
markers = {"paper": "o", "paper-1e-4": "s"}
for regime in ("paper", "paper-1e-4"):
    p2 = rates[regime]["p2"]
    ns = np.array(sizes[regime])
    low, middle, high = curve(regime, 4.0, "uniform", "direct", (25, 50, 75))
    axes[0].fill_between(ns * np.sqrt(p2), low, high, color=gray, alpha=0.15, linewidth=0)
    axes[0].plot(
        ns * np.sqrt(p2), middle, color=gray, linewidth=1.2, marker=markers[regime], markersize=3.5,
        markerfacecolor=gray if regime == "paper" else "white", label=titles[regime],
    )
    for family in families:
        x = ns * p2
        low, middle, high = curve(regime, 4.0, family, "rate", (25, 50, 75)) / x
        t = curve(regime, 4.0, family, "t")[0]
        axes[1].fill_between(x, low, high, color=colors[family], alpha=0.15, linewidth=0)
        axes[1].plot(x, middle, color=colors[family], linewidth=1.2, linestyle="-" if regime == "paper" else "--")
        for one, face in ((t == 1, colors[family]), (t > 1, "white")):
            axes[1].scatter(x[one], middle[one], facecolor=face, edgecolor=colors[family], marker=markers[regime], s=14, zorder=3)
axes[0].set_xlabel(r"$n \sqrt{p_2}$")
axes[0].set_ylabel("direct logical error rate")
axes[0].legend(frameon=False, fontsize=7)
axes[1].set_xscale("log")
axes[1].set_xlabel(r"$n p_2$")
axes[1].set_ylabel(r"CliNR logical error rate / $n p_2$")
for family in families:
    axes[1].plot([], [], color=colors[family], linewidth=1.2, label=family)
axes[1].plot([], [], color=gray, linewidth=1.2, linestyle="-", label=r"$p_2 = 10^{-3}$")
axes[1].plot([], [], color=gray, linewidth=1.2, linestyle="--", label=r"$p_2 = 10^{-4}$")
axes[1].legend(frameon=False, fontsize=7)
for ax in axes:
    style(ax)
plt.show()
```

The direct implementation scales as predicted.
Its rate is nearly $1 - e^{-s \bar p}$, with $\bar p$ the mean fault probability of a gate, so with $s \propto n^2$ it depends on $n$ and $p$ through $n \sqrt{p}$ alone, and the two curves of the left panel coincide.
Interpolating linearly between sizes, the rate reaches $0.1$ at $n =$ {glue:text}`reach_direct_0_1_3:.1f` for $p_2 = 10^{-3}$ and $n =$ {glue:text}`reach_direct_0_1_4:.1f` for $p_2 = 10^{-4}$, a ratio of {glue:text}`reach_direct_0_1_ratio:.2f` (95% bootstrap interval over circuits {glue:text}`reach_direct_0_1_low:.2f` to {glue:text}`reach_direct_0_1_high:.2f`).

CliNR's scaling depends on whether one chunk fits the budget.
With $r = \lfloor \log_2(s/n) \rfloor$ checks, $s 2^{-r}$ lies between $n$ and $2n$, so in Theorem 1 the faults of a chunk that escape its checks, those of its checks, and those of its teleportation each contribute a term of order $n p$.
With one chunk the rate is therefore proportional to $n p$, up to the slow growth of $r$ and the fall of the acceptance with $s p$, and the reach stretches by nearly 10, not 4.6.
The right panel shows this: while $t = 1$, the rate divided by $n p_2$ stays between {glue:text}`per_np_low_uniform:.1f` and {glue:text}`per_np_high_uniform:.1f` for uniform checks and between {glue:text}`per_np_low_restricted:.1f` and {glue:text}`per_np_high_restricted:.1f` for restricted ones, under both rates.
Once $s p$ is large, a chunk is accepted often enough for the budget only if it has $O(1/p)$ gates, so $t$ grows in proportion to $s p$, and the $t$ teleportations contribute $t n p \propto s n p^2 \propto n^3 p^2$.
The reach then scales as $p^{-2/3}$, the exponent of Theorem 2, whose own choice $t = \lfloor \sqrt{s/n} \rfloor$ gives the same.

The measured reaches follow the two regimes.
At a threshold of $0.05$, which both rates reach with one chunk, the ratio is {glue:text}`reach_uniform_0_05_ratio:.1f` for uniform checks ({glue:text}`reach_uniform_0_05_low:.1f` to {glue:text}`reach_uniform_0_05_high:.1f`) and {glue:text}`reach_restricted_0_05_ratio:.1f` for restricted ones ({glue:text}`reach_restricted_0_05_low:.1f` to {glue:text}`reach_restricted_0_05_high:.1f`).
At $0.2$, which restricted checks reach at $n =$ {glue:text}`reach_restricted_0_2_4:.0f` for $p_2 = 10^{-4}$ with two chunks, the ratio is {glue:text}`reach_restricted_0_2_ratio:.1f` ({glue:text}`reach_restricted_0_2_low:.1f` to {glue:text}`reach_restricted_0_2_high:.1f`), near the asymptotic 4.6.
Uniform checks do not reach $0.2$ at $p_2 = 10^{-4}$ within the sizes run.

## Against the paper's Fig. 4

```{code-cell} python
---
tags: [remove-cell]
---
# The maps: the median over circuits of p_direct - p_CliNR in each cell, where at least half have a t.
panels = (("paper-1e-4", 2.0), ("paper-1e-4", 4.0), ("forte-like", 4.0))
maps = {}
for regime, budget in panels:
    for family in families:
        grid = np.full((len(map_sizes[regime]), len(alphas)), np.nan)
        for i, n in enumerate(map_sizes[regime]):
            for j, alpha in enumerate(alphas):
                outcomes = cell("maps", regime, budget, family, n, alpha)
                if len(fitted(outcomes)) >= len(outcomes) / 2:
                    grid[i, j] = np.median([row["direct"] - row["rate"] for row in fitted(outcomes)])
        maps[regime, budget, family] = grid

for family in families:
    grid = maps["paper-1e-4", 4.0, family]
    for i, n in enumerate(map_sizes["paper-1e-4"]):
        positive = np.nonzero(grid[i] > 0)[0]
        glue(key("boundary", family, n), alphas[positive[0]] if len(positive) else np.nan, display=False)
glue("map_best", float(np.nanmax([maps["paper-1e-4", 4.0, family] for family in families])), display=False)
glue("map_direct_max", max(row["direct"] for row in rows if (row["part"], row["regime"]) == ("maps", "paper-1e-4")), display=False)
fits_two = np.concatenate([~np.isnan(maps["paper-1e-4", 2.0, family]) for family in families])
glue("two_lowest", alphas[int(np.nonzero(fits_two.any(axis=0))[0][0])], display=False)
glue("two_restricted_worst", float(np.nanmin(maps["paper-1e-4", 2.0, "restricted"])), display=False)

forte = {family: maps["forte-like", 4.0, family] for family in families}
glue("forte_cells", int(sum((~np.isnan(grid)).sum() for grid in forte.values())), display=False)
glue("forte_positive", int(sum((grid > 0).sum() for grid in forte.values())), display=False)
glue("forte_best", float(np.nanmax(forte["restricted"])), display=False)
glue("forte_worst", float(np.nanmin(forte["restricted"])), display=False)
```

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-payoff-maps
    caption: |-
      Median over circuits of $p^{\mathrm{direct}} - p^{\mathrm{CliNR}}$ for circuits of $n^\alpha$ gates drawn from $\{H, S, \mathrm{CX}\}$, under the paper's rule and a budget of 2 or 4 on the gate overhead, for each family of checks; blue where CliNR is better.
      Blank cells are those where the rule finds a $t$ on fewer than half the circuits.
      The rule never fits a budget of 2 at the Forte-like rates.
---
diverging = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])
fig, axes = plt.subplots(3, 2, figsize=(7.5, 8.0), layout="constrained")
for row, (regime, budget) in enumerate(panels):
    limit = max(np.nanmax(np.abs(maps[regime, budget, family])) for family in families)
    for column, family in enumerate(families):
        ax = axes[row, column]
        image = ax.pcolormesh(
            alphas, map_sizes[regime], maps[regime, budget, family], cmap=diverging,
            vmin=-limit, vmax=limit, shading="nearest", edgecolors="white", linewidth=1,
        )
        ax.set_title(f"{titles[regime]}, budget {budget:.0f}, {family}", fontsize=8)
        ax.set_xticks(alphas[::2])
        ax.set_yticks(map_sizes[regime])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row == len(panels) - 1:
            ax.set_xlabel(r"shape $\alpha$")
        if column == 0:
            ax.set_ylabel("qubits $n$")
    fig.colorbar(image, ax=axes[row, :], label=r"$p^{\mathrm{direct}} - p^{\mathrm{CliNR}}$", shrink=0.9)
plt.show()
```

The paper's Fig. 4 maps the difference of the two rates at $p_2 = 10^{-4}$, without stating its budget, and finds the direct implementation competitive at small $n$ and $\alpha$ and CliNR favored as either grows.
At a budget of 4 we find the same: the smallest $\alpha$ at which CliNR is better falls from {glue:text}`boundary_uniform_10:.1f` at $n = 10$ to {glue:text}`boundary_uniform_55:.1f` at $n = 55$ with uniform checks, and from {glue:text}`boundary_restricted_15:.1f` at $n = 15$ to {glue:text}`boundary_restricted_55:.1f` at $n = 55$ with restricted ones.
At a budget of 2 the rule fits only for $\alpha$ of at least {glue:text}`two_lowest:.1f`, and at small $n$ only near $\alpha = 2$, so the edge of the map is set by the budget rather than by the sign of the difference; within it uniform checks are better everywhere, and restricted checks lose at most {glue:text}`two_restricted_worst:.3f`, at the smallest $\alpha$ that fits.

The magnitudes differ.
Our largest difference is {glue:text}`map_best:.2f`, while the paper's color scale runs to $0.30$; but no circuit of ours has a direct rate above {glue:text}`map_direct_max:.2f`, so no difference here could reach it.
The paper's model is thus the noisier one, as its idle noise would make it.

## At Forte-like rates

Under the Forte-like rates the rule never fits a budget of 2, on either part of the experiment.

At a budget of 4, CliNR pays off on random Clifford circuits by more than a few percent only at $n = 10$ and $15$.
At $n = 10$ the gain is {glue:text}`gain_forte_like_4_uniform_10:.2f` with uniform checks and {glue:text}`gain_forte_like_4_restricted_10:.2f` with restricted ones; at $n = 15$ it is {glue:text}`gain_forte_like_4_restricted_15:.2f` with restricted checks, and {glue:text}`gain_forte_like_4_uniform_15:.2f` with uniform ones on the {glue:text}`fits_forte_like_4_uniform_15` circuits where they fit.
At $n = 5$ CliNR loses, with gains of {glue:text}`gain_forte_like_4_uniform_5:.2f` and {glue:text}`gain_forte_like_4_restricted_5:.2f`, and from $n = 20$ uniform checks no longer fit while restricted ones gain at most {glue:text}`gain_forte_like_4_restricted_20:.2f`.
The maps show the same at smaller sizes: of the {glue:text}`forte_cells` cells where the rule fits, CliNR is better in {glue:text}`forte_positive`, all with restricted checks at $\alpha$ of 1.9 or 2.0 and by at most {glue:text}`forte_best:.2f`, and elsewhere worse by as much as {glue:text}`forte_worst:.2f`.
Since a synthesized Clifford has about $1.5 n^2$ gates, the curves sit just above the top row of the map, and together they say that CliNR pays off at these rates only for circuits of about $n^2$ gates or more, at around 10 to 15 qubits.

```{code-cell} python
---
tags: [remove-cell]
---
# The teleportation after a chunk's last check, which no check protects.
for regime in rates:
    text = data["circuits"][payoff.circuit_key(0, 10, 0.0, 0)]
    terms = bounds.plog_terms(stim.Circuit(text), 10, 1, 1, NoiseModel(**rates[regime]))
    teleport = terms[0].teleport
    glue(key("teleport", regime), float(teleport), display=False)
    glue(key("teleport_share", regime), teleport / float(np.median([row["direct"] for row in cell("curves", regime, 4.0, "uniform", 10)])), display=False)
glue("teleport_measurement", 1 - (1 - rates["forte-like"]["p_m"]) ** (2 * 10), display=False)
```

The cost that stops CliNR here is the teleportation.
After a chunk's last check come $n$ CX, $n$ Hadamards and $2n$ measurements that no check protects.
At $n = 10$, the probability that one of them faults is {glue:text}`teleport_forte_like:.3f` under the Forte-like rates, {glue:text}`teleport_share_forte_like:.0%` of the direct implementation's rate, against {glue:text}`teleport_paper:.3f`, or {glue:text}`teleport_share_paper:.0%`, under the paper's rates at $p_2 = 10^{-3}$.
The measurements alone fault with probability {glue:text}`teleport_measurement:.3f`, since $p_m$ exceeds $p_2$ in this regime, while the paper's rates put it at $p_2/10$.
This leans on the split of state preparation and measurement error assumed in {doc}`families`, which places most of it on measurement; error moved to preparation would fall where the checks can catch it, and CliNR would do better.

Thus, under the paper's rates, this implementation reproduces the shape of the paper's two figures and the scaling of its Theorem 2, with gains larger than the paper reports.
Under the Forte-like rates, CliNR with the paper's rule pays off only for dense circuits of 10 to 15 qubits, at a budget of 4, and by a factor of at most about 1.6.
