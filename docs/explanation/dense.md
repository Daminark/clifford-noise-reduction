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

# When CZNR pays off

The paper's Fig. 7 compares CZNR with the direct implementation on dense blocks of `CZ` gates, at a budget of 2 on the gate overhead.
We repeat it, add a budget of 4 and the Forte-like rates of {doc}`families`, and run CliNR on the same blocks, since {doc}`cznr` shows CliNR to be the worse of the two only to first order.
We first describe the experiment, then set our results against Fig. 7 and CZNR against CliNR, and close with the Forte-like rates.

```{code-cell} python
---
tags: [remove-cell]
---
# Loads the results of scripts/dense.py and computes the figures quoted below.
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_inline.backend_inline import set_matplotlib_formats
from myst_nb import glue

from clinr import estimators

set_matplotlib_formats("svg")
root = Path.cwd().parents[1]
for name in ("regret", "payoff", "dense"):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
dense = sys.modules["dense"]
data = dense.payoff.load(root / "results" / "dense.json.gz")
config = data["config"]
rows = dense.outcomes(data)
sizes = data["sizes"]
arms = list(dense.ARMS)
outcome = {(row["regime"], row["budget"], row["arm"], row["n"]): row for row in rows}


def key(*parts):
    return "_".join(str(part) for part in parts).replace("-", "_").replace(".", "_")


def gain(regime, budget, arm, n):
    """The direct implementation's rate over the arm's, nan without a t."""
    row = outcome[regime, budget, arm, n]
    return row["direct"] / row["rate"] if row["t"] is not None else np.nan


def fitted(regime, budget, arm):
    """The sizes at which the rule finds a t."""
    return [n for n in sizes[regime] if outcome[regime, budget, arm, n]["t"] is not None]


glue("pilot_draws", config["pilot_draws"], display=False)
glue("pilot_shots", f"{config['pilot_shots']:,}", display=False)
glue("draws", config["draws"], display=False)
glue("shots", f"{config['shots']:,}", display=False)
glue("direct_shots", f"{config['direct_shots']:,}", display=False)
glue("t_max", config["t_max"], display=False)
for regime in sizes:
    glue(key("largest", regime), sizes[regime][-1], display=False)
for (regime, budget, arm, n), row in outcome.items():
    name = key(regime, int(budget), arm, n)
    glue(f"gain_{name}", float(gain(regime, budget, arm, n)), display=False)
    glue(f"rate_{name}", float(row["rate"]), display=False)
    glue(f"overhead_{name}", float(row["overhead"]), display=False)
for regime in sizes:
    for budget in config["budgets"]:
        for arm in arms:
            ns = fitted(regime, budget, arm)
            name = key(regime, int(budget), arm)
            glue(f"first_{name}", ns[0] if ns else 0, display=False)
            glue(f"last_{name}", ns[-1] if ns else 0, display=False)
            gains = [gain(regime, budget, arm, n) for n in ns]
            glue(f"best_{name}", float(max(gains)) if gains else np.nan, display=False)

# The statistical uncertainty of single estimates.
widths = []
for name, unit in data["units"].items():
    regime, n, arm = name.split("/")
    for record in unit["estimates"].values():
        rate, low, high = estimators.logical_error_rate(dense.estimates_of(record, int(n), arm))
        widths.append((high - low) / 2 / rate)
glue("width_median", float(np.median(widths)), display=False)
glue("width_max", float(np.max(widths)), display=False)

# The chosen pairs whose fresh estimate is over the budget the pilot met.
chosen, over = 0, []
for name, unit in data["units"].items():
    regime, n, arm = name.split("/")
    for budget, t in unit["choice"].items():
        if t is None:
            continue
        chosen += 1
        draws = dense.estimates_of(unit["estimates"][str(t)], int(n), arm)
        overhead, low, _ = estimators.gate_overhead_over_draws(draws)
        if overhead > float(budget):
            over.append((overhead / float(budget) - 1, low <= float(budget)))
glue("chosen", f"{chosen:,}", display=False)
glue("over", len(over), display=False)
glue("over_worst", max(excess for excess, _ in over), display=False)
glue("over_within", sum(within for _, within in over), display=False)
glue("over_beyond", sum(not within for _, within in over), display=False)
glue("direct_errors", min(record["errors"] for record in data["direct"].values()), display=False)

# The least overhead the pilots found for native CZNR at the paper's rates.
for n in sizes["paper"]:
    pilots = data["units"][f"paper/{n}/cznr-native"]["pilots"]
    glue(f"least_overhead_{n}", min(value[1] for value in pilots.values()), display=False)

# CliNR's rate over CZNR's at a budget of 4, wherever both fit.
for regime in sizes:
    for arm in ("uniform", "restricted"):
        ratios = [
            outcome[regime, 4.0, arm, n]["rate"] / outcome[regime, 4.0, "cznr", n]["rate"]
            for n in sizes[regime]
            if n in fitted(regime, 4.0, arm) and n in fitted(regime, 4.0, "cznr")
        ]
        glue(key("ratio", regime, arm, "low"), float(min(ratios)), display=False)
        glue(key("ratio", regime, arm, "high"), float(max(ratios)), display=False)
        glue(key("ratio", regime, arm, "median"), float(np.median(ratios)), display=False)
        glue(key("ratio", regime, arm, "count"), len(ratios), display=False)
```

## The experiment

The data come from `scripts/dense.py`, which writes `results/dense.json.gz` with its settings, the versions that produced it, the circuits, and the counts of every draw.

The circuits are complete graphs, one `CZ` on every pair of $n$ qubits, so $s = n(n - 1)/2$, as in Fig. 7.
The noise regimes are those of {doc}`payoff`: {func}`clinr.noise_models.asymmetric_noise` at $p_2 = 10^{-3}$ and $10^{-4}$, and the Forte-like rates, none with idle noise.
The sizes run $n = 5, 10, 15, \dots$, in steps of 10 beyond $n = 80$, until the direct implementation's bound reaches $0.9$: to $n =$ {glue:text}`largest_paper`, {glue:text}`largest_paper_1e_4` and {glue:text}`largest_forte_like`.

Four schemes run on each block, every one by the paper's rule as in {doc}`payoff`: $r = \lfloor \log_2(s/n) \rfloor$ checks, at least 1, and the smallest $t$ up to {glue:text}`t_max` whose overhead, judged by a pilot over {glue:text}`pilot_draws` draws of {glue:text}`pilot_shots` shots, is within the budget.

- **CZNR**, with uniform checks and $|+\rangle$ prepared as a reset and a Hadamard, as everywhere else in the package.
- **CZNR with native preparation**, $|+\rangle$ prepared in one operation, as the paper counts it.
- **CliNR with uniform checks.**
- **CliNR with restricted checks**, the family of the paper's own CliNR numerics.

A chosen pair is estimated afresh over {glue:text}`draws` draws of {glue:text}`shots` shots, and the direct implementation over {glue:text}`direct_shots` shots.
Judged by their fresh estimates, {glue:text}`over` of the {glue:text}`chosen` chosen pairs are over their budget, by at most {glue:text}`over_worst:.1%`, since a pilot over few draws can fall within the budget at a $t$ just over it, and the first $t$ to fall within is more likely one whose pilot fell low.
The budget lies inside the fresh estimate's 95% interval for {glue:text}`over_within` of them, where the excess is within that estimate's own uncertainty; the interval lies wholly above it for the remaining {glue:text}`over_beyond`.
Every rate is a logical error rate per run of the circuit, and a scheme's *gain* is the direct implementation's rate over its own.
The 95% interval of a single estimate has a median half-width of {glue:text}`width_median:.0%` of the rate, and at most {glue:text}`width_max:.0%`; every direct estimate counts at least {glue:text}`direct_errors` logical errors.
There is one circuit per size, so these intervals are the whole of the statistical uncertainty.
We begin with the paper's own comparison, at a budget of 2.

## Against the paper's Fig. 7

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-dense-fig7
    caption: |-
      Logical error rate of the complete graph on $n$ qubits at a budget of 2 on the gate overhead, directly and under each scheme with the paper's rule.
      A scheme is drawn where the rule finds a $t$ within the budget.
---
plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
gray = "#52514e"
styles = {
    "cznr": {"color": "#1baf7a", "linestyle": "-", "markerfacecolor": "#1baf7a", "label": "CZNR"},
    "cznr-native": {"color": "#1baf7a", "linestyle": ":", "markerfacecolor": "white", "label": "CZNR, native $|+\\rangle$"},
    "uniform": {"color": "#2a78d6", "linestyle": "-", "markerfacecolor": "#2a78d6", "label": "CliNR, uniform"},
    "restricted": {"color": "#eb6834", "linestyle": "-", "markerfacecolor": "#eb6834", "label": "CliNR, restricted"},
}
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


def rates(regime, budget, arm):
    return [outcome[regime, budget, arm, n]["rate"] for n in sizes[regime]]


fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
for ax, regime in zip(axes, ("paper", "paper-1e-4")):
    ns = sizes[regime]
    ax.plot(ns, [outcome[regime, 2.0, "cznr", n]["direct"] for n in ns], color=gray, linestyle="--", linewidth=1.5, label="direct")
    for arm in arms:
        ax.plot(ns, rates(regime, 2.0, arm), linewidth=1.5, marker="o", markersize=3.5, **styles[arm])
    ax.set_yscale("log")
    ax.set_title(titles[regime])
    ax.set_xlabel("qubits $n$")
    style(ax)
axes[0].set_ylabel("logical error rate")
axes[1].legend(frameon=False, fontsize=7)
plt.show()
```

Fig. 7 draws CZNR within a budget of 2 from $n = 10$ to $50$ at $p_2 = 10^{-3}$ and from $n = 10$ to $85$ at $10^{-4}$.
At $10^{-4}$ we find much the same: CZNR fits from $n =$ {glue:text}`first_paper_1e_4_2_cznr` on, and past the end of Fig. 7's range to $n =$ {glue:text}`last_paper_1e_4_2_cznr`, with a gain of up to {glue:text}`best_paper_1e_4_2_cznr:.1f`, at $n = 80$.

At $10^{-3}$ we do not.
With $|+\rangle$ prepared as a reset and a Hadamard, the rule never fits a budget of 2: at $n = 20$, for instance, the overhead at $t = 1$ is {glue:text}`overhead_paper_4_cznr_20:.2f`.
Prepared natively, as the paper counts it, CZNR fits from $n =$ {glue:text}`first_paper_2_cznr_native` to {glue:text}`last_paper_2_cznr_native`, with gains of {glue:text}`gain_paper_2_cznr_native_15:.1f`, {glue:text}`gain_paper_2_cznr_native_20:.1f` and {glue:text}`gain_paper_2_cznr_native_25:.1f`.
Beyond, no $t$ brings the overhead within 2: the least the pilots found is {glue:text}`least_overhead_30:.2f` at $n = 30$ and {glue:text}`least_overhead_50:.2f` at $n = 50$, since with one chunk the acceptance is low, and with several the cost that every chunk pays, whatever its size, adds up.
Our overhead counts every operation of every attempt, restarts included; the paper does not state how it counts the overhead that keeps $n = 50$ within a budget of 2, and we leave the difference unexplained.
A budget of 2 thus leaves CZNR little room at $10^{-3}$, so to compare the schemes we turn to a budget of 4.

## CZNR against CliNR

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-dense-gain
    caption: |-
      Gain of each scheme over the direct implementation on the complete graph on $n$ qubits, at a budget of 4 on the gate overhead.
      A scheme is drawn where the rule finds a $t$ within the budget; above the gray line it pays off.
      The jumps fall where the rule's $r$ or $t$ changes.
---
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5), layout="constrained")
for ax, regime in zip(axes, titles):
    ns = sizes[regime]
    for arm in arms:
        ax.plot(ns, [gain(regime, 4.0, arm, n) for n in ns], linewidth=1.5, marker="o", markersize=3.5, **styles[arm])
    ax.axhline(1, color=gray, linewidth=0.8)
    ax.set_title(titles[regime])
    ax.set_xlabel("qubits $n$")
    style(ax)
axes[0].set_ylabel("gain over direct")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False, fontsize=7)
plt.show()
```

CZNR does better than CliNR with uniform checks wherever both fit, under every regime.
At a budget of 4, CliNR's rate is {glue:text}`ratio_paper_uniform_low:.2f` to {glue:text}`ratio_paper_uniform_high:.2f` times CZNR's at $p_2 = 10^{-3}$, over the {glue:text}`ratio_paper_uniform_count` sizes where both fit, and {glue:text}`ratio_paper_1e_4_uniform_low:.2f` to {glue:text}`ratio_paper_1e_4_uniform_high:.2f` times at $10^{-4}$, with a median of {glue:text}`ratio_paper_1e_4_uniform_median:.2f`.
CliNR also fits over a shorter range, since its overhead is higher: at $p_2 = 10^{-3}$ it stops fitting after $n =$ {glue:text}`last_paper_4_uniform`, while CZNR fits to $n =$ {glue:text}`last_paper_4_cznr`.
So the first-order comparison of {doc}`cznr` holds at these rates too, which reach well past first order.

Native preparation saves about $n$ operations per chunk, and at the same $t$ leaves the logical error rate within the noise: it matters only where the overhead is close to the budget, as at a budget of 2 above.

CliNR with restricted checks is the worst of the four.
On a complete graph its propagated $ZZ$ checks have weight 2 and the others weight $n + 1$, so its checks are cheap and it fits at larger sizes than either uniform scheme.
However, one draw may miss a damaging fault with probability up to $1 - 2/(3n)$ ({doc}`checks-and-bounds`), against $1/2$ for a uniform check, and the rule's few checks do not make up the difference: its gain at a budget of 4 never exceeds {glue:text}`best_paper_4_restricted:.1f` at $p_2 = 10^{-3}$ and {glue:text}`best_paper_1e_4_4_restricted:.1f` at $10^{-4}$, against {glue:text}`best_paper_4_cznr:.1f` and {glue:text}`best_paper_1e_4_4_cznr:.1f` for CZNR.

## At Forte-like rates

No scheme fits a budget of 2 under the Forte-like rates, at any size.

At a budget of 4, CZNR pays off from $n = 10$ to $25$, with gains of {glue:text}`gain_forte_like_4_cznr_10:.2f`, {glue:text}`gain_forte_like_4_cznr_15:.2f`, {glue:text}`gain_forte_like_4_cznr_20:.2f` and {glue:text}`gain_forte_like_4_cznr_25:.2f`, and with native preparation also at $n = 30$, with {glue:text}`gain_forte_like_4_cznr_native_30:.2f`.
CliNR with uniform checks fits only at $n = 15$, where its gain is {glue:text}`gain_forte_like_4_uniform_15:.2f`, and CliNR with restricted checks never gains more than {glue:text}`best_forte_like_4_restricted:.2f`.
On random Clifford circuits at these rates, the best gain of CliNR is about 1.6 ({doc}`payoff`), so a dense block of `CZ` gates is where noise reduction at Forte-like rates has the most to offer, and only CZNR delivers it.

Thus CZNR reproduces Fig. 7 at $p_2 = 10^{-4}$ and only in part at $10^{-3}$, where the budget of 2 sits at the edge of what it needs.
On these circuits it beats CliNR under every regime, as the first-order argument predicts, and it is the one scheme that pays off at the Forte-like rates, by up to {glue:text}`best_forte_like_4_cznr:.1f`.
