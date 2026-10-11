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

# Uniform or restricted checks

The paper bounds CliNR with checks drawn uniformly from the stabilizer group of the resource state (Theorem 1), and runs its numerics with checks drawn from the propagated Bell stabilizers (Sec. V), the two families of {doc}`checks-and-bounds`.
It never runs both, so which family to use is left open.
We answer this by measurement: for each family, noise regime, budget on the gate overhead and circuit size, we find the implementation with the least measured logical error rate, and compare.
We begin with the experiment, then compare the families, trace the comparison to the cost and the strength of a single check, and close with what the bounds would have chosen instead.

```{code-cell} python
---
tags: [remove-cell]
---
# Loads the results of scripts/regret.py and computes the figures quoted below.
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
from matplotlib_inline.backend_inline import set_matplotlib_formats
from myst_nb import glue

from clinr import accounting

set_matplotlib_formats("svg")
root = Path.cwd().parents[1]
spec = importlib.util.spec_from_file_location("regret", root / "scripts" / "regret.py")
regret = importlib.util.module_from_spec(spec)
sys.modules["regret"] = regret
spec.loader.exec_module(regret)
data = json.loads((root / "results" / "regret.json").read_text(encoding="utf-8"))
config = data["config"]
rows = regret.outcomes(data)
regimes = [name for name, _ in config["regimes"]]
sizes = config["sizes"]
ceilings = config["ceilings"]
families = list(regret.FAMILIES)
circuits_per_size = config["circuits_per_size"]


def values(name, regime, ceiling, family, n):
    """Return a quantity over the circuits of one cell, in circuit order."""
    cell = sorted(
        (row for row in rows
         if (row["regime"], row["ceiling"], row["family"], row["n"])
         == (regime, ceiling, family, n)),
        key=lambda row: row["circuit"],
    )
    return np.array([row[name] for row in cell])


def median(name, regime, ceiling, family, n):
    return float(np.nanmedian(values(name, regime, ceiling, family, n)))


def wins(regime, ceiling, n):
    """Return the circuits on which each family's best pair is the lower."""
    uniform = values("best_pair", regime, ceiling, "uniform", n)
    restricted = values("best_pair", regime, ceiling, "restricted", n)
    both = ~np.isnan(uniform) & ~np.isnan(restricted)
    return int((uniform[both] < restricted[both]).sum()), int((restricted[both] < uniform[both]).sum())


for n in sizes:
    counts = [sum(accounting.gate_counts(stim.Circuit(text))) for text in data["circuits"][str(n)]]
    glue(f"gates_{n}", f"{min(counts)} to {max(counts)}", display=False)
glue("circuits_per_size", circuits_per_size, display=False)
glue("draws", config["draws"], display=False)
glue("shots", f"{config['shots']:,}", display=False)
glue("reestimate_draws", config["reestimate_draws"], display=False)
glue("direct_shots", f"{config['direct_shots']:,}", display=False)
glue("t_max", max(config["ts"]), display=False)
glue("r_max", max(config["rs"]), display=False)

for regime in regimes:
    for ceiling in ceilings:
        for n in sizes:
            key = f"{regime}_{ceiling:.0f}_{n}".replace("-", "_")
            glue(f"direct_{key}", median("direct", regime, ceiling, "uniform", n), display=False)
            for family in families:
                glue(f"{family}_{key}", median("best_pair", regime, ceiling, family, n), display=False)
            uniform_wins, restricted_wins = wins(regime, ceiling, n)
            glue(f"uniform_wins_{key}", uniform_wins, display=False)
            glue(f"restricted_wins_{key}", restricted_wins, display=False)

best_pairs = [row["best_pair_pick"] for row in rows if isinstance(row["best_pair_pick"], list)]
glue("one_chunk", sum(pick[0] == 1 for pick in best_pairs) / len(best_pairs), display=False)
for family in families:
    picks = values("best_pair_pick", "paper", 4.0, family, 25)
    glue(f"r_{family}", int(Counter(pick[1] for pick in picks).most_common(1)[0][0]), display=False)
    weights = [
        sum(weight) / sum(checks)
        for key, record in data["grid"].items()
        if key.startswith(f"paper/25/") and f"/{family}/" in key and sum(record["chunk_checks"]) > 0
        for weight, checks in [(w, record["chunk_checks"]) for w in record["chunk_weight"]]
    ]
    glue(f"weight_{family}", float(np.mean(weights)), display=False)
bound_direct = Counter(
    (row["family"], row["bound_pick"] == regret.DIRECT) for row in rows
)
glue("restricted_bound_direct", bound_direct["restricted", True], display=False)
cells = [
    wins(regime, ceiling, n)
    for regime in regimes for ceiling in ceilings for n in sizes
    if not np.isnan(np.concatenate([values("best_pair", regime, ceiling, family, n) for family in families])).any()
]
glue("full_cells", len(cells), display=False)
glue("unanimous", sum(circuits_per_size in pair for pair in cells), display=False)
gains = [
    1 - median("best_pair", "forte-like", 2.0, family, n) / median("direct", "forte-like", 2.0, family, n)
    for family in families for n in sizes
]
glue("forte_gain", max(gains), display=False)
glue("restricted_cases", bound_direct["restricted", True] + bound_direct["restricted", False], display=False)
```

## The experiment

The data come from `scripts/regret.py`, which writes `results/regret.json` with its settings, the versions that produced it, the circuits, and the counts of every draw.
It takes {glue:text}`circuits_per_size` random Clifford circuits at each of $n = 3, 8, 15, 25$ qubits from {func}`clinr.circuits.random_cliffords`, as the paper does, of {glue:text}`gates_3`, {glue:text}`gates_8`, {glue:text}`gates_15` and {glue:text}`gates_25` gates respectively.

There are two noise regimes.
The first is the paper's: {func}`clinr.noise_models.asymmetric_noise` at $10^{-3}$, so two-qubit gates fault with probability $10^{-3}$ and everything else with $10^{-4}$.
The second, which we call Forte-like, takes the rates IonQ publishes for its Forte Enterprise system: $2 \times 10^{-4}$ for single-qubit gates, $4 \times 10^{-3}$ for two-qubit gates, and $5 \times 10^{-3}$ for state preparation and measurement together, which we place mostly on measurement ($4.8 \times 10^{-3}$, with $2 \times 10^{-4}$ on preparation), where published error budgets for trapped ions put most of it.

For every pair $(t, r)$ with $t \le$ {glue:text}`t_max` and $r$ up to {glue:text}`r_max`, and each family, the script estimates the logical error rate and the gate overhead over {glue:text}`draws` draws of the checks, of {glue:text}`shots` shots each.
Under a budget of 2 or 4 on the measured overhead, the best pair is the one with the least estimated rate; since the least of many noisy estimates is biased low, the rate we report for it comes from {glue:text}`reestimate_draws` fresh draws.
The direct implementation is estimated twice with {glue:text}`direct_shots` shots, and we report the second estimate.
How the estimates and their intervals are formed is the subject of {doc}`estimation`.

## Which family wins

The figure shows the least measured logical error rate for each family, against the direct implementation.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-families
    caption: |-
      Least measured logical error rate of CliNR under a budget on the gate overhead, for each family of checks, against the direct implementation.
      Lines join the medians over the circuits of each size; bands span their interquartile range.
---
plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
colors = {"uniform": "#2a78d6", "restricted": "#eb6834"}
titles = {"paper": "Paper's rates", "forte-like": "Forte-like rates"}
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True, layout="constrained")
for i, regime in enumerate(regimes):
    for j, ceiling in enumerate(ceilings):
        ax = axes[i, j]
        direct = [median("direct", regime, ceiling, "uniform", n) for n in sizes]
        ax.plot(sizes, direct, color="#52514e", linestyle="--", linewidth=1.5, label="direct")
        for family in families:
            table = np.array([values("best_pair", regime, ceiling, family, n) for n in sizes])
            middle = np.nanmedian(table, axis=1)
            low, high = np.nanpercentile(table, [25, 75], axis=1)
            ax.fill_between(sizes, low, high, color=colors[family], alpha=0.18, linewidth=0)
            ax.plot(sizes, middle, color=colors[family], linewidth=1.5, marker="o", markersize=4, label=family)
        ax.set_yscale("log")
        ax.set_title(f"{titles[regime]}, overhead at most {ceiling:.0f}")
        ax.grid(color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i == 1:
            ax.set_xlabel("qubits $n$")
        if j == 0:
            ax.set_ylabel("logical error rate")
        ax.set_xticks(sizes)
axes[0, 0].legend(frameon=False)
plt.show()
```

Under the paper's rates the answer depends on the budget.
At an overhead of 4, uniform checks win at every size and on every circuit: at $n = 25$ the best uniform pair reaches {glue:text}`uniform_paper_4_25:.3f` and the best restricted one {glue:text}`restricted_paper_4_25:.3f`, against {glue:text}`direct_paper_4_25:.3f` for the direct implementation.
At an overhead of 2, restricted checks win from $n = 15$ on: {glue:text}`restricted_paper_2_25:.3f` against {glue:text}`uniform_paper_2_25:.3f` at $n = 25$, lower on {glue:text}`restricted_wins_paper_2_25` of the {glue:text}`circuits_per_size` circuits.
At $n = 8$ the two are even, uniform checks lower on {glue:text}`uniform_wins_paper_2_8` circuits and restricted ones on {glue:text}`restricted_wins_paper_2_8`, and at $n = 3$ only some circuits admit any pair within the budget, none of them better than direct.

The Forte-like regime shows the same crossover at a different place.
At an overhead of 4, uniform checks win at $n = 3$ and $8$ and restricted ones at $n = 15$ and $25$, on every circuit in each case; at $n = 15$, for instance, {glue:text}`restricted_forte_like_4_15:.3f` against {glue:text}`uniform_forte_like_4_15:.3f`, with the direct implementation at {glue:text}`direct_forte_like_4_15:.3f`.
At an overhead of 2, no pair improves on the direct implementation by more than {glue:text}`forte_gain:.0%` of its rate.

Two features of the comparison make it robust.
Both families run on the same circuits, so the comparison is paired, and in {glue:text}`unanimous` of the {glue:text}`full_cells` cells where every circuit admits a pair for both families, one family is lower on every circuit.
The winner is also found with one chunk: {glue:text}`one_chunk:.0%` of the best pairs have $t = 1$, so within this range of sizes, splitting the circuit costs more in teleportations than it recovers in acceptance.

## Why the budget decides

The comparison is a trade between what a check costs and what it catches, and the budget sets the exchange rate.

A uniform check catches a damaging fault with probability exactly $1/2$, while a restricted check may catch it with probability as low as $2/(3n)$ ({doc}`checks-and-bounds`).
A restricted check is also cheaper: at $n = 25$ the restricted checks drawn in the experiment have weight {glue:text}`weight_restricted:.1f` on average, against {glue:text}`weight_uniform:.1f` for uniform ones, and each unit of weight is a two-qubit gate that both costs overhead and can fault.

With room in the budget, the stronger check wins.
At an overhead of 4 under the paper's rates, the best pairs at $n = 25$ use {glue:text}`r_uniform` uniform checks or {glue:text}`r_restricted` restricted ones, and the larger number of restricted checks does not make up for their weaker detection.
When the budget binds, as at an overhead of 2, or at larger sizes under the Forte-like rates, where restarts consume much of it, only a few checks fit, and the lighter family fits more of them.

## What the bounds would choose

The comparison above uses measurement to choose, and the bounds see a different picture.
For the restricted family, {func}`clinr.bounds.choose_tr` recommends the direct implementation in {glue:text}`restricted_bound_direct` of {glue:text}`restricted_cases` cases: its worst-case miss of $1 - 2/(3n)$ per check leaves the bound above the direct one for every pair within either budget.
A reader guided by the bounds alone would therefore never find the regime in which the paper's own family is the better one.
How much this caution costs, for both families, is the subject of {doc}`regret`.
