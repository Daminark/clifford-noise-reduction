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

# How tight the bounds are

{doc}`regret` found the bound on the direct implementation nearly exact, and the bound on CliNR far above the measured rate: at the pairs with checks that measurement chose, between {glue:text}`slack_low:.1f` and {glue:text}`slack_high:.1f` times.
We take the CliNR bound apart to see where that slack comes from.
We first describe how each ingredient of the bound is replaced by what actually happens, then read off the slack ingredient by ingredient, look closely at the one that dominates for restricted checks, and close with the bound on the gate overhead.

```{code-cell} python
---
tags: [remove-cell]
---
# Loads the results of scripts/sharpness.py and computes the figures quoted below.
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
from matplotlib_inline.backend_inline import set_matplotlib_formats
from myst_nb import glue

from clinr import accounting

set_matplotlib_formats("svg")
root = Path.cwd().parents[1]
sys.path.insert(0, str(root / "scripts"))
for name in ("regret", "sharpness"):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
regret, sharpness = sys.modules["regret"], sys.modules["sharpness"]
results = json.loads((root / "results" / "regret.json").read_text(encoding="utf-8"))
data = json.loads((root / "results" / "sharpness.json").read_text(encoding="utf-8"))
rows = sharpness.waterfall(data, results)
regimes = [name for name, _ in results["config"]["regimes"]]
sizes = results["config"]["sizes"]
families = list(regret.FAMILIES)
stages = sharpness.STAGES
factor_names = ["harmless faults", "detection", "acceptance", "multiple faults"]


def select(regime=None, family=None, n=None):
    return [
        row for row in rows
        if (regime is None or row["regime"] == regime)
        and (family is None or row["family"] == family)
        and (n is None or row["n"] == n)
    ]


def factors(cell):
    """Return each point's ratio of one stage to the next."""
    values = np.array([[row["stages"][stage] for stage in stages] for row in cell])
    return values[:, :-1] / values[:, 1:]


def region_ratio(cell, region, upper, lower):
    return float(np.median([row["regions"][lower][region] / row["regions"][upper][region] for row in cell]))


glue("points", len(rows), display=False)
slack = [row["stages"]["bound"] / row["stages"]["measured"] for row in rows]
glue("slack_low", min(slack), display=False)
glue("slack_high", max(slack), display=False)
glue("draws", data["config"]["draws"], display=False)
uniform = select(family="uniform")
restricted = select(family="restricted")
glue("resource_damaging", region_ratio(uniform, "resource", "bound", "damaging"), display=False)
glue("resource_caught", region_ratio(uniform, "resource", "damaging", "detection"), display=False)
glue("check_damaging", region_ratio(uniform, "check", "bound", "damaging"), display=False)
glue("check_escaping", region_ratio(uniform, "check", "damaging", "detection"), display=False)
glue("teleport_damaging", region_ratio(select(), "teleport", "bound", "damaging"), display=False)
glue("restricted_escaping", region_ratio(select("paper", "restricted", 25), "resource", "damaging", "detection"), display=False)
for regime in regimes:
    for family in families:
        for n in sizes:
            medians = np.median(factors(select(regime, family, n)), axis=0)
            key = f"{regime}_{family}_{n}".replace("-", "_")
            for name, value in zip(("harmless", "detection", "acceptance", "multiple"), medians):
                glue(f"{name}_{key}", float(value), display=False)
overhead = np.array([
    [row["overhead"]["bound"] / row["overhead"]["acceptance"],
     row["overhead"]["acceptance"] / row["overhead"]["measured"]]
    for row in rows
])
glue("overhead_acceptance_max", float(overhead[:, 0].max()), display=False)
glue("overhead_weight_max", float(overhead[:, 1].max()), display=False)
glue("overhead_weight_min", float(overhead[:, 1].min()), display=False)
for n in sizes:
    counts = [sum(accounting.gate_counts(stim.Circuit(text))) for text in results["circuits"][str(n)]]
    glue(f"alpha_{n}", float(np.log(np.median(counts)) / np.log(n)), display=False)
```

## Replacing the bound's ingredients

For one chunk, the bound of {doc}`checks-and-bounds` is

$$\frac{g_{\mathrm{resource}} \, q^r + g_{\mathrm{check}} \, e_r + g_{\mathrm{teleport}}}{1 - g_{\mathrm{protected}}},$$

and {func}`clinr.bounds.plog_terms` returns these terms for each chunk.
Each ingredient is a worst case, and we replace them in turn, at each of the {glue:text}`points` pairs measurement chose in the regret experiment:

1. **Harmless faults.**
   Each $g$ charges every fault of its region, whether or not it damages the output.
   We scale it by the fraction of the region's faults that do.
1. **Detection.**
   $q^r$ and $e_r$ assume that every check misses with the family's worst-case probability, and that a fault in a check gadget is seen only by the later checks.
   We replace both by the faults that escape the checks actually drawn.
1. **Acceptance.**
   $1 - g_{\mathrm{protected}}$ is the probability that nothing in the protected region faults, which is less than the probability that the checks pass.
   We replace it by the measured acceptance.
1. **Multiple faults.**
   The first two replacements count faults one at a time.
   The measured logical error rate accounts for every combination.

The fractions in the first two steps are exact to first order in the error rates for each draw of the checks: {func}`clinr.first_order.faults` lists every fault of the circuit as built, with the region it occurs in and whether it flips a check or the output.
They are averaged over {glue:text}`draws` draws of the checks, and applied to the exact fault probabilities $g$ rather than to first-order sums, which exceed them once several faults per run are likely.
The measured acceptance and logical error rate are the fresh estimates of the regret experiment.

## Where the slack is

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-waterfall
    caption: |-
      The ratio of the bound to the measured logical error rate, as the product of the four factors of the replacements above.
      Each segment is the median of its factor over the circuits of one size; U and R mark uniform and restricted checks.
---
plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
colors = ["#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
titles = {"paper": "Paper's rates", "forte-like": "Forte-like rates"}
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True, layout="constrained")
for ax, regime in zip(axes, regimes):
    positions, labels = [], []
    for i, n in enumerate(sizes):
        for j, family in enumerate(families):
            x = 3 * i + j
            medians = np.median(factors(select(regime, family, n)), axis=0)
            bottom = 1.0
            for k, value in enumerate(np.maximum(medians, 1.0)):
                ax.bar(x, bottom * value - bottom, bottom=bottom, width=0.8,
                       color=colors[k], edgecolor="white", linewidth=1,
                       label=factor_names[k] if (i, j) == (0, 0) else None)
                bottom *= value
            positions.append(x)
            labels.append(f"{family[0].upper()}\n{n}" if j == 0 else family[0].upper())
    ax.set_yscale("log")
    ax.set_xticks(positions, labels)
    ax.set_title(titles[regime])
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("bound / measured")
axes[0].set_xlabel("checks and qubits $n$")
axes[1].set_xlabel("checks and qubits $n$")
axes[1].legend(frameon=False, loc="upper left")
plt.show()
```

With uniform checks, the resource term is nearly exact.
Of the resource's faults, a fraction of {glue:text}`resource_damaging:.2f` damages the output, and the checks let through exactly the worst case: the ratio of escaping to charged faults is {glue:text}`resource_caught:.2f`.
This is the argument of {doc}`checks-and-bounds` at work: exactly half of the stabilizer group anticommutes with any damaging fault, so a uniform check misses it with probability exactly $1/2$.
The slack lies in the other terms.
The check term charges each gadget at the family's worst-case weight, and charges a fault in gadget $j$ only the misses of the later checks.
Lighter drawn checks and harmless faults bring it to {glue:text}`check_damaging:.2f` of its charge, and the escapes to {glue:text}`check_escaping:.2f` of what remains: a fault in a gadget that damages the resource usually also flips that gadget's own outcome, a detection the bound does not count.
Of the teleportation's faults, a fraction of {glue:text}`teleport_damaging:.2f` damages the output.

With restricted checks, detection dominates.
Under the paper's rates the detection factor is {glue:text}`detection_paper_restricted_15:.1f` at $n = 15$ and {glue:text}`detection_paper_restricted_25:.1f` at $n = 25$, against {glue:text}`detection_paper_uniform_15:.1f` and {glue:text}`detection_paper_uniform_25:.1f` with uniform checks: at $n = 25$, the drawn checks let through only {glue:text}`restricted_escaping:.0%` of what the worst case allows of the resource's damaging faults.
The next section shows which faults attain the worst case.

The last two factors grow with the number of faults per run.
Under the Forte-like rates at $n = 25$, the acceptance factor reaches {glue:text}`acceptance_forte_like_uniform_25:.1f` and the multiple-fault factor {glue:text}`multiple_forte_like_uniform_25:.1f` with uniform checks.
The bound on the acceptance is the probability that no operation faults at all, while most faults leave the checks passing; and when several faults are likely in one run, counting them one at a time overstates the rate.

## Detection by position

The restricted family's worst case, a miss of $1 - 2/(3n)$ per check, is attained, but only by some faults.
Write a fault on the resource half that carries the chunk as a Pauli $E$ after the first $g$ gates of the chunk $C$, and $E_0 = C_{\le g}^\dagger E \, C_{\le g}$ for the same fault moved back to the start of $C$.
A restricted check $P_i \otimes C P_i C^\dagger$ anticommutes with $E$ exactly when $P_i$ anticommutes with $E_0$ on qubit $i$, and on each qubit where $E_0$ acts, two of the three $P_i$ do.
So one restricted check catches the fault with probability $2 \, \mathrm{wt}(E_0) / (3n)$.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-positions
    caption: |-
      Probability that one check catches a damaging fault in the resource, by where the fault occurs: in the preparation of the Bell pairs, or among the gates of the chunk, in ten equal parts in time order.
      Means over three circuits per size and their checks, under the paper's rates, for random Clifford circuits and random gate sequences of the same size.
      The dotted line is the restricted family's worst case, $2/(3n)$.
---
bins = data["config"]["bins"]
x = np.arange(bins + 1)
styles = {
    ("uniform", "clifford"): ("#2a78d6", "-"),
    ("uniform", "sequence"): ("#2a78d6", "--"),
    ("restricted", "clifford"): ("#eb6834", "-"),
    ("restricted", "sequence"): ("#eb6834", "--"),
}
position_sizes = data["config"]["position_sizes"]
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2), sharex=True, sharey=True, layout="constrained")
for ax, n in zip(axes.flat, position_sizes):
    for (family, generator), (color, style) in styles.items():
        profiles = [
            np.array(profile["caught"]) / np.array(profile["damaging"])
            for key, profile in data["positions"].items()
            if key.startswith(f"{n}/") and key.endswith(f"/{generator}/{family}")
        ]
        ax.plot(x, np.mean(profiles, axis=0), color=color, linestyle=style,
                linewidth=1.5, marker="o", markersize=3,
                label=f"{family}, {'random Clifford' if generator == 'clifford' else 'gate sequence'}")
    ax.axhline(2 / (3 * n), color="#52514e", linestyle=":", linewidth=1)
    ax.set_title(f"$n = {n}$")
    ax.set_ylim(0, 0.6)
    ax.set_xticks([0, 1, bins], ["prep", "start", "end"])
    ax.grid(color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("probability one check catches")
for ax in axes[1, :]:
    ax.set_xlabel("where the fault occurs")
axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
plt.show()
```

A fault in the preparation of the Bell pairs touches a single pair, on which two of its three checks anticommute with it, and attains the worst case.
A fault early in $C$ is moved back through few gates and stays light, while a fault late in $C$ is moved back through most of the circuit, and a random Clifford spreads it to weight about $3n/4$, where the catch reaches $1/2$, the uniform family's.
Uniform checks catch every damaging fault with probability $1/2$ wherever it occurs.
So the worst case of the restricted family is a property of the first few layers of the chunk, and the bound charges it to every fault.

How the circuit is drawn matters for the same reason.
At the same number of gates, a random sequence of $H$, $S$ and $CX$ spreads a fault more slowly than a random Clifford synthesized from its tableau, so its catch rises later in the chunk, and restricted checks detect less on it.
The sizes of the experiment also vary the shape of the circuits: a synthesized random Clifford has about $1.5 n^2$ gates, so $s = n^\alpha$ with $\alpha$ from {glue:text}`alpha_3:.1f` at $n = 3$ to {glue:text}`alpha_25:.1f` at $n = 25$, and part of what varies with $n$ in these results varies with $\alpha$.

## The bound on the gate overhead

The overhead bound has two ingredients that differ from the measured overhead: the acceptance bound, and the worst-case weight of every check.
Replacing the acceptance bound by the measured acceptance lowers the overhead bound by a factor of up to {glue:text}`overhead_acceptance_max:.1f`, while the drawn weights lower it by only {glue:text}`overhead_weight_min:.2f` to {glue:text}`overhead_weight_max:.2f` times.
Thus, as for the logical error rate, the slack of the overhead bound lies in counting every fault as a restart.
