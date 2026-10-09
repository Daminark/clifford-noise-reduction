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

# What the bounds cost

{func}`clinr.bounds.choose_tr` chooses an implementation by the bounds of {doc}`checks-and-bounds`, which hold for every circuit but say nothing about how far above the true values they sit.
A choice made from them is safe, in that its logical error rate is at most its bound, but it can be worse than the choice measurement would make.
We measure how much worse, on the experiment of {doc}`families`, and compare with the rule the paper uses to choose.
We first set out the three ways of choosing and the measure of loss, then report the loss, and close by tracing it to the two bounds being compared.

```{code-cell} python
---
tags: [remove-cell]
---
# Loads the results of scripts/regret.py and computes the figures quoted below.
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
spec = importlib.util.spec_from_file_location("regret", root / "scripts" / "regret.py")
regret = importlib.util.module_from_spec(spec)
sys.modules["regret"] = regret
spec.loader.exec_module(regret)
data = json.loads((root / "results" / "regret.json").read_text(encoding="utf-8"))
config = data["config"]
rows = regret.outcomes(data)
regimes = [name for name, _ in config["regimes"]]
rates = dict((name, noise) for name, noise in config["regimes"])
sizes = config["sizes"]
ceilings = config["ceilings"]
families = list(regret.FAMILIES)


def cell(regime, ceiling, family, n):
    return [
        row for row in rows
        if (row["regime"], row["ceiling"], row["family"], row["n"]) == (regime, ceiling, family, n)
    ]


def regret_of(choice, regime, ceiling, family, n, difference=False):
    """Return the median loss of a choice and the number of circuits it was made on."""
    losses = [
        row[choice] - row["best"] if difference else row[choice] / row["best"]
        for row in cell(regime, ceiling, family, n)
        if not np.isnan(row[choice])
    ]
    return (float(np.median(losses)) if losses else np.nan), len(losses)


for regime in regimes:
    for ceiling in ceilings:
        for family in families:
            for n in sizes:
                key = f"{regime}_{ceiling:.0f}_{family}_{n}".replace("-", "_")
                for choice in ("bound", "rule"):
                    ratio, made = regret_of(choice, regime, ceiling, family, n)
                    difference, _ = regret_of(choice, regime, ceiling, family, n, difference=True)
                    glue(f"{choice}_{key}", ratio, display=False)
                    glue(f"{choice}_difference_{key}", difference, display=False)
                    glue(f"{choice}_made_{key}", made, display=False)

# How close each bound sits to what it bounds.
direct_slack = []
pair_slack = {family: [] for family in families}
outside = []
for row in rows:
    case = regret._key(row["regime"], row["n"], row["circuit"], row["family"])
    noise = rates[row["regime"]]
    text = data["circuits"][str(row["n"])][row["circuit"]]
    s1, s2 = accounting.gate_counts(stim.Circuit(text))
    # The direct implementation's bound: the probability that some gate faults.
    direct_bound = 1 - (1 - noise["p1"]) ** s1 * (1 - noise["p2"]) ** s2
    direct_slack.append(direct_bound / row["direct"])
    pick = row["best_pick"]
    if isinstance(pick, list):
        plog, overhead = data["bounds"][case]["grid"][regret._key(*pick)]
        pair_slack[row["family"]].append(plog / row["best"])
        outside.append(overhead > row["ceiling"])
glue("direct_slack", max(abs(value - 1) for value in direct_slack), display=False)
for family in families:
    glue(f"slack_low_{family}", min(pair_slack[family]), display=False)
    glue(f"slack_high_{family}", max(pair_slack[family]), display=False)
glue("outside", sum(outside), display=False)
glue("best_pairs", len(outside), display=False)
```

## Three ways to choose

Each way of choosing picks, for a given circuit, noise regime, family of checks and budget on the gate overhead, either the direct implementation or a pair $(t, r)$.

- **By the bounds**: {func}`clinr.bounds.choose_tr` takes the least bound on the logical error rate among the direct implementation and the pairs whose overhead bound is within the budget.
- **By the paper's rule**: $r = \lfloor \log_2(s/n) \rfloor$ checks, the choice in the proof of the paper's Theorem 2, and the smallest $t$ whose overhead is within the budget, as in the paper's Fig. 2. We judge the overhead by measurement. The rule always chooses CliNR, and may find no $t$ at all.
- **By measurement**: the least measured logical error rate among the direct implementation and the pairs whose measured overhead is within the budget, as in {doc}`families`.

Measurement is the best a user could do with full knowledge of the grid, so we measure every choice against it.
The **regret** of a choice is the ratio of its logical error rate to that of the measured choice, both read from fresh estimates; a regret of 1 means nothing was lost.
We also report the difference of the two rates, the logical errors lost per run.

## How much each choice loses

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-regret
    caption: |-
      Regret of choosing by the bounds (solid) and by the paper's rule (dashed), for each family of checks.
      Lines join the medians over the circuits of each size; the rule's are over the circuits on which some $t$ fits the budget, and are drawn where it fits on at least half of them.
---
plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
colors = {"uniform": "#2a78d6", "restricted": "#eb6834"}
titles = {"paper": "Paper's rates", "forte-like": "Forte-like rates"}
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True, sharey=True, layout="constrained")
for i, regime in enumerate(regimes):
    for j, ceiling in enumerate(ceilings):
        ax = axes[i, j]
        for family in families:
            for choice, style in (("bound", "-"), ("rule", "--")):
                points = [regret_of(choice, regime, ceiling, family, n) for n in sizes]
                half = config["circuits_per_size"] / 2
                ys = [ratio if made >= half else np.nan for ratio, made in points]
                ax.plot(
                    sizes, ys, color=colors[family], linestyle=style, linewidth=1.5,
                    marker="o", markersize=4, markerfacecolor="white" if choice == "rule" else colors[family],
                    label=f"{family}, {'bounds' if choice == 'bound' else 'paper' + chr(39) + 's rule'}",
                )
        ax.axhline(1, color="#52514e", linewidth=0.8)
        ax.set_title(f"{titles[regime]}, overhead at most {ceiling:.0f}")
        ax.grid(color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i == 1:
            ax.set_xlabel("qubits $n$")
        if j == 0:
            ax.set_ylabel("regret")
        ax.set_xticks(sizes)
axes[0, 0].legend(frameon=False)
plt.show()
```

With uniform checks under the paper's rates and an overhead of 4, choosing by the bounds loses little: a regret of {glue:text}`bound_paper_4_uniform_8:.2f`, {glue:text}`bound_paper_4_uniform_15:.2f` and {glue:text}`bound_paper_4_uniform_25:.2f` at $n = 8$, $15$ and $25$, or {glue:text}`bound_difference_paper_4_uniform_25:.3f` in logical error rate at $n = 25$.
This is the regime the bounds were made for, and there they do as well as the paper's rule or better, which loses {glue:text}`rule_paper_4_uniform_8:.2f`, {glue:text}`rule_paper_4_uniform_15:.2f` and {glue:text}`rule_paper_4_uniform_25:.2f`.

With restricted checks the bounds recommend the direct implementation throughout, as {doc}`families` found, and the regret is that of not using CliNR at all.
Under the paper's rates and an overhead of 4 it reaches {glue:text}`bound_paper_4_restricted_25:.2f` at $n = 25$, a difference of {glue:text}`bound_difference_paper_4_restricted_25:.2f` in logical error rate.
The paper's rule, which needs no bound, loses {glue:text}`rule_paper_4_restricted_25:.2f` there.
The same holds under the Forte-like rates at an overhead of 4, for both families, where the bounds keep the direct implementation and lose up to {glue:text}`bound_forte_like_4_restricted_15:.2f` at $n = 15$.

At an overhead of 2 the picture is mixed.
Under the paper's rates, the bounds still find a pair with uniform checks at $n = 8$ and $15$, but fall back to the direct implementation at $n = 25$, with a regret of {glue:text}`bound_paper_2_uniform_25:.2f`; the paper's rule never fits this budget with uniform checks, since with its $r$ checks no $t$ brings the overhead within it.
Under the Forte-like rates there is almost nothing to lose, because no pair does more than marginally better than the direct implementation.

The paper's rule has the opposite failing at the smallest size.
At $n = 3$ under the Forte-like rates the direct implementation is the best choice, and the bounds choose it on every circuit, while the rule, which always chooses CliNR, loses {glue:text}`rule_forte_like_4_uniform_3:.2f` with uniform checks at an overhead of 4.

## Where the regret comes from

The bounds compare two numbers of very different quality.
The bound on the direct implementation, the probability that some gate faults, is nearly exact: on every circuit and regime it agrees with the measured rate to within {glue:text}`direct_slack:.1%`, about the noise of the estimate, since a single fault in a Clifford circuit leaves a nontrivial Pauli on the output, and so a logical error.
The bound on CliNR at the measured choice is {glue:text}`slack_low_uniform:.1f` to {glue:text}`slack_high_uniform:.1f` times the measured rate with uniform checks, and {glue:text}`slack_low_restricted:.1f` to {glue:text}`slack_high_restricted:.1f` times with restricted ones, whose worst-case miss per check is far from its typical one.
A comparison of a nearly exact number with a loose one favors the former, so choosing by the bounds leans toward the direct implementation.

The budget adds a second, independent loss.
The overhead bound replaces each chunk's acceptance probability by the probability that nothing in its protected region faults, which is lower, so it overstates the overhead.
Of the {glue:text}`best_pairs` measured choices that are pairs, {glue:text}`outside` have an overhead bound above the budget, so the bounds cannot choose them at all, however good their logical error bound.

Thus the bounds are reliable as guarantees and conservative as guides.
Where they are tight enough, with uniform checks and room in the budget, they choose nearly as well as measurement; elsewhere a simple rule with no guarantee chooses better.
