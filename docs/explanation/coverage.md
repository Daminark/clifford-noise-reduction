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

# How often the rate's interval covers

The Korn-Graubard interval of {doc}`estimation` covers the logical error rate about as often as it claims when errors are plentiful, more often when they are few, and markedly less often when the draws of the checks differ a great deal.
We describe the simulation first, then read the table.

```{code-cell} python
---
tags: [remove-cell]
---
# Loads the results of scripts/coverage.py and glues the figures quoted below.
import json
from pathlib import Path

from myst_nb import glue

root = Path.cwd().parents[1]
data = json.loads((root / "results" / "coverage.json").read_text(encoding="utf-8"))
rows = data["scenarios"]
glue("replications", f"{rows[0]['replications']:,}", display=False)
glue("shots", f"{rows[0]['shots']:,}", display=False)
for i, row in enumerate(rows):
    glue(f"rate_{i}", f"{row['rate']:g}", display=False)
    glue(f"sigma_{i}", row["sigma"], display=False)
    glue(f"draws_{i}", row["draws"], display=False)
    glue(f"errors_{i}", f"{row['mean_errors']:.0f}", display=False)
    glue(f"coverage_{i}", f"{100 * row['coverage']:.1f}%", display=False)
    glue(f"mcse_{i}", f"{100 * row['mcse']:.1f}", display=False)
```

## The simulation

`scripts/coverage.py` builds each scenario from draws of the checks whose logical error rates, per shot, are lognormal about a mean rate, with a given standard deviation of their logarithm.
Each draw samples its errors over {glue:text}`shots` shots, every one accepted, so the rate per accepted shot is the rate per shot, and the mean rate is the true rate the interval should cover.
Each scenario is repeated {glue:text}`replications` times, and the Monte Carlo standard error of each coverage $c$ is $\sqrt{c(1 - c)/R}$ over the $R$ repetitions.
The seed and the versions that produced the table are in `results/coverage.json`.

The scenarios cross plentiful and few errors with moderate and heavy spread between draws.
The log-standard deviation of 1.5 for heavy spread, a coefficient of variation near 3 between draws, was chosen before the simulation was run, though not written down first.
An earlier simulation, before this one, found a coverage of 91.8% under heavy spread, but its definition of heavy spread was not saved, so the two figures cannot be read together.

## The coverage

| Scenario                 | Mean rate           | Log-standard deviation | Draws                | Errors, on average    | Coverage at 95%         | Monte Carlo SE (points) |
| ------------------------ | ------------------- | ---------------------- | -------------------- | --------------------- | ----------------------- | ----------------------- |
| Plentiful errors         | {glue:text}`rate_0` | {glue:text}`sigma_0`   | {glue:text}`draws_0` | {glue:text}`errors_0` | {glue:text}`coverage_0` | {glue:text}`mcse_0`     |
| Few errors               | {glue:text}`rate_1` | {glue:text}`sigma_1`   | {glue:text}`draws_1` | {glue:text}`errors_1` | {glue:text}`coverage_1` | {glue:text}`mcse_1`     |
| Heavy spread             | {glue:text}`rate_2` | {glue:text}`sigma_2`   | {glue:text}`draws_2` | {glue:text}`errors_2` | {glue:text}`coverage_2` | {glue:text}`mcse_2`     |
| Few errors, heavy spread | {glue:text}`rate_3` | {glue:text}`sigma_3`   | {glue:text}`draws_3` | {glue:text}`errors_3` | {glue:text}`coverage_3` | {glue:text}`mcse_3`     |

With few errors the interval widens rather than undercovers, since it falls back on the exact interval for the shots it has, and Clopper-Pearson is conservative.
Under heavy spread it falls short, at {glue:text}`coverage_2`: the variance between draws is estimated from {glue:text}`draws_2` of them, on 9 degrees of freedom, the known weakness of design-based intervals with few clusters.
The remedy is more draws where the design effect is large, not a different interval ({doc}`../adr/0005-korn-graubard-intervals-for-the-logical-error-rate`).
Few errors with heavy spread covers at {glue:text}`coverage_3`, since with about one error the interval rests on the exact fallback.
