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

# Average over repeated draws of the checks

One call to {func}`clinr.estimators.estimate_clinr` draws the checks once and shares them across all its shots, while the protocol draws fresh checks each time it runs.
This guide estimates the protocol's logical error rate and gate overhead over several draws, with an interval that accounts for the differences between them.

## Take several draws

Call {func}`clinr.estimators.estimate_clinr` repeatedly with the same circuit, `n`, `t`, `r` and options; each call draws new checks.

```{code-cell} python
import numpy as np
import stim

from clinr import estimators, noise_models

circuit = stim.Circuit("""
REPEAT 8 {
    H 0
    CX 0 1
    S 1
    CX 1 2
    H 2
    CX 2 3
    S 3
    CX 3 0
}
""")
n, t, r = 4, 1, 6
noise = noise_models.NoiseModel(p1=1e-4, p2=1e-3, p_m=5e-4, p_prep=2e-4)
rng = np.random.default_rng(7)
draws = [
    estimators.estimate_clinr(circuit, n, t, r, noise, 20_000, rng)
    for _ in range(8)
]
```

## Summarize them

{func}`clinr.estimators.cluster_interval` gives the mean of the per-draw rates with a Student t interval over draws.
{func}`clinr.estimators.design_effect` compares the spread between draws with the spread within them; well above 1, a Wilson interval on the pooled shots would be too narrow.
{func}`clinr.estimators.mean_gate_overhead` averages the per-draw overheads.

```{code-cell} python
rate, low, high = estimators.cluster_interval(draws)
print(f"logical error rate: {rate:.4f}  [{low:.4f}, {high:.4f}]")
print(f"design effect: {estimators.design_effect(draws):.2f}")
print(f"gate overhead: {estimators.mean_gate_overhead(draws):.2f}")
```

All three raise `ValueError` if the draws come from different circuits' gate counts, numbers of checks, `n` or options.
