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

One call to {func}`clinr.estimators.estimate_clinr` draws the checks once and shares them across all its shots, while the protocol draws new checks for every attempt.
This guide estimates the protocol's logical error rate and gate overhead over several draws, with an interval that accounts for the differences between them.

## Take several draws

Call {func}`clinr.estimators.estimate_clinr` repeatedly with the same circuit, `n`, `t`, `r` and options; each call draws new checks.
The precision comes mostly from the number of draws rather than the shots per draw, so prefer many draws of modest size.

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
    estimators.estimate_clinr(circuit, n, t, r, noise, 4_000, rng)
    for _ in range(40)
]
```

## Summarize them

{func}`clinr.estimators.logical_error_rate` and {func}`clinr.estimators.gate_overhead_over_draws` give the protocol's logical error rate and gate overhead, with the checks redrawn for every attempt, each with a Student t interval over the draws.
Pass `checks="kept"` for the values with one check set kept for every attempt.
{func}`clinr.estimators.design_effect` compares the spread between draws with the spread within them; well above 1, a Wilson interval on the pooled shots would be too narrow.

```{code-cell} python
rate, low, high = estimators.logical_error_rate(draws)
print(f"logical error rate: {rate:.4f}  [{low:.4f}, {high:.4f}]")
overhead, low, high = estimators.gate_overhead_over_draws(draws)
print(f"gate overhead: {overhead:.3f}  [{low:.3f}, {high:.3f}]")
print(f"design effect: {estimators.design_effect(draws):.2f}")
```

All three raise `ValueError` if the draws come from different gate counts, numbers of checks, `n` or options.
