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

# Sweep a fixed set of random Clifford circuits

{func}`clinr.circuits.random_clifford` draws uniformly random Cliffords, but Stim's sampler behind it takes no seed.
This guide draws a set once, saves it, and runs both implementations over it.

```{code-cell} python
---
tags: [remove-cell]
---
import os
import tempfile

os.chdir(tempfile.mkdtemp())
```

## Draw and save the circuits

```{code-cell} python
from clinr import circuits

n = 3
circuits.save_circuits(circuits.random_cliffords(n, 5), "cliffords.json")
```

## Load them

Later runs load the same circuits instead of drawing new ones:

```{code-cell} python
cliffords = circuits.load_circuits("cliffords.json")
```

For circuits reproducible from a seed instead, use {func}`clinr.circuits.random_gate_sequence`, which takes a NumPy generator.

## Run both implementations over the set

{func}`clinr.estimators.sweep_direct` and {func}`clinr.estimators.sweep_clinr` return one estimate per circuit, in order:

```{code-cell} python
import numpy as np

from clinr import estimators, noise_models

noise = noise_models.asymmetric_noise(1e-3)
rng = np.random.default_rng(7)
direct = estimators.sweep_direct(cliffords, n, noise, 20_000, rng)
clinr = estimators.sweep_clinr(cliffords, n, 1, 4, noise, 20_000, rng)
for index, (plain, protected) in enumerate(zip(direct, clinr)):
    print(f"circuit {index}: {protected.s1 + protected.s2} gates, "
          f"direct {plain.p_log:.4f}, CliNR {protected.p_log:.4f}")
```
