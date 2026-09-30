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

# Getting started

In this tutorial you run a small Clifford circuit two ways, directly and under Clifford Noise Reduction (CliNR), and compare the logical error rates.
Along the way you compute the analytic bounds, measure the gate overhead, and let the package choose the protocol's parameters.
Everything runs in a few seconds.

## Install

The package is not on PyPI; install it from GitHub:

```bash
pip install git+https://github.com/Daminark/clifford-noise-reduction
```

## A circuit and a noise model

Every random draw in the package takes a NumPy generator, so seed one and draw a random Clifford circuit of 30 gates on 3 qubits.

```{code-cell} python
import numpy as np

from clinr import accounting, bounds, circuits, estimators, noise_models

rng = np.random.default_rng(2026)
n = 3
circuit = circuits.random_gate_sequence(n, 30, rng)
s1, s2 = accounting.gate_counts(circuit)
print(f"{s1} single-qubit gates, {s2} two-qubit gates")
```

Choose a noise model: two-qubit gates fail with probability `1e-3`, and single-qubit gates, measurements and preparations ten times less often.

```{code-cell} python
noise = noise_models.asymmetric_noise(1e-3)
noise
```

## Run it directly

Sample the circuit as it stands.
The logical error rate is the fraction of shots in which the implemented channel is not the circuit's, with a 95% Wilson interval.

```{code-cell} python
direct = estimators.estimate_direct(circuit, n, noise, 200_000, rng)
rate, low, high = direct.interval()
print(f"direct: {rate:.4f}  [{low:.4f}, {high:.4f}]")
```

## Run it with CliNR

CliNR splits the circuit into `t` chunks and applies each through a resource state guarded by `r` stabilizer checks.
Take one chunk and four checks.
A shot counts only if every check passes, so the estimate also reports the acceptance rate.

```{code-cell} python
clinr = estimators.estimate_clinr(circuit, n, 1, 4, noise, 200_000, rng)
rate, low, high = clinr.interval()
print(f"CliNR, t = 1, r = 4: {rate:.4f}  [{low:.4f}, {high:.4f}]")
print(f"acceptance: {clinr.acceptance:.3f}")
```

The error rate is lower, and the price is operations.
The gate overhead is the expected number of operations per gate of the circuit, restarts included.

```{code-cell} python
print(f"gate overhead: {estimators.gate_overhead(clinr):.2f}")
```

## Compare with the bounds

The package also bounds both quantities analytically, from the circuit's gate counts alone.
The logical error bound holds on average over the random draw of the checks, and here it sits well above what you measured, and above even the direct rate.
It is loose; how loose, and whether that matters for choosing parameters, is what the results pages examine.

```{code-cell} python
print(f"logical error bound: {bounds.plog_bound(circuit, n, 1, 4, noise):.4f}")
print(f"gate overhead bound: {bounds.overhead_bound(circuit, n, 1, 4, noise):.2f}")
```

## Vary the number of checks

More checks catch more faults in the resource state, but every check is noisy too, and more checks mean more restarts.

```{code-cell} python
for r in range(0, 9, 2):
    estimate = estimators.estimate_clinr(circuit, n, 1, r, noise, 100_000, rng)
    print(
        f"r = {r}: measured {estimate.p_log:.4f}, "
        f"bound {bounds.plog_bound(circuit, n, 1, r, noise):.4f}, "
        f"overhead {estimators.gate_overhead(estimate):.2f}"
    )
```

## Let the package choose

{func}`clinr.bounds.choose_tr` searches for the pair `(t, r)` with the smallest logical error bound whose overhead bound stays within a ceiling you set.

```{code-cell} python
t, r = bounds.choose_tr(circuit, n, noise, max_overhead=10, r_max=10)
chosen = estimators.estimate_clinr(circuit, n, t, r, noise, 200_000, rng)
print(f"chosen: t = {t}, r = {r}, measured {chosen.p_log:.4f}")
```

## Next steps

Each estimate here uses one random draw of the checks, shared by all its shots.
The {doc}`../reference/api` covers the functions for repeated draws, {func}`clinr.estimators.sweep_clinr` and {func}`clinr.estimators.cluster_interval`, and every option used above.
