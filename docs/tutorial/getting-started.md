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

In this tutorial we run a small Clifford circuit two ways, directly and under Clifford Noise Reduction (CliNR), and compare the logical error rates.
Along the way we compute the analytic bounds, measure the cost in operations, and let the package choose the protocol's parameters.
Everything runs in a few seconds, and every result comes from one seeded generator, so running the page again from the top on the same installation gives the same numbers.

## Install

The package needs Python 3.14 or later and is not on PyPI; install it from GitHub:

```bash
pip install git+https://github.com/Daminark/clifford-noise-reduction
```

## A circuit and a noise model

Every random draw in the package takes a NumPy generator, so we seed one first, then draw a random Clifford circuit of 30 gates on 3 qubits.

```{code-cell} python
import numpy as np

from clinr import accounting, bounds, circuits, estimators, noise_models

rng = np.random.default_rng(2026)
n = 3
circuit = circuits.random_gate_sequence(n, 30, rng)
s1, s2 = accounting.gate_counts(circuit)
print(f"{s1} single-qubit gates, {s2} two-qubit gates")
```

Next, a noise model: two-qubit gates fail with probability `1e-3`, and single-qubit gates, measurements and preparations ten times less often.

```{code-cell} python
noise = noise_models.asymmetric_noise(1e-3)
noise
```

## Run it directly

First we sample the circuit as it stands.
The logical error rate is the fraction of shots in which the implemented channel is not the circuit's; the estimate gives it with a 95% interval.

```{code-cell} python
direct = estimators.estimate_direct(circuit, n, noise, 200_000, rng)
rate, low, high = direct.interval()
print(f"direct: {rate:.4f}  [{low:.4f}, {high:.4f}]")
```

## Run it with CliNR

CliNR splits the circuit into `t` chunks and applies each through a resource state guarded by `r` stabilizer checks.
We take one chunk and four checks.
A shot counts only if every check passes, so the estimate also reports the fraction of shots accepted.

```{code-cell} python
clinr = estimators.estimate_clinr(circuit, n, 1, 4, noise, 200_000, rng)
rate, low, high = clinr.interval()
print(f"CliNR, t = 1, r = 4: {rate:.4f}  [{low:.4f}, {high:.4f}]")
print(f"acceptance: {clinr.acceptance:.3f}")
```

Notice that the logical error rate is well below the direct one, and that nearly every shot is accepted.
The price is operations: the gate overhead is the expected number of operations per gate of the circuit, restarts included.

```{code-cell} python
print(f"gate overhead: {estimators.gate_overhead(clinr):.2f}")
```

## Compare with the bounds

The package also bounds both quantities analytically, from the circuit's gate counts alone.

```{code-cell} python
plog = bounds.plog_bound(circuit, n, 1, 4, noise)
overhead = bounds.overhead_bound(circuit, n, 1, 4, noise)
print(f"logical error bound: {plog:.4f}")
print(f"gate overhead bound: {overhead:.2f}")
```

Both bounds sit above what we measured.
Notice that the logical error bound sits above even the direct rate: it is a guarantee, and a loose one.

## Vary the number of checks

Now we repeat the CliNR run with 0 to 8 checks.
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

The output should show the measured rate falling sharply over the first few checks and then leveling off, while the overhead keeps climbing and the bound barely moves.
Notice also that with no checks, CliNR does worse than running the circuit directly: the resource state and the teleportation only add operations, and so faults.

## Let the package choose

Finally, {func}`clinr.bounds.choose_tr` searches for the pair `(t, r)` with the smallest logical error bound whose overhead bound stays within a ceiling we set.

```{code-cell} python
t, r = bounds.choose_tr(circuit, n, noise, max_overhead=10, r_max=10)
chosen = estimators.estimate_clinr(circuit, n, t, r, noise, 200_000, rng)
print(f"chosen: t = {t}, r = {r}, measured {chosen.p_log:.4f}")
```

## What we did

We ran a Clifford circuit under CliNR, brought its logical error rate well below the direct rate at about three operations per gate, compared both figures with their analytic bounds, and let the package choose the protocol's parameters.
Every function used here, with all its options, is in the {doc}`../reference/api`.
