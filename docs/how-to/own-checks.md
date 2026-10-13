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

# Measure checks of your own choosing

{func}`clinr.estimators.estimate_clinr` and {func}`clinr.estimators.estimate_cznr` draw their checks at random.
This guide evaluates checks you choose instead, here the lightest stabilizers of each chunk's resource state, in two ways: by sampling with {func}`clinr.estimators.estimate_clinr_given`, and exactly to first order in the error rates with {mod}`clinr.first_order`.
The last section does the same for CZNR with {func}`clinr.estimators.estimate_cznr_given`.
The analytic bounds cover only the two random families.

## Find the stabilizers of each chunk's resource state

Checks are written in logical coordinates: length `2n`, with the resource's bare half on `[0, n)` and the half carrying the chunk on `[n, 2n)`.
Split the circuit into its chunks, and get each chunk's `2n` stabilizer generators in that layout:

```{code-cell} python
import itertools

import stim

from clinr import noise_models, protocol

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
n, t, r = 4, 1, 4
bare, image = list(range(n)), list(range(n, 2 * n))
chunks = protocol.chunk_circuit(circuit, n, t)
generators = [
    protocol.stabilizer_gens(chunk.to_tableau(), bare, image, 2 * n)
    for chunk in chunks
]
```

{func}`clinr.checks.restricted_family` gives the paper's low-weight family in the same layout.

## Choose the checks

Any non-identity product of a chunk's generators is a valid check.
Enumerate each chunk's group and keep its `r` lightest elements:

```{code-cell} python
def lightest(gens, count):
    elements = []
    for bits in itertools.product([0, 1], repeat=len(gens)):
        element = stim.PauliString(len(gens[0]))
        for bit, gen in zip(bits, gens):
            if bit:
                element *= gen
        if element.weight > 0:
            elements.append(element)
    return sorted(elements, key=lambda element: element.weight)[:count]


checks = [lightest(gens, r) for gens in generators]
print([check.weight for check in checks[0]])
```

Every chunk needs the same number of checks.

## Estimate the rates

{func}`clinr.estimators.estimate_clinr_given` builds the circuit with {func}`clinr.protocol.build_clinr_tr`, which checks that each check stabilizes its chunk's resource, and samples it.
A shot is accepted when every check passes, and a logical error is an accepted shot whose output is in error:

```{code-cell} python
import numpy as np

from clinr import estimators

noise = noise_models.NoiseModel(p1=1e-4, p2=1e-3, p_m=5e-4, p_prep=2e-4)
estimate = estimators.estimate_clinr_given(
    circuit, n, checks, noise, 100_000, np.random.default_rng(7)
)
rate, lo, hi = estimate.interval()
print(f"acceptance: {estimate.acceptance:.3f}")
print(f"logical error rate: {rate:.4f}, 95% interval {lo:.4f} to {hi:.4f}")
print(f"gate overhead: {estimators.gate_overhead(estimate):.2f}")
```

The interval is for the rate given these checks.
{func}`clinr.estimators.gate_overhead` charges each chunk its checks' weight and its measured acceptance.
An estimate with given checks is not a draw of them, so the statistics over draws, such as {func}`clinr.estimators.logical_error_rate`, reject it.

## Compute the rates to first order

To first order in the error rates, {func}`clinr.first_order.faults` gives the acceptance and the logical error rate without sampling, and attributes the undetected logical errors to the regions of the circuit, by the tags {func}`clinr.protocol.build_clinr_tr` puts on its noise channels:

```{code-cell} python
from collections import defaultdict

from clinr import first_order

built = protocol.build_clinr_tr(circuit, n, checks, noise)
faults = first_order.faults(built)
rejected = sum(fault.probability for fault in faults if fault.detected)
print(f"acceptance: {1 - rejected:.3f}")
print(f"logical error rate: {first_order.logical_error_rate(faults):.4f}")
by_region = defaultdict(float)
for fault in faults:
    if fault.logical and not fault.detected:
        by_region[fault.tag.split("/")[0]] += fault.probability
print({region: round(rate, 5) for region, rate in by_region.items()})
```

The first-order values differ from the sampled ones by the effect of several faults in one shot, and by the shot noise of the sample.

## Measure CZNR with checks of your own

For a circuit of `CZ` gates, CZNR's resource is each chunk's graph state, on `n` qubits.
{func}`clinr.protocol.graph_state_gens` gives its generators in that layout, and {func}`clinr.estimators.estimate_cznr_given` takes checks built from them:

```{code-cell} python
cz_circuit = stim.Circuit("""
REPEAT 6 {
    CZ 0 1 1 2 2 3 3 0 0 2 1 3
}
""")
cz_chunks = protocol.chunk_circuit(cz_circuit, n, 2)
cz_checks = [
    lightest(protocol.graph_state_gens(chunk.to_tableau()), 2)
    for chunk in cz_chunks
]
cz_estimate = estimators.estimate_cznr_given(
    cz_circuit, n, cz_checks, noise, 100_000, np.random.default_rng(7)
)
rate, lo, hi = cz_estimate.interval()
print([check.weight for check in cz_checks[0]])
print(f"acceptance: {cz_estimate.acceptance:.3f}")
print(f"logical error rate: {rate:.4f}, 95% interval {lo:.4f} to {hi:.4f}")
print(f"gate overhead: {estimators.gate_overhead(cz_estimate):.2f}")
```
