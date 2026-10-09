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

{func}`clinr.estimators.estimate_clinr` draws its checks at random.
This guide evaluates checks you choose instead, here the lightest stabilizers of the resource state: by sampling the circuit with Stim, exactly to first order in the error rates with {mod}`clinr.first_order`, and in operations with {mod}`clinr.accounting`.
The analytic bounds and {func}`clinr.estimators.gate_overhead` cover only the two random families.

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
        element = stim.PauliString(2 * n)
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

## Build the circuit and sample it

{func}`clinr.protocol.build_clinr_tr` checks that each check stabilizes its chunk's resource, and emits one detector per check, in chunk order, and `2n` observables.
A shot is accepted when no detector fires, and a logical error is an accepted shot with any observable flipped:

```{code-cell} python
noise = noise_models.NoiseModel(p1=1e-4, p2=1e-3, p_m=5e-4, p_prep=2e-4)
built = protocol.build_clinr_tr(circuit, n, checks, noise)
sampler = built.compile_detector_sampler(seed=7)
detectors, observables = sampler.sample(100_000, separate_observables=True)
accepted = ~detectors.any(axis=1)
print(f"acceptance: {accepted.mean():.3f}")
print(f"logical error rate: {observables[accepted].any(axis=1).mean():.4f}")
```

## Compute the rates to first order

To first order in the error rates, {func}`clinr.first_order.faults` gives the same two numbers without sampling, and attributes the undetected logical errors to the regions of the circuit, by the tags {func}`clinr.protocol.build_clinr_tr` puts on its noise channels:

```{code-cell} python
from collections import defaultdict

from clinr import first_order

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

The first-order values are exact for these checks, and differ from the sampled ones by the effect of several faults in one shot.

## Price the overhead

{func}`clinr.accounting.expected_overhead` takes each chunk's gate counts, check region and acceptance.
The detectors come in chunk order, `r` per chunk, so each chunk's acceptance is the fraction of shots in which none of its own fires:

```{code-cell} python
from clinr import accounting

per_chunk = detectors.reshape(len(detectors), t, r)
acceptance = [float(rate) for rate in (~per_chunk.any(axis=2)).mean(axis=0)]
overhead = accounting.expected_overhead(
    [accounting.gate_counts(chunk) for chunk in chunks],
    n,
    accounting.prep_region(n, native_plus=False),
    [
        accounting.check_region(
            r, sum(check.weight for check in chunk_checks), native_plus=False
        )
        for chunk_checks in checks
    ],
    acceptance,
)
print(f"gate overhead: {overhead:.2f}")
```
