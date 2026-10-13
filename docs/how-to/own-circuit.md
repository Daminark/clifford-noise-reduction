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

# Run CliNR on your own circuit

This guide estimates and bounds CliNR on a Clifford circuit you supply.

## Write or load the circuit

The circuit must consist of single- and two-qubit unitary gates on qubits `0` to `n - 1`.
`REPEAT` blocks are unrolled, and annotations such as `TICK` are ignored; measurements, resets and noise channels are rejected (see {func}`clinr.protocol.gate_list`).

Write the circuit in Stim's text format, or load it with `stim.Circuit.from_file`:

```{code-cell} python
import numpy as np
import stim

from clinr import bounds, estimators, noise_models

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
```

## Declare the number of qubits

Pass the number of data qubits `n` explicitly.
It is not inferred from the circuit, so a circuit that leaves its last qubits untouched is still treated as an `n`-qubit circuit.

```{code-cell} python
n = 4
```

## Set the noise model

Give the four error rates: depolarizing after single- and two-qubit gates, a flipped measurement outcome, and a preparation in the orthogonal state.

```{code-cell} python
noise = noise_models.NoiseModel(p1=1e-4, p2=1e-3, p_m=5e-4, p_prep=2e-4)
```

{func}`clinr.noise_models.uniform_noise` and {func}`clinr.noise_models.asymmetric_noise` build common cases from one rate.

## Choose the number of chunks and checks

Pick `t` and `r` yourself, or let {func}`clinr.bounds.choose_tr` pick the pair with the smallest logical error bound under a ceiling on the overhead bound.
It returns `None` when the direct implementation's bound is smaller than every pair's within the ceiling; raise the ceiling, or run the circuit directly.
If the returned `r` equals `r_max`, the cap may be binding; raise it and search again.

```{code-cell} python
choice = bounds.choose_tr(circuit, n, noise, max_overhead=8, r_max=8)
assert choice is not None, "the bounds favor the direct implementation"
t, r = choice
plog = bounds.plog_bound(circuit, n, t, r, noise)
overhead = bounds.overhead_bound(circuit, n, t, r, noise)
print(f"t = {t}, r = {r}")
print(f"bounds: logical error {plog:.4f}, gate overhead {overhead:.2f}")
```

## Estimate both implementations

```{code-cell} python
rng = np.random.default_rng(7)
direct = estimators.estimate_direct(circuit, n, noise, 100_000, rng)
clinr = estimators.estimate_clinr(circuit, n, t, r, noise, 100_000, rng)
print(f"direct: {direct.p_log:.4f}")
overhead = estimators.gate_overhead(clinr)
print(f"CliNR:  {clinr.p_log:.4f}, gate overhead {overhead:.2f}")
```

## Change the protocol's options

Three keyword options select variants of the protocol: `restricted_checks` draws checks from the low-weight family of the paper's Sec. V, `native_plus` prepares `|+>` in one operation, and `noisy_checks=False` measures checks without noise.
Pass the same options to {func}`clinr.bounds.choose_tr`, the bounds and the estimators, so that all three describe the same protocol.

```{code-cell} python
options = {"restricted_checks": True}
restricted = estimators.estimate_clinr(
    circuit, n, t, r, noise, 100_000, rng, **options
)
plog = bounds.plog_bound(circuit, n, t, r, noise, **options)
print(f"restricted checks: {restricted.p_log:.4f}, bound {plog:.4f}")
```
