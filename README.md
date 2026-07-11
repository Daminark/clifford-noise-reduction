# CliNR: Clifford Noise Reduction in Stim

An implementation and simulation study of Clifford Noise Reduction (CliNR), the noise-reduction scheme of Delfosse and Tham ([arXiv:2407.06583](https://arxiv.org/abs/2407.06583)), built on [Stim](https://github.com/quantumlib/Stim).

CliNR lowers the logical error rate of a Clifford circuit `C` by applying it through gate teleportation.
Rather than running `C` directly on the data, it prepares a resource state carrying `C` offline, checks that resource with a few random stabilizer measurements, and teleports the data through only if the checks pass.
Because `C` is Clifford, the teleportation byproduct is a Pauli read directly off `C`'s tableau, so the whole scheme is a stabilizer circuit.
That makes it classically simulable at scale, which is what allows a meaningful logical-error-rate estimate on a laptop.

This repository implements the `t = 1` case (Lemma 1 of the paper) and uses it to demonstrate the mechanism:

- gate teleportation of an arbitrary Clifford, verified for all inputs at once via a Choi (channel-state) test;
- offline fault detection, where `r` random stabilizer checks catch resource faults with probability `1 - 2^{-r}`;
- the effect of `r` on the logical error rate, which falls with more checks and saturates at the floor set by the unprotected teleportation gadget.

## What's here

| file | role |
|------|------|
| `clinr.py` | Circuit construction. Builds `CliNR_{1,r}(C)` with the Choi test harness and the circuit-level noise model. Exact at `p = 0`. |
| `estimators.py` | Monte-Carlo evaluation: `p_logical` (logical error rate) and `catch_rate` (detection probability), pooled over random check sets. |
| `demo.ipynb` | Worked example and the two core figures. Start here. |

## Quickstart

```bash
uv sync
uv run jupyter lab demo.ipynb
```

or, minimally, in Python:

```python
import numpy as np, stim
import clinr, estimators

# a small random Clifford circuit on n qubits
n = 4
rng = np.random.default_rng(0)
C = stim.Circuit()
for _ in range(60):
    which = rng.integers(3)
    if which == 0:   C.append("H", [int(rng.integers(n))])
    elif which == 1: C.append("S", [int(rng.integers(n))])
    else:
        a, b = rng.choice(n, 2, replace=False)
        C.append("CX", [int(a), int(b)])

# estimate the CliNR logical error rate with r = 3 checks
est = estimators.p_logical(C, n, r=3, p1=1e-4, p2=1e-3, p_m=1e-4)
print(est)
print(est.acceptance)
```

## Noise model

Circuit-level depolarizing noise (paper, Sec. II): `p1` after preparations and single-qubit gates, `p2` after controlled-Paulis, `p_m` for measurement flips.
Idle qubits are noiseless.
The three rates are independent parameters here.
The demo pegs them to a single scale (`p1 = p_m = p2/10`) in the noise regime it explores.

## Scope

This is a study of the `t = 1` scheme (Lemma 1), a single teleported block.
The general `t`-block construction (Theorem 1), which partitions large circuits to bound the restart overhead, enables scaling and is the immediate next step.