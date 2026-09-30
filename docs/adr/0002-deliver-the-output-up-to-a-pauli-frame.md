# 2. Deliver the output up to a Pauli frame

## Status

Accepted.

## Context

Gate teleportation leaves a Pauli correction on the output, which depends on the Bell measurement.
The paper applies it as `n` Pauli gates after each chunk, and charges them in its `5n` operations per teleportation.
Between Clifford operations a control stack does not apply such a correction: it tracks it in software, since a Pauli passes through a Clifford gate by a classically computable update, and applies it only where the state is read out or reaches a non-Clifford gate.

## Decision

The circuits never apply the correction.
It is tracked as a Pauli frame, updated through each later chunk and folded into the observables at readout, so the circuit implements `C` up to that frame.

## Consequences

The correction is not a fault location, and a teleportation costs `4n` operations rather than the paper's `5n`, in both the circuit and the bounds.
A caller who needs the corrected state pays for applying the correction outside this model.
