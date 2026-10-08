# Differences from the paper

This package implements the protocol of Delfosse and Tham ([arXiv:2407.06583](https://arxiv.org/abs/2407.06583)), and departs from the paper in its noise model, in the circuits it builds, in its bounds and in how it estimates.
This page lists the departures; the page named under each table explains them.

## The noise model

|                               | Paper                                                                  | This package                                                                     |
| ----------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Error rates                   | one rate $p$ in the analysis; $p_1 = p_2/10$ in the numerics of Sec. V | four rates, $p_1$, $p_2$, $p_m$ and $p_{\mathrm{prep}}$                          |
| $\lvert +\rangle$ preparation | one operation, so a check of weight $w$ costs $w + 3$                  | a reset and a Hadamard, so $w + 4$; one `RX` and $w + 3$ with `native_plus=True` |
| Preparation error             | depolarizing at $p$                                                    | a flip at $p_{\mathrm{prep}}$                                                    |
| Idle qubits                   | noiseless in Sec. II and Theorem 1; depolarizing at $p_1$ in Sec. V    | noiseless                                                                        |
| Gates                         | single-qubit Cliffords and controlled Paulis                           | any single- or two-qubit unitary gate, each one fault location                   |

Explained in {doc}`circuits`.

## The circuits

|                | Paper                                                                   | This package                                                                                                                              |
| -------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Correction $Q$ | $n$ Pauli gates after every chunk, so $5n$ operations per teleportation | tracked as a Pauli frame, so $4n$ operations per teleportation ({doc}`ADR 2 <../adr/0002-deliver-the-output-up-to-a-pauli-frame>`)        |
| Failed checks  | the attempt restarts at the first check that fails                      | all $r$ checks are measured before deciding, which accepts the same attempts ({doc}`ADR 1 <../adr/0001-run-every-check-before-deciding>`) |
| Chunks         | at most $s$ in Theorem 1                                                | any $t \ge 1$, each empty chunk paying its teleportation ({doc}`ADR 3 <../adr/0003-allow-more-chunks-than-gates>`)                        |
| Restarts       | a rejected resource is prepared again                                   | one attempt per chunk, with post-selection on every check passing, which gives the same output distribution ({doc}`the-scheme`)           |
| Qubits         | $3n + 1$                                                                | $4n + 1$ in simulation, the extra $n$ a noiseless Choi reference that reads out the channel                                               |

Explained in {doc}`circuits`, except where a row names another page.

## The checks and the bounds

|                                 | Paper                                                                                                     | This package                                                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Family of checks                | uniform over the stabilizer group in Theorem 1; the propagated Bell stabilizers in the numerics of Sec. V | uniform by default; the propagated Bell stabilizers with `restricted_checks=True` |
| Bound for the restricted family | none                                                                                                      | a miss of at most $1 - 2/(3n)$ per draw                                           |
| Error rates in the bounds       | one                                                                                                       | four, one per kind of operation                                                   |
| Chunk sizes in the bounds       | every chunk charged $s_0 = \lceil s/t \rceil$ gates                                                       | each chunk charged its own gates                                                  |
| Operations in the bounds        | as in the paper's circuit                                                                                 | as in the circuit built                                                           |
| Bounds without checks           | conditioned on acceptance, as with checks                                                                 | unconditioned, since every attempt is accepted                                    |

Explained in {doc}`checks-and-bounds`.

## Estimation

|                     | Paper                                                                                                | This package                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Checks across shots | new checks for every attempt (Sec. III); one set per batch of $10^3$ shots in the numerics of Sec. V | one draw per estimate; the protocol with new checks estimated over many draws, or with kept checks on request |

Explained in {doc}`estimation`.
