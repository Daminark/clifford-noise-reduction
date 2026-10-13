# Differences from the paper

This package implements the protocol of Delfosse and Tham ([arXiv:2407.06583](https://arxiv.org/abs/2407.06583)), and departs from the paper in its noise model, in the circuits it builds, in its bounds, in its treatment of CZNR and in how it estimates.
We list the departures here, and the page named under each table explains them.

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

|                | Paper                                                                   | This package                                                                                                                               |
| -------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Correction $Q$ | $n$ Pauli gates after every chunk, so $5n$ operations per teleportation | tracked as a Pauli frame, so $4n$ operations per teleportation ({doc}`ADR 2 <../adr/0002-deliver-the-output-up-to-a-pauli-frame>`)         |
| Failed checks  | the attempt restarts at the first check that fails                      | all $r$ checks are measured before deciding, which accepts the same attempts ({doc}`ADR 1 <../adr/0001-run-every-check-before-deciding>`)  |
| Chunks         | at most $s$ in Theorem 1                                                | any $t \ge 1$, each empty chunk paying its preparation, checks and teleportation ({doc}`ADR 3 <../adr/0003-allow-more-chunks-than-gates>`) |
| Restarts       | a rejected resource is prepared again                                   | one attempt per chunk, with post-selection on every check passing, which gives the same output distribution ({doc}`the-scheme`)            |
| Qubits         | $3n + 1$                                                                | $4n + 1$ in simulation, the extra $n$ a noiseless Choi reference that reads out the channel                                                |

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

## CZNR

|                        | Paper (App. E)                                                | This package                                                     |
| ---------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| Family of checks       | uniform over the graph state's stabilizer group               | the same; the restricted family is CliNR's only                  |
| Teleportation          | $3n$ operations, with $n$ Pauli gates applying the correction | $2n$, the correction tracked as a Pauli frame                    |
| Qubits                 | $2n + 1$                                                      | $3n + 1$ in simulation, the extra $n$ a noiseless Choi reference |
| Statement of Theorem 3 | $g_p(n + s)$ in the resource term, and both $m$ and $m_0$     | read as $g_p(n + s_0)$ and $m_0$, as in Theorem 1                |

The bounds depart from Theorem 3 as those for CliNR depart from Theorem 1.
Explained in {doc}`cznr`.

## Estimation

|                     | Paper                                                                                                | This package                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Checks across shots | new checks for every attempt (Sec. III); one set per batch of $10^3$ shots in the numerics of Sec. V | one draw per estimate; the protocol with new checks estimated over many draws, or with kept checks on request |

Explained in {doc}`estimation`.
