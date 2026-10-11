# CZNR, for circuits of CZ gates

When the circuit consists of `CZ` gates only, CliNR simplifies: the resource state becomes an $n$-qubit graph state, and the data is teleported through it by a one-bit teleportation.
This is CZNR, the variant of the paper's App. E (Delfosse and Tham, [arXiv:2407.06583](https://arxiv.org/abs/2407.06583)), which {func}`clinr.protocol.build_cznr_tr` builds.
This page explains why the simplification works, what a check detects, how the paper's Theorem 3 compares with its Theorem 1, and what can be said of CZNR against CliNR on the same circuit.

## Graph states and one-bit teleportation

A sequence of `CZ` gates on $n$ qubits defines a graph $G$ with an edge between $i$ and $j$ for each `CZ` on the pair, two edges on one pair cancelling since $\mathrm{CZ}^2 = I$.
Its unitary $U_G$ is diagonal in the computational basis.
The *graph state* $|G\rangle = U_G |+\rangle^{\otimes n}$ is stabilized by the generators

$$K_i = X_i \prod_{j \in N(i)} Z_j,$$

with $N(i)$ the neighbors of $i$, which {func}`clinr.protocol.graph_state_gens` returns.

The one-bit teleportation of a qubit $|\psi\rangle$ prepares a fresh qubit in $|+\rangle$, applies a `CX` controlled on the fresh qubit and targeting $|\psi\rangle$, and measures the old qubit in the $Z$ basis.
With outcome $z$, the fresh qubit holds $X^z |\psi\rangle$.
Applied to $n$ qubits at once, it leaves $X^z |\psi\rangle$ on the fresh block, with $z$ the vector of outcomes.

Now apply $U_G$ to the fresh block after the teleportation.
The `CX` gates touch the fresh block only through their controls, and $U_G$ is diagonal, so it commutes with them and can be moved to before them, onto the fresh $|+\rangle^{\otimes n}$: the fresh block is then the graph state $|G\rangle$, prepared before the data arrives (the paper's Lemma 3).
The output is

$$U_G X^z |\psi\rangle = \left(U_G X^z U_G^\dagger\right) U_G |\psi\rangle, \qquad U_G X_i U_G^\dagger = X_i \prod_{j \in N(i)} Z_j,$$

so the circuit applies $U_G$ up to the Pauli $\prod_i (X_i Z_{N(i)})^{z_i}$, one Pauli per qubit.
As for CliNR, the package tracks this correction as a Pauli frame rather than applying it ({doc}`../adr/0002-deliver-the-output-up-to-a-pauli-frame`).

Each chunk therefore needs $n$ resource qubits where CliNR needs $2n$, and its teleportation is $n$ `CX` and $n$ measurements where CliNR's is $n$ `CX`, $n$ Hadamards and $2n$ measurements.
The circuits built for simulation have $3n + 1$ qubits, the extra $n$ a noiseless Choi reference, as in {doc}`circuits`.

## What a check detects

The argument of {doc}`checks-and-bounds` carries over.
The stabilizer group of $|G\rangle$ has $2^n$ elements and is a maximal commuting group, so an error that commutes with all of it leaves the state unchanged, and an error that damages it anticommutes with exactly half of it.
A check drawn uniformly from the group, as {func}`clinr.checks.draw_cznr_checks` draws it, therefore misses a damaging error with probability $1/2$.
It acts on at most $n$ qubits, and on a graph without isolated vertices on $3n/4$ of them on average, since each qubit's Pauli is then uniform over $I$, $X$, $Y$ and $Z$.

CZNR draws uniform checks only, as App. E and Theorem 3 do.
The restricted family of CliNR's numerics comes from pushing the low-weight stabilizers of the initial state through the circuit, and for CZNR that recipe gives the generators $K_i$, of weight $1 + \deg(i)$.
On a dense graph they are heavier than a uniform check, $n$ on the complete graph against $3n/4$, and weaker: $Z$ on one qubit anticommutes with one generator only, so a generator drawn uniformly misses it with probability $1 - 1/n$.

## Theorem 3 against Theorem 1

Theorem 3 is Theorem 1 with CZNR's regions in place of CliNR's.
For one error rate $p$, $g_p(x) = 1 - (1 - p)^x$ and $s_0 = \lceil s/t \rceil$, Theorem 1 bounds the logical error rate by

$$t \cdot \frac{g_p(3n + s_0) 2^{-r} + 2 g_p(2n + 3) + g_p(5n)}{(1 - p)^{m_0}}, \qquad m_0 = 3n + s_0 + (2n + 3) r,$$

and Theorem 3 by

$$t \cdot \frac{g_p(n + s_0) 2^{-r} + 2 g_p(n + 3) + g_p(3n)}{(1 - p)^{m_0}}, \qquad m_0 = n + s_0 + (n + 3) r.$$

Term by term:

| Term             | CliNR, Theorem 1                           | CZNR, Theorem 3                             |
| ---------------- | ------------------------------------------ | ------------------------------------------- |
| Resource         | $3n + s_0$: $n$ Bell pairs, then the chunk | $n + s_0$: $n$ preparations, then the chunk |
| One check gadget | $2n + 3$, a check of weight up to $2n$     | $n + 3$, a check of weight up to $n$        |
| Teleportation    | $5n$, with $n$ Pauli gates of correction   | $3n$, with $n$ Pauli gates of correction    |
| Gate overhead    | $10n/s_0 + 2 m_0 / (s_0 (1 - p)^{m_0})$    | $6n/s_0 + 2 m_0 / (s_0 (1 - p)^{m_0})$      |

Two misprints in the statement of Theorem 3 are corrected here: it writes $g_p(n + s)$ for the resource term, where the counterpart of Theorem 1's $g_p(3n + s_0)$ is $g_p(n + s_0)$, and it defines $m$ but uses both $m$ and $m_0$.
Both overhead bounds use $t s_0 \le 2s$, which holds for $t \le s$, the range Theorem 1 takes.

{func}`clinr.bounds.cznr_plog_bound` and {func}`clinr.bounds.cznr_overhead_bound` rederive Theorem 3 with the departures {doc}`checks-and-bounds` lists for Theorem 1: four rates, each chunk's own size, no conditioning without checks, and the preparation convention, under which $|+\rangle$ costs a reset and a Hadamard unless `native_plus=True`.
The teleportation is charged $2n$, the paper's $3n$ without the correction.

## CZNR against CliNR on the same circuit

At the same $t$ and $r$, CZNR runs fewer operations of every kind in every region than CliNR: a preparation of $2n$ operations against $4n$, the same gates, checks of weight at most $n$ against $2n$ with the same miss of $1/2$, and a teleportation of $2n$ against $4n$.
Each term of its bound is therefore the smaller, and so is its overhead bound; the test suite checks both over random circuits of `CZ` gates and parameters.

The same holds of the logical error rate to first order in the error rates, which is the probability of a single fault that damages the output and escapes every check ({mod}`clinr.first_order`): CZNR has fewer such faults in every region, and the test suite checks its first-order rate against CliNR's on complete graphs.
It does not follow at higher orders.
The two circuits are not one a part of the other, two faults can cancel, and the logical error rate is conditioned on acceptance, so having fewer fault locations does not by itself make the rate lower.
How the schemes compare at the rates where those effects matter is a question for measurement.
