"""Circuit-independent analytic bounds for CliNR (arXiv:2407.06583, Theorem 1).

Re-derives the Theorem 1 bounds on the logical error rate and the gate
overhead under a four-rate noise model, replacing the paper's collapsed
``p1 = p2 = p_m = p_prep = p``. The bounds are deterministic: no sampling, no RNG.

Rates are carried by a :class:`~noise_models.NoiseModel`; choosing a regime is
the caller's job.

Fault locations are counted for the circuit as built: ``|+>`` preparation is
a reset followed by a Hadamard, preparations are charged separately from
single-qubit gates, the Pauli correction is tracked classically
and contributes nothing, and stabilizers are drawn uniformly from the full
resource stabilizer group so a check has weight at most ``2n``.
"""

from __future__ import annotations

import math

import stim

import accounting
import clinr
from noise_models import NoiseModel

__all__ = [
    "fault_prob",
    "plog_bound",
    "overhead_bound",
    "choose_tr",
]


def _no_fault_prob(
    noise: NoiseModel, n1: int, n2: int, nm: int, n_prep: int = 0
) -> float:
    """Probability that none of the given operations faults."""
    return (
        (1.0 - noise.p1) ** n1
        * (1.0 - noise.p2) ** n2
        * (1.0 - noise.p_m) ** nm
        * (1.0 - noise.p_prep) ** n_prep
    )


def fault_prob(
    noise: NoiseModel, n1: int, n2: int, nm: int, n_prep: int = 0
) -> float:
    """Probability that at least one of the given operations faults.

    Generalizes ``g_p(x) = 1 - (1 - p)^x`` to typed operation counts, and
    reduces to it when the four rates are equal.

    Args:
        noise: The four rates.
        n1: Count of single-qubit gates.
        n2: Count of two-qubit gates.
        nm: Count of measurements.
        n_prep: Count of preparations.

    Raises:
        ValueError: if any count is negative.
    """
    if min(n1, n2, nm, n_prep) < 0:
        raise ValueError(
            f"counts must be non-negative, got ({n1}, {n2}, {nm}, {n_prep})"
        )
    return 1.0 - _no_fault_prob(noise, n1, n2, nm, n_prep)


def _detection(n: int, restricted_checks: bool) -> tuple[float, int]:
    """``(miss, check_weight)`` for the selected check family.

    Uniform draws from the full group detect a damaging fault with probability
    exactly ``1/2``. The restricted family's ``3n`` elements come in triples
    ``(gx, gz, gx gz)`` per Bell pair, and anticommutation is a homomorphism,
    so a fault anticommutes with either none or exactly two of each triple:
    one it is caught by at all, it is caught by at least ``2`` of the ``3n``.
    """
    if restricted_checks:
        return 1.0 - 2.0 / (3 * n), n + 1
    return 0.5, 2 * n


def _chunk_plog_bound(
    noise: NoiseModel,
    n: int,
    s1: int,
    s2: int,
    r: int,
    family: tuple[float, int],
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Bound the logical error rate contributed by one chunk.

    Regions are those of :mod:`accounting`: the resource preparation together
    with the chunk's own gates, each of ``r`` checks, and the teleportation.
    ``family`` is ``(miss, w)`` from :func:`_detection`: the
    probability that one check fails to detect a damaging fault, and the
    worst-case check weight. They travel together so that a miss probability
    cannot be paired with another family's weight.

    A fault in check ``i``'s own gadget can only be caught by checks
    ``i+1..r``, so those faults contribute ``(1 - miss**r) / (1 - miss)``
    times one check's fault probability. That factor is at most ``2`` for
    uniform draws and at most ``3n/2`` for the restricted family, and it
    vanishes at ``r = 0``.
    """
    miss, w = family
    prep = accounting.prep_region(n, native_plus)
    resource = accounting.add(prep, (s1, s2, 0, 0))
    resource_fault = fault_prob(noise, *resource)
    check_fault = fault_prob(
        noise, *accounting.check_region(1, w, noisy_checks, native_plus)
    )
    teleport_fault = fault_prob(noise, *accounting.teleport_region(n))

    escapes = (1.0 - miss**r) / (1.0 - miss)
    numerator = (
        resource_fault * miss**r + check_fault * escapes + teleport_fault
    )
    checks = accounting.check_region(r, r * w, noisy_checks, native_plus)
    no_fault = _no_fault_prob(
        noise, *accounting.protected_region(prep, s1, s2, checks)
    )
    if no_fault == 0.0:
        return math.inf
    return numerator / no_fault


def _plog_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    n: int,
    r: int,
    family: tuple[float, int],
    noisy_checks: bool,
    noisy_q: bool,
    native_plus: bool,
) -> float:
    """Sum the per-chunk bounds for chunks with the given gate counts."""
    total = 0.0
    for s1, s2 in counts:
        total += _chunk_plog_bound(
            noise, n, s1, s2, r, family, noisy_checks, native_plus
        )
    if noisy_q:
        total += fault_prob(noise, *accounting.frame_region(n))
    return total


def _overhead_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    n: int,
    r: int,
    w: int,
    noisy_checks: bool,
    noisy_q: bool,
    native_plus: bool,
) -> float:
    """Bound the expected operations per gate for chunks with these counts.

    Every chunk is charged worst-case check weight, and its acceptance
    probability is replaced by the probability that nothing in its protected
    region faults, which is a lower bound on it.
    """
    prep = accounting.prep_region(n, native_plus)
    checks = accounting.check_region(r, r * w, noisy_checks, native_plus)
    no_fault = [
        _no_fault_prob(
            noise, *accounting.protected_region(prep, s1, s2, checks)
        )
        for s1, s2 in counts
    ]
    return accounting.expected_overhead(
        counts, n, prep, [checks] * len(counts), no_fault, noisy_q
    )


def plog_bound(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    *,
    noisy_checks: bool = True,
    restricted_checks: bool = False,
    noisy_q: bool = False,
    native_plus: bool = False,
) -> float:
    """Bound the logical error rate of ``CliNR_{t,r}`` applied to ``circuit``.

    The chunks are counted exactly rather than assumed equal, so the bound is
    the sum of per-chunk bounds.

    The teleportation byproduct is tracked as a Pauli frame and resolved for
    free at readout, so by default it costs nothing. Setting ``noisy_q`` models
    a deployment that instead resolves the frame into a live corrected state by
    applying a Pauli once at the end of the chain, adding ``n`` single-qubit
    fault locations. These occur after the last check, so they are undetectable
    and do not affect any chunk's restart probability.

    Args:
        circuit: The Clifford circuit to be implemented.
        n: Number of data qubits.
        t: Number of chunks.
        r: Number of stabilizer checks per chunk.
        noise: The four rates.
        noisy_checks: If True, charge each check the fault locations of the
            ancilla gadget; if False, treat it as an idealized projective
            measurement with no fault locations, matching ``noisy_checks`` in
            :func:`clinr.build_clinr_tr`.
        restricted_checks: If True, price the weight-``(n+1)`` family of
            :func:`clinr.restricted_family`. That family generates the group
            but is not uniform in it, so the per-check miss probability is
            bounded by ``1 - 2/(3n)`` rather than ``1/2``.
        noisy_q: If True, add the one-time frame-resolution cost.
        native_plus: If True, price ``|+>`` as a single native preparation,
            the paper's Sec. II convention, dropping ``n`` single-qubit fault
            locations per chunk; matching ``native_plus`` in
            :func:`clinr.build_clinr_tr`.

    Returns:
        An upper bound on the logical error rate, not clipped to ``[0, 1]``.

    Raises:
        ValueError: if ``n < 1`` or ``r < 0``, or from
            :func:`clinr.chunk_circuit` if ``t < 1`` or ``circuit`` is wider
            than ``n``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    family = _detection(n, restricted_checks)
    chunks = clinr.chunk_circuit(circuit, n, t)
    counts = [accounting.gate_counts(c) for c in chunks]
    return _plog_from_counts(
        counts, noise, n, r, family, noisy_checks, noisy_q, native_plus
    )


def overhead_bound(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    *,
    noisy_checks: bool = True,
    restricted_checks: bool = False,
    noisy_q: bool = False,
    native_plus: bool = False,
) -> float:
    """Bound the gate overhead of ``CliNR_{t,r}`` applied to ``circuit``.

    Each chunk contributes the ``4n`` operations that finish its gate
    teleportation (``n`` controlled-Paulis, ``n`` Hadamards and ``2n``
    measurements) plus an expected ``m / (1 - p_res)`` operations across
    restarts, where ``m`` counts the operations up to and including the last
    check. Every attempt runs all ``r`` checks, so ``m`` is an attempt's exact
    cost and the geometric expectation is exact rather than an upper bound. The paper's ``5n`` also charges the ``n`` Pauli corrections, which
    are tracked classically here and priced by ``noisy_q`` instead. The total
    is divided by the size of ``circuit``.

    Args:
        circuit: The Clifford circuit to be implemented.
        n: Number of data qubits.
        t: Number of chunks.
        r: Number of stabilizer checks per chunk.
        noise: The four rates.
        noisy_checks: If True, charge each check the operations of the ancilla
            gadget; if False, an idealized check costs nothing.
        restricted_checks: If True, price the weight-``(n+1)`` restricted family;
            the worst-case check weight drops from ``2n`` to ``n + 1``.
        noisy_q: If True, add the ``n`` gates that resolve the tracked Pauli
            frame into a live corrected state once at the end of the chain.
        native_plus: If True, price ``|+>`` as a single native preparation,
            dropping ``n`` operations per chunk.

    Returns:
        An upper bound on the expected gate overhead.

    Raises:
        ValueError: if ``circuit`` has no gates, or on the same conditions as
            :func:`plog_bound`.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    _, w = _detection(n, restricted_checks)
    chunks = clinr.chunk_circuit(circuit, n, t)
    counts = [accounting.gate_counts(c) for c in chunks]
    return _overhead_from_counts(
        counts, noise, n, r, w, noisy_checks, noisy_q, native_plus
    )


def choose_tr(
    circuit: stim.Circuit,
    n: int,
    noise: NoiseModel,
    *,
    max_overhead: float,
    r_max: int,
    t_max: int | None = None,
    noisy_checks: bool = True,
    restricted_checks: bool = False,
    noisy_q: bool = False,
    native_plus: bool = False,
) -> tuple[int, int]:
    """Choose ``(t, r)`` minimizing the bound on the logical error rate.

    The search is exhaustive over ``1 <= t <= t_max`` and ``0 <= r <= r_max``,
    restricted to pairs whose gate-overhead bound is at most ``max_overhead``.

    Args:
        circuit: The Clifford circuit to be implemented.
        n: Number of data qubits.
        noise: The four rates.
        max_overhead: Largest acceptable gate-overhead bound.
        r_max: Largest number of checks per chunk to consider, a search cap
            rather than a physical limit.
        t_max: Largest number of chunks to consider. Defaults to the gate
            count of ``circuit``; more chunks than gates are empty and only
            add cost, so the default searches every useful ``t``.
        noisy_checks: If True, charge each check the ancilla gadget's cost in
            both bounds.
        restricted_checks: If True, price the restricted check family in both bounds.
        noisy_q: If True, evaluate both bounds under one-time frame resolution,
            so the objective and the overhead constraint share the assumption.
        native_plus: If True, price ``|+>`` as a single native preparation in
            both bounds.

    Returns:
        The pair ``(t, r)`` with the smallest logical error rate bound among
        those meeting the overhead constraint. A result with ``r == r_max``
        may mean the cap is binding.

    Raises:
        ValueError: if ``r_max < 0``, if ``t_max < 1``, or if no pair meets
            the overhead constraint.
    """
    if r_max < 0:
        raise ValueError(f"r_max must be non-negative, got {r_max}")
    s1_all, s2_all = accounting.gate_counts(circuit)
    size = s1_all + s2_all
    upper_t = size if t_max is None else t_max
    if upper_t < 1:
        raise ValueError(f"t_max must be at least 1, got {upper_t}")

    family = _detection(n, restricted_checks)
    _, w = family
    best: tuple[int, int] | None = None
    best_plog = float("inf")
    for t in range(1, upper_t + 1):
        chunks = clinr.chunk_circuit(circuit, n, t)
        counts = [accounting.gate_counts(c) for c in chunks]
        for r in range(0, r_max + 1):
            if _overhead_from_counts(
                counts, noise, n, r, w, noisy_checks, noisy_q, native_plus
            ) > max_overhead:
                continue
            value = _plog_from_counts(
                counts, noise, n, r, family, noisy_checks, noisy_q,
                native_plus,
            )
            if value < best_plog:
                best_plog, best = value, (t, r)
    if best is None:
        raise ValueError(
            f"no (t, r) achieves gate overhead at most {max_overhead}"
        )
    return best
