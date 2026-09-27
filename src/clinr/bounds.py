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
from typing import NamedTuple

import stim

from . import accounting, protocol
from .noise_models import NoiseModel

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


class _Family(NamedTuple):
    """A chunk's check family, valid only for the ``r`` it was computed at.

    Attributes:
        r: Number of checks per chunk.
        miss: Worst-case probability that a damaging resource fault commutes
            with all ``r`` checks.
        escapes: Summed over the check gadgets, the probability that a fault
            arising in that gadget commutes with every later check.
        weight: Worst-case weight of one check.
    """

    r: int
    miss: float
    escapes: float
    weight: int


def _check_r(n: int, r: int, name: str = "r") -> None:
    """Reject a width or check count the bounds are not defined for.

    ``name`` appears in the message, for callers whose parameter is not
    literally ``r``. Checks are drawn with replacement, so ``r`` has no upper
    limit.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if r < 0:
        raise ValueError(f"{name} must be non-negative, got {r}")


def _check_weight(n: int, restricted_checks: bool) -> int:
    """Worst-case weight of one check from the selected family."""
    return n + 1 if restricted_checks else 2 * n


def _detection(
    n: int, r: int, restricted_checks: bool
) -> tuple[float, float]:
    """``(miss, escapes)`` for ``r`` independent draws.

    The draws are independent random variables, so the misses multiply, and a
    fault arising in gadget ``j`` faces ``r - j`` genuinely fresh draws even
    when that fault's residual depends on check ``j`` itself. The per-draw
    miss is exact for what :func:`checks.draw_checks` samples: the full group
    rejects the identity, and the restricted family's worst case is a fault
    visible to a single Bell triple.
    """
    if restricted_checks:
        q = (3 * n - 2) / (3 * n)
    else:
        q = (2 ** (2 * n - 1) - 1) / (2 ** (2 * n) - 1)
    miss, escapes = 1.0, 0.0
    for _ in range(r):
        escapes += miss
        miss *= q
    return miss, escapes


def _check_family(n: int, r: int, restricted_checks: bool) -> _Family:
    """Return the detection probabilities and weight of ``r`` checks.

    ``miss`` and ``escapes`` are exact for the draws
    :func:`checks.draw_checks` makes.
    """
    miss, escapes = _detection(n, r, restricted_checks)
    return _Family(r, miss, escapes, _check_weight(n, restricted_checks))


def _chunk_plog_bound(
    noise: NoiseModel,
    n: int,
    s1: int,
    s2: int,
    family: _Family,
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Bound the logical error rate contributed by one chunk.

    Regions are those of :mod:`accounting`: the resource preparation together
    with the chunk's own gates, each of ``r`` checks, and the teleportation.
    ``family`` is :func:`_check_family`, which carries the ``r`` its
    probabilities were computed at, so no caller can pair them wrongly.

    A fault in check ``i``'s own gadget can only be caught by checks
    ``i+1..r``, which is what ``escapes`` sums; it vanishes at ``r = 0`` and
    is at most ``2`` for uniform draws.
    """
    r, miss, escapes, w = family
    prep = accounting.prep_region(n, native_plus)
    resource = accounting.add(
        prep, accounting.Region(n1=s1, n2=s2, nm=0, n_prep=0)
    )
    resource_fault = fault_prob(noise, *resource)
    check_fault = fault_prob(
        noise, *accounting.check_region(1, w, noisy_checks, native_plus)
    )
    teleport_fault = fault_prob(noise, *accounting.teleport_region(n))

    numerator = (
        resource_fault * miss + check_fault * escapes + teleport_fault
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
    family: _Family,
    noisy_checks: bool,
    noisy_q: bool,
    native_plus: bool,
) -> float:
    """Sum the per-chunk bounds for chunks with the given gate counts."""
    total = 0.0
    for s1, s2 in counts:
        total += _chunk_plog_bound(
            noise, n, s1, s2, family, noisy_checks, native_plus
        )
    if noisy_q:
        total += fault_prob(noise, *accounting.frame_region(n))
    return total


def _overhead_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    n: int,
    family: _Family,
    noisy_checks: bool,
    noisy_q: bool,
    native_plus: bool,
) -> float:
    """Bound the expected operations per gate for chunks with these counts.

    Every chunk is charged worst-case check weight, and its acceptance
    probability is replaced by the probability that nothing in its protected
    region faults, which is a lower bound on it.
    """
    r, _, _, w = family
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
            :func:`protocol.build_clinr_tr`.
        restricted_checks: If True, price the weight-``(n+1)`` family of
            :func:`checks.restricted_family`, whose ``3n`` elements detect a
            damaging fault less often at a given ``r`` than draws from the
            whole group.
        noisy_q: If True, add the one-time frame-resolution cost.
        native_plus: If True, price ``|+>`` as a single native preparation,
            the paper's Sec. II convention, dropping ``n`` single-qubit fault
            locations per chunk; matching ``native_plus`` in
            :func:`protocol.build_clinr_tr`.

    Returns:
        An upper bound on the logical error rate, not clipped to ``[0, 1]``.

    Raises:
        ValueError: if ``n < 1`` or ``r < 0``, or from
            :func:`protocol.chunk_circuit` if ``t < 1`` or ``circuit`` is
            wider than ``n``.
    """
    _check_r(n, r)
    family = _check_family(n, r, restricted_checks)
    chunks = protocol.chunk_circuit(circuit, n, t)
    counts = [accounting.gate_counts(c) for c in chunks]
    return _plog_from_counts(
        counts, noise, n, family, noisy_checks, noisy_q, native_plus
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
        restricted_checks: If True, price the weight-``(n+1)`` restricted
            family; the worst-case check weight drops from ``2n`` to
            ``n + 1``.
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
    _check_r(n, r)
    family = _check_family(n, r, restricted_checks)
    chunks = protocol.chunk_circuit(circuit, n, t)
    counts = [accounting.gate_counts(c) for c in chunks]
    return _overhead_from_counts(
        counts, noise, n, family, noisy_checks, noisy_q, native_plus
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
        restricted_checks: If True, price the restricted check family in
            both bounds.
        noisy_q: If True, evaluate both bounds under one-time frame resolution,
            so the objective and the overhead constraint share the assumption.
        native_plus: If True, price ``|+>`` as a single native preparation in
            both bounds.

    Returns:
        The pair ``(t, r)`` with the smallest logical error rate bound among
        those meeting the overhead constraint. A result with ``r == r_max``
        may mean the cap is binding.

    Raises:
        ValueError: if ``n < 1``, if ``r_max < 0``, if ``t_max < 1``, or if
            no pair meets the overhead constraint.
    """
    _check_r(n, r_max, "r_max")
    s1_all, s2_all = accounting.gate_counts(circuit)
    size = s1_all + s2_all
    upper_t = size if t_max is None else t_max
    if upper_t < 1:
        raise ValueError(f"t_max must be at least 1, got {upper_t}")

    families = [
        _check_family(n, r, restricted_checks) for r in range(r_max + 1)
    ]
    best: tuple[int, int] | None = None
    best_plog = float("inf")
    for t in range(1, upper_t + 1):
        chunks = protocol.chunk_circuit(circuit, n, t)
        counts = [accounting.gate_counts(c) for c in chunks]
        for r, family in enumerate(families):
            if _overhead_from_counts(
                counts, noise, n, family, noisy_checks, noisy_q, native_plus
            ) > max_overhead:
                continue
            value = _plog_from_counts(
                counts, noise, n, family, noisy_checks, noisy_q, native_plus
            )
            if value < best_plog:
                best_plog, best = value, (t, r)
    if best is None:
        raise ValueError(
            f"no (t, r) achieves gate overhead at most {max_overhead}"
        )
    return best
