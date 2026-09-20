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

import clinr
from noise_models import NoiseModel

__all__ = [
    "gate_counts",
    "fault_prob",
    "plog_bound",
    "overhead_bound",
    "choose_tr",
]


def gate_counts(circuit: stim.Circuit) -> tuple[int, int]:
    """Return ``(s1, s2)``, the single- and two-qubit gate counts.

    Counted over :func:`clinr.gate_list`, so identities are ignored and fused
    instructions are split.

    Raises:
        ValueError: propagated from :func:`clinr.gate_list` if an instruction
            is malformed.
    """
    s1 = s2 = 0
    for _, targets in clinr.gate_list(circuit):
        if len(targets) == 2:
            s2 += 1
        else:
            s1 += 1
    return s1, s2


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


_ZERO = (0, 0, 0, 0)


def _check_region(
    count: int, weight: int, noisy_checks: bool, native_plus: bool
) -> tuple[int, int, int, int]:
    """Operation counts ``(n1, n2, nm, n_prep)`` for ``count`` checks.

    ``weight`` is the total weight of those checks: a weight-``w`` check is
    measured by a gadget of one preparation, ``w`` controlled-Paulis, one
    measurement, and two single-qubit gates, or one under ``native_plus``,
    where an ``RX`` replaces the reset and the Hadamard that follows it. An
    idealized check is a projective measurement with no fault locations, so
    every count is zero.
    """
    if not noisy_checks:
        return (0, 0, 0, 0)
    return ((1 if native_plus else 2) * count, weight, count, count)


def _prep_region(n: int, native_plus: bool) -> tuple[int, int, int, int]:
    """Operation counts ``(n1, n2, nm, n_prep)`` for one resource preparation.

    ``2n`` resets and ``n`` controlled-Paulis either way. ``|+>`` costs ``n``
    further single-qubit gates as ``R`` then ``H``, and none as a native
    ``RX``.
    """
    return (0 if native_plus else n, n, 0, 2 * n)


def _protected_counts(
    prep: tuple[int, int, int, int],
    s1: int,
    s2: int,
    checks: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Operation counts for one chunk up to and including its last check.

    Args:
        prep: Counts for the resource preparation (:func:`_prep_region`).
        s1: Single-qubit gate count of the chunk.
        s2: Two-qubit gate count of the chunk.
        checks: Counts for all of the chunk's checks (:func:`_check_region`).
    """
    p1, p2, pm, pp = prep
    c1, c2, cm, cp = checks
    return (p1 + s1 + c1, p2 + s2 + c2, pm + cm, pp + cp)


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
    w: int,
    miss: float,
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Bound the logical error rate contributed by one chunk.

    Regions, for the circuit as built, as ``(n1, n2, nm, n_prep)``: preparation
    (:func:`_prep_region`) together with application of the chunk
    (``s1``, ``s2``, ``0``, ``0``), each of ``r`` checks
    (:func:`_check_region`), and the teleportation (``n``, ``n``, ``2n``,
    ``0``). ``miss`` is the probability that one check fails to detect a
    damaging fault.

    A fault in check ``i``'s own gadget can only be caught by checks
    ``i+1..r``, so those faults contribute ``(1 - miss**r) / (1 - miss)``
    times one check's fault probability. That factor is at most ``2`` for
    uniform draws and at most ``3n/2`` for the restricted family, and it
    vanishes at ``r = 0``.
    """
    prep = _prep_region(n, native_plus)
    prep_fault = fault_prob(noise, *_protected_counts(prep, s1, s2, _ZERO))
    check_fault = fault_prob(
        noise, *_check_region(1, w, noisy_checks, native_plus)
    )
    teleport_fault = fault_prob(noise, n, n, 2 * n, 0)

    escapes = (1.0 - miss**r) / (1.0 - miss)
    numerator = prep_fault * miss**r + check_fault * escapes + teleport_fault
    no_restart = _no_fault_prob(
        noise,
        *_protected_counts(
            prep, s1, s2, _check_region(r, r * w, noisy_checks, native_plus)
        ),
    )
    if no_restart == 0.0:
        return math.inf
    return numerator / no_restart


def _plog_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    n: int,
    r: int,
    w: int,
    miss: float,
    noisy_checks: bool,
    noisy_q: bool,
    native_plus: bool,
) -> float:
    """Sum the per-chunk bounds for chunks with the given gate counts."""
    total = 0.0
    for s1, s2 in counts:
        total += _chunk_plog_bound(
            noise, n, s1, s2, r, w, miss, noisy_checks, native_plus
        )
    if noisy_q:
        total += fault_prob(noise, n, 0, 0, 0)
    return total


def _overhead_from_parts(
    counts: list[tuple[int, int]],
    n: int,
    prep: tuple[int, int, int, int],
    regions: list[tuple[int, int, int, int]],
    no_restarts: list[float],
    noisy_q: bool,
    size: int,
) -> float:
    """Expected operations per gate, given a per-chunk no-restart probability.

    ``prep`` is the resource preparation's counts (:func:`_prep_region`),
    shared by every chunk. ``counts``, ``regions`` and ``no_restarts`` are in
    chunk order. The no-restart probabilities may be bounded analytically or
    measured, which is the only difference between the analytic and empirical
    gate overheads.

    Raises:
        ValueError: if the three sequences differ in length.
    """
    expected = 0.0
    for (s1, s2), region, no_restart in zip(
        counts, regions, no_restarts, strict=True
    ):
        if no_restart == 0.0:
            return math.inf
        ops = sum(_protected_counts(prep, s1, s2, region))
        expected += 4 * n + ops / no_restart
    if noisy_q:
        expected += n
    return expected / size


def _overhead_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    n: int,
    r: int,
    w: int,
    noisy_checks: bool,
    noisy_q: bool,
    native_plus: bool,
    size: int,
) -> float:
    """Expected operations per gate for chunks with the given gate counts."""
    prep = _prep_region(n, native_plus)
    region = _check_region(r, r * w, noisy_checks, native_plus)
    regions = [region] * len(counts)
    no_restarts = [
        _no_fault_prob(noise, *_protected_counts(prep, s1, s2, region))
        for s1, s2 in counts
    ]
    return _overhead_from_parts(
        counts, n, prep, regions, no_restarts, noisy_q, size
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
    miss, w = _detection(n, restricted_checks)
    counts = [gate_counts(c) for c in clinr.chunk_circuit(circuit, n, t)]
    return _plog_from_counts(
        counts, noise, n, r, w, miss, noisy_checks, noisy_q, native_plus
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
    s1_all, s2_all = gate_counts(circuit)
    size = s1_all + s2_all
    if size == 0:
        raise ValueError("circuit must contain at least one gate")

    counts = [gate_counts(c) for c in clinr.chunk_circuit(circuit, n, t)]
    return _overhead_from_counts(
        counts, noise, n, r, w, noisy_checks, noisy_q, native_plus, size
    )


def choose_tr(
    circuit: stim.Circuit,
    n: int,
    noise: NoiseModel,
    *,
    max_overhead: float = 2.0,
    t_max: int | None = None,
    r_max: int = 20,
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
        t_max: Largest number of chunks to consider. Defaults to the gate
            count of ``circuit``.
        r_max: Largest number of checks per chunk to consider. A search cap,
            not a physical limit. The optimum sits at roughly
            ``log_miss(floor * c / prep)``, where ``c`` is the per-check
            fault probability and ``floor`` the ``r``-independent part of the
            numerator, so under uniform draws (``miss = 1/2``) it is a base-2
            logarithm of small quantities and stays in single digits. Under
            restricted draws ``miss = 1 - 2/(3n)``, which multiplies that
            logarithm by about ``1.5n``; the default is already too small at
            ``n = 3``.
        noisy_checks: If True, charge each check the ancilla gadget's cost in
            both bounds.
        restricted_checks: If True, price the restricted check family in both bounds.
        noisy_q: If True, evaluate both bounds under one-time frame resolution,
            so the objective and the overhead constraint share the assumption.
        native_plus: If True, price ``|+>`` as a single native preparation in
            both bounds.

    Returns:
        The pair ``(t, r)`` with the smallest logical error rate bound among
        those meeting the overhead constraint.

    Raises:
        ValueError: if ``r_max < 0``, if ``t_max < 1``, or if no pair meets
            the overhead constraint.
    """
    if r_max < 0:
        raise ValueError(f"r_max must be non-negative, got {r_max}")
    s1_all, s2_all = gate_counts(circuit)
    size = s1_all + s2_all
    upper_t = size if t_max is None else t_max
    if upper_t < 1:
        raise ValueError(f"t_max must be at least 1, got {upper_t}")

    miss, w = _detection(n, restricted_checks)
    best: tuple[int, int] | None = None
    best_plog = float("inf")
    for t in range(1, upper_t + 1):
        counts = [gate_counts(c) for c in clinr.chunk_circuit(circuit, n, t)]
        for r in range(0, r_max + 1):
            if _overhead_from_counts(
                counts, noise, n, r, w, noisy_checks, noisy_q, native_plus,
                size,
            ) > max_overhead:
                continue
            value = _plog_from_counts(
                counts, noise, n, r, w, miss, noisy_checks, noisy_q,
                native_plus,
            )
            if value < best_plog:
                best_plog, best = value, (t, r)
    if best is None:
        raise ValueError(
            f"no (t, r) achieves gate overhead at most {max_overhead}"
        )
    return best
