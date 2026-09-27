"""Operation counts for CliNR as built, shared by bounds and estimators.

A region is a count of operations ``(n1, n2, nm, n_prep)``: single-qubit
gates, two-qubit gates, measurements and preparations. The bounds turn a
region into a fault probability; the gate overhead turns it into an operation
count. Keeping the counts here means both read the same numbers for the same
circuit.

Per chunk the protocol has four regions: the resource preparation, the
chunk's own gates, its checks, and the teleportation after the last check.
Frame resolution, when priced, is one further region for the whole chain.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import stim

from . import protocol


class Region(NamedTuple):
    """Operation counts for one region of the built circuit.

    Attributes:
        n1: Single-qubit gates.
        n2: Two-qubit gates.
        nm: Measurements.
        n_prep: Preparations.
    """

    n1: int
    n2: int
    nm: int
    n_prep: int


ZERO = Region(0, 0, 0, 0)

__all__ = [
    "Region",
    "ZERO",
    "gate_counts",
    "add",
    "prep_region",
    "check_region",
    "teleport_region",
    "frame_region",
    "protected_region",
    "expected_overhead",
]


def gate_counts(circuit: stim.Circuit) -> tuple[int, int]:
    """Return ``(s1, s2)``, the single- and two-qubit gate counts.

    Counted over :func:`protocol.gate_list`, so identities are ignored
    and fused instructions are split.

    Raises:
        ValueError: propagated from :func:`protocol.gate_list` if an
            instruction is malformed.
    """
    s1 = s2 = 0
    for _, targets in protocol.gate_list(circuit):
        if len(targets) == 2:
            s2 += 1
        else:
            s1 += 1
    return s1, s2


def add(*regions: Region) -> Region:
    """Elementwise sum of regions."""
    return Region(*(sum(counts) for counts in zip(*regions, strict=True)))


def prep_region(n: int, native_plus: bool) -> Region:
    """One resource preparation of ``n`` Bell pairs.

    ``2n`` resets and ``n`` controlled-Paulis either way. ``|+>`` costs ``n``
    further single-qubit gates as ``R`` then ``H``, and none as a native
    ``RX``.
    """
    return Region(n1=0 if native_plus else n, n2=n, nm=0, n_prep=2 * n)


def check_region(
    count: int, weight: int, noisy_checks: bool, native_plus: bool
) -> Region:
    """``count`` checks of total weight ``weight``.

    A weight-``w`` check is measured by a gadget of one preparation, ``w``
    controlled-Paulis, one measurement, and two single-qubit gates, or one
    under ``native_plus``, where an ``RX`` replaces the reset and the Hadamard
    that follows it. An idealized check is a projective measurement with no
    fault locations, so every count is zero.
    """
    if not noisy_checks:
        return ZERO
    return Region(
        n1=(1 if native_plus else 2) * count,
        n2=weight,
        nm=count,
        n_prep=count,
    )


def teleport_region(n: int) -> Region:
    """The teleportation after a chunk's last check.

    ``n`` controlled-Paulis, ``n`` Hadamards and ``2n`` measurements. The
    Pauli correction is tracked classically, so it is not part of this region.
    """
    return Region(n1=n, n2=n, nm=2 * n, n_prep=0)


def frame_region(n: int) -> Region:
    """Resolving the tracked Pauli frame once, at the end of the chain."""
    return Region(n1=n, n2=0, nm=0, n_prep=0)


def protected_region(
    prep: Region, s1: int, s2: int, checks: Region
) -> Region:
    """One attempt at a chunk, up to and including its last check.

    Args:
        prep: The resource preparation (:func:`prep_region`).
        s1: Single-qubit gate count of the chunk.
        s2: Two-qubit gate count of the chunk.
        checks: All of the chunk's checks (:func:`check_region`).
    """
    return add(prep, Region(n1=s1, n2=s2, nm=0, n_prep=0), checks)


def expected_overhead(
    counts: list[tuple[int, int]],
    n: int,
    prep: Region,
    check_regions: list[Region],
    acceptance: list[float],
    noisy_q: bool,
) -> float:
    """Expected operations per gate, given each chunk's acceptance probability.

    Every attempt runs all of its chunk's checks, so an attempt at chunk ``k``
    costs exactly ``m_k`` operations and the number of attempts is geometric
    with mean ``1 / q_k``. Each chunk also pays its teleportation once, and
    frame resolution is paid once for the chain when ``noisy_q``. The total is
    divided by the number of gates across the chunks, which is the size of the
    circuit they came from. The acceptance probabilities may be bounded
    analytically or measured, which is the only difference between the
    analytic and empirical gate overheads.

    Args:
        counts: Gate counts ``(s1, s2)`` per chunk, in chunk order.
        n: Number of data qubits.
        prep: The resource preparation, shared by every chunk.
        check_regions: Each chunk's checks, in chunk order.
        acceptance: Each chunk's acceptance probability ``q_k``, in chunk
            order.
        noisy_q: If True, add the one-time frame resolution.

    Returns:
        The expected overhead, or ``math.inf`` if any ``q_k`` is zero.

    Raises:
        ValueError: if the three per-chunk sequences differ in length, or if
            the chunks contain no gates.
    """
    size = sum(s1 + s2 for s1, s2 in counts)
    if size == 0:
        raise ValueError("circuit must contain at least one gate")
    tail = sum(teleport_region(n))
    expected = 0.0
    for (s1, s2), checks, q in zip(
        counts, check_regions, acceptance, strict=True
    ):
        if q == 0.0:
            return math.inf
        expected += tail + sum(protected_region(prep, s1, s2, checks)) / q
    if noisy_q:
        expected += sum(frame_region(n))
    return expected / size
