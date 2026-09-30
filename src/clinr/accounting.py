"""Operation counts for the circuits :mod:`clinr.protocol` builds.

A :class:`Region` counts the operations of one part of a built circuit by
kind. :mod:`clinr.bounds` turns a region into a fault probability and
:mod:`clinr.estimators` into an operation count, so both read the same
numbers for the same circuit.

Per chunk the protocol has four regions: the resource preparation, the
chunk's own gates, its checks, and the teleportation after the last check.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import stim

from . import protocol


class Region(NamedTuple):
    """Operation counts for one region of a built circuit.

    Parameters
    ----------
    n1 : int
        Single-qubit gates.
    n2 : int
        Two-qubit gates.
    nm : int
        Measurements.
    n_prep : int
        Preparations.
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
    "protected_region",
    "expected_overhead",
]


def _check_n(n: int) -> None:
    """Raise ``ValueError`` unless ``n`` is at least 1."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")


def gate_counts(circuit: stim.Circuit) -> tuple[int, int]:
    """Return ``(s1, s2)``, the numbers of single- and two-qubit gates.

    Parameters
    ----------
    circuit : stim.Circuit
        As accepted by :func:`clinr.protocol.gate_list`, whose gates are the
        ones counted.

    Returns
    -------
    tuple of (int, int)
    """
    s1 = s2 = 0
    for _, targets in protocol.gate_list(circuit):
        if len(targets) == 2:
            s2 += 1
        else:
            s1 += 1
    return s1, s2


def add(*regions: Region) -> Region:
    """Return the elementwise sum of ``regions``, or :data:`ZERO` if none."""
    if not regions:
        return ZERO
    return Region(*(sum(counts) for counts in zip(*regions, strict=True)))


def prep_region(n: int, native_plus: bool) -> Region:
    """Return the region of one resource preparation of ``n`` Bell pairs.

    ``2n`` preparations and ``n`` ``CX`` either way, plus ``n`` Hadamards
    unless ``native_plus``, when ``|+>`` is one ``RX`` rather than ``R``
    then ``H``.

    Parameters
    ----------
    n : int
        Number of Bell pairs. Must be at least 1.
    native_plus : bool

    Returns
    -------
    Region

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    _check_n(n)
    return Region(n1=0 if native_plus else n, n2=n, nm=0, n_prep=2 * n)


def check_region(
    count: int, weight: int, noisy_checks: bool, native_plus: bool
) -> Region:
    """Return the region of ``count`` checks of total weight ``weight``.

    A noisy weight-``w`` check is one preparation, two Hadamards (one under
    ``native_plus``), ``w`` controlled Paulis and one measurement: ``w + 4``
    operations, or ``w + 3``. An idealized check has no fault locations.

    Parameters
    ----------
    count : int
        Number of checks. Must be non-negative.
    weight : int
        Their total weight. Must be at least ``count``, since no check is
        the identity.
    noisy_checks : bool
        If False, the checks are idealized and the region is :data:`ZERO`.
    native_plus : bool

    Returns
    -------
    Region

    Raises
    ------
    ValueError
        If ``count`` is negative or ``weight`` is less than ``count``.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if weight < count:
        raise ValueError(
            f"weight must be at least count = {count}, got {weight}"
        )
    if not noisy_checks:
        return ZERO
    return Region(
        n1=(1 if native_plus else 2) * count,
        n2=weight,
        nm=count,
        n_prep=count,
    )


def teleport_region(n: int) -> Region:
    """Return the region of the teleportation after a chunk's last check.

    ``n`` ``CX``, ``n`` Hadamards and ``2n`` measurements: the ``4n`` of the
    paper's ``5n`` that remain when the Pauli correction is tracked
    classically ([1]_, proof of Lemma 1).

    Parameters
    ----------
    n : int
        Number of data qubits. Must be at least 1.

    Returns
    -------
    Region

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    _check_n(n)
    return Region(n1=n, n2=n, nm=2 * n, n_prep=0)


def protected_region(prep: Region, s1: int, s2: int, checks: Region) -> Region:
    """Return the region of one attempt at a chunk, through its last check.

    Parameters
    ----------
    prep : Region
        The resource preparation, from :func:`prep_region`.
    s1 : int
        Single-qubit gates in the chunk. Must be non-negative.
    s2 : int
        Two-qubit gates in the chunk. Must be non-negative.
    checks : Region
        All of the chunk's checks, from :func:`check_region`.

    Returns
    -------
    Region

    Raises
    ------
    ValueError
        If ``s1`` or ``s2`` is negative.
    """
    if s1 < 0 or s2 < 0:
        raise ValueError(
            f"gate counts must be non-negative, got s1 = {s1}, s2 = {s2}"
        )
    return add(prep, Region(n1=s1, n2=s2, nm=0, n_prep=0), checks)


def expected_overhead(
    counts: list[tuple[int, int]],
    n: int,
    prep: Region,
    check_regions: list[Region],
    acceptance: list[float],
) -> float:
    """Return the expected number of operations per gate of the circuit.

    Every attempt at chunk ``k`` runs all of its checks, so it costs the
    ``m_k`` operations of its :func:`protected_region`, and the number of
    attempts is geometric with mean ``1 / q_k``. Each chunk also pays its
    teleportation once. This is Eq. (B4) of [1]_ with the acceptance ``q_k`` in
    place of ``1 - p_res`` and the teleportation's ``4n`` in place of
    ``5n``. The total is divided by the number of gates across the chunks.

    Parameters
    ----------
    counts : list of tuple of (int, int)
        ``(s1, s2)`` for each chunk, in chunk order, not all zero.
    n : int
        Number of data qubits. Must be at least 1.
    prep : Region
        The resource preparation, the same for every chunk.
    check_regions : list of Region
        Each chunk's checks, in chunk order.
    acceptance : list of float
        Each chunk's acceptance probability ``q_k``, in chunk order, in
        [0, 1]. Analytic bounds and measured frequencies are both accepted;
        the result is a bound or an estimate accordingly.

    Returns
    -------
    float
        The expected overhead, or ``math.inf`` if any ``q_k`` is zero.

    Raises
    ------
    ValueError
        If ``counts``, ``check_regions`` and ``acceptance`` differ in
        length, if the chunks hold no gates, or if an acceptance probability
        lies outside [0, 1].
    """
    lengths = {len(counts), len(check_regions), len(acceptance)}
    if len(lengths) > 1:
        raise ValueError(
            "counts, check_regions and acceptance must have one entry per "
            f"chunk, got {len(counts)}, {len(check_regions)} and "
            f"{len(acceptance)}"
        )
    size = sum(s1 + s2 for s1, s2 in counts)
    if size == 0:
        raise ValueError("the chunks must hold at least one gate")
    if not all(0.0 <= q <= 1.0 for q in acceptance):
        raise ValueError(
            f"acceptance probabilities must lie in [0, 1], got {acceptance}"
        )
    tail = sum(teleport_region(n))
    expected = 0.0
    for (s1, s2), checks, q in zip(
        counts, check_regions, acceptance, strict=True
    ):
        if q == 0.0:
            return math.inf
        expected += tail + sum(protected_region(prep, s1, s2, checks)) / q
    return expected / size
