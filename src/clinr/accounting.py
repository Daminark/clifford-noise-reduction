"""Operation counts for the circuits :mod:`clinr.protocol` builds.

A :class:`Region` counts the operations of one part of a built circuit by
kind. :mod:`clinr.bounds` turns a region into a fault probability and
:mod:`clinr.estimators` into an operation count, so both read the same
numbers for the same circuit.

Per chunk the protocol of [1]_ has four regions: the resource preparation,
the chunk's own gates, its checks, and the teleportation after the last
check. CZNR, its variant for circuits of ``CZ`` gates ([1]_, App. E), has
the same four, with a smaller preparation and teleportation. A
:class:`Scheme` holds the two that differ.

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


#: The empty region, which :func:`add` returns for no regions.
ZERO = Region(0, 0, 0, 0)

__all__ = [
    "Region",
    "ZERO",
    "gate_counts",
    "add",
    "prep_region",
    "check_region",
    "teleport_region",
    "cznr_prep_region",
    "cznr_teleport_region",
    "Scheme",
    "clinr_scheme",
    "cznr_scheme",
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


def check_region(count: int, weight: int, native_plus: bool) -> Region:
    """Return the region of ``count`` checks of total weight ``weight``.

    A weight-``w`` check is one preparation, two Hadamards (one under
    ``native_plus``), ``w`` controlled Paulis and one measurement: ``w + 4``
    operations, or ``w + 3``. The count is the same whether or not the
    checks are noisy; noiseless checks only have no fault locations.

    Parameters
    ----------
    count : int
        Number of checks. Must be non-negative.
    weight : int
        Their total weight. Must be non-negative.
    native_plus : bool

    Returns
    -------
    Region

    Raises
    ------
    ValueError
        If ``count`` or ``weight`` is negative.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if weight < 0:
        raise ValueError(f"weight must be non-negative, got {weight}")
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
    classically (arXiv:2407.06583, proof of Lemma 1).

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


def cznr_prep_region(n: int, native_plus: bool) -> Region:
    """Return the region of one graph-state preparation's ``|+>`` states.

    ``n`` preparations, plus ``n`` Hadamards unless ``native_plus``. The
    graph state's ``CZ`` gates are the chunk's own and are counted with it.

    Parameters
    ----------
    n : int
        Number of qubits of the graph state. Must be at least 1.
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
    return Region(n1=0 if native_plus else n, n2=0, nm=0, n_prep=n)


def cznr_teleport_region(n: int) -> Region:
    """Return the region of CZNR's one-bit teleportation of ``n`` qubits.

    ``n`` ``CX`` and ``n`` measurements: the ``2n`` of the paper's ``3n``
    (arXiv:2407.06583, App. E) that remain when the Pauli correction is
    tracked classically.

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
    return Region(n1=0, n2=n, nm=n, n_prep=0)


class Scheme(NamedTuple):
    """The regions a scheme adds around each chunk: CliNR's or CZNR's.

    Parameters
    ----------
    prep : Region
        The resource preparation, the same for every chunk.
    teleport : Region
        The teleportation after a chunk's last check.
    """

    prep: Region
    teleport: Region


def clinr_scheme(n: int, native_plus: bool) -> Scheme:
    """Return CliNR's regions on ``n`` data qubits.

    :func:`prep_region` and :func:`teleport_region`.

    Parameters
    ----------
    n : int
        Number of data qubits. Must be at least 1.
    native_plus : bool

    Returns
    -------
    Scheme
    """
    return Scheme(prep_region(n, native_plus), teleport_region(n))


def cznr_scheme(n: int, native_plus: bool) -> Scheme:
    """Return CZNR's regions on ``n`` data qubits.

    :func:`cznr_prep_region` and :func:`cznr_teleport_region`.

    Parameters
    ----------
    n : int
        Number of data qubits. Must be at least 1.
    native_plus : bool

    Returns
    -------
    Scheme
    """
    return Scheme(cznr_prep_region(n, native_plus), cznr_teleport_region(n))


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
    scheme: Scheme,
    check_regions: list[Region],
    acceptance: list[float],
) -> float:
    """Return the expected number of operations per gate of the circuit.

    Every attempt at chunk ``k`` runs all of its checks, so it costs the
    ``m_k`` operations of its :func:`protected_region`, and the number of
    attempts is geometric with mean ``1 / q_k``. Each chunk also pays its
    teleportation once. This is Eq. (B4) of arXiv:2407.06583 with the
    acceptance ``q_k`` in place of ``1 - p_res`` and the scheme's
    teleportation, ``4n`` for CliNR, in place of ``5n``. The total is
    divided by the number of gates across the chunks.

    Parameters
    ----------
    counts : list of tuple of (int, int)
        ``(s1, s2)`` for each chunk, in chunk order, not all zero.
    scheme : Scheme
        The preparation and teleportation of every chunk, from
        :func:`clinr_scheme` or :func:`cznr_scheme`.
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
    tail = sum(scheme.teleport)
    expected = 0.0
    for (s1, s2), checks, q in zip(
        counts, check_regions, acceptance, strict=True
    ):
        if q == 0.0:
            return math.inf
        expected += (
            tail + sum(protected_region(scheme.prep, s1, s2, checks)) / q
        )
    return expected / size
