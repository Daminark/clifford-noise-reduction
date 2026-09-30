"""Analytic bounds on the logical error rate and gate overhead of CliNR.

:func:`plog_bound` and :func:`overhead_bound` rederive Theorem 1 of [1]_
for the circuits :mod:`clinr.protocol` builds, and :func:`choose_tr` picks
``(t, r)`` by them. The bounds are deterministic.

They differ from Theorem 1 in what they count, not in how:

- four rates, one per kind of operation (:mod:`clinr.accounting`), where
  the paper has one;
- the exact size of each chunk, where the paper charges every chunk the
  largest;
- ``4n`` operations to finish a teleportation, where the paper's ``5n``
  includes applying the Pauli correction, which is tracked here instead;
- the exact probability that ``r`` checks all miss a damaging fault, and
  the sum over gadgets of the probability that a fault arising in one
  escapes every later check, for whichever family
  :func:`clinr.checks.draw_checks` draws from, where the paper has ``2^-r``
  and ``2``.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import stim

from . import accounting, protocol
from .accounting import Region
from .noise_models import NoiseModel

__all__ = [
    "plog_bound",
    "overhead_bound",
    "choose_tr",
]


def _no_fault_prob(noise: NoiseModel, region: Region) -> float:
    """Return the probability that no operation of ``region`` faults."""
    assert min(region) >= 0, f"negative counts in {region}"
    return (
        (1.0 - noise.p1) ** region.n1
        * (1.0 - noise.p2) ** region.n2
        * (1.0 - noise.p_m) ** region.nm
        * (1.0 - noise.p_prep) ** region.n_prep
    )


def _fault_prob(noise: NoiseModel, region: Region) -> float:
    """Return the probability that some operation of ``region`` faults.

    The paper's ``g_p(x) = 1 - (1 - p)^x``, with one rate per kind.
    """
    return 1.0 - _no_fault_prob(noise, region)


class _Family(NamedTuple):
    """The detection numbers of ``r`` checks, valid only at that ``r``.

    ``miss`` is the worst-case probability that a damaging fault in the
    resource commutes with all ``r`` checks; ``escapes`` sums, over the
    gadgets, the probability that a fault arising in one commutes with every
    later check; ``weight`` is the worst-case weight of one check.
    """

    r: int
    miss: float
    escapes: float
    weight: int


def _check_sizes(n: int, r: int, name: str = "r") -> None:
    """Raise ``ValueError`` unless ``n >= 1`` and ``r >= 0``, naming ``r``."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if r < 0:
        raise ValueError(f"{name} must be non-negative, got {r}")


def _check_family(n: int, r: int, restricted_checks: bool) -> _Family:
    """Return the :class:`_Family` of ``r`` draws from the chosen family.

    The per-draw worst-case miss ``q`` is ``1 - 2/(3n)`` for the restricted
    family and ``(2^(2n-1) - 1)/(2^(2n) - 1)`` for the full group without
    the identity; ``miss = q^r`` and ``escapes = 1 + q + ... + q^(r-1)``.
    Requires ``n >= 1`` and ``r >= 0``.
    """
    # Exact for what draw_checks samples. The draws are independent, so the
    # misses multiply, and a fault arising in gadget j faces r - j fresh
    # draws even when its residual depends on check j itself.
    if restricted_checks:
        q = (3 * n - 2) / (3 * n)
        weight = n + 1
    else:
        q = (2 ** (2 * n - 1) - 1) / (2 ** (2 * n) - 1)
        weight = 2 * n
    miss, escapes = 1.0, 0.0
    for _ in range(r):
        escapes += miss
        miss *= q
    return _Family(r, miss, escapes, weight)


def _chunk_plog_bound(
    noise: NoiseModel,
    n: int,
    s1: int,
    s2: int,
    family: _Family,
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Return the bound on one chunk's logical error rate.

    Lemma 1 of arXiv:2407.06583, Eq. (B1), with the regions of
    :mod:`clinr.accounting` and ``family`` in place of ``2^-r`` and ``2``.
    """
    r, miss, escapes, w = family
    prep = accounting.prep_region(n, native_plus)
    resource = accounting.add(prep, Region(n1=s1, n2=s2, nm=0, n_prep=0))
    one_check = accounting.check_region(1, w, noisy_checks, native_plus)
    numerator = (
        _fault_prob(noise, resource) * miss
        + _fault_prob(noise, one_check) * escapes
        + _fault_prob(noise, accounting.teleport_region(n))
    )
    checks = accounting.check_region(r, r * w, noisy_checks, native_plus)
    no_fault = _no_fault_prob(
        noise, accounting.protected_region(prep, s1, s2, checks)
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
    native_plus: bool,
) -> float:
    """Return the sum of the chunk bounds for chunks of these gate counts."""
    total = 0.0
    for s1, s2 in counts:
        total += _chunk_plog_bound(
            noise, n, s1, s2, family, noisy_checks, native_plus
        )
    return total


def _overhead_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    n: int,
    family: _Family,
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Return the overhead bound for chunks of these gate counts.

    Each chunk is charged the worst-case check weight, and its acceptance
    probability the probability that nothing in its protected region
    faults, which is at most the true acceptance.
    """
    r, _, _, w = family
    prep = accounting.prep_region(n, native_plus)
    checks = accounting.check_region(r, r * w, noisy_checks, native_plus)
    no_fault = [
        _no_fault_prob(
            noise, accounting.protected_region(prep, s1, s2, checks)
        )
        for s1, s2 in counts
    ]
    return accounting.expected_overhead(
        counts, n, prep, [checks] * len(counts), no_fault
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
    native_plus: bool = False,
) -> float:
    """Bound the logical error rate of ``CliNR_{t,r}`` applied to ``circuit``.

    The sum over chunks of the bound of Lemma 1 of arXiv:2407.06583, Eq.
    (B1), with each chunk counted exactly, where Theorem 1 multiplies the
    largest chunk's bound by ``t``.

    Parameters
    ----------
    circuit : stim.Circuit
        The Clifford circuit ``C``, as accepted by
        :func:`clinr.protocol.chunk_circuit`, on qubits ``[0, n)``.
    n : int
        Number of data qubits. Must be at least 1.
    t : int
        Number of chunks. Must be at least 1.
    r : int
        Number of checks per chunk. Must be non-negative.
    noise : NoiseModel
    noisy_checks : bool, optional
        If True, charge each check the fault locations of the ancilla
        gadget; if False, none, as for ``noisy_checks`` in
        :func:`clinr.protocol.build_clinr_tr`.
    restricted_checks : bool, optional
        If True, price draws from :func:`clinr.checks.restricted_family`;
        if False, from the full stabilizer group.
    native_plus : bool, optional
        If True, price ``|+>`` as one preparation, as in
        :func:`clinr.protocol.build_clinr_tr`.

    Returns
    -------
    float
        An upper bound on the logical error rate of the circuit that
        :func:`clinr.protocol.build_clinr_tr` builds with the same options,
        averaged over the checks :func:`clinr.checks.draw_checks` draws.
        Not clipped to [0, 1], and infinite when some chunk can never be
        accepted.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``r < 0``.
    """
    _check_sizes(n, r)
    family = _check_family(n, r, restricted_checks)
    chunks = protocol.chunk_circuit(circuit, n, t)
    counts = [accounting.gate_counts(chunk) for chunk in chunks]
    return _plog_from_counts(
        counts, noise, n, family, noisy_checks, native_plus
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
    native_plus: bool = False,
) -> float:
    """Bound the gate overhead of ``CliNR_{t,r}`` applied to ``circuit``.

    :func:`clinr.accounting.expected_overhead` with every check at its
    family's worst-case weight and each chunk's acceptance probability
    replaced by the probability that nothing in its protected region
    faults, which is at most the true acceptance.

    Parameters
    ----------
    circuit : stim.Circuit
        The Clifford circuit ``C``, as accepted by
        :func:`clinr.protocol.chunk_circuit`, on qubits ``[0, n)``, with at
        least one gate.
    n, t, r, noise, noisy_checks, restricted_checks, native_plus
        As in :func:`plog_bound`.

    Returns
    -------
    float
        An upper bound on the expected number of operations per gate of
        ``circuit``, infinite when some chunk can never be accepted.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``r < 0``.
    """
    _check_sizes(n, r)
    family = _check_family(n, r, restricted_checks)
    chunks = protocol.chunk_circuit(circuit, n, t)
    counts = [accounting.gate_counts(chunk) for chunk in chunks]
    return _overhead_from_counts(
        counts, noise, n, family, noisy_checks, native_plus
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
    native_plus: bool = False,
) -> tuple[int, int]:
    """Return the ``(t, r)`` with the least logical error rate bound.

    The search is exhaustive over ``1 <= t <= t_max`` and
    ``0 <= r <= r_max``, among the pairs whose overhead bound is at most
    ``max_overhead``; ties go to the smaller ``t``, then the smaller ``r``.

    Parameters
    ----------
    circuit : stim.Circuit
        The Clifford circuit ``C``, as accepted by
        :func:`clinr.protocol.chunk_circuit`, on qubits ``[0, n)``, with at
        least one gate.
    n : int
        Number of data qubits. Must be at least 1.
    noise : NoiseModel
    max_overhead : float
        Largest acceptable overhead bound. Must be positive.
    r_max : int
        Largest number of checks per chunk to try. Must be non-negative.
    t_max : int, optional
        Largest number of chunks to try, at least 1. Defaults to the gate
        count of ``circuit``, beyond which chunks are empty and only add
        cost.
    noisy_checks, restricted_checks, native_plus : bool, optional
        As in :func:`plog_bound`, applied to both bounds.

    Returns
    -------
    tuple of (int, int)
        ``(t, r)``. A result with ``r == r_max`` may mean the cap binds.

    Raises
    ------
    ValueError
        If ``n < 1``, ``r_max < 0``, ``t_max < 1``, ``max_overhead`` is not
        positive, ``circuit`` has no gates, or no pair meets the overhead
        limit.
    """
    _check_sizes(n, r_max, "r_max")
    if not max_overhead > 0.0:
        raise ValueError(f"max_overhead must be positive, got {max_overhead}")
    size = sum(accounting.gate_counts(circuit))
    if size == 0:
        raise ValueError("circuit must contain at least one gate")
    upper_t = size if t_max is None else t_max
    if upper_t < 1:
        raise ValueError(f"t_max must be at least 1, got {upper_t}")

    families = [
        _check_family(n, r, restricted_checks) for r in range(r_max + 1)
    ]
    best: tuple[int, int] | None = None
    best_plog = math.inf
    for t in range(1, upper_t + 1):
        chunks = protocol.chunk_circuit(circuit, n, t)
        counts = [accounting.gate_counts(chunk) for chunk in chunks]
        for r, family in enumerate(families):
            options = (noise, n, family, noisy_checks, native_plus)
            if _overhead_from_counts(counts, *options) > max_overhead:
                continue
            value = _plog_from_counts(counts, *options)
            if value < best_plog:
                best_plog, best = value, (t, r)
    if best is None:
        raise ValueError(
            f"no (t, r) achieves gate overhead at most {max_overhead}"
        )
    return best
