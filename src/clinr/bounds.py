"""Analytic bounds on the logical error rate and gate overhead of CliNR.

:func:`plog_bound` and :func:`overhead_bound` rederive Theorem 1 of [1]_
for the circuits :mod:`clinr.protocol` builds, and :func:`choose_tr` picks
``(t, r)``, or the direct implementation, by them. The bounds are
deterministic.

They differ from Theorem 1 mostly in what they count:

- four rates, one per kind of operation (:mod:`clinr.accounting`), where
  the paper has one;
- the exact size of each chunk, where the paper charges every chunk the
  largest;
- ``4n`` operations to finish a teleportation, where the paper's ``5n``
  includes applying the Pauli correction, which is tracked here instead
  (``2n`` for CZNR, where the paper has ``3n``);
- the probability that ``r`` checks all miss a damaging fault, and the sum
  over gadgets of the probability that a fault arising in one escapes every
  later check, for whichever family :func:`clinr.checks.draw_checks` draws
  from, where the paper has ``2^-r`` and ``2`` for the full group;
- no conditioning on acceptance when ``r == 0``, since every attempt is
  then accepted.

:func:`cznr_plog_bound` and :func:`cznr_overhead_bound` do the same for
CZNR, rederiving Theorem 3 of [1]_ with the same departures.

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
    "ChunkTerms",
    "direct_plog_bound",
    "plog_bound",
    "plog_terms",
    "overhead_bound",
    "choose_tr",
    "cznr_plog_bound",
    "cznr_overhead_bound",
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
    """``r`` draws from a family of checks.

    ``q`` is the worst-case probability that one draw misses a damaging
    fault, and ``weight`` the worst-case weight of one check.
    """

    r: int
    q: float
    weight: int


class ChunkTerms(NamedTuple):
    """The terms of one chunk's bound on the logical error rate, ``r >= 1``.

    The chunk's bound is
    ``(resource * miss + check * escapes + teleport) / acceptance``, with
    ``miss = q**r`` and ``escapes = 1 + q + ... + q**(r - 1)`` for ``q`` the
    per-check miss.

    Parameters
    ----------
    resource : float
        Probability that some operation of the resource faults.
    check : float
        Probability that some operation of one check gadget faults, at the
        family's worst-case weight; 0 for noiseless checks.
    teleport : float
        Probability that some operation of the teleportation faults.
    acceptance : float
        Probability that no operation of the protected region faults, a
        lower bound on the chunk's acceptance.
    checks : int
        The number of checks ``r``.
    per_check_miss : float
        The worst-case probability ``q`` that one check misses a damaging
        fault: ``1/2`` for uniform checks, ``1 - 2/(3n)`` for restricted ones.
    """

    resource: float
    check: float
    teleport: float
    acceptance: float
    checks: int
    per_check_miss: float

    @property
    def miss(self) -> float:
        """The worst-case probability that all ``r`` checks miss."""
        miss = 1.0
        for _ in range(self.checks):
            miss *= self.per_check_miss
        return miss

    @property
    def escapes(self) -> float:
        """The sum over gadgets of the probability that later checks miss."""
        escapes, miss = 0.0, 1.0
        for _ in range(self.checks):
            escapes += miss
            miss *= self.per_check_miss
        return escapes

    @property
    def bound(self) -> float:
        """The chunk's bound, infinite if no attempt can be accepted."""
        if self.acceptance == 0.0:
            return math.inf
        numerator = (
            self.resource * self.miss
            + self.check * self.escapes
            + self.teleport
        )
        return numerator / self.acceptance


def _check_sizes(n: int, r: int, name: str = "r") -> None:
    """Raise ``ValueError`` unless ``n >= 1`` and ``r >= 0``, naming ``r``."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if r < 0:
        raise ValueError(f"{name} must be non-negative, got {r}")


def _check_family(n: int, r: int, restricted_checks: bool) -> _Family:
    """Return the :class:`_Family` of ``r`` draws from the chosen family.

    The per-draw worst-case miss ``q`` is ``1 - 2/(3n)`` for the restricted
    family and ``1/2`` for the full group, identity included.
    Requires ``n >= 1`` and ``r >= 0``.
    """
    # Exact for what draw_checks samples. The draws are independent, so the
    # misses multiply (ChunkTerms.miss), and a fault arising in gadget j
    # faces r - j fresh draws even when its residual depends on check j
    # itself (ChunkTerms.escapes).
    if restricted_checks:
        return _Family(r, (3 * n - 2) / (3 * n), n + 1)
    return _Family(r, 0.5, 2 * n)


def _faulty(checks: Region, noisy_checks: bool) -> Region:
    """Return the fault locations among ``checks``: none if noiseless."""
    return checks if noisy_checks else accounting.ZERO


def _chunk_plog_bound(
    noise: NoiseModel,
    scheme: accounting.Scheme,
    s1: int,
    s2: int,
    family: _Family,
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Return the bound on one chunk's logical error rate.

    Lemma 1 of arXiv:2407.06583, Eq. (B1), with the regions of
    :mod:`clinr.accounting` and ``family`` in place of ``2^-r`` and ``2``.
    With no checks every attempt is accepted, so the conditioning drops out
    and the bound is the probability that the resource or the teleportation
    faults.
    """
    if family.r == 0:
        resource = accounting.add(
            scheme.prep, Region(n1=s1, n2=s2, nm=0, n_prep=0)
        )
        return _fault_prob(noise, accounting.add(resource, scheme.teleport))
    return _chunk_terms(
        noise, scheme, s1, s2, family, noisy_checks, native_plus
    ).bound


def _chunk_terms(
    noise: NoiseModel,
    scheme: accounting.Scheme,
    s1: int,
    s2: int,
    family: _Family,
    noisy_checks: bool,
    native_plus: bool,
) -> ChunkTerms:
    """Return the terms of one chunk's bound; requires ``family.r >= 1``."""
    r, q, w = family
    assert r >= 1, "the bound without checks has no terms"
    prep = scheme.prep
    resource = accounting.add(prep, Region(n1=s1, n2=s2, nm=0, n_prep=0))
    one_check = _faulty(
        accounting.check_region(1, w, native_plus), noisy_checks
    )
    checks = _faulty(
        accounting.check_region(r, r * w, native_plus), noisy_checks
    )
    return ChunkTerms(
        resource=_fault_prob(noise, resource),
        check=_fault_prob(noise, one_check),
        teleport=_fault_prob(noise, scheme.teleport),
        acceptance=_no_fault_prob(
            noise, accounting.protected_region(prep, s1, s2, checks)
        ),
        checks=r,
        per_check_miss=q,
    )


def _plog_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    scheme: accounting.Scheme,
    families: list[_Family],
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Return the sum of the chunk bounds, one family per chunk."""
    total = 0.0
    for (s1, s2), family in zip(counts, families, strict=True):
        total += _chunk_plog_bound(
            noise, scheme, s1, s2, family, noisy_checks, native_plus
        )
    return total


def _overhead_from_counts(
    counts: list[tuple[int, int]],
    noise: NoiseModel,
    scheme: accounting.Scheme,
    families: list[_Family],
    noisy_checks: bool,
    native_plus: bool,
) -> float:
    """Return the overhead bound for chunks of these gate counts.

    Each chunk is charged the worst-case check weight, and its acceptance
    probability the probability that nothing in its protected region
    faults, which is at most the true acceptance. With no checks every
    attempt is accepted, so the acceptance is 1 and the overhead exact.
    """
    regions, no_fault = [], []
    for (s1, s2), (r, _, w) in zip(counts, families, strict=True):
        checks = accounting.check_region(r, r * w, native_plus)
        regions.append(checks)
        protected = accounting.protected_region(
            scheme.prep, s1, s2, _faulty(checks, noisy_checks)
        )
        no_fault.append(1.0 if r == 0 else _no_fault_prob(noise, protected))
    return accounting.expected_overhead(counts, scheme, regions, no_fault)


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
        :func:`clinr.protocol.build_clinr_tr`. The checks' operations
        count toward the overhead either way.
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
        Not clipped to [0, 1], and infinite when some chunk with checks can
        never be accepted. With ``r == 0`` it is the probability that some
        operation outside the checks faults, since no attempt is rejected.

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
        counts,
        noise,
        accounting.clinr_scheme(n, native_plus),
        [family] * t,
        noisy_checks,
        native_plus,
    )


def direct_plog_bound(circuit: stim.Circuit, noise: NoiseModel) -> float:
    """Bound the logical error rate of the direct implementation of a circuit.

    The probability that some gate of ``circuit`` faults, the paper's
    ``g_p(s)`` with one rate per kind of gate. A logical error needs a
    fault, so this is an upper bound, and nearly exact to first order: a
    single fault in a Clifford circuit leaves a nontrivial Pauli on its
    output.

    Parameters
    ----------
    circuit : stim.Circuit
        The Clifford circuit ``C``, as accepted by
        :func:`clinr.accounting.gate_counts`.
    noise : NoiseModel

    Returns
    -------
    float
        An upper bound on the logical error rate of the circuit that
        :func:`clinr.protocol.build_direct` builds.
    """
    s1, s2 = accounting.gate_counts(circuit)
    return _fault_prob(noise, Region(n1=s1, n2=s2, nm=0, n_prep=0))


def plog_terms(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    *,
    noisy_checks: bool = True,
    restricted_checks: bool = False,
    native_plus: bool = False,
) -> list[ChunkTerms]:
    """Return the terms of :func:`plog_bound`, one entry per chunk.

    :func:`plog_bound` is the sum of the entries' :attr:`ChunkTerms.bound`.
    Without checks the bound is a single fault probability, with no terms.

    Parameters
    ----------
    circuit, n, t, noise, noisy_checks, restricted_checks, native_plus
        As in :func:`plog_bound`.
    r : int
        Number of checks per chunk. Must be at least 1.

    Returns
    -------
    list of ChunkTerms
        The terms of each chunk, in chunk order.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``r < 1``.
    """
    _check_sizes(n, r)
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}")
    family = _check_family(n, r, restricted_checks)
    return [
        _chunk_terms(
            noise,
            accounting.clinr_scheme(n, native_plus),
            *accounting.gate_counts(chunk),
            family,
            noisy_checks,
            native_plus,
        )
        for chunk in protocol.chunk_circuit(circuit, n, t)
    ]


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
    faults, which is at most the true acceptance. With ``r == 0`` no attempt
    is rejected, so the acceptance is 1 and the value is exact.

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
        ``circuit``, infinite when some chunk with checks can never be
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
    return _overhead_from_counts(
        counts,
        noise,
        accounting.clinr_scheme(n, native_plus),
        [family] * t,
        noisy_checks,
        native_plus,
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
) -> tuple[int, int] | None:
    """Return the implementation with the least logical error rate bound.

    The candidates are the direct implementation, whose bound is the
    probability that one of the gates of ``circuit`` faults and whose
    overhead is 1, and every ``CliNR_{t,r}`` with ``1 <= t <= t_max`` and
    ``0 <= r <= r_max`` whose overhead bound is at most ``max_overhead``.
    Ties go to the direct implementation, then to the smaller ``t``, then
    to the smaller ``r``.

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
        Largest acceptable overhead bound. Must be at least 1, the overhead
        of the direct implementation.
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
    tuple of (int, int) or None
        ``(t, r)``, or ``None`` when the direct implementation's bound is
        the least. A result with ``r == r_max`` may mean the cap binds.

    Raises
    ------
    ValueError
        If ``n < 1``, ``r_max < 0``, ``t_max < 1``, ``max_overhead`` is less
        than 1, or ``circuit`` has no gates.
    """
    _check_sizes(n, r_max, "r_max")
    if not max_overhead >= 1.0:
        raise ValueError(
            f"max_overhead must be at least 1, got {max_overhead}"
        )
    s1, s2 = accounting.gate_counts(circuit)
    if s1 + s2 == 0:
        raise ValueError("circuit must contain at least one gate")
    upper_t = s1 + s2 if t_max is None else t_max
    if upper_t < 1:
        raise ValueError(f"t_max must be at least 1, got {upper_t}")

    families = [
        _check_family(n, r, restricted_checks) for r in range(r_max + 1)
    ]
    best: tuple[int, int] | None = None
    best_plog = direct_plog_bound(circuit, noise)
    for t in range(1, upper_t + 1):
        chunks = protocol.chunk_circuit(circuit, n, t)
        counts = [accounting.gate_counts(chunk) for chunk in chunks]
        for r, family in enumerate(families):
            options = (
                noise,
                accounting.clinr_scheme(n, native_plus),
                [family] * t,
                noisy_checks,
                native_plus,
            )
            if _overhead_from_counts(counts, *options) > max_overhead:
                continue
            value = _plog_from_counts(counts, *options)
            if value < best_plog:
                best_plog, best = value, (t, r)
    return best


def cznr_plog_bound(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    *,
    noisy_checks: bool = True,
    native_plus: bool = False,
) -> float:
    """Bound the logical error rate of ``CZNR_{t,r}`` applied to ``circuit``.

    The bound of :func:`plog_bound`, with CZNR's regions in place of
    CliNR's: Theorem 3 of arXiv:2407.06583 with the same departures from
    the paper as :func:`plog_bound`'s from Theorem 1. A check drawn
    uniformly from the graph state's stabilizer group misses a damaging
    fault with probability ``1/2`` and weighs at most ``n``.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit ``C``, of ``CZ`` gates only, on qubits ``[0, n)``.
    n, t, r, noise, noisy_checks, native_plus
        As in :func:`plog_bound`.

    Returns
    -------
    float
        An upper bound on the logical error rate of the circuit that
        :func:`clinr.protocol.build_cznr_tr` builds with the same options,
        averaged over the checks :func:`clinr.checks.draw_cznr_checks`
        draws, as for :func:`plog_bound`.

    Raises
    ------
    ValueError
        If ``n < 1``, ``r < 0`` or ``circuit`` has a gate other than ``CZ``.
    """
    _check_sizes(n, r)
    protocol.require_cz_only(circuit)
    chunks = protocol.chunk_circuit(circuit, n, t)
    return _plog_from_counts(
        [accounting.gate_counts(chunk) for chunk in chunks],
        noise,
        accounting.cznr_scheme(n, native_plus),
        [_Family(r, 0.5, n)] * len(chunks),
        noisy_checks,
        native_plus,
    )


def cznr_overhead_bound(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    *,
    noisy_checks: bool = True,
    native_plus: bool = False,
) -> float:
    """Bound the gate overhead of ``CZNR_{t,r}`` applied to ``circuit``.

    :func:`overhead_bound` with CZNR's regions and the checks of
    :func:`cznr_plog_bound`.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit ``C``, of ``CZ`` gates only, on qubits ``[0, n)``, with
        at least one gate.
    n, t, r, noise, noisy_checks, native_plus
        As in :func:`cznr_plog_bound`.

    Returns
    -------
    float
        An upper bound on the expected number of operations per gate of
        ``circuit``, infinite when some chunk with checks can never be
        accepted.

    Raises
    ------
    ValueError
        If ``n < 1``, ``r < 0`` or ``circuit`` has a gate other than ``CZ``.
    """
    _check_sizes(n, r)
    protocol.require_cz_only(circuit)
    chunks = protocol.chunk_circuit(circuit, n, t)
    return _overhead_from_counts(
        [accounting.gate_counts(chunk) for chunk in chunks],
        noise,
        accounting.cznr_scheme(n, native_plus),
        [_Family(r, 0.5, n)] * len(chunks),
        noisy_checks,
        native_plus,
    )
