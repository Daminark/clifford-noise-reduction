"""Choosing which stabilizers to measure.

The verification sequence of a CliNR chunk is an ordered ``r``-tuple of
stabilizers of its resource state, and which tuple is drawn is a policy
separate from the circuit that measures it. This module holds the policy;
:mod:`protocol` holds the construction, and :mod:`bounds` holds the
detection probabilities the policy implies.

Two families are offered, both drawn uniformly over the ordered *linearly
independent* tuples, following arXiv:2407.06583 Sec. III: the full
stabilizer group, whose elements have weight up to ``2n``, and the
restricted family of arXiv:2407.06583 Sec. V, the ``3n`` propagated Bell
stabilizers, whose elements have weight at most ``n + 1`` and are therefore
cheaper to measure. Independence caps ``r`` at ``2n`` for either family.
"""

from __future__ import annotations

import numpy as np
import stim

from . import protocol

__all__ = [
    "random_stabilizer",
    "restricted_family",
    "draw_checks",
]


# ---------------------------------------------------------------------------
# Check families
# ---------------------------------------------------------------------------
def random_stabilizer(
    gens: list[stim.PauliString], rng: np.random.Generator
) -> stim.PauliString:
    """Sample a uniformly random *non-identity* element of ``⟨gens⟩``.

    The stabilizer group is elementary abelian (``S ≅ F_2^{2n}`` with ``gens`` a
    basis), so an independent fair coin per generator yields a uniform group
    element. The identity element is rejected: it has no valid ``MPP`` targets
    and carries no information as a check.

    The returned Pauli has the width of ``gens``.

    Args:
        gens: Independent generators, as returned by
            :func:`protocol.stabilizer_gens`.
        rng: A NumPy generator. Passing the generator (rather than a seed) lets
            a caller thread one stream through repeated draws so that the draws
            are independent; the explicit type is enforced to reject objects
            whose ``.random`` has a different contract.

    Raises:
        TypeError: if ``rng`` is not a ``numpy.random.Generator``.
        ValueError: if ``gens`` is empty.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not gens:
        raise ValueError("gens must not be empty")
    total = len(gens[0])
    while True:
        take = rng.random(len(gens)) < 0.5
        product = stim.PauliString(total)
        for keep, gen in zip(take, gens, strict=True):
            if keep:
                product *= gen
        if product.weight > 0:
            return product


def restricted_family(
    tableau: stim.Tableau, n: int, bare: list[int], image: list[int], total: int
) -> list[stim.PauliString]:
    """The weight-``(n+1)`` check family used by the paper's Sec. V numerics.

    The weight-2 Bell stabilizers propagated through ``I ⊗ C``: for each pair,
    the images of ``XX``, ``ZZ`` and ``YY``. Propagation commutes with taking
    products, so the third is the product of the first two, up to a phase that
    nothing downstream reads.

    Returns ``3n`` elements, each of weight 1 on ``bare`` and at most ``n`` on
    ``image``, none of them the identity. Arguments are as in
    :func:`protocol.stabilizer_gens`.
    """
    gens = protocol.stabilizer_gens(tableau, n, bare, image, total)
    family: list[stim.PauliString] = []
    for i in range(n):
        gx, gz = gens[2 * i], gens[2 * i + 1]
        family += [gx, gz, gx * gz]
    return family



# ---------------------------------------------------------------------------
# Drawing an independent tuple
# ---------------------------------------------------------------------------
def _reduce(
    basis: dict[int, np.ndarray], check: stim.PauliString
) -> np.ndarray | None:
    """Reduce ``check`` against ``basis``; ``None`` if it lies in the span.

    ``basis`` maps each pivot position to a row whose first set bit is that
    pivot, so eliminating in increasing pivot order never reintroduces a bit
    already cleared. A returned vector is the reduced row to store under its
    own pivot.
    """
    xs, zs = check.to_numpy()
    vec = np.concatenate([xs, zs])
    for pivot in sorted(basis):
        if vec[pivot]:
            vec = vec ^ basis[pivot]
    return vec if vec.any() else None


def _draw_uniform(
    gens: list[stim.PauliString], r: int, rng: np.random.Generator
) -> list[stim.PauliString]:
    """``r`` independent stabilizers, uniform over ordered independent tuples.

    Rejecting dependent candidates one at a time is exactly uniform here: for
    a prefix of rank ``j`` the acceptable set has ``2^{2n} - 2^j`` elements
    whatever the prefix is, so every ordered independent tuple is reached with
    the same probability.
    """
    basis: dict[int, np.ndarray] = {}
    drawn: list[stim.PauliString] = []
    while len(drawn) < r:
        candidate = random_stabilizer(gens, rng)
        vec = _reduce(basis, candidate)
        if vec is None:
            continue
        basis[int(np.argmax(vec))] = vec
        drawn.append(candidate)
    return drawn


def _draw_restricted(
    pool: list[stim.PauliString], r: int, rng: np.random.Generator
) -> list[stim.PauliString]:
    """``r`` independent members of ``pool``, uniform over ordered tuples.

    Rejection one at a time would *not* be uniform over the restricted family:
    a prefix holding both members of one Bell triple leaves fewer acceptable
    continuations than one spread across triples. Drawing a whole ordered
    ``r``-subset and rejecting it entire conditions the uniform distribution
    on independence, which is uniform over what remains.
    """
    while True:
        drawn = [pool[int(i)] for i in rng.permutation(len(pool))[:r]]
        basis: dict[int, np.ndarray] = {}
        for check in drawn:
            vec = _reduce(basis, check)
            if vec is None:
                break
            basis[int(np.argmax(vec))] = vec
        else:
            return drawn


def draw_checks(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    rng: np.random.Generator,
    *,
    restricted_checks: bool = False,
) -> list[list[stim.PauliString]]:
    """Draw ``r`` independent checks for each of the ``t`` chunks of ``circuit``.

    Each check is a length-``2n`` Pauli on the resource state of its chunk,
    with positions ``[0, n)`` on the bare half and ``[n, 2n)`` on the half
    carrying the chunk.

    The ``r`` checks of a chunk are linearly independent, following the
    paper's Sec. III, and are uniform over the ordered independent tuples.
    Independence caps ``r`` at ``2n``, the rank of the resource stabilizer
    group, and at ``r = 2n`` the checks span the group, so no damaging fault
    can commute with all of them.

    Args:
        circuit: The Clifford circuit ``C`` over ``n`` qubits.
        n: Number of data qubits.
        t: Number of chunks.
        r: Number of checks per chunk.
        rng: NumPy generator, drawn from in chunk order.
        restricted_checks: If True, draw from :func:`restricted_family`;
            otherwise draw from the non-identity elements of the full
            stabilizer group.

    Returns:
        ``t`` lists of ``r`` checks, in chunk order.

    Raises:
        ValueError: if ``r`` is negative or exceeds ``2n``, or from
            :func:`protocol.chunk_circuit`.
    """
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    if r > 2 * n:
        raise ValueError(
            f"r must be at most 2n = {2 * n} for independent checks, got {r}"
        )
    bare, image = list(range(n)), list(range(n, 2 * n))
    checks = []
    for chunk in protocol.chunk_circuit(circuit, n, t):
        tableau = chunk.to_tableau()
        if restricted_checks:
            pool = restricted_family(tableau, n, bare, image, 2 * n)
            checks.append(_draw_restricted(pool, r, rng))
        else:
            gens = protocol.stabilizer_gens(tableau, n, bare, image, 2 * n)
            checks.append(_draw_uniform(gens, r, rng))
    return checks
