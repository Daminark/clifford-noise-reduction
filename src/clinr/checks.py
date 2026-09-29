"""Choosing which stabilizers to measure.

The verification sequence of a CliNR chunk is an ordered ``r``-tuple of
stabilizers of its resource state, and which tuple is drawn is a policy
separate from the circuit that measures it. This module holds the policy;
:mod:`protocol` holds the construction, and :mod:`bounds` holds the
detection probabilities the policy implies.

Two families are offered, both drawn independently and identically: the
full stabilizer group, whose elements have weight up to ``2n``, and the
restricted family of arXiv:2407.06583 Sec. V, the ``3n`` propagated Bell
stabilizers, whose elements have weight at most ``n + 1`` and are therefore
cheaper to measure.
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
    gens = protocol.stabilizer_gens(tableau, bare, image, total)
    family: list[stim.PauliString] = []
    for i in range(n):
        gx, gz = gens[2 * i], gens[2 * i + 1]
        family += [gx, gz, gx * gz]
    return family



# ---------------------------------------------------------------------------
# Drawing a chunk's checks
# ---------------------------------------------------------------------------
def draw_checks(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    rng: np.random.Generator,
    *,
    restricted_checks: bool = False,
) -> list[list[stim.PauliString]]:
    """Draw ``r`` checks for each of the ``t`` chunks of ``circuit``.

    Each check is a length-``2n`` Pauli on the resource state of its chunk,
    with positions ``[0, n)`` on the bare half and ``[n, 2n)`` on the half
    carrying the chunk.

    The ``r`` checks of a chunk are drawn independently and identically,
    with replacement, which is the convention Appendix B of the paper proves
    under: a damaging fault makes the ``r`` outcomes a uniform bit-string, so
    the per-draw misses multiply. A check may therefore be drawn twice.

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
        ValueError: if ``r`` is negative, or from
            :func:`protocol.chunk_circuit`.
    """
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    bare, image = list(range(n)), list(range(n, 2 * n))
    checks = []
    for chunk in protocol.chunk_circuit(circuit, n, t):
        tableau = chunk.to_tableau()
        if restricted_checks:
            pool = restricted_family(tableau, n, bare, image, 2 * n)
            checks.append(
                [pool[int(rng.integers(len(pool)))] for _ in range(r)]
            )
        else:
            gens = protocol.stabilizer_gens(tableau, bare, image, 2 * n)
            checks.append([random_stabilizer(gens, rng) for _ in range(r)])
    return checks
