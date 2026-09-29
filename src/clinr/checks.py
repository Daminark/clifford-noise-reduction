"""Choosing which stabilizers of a chunk's resource state to measure.

The ``r`` checks of a chunk are drawn independently and identically, with
replacement, from one of two families: the non-identity elements of the full
stabilizer group, whose weight is at most ``2n``, as the proof of Theorem 1
of [1]_ assumes; or the restricted family of [1]_, Sec. V, the ``3n``
stabilizers propagated from the Bell pairs, whose weight is at most
``n + 1``. :mod:`clinr.protocol` measures the checks and :mod:`clinr.bounds`
prices what each family detects.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
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


def _check_rng(rng: np.random.Generator) -> None:
    """Raise ``TypeError`` unless ``rng`` is a ``numpy.random.Generator``."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )


# ---------------------------------------------------------------------------
# Check families
# ---------------------------------------------------------------------------
def random_stabilizer(
    gens: list[stim.PauliString], rng: np.random.Generator
) -> stim.PauliString:
    """Return a uniformly random non-identity element of the group of ``gens``.

    Each generator is included in the product with probability 1/2, which is
    uniform over the group, and the identity is redrawn.

    Parameters
    ----------
    gens : list of stim.PauliString
        Pairwise commuting generators of one length, not all the identity,
        such as those of :func:`clinr.protocol.stabilizer_gens`.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    stim.PauliString
        Of the length of ``gens``, with a sign that nothing downstream reads.

    Raises
    ------
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``.
    ValueError
        If ``gens`` is empty, if its elements differ in length, or if every
        element is the identity.
    """
    _check_rng(rng)
    if not gens:
        raise ValueError("gens must not be empty")
    total = len(gens[0])
    if any(len(gen) != total for gen in gens):
        raise ValueError(
            f"gens must share one length, got {sorted({len(g) for g in gens})}"
        )
    if all(gen.weight == 0 for gen in gens):
        raise ValueError("gens generate only the identity")
    while True:
        take = rng.random(len(gens)) < 0.5
        product = stim.PauliString(total)
        for keep, gen in zip(take, gens, strict=True):
            if keep:
                product *= gen
        if product.weight > 0:
            return product


def restricted_family(
    tableau: stim.Tableau, bare: list[int], image: list[int], total: int
) -> list[stim.PauliString]:
    """Return the restricted check family of arXiv:2407.06583, Sec. V.

    With ``n = len(tableau)``, the family is the images of the Bell
    stabilizers ``XX``, ``ZZ`` and ``YY`` of each pair under ``I ⊗ C``, in
    that order, pair by pair. The third of each triple is the product of the
    first two, up to a sign that nothing downstream reads.

    Parameters
    ----------
    tableau, bare, image, total
        As in :func:`clinr.protocol.stabilizer_gens`.

    Returns
    -------
    list of stim.PauliString
        ``3n`` stabilizers of the resource state, none the identity, each of
        weight 1 on ``bare`` and at most ``n`` on ``image``.
    """
    gens = protocol.stabilizer_gens(tableau, bare, image, total)
    family: list[stim.PauliString] = []
    for i in range(len(tableau)):
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

    The ``r`` checks of a chunk are drawn independently and identically,
    with replacement, the convention of arXiv:2407.06583, App. B, so a check
    may be drawn twice.

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
    rng : numpy.random.Generator
        Source of randomness, drawn from in chunk order.
    restricted_checks : bool, optional
        If True, draw uniformly from :func:`restricted_family`; if False,
        from the non-identity elements of the full stabilizer group, with
        :func:`random_stabilizer`.

    Returns
    -------
    list of list of stim.PauliString
        ``t`` lists of ``r`` checks, in chunk order, as
        :func:`clinr.protocol.build_clinr_tr` takes them: length ``2n``,
        the bare half on ``[0, n)`` and the half carrying the chunk on
        ``[n, 2n)``.

    Raises
    ------
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``.
    ValueError
        If ``r`` is negative.
    """
    _check_rng(rng)
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    bare, image = list(range(n)), list(range(n, 2 * n))
    checks = []
    for chunk in protocol.chunk_circuit(circuit, n, t):
        tableau = chunk.to_tableau()
        if restricted_checks:
            pool = restricted_family(tableau, bare, image, 2 * n)
            checks.append(
                [pool[int(rng.integers(len(pool)))] for _ in range(r)]
            )
        else:
            gens = protocol.stabilizer_gens(tableau, bare, image, 2 * n)
            checks.append([random_stabilizer(gens, rng) for _ in range(r)])
    return checks
