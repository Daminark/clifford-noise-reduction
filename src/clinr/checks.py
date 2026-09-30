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
    "restricted_family",
    "draw_checks",
]


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
        If True, draw uniformly from :func:`restricted_family`. If False,
        draw uniformly from the non-identity elements of the full
        stabilizer group: each generator of
        :func:`clinr.protocol.stabilizer_gens` enters the product with
        probability 1/2, which is uniform over the group, and the identity
        is redrawn.

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
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    bare, image = list(range(n)), list(range(n, 2 * n))
    checks = []
    for chunk in protocol.chunk_circuit(circuit, n, t):
        tableau = chunk.to_tableau()
        if restricted_checks:
            family = restricted_family(tableau, bare, image, 2 * n)
            checks.append(
                [family[int(rng.integers(len(family)))] for _ in range(r)]
            )
            continue
        gens = protocol.stabilizer_gens(tableau, bare, image, 2 * n)
        chunk_checks: list[stim.PauliString] = []
        while len(chunk_checks) < r:
            product = stim.PauliString(2 * n)
            for keep, gen in zip(
                rng.random(len(gens)) < 0.5, gens, strict=True
            ):
                if keep:
                    product *= gen
            if product.weight > 0:
                chunk_checks.append(product)
        checks.append(chunk_checks)
    return checks
