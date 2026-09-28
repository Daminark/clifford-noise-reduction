"""Circuit-level noise models.

A :class:`NoiseModel` holds the four error rates that :mod:`clinr.protocol`
builds noisy circuits with and :mod:`clinr.bounds` prices fault locations
against.
:func:`uniform_noise` and :func:`asymmetric_noise` give the two regimes of
[1]_: one rate for every operation, and two-qubit gates noisier than the rest.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NoiseModel",
    "uniform_noise",
    "asymmetric_noise",
]


def _check_rate(name: str, rate: float) -> None:
    """Raise ``ValueError``, naming ``name``, unless ``rate`` is in [0, 1]."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {rate}")


@dataclass(frozen=True)
class NoiseModel:
    """Four circuit-level error rates.

    Idle qubits are noiseless: the model has no idle rate.

    Parameters
    ----------
    p1 : float
        Depolarizing rate after each single-qubit gate, in [0, 1].
    p2 : float
        Depolarizing rate after each two-qubit gate, in [0, 1].
    p_m : float
        Probability that a measurement outcome is flipped, in [0, 1].
    p_prep : float
        Probability that a preparation lands in the orthogonal state, in
        [0, 1].

    Raises
    ------
    ValueError
        If any rate lies outside [0, 1].
    """

    p1: float
    p2: float
    p_m: float
    p_prep: float

    def __post_init__(self) -> None:
        """Check that every rate lies in [0, 1]."""
        for name in ("p1", "p2", "p_m", "p_prep"):
            _check_rate(name, getattr(self, name))


def uniform_noise(p: float) -> NoiseModel:
    """Return the model with every rate equal to ``p``.

    This is the single-rate model of arXiv:2407.06583, Sec. II, under which
    Theorem 1 is stated.

    Parameters
    ----------
    p : float
        Error rate of every operation, in [0, 1].

    Returns
    -------
    NoiseModel

    Raises
    ------
    ValueError
        If ``p`` lies outside [0, 1].
    """
    _check_rate("p", p)
    return NoiseModel(p1=p, p2=p, p_m=p, p_prep=p)


def asymmetric_noise(p2: float, ratio: float = 10.0) -> NoiseModel:
    """Return the model with two-qubit gates ``ratio`` times noisier.

    Every operation other than a two-qubit gate gets rate ``p2 / ratio``.
    The default ``ratio=10`` is the regime of the numerics in
    arXiv:2407.06583, Sec. V, which also puts noise on idle qubits; this
    model has none.

    Parameters
    ----------
    p2 : float
        Depolarizing rate after each two-qubit gate, in [0, 1].
    ratio : float, optional
        How many times noisier two-qubit gates are than every other
        operation. Must be positive, and at least ``p2``.

    Returns
    -------
    NoiseModel

    Raises
    ------
    ValueError
        If ``p2`` lies outside [0, 1], if ``ratio`` is not positive, or if
        ``p2 / ratio`` exceeds 1.
    """
    _check_rate("p2", p2)
    if not ratio > 0.0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    rest = p2 / ratio
    if rest > 1.0:
        raise ValueError(
            f"p2 / ratio must not exceed 1, got {p2} / {ratio} = {rest}"
        )
    return NoiseModel(p1=rest, p2=p2, p_m=rest, p_prep=rest)
