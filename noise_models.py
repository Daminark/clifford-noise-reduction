"""Circuit-level noise models.

A ``NoiseModel`` carries the four rates that :mod:`clinr` builds circuits with
and that :mod:`bounds` prices fault locations against. Named regimes are
supplied as factories: choosing a regime is the caller's job, and the
construction and analysis layers only ever consume one.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NoiseModel",
    "uniform_noise",
    "asymmetric_noise",
]


@dataclass(frozen=True)
class NoiseModel:
    """Four circuit-level error rates.

    Attributes:
        p1: Depolarizing rate after single-qubit gates.
        p2: Depolarizing rate after two-qubit gates.
        p_m: Measurement flip probability.
        p_prep: Preparation bit-flip probability.

    Raises:
        ValueError: if any rate lies outside ``[0, 1]``.
    """

    p1: float
    p2: float
    p_m: float
    p_prep: float

    def __post_init__(self) -> None:
        for name, rate in (
            ("p1", self.p1),
            ("p2", self.p2),
            ("p_m", self.p_m),
            ("p_prep", self.p_prep),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {rate}")


def uniform_noise(p: float) -> NoiseModel:
    """Collapse all four rates to ``p``, as in the paper's analysis."""
    return NoiseModel(p1=p, p2=p, p_m=p, p_prep=p)


def asymmetric_noise(p2: float, ratio: float = 10.0) -> NoiseModel:
    """Rates with two-qubit gates ``ratio`` times noisier than every other
    operation.

    ``ratio=10`` reproduces the regime of the paper's Sec. V numerics.

    Raises:
        ValueError: if ``ratio`` is not positive.
    """
    if ratio <= 0.0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    rest = p2 / ratio
    return NoiseModel(p1=rest, p2=p2, p_m=rest, p_prep=rest)
