"""Monte Carlo estimates of CliNR and of the direct implementation.

:func:`estimate_clinr` and :func:`estimate_direct` sample the circuits
:mod:`clinr.protocol` builds and return an :class:`Estimate` of raw counts,
from which the point estimates, their intervals and the measured gate
overhead are all computed, so a sweep never has to be rerun for another
statistic.

A CliNR shot is accepted when every check of every chunk passes, and the
logical error rate is taken over accepted shots. This conditioning gives the
output distribution of the protocol, which re-prepares a rejected resource
offline and never lets a failed attempt touch the data; the cost of
re-preparation appears in :func:`gate_overhead` instead.

One estimate is one draw of the checks, and its shots share those checks.
:func:`cluster_interval`, :func:`design_effect` and
:func:`mean_gate_overhead` treat draws, not shots, as the unit of
observation.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import stim
from scipy import stats

from . import accounting, checks, protocol
from .noise_models import NoiseModel

__all__ = [
    "wilson_interval",
    "Estimate",
    "estimate_clinr",
    "estimate_direct",
    "sweep_clinr",
    "sweep_direct",
    "gate_overhead",
    "cluster_interval",
    "design_effect",
    "mean_gate_overhead",
]


def _check_confidence(confidence: float) -> None:
    """Raise ``ValueError`` unless ``0 < confidence < 1``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must lie strictly between 0 and 1, got {confidence}"
        )


def _check_rng(rng: np.random.Generator) -> None:
    """Raise ``TypeError`` unless ``rng`` is a ``numpy.random.Generator``."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )


def wilson_interval(
    errors: int, trials: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)``, a Wilson score interval for a proportion.

    Parameters
    ----------
    errors : int
        Number of successes. Must be between 0 and ``trials``.
    trials : int
        Number of independent trials. Must be non-negative.
    confidence : float, optional
        Confidence level, strictly between 0 and 1.

    Returns
    -------
    tuple of (float, float, float)
        The point estimate ``errors / trials`` and the interval's bounds,
        which lie in [0, 1]. With no trials the point estimate is ``nan``
        and the interval is ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``trials`` or ``errors`` is negative, if ``errors`` exceeds
        ``trials``, or if ``confidence`` is not strictly between 0 and 1.
    """
    if trials < 0:
        raise ValueError(f"trials must be non-negative, got {trials}")
    if errors < 0:
        raise ValueError(f"errors must be non-negative, got {errors}")
    if errors > trials:
        raise ValueError(f"errors ({errors}) cannot exceed trials ({trials})")
    _check_confidence(confidence)
    if trials == 0:
        return math.nan, 0.0, 1.0
    interval = stats.binomtest(errors, trials).proportion_ci(
        confidence, method="wilson"
    )
    return errors / trials, float(interval.low), float(interval.high)


@dataclass(frozen=True)
class Estimate:
    """Counts from one Monte Carlo run on one circuit.

    Parameters
    ----------
    shots : int
        Shots sampled. Must be positive.
    accepted : int
        Shots in which every check passed, at most ``shots``.
    errors : int
        Accepted shots with a nontrivial output error, at most
        ``accepted``.
    s1, s2 : int
        Single- and two-qubit gate counts of the circuit.
    n : int
        Number of data qubits. Must be at least 1.
    chunk_accepted : tuple of int, optional
        For each chunk, in order, the shots its own checks accepted:
        between ``accepted`` and ``shots``. A chunk without checks accepts
        every shot.
    chunk_checks : tuple of int, optional
        Checks measured in each chunk.
    chunk_weight : tuple of int, optional
        Total weight of each chunk's checks: at least its number of checks,
        since no check is the identity, and at most that number times the
        family's worst-case weight, ``n + 1`` if ``restricted_checks`` and
        ``2n`` otherwise.
    chunk_counts : tuple of tuple of (int, int), optional
        Gate counts ``(s1, s2)`` of each chunk, summing to ``(s1, s2)``.
    noisy_checks, restricted_checks, native_plus : bool, optional
        The options the circuit was built and its checks drawn with, as in
        :func:`estimate_clinr`. The direct implementation keeps the
        defaults.

    The four per-chunk tuples have one length: the number of chunks, or
    zero for the direct implementation. All counts are non-negative.

    Raises
    ------
    ValueError
        If any of the constraints above fails.
    """

    shots: int
    accepted: int
    errors: int
    s1: int
    s2: int
    n: int
    chunk_accepted: tuple[int, ...] = ()
    chunk_checks: tuple[int, ...] = ()
    chunk_weight: tuple[int, ...] = ()
    chunk_counts: tuple[tuple[int, int], ...] = ()
    noisy_checks: bool = True
    restricted_checks: bool = False
    native_plus: bool = False

    def __post_init__(self) -> None:
        """Check the constraints stated in the class docstring."""
        if self.shots < 1:
            raise ValueError(f"shots must be positive, got {self.shots}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")
        for name, value in (
            ("accepted", self.accepted),
            ("errors", self.errors),
            ("s1", self.s1),
            ("s2", self.s2),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        for name, values in (
            ("chunk_accepted", self.chunk_accepted),
            ("chunk_checks", self.chunk_checks),
            ("chunk_weight", self.chunk_weight),
            (
                "chunk_counts",
                tuple(count for pair in self.chunk_counts for count in pair),
            ),
        ):
            if any(value < 0 for value in values):
                raise ValueError(f"{name} must be non-negative, got {values}")
        if self.accepted > self.shots:
            raise ValueError(
                f"accepted ({self.accepted}) cannot exceed shots "
                f"({self.shots})"
            )
        if self.errors > self.accepted:
            raise ValueError(
                f"errors ({self.errors}) cannot exceed accepted "
                f"({self.accepted})"
            )
        if any(chunk > self.shots for chunk in self.chunk_accepted):
            raise ValueError(
                f"no chunk can accept more than the {self.shots} shots, got "
                f"{self.chunk_accepted}"
            )
        if any(chunk < self.accepted for chunk in self.chunk_accepted):
            raise ValueError(
                "every chunk accepts at least the "
                f"{self.accepted} shots accepted overall, got "
                f"{self.chunk_accepted}"
            )
        worst = self.n + 1 if self.restricted_checks else 2 * self.n
        if any(
            not checks <= weight <= checks * worst
            for weight, checks in zip(
                self.chunk_weight, self.chunk_checks, strict=False
            )
        ):
            raise ValueError(
                "each chunk's check weight must lie between its number of "
                f"checks and {worst} times it, got weights "
                f"{self.chunk_weight} for checks {self.chunk_checks}"
            )
        lengths = {
            len(self.chunk_accepted),
            len(self.chunk_checks),
            len(self.chunk_weight),
            len(self.chunk_counts),
        }
        if len(lengths) > 1:
            raise ValueError(
                "per-chunk tuples disagree in length: "
                f"{len(self.chunk_accepted)}, {len(self.chunk_checks)}, "
                f"{len(self.chunk_weight)}, {len(self.chunk_counts)}"
            )
        if self.chunk_counts:
            totals = (
                sum(s1 for s1, _ in self.chunk_counts),
                sum(s2 for _, s2 in self.chunk_counts),
            )
            if totals != (self.s1, self.s2):
                raise ValueError(
                    f"chunk gate counts sum to {totals}, not "
                    f"{(self.s1, self.s2)}"
                )

    @property
    def size(self) -> int:
        """Gate count of the circuit."""
        return self.s1 + self.s2

    @property
    def acceptance(self) -> float:
        """Fraction of shots in which every check passed."""
        return self.accepted / self.shots

    @property
    def p_log(self) -> float:
        """Logical error rate among accepted shots; ``nan`` if none."""
        return math.nan if self.accepted == 0 else self.errors / self.accepted

    def interval(self, confidence: float = 0.95) -> tuple[float, float, float]:
        """Return :func:`wilson_interval` of ``errors`` in ``accepted``."""
        return wilson_interval(self.errors, self.accepted, confidence)


def _tally(
    circuit: stim.Circuit,
    shots: int,
    rng: np.random.Generator,
    checks_per_chunk: list[int],
) -> tuple[int, int, tuple[int, ...]]:
    """Sample ``circuit``; return ``(accepted, errors, chunk_accepted)``.

    ``checks_per_chunk`` holds the number of detectors each chunk owns, in
    detector order, and sums to the circuit's detector count.
    """
    assert sum(checks_per_chunk) == circuit.num_detectors, (
        f"{checks_per_chunk} checks for {circuit.num_detectors} detectors"
    )
    seed = int(rng.integers(1 << 63))
    detectors, observables = circuit.compile_detector_sampler(
        seed=seed
    ).sample(shots, separate_observables=True)

    mask = ~detectors.any(axis=1)
    accepted_by_chunk = []
    start = 0
    for count in checks_per_chunk:
        window = detectors[:, start : start + count]
        accepted_by_chunk.append(int((~window.any(axis=1)).sum()))
        start += count
    per_chunk = tuple(accepted_by_chunk)

    accepted = int(mask.sum())
    errors = int(observables[mask].any(axis=1).sum())
    return accepted, errors, per_chunk


def estimate_clinr(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    shots: int,
    rng: np.random.Generator,
    *,
    noisy_checks: bool = True,
    restricted_checks: bool = False,
    native_plus: bool = False,
) -> Estimate:
    """Sample ``CliNR_{t,r}`` applied to ``circuit`` under fresh checks.

    The checks are drawn by :func:`clinr.checks.draw_checks` and the circuit
    built by :func:`clinr.protocol.build_clinr_tr`. ``rng`` supplies the
    draws and then seeds Stim's sampler, so one generator drives the whole
    estimate.

    Parameters
    ----------
    circuit, n, t, r
        As in :func:`clinr.checks.draw_checks`.
    noise : NoiseModel
    shots : int
        Shots to sample. Must be positive.
    rng : numpy.random.Generator
        Source of randomness.
    noisy_checks, native_plus : bool, optional
        As in :func:`clinr.protocol.build_clinr_tr`.
    restricted_checks : bool, optional
        As in :func:`clinr.checks.draw_checks`.

    Returns
    -------
    Estimate
        With one entry per chunk in each per-chunk tuple.

    Raises
    ------
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``.
    ValueError
        If ``shots`` is not positive.
    """
    _check_rng(rng)
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    drawn = checks.draw_checks(
        circuit, n, t, r, rng, restricted_checks=restricted_checks
    )
    built = protocol.build_clinr_tr(
        circuit,
        n,
        drawn,
        noise,
        noisy_checks=noisy_checks,
        native_plus=native_plus,
    )
    per_chunk_checks = [len(chunk_checks) for chunk_checks in drawn]
    accepted, errors, per_chunk = _tally(built, shots, rng, per_chunk_checks)
    s1, s2 = accounting.gate_counts(circuit)
    chunk_counts = tuple(
        accounting.gate_counts(chunk)
        for chunk in protocol.chunk_circuit(circuit, n, len(drawn))
    )
    return Estimate(
        shots,
        accepted,
        errors,
        s1,
        s2,
        n,
        per_chunk,
        tuple(per_chunk_checks),
        tuple(sum(check.weight for check in chunk) for chunk in drawn),
        chunk_counts,
        noisy_checks=noisy_checks,
        restricted_checks=restricted_checks,
        native_plus=native_plus,
    )


def estimate_direct(
    circuit: stim.Circuit,
    n: int,
    noise: NoiseModel,
    shots: int,
    rng: np.random.Generator,
) -> Estimate:
    """Sample the direct implementation of ``circuit``.

    The circuit is built by :func:`clinr.protocol.build_direct`; ``rng``
    seeds Stim's sampler.

    Parameters
    ----------
    circuit, n
        As in :func:`clinr.protocol.build_direct`.
    noise : NoiseModel
    shots : int
        Shots to sample. Must be positive.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    Estimate
        With every shot accepted and empty per-chunk tuples.

    Raises
    ------
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``.
    ValueError
        If ``shots`` is not positive.
    """
    _check_rng(rng)
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    built = protocol.build_direct(circuit, n, noise)
    accepted, errors, _ = _tally(built, shots, rng, [])
    s1, s2 = accounting.gate_counts(circuit)
    return Estimate(shots, accepted, errors, s1, s2, n)


def sweep_clinr(
    circuits: list[stim.Circuit],
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    shots: int,
    rng: np.random.Generator,
    *,
    noisy_checks: bool = True,
    restricted_checks: bool = False,
    native_plus: bool = False,
) -> list[Estimate]:
    """Return :func:`estimate_clinr` of each of ``circuits``, in order.

    One generator is threaded through the sweep, so the circuits' checks
    and shots are drawn independently. Every other argument is as in
    :func:`estimate_clinr`.

    Raises
    ------
    ValueError
        If ``circuits`` is empty.
    """
    if not circuits:
        raise ValueError("circuits must not be empty")
    return [
        estimate_clinr(
            circuit,
            n,
            t,
            r,
            noise,
            shots,
            rng,
            noisy_checks=noisy_checks,
            restricted_checks=restricted_checks,
            native_plus=native_plus,
        )
        for circuit in circuits
    ]


def sweep_direct(
    circuits: list[stim.Circuit],
    n: int,
    noise: NoiseModel,
    shots: int,
    rng: np.random.Generator,
) -> list[Estimate]:
    """Return :func:`estimate_direct` of each of ``circuits``, in order.

    One generator is threaded through the sweep, so the circuits' shots are
    drawn independently. Every other argument is as in
    :func:`estimate_direct`.

    Raises
    ------
    ValueError
        If ``circuits`` is empty.
    """
    if not circuits:
        raise ValueError("circuits must not be empty")
    return [
        estimate_direct(circuit, n, noise, shots, rng) for circuit in circuits
    ]


def _setting(estimate: Estimate) -> dict[str, int | bool]:
    """Return the ``n`` and options that ``estimate`` was taken with."""
    return {
        "n": estimate.n,
        "noisy_checks": estimate.noisy_checks,
        "restricted_checks": estimate.restricted_checks,
        "native_plus": estimate.native_plus,
    }


def _comparable(estimates: list[Estimate]) -> None:
    """Raise ``ValueError`` unless ``estimates`` are draws for one setting.

    That is: at least one estimate, the same gate counts, checks per chunk,
    ``n`` and options throughout, and at least one accepted shot each. Shot
    counts and check weights may differ between draws.
    """
    if not estimates:
        raise ValueError("estimates must not be empty")
    first = estimates[0]
    for other in estimates:
        if (other.s1, other.s2) != (first.s1, first.s2):
            raise ValueError(
                f"gate counts disagree: {(first.s1, first.s2)} "
                f"and {(other.s1, other.s2)}"
            )
        if other.chunk_checks != first.chunk_checks:
            raise ValueError(
                f"checks per chunk disagree: {first.chunk_checks} "
                f"and {other.chunk_checks}"
            )
        if _setting(other) != _setting(first):
            raise ValueError(
                f"n or options disagree: {_setting(first)} "
                f"and {_setting(other)}"
            )
        if other.accepted == 0:
            raise ValueError("an estimate accepted no shots")


def cluster_interval(
    estimates: list[Estimate], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for the logical error rate over draws.

    The point estimate is the mean of the per-draw rates, and the interval
    the two-sided Student t interval on that mean, clamped to [0, 1]. A
    Wilson interval on the pooled shots understates the uncertainty whenever
    draws differ systematically, which :func:`design_effect` measures.

    Parameters
    ----------
    estimates : list of Estimate
        Draws for one setting: the same gate counts, checks per chunk,
        ``n`` and options, each with at least one accepted shot.
    confidence : float, optional
        Confidence level, strictly between 0 and 1.

    Returns
    -------
    tuple of (float, float, float)
        With a single draw the bounds are ``nan``.

    Raises
    ------
    ValueError
        If ``estimates`` is empty, disagree in gate counts, checks per
        chunk, ``n`` or options, or include one that accepted no shots, or
        if ``confidence`` is not strictly between 0 and 1.
    """
    _comparable(estimates)
    _check_confidence(confidence)
    rates = [estimate.p_log for estimate in estimates]
    point = sum(rates) / len(rates)
    if len(rates) < 2:
        return point, math.nan, math.nan
    df = len(rates) - 1
    spread = math.sqrt(
        sum((rate - point) ** 2 for rate in rates) / df / len(rates)
    )
    quantile = float(stats.t.ppf((1.0 + confidence) / 2.0, df))
    return (
        point,
        max(0.0, point - quantile * spread),
        min(1.0, point + quantile * spread),
    )


def design_effect(estimates: list[Estimate]) -> float:
    """Return the between-draw variance over the mean within-draw variance.

    The within-draw variance of a draw is the binomial ``p (1 - p) / a`` at
    its rate ``p`` and accepted count ``a``. A value near 1 means the draws
    vary no more than independent shots would, so a Wilson interval on the
    pooled shots is honest; above 1, clustered intervals are wider by its
    square root.

    Parameters
    ----------
    estimates : list of Estimate
        At least two draws, as in :func:`cluster_interval`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        On the conditions of :func:`cluster_interval` on ``estimates``, if
        fewer than two are given, or if every draw's rate is 0 or 1, which
        makes the within-draw variance zero.

    Notes
    -----
    A draw that saw no errors contributes zero within-draw variance, so the
    ratio is biased upward when errors are rare.
    """
    _comparable(estimates)
    if len(estimates) < 2:
        raise ValueError("design effect needs at least two estimates")
    rates = [estimate.p_log for estimate in estimates]
    mean = sum(rates) / len(rates)
    between = sum((rate - mean) ** 2 for rate in rates) / (len(rates) - 1)
    within = sum(
        rate * (1.0 - rate) / estimate.accepted
        for rate, estimate in zip(rates, estimates, strict=True)
    ) / len(estimates)
    if within == 0.0:
        raise ValueError(
            "design effect is undefined: every draw saw no errors or only "
            "errors, so the within-draw variance is zero"
        )
    return between / within


def gate_overhead(estimate: Estimate) -> float:
    """Return the gate overhead measured by ``estimate``.

    :func:`clinr.accounting.expected_overhead`, as in
    :func:`clinr.bounds.overhead_bound`, but with each chunk's checks at the
    weight actually drawn and its acceptance probability the fraction of
    shots its checks accepted. Every attempt runs all of a chunk's checks,
    so that fraction is the only estimated quantity; everything else is
    read from ``estimate``.

    Parameters
    ----------
    estimate : Estimate
        From :func:`estimate_clinr`, with at least one gate and every chunk
        accepting at least one shot.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``estimate`` has no chunks or a chunk accepted no shots.
    """
    if not estimate.chunk_accepted:
        raise ValueError("estimate has no chunks")
    if any(accepted == 0 for accepted in estimate.chunk_accepted):
        raise ValueError(
            "a chunk accepted no shots, so its restart probability is "
            "unmeasurable"
        )

    regions = [
        accounting.check_region(
            count, weight, estimate.noisy_checks, estimate.native_plus
        )
        for count, weight in zip(
            estimate.chunk_checks, estimate.chunk_weight, strict=True
        )
    ]
    return accounting.expected_overhead(
        list(estimate.chunk_counts),
        estimate.n,
        accounting.prep_region(estimate.n, estimate.native_plus),
        regions,
        [accepted / estimate.shots for accepted in estimate.chunk_accepted],
    )


def mean_gate_overhead(estimates: list[Estimate]) -> float:
    """Return the mean of :func:`gate_overhead` over draws.

    Each draw counts equally, whatever its number of shots. This is not the
    overhead of the pooled counts, since each draw has its own check weights
    and acceptance.

    Parameters
    ----------
    estimates : list of Estimate
        Draws as in :func:`cluster_interval`, each as in
        :func:`gate_overhead`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        On the conditions of :func:`cluster_interval` on ``estimates``.
    """
    _comparable(estimates)
    overheads = [gate_overhead(estimate) for estimate in estimates]
    return sum(overheads) / len(overheads)
