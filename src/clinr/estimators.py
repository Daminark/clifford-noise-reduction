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
    ci = stats.binomtest(errors, trials).proportion_ci(
        confidence, method="wilson"
    )
    return errors / trials, float(ci.low), float(ci.high)


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
    chunk_accepted : tuple of int, optional
        For each chunk, in order, the shots its own checks accepted:
        between ``accepted`` and ``shots``. A chunk without checks accepts
        every shot.
    chunk_checks : tuple of int, optional
        Checks measured in each chunk.
    chunk_weight : tuple of int, optional
        Total weight of each chunk's checks, at least its number of checks,
        since no check is the identity.
    chunk_counts : tuple of tuple of (int, int), optional
        Gate counts ``(s1, s2)`` of each chunk, summing to ``(s1, s2)``.

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
    chunk_accepted: tuple[int, ...] = ()
    chunk_checks: tuple[int, ...] = ()
    chunk_weight: tuple[int, ...] = ()
    chunk_counts: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        """Check the constraints stated in the class docstring."""
        if self.shots < 1:
            raise ValueError(f"shots must be positive, got {self.shots}")
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
            ("chunk_counts", tuple(c for p in self.chunk_counts for c in p)),
        ):
            if any(v < 0 for v in values):
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
        if any(a > self.shots for a in self.chunk_accepted):
            raise ValueError(
                f"no chunk can accept more than the {self.shots} shots, got "
                f"{self.chunk_accepted}"
            )
        if any(a < self.accepted for a in self.chunk_accepted):
            raise ValueError(
                "every chunk accepts at least the "
                f"{self.accepted} shots accepted overall, got "
                f"{self.chunk_accepted}"
            )
        if any(
            w < c
            for w, c in zip(self.chunk_weight, self.chunk_checks, strict=False)
        ):
            raise ValueError(
                "each chunk's check weight must be at least its number of "
                f"checks, got weights {self.chunk_weight} for checks "
                f"{self.chunk_checks}"
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
    noisy_q: bool = False,
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
    noisy_checks, noisy_q, native_plus : bool, optional
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
        noisy_q=noisy_q,
        native_plus=native_plus,
    )
    per_chunk_checks = [len(c) for c in drawn]
    accepted, errors, per_chunk = _tally(built, shots, rng, per_chunk_checks)
    s1, s2 = accounting.gate_counts(circuit)
    chunk_counts = tuple(
        accounting.gate_counts(c)
        for c in protocol.chunk_circuit(circuit, n, len(drawn))
    )
    return Estimate(
        shots,
        accepted,
        errors,
        s1,
        s2,
        per_chunk,
        tuple(per_chunk_checks),
        tuple(sum(p.weight for p in c) for c in drawn),
        chunk_counts,
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
    return Estimate(shots, accepted, errors, s1, s2)


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
    noisy_q: bool = False,
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
            c,
            n,
            t,
            r,
            noise,
            shots,
            rng,
            noisy_checks=noisy_checks,
            restricted_checks=restricted_checks,
            noisy_q=noisy_q,
            native_plus=native_plus,
        )
        for c in circuits
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
    return [estimate_direct(c, n, noise, shots, rng) for c in circuits]


def _comparable(estimates: list[Estimate]) -> None:
    """Raise ``ValueError`` unless ``estimates`` are draws for one setting.

    That is: at least one estimate, the same gate counts and checks per
    chunk throughout, and at least one accepted shot each. Shot counts and
    check weights may differ between draws.
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
        Draws for one setting: the same gate counts and checks per chunk,
        each with at least one accepted shot.
    confidence : float, optional
        Confidence level, strictly between 0 and 1.

    Returns
    -------
    tuple of (float, float, float)
        With a single draw the bounds are ``nan``.

    Raises
    ------
    ValueError
        If ``estimates`` is empty, disagree in gate counts or checks per
        chunk, or include one that accepted no shots, or if ``confidence``
        is not strictly between 0 and 1.
    """
    _comparable(estimates)
    _check_confidence(confidence)
    rates = [e.p_log for e in estimates]
    point = sum(rates) / len(rates)
    if len(rates) < 2:
        return point, math.nan, math.nan
    df = len(rates) - 1
    spread = math.sqrt(sum((r - point) ** 2 for r in rates) / df / len(rates))
    t = float(stats.t.ppf((1.0 + confidence) / 2.0, df))
    return point, max(0.0, point - t * spread), min(1.0, point + t * spread)


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
    rates = [e.p_log for e in estimates]
    mean = sum(rates) / len(rates)
    between = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
    within = sum(
        r * (1.0 - r) / e.accepted
        for r, e in zip(rates, estimates, strict=True)
    ) / len(estimates)
    if within == 0.0:
        raise ValueError(
            "design effect is undefined: every draw saw no errors or only "
            "errors, so the within-draw variance is zero"
        )
    return between / within


def gate_overhead(
    n: int,
    estimate: Estimate,
    *,
    noisy_checks: bool = True,
    noisy_q: bool = False,
    native_plus: bool = False,
) -> float:
    """Return the gate overhead measured by ``estimate``.

    :func:`clinr.accounting.expected_overhead`, as in
    :func:`clinr.bounds.overhead_bound`, but with each chunk's checks at the
    weight actually drawn and its acceptance probability the fraction of
    shots its checks accepted. Every attempt runs all of a chunk's checks,
    so that fraction is the only estimated quantity.

    Parameters
    ----------
    n : int
        Number of data qubits, as passed to :func:`estimate_clinr`. Must be
        at least 1.
    estimate : Estimate
        From :func:`estimate_clinr`, with at least one gate and every chunk
        accepting at least one shot.
    noisy_checks, noisy_q, native_plus : bool, optional
        As passed to :func:`estimate_clinr`.

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
    if any(a == 0 for a in estimate.chunk_accepted):
        raise ValueError(
            "a chunk accepted no shots, so its restart probability is "
            "unmeasurable"
        )

    regions = [
        accounting.check_region(count, weight, noisy_checks, native_plus)
        for count, weight in zip(
            estimate.chunk_checks, estimate.chunk_weight, strict=True
        )
    ]
    return accounting.expected_overhead(
        list(estimate.chunk_counts),
        n,
        accounting.prep_region(n, native_plus),
        regions,
        [a / estimate.shots for a in estimate.chunk_accepted],
        noisy_q,
    )


def mean_gate_overhead(
    n: int,
    estimates: list[Estimate],
    *,
    noisy_checks: bool = True,
    noisy_q: bool = False,
    native_plus: bool = False,
) -> float:
    """Return the mean of :func:`gate_overhead` over draws.

    Each draw counts equally, whatever its number of shots. This is not the
    overhead of the pooled counts, since each draw has its own check weights
    and acceptance.

    Parameters
    ----------
    n, noisy_checks, noisy_q, native_plus
        As in :func:`gate_overhead`.
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
    overheads = [
        gate_overhead(
            n,
            e,
            noisy_checks=noisy_checks,
            noisy_q=noisy_q,
            native_plus=native_plus,
        )
        for e in estimates
    ]
    return sum(overheads) / len(overheads)
