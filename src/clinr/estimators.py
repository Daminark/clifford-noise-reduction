"""Monte Carlo estimates of CliNR, CZNR and the direct implementation.

:func:`estimate_clinr`, :func:`estimate_cznr` and :func:`estimate_direct`
sample the circuits :mod:`clinr.protocol` builds and return an
:class:`Estimate` of raw counts, from which the point estimates, their
intervals and the measured gate overhead are all computed, so a sweep
never has to be rerun for another statistic.

A CliNR shot is accepted when every check of every chunk passes, and the
logical error rate is taken over accepted shots. This conditioning gives the
output distribution of the protocol of [1]_, which re-prepares a rejected
resource offline and never lets a failed attempt touch the data; the cost of
re-preparation appears in :func:`gate_overhead` instead.

One estimate is one draw of the checks, and its shots share those checks.
:func:`logical_error_rate`, :func:`gate_overhead_over_draws` and
:func:`design_effect` treat draws, not shots, as the unit of observation.
The protocol draws new checks for every attempt, which the first two
estimate by default from the same draws; they also give the rates with one
check set kept for every attempt.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import stim
from scipy import stats

from . import _validation, accounting, protocol
from .checks import draw_checks, draw_cznr_checks
from .noise_models import NoiseModel

__all__ = [
    "wilson_interval",
    "Estimate",
    "estimate_clinr",
    "estimate_cznr",
    "estimate_direct",
    "sweep_clinr",
    "sweep_direct",
    "gate_overhead",
    "logical_error_rate",
    "gate_overhead_over_draws",
    "design_effect",
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
        Number of errors, the events whose proportion is estimated. Must
        be between 0 and ``trials``.
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
    TypeError
        If ``errors`` or ``trials`` is not an integer.
    ValueError
        If ``trials`` or ``errors`` is negative, if ``errors`` exceeds
        ``trials``, or if ``confidence`` is not strictly between 0 and 1.
    """
    _validation.require_int(errors, "errors")
    _validation.require_int(trials, "trials")
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

    The four per-chunk tuples have one length: the number of chunks, or
    zero for the direct implementation. All counts are non-negative.

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
        Total weight of each chunk's checks: at most its number of checks
        times the family's worst-case weight, ``n`` if ``cznr``, ``n + 1``
        if ``restricted_checks`` and ``2n`` otherwise, and at least its
        number of checks if ``restricted_checks``, since that family
        excludes the identity.
    chunk_counts : tuple of tuple of (int, int), optional
        Gate counts ``(s1, s2)`` of each chunk, summing to ``(s1, s2)``.
    noisy_checks, restricted_checks, native_plus : bool, optional
        The options the circuit was built and its checks drawn with, as in
        :func:`estimate_clinr`. The direct implementation keeps the
        defaults.
    cznr : bool, optional
        True for an estimate of CZNR (:func:`estimate_cznr`), whose checks
        are drawn uniformly, so ``restricted_checks`` is then False.

    Raises
    ------
    TypeError
        If a count is not an integer or an option is not a bool.
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
    cznr: bool = False

    def __post_init__(self) -> None:
        """Check the constraints stated in the class docstring."""
        for name in ("shots", "accepted", "errors", "s1", "s2", "n"):
            _validation.require_int(getattr(self, name), name)
        for name in ("chunk_accepted", "chunk_checks", "chunk_weight"):
            for value in getattr(self, name):
                _validation.require_int(value, f"each entry of {name}")
        for pair in self.chunk_counts:
            for value in pair:
                _validation.require_int(value, "each entry of chunk_counts")
        for name in (
            "noisy_checks",
            "restricted_checks",
            "native_plus",
            "cznr",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(
                    f"{name} must be a bool, got "
                    f"{type(getattr(self, name)).__name__}"
                )
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
        if any(
            checks == 0 and accepted != self.shots
            for checks, accepted in zip(
                self.chunk_checks, self.chunk_accepted, strict=True
            )
        ):
            raise ValueError(
                "a chunk without checks accepts every shot, got "
                f"{self.chunk_accepted} for checks {self.chunk_checks}"
            )
        if self.cznr and self.restricted_checks:
            raise ValueError(
                "CZNR draws uniform checks only, so cznr and "
                "restricted_checks cannot both be set"
            )
        least = 1 if self.restricted_checks else 0
        if self.cznr:
            worst = self.n
        else:
            worst = self.n + 1 if self.restricted_checks else 2 * self.n
        if any(
            not least * checks <= weight <= worst * checks
            for weight, checks in zip(
                self.chunk_weight, self.chunk_checks, strict=True
            )
        ):
            raise ValueError(
                f"each chunk's check weight must lie between {least} and "
                f"{worst} times its number of checks, got weights "
                f"{self.chunk_weight} for checks {self.chunk_checks}"
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
        If ``rng`` is not a ``numpy.random.Generator``, ``shots`` is not an
        integer, or ``noise`` is not a
        :class:`clinr.noise_models.NoiseModel`; or as the functions it calls
        raise for the other arguments.
    ValueError
        If ``shots`` is not positive, or as the functions it calls raise
        for the other arguments.
    """
    _check_rng(rng)
    _validation.require_int(shots, "shots")
    _validation.require_noise(noise)
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    drawn = draw_checks(
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


def estimate_cznr(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    shots: int,
    rng: np.random.Generator,
    *,
    noisy_checks: bool = True,
    native_plus: bool = False,
) -> Estimate:
    """Sample ``CZNR_{t,r}`` applied to a ``CZ`` circuit under fresh checks.

    :func:`estimate_clinr` for CZNR: the checks are drawn by
    :func:`clinr.checks.draw_cznr_checks` and the circuit built by
    :func:`clinr.protocol.build_cznr_tr`.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit ``C``, of ``CZ`` gates only, on qubits ``[0, n)``.
    n, t, r, noise, shots, rng
        As in :func:`estimate_clinr`.
    noisy_checks, native_plus : bool, optional
        As in :func:`clinr.protocol.build_cznr_tr`.

    Returns
    -------
    Estimate
        With ``cznr`` set, and one entry per chunk in each per-chunk tuple.

    Raises
    ------
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``, ``shots`` is not an
        integer, or ``noise`` is not a
        :class:`clinr.noise_models.NoiseModel`; or as the functions it calls
        raise for the other arguments.
    ValueError
        If ``shots`` is not positive, or as the functions it calls raise
        for the other arguments.
    """
    _check_rng(rng)
    _validation.require_int(shots, "shots")
    _validation.require_noise(noise)
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    drawn = draw_cznr_checks(circuit, n, t, r, rng)
    built = protocol.build_cznr_tr(
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
        native_plus=native_plus,
        cznr=True,
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
        If ``rng`` is not a ``numpy.random.Generator``, ``shots`` is not an
        integer, or ``noise`` is not a
        :class:`clinr.noise_models.NoiseModel`; or as the functions it calls
        raise for the other arguments.
    ValueError
        If ``shots`` is not positive, or as the functions it calls raise
        for the other arguments.
    """
    _check_rng(rng)
    _validation.require_int(shots, "shots")
    _validation.require_noise(noise)
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
        "cznr": estimate.cznr,
    }


def _comparable(estimates: list[Estimate]) -> None:
    """Raise ``ValueError`` unless ``estimates`` are draws for one setting.

    That is: at least one estimate, and the same gate counts, overall and
    per chunk, checks per chunk, ``n`` and options throughout. Shot counts
    and check weights may differ between draws.
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
        if other.chunk_counts != first.chunk_counts:
            raise ValueError(
                f"gate counts per chunk disagree: {first.chunk_counts} "
                f"and {other.chunk_counts}"
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


def _each_accepted(estimates: list[Estimate]) -> None:
    """Raise ``ValueError`` if some estimate accepted no shots."""
    if any(estimate.accepted == 0 for estimate in estimates):
        raise ValueError("an estimate accepted no shots")


def _check_checks(checks: str) -> None:
    """Raise ``ValueError`` unless ``checks`` is ``"fresh"`` or ``"kept"``."""
    if checks not in ("fresh", "kept"):
        raise ValueError(f"checks must be 'fresh' or 'kept', got {checks!r}")


def _half_width(deviations: list[float], confidence: float) -> float:
    """Return the Student t half-width from per-draw deviations.

    Each deviation is a draw's first-order contribution to the point
    estimate; they sum to zero, and the sum of their squares over
    ``J (J - 1)``, for ``J`` draws, estimates its variance. Requires at
    least two draws.
    """
    draws = len(deviations)
    assert draws >= 2, "a half-width needs at least two draws"
    variance = sum(d * d for d in deviations) / (draws * (draws - 1))
    quantile = float(stats.t.ppf((1.0 + confidence) / 2.0, draws - 1))
    return quantile * math.sqrt(variance)


def logical_error_rate(
    estimates: list[Estimate],
    confidence: float = 0.95,
    *,
    checks: Literal["fresh", "kept"] = "fresh",
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for the logical error rate over draws.

    With ``checks="fresh"``, the rate of the protocol of arXiv:2407.06583,
    Sec. III, which draws new checks for every attempt. Check sets that
    accept more often are then over-represented among accepted attempts,
    so the point estimate is the mean over draws of the fraction of shots
    accepted with an error, over the mean of the fraction accepted. With
    ``checks="kept"``, the rate with one check set kept for every attempt,
    averaged over check sets: the mean of the per-draw rates. Each draw
    counts equally, whatever its number of shots.

    The draws are independent clusters of correlated shots, so the
    interval is the Korn-Graubard interval for a proportion from a
    clustered sample (Korn and Graubard, Survey Methodology 24, 193
    (1998)), as the NCHS data presentation standards for proportions
    specify it. The variance of the point estimate is estimated over the
    draws by the delta method, so it counts the spread between draws as
    well as shot noise. Dividing the binomial variance at the point
    estimate by it gives an effective number of shots, scaled by the ratio
    of the Student t quantiles at the shots' and the draws' degrees of
    freedom, and capped at the shots accepted. The interval is the
    Clopper-Pearson interval at that effective number. When the variance
    carries no information, at a point estimate of 0 or 1 or with draws
    that do not differ, the effective number is the shots accepted.

    Parameters
    ----------
    estimates : list of Estimate
        Draws for one setting: the same gate counts, checks per chunk,
        ``n`` and options. With ``checks="kept"`` each must have accepted
        a shot; with ``"fresh"``, at least one.
    confidence : float, optional
        Confidence level, strictly between 0 and 1.
    checks : {"fresh", "kept"}, optional
        Whether the checks are redrawn for every attempt or kept.

    Returns
    -------
    tuple of (float, float, float)
        With a single draw the bounds are ``nan``.

    Raises
    ------
    ValueError
        If ``estimates`` is empty or its entries disagree in gate counts,
        checks per chunk, ``n`` or options; if they accepted too few
        shots, as above; if ``confidence`` is not strictly between 0 and 1;
        or if ``checks`` is neither ``"fresh"`` nor ``"kept"``.
    """
    _comparable(estimates)
    _check_confidence(confidence)
    _check_checks(checks)
    if checks == "fresh":
        if all(estimate.accepted == 0 for estimate in estimates):
            raise ValueError("no estimate accepted a shot")
        failed = [estimate.errors / estimate.shots for estimate in estimates]
        passed = [estimate.accepted / estimate.shots for estimate in estimates]
        mean_passed = sum(passed) / len(passed)
        point = sum(failed) / sum(passed)
        deviations = [
            (fail - point * accept) / mean_passed
            for fail, accept in zip(failed, passed, strict=True)
        ]
    else:
        _each_accepted(estimates)
        rates = [estimate.p_log for estimate in estimates]
        point = sum(rates) / len(rates)
        deviations = [rate - point for rate in rates]
    if len(estimates) < 2:
        return point, math.nan, math.nan
    draws = len(estimates)
    variance = sum(d * d for d in deviations) / (draws * (draws - 1))
    shots = sum(estimate.accepted for estimate in estimates)
    return (point, *_korn_graubard(point, variance, shots, draws, confidence))


def _korn_graubard(
    point: float, variance: float, shots: int, draws: int, confidence: float
) -> tuple[float, float]:
    """Return the Korn-Graubard bounds for a proportion over draws.

    ``point`` is the estimate, ``variance`` its variance over the
    ``draws`` clusters, and ``shots`` the units behind it. Requires at
    least two draws and one shot.
    """
    assert draws >= 2, "the interval needs at least two draws"
    assert shots >= 1, "the interval needs at least one shot"
    tail = (1.0 + confidence) / 2.0
    effective = float(shots)
    if 0.0 < point < 1.0 and variance > 0.0:
        # The t quantile at the shots' degrees of freedom is at most the one
        # at the draws', so the ratio shrinks the effective number.
        ratio = float(
            stats.t.ppf(tail, max(shots, draws) - 1)
            / stats.t.ppf(tail, draws - 1)
        )
        effective = min(effective, point * (1.0 - point) / variance * ratio**2)
    errors = point * effective
    lo = (
        0.0
        if errors == 0.0
        else float(stats.beta.ppf(1.0 - tail, errors, effective - errors + 1))
    )
    hi = (
        1.0
        if errors == effective
        else float(stats.beta.ppf(tail, errors + 1, effective - errors))
    )
    return lo, hi


def design_effect(estimates: list[Estimate]) -> float:
    """Return the between-draw variance over the mean within-draw variance.

    The within-draw variance of a draw is the binomial ``p (1 - p) / a`` at
    its rate ``p`` and accepted count ``a``. A value near 1 means the draws
    vary no more than independent shots would, so a Wilson interval on the
    pooled shots is honest; above 1, an interval over draws is wider than
    the pooled one by about the square root of the value.

    Parameters
    ----------
    estimates : list of Estimate
        At least two draws, as in :func:`logical_error_rate` with
        ``checks="kept"``.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``estimates`` fails the conditions of :func:`logical_error_rate`
        with ``checks="kept"``, if fewer than two are given, or if every
        draw's rate is 0 or 1, which makes the within-draw variance zero.

    Notes
    -----
    When errors are rare, few draws see any, and the ratio is noisy.
    """
    _comparable(estimates)
    _each_accepted(estimates)
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


def _check_has_chunks(estimate: Estimate) -> None:
    """Raise ``ValueError`` if ``estimate`` is of the direct implementation."""
    if not estimate.chunk_accepted:
        raise ValueError("estimate has no chunks")


def _check_overhead_input(estimate: Estimate) -> None:
    """Raise ``ValueError`` unless every chunk of ``estimate`` accepted."""
    _check_has_chunks(estimate)
    if any(accepted == 0 for accepted in estimate.chunk_accepted):
        raise ValueError(
            "a chunk accepted no shots, so its restart probability is "
            "unmeasurable"
        )


def _scheme(estimate: Estimate) -> accounting.Scheme:
    """Return the scheme ``estimate`` was made with."""
    if estimate.cznr:
        return accounting.cznr_scheme(estimate.n, estimate.native_plus)
    return accounting.clinr_scheme(estimate.n, estimate.native_plus)


def _check_regions(estimate: Estimate) -> list[accounting.Region]:
    """Return each chunk's check region at the weights ``estimate`` drew."""
    return [
        accounting.check_region(count, weight, estimate.native_plus)
        for count, weight in zip(
            estimate.chunk_checks, estimate.chunk_weight, strict=True
        )
    ]


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
        From :func:`estimate_clinr` or :func:`estimate_cznr`, with at least
        one gate and every chunk accepting at least one shot.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``estimate`` has no chunks or a chunk accepted no shots.
    """
    _check_overhead_input(estimate)
    return accounting.expected_overhead(
        list(estimate.chunk_counts),
        _scheme(estimate),
        _check_regions(estimate),
        [accepted / estimate.shots for accepted in estimate.chunk_accepted],
    )


def gate_overhead_over_draws(
    estimates: list[Estimate],
    confidence: float = 0.95,
    *,
    checks: Literal["fresh", "kept"] = "fresh",
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for the gate overhead over draws.

    With ``checks="fresh"``, the overhead of the protocol of
    arXiv:2407.06583, Sec. III, which draws new checks for every attempt:
    its attempts are then independent and identically distributed, so by
    Wald's identity each chunk costs its teleportation plus the mean over
    draws of its protected region's operations, divided by the mean over
    draws of its acceptance. With ``checks="kept"``, the mean over draws of
    :func:`gate_overhead`, each draw's checks kept for every attempt. Each
    draw counts equally, whatever its number of shots. The interval is as in
    :func:`logical_error_rate`, without the clamp.

    Parameters
    ----------
    estimates : list of Estimate
        Draws as in :func:`logical_error_rate`, of CliNR or CZNR rather
        than the direct implementation. With ``checks="kept"`` every chunk
        of each must have accepted a shot; with ``"fresh"``, every chunk of
        at least one.
    confidence : float, optional
        Confidence level, strictly between 0 and 1.
    checks : {"fresh", "kept"}, optional
        Whether the checks are redrawn for every attempt or kept.

    Returns
    -------
    tuple of (float, float, float)
        With a single draw the bounds are ``nan``.

    Raises
    ------
    ValueError
        On the conditions of :func:`logical_error_rate` other than those on
        accepted shots, if an estimate has no chunks, or if the chunks
        accepted too few shots, as above.
    """
    _comparable(estimates)
    _check_confidence(confidence)
    _check_checks(checks)
    for estimate in estimates:
        _check_has_chunks(estimate)
    if checks == "kept":
        overheads = [gate_overhead(estimate) for estimate in estimates]
        point = sum(overheads) / len(overheads)
        deviations = [overhead - point for overhead in overheads]
    else:
        first = estimates[0]
        scheme = _scheme(first)
        # Each draw's operations per attempt and acceptance, chunk by chunk.
        operations = [
            [
                sum(accounting.protected_region(scheme.prep, s1, s2, region))
                for (s1, s2), region in zip(
                    estimate.chunk_counts,
                    _check_regions(estimate),
                    strict=True,
                )
            ]
            for estimate in estimates
        ]
        acceptance = [
            [accepted / estimate.shots for accepted in estimate.chunk_accepted]
            for estimate in estimates
        ]
        draws, chunks = len(estimates), len(first.chunk_accepted)
        mean_ops = [
            sum(row[k] for row in operations) / draws for k in range(chunks)
        ]
        mean_acc = [
            sum(row[k] for row in acceptance) / draws for k in range(chunks)
        ]
        if 0.0 in mean_acc:
            raise ValueError("a chunk accepted no shots in any estimate")
        # At fixed acceptance the overhead is linear in each chunk's
        # operation count, so averaging it over the draws averages the
        # operations.
        point = (
            sum(
                accounting.expected_overhead(
                    list(estimate.chunk_counts),
                    scheme,
                    _check_regions(estimate),
                    mean_acc,
                )
                for estimate in estimates
            )
            / draws
        )
        size = first.s1 + first.s2
        deviations = [
            sum(
                ops[k] / mean_acc[k] - mean_ops[k] * acc[k] / mean_acc[k] ** 2
                for k in range(chunks)
            )
            / size
            for ops, acc in zip(operations, acceptance, strict=True)
        ]
    if len(estimates) < 2:
        return point, math.nan, math.nan
    half = _half_width(deviations, confidence)
    return point, point - half, point + half
