"""Monte Carlo estimators for CliNR (arXiv:2407.06583).

Estimators return raw counts rather than scalars. A count record is a
sufficient statistic: the point estimate, its confidence interval, and any
reduction over a sweep of circuits can be reconstructed from it, so a sweep
never has to be repeated to compute a different statistic.

Every reported proportion carries a Wilson score interval. The Wald interval
is unusable here: the logical error rate sits near zero, where Wald's lower
bound leaves ``[0, 1]`` and collapses to a point when no errors are seen.

Acceptance is modelled by post-selection. In the protocol a chunk whose
resource fails its checks is re-prepared offline, so the resource consumed is
always one that passed, and the data is never touched by a failed attempt.
Conditioning on all checks passing therefore reproduces the protocol's state,
and the cost of re-preparation appears in the gate overhead instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import stim

from . import accounting, checks, protocol
from .noise_models import NoiseModel

__all__ = [
    "Z_95",
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

# Normal quantile for a two-sided 95% interval.
Z_95 = 1.959963984540054

# Student t quantiles for a two-sided 95% interval, indexed by degrees of
# freedom (entry 0 unused). Beyond the table the normal quantile is used: it
# is about 4% narrower at 31 degrees of freedom and under 1% past 130.
_T_95 = (
    math.nan,
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def wilson_interval(
    errors: int, trials: int, z: float = Z_95
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)``, a Wilson score interval for a proportion.

    Args:
        errors: Number of successes observed.
        trials: Number of independent trials.
        z: Normal quantile setting the confidence level.

    Returns:
        The point estimate ``errors / trials`` and the interval bounds, both
        clipped to ``[0, 1]``. With no trials the point estimate is ``nan``
        and the interval is ``(0, 1)``.

    Raises:
        ValueError: if ``trials`` or ``errors`` is negative, if ``errors``
            exceeds ``trials``, or if ``z`` is not positive.
    """
    if trials < 0:
        raise ValueError(f"trials must be non-negative, got {trials}")
    if errors < 0:
        raise ValueError(f"errors must be non-negative, got {errors}")
    if errors > trials:
        raise ValueError(f"errors ({errors}) cannot exceed trials ({trials})")
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    if trials == 0:
        return math.nan, 0.0, 1.0

    z2 = z * z
    centre = (errors + z2 / 2.0) / (trials + z2)
    spread = math.sqrt(errors * (trials - errors) / trials + z2 / 4.0)
    half = z * spread / (trials + z2)
    return errors / trials, max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Estimate:
    """Counts from one Monte Carlo run on a single circuit.

    Attributes:
        shots: Number of shots sampled.
        accepted: Shots in which every check returned a trivial outcome.
        errors: Accepted shots carrying a non-trivial output error.
        s1: Single-qubit gate count of the circuit.
        s2: Two-qubit gate count of the circuit.
        chunk_accepted: Shots accepted by each chunk's checks alone, in chunk
            order. A chunk with no checks accepts every shot.
        chunk_checks: Number of checks measured in each chunk, in chunk order.
        chunk_weight: Total weight of those checks, in chunk order.
        chunk_counts: Gate counts ``(s1, s2)`` of each chunk, in chunk order.

    The four per-chunk tuples have one length: the number of chunks, or zero
    for the direct implementation, which has none. Carrying the chunk gate
    counts is what lets :func:`gate_overhead` work from the estimate alone,
    rather than being handed a circuit it has to trust.

    Raises:
        ValueError: if the per-chunk tuples have different lengths, or if the
            chunk gate counts do not sum to ``(s1, s2)``.
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
        """Total gate count of the circuit."""
        return self.s1 + self.s2

    @property
    def acceptance(self) -> float:
        """Fraction of shots in which every check passed."""
        return self.accepted / self.shots

    @property
    def p_log(self) -> float:
        """Point estimate of the logical error rate among accepted shots."""
        return math.nan if self.accepted == 0 else self.errors / self.accepted

    def interval(self, z: float = Z_95) -> tuple[float, float, float]:
        """Return ``(point, lo, hi)`` for the logical error rate.

        The accepted count is the effective sample size, so rejecting more
        shots widens the interval.
        """
        return wilson_interval(self.errors, self.accepted, z)


def _tally(
    circuit: stim.Circuit,
    shots: int,
    rng: np.random.Generator,
    checks_per_chunk: list[int],
) -> tuple[int, int, tuple[int, ...]]:
    """Sample ``circuit`` and return ``(accepted, errors, chunk_accepted)``.

    ``checks_per_chunk`` gives the number of detectors each chunk owns, in
    detector order.
    """
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
    """Estimate the logical error rate of ``CliNR_{t,r}`` applied to ``circuit``.

    A fresh set of checks is drawn for the returned circuit, and the estimate
    is conditioned on all of them passing. Draws are taken from ``rng``, which
    also seeds the sampler, so one generator drives the whole estimate.

    Args:
        circuit: The Clifford circuit to implement.
        n: Number of data qubits.
        t: Number of chunks.
        r: Number of checks per chunk.
        noise: The four rates.
        shots: Number of shots to sample.
        rng: NumPy generator.
        noisy_checks: If True, measure checks with the noisy gadget.
        restricted_checks: If True, draw checks from the weight-``(n+1)``
            family instead of uniformly from the full stabilizer group.
        noisy_q: If True, resolve the tracked Pauli frame into a live corrected
            state once at the end of the chain, at its noise cost.
        native_plus: If True, prepare ``|+>`` with a single ``RX`` rather than
            ``R`` then ``H``.

    Returns:
        An :class:`Estimate` holding the counts.

    Raises:
        ValueError: if ``shots`` is not positive.
    """
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    drawn = checks.draw_checks(
        circuit, n, t, r, rng, restricted_checks=restricted_checks
    )
    built = protocol.build_clinr_tr(
        circuit, n, drawn, noise, noisy_checks=noisy_checks,
        noisy_q=noisy_q, native_plus=native_plus,
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
    """Estimate the logical error rate of the direct implementation.

    The direct implementation has no checks, so every shot is accepted and the
    logical error rate is taken over all shots.

    Args:
        circuit: The Clifford circuit to implement.
        n: Number of data qubits.
        noise: The four rates.
        shots: Number of shots to sample.
        rng: NumPy generator, used to seed the sampler.

    Returns:
        An :class:`Estimate` holding the counts, with ``accepted == shots``.

    Raises:
        ValueError: if ``shots`` is not positive.
    """
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
    """Estimate each of ``circuits`` under one set of parameters.

    One generator is threaded through the sweep, so each circuit's checks and
    shots are drawn independently. Arguments other than ``circuits`` are as in
    :func:`estimate_clinr`, and one :class:`Estimate` is returned per circuit,
    in the order given.

    Raises:
        ValueError: if ``circuits`` is empty.
    """
    if not circuits:
        raise ValueError("circuits must not be empty")
    return [
        estimate_clinr(
            c, n, t, r, noise, shots, rng,
            noisy_checks=noisy_checks, restricted_checks=restricted_checks,
            noisy_q=noisy_q, native_plus=native_plus,
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
    """Estimate the direct implementation of each of ``circuits``.

    One generator is threaded through the sweep, so each circuit's shots are
    drawn independently. Arguments other than ``circuits`` are as in
    :func:`estimate_direct`, and one :class:`Estimate` is returned per circuit,
    in the order given.

    Raises:
        ValueError: if ``circuits`` is empty.
    """
    if not circuits:
        raise ValueError("circuits must not be empty")
    return [estimate_direct(c, n, noise, shots, rng) for c in circuits]


def _comparable(estimates: list[Estimate]) -> None:
    """Check that estimates describe one circuit under one set of parameters.

    Shot counts may differ: the draw, not the shot, is the unit of
    observation. Check weights may differ too, since that variation is what
    repeated draws are for.

    Raises:
        ValueError: if ``estimates`` is empty, if the gate counts or the
            per-chunk check counts disagree, or if any estimate accepted no
            shots.
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


def cluster_interval(estimates: list[Estimate]) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for the logical error rate over check sets.

    Each estimate is one draw of the check sets, so a draw is the unit of
    observation and its shots are not independent of each other. The point
    estimate is the mean of the per-draw rates and the interval is the
    two-sided 95% Student t interval on those means, clamped to ``[0, 1]``.
    Pooling the shots instead and taking a Wilson interval understates the
    uncertainty whenever draws differ systematically
    (:func:`design_effect` measures that).

    The confidence level is fixed at 95%, the level the ``t`` table covers.
    With one draw the bounds are ``nan``.

    Raises:
        ValueError: on the conditions of :func:`_comparable`.
    """
    _comparable(estimates)
    rates = [e.p_log for e in estimates]
    point = sum(rates) / len(rates)
    if len(rates) < 2:
        return point, math.nan, math.nan
    df = len(rates) - 1
    spread = math.sqrt(
        sum((r - point) ** 2 for r in rates) / df / len(rates)
    )
    t = _T_95[df] if df < len(_T_95) else Z_95
    return point, max(0.0, point - t * spread), min(1.0, point + t * spread)


def design_effect(estimates: list[Estimate]) -> float:
    """Return the between-draw variance over the mean within-draw variance.

    One means the draws are no more variable than independent sampling would
    make them, so a Wilson interval on the pooled shots would be honest.
    Larger values say clustered intervals should be wider by its square root.

    Raises:
        ValueError: on the conditions of :func:`_comparable`, or if fewer than
            two estimates are given.
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
    return between / within


def gate_overhead(
    n: int,
    estimate: Estimate,
    *,
    noisy_checks: bool = True,
    noisy_q: bool = False,
    native_plus: bool = False,
) -> float:
    """Estimate the gate overhead from an estimate's measured acceptance.

    Identical to :func:`bounds.overhead_bound` except that each chunk is
    charged the weight of the checks actually drawn, and its restart
    probability is taken from its measured acceptance rather than bounded
    analytically; both call the same operation-count helper. Since every
    attempt runs all of a chunk's checks, that restart probability is the only
    estimated quantity. Everything but ``n`` is read from ``estimate``: the
    chunks' gate counts, their checks and their measured acceptance, so no
    circuit has to be supplied and trusted to be the one that was sampled.

    Args:
        n: Number of data qubits.
        estimate: Counts from :func:`estimate_clinr`.
        noisy_checks: If True, charge each check the ancilla gadget's
            operations; if False, an idealized check costs nothing. Must match
            the value passed to :func:`estimate_clinr`.
        noisy_q: If True, add the ``n`` gates that resolve the tracked Pauli
            frame into a live corrected state once at the end of the chain.
        native_plus: If True, charge ``|+>`` as a single native preparation.
            Must match the value passed to :func:`estimate_clinr`.

    Returns:
        The estimated expected gate overhead.

    Raises:
        ValueError: if ``estimate`` has no chunks, if any chunk accepted no
            shots, or if the chunks contain no gates.
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
    """Mean of :func:`gate_overhead` over check-set draws.

    Each draw counts equally, whatever its number of shots, because the
    marginal is taken over draws. The mean of the per-draw overheads is not
    the overhead of the summed counts: each draw has its own check weights and
    its own restart probability.

    Raises:
        ValueError: on the conditions of :func:`_comparable`, or from
            :func:`gate_overhead`.
    """
    _comparable(estimates)
    overheads = [
        gate_overhead(
            n, e, noisy_checks=noisy_checks, noisy_q=noisy_q,
            native_plus=native_plus,
        )
        for e in estimates
    ]
    return sum(overheads) / len(overheads)
