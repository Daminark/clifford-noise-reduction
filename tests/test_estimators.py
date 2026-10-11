"""Tests for the Monte Carlo estimators in ``clinr.estimators``."""

import math
from typing import Any

import numpy as np
import pytest
import stim
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from clinr import (
    bounds,
    checks,
    circuits,
    estimators,
    first_order,
    noise_models,
    protocol,
)

NOISELESS = noise_models.uniform_noise(0.0)
# Low enough that two faults in one shot are rare, so the estimates agree with
# the first-order values to about 2% relative.
LOW_NOISE = noise_models.NoiseModel(p1=2e-4, p2=2e-3, p_m=1e-3, p_prep=4e-4)


def seeded_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return a seeded random ``{H, S, CX}`` circuit on ``n`` qubits."""
    return circuits.random_gate_sequence(n, size, np.random.default_rng(seed))


def draw(errors: int, accepted: int, **fields) -> estimators.Estimate:
    """Return a hand-built one-chunk estimate at n = 1: 10 gates, 2 checks.

    At ``n = 1`` every check has weight 2, so the two weigh 4 in total.

    ``chunk_accepted`` defaults to ``accepted``; any field can be overridden.
    """
    values: dict[str, Any] = {
        "shots": 1000,
        "accepted": accepted,
        "errors": errors,
        "s1": 6,
        "s2": 4,
        "n": 1,
        "chunk_accepted": (accepted,),
        "chunk_checks": (2,),
        "chunk_weight": (4,),
        "chunk_counts": ((6, 4),),
    }
    return estimators.Estimate(**(values | fields))


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------
def test_wilson_interval_at_half_successes_has_the_closed_form_width():
    """At k = n/2 the Wilson interval is 1/2 +- z / (2 sqrt(n + z^2))."""
    z = 1.959963984540054
    half = z / (2 * math.sqrt(10 + z**2))

    assert estimators.wilson_interval(5, 10) == pytest.approx(
        (0.5, 0.5 - half, 0.5 + half)
    )


def test_wilson_interval_with_no_successes_has_the_closed_form_upper_end():
    """At k = 0 the Wilson interval is [0, z^2 / (n + z^2)]."""
    z2 = 1.959963984540054**2

    assert estimators.wilson_interval(0, 10) == pytest.approx(
        (0.0, 0.0, z2 / (10 + z2))
    )


def test_wilson_interval_with_no_trials_is_the_whole_unit_interval():
    point, lo, hi = estimators.wilson_interval(0, 0)

    assert math.isnan(point)
    assert (lo, hi) == (0.0, 1.0)


@given(
    trials=st.integers(1, 10**6),
    fraction=st.floats(0, 1),
    lower=st.floats(0.5, 0.9),
    higher=st.floats(0.91, 0.999),
)
def test_wilson_intervals_contain_the_estimate_and_nest_by_confidence(
    trials, fraction, lower, higher
):
    errors = round(fraction * trials)

    point, lo, hi = estimators.wilson_interval(errors, trials, lower)
    _, wide_lo, wide_hi = estimators.wilson_interval(errors, trials, higher)

    assert 0.0 <= wide_lo <= lo <= point <= hi <= wide_hi <= 1.0


@pytest.mark.parametrize(
    ("errors", "trials", "confidence", "message"),
    [
        (0, -1, 0.95, "trials must be non-negative"),
        (-1, 5, 0.95, "errors must be non-negative"),
        (6, 5, 0.95, "cannot exceed trials"),
        (1, 0, 0.95, "cannot exceed trials"),
        (1, 5, 0.0, "strictly between 0 and 1"),
        (1, 5, 1.0, "strictly between 0 and 1"),
    ],
)
def test_wilson_interval_rejects_invalid_arguments(
    errors, trials, confidence, message
):
    with pytest.raises(ValueError, match=message):
        estimators.wilson_interval(errors, trials, confidence)


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"shots": 0, "accepted": 0, "chunk_accepted": (0,)}, "shots must"),
        ({"errors": -1}, "errors must be non-negative"),
        ({"s1": -6, "chunk_counts": ((-6, 4),)}, "s1 must be non-negative"),
        ({"chunk_weight": (-4,)}, "chunk_weight must be non-negative"),
        ({"n": 0}, "n must be at least 1"),
        ({"accepted": 1001}, "cannot exceed shots"),
        ({"errors": 901}, "cannot exceed accepted"),
        ({"chunk_accepted": (1001,)}, "no chunk can accept more"),
        ({"chunk_accepted": (899,)}, "every chunk accepts at least"),
        ({"chunk_checks": (2, 2)}, "disagree in length"),
        ({"chunk_counts": ((6, 3),)}, "chunk gate counts sum to"),
        ({"chunk_weight": (5,)}, "between 0 and 2 times"),
        (
            {"n": 2, "restricted_checks": True, "chunk_weight": (1,)},
            "between 1 and 3 times",
        ),
        (
            {"n": 2, "restricted_checks": True, "chunk_weight": (7,)},
            "between 1 and 3 times",
        ),
    ],
)
def test_estimate_rejects_inconsistent_counts(fields, message):
    counts = {"errors": 10, "accepted": 900} | fields

    with pytest.raises(ValueError, match=message):
        draw(**counts)


def test_estimate_reports_acceptance_and_logical_error_rate():
    estimate = draw(9, 900)

    assert estimate.size == 10
    assert estimate.acceptance == 0.9
    assert estimate.p_log == 0.01


def test_estimate_with_nothing_accepted_has_an_undefined_error_rate():
    assert math.isnan(draw(0, 0).p_log)


# ---------------------------------------------------------------------------
# estimate_clinr and estimate_direct
# ---------------------------------------------------------------------------
@given(
    n=st.integers(1, 3),
    size=st.integers(1, 25),
    t=st.integers(1, 3),
    r=st.integers(0, 3),
    restricted_checks=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_estimate_at_zero_noise_accepts_every_shot_without_error(
    n, size, t, r, restricted_checks, seed
):
    estimate = estimators.estimate_clinr(
        seeded_circuit(n, size, seed),
        n,
        t,
        r,
        NOISELESS,
        200,
        np.random.default_rng(seed),
        restricted_checks=restricted_checks,
    )

    assert (estimate.accepted, estimate.errors) == (200, 0)
    assert estimate.chunk_accepted == (200,) * t
    assert estimate.chunk_checks == (r,) * t
    assert estimate.size == size
    # Every check weighs at most its family's worst case, and a restricted
    # check at least 1, since only the full group contains the identity.
    least, worst = (1, n + 1) if restricted_checks else (0, 2 * n)
    assert all(
        least * r <= weight <= worst * r for weight in estimate.chunk_weight
    )


def test_each_chunk_records_its_own_acceptance_not_the_joint_one():
    """Each chunk's checks pass more often than all three chunks' together.

    At 20000 shots each chunk accepts about 2000 more shots than the three
    jointly, against a shot-noise spread of about 50, so the strict
    inequality does not fail by chance.
    """
    estimate = estimators.estimate_clinr(
        seeded_circuit(3, 20, seed=1),
        3,
        3,
        3,
        noise_models.uniform_noise(2e-3),
        20000,
        np.random.default_rng(2),
    )

    assert all(chunk > estimate.accepted for chunk in estimate.chunk_accepted)


def test_estimate_repeats_for_the_same_seed():
    circuit = seeded_circuit(3, 20, seed=1)
    noise = noise_models.uniform_noise(2e-3)

    first, second = (
        estimators.estimate_clinr(
            circuit, 3, 2, 2, noise, 2000, np.random.default_rng(4)
        )
        for _ in range(2)
    )

    assert first == second


def test_direct_estimate_at_zero_noise_has_no_chunks_and_no_errors():
    estimate = estimators.estimate_direct(
        seeded_circuit(3, 20, seed=1),
        3,
        NOISELESS,
        200,
        np.random.default_rng(0),
    )

    assert estimate.shots == estimate.accepted == 200
    assert estimate.errors == 0
    assert estimate.chunk_accepted == ()


def test_clinr_estimate_rejects_a_shot_count_below_one():
    circuit, rng = seeded_circuit(2, 5, seed=0), np.random.default_rng(0)

    with pytest.raises(ValueError, match="shots must be positive"):
        estimators.estimate_clinr(circuit, 2, 1, 1, NOISELESS, 0, rng)


def test_direct_estimate_rejects_a_shot_count_below_one():
    circuit, rng = seeded_circuit(2, 5, seed=0), np.random.default_rng(0)

    with pytest.raises(ValueError, match="shots must be positive"):
        estimators.estimate_direct(circuit, 2, NOISELESS, 0, rng)


def test_clinr_estimate_rejects_a_legacy_random_state():
    circuit, rng = seeded_circuit(2, 5, seed=0), np.random.RandomState(0)

    with pytest.raises(TypeError, match=r"numpy\.random\.Generator"):
        estimators.estimate_clinr(
            circuit,
            2,
            1,
            1,
            NOISELESS,
            10,
            rng,  # type: ignore[arg-type]
        )


def test_direct_estimate_rejects_a_legacy_random_state():
    circuit, rng = seeded_circuit(2, 5, seed=0), np.random.RandomState(0)

    with pytest.raises(TypeError, match=r"numpy\.random\.Generator"):
        estimators.estimate_direct(
            circuit,
            2,
            NOISELESS,
            10,
            rng,  # type: ignore[arg-type]
        )


def test_clinr_sweep_rejects_an_empty_list_of_circuits():
    with pytest.raises(ValueError, match="circuits must not be empty"):
        estimators.sweep_clinr(
            [], 2, 1, 1, NOISELESS, 10, np.random.default_rng(0)
        )


def test_direct_sweep_rejects_an_empty_list_of_circuits():
    with pytest.raises(ValueError, match="circuits must not be empty"):
        estimators.sweep_direct([], 2, NOISELESS, 10, np.random.default_rng(0))


def test_sweeps_estimate_each_circuit_in_turn_with_the_options_given():
    """Circuits of different sizes, so the order of the results shows."""
    batch = [seeded_circuit(2, size, size) for size in (3, 5, 8)]
    options = {
        "noisy_checks": False,
        "restricted_checks": True,
        "native_plus": True,
    }

    clinr = estimators.sweep_clinr(
        batch, 2, 1, 1, NOISELESS, 10, np.random.default_rng(0), **options
    )
    direct = estimators.sweep_direct(
        batch, 2, NOISELESS, 10, np.random.default_rng(0)
    )

    assert [estimate.size for estimate in clinr] == [3, 5, 8]
    assert [estimate.size for estimate in direct] == [3, 5, 8]
    assert all(
        (e.noisy_checks, e.restricted_checks, e.native_plus)
        == (False, True, True)
        for e in clinr
    )
    assert [estimate.chunk_checks for estimate in clinr] == [(1,)] * 3
    assert [estimate.errors for estimate in direct] == [0, 0, 0]


def test_cznr_estimate_passes_noiseless_checks_to_the_builder():
    """Noiseless checks at full noise elsewhere: a check cannot misfire."""
    circuit = stim.Circuit("CZ 0 1")
    noise = noise_models.NoiseModel(p1=0.0, p2=0.0, p_m=0.5, p_prep=0.0)

    estimate = estimators.estimate_cznr(
        circuit,
        2,
        1,
        2,
        noise,
        200,
        np.random.default_rng(0),
        noisy_checks=False,
    )

    assert estimate.noisy_checks is False
    assert estimate.chunk_accepted == (200,)


def test_estimate_interval_is_the_wilson_interval_of_its_counts():
    estimate = draw(errors=3, accepted=900)

    assert estimate.interval(0.9) == estimators.wilson_interval(3, 900, 0.9)


# ---------------------------------------------------------------------------
# gate_overhead and gate_overhead_over_draws
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("accepted", "expected"), [(1000, 3.0), (500, 5.6)])
def test_gate_overhead_on_a_hand_worked_estimate(accepted, expected):
    """Worked by hand: (4 + 26 / acceptance) / 10.

    One chunk at n = 1 of 10 gates, with two checks of total weight 4.
    Protected region: preparation 4, gates 10, checks 2*2 + 4 + 2 + 2 = 12,
    so 26 per attempt; teleportation 4 once.
    """
    assert estimators.gate_overhead(draw(0, accepted)) == pytest.approx(
        expected
    )


def test_gate_overhead_charges_each_chunk_its_own_acceptance():
    """Worked by hand: (4 + 26 + 4 + 16 / 0.5) / 10 = 6.6.

    The first chunk holds all 10 gates and always accepts (4 + 26 per
    pass); the second is empty and accepts half the time, so its 4 + 0 + 12
    protected operations are paid twice on average, plus its teleportation.
    Charging both chunks the joint acceptance of 0.5 would give 9.2.
    """
    estimate = draw(
        0,
        500,
        chunk_accepted=(1000, 500),
        chunk_checks=(2, 2),
        chunk_weight=(4, 4),
        chunk_counts=((6, 4), (0, 0)),
    )

    assert estimators.gate_overhead(estimate) == pytest.approx(6.6)


def test_gate_overhead_reads_the_options_from_the_estimate():
    """Worked by hand, as the one-chunk example with every option changed.

    Noiseless checks still cost their operations, and a native |+> drops one
    Hadamard from the preparation and from each check gadget: protected
    region 3 + 10 + 10 and teleportation 4, so (4 + 23) / 10 = 2.7.
    """
    estimate = draw(0, 1000, noisy_checks=False, native_plus=True)

    assert estimators.gate_overhead(estimate) == pytest.approx(2.7)


def test_kept_gate_overhead_averages_draws_rather_than_pooling_them():
    """Worked by hand: the per-draw overheads are 3.0 and 5.6.

    Pooling the counts would give acceptance 0.75 and an overhead of
    (4 + 26 / 0.75) / 10 = 3.87 instead.
    """
    draws = [draw(0, 1000), draw(0, 500)]

    point, _, _ = estimators.gate_overhead_over_draws(draws, checks="kept")

    assert point == pytest.approx(4.3)


def test_fresh_gate_overhead_averages_acceptance_over_draws_not_shots():
    """Worked by hand: (4 + 26 / 0.75) / 10 = 3.87.

    Acceptance 1 on 1000 shots and 0.5 on 500 average to 0.75 per draw,
    where pooling the shots would give 1250 / 1500 and an overhead of 3.52.
    Each attempt costs 26 operations, as in the one-chunk example.
    """
    draws = [draw(0, 1000), draw(0, 250, shots=500)]

    point, _, _ = estimators.gate_overhead_over_draws(draws)

    assert point == pytest.approx((4 + 26 / 0.75) / 10)


def test_gate_overhead_over_draws_on_a_hand_worked_pair():
    """Worked by hand, both readings, with their t intervals.

    The second draw's checks weigh 2 rather than 4, so its attempts cost 24
    operations against 26, and it accepts half its shots.
    Fresh: (4 + 25 / 0.75) / 10 = 3.73; the delta-method deviations are
    (26 / 0.75 - 25 / 0.75 ** 2) / 10 = -44/45 and +44/45, so the standard
    error is 44/45.
    Kept: the per-draw overheads 3.0 and (4 + 24 / 0.5) / 10 = 5.2, with
    mean 4.1 and standard error 1.1.
    t = 12.706 at one degree of freedom.
    """
    draws = [draw(0, 1000), draw(0, 500, chunk_weight=(2,))]
    t = float(stats.t.ppf(0.975, 1))

    fresh = estimators.gate_overhead_over_draws(draws)
    kept = estimators.gate_overhead_over_draws(draws, checks="kept")

    fresh_point, kept_point = (4 + 25 / 0.75) / 10, 4.1
    assert fresh == pytest.approx(
        (fresh_point, fresh_point - t * 44 / 45, fresh_point + t * 44 / 45)
    )
    assert kept == pytest.approx(
        (kept_point, kept_point - t * 1.1, kept_point + t * 1.1)
    )


@pytest.mark.parametrize("checks", ["fresh", "kept"])
def test_gate_overhead_over_one_draw_has_undefined_bounds(checks):
    point, lo, hi = estimators.gate_overhead_over_draws(
        [draw(0, 1000)], checks=checks
    )

    assert point == pytest.approx(3.0)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_fresh_gate_overhead_counts_a_draw_whose_chunk_accepted_nothing():
    """Worked by hand: mean acceptance 0.5, so (4 + 26 / 0.5) / 10 = 5.6.

    Under fresh draws such a check set is simply one that never passes; with
    the checks kept, its overhead is infinite and the reading undefined.
    """
    draws = [draw(0, 1000), draw(0, 0)]

    point, _, _ = estimators.gate_overhead_over_draws(draws)

    assert point == pytest.approx(5.6)
    with pytest.raises(ValueError, match="a chunk accepted no shots"):
        estimators.gate_overhead_over_draws(draws, checks="kept")


def test_fresh_gate_overhead_rejects_a_chunk_that_never_accepted():
    with pytest.raises(ValueError, match="accepted no shots in any"):
        estimators.gate_overhead_over_draws([draw(0, 0), draw(0, 0)])


def test_gate_overhead_over_draws_rejects_draws_without_chunks():
    direct = estimators.Estimate(100, 100, 3, 6, 4, 1)

    with pytest.raises(ValueError, match="estimate has no chunks"):
        estimators.gate_overhead_over_draws([direct, direct])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gate_overhead_at_zero_noise_is_at_most_the_bound(n):
    """With certain acceptance only the check weights differ.

    The estimate charges the drawn weight and the bound the worst case.
    """
    circuit = seeded_circuit(n, 20, seed=n)
    estimate = estimators.estimate_clinr(
        circuit, n, 2, 3, NOISELESS, 100, np.random.default_rng(n)
    )

    measured = estimators.gate_overhead(estimate)
    bound = bounds.overhead_bound(circuit, n, 2, 3, NOISELESS)

    assert measured <= bound


def test_gate_overhead_at_zero_noise_meets_the_bound_at_worst_case_weight():
    """At n = 1 every restricted check has weight 2 = n + 1, the worst case."""
    circuit = seeded_circuit(1, 20, seed=1)
    estimate = estimators.estimate_clinr(
        circuit,
        1,
        2,
        3,
        NOISELESS,
        100,
        np.random.default_rng(1),
        restricted_checks=True,
    )

    measured = estimators.gate_overhead(estimate)
    bound = bounds.overhead_bound(
        circuit, 1, 2, 3, NOISELESS, restricted_checks=True
    )

    assert measured == pytest.approx(bound)


def test_gate_overhead_rejects_an_estimate_without_chunks():
    direct = estimators.Estimate(100, 100, 3, 6, 4, 1)

    with pytest.raises(ValueError, match="estimate has no chunks"):
        estimators.gate_overhead(direct)


def test_gate_overhead_rejects_a_chunk_that_accepted_nothing():
    with pytest.raises(ValueError, match="a chunk accepted no shots"):
        estimators.gate_overhead(draw(0, 0))


# ---------------------------------------------------------------------------
# Statistics over check-set draws
# ---------------------------------------------------------------------------
# 800 of 1000 shots accepted, so a statistic using shots where it should
# use accepted shots gives a different answer.
SPREAD = [draw(80, 800), draw(96, 800), draw(112, 800)]


def clopper_pearson_by_f(
    errors: float, shots: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Return the Clopper-Pearson bounds in the F form of the NCHS standards.

    An oracle independent of the package's beta-quantile form; ``errors``
    and ``shots`` may be effective, not whole, counts.
    """
    alpha = 1 - confidence
    lo = 0.0
    if errors > 0:
        f = stats.f.ppf(alpha / 2, 2 * errors, 2 * (shots - errors + 1))
        lo = 1 / (1 + (shots - errors + 1) / (errors * f))
    hi = 1.0
    if errors < shots:
        f = stats.f.ppf(1 - alpha / 2, 2 * (errors + 1), 2 * (shots - errors))
        hi = 1 / (1 + (shots - errors) / ((errors + 1) * f))
    return lo, hi


def t_ratio_squared(shots: int, draws: int) -> float:
    """Return the squared ratio of the 97.5% t quantiles, shots over draws."""
    return float(
        (stats.t.ppf(0.975, shots - 1) / stats.t.ppf(0.975, draws - 1)) ** 2
    )


def test_kept_logical_error_rate_is_a_korn_graubard_interval_on_draw_rates():
    """Worked by hand from rates 0.10, 0.12 and 0.14 over 800 shots each.

    Mean 0.12 with variance 0.02 ** 2 / 3 over the draws, so the binomial
    variance 0.12 * 0.88 over it is 792 effective shots; the t quantiles at
    2399 and 2 degrees of freedom scale that by 0.2077, to 164.51.
    """
    effective = 792 * t_ratio_squared(2400, 3)

    assert effective == pytest.approx(164.51, abs=0.01)
    assert estimators.logical_error_rate(
        SPREAD, checks="kept"
    ) == pytest.approx(
        (0.12, *clopper_pearson_by_f(0.12 * effective, effective))
    )


def test_fresh_logical_error_rate_weights_draws_by_their_acceptance():
    """Worked by hand, both readings.

    Rates 10/1000 and 40/500, with acceptance 1 and 0.5.
    Fresh: mean fraction failed 0.025 over mean fraction accepted 0.75,
    that is 1/30; the delta-method deviations, (0.01 - 1/30 * 1) / 0.75
    and (0.04 - 1/30 * 0.5) / 0.75, are -7/225 and +7/225, so the standard
    error is 7/225 and the effective number of the 1500 accepted shots is
    (1/30) (29/30) / (7/225) ** 2 = 33.29, scaled by the t quantiles at 1499
    and 1 degrees of freedom. Kept: the mean of 0.01 and 0.08, that is
    0.045.
    """
    draws = [draw(10, 1000), draw(40, 500)]
    effective = (
        (1 / 30) * (29 / 30) / (7 / 225) ** 2 * t_ratio_squared(1500, 2)
    )

    point, lo, hi = estimators.logical_error_rate(draws)

    assert (point, lo, hi) == pytest.approx(
        (1 / 30, *clopper_pearson_by_f(effective / 30, effective))
    )
    assert estimators.logical_error_rate(draws, checks="kept")[
        0
    ] == pytest.approx(0.045)


def test_fresh_logical_error_rate_counts_a_draw_that_accepted_nothing():
    """A check set that never passes lowers the mean acceptance only."""
    draws = [draw(10, 1000), draw(0, 0)]

    point, _, _ = estimators.logical_error_rate(draws)

    assert point == pytest.approx(0.01)
    with pytest.raises(ValueError, match="accepted no shots"):
        estimators.logical_error_rate(draws, checks="kept")


def test_fresh_logical_error_rate_rejects_draws_that_all_accepted_nothing():
    with pytest.raises(ValueError, match="no estimate accepted a shot"):
        estimators.logical_error_rate([draw(0, 0), draw(0, 0)])


@pytest.mark.parametrize("checks", ["fresh", "kept"])
def test_logical_error_rate_without_errors_does_not_collapse(checks):
    """No draw saw an error, so the interval is the exact binomial one.

    Its upper end at 0 errors in 3000 shots is 1 - 0.025 ** (1 / 3000).
    """
    draws = [draw(0, 1000)] * 3

    point, lo, hi = estimators.logical_error_rate(draws, checks=checks)

    assert (point, lo) == (0.0, 0.0)
    assert hi == pytest.approx(1 - 0.025 ** (1 / 3000))


@pytest.mark.parametrize("checks", ["fresh", "kept"])
def test_logical_error_rate_of_identical_draws_is_the_pooled_exact_interval(
    checks,
):
    """Draws that do not differ show no clustering, so every shot counts."""
    draws = [draw(10, 1000)] * 3
    exact = stats.binomtest(30, 3000).proportion_ci(0.95, method="exact")

    assert estimators.logical_error_rate(draws, checks=checks) == (
        pytest.approx((0.01, exact.low, exact.high))
    )


@pytest.mark.slow
@pytest.mark.parametrize("checks", ["fresh", "kept"])
def test_logical_error_rate_covers_a_rare_rate_that_varies_between_draws(
    checks,
):
    """At about one expected error over 10 draws, coverage stays near 95%.

    Each draw's rate is lognormal about 1e-4, with a log-standard deviation
    of 0.5, over 1000 shots. 2000 repetitions put the coverage within about
    0.005; the former Student t interval covered 62% here.
    """
    rng = np.random.default_rng(3)
    mean, sigma = 1e-4, 0.5
    covered = 0
    for _ in range(2000):
        rates = rng.lognormal(math.log(mean) - sigma**2 / 2, sigma, 10)
        draws = [
            estimators.Estimate(1000, 1000, int(errors), 1, 0, 1)
            for errors in rng.binomial(1000, rates)
        ]
        _, lo, hi = estimators.logical_error_rate(draws, checks=checks)
        covered += lo <= mean <= hi

    assert covered / 2000 >= 0.94


@pytest.mark.parametrize("checks", ["fresh", "kept"])
def test_logical_error_rate_of_one_draw_has_undefined_bounds(checks):
    point, lo, hi = estimators.logical_error_rate(
        [draw(100, 1000)], checks=checks
    )

    assert point == 0.1
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_design_effect_compares_between_and_within_draw_variance():
    """Worked by hand: between-draw variance 0.0004 over 0.316 / 2400.

    The mean within-draw variance is
    (0.1 * 0.9 + 0.12 * 0.88 + 0.14 * 0.86) / 3 / 800 = 0.316 / 2400.
    """
    assert estimators.design_effect(SPREAD) == pytest.approx(0.96 / 0.316)


def test_design_effect_rejects_a_single_draw():
    with pytest.raises(ValueError, match="at least two estimates"):
        estimators.design_effect([draw(100, 1000)])


def test_design_effect_is_undefined_when_no_draw_saw_an_error():
    with pytest.raises(ValueError, match="design effect is undefined"):
        estimators.design_effect([draw(0, 1000), draw(0, 900)])


@pytest.mark.parametrize(
    ("draws", "message"),
    [
        ([], "estimates must not be empty"),
        (
            [draw(1, 900), draw(1, 900, s1=5, chunk_counts=((5, 4),))],
            "gate counts disagree",
        ),
        (
            [draw(1, 900), draw(1, 900, chunk_checks=(3,))],
            "checks per chunk disagree",
        ),
        ([draw(1, 900), draw(1, 900, noisy_checks=False)], "options disagree"),
        ([draw(1, 900), draw(1, 900, n=2)], "options disagree"),
    ],
)
@pytest.mark.parametrize(
    "statistic",
    [
        estimators.logical_error_rate,
        estimators.gate_overhead_over_draws,
        estimators.design_effect,
    ],
)
def test_statistics_over_draws_reject_incomparable_estimates(
    statistic, draws, message
):
    with pytest.raises(ValueError, match=message):
        statistic(draws)


@pytest.mark.parametrize(
    "statistic",
    [estimators.logical_error_rate, estimators.gate_overhead_over_draws],
)
def test_statistics_over_draws_reject_a_confidence_outside_the_unit_interval(
    statistic,
):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        statistic(SPREAD, 1.5)


@pytest.mark.parametrize(
    "statistic",
    [estimators.logical_error_rate, estimators.gate_overhead_over_draws],
)
def test_statistics_over_draws_reject_an_unknown_checks_option(statistic):
    with pytest.raises(ValueError, match="'fresh' or 'kept'"):
        statistic(SPREAD, checks="redrawn")


def test_design_effect_rejects_a_draw_that_accepted_nothing():
    with pytest.raises(ValueError, match="accepted no shots"):
        estimators.design_effect([draw(1, 900), draw(0, 0)])


def assert_close_to_first_order(measured: float, expected: float, count: int):
    """Assert ``measured`` agrees with ``expected``, on both sides.

    The allowance is 5 binomial standard errors over ``count`` trials, plus
    5% of ``expected`` for the effect of several faults in one shot.
    """
    standard_error = math.sqrt(expected * (1 - expected) / count)
    assert abs(measured - expected) <= 5 * standard_error + 0.05 * expected


def test_direct_estimate_counts_the_errors_first_order_predicts():
    """An oracle on both sides: a rate of zero, or of every shot, fails."""
    n = 3
    circuit = seeded_circuit(n, 20, seed=1)
    estimate = estimators.estimate_direct(
        circuit, n, LOW_NOISE, 100_000, np.random.default_rng(5)
    )
    built = protocol.build_direct(circuit, n, LOW_NOISE)
    expected = first_order.logical_error_rate(first_order.faults(built))
    assert_close_to_first_order(estimate.p_log, expected, estimate.accepted)


@pytest.mark.parametrize("restricted_checks", [False, True])
def test_clinr_estimate_counts_the_errors_first_order_predicts(
    restricted_checks,
):
    """The same generator, seeded alike, redraws the estimate's checks."""
    n, t, r = 3, 2, 3
    circuit = seeded_circuit(n, 20, seed=1)
    options = {"restricted_checks": restricted_checks}
    estimate = estimators.estimate_clinr(
        circuit,
        n,
        t,
        r,
        LOW_NOISE,
        100_000,
        np.random.default_rng(5),
        **options,
    )
    drawn = checks.draw_checks(
        circuit, n, t, r, np.random.default_rng(5), **options
    )
    built = protocol.build_clinr_tr(circuit, n, drawn, LOW_NOISE)
    expected = first_order.logical_error_rate(first_order.faults(built))
    assert_close_to_first_order(estimate.p_log, expected, estimate.accepted)


def test_cznr_estimate_counts_the_errors_first_order_predicts():
    n, t, r = 4, 2, 2
    circuit = stim.Circuit("CZ 0 1 1 2 0 2 0 1 2 3 1 3")
    estimate = estimators.estimate_cznr(
        circuit, n, t, r, LOW_NOISE, 100_000, np.random.default_rng(5)
    )
    drawn = checks.draw_cznr_checks(circuit, n, t, r, np.random.default_rng(5))
    built = protocol.build_cznr_tr(circuit, n, drawn, LOW_NOISE)
    expected = first_order.logical_error_rate(first_order.faults(built))
    assert_close_to_first_order(estimate.p_log, expected, estimate.accepted)


def test_each_chunk_rejects_the_shots_its_own_faults_predict():
    """Chunk 0 holds 30 noisy CX and chunk 1 30 quieter S, so they differ.

    A chunk's detectors see only the faults of its own resource and checks,
    whose regions are tagged with its index.
    """
    n, t, r, shots = 2, 2, 2, 100_000
    circuit = stim.Circuit("CX 0 1\n" * 30 + "S 0\n" * 30)
    estimate = estimators.estimate_clinr(
        circuit, n, t, r, LOW_NOISE, shots, np.random.default_rng(9)
    )
    drawn = checks.draw_checks(circuit, n, t, r, np.random.default_rng(9))
    found = first_order.faults(
        protocol.build_clinr_tr(circuit, n, drawn, LOW_NOISE)
    )
    for k in range(t):
        expected = sum(
            fault.probability
            for fault in found
            if fault.detected and fault.tag.split("/")[1] == str(k)
        )
        rejected = 1 - estimate.chunk_accepted[k] / shots
        assert_close_to_first_order(rejected, expected, shots)
