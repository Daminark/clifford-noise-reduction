"""Tests for the Monte Carlo estimators in ``clinr.estimators``."""

import math

import numpy as np
import pytest
import stim
from hypothesis import given
from hypothesis import strategies as st

from clinr import bounds, circuits, estimators, noise_models

NOISELESS = noise_models.uniform_noise(0.0)


def seeded_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return a seeded random ``{H, S, CX}`` circuit on ``n`` qubits."""
    return circuits.random_gate_sequence(n, size, np.random.default_rng(seed))


def draw(errors: int, accepted: int, **fields) -> estimators.Estimate:
    """Return a hand-built one-chunk estimate at n = 1: 10 gates, 2 checks.

    At ``n = 1`` every check has weight 2, so the two weigh 4 in total.

    ``chunk_accepted`` defaults to ``accepted``; any field can be overridden.
    """
    values = {
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
        ({"chunk_weight": (1,)}, "between its number of checks and 2"),
        ({"chunk_weight": (5,)}, "between its number of checks and 2"),
        (
            {"n": 2, "restricted_checks": True, "chunk_weight": (7,)},
            "between its number of checks and 3",
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
        seeded_circuit(n, size, seed), n, t, r, NOISELESS, 200,
        np.random.default_rng(seed), restricted_checks=restricted_checks,
    )

    assert (estimate.accepted, estimate.errors) == (200, 0)
    assert estimate.chunk_accepted == (200,) * t
    assert estimate.chunk_checks == (r,) * t
    assert estimate.size == size
    # Every check has weight between 1 and its family's worst case.
    worst = n + 1 if restricted_checks else 2 * n
    assert all(r <= w <= r * worst for w in estimate.chunk_weight)


def test_each_chunk_records_its_own_acceptance_not_the_joint_one():
    """Each chunk's checks pass more often than all three chunks' together.

    At 20000 shots each chunk accepts about 2000 more shots than the three
    jointly, against a shot-noise spread of about 50, so the strict
    inequality does not fail by chance.
    """
    estimate = estimators.estimate_clinr(
        seeded_circuit(3, 20, seed=1), 3, 3, 3,
        noise_models.uniform_noise(2e-3), 20000, np.random.default_rng(2),
    )

    assert all(a > estimate.accepted for a in estimate.chunk_accepted)


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
        seeded_circuit(3, 20, seed=1), 3, NOISELESS, 200,
        np.random.default_rng(0),
    )

    assert estimate.shots == estimate.accepted == 200
    assert estimate.errors == 0
    assert estimate.chunk_accepted == ()


@pytest.mark.parametrize("estimate", ["clinr", "direct"])
def test_estimates_reject_a_shot_count_below_one(estimate):
    circuit, rng = seeded_circuit(2, 5, seed=0), np.random.default_rng(0)

    with pytest.raises(ValueError, match="shots must be positive"):
        if estimate == "clinr":
            estimators.estimate_clinr(circuit, 2, 1, 1, NOISELESS, 0, rng)
        else:
            estimators.estimate_direct(circuit, 2, NOISELESS, 0, rng)


@pytest.mark.parametrize("estimate", ["clinr", "direct"])
def test_estimates_reject_a_legacy_random_state(estimate):
    circuit, rng = seeded_circuit(2, 5, seed=0), np.random.RandomState(0)

    with pytest.raises(TypeError, match=r"numpy\.random\.Generator"):
        if estimate == "clinr":
            estimators.estimate_clinr(circuit, 2, 1, 1, NOISELESS, 10, rng)
        else:
            estimators.estimate_direct(circuit, 2, NOISELESS, 10, rng)


@pytest.mark.parametrize("sweep", ["clinr", "direct"])
def test_sweeps_reject_an_empty_list_of_circuits(sweep):
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="circuits must not be empty"):
        if sweep == "clinr":
            estimators.sweep_clinr([], 2, 1, 1, NOISELESS, 10, rng)
        else:
            estimators.sweep_direct([], 2, NOISELESS, 10, rng)


# ---------------------------------------------------------------------------
# gate_overhead and mean_gate_overhead
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
        0, 500, chunk_accepted=(1000, 500), chunk_checks=(2, 2),
        chunk_weight=(4, 4), chunk_counts=((6, 4), (0, 0)),
    )

    assert estimators.gate_overhead(estimate) == pytest.approx(6.6)


def test_gate_overhead_reads_the_options_from_the_estimate():
    """Worked by hand, as the one-chunk example with every option changed.

    Idealized checks cost nothing, and a native |+> drops the preparation's
    Hadamard: protected region 3 + 10, teleportation 4, and 1 more for the
    frame, so (4 + 13 + 1) / 10 = 1.8.
    """
    estimate = draw(
        0, 1000, noisy_checks=False, native_plus=True, noisy_q=True
    )

    assert estimators.gate_overhead(estimate) == pytest.approx(1.8)


def test_mean_gate_overhead_averages_draws_rather_than_pooling_them():
    """Worked by hand: the per-draw overheads are 3.0 and 5.6.

    Pooling the counts would give acceptance 0.75 and an overhead of
    (4 + 26 / 0.75) / 10 = 3.87 instead.
    """
    draws = [draw(0, 1000), draw(0, 500)]

    assert estimators.mean_gate_overhead(draws) == pytest.approx(4.3)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gate_overhead_at_zero_noise_is_at_most_the_bound(n):
    """With certain acceptance only the check weights differ.

    The estimate charges the drawn weight and the bound the worst case.
    At n = 1 every non-identity stabilizer has weight 2 = 2n, so the two
    coincide.
    """
    circuit = seeded_circuit(n, 20, seed=n)
    estimate = estimators.estimate_clinr(
        circuit, n, 2, 3, NOISELESS, 100, np.random.default_rng(n)
    )

    measured = estimators.gate_overhead(estimate)
    bound = bounds.overhead_bound(circuit, n, 2, 3, NOISELESS)

    if n == 1:
        assert measured == pytest.approx(bound)
    else:
        assert measured <= bound


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


def test_cluster_interval_is_a_student_t_interval_on_the_draw_means():
    """Worked by hand from rates 0.10, 0.12 and 0.14.

    Mean 0.12, standard error 0.02 / sqrt(3), and t = 4.3027 at two degrees
    of freedom.
    """
    half = 4.302652729749462 * 0.02 / math.sqrt(3)

    assert estimators.cluster_interval(SPREAD) == pytest.approx(
        (0.12, 0.12 - half, 0.12 + half)
    )


def test_cluster_interval_is_clamped_at_zero():
    draws = [draw(10, 1000), draw(20, 1000), draw(30, 1000)]

    assert estimators.cluster_interval(draws)[1] == 0.0


def test_cluster_interval_of_one_draw_has_undefined_bounds():
    point, lo, hi = estimators.cluster_interval([draw(100, 1000)])

    assert point == 0.1
    assert math.isnan(lo) and math.isnan(hi)


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
        ([draw(1, 900), draw(1, 900, s1=5, chunk_counts=((5, 4),))],
         "gate counts disagree"),
        ([draw(1, 900), draw(1, 900, chunk_checks=(3,))],
         "checks per chunk disagree"),
        ([draw(1, 900), draw(1, 900, noisy_checks=False)], "options disagree"),
        ([draw(1, 900), draw(1, 900, n=2)], "options disagree"),
        ([draw(1, 900), draw(0, 0)], "accepted no shots"),
    ],
)
def test_statistics_over_draws_reject_incomparable_estimates(draws, message):
    with pytest.raises(ValueError, match=message):
        estimators.cluster_interval(draws)


def test_cluster_interval_rejects_a_confidence_outside_the_unit_interval():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        estimators.cluster_interval(SPREAD, 1.5)
