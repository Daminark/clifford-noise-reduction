"""Tests for the analytic bounds in ``clinr.bounds``."""

import math

import numpy as np
import pytest
import stim
from hypothesis import assume, given
from hypothesis import strategies as st

from clinr import bounds, circuits, estimators, noise_models

P = 0.01
UNIFORM = noise_models.uniform_noise(P)


def seeded_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return a seeded random ``{H, S, CX}`` circuit on ``n`` qubits."""
    return circuits.random_gate_sequence(n, size, np.random.default_rng(seed))


def g(p: float, locations: int) -> float:
    """Return the probability that one of ``locations`` operations faults."""
    return 1 - (1 - p) ** locations


def paper_plog(n: int, s: int, t: int, r: int, p: float) -> float:
    """Return Theorem 1's logical error bound, arXiv:2407.06583 Eq. (2)."""
    s0 = -(-s // t)
    m0 = 3 * n + s0 + (2 * n + 3) * r
    numerator = g(p, 3 * n + s0) * 2.0**-r + 2 * g(p, 2 * n + 3) + g(p, 5 * n)
    return t * numerator / (1 - p) ** m0


def paper_overhead(n: int, s: int, t: int, r: int, p: float) -> float:
    """Return Theorem 1's gate overhead bound."""
    s0 = -(-s // t)
    m0 = 3 * n + s0 + (2 * n + 3) * r
    return 10 * n / s0 + 2 * m0 / (s0 * (1 - p) ** m0)


# ---------------------------------------------------------------------------
# Against the paper
# ---------------------------------------------------------------------------
# The paper's conventions: one rate, a native |+> preparation, and uniform
# checks; the Pauli correction the paper charges is resolved for free here.
THEOREM_1 = {
    "n": st.integers(1, 5),
    "size": st.integers(1, 150),
    "t": st.integers(1, 6),
    "r": st.integers(0, 8),
    "p": st.floats(1e-5, 1e-2),
    "seed": st.integers(0, 2**32 - 1),
}


@given(**THEOREM_1)
def test_plog_bound_never_exceeds_theorem_1(n, size, t, r, p, seed):
    circuit = seeded_circuit(n, size, seed)
    noise = noise_models.uniform_noise(p)

    plog = bounds.plog_bound(circuit, n, t, r, noise, native_plus=True)

    assert plog <= paper_plog(n, size, t, r, p)


@given(**THEOREM_1)
def test_overhead_bound_never_exceeds_theorem_1_where_it_applies(
    n, size, t, r, p, seed
):
    """Compared only where Theorem 1's overhead formula holds.

    The proof of Theorem 1 of arXiv:2407.06583 uses t * ceil(s/t) <= 2s.
    Beyond it, empty chunks each still pay a teleportation, which this
    implementation charges and the paper's formula does not cover.
    """
    assume(t * -(-size // t) <= 2 * size)
    circuit = seeded_circuit(n, size, seed)
    noise = noise_models.uniform_noise(p)

    overhead = bounds.overhead_bound(circuit, n, t, r, noise, native_plus=True)

    assert overhead <= paper_overhead(n, size, t, r, p)


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------
def test_bounds_on_a_hand_worked_uniform_example():
    """Worked by hand at n = 1, C = H, t = 1 and r = 2 uniform checks.

    Worst-case check weight 2n = 2.
    Per-draw miss (2^1 - 1)/(2^2 - 1) = 1/3: miss 1/9, escapes 1 + 1/3.
    Locations: preparation 4 (2 resets, H, CX), resource with C 5, one
    check gadget 6, teleportation 4, protected region 4 + 1 + 12 = 17.
    """
    circuit = stim.Circuit("H 0")
    plog = (g(P, 5) / 9 + g(P, 6) * 4 / 3 + g(P, 4)) / (1 - P) ** 17
    overhead = (4 + 17 / (1 - P) ** 17) / 1

    assert bounds.plog_bound(circuit, 1, 1, 2, UNIFORM) == pytest.approx(plog)
    assert bounds.overhead_bound(circuit, 1, 1, 2, UNIFORM) == pytest.approx(
        overhead
    )


def test_bounds_on_a_hand_worked_restricted_example():
    """Worked by hand at n = 2, C = CX, t = 1 and r = 1 restricted check.

    Worst-case check weight n + 1 = 3.
    Per-draw miss 1 - 2/(3n) = 2/3; escapes 1.
    Locations: preparation 8, resource with C 9, one check gadget 7,
    teleportation 8, protected region 8 + 1 + 7 = 16.
    """
    circuit = stim.Circuit("CX 0 1")
    plog = (g(P, 9) * 2 / 3 + g(P, 7) + g(P, 8)) / (1 - P) ** 16
    overhead = (8 + 16 / (1 - P) ** 16) / 1

    assert bounds.plog_bound(
        circuit, 2, 1, 1, UNIFORM, restricted_checks=True
    ) == pytest.approx(plog)
    assert bounds.overhead_bound(
        circuit, 2, 1, 1, UNIFORM, restricted_checks=True
    ) == pytest.approx(overhead)


def test_plog_bound_charges_each_kind_of_operation_at_its_own_rate():
    """Worked by hand at n = 1, C = H, t = 1 and r = 0, four distinct rates.

    The resource is 2 preparations, 2 single-qubit gates (the |+> Hadamard
    and C) and a CX; the teleportation is a CX, a Hadamard and 2
    measurements. With no checks the protected region is the resource, so
    swapping any two rates changes the value.
    """
    p1, p2, p_m, p_prep = 0.001, 0.002, 0.003, 0.004
    noise = noise_models.NoiseModel(p1=p1, p2=p2, p_m=p_m, p_prep=p_prep)
    resource_clean = (1 - p1) ** 2 * (1 - p2) * (1 - p_prep) ** 2
    teleport_clean = (1 - p1) * (1 - p2) * (1 - p_m) ** 2
    plog = (1 - resource_clean + 1 - teleport_clean) / resource_clean

    bound = bounds.plog_bound(stim.Circuit("H 0"), 1, 1, 0, noise)

    assert bound == pytest.approx(plog)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------
RATES = ("p1", "p2", "p_m", "p_prep")


@given(
    rates=st.tuples(*[st.floats(0.0, 0.01)] * 4),
    which=st.sampled_from(RATES),
    bump=st.floats(1e-6, 0.01),
    t=st.integers(1, 3),
    r=st.integers(0, 5),
    restricted_checks=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_bounds_do_not_decrease_when_any_rate_rises(
    rates, which, bump, t, r, restricted_checks, seed
):
    circuit = seeded_circuit(3, 20, seed)
    lower = noise_models.NoiseModel(*rates)
    raised = dict(zip(RATES, rates, strict=True))
    raised[which] += bump
    higher = noise_models.NoiseModel(**raised)

    for bound in (bounds.plog_bound, bounds.overhead_bound):
        assert bound(
            circuit, 3, t, r, higher, restricted_checks=restricted_checks
        ) >= bound(
            circuit, 3, t, r, lower, restricted_checks=restricted_checks
        )


@pytest.mark.parametrize("restricted_checks", [False, True])
def test_idealized_checks_never_make_the_bound_worse(restricted_checks):
    circuit = seeded_circuit(3, 30, seed=2)
    values = [
        bounds.plog_bound(
            circuit,
            3,
            1,
            r,
            UNIFORM,
            noisy_checks=False,
            restricted_checks=restricted_checks,
        )
        for r in range(10)
    ]

    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Soundness against measurement
# ---------------------------------------------------------------------------
SOUNDNESS_CASES = [
    (n, t, r) for n in (2, 3, 8) for t in (1, 2) for r in (0, 2, 4)
] + [(25, 1, r) for r in (0, 4)]


@pytest.mark.parametrize(
    ("draws", "shots"),
    [
        pytest.param(8, 20000, id="quick"),
        pytest.param(32, 50000, id="precise", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(("n", "t", "r"), SOUNDNESS_CASES)
@pytest.mark.parametrize("restricted_checks", [False, True])
@pytest.mark.parametrize("noisy_checks", [True, False])
def test_measured_logical_error_rate_never_exceeds_the_bound(
    n, t, r, restricted_checks, noisy_checks, draws, shots
):
    """Mean over random check draws, with 5 standard errors of slack.

    The bound covers the protocol, which draws its checks at random, so the
    measurement averages over draws; one fixed check set may do worse.
    The standard error is that of the mean of the per-draw rates, so it
    counts the spread between draws, which dominates, as well as shot noise.
    The slack is at most 10% of the bound in the quick tier (8 draws of 2e4
    shots) and 4% in the precise tier (32 draws of 5e4 shots), so an unsound
    bound passes only if it understates the rate by less than that.
    """
    noise = noise_models.asymmetric_noise(2e-3)
    circuit = seeded_circuit(n, 16 * n, seed=n)
    rng = np.random.default_rng(1000 * n + 100 * t + 10 * r)
    estimates = [
        estimators.estimate_clinr(
            circuit,
            n,
            t,
            r,
            noise,
            shots,
            rng,
            restricted_checks=restricted_checks,
            noisy_checks=noisy_checks,
        )
        for _ in range(draws)
    ]
    rates = [estimate.p_log for estimate in estimates]
    mean = sum(rates) / len(rates)
    standard_error = float(np.std(rates, ddof=1)) / math.sqrt(len(rates))

    bound = bounds.plog_bound(
        circuit,
        n,
        t,
        r,
        noise,
        restricted_checks=restricted_checks,
        noisy_checks=noisy_checks,
    )

    assert mean <= bound + 5 * standard_error


# ---------------------------------------------------------------------------
# choose_tr
# ---------------------------------------------------------------------------
@given(
    n=st.integers(1, 3),
    size=st.integers(1, 20),
    p=st.sampled_from([1e-4, 1e-3, 5e-3]),
    max_overhead=st.floats(1.5, 30.0),
    r_max=st.integers(0, 5),
    t_max=st.integers(1, 4),
    restricted_checks=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_choose_tr_returns_the_best_pair_within_the_overhead_limit(
    n, size, p, max_overhead, r_max, t_max, restricted_checks, seed
):
    circuit = seeded_circuit(n, size, seed)
    noise = noise_models.uniform_noise(p)
    options = {"restricted_checks": restricted_checks}
    feasible = [
        (t, r)
        for t in range(1, t_max + 1)
        for r in range(r_max + 1)
        if bounds.overhead_bound(circuit, n, t, r, noise, **options)
        <= max_overhead
    ]

    def choose() -> tuple[int, int]:
        return bounds.choose_tr(
            circuit,
            n,
            noise,
            max_overhead=max_overhead,
            r_max=r_max,
            t_max=t_max,
            **options,
        )

    if not feasible:
        with pytest.raises(ValueError, match="no \\(t, r\\) achieves"):
            choose()
        return
    best = min(
        feasible,
        key=lambda tr: bounds.plog_bound(
            circuit, n, tr[0], tr[1], noise, **options
        ),
    )

    assert choose() == best


@pytest.mark.parametrize(
    ("n", "r_max", "t_max", "message"),
    [
        (0, 2, 2, "n must be at least 1"),
        (2, -1, 2, "r_max must be non-negative"),
        (2, 2, 0, "t_max must be at least 1"),
    ],
)
def test_choose_tr_rejects_invalid_search_ranges(n, r_max, t_max, message):
    with pytest.raises(ValueError, match=message):
        bounds.choose_tr(
            stim.Circuit("CX 0 1"),
            n,
            UNIFORM,
            max_overhead=10.0,
            r_max=r_max,
            t_max=t_max,
        )


@pytest.mark.parametrize(
    ("n", "r", "message"),
    [(0, 1, "n must be at least 1"), (2, -1, "r must be non-negative")],
)
def test_bounds_reject_invalid_sizes(n, r, message):
    for bound in (bounds.plog_bound, bounds.overhead_bound):
        with pytest.raises(ValueError, match=message):
            bound(stim.Circuit("CX 0 1"), n, 1, r, UNIFORM)


@pytest.mark.parametrize("max_overhead", [0.0, -1.0, math.nan])
def test_choose_tr_rejects_an_overhead_limit_that_is_not_positive(
    max_overhead,
):
    with pytest.raises(ValueError, match="max_overhead must be positive"):
        bounds.choose_tr(
            stim.Circuit("CX 0 1"),
            2,
            UNIFORM,
            max_overhead=max_overhead,
            r_max=2,
        )


def test_choose_tr_rejects_a_circuit_without_gates():
    with pytest.raises(ValueError, match="at least one gate"):
        bounds.choose_tr(
            stim.Circuit(), 2, UNIFORM, max_overhead=10.0, r_max=2
        )


def test_overhead_bound_rejects_a_circuit_without_gates():
    with pytest.raises(ValueError, match="at least one gate"):
        bounds.overhead_bound(stim.Circuit(), 2, 1, 1, UNIFORM)


def test_resolving_the_frame_adds_n_single_qubit_locations_once():
    """Worked by hand at n = 2 and p = 0.01, for any t.

    The logical error rate gains 1 - 0.99^2, and the overhead gains 2
    operations over the circuit's 4 gates.
    """
    circuit = seeded_circuit(2, 4, seed=0)

    for t in (1, 2, 4):
        free, charged = (
            bounds.plog_bound(circuit, 2, t, 2, UNIFORM, noisy_q=noisy_q)
            for noisy_q in (False, True)
        )
        cheap, costly = (
            bounds.overhead_bound(circuit, 2, t, 2, UNIFORM, noisy_q=noisy_q)
            for noisy_q in (False, True)
        )
        assert charged - free == pytest.approx(1 - 0.99**2)
        assert costly - cheap == pytest.approx(2 / 4)


def test_bounds_are_infinite_when_no_attempt_can_succeed():
    circuit = seeded_circuit(2, 10, seed=0)
    certain_fault = noise_models.uniform_noise(1.0)

    assert bounds.plog_bound(circuit, 2, 1, 1, certain_fault) == math.inf
    assert bounds.overhead_bound(circuit, 2, 1, 1, certain_fault) == math.inf
