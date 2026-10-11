"""Tests for the analytic bounds in ``clinr.bounds``."""

import math
from typing import Any

import numpy as np
import pytest
import stim
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy import stats

from clinr import accounting, bounds, circuits, estimators, noise_models

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
# checks; the Pauli correction the paper charges is tracked as a Pauli frame
# here, at no cost.
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
    Per-draw miss 1/2, the identity included: miss 1/4, escapes 1 + 1/2.
    Locations: preparation 4 (2 resets, H, CX), resource with C 5, one
    check gadget 6, teleportation 4, protected region 4 + 1 + 12 = 17.
    """
    circuit = stim.Circuit("H 0")
    plog = (g(P, 5) / 4 + g(P, 6) * 3 / 2 + g(P, 4)) / (1 - P) ** 17
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
    measurements. With no checks nothing is rejected, so the bound is the
    probability that one of these faults, and swapping any two rates
    changes the value.
    """
    p1, p2, p_m, p_prep = 0.001, 0.002, 0.003, 0.004
    noise = noise_models.NoiseModel(p1=p1, p2=p2, p_m=p_m, p_prep=p_prep)
    resource_clean = (1 - p1) ** 2 * (1 - p2) * (1 - p_prep) ** 2
    teleport_clean = (1 - p1) * (1 - p2) * (1 - p_m) ** 2
    plog = 1 - resource_clean * teleport_clean

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
    # From r = 1 on; the bound without checks is a different, unconditioned
    # formula and can sit below the bound with one check.
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
        for r in range(1, 10)
    ]

    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Soundness against measurement
# ---------------------------------------------------------------------------
def soundness_cases() -> list[Any]:
    """Return the cases, each once: without checks the options do nothing.

    ``n = 25`` runs in the precise tier only, since each case takes over a
    second.
    """
    tiers = [
        ("quick", 8, 20000, ()),
        ("precise", 32, 50000, (pytest.mark.slow,)),
    ]
    options = [
        (restricted, noisy)
        for restricted in (False, True)
        for noisy in (True, False)
    ]
    sizes = [(n, t) for n in (2, 3, 8) for t in (1, 2)] + [(25, 1)]
    cases = []
    for tier, draws, shots, marks in tiers:
        for n, t in sizes:
            if n == 25 and tier == "quick":
                continue
            for r in (0, 4) if n == 25 else (0, 2, 4):
                for restricted, noisy in options if r else options[:1]:
                    cases.append(
                        pytest.param(
                            n,
                            t,
                            r,
                            restricted,
                            noisy,
                            draws,
                            shots,
                            marks=marks,
                            id=f"{tier}-{n}-{t}-{r}-{restricted}-{noisy}",
                        )
                    )
    return cases


@pytest.mark.parametrize(
    ("n", "t", "r", "restricted_checks", "noisy_checks", "draws", "shots"),
    soundness_cases(),
)
def test_measured_logical_error_rate_never_exceeds_the_bound(
    n, t, r, restricted_checks, noisy_checks, draws, shots
):
    """The protocol's rate over random check draws, within 5 standard errors.

    The bound covers the protocol, which draws new checks for every attempt,
    so the measurement is the fresh-draw rate over draws; one fixed check
    set may do worse. Its standard error is taken over the draws, so it
    counts the spread between draws, which dominates, as well as shot noise.
    The quick tier (8 draws of 2e4 shots) is coarse: in the noisiest cases
    the 5 standard errors reach about half the bound, so it catches only a
    grossly unsound bound. The precise tier (32 draws of 5e4 shots) is the
    tighter check.
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
    rate, _, high = estimators.logical_error_rate(estimates)
    standard_error = (high - rate) / float(stats.t.ppf(0.975, draws - 1))

    bound = bounds.plog_bound(
        circuit,
        n,
        t,
        r,
        noise,
        restricted_checks=restricted_checks,
        noisy_checks=noisy_checks,
    )

    assert rate <= bound + 5 * standard_error


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

    s1, s2 = accounting.gate_counts(circuit)
    # The direct implementation comes first, so it wins ties, as do the
    # smaller t and then the smaller r among the pairs.
    candidates = [(g(p, s1 + s2), None)] + [
        (bounds.plog_bound(circuit, n, t, r, noise, **options), (t, r))
        for t, r in feasible
    ]
    best = min(candidates, key=lambda candidate: candidate[0])[1]

    assert (
        bounds.choose_tr(
            circuit,
            n,
            noise,
            max_overhead=max_overhead,
            r_max=r_max,
            t_max=t_max,
            **options,
        )
        == best
    )


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


@pytest.mark.parametrize("max_overhead", [0.0, 0.99, -1.0, math.nan])
def test_choose_tr_rejects_an_overhead_limit_below_the_direct_one(
    max_overhead,
):
    with pytest.raises(ValueError, match="max_overhead must be at least 1"):
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


def test_overhead_bound_is_exact_without_checks():
    """Worked by hand at n = 1, C = H, t = 1 and r = 0.

    The resource is 2 preparations, the |+> Hadamard, a CX and C, so 5
    operations, and the teleportation 4; with no checks nothing is
    rejected, so the overhead is (5 + 4) / 1 whatever the noise.
    """
    noisy = noise_models.uniform_noise(0.3)

    assert bounds.overhead_bound(stim.Circuit("H 0"), 1, 1, 0, noisy) == 9.0


@given(
    n=st.integers(1, 3),
    size=st.integers(1, 20),
    p=st.sampled_from([1e-4, 1e-3, 5e-3]),
    t=st.integers(1, 4),
    seed=st.integers(0, 2**32 - 1),
)
def test_no_checks_never_bound_below_the_direct_implementation(
    n, size, p, t, seed
):
    circuit = seeded_circuit(n, size, seed)
    s1, s2 = accounting.gate_counts(circuit)
    noise = noise_models.uniform_noise(p)

    assert bounds.plog_bound(circuit, n, t, 0, noise) > g(p, s1 + s2)


def test_choose_tr_recommends_a_pair_whose_bound_beats_the_direct_one():
    # 200 gates on 3 qubits at 1e-4: every check's cost is small against
    # the circuit, so some pair's bound falls below the direct one.
    circuit = seeded_circuit(3, 200, seed=4)
    noise = noise_models.asymmetric_noise(1e-4)

    choice = bounds.choose_tr(
        circuit, 3, noise, max_overhead=4.0, r_max=6, t_max=20
    )

    assert choice is not None
    t, r = choice
    assert bounds.plog_bound(circuit, 3, t, r, noise) < (
        bounds.direct_plog_bound(circuit, noise)
    )
    assert bounds.overhead_bound(circuit, 3, t, r, noise) <= 4.0


def test_choose_tr_recommends_the_direct_implementation_when_it_is_best():
    # One gate is cheaper to run directly than any CliNR implementation.
    assert (
        bounds.choose_tr(
            stim.Circuit("H 0"), 1, UNIFORM, max_overhead=100.0, r_max=4
        )
        is None
    )


def test_bounds_are_infinite_when_no_attempt_can_succeed():
    circuit = seeded_circuit(2, 10, seed=0)
    certain_fault = noise_models.uniform_noise(1.0)

    assert bounds.plog_bound(circuit, 2, 1, 1, certain_fault) == math.inf
    assert bounds.overhead_bound(circuit, 2, 1, 1, certain_fault) == math.inf


@given(
    n=st.integers(1, 4),
    size=st.integers(0, 30),
    t=st.integers(1, 3),
    r=st.integers(1, 6),
    noisy_checks=st.booleans(),
    restricted_checks=st.booleans(),
    native_plus=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_plog_terms_add_up_to_the_bound(
    n, size, t, r, noisy_checks, restricted_checks, native_plus, seed
):
    circuit = seeded_circuit(n, size, seed)
    options = {
        "noisy_checks": noisy_checks,
        "restricted_checks": restricted_checks,
        "native_plus": native_plus,
    }

    terms = bounds.plog_terms(circuit, n, t, r, UNIFORM, **options)

    assert len(terms) == t
    assert sum(term.bound for term in terms) == pytest.approx(
        bounds.plog_bound(circuit, n, t, r, UNIFORM, **options), rel=1e-12
    )
    for term in terms:
        assert term.checks == r
        if not noisy_checks:
            assert term.check == 0.0


@pytest.mark.parametrize("n", [1, 2, 5])
def test_plog_terms_carry_each_familys_worst_case_miss(n):
    circuit = seeded_circuit(n, 10, seed=1)

    (uniform,) = bounds.plog_terms(circuit, n, 1, 3, UNIFORM)
    (restricted,) = bounds.plog_terms(
        circuit, n, 1, 3, UNIFORM, restricted_checks=True
    )

    assert uniform.per_check_miss == 0.5
    assert uniform.miss == 0.125
    assert uniform.escapes == 1.75
    assert restricted.per_check_miss == pytest.approx(1 - 2 / (3 * n))


def test_plog_terms_reject_a_bound_without_checks():
    with pytest.raises(ValueError, match="r must be at least 1"):
        bounds.plog_terms(stim.Circuit("H 0"), 1, 1, 0, UNIFORM)


def test_direct_plog_bound_charges_each_gate_at_its_own_rate():
    noise = noise_models.NoiseModel(p1=0.001, p2=0.002, p_m=0.3, p_prep=0.4)
    circuit = stim.Circuit("H 0\nS 1\nCX 0 1")

    bound = bounds.direct_plog_bound(circuit, noise)

    assert bound == pytest.approx(1 - (1 - 0.001) ** 2 * (1 - 0.002))
    assert bounds.direct_plog_bound(stim.Circuit(), noise) == 0.0
