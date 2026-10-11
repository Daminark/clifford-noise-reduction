"""Tests for CZNR, the variant of CliNR for circuits of CZ gates.

Its builder lives in ``clinr.protocol``, its checks in ``clinr.checks``,
its regions in ``clinr.accounting``, its bounds in ``clinr.bounds`` and its
estimator in ``clinr.estimators``.
"""

import itertools
import math

import numpy as np
import pytest
import stim
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from clinr import (
    accounting,
    bounds,
    checks,
    estimators,
    first_order,
    noise_models,
    protocol,
)

NOISELESS = noise_models.uniform_noise(0.0)
# Four distinct rates, so every noise channel can be attributed to its kind.
NOISE = noise_models.NoiseModel(p1=0.001, p2=0.002, p_m=0.003, p_prep=0.004)


def cz_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return ``size`` CZ gates on random distinct pairs of ``n`` qubits."""
    rng = np.random.default_rng(seed)
    circuit = stim.Circuit()
    for _ in range(size if n > 1 else 0):
        a, b = rng.choice(n, 2, replace=False)
        circuit.append("CZ", [int(a), int(b)])
    return circuit


def complete_graph(n: int) -> stim.Circuit:
    """Return one CZ on every pair of ``n`` qubits."""
    circuit = stim.Circuit()
    for a, b in itertools.combinations(range(n), 2):
        circuit.append("CZ", [a, b])
    return circuit


# ---------------------------------------------------------------------------
# Graph states and checks
# ---------------------------------------------------------------------------
def test_graph_state_gens_of_a_path_are_x_times_z_on_the_neighbors():
    # The path 0 - 1 - 2, with the edge 0 - 2 applied twice, which cancels.
    circuit = stim.Circuit("CZ 0 1 1 2 0 2 2 0")
    tableau = protocol.chunk_circuit(circuit, 3, 1)[0].to_tableau()

    gens = protocol.graph_state_gens(tableau)

    assert [str(gen)[1:] for gen in gens] == ["XZ_", "ZXZ", "_ZX"]


def test_drawn_checks_stabilize_their_chunk():
    n, t, r = 4, 3, 5
    circuit = cz_circuit(n, 12, seed=1)

    drawn = checks.draw_cznr_checks(circuit, n, t, r, np.random.default_rng(1))

    assert [len(chunk_checks) for chunk_checks in drawn] == [r] * t
    for chunk, chunk_checks in zip(
        protocol.chunk_circuit(circuit, n, t), drawn, strict=True
    ):
        gens = protocol.graph_state_gens(chunk.to_tableau())
        for check in chunk_checks:
            assert len(check) == n
            assert all(check.commutes(gen) for gen in gens)


def test_a_uniform_check_catches_every_damaging_fault_half_the_time():
    """Exhaustively over Paulis on the graph state of the complete graph."""
    n = 3
    tableau = protocol.chunk_circuit(complete_graph(n), n, 1)[0].to_tableau()
    gens = protocol.graph_state_gens(tableau)
    group = []
    for bits in itertools.product([0, 1], repeat=n):
        element = stim.PauliString(n)
        for bit, gen in zip(bits, gens, strict=True):
            if bit:
                element *= gen
        group.append(element)

    for letters in itertools.product("IXYZ", repeat=n):
        error = stim.PauliString("".join(letters))
        if all(error.commutes(element) for element in group):
            continue  # harmless: a stabilizer, up to sign
        caught = sum(not error.commutes(element) for element in group)
        assert caught == len(group) // 2


def unsigned(pauli: stim.PauliString) -> str:
    """Return a Pauli string without its sign."""
    return str(pauli)[1:]


@pytest.mark.parametrize("n", [2, 3])
def test_uniform_cznr_checks_are_uniform_over_the_group(n):
    """Chi-square test at p > 1e-6, about a five-sigma departure.

    With 1000 expected draws per element, an element never drawn, or drawn
    at twice its share, gives a p-value far below the threshold.
    """
    circuit = complete_graph(n)
    gens = protocol.graph_state_gens(
        protocol.chunk_circuit(circuit, n, 1)[0].to_tableau()
    )
    group = {}
    for bits in itertools.product([0, 1], repeat=n):
        element = stim.PauliString(n)
        for bit, gen in zip(bits, gens, strict=True):
            if bit:
                element *= gen
        group[unsigned(element)] = 0
    draws = 1000 * 2**n

    (drawn,) = checks.draw_cznr_checks(
        circuit, n, 1, draws, np.random.default_rng(0)
    )

    for check in drawn:
        group[unsigned(check)] += 1
    assert stats.chisquare(list(group.values())).pvalue > 1e-6


def test_uniform_cznr_checks_miss_a_single_z_at_rate_one_half():
    """Five-sigma binomial bounds on single draws and on triples.

    Z on one qubit anticommutes with one generator only, so a uniform check
    misses it with probability 1/2. At 30000 draws, 5 sigma on the
    single-draw rate is about 0.014, inside the gap to the 0.45 a sampler
    that kept each generator with probability 0.55 would give. The triple
    rate checks independence: a sampler that reused draws would miss
    triples at a rate nearer 1/4 than 1/8.
    """
    n, draws = 3, 30000
    (drawn,) = checks.draw_cznr_checks(
        complete_graph(n), n, 1, draws, np.random.default_rng(7)
    )

    single_z = stim.PauliString("Z" + "_" * (n - 1))
    missed = [check.commutes(single_z) for check in drawn]
    single = sum(missed) / draws
    triples = [all(missed[i : i + 3]) for i in range(0, draws, 3)]
    triple = sum(triples) / len(triples)

    assert abs(single - 1 / 2) < 5 * math.sqrt(1 / 4 / draws)
    assert abs(triple - 1 / 8) < 5 * math.sqrt(7 / 64 / len(triples))


@pytest.mark.parametrize(
    ("r", "rng", "error", "message"),
    [
        (-1, np.random.default_rng(0), ValueError, "non-negative"),
        (1, 0, TypeError, "numpy.random.Generator"),
    ],
)
def test_draw_cznr_checks_rejects_invalid_arguments(r, rng, error, message):
    with pytest.raises(error, match=message):
        checks.draw_cznr_checks(stim.Circuit("CZ 0 1"), 2, 1, r, rng)


def test_graph_state_gens_rejects_an_empty_tableau():
    with pytest.raises(ValueError, match="at least one qubit"):
        protocol.graph_state_gens(stim.Tableau(0))


def test_draw_cznr_checks_rejects_gates_other_than_cz():
    with pytest.raises(ValueError, match="CZ gates only"):
        checks.draw_cznr_checks(
            stim.Circuit("CX 0 1"), 2, 1, 1, np.random.default_rng(0)
        )


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------
def assert_exact(built: stim.Circuit) -> None:
    """At zero noise every detector and observable is deterministic.

    Stim refuses to build a detector error model for a circuit with a
    non-deterministic detector or observable, and reports both against the
    noiseless reference, so a model with no errors is the exactness claim.
    A teleportation whose Pauli frame were tracked wrongly would leave an
    observable random.
    """
    assert built.detector_error_model().num_errors == 0


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("t", [1, 2, 3])
@pytest.mark.parametrize("r", [0, 2])
@pytest.mark.parametrize("native_plus", [False, True])
def test_cznr_is_exact_at_zero_noise(n, t, r, native_plus):
    circuit = cz_circuit(n, 10, seed=n + t)
    drawn = checks.draw_cznr_checks(circuit, n, t, r, np.random.default_rng(2))

    built = protocol.build_cznr_tr(
        circuit, n, drawn, NOISELESS, native_plus=native_plus
    )

    assert_exact(built)
    assert built.num_detectors == t * r
    assert built.num_observables == 2 * n
    assert built.num_qubits == 3 * n + (1 if r else 0)


@pytest.mark.parametrize(
    ("circuit", "drawn", "message"),
    [
        (stim.Circuit("CX 0 1"), [[]], "CZ gates only"),
        (
            stim.Circuit("CZ 0 1"),
            [[], [stim.PauliString("XZ")]],
            "same number",
        ),
        (stim.Circuit("CZ 0 1"), [[stim.PauliString("XZI")]], "length 3"),
        (
            stim.Circuit("CZ 0 1"),
            [[stim.PauliString("XI")]],
            "does not stabilize",
        ),
    ],
)
def test_cznr_builder_rejects_invalid_input(circuit, drawn, message):
    with pytest.raises(ValueError, match=message):
        protocol.build_cznr_tr(circuit, 2, drawn, NOISELESS)


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------
def fault_locations(built: stim.Circuit) -> dict[str, accounting.Region]:
    """Count the noisy operations in ``built`` by kind, for each tag."""
    counts: dict[str, list[int]] = {}
    for index in range(len(built)):
        inst = built[index]
        assert isinstance(inst, stim.CircuitInstruction)
        targets = len(inst.targets_copy())
        kinds = counts.setdefault(inst.tag, [0, 0, 0, 0])
        if inst.name == "DEPOLARIZE1":
            kinds[0] += targets
        elif inst.name == "DEPOLARIZE2":
            kinds[1] += targets // 2
        elif inst.name == "M" and inst.gate_args_copy():
            kinds[2] += targets
        elif inst.name in ("X_ERROR", "Z_ERROR"):
            kinds[3] += targets
    return {
        tag: accounting.Region(*kinds)
        for tag, kinds in counts.items()
        if any(kinds)
    }


@given(
    n=st.integers(1, 4),
    size=st.integers(0, 20),
    t=st.integers(1, 3),
    r=st.integers(0, 3),
    native_plus=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_cznr_regions_count_every_fault_location_the_builder_emits(
    n, size, t, r, native_plus, seed
):
    circuit = cz_circuit(n, size, seed)
    drawn = checks.draw_cznr_checks(
        circuit, n, t, r, np.random.default_rng(seed)
    )
    built = protocol.build_cznr_tr(
        circuit, n, drawn, NOISE, native_plus=native_plus
    )

    expected = {}
    for k, (chunk, chunk_checks) in enumerate(
        zip(protocol.chunk_circuit(circuit, n, t), drawn, strict=True)
    ):
        s1, s2 = accounting.gate_counts(chunk)
        expected[f"prep/{k}"] = accounting.cznr_prep_region(n, native_plus)
        expected[f"gates/{k}"] = accounting.Region(
            n1=s1, n2=s2, nm=0, n_prep=0
        )
        for j, check in enumerate(chunk_checks):
            expected[f"check/{k}/{j}"] = accounting.check_region(
                1, check.weight, native_plus
            )
        expected[f"teleport/{k}"] = accounting.cznr_teleport_region(n)
    expected = {tag: region for tag, region in expected.items() if any(region)}

    assert fault_locations(built) == expected


def test_first_order_attributes_cznr_faults_to_every_region():
    circuit = complete_graph(3)
    drawn = checks.draw_cznr_checks(circuit, 3, 1, 2, np.random.default_rng(3))
    built = protocol.build_cznr_tr(circuit, 3, drawn, NOISE)

    regions = {fault.tag.split("/")[0] for fault in first_order.faults(built)}

    assert regions == {"prep", "gates", "check", "teleport"}


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
def test_cznr_plog_bound_without_checks_on_one_cz():
    """Worked by hand at n = 2, C = CZ 0 1, t = 1, r = 0.

    The resource is two |+> (two resets and two Hadamards) and the CZ; the
    teleportation is two CX and two measurements. With no checks nothing
    is rejected, so the bound is the probability that one of these faults.
    """
    p1, p2, p_m, p_prep = 0.001, 0.002, 0.003, 0.004
    clean = (1 - p_prep) ** 2 * (1 - p1) ** 2 * (1 - p2) ** 3 * (1 - p_m) ** 2

    bound = bounds.cznr_plog_bound(stim.Circuit("CZ 0 1"), 2, 1, 0, NOISE)

    assert bound == pytest.approx(1 - clean)


def test_cznr_plog_bound_with_one_uniform_check_on_one_cz():
    """Worked by hand at n = 2, C = CZ 0 1, t = 1, r = 1, uniform checks.

    The check is charged at weight n = 2: a reset, two Hadamards, two
    controlled Paulis and a measurement. One check misses with probability
    1/2, its own faults escape with probability 1, and the bound divides by
    the probability that nothing in the resource or the check faults.
    """
    p1, p2, p_m, p_prep = 0.001, 0.002, 0.003, 0.004
    resource = (1 - p_prep) ** 2 * (1 - p1) ** 2 * (1 - p2)
    check = (1 - p_prep) * (1 - p1) ** 2 * (1 - p2) ** 2 * (1 - p_m)
    teleport = (1 - p2) ** 2 * (1 - p_m) ** 2
    expected = ((1 - resource) / 2 + (1 - check) + (1 - teleport)) / (
        resource * check
    )

    bound = bounds.cznr_plog_bound(stim.Circuit("CZ 0 1"), 2, 1, 1, NOISE)

    assert bound == pytest.approx(expected)


def test_cznr_overhead_bound_is_exact_without_checks():
    # Two |+> (4 operations), one CZ and a teleportation of 4: 9 per gate.
    noisy = noise_models.uniform_noise(0.3)

    bound = bounds.cznr_overhead_bound(stim.Circuit("CZ 0 1"), 2, 1, 0, noisy)

    assert bound == 9.0


@pytest.mark.parametrize(
    "function", [bounds.cznr_plog_bound, bounds.cznr_overhead_bound]
)
def test_cznr_bounds_reject_gates_other_than_cz(function):
    with pytest.raises(ValueError, match="CZ gates only"):
        function(stim.Circuit("CX 0 1"), 2, 1, 1, NOISE)


@given(
    n=st.integers(1, 6),
    size=st.integers(1, 30),
    t=st.integers(1, 4),
    r=st.integers(0, 6),
    noisy_checks=st.booleans(),
    native_plus=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_cznr_bounds_never_exceed_clinr_ones_on_the_same_circuit(
    n, size, t, r, noisy_checks, native_plus, seed
):
    """At the same t and r, CZNR does less in every region.

    Its preparation, checks and teleportation each have fewer operations
    of every kind than CliNR's, with the same miss of 1/2 per uniform
    check, so each term of its bound is the smaller.
    """
    circuit = cz_circuit(n, size, seed)
    options = {"noisy_checks": noisy_checks, "native_plus": native_plus}

    assert bounds.cznr_plog_bound(
        circuit, n, t, r, NOISE, **options
    ) <= bounds.plog_bound(circuit, n, t, r, NOISE, **options)
    if size and n > 1:
        assert bounds.cznr_overhead_bound(
            circuit, n, t, r, NOISE, **options
        ) <= bounds.overhead_bound(circuit, n, t, r, NOISE, **options)


@pytest.mark.parametrize(
    ("n", "t", "r"), [(3, 1, 1), (4, 1, 2), (6, 1, 4), (6, 2, 2), (5, 3, 1)]
)
def test_cznr_first_order_rate_is_below_clinr_on_complete_graphs(n, t, r):
    """Means over 40 draws of each scheme's checks, 5 standard errors apart.

    To first order, the logical error rate is the probability of a single
    fault that damages the output and escapes every check; CZNR has fewer
    such faults in every region.
    """
    circuit = complete_graph(n)
    rng = np.random.default_rng(n + t + r)
    draws = 40

    def rate(built: stim.Circuit) -> float:
        return first_order.logical_error_rate(first_order.faults(built))

    cznr = [
        rate(
            protocol.build_cznr_tr(
                circuit,
                n,
                checks.draw_cznr_checks(circuit, n, t, r, rng),
                NOISE,
            )
        )
        for _ in range(draws)
    ]
    clinr = [
        rate(
            protocol.build_clinr_tr(
                circuit, n, checks.draw_checks(circuit, n, t, r, rng), NOISE
            )
        )
        for _ in range(draws)
    ]

    gap = np.mean(clinr) - np.mean(cznr)
    standard_error = math.sqrt(
        (np.var(cznr, ddof=1) + np.var(clinr, ddof=1)) / draws
    )
    assert gap > 5 * standard_error


@pytest.mark.parametrize(("n", "t", "r"), [(3, 1, 2), (6, 1, 4), (6, 2, 2)])
def test_measured_cznr_rate_never_exceeds_the_bound(n, t, r):
    """The fresh-draw rate over draws, with 5 standard errors, as for CliNR."""
    noise = noise_models.asymmetric_noise(2e-3)
    circuit = complete_graph(n)
    rng = np.random.default_rng(10 * n + r)
    draws = 8
    estimates = [
        estimators.estimate_cznr(circuit, n, t, r, noise, 20_000, rng)
        for _ in range(draws)
    ]
    rate, _, high = estimators.logical_error_rate(estimates)
    standard_error = (high - rate) / float(stats.t.ppf(0.975, draws - 1))

    bound = bounds.cznr_plog_bound(circuit, n, t, r, noise)

    assert rate <= bound + 5 * standard_error


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------
def test_cznr_gate_overhead_at_zero_noise_counts_the_built_operations():
    # Complete graph on 3 qubits, t = 1, r = 2: |+> on 3 (6), 3 CZ, two
    # checks of total weight W, each with 4 more operations (8 + W), and
    # the teleportation (6).
    n = 3
    estimate = estimators.estimate_cznr(
        complete_graph(n), n, 1, 2, NOISELESS, 100, np.random.default_rng(0)
    )
    weight = estimate.chunk_weight[0]

    assert estimate.cznr
    assert estimate.acceptance == 1.0
    assert estimators.gate_overhead(estimate) == pytest.approx(
        (6 + 3 + 8 + weight + 6) / 3
    )


@pytest.mark.parametrize("mode", ["fresh", "kept"])
def test_cznr_gate_overhead_over_draws_at_zero_noise_counts_the_operations(
    mode,
):
    # As in the test above, with the drawn weight averaged over the draws:
    # every shot is accepted, so both modes average the operations.
    n = 3
    rng = np.random.default_rng(1)
    draws = [
        estimators.estimate_cznr(
            complete_graph(n), n, 1, 2, NOISELESS, 50, rng
        )
        for _ in range(6)
    ]
    weight = sum(draw.chunk_weight[0] for draw in draws) / len(draws)

    point, _, _ = estimators.gate_overhead_over_draws(draws, checks=mode)

    assert point == pytest.approx((6 + 3 + 8 + weight + 6) / 3)


def test_estimate_cznr_rejects_a_non_positive_number_of_shots():
    with pytest.raises(ValueError, match="shots must be positive"):
        estimators.estimate_cznr(
            stim.Circuit("CZ 0 1"),
            2,
            1,
            1,
            NOISELESS,
            0,
            np.random.default_rng(0),
        )


def test_cznr_estimates_are_not_comparable_with_clinr_ones():
    circuit = stim.Circuit("CZ 0 1")
    rng = np.random.default_rng(0)
    cznr = estimators.estimate_cznr(circuit, 2, 1, 1, NOISELESS, 10, rng)
    clinr = estimators.estimate_clinr(circuit, 2, 1, 1, NOISELESS, 10, rng)

    with pytest.raises(ValueError, match="options disagree"):
        estimators.logical_error_rate([cznr, clinr])


def test_a_cznr_estimate_rejects_checks_heavier_than_n():
    with pytest.raises(ValueError, match="between 0 and 2"):
        estimators.Estimate(
            100,
            100,
            0,
            0,
            1,
            2,
            chunk_accepted=(100,),
            chunk_checks=(1,),
            chunk_weight=(3,),
            chunk_counts=((0, 1),),
            cznr=True,
        )


def test_a_cznr_estimate_rejects_restricted_checks():
    with pytest.raises(ValueError, match="uniform checks only"):
        estimators.Estimate(
            100, 100, 0, 0, 1, 2, restricted_checks=True, cznr=True
        )
