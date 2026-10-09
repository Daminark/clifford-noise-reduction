"""Tests for CZNR, the variant of CliNR for circuits of CZ gates.

Its builder lives in ``clinr.protocol``, its checks in ``clinr.checks``,
its regions in ``clinr.accounting``, its bounds in ``clinr.bounds`` and its
estimator in ``clinr.estimators``.
"""

import itertools

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


@pytest.mark.parametrize("restricted_checks", [False, True])
def test_drawn_checks_stabilize_their_chunk_and_restricted_ones_are_gens(
    restricted_checks,
):
    n, t, r = 4, 3, 5
    circuit = cz_circuit(n, 12, seed=1)

    drawn = checks.draw_cznr_checks(
        circuit,
        n,
        t,
        r,
        np.random.default_rng(1),
        restricted_checks=restricted_checks,
    )

    assert [len(chunk_checks) for chunk_checks in drawn] == [r] * t
    for chunk, chunk_checks in zip(
        protocol.chunk_circuit(circuit, n, t), drawn, strict=True
    ):
        gens = protocol.graph_state_gens(chunk.to_tableau())
        for check in chunk_checks:
            assert len(check) == n
            assert all(check.commutes(gen) for gen in gens)
            if restricted_checks:
                assert any(check == gen for gen in gens)


def test_a_generator_catches_a_damaging_fault_at_least_one_time_in_n():
    """Exhaustively over Paulis on the graph state of the complete graph.

    A uniform element of the group catches every damaging fault with
    probability exactly 1/2; a uniform generator with probability at least
    1/n, attained by a single Z.
    """
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

    least = 1.0
    for letters in itertools.product("IXYZ", repeat=n):
        error = stim.PauliString("".join(letters))
        if all(error.commutes(element) for element in group):
            continue  # harmless: a stabilizer, up to sign
        caught = sum(not error.commutes(element) for element in group)
        assert caught == len(group) // 2
        least = min(least, sum(not error.commutes(g) for g in gens) / n)

    assert least == pytest.approx(1 / n)
    single_z = stim.PauliString("Z" + "I" * (n - 1))
    assert sum(not single_z.commutes(gen) for gen in gens) == 1


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
@pytest.mark.parametrize("restricted_checks", [False, True])
@pytest.mark.parametrize("native_plus", [False, True])
def test_cznr_is_exact_at_zero_noise(n, t, r, restricted_checks, native_plus):
    circuit = cz_circuit(n, 10, seed=n + t)
    drawn = checks.draw_cznr_checks(
        circuit,
        n,
        t,
        r,
        np.random.default_rng(2),
        restricted_checks=restricted_checks,
    )

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
    restricted_checks=st.booleans(),
    native_plus=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_cznr_regions_count_every_fault_location_the_builder_emits(
    n, size, t, r, restricted_checks, native_plus, seed
):
    circuit = cz_circuit(n, size, seed)
    drawn = checks.draw_cznr_checks(
        circuit,
        n,
        t,
        r,
        np.random.default_rng(seed),
        restricted_checks=restricted_checks,
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


def test_restricted_bound_charges_the_largest_generator_of_the_chunk():
    # Each chunk's check is charged at its largest generator's weight: 4
    # for the star centered at 0, 3 for the path, with as many gates.
    star = stim.Circuit("CZ 0 1 0 2 0 3")
    path = stim.Circuit("CZ 0 1 1 2 2 3")
    noise = noise_models.uniform_noise(1e-3)

    def overhead(circuit: stim.Circuit) -> float:
        return bounds.cznr_overhead_bound(
            circuit, 4, 1, 1, noise, restricted_checks=True
        )

    assert overhead(star) > overhead(path)


@pytest.mark.parametrize(
    "function", [bounds.cznr_plog_bound, bounds.cznr_overhead_bound]
)
def test_cznr_bounds_reject_gates_other_than_cz(function):
    with pytest.raises(ValueError, match="CZ gates only"):
        function(stim.Circuit("CX 0 1"), 2, 1, 1, NOISE)


@pytest.mark.parametrize(("n", "t", "r"), [(3, 1, 2), (6, 1, 4), (6, 2, 2)])
@pytest.mark.parametrize("restricted_checks", [False, True])
def test_measured_cznr_rate_never_exceeds_the_bound(
    n, t, r, restricted_checks
):
    """The fresh-draw rate over draws, with 5 standard errors, as for CliNR."""
    noise = noise_models.asymmetric_noise(2e-3)
    circuit = complete_graph(n)
    rng = np.random.default_rng(10 * n + r)
    draws = 8
    estimates = [
        estimators.estimate_cznr(
            circuit,
            n,
            t,
            r,
            noise,
            20_000,
            rng,
            restricted_checks=restricted_checks,
        )
        for _ in range(draws)
    ]
    rate, _, high = estimators.logical_error_rate(estimates)
    standard_error = (high - rate) / float(stats.t.ppf(0.975, draws - 1))

    bound = bounds.cznr_plog_bound(
        circuit, n, t, r, noise, restricted_checks=restricted_checks
    )

    assert rate <= bound + 5 * standard_error


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------
def test_cznr_gate_overhead_at_zero_noise_counts_the_built_operations():
    # Complete graph on 3 qubits, t = 1, r = 2 generators of weight 3:
    # |+> on 3 (6), 3 CZ, two checks of 3 + 4 (14), teleportation (6).
    n = 3
    estimate = estimators.estimate_cznr(
        complete_graph(n),
        n,
        1,
        2,
        NOISELESS,
        100,
        np.random.default_rng(0),
        restricted_checks=True,
    )

    assert estimate.cznr
    assert estimate.acceptance == 1.0
    assert estimators.gate_overhead(estimate) == pytest.approx(
        (6 + 3 + 14 + 6) / 3
    )


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
