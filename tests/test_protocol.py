"""Tests for circuit construction in ``clinr.protocol``."""

import itertools

import numpy as np
import pytest
import stim
from hypothesis import given, settings
from hypothesis import strategies as st

from clinr import checks, circuits, noise_models, protocol

NOISELESS = noise_models.uniform_noise(0.0)


def seeded_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return a seeded random ``{H, S, CX}`` circuit on ``n`` qubits."""
    return circuits.random_gate_sequence(n, size, np.random.default_rng(seed))


def full_tableau(circuit: stim.Circuit, n: int) -> stim.Tableau:
    """Return the tableau of ``circuit``, padded to width ``n``."""
    return protocol.chunk_circuit(circuit, n, 1)[0].to_tableau()


def one_instruction(name: str) -> stim.Circuit:
    """Return some valid single instruction of gate ``name``.

    Stim's gates differ in the targets and arguments they accept, so this
    tries the common shapes in turn. It fails loudly for a gate none of them
    fits, which is what a new stim gate should do to this suite.
    """
    shapes = [
        "{} 0", "{} 0 1", "{}(0.1) 0", "{}(0.1) 0 1", "{} X0", "{}(0.1) X0",
        "{}(0.1,0.1,0.1) 0", "{}(0.1,0.1,0.1,0.1) 0",
        "{}(" + ",".join(["0.01"] * 15) + ") 0 1",
        "{}", "{}(0)", "{}(1)", "{}(1,2) 0",
    ]
    for shape in shapes:
        try:
            return stim.Circuit(shape.format(name))
        except ValueError:
            continue
    raise AssertionError(f"no test instruction shape fits stim gate {name}")


# ---------------------------------------------------------------------------
# gate_list
# ---------------------------------------------------------------------------
GATES = stim.gate_data()
UNITARY_1Q_2Q = sorted(
    name for name, data in GATES.items()
    if data.is_unitary
    and (data.is_single_qubit_gate or data.is_two_qubit_gate)
)
OPERATIONS = sorted(
    name for name, data in GATES.items()
    if not data.is_unitary
    and (data.is_noisy_gate or data.produces_measurements or data.is_reset)
)
ANNOTATIONS = sorted(
    name for name, data in GATES.items()
    if not data.is_unitary
    and not (data.is_noisy_gate or data.produces_measurements or data.is_reset)
    and name != "REPEAT"  # a block, not an instruction; tested separately
)
OTHER_UNITARIES = sorted(
    name for name, data in GATES.items()
    if data.is_unitary
    and not (data.is_single_qubit_gate or data.is_two_qubit_gate)
)


@pytest.mark.parametrize("name", UNITARY_1Q_2Q)
def test_gate_list_keeps_exactly_the_gates_that_act_nontrivially(name):
    data = GATES[name]
    width = 1 if data.is_single_qubit_gate else 2
    targets = list(range(width))
    is_identity = data.tableau == stim.Tableau(width)

    instruction = f"{name} " + " ".join(map(str, targets))

    gates = protocol.gate_list(stim.Circuit(instruction))

    assert gates == ([] if is_identity else [(name, targets)])


@pytest.mark.parametrize("name", OPERATIONS)
def test_gate_list_rejects_operations_that_are_not_unitary_gates(name):
    with pytest.raises(ValueError, match="not a unitary gate"):
        protocol.gate_list(one_instruction(name))


@pytest.mark.parametrize("name", ANNOTATIONS)
def test_gate_list_drops_annotations(name):
    assert protocol.gate_list(one_instruction(name)) == []


@pytest.mark.parametrize("name", OTHER_UNITARIES)
def test_gate_list_rejects_unitaries_on_more_than_two_qubits(name):
    with pytest.raises(ValueError, match="neither a single- nor a two-qubit"):
        protocol.gate_list(one_instruction(name))


def test_gate_list_splits_fused_instructions_by_arity():
    circuit = stim.Circuit("CX 0 1 2 3\nH 0 1 2")

    assert protocol.gate_list(circuit) == [
        ("CX", [0, 1]), ("CX", [2, 3]), ("H", [0]), ("H", [1]), ("H", [2]),
    ]


def test_gate_list_flattens_repeat_blocks():
    circuit = stim.Circuit("REPEAT 2 {\n    S 0\n}")

    assert protocol.gate_list(circuit) == [("S", [0]), ("S", [0])]


def test_gate_list_rejects_a_gate_controlled_by_a_sweep_bit():
    with pytest.raises(ValueError, match="non-qubit target"):
        protocol.gate_list(stim.Circuit("CX sweep[0] 1"))


# ---------------------------------------------------------------------------
# chunk_circuit
# ---------------------------------------------------------------------------
@settings(deadline=None)
@given(
    n=st.integers(1, 3),
    size=st.integers(0, 40),
    t=st.integers(1, 45),
    seed=st.integers(0, 2**32 - 1),
)
def test_chunks_partition_the_gates_into_balanced_contiguous_pieces(
    n, size, t, seed
):
    circuit = seeded_circuit(n, size, seed)

    chunks = protocol.chunk_circuit(circuit, n, t)

    sizes = [len(protocol.gate_list(chunk)) for chunk in chunks]
    assert len(chunks) == t
    assert [g for chunk in chunks for g in protocol.gate_list(chunk)] == (
        protocol.gate_list(circuit)
    )
    assert sum(sizes) == size
    assert sizes == sorted(sizes, reverse=True)
    assert max(sizes) - min(sizes) <= 1
    assert all(chunk.num_qubits == n for chunk in chunks)


@pytest.mark.parametrize(
    ("n", "t", "message"),
    [(0, 1, "n must be at least 1"), (2, 0, "t must be at least 1")],
)
def test_chunk_circuit_rejects_invalid_sizes(n, t, message):
    with pytest.raises(ValueError, match=message):
        protocol.chunk_circuit(stim.Circuit("H 0"), n, t)


def test_chunk_circuit_rejects_a_circuit_wider_than_n():
    with pytest.raises(ValueError, match="exceeding n = 2"):
        protocol.chunk_circuit(stim.Circuit("CX 0 2"), 2, 1)


# ---------------------------------------------------------------------------
# stabilizer_gens
# ---------------------------------------------------------------------------
@settings(deadline=None)
@given(
    n=st.integers(1, 3),
    size=st.integers(0, 30),
    seed=st.integers(0, 2**32 - 1),
)
def test_stabilizer_gens_generate_the_stabilizer_group_of_the_resource(
    n, size, seed
):
    circuit = seeded_circuit(n, size, seed)
    bare, image = list(range(n)), list(range(n, 2 * n))

    gens = protocol.stabilizer_gens(
        full_tableau(circuit, n), n, bare, image, 2 * n
    )

    # The resource state, prepared independently by simulation: Bell pairs
    # joining bare to image, then the circuit applied to image.
    resource = stim.Circuit()
    for i in range(n):
        resource.append("H", [bare[i]])
        resource.append("CX", [bare[i], image[i]])
    for name, targets in protocol.gate_list(circuit):
        resource.append(name, [image[q] for q in targets])
    simulator = stim.TableauSimulator()
    simulator.do(resource)

    assert len(gens) == 2 * n
    assert all(
        abs(simulator.peek_observable_expectation(g)) == 1 for g in gens
    )
    # Raises unless the 2n generators commute and are independent.
    stim.Tableau.from_stabilizers(gens)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_choi_observables_detect_every_error_on_the_output(n):
    circuit = seeded_circuit(n, 25, seed=n)
    ref, data = list(range(n)), list(range(n, 2 * n))
    observables = protocol.stabilizer_gens(
        full_tableau(circuit, n), n, ref, data, 2 * n
    )

    for letters in itertools.product("IXYZ", repeat=n):
        if set(letters) == {"I"}:
            continue
        error = stim.PauliString("I" * n + "".join(letters))
        assert any(not error.commutes(obs) for obs in observables), letters


# ---------------------------------------------------------------------------
# build_clinr_tr and build_direct
# ---------------------------------------------------------------------------
def assert_exact(built: stim.Circuit) -> None:
    """At zero noise every detector and observable is deterministic.

    Stim refuses to build a detector error model for a circuit with a
    non-deterministic detector or observable, and reports both against the
    noiseless reference, so a model with no errors is the exactness claim.
    """
    assert built.detector_error_model().num_errors == 0


@pytest.mark.parametrize("t", [1, 2, 3])
@pytest.mark.parametrize("r", [0, 1, 3])
@pytest.mark.parametrize("restricted_checks", [False, True])
@pytest.mark.parametrize("noisy_checks", [True, False])
@pytest.mark.parametrize("native_plus", [False, True])
def test_clinr_is_exact_at_zero_noise(
    t, r, restricted_checks, noisy_checks, native_plus
):
    n = 3
    circuit = seeded_circuit(n, 20, seed=7)
    drawn = checks.draw_checks(
        circuit, n, t, r, np.random.default_rng(11),
        restricted_checks=restricted_checks,
    )

    built = protocol.build_clinr_tr(
        circuit, n, drawn, NOISELESS,
        noisy_checks=noisy_checks, native_plus=native_plus,
    )

    assert built.num_detectors == t * r
    assert built.num_observables == 2 * n
    assert_exact(built)


@settings(deadline=None, max_examples=50)
@given(
    n=st.integers(1, 3),
    size=st.integers(0, 30),
    t=st.integers(1, 3),
    r=st.integers(0, 3),
    restricted_checks=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_clinr_is_exact_at_zero_noise_for_any_circuit(
    n, size, t, r, restricted_checks, seed
):
    circuit = seeded_circuit(n, size, seed)
    drawn = checks.draw_checks(
        circuit, n, t, r, np.random.default_rng(seed),
        restricted_checks=restricted_checks,
    )

    assert_exact(protocol.build_clinr_tr(circuit, n, drawn, NOISELESS))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_direct_implementation_is_exact_at_zero_noise(n):
    built = protocol.build_direct(seeded_circuit(n, 25, seed=n), n, NOISELESS)

    assert built.num_detectors == 0
    assert built.num_observables == 2 * n
    assert_exact(built)


def valid_checks(n: int, t: int, r: int) -> list[list[stim.PauliString]]:
    """Return uniform checks for the circuit the guard tests build."""
    return checks.draw_checks(
        seeded_circuit(n, 10, seed=3), n, t, r, np.random.default_rng(5)
    )


def test_builder_rejects_chunks_with_different_numbers_of_checks():
    circuit = seeded_circuit(2, 10, seed=3)
    ragged = valid_checks(2, 2, 2)
    ragged[1] = ragged[1][:1]

    with pytest.raises(ValueError, match="same number of checks"):
        protocol.build_clinr_tr(circuit, 2, ragged, NOISELESS)


def test_builder_rejects_a_check_of_the_wrong_length():
    circuit = seeded_circuit(2, 10, seed=3)
    drawn = [[stim.PauliString("XXXXX")]]

    with pytest.raises(ValueError, match="expected 4"):
        protocol.build_clinr_tr(circuit, 2, drawn, NOISELESS)


def test_builder_rejects_the_identity_as_a_check():
    circuit = seeded_circuit(2, 10, seed=3)
    drawn = [[stim.PauliString(4)]]

    with pytest.raises(ValueError, match="is the identity"):
        protocol.build_clinr_tr(circuit, 2, drawn, NOISELESS)


def test_builder_rejects_a_check_that_does_not_stabilize_the_resource():
    circuit = seeded_circuit(2, 10, seed=3)
    # X on one bare qubit anticommutes with that qubit's Z-type generator.
    drawn = [[stim.PauliString("X___")]]

    with pytest.raises(ValueError, match="does not stabilize"):
        protocol.build_clinr_tr(circuit, 2, drawn, NOISELESS)


def test_direct_builder_rejects_n_below_one():
    with pytest.raises(ValueError, match="n must be at least 1"):
        protocol.build_direct(stim.Circuit(), 0, NOISELESS)


def test_direct_builder_rejects_a_circuit_wider_than_n():
    with pytest.raises(ValueError, match="exceeding n = 1"):
        protocol.build_direct(stim.Circuit("CX 0 1"), 1, NOISELESS)
