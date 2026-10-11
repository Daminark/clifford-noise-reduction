"""Tests for the random circuit generators in ``clinr.circuits``."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from clinr import circuits, protocol


@given(
    n=st.integers(1, 4),
    size=st.integers(0, 50),
    seed=st.integers(0, 2**32 - 1),
)
def test_random_gate_sequence_draws_size_gates_from_h_s_and_cx(n, size, seed):
    rng = np.random.default_rng(seed)

    circuit = circuits.random_gate_sequence(n, size, rng)

    gates = protocol.gate_list(circuit)
    assert len(gates) == size
    alphabet = {"H", "S"} if n == 1 else {"H", "S", "CX"}
    assert {name for name, _ in gates} <= alphabet
    for _, targets in gates:
        assert all(0 <= qubit < n for qubit in targets)
        assert len(set(targets)) == len(targets)


def test_random_gate_sequence_repeats_for_the_same_seed():
    first, second = (
        circuits.random_gate_sequence(4, 30, np.random.default_rng(9))
        for _ in range(2)
    )

    assert first == second


@pytest.mark.parametrize(
    ("n", "size", "message"),
    [(0, 5, "n must be at least 1"), (2, -1, "size must be non-negative")],
)
def test_random_gate_sequence_rejects_invalid_sizes(n, size, message):
    with pytest.raises(ValueError, match=message):
        circuits.random_gate_sequence(n, size, np.random.default_rng(0))


def test_random_gate_sequence_rejects_a_legacy_random_state():
    with pytest.raises(TypeError, match=r"numpy\.random\.Generator"):
        circuits.random_gate_sequence(
            2,
            5,
            np.random.RandomState(0),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("n", [1, 2, 5])
def test_random_clifford_synthesizes_over_h_s_and_cx(n):
    circuit = circuits.random_clifford(n)

    assert circuit.num_qubits <= n
    names = {name for name, _ in protocol.gate_list(circuit)}
    assert names <= {"H", "S", "CX"}


def test_random_clifford_rejects_n_below_one():
    with pytest.raises(ValueError, match="n must be at least 1"):
        circuits.random_clifford(0)


def test_random_cliffords_returns_count_circuits():
    assert len(circuits.random_cliffords(2, 3)) == 3


def test_random_cliffords_rejects_a_count_below_one():
    with pytest.raises(ValueError, match="count must be positive"):
        circuits.random_cliffords(2, 0)


def test_saved_circuits_load_back_unchanged(tmp_path):
    saved = [
        circuits.random_gate_sequence(3, 20, np.random.default_rng(seed))
        for seed in range(3)
    ]
    path = tmp_path / "circuits.json"

    circuits.save_circuits(saved, path)

    assert circuits.load_circuits(path) == saved


def test_save_circuits_rejects_an_empty_list(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        circuits.save_circuits([], tmp_path / "empty.json")


def test_random_cliffords_rejects_n_below_one():
    with pytest.raises(ValueError, match="n must be at least 1"):
        circuits.random_cliffords(0, 3)
