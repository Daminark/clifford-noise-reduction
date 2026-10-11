# The wrong types are the point of these tests.
# mypy: disable-error-code="arg-type, list-item"
"""Tests that the public functions reject arguments of the wrong type.

Each case passes one argument of the wrong type, the rest valid, and checks
that the error is a ``TypeError`` naming that argument rather than an
unrelated failure further in.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import stim

from clinr import (
    accounting,
    bounds,
    checks,
    circuits,
    estimators,
    noise_models,
    protocol,
)

NOISE = noise_models.uniform_noise(1e-3)
CIRCUIT = stim.Circuit("H 0\nCX 0 1\nS 1")
CZ = stim.Circuit("CZ 0 1")


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def clinr_checks() -> list[list[stim.PauliString]]:
    return checks.draw_checks(CIRCUIT, 2, 1, 1, rng())


def estimate(**fields: Any) -> estimators.Estimate:
    values: dict[str, Any] = {
        "shots": 10,
        "accepted": 10,
        "errors": 0,
        "s1": 2,
        "s2": 1,
        "n": 2,
    }
    values.update(fields)
    return estimators.Estimate(**values)


CASES: list[tuple[str, Callable[[], object], str]] = [
    ("gate_list", lambda: protocol.gate_list("H 0"), "circuit"),
    ("chunk_circuit n", lambda: protocol.chunk_circuit(CIRCUIT, 2.0, 1), "n"),
    ("chunk_circuit t", lambda: protocol.chunk_circuit(CIRCUIT, 2, True), "t"),
    (
        "stabilizer_gens total",
        lambda: protocol.stabilizer_gens(
            stim.Tableau(1),
            [0],
            [1],
            2.5,
        ),
        "total",
    ),
    (
        "build_direct n",
        lambda: protocol.build_direct(CIRCUIT, 2.0, NOISE),
        "n",
    ),
    (
        "build_direct noise",
        lambda: protocol.build_direct(CIRCUIT, 2, 1e-3),
        "noise",
    ),
    (
        "build_clinr_tr flat checks",
        lambda: protocol.build_clinr_tr(CIRCUIT, 2, clinr_checks()[0], NOISE),
        "checks",
    ),
    (
        "build_clinr_tr string check",
        lambda: protocol.build_clinr_tr(CIRCUIT, 2, [["XXXX"]], NOISE),
        "check",
    ),
    (
        "build_cznr_tr noise",
        lambda: protocol.build_cznr_tr(CZ, 2, [[]], None),
        "noise",
    ),
    (
        "draw_checks r",
        lambda: checks.draw_checks(CIRCUIT, 2, 1, 1.5, rng()),
        "r",
    ),
    (
        "draw_cznr_checks r",
        lambda: checks.draw_cznr_checks(CZ, 2, 1, "1", rng()),
        "r",
    ),
    (
        "random_gate_sequence n",
        lambda: circuits.random_gate_sequence(True, 3, rng()),
        "n",
    ),
    (
        "random_gate_sequence size",
        lambda: circuits.random_gate_sequence(2, 3.0, rng()),
        "size",
    ),
    ("random_clifford n", lambda: circuits.random_clifford(2.0), "n"),
    (
        "random_cliffords count",
        lambda: circuits.random_cliffords(2, 1.0),
        "count",
    ),
    (
        "save_circuits element",
        lambda: circuits.save_circuits(["H 0"], "unused.json"),
        "circuits",
    ),
    ("prep_region n", lambda: accounting.prep_region(1.5, False), "n"),
    (
        "check_region count",
        lambda: accounting.check_region(1.5, 2, False),
        "count",
    ),
    (
        "check_region weight",
        lambda: accounting.check_region(1, 2.0, False),
        "weight",
    ),
    (
        "protected_region s1",
        lambda: accounting.protected_region(
            accounting.ZERO,
            1.0,
            0,
            accounting.ZERO,
        ),
        "s1",
    ),
    (
        "plog_bound r",
        lambda: bounds.plog_bound(CIRCUIT, 2, 1, 1.5, NOISE),
        "r",
    ),
    (
        "overhead_bound r",
        lambda: bounds.overhead_bound(CIRCUIT, 2, 1, 1.5, NOISE),
        "r",
    ),
    (
        "overhead_bound t",
        lambda: bounds.overhead_bound(CIRCUIT, 2, 1.0, 1, NOISE),
        "t",
    ),
    (
        "cznr_plog_bound noise",
        lambda: bounds.cznr_plog_bound(CZ, 2, 1, 1, "noise"),
        "noise",
    ),
    (
        "direct_plog_bound noise",
        lambda: bounds.direct_plog_bound(CIRCUIT, 1e-3),
        "noise",
    ),
    (
        "choose_tr t_max",
        lambda: bounds.choose_tr(
            CIRCUIT,
            2,
            NOISE,
            max_overhead=4,
            r_max=2,
            t_max=2.0,
        ),
        "t_max",
    ),
    (
        "choose_tr max_overhead",
        lambda: bounds.choose_tr(
            CIRCUIT,
            2,
            NOISE,
            max_overhead="4",
            r_max=2,
        ),
        "max_overhead",
    ),
    (
        "wilson_interval errors",
        lambda: estimators.wilson_interval(2.5, 10),
        "errors",
    ),
    ("Estimate shots", lambda: estimate(shots=10.0), "shots"),
    ("Estimate n", lambda: estimate(n=True), "n"),
    ("Estimate option", lambda: estimate(native_plus=1), "native_plus"),
    (
        "estimate_direct shots",
        lambda: estimators.estimate_direct(CIRCUIT, 2, NOISE, 10.0, rng()),
        "shots",
    ),
    (
        "estimate_clinr noise",
        lambda: estimators.estimate_clinr(CIRCUIT, 2, 1, 1, 1e-3, 10, rng()),
        "noise",
    ),
    ("NoiseModel rate", lambda: noise_models.NoiseModel("0.1", 0, 0, 0), "p1"),
    ("uniform_noise p", lambda: noise_models.uniform_noise(None), "p"),
    (
        "asymmetric_noise ratio",
        lambda: noise_models.asymmetric_noise(1e-3, "10"),
        "ratio",
    ),
]


@pytest.mark.parametrize(
    ("call", "name"),
    [(call, name) for _, call, name in CASES],
    ids=[case for case, _, _ in CASES],
)
def test_public_functions_reject_an_argument_of_the_wrong_type(call, name):
    with pytest.raises(TypeError, match=rf"\b{name}\b"):
        call()


def test_numpy_integers_are_accepted_where_integers_are():
    built = protocol.chunk_circuit(CIRCUIT, np.int64(2), np.int64(1))

    assert len(built) == 1


@pytest.mark.parametrize(
    "builder", [protocol.build_clinr_tr, protocol.build_cznr_tr]
)
def test_builders_reject_checks_for_no_chunks(builder):
    with pytest.raises(ValueError, match="checks must hold one list"):
        builder(CZ, 2, [], NOISE)


def test_a_chunk_without_checks_must_accept_every_shot():
    with pytest.raises(ValueError, match="without checks accepts every"):
        estimate(
            accepted=5,
            chunk_accepted=(5,),
            chunk_checks=(0,),
            chunk_weight=(0,),
            chunk_counts=((2, 1),),
        )


def test_statistics_over_draws_reject_draws_whose_chunks_differ():
    first = estimate(
        chunk_accepted=(10, 10),
        chunk_checks=(1, 1),
        chunk_weight=(2, 2),
        chunk_counts=((2, 0), (0, 1)),
    )
    second = estimate(
        chunk_accepted=(10, 10),
        chunk_checks=(1, 1),
        chunk_weight=(2, 2),
        chunk_counts=((0, 1), (2, 0)),
    )

    with pytest.raises(ValueError, match="gate counts per chunk disagree"):
        estimators.logical_error_rate([first, second])


def test_graph_state_gens_rejects_a_tableau_not_of_cz_gates():
    tableau = stim.Circuit("H 0\nCX 0 1").to_tableau()

    with pytest.raises(ValueError, match="not that of a circuit of CZ gates"):
        protocol.graph_state_gens(tableau)


def test_annotations_do_not_widen_the_circuit():
    """A qubit only an annotation mentions is not a qubit the circuit uses."""
    circuit = stim.Circuit("QUBIT_COORDS(0, 0) 5\nH 0\nCX 0 1")

    built = protocol.build_direct(circuit, 2, NOISE)

    assert built.num_observables == 4


def test_expected_overhead_validates_every_chunk_before_returning():
    """A zero acceptance would make the result infinite, but not first."""
    scheme = accounting.clinr_scheme(1, native_plus=False)

    with pytest.raises(ValueError, match="non-negative"):
        accounting.expected_overhead(
            [(1, 0), (2, -1)],
            scheme,
            [accounting.ZERO, accounting.ZERO],
            [0.0, 0.5],
        )


def test_expected_overhead_rejects_a_region_with_a_negative_count():
    scheme = accounting.clinr_scheme(1, native_plus=False)

    with pytest.raises(ValueError, match="operation counts"):
        accounting.expected_overhead(
            [(1, 0)], scheme, [accounting.Region(0, -1, 0, 0)], [1.0]
        )
