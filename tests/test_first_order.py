"""Tests for the first-order analysis in ``clinr.first_order``."""

import itertools
import math
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pytest
import stim
from hypothesis import given
from hypothesis import strategies as st

from clinr import checks, circuits, first_order, noise_models, protocol

# Four distinct rates, so each fault's probability identifies its channel.
NOISE = noise_models.NoiseModel(p1=0.001, p2=0.002, p_m=0.003, p_prep=0.004)
NOISE_CHANNELS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}


def seeded_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return a seeded random ``{H, S, CX}`` circuit on ``n`` qubits."""
    return circuits.random_gate_sequence(n, size, np.random.default_rng(seed))


def noiseless(circuit: stim.Circuit, start: int, stop: int) -> stim.Circuit:
    """Return instructions ``[start, stop)`` of ``circuit`` without noise."""
    out = stim.Circuit()
    for index in range(start, stop):
        instruction = circuit[index]
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name in NOISE_CHANNELS:
            continue
        if instruction.name == "M":
            out.append("M", instruction.targets_copy())
        else:
            out.append(instruction)
    return out


def components(
    instruction: stim.CircuitInstruction,
) -> Iterator[tuple[float, stim.Circuit]]:
    """Yield each fault of a noise instruction as ``(probability, circuit)``.

    The circuit applies the fault with certainty, in place of the
    instruction, as noise: Stim reports detector and observable flips
    relative to a noiseless reference sample, and a fault applied as a gate
    would become part of that reference.
    """
    targets = [target.value for target in instruction.targets_copy()]
    rate = instruction.gate_args_copy()[0]
    name = instruction.name
    if name == "DEPOLARIZE1":
        for qubit in targets:
            for pauli in "XYZ":
                yield rate / 3, stim.Circuit(f"{pauli}_ERROR(1) {qubit}")
    elif name == "DEPOLARIZE2":
        for a, b in itertools.batched(targets, 2, strict=True):
            for pa, pb in itertools.product("IXYZ", repeat=2):
                if pa == pb == "I":
                    continue
                fault = stim.Circuit()
                for pauli, qubit in ((pa, a), (pb, b)):
                    if pauli != "I":
                        fault.append(f"{pauli}_ERROR", [qubit], 1.0)
                yield rate / 15, fault
    elif name in ("X_ERROR", "Z_ERROR"):
        for qubit in targets:
            yield rate, stim.Circuit(f"{name}(1) {qubit}")
    else:  # a noisy measurement: flip one outcome, measure the rest bare
        assert name == "M"
        for flipped in targets:
            injected = stim.Circuit()
            for qubit in targets:
                injected.append("M", [qubit], 1.0 if qubit == flipped else 0.0)
            yield rate, injected


def reference(circuit: stim.Circuit) -> dict[int, tuple[float, float]]:
    """Inject each fault alone and record its effect, for each instruction.

    Returns, for each noise instruction, the total probability of its faults
    that flip an observable and no detector, and of those that flip a
    detector. This is the naive version of :func:`clinr.first_order.faults`.
    """
    effects: dict[int, tuple[float, float]] = {}
    for index in range(len(circuit)):
        instruction = circuit[index]
        assert isinstance(instruction, stim.CircuitInstruction)
        noisy = instruction.name == "M" and bool(instruction.gate_args_copy())
        if instruction.name not in NOISE_CHANNELS and not noisy:
            continue
        # A measurement's own record is the one flipped, so it is replaced;
        # a channel follows its gate, so the fault is inserted after it.
        before = noiseless(circuit, 0, index)
        after = noiseless(circuit, index + 1, len(circuit))
        undetected = detected = 0.0
        for probability, fault in components(instruction):
            sampler = (before + fault + after).compile_detector_sampler()
            flips, observables = sampler.sample(1, separate_observables=True)
            if flips.any():
                detected += probability
            elif observables.any():
                undetected += probability
        effects[index] = (undetected, detected)
    return effects


def by_instruction(
    found: list[first_order.Fault],
) -> dict[int, list[float]]:
    """Sum the faults of each instruction as :func:`reference` does."""
    effects: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for fault in found:
        if fault.detected:
            effects[fault.instruction][1] += fault.probability
        elif fault.logical:
            effects[fault.instruction][0] += fault.probability
    return effects


def test_every_fault_of_one_gate_is_a_logical_error():
    # The Choi state of one qubit detects every nontrivial Pauli.
    found = first_order.faults(
        protocol.build_direct(stim.Circuit("H 0"), 1, NOISE)
    )

    assert len(found) == 3
    for fault in found:
        assert fault.tag == "gates/0"
        assert fault.probability == pytest.approx(NOISE.p1 / 3)
        assert fault.logical
        assert not fault.detected
    assert first_order.logical_error_rate(found) == pytest.approx(NOISE.p1)


@given(
    n=st.integers(1, 2),
    size=st.integers(0, 6),
    t=st.integers(1, 2),
    r=st.integers(0, 2),
    restricted_checks=st.booleans(),
    native_plus=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_faults_agree_with_injecting_each_fault_alone(
    n, size, t, r, restricted_checks, native_plus, seed
):
    circuit = seeded_circuit(n, size, seed)
    drawn = checks.draw_checks(
        circuit,
        n,
        t,
        r,
        np.random.default_rng(seed),
        restricted_checks=restricted_checks,
    )
    built = protocol.build_clinr_tr(
        circuit, n, drawn, NOISE, native_plus=native_plus
    )

    found = by_instruction(first_order.faults(built))
    expected = reference(built)

    for index, (undetected, detected) in expected.items():
        assert found[index][0] == pytest.approx(undetected, abs=1e-15)
        assert found[index][1] == pytest.approx(detected, abs=1e-15)
    assert set(found) <= set(expected)


def test_faults_are_tagged_by_region_and_sorted_in_time():
    circuit = seeded_circuit(2, 12, seed=3)
    drawn = checks.draw_checks(circuit, 2, 2, 2, np.random.default_rng(3))
    built = protocol.build_clinr_tr(circuit, 2, drawn, NOISE)

    found = first_order.faults(built)

    assert [fault.instruction for fault in found] == sorted(
        fault.instruction for fault in found
    )
    regions = {fault.tag.split("/")[0] for fault in found}
    assert regions == {"prep", "gates", "check", "teleport"}
    for fault in found:
        assert built[fault.instruction].tag == fault.tag


def test_noiseless_checks_contribute_no_faults():
    circuit = seeded_circuit(2, 12, seed=4)
    drawn = checks.draw_checks(circuit, 2, 1, 3, np.random.default_rng(4))
    built = protocol.build_clinr_tr(
        circuit, 2, drawn, NOISE, noisy_checks=False
    )

    found = first_order.faults(built)

    assert found
    assert not any(fault.tag.startswith("check/") for fault in found)


def test_logical_error_rate_matches_sampling_at_low_noise():
    # Second-order terms are below the sampling error at this rate.
    noise = noise_models.asymmetric_noise(1e-4)
    rng = np.random.default_rng(5)
    circuit = seeded_circuit(3, 30, seed=5)
    drawn = checks.draw_checks(circuit, 3, 1, 2, rng)
    built = protocol.build_clinr_tr(circuit, 3, drawn, noise)
    shots = 400_000

    flips, observables = built.compile_detector_sampler(seed=5).sample(
        shots, separate_observables=True
    )
    accepted = ~flips.any(axis=1)
    rate = observables[accepted].any(axis=1).mean()
    expected = first_order.logical_error_rate(first_order.faults(built))

    assert abs(rate - expected) < 5 * math.sqrt(expected / shots)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("REPEAT 2 {\n    X_ERROR(0.1) 0\n}", "REPEAT"),
        ("PAULI_CHANNEL_1(0.1, 0, 0) 0", "unsupported noise channel"),
    ],
)
def test_faults_reject_circuits_outside_the_supported_set(text, message):
    with pytest.raises(ValueError, match=message):
        first_order.faults(stim.Circuit(text))
