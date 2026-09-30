"""Tests for the operation accounting in ``clinr.accounting``."""

import math

import numpy as np
import pytest
import stim
from hypothesis import given
from hypothesis import strategies as st

from clinr import accounting, checks, circuits, noise_models, protocol

# Four distinct rates, so every noise channel in a built circuit can be
# attributed to the kind of operation that emitted it.
NOISE = noise_models.NoiseModel(p1=0.001, p2=0.002, p_m=0.003, p_prep=0.004)
NOISE_CHANNELS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}
ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "TICK"}


def seeded_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return a seeded random ``{H, S, CX}`` circuit on ``n`` qubits."""
    return circuits.random_gate_sequence(n, size, np.random.default_rng(seed))


def fault_locations(built: stim.Circuit) -> accounting.Region:
    """Count the noisy operations in ``built``, by kind, from its channels."""
    n1 = n2 = nm = n_prep = 0
    for inst in built.flattened():  # type: ignore[attr-defined]  # stubs omit iteration
        targets = len(inst.targets_copy())
        args = inst.gate_args_copy()
        if inst.name == "DEPOLARIZE1":
            assert args == [NOISE.p1]
            n1 += targets
        elif inst.name == "DEPOLARIZE2":
            assert args == [NOISE.p2]
            n2 += targets // 2
        elif inst.name in ("X_ERROR", "Z_ERROR"):
            assert args == [NOISE.p_prep]
            n_prep += targets
        elif inst.name == "M":
            assert args == [NOISE.p_m]
            nm += targets
    return accounting.Region(n1=n1, n2=n2, nm=nm, n_prep=n_prep)


def protocol_operations(built: stim.Circuit, n: int) -> int:
    """Count the operations in ``built`` that belong to the protocol.

    Excluded: noise channels and annotations, which are not operations;
    ``MPP``, which is an idealized check or the harness readout; and anything
    touching the Choi reference block ``[3n, 4n)``, which is test apparatus.
    """
    reference = set(range(3 * n, 4 * n))
    count = 0
    for inst in built.flattened():  # type: ignore[attr-defined]  # stubs omit iteration
        if inst.name in NOISE_CHANNELS | ANNOTATIONS or inst.name == "MPP":
            continue
        targets = [target.value for target in inst.targets_copy()]
        if reference & set(targets):
            continue
        width = 2 if stim.gate_data(inst.name).is_two_qubit_gate else 1
        count += len(targets) // width
    return count


def build(
    circuit: stim.Circuit,
    n: int,
    t: int,
    r: int,
    restricted_checks: bool,
    **flags: bool,
) -> tuple[list[list[stim.PauliString]], stim.Circuit]:
    """Draw checks and build ``CliNR_{t,r}``; return the checks and circuit."""
    drawn = checks.draw_checks(
        circuit, n, t, r, np.random.default_rng(1),
        restricted_checks=restricted_checks,
    )
    return drawn, protocol.build_clinr_tr(circuit, n, drawn, NOISE, **flags)


def check_regions(
    drawn: list[list[stim.PauliString]], noisy_checks: bool, native_plus: bool
) -> list[accounting.Region]:
    """Return each chunk's check region, charged at the drawn weights."""
    return [
        accounting.check_region(
            len(chunk), sum(p.weight for p in chunk), noisy_checks, native_plus
        )
        for chunk in drawn
    ]


@given(
    n=st.integers(1, 3),
    size=st.integers(0, 30),
    t=st.integers(1, 3),
    r=st.integers(0, 3),
    restricted_checks=st.booleans(),
    noisy_checks=st.booleans(),
    noisy_q=st.booleans(),
    native_plus=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_regions_count_every_fault_location_the_builder_emits(
    n, size, t, r, restricted_checks, noisy_checks, noisy_q, native_plus, seed
):
    circuit = seeded_circuit(n, size, seed)
    drawn, built = build(
        circuit, n, t, r, restricted_checks,
        noisy_checks=noisy_checks, noisy_q=noisy_q, native_plus=native_plus,
    )
    prep = accounting.prep_region(n, native_plus)
    counts = [
        accounting.gate_counts(chunk)
        for chunk in protocol.chunk_circuit(circuit, n, t)
    ]

    regions = []
    for (s1, s2), chunk_checks in zip(
        counts, check_regions(drawn, noisy_checks, native_plus), strict=True
    ):
        regions.append(accounting.protected_region(prep, s1, s2, chunk_checks))
        regions.append(accounting.teleport_region(n))
    if noisy_q:
        regions.append(accounting.frame_region(n))

    assert fault_locations(built) == accounting.add(*regions)


@given(
    n=st.integers(1, 3),
    size=st.integers(1, 30),
    t=st.integers(1, 3),
    r=st.integers(0, 3),
    restricted_checks=st.booleans(),
    noisy_checks=st.booleans(),
    native_plus=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_overhead_with_certain_acceptance_counts_the_built_operations(
    n, size, t, r, restricted_checks, noisy_checks, native_plus, seed
):
    """Acceptance 1 makes the expected count that of one pass, built exactly.

    With every chunk accepted on its first attempt, the expected number of
    operations is the number in one pass through the built circuit, so the
    builder is the oracle.
    """
    circuit = seeded_circuit(n, size, seed)
    drawn, built = build(
        circuit, n, t, r, restricted_checks,
        noisy_checks=noisy_checks, native_plus=native_plus,
    )
    counts = [
        accounting.gate_counts(chunk)
        for chunk in protocol.chunk_circuit(circuit, n, t)
    ]

    overhead = accounting.expected_overhead(
        counts,
        n,
        accounting.prep_region(n, native_plus),
        check_regions(drawn, noisy_checks, native_plus),
        [1.0] * t,
        noisy_q=False,
    )

    assert overhead * size == pytest.approx(protocol_operations(built, n))


@pytest.mark.parametrize(
    ("noisy_q", "expected"), [(False, 10.0), (True, 31 / 3)]
)
def test_expected_overhead_on_a_hand_worked_chunk(noisy_q, expected):
    """Worked by hand: (26 + 4) / 3 = 10, and 31 / 3 with frame resolution.

    One chunk at n = 1 of two single-qubit gates and one CX, with one
    weight-2 check, accepted half the time.
    Protected region: preparation 4, gates 3, check 6, so 13 per attempt and
    26 expected; teleportation adds 4 once, and frame resolution 1 more.
    """
    overhead = accounting.expected_overhead(
        [(2, 1)],
        1,
        accounting.Region(n1=1, n2=1, nm=0, n_prep=2),
        [accounting.Region(n1=2, n2=2, nm=1, n_prep=1)],
        [0.5],
        noisy_q,
    )

    assert overhead == pytest.approx(expected)


def test_expected_overhead_is_infinite_when_a_chunk_never_accepts():
    overhead = accounting.expected_overhead(
        [(2, 1), (1, 0)], 1, accounting.ZERO, [accounting.ZERO] * 2,
        [0.9, 0.0], noisy_q=False,
    )

    assert overhead == math.inf


def test_expected_overhead_rejects_chunks_without_gates():
    with pytest.raises(ValueError, match="at least one gate"):
        accounting.expected_overhead(
            [(0, 0)], 1, accounting.ZERO, [accounting.ZERO], [1.0], False
        )


def test_expected_overhead_rejects_per_chunk_sequences_of_unequal_length():
    with pytest.raises(ValueError, match="one entry per chunk"):
        accounting.expected_overhead(
            [(2, 1), (1, 0)], 1, accounting.ZERO, [accounting.ZERO],
            [1.0, 1.0], False,
        )


@pytest.mark.parametrize("q", [-0.1, 1.1, math.nan])
def test_expected_overhead_rejects_an_acceptance_outside_the_unit_interval(q):
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        accounting.expected_overhead(
            [(2, 1)], 1, accounting.ZERO, [accounting.ZERO], [q], False
        )


@pytest.mark.parametrize(
    ("region", "args"),
    [
        (accounting.prep_region, (0, False)),
        (accounting.teleport_region, (0,)),
        (accounting.frame_region, (0,)),
    ],
)
def test_regions_reject_n_below_one(region, args):
    with pytest.raises(ValueError, match="n must be at least 1"):
        region(*args)


@pytest.mark.parametrize(
    ("count", "weight", "message"),
    [(-1, 0, "count must be non-negative"), (3, 2, "at least count = 3")],
)
def test_check_region_rejects_impossible_counts(count, weight, message):
    with pytest.raises(ValueError, match=message):
        accounting.check_region(count, weight, True, False)


@pytest.mark.parametrize(("s1", "s2"), [(-1, 0), (0, -1)])
def test_protected_region_rejects_negative_gate_counts(s1, s2):
    with pytest.raises(ValueError, match="must be non-negative"):
        accounting.protected_region(accounting.ZERO, s1, s2, accounting.ZERO)


def test_the_sum_of_no_regions_is_zero():
    assert accounting.add() == accounting.ZERO


def test_gate_counts_split_by_arity_and_skip_identities_and_annotations():
    circuit = stim.Circuit("H 0 1 2\nCX 0 1 1 2\nI 0\nTICK\nS 2")

    assert accounting.gate_counts(circuit) == (4, 2)
