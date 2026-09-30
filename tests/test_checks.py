"""Tests for check selection in ``clinr.checks``."""

import itertools
import math

import numpy as np
import pytest
import stim
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from clinr import checks, circuits, protocol


def seeded_circuit(n: int, size: int, seed: int) -> stim.Circuit:
    """Return a seeded random ``{H, S, CX}`` circuit on ``n`` qubits."""
    return circuits.random_gate_sequence(n, size, np.random.default_rng(seed))


def resource(circuit: stim.Circuit, n: int):
    """Return the tableau, generators and restricted family of the resource.

    Laid out as ``draw_checks`` lays out its checks: the bare half on
    ``[0, n)`` and the half carrying the circuit on ``[n, 2n)``.
    """
    tableau = protocol.chunk_circuit(circuit, n, 1)[0].to_tableau()
    bare, image = list(range(n)), list(range(n, 2 * n))
    gens = protocol.stabilizer_gens(tableau, bare, image, 2 * n)
    family = checks.restricted_family(tableau, bare, image, 2 * n)
    return tableau, gens, family


def unsigned(pauli: stim.PauliString) -> str:
    """Return ``pauli`` as letters, without its sign."""
    return str(pauli)[1:]


def group_elements(gens: list[stim.PauliString]) -> list[stim.PauliString]:
    """Return every non-identity product of ``gens``."""
    elements = []
    for mask in range(1, 2 ** len(gens)):
        product = stim.PauliString(len(gens[0]))
        for bit, gen in enumerate(gens):
            if mask >> bit & 1:
                product *= gen
        elements.append(product)
    return elements


def bare_z(n: int) -> stim.PauliString:
    """Return ``Z`` on the first bare qubit, a fault visible to one triple.

    It anticommutes with the propagated ``XX`` and ``YY`` of the first Bell
    pair and commutes with everything else in the restricted family, so it
    attains that family's worst case, whatever the circuit.
    """
    return stim.PauliString("Z" + "_" * (2 * n - 1))


# ---------------------------------------------------------------------------
# restricted_family
# ---------------------------------------------------------------------------
@given(
    n=st.integers(1, 4),
    size=st.integers(0, 30),
    seed=st.integers(0, 2**32 - 1),
)
def test_restricted_family_is_the_propagated_bell_triples(n, size, seed):
    _, gens, family = resource(seeded_circuit(n, size, seed), n)

    assert len(family) == 3 * n
    for element in family:
        assert sum(element[q] != 0 for q in range(n)) == 1
        assert sum(element[q] != 0 for q in range(n, 2 * n)) <= n
        assert all(element.commutes(g) for g in gens)
    for gx, gz, third in itertools.batched(family, 3, strict=True):
        assert unsigned(third) == unsigned(gx * gz)


# ---------------------------------------------------------------------------
# The worst-case miss the bounds charge, by enumeration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3])
def test_worst_case_per_draw_miss_is_the_constant_the_bounds_use(n):
    _, gens, family = resource(seeded_circuit(n, 25, seed=n), n)
    group = group_elements(gens)
    uniform_miss = (2 ** (2 * n - 1) - 1) / (2 ** (2 * n) - 1)
    restricted_miss = 1 - 2 / (3 * n)

    worst_restricted = 0.0
    for letters in itertools.product("IXYZ", repeat=2 * n):
        fault = stim.PauliString("".join(letters))
        if all(fault.commutes(g) for g in gens):
            continue  # a stabilizer, up to sign: it does no damage
        uniform = sum(fault.commutes(s) for s in group) / len(group)
        restricted = sum(fault.commutes(s) for s in family) / len(family)
        # Half the group anticommutes, and the identity is never drawn.
        assert uniform == pytest.approx(uniform_miss)
        worst_restricted = max(worst_restricted, restricted)

    assert worst_restricted == pytest.approx(restricted_miss)


# ---------------------------------------------------------------------------
# draw_checks
# ---------------------------------------------------------------------------
@given(
    n=st.integers(1, 3),
    size=st.integers(0, 30),
    t=st.integers(1, 4),
    r=st.integers(0, 4),
    restricted_checks=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_draw_checks_draws_stabilizers_of_each_chunks_resource(
    n, size, t, r, restricted_checks, seed
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

    assert len(drawn) == t
    for chunk, chunk_checks in zip(
        protocol.chunk_circuit(circuit, n, t), drawn, strict=True
    ):
        _, gens, family = resource(chunk, n)
        allowed = {unsigned(p) for p in family}
        assert len(chunk_checks) == r
        for check in chunk_checks:
            assert len(check) == 2 * n
            assert check.weight > 0
            assert all(check.commutes(g) for g in gens)
            if restricted_checks:
                assert unsigned(check) in allowed


@pytest.mark.parametrize("n", [1, 2])
def test_uniform_checks_are_uniform_over_the_non_identity_elements(n):
    """Chi-square test at p > 1e-6, about a five-sigma departure.

    With 1000 expected draws per element, an element never drawn, or drawn
    at twice its share, gives a p-value far below the threshold.
    """
    circuit = seeded_circuit(n, 20, seed=n)
    _, gens, _ = resource(circuit, n)
    draws = 1000 * (4**n - 1)

    (drawn,) = checks.draw_checks(
        circuit, n, 1, draws, np.random.default_rng(0)
    )

    tally = {unsigned(g): 0 for g in group_elements(gens)}
    for check in drawn:
        tally[unsigned(check)] += 1
    assert stats.chisquare(list(tally.values())).pvalue > 1e-6


def test_draw_checks_repeats_its_draws_for_the_same_seed():
    circuit = seeded_circuit(3, 20, seed=0)

    first, second = (
        checks.draw_checks(circuit, 3, 2, 4, np.random.default_rng(42))
        for _ in range(2)
    )

    assert first == second


@pytest.mark.parametrize("restricted_checks", [False, True])
def test_drawn_checks_miss_a_worst_case_fault_at_the_closed_form_rate(
    restricted_checks,
):
    """Five-sigma binomial bounds on single draws and on triples.

    At 30000 draws, 5 sigma on the single-draw rate is about 0.014, well
    inside the gap of 0.033 between 7/15 and the 1/2 a sampler that could
    return the identity would give.
    The triple rate q**3, at about 0.015 for 5 sigma, checks independence:
    a sampler that reused draws would miss triples at a rate nearer q**2.
    """
    n, draws = 2, 30000
    circuit = seeded_circuit(n, 20, seed=5)
    q = 1 - 2 / (3 * n) if restricted_checks else 7 / 15
    (drawn,) = checks.draw_checks(
        circuit,
        n,
        1,
        draws,
        np.random.default_rng(7),
        restricted_checks=restricted_checks,
    )

    missed = [check.commutes(bare_z(n)) for check in drawn]
    single = sum(missed) / draws
    # Draws are independent, so three in a row all miss at rate q**3.
    triples = [all(missed[i : i + 3]) for i in range(0, draws, 3)]
    triple = sum(triples) / len(triples)

    assert abs(single - q) < 5 * math.sqrt(q * (1 - q) / draws)
    assert abs(triple - q**3) < 5 * math.sqrt(q**3 * (1 - q**3) / len(triples))


@pytest.mark.parametrize("restricted_checks", [False, True])
def test_draw_checks_rejects_a_legacy_random_state(restricted_checks):
    with pytest.raises(TypeError, match=r"numpy\.random\.Generator"):
        checks.draw_checks(
            seeded_circuit(2, 5, seed=0),
            2,
            1,
            2,
            np.random.RandomState(0),  # type: ignore[arg-type]
            restricted_checks=restricted_checks,
        )


def test_draw_checks_rejects_a_negative_check_count():
    with pytest.raises(ValueError, match="r must be non-negative"):
        checks.draw_checks(
            seeded_circuit(2, 5, seed=0), 2, 1, -1, np.random.default_rng(0)
        )
