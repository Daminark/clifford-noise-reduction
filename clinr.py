"""CliNR_{1,r} circuit construction (arXiv:2407.06583, Lemma 1).

Builds gate-teleportation circuits that apply a Clifford ``C`` through a
resource state guarded by ``r`` random stabilizer checks. Circuits are exact
at ``p = 0``; statistical estimation lives in the ``estimators`` module.
Scope is ``t = 1`` only.

Register layout (``4n + 1`` qubits: four blocks of ``n`` plus one ancilla)::

    b1  [0,   n)     data input
    b2  [n,  2n)     resource half teleported against
    b3  [2n, 3n)     resource half carrying C; the output
    ref [3n, 4n)     Choi witness -- not physical (test apparatus only)
    anc 4n           ancilla for stabilizer measurement

Noise (Sec. II): ``p1`` after preps and single-qubit gates (DEPOLARIZE1),
``p2`` after controlled-Paulis (DEPOLARIZE2), ``p_m`` for measurement flips
(X error before ``M``); idle qubits noiseless.
"""

from __future__ import annotations

import numpy as np
import stim

__all__ = [
    "blocks",
    "stabilizer_gens",
    "random_stabilizer",
    "build_clinr_1r",
]

_PAULI_CHR = "IXYZ"
_TWO_QUBIT_GATES = frozenset({"CX", "CY", "CZ", "CNOT", "SWAP", "ISWAP"})


# ---------------------------------------------------------------------------
# Noise injectors: each appends only noise, and nothing when its rate is 0.
# ---------------------------------------------------------------------------
def _noise_after_gate(
    circ: stim.Circuit, gate: str, targets: list[int], p1: float, p2: float
) -> None:
    """Append depolarizing noise following a unitary ``gate`` on ``targets``.

    Stim fuses consecutive identical gate names into a single instruction, so
    ``targets`` may contain several gates' worth of qubits. We therefore chunk
    ``targets`` by the gate's arity and apply one depolarizing channel per gate:
    DEPOLARIZE1 at rate ``p1`` for single-qubit gates, DEPOLARIZE2 at rate
    ``p2`` for two-qubit gates.

    Raises:
        ValueError: if ``targets`` is not a whole number of gates wide.
    """
    width = 2 if gate in _TWO_QUBIT_GATES else 1
    if len(targets) % width:
        raise ValueError(
            f"{gate!r} acts on {width} qubit(s) but got {len(targets)} targets"
        )
    channel = "DEPOLARIZE2" if width == 2 else "DEPOLARIZE1"
    rate = p2 if width == 2 else p1
    for i in range(0, len(targets), width):
        circ.append(channel, targets[i : i + width], rate)


def _noise_after_prep(circ: stim.Circuit, targets: list[int], p1: float) -> None:
    """Append a single-qubit depolarizing error after preparing ``targets``."""
    if p1 > 0:
        circ.append("DEPOLARIZE1", targets, p1)


def _noise_before_measure(circ: stim.Circuit, targets: list[int], p_m: float) -> None:
    """Model measurement misreport as an X error just before a Z-basis ``M``."""
    if p_m > 0:
        circ.append("X_ERROR", targets, p_m)


def _apply_noisy(
    circ: stim.Circuit, gate: str, targets: list[int], p1: float, p2: float
) -> None:
    """Append ``gate`` on ``targets`` immediately followed by its gate noise."""
    circ.append(gate, targets)
    if p1 or p2:
        _noise_after_gate(circ, gate, targets, p1, p2)


# ---------------------------------------------------------------------------
# Block layout and Pauli helpers
# ---------------------------------------------------------------------------
def blocks(n: int) -> tuple[list[int], list[int], list[int], list[int], int]:
    """Return the index layout ``(b1, b2, b3, ref, anc)`` for ``n`` data qubits.

    ``b1, b2, b3, ref`` are length-``n`` index lists for the four blocks and
    ``anc`` is the single ancilla index. See the module docstring for the roles.
    """
    b1 = list(range(0, n))
    b2 = list(range(n, 2 * n))
    b3 = list(range(2 * n, 3 * n))
    ref = list(range(3 * n, 4 * n))
    anc = 4 * n
    return b1, b2, b3, ref, anc


def _pauli_to_mpp_targets(pauli: stim.PauliString) -> list:
    """Convert a Pauli string into an ``MPP`` target list (combiner-joined).

    Identity positions are skipped; each non-identity position contributes a
    single-qubit Pauli target, and adjacent targets are joined by combiners so
    that ``MPP`` measures the *product* observable. The overall sign of
    ``pauli`` is ignored: ``MPP`` measures the unsigned product and the sign is
    recovered downstream by Stim's detector/observable reference values.
    """
    ctor = {1: stim.target_x, 2: stim.target_y, 3: stim.target_z}
    targets: list = []
    for qubit in range(len(pauli)):
        code = pauli[qubit]
        if code == 0:
            continue
        if targets:
            targets.append(stim.target_combiner())
        targets.append(ctor[code](qubit))
    return targets


def _apply_pauli_controlled(
    circ: stim.Circuit, pauli: stim.PauliString, targets: list[int], rec: int
) -> None:
    """Apply ``pauli`` (raised to a measured bit) to ``targets``, classically.

    Each non-identity factor of ``pauli`` is applied as a Pauli controlled on
    the measurement record ``rec``. The control is a *classical* measurement
    outcome, so a global sign on ``pauli`` is an unobservable global phase and
    is safely ignored.

    Args:
        pauli: An ``n``-qubit Pauli whose ``j``-th factor acts on ``targets[j]``.
        targets: Physical qubit indices, one per position of ``pauli``.
        rec: Measurement-record offset (e.g. ``-1``) supplying the control bit.
    """
    controlled = {1: "CX", 2: "CY", 3: "CZ"}
    for j in range(len(pauli)):
        code = pauli[j]
        if code == 0:
            continue
        circ.append(controlled[code], [stim.target_rec(rec), targets[j]])


# ---------------------------------------------------------------------------
# Resource-state stabilizers
# ---------------------------------------------------------------------------
def stabilizer_gens(
    tableau: stim.Tableau, n: int, b2: list[int], b3: list[int], total: int
) -> list[stim.PauliString]:
    """Generators of the resource-state stabilizer group, in closed form.

    The resource ``(I ⊗ C)|Φ+⟩^{⊗n}`` on blocks 2 and 3 is stabilized by the
    Bell generators conjugated by ``C`` on block 3::

        X_{b2[i]} ⊗ (C X_i C†)_{b3}     and     Z_{b2[i]} ⊗ (C Z_i C†)_{b3}

    for ``i = 0, ..., n-1``, taken from the tableau. Signs are dropped: they do
    not affect measurement outcomes or the anticommutation that drives
    detection.

    Args:
        tableau: The tableau of the Clifford ``C``.
        n: Number of data qubits.
        b2, b3: Index lists for blocks 2 and 3.
        total: Full register width (length of each returned Pauli string).

    Returns:
        A list of ``2n`` independent generators as length-``total`` Paulis.
    """
    gens: list[stim.PauliString] = []
    for i in range(n):
        for base, conjugated in (("X", tableau.x_output(i)), ("Z", tableau.z_output(i))):
            letters = ["I"] * total
            letters[b2[i]] = base
            for j in range(n):
                letters[b3[j]] = _PAULI_CHR[conjugated[j]]
            gens.append(stim.PauliString("".join(letters)))
    return gens


def random_stabilizer(
    gens: list[stim.PauliString], total: int, rng: np.random.Generator
) -> stim.PauliString:
    """Sample a uniformly random *non-identity* element of ``⟨gens⟩``.

    Each generator is included independently with probability 1/2, which samples
    the group uniformly; the uniformity is what the detection guarantee relies
    on, so preserve it if you change the sampling. The identity element is
    rejected: it has no valid ``MPP`` targets and carries no information as a
    check.

    Args:
        gens: Independent generators (as returned by :func:`stabilizer_gens`).
        total: Register width, i.e. the length of the returned Pauli string.
        rng: A NumPy generator. Passing the generator (rather than a seed) lets
            a caller thread one stream through repeated draws so that the draws
            are independent; the explicit type is enforced to reject objects
            whose ``.random`` has a different contract.

    Raises:
        TypeError: if ``rng`` is not a ``numpy.random.Generator``.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    while True:
        take = rng.random(len(gens)) < 0.5
        product = stim.PauliString(total)
        for keep, gen in zip(take, gens):
            if keep:
                product *= gen
        if product.weight > 0:
            return product


# ---------------------------------------------------------------------------
# Stabilizer-measurement gadget
# ---------------------------------------------------------------------------
def _measure_stabilizer_noisy(
    circ: stim.Circuit,
    pauli: stim.PauliString,
    anc: int,
    p1: float,
    p2: float,
    p_m: float,
) -> None:
    """Measure ``pauli`` via a noisy ancilla-based gadget, appending one record.

    Realizes the measurement as reset -> H -> (controlled-Pauli per support
    qubit) -> H -> measure, with circuit-level noise on every operation. This is
    the faithful, noisy stand-in for Stim's idealized ``MPP``: a weight-``w``
    stabilizer costs ``~ w + 3`` noisy locations. Use ``MPP`` directly instead
    for an idealized, noiseless check.
    """
    controlled = {1: "CX", 2: "CY", 3: "CZ"}
    circ.append("R", [anc])
    _noise_after_prep(circ, [anc], p1)
    _apply_noisy(circ, "H", [anc], p1, p2)
    for j in range(len(pauli)):
        code = pauli[j]
        if code == 0:
            continue
        _apply_noisy(circ, controlled[code], [anc, j], p1, p2)
    _apply_noisy(circ, "H", [anc], p1, p2)
    _noise_before_measure(circ, [anc], p_m)
    circ.append("M", [anc])


# ---------------------------------------------------------------------------
# CliNR_{1,r} circuit builder
# ---------------------------------------------------------------------------
def build_clinr_1r(
    circuit: stim.Circuit,
    n: int,
    r: int,
    p1: float,
    p2: float,
    p_m: float,
    rng: np.random.Generator,
    *,
    noisy_checks: bool = True,
) -> stim.Circuit:
    """Build ``CliNR_{1,r}(C)`` with the Choi channel-test harness.

    The returned circuit teleports the data through a resource state carrying
    ``C``, guarded by ``r`` random stabilizer checks, and includes the reference
    register and observables needed to read out the logical error rate. It is a
    pure stabilizer circuit and is exact at ``p1 = p2 = p_m = 0``.

    Args:
        circuit: The Clifford circuit ``C`` to implement, over ``n`` qubits and
            built from Clifford gates Stim recognizes (e.g. H, S, CX).
        n: Number of data qubits.
        r: Number of random stabilizer checks (``r = 0`` gives plain gate
            teleportation with no error detection).
        p1: Single-qubit / preparation depolarizing rate.
        p2: Two-qubit depolarizing rate.
        p_m: Measurement flip probability.
        rng: NumPy generator used to draw the ``r`` random checks.
        noisy_checks: If True (default), measure each check with the explicit
            noisy ancilla gadget so measurement faults are modeled (faithful to
            Lemma 1). If False, use Stim's idealized ``MPP``.

    Returns:
        A ``stim.Circuit`` on ``4n + 1`` qubits with ``r`` detectors (the
        checks) and ``2n`` observables (the Choi stabilizers).

    Detectors and observables:
        Each check is wrapped in a ``DETECTOR``; a fired detector marks a run to
        be discarded (the "restart"). The ``2n`` Choi stabilizers are declared
        via ``OBSERVABLE_INCLUDE``; a logical error is any observable flip among
        accepted runs.
    """
    tableau = circuit.to_tableau()
    b1, b2, b3, ref, anc = blocks(n)
    total = 4 * n + 1
    out = stim.Circuit()

    # --- Choi input: Bell-pair each reference qubit with its data qubit. ---
    # This is the (noiseless, non-physical) measuring apparatus, not part of
    # the protocol; it lets us test the channel for all inputs at once.
    for i in range(n):
        out.append("H", [ref[i]])
        out.append("CX", [ref[i], b1[i]])

    # --- Protected region: prepare the resource state (I ⊗ C)|Φ+⟩^{⊗n}. ---
    # Bell pairs across blocks 2 and 3, then C applied to block 3. All of this
    # is noisy and is exactly what the stabilizer checks below protect.
    for i in range(n):
        out.append("R", [b2[i], b3[i]])
        _noise_after_prep(out, [b2[i], b3[i]], p1)
        _apply_noisy(out, "H", [b2[i]], p1, p2)
        _apply_noisy(out, "CX", [b2[i], b3[i]], p1, p2)
    for inst in circuit.flattened():
        data_targets = [b3[t.value] for t in inst.targets_copy()]
        _apply_noisy(out, inst.name, data_targets, p1, p2)

    # --- Offline fault detection: r random stabilizer checks. ---
    # Placed BEFORE the teleportation CNOTs so a detected fault only spoils the
    # resource (a "restart"), never the data -- and so the measurement-record
    # offsets used by the correction below stay fixed.
    gens = stabilizer_gens(tableau, n, b2, b3, total)
    for _ in range(r):
        check = random_stabilizer(gens, total, rng)
        if noisy_checks:
            _measure_stabilizer_noisy(out, check, anc, p1, p2, p_m)
        else:
            out.append("MPP", _pauli_to_mpp_targets(check))
        out.append("DETECTOR", [stim.target_rec(-1)])

    # --- Teleportation (unprotected): Bell-measure block 1 against block 2. ---
    for i in range(n):
        _apply_noisy(out, "CX", [b1[i], b2[i]], p1, p2)
    for i in range(n):
        _apply_noisy(out, "H", [b1[i]], p1, p2)
    _noise_before_measure(out, b1 + b2, p_m)
    out.append("M", b1)  # outcomes m0_i -> rec[-2n + i]
    out.append("M", b2)  # outcomes m1_i -> rec[-n  + i]

    # --- Classically-controlled Pauli correction Q. ---
    # The teleportation byproduct is C X^{m1} Z^{m0}; correcting it leaves C
    # standing, with each factor read off the tableau as (C X_i C†), (C Z_i C†).
    for i in range(n):
        _apply_pauli_controlled(out, tableau.x_output(i), b3, -n + i)   # ^ m1_i
        _apply_pauli_controlled(out, tableau.z_output(i), b3, -2 * n + i)  # ^ m0_i

    # --- Choi observables: the 2n stabilizers of (I ⊗ C)|Φ+⟩ on (ref, b3). ---
    # All 2n preserved  <=>  the implemented channel equals C.
    observable = 0
    for i in range(n):
        for base, conjugated in (("X", tableau.x_output(i)), ("Z", tableau.z_output(i))):
            letters = ["I"] * total
            letters[ref[i]] = base
            for j in range(n):
                letters[b3[j]] = _PAULI_CHR[conjugated[j]]
            out.append("MPP", _pauli_to_mpp_targets(stim.PauliString("".join(letters))))
            out.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], observable)
            observable += 1

    return out