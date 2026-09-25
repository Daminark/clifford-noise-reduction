"""CliNR_{t,r} circuit construction (arXiv:2407.06583, Theorem 1).

Builds gate-teleportation circuits that apply a Clifford ``C`` through
resource states guarded by ``r`` random stabilizer checks. The circuit is
split into ``t`` chunks, each teleported through its own checked resource.
All ``r`` checks of a chunk are measured before its resource is accepted or
discarded; the scheme does not abort at the first nontrivial outcome.
Circuits are exact at ``p = 0``, which is the correctness guarantee the test
suite asserts.

Register layout (``4n + 1`` qubits: four blocks of ``n`` plus one ancilla)::

    b1  [0,   n)     data input
    b2  [n,  2n)     resource half teleported against
    b3  [2n, 3n)     resource half carrying C; the output
    ref [3n, 4n)     Choi witness -- not physical (test apparatus only)
    anc 4n           ancilla for stabilizer measurement

Noise (Sec. II) is carried by a :class:`~noise_models.NoiseModel`: ``p1``
after single-qubit gates (DEPOLARIZE1), ``p2`` after controlled-Paulis
(DEPOLARIZE2), ``p_m`` for measurement flips (the reported outcome of ``M``),
and ``p_prep`` after preparations (X error); idle qubits noiseless.

Three modelling choices depart from the paper and are counted as such by the
analytic bounds:

1. ``|+>`` is prepared as a reset followed by a Hadamard, two fault
   locations, where the paper's Sec. II charges one. Blocks 2 and 3 therefore
   contribute ``2n`` preparations plus ``n`` single-qubit locations per chunk
   rather than ``2n`` locations total. Preparation error is a bit flip rather
   than a depolarizing channel.
2. The Pauli correction ``Q`` of Eq. (1) is tracked classically rather than
   applied, so it is not a fault location. The paper charges ``n``.
3. Checks are drawn uniformly from the resource stabilizer group by default,
   as the proof of Theorem 1 requires, and so have weight up to ``2n``. The
   paper's Sec. V numerics instead restrict to the weight-``(n+1)`` propagated
   Bell stabilizers, available here via ``restricted_checks``. That family does
   not carry the ``2^-r`` guarantee; it generates the group, so a fault escapes
   one check with probability at most ``1 - 2/(3n)``.

The Choi harness on ``ref`` is noiseless: it is measuring apparatus, not
protocol, and does not appear in the bounds.
"""

from __future__ import annotations

import stim

from .noise_models import NoiseModel

__all__ = [
    "blocks",
    "gate_list",
    "stabilizer_gens",
    "chunk_circuit",
    "build_clinr_tr",
    "build_direct",
]

# Map Stim's Pauli codes (as returned by PauliString indexing) to letters.
_PAULI_CHR = "IXYZ"

# Unitary gates whose tableau is the identity. These pad a chunk to full width
# and are not fault locations. The test suite asserts this set is exhaustive.
_IDENTITY_GATES = frozenset({"I", "II"})


def _is_two_qubit(name: str) -> bool:
    """Whether ``name`` is a two-qubit gate, according to Stim's gate data."""
    return stim.gate_data(name).is_two_qubit_gate


# ---------------------------------------------------------------------------
# Operation emitters: each appends its operation, and noise when the rate is
# non-zero.
# ---------------------------------------------------------------------------
def _prep_noisy(
    circ: stim.Circuit, targets: list[int], noise: NoiseModel
) -> None:
    """Append ``R`` on ``targets`` followed by an ``X_ERROR`` at ``p_prep``."""
    circ.append("R", targets)
    if noise.p_prep > 0:
        circ.append("X_ERROR", targets, noise.p_prep)


def _measure_noisy(
    circ: stim.Circuit, targets: list[int], noise: NoiseModel
) -> None:
    """Append a ``Z``-basis ``M`` on ``targets`` whose outcome flips at ``p_m``."""
    if noise.p_m > 0:
        circ.append("M", targets, noise.p_m)
    else:
        circ.append("M", targets)


def _apply_noisy(
    circ: stim.Circuit, gate: str, targets: list[int], noise: NoiseModel
) -> None:
    """Append ``gate`` on ``targets`` followed by its depolarizing noise.

    DEPOLARIZE1 at rate ``p1`` for single-qubit gates, DEPOLARIZE2 at rate
    ``p2`` for two-qubit gates. ``targets`` may hold several fused gates; a
    channel instruction broadcasts per gate, so it is passed through whole.
    """
    circ.append(gate, targets)
    two = _is_two_qubit(gate)
    rate = noise.p2 if two else noise.p1
    if rate > 0:
        circ.append("DEPOLARIZE2" if two else "DEPOLARIZE1", targets, rate)


def _prep_plus_noisy(
    circ: stim.Circuit,
    targets: list[int],
    noise: NoiseModel,
    *,
    native_plus: bool,
) -> None:
    """Prepare ``targets`` in ``|+>``, by whichever convention is in force.

    ``RX`` followed by a ``Z_ERROR`` at ``p_prep`` when ``native_plus``, one
    fault location; otherwise ``R`` then ``H``, two locations, with the reset
    error a bit flip and the Hadamard depolarizing.
    """
    if native_plus:
        circ.append("RX", targets)
        if noise.p_prep > 0:
            circ.append("Z_ERROR", targets, noise.p_prep)
    else:
        _prep_noisy(circ, targets, noise)
        _apply_noisy(circ, "H", targets, noise)


# ---------------------------------------------------------------------------
# Gate flattening
# ---------------------------------------------------------------------------
def gate_list(circuit: stim.Circuit) -> list[tuple[str, list[int]]]:
    """Flatten ``circuit`` into one entry per gate, as ``(name, targets)``.

    Fused instructions are split by the gate's arity into individual gates.
    Identity gates are dropped, as are annotations (instructions with no state
    effect, no measurement record and no error): neither is a fault location.

    Raises:
        ValueError: if an instruction is an operation other than a unitary
            gate, if a gate is neither single- nor two-qubit, or if any of its
            targets is not a bare qubit.
    """
    gates: list[tuple[str, list[int]]] = []
    for inst in circuit.flattened():
        if inst.name in _IDENTITY_GATES:
            continue
        data = stim.gate_data(inst.name)
        if not data.is_unitary:
            if data.is_noisy_gate or data.produces_measurements or data.is_reset:
                raise ValueError(f"{inst.name!r} is not a unitary gate")
            continue
        if not (data.is_single_qubit_gate or data.is_two_qubit_gate):
            raise ValueError(
                f"{inst.name!r} is neither a single- nor a two-qubit gate"
            )
        raw = inst.targets_copy()
        if not all(target.is_qubit_target for target in raw):
            raise ValueError(f"{inst.name!r} has a non-qubit target")
        targets = [target.value for target in raw]
        width = 2 if _is_two_qubit(inst.name) else 1
        gates.extend(
            (inst.name, targets[i : i + width])
            for i in range(0, len(targets), width)
        )
    return gates


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


def _pad_to_width(circuit: stim.Circuit, n: int) -> stim.Circuit:
    """Return ``circuit`` prefixed with an identity on all ``n`` qubits.

    Stim sizes a circuit's tableau by the highest qubit index it mentions, so
    the padding pins the tableau to width ``n`` even when ``circuit`` leaves
    some qubits untouched.

    Raises:
        ValueError: if ``circuit`` acts on a qubit index of ``n`` or more.
    """
    if circuit.num_qubits > n:
        raise ValueError(
            f"circuit acts on {circuit.num_qubits} qubits, exceeding n = {n}"
        )
    padded = stim.Circuit()
    padded.append("I", list(range(n)))
    padded += circuit
    return padded


# ---------------------------------------------------------------------------
# Resource-state stabilizers
# ---------------------------------------------------------------------------
def stabilizer_gens(
    tableau: stim.Tableau, n: int, bare: list[int], image: list[int], total: int
) -> list[stim.PauliString]:
    """Generators of the stabilizer group of ``(I ⊗ C)|Φ+⟩^{⊗n}``, in closed form.

    With ``n`` Bell pairs joining ``bare`` to ``image`` and ``C`` applied to
    ``image``, the state is stabilized by the Bell generators conjugated by
    ``C``::

        X_{bare[i]} ⊗ (C X_i C†)_{image}   and   Z_{bare[i]} ⊗ (C Z_i C†)_{image}

    for ``i = 0, ..., n-1``, taken from the tableau. Signs are dropped: they do
    not affect measurement outcomes or the anticommutation that drives
    detection.

    Args:
        tableau: The tableau of the Clifford ``C``.
        n: Number of data qubits.
        bare: Index list for the block ``C`` does not act on.
        image: Index list for the block carrying ``C``.
        total: Full register width (length of each returned Pauli string).

    Returns:
        A list of ``2n`` independent generators as length-``total`` Paulis.
    """
    gens: list[stim.PauliString] = []
    for i in range(n):
        for base, conjugated in (("X", tableau.x_output(i)), ("Z", tableau.z_output(i))):
            letters = ["I"] * total
            letters[bare[i]] = base
            for j in range(n):
                letters[image[j]] = _PAULI_CHR[conjugated[j]]
            gens.append(stim.PauliString("".join(letters)))
    return gens


# ---------------------------------------------------------------------------
# Stabilizer-measurement gadget
# ---------------------------------------------------------------------------
def _measure_stabilizer_noisy(
    circ: stim.Circuit,
    pauli: stim.PauliString,
    anc: int,
    noise: NoiseModel,
    *,
    native_plus: bool = False,
) -> None:
    """Measure ``pauli`` via a noisy ancilla-based gadget, appending one record.

    Realizes the measurement as reset -> H -> (controlled-Pauli per support
    qubit) -> H -> measure, with circuit-level noise on every operation. This is
    the faithful, noisy stand-in for Stim's idealized ``MPP``: a weight-``w``
    stabilizer costs ``w + 4`` locations, or ``w + 3`` with ``native_plus``,
    where a single ``RX`` replaces the reset and the first Hadamard. Use
    ``MPP`` directly instead for an idealized, noiseless check.

    Args:
        circ: Circuit to append to.
        pauli: The stabilizer to measure.
        anc: Ancilla index, outside the support of ``pauli``.
        noise: The four rates.
        native_plus: If True, prepare the ancilla in ``|+>`` with a single
            ``RX`` rather than ``R`` then ``H``.

    Raises:
        ValueError: if ``anc`` lies in the support of ``pauli``.
    """
    if pauli[anc] != 0:
        raise ValueError(
            f"ancilla {anc} lies in the support of the measured stabilizer"
        )
    controlled = {1: "CX", 2: "CY", 3: "CZ"}
    _prep_plus_noisy(circ, [anc], noise, native_plus=native_plus)
    for j in range(len(pauli)):
        code = pauli[j]
        if code == 0:
            continue
        _apply_noisy(circ, controlled[code], [anc, j], noise)
    _apply_noisy(circ, "H", [anc], noise)
    _measure_noisy(circ, [anc], noise)


# ---------------------------------------------------------------------------
# CliNR_{1,r} circuit builder
# ---------------------------------------------------------------------------
def chunk_circuit(circuit: stim.Circuit, n: int, t: int) -> list[stim.Circuit]:
    """Split ``circuit`` into ``t`` contiguous sub-circuits (the CliNR chunks).

    The gate list of ``circuit`` (:func:`gate_list`, so identities are dropped
    and fused instructions are split) is cut into ``t`` pieces of size
    ``s0 = ceil(s/t)`` (the first ``s mod t`` chunks) and ``s0 - 1`` (the
    rest), so the sizes sum to ``s``, matching the paper. Each chunk is padded
    with an identity on all ``n`` qubits so that its tableau is full width
    ``n`` even when the chunk's gates touch only some qubits.

    Exactly ``t`` chunks are returned. If ``t`` exceeds the gate count the
    surplus chunks are empty, and each still costs a full teleportation round.

    Args:
        circuit: The Clifford circuit ``C`` to split.
        n: Number of data qubits.
        t: Number of chunks.

    Returns:
        A list of ``t`` sub-circuits whose concatenation equals ``circuit``.

    Raises:
        ValueError: if ``n < 1``, ``t < 1``, or ``circuit`` acts on a qubit
            index of ``n`` or more.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if t < 1:
        raise ValueError(f"t must be at least 1, got {t}")
    gates = gate_list(circuit)
    s = len(gates)
    s0 = -(-s // t)                       # ceil division
    n_full = s % t or t                   # count of size-s0 chunks
    chunks: list[stim.Circuit] = []
    idx = 0
    for k in range(t):
        size = s0 if k < n_full else s0 - 1
        sub = stim.Circuit()
        for name, targets in gates[idx : idx + size]:
            sub.append(name, targets)
        chunks.append(_pad_to_width(sub, n))
        idx += size
    return chunks


def _embed(
    check: stim.PauliString, bare: list[int], image: list[int], total: int
) -> stim.PauliString:
    """Place a length-``2n`` check onto the ``bare`` and ``image`` blocks."""
    n = len(bare)
    out = stim.PauliString(total)
    for i in range(n):
        out[bare[i]] = check[i]
        out[image[i]] = check[n + i]
    return out


def build_direct(
    circuit: stim.Circuit,
    n: int,
    noise: NoiseModel,
) -> stim.Circuit:
    """Build the direct implementation of ``C`` with the Choi channel test.

    ``C`` is applied to the data block with circuit-level noise and no
    teleportation, resource state, or checks. This is the baseline the CliNR
    implementation is compared against.

    Args:
        circuit: The Clifford circuit ``C`` over ``n`` qubits.
        n: Number of data qubits.
        noise: The circuit-level error rates.

    Returns:
        A ``stim.Circuit`` with no detectors and ``2n`` observables, exact at
        zero noise.

    Raises:
        ValueError: if ``n < 1``, or if ``circuit`` acts on more than ``n``
            qubits.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    full_tableau = _pad_to_width(circuit, n).to_tableau()
    b1, _, _, ref, _ = blocks(n)
    total = 4 * n + 1
    out = stim.Circuit()

    for i in range(n):
        out.append("H", [ref[i]])
        out.append("CX", [ref[i], b1[i]])

    for name, targets in gate_list(circuit):
        _apply_noisy(out, name, [b1[q] for q in targets], noise)

    for observable, gen in enumerate(
        stabilizer_gens(full_tableau, n, ref, b1, total)
    ):
        out.append("MPP", _pauli_to_mpp_targets(gen))
        out.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], observable)

    return out


def build_clinr_tr(
    circuit: stim.Circuit,
    n: int,
    checks: list[list[stim.PauliString]],
    noise: NoiseModel,
    *,
    noisy_checks: bool = True,
    noisy_q: bool = False,
    native_plus: bool = False,
) -> stim.Circuit:
    """Build ``CliNR_{t,r}(C)`` with the Choi channel-test harness.

    ``C`` is split into ``t`` chunks (:func:`chunk_circuit`); each chunk is
    applied by gate teleportation through its own checked resource state, and
    the data is relabelled between chunks so that one chunk's output block feeds
    the next chunk's input (a software swap, no gates). A single Choi harness
    spans the whole chain: one reference register Bell-paired to the first
    input, and one readout against the final output using the *full* circuit's
    tableau. The result is a pure stabilizer circuit, exact at zero noise.
    Passing ``t=1`` recovers the single-block scheme (Lemma 1).

    The teleportation byproduct is tracked as a Pauli frame rather than
    applied: each measurement record is associated with the Pauli it would
    control, the frame is conjugated through each subsequent chunk, and the
    records that anticommute with an observable are folded into it at readout.

    Args:
        circuit: The Clifford circuit ``C`` over ``n`` qubits.
        n: Number of data qubits.
        checks: One list of ``r`` checks per chunk, the same ``r`` for every
            chunk, in the format returned by :func:`draw_checks`. The number
            of chunks ``t`` is ``len(checks)``. Any stabilizers of the
            resource state give a valid circuit; the bounds in :mod:`bounds`
            cover only the two distributions :func:`draw_checks` implements.
        noise: The circuit-level error rates.
        noisy_checks: If True, measure checks with the explicit noisy gadget;
            if False, use idealized ``MPP``.
        noisy_q: If True, add a one-time depolarizing layer on the final data
            block, modelling physical resolution of the tracked Pauli frame
            into a live corrected state. Placed after all checks, so it never
            triggers a restart. Left False, the frame is resolved for free at
            readout.
        native_plus: If True, prepare every ``|+>`` with a single ``RX``, the
            paper's Sec. II convention: the ``n`` Hadamards per chunk and the
            first Hadamard of each check gadget go away, bringing a check to
            the paper's ``w + 3`` locations. If False, ``|+>`` is ``R`` then
            ``H``.

    Returns:
        A ``stim.Circuit`` on ``4n + 1`` qubits with one detector per check, in
        chunk order, and ``2n`` observables (the Choi stabilizers of the full
        ``C``).

    Raises:
        ValueError: if the chunks have different numbers of checks; if a
            check does not have length ``2n``, is the identity, or does not
            stabilize its chunk's resource state; or from
            :func:`chunk_circuit`.
    """
    if len({len(chunk_checks) for chunk_checks in checks}) > 1:
        raise ValueError(
            "every chunk needs the same number of checks, got "
            f"{[len(chunk_checks) for chunk_checks in checks]}"
        )
    full_tableau = _pad_to_width(circuit, n).to_tableau()
    chunks = chunk_circuit(circuit, n, len(checks))
    b1, b2, b3, ref, anc = blocks(n)
    total = 4 * n + 1
    out = stim.Circuit()
    frame: dict[int, stim.PauliString] = {}

    # Choi input on (ref, b1): entangle each reference qubit with the first
    # chunk's input. Noiseless -- this is measuring apparatus, not protocol.
    for i in range(n):
        out.append("H", [ref[i]])
        out.append("CX", [ref[i], b1[i]])

    data_in = b1  # block the current chunk reads its input from
    for k, (chunk, chunk_checks) in enumerate(
        zip(chunks, checks, strict=True)
    ):
        tableau = chunk.to_tableau()
        gens = stabilizer_gens(
            tableau, n, list(range(n)), list(range(n, 2 * n)), 2 * n
        )
        for check in chunk_checks:
            if len(check) != 2 * n:
                raise ValueError(
                    f"chunk {k}: check has length {len(check)}, expected {2 * n}"
                )
            if check.weight == 0:
                raise ValueError(f"chunk {k}: check is the identity")
            if not all(check.commutes(g) for g in gens):
                raise ValueError(
                    f"chunk {k}: check {check} does not stabilize the resource"
                )

        # Protected region: prepare this chunk's resource (I x C_k)|Phi+>.
        for i in range(n):
            _prep_plus_noisy(out, [b2[i]], noise, native_plus=native_plus)
            _prep_noisy(out, [b3[i]], noise)
            _apply_noisy(out, "CX", [b2[i], b3[i]], noise)
        for name, targets in gate_list(chunk):
            _apply_noisy(out, name, [b3[q] for q in targets], noise)

        # Offline checks on this chunk's resource.
        for logical in chunk_checks:
            check = _embed(logical, b2, b3, total)
            if noisy_checks:
                _measure_stabilizer_noisy(
                    out, check, anc, noise, native_plus=native_plus
                )
            else:
                out.append("MPP", _pauli_to_mpp_targets(check))
            out.append("DETECTOR", [stim.target_rec(-1)])

        # Teleport the data through this chunk: Bell-measure data_in vs b2.
        for i in range(n):
            _apply_noisy(out, "CX", [data_in[i], b2[i]], noise)
        for i in range(n):
            _apply_noisy(out, "H", [data_in[i]], noise)
        frame = {index: tableau(pauli) for index, pauli in frame.items()}
        base = out.num_measurements
        _measure_noisy(out, data_in, noise)  # m0_i at absolute index base + i
        _measure_noisy(out, b2, noise)       # m1_i at absolute index base + n + i
        for i in range(n):
            frame[base + n + i] = tableau.x_output(i)
            frame[base + i] = tableau.z_output(i)

        # Relabel: this chunk's output (b3) is the next chunk's input; reuse the
        # spent input register as the next chunk's resource-output scratch.
        data_in, b3 = b3, data_in

    # Choi observables between ref and the final data block, using the full
    # circuit's tableau: all 2n preserved  <=>  the channel equals C.
    # Resolving the tracked frame into a live corrected state would apply one
    # Pauli to the final block; its noise is a single depolarizing layer here,
    # equivalent in distribution because depolarizing noise is Pauli-symmetric.
    if noisy_q and noise.p1 > 0:
        out.append("DEPOLARIZE1", data_in, noise.p1)

    for observable, gen in enumerate(
        stabilizer_gens(full_tableau, n, ref, data_in, total)
    ):
        out.append("MPP", _pauli_to_mpp_targets(gen))
        # the generator's data-block part, back in logical indexing
        conj = stim.PauliString([gen[q] for q in data_in])
        targets = [stim.target_rec(-1)]
        for index, pauli in frame.items():
            if not pauli.commutes(conj):
                targets.append(stim.target_rec(index - out.num_measurements))
        out.append("OBSERVABLE_INCLUDE", targets, observable)

    return out
