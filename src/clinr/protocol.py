"""Construction of the CliNR circuits and the direct baseline.

:func:`build_clinr_tr` builds ``CliNR_{t,r}(C)`` of [1]_, Sec. III: ``C`` is
split into ``t`` chunks, and each chunk is applied by gate teleportation
through its own resource state, guarded by ``r`` stabilizer checks that the
caller supplies (:func:`clinr.checks.draw_checks` draws them).
All ``r`` checks of a chunk are measured before its resource is accepted or
discarded, where the paper restarts at the first nontrivial outcome; the two
accept the same shots.
:func:`build_direct` builds the baseline, ``C`` applied directly.
Both carry a Choi harness that reads out whether the implemented channel is
``C``, and both are exact at zero noise: every detector and observable is
deterministic.

Register layout, ``4n + 1`` qubits::

    b1  [0,   n)     data input of the first chunk
    b2  [n,  2n)     resource half the data is teleported against
    b3  [2n, 3n)     resource half carrying the chunk
    ref [3n, 4n)     Choi reference, noiseless apparatus
    anc 4n           ancilla for the check gadget

After each chunk, ``b1`` and ``b3`` exchange roles by relabeling, with no
gates, so the output ends on ``b3`` when ``t`` is odd and on ``b1`` when
``t`` is even.

Noise follows a :class:`clinr.noise_models.NoiseModel`: ``DEPOLARIZE1`` at
``p1`` after single-qubit gates, ``DEPOLARIZE2`` at ``p2`` after two-qubit
gates, a flipped outcome at ``p_m`` for each measurement, and at ``p_prep``
an ``X_ERROR`` after ``R`` or a ``Z_ERROR`` after ``RX``. Idle qubits and the
Choi harness are noiseless.

Three modeling choices depart from [1]_, Sec. II, and the analytic bounds
count them as such:

1. By default ``|+>`` is a reset followed by a Hadamard, two fault
   locations where the paper charges one; ``native_plus=True`` prepares it
   with one ``RX``, as the paper does.
2. A preparation error is a flip with probability ``p_prep``, where the
   paper's depolarizing error flips a fresh preparation with probability
   ``2p/3``.
3. The Pauli correction ``Q`` of Eq. (1) is never applied: it is tracked
   as a Pauli frame and folded into the observables, so the circuit
   implements ``C`` up to that frame and ``Q`` is not a fault location,
   where the paper applies and charges it after every chunk.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
"""

from __future__ import annotations

import stim

from .noise_models import NoiseModel

__all__ = [
    "gate_list",
    "stabilizer_gens",
    "chunk_circuit",
    "build_clinr_tr",
    "build_direct",
]

# Stim's Pauli codes, as returned by indexing a PauliString, to letters.
_PAULI_CHR = "IXYZ"

# Unitary gates whose tableau is the identity. They pad a chunk to full width
# and are not fault locations. The test suite checks this set is exhaustive.
_IDENTITY_GATES = frozenset({"I", "II"})

# The noise model of a check gadget when checks are noiseless.
_NOISELESS = NoiseModel(p1=0.0, p2=0.0, p_m=0.0, p_prep=0.0)


def _is_two_qubit(name: str) -> bool:
    """Return whether Stim's gate data marks ``name`` as a two-qubit gate."""
    return bool(stim.gate_data(name).is_two_qubit_gate)


# ---------------------------------------------------------------------------
# Operation emitters: each appends its operation, and noise when the rate is
# non-zero.
# ---------------------------------------------------------------------------
def _prep_noisy(
    circ: stim.Circuit, targets: list[int], noise: NoiseModel, tag: str
) -> None:
    """Append ``R`` on ``targets`` followed by an ``X_ERROR`` at ``p_prep``."""
    circ.append("R", targets)
    if noise.p_prep > 0:
        circ.append("X_ERROR", targets, noise.p_prep, tag=tag)


def _measure_noisy(
    circ: stim.Circuit, targets: list[int], noise: NoiseModel, tag: str
) -> None:
    """Append ``M`` on ``targets``, each outcome flipped at ``p_m``."""
    if noise.p_m > 0:
        circ.append("M", targets, noise.p_m, tag=tag)
    else:
        circ.append("M", targets)


def _apply_noisy(
    circ: stim.Circuit,
    gate: str,
    targets: list[int],
    noise: NoiseModel,
    tag: str,
) -> None:
    """Append ``gate`` on ``targets``, then its depolarizing noise.

    ``targets`` may hold several fused gates; the channel broadcasts per gate
    in the same way, so it takes the same target list.
    """
    circ.append(gate, targets)
    two = _is_two_qubit(gate)
    rate = noise.p2 if two else noise.p1
    if rate > 0:
        name = "DEPOLARIZE2" if two else "DEPOLARIZE1"
        circ.append(name, targets, rate, tag=tag)


def _prep_plus_noisy(
    circ: stim.Circuit,
    targets: list[int],
    noise: NoiseModel,
    tag: str,
    *,
    native_plus: bool,
) -> None:
    """Prepare ``targets`` in ``|+>``: ``RX`` if ``native_plus``, else R, H."""
    if native_plus:
        circ.append("RX", targets)
        if noise.p_prep > 0:
            circ.append("Z_ERROR", targets, noise.p_prep, tag=tag)
    else:
        _prep_noisy(circ, targets, noise, tag)
        _apply_noisy(circ, "H", targets, noise, tag)


# ---------------------------------------------------------------------------
# Gate flattening
# ---------------------------------------------------------------------------
def gate_list(circuit: stim.Circuit) -> list[tuple[str, list[int]]]:
    """Return the gates of ``circuit``, one ``(name, targets)`` per gate.

    ``REPEAT`` blocks are unrolled and fused instructions are split by the
    gate's arity. Identity gates and annotations (instructions with no effect
    on the state, no measurement record and no error) are dropped, since
    neither is a fault location.

    Parameters
    ----------
    circuit : stim.Circuit
        A circuit of single- and two-qubit unitary gates on qubit targets,
        possibly with annotations.

    Returns
    -------
    list of tuple of (str, list of int)
        The gates in order.

    Raises
    ------
    ValueError
        If an instruction is a noise channel, a measurement or a reset, if a
        unitary acts on more than two qubits, or if a gate has a target
        other than a qubit.
    """
    gates: list[tuple[str, list[int]]] = []
    # Circuit iterates through __getitem__, which Stim's stubs do not model.
    for inst in circuit.flattened():  # type: ignore[attr-defined]
        if inst.name in _IDENTITY_GATES:
            continue
        data = stim.gate_data(inst.name)
        if not data.is_unitary:
            if (
                data.is_noisy_gate
                or data.produces_measurements
                or data.is_reset
            ):
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
def _blocks(n: int) -> tuple[list[int], list[int], list[int], list[int], int]:
    """Return ``(b1, b2, b3, ref, anc)`` of the module's register layout."""
    b1 = list(range(0, n))
    b2 = list(range(n, 2 * n))
    b3 = list(range(2 * n, 3 * n))
    ref = list(range(3 * n, 4 * n))
    anc = 4 * n
    return b1, b2, b3, ref, anc


def _pauli_to_mpp_targets(pauli: stim.PauliString) -> list[stim.GateTarget]:
    """Return ``MPP`` targets measuring the product ``pauli``, sign ignored.

    ``pauli`` must not be the identity. The sign does not matter because
    Stim compares detectors and observables against a noiseless reference.
    """
    assert pauli.weight > 0, "MPP of the identity"
    ctor = {1: stim.target_x, 2: stim.target_y, 3: stim.target_z}
    targets: list[stim.GateTarget] = []
    for qubit in range(len(pauli)):
        code = pauli[qubit]
        if code == 0:
            continue
        if targets:
            targets.append(stim.target_combiner())
        targets.append(ctor[code](qubit))
    return targets


def _pad_to_width(circuit: stim.Circuit, n: int) -> stim.Circuit:
    """Return ``circuit`` after an identity on all ``n`` qubits.

    Stim sizes a tableau by the highest qubit index a circuit mentions, so
    the padding makes the tableau exactly ``n`` wide. This is where the
    public builders reject a circuit wider than ``n``.
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
    tableau: stim.Tableau, bare: list[int], image: list[int], total: int
) -> list[stim.PauliString]:
    """Return generators of the stabilizer group of the resource state.

    With ``n = len(tableau)``, the resource is ``n`` Bell pairs joining
    ``bare[i]`` to ``image[i]``, with the Clifford ``C`` of ``tableau``
    applied to ``image``. Its stabilizer group is generated by
    ``X_{bare[i]} (C X_i C^dag)_{image}`` and
    ``Z_{bare[i]} (C Z_i C^dag)_{image}`` for ``i`` in ``range(n)``.

    Parameters
    ----------
    tableau : stim.Tableau
        Tableau of ``C``, on at least one qubit. A tableau taken from a
        circuit is only as wide as the highest qubit the circuit touches, so
        pad the circuit first (the chunks of :func:`chunk_circuit` are).
    bare : list of int
        The ``n`` indices of the half ``C`` does not act on.
    image : list of int
        The ``n`` indices of the half carrying ``C``.
    total : int
        Length of the returned Pauli strings. Every index in ``bare`` and
        ``image`` must lie in ``[0, total)``, and no index may repeat.

    Returns
    -------
    list of stim.PauliString
        ``2n`` independent, pairwise commuting generators, ordered
        ``X``-type then ``Z``-type for each pair in turn. Signs are dropped;
        detection depends only on commutation.

    Raises
    ------
    ValueError
        If ``tableau`` acts on no qubits, if ``bare`` or ``image`` does not
        hold ``len(tableau)`` indices, or if the indices repeat or fall
        outside ``[0, total)``.
    """
    n = len(tableau)
    if n < 1:
        raise ValueError("tableau must act on at least one qubit")
    if len(bare) != n or len(image) != n:
        raise ValueError(
            f"bare and image must each hold {n} indices, one per qubit of "
            f"the tableau, got {len(bare)} and {len(image)}"
        )
    indices = [*bare, *image]
    if len(set(indices)) != 2 * n:
        raise ValueError(f"bare and image repeat an index: {indices}")
    if not all(0 <= index < total for index in indices):
        raise ValueError(f"indices must lie in [0, {total}), got {indices}")

    gens: list[stim.PauliString] = []
    for i in range(n):
        for base, conjugated in (
            ("X", tableau.x_output(i)),
            ("Z", tableau.z_output(i)),
        ):
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
    tag: str,
    *,
    native_plus: bool = False,
) -> None:
    """Append a noisy ancilla measurement of ``pauli``, one record.

    The gadget is ``|+>`` on ``anc``, a controlled Pauli from ``anc`` to each
    support qubit, ``H``, then ``M``: ``w + 4`` fault locations for weight
    ``w``, or ``w + 3`` with ``native_plus``. ``anc`` must lie outside the
    support of ``pauli``.
    """
    assert pauli[anc] == 0, f"ancilla {anc} in the support of {pauli}"
    controlled = {1: "CX", 2: "CY", 3: "CZ"}
    _prep_plus_noisy(circ, [anc], noise, tag, native_plus=native_plus)
    for qubit in range(len(pauli)):
        code = pauli[qubit]
        if code == 0:
            continue
        _apply_noisy(circ, controlled[code], [anc, qubit], noise, tag)
    _apply_noisy(circ, "H", [anc], noise, tag)
    _measure_noisy(circ, [anc], noise, tag)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_circuit(circuit: stim.Circuit, n: int, t: int) -> list[stim.Circuit]:
    """Split ``circuit`` into ``t`` contiguous chunks.

    The gates of :func:`gate_list` are cut in order into ``t`` pieces, the
    first ``s mod t`` of size ``ceil(s/t)`` and the rest one smaller, as in
    arXiv:2407.06583, Sec. III. When ``t`` exceeds the gate count ``s`` the
    surplus chunks are empty.

    Parameters
    ----------
    circuit : stim.Circuit
        The Clifford circuit ``C``, as accepted by :func:`gate_list`, on
        qubits ``[0, n)``.
    n : int
        Number of data qubits. Must be at least 1.
    t : int
        Number of chunks. Must be at least 1.

    Returns
    -------
    list of stim.Circuit
        ``t`` chunks whose gates, concatenated, are :func:`gate_list` of
        ``circuit``. Each begins with an identity on all ``n`` qubits, so
        its tableau is ``n`` wide.

    Raises
    ------
    ValueError
        If ``n < 1``, ``t < 1``, or ``circuit`` acts on a qubit index of
        ``n`` or more.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if t < 1:
        raise ValueError(f"t must be at least 1, got {t}")
    gates = gate_list(circuit)
    s = len(gates)
    s0 = -(-s // t)  # ceil(s / t)
    n_full = s % t or t  # how many chunks have size s0
    chunks: list[stim.Circuit] = []
    idx = 0
    for k in range(t):
        size = s0 if k < n_full else s0 - 1
        sub = stim.Circuit()
        for name, targets in gates[idx : idx + size]:
            sub.append(name, targets)
        chunks.append(_pad_to_width(sub, n))
        idx += size
    assert idx == s, f"chunks hold {idx} of {s} gates"
    return chunks


def _embed(
    check: stim.PauliString, bare: list[int], image: list[int], total: int
) -> stim.PauliString:
    """Place a length-``2n`` check onto ``bare`` then ``image``."""
    n = len(bare)
    assert len(check) == 2 * n, f"check of length {len(check)}, n = {n}"
    out = stim.PauliString(total)
    for i in range(n):
        out[bare[i]] = check[i]
        out[image[i]] = check[n + i]
    return out


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_direct(
    circuit: stim.Circuit,
    n: int,
    noise: NoiseModel,
) -> stim.Circuit:
    """Build the direct implementation of ``C`` with the Choi harness.

    ``C`` is applied to ``b1`` with circuit-level noise, with no resource
    state, teleportation or checks.

    Parameters
    ----------
    circuit : stim.Circuit
        The Clifford circuit ``C``, as accepted by :func:`gate_list`, on
        qubits ``[0, n)``.
    n : int
        Number of data qubits. Must be at least 1.
    noise : NoiseModel

    Returns
    -------
    stim.Circuit
        On ``4n + 1`` qubits, with no detectors and ``2n`` observables, the
        Choi stabilizers of ``C``; exact at zero noise. Every noise channel
        is tagged ``gates/0``.

    Raises
    ------
    ValueError
        If ``n < 1``, or if ``circuit`` acts on a qubit index of ``n`` or
        more.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    full_tableau = _pad_to_width(circuit, n).to_tableau()
    b1, _, _, ref, _ = _blocks(n)
    total = 4 * n + 1
    out = stim.Circuit()

    for i in range(n):
        out.append("H", [ref[i]])
        out.append("CX", [ref[i], b1[i]])

    for name, targets in gate_list(circuit):
        _apply_noisy(
            out, name, [b1[qubit] for qubit in targets], noise, "gates/0"
        )

    for observable, gen in enumerate(
        stabilizer_gens(full_tableau, ref, b1, total)
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
    native_plus: bool = False,
) -> stim.Circuit:
    """Build ``CliNR_{t,r}(C)`` with the Choi harness.

    The chunks are those of :func:`chunk_circuit` with ``t = len(checks)``.
    The teleportation byproduct is tracked as a Pauli frame and folded into
    the observables, so the circuit needs no feedback. ``t = 1`` is the
    circuit of Lemma 1 of arXiv:2407.06583.

    Parameters
    ----------
    circuit : stim.Circuit
        The Clifford circuit ``C``, as accepted by :func:`gate_list`, on
        qubits ``[0, n)``.
    n : int
        Number of data qubits. Must be at least 1.
    checks : list of list of stim.PauliString
        One list of ``r`` checks per chunk, the same ``r`` for every chunk,
        laid out as :func:`clinr.checks.draw_checks` returns them: length
        ``2n``, the bare half on ``[0, n)`` and the half carrying the chunk
        on ``[n, 2n)``. Each must be a stabilizer of its chunk's resource
        state; the identity is allowed. There must be at least one chunk.
    noise : NoiseModel
    noisy_checks : bool, optional
        If True, measure each check with the noisy ancilla gadget; if False,
        with the same gadget without noise, so that the checks detect
        faults but cause none.
    native_plus : bool, optional
        If True, prepare every ``|+>`` with one ``RX``, as in
        arXiv:2407.06583, Sec. II, so that a weight-``w`` check costs
        ``w + 3`` locations rather than ``w + 4``. If False, ``|+>`` is
        ``R`` then ``H``.

    Returns
    -------
    stim.Circuit
        On ``4n + 1`` qubits, with one detector per check in chunk order and
        ``2n`` observables, the Choi stabilizers of ``C``; exact at zero
        noise. Every noise channel is tagged with the region it belongs to
        in chunk ``k``, counted from 0, as :mod:`clinr.accounting` divides
        it: ``prep/k`` for the preparation of the resource, ``gates/k`` for
        the chunk's gates, ``check/k/j`` for its ``j``-th check, and
        ``teleport/k`` for the teleportation.

    Raises
    ------
    ValueError
        If the chunks have different numbers of checks, or if a check does
        not have length ``2n`` or does not stabilize its chunk's resource
        state.
    """
    if len({len(chunk_checks) for chunk_checks in checks}) > 1:
        raise ValueError(
            "every chunk needs the same number of checks, got "
            f"{[len(chunk_checks) for chunk_checks in checks]}"
        )
    full_tableau = _pad_to_width(circuit, n).to_tableau()
    chunks = chunk_circuit(circuit, n, len(checks))
    b1, b2, b3, ref, anc = _blocks(n)
    total = 4 * n + 1
    out = stim.Circuit()
    frame: dict[int, stim.PauliString] = {}

    # Choi input: pair each reference qubit with the first chunk's input.
    for i in range(n):
        out.append("H", [ref[i]])
        out.append("CX", [ref[i], b1[i]])

    data_in = b1  # the block the current chunk reads its input from
    for k, (chunk, chunk_checks) in enumerate(
        zip(chunks, checks, strict=True)
    ):
        tableau = chunk.to_tableau()
        gens = stabilizer_gens(
            tableau, list(range(n)), list(range(n, 2 * n)), 2 * n
        )
        for check in chunk_checks:
            if len(check) != 2 * n:
                raise ValueError(
                    f"chunk {k}: check has length {len(check)}, "
                    f"expected {2 * n}"
                )
            if not all(check.commutes(gen) for gen in gens):
                raise ValueError(
                    f"chunk {k}: check {check} does not stabilize the resource"
                )

        # Protected region: prepare this chunk's resource (I x C_k)|Phi+>.
        prep, gates, teleport = f"prep/{k}", f"gates/{k}", f"teleport/{k}"
        for i in range(n):
            _prep_plus_noisy(
                out, [b2[i]], noise, prep, native_plus=native_plus
            )
            _prep_noisy(out, [b3[i]], noise, prep)
            _apply_noisy(out, "CX", [b2[i], b3[i]], noise, prep)
        for name, targets in gate_list(chunk):
            _apply_noisy(
                out, name, [b3[qubit] for qubit in targets], noise, gates
            )

        # Offline checks on this chunk's resource.
        for j, logical in enumerate(chunk_checks):
            check = _embed(logical, b2, b3, total)
            _measure_stabilizer_noisy(
                out,
                check,
                anc,
                noise if noisy_checks else _NOISELESS,
                f"check/{k}/{j}",
                native_plus=native_plus,
            )
            out.append("DETECTOR", [stim.target_rec(-1)])

        # Teleport the data through this chunk: Bell-measure data_in vs b2.
        for i in range(n):
            _apply_noisy(out, "CX", [data_in[i], b2[i]], noise, teleport)
        for i in range(n):
            _apply_noisy(out, "H", [data_in[i]], noise, teleport)
        frame = {index: tableau(pauli) for index, pauli in frame.items()}
        base = out.num_measurements
        _measure_noisy(out, data_in, noise, teleport)  # m0_i at base + i
        _measure_noisy(out, b2, noise, teleport)  # m1_i at base + n + i
        for i in range(n):
            frame[base + n + i] = tableau.x_output(i)
            frame[base + i] = tableau.z_output(i)

        # This chunk's output (b3) is the next chunk's input, and the spent
        # input register becomes the next resource's second half.
        data_in, b3 = b3, data_in

    # Choi observables against the full circuit's tableau: all 2n preserved
    # exactly when the implemented channel is C.
    for observable, gen in enumerate(
        stabilizer_gens(full_tableau, ref, data_in, total)
    ):
        out.append("MPP", _pauli_to_mpp_targets(gen))
        # The generator's output-block part, in logical indexing.
        conj = stim.PauliString([gen[qubit] for qubit in data_in])
        records = [stim.target_rec(-1)]
        for index, pauli in frame.items():
            if not pauli.commutes(conj):
                records.append(stim.target_rec(index - out.num_measurements))
        out.append("OBSERVABLE_INCLUDE", records, observable)

    return out
