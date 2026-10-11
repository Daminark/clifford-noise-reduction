"""Random Clifford circuits to protect.

:func:`random_gate_sequence` draws gates uniformly from ``{H, S, CX}``, the
construction of [1]_, App. D, and is reproducible from a NumPy generator.
:func:`random_clifford` samples a uniformly random Clifford [2]_ and
synthesizes it over ``{H, S, CX}``, the construction of [1]_, Sec. V; Stim's
sampler takes no seed, so these circuits are not reproducible, and
:func:`save_circuits` and :func:`load_circuits` keep a generated set.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
.. [2] S. Bravyi and D. Maslov, "Hadamard-free circuits expose the structure
   of the Clifford group", arXiv:2003.09412 (2020).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import stim

from . import _validation

__all__ = [
    "random_gate_sequence",
    "random_clifford",
    "random_cliffords",
    "save_circuits",
    "load_circuits",
]


def random_gate_sequence(
    n: int, size: int, rng: np.random.Generator
) -> stim.Circuit:
    """Return ``size`` gates drawn uniformly from ``{H, S, CX}``.

    Each gate is ``H``, ``S`` or ``CX`` with equal probability (``H`` or
    ``S`` when ``n = 1``); a single-qubit gate acts on a uniformly random
    qubit, and a ``CX`` on a uniformly random ordered pair of distinct
    qubits. This is the construction of arXiv:2407.06583, App. D.

    Parameters
    ----------
    n : int
        Number of qubits. Must be at least 1.
    size : int
        Number of gates. Must be non-negative.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    stim.Circuit
        The gates in order. Qubits no gate touches are left implicit, so
        ``num_qubits`` may be less than ``n``.

    Raises
    ------
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``, or ``n`` or
        ``size`` is not an integer.
    ValueError
        If ``n < 1`` or ``size < 0``.

    Notes
    -----
    The same generator state gives the same circuit within one NumPy
    version; NumPy does not promise identical streams across versions.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    _validation.require_int(n, "n")
    _validation.require_int(size, "size")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")

    alphabet = ["H", "S"] if n == 1 else ["H", "S", "CX"]
    circuit = stim.Circuit()
    choices = rng.integers(len(alphabet), size=size)
    for name in (alphabet[choice] for choice in choices):
        if name == "CX":
            control, target = rng.choice(n, 2, replace=False)
            circuit.append("CX", [int(control), int(target)])
        else:
            circuit.append(name, [int(rng.integers(n))])
    return circuit


def random_clifford(n: int) -> stim.Circuit:
    """Return a uniformly random ``n``-qubit Clifford over ``{H, S, CX}``.

    The Clifford is sampled by Stim's implementation of Bravyi and Maslov's
    algorithm, as in arXiv:2407.06583, Sec. V, and synthesized by Gaussian
    elimination into ``O(n^2)`` gates.

    Parameters
    ----------
    n : int
        Number of qubits. Must be at least 1.

    Returns
    -------
    stim.Circuit

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n < 1``.

    Notes
    -----
    Not reproducible: Stim's sampler takes no seed, and Stim documents its
    synthesis as unstable across versions and possibly randomized. Keep a
    generated set with :func:`save_circuits` for anything that must be
    rerun.
    """
    _validation.require_int(n, "n")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return stim.Tableau.random(n).to_circuit(method="elimination")


def random_cliffords(n: int, count: int) -> list[stim.Circuit]:
    """Return ``count`` independent :func:`random_clifford` circuits.

    Parameters
    ----------
    n : int
        Number of qubits. Must be at least 1.
    count : int
        Number of circuits. Must be at least 1.

    Returns
    -------
    list of stim.Circuit

    Raises
    ------
    TypeError
        If ``n`` or ``count`` is not an integer.
    ValueError
        If ``n < 1`` or ``count < 1``.
    """
    _validation.require_int(n, "n")
    _validation.require_int(count, "count")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    return [random_clifford(n) for _ in range(count)]


def save_circuits(circuits: list[stim.Circuit], path: str | Path) -> None:
    """Write ``circuits`` to ``path``, replacing any existing file.

    The file is a JSON list of Stim circuit texts, in order, encoded as
    UTF-8.

    Parameters
    ----------
    circuits : list of stim.Circuit
        Must not be empty.
    path : str or pathlib.Path

    Raises
    ------
    TypeError
        If an element of ``circuits`` is not a ``stim.Circuit``.
    ValueError
        If ``circuits`` is empty.
    """
    if not circuits:
        raise ValueError("circuits must not be empty")
    for circuit in circuits:
        _validation.require_circuit(circuit, "each element of circuits")
    text = json.dumps([str(circuit) for circuit in circuits], indent=1)
    Path(path).write_text(text, encoding="utf-8")


def load_circuits(path: str | Path) -> list[stim.Circuit]:
    """Return the circuits written to ``path`` by :func:`save_circuits`.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    list of stim.Circuit
        In the order they were saved.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [stim.Circuit(circuit) for circuit in json.loads(text)]
