"""Random Clifford circuit generators.

Two constructions, matching the two the paper uses.

:func:`random_gate_sequence` draws gates uniformly from ``{H, S, CX}``, the
Appendix D construction, and is fully seeded: the circuits a given
``numpy.random.Generator`` produces are reproducible across machines and
library versions.

:func:`random_clifford` samples a uniformly random Clifford and synthesises it,
the Sec. V construction. This one is **not** reproducible: Stim's sampler
takes no seed, and Stim documents its synthesis as unstable across versions
and possibly randomized, so fixing the tableau would not be enough either.
:func:`save_circuits` and :func:`load_circuits` exist so that a generated set
can be committed and reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import stim

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
    """Sample ``size`` gates uniformly from ``{H, S, CX}`` on ``n`` qubits.

    Targets are uniform, and the two targets of a ``CX`` are distinct. With
    ``n = 1`` only ``H`` and ``S`` are drawn. Qubits the sample never touches
    are left implicit, so the circuit may act on fewer than ``n`` qubits.

    Raises:
        TypeError: if ``rng`` is not a ``numpy.random.Generator``.
        ValueError: if ``n < 1`` or ``size < 0``.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")

    alphabet = ["H", "S"] if n == 1 else ["H", "S", "CX"]
    circuit = stim.Circuit()
    for name in (alphabet[k] for k in rng.integers(len(alphabet), size=size)):
        if name == "CX":
            a, b = rng.choice(n, 2, replace=False)
            circuit.append("CX", [int(a), int(b)])
        else:
            circuit.append(name, [int(rng.integers(n))])
    return circuit


def random_clifford(n: int) -> stim.Circuit:
    """Synthesise a uniformly random ``n``-qubit Clifford over ``{H, S, CX}``.

    Sampling is Stim's, which implements Bravyi-Maslov and takes no seed, and
    Stim's synthesis is documented as unstable across versions and possibly
    randomized, so repeated calls differ and a fixed tableau would not help.
    Commit a generated set with :func:`save_circuits` for anything that must
    be reproducible.

    Raises:
        ValueError: if ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return stim.Tableau.random(n).to_circuit(method="elimination")


def random_cliffords(n: int, count: int) -> list[stim.Circuit]:
    """Return ``count`` independent :func:`random_clifford` circuits.

    Raises:
        ValueError: if ``count < 1``, or as :func:`random_clifford`.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    return [random_clifford(n) for _ in range(count)]


def save_circuits(circuits: list[stim.Circuit], path: str | Path) -> None:
    """Write ``circuits`` to ``path`` as a JSON list of Stim circuit texts.

    Raises:
        ValueError: if ``circuits`` is empty.
    """
    if not circuits:
        raise ValueError("circuits must not be empty")
    Path(path).write_text(json.dumps([str(c) for c in circuits], indent=1))


def load_circuits(path: str | Path) -> list[stim.Circuit]:
    """Read circuits written by :func:`save_circuits`."""
    return [stim.Circuit(text) for text in json.loads(Path(path).read_text())]
