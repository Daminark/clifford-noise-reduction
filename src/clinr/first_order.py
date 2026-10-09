"""First-order analysis of the circuits :mod:`clinr.protocol` builds.

A fault here is one outcome of one noise channel at one location: one of the
three Paulis of a ``DEPOLARIZE1``, one of the fifteen of a ``DEPOLARIZE2``,
the flip of an ``X_ERROR`` or ``Z_ERROR``, or the flipped outcome of a noisy
measurement. :func:`faults` lists every fault of a built circuit that flips
some detector or some observable, with its probability, the region its
channel is tagged with, and its effect, as Stim's explanation of the
circuit's detector error model reports them. Faults that flip nothing are
harmless and left out.

To first order in the error rates, at most one fault occurs per shot, so the
logical error rate is the total probability of the faults that flip an
observable and no detector (:func:`logical_error_rate`), and the rejection
rate that of the faults that flip a detector. The estimators of
:mod:`clinr.estimators` measure the same quantities to all orders by
sampling; the first-order values are exact for one draw of the checks,
attribute every contribution to a region, and cost one pass over the
circuit.
"""

from __future__ import annotations

from typing import NamedTuple

import stim

__all__ = ["Fault", "faults", "logical_error_rate"]

# The number of equally likely outcomes of each channel the builders emit.
_OUTCOMES = {
    "DEPOLARIZE1": 3,
    "DEPOLARIZE2": 15,
    "X_ERROR": 1,
    "Z_ERROR": 1,
    "M": 1,
}


class Fault(NamedTuple):
    """One fault of a circuit with an effect on its detectors or observables.

    Parameters
    ----------
    tag : str
        The tag of the noise channel, which names its region (see
        :func:`clinr.protocol.build_clinr_tr`).
    instruction : int
        Index of the channel's instruction in the circuit, so faults sort
        in time order.
    probability : float
        Probability of this fault.
    logical : bool
        Whether it flips some observable.
    detected : bool
        Whether it flips some detector.
    """

    tag: str
    instruction: int
    probability: float
    logical: bool
    detected: bool


def faults(circuit: stim.Circuit) -> list[Fault]:
    """Return every fault of ``circuit`` that flips a detector or observable.

    Parameters
    ----------
    circuit : stim.Circuit
        A circuit built by :mod:`clinr.protocol`, or any circuit without
        ``REPEAT`` blocks whose noise channels are ``DEPOLARIZE1``,
        ``DEPOLARIZE2``, ``X_ERROR``, ``Z_ERROR`` or noisy ``M``, and whose
        detectors and observables are deterministic without noise.

    Returns
    -------
    list of Fault
        One entry per fault with an effect, ordered by instruction.

    Raises
    ------
    ValueError
        If ``circuit`` has a ``REPEAT`` block or another noise channel.
    """
    for index in range(len(circuit)):
        instruction = circuit[index]
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("circuit must not contain REPEAT blocks")
        noisy = stim.gate_data(instruction.name).is_noisy_gate
        if (
            noisy
            and instruction.gate_args_copy()
            and (instruction.name not in _OUTCOMES)
        ):
            raise ValueError(
                f"unsupported noise channel {instruction.name!r} in circuit"
            )
    found = []
    for error in circuit.explain_detector_error_model_errors(
        reduce_to_one_representative_error=False
    ):
        targets = [term.dem_target for term in error.dem_error_terms]
        logical = any(target.is_logical_observable_id() for target in targets)
        detected = any(target.is_relative_detector_id() for target in targets)
        for location in error.circuit_error_locations:
            index = location.stack_frames[0].instruction_offset
            instruction = circuit[index]
            assert isinstance(instruction, stim.CircuitInstruction)
            rate = instruction.gate_args_copy()[0]
            found.append(
                Fault(
                    tag=location.noise_tag,
                    instruction=index,
                    probability=rate / _OUTCOMES[instruction.name],
                    logical=logical,
                    detected=detected,
                )
            )
    found.sort(key=lambda fault: fault.instruction)
    return found


def logical_error_rate(found: list[Fault]) -> float:
    """Return the first-order logical error rate of a list of faults.

    Parameters
    ----------
    found : list of Fault
        Faults, as :func:`faults` returns them, or any subset of them, such
        as those of one region.

    Returns
    -------
    float
        The total probability of the faults that flip an observable and no
        detector: the logical error rate to first order in the error rates,
        accepted or not, since the two differ only at second order.
    """
    return sum(
        fault.probability
        for fault in found
        if fault.logical and not fault.detected
    )
