"""Type checks shared by the public functions.

Each raises ``TypeError`` naming the argument and the type received, so a
caller's mistake surfaces at the public boundary rather than as an
unrelated error deeper in the package or in Stim.
"""

from __future__ import annotations

import numbers

import stim

from .noise_models import NoiseModel


def require_int(value: object, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is an integer, and not a bool.

    NumPy integers are accepted.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        )


def require_real(value: object, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a real number, not a bool."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")


def require_circuit(value: object, name: str = "circuit") -> None:
    """Raise ``TypeError`` unless ``value`` is a ``stim.Circuit``."""
    if not isinstance(value, stim.Circuit):
        raise TypeError(
            f"{name} must be a stim.Circuit, got {type(value).__name__}"
        )


def require_noise(value: object) -> None:
    """Raise ``TypeError`` unless ``value`` is a ``NoiseModel``."""
    if not isinstance(value, NoiseModel):
        raise TypeError(
            f"noise must be a NoiseModel, got {type(value).__name__}"
        )


def require_checks(value: object, length: int) -> None:
    """Raise unless ``value`` is a non-empty list of lists of Pauli strings.

    ``TypeError`` for the wrong structure, ``ValueError`` for no chunks;
    ``length`` is the length each check must have, named in the message
    of the caller's own length check.
    """
    if not isinstance(value, list) or not all(
        isinstance(chunk, list) for chunk in value
    ):
        raise TypeError(
            "checks must be a list with one list of checks per chunk"
        )
    if not value:
        raise ValueError("checks must hold one list per chunk, got none")
    for chunk in value:
        for check in chunk:
            if not isinstance(check, stim.PauliString):
                raise TypeError(
                    "each check must be a stim.PauliString of length "
                    f"{length}, got {type(check).__name__}"
                )
