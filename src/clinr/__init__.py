"""Clifford Noise Reduction (CliNR) in Stim.

The scheme of Delfosse and Tham, arXiv:2407.06583, implemented as six
modules: ``protocol`` builds the circuits, ``accounting`` counts the
operations they contain, ``bounds`` turns those counts into deterministic
analytic bounds, ``estimators`` measures the same quantities by Monte Carlo,
``circuits`` supplies random Clifford inputs, and ``noise_models`` holds the
four rates.

Modules are imported as modules and types by name, so ``clinr.bounds`` and
``clinr.protocol`` are the intended spellings rather than a flat namespace of
re-exported functions.
"""

from . import (
    accounting,
    bounds,
    circuits,
    estimators,
    noise_models,
    protocol,
)

__all__ = [
    "protocol",
    "accounting",
    "bounds",
    "estimators",
    "circuits",
    "noise_models",
]
