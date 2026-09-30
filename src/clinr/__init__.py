"""Clifford Noise Reduction (CliNR) in Stim.

An implementation of the scheme of [1]_ in seven modules: ``protocol``
builds the CliNR circuits and the direct baseline, ``checks`` draws the
stabilizer checks they measure, ``accounting`` counts the operations of the
built circuits, ``bounds`` turns those counts into analytic bounds on the
logical error rate and the gate overhead, ``estimators`` measures the same
quantities by Monte Carlo, ``circuits`` supplies random Clifford circuits,
and ``noise_models`` holds the error rates.

The modules are the namespace: ``clinr.bounds.plog_bound``, not a
re-export from ``clinr``.

References
----------
.. [1] N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford
   circuits", arXiv:2407.06583 (2024).
"""

from . import (
    accounting,
    bounds,
    checks,
    circuits,
    estimators,
    noise_models,
    protocol,
)

__all__ = [
    "protocol",
    "checks",
    "accounting",
    "bounds",
    "estimators",
    "circuits",
    "noise_models",
]
