"""Monte-Carlo estimators for CliNR circuits (evaluation layer).

Builds on the ``clinr`` construction layer to estimate the quantities that
characterize CliNR performance:

  * :func:`p_logical` -- the logical error rate P(logical error | accepted),
  * :func:`catch_rate` -- P(a planted resource fault is detected).

Both are pooled over independent draws of the random check set. A single
``stim.Circuit`` hard-codes one check set, but the protocol draws a fresh set on
every run, so an honest estimate averages over check sets as well as over noise.
Pooling counts (a ratio of summed counts, not a mean of per-draw ratios) is the
correct way to combine draws when acceptance itself depends on the draw.

Each estimator returns an :class:`Estimate` carrying the value, its standard
error, and the sample sizes behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

import clinr

__all__ = ["Estimate", "p_logical", "catch_rate"]


@dataclass(frozen=True)
class Estimate:
    """A Monte-Carlo estimate with its standard error and sample sizes.

    Attributes:
        value: The estimated probability.
        stderr: Binomial standard error ``sqrt(value*(1-value)/n_used)``.
        n_used: Number of shots the estimate is computed from (e.g. accepted
            shots, for a postselected quantity).
        n_total: Total shots drawn (``n_used <= n_total``).
    """

    value: float
    stderr: float
    n_used: int
    n_total: int

    @property
    def acceptance(self) -> float:
        """Fraction of drawn shots that were used (e.g. accepted)."""
        return self.n_used / self.n_total if self.n_total else float("nan")

    def __str__(self) -> str:
        return f"{self.value:.5f} +/- {self.stderr:.5f} (n={self.n_used})"


def _binom_stderr(value: float, n: int) -> float:
    """Standard error of a Bernoulli proportion; 0 when no samples."""
    return (value * (1.0 - value) / n) ** 0.5 if n > 0 else 0.0


def p_logical(
    circuit: stim.Circuit,
    n: int,
    r: int,
    p1: float,
    p2: float,
    p_m: float,
    *,
    n_draws: int = 16,
    shots: int = 20_000,
    seed: int = 0,
    noisy_checks: bool = True,
) -> Estimate:
    """Estimate the CliNR logical error rate ``P(logical error | accepted)``.

    For each of ``n_draws`` independent check sets, builds ``CliNR_{1,r}`` and
    samples ``shots`` times. A shot is *accepted* if no detector fired (the
    restart condition) and is a *logical error* if any Choi observable flipped.
    The estimate is the pooled ratio ``sum(errors) / sum(accepted)`` over all
    draws.

    Args:
        circuit: The Clifford circuit ``C`` to implement.
        n: Number of data qubits.
        r: Number of stabilizer checks.
        p1, p2, p_m: Single-qubit, two-qubit, and measurement error rates.
        n_draws: Number of independent random check sets to average over.
        shots: Shots per check set.
        seed: Base seed; independent substreams are spawned per draw.
        noisy_checks: Passed through to :func:`clinr.build_clinr_1r`.

    Returns:
        An :class:`Estimate` whose ``value`` is the logical error rate,
        ``n_used`` the total accepted shots, and ``acceptance`` the accept rate.
    """
    seeds = np.random.SeedSequence(seed).spawn(n_draws)
    total_accept = 0
    total_error = 0
    for sub in seeds:
        rng = np.random.default_rng(sub)
        circ = clinr.build_clinr_1r(
            circuit, n, r, p1, p2, p_m, rng, noisy_checks=noisy_checks
        )
        dets, obs = circ.compile_detector_sampler().sample(
            shots, separate_observables=True
        )
        accepted = ~dets.any(axis=1) if dets.shape[1] else np.ones(shots, bool)
        total_accept += int(accepted.sum())
        total_error += int(obs[accepted].any(axis=1).sum())

    value = total_error / total_accept if total_accept else float("nan")
    return Estimate(
        value=value,
        stderr=_binom_stderr(value, total_accept),
        n_used=total_accept,
        n_total=n_draws * shots,
    )


def catch_rate(
    circuit: stim.Circuit,
    n: int,
    r: int,
    *,
    n_trials: int = 20_000,
    seed: int = 0,
) -> Estimate:
    """Estimate ``P(a random resource fault is detected)`` for ``r`` checks.

    Tests the detection guarantee directly. For each trial we draw a random
    non-trivial Pauli fault on the resource (blocks 2 and 3) and an independent
    set of ``r`` random stabilizer checks, and record whether the fault
    anticommutes with at least one check (i.e. is detected). Detection is exact
    per trial -- it is a commutation question -- so no circuit sampling is
    needed. For ``r`` uniformly random checks the rate approaches
    ``1 - 2**(-r)``.

    Args:
        circuit: The Clifford circuit ``C`` (only its tableau matters here).
        n: Number of data qubits.
        r: Number of stabilizer checks (``r = 0`` detects nothing).
        n_trials: Number of independent (fault, check set) trials.
        seed: Base seed for the trials' random generator.

    Returns:
        An :class:`Estimate` whose ``value`` is the detection probability.
    """
    tableau = circuit.to_tableau()
    b1, b2, b3, ref, anc = clinr.blocks(n)
    total = 4 * n + 1
    resource_qubits = b2 + b3
    gens = clinr.stabilizer_gens(tableau, n, b2, b3, total)

    rng = np.random.default_rng(seed)
    detected = 0
    for _ in range(n_trials):
        fault = _random_pauli_on(resource_qubits, total, rng)
        checks = [clinr.random_stabilizer(gens, total, rng) for _ in range(r)]
        if any(not fault.commutes(chk) for chk in checks):
            detected += 1

    value = detected / n_trials
    return Estimate(
        value=value,
        stderr=_binom_stderr(value, n_trials),
        n_used=n_trials,
        n_total=n_trials,
    )


def _random_pauli_on(
    support: list[int], total: int, rng: np.random.Generator
) -> stim.PauliString:
    """A uniformly random non-identity Pauli supported within ``support``."""
    codes = np.zeros(total, dtype=np.uint8)
    while True:
        codes[support] = rng.integers(0, 4, size=len(support), dtype=np.uint8)
        if codes.any():
            break
    return stim.PauliString("".join("IXYZ"[c] for c in codes))