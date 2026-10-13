"""Measure when CZNR pays off on dense blocks of CZ gates.

The circuits are complete graphs: one ``CZ`` on every pair of ``n`` qubits,
``s = n(n - 1)/2`` gates, as in the paper's Fig. 7 (arXiv:2407.06583,
App. E). Under three noise regimes, the script estimates the logical error
rate of the direct implementation and of three schemes at a budget of 2 and
of 4 on the gate overhead: CZNR, with checks drawn uniformly from the graph
state's group, and CliNR, with uniform and with restricted checks. CZNR
runs twice, with ``|+>`` prepared as a reset and a Hadamard, as everywhere
else in the package, and with one native preparation, as the paper counts
it, since at the paper's rates the budget of 2 falls between the two.
Under each regime the sizes run in steps, coarser beyond a set size, until the
direct implementation's bound, which is nearly exact, reaches a set logical
error rate.

Each scheme follows the paper's rule (Fig. 2 and Fig. 7):
``r = floor(log2(s / n))`` checks, at least one, and the smallest ``t``
whose overhead is within the budget. The overhead of each ``t`` is judged
by a small pilot estimate, and the chosen pair is then estimated afresh. A
budget that no ``t`` meets costs only its pilots.

Run from the repository root::

    uv run python scripts/dense.py

The results go to ``results/dense.json.gz`` with the settings, the seed,
the versions and platform that produced them, the circuits, and the counts
of every draw. The run saves as it goes and resumes from the file.
``--quick`` runs a small configuration, writing
``results/dense-quick.json.gz`` unless ``--output`` says otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import itertools
import math
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import payoff
import regret
import stim

from clinr import bounds, estimators
from clinr.noise_models import NoiseModel

#: The schemes compared with the direct implementation: whether each is
#: CZNR, whether its checks are restricted, and whether it prepares ``|+>``
#: natively.
ARMS: dict[str, tuple[bool, bool, bool]] = {
    "cznr": (True, False, False),
    "cznr-native": (True, False, True),
    "uniform": (False, False, False),
    "restricted": (False, True, False),
}

# Phases of the run, which separate its random streams.
DIRECT, PILOT, FINAL = 1, 2, 3

SEEDING = (
    "each generator is seeded by numpy.random.SeedSequence(config.seed, "
    "spawn_key=key), where key is (1, regime index, n) for the direct "
    "implementation, and (regime index, n, arm index) followed by (2, t) "
    "for a pilot and (3, t) for a final estimate"
)


@dataclasses.dataclass(frozen=True)
class Config:
    """The settings of a run, saved with its results."""

    regimes: tuple[tuple[str, NoiseModel], ...]
    first_size: int
    size_step: int
    coarse_after: int
    coarse_step: int
    stop_direct: float
    max_size: int
    budgets: tuple[float, ...]
    arms: tuple[str, ...]
    t_max: int
    pilot_draws: int
    pilot_shots: int
    draws: int
    shots: int
    direct_shots: int
    seed: int


FULL = Config(
    regimes=payoff.REGIMES,
    first_size=5,
    size_step=5,
    coarse_after=80,
    coarse_step=10,
    stop_direct=0.9,
    max_size=250,
    budgets=(2.0, 4.0),
    arms=tuple(ARMS),
    t_max=12,
    pilot_draws=10,
    pilot_shots=1_000,
    draws=10,
    shots=10_000,
    direct_shots=100_000,
    seed=20261011,
)
QUICK = Config(
    regimes=payoff.REGIMES[:1],
    first_size=3,
    size_step=1,
    coarse_after=3,
    coarse_step=1,
    stop_direct=0.0,
    max_size=4,
    budgets=(2.0, 4.0),
    arms=tuple(ARMS),
    t_max=2,
    pilot_draws=2,
    pilot_shots=200,
    draws=2,
    shots=500,
    direct_shots=2_000,
    seed=1,
)


def complete_graph(n: int) -> stim.Circuit:
    """Return one ``CZ`` on every pair of ``n`` qubits, in lexical order."""
    circuit = stim.Circuit()
    for a, b in itertools.combinations(range(n), 2):
        circuit.append("CZ", [a, b])
    return circuit


def run_point(
    text: str,
    n: int,
    t: int,
    r: int,
    noise: dict[str, float],
    arm: str,
    draws: int,
    shots: int,
    seed: np.random.SeedSequence,
) -> dict[str, Any]:
    """Estimate one pair ``(t, r)`` of an arm over ``draws`` draws."""
    circuit = stim.Circuit(text)
    rng = np.random.default_rng(seed)
    cznr, restricted, native_plus = ARMS[arm]
    model = NoiseModel(**noise)
    estimates = [
        estimators.estimate_cznr(
            circuit, n, t, r, model, shots, rng, native_plus=native_plus
        )
        if cznr
        else estimators.estimate_clinr(
            circuit,
            n,
            t,
            r,
            model,
            shots,
            rng,
            restricted_checks=restricted,
            native_plus=native_plus,
        )
        for _ in range(draws)
    ]
    first = estimates[0]
    return {
        "shots": shots,
        "s1": first.s1,
        "s2": first.s2,
        "chunk_counts": [list(pair) for pair in first.chunk_counts],
        "chunk_checks": list(first.chunk_checks),
        # One entry per draw, in the order drawn.
        "accepted": [estimate.accepted for estimate in estimates],
        "errors": [estimate.errors for estimate in estimates],
        "chunk_accepted": [list(e.chunk_accepted) for e in estimates],
        "chunk_weight": [list(e.chunk_weight) for e in estimates],
    }


def estimates_of(
    record: dict[str, Any], n: int, arm: str
) -> list[estimators.Estimate]:
    """Rebuild the estimates of one point of an arm from its record."""
    cznr, restricted, native_plus = ARMS[arm]
    return [
        estimators.Estimate(
            record["shots"],
            accepted,
            errors,
            record["s1"],
            record["s2"],
            n,
            tuple(chunk_accepted),
            tuple(record["chunk_checks"]),
            tuple(chunk_weight),
            tuple(tuple(pair) for pair in record["chunk_counts"]),
            restricted_checks=restricted,
            native_plus=native_plus,
            cznr=cznr,
        )
        for accepted, errors, chunk_accepted, chunk_weight in zip(
            record["accepted"],
            record["errors"],
            record["chunk_accepted"],
            record["chunk_weight"],
            strict=True,
        )
    ]


def measured(record: dict[str, Any], n: int, arm: str) -> tuple[float, float]:
    """Return the fresh-check logical error rate and overhead of a point.

    Both are ``nan`` when the draws accepted too few shots to estimate them.
    """
    draws = estimates_of(record, n, arm)
    try:
        rate, _, _ = estimators.logical_error_rate(draws)
        overhead, _, _ = estimators.gate_overhead_over_draws(draws)
    except ValueError:
        return float("nan"), float("nan")
    return rate, overhead


def run_unit(
    text: str,
    n: int,
    noise: dict[str, float],
    arm: str,
    budgets: tuple[float, ...],
    settings: dict[str, int],
    spawn_key: tuple[int, ...],
) -> dict[str, Any]:
    """Choose ``t`` by the paper's rule for each budget, and estimate it.

    As :func:`payoff.run_unit`, for one arm.
    """
    size = n * (n - 1) // 2
    r = payoff.rule_r(size, n)
    seed = settings["seed"]
    pilots: dict[int, list[float]] = {}
    choice: dict[str, int | None] = {}
    for budget in budgets:
        choice[str(budget)] = None
        for t in range(1, min(settings["t_max"], size) + 1):
            if t not in pilots:
                record = run_point(
                    text,
                    n,
                    t,
                    r,
                    noise,
                    arm,
                    settings["pilot_draws"],
                    settings["pilot_shots"],
                    np.random.SeedSequence(
                        seed, spawn_key=(*spawn_key, PILOT, t)
                    ),
                )
                pilots[t] = list(measured(record, n, arm))
            if pilots[t][1] <= budget:
                choice[str(budget)] = t
                break
    estimates = {}
    for t in sorted({t for t in choice.values() if t is not None}):
        estimates[str(t)] = run_point(
            text,
            n,
            t,
            r,
            noise,
            arm,
            settings["draws"],
            settings["shots"],
            np.random.SeedSequence(seed, spawn_key=(*spawn_key, FINAL, t)),
        )
    return {
        "s": size,
        "r": r,
        "pilots": {str(t): value for t, value in pilots.items()},
        "choice": choice,
        "estimates": estimates,
    }


def sizes_of(config: Config, noise: NoiseModel) -> list[int]:
    """Return the sizes of a regime, by the stopping rule."""
    sizes = []
    n = config.first_size
    while True:
        sizes.append(n)
        direct = bounds.direct_plog_bound(complete_graph(n), noise)
        if direct >= config.stop_direct or n >= config.max_size:
            return sizes
        n += (
            config.coarse_step
            if n >= config.coarse_after
            else config.size_step
        )


def run(config: Config, path: Path, workers: int) -> None:
    """Run the experiment, resuming from ``path`` if it exists."""
    state = payoff.Results(path, config, experiment="dense", seeding=SEEDING)
    data = state.data
    for regime, model in config.regimes:
        if regime not in data["sizes"]:
            data["sizes"][regime] = sizes_of(config, model)
        for n in data["sizes"][regime]:
            data["circuits"].setdefault(str(n), str(complete_graph(n)))
    state.save()

    settings = {
        "seed": config.seed,
        "t_max": config.t_max,
        "pilot_draws": config.pilot_draws,
        "pilot_shots": config.pilot_shots,
        "draws": config.draws,
        "shots": config.shots,
    }
    jobs: list[tuple[str, str, Any, tuple[Any, ...]]] = []
    for i, (regime, model) in enumerate(config.regimes):
        noise = dataclasses.asdict(model)
        for n in data["sizes"][regime]:
            text = data["circuits"][str(n)]
            case = regret.record_key(regime, n)
            if case not in data["direct"]:
                seed = np.random.SeedSequence(
                    config.seed, spawn_key=(DIRECT, i, n)
                )
                jobs.append(
                    (
                        "direct",
                        case,
                        payoff.run_direct,
                        (text, n, noise, config.direct_shots, seed),
                    )
                )
            for j, arm in enumerate(config.arms):
                key = regret.record_key(case, arm)
                if key in data["units"]:
                    continue
                jobs.append(
                    (
                        "units",
                        key,
                        run_unit,
                        (
                            text,
                            n,
                            noise,
                            arm,
                            config.budgets,
                            settings,
                            (i, n, j),
                        ),
                    )
                )

    with contextlib.ExitStack() as stack:
        pool = (
            stack.enter_context(ProcessPoolExecutor(max_workers=workers))
            if workers > 1
            else None
        )

        def finished() -> Iterator[tuple[str, str, Any]]:
            if pool is None:
                for table, key, function, arguments in jobs:
                    yield table, key, function(*arguments)
                return
            futures = {
                pool.submit(function, *arguments): (table, key)
                for table, key, function, arguments in jobs
            }
            try:
                for future in as_completed(futures):
                    yield (*futures[future], future.result())
            finally:
                # On a failure, drop the queued jobs rather than run them
                # only to discard their results.
                for future in futures:
                    future.cancel()

        try:
            for done, (table, key, result) in enumerate(finished(), 1):
                data[table][key] = result
                state.save(unless_recent=True)
                print(f"{done}/{len(jobs)} {table} {key}", flush=True)
        finally:
            state.save()  # keep what finished before any failure

    data["runs"][-1]["finished"] = datetime.datetime.now(
        datetime.UTC
    ).isoformat()
    state.save()


def outcomes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the outcome of every regime, size, arm and budget.

    Parameters
    ----------
    data : dict
        A finished results file, as loaded from JSON.

    Returns
    -------
    list of dict
        One record per regime, size, arm and budget, with keys ``regime``,
        ``n``, ``arm``, ``budget``, ``s``, ``r`` and ``t`` (``None`` when no
        ``t`` up to ``t_max`` is within the budget); ``direct``, the direct
        implementation's logical error rate; and ``rate`` and ``overhead``,
        the arm's at the chosen ``t`` from the fresh estimate, ``nan`` when
        there is none.
    """
    rows = []
    for key, unit in data["units"].items():
        regime, size, arm = key.split("/")
        direct = data["direct"][regret.record_key(regime, size)]
        n = int(size)
        for budget, t in unit["choice"].items():
            rate = overhead = math.nan
            if t is not None:
                rate, overhead = measured(unit["estimates"][str(t)], n, arm)
            rows.append(
                {
                    "regime": regime,
                    "n": n,
                    "arm": arm,
                    "budget": float(budget),
                    "s": unit["s"],
                    "r": unit["r"],
                    "t": t,
                    "direct": direct["errors"] / direct["shots"],
                    "rate": rate,
                    "overhead": overhead,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="run a small configuration"
    )
    parser.add_argument("--output", type=Path, help="the results file")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes to run in parallel",
    )
    args = parser.parse_args(argv)
    config = QUICK if args.quick else FULL
    name = "dense-quick.json.gz" if args.quick else "dense.json.gz"
    run(config, args.output or Path("results") / name, args.workers)


if __name__ == "__main__":
    main(sys.argv[1:])
