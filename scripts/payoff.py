"""Measure when CliNR pays off, under the paper's rule for ``(t, r)``.

Two parts. The curves: for random Clifford circuits of each size, under
three noise regimes, the logical error rate of the direct implementation
and of CliNR at a budget of 2 and of 4 on the gate overhead. Under each
regime the sizes run in steps, coarser beyond a set size, until the direct
implementation's bound, which is nearly exact, reaches a set logical error
rate on the median circuit. The maps: the same comparison at both budgets
for random gate sequences of ``n^alpha`` gates, over a grid of ``n`` and
``alpha``, under two regimes.

CliNR's parameters follow the paper's rule (arXiv:2407.06583, Fig. 2):
``r = floor(log2(s / n))`` checks, at least one, and the smallest ``t``
whose overhead is within the budget. The overhead of each ``t`` is judged
by a small pilot estimate, and the chosen pair is then estimated afresh.

Run from the repository root::

    uv run python scripts/payoff.py

The results go to ``results/payoff.json.gz``, compressed since the largest
circuits run to tens of thousands of gates, with the settings, the seed,
the versions and platform that produced them, the circuits, and the counts
of every draw. The run saves as it goes and resumes from the file.
``--quick`` runs a small configuration, writing
``results/payoff-quick.json.gz`` unless ``--output`` says otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import gzip
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import regret
import stim

from clinr import accounting, bounds, circuits, estimators
from clinr.noise_models import NoiseModel, asymmetric_noise

REGIMES: tuple[tuple[str, NoiseModel], ...] = (
    ("paper", asymmetric_noise(1e-3)),
    ("paper-1e-4", asymmetric_noise(1e-4)),
    ("forte-like", NoiseModel(p1=2e-4, p2=4e-3, p_m=4.8e-3, p_prep=2e-4)),
)

# Phases of the run, which separate its random streams.
CIRCUITS, DIRECT, PILOT, FINAL = 0, 1, 2, 3

SEEDING = (
    "each generator is seeded by numpy.random.SeedSequence("
    "config.seed, spawn_key=key), where key is (0, part, n, "
    "alpha index, circuit index) for a gate sequence of a "
    "map, (1, part, regime index, n, alpha index, circuit "
    "index) for the direct implementation, and (part, "
    "regime index, n, alpha index, circuit index, family "
    "index) followed by (2, t) for a pilot and (3, t) for a "
    "final estimate; part is 0 for the curves and 1 for the "
    "maps, and the alpha index is 0 on the curves"
)


@dataclasses.dataclass(frozen=True)
class Config:
    """The settings of a run, saved with its results."""

    regimes: tuple[tuple[str, NoiseModel], ...]
    curve_regimes: tuple[str, ...]
    first_size: int
    size_step: int
    coarse_after: int
    coarse_step: int
    stop_direct: float
    max_size: int
    curve_circuits: int
    curve_budgets: tuple[float, ...]
    map_sizes: tuple[tuple[str, tuple[int, ...]], ...]
    alphas: tuple[float, ...]
    map_circuits: int
    map_budgets: tuple[float, ...]
    t_max: int
    pilot_draws: int
    pilot_shots: int
    draws: int
    shots: int
    direct_shots: int
    seed: int


FULL = Config(
    regimes=REGIMES,
    curve_regimes=("paper", "paper-1e-4", "forte-like"),
    first_size=5,
    size_step=5,
    coarse_after=80,
    coarse_step=10,
    stop_direct=0.9,
    max_size=200,
    curve_circuits=10,
    curve_budgets=(2.0, 4.0),
    map_sizes=(
        ("paper-1e-4", tuple(range(5, 56, 5))),
        ("forte-like", tuple(range(5, 31, 5))),
    ),
    alphas=(1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0),
    map_circuits=10,
    map_budgets=(2.0, 4.0),
    t_max=12,
    pilot_draws=3,
    pilot_shots=1_000,
    draws=10,
    shots=10_000,
    direct_shots=100_000,
    seed=20261010,
)
QUICK = Config(
    regimes=REGIMES,
    curve_regimes=("paper",),
    first_size=3,
    size_step=1,
    coarse_after=3,
    coarse_step=1,
    stop_direct=0.0,
    max_size=3,
    curve_circuits=1,
    curve_budgets=(2.0, 4.0),
    map_sizes=(("forte-like", (3,)),),
    alphas=(1.6, 2.0),
    map_circuits=1,
    map_budgets=(4.0,),
    t_max=2,
    pilot_draws=2,
    pilot_shots=200,
    draws=2,
    shots=500,
    direct_shots=2_000,
    seed=1,
)


def rule_r(size: int, n: int) -> int:
    """Return the paper's number of checks, ``floor(log2(size / n))``.

    The result is at least 1, since the paper takes ``r >= 1``.
    """
    return max(regret.paper_r(size, n), 1)


def run_unit(
    text: str,
    n: int,
    noise: dict[str, float],
    restricted: bool,
    budgets: tuple[float, ...],
    settings: dict[str, int],
    spawn_key: tuple[int, ...],
) -> dict[str, Any]:
    """Choose ``t`` by the paper's rule for each budget, and estimate it.

    The pilot estimates the overhead of ``t = 1, 2, ...`` in turn, each
    ``t`` at most once, until one is within the budget or ``t`` passes
    ``t_max`` or the gate count. Each chosen ``t`` is then estimated with
    fresh draws.
    """
    size = sum(accounting.gate_counts(stim.Circuit(text)))
    r = rule_r(size, n)
    seed = settings["seed"]
    pilots: dict[int, list[float]] = {}
    choice: dict[str, int | None] = {}
    for budget in budgets:
        choice[str(budget)] = None
        for t in range(1, min(settings["t_max"], size) + 1):
            if t not in pilots:
                record = regret.run_point(
                    text,
                    n,
                    t,
                    r,
                    noise,
                    restricted,
                    settings["pilot_draws"],
                    settings["pilot_shots"],
                    np.random.SeedSequence(
                        seed, spawn_key=(*spawn_key, PILOT, t)
                    ),
                )
                rate, overhead = regret.measured(record, n, restricted)
                pilots[t] = [rate, overhead]
            if pilots[t][1] <= budget:
                choice[str(budget)] = t
                break
    estimates = {}
    for t in sorted({t for t in choice.values() if t is not None}):
        estimates[str(t)] = regret.run_point(
            text,
            n,
            t,
            r,
            noise,
            restricted,
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


def run_direct(
    text: str,
    n: int,
    noise: dict[str, float],
    shots: int,
    seed: np.random.SeedSequence,
) -> dict[str, int]:
    """Estimate the direct implementation of one circuit."""
    estimate = estimators.estimate_direct(
        stim.Circuit(text),
        n,
        NoiseModel(**noise),
        shots,
        np.random.default_rng(seed),
    )
    return {"shots": estimate.shots, "errors": estimate.errors}


class Results:
    """The results file of a run, loaded, extended and saved."""

    def __init__(
        self,
        path: Path,
        config: Any,
        *,
        experiment: str = "payoff",
        seeding: str = SEEDING,
    ) -> None:
        self.path = path
        self.saved = -math.inf
        settings = json.loads(json.dumps(dataclasses.asdict(config)))
        if path.exists():
            self.data = load(path)
            if self.data["config"] != settings:
                raise SystemExit(
                    f"{path} was made with other settings; "
                    "move it or pass --output"
                )
        else:
            self.data = {
                "experiment": experiment,
                "config": settings,
                "seeding": seeding,
                "sizes": {},
                "circuits": {},
                "direct": {},
                "units": {},
                "runs": [],
            }
        self.data["runs"].append(
            {
                "started": datetime.datetime.now(datetime.UTC).isoformat(),
                "provenance": regret.provenance(),
            }
        )
        self.save()

    def save(self, *, unless_recent: bool = False) -> None:
        """Write the file atomically, at most once a minute if asked."""
        now = time.monotonic()
        if unless_recent and now - self.saved < 60:
            return
        self.saved = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        text = json.dumps(self.data).encode()
        if self.path.suffix == ".gz":
            text = gzip.compress(text, mtime=0)
        temporary.write_bytes(text)
        temporary.replace(self.path)


def load(path: Path) -> dict[str, Any]:
    """Read a results file, compressed with gzip if its name ends in .gz."""
    text = path.read_bytes()
    if path.suffix == ".gz":
        text = gzip.decompress(text)
    data: dict[str, Any] = json.loads(text)
    return data


def cases(
    config: Config, sizes: dict[str, list[int]]
) -> Iterator[tuple[int, str, int, int, float, int]]:
    """Yield ``(part, regime, n, alpha index, alpha, circuits)``.

    ``sizes`` holds the sizes of the curves under each regime. The curves
    have ``alpha`` 0, meaning random Clifford circuits.
    """
    for regime in config.curve_regimes:
        for n in sizes[regime]:
            yield 0, regime, n, 0, 0.0, config.curve_circuits
    for regime, map_sizes in config.map_sizes:
        for n in map_sizes:
            for a, alpha in enumerate(config.alphas):
                yield 1, regime, n, a, alpha, config.map_circuits


def circuit_key(part: int, n: int, alpha: float, c: int) -> str:
    """Return the key of a circuit, shared by every regime."""
    if part == 0:
        return regret._key("clifford", n, c)
    return regret._key("sequence", n, alpha, c)


def run(config: Config, path: Path, workers: int) -> None:
    """Run the experiment, resuming from ``path`` if it exists."""
    state = Results(path, config)
    data = state.data
    regimes = dict(config.regimes)
    regime_index = {name: i for i, (name, _) in enumerate(config.regimes)}
    family_index = {name: j for j, name in enumerate(regret.FAMILIES)}

    # The random Cliffords of the curves, generated once and saved, since
    # Stim's sampler takes no seed, and shared by every regime. Each regime
    # takes sizes until its direct bound on the median circuit reaches
    # stop_direct.
    for regime in config.curve_regimes:
        if regime in data["sizes"]:
            continue
        sizes = []
        n = config.first_size
        while True:
            texts = []
            for c in range(config.curve_circuits):
                key = circuit_key(0, n, 0.0, c)
                if key not in data["circuits"]:
                    data["circuits"][key] = str(circuits.random_clifford(n))
                texts.append(data["circuits"][key])
            sizes.append(n)
            direct = np.median(
                [
                    bounds.direct_plog_bound(
                        stim.Circuit(text), regimes[regime]
                    )
                    for text in texts
                ]
            )
            if direct >= config.stop_direct or n >= config.max_size:
                break
            if n >= config.coarse_after:
                n += config.coarse_step
            else:
                n += config.size_step
        data["sizes"][regime] = sizes
    state.save()

    # Seeded gate sequences of n^alpha gates for the maps.
    for part, _, n, a, alpha, count in cases(config, data["sizes"]):
        for c in range(count):
            key = circuit_key(part, n, alpha, c)
            if part == 0 or key in data["circuits"]:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    config.seed, spawn_key=(CIRCUITS, part, n, a, c)
                )
            )
            circuit = circuits.random_gate_sequence(n, round(n**alpha), rng)
            data["circuits"][key] = str(circuit)
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
    for part, regime, n, a, alpha, count in cases(config, data["sizes"]):
        i = regime_index[regime]
        noise = dataclasses.asdict(regimes[regime])
        budgets = config.curve_budgets if part == 0 else config.map_budgets
        for c in range(count):
            text = data["circuits"][circuit_key(part, n, alpha, c)]
            case = regret._key(
                "curves" if part == 0 else "maps", regime, n, alpha, c
            )
            if case not in data["direct"]:
                seed = np.random.SeedSequence(
                    config.seed, spawn_key=(DIRECT, part, i, n, a, c)
                )
                jobs.append(
                    (
                        "direct",
                        case,
                        run_direct,
                        (text, n, noise, config.direct_shots, seed),
                    )
                )
            for family, restricted in regret.FAMILIES.items():
                key = regret._key(case, family)
                if key in data["units"]:
                    continue
                spawn_key = (part, i, n, a, c, family_index[family])
                jobs.append(
                    (
                        "units",
                        key,
                        run_unit,
                        (
                            text,
                            n,
                            noise,
                            restricted,
                            budgets,
                            settings,
                            spawn_key,
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
            for future in as_completed(futures):
                yield (*futures[future], future.result())

        for done, (table, key, result) in enumerate(finished(), 1):
            data[table][key] = result
            state.save(unless_recent=True)
            print(f"{done}/{len(jobs)} {table} {key}", flush=True)

    data["runs"][-1]["finished"] = datetime.datetime.now(
        datetime.UTC
    ).isoformat()
    state.save()


def outcomes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the outcome of every circuit, family and budget.

    Parameters
    ----------
    data : dict
        A finished results file, as loaded from JSON.

    Returns
    -------
    list of dict
        One record per circuit, family and budget, with keys ``part``
        (``"curves"`` or ``"maps"``), ``regime``, ``n``, ``alpha`` (0 for
        the random Cliffords of the curves), ``circuit``, ``family``,
        ``budget``, ``s``, ``r`` and ``t`` (``None`` when no ``t`` up to
        ``t_max`` is within the budget); ``direct``, the direct
        implementation's logical error rate; and ``rate`` and
        ``overhead``, CliNR's at the chosen ``t`` from the fresh estimate,
        ``nan`` when there is none.
    """
    rows = []
    for key, unit in data["units"].items():
        part, regime, size, alpha, c, family = key.split("/")
        case = regret._key(part, regime, size, alpha, c)
        direct = data["direct"][case]
        n = int(size)
        restricted = regret.FAMILIES[family]
        for budget, t in unit["choice"].items():
            rate = overhead = math.nan
            if t is not None:
                record = unit["estimates"][str(t)]
                rate, overhead = regret.measured(record, n, restricted)
            rows.append(
                {
                    "part": part,
                    "regime": regime,
                    "n": n,
                    "alpha": float(alpha),
                    "circuit": int(c),
                    "family": family,
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
    name = "payoff-quick.json.gz" if args.quick else "payoff.json.gz"
    run(config, args.output or Path("results") / name, args.workers)


if __name__ == "__main__":
    main(sys.argv[1:])
