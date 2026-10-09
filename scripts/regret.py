"""Measure the regret of choosing ``(t, r)`` by the bound.

For each circuit, noise regime, check family and overhead ceiling, this
records three choices of implementation under the ceiling: the one
:func:`clinr.bounds.choose_tr` picks, a pair or the direct implementation;
the paper's rule, ``r = floor(log2(s / n))`` and the smallest ``t`` whose
overhead fits, by measurement and by the bound; and the best few that
measurement finds, the direct implementation among them. Every pair of a
grid is estimated over draws of the checks, and every chosen pair is then
estimated again with independent draws, as is the direct implementation,
so that the measured choice is not flattered by the minimum of noisy
estimates.

Run from the repository root::

    uv run python scripts/regret.py

The results go to ``results/regret.json`` with the settings, the seed, the
versions and platform that produced them, the circuits, and the counts of
every draw. The run saves after every point and resumes from the file, so
it can be stopped and started again. ``--quick`` runs a small grid to
check the script, writing ``results/regret-quick.json`` unless ``--output``
says otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import stim

from clinr import accounting, bounds, circuits, estimators
from clinr.noise_models import NoiseModel, asymmetric_noise

# The paper's numerics (arXiv:2407.06583, Sec. V) without idle noise, and a
# regime from IonQ's published Forte Enterprise figures: single-qubit gate
# error 2e-4, two-qubit gate error 4e-3, and SPAM error 5e-3, placed mostly
# on measurement, where published trapped-ion error budgets put most of it.
REGIMES: tuple[tuple[str, NoiseModel], ...] = (
    ("paper", asymmetric_noise(1e-3)),
    ("forte-like", NoiseModel(p1=2e-4, p2=4e-3, p_m=4.8e-3, p_prep=2e-4)),
)
FAMILIES: dict[str, bool] = {"uniform": False, "restricted": True}

# How a choice records the direct implementation.
DIRECT = "direct"


@dataclasses.dataclass(frozen=True)
class Config:
    """The settings of a run, saved with its results."""

    regimes: tuple[tuple[str, NoiseModel], ...]
    sizes: tuple[int, ...]
    circuits_per_size: int
    ts: tuple[int, ...]
    rs: tuple[int, ...]
    ceilings: tuple[float, ...]
    draws: int
    reestimate_draws: int
    top: int
    shots: int
    direct_shots: int
    seed: int


FULL = Config(
    regimes=REGIMES,
    sizes=(3, 8, 15, 25),
    circuits_per_size=10,
    ts=(1, 2, 3, 4, 5, 6),
    rs=(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24),
    ceilings=(2.0, 4.0),
    draws=20,
    reestimate_draws=80,
    top=3,
    shots=20_000,
    direct_shots=1_000_000,
    seed=20261007,
)
QUICK = Config(
    regimes=REGIMES,
    sizes=(2,),
    circuits_per_size=1,
    ts=(1, 2),
    rs=(0, 1, 2, 3),
    ceilings=(2.0, 4.0),
    draws=3,
    reestimate_draws=3,
    top=1,
    shots=500,
    direct_shots=2_000,
    seed=1,
)

# Phases of the run, which separate the random streams of the grid, the
# re-estimates, and the two estimates of the direct implementation.
GRID, REESTIMATE, DIRECT_FIRST, DIRECT_AGAIN = 0, 1, 2, 3


def _key(*parts: object) -> str:
    """Return the key of a record in the results file."""
    return "/".join(str(part) for part in parts)


def _seed(config: Config, *spawn_key: int) -> np.random.SeedSequence:
    """Return the seed of one point, spawned from the run's seed."""
    return np.random.SeedSequence(config.seed, spawn_key=spawn_key)


def paper_r(size: int, n: int) -> int:
    """Return the paper's number of checks, ``floor(log2(size / n))``.

    Arithmetic on integers, so exact; 0 when ``size < 2 n``.
    """
    return max((size // n).bit_length() - 1, 0)


def _git(*args: str) -> str | None:
    """Return the output of a git command in the repository, or None."""
    try:
        done = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return done.stdout.strip()


def _processor() -> str:
    """Return the processor's model name, as precisely as the OS gives it."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def provenance() -> dict[str, Any]:
    """Return the versions, commit and platform the run used."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": None if status is None else bool(status),
        "python": platform.python_version(),
        "packages": {
            name: metadata.version(name)
            for name in ("clifford-noise-reduction", "stim", "numpy")
        },
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": _processor(),
        "cpus": os.cpu_count(),
    }


def run_point(
    text: str,
    n: int,
    t: int,
    r: int,
    noise: dict[str, float],
    restricted: bool,
    draws: int,
    shots: int,
    seed: np.random.SeedSequence,
) -> dict[str, Any]:
    """Estimate one pair ``(t, r)`` over ``draws`` draws of the checks."""
    circuit = stim.Circuit(text)
    rng = np.random.default_rng(seed)
    estimates = [
        estimators.estimate_clinr(
            circuit,
            n,
            t,
            r,
            NoiseModel(**noise),
            shots,
            rng,
            restricted_checks=restricted,
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
    record: dict[str, Any], n: int, restricted: bool
) -> list[estimators.Estimate]:
    """Rebuild the estimates of one point from its record."""
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
        )
        for accepted, errors, chunk_accepted, chunk_weight in zip(
            record["accepted"],
            record["errors"],
            record["chunk_accepted"],
            record["chunk_weight"],
            strict=True,
        )
    ]


def measured(
    record: dict[str, Any], n: int, restricted: bool
) -> tuple[float, float]:
    """Return the fresh-check logical error rate and overhead of a point.

    Both are ``nan`` when the draws accepted too few shots to estimate them.
    """
    draws = estimates_of(record, n, restricted)
    try:
        rate, _, _ = estimators.logical_error_rate(draws)
        overhead, _, _ = estimators.gate_overhead_over_draws(draws)
    except ValueError:
        return float("nan"), float("nan")
    return rate, overhead


Pick = list[int] | str | None


def outcomes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the measured outcome of every choice in a results file.

    Each outcome is read from the fresh estimates (the re-estimates, and
    the second estimate of the direct implementation), never from the grid
    the measured choice was selected on.

    Parameters
    ----------
    data : dict
        A finished results file, as loaded from JSON.

    Returns
    -------
    list of dict
        One record per case and ceiling, with keys ``regime``, ``n``,
        ``circuit``, ``family`` and ``ceiling``; ``direct``, the rate of the
        direct implementation; and, for each choice, its pick and its rate:
        ``best`` (the measured choice), ``best_pair`` (the best pair by
        measurement), ``bound``, ``rule`` and ``rule_by_bound``. A pick is
        ``[t, r]``, ``"direct"``, or ``None`` when there is none, in which
        case its rate is ``nan``.
    """

    def rate(case: str, n: int, family: str, pick: Pick) -> float:
        if pick is None:
            return math.nan
        if pick == DIRECT:
            regime, _, c, _ = case.split("/")
            record = data["direct_reestimates"][_key(regime, n, c)]
            return float(record["errors"] / record["shots"])
        assert isinstance(pick, list)
        record = data["reestimates"][_key(case, *pick)]
        return measured(record, n, FAMILIES[family])[0]

    rows = []
    for key, choice in data["choices"].items():
        case, ceiling = key.rsplit("/", 1)
        regime, size, c, family = case.split("/")
        n = int(size)
        pairs = [pick for pick in choice["measured"] if pick != DIRECT]
        picks: dict[str, Pick] = {
            "best": choice["measured"][0],
            "best_pair": pairs[0] if pairs else None,
            "bound": choice["bound"],
            "rule": choice["rule"],
            "rule_by_bound": choice["rule_by_bound"],
        }
        row: dict[str, Any] = {
            "regime": regime,
            "n": n,
            "circuit": int(c),
            "family": family,
            "ceiling": float(ceiling),
            "direct": rate(case, n, family, DIRECT),
        }
        for name, pick in picks.items():
            row[f"{name}_pick"] = pick
            row[name] = rate(case, n, family, pick)
        rows.append(row)
    return rows


class Run:
    """The results file of a run, loaded, extended and saved."""

    def __init__(self, path: Path, config: Config) -> None:
        self.path = path
        self.config = config
        self.saved = -math.inf
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            saved = self.data["config"]
            if saved != json.loads(json.dumps(dataclasses.asdict(config))):
                raise SystemExit(
                    f"{path} was made with other settings; "
                    "move it or pass --output"
                )
        else:
            self.data = {
                "experiment": "regret",
                "config": dataclasses.asdict(config),
                "circuits": {},
                "direct": {},
                "direct_reestimates": {},
                "bounds": {},
                "grid": {},
                "choices": {},
                "reestimates": {},
                "seeding": (
                    "each point's generator is seeded by "
                    "numpy.random.SeedSequence(config.seed, spawn_key=(phase, "
                    "regime index, n, circuit index, family index, t, r)), "
                    "with phase 0 for the grid and 1 for the re-estimates, "
                    "and the direct implementation's by spawn_key=(phase, "
                    "regime index, n, circuit index), with phase 2 for the "
                    "first estimate and 3 for the re-estimate"
                ),
                "runs": [],
            }
        self.data["runs"].append(
            {
                "started": datetime.datetime.now(datetime.UTC).isoformat(),
                "provenance": provenance(),
            }
        )

    def save(self, *, unless_recent: bool = False) -> None:
        """Write the file atomically, so an interruption cannot corrupt it.

        With ``unless_recent``, skip the write if the last one was less than
        a minute ago.
        """
        now = time.monotonic()
        if unless_recent and now - self.saved < 60:
            return
        self.saved = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data), encoding="utf-8")
        temporary.replace(self.path)

    def circuits(self, n: int) -> list[str]:
        """Return the circuits of size ``n``, generating them once."""
        saved = self.data["circuits"]
        if str(n) not in saved:
            generated = circuits.random_cliffords(
                n, self.config.circuits_per_size
            )
            saved[str(n)] = [str(circuit) for circuit in generated]
            self.save()
        texts: list[str] = saved[str(n)]
        return texts


def cases(config: Config) -> Iterator[tuple[int, str, int, int, str, int]]:
    """Yield ``(regime index, regime, n, circuit, family, family index)``."""
    for i, (regime, _) in enumerate(config.regimes):
        for n in config.sizes:
            for c in range(config.circuits_per_size):
                for j, family in enumerate(FAMILIES):
                    yield i, regime, n, c, family, j


def run(config: Config, path: Path, workers: int) -> None:
    """Run the experiment, resuming from ``path`` if it exists."""
    state = Run(path, config)
    data = state.data
    regimes = dict(config.regimes)
    texts = {n: state.circuits(n) for n in config.sizes}
    rule_rs = {
        (n, c): paper_r(sum(accounting.gate_counts(stim.Circuit(text))), n)
        for n in config.sizes
        for c, text in enumerate(texts[n])
    }
    missing = sorted(set(rule_rs.values()) - set(config.rs))
    if missing:
        raise SystemExit(f"the paper's rule needs r in {missing}, not in rs")

    # The direct implementation, twice per regime and circuit.
    for i, (regime, noise) in enumerate(config.regimes):
        for n in config.sizes:
            for c, text in enumerate(texts[n]):
                key = _key(regime, n, c)
                for table, phase in (
                    ("direct", DIRECT_FIRST),
                    ("direct_reestimates", DIRECT_AGAIN),
                ):
                    if key in data[table]:
                        continue
                    rng = np.random.default_rng(_seed(config, phase, i, n, c))
                    direct = estimators.estimate_direct(
                        stim.Circuit(text), n, noise, config.direct_shots, rng
                    )
                    data[table][key] = {
                        "shots": direct.shots,
                        "errors": direct.errors,
                    }
    state.save()

    # The bounds at every grid pair, and the bound's choice per ceiling.
    for _, regime, n, c, family, _ in cases(config):
        key = _key(regime, n, c, family)
        if key in data["bounds"]:
            continue
        circuit, noise = stim.Circuit(texts[n][c]), regimes[regime]
        restricted = FAMILIES[family]
        grid = {
            _key(t, r): [
                bounds.plog_bound(
                    circuit, n, t, r, noise, restricted_checks=restricted
                ),
                bounds.overhead_bound(
                    circuit, n, t, r, noise, restricted_checks=restricted
                ),
            ]
            for t in config.ts
            for r in config.rs
        }
        choice: dict[str, list[int] | str] = {}
        for ceiling in config.ceilings:
            picked = bounds.choose_tr(
                circuit,
                n,
                noise,
                max_overhead=ceiling,
                r_max=max(config.rs),
                t_max=max(config.ts),
                restricted_checks=restricted,
            )
            choice[str(ceiling)] = DIRECT if picked is None else list(picked)
        data["bounds"][key] = {"grid": grid, "choice": choice}
        state.save()

    def submit_all(
        pool: ProcessPoolExecutor | None,
        jobs: list[tuple[str, str, dict[str, Any]]],
        draws: int,
    ) -> None:
        """Run ``jobs``, in ``pool`` if given, saving results as they come."""

        def arguments(args: dict[str, Any]) -> tuple[Any, ...]:
            return (
                texts[args["n"]][args["c"]],
                args["n"],
                args["t"],
                args["r"],
                dataclasses.asdict(regimes[args["regime"]]),
                FAMILIES[args["family"]],
                draws,
                config.shots,
                args["seed"],
            )

        def finished() -> Iterator[tuple[str, str, dict[str, Any]]]:
            if pool is None:
                for table, key, args in jobs:
                    yield table, key, run_point(*arguments(args))
                return
            futures = {
                pool.submit(run_point, *arguments(args)): (table, key)
                for table, key, args in jobs
            }
            for future in as_completed(futures):
                yield (*futures[future], future.result())

        for done, (table, key, result) in enumerate(finished(), 1):
            data[table][key] = result
            state.save(unless_recent=True)
            print(f"{table} {done}/{len(jobs)}: {key}", flush=True)

    with contextlib.ExitStack() as stack:
        pool = (
            stack.enter_context(ProcessPoolExecutor(max_workers=workers))
            if workers > 1
            else None
        )
        # Every pair of the grid.
        jobs = []
        for i, regime, n, c, family, j in cases(config):
            for t in config.ts:
                for r in config.rs:
                    key = _key(regime, n, c, family, t, r)
                    if key not in data["grid"]:
                        seed = _seed(config, GRID, i, n, c, j, t, r)
                        args = {
                            "regime": regime,
                            "n": n,
                            "c": c,
                            "family": family,
                            "t": t,
                            "r": r,
                            "seed": seed,
                        }
                        jobs.append(("grid", key, args))
        submit_all(pool, jobs, config.draws)
        state.save()

        # The choices, and the pairs to estimate again.
        jobs = []
        for i, regime, n, c, family, j in cases(config):
            restricted = FAMILIES[family]
            case = _key(regime, n, c, family)
            record = data["direct"][_key(regime, n, c)]
            direct_rate = record["errors"] / record["shots"]
            points = {}
            for t in config.ts:
                for r in config.rs:
                    record = data["grid"][_key(case, t, r)]
                    points[t, r] = measured(record, n, restricted)
            rule_r = rule_rs[n, c]
            grid_bounds = data["bounds"][case]["grid"]
            again: set[tuple[int, int]] = set()
            for ceiling in config.ceilings:
                # The direct implementation first, so that it wins ties.
                candidates: list[tuple[float, list[int] | str]] = [
                    (direct_rate, DIRECT)
                ] + [
                    (rate, [t, r])
                    for (t, r), (rate, overhead) in points.items()
                    if overhead <= ceiling and not math.isnan(rate)
                ]
                candidates.sort(key=lambda candidate: candidate[0])
                top = [pick for _, pick in candidates[: config.top]]
                # The paper's rule: the smallest t whose overhead fits.
                rule = next(
                    (
                        [t, rule_r]
                        for t in config.ts
                        if points[t, rule_r][1] <= ceiling
                        and not math.isnan(points[t, rule_r][0])
                    ),
                    None,
                )
                rule_by_bound = next(
                    (
                        [t, rule_r]
                        for t in config.ts
                        if grid_bounds[_key(t, rule_r)][1] <= ceiling
                    ),
                    None,
                )
                picked = data["bounds"][case]["choice"][str(ceiling)]
                # Pairs are [t, r], "direct" is the direct implementation,
                # and None a rule that no t in the grid satisfies.
                data["choices"][_key(case, ceiling)] = {
                    "bound": picked,
                    "rule": rule,
                    "rule_by_bound": rule_by_bound,
                    "measured": top,
                }
                for pick in [picked, rule, rule_by_bound, *top]:
                    if isinstance(pick, list):
                        again.add((pick[0], pick[1]))
            for t, r in sorted(again):
                key = _key(case, t, r)
                if key not in data["reestimates"]:
                    seed = _seed(config, REESTIMATE, i, n, c, j, t, r)
                    args = {
                        "regime": regime,
                        "n": n,
                        "c": c,
                        "family": family,
                        "t": t,
                        "r": r,
                        "seed": seed,
                    }
                    jobs.append(("reestimates", key, args))
        state.save()
        submit_all(pool, jobs, config.reestimate_draws)

    data["runs"][-1]["finished"] = datetime.datetime.now(
        datetime.UTC
    ).isoformat()
    state.save()


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="run a small grid"
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
    name = "regret-quick.json" if args.quick else "regret.json"
    path = args.output or Path("results") / name
    run(config, path, args.workers)


if __name__ == "__main__":
    main(sys.argv[1:])
