"""Measure how far the bound on the logical error rate sits above it.

Two parts, both on the circuits and noise regimes of ``scripts/regret.py``,
whose results file they read.

The waterfall: for each case of the regret experiment and each pair
``(t, r)`` with ``r >= 1`` that measurement chose there, the checks are
drawn many times and the faults of each draw listed with
:func:`clinr.first_order.faults`. Averaged over the draws, they give, per
chunk and region, the probability of a fault that damages the output and of
one that does so undetected, to set against the terms of
:func:`clinr.bounds.plog_terms`. The measured acceptance and logical error
rate come from the regret experiment's fresh estimates.

The positions: for one check (``t = r = 1``), the probability that the
check catches a damaging fault, by the fault's position in the circuit, for
random Clifford circuits and for random gate sequences of the same size.

Run from the repository root, after ``scripts/regret.py``::

    uv run python scripts/sharpness.py

The results go to ``results/sharpness.json`` with the settings, the seed,
the versions and platform, and a digest of the regret results they used.
Each point is saved as it finishes, and a run resumes from the file.
``--quick`` uses a small configuration, for checking the script against a
quick regret run.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import regret
import stim

from clinr import accounting, bounds, checks, circuits, first_order, protocol
from clinr.noise_models import NoiseModel

# The regions of a chunk that the bound charges separately.
REGIONS = ("resource", "check", "teleport")
# The builder's tags, by the region of the bound they belong to.
TAG_REGIONS = {
    "prep": "resource",
    "gates": "resource",
    "check": "check",
    "teleport": "teleport",
}
GENERATORS = ("clifford", "sequence")
# Phases of the run, which separate the random streams of its parts.
WATERFALL, POSITIONS, SEQUENCES = 0, 1, 2


@dataclasses.dataclass(frozen=True)
class Config:
    """The settings of a run, saved with its results."""

    draws: int
    position_sizes: tuple[int, ...]
    position_circuits: int
    position_draws: int
    bins: int
    seed: int


FULL = Config(
    draws=40,
    position_sizes=(3, 8, 15, 25),
    position_circuits=3,
    position_draws=100,
    bins=10,
    seed=20261009,
)
QUICK = Config(
    draws=2,
    position_sizes=(2,),
    position_circuits=1,
    position_draws=3,
    bins=4,
    seed=1,
)


def _region(tag: str) -> tuple[str, int]:
    """Return the bound's region and the chunk of a builder tag."""
    kind, chunk = tag.split("/")[:2]
    return TAG_REGIONS[kind], int(chunk)


def tally(built: stim.Circuit, t: int, r: int) -> dict[str, list[list[float]]]:
    """Sum the faults of one draw by chunk and region.

    Returns, under ``damaging`` and ``undetected``, a ``t`` by 3 table of
    the probability of a fault that flips an observable, with or without a
    detector, and of one that flips an observable and no detector, in the
    order of :data:`REGIONS`; under ``gadgets``, a ``t`` by ``r`` table of
    the damaging probability of each check gadget; and under ``detected``,
    the probability of a fault that flips a detector of each chunk.
    """
    damaging = [[0.0] * len(REGIONS) for _ in range(t)]
    undetected = [[0.0] * len(REGIONS) for _ in range(t)]
    gadgets = [[0.0] * r for _ in range(t)]
    detected = [0.0] * t
    for fault in first_order.faults(built):
        region, k = _region(fault.tag)
        column = REGIONS.index(region)
        if fault.detected:
            detected[k] += fault.probability
        if not fault.logical:
            continue
        damaging[k][column] += fault.probability
        if region == "check":
            gadgets[k][int(fault.tag.split("/")[2])] += fault.probability
        if not fault.detected:
            undetected[k][column] += fault.probability
    return {
        "damaging": damaging,
        "undetected": undetected,
        "gadgets": gadgets,
        "detected": [detected],
    }


def run_waterfall_point(
    text: str,
    n: int,
    t: int,
    r: int,
    noise: NoiseModel,
    restricted: bool,
    draws: int,
    seed: np.random.SeedSequence,
) -> dict[str, Any]:
    """Average the fault tallies of one pair over ``draws`` draws."""
    circuit = stim.Circuit(text)
    rng = np.random.default_rng(seed)
    total: dict[str, np.ndarray] = {}
    for _ in range(draws):
        drawn = checks.draw_checks(
            circuit, n, t, r, rng, restricted_checks=restricted
        )
        built = protocol.build_clinr_tr(circuit, n, drawn, noise)
        for name, table in tally(built, t, r).items():
            total[name] = total.get(name, 0) + np.array(table)
    return {name: (table / draws).tolist() for name, table in total.items()}


def positions(
    built: stim.Circuit, bins: int
) -> tuple[list[float], list[float]]:
    """Return the damaging and caught probability of one draw, by position.

    Position 0 is the preparation of the resource; positions 1 to ``bins``
    divide the noise locations of the chunk's gates into equal parts, in
    time order.
    """
    gates = [
        index
        for index in range(len(built))
        if getattr(built[index], "tag", "") == "gates/0"
    ]
    damaging = [0.0] * (bins + 1)
    caught = [0.0] * (bins + 1)
    for fault in first_order.faults(built):
        if not fault.logical or not fault.tag.endswith("/0"):
            continue
        kind = fault.tag.split("/")[0]
        if kind == "prep":
            position = 0
        elif kind == "gates":
            rank = gates.index(fault.instruction)
            position = 1 + rank * bins // len(gates)
        else:
            continue
        damaging[position] += fault.probability
        if fault.detected:
            caught[position] += fault.probability
    return damaging, caught


def run_position_point(
    text: str,
    n: int,
    noise: NoiseModel,
    restricted: bool,
    draws: int,
    bins: int,
    seed: np.random.SeedSequence,
) -> dict[str, list[float]]:
    """Average the position profile of one circuit over ``draws`` checks."""
    circuit = stim.Circuit(text)
    rng = np.random.default_rng(seed)
    damaging = np.zeros(bins + 1)
    caught = np.zeros(bins + 1)
    for _ in range(draws):
        drawn = checks.draw_checks(
            circuit, n, 1, 1, rng, restricted_checks=restricted
        )
        built = protocol.build_clinr_tr(circuit, n, drawn, noise)
        profile = positions(built, bins)
        damaging += profile[0]
        caught += profile[1]
    return {
        "damaging": (damaging / draws).tolist(),
        "caught": (caught / draws).tolist(),
    }


STAGES = ("bound", "damaging", "detection", "acceptance", "measured")


def waterfall(
    data: dict[str, Any], results: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the bound at each stage of its replacement by what happens.

    Parameters
    ----------
    data : dict
        A finished sharpness results file, as loaded from JSON.
    results : dict
        The regret results it was computed from.

    Returns
    -------
    list of dict
        One record per waterfall point, with keys ``regime``, ``n``,
        ``circuit``, ``family``, ``t`` and ``r``; ``stages``, the value at
        each of :data:`STAGES`; ``regions``, the contribution of each of
        :data:`REGIONS` at the first three stages; and ``overhead``, the
        overhead bound, the overhead with the measured acceptance and
        worst-case check weights, and the measured overhead.
        The stages replace, in turn: the probability that a region faults
        with the probability that it faults damagingly, at the worst-case
        detection (``damaging``); the worst-case detection with the faults
        that escape the drawn checks (``detection``); the bound on each
        chunk's acceptance with its measured acceptance (``acceptance``);
        and the rate so obtained with the measured one (``measured``).
        The first two replacements scale each of the bound's fault
        probabilities by a fraction of first-order quantities: the
        damaging, or escaping, fault probability of the region over the
        total of its fault rates. Scaling keeps the fault probabilities
        exact where first-order sums would exceed them, so that each
        replacement isolates one source of slack.
    """
    regret_config = results["config"]
    regimes = {
        name: NoiseModel(**noise) for name, noise in regret_config["regimes"]
    }
    rows = []
    for key, point in data["waterfall"].items():
        regime, size, c, family, ts, rs = key.split("/")
        n, t, r = int(size), int(ts), int(rs)
        restricted = regret.FAMILIES[family]
        circuit = stim.Circuit(results["circuits"][size][int(c)])
        noise = regimes[regime]
        terms = bounds.plog_terms(
            circuit, n, t, r, noise, restricted_checks=restricted
        )
        record = results["reestimates"][key]
        rate, overhead = regret.measured(record, n, restricted)
        acceptance = np.mean(
            np.array(record["chunk_accepted"]) / record["shots"], axis=0
        )
        stages = dict.fromkeys(STAGES, 0.0)
        regions = {stage: dict.fromkeys(REGIONS, 0.0) for stage in STAGES[:3]}
        counts = [
            accounting.gate_counts(chunk)
            for chunk in protocol.chunk_circuit(circuit, n, t)
        ]
        worst = n + 1 if restricted else 2 * n
        prep = accounting.prep_region(n, native_plus=False)
        for k, term in enumerate(terms):
            q = term.per_check_miss
            s1, s2 = counts[k]
            # Each region's fault probability, and the total of its rates.
            exact = {
                "resource": term.resource,
                "check": term.check,
                "teleport": term.teleport,
            }
            rates = {
                "resource": _rate_sum(
                    noise,
                    accounting.add(
                        prep, accounting.Region(n1=s1, n2=s2, nm=0, n_prep=0)
                    ),
                ),
                "check": _rate_sum(
                    noise, accounting.check_region(1, worst, False)
                ),
                "teleport": _rate_sum(noise, accounting.teleport_region(n)),
            }
            resource, _, teleport = point["damaging"][k]
            harmful = {
                "resource": resource * term.miss,
                "check": sum(
                    gadget * q ** (r - 1 - j)
                    for j, gadget in enumerate(point["gadgets"][k])
                ),
                "teleport": teleport,
            }
            charged = {
                "resource": term.resource * term.miss,
                "check": term.check * term.escapes,
                "teleport": term.teleport,
            }
            damaging = {
                region: exact[region] * harmful[region] / rates[region]
                for region in REGIONS
            }
            escaped = {
                region: exact[region] * escaping / rates[region]
                for region, escaping in zip(
                    REGIONS, point["undetected"][k], strict=True
                )
            }
            for stage, values in (
                ("bound", charged),
                ("damaging", damaging),
                ("detection", escaped),
            ):
                for region, value in values.items():
                    regions[stage][region] += value / term.acceptance
                stages[stage] += sum(values.values()) / term.acceptance
            stages["acceptance"] += sum(escaped.values()) / float(
                acceptance[k]
            )
        stages["measured"] = rate
        rows.append(
            {
                "regime": regime,
                "n": n,
                "circuit": int(c),
                "family": family,
                "t": t,
                "r": r,
                "stages": stages,
                "regions": regions,
                "overhead": {
                    "bound": bounds.overhead_bound(
                        circuit, n, t, r, noise, restricted_checks=restricted
                    ),
                    "acceptance": accounting.expected_overhead(
                        counts,
                        accounting.clinr_scheme(n, native_plus=False),
                        [accounting.check_region(r, r * worst, False)] * t,
                        [float(value) for value in acceptance],
                    ),
                    "measured": overhead,
                },
            }
        )
    return rows


def _rate_sum(noise: NoiseModel, region: accounting.Region) -> float:
    """Return the total of the fault rates of a region's operations."""
    return (
        region.n1 * noise.p1
        + region.n2 * noise.p2
        + region.nm * noise.p_m
        + region.n_prep * noise.p_prep
    )


def digest(path: Path) -> str:
    """Return the SHA-256 of a file, to tie these results to their input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(data: dict[str, Any], path: Path) -> None:
    """Write ``data`` atomically, so an interruption cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def run(config: Config, source: Path, path: Path) -> None:
    """Run both parts, resuming from ``path`` if it exists."""
    settings = json.loads(json.dumps(dataclasses.asdict(config)))
    source_digest = digest(source)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["config"] != settings or data["regret"] != source_digest:
            raise SystemExit(
                f"{path} was made with other settings or other regret "
                "results; move it or pass --output"
            )
    else:
        data = {
            "experiment": "sharpness",
            "config": settings,
            "regret": source_digest,
            "seeding": (
                "each point's generator is seeded by "
                "numpy.random.SeedSequence(config.seed, spawn_key=(phase, "
                "regime index, n, circuit index, family index, t, r)) with "
                "phase 0 for the waterfall; spawn_key=(1, n, circuit index, "
                "family index, generator index) for the positions; and "
                "spawn_key=(2, n, circuit index) for the gate sequences"
            ),
            "waterfall": {},
            "sequences": {},
            "positions": {},
            "runs": [],
        }
    data["runs"].append(
        {
            "started": datetime.datetime.now(datetime.UTC).isoformat(),
            "provenance": regret.provenance(),
        }
    )
    save(data, path)
    results = json.loads(source.read_text(encoding="utf-8"))
    regret_config = results["config"]
    regimes = {
        name: NoiseModel(**noise) for name, noise in regret_config["regimes"]
    }
    regime_index = {
        name: i for i, (name, _) in enumerate(regret_config["regimes"])
    }
    family_index = {name: j for j, name in enumerate(regret.FAMILIES)}

    # The waterfall, at each pair measurement chose, with checks.
    pairs: defaultdict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in regret.outcomes(results):
        pick = row["best_pair_pick"]
        if isinstance(pick, list) and pick[1] >= 1:
            case = regret._key(
                row["regime"], row["n"], row["circuit"], row["family"]
            )
            pairs[case].add((pick[0], pick[1]))
    for case, chosen in sorted(pairs.items()):
        regime, size_key, circuit_key, family = case.split("/")
        n, c = int(size_key), int(circuit_key)
        for t, r in sorted(chosen):
            key = regret._key(case, t, r)
            if key in data["waterfall"]:
                continue
            seed = np.random.SeedSequence(
                config.seed,
                spawn_key=(
                    WATERFALL,
                    regime_index[regime],
                    n,
                    c,
                    family_index[family],
                    t,
                    r,
                ),
            )
            data["waterfall"][key] = run_waterfall_point(
                results["circuits"][size_key][c],
                n,
                t,
                r,
                regimes[regime],
                regret.FAMILIES[family],
                config.draws,
                seed,
            )
            save(data, path)
            print(f"waterfall {key}", flush=True)

    # The positions, under the paper's rates, for both generators.
    noise = regimes[regret_config["regimes"][0][0]]
    for n in config.position_sizes:
        texts = results["circuits"][str(n)][: config.position_circuits]
        for c, text in enumerate(texts):
            # A gate sequence of the same size, the same for both families.
            sequence_key = regret._key(n, c)
            if sequence_key not in data["sequences"]:
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        config.seed, spawn_key=(SEQUENCES, n, c)
                    )
                )
                gate_count = sum(accounting.gate_counts(stim.Circuit(text)))
                data["sequences"][sequence_key] = str(
                    circuits.random_gate_sequence(n, gate_count, rng)
                )
            generated = {
                "clifford": text,
                "sequence": data["sequences"][sequence_key],
            }
            for g, generator in enumerate(GENERATORS):
                for family, restricted in regret.FAMILIES.items():
                    key = regret._key(n, c, generator, family)
                    if key in data["positions"]:
                        continue
                    seed = np.random.SeedSequence(
                        config.seed,
                        spawn_key=(POSITIONS, n, c, family_index[family], g),
                    )
                    data["positions"][key] = run_position_point(
                        generated[generator],
                        n,
                        noise,
                        restricted,
                        config.position_draws,
                        config.bins,
                        seed,
                    )
                    save(data, path)
                    print(f"positions {key}", flush=True)

    data["runs"][-1]["finished"] = datetime.datetime.now(
        datetime.UTC
    ).isoformat()
    save(data, path)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="run a small configuration"
    )
    parser.add_argument(
        "--regret", type=Path, help="the regret results to read"
    )
    parser.add_argument("--output", type=Path, help="the results file")
    args = parser.parse_args(argv)
    config = QUICK if args.quick else FULL
    suffix = "-quick" if args.quick else ""
    source = args.regret or Path("results") / f"regret{suffix}.json"
    path = args.output or Path("results") / f"sharpness{suffix}.json"
    run(config, source, path)


if __name__ == "__main__":
    main(sys.argv[1:])
