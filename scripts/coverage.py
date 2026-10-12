"""Measure the coverage of the interval on the logical error rate over draws.

Each scenario draws a rate for each of its draws, lognormal about a mean
rate, samples each draw's errors over its shots, and checks whether
:func:`clinr.estimators.logical_error_rate` covers the mean rate at 95%.
Each coverage is reported with its Monte Carlo standard error (Morris,
White and Crowther, Statistics in Medicine 38, 2074 (2019)).

Run from the repository root::

    uv run python scripts/coverage.py

The results go to ``results/coverage.json``, with the settings, the seed and
the versions that produced them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import regret

from clinr import estimators


@dataclasses.dataclass(frozen=True)
class Scenario:
    """Draws whose rates are lognormal about ``rate``."""

    name: str
    rate: float
    sigma: float
    draws: int
    shots: int


SCENARIOS = (
    Scenario("plentiful errors", 2e-2, 0.5, 40, 1000),
    Scenario("few errors", 1e-4, 0.5, 10, 1000),
    Scenario("heavy spread", 2e-2, 1.5, 10, 1000),
    Scenario("few errors, heavy spread", 1e-4, 1.5, 10, 1000),
)
REPLICATIONS = 10_000
SEED = 20261014


def coverage(
    scenario: Scenario, replications: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Return the coverage of the 95% interval in ``scenario``."""
    location = math.log(scenario.rate) - scenario.sigma**2 / 2
    covered = 0
    errors = 0
    for _ in range(replications):
        rates = rng.lognormal(location, scenario.sigma, scenario.draws)
        counts = rng.binomial(scenario.shots, np.minimum(rates, 1.0))
        errors += int(counts.sum())
        shots = scenario.shots
        draws = [
            estimators.Estimate(shots, shots, int(e), 1, 0, 1) for e in counts
        ]
        _, low, high = estimators.logical_error_rate(draws)
        covered += low <= scenario.rate <= high
    share = covered / replications
    return {
        **dataclasses.asdict(scenario),
        "replications": replications,
        "mean_errors": errors / replications,
        "coverage": share,
        "mcse": math.sqrt(share * (1 - share) / replications),
    }


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run every scenario."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=Path("results/coverage.json")
    )
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    rows = [coverage(s, args.replications, rng) for s in SCENARIOS]
    out = {
        "experiment": "coverage",
        "seed": SEED,
        "provenance": regret.provenance(),
        "scenarios": rows,
    }
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['name']}: {row['coverage']:.3f} "
            f"(MC SE {row['mcse']:.3f}), {row['mean_errors']:.1f} errors"
        )


if __name__ == "__main__":
    main()
