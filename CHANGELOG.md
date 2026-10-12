# Changelog

All notable changes to this project are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first release.
It replaces the earlier demonstration of the scheme with one chunk, which had no version.

### Added

- `clinr.protocol`: the circuits of the direct implementation, of CliNR, for any circuit of single- and two-qubit Clifford gates split into $t$ chunks with $r$ checks each, and of CZNR, for circuits of `CZ` gates, each with a Choi harness that reads out the implemented channel.
- `clinr.checks`: checks drawn uniformly from the resource state's stabilizer group, or for CliNR from the restricted family of the paper's numerics.
- `clinr.bounds`: bounds on the logical error rate and the gate overhead, rederiving the paper's Theorems 1 and 3 with four error rates, and a choice of $(t, r)$ or the direct implementation by them.
- `clinr.accounting`: the operation counts of each region of a built circuit, which the bounds and the estimators share.
- `clinr.estimators`: estimates of the logical error rate and the gate overhead, over draws of the checks, with confidence intervals; one draw's interval is for the rate given its checks.
- `clinr.first_order`: every single fault of a built circuit, with its region and whether it is caught, and the logical error rate to first order.
- `clinr.circuits` and `clinr.noise_models`: random circuits, and the noise models of the paper and of the experiments.
- Four experiments, each one script writing one results file: the regret of choosing by the bounds, how tight the bounds are, when CliNR pays off, and when CZNR pays off on dense blocks of `CZ` gates.
- A simulation of how often the interval on the logical error rate covers, `scripts/coverage.py`, writing `results/coverage.json`.
- Support for Python 3.13 and later, the floor that SPEC 0 sets at this release; CI runs the tests on 3.13 and on 3.14.
- Documentation: a tutorial, how-to guides, explanations of the scheme, the bounds and every result, decision records, and the API reference.

### Results

Each results file records the commit it was run at: `results/regret.json` at `75d16fc`, `results/sharpness.json` at `e80d396`, `results/payoff.json.gz` at `84d7157`, `results/dense.json.gz` at `d05e2a4` and `results/coverage.json` at `1dc9b26`.
The commits since change nothing the results depend on: they add readers of the results files, code that the experiments do not run, refactors that leave the bounds' values unchanged, and signed checks and readouts, which leave every circuit's detector error model and seeded samples unchanged.
One commit changes the bounds' values, by computing the probability that some operation faults so that it keeps its precision at rates far below machine precision: over the grid of `results/regret.json` the bounds move by less than $10^{-13}$ of their value, and no choice of `choose_tr` changes.
The pages of `payoff.py` and `dense.py` state how many of the chosen pairs their fresh estimates show over budget, and by how much at most.
