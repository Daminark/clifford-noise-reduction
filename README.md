# CliNR: Clifford Noise Reduction in Stim

[![CI](https://github.com/Daminark/clifford-noise-reduction/actions/workflows/ci.yml/badge.svg)](https://github.com/Daminark/clifford-noise-reduction/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://daminark.github.io/clifford-noise-reduction)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

An implementation in [Stim](https://github.com/quantumlib/Stim) of Clifford Noise Reduction (CliNR), the scheme of Delfosse and Tham ([arXiv:2407.06583](https://arxiv.org/abs/2407.06583)), and a simulation study of it.

CliNR lowers the logical error rate of a Clifford circuit without error correction.
It splits the circuit into chunks and applies each by gate teleportation, through a resource state that is prepared offline and checked by a few random stabilizer measurements, so that a faulty resource is discarded before the data ever touches it.
The package builds these circuits for any Clifford circuit, together with the variant for circuits of `CZ` gates, CZNR; computes the paper's analytic bounds on the logical error rate and the gate overhead; and estimates both by sampling.

The study asks how good the bounds are as guides, which family of checks to draw, and when the scheme pays off, under the paper's noise rates and under rates modeled on a current trapped-ion system.
The answers are collected in the [summary of findings](https://daminark.github.io/clifford-noise-reduction/explanation/findings.html), and each result is reproduced by one script.
An earlier version, with one chunk, was listed as a top project of the Erdős Institute Quantum Computing Boot Camp.

## Installation

The package needs Python 3.13 or later and is installed from GitHub:

```bash
pip install git+https://github.com/Daminark/clifford-noise-reduction
```

To work on it, clone the repository and run `uv sync`; [CONTRIBUTING.md](CONTRIBUTING.md) lists the checks.

## Usage

Compare a small random circuit run directly and under CliNR with one chunk and four checks:

```python
import numpy as np

from clinr import bounds, circuits, estimators, noise_models

rng = np.random.default_rng(2026)
n = 3
circuit = circuits.random_gate_sequence(n, 30, rng)
noise = noise_models.asymmetric_noise(1e-3)

direct = estimators.estimate_direct(circuit, n, noise, 200_000, rng)
clinr = estimators.estimate_clinr(circuit, n, 1, 4, noise, 200_000, rng)
print(f"direct: {direct.interval()[0]:.4f}")  # direct: 0.0129
print(f"CliNR:  {clinr.interval()[0]:.4f}")  # CliNR:  0.0075
print(f"gate overhead: {estimators.gate_overhead(clinr):.2f}")  # 3.12
print(f"bound: {bounds.plog_bound(circuit, n, 1, 4, noise):.4f}")  # 0.0176
```

The [tutorial](https://daminark.github.io/clifford-noise-reduction/tutorial/getting-started.html) walks through this example, the bounds, and choosing the number of chunks and checks.

## Documentation

The documentation is at [daminark.github.io/clifford-noise-reduction](https://daminark.github.io/clifford-noise-reduction): a tutorial, how-to guides for your own circuits and checks and for [reproducing the results](https://daminark.github.io/clifford-noise-reduction/how-to/reproduce-results.html), explanations of the scheme, the bounds and every result, and the API reference.
The departures from the paper are listed on [one page](https://daminark.github.io/clifford-noise-reduction/explanation/differences.html).

## Related work

The scheme is that of N. Delfosse and E. Tham, "Low-cost noise reduction for Clifford circuits", [arXiv:2407.06583](https://arxiv.org/abs/2407.06583), Physical Review Letters 134, 090603 (2025).
Its extensions, Optimized, Recursive and Distributed CliNR, and a study on trapped-ion hardware are placed relative to this package on the [related work](https://daminark.github.io/clifford-noise-reduction/explanation/related-work.html) page.

## Citation

To cite the package, use the metadata in [CITATION.cff](CITATION.cff), which GitHub offers as "Cite this repository"; please also cite the paper above.

## License

[MIT](LICENSE).
