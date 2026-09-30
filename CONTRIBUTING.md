# Contributing

Open an issue before a large change.
Pull requests target `main` and must pass CI.

## Setup

```bash
uv sync
uv tool install pre-commit
pre-commit install
```

Add a dependency with `uv add` (or `uv add --dev`) and commit the updated `uv.lock`.

## Checks

The checks are [nox](https://nox.thea.codes) sessions, which CI runs too.

```bash
uvx nox                 # lint, typecheck, the fast tests and the docs, as on every push
uvx nox -s tests_full   # every test, as on pull requests
uvx nox -l              # list the sessions
```

## Conventions

- Mathematical quantities follow the notation of Delfosse and Tham, arXiv:2407.06583: `n` data qubits, `s` gates, `t` chunks, `r` checks per chunk, error rates `p`.
  Other names, loop variables included, say what they hold.
- Public functions validate their arguments and raise `ValueError` or `TypeError`, naming the argument.
  Private functions (leading underscore) state their preconditions and check the cheap ones with `assert`.
- An argument that can be computed from another is computed, not passed.
- A name is public only if it serves a caller's goal on its own or is a contract other modules and the tests rely on; otherwise it is private.
- Every stochastic function takes a `numpy.random.Generator` named `rng`.
  The exception is `circuits.random_clifford`, since Stim's Clifford sampler takes no seed; keep generated circuits with `circuits.save_circuits`.
- Docstrings use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) and cite the section or equation of the paper they implement.
  `Raises` lists only what the function raises itself.
- Tests go through the public API and are named for the behavior they check.
  Expected values are hand-worked or come from an independent oracle, never from the formula under test.
  Statistical tests allow 5 standard errors; property tests draw an integer seed from Hypothesis.
  Tests that take more than about a second are marked `slow`.
- Markdown prose uses [semantic line breaks](https://sembr.org).
- Commit messages have an imperative subject line and a body saying why.

Design decisions are recorded in [`docs/adr/`](docs/adr/).
