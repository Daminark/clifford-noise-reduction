# Reproduce the results

Each results page renders one file in `results/`, written by one script in `scripts/`.
This guide reruns the scripts and rebuilds the pages from their output.

## Run the experiments

Run each script from the repository root, in this order, since `sharpness.py` reads the results of `regret.py`:

```bash
uv sync --locked
uv run python scripts/regret.py
uv run python scripts/sharpness.py
uv run python scripts/payoff.py
uv run python scripts/dense.py
```

| Script                 | Writes                   | Rendered by                                                  | Run at commit | Time   |
| ---------------------- | ------------------------ | ------------------------------------------------------------ | ------------- | ------ |
| `scripts/regret.py`    | `results/regret.json`    | {doc}`../explanation/families`, {doc}`../explanation/regret` | `75d16fc`     | 73 min |
| `scripts/sharpness.py` | `results/sharpness.json` | {doc}`../explanation/sharpness`                              | `e80d396`     | 41 min |
| `scripts/payoff.py`    | `results/payoff.json.gz` | {doc}`../explanation/payoff`                                 | `84d7157`     | 4.3 h  |
| `scripts/dense.py`     | `results/dense.json.gz`  | {doc}`../explanation/dense`                                  | `d05e2a4`     | 1.7 h  |

The times are for one machine with 16 CPUs; `regret.py`, `payoff.py` and `dense.py` take `--workers`, the number of processes, which defaults to the number of CPUs.
A script saves as it goes and resumes from its results file, so it can be stopped and started again.
`--quick` runs a small configuration, writing a separate file, to check a script before a full run; `sharpness.py --quick` reads the quick results of `regret.py`.

The circuits of `regret.py` and of the curves of `payoff.py` are random Cliffords from Stim's sampler, which takes no seed, so a rerun draws new circuits and reproduces the results statistically rather than exactly.
Everything else is seeded, and Stim reproduces seeded samples exactly only with the same version on the same machine.

## Rebuild the pages

```bash
rm -rf docs/_build
uvx nox -s docs
```

The pages are executed when the site is built, and the build caches each page by its source, not by the results it reads, so remove the cache after replacing a results file.

## Check which code produced a file

Each results file records, under `runs`, the commit it was run at, whether the work tree had uncommitted changes, the versions of Python, the package, Stim and NumPy, and the platform.
The commits since each of the runs above add code without changing what the scripts compute: readers of the results files, the CZNR variant, tags on noise channels, refactors that leave the bounds' values unchanged, and checks and readouts measured with their signs, which leaves every circuit's detector error model and seeded samples unchanged.
