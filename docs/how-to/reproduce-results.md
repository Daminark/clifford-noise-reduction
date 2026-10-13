# Reproduce the results

Each results page renders one file in `results/`, written by one script in `scripts/`.
This guide reruns the scripts and rebuilds the pages from their output.

## Run the experiments

The repository carries the results of the runs in the table below, and a script will not extend a results file that a run at another commit began, so a rerun writes its files elsewhere with `--output`.
Run each script from the repository root, in this order, since `sharpness.py` reads the results of `regret.py`, here through `--regret`:

```bash
uv sync --locked
mkdir -p results/rerun
uv run python scripts/regret.py --output results/rerun/regret.json
uv run python scripts/sharpness.py --regret results/rerun/regret.json --output results/rerun/sharpness.json
uv run python scripts/coverage.py --output results/rerun/coverage.json
uv run python scripts/payoff.py --output results/rerun/payoff.json.gz
uv run python scripts/dense.py --output results/rerun/dense.json.gz
```

| Script                 | Writes                   | Rendered by                                                  | Run at commit | Time   |
| ---------------------- | ------------------------ | ------------------------------------------------------------ | ------------- | ------ |
| `scripts/regret.py`    | `results/regret.json`    | {doc}`../explanation/families`, {doc}`../explanation/regret` | `75d16fc`     | 73 min |
| `scripts/sharpness.py` | `results/sharpness.json` | {doc}`../explanation/sharpness`                              | `e80d396`     | 41 min |
| `scripts/coverage.py`  | `results/coverage.json`  | {doc}`../explanation/coverage`                               | `1dc9b26`     | 8 s    |
| `scripts/payoff.py`    | `results/payoff.json.gz` | {doc}`../explanation/payoff`                                 | `84d7157`     | 4.3 h  |
| `scripts/dense.py`     | `results/dense.json.gz`  | {doc}`../explanation/dense`                                  | `d05e2a4`     | 1.7 h  |

The times are for one machine with 16 CPUs; `regret.py`, `payoff.py` and `dense.py` take `--workers`, the number of processes, which defaults to the number of CPUs.
`regret.py`, `sharpness.py`, `payoff.py` and `dense.py` save as they go and resume from their output file, so a run can be stopped and started again at the same commit.
They refuse to resume a file begun at another commit or made with other settings, and `sharpness.py` also refuses one made from other regret results.
`coverage.py` runs in seconds and overwrites its output file rather than resuming it.
The files go under `results/` because a run records whether the work tree has uncommitted changes or untracked files outside it.
`--quick` runs a small configuration, writing a separate file, to check a script before a full run; `sharpness.py --quick` reads the quick results of `regret.py`.

The circuits of `regret.py` and of the curves of `payoff.py` are random Cliffords from Stim's sampler, which takes no seed, so a rerun draws new circuits and reproduces the results statistically rather than exactly.
Everything else is seeded, and Stim reproduces seeded samples exactly only with the same version on the same machine.

## Rebuild the pages

The pages read the files of the table under their names in `results/`, so they come to read a rerun's files once those replace them:

```bash
mv results/rerun/* results/
```

Then build the site:

```bash
rm -rf docs/_build
uvx nox -s docs
```

The pages are executed when the site is built, and the build caches each page by its source, not by the results it reads, so remove the cache after replacing a results file.

## Check which code produced a file

Each results file records, under `runs`, the commit it was run at, whether the work tree had uncommitted changes or untracked files outside `results/`, the versions of Python, the package, Stim and NumPy, and the platform.
The commits since each of the runs above add code without changing what the scripts compute: readers of the results files, the CZNR variant, tags on noise channels, refactors that leave the bounds' values unchanged, and checks and readouts measured with their signs, which leaves every circuit's detector error model and seeded samples unchanged.
One commit changes the bounds' values, by computing the probability that some operation faults so that it keeps its precision at rates far below machine precision: over the grid of `results/regret.json` the bounds move by less than $10^{-13}$ of their value, and no choice of `choose_tr` changes.
