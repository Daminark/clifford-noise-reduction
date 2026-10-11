# Summary of findings

This package implements CliNR and its variant CZNR in Stim, and asks four questions of them by simulation.
Each answer below is measured on the page it links to.
Every rate is a logical error rate per run of the circuit, under circuit-level noise without idle noise.
The paper's rates are $p_2 = 10^{-3}$ for two-qubit gates and $p_2/10$ for every other operation, unless another $p_2$ is named; the Forte-like rates are those of IonQ's published figures for its Forte Enterprise system, as {doc}`families` sets them out.

## Which checks to draw

With room in the budget on the gate overhead, uniform checks win; when the budget binds, the paper's restricted ones do.
At an overhead of 4 and $n = 25$ under the paper's rates, the best uniform pair reaches {glue:text}`families.md::uniform_paper_4_25:.3f` and the best restricted one {glue:text}`families.md::restricted_paper_4_25:.3f`, against {glue:text}`families.md::direct_paper_4_25:.3f` for the direct implementation; at an overhead of 2, restricted checks win from $n = 15$ ({doc}`families`).

## What the bounds are worth

The bound on the direct implementation is nearly exact, while the bound on CliNR sits {glue:text}`sharpness.md::slack_low:.1f` to {glue:text}`sharpness.md::slack_high:.1f` times above the measured rate.
For restricted checks the largest part of that is detection: the worst-case catch of one check is $2/(3n)$, far below what drawn checks catch, a factor of {glue:text}`sharpness.md::detection_paper_restricted_25:.1f` at $n = 25$ ({doc}`sharpness`).
So choosing by the bounds loses little with uniform checks, a regret of {glue:text}`regret.md::bound_paper_4_uniform_25:.2f` at $n = 25$, and a factor of {glue:text}`regret.md::bound_paper_4_restricted_25:.1f` with restricted ones, for which the bounds never recommend CliNR ({doc}`regret`).

## When CliNR pays off

On random Clifford circuits under the paper's rule, CliNR reproduces the shape of the paper's Fig. 2 and Fig. 4 with larger gains.
At matched circuit size the gain is {glue:text}`payoff.md::gain_paper_2_restricted_20:.1f` against the paper's 2 at $p_2 = 10^{-3}$, and {glue:text}`payoff.md::gain_paper_1e_4_2_restricted_50:.1f` against its 4 at $10^{-4}$.
The reach of CliNR scales as Theorem 2 predicts once the budget forces several chunks, and nearly as $1/p$ while one chunk fits.
At Forte-like rates CliNR pays off only for dense circuits of 10 to 15 qubits, at an overhead of 4, by at most {glue:text}`payoff.md::gain_forte_like_4_uniform_15:.1f` ({doc}`payoff`).

## CZNR on blocks of CZ gates

At the same parameters, CZNR's bounds are below CliNR's on every circuit of `CZ` gates, and so, averaged over the checks, is its logical error rate to first order ({doc}`cznr`); on complete graphs it stays ahead well past first order.
At an overhead of 4 and $p_2 = 10^{-4}$, CliNR's rate with uniform checks is {glue:text}`dense.md::ratio_paper_1e_4_uniform_low:.2f` to {glue:text}`dense.md::ratio_paper_1e_4_uniform_high:.2f` times CZNR's.
CZNR reproduces the paper's Fig. 7 at $p_2 = 10^{-4}$, and at $10^{-3}$ only between $n = 15$ and $25$, with $|+\rangle$ prepared natively.
At Forte-like rates it is the one scheme that pays off on these circuits, by up to {glue:text}`dense.md::best_forte_like_4_cznr:.1f`, against CliNR's best of {glue:text}`payoff.md::gain_forte_like_4_uniform_15:.1f` on random Clifford circuits ({doc}`dense`).
