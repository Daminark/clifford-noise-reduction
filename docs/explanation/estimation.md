---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Estimation

{mod}`clinr.estimators` turns sampled shots into the protocol's logical error rate and gate overhead, with intervals.
We explain what each estimator estimates and why its interval covers what it claims to, building on the argument of {doc}`the-scheme` that the shots in which every check passes have the protocol's output distribution.
We begin with one draw of the checks, then turn to many draws and their intervals, and close with the design effect, which measures how much the draws differ.

```{code-cell} python
---
tags: [remove-cell]
---
# Computes the figures quoted in the last section.
import numpy as np
from myst_nb import glue

from clinr import circuits, estimators, noise_models

rng = np.random.default_rng(2026)
n, t, r = 3, 2, 4
circuit = circuits.random_gate_sequence(n, 30, rng)
noise = noise_models.uniform_noise(2e-3)
for family, restricted in (("uniform", False), ("restricted", True)):
    draws = [
        estimators.estimate_clinr(
            circuit, n, t, r, noise, 20_000, rng, restricted_checks=restricted
        )
        for _ in range(40)
    ]
    rate, low, high = estimators.logical_error_rate(draws)
    _, pooled_low, pooled_high = estimators.wilson_interval(
        sum(draw.errors for draw in draws), sum(draw.accepted for draw in draws)
    )
    kept, _, _ = estimators.logical_error_rate(draws, checks="kept")
    glue(f"{family}_effect", estimators.design_effect(draws), display=False)
    glue(f"{family}_widening", (high - low) / (pooled_high - pooled_low), display=False)
    glue(f"{family}_fresh", rate, display=False)
    glue(f"{family}_kept", kept, display=False)
```

## One draw of the checks

{func}`clinr.estimators.estimate_clinr` draws the checks once, builds the circuit and samples it, and the {class}`clinr.estimators.Estimate` it returns holds the counts.
A shot is accepted when every detector passes, and the draw's logical error rate is the fraction of accepted shots in which some observable flips.

Its interval is the Wilson score interval, {func}`clinr.estimators.wilson_interval`.
Logical error rates sit near zero, where the Wald interval, the point estimate plus or minus a multiple of its standard error, leaves $[0, 1]$ and collapses to a point when no error is seen; the Wilson interval does neither.
The interval is honest for the draw, since given the checks the shots are independent.

For each chunk, the estimate also counts the shots that the chunk's own detectors accept.
These are marginals, not factors of the overall acceptance.
Chunk $k$ restarts on its own checks only, so its expected number of attempts is $1/a_k$, with $a_k$ its marginal acceptance, and the overhead is a sum over the chunks, whose expectation does not depend on how the chunks are correlated.

{func}`clinr.estimators.gate_overhead` evaluates the overhead of {doc}`checks-and-bounds`,

$$\omega = \frac{1}{s} \sum_k \left(4n + \frac{m_k}{a_k}\right),$$

with each chunk's operations per attempt $m_k$ counted at the check weights drawn, and each $a_k$ measured; for CZNR the teleportation is $2n$ in place of $4n$.
Every attempt measures all its checks ({doc}`../adr/0001-run-every-check-before-deciding`), so $m_k$ is exact, and the acceptance is the only estimated quantity.
By Jensen's inequality, the reciprocal of an estimated acceptance over $N$ shots is biased upward, by a relative $(1 - a_k)/(N a_k)$ to first order: $5 \times 10^{-8}$ at $a_k = 0.99$ and $N = 2 \times 10^5$.

The estimate of one draw is honest, but one draw is not the protocol, which draws its checks anew.

## Many draws

The protocol draws new checks for every attempt, while all the shots of one estimate share one draw.
So one estimate measures a different protocol, which keeps one set of checks for every attempt, and its shots are not independent samples of the actual protocol, since they are clustered by draw.
Repeated calls give independent draws.
{func}`clinr.estimators.logical_error_rate`, {func}`clinr.estimators.gate_overhead_over_draws` and {func}`clinr.estimators.design_effect` take a list of estimates and treat the draw, not the shot, as the unit of observation.

For a draw $D$ of the checks, write $a(D)$ for the probability that an attempt passes them, $p(D)$ for the logical error rate given that it passes, and $m(D)$ for the operations of an attempt; for the overhead, all three are per chunk.
With new checks for every attempt (*fresh*) and with one set kept for every attempt (*kept*), the logical error rate and each chunk's cost $\omega_k$, of which $\omega$ is the sum over $s$, are:

| Checks | Logical error rate               | Cost of chunk $k$                        |
| ------ | -------------------------------- | ---------------------------------------- |
| fresh  | $\mathbb{E}[ap] / \mathbb{E}[a]$ | $4n + \mathbb{E}[m_k] / \mathbb{E}[a_k]$ |
| kept   | $\mathbb{E}[p]$                  | $4n + \mathbb{E}[m_k / a_k]$             |

With fresh checks, a draw is more likely to be the one in use the more often it passes, so the accepted attempts over-represent the draws that pass often, and the logical error rate is a ratio of means.
Fresh attempts are independent and identically distributed, so Wald's identity gives each chunk's cost from the mean operations and the mean acceptance.
With kept checks, each run of the protocol draws once, so the logical error rate is the plain mean over draws, and each draw pays its own $m_k / a_k$.

Fresh checks are the paper's protocol (Sec. III) and what the bounds cover, so they are the default; `checks="kept"` gives the other reading.
The two logical error rates differ by $\mathrm{Cov}(a, p) / \mathbb{E}[a]$, which is small when the draws differ little.

Shared-draw estimates suffice for both readings.
A draw's shots estimate $a(D_j)$ and $a(D_j) p(D_j)$ without bias, by the fractions of its shots accepted and accepted with an error, and its $m(D_j)$ is counted exactly.
The draws are independent samples of $D$, so means over draws estimate each expectation in the table.
A draw counts once, whatever its number of shots.
A draw that accepted no shots still counts toward the fresh readings, where it lowers the mean acceptance; the kept readings need every draw to have accepted, since $p$ and $m_k / a_k$ are otherwise undefined.

## Intervals over draws

Each interval is a Student $t$ interval with $J - 1$ degrees of freedom over $J$ draws, with the variance estimated from the spread between the draws.
Each draw's counts carry their own shot noise, so that spread includes the shot noise along with the variation between draws of the checks.

The kept readings are means over draws, so their variance is the sample variance of the per-draw values divided by $J$.
The fresh readings are ratios of means, so their variance comes from the delta method.
For the logical error rate, with $e_j$ and $a_j$ the fractions of draw $j$'s shots accepted with an error and accepted, $\bar a$ the mean of the $a_j$, and the estimate $\hat R = \sum_j e_j / \sum_j a_j$, it is the standard variance of a ratio estimator,

$$\widehat{\mathrm{Var}}(\hat R) = \frac{1}{J (J - 1)} \sum_j \left(\frac{e_j - \hat R a_j}{\bar a}\right)^2.$$

The fresh overhead is a sum over the chunks of such ratios, and its deviations sum theirs.
The interval on the logical error rate is clamped to $[0, 1]$, and with a single draw both bounds are `nan`.

A ratio estimator is biased by $O(1/J)$, as is any estimate of a probability given acceptance, even with new checks drawn for every shot.
The standard error shrinks only like $1/\sqrt{J}$, so with many draws the bias is far below it.
Precision and bias therefore both come from the number of draws, not the shots per draw, which is why {doc}`../how-to/repeated-draws` takes many draws of modest size.

## The design effect

{func}`clinr.estimators.design_effect` divides the variance between the draws' rates by the mean variance within them, $p_j (1 - p_j) / A_j$ for a draw with rate $p_j$ over $A_j$ accepted shots.
Near 1, the draws differ no more than independent shots would, and a Wilson interval on the pooled shots of every draw would be honest.
Above 1, the pooled interval is too narrow, by about the square root of the design effect.
A draw that saw no errors adds nothing to the variance within draws, so the ratio is biased upward when errors are rare.

The effect depends on the setting, so it is measured rather than assumed.
At $n = 3$, with 30 random gates, $t = 2$, $r = 4$ and {func}`clinr.noise_models.uniform_noise` at $2 \times 10^{-3}$, 40 uniform draws of $2 \times 10^4$ shots give a design effect of {glue:text}`uniform_effect:.1f`, and an interval over draws {glue:text}`uniform_widening:.1f` times as wide as the pooled Wilson interval.
The restricted family, whose checks differ more from one another, gives {glue:text}`restricted_effect:.1f` and {glue:text}`restricted_widening:.1f` times.
The fresh and kept logical error rates still agree closely: {glue:text}`uniform_fresh:.5f` against {glue:text}`uniform_kept:.5f` with uniform checks, and {glue:text}`restricted_fresh:.5f` against {glue:text}`restricted_kept:.5f` with restricted ones.

The paper's own numerics (Sec. V) reuse each set of checks for a batch of $10^3$ shots, for expedience, which the authors justify for random Clifford circuits, since those scramble faults quickly, and expect may fail for shallower structured circuits.
The design effect measures the clustering that such reuse introduces.
