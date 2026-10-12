# 5. Korn-Graubard intervals for the logical error rate over draws

## Status

Accepted.

## Context

The shots of one estimate share one draw of the checks, and the draws are independent, so the shots over many draws are a clustered sample.
The logical error rate over draws was given a Student t interval, the estimate plus or minus a t quantile times a standard error estimated from the spread between draws.
That form assumes a nearly normal estimate whose variance is estimated well, and a logical error rate near zero meets neither: its distribution is lopsided, and draws that see few errors also report a small variance.
When no draw sees an error, the interval collapses to the point 0.
In a simulation with about one expected error over 10 draws, it covered the true rate in 62% of trials.

## Decision

`estimators.logical_error_rate` returns the Korn-Graubard interval for a proportion from a clustered sample (Korn and Graubard, Survey Methodology 24, 193 (1998)), as the NCHS data presentation standards for proportions (Parker et al., Vital and Health Statistics 2(175), 2017) specify it.
The variance over draws gives an effective number of shots, scaled by the ratio of the t quantiles at the shots' and the draws' degrees of freedom and capped at the shots accepted, and the interval is the Clopper-Pearson interval at that number.
When the variance carries no information about the clustering, at an estimate of 0 or 1 or with draws that do not differ, the effective number is the shots accepted.
The gate overhead keeps the Student t interval: it is a ratio of means far from any boundary, whose variance does not depend on its value.

## Consequences

The interval never collapses and stays within [0, 1], and with few draws or few errors it widens rather than undercovers; at about one expected error over 10 draws it covers in about 99% of trials, and with plentiful errors in about 95%.
Clopper-Pearson is conservative, so with few effective shots the interval is wider than nominal.
The point estimates are unchanged, so no result moves; the intervals quoted on the results pages change slightly.

## Note, 2026-10-12

The reason given above for keeping the Student t interval on the gate overhead is wrong: a ratio's variance does depend on its value.
What justifies the interval is that each chunk's mean acceptance, a denominator, is estimated precisely relative to its size, so the ratio is nearly linear in the means over draws; Cochran's rule (Sampling Techniques, 3rd ed., ch. 6) takes this to hold when each mean acceptance has a coefficient of variation below about 0.1.
The correction comes from the statistics audit of the experiments, which also asked that the interval's coverage be measured with its Monte Carlo errors; the decision is unchanged.
That measurement is in {doc}`../explanation/coverage`.
