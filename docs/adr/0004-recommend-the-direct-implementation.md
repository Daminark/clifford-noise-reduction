# 4. Recommend the direct implementation when its bound is least

## Status

Accepted.

## Context

`bounds.choose_tr` searched only the pairs `(t, r)`, and raised an error when none met the overhead ceiling.
It so recommended CliNR even where every pair's bound exceeded the direct implementation's, and measured logical error rates confirmed that some of those picks do worse than running the circuit directly.
The direct implementation always meets a ceiling of at least 1, its own overhead.

## Decision

`bounds.choose_tr` compares the pairs with the direct implementation, whose bound is the probability that some gate of the circuit faults, and returns `None` when that bound is least.
Ties go to the direct implementation.
A ceiling below 1 is an error, and a ceiling that no pair meets is not.

## Consequences

Callers handle a `None` return, and an overhead ceiling now always has an answer.
The recommendation is no worse than the direct implementation by the bounds, but only by the bounds: where the bounds with checks are loose, as for the restricted family, the direct implementation can be recommended even though some pair does better when measured.
