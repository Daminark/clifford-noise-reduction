# 1. Run every check of an attempt before deciding

## Status

Accepted.

## Context

The paper restarts an attempt at the first check that fails.
Running all `r` checks and then deciding accepts exactly the same attempts, but costs a different number of operations.
Stim builds the protocol as one circuit with a detector per check and post-selects on all of them, which is the run-all version.

## Decision

An attempt at a chunk measures all `r` of its checks, and the chunk is accepted when all of them pass.

## Consequences

Every attempt costs the same number of operations, so the expected cost is that cost over the acceptance probability, exactly rather than as an upper bound, and `estimators.gate_overhead` needs nothing but the measured acceptance.
Stopping at the first failure would save little, at most about 3% of operations in the cases measured, at the price of up to `r` classical round trips per attempt, which this operation-counting model cannot price.
