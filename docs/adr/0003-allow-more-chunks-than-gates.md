# 3. Allow more chunks than gates

## Status

Accepted.

## Context

Theorem 1 of arXiv:2407.06583 splits a circuit of `s` gates into `t` chunks.
When `t` exceeds `s`, some chunks are empty, and each of them still pays a resource preparation, its checks and a teleportation.
The proof of the theorem's overhead bound uses `t ceil(s/t) <= 2s`, which fails for `t > 2s`, so the formula does not cover that range; its bound on the logical error rate holds for every `t`.

## Decision

`protocol.chunk_circuit` accepts any `t >= 1` and returns empty chunks when `t > s`, and the bounds price each empty chunk's preparation, checks and teleportation.
No warning is raised: the configuration is valid and correctly priced, and `bounds.choose_tr` never selects it by default.

## Consequences

The tests compare `bounds.overhead_bound` with Theorem 1 only where `t ceil(s/t) <= 2s`.
Beyond it the paper's formula can fall below the true overhead: over 3000 random cases, 1017 of the 1124 outside that range exceed it, and none of the 1876 inside.
