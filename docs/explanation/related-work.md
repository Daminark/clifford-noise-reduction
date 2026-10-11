# Related work

Delfosse and Tham introduced CliNR in [arXiv:2407.06583](https://arxiv.org/abs/2407.06583), published in Physical Review Letters 134, 090603 (2025), the paper this package implements.
We place the package among the work that followed: three papers that extend the scheme, then one study on hardware, and we close with how these works realize the scheme's restarts.

## Extensions of the scheme

**Optimized CliNR** (Tham and Delfosse, [arXiv:2504.13356](https://arxiv.org/abs/2504.13356), published in Quantum 9, 1829 (2025)) chooses which stabilizers to measure, the *verification sequence*.
It searches over sequences by tabu search, either on the logical error rate directly or first on a proxy for it that is cheaper to evaluate, whose symmetries shrink the search space.
It also runs the CZNR variant of the scheme, for circuits of `CZ` gates, on a 36-qubit trapped-ion computer without mid-circuit measurements, where it reaches breakeven.
Choosing the stabilizers is a different decision from choosing $(t, r)$.
This package draws its checks at random, from the whole stabilizer group or from the restricted family, and {doc}`../how-to/own-checks` shows how to measure checks chosen otherwise.

**Recursive CliNR** (Brodutch, Baimetov, Tham and Delfosse, [arXiv:2511.22624](https://arxiv.org/abs/2511.22624)) addresses the faults the scheme cannot catch, those in the teleportation.
A CliNR circuit is itself a Clifford circuit, so it can be the circuit inside a larger one, and nesting to depth $D$ uses $(2D + 1)n + 1$ qubits.
The paper shows that a recursive implementation has a vanishing logical error rate whenever $np \to 0$, where the scheme here needs $snp^2 \to 0$.
The paper estimates logical error rates and overheads with a Markov model over the preparation, verification and injection of each resource, compared against Stim simulations.
This package implements the scheme without nesting, a tree of depth one in their terms.

**Distributed CliNR** (Dobbs, Delfosse and Brodutch, [arXiv:2512.10693](https://arxiv.org/abs/2512.10693)) prepares and checks the resource of each chunk on a different processor, in parallel, on a ring of processors whose links produce Bell pairs slowly.
In circuit-level simulations of 85-qubit random Clifford circuits over four processors, it reaches a lower logical error rate and a shorter depth than both the direct implementation and CliNR on one processor.
Its depth analysis restarts a resource only after all $r$ checks, which the authors note simplifies the proof without much added depth, since $r$ is small.
This package makes the same choice for its cost model ({doc}`../adr/0001-run-every-check-before-deciding`).

## On hardware

Brown and coauthors, from qBraid, IonQ and NVIDIA ([arXiv:2605.06792](https://arxiv.org/abs/2605.06792)), apply CliNR to an encoded Trotter step of a fermionic Hamiltonian on a barium trapped-ion system, with the checks read out by mid-circuit measurement, and report a logical error rate up to 54% lower than direct execution.
They also show, in simulation, that a machine-learning model can pick pairs of stabilizers that outperform random pairs.

On their hardware, deferring a check's readout to the end of the circuit removed most of the benefit.
Their calibrated simulation, whose noise is Pauli and whose idle errors on the ancillas commute with the measurement, showed no such difference, and they attribute the hardware effect to noise their model leaves out, such as crosstalk or other non-Pauli effects.
In a model where waiting adds no noise that matters, when a check is read out cannot change the statistics, which is what lets post-selection stand in for restarts here ({doc}`the-scheme`).
Their hardware result marks that equivalence as a property of the noise model.

## How restarts are realized

Neither hardware experiment restarts a rejected resource in real time.
The Optimized CliNR experiment runs every circuit to completion, discards the shots whose checks fired, and infers the restart rate as the fraction discarded.
The study of Brown and coauthors measures the checks mid-circuit but still post-selects, and names real-time feed-forward as a next step.
Recursive CliNR's Stim simulations, by contrast, run each circuit to completion with every restart included.
This package post-selects, as the experiments do, and recovers the expected number of attempts from each chunk's acceptance ({doc}`estimation`).
