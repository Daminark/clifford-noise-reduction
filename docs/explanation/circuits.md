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

# How the circuits are built

{func}`clinr.protocol.build_clinr_tr` and {func}`clinr.protocol.build_direct` return the Stim circuits that the estimators sample.
This page explains what those circuits contain beyond the scheme of {doc}`the-scheme`: the harness that reads out whether a circuit implemented $C$, the registers, the Pauli frame that tracks the corrections, and the noise model.

````{code-cell} python
---
tags: [remove-cell]
---
# Builds the figures and excerpts on this page and checks the claims made
# about them; the page fails to build if one does not hold.
import base64
import inspect
import itertools
import textwrap

import numpy as np
import stim
from IPython.display import HTML, Markdown
from myst_nb import glue

from clinr import protocol
from clinr.noise_models import NoiseModel


def show(circuit, alt):
    svg = str(circuit.diagram("timeline-svg"))
    width = int(svg.split('viewBox="0 0 ', 1)[1].split()[0])
    encoded = base64.b64encode(svg.encode()).decode()
    # These diagrams are too wide to shrink to the column and stay legible,
    # so they keep 80% of their size and scroll sideways.
    return HTML(
        '<div style="overflow-x: auto">'
        f'<img src="data:image/svg+xml;base64,{encoded}" alt="{alt}" '
        f'style="width: {0.8 * width:.0f}px; max-width: none; '
        'background: white; padding: 0.5em; box-sizing: border-box">'
        "</div>"
    )


def excerpt(function, code):
    assert code in inspect.getsource(function), "excerpt drifted from source"
    return Markdown(f"```python\n{textwrap.dedent(code)}```")


n = 2
chunk = stim.Circuit("H 0\nCX 0 1")
silent = NoiseModel(p1=0.0, p2=0.0, p_m=0.0, p_prep=0.0)
direct = protocol.build_direct(chunk, n, silent)
clinr = protocol.build_clinr_tr(chunk, n, [[stim.PauliString("X_Z_")]], silent)

# Completeness of the readout: every nontrivial Pauli on the output block
# anticommutes with some Choi generator.
b1, _, _, ref, _ = protocol._blocks(n)
gens = protocol.stabilizer_gens(chunk.to_tableau(), ref, b1, 4 * n + 1)
for codes in itertools.product(range(4), repeat=n):
    if any(codes):
        error = stim.PauliString(4 * n + 1)
        for qubit, code in zip(b1, codes):
            error[qubit] = code
        assert any(not error.commutes(gen) for gen in gens)

# Exactness at zero noise, and what stripping the frame does to it.
for circuit in (direct, clinr):
    assert not circuit.compile_detector_sampler(seed=0).sample(1000, separate_observables=True)[1].any()
stripped = stim.Circuit()
for instruction in clinr:
    if instruction.name == "OBSERVABLE_INCLUDE":
        stripped.append(instruction.name, instruction.targets_copy()[:1], instruction.gate_args_copy())
    else:
        stripped.append(instruction)
_, flips = stripped.compile_detector_sampler(seed=0).sample(100_000, separate_observables=True)
glue("stripped", float(flips.any(axis=1).mean()), display=False)
````

## What the logical error rate measures

A circuit implements $C$ only if it does so on every input, so no single input state can test it.
The circuits test every input at once through the Choi-Jamiołkowski isomorphism, which turns the question of which operation a circuit applies into the question of which state it prepares.

### The Choi state

For an operator $A$ on $n$ qubits, let

$$|A\rangle\rangle = (I \otimes A)|\Phi^+\rangle^{\otimes n} = 2^{-n/2} \sum_i |i\rangle \otimes A|i\rangle,$$

summing over the computational basis: the result of applying $A$ to one half of $n$ Bell pairs and leaving the other half, the *reference*, alone.
Its amplitude on $|i\rangle|j\rangle$ is $2^{-n/2} \langle j|A|i\rangle$, so $|A\rangle\rangle$ is the matrix of $A$ written out as a vector, and $A \mapsto |A\rangle\rangle$ is a linear bijection from operators on $n$ qubits to vectors on $2n$.
It also turns the Hilbert-Schmidt inner product into the inner product of states:

$$\langle\langle A|B\rangle\rangle = 2^{-n} \operatorname{Tr}(A^\dagger B).$$

Since the map is injective, two unitaries prepare the same state up to a phase exactly when they are equal up to a phase.
So a circuit applies $C$ exactly when it takes $|\Phi^+\rangle^{\otimes n}$ to the *Choi state* of $C$, $|\psi_C\rangle = (I \otimes C)|\Phi^+\rangle^{\otimes n}$, which is the state the resource of {doc}`the-scheme` is in, with the reference as its bare half.
One input settles every input.

### Reading it out with stabilizers

The harness prepares the Bell pairs and tests the state.
Before anything else, it joins each of the $n$ input qubits in a Bell pair to a qubit of a reference block; the circuit then runs on the input qubits; and the harness ends by measuring, with `MPP`, the $2n$ generators of the stabilizer group $S$ of $|\psi_C\rangle$, each declared an observable.
These are the propagated Bell stabilizers $X_i \otimes C X_i C^\dagger$ and $Z_i \otimes C Z_i C^\dagger$, read off the tableau of $C$, as for the resource.

The noise is Pauli, so in each shot the faults propagate to a Pauli error $E$ on the output, after $C$.
The shot applies $EC$, with the Pauli frame below taken into account, and leaves $(I \otimes E)|\psi_C\rangle$.
If $E$ is the identity, every generator reads $+1$.
If not, $I \otimes E$ is not in $S$, since the only element of $S$ acting on the output alone is the identity: the reference parts of the generators, the $X_i$ and $Z_i$, are independent.
A stabilizer state's group is its own normalizer in the Pauli group, up to phases, so $I \otimes E$ anticommutes with some generator, and that generator reads $-1$.
So some observable flips exactly when $E$ is not the identity, and the logical error rate is the probability, given acceptance, that it is not.
The two cases are as far apart as states can be: $\langle\langle C|EC\rangle\rangle = 2^{-n} \operatorname{Tr}(E) = 0$ for every Pauli $E$ other than the identity.

### In the language of channels

The same isomorphism, stated for density operators, covers noise of every kind.
A noisy implementation applies a channel $\mathcal{E}$, here the mixture over shots of the operations $EC$, and its Choi state is the density operator

$$J(\mathcal{E}) = (\mathcal{I} \otimes \mathcal{E})\left((|\Phi^+\rangle\langle\Phi^+|)^{\otimes n}\right) = 2^{-n} \sum_{i, j} |i\rangle\langle j| \otimes \mathcal{E}(|i\rangle\langle j|),$$

which lists the action of $\mathcal{E}$ on the matrix units $|i\rangle\langle j|$, as $|A\rangle\rangle$ lists the matrix of $A$.
The map $\mathcal{E} \mapsto J(\mathcal{E})$ is again a linear bijection.
It takes completely positive maps to positive operators, and channels to density operators whose reduced state on the reference is maximally mixed; this correspondence is the channel-state duality.
The unitary channel $\rho \mapsto C \rho C^\dagger$ goes to $|\psi_C\rangle\langle\psi_C|$.

The projector onto $|\psi_C\rangle$ is $\prod_i (I + g_i)/2$ over the generators $g_i$, so for any channel the probability that every generator reads $+1$ is $\langle\psi_C| J(\mathcal{E}) |\psi_C\rangle$, the *entanglement fidelity* of $\mathcal{E}$ with $C$.
The logical error rate is one minus the entanglement fidelity; for the Pauli mixture here, that is the probability that $E$ is not the identity, as above.

The reference block and the Bell pairs that join it to the input are noiseless.
They are apparatus for measuring the logical error rate, not part of the scheme, and noise in them would be charged to the circuit under test.
Both builders use the same harness, so CliNR and the direct implementation are measured the same way.

## The registers

| Block | Qubits     | Role                              |
| ----- | ---------- | --------------------------------- |
| `b1`  | $[0, n)$   | the data input of the first chunk |
| `b2`  | $[n, 2n)$  | the bare half of each resource    |
| `b3`  | $[2n, 3n)$ | the image half of each resource   |
| `ref` | $[3n, 4n)$ | the Choi reference                |
| `anc` | $4n$       | the ancilla of every check        |

For each chunk, the circuit prepares the resource on `b2` and `b3`, measures the checks with `anc`, one detector per check, and Bell-measures the data block against `b2`.
The output lands on `b3`, so the builder swaps the names of the data block and `b3`, with no gates: the old data block, just measured, is reset to hold the next resource's image half.
The output ends on `b3` when $t$ is odd and on `b1` when $t$ is even.
The circuits use $4n + 1$ qubits, where the scheme uses $3n + 1$; the reference makes the difference.

Below are both circuits at $n = 2$ for the $C$ of {doc}`the-scheme`, drawn without noise so that no noise channels crowd them.
First the direct implementation, which applies $C$ to `q0` and `q1`:

```{code-cell} python
---
tags: [remove-input]
---
show(direct, "Timeline diagram of the direct implementation: Choi pairs on q6 and q7, C on q0 and q1, then four MPP observables")
```

Then CliNR$(1, 1)$, with the check $X$ on `q2` times $Z$ on `q4` measured by `q8`:

```{code-cell} python
---
tags: [remove-input]
---
show(clinr, "Timeline diagram of CliNR(1, 1): Choi pairs, the resource on q2 to q5, one check by ancilla q8, the Bell measurements, then four MPP observables")
```

Both open with the Choi pairs on `q6` and `q7` and close with the four `MPP`s of the readout.
In the second, each observable also includes some of the Bell outcomes: that is the Pauli frame.

## The Pauli frame

The builder handles the corrections with a Pauli frame, as stabilizer simulations and error-correction control stacks commonly do: it never applies them, and tracks classically what they are ({doc}`../adr/0002-deliver-the-output-up-to-a-pauli-frame`).

Write $Q_k$ for the correction of chunk $k$, a Pauli fixed by that chunk's $2n$ Bell outcomes.
Left unapplied, it stays on the data: after the first chunk the data holds $Q_1 C_1|\psi\rangle$ in place of $C_1|\psi\rangle$.
The second chunk teleports that state and applies $C_2$, which leaves

$$Q_2 C_2 Q_1 C_1|\psi\rangle = Q_2 (C_2 Q_1 C_2^\dagger) C_2 C_1|\psi\rangle,$$

and $C_2 Q_1 C_2^\dagger$ is again a Pauli, since $C_2$ is a Clifford circuit.
After all $t$ chunks the output is $Q C|\psi\rangle$, with

$$Q = \prod_{k=1}^{t} C_{>k} Q_k C_{>k}^\dagger, \qquad C_{>k} = C_t \cdots C_{k+1},$$

up to a phase: each chunk's correction, carried through the chunks after it.

### What the builder records

$Q$ depends on the outcomes of each shot, but how it depends on them is fixed when the circuit is built.
By {doc}`the-scheme`, $Q_k = \prod_i (C_k X_i C_k^\dagger)^{b_i} (C_k Z_i C_k^\dagger)^{d_i}$ up to a phase, with $b_i$ the outcome of bare qubit $i$ and $d_i$ that of data qubit $i$.
So $Q = \prod_j P_j^{o_j}$ over every Bell outcome $o_j$, with $j$ its measurement record and $P_j$ a fixed Pauli.
The builder keeps the *frame*, the map from each record $j$ to its $P_j$, written on the $n$ logical qubits; in a shot, $Q$ is the product of the $P_j$ whose outcomes are 1.

The frame is built chunk by chunk.
When chunk $k$ is teleported, the entries already in the frame are conjugated by $C_k$.
Conjugation is a homomorphism, so

$$C_k \left(\prod_j P_j^{o_j}\right) C_k^\dagger = \prod_j (C_k P_j C_k^\dagger)^{o_j},$$

and conjugating each entry carries the whole map through $C_k$ without the outcomes.
Then the chunk's own $2n$ records join: the bare outcome $i$ with $C_k X_i C_k^\dagger$, and the data outcome $i$ with $C_k Z_i C_k^\dagger$.

```{code-cell} python
---
tags: [remove-input]
---
excerpt(protocol.build_clinr_tr, """\
        frame = {index: tableau(pauli) for index, pauli in frame.items()}
        base = out.num_measurements
        _measure_noisy(out, data_in, noise)  # m0_i at index base + i
        _measure_noisy(out, b2, noise)  # m1_i at index base + n + i
        for i in range(n):
            frame[base + n + i] = tableau.x_output(i)
            frame[base + i] = tableau.z_output(i)
""")
```

What the frame is used for depends on what the output is for: a Pauli measurement, or a live state.

### Reading out a Pauli measurement

When the output is measured in a Pauli basis, as here and as is usual in error correction, stabilizer simulation and teleportation-based circuits, $Q$ never has to be applied.
Measuring a Pauli $O$ on $Q C|\psi\rangle$ gives its outcome on $C|\psi\rangle$, flipped exactly when $Q$ anticommutes with $O$.
Since $Q = \prod_j P_j^{o_j}$, the flip is the XOR of the outcomes $o_j$ whose $P_j$ anticommutes with $O$.
`OBSERVABLE_INCLUDE` XORs its targets, so including those records with each `MPP` of the readout corrects its outcome.
The builder reads the output part of each Choi generator back into logical indexing, and turns each record's absolute index into Stim's lookback from the current end of the record:

```{code-cell} python
---
tags: [remove-input]
---
excerpt(protocol.build_clinr_tr, """\
        conj = stim.PauliString([gen[qubit] for qubit in data_in])
        records = [stim.target_rec(-1)]
        for index, pauli in frame.items():
            if not pauli.commutes(conj):
                records.append(stim.target_rec(index - out.num_measurements))
        out.append("OBSERVABLE_INCLUDE", records, observable)
""")
```

Every built circuit is exact at zero noise, which the test suite checks, and exactness tests the frame: the pairing of outcomes with Paulis, the order of the conjugation, and the index arithmetic.
Strip the frame from the observables of CliNR$(1, 1)$ above, and some observable flips in {glue:text}`stripped:.1%` of zero-noise shots.

### Producing a live state

When the output must be the state $C|\psi\rangle$ itself, before a non-Clifford gate for example, $Q$ has to be applied, as a single-qubit Pauli gate on each qubit where it acts.
The paper applies each chunk's correction $Q_k$ after that chunk; with the frame, the accumulated $Q$ can instead be applied once, at the end.

Every correction is uniform over the $4^n$ Paulis, independently of the faults.
Chunk $k$'s $2n$ Bell outcomes map one to one onto the $n$-qubit Paulis, and each outcome is its ideal value, uniform and independent of the faults and of the other outcomes, XOR a function of the faults.
So each $Q_k$ is uniform and independent of the faults and of the other corrections, and so is the accumulated $Q$; each acts nontrivially on each qubit with probability $3/4$, independently across qubits.
On hardware, which applies only the nontrivial gates, the two ways cost:

| Correction                         | Pauli gates applied, on average |
| ---------------------------------- | ------------------------------- |
| after every chunk, as in the paper | $3tn/4$                         |
| once at the end                    | $3n/4$                          |

Each gate applied is a single-qubit fault location at $p_1$, and how many are applied does not depend on the faults.
The paper's accounting charges all $n$ gates of every correction, which is its $5n$ operations per teleportation against our $4n$.

Either way, the frame removes only the Pauli gates of the correction.
The $4n$ operations of each Bell measurement stay noisy and unchecked; they are the injection step of the recursive variant of the scheme ([arXiv:2511.22624](https://arxiv.org/abs/2511.22624)), and their faults bound the logical error rate from below there as here.

## The noise model

A {class}`clinr.noise_models.NoiseModel` holds four rates, and the builders place noise as follows:

- a `DEPOLARIZE1` at $p_1$ after each single-qubit gate, and a `DEPOLARIZE2` at $p_2$ after each two-qubit gate;
- each measurement outcome flipped with probability $p_m$;
- each preparation flipped with probability $p_{\mathrm{prep}}$: an `X_ERROR` after `R`, or a `Z_ERROR` after `RX`.

The paper analyzes one rate $p$ and runs its numerics with single-qubit operations ten times less noisy than two-qubit gates; four rates price each kind of operation separately, and {func}`clinr.noise_models.uniform_noise` and {func}`clinr.noise_models.asymmetric_noise` give the paper's two regimes.

Two choices depart from the paper's Sec. II.

- **$|+\rangle$ is a reset and a Hadamard**, two fault locations where the paper charges one.
  This includes the ancilla of each check, so a check of weight $w$ costs $w + 4$ operations against the paper's $w + 3$.
  With `native_plus=True`, $|+\rangle$ is one `RX`, as in the paper.
- **A preparation fault is a flip**, where the paper depolarizes.
  A depolarizing fault after a fresh $|0\rangle$ acts as a flip with probability $2p/3$, since $Z$ leaves $|0\rangle$ unchanged, so charging the flip directly makes the rate in the bounds the probability of a fault that matters.

Idle qubits are noiseless, as in the paper's Theorem 1.
Idle noise needs a schedule, and a faithful one depends on how long each operation takes on the hardware.
A schedule of equal-duration steps is furthest off for the checks, which are $r$ sequential mid-circuit measurements, so it would undercount CliNR's idle time most.
The paper's Sec. V charges idle qubits at $p_1$ without stating its schedule.
The package leaves idle noise out, and {doc}`the-scheme` explains why sampling restarts by post-selection relies on that.

The builders accept any single- or two-qubit unitary gate and count fault locations for the circuit as given; decomposing into a hardware's native gates is the caller's.
