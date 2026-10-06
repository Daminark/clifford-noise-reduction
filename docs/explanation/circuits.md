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
This page explains what those circuits contain beyond the scheme of {doc}`the-scheme`: the harness that reads out whether a circuit implemented $C$, the registers, the Pauli frame that stands in for the corrections, and the noise model.

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

A circuit implements $C$ when it does so on every input, so one input state cannot test it.
The circuits test every input at once with the Choi state.
Before anything else, each of the $n$ input qubits is joined in a Bell pair to a qubit of a *reference* block.
If the circuit then applies $C$, the reference and the output end in $(I \otimes C)|\Phi^+\rangle^{\otimes n}$, the state the resource of {doc}`the-scheme` is in, with the reference as its bare half.
The circuit ends by measuring that state's $2n$ stabilizer generators with `MPP`, each declared an observable, and without noise all of them come out $+1$.

Under Pauli noise, the faults of a shot propagate to a Pauli error $E$ on the output, after $C$, and an observable flips exactly when $E$ anticommutes with it.
If $E$ commuted with all $2n$ generators, it would lie in their group up to a phase.
The only element of that group acting on the output alone is the identity, since the reference parts of the generators, the $X_i$ and $Z_i$, are independent.
So the observables catch every nontrivial $E$, and the logical error rate, the probability given acceptance that some observable flips, is the probability that $E$ is not the identity.
Equivalently, it is one minus the entanglement fidelity of the implemented channel with $C$.

The reference block and the Bell pairs that join it to the input are noiseless, since they are apparatus rather than part of the scheme.
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

The builder never applies the correction $Q$ ({doc}`../adr/0002-deliver-the-output-up-to-a-pauli-frame`), and it could not write $Q$ into the circuit, since $Q$ depends on the outcomes of each shot.
What is fixed is how it depends on them: each Bell outcome $o_j$, with $j$ its measurement record, contributes a fixed Pauli $P_j$, and $Q = \prod_j P_j^{o_j}$.
The builder keeps the *frame*, the map from each record $j$ to its $P_j$, written on the $n$ logical qubits.

When chunk $k$ is teleported, the earlier entries sit on the data before $C_k$ acts, so each is conjugated by $C_k$.
Conjugation is a homomorphism, so

$$C_k Q C_k^\dagger = \prod_j (C_k P_j C_k^\dagger)^{o_j},$$

and conjugating each entry updates the whole map without the outcomes.
Then the chunk's own $2n$ records join, already expressed after $C_k$: the bare outcome $i$ with $C_k X_i C_k^\dagger$, and the data outcome $i$ with $C_k Z_i C_k^\dagger$.

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

At readout, measuring an observable $O$ on $Q C|\psi\rangle$ gives its outcome on $C|\psi\rangle$, flipped exactly when $Q$ anticommutes with $O$.
Since $Q = \prod_j P_j^{o_j}$, the flip is the XOR of the outcomes $o_j$ whose $P_j$ anticommutes with $O$.
`OBSERVABLE_INCLUDE` XORs its targets, so including those records with the `MPP` takes $Q$ out of the readout.
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

### What a live state costs

Tracking the frame leaves the output in $Q C|\psi\rangle$, with $Q$ known.
A caller who needs $C|\psi\rangle$ itself, before a non-Clifford gate for example, applies $Q$ as $n$ single-qubit Pauli gates.
The paper applies the correction after every chunk and charges its $n$ gates each time, $5n$ operations per teleportation against our $4n$.

Applied once at the end, the correction costs, in distribution, one `DEPOLARIZE1` at $3p_1/4$ on each output qubit.
The frame resolved at the end is uniform over the $4^n$ Paulis and independent of every fault and every check outcome.
The last chunk's $2n$ Bell outcomes map one to one onto the $n$-qubit Paulis, and each outcome is its ideal value, uniform and independent of the faults, XOR a function of the faults.
So each qubit's gate is nontrivial with probability $3/4$, independently of everything else, and a depolarizing fault at $p_1$ after a gate applied with probability $3/4$ is a depolarizing fault at $3p_1/4$.
A simulation that rebuilt the frame shot by shot from sampled records agrees, both on the uniformity and on the logical error rate.

Tracking the frame removes only the $n$ Pauli gates of the correction.
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
