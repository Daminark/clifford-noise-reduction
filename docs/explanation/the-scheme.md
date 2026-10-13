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

# The scheme

Clifford Noise Reduction (CliNR) lowers the logical error rate of a Clifford circuit by applying it through gate teleportation, with the resource state checked before the data touches it.
We describe the scheme of Delfosse and Tham ([arXiv:2407.06583](https://arxiv.org/abs/2407.06583)) in the order it is built.
We first place it among the ways of handling noise, then describe how gate teleportation carries a Clifford circuit and what one attempt at a chunk does, and close with why the package can sample the scheme with post-selection in place of restarts.
What the checks catch, and the bounds that follow, are the subject of {doc}`checks-and-bounds`.

```{code-cell} python
---
tags: [remove-cell]
---
# Builds the diagrams on this page and checks that each circuit does what the
# text says; the page fails to build if one does not.
import base64

import stim
from IPython.display import HTML


def show(circuit, alt):
    svg = str(circuit.diagram("timeline-svg"))
    width = int(svg.split('viewBox="0 0 ', 1)[1].split()[0])
    encoded = base64.b64encode(svg.encode()).decode()
    return HTML(
        f'<img src="data:image/svg+xml;base64,{encoded}" alt="{alt}" '
        f'style="width: 100%; max-width: {width}px; background: white; '
        'padding: 0.5em; box-sizing: border-box">'
    )


teleport = stim.Circuit(
    """
    H 1
    CX 1 2
    CX 0 1
    H 0
    M 0 1
    CX rec[-1] 2
    CZ rec[-2] 2
    """
)
for one_qubit in stim.Tableau.iter_all(1):
    sim = stim.TableauSimulator(seed=0)
    sim.do_tableau(one_qubit, [0])
    sim.do_circuit(teleport)
    expected = one_qubit.z_output(0)
    observable = stim.PauliString(3)
    observable[2] = expected[0]
    observable.sign = expected.sign
    assert sim.peek_observable_expectation(observable) == 1

n = 2
chunk = stim.Circuit("H 0\nCX 0 1")
tableau = chunk.to_tableau()
data, bare, image, ancilla = [0, 1], [2, 3], [4, 5], 6


def resource():
    c = stim.Circuit()
    for b, i in zip(bare, image):
        c.append("H", [b])
        c.append("CX", [b, i])
    for instruction in chunk:
        c.append(instruction.name, [image[t.value] for t in instruction.targets_copy()])
    return c


def gate_teleport(check=None, corrections=True):
    c = resource()
    if check is not None:
        c.append("R", [ancilla])
        c.append("H", [ancilla])
        for q, code in zip(bare + image, check):
            if code:
                c.append("C" + "_XYZ"[code], [ancilla, q])
        c.append("H", [ancilla])
        c.append("M", [ancilla])
        c.append("DETECTOR", [stim.target_rec(-1)])
    for d, b in zip(data, bare):
        c.append("CX", [d, b])
    c.append("H", data)
    c.append("M", data + bare)
    if corrections:
        # The outcome of bare qubit i controls C X_i C^dag, that of data
        # qubit i controls C Z_i C^dag.
        for i in range(n):
            for pauli, rec in ((tableau.x_output(i), -n + i), (tableau.z_output(i), -2 * n + i)):
                for q, code in zip(image, pauli):
                    if code:
                        c.append("C" + "_XYZ"[code], [stim.target_rec(rec), q])
    return c


# The check X on q2 times Z on q4 is X_0 (x) C X_0 C^dag, a stabilizer.
check = stim.PauliString("X_") + tableau.x_output(0)
assert check == stim.PauliString("X_Z_")
sim = stim.TableauSimulator()
sim.do_circuit(resource())
padded = stim.PauliString(7)
for q, code in zip(bare + image, check):
    padded[q] = code
assert sim.peek_observable_expectation(padded) == 1

for k, two_qubit in enumerate(stim.Tableau.iter_all(2, unsigned=True)):
    if k % 15:
        continue
    for circuit in (gate_teleport(), gate_teleport(check)):
        sim = stim.TableauSimulator(seed=k)
        sim.do_tableau(two_qubit, data)
        sim.do_circuit(circuit)
        if circuit.num_detectors:
            assert not sim.current_measurement_record()[-5]
        whole = two_qubit.then(tableau)
        for i in range(n):
            expected = whole.z_output(i)
            observable = stim.PauliString(7)
            for q, code in zip(image, expected):
                observable[q] = code
            observable.sign = expected.sign
            assert sim.peek_observable_expectation(observable) == 1
```

## Where CliNR sits

Quantum error correction encodes each logical qubit in many physical qubits and corrects faults as the computation runs.
Below a threshold error rate it suppresses the logical error rate as far as needed, at a cost of tens or hundreds of physical qubits per logical qubit.
Quantum error mitigation needs few or no extra qubits, and instead post-processes data from many runs of variants of the circuit; its sampling cost generally grows exponentially with the size of the circuit.

Coherent Pauli checks measure a few parity checks across the circuit and discard the run when one fires.
More checks catch more faults, but every discarded run is a restart from scratch, and the sampling overhead grows exponentially with the number of checks.

CliNR avoids restarting from scratch by moving the checks off the data: it is a teleported version of coherent Pauli checks (the paper's Sec. I).
Its checks act on a resource state prepared apart from the data, so a check that fires discards only the resource, which is prepared again while the data waits, and no run is discarded.
The scheme is not fault-tolerant, since faults during the teleportation reach the data unchecked, but it is cheap: $3n + 1$ qubits for an $n$-qubit circuit.
For a circuit of $s$ gates at physical error rate $p$, the direct implementation has a vanishing logical error rate when $sp \to 0$.
The paper's Theorem 2 gives CliNR one when $snp^2 \to 0$, using on average $2s + o(s)$ gates, so CliNR reaches circuits with $ns = o(1/p^2)$, where the direct implementation needs $s = o(1/p)$.

The scheme rests on gate teleportation, so we begin with teleportation itself.

## Teleportation

Teleportation moves a qubit's state onto another qubit, using a Bell pair $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$ and two measurements.
In the circuit below, `q0` holds the input $|\psi\rangle$, and the first two gates prepare `q1` and `q2` as a Bell pair.
A Bell measurement of `q0` and `q1`, which is a `CX`, an `H` and two measurements, leaves `q2` in $P|\psi\rangle$ for a Pauli $P$ that the two outcomes determine.
The two classically controlled gates at the end apply $P$ again, which leaves `q2` in $|\psi\rangle$ up to a global phase.

```{code-cell} python
---
tags: [remove-input]
---
show(teleport, "Timeline diagram of teleporting q0 onto q2 through a Bell pair on q1 and q2")
```

## Gate teleportation

Gate teleportation applies a circuit $C$ to the state in the course of teleporting it.
Take $n$ Bell pairs, call one qubit of each pair its *bare* half and the other its *image* half, and apply $C$ to the image halves.
The result is the *resource state* $(I \otimes C)|\Phi^+\rangle^{\otimes n}$.
Teleporting $n$ data qubits through it, by a Bell measurement of each data qubit with its bare partner, leaves the image half in

$$C P|\psi\rangle = (C P C^\dagger) C|\psi\rangle$$

for a Pauli $P$ that the outcomes determine, since $C$ acts on the image half only and so commutes with the measurements.
What remains is to undo $Q = C P C^\dagger$.
When $C$ is a Clifford circuit, $Q$ is again a Pauli, so it is computed from the outcomes in classical software and undone by at most $n$ single-qubit Pauli gates.
When $C$ is not, $Q$ is generally not a Pauli, and undoing it takes a longer sequence of noisy gates; this is why the scheme is for Clifford circuits.

Notice that all of $C$ goes into the resource before the data arrives.
The resource can therefore be checked, and if need be thrown away, without disturbing the data, which is what CliNR exploits.

The circuit below gate-teleports $C$ = `H 0`, `CX 0 1` at $n = 2$, with the data on `q0` and `q1`, the bare half on `q2` and `q3`, and the image half on `q4` and `q5`.
The first gates prepare the resource and the middle ones are the Bell measurements.
The classically controlled gates at the end undo $Q$: the outcome of bare qubit $i$ controls $C X_i C^\dagger$, and that of data qubit $i$ controls $C Z_i C^\dagger$.

```{code-cell} python
---
tags: [remove-input]
---
show(gate_teleport(), "Timeline diagram of gate-teleporting H 0, CX 0 1 from q0 and q1 onto q4 and q5")
```

The package never applies the correction: it tracks $Q$ as a Pauli frame and folds it into the readout ({doc}`../adr/0002-deliver-the-output-up-to-a-pauli-frame`).
We now insert the checks.

## One attempt at a chunk

CliNR$(1, r)$ applies the whole circuit as one chunk.
An attempt at it runs as follows.

1. Prepare the resource state on $2n$ qubits.
1. Measure $r$ of its stabilizers, the *checks*, drawn at random.
   Each check uses one ancilla, which is prepared in $|+\rangle$, controls the check's Pauli on each qubit of its support, and is measured in the $X$ basis.
   Without a fault, every check has the outcome $+1$.
1. If any check has the outcome $-1$, discard the resource and start a new attempt, with new checks.
   The data waits untouched.
1. Otherwise, teleport the data through the resource.

The data, the resource and the ancilla make $3n + 1$ qubits.
A fault while preparing the resource or measuring a check can be caught by a later check.
A fault in the teleportation, $4n$ operations on the data and the bare half, cannot.
{doc}`checks-and-bounds` works out how likely a fault is to escape the checks, and the bounds that follow.

The circuit below is one attempt at the chunk above, with $r = 1$.
Its check is the propagated $XX$ stabilizer of the first pair, $X_0 \otimes C X_0 C^\dagger$, which for this $C$ is $X$ on `q2` times $Z$ on `q4`.
The ancilla `q6` measures it, and the `DETECTOR` marks the outcome as one that is $+1$ unless a fault occurs.
The corrections are left out, as in the package.

```{code-cell} python
---
tags: [remove-input]
---
show(
    gate_teleport(check, corrections=False),
    "Timeline diagram of one attempt: the resource on q2 to q5, a check measured by ancilla q6, then the Bell measurements",
)
```

The paper ends an attempt at the first check that fires.
The package measures all $r$ before deciding, which accepts the same attempts ({doc}`../adr/0001-run-every-check-before-deciding`).
One chunk suffices for a short circuit, but in a long one most attempts hold a fault and the expected number of attempts grows quickly, which leads us to split the circuit.

## Many chunks

CliNR$(t, r)$ splits $C$ into $t$ consecutive chunks $C_1, \dots, C_t$ and applies each by its own attempts, through its own resource state $(I \otimes C_k)|\Phi^+\rangle^{\otimes n}$.
After chunk $k$, the data sits on that chunk's image half, and the qubits that held the data and the bare half have been measured.
They are reset to hold the next resource, so the scheme uses $3n + 1$ qubits for any $t$.
The circuits the package builds add $n$ noiseless reference qubits, which read out the channel the circuit implements and are not part of the scheme.

The two parameters pull against each other.

- More chunks make each resource shorter, so an attempt is less likely to hold a fault and passes more often.
  But each chunk adds a teleportation, whose $4n$ operations go unchecked, and $r$ checks of its own.
- More checks let fewer faults in the resource escape, since each check misses a damaging fault with probability at most $q < 1$.
  But each check is a gadget that can fault itself, and each lowers the probability that an attempt passes, which raises the number of attempts.

{func}`clinr.bounds.choose_tr` searches the pairs $(t, r)$ and the direct implementation for the least bound on the logical error rate, under a cap on the bound on the gate overhead; the {doc}`tutorial <../tutorial/getting-started>` runs it.

## Why post-selection stands in for restarts

The package does not loop.
It builds each chunk once, with a detector for each check, samples the whole circuit with Stim, and keeps the shots in which no detector fires.
The kept shots have the output distribution of the protocol, for three reasons.

- A chunk's detectors see only the faults in that chunk's resource and checks, since the resource and the ancilla start freshly reset and the data reaches the resource only after the checks.
- Faults at different locations are independent.
  So keeping the shots in which every detector passes conditions each chunk's resource and check faults on that chunk passing, independently across chunks, and leaves the teleportation faults as they are.
- In the protocol, the attempt that is used is the first to pass.
  Attempts are independent and identically distributed, so its faults are distributed as those of one attempt conditioned on passing, which is what the kept shots sample.

What post-selection loses is the number of attempts.
At chunk $k$ it is geometric with mean $1/a_k$, where $a_k$ is the probability that an attempt passes; the fraction of shots in which that chunk's detectors pass estimates $a_k$, and {func}`clinr.estimators.gate_overhead` turns it into a cost.

Two assumptions carry this argument.
The first is that idle qubits are noiseless, as in the paper's Theorem 1, so the data's wait costs nothing.
With idle noise, the number of attempts would matter to the data, and post-selection would not capture it.

The second concerns the checks.
One estimate samples one draw of the checks, so its kept shots follow a protocol that keeps that draw for every attempt.
Under new checks for every attempt, the draw in use is the one that passed, and a draw that passes more often is more likely to be it.
With $a$ the probability that a draw passes and $p$ the logical error rate on the shots it accepts, the protocol's logical error rate is $\mathbb{E}[ap]/\mathbb{E}[a]$ over draws, where keeping one draw gives $\mathbb{E}[p]$.
{func}`clinr.estimators.logical_error_rate` and {func}`clinr.estimators.gate_overhead_over_draws` estimate the protocol's rates under new checks from many draws by default, and under kept checks with `checks="kept"`.
