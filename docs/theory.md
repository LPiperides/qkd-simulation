# Theory notes

Derivations of every number the test suite checks and every analytic curve
drawn in the figures. Notation: `p` is the depolarizing probability of one
hop of the channel, `Q` is the quantum bit error rate on the sifted key,
`h2(x) = -x log2 x - (1-x) log2 (1-x)` is the binary entropy.

## 1. States, bases and the channel

Every protocol uses the same four single-qubit states:

| basis | name | bit index 0 | bit index 1 |
|---|---|---|---|
| `+` | computational / Z | `\|0>` | `\|1>` |
| `x` | diagonal / X | `\|+> = (\|0> + \|1>)/sqrt2` | `\|-> = (\|0> - \|1>)/sqrt2` |

A state measured in its own basis returns its index with certainty; measured
in the other basis it returns 0 or 1 with probability 1/2 each.

**Channel.** Each hop applies the depolarizing map `E(rho) = (1-p) rho + p I/2`.
With probability `p` the qubit is replaced by the maximally mixed state, which
returns a uniformly random outcome in *any* basis. Hence

* measured in the preparation basis: flip probability `p/2`;
* measured in the conjugate basis: uniform, independent of `p`.

In the simulator the map is attached to an identity gate that every prepared
circuit ends with. The state `|0>` is prepared with no gates at all, so a
noise model attached to the `x` and `h` gates would let a quarter of the
transmissions through untouched and bias every QBER below.

**Intercept-and-resend.** Eve measures every qubit in a random basis and
re-sends the eigenstate she observed. With probability 1/2 her basis matches
Alice's and the state is forwarded unchanged. Otherwise she forwards a
conjugate-basis eigenstate, which is indistinguishable from the maximally mixed
state as far as any single later measurement is concerned. The attack is
therefore statistically identical to a depolarizing channel with `p = 1/2`,
and every "with Eve" row below is the corresponding noise formula at `p = 1/2`.

## 2. BB84

Alice draws a bit and a basis uniformly; Bob measures in a uniform basis.
Sifting keeps the rounds where the bases agree.

* retention `= 1/2`
* QBER: in a kept round Bob measures in the preparation basis, so `Q = p/2`.
* intercept-and-resend (`p = 1/2`): `Q = 1/4`.
* Eve's agreement with Alice on the sifted key: she is right whenever her basis
  matched (probability 1/2) and half the time otherwise: `1/2 + 1/4 = 3/4`.

## 3. B92

Alice sends `|0>` for bit 0 and `|->` for bit 1. Bob measures in a uniform
basis and keeps a round only when his outcome is impossible for one of the two
states:

| Bob's basis | outcome | ruled out | conclusion |
|---|---|---|---|
| `+` | `\|1>` | `\|0>` | Alice sent `\|->`, bit 1 |
| `x` | `\|+>` | `\|->` | Alice sent `\|0>`, bit 0 |

Take Alice's bit to be 0 (`|0>`); bit 1 is symmetric.

* Bob in `+` (probability 1/2): outcome `|1>` only via the channel, probability
  `p/2`. Conclusive and **wrong**.
* Bob in `x` (probability 1/2): outcome `|+>` with probability 1/2 regardless of
  `p` (the depolarized state is uniform in `x`, the clean state is uniform in
  `x`). Conclusive and **right**.

Per round: `P(right) = 1/4`, `P(wrong) = p/4`, so

* retention `= (1 + p)/4`
* QBER `Q = (p/4) / ((1+p)/4) = p/(1+p)`
* intercept-and-resend: retention `3/8`, `Q = 1/3`.

## 4. SARG04

Alice sends one of the four BB84 states. **The bit is the basis**: Z states
carry bit 0, X states carry bit 1. She announces an unordered pair `{z, x}`
of one Z state and one X state containing the state she sent; the other member
is drawn at random. Bob measures in a uniform basis and is conclusive when his
outcome is orthogonal to one member of the pair:

| Bob's basis | condition | ruled out | conclusion |
|---|---|---|---|
| `+` | outcome is not the announced Z state | the Z state | Alice sent the X state, bit 1 |
| `x` | outcome is not the announced X state | the X state | Alice sent the Z state, bit 0 |

Let Alice's state be `a` in basis `A`, the decoy be `r` in the other basis.

* Bob measures in `A` (probability 1/2): his outcome differs from `a` only via the
  channel, probability `p/2`. Then he rules out `a` and concludes the decoy was
  sent: conclusive and **wrong**.
* Bob measures in the other basis (probability 1/2): his outcome is uniform, so it
  differs from `r` with probability 1/2. Then he rules out `r` and concludes `a`:
  conclusive and **right**.

The bookkeeping is identical to B92: retention `(1 + p)/4`, `Q = p/(1+p)`,
and under intercept-and-resend retention `3/8` with `Q = 1/3`.

**Why the bit convention matters.** If the bit is instead read as the index
of the state inside its basis (so `|0>` and `|+>` both mean 0), Bob's wrong
deductions still land on the correct bit half of the time, because the decoy's
index is random. The apparent error rate under attack drops to
`(1/8 x 1/2) / (3/8) = 1/6`, and the protocol looks twice as tolerant of
eavesdropping as it really is. The regression test in `tests/test_sarg04.py`
computes both conventions on the same transcript.

## 5. Mutual information

Over the sifted key, Alice's bit `A` and Bob's bit `B` form a binary symmetric
channel with crossover `Q`, so `I(A:B) = 1 - h2(Q)`.

Eve's guess `E` is compared with `A` the same way, using the empirical joint
distribution (an inconclusive guess is its own symbol). For BB84 under
intercept-and-resend, `E` agrees with `A` on 3/4 of the sifted rounds, so

`I(A:E) = 1 - h2(1/4) = 0.189 bits = I(A:B)`.

Eve knows exactly as much about the key as Bob does. By the Csiszár-Körner
condition a positive secret-key rate through one-way post-processing requires
`I(A:B) > I(A:E)`, so no key can be distilled from a fully intercepted BB84
link; this is the operational meaning of the 25% QBER signature.

B92 and SARG04 are worse off. Eve applies the same discrimination rule as
Bob, but to a *clean* state, so whenever she is conclusive she is right. On
the rounds Bob keeps she is conclusive half the time (work through the
branches of section 3 with Eve's basis included), which gives
`I(A:E) = 1/2` against Bob's `I(A:B) = 1 - h2(1/3) = 0.082`. The
simulation reproduces both numbers at `p = 0` in
`figures/mutual_information_vs_noise.png`.

## 6. Secret-key rate

Error correction reveals at least `h2(Q)` bits per sifted bit over the public
channel (a practical code reveals `f_ec h2(Q)` with `f_ec >= 1`); privacy
amplification against the optimal collective attack on BB84 costs another
`h2(Q)` (Shor & Preskill, 2000). Per sent qubit:

`r = retention x max(0, 1 - (1 + f_ec) h2(Q))`.

For `f_ec = 1` the rate reaches zero at `h2(Q) = 1/2`, i.e. `Q* = 0.1100`;
for the Cascade-like `f_ec = 1.16`, `Q* = 0.098`.

Translating the threshold into channel noise using the QBER formulas above:

| protocol | `Q(p)` | `p` at which `r = 0` (`f_ec = 1`) |
|---|---|---|
| BB84 | `p/2` | `0.220` |
| B92 / SARG04 | `p/(1+p)` | `0.124` |

The bound is a theorem for BB84. Applying it to B92 and SARG04 gives a fair
like-for-like comparison of how quickly each protocol's raw error rate
consumes the key budget, but their rigorous single-photon security proofs
have different thresholds, so those two curves are labelled heuristic.

## References

* C. H. Bennett and G. Brassard, *Quantum cryptography: public key distribution
  and coin tossing*, Proc. IEEE Int. Conf. on Computers, Systems and Signal
  Processing, 175 (1984).
* C. H. Bennett, *Quantum cryptography using any two nonorthogonal states*,
  Phys. Rev. Lett. 68, 3121 (1992).
* V. Scarani, A. Acín, G. Ribordy and N. Gisin, *Quantum cryptography protocols
  robust against photon number splitting attacks for weak laser pulse
  implementations*, Phys. Rev. Lett. 92, 057901 (2004).
* P. W. Shor and J. Preskill, *Simple proof of security of the BB84 quantum key
  distribution protocol*, Phys. Rev. Lett. 85, 441 (2000).
* I. Csiszár and J. Körner, *Broadcast channels with confidential messages*,
  IEEE Trans. Inf. Theory 24, 339 (1978).
