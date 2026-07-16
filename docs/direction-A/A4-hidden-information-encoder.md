# Sprint A4 — Structured hidden-information encoder

> **Status:** implemented; gate **FAIL** on G4 (2026-07-13). Detailed design for
> Sprint A4 of the [Direction A sprint plan](sprint-plan.md). The implementation
> remains additive and default-off; the flat encoder stays the default.

## 1. Purpose

A4 replaces the learner's undifferentiated MLP-over-104-floats with an encoder
whose architecture matches the information structure of Heat:

- the acting seat's **private hand** is represented by learned card-token
  embeddings and attention pooling;
- own state, track state, phase context and **public opponent state** are encoded
  separately;
- the two streams are fused into the state embedding consumed by either the A0
  masked action head or the A3 dot-product head and by the value head.

The sprint is successful only if this structure improves or matches learning
at equal training budget without leaking an opponent's private cards.

## 2. Existing information boundary

The frozen `encode_observation` layout already provides a useful, auditable
boundary:

| slice | width | visibility | contents |
|---|---:|---|---|
| `0:8` | 8 | private | own hand histogram: S1–S4, Heat, Stress, U0, U5 |
| `8:104` | 96 | own/public | gear, kinematics, own deck summary, whole track, adrenaline/rank, public opponent slots, phase/round context |

Opponent slots contain only presence, relative position, gear, lap delta and
finished status. No opponent hand, draw pile or discard pile is present. A4
therefore consumes the current observation rather than introducing a second
rollout representation: the buffer, collector, PPO update, agent adapters and
legality codec remain unchanged.

The current public contract does **not** yet encode played-card history or
opponent hand counts named in the high-level sprint sketch. Adding those is an
observation-contract change and belongs with A8's full-rules public-history
work. A4 establishes the structured boundary now and its leakage tests make
that later extension safe. This is preferable to deriving public features from
private engine fields inside the network.

## 3. Design

### 3.1 Hand encoder

The hand block is an eight-token histogram with counts divided by seven. The
encoder reconstructs integer multiplicities with `round(x * 7)` and looks up a
learned embedding for each token type.

A learned query scores the eight token embeddings. Token multiplicity enters
attention as `score + log(count)`, which is mathematically equivalent to
expanding a token into repeated identical card slots. Absent tokens are masked.
An explicit learned `empty_hand` vector handles forced/terminal empty-hand
states. The pooled embedding is concatenated with the raw histogram so absolute
counts are retained (attention alone represents relative composition).

This representation is permutation-invariant, distinguishes hand
compositions/counts and uses no physical card IDs — matching the action codec's
value-multiset semantics.

### 3.2 Public encoder and fusion

The remaining 96 values feed a separate MLP using `A0Config.hidden_sizes`. Its
output is concatenated with the attention-pooled hand and raw hand histogram,
then projected through a `tanh` fusion layer to the normal trunk width. Existing
masked and dot-product action heads consume that embedding unchanged; the value
head reads the same fused state.

### 3.3 Selection and compatibility

`A0Config.encoder` selects `"flat"` (existing behavior, default) or
`"structured"`. `build_policy` supports both encoders with both action heads:

| encoder | masked head | dot-product head |
|---|---|---|
| flat | existing A0 baseline | existing A3 arm |
| structured | primary A4 arm | A3 re-read after A4 |

The `act` / `evaluate` signatures and observation/action dimensions remain
frozen. Checkpoints store the encoder name; old Direction-A checkpoints without
that field load as `flat`. The A5 CLI gains `--encoder` for training and gate
runs. No default changes before the gate passes.

## 4. Scope

**In scope**

- `StructuredObservationEncoder` and policy-factory wiring.
- Masked- and dot-product-head compatibility.
- A7 checkpoint persistence of the encoder choice.
- A5 CLI selection.
- Unit tests for shape, legality, gradients, empty hands, count sensitivity,
  permutation/value-multiset semantics and private-information leakage.
- Tiny-Heat flat-versus-structured ablation script and recorded results.

**Out of scope**

- Changing `OBS_DIM` or the engine-facing observation codec.
- Exposing opponent private cards.
- Adding played-card history/opponent hand-count public features (A8 contract
  extension).
- Tuning the dot-product action representation; A4 only re-runs it with the new
  state encoder after the primary masked-head gate.
- Full-rules/domain-randomized training (A8).

## 5. Acceptance gate

Run the A5 pool recipe on Tiny-Heat, masked head, seeds `{0,1,2}`, comparing
`encoder=flat` with `encoder=structured` at identical timesteps and PPO
hyperparameters.

- **G1 — interface and legality.** Both action heads satisfy the frozen policy
  interface, assign zero probability to illegal actions and train with finite
  losses using the structured encoder.
- **G2 — no private leakage.** Mutating an opponent's hand/deck while holding
  public state fixed leaves the acting seat observation and structured embedding
  byte/numerically identical. Mutating the acting seat's hand changes its hand
  representation.
- **G3 — sample efficiency.** At the fixed 40k-step checkpoint, structured mean
  win rate versus weak is no worse than flat by more than 5 percentage points,
  and structured wins on either mean win rate or mean area under the periodic
  evaluation curve. Report every seed; no tuning around a failure.
- **G4 — no collapse.** Every structured seed passes A5 Stage-1, retains minimum
  entropy at or above `0.40`, and has final-vs-weak regression no worse than 15
  points.
- **G5 — engineering.** New tests, full suite, Ruff and strict mypy are green;
  flat remains the default until G1–G4 pass.

If G3 fails, keep the implementation available for diagnosis but retain the
flat default. If G3 passes, flip `A0Config.encoder` and the CLI default to
`structured` in the gate-results commit.

## 6. Tests

`tests/test_a4_structured_encoder.py` covers:

1. output shape/finiteness, empty hand and hand-count sensitivity;
2. frozen policy shapes, legal masking and gradient flow through card, public
   and fusion modules;
3. factory support for masked/dot-product × flat/structured and invalid values;
4. opponent-private-state leakage invariance and own-hand sensitivity;
5. a small multi-seat PPO smoke with finite losses;
6. checkpoint round-trip for a structured policy (in the A7 checkpoint tests).

## 7. Implementation order

1. Land this design and encoder behind `encoder="structured"`.
2. Land correctness/leakage tests and the A5 CLI/checkpoint wiring.
3. Add `scripts/a4_gate.py`, run the three-seed ablation, and append the complete
   result table here.
4. Re-run A3's head comparison with the structured encoder as a separate
   secondary result; do not let it alter the primary A4 verdict.

## 8. Implementation progress (2026-07-13)

The first implementation slice is complete and default-off:

- structured own-hand attention/public-state encoder implemented;
- masked and dot-product heads wired through the common factory;
- `A0Config.encoder`, A5 `--encoder`, and A7 checkpoint round-tripping wired;
- leakage, count-sensitivity, legality, gradient, checkpoint and multi-seat
  smoke tests added;
- `scripts/a4_gate.py` added for the fixed flat-versus-structured experiment.

Verification: **7/7 A4 tests pass; 31/31 directly affected regression tests
pass; full suite 1064 passed; Ruff clean; strict mypy clean** on the new/edited
source files.

## 9. Gate results (2026-07-13) — G4 FAIL, flat stays default

Setup: `scripts/a4_gate.py` defaults — Tiny-Heat, 2 players, masked head, A5
pool recipe, 150k steps, seeds `{0,1,2}`. Both arms use identical PPO settings;
only `A0Config.encoder` differs. Total wall time was 917.8s (15m18s).

| encoder | seed | wr@40k | AUC | final | peak | regression | min entropy | engaged? | stage1 | wall(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| flat | 0 | 0.800 | 0.804 | 0.850 | 0.950 | 0.100 | 0.520 | no | pass | 130 |
| flat | 1 | 0.800 | 0.795 | 0.875 | 0.875 | 0.000 | 0.460 | no | pass | 140 |
| flat | 2 | 0.825 | 0.786 | 0.750 | 0.950 | 0.200 | 0.511 | no | pass | 126 |
| structured | 0 | 0.850 | 0.814 | 0.800 | 0.950 | 0.150 | 0.481 | no | pass | 171 |
| structured | 1 | 0.775 | 0.757 | 0.900 | 0.900 | 0.000 | 0.506 | no | pass | 179 |
| structured | 2 | 0.850 | 0.779 | 0.600 | 0.850 | **0.250** | 0.575 | no | pass | 169 |

| encoder | wr@40k | AUC | final | regression | min entropy |
|---|---|---|---|---|---|
| flat | 0.808 ± 0.012 | 0.795 ± 0.007 | 0.825 ± 0.054 | 0.100 ± 0.082 | 0.497 ± 0.026 |
| structured | 0.825 ± 0.035 | 0.783 ± 0.024 | 0.767 ± 0.125 | 0.133 ± 0.103 | 0.520 ± 0.040 |

### Verdicts

- **G1 — interface and legality: PASS.** Both heads pass the structured-policy
  interface, zero-illegal-mass, gradient and training-smoke tests.
- **G2 — no private leakage: PASS.** Opponent private hand/deck mutation leaves
  the acting-seat observation and structured embedding identical; own-hand
  mutation changes them.
- **G3 — sample efficiency: PASS, narrowly.** Structured is not worse at 40k;
  it is **+1.7 points** on the three-seed mean (0.825 vs 0.808), satisfying the
  fixed-checkpoint comparison. It does not win the broader curve: AUC is 0.783
  vs 0.795 and final mean is 0.767 vs 0.825. The pass is therefore literal but
  weak evidence, not a reason by itself to change the default.
- **G4 — no collapse: FAIL.** All structured seeds pass Stage-1, entropy stays
  above 0.40 and the controller never engages, but seed 2 regresses **25 points**
  (0.850 peak to 0.600 final), exceeding the predeclared 15-point limit.
- **G5 — engineering: PASS.** Full suite 1064 passed; Ruff and strict mypy are
  clean; checkpoint compatibility is covered; flat behavior remains default.

### Decision

The sprint does **not graduate** because G4 is an every-seed requirement. Keep
`A0Config.encoder="flat"` and the CLI default unchanged. The structured arm is
also about 31% slower in this run (mean 173s vs 132s), has lower AUC, lower final
mean and much higher final variance despite its small 40k advantage. Preserve
the implementation as an experimental arm, but do not route A8 through it
without a new design hypothesis and a separately declared re-test.
