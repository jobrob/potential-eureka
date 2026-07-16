# Sprint A3 — Legal-action dot-product head

> **Status:** **implemented and complete; gate G3 failed** (2026-07-09). The
> dot-product head remains available but default-off; the masked head remains the
> Direction A default. Retained because the failed gate is an active design constraint.

## 1. Purpose

The A0/A2 policy scores actions with a free `Linear(hidden, 516)` output layer: every
flat action index owns its own untied weight vector, so nothing learned about "play
{S3,S3}" transfers to "play {S3,S4}", and rarely-legal indices train on almost no
data — the **wide-head generalization tax** the Big 2 recipe removes. A3 replaces
that head with **feature-derived action scoring**: embed each action's *features*
(kind, gear, card-token histogram, react fields…), score by dot-product against a
state embedding, softmax over the legal set. Actions now share statistical strength
through their features, which is the property Direction A needs for the full game
(and the interface A4's structured encoder will feed).

## 2. Key design decision — static feature table, full-width scoring

The flat action space (`spaces.py`: GEAR 4 + CARDS 494 + REACT 8 + SLIPSTREAM 2 +
DISCARD 8 = 516) has **fully index-intrinsic semantics**: the card sub-range
enumerates frozen value-multisets (`_CARD_MULTISETS`), REACT is a fixed 8-slot table,
etc. Nothing about an action's identity depends on state (state-dependent context —
e.g. the heat cost of shifting to gear 4 *now* — is the observation encoder's job).

Therefore:

1. **Precompute a static feature table** `(ACTION_DIM, F)` once at import.
2. Each forward: `E = MLP_a(table)` → `(ACTION_DIM, d)` action embeddings;
   `e = W_s(trunk(obs))` → `(B, d)` state embedding;
   `logits = e @ E.T / sqrt(d)`; mask illegal → `Categorical`.
3. **Score all 516 and mask, rather than gather-then-score only legal rows.** The
   sprint plan's literal wording is "rank only currently-legal actions"; the two are
   mathematically identical after the softmax (illegal logits are `-inf` either way),
   and at 516 actions a `(B,d)@(d,516)` matmul is trivially cheap — the ragged
   gather/pad machinery would buy nothing today. What the sprint is *for* — removing
   the untied-per-index parameters — is delivered by the feature-derived embeddings.
   The sparse-gather optimization stays trivial to add later if the action space ever
   grows (the head already factors through per-action feature rows). Flagged as a
   deliberate design trade; the §5 gate (curve parity at equal compute) tests what
   matters.

Consequence: **`act`/`evaluate` signatures, the buffer, the trainer, and both
collectors are untouched.** The A0 "swap internals only" promise holds literally for
this sprint (the mask stays a `(B, ACTION_DIM)` bool).

## 3. Scope

**In scope**
- `src/heat/ml/selfplay/action_features.py` — the static per-action feature table.
- `DotProductPolicy` in `src/heat/ml/selfplay/policy.py` (same frozen interface) + a
  small `build_policy(config)` factory; `A0Config.head: str = "masked"` selector.
- `--head {masked,dotprod}` on both training CLIs; wiring in `train`/`train_multiseat`.
- Tests + the §5 head-comparison gate experiment (a script, run manually).

**Out of scope**
- Structured/hidden-info observation encoder → **A4** (the state side stays the flat
  MLP trunk; A4 replaces the trunk under the same `W_s` projection).
- *Dynamic* per-action features (e.g. current heat cost of a gear shift) — would
  force per-step feature storage in the buffer; revisit at A4 if the gate suggests
  the static table is limiting.
- Any anti-collapse / learning-quality work → **A5**. Removing the old masked head —
  it stays as the baseline and default until A3's gate passes.

## 4. Design

### 4.1 `action_features.py` — the static table

`action_feature_table() -> np.ndarray` of shape `(ACTION_DIM, ACTION_FEAT_DIM)`
float32, all values in [0, 1], built from the codec's frozen internals
(`_CARD_MULTISETS`, `_REACT_TABLE`, the `spaces` offsets — import the private names;
they are frozen contracts guarded by the codec's own asserts). Blocks (zero where
not applicable):

| block | dims | content |
|---|---|---|
| kind one-hot | 5 | GEAR / CARDS / REACT / SLIPSTREAM / DISCARD |
| gear | 4 | one-hot target gear 1–4 |
| cards histogram | 8 | count of each token (`_TOKEN_ALPHABET` order: H,S1,S2,S3,S4,ST,U0,U5) / 4 |
| cards size | 1 | multiset size / 4 |
| cards value sum | 1 | sum of deterministic printed values (S1–4→1–4, U0→0, U5→5, ST/H→0) / 20 |
| react | 4 | cooldown_count/4, boost, adrenaline_speed, adrenaline_cooldown |
| slipstream | 1 | 1.0 = take, 0.0 = decline (kind one-hot already marks the block) |
| discard | 1 | k / 7 |

`ACTION_FEAT_DIM = 25`. Module-level asserts: table shape, no NaN, kind blocks
mutually exclusive, and a spot-check row (e.g. the `("S3","S3")` multiset's histogram).

### 4.2 `DotProductPolicy`

Same constructor surface as `HeatPolicy` plus `embed_dim: int = 64` and
`action_mlp_hidden: tuple[int, ...] = (64,)`:

- trunk: identical MLP-over-obs shape as `HeatPolicy` (equal compute for the gate).
- `self.register_buffer("action_features", torch.from_numpy(action_feature_table()))`
  (moves with `.to(device)`, saved in checkpoints, not a parameter).
- action MLP: `25 → action_mlp_hidden → embed_dim` (Tanh, matching house style);
  recomputed each forward (weights change per update; 516×25 is trivial).
- state projection: `Linear(trunk_out, embed_dim)`.
- logits scaled by `1/sqrt(embed_dim)`; mask + `Categorical` + value head exactly as
  `HeatPolicy._distribution_and_value` (value head reads the trunk latent, unchanged).
- `act` / `evaluate`: byte-for-byte the same contracts as `HeatPolicy`.

`build_policy(config: A0Config) -> nn.Module` maps `config.head` →
`HeatPolicy` (`"masked"`, default) / `DotProductPolicy` (`"dotprod"`); `train` and
`train_multiseat` construct through it (type stays `HeatPolicy`-compatible; both
classes satisfy the same protocol — annotate with a small `Protocol` if mypy needs it,
do not force a common base class).

Note the head is *smaller* than the masked one (~21k vs ~134k head params at 256
trunk width) — "equal compute" for the gate means same trunk, same steps, same
hyperparameters; do not compensate elsewhere.

### 4.3 CLI / config

`A0Config.head: str = "masked"`; `--head {masked,dotprod}` on
`scripts/train_selfplay_a0.py` and `scripts/train_selfplay_a2.py`. Default stays
`masked` until the gate passes (flip the default in a later sprint, not this one).

### 4.4 Gate experiment script

`scripts/compare_heads_a3.py`: for each head × seeds {0,1,2}, run the A0 G2-style
probe — Tiny-Heat, 2 players, weak `HeuristicAgent` opponent, 60k steps, CPU — and
report per-run the mean episode return over the final 10 iterations, plus the
per-head mean ± std and wall time. (~6 runs × ~1 min; single process.) Print a
markdown table; no pass/fail logic in the script — the gate call is made by a human
reading the table.

## 5. Acceptance gate (exit criteria)

- **G1 — interface + legality.** `DotProductPolicy.act/evaluate` match the frozen
  contracts; **zero probability mass on illegal actions** (same assert pattern as the
  A0 smoke test); logp consistent between `act` and `evaluate`.
- **G2 — feature table correct.** Table shape/range asserts pass; spot-check rows for
  a known gear, card multiset, react slot, and discard index are exactly right.
- **G3 — the learning-curve gate (the point of A3).** On the §4.4 probe, the
  dot-product head's final-window mean return **matches or beats the masked head's**
  across seeds (overlapping ±1 std counts as "matches"; a consistent shortfall is a
  FAIL — stop and report, do not tune around it). Both heads must also run through
  `train_multiseat` (self-play smoke) without error.
- **G4 — non-regression.** Full suite green; default behavior unchanged
  (`head="masked"` everywhere unless asked); `ruff` clean; mypy on new files matches
  repo practice.

## 6. Tests (`tests/test_dotprod_policy.py`)

1. Feature-table spot checks (G2): gear-3 row, a known multiset row (histogram +
   size + value sum), react slot 6, discard k=5; mutual-exclusivity of kind blocks.
2. No-illegal-mass + shape/dtype conformance for `DotProductPolicy` (reuse the A0
   smoke-test pattern with random masks; both `act` and `evaluate`).
3. `act`/`evaluate` logp consistency on a fixed batch.
4. Gradient flow: a dummy PPO-style loss backpropagates non-zero grads into the
   action MLP, state projection, and trunk.
5. `build_policy` factory: correct class per `config.head`; unknown head raises.
6. 2-iteration `train(head="dotprod")` and `train_multiseat(head="dotprod")` smokes
   complete with finite losses.

## 7. Notes for the implementer

- Match conventions: `from __future__ import annotations`, house-style docstrings,
  full type hints, `PYTHONPATH=src` layout.
- Only edits to existing files: `policy.py` (add class + factory), `ppo.py`
  (config field + build via factory), `multiseat.py` (build via factory), the two
  CLIs. Everything else additive.
- Keep CI fast: tests use tiny trunks; the §4.4 comparison is a script, not a test.
- Run before committing: the new tests, the full suite, and the §4.4 gate script;
  paste the comparison table into your report **and** append it to this doc as a
  `## 8. Gate results` section (with the G3 verdict) alongside the code changes.
- Commit protocol (this sprint is committed, unlike A0–A2's tree-only convention):
  commit 1 = this design doc alone (`Design Sprint A3: legal-action dot-product
  head`); commit 2 = implementation + tests + gate results appended to this doc
  (`Implement Sprint A3: ...`), only after the full suite is green and the gate
  script has run. End both commit messages with the trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## 8. Gate results

Ran `scripts/compare_heads_a3.py` (Tiny-Heat, 2 players, weak `HeuristicAgent`
opponent, 60k steps, CPU, shared `256×256` trunk, `n_steps=2048`) for both heads
across seeds {0, 1, 2}. Metric = mean episode return over the final 10 iterations.

### Per-run (mean episode return over final 10 iters)

| head | seed | final-window mean return | wall (s) |
|---|---|---|---|
| masked | 0 | +0.8612 | 63.6 |
| masked | 1 | +0.8723 | 65.3 |
| masked | 2 | +0.8510 | 46.8 |
| dotprod | 0 | +0.5884 | 51.0 |
| dotprod | 1 | +0.6387 | 50.7 |
| dotprod | 2 | +0.6547 | 51.2 |

### Per-head summary

| head | mean ± std | wall total (s) |
|---|---|---|
| masked | +0.8615 ± 0.0087 | 175.7 |
| dotprod | +0.6273 ± 0.0282 | 153.0 |

### G3 verdict — **FAIL**

The dot-product head **consistently underperforms** the masked head. All three
dotprod seeds (+0.588, +0.639, +0.655) land well below all three masked seeds
(+0.851, +0.861, +0.872); the per-head ±1 std bands do **not** overlap
(masked ≥ 0.8528 vs dotprod ≤ 0.6555). This is a clear shortfall, not noise.

Per §5/§7, we **stop and report honestly rather than tune around it**. The other
gates pass: G1 (interface + zero illegal mass + act/evaluate logp consistency),
G2 (feature-table spot checks), and G4 (full suite green, `ruff`/`mypy --strict`
clean, both heads run through `train`/`train_multiseat`) all hold. But G3 is the
*point* of A3, so A3 does **not** graduate: the default head stays `"masked"`
everywhere, and `"dotprod"` remains an opt-in behind `--head dotprod` / a config
field for future investigation.

Interpretation (non-authoritative, for the next sprint): on this small probe the
free per-index head can memorize the tiny reachable action set outright, so the
feature-sharing that should *help* in the full game instead only adds a
bottleneck (25-dim features → 64-dim embedding) with no payoff yet. The generic
static features may also be too coarse to separate high-value plays. Likely
follow-ups before re-testing: richer/learned action features, a larger
`embed_dim`, or (per §3 scope) waiting for the A4 structured encoder — and
re-running this gate on a harder bed where the wide head's generalization tax
actually bites.
