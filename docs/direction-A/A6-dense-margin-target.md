# Sprint A6 — First dense target: terminal margin

> **Status:** design (2026-07-10). Detailed design for Sprint A6 of the
> [Direction A sprint plan](sprint-plan.md). Implementable spec; read this plus
> `src/heat/ml/spaces.py` (`step_reward`, `_placement_reward`),
> `src/heat/ml/selfplay/multiseat.py` (`_play_one_game`'s game-end completion),
> `src/heat/ml/selfplay/recipe.py` (`A5Config`, `train_selfplay_a5`), and the
> [A5 gate results](A5-anticollapse-recipe.md) §8 (the baseline this compares to).

## 1. Purpose

The C6 postmortem named the low-SNR ±1 terminal target a root cause of flat learning:
a placement-only reward says *whether* you won, never *by how much*, so the value
function cannot distinguish a dominant position from a lucky squeaker. A6 adds the
Phase-0 dense signal from Direction C: a **terminal margin** term — the normalized
progress lead over the best opponent at game end — added to the terminal reward, so
GAE returns (and therefore the PPO value loss AND advantages) carry graded
information. Remaining Direction-C aux heads stay in Phase 1 (A8).

## 2. Key design decisions

1. **Augment the terminal reward, not a second value channel.** The sprint
   deliverable is "margin value target wired into the PPO value loss"; the minimal
   faithful mechanism is `terminal_reward = placement + margin_coef * margin`, which
   reaches the value loss through the existing GAE returns. A separate value-only
   reward stream would need a second returns channel in the buffer — machinery A6
   doesn't justify. (If the gate someday shows the policy-gradient side is *hurt* by
   the margin term while the value side benefits, that split becomes an A8 option.)
2. **Margin is paid on truncation too.** Placement is termination-only (existing
   semantics, untouched), but a progress differential is meaningful at a time-limit
   cutoff — paying it there adds signal exactly where the sparse target has none.
3. **Opt-in, default off.** `margin_coef: float = 0.0` — every committed behavior
   (A0–A5, all gates) is byte-identical until a caller opts in. The A6 gate decides
   whether the recipe default flips (only `A5Config`'s default may flip, and only on
   a PASS).

## 3. Scope

**In scope**
- `terminal_margin(state, player_id) -> float` in `src/heat/ml/spaces.py` (pure).
- `margin_coef` plumbing: `MultiSeatCollector(..., margin_coef=0.0)` applied at
  game-end completion; `A5Config.margin_coef` passed through by `train_selfplay_a5`;
  `--margin-coef` on `scripts/train_selfplay_a5.py`.
- `scripts/a6_gate.py` (the A/B sample-efficiency experiment) + tests.

**Out of scope**
- Aux prediction heads (finishing order, per-corner spin, opponent next move) → A8.
- Any change to `_placement_reward`, `step_reward`'s existing modes, the SB3 path,
  or `HeatEnv` — the margin term lives in the Direction-A collector only.
- Reward-hyperparameter search beyond the single default `margin_coef=0.5` arm.

## 4. Design

### 4.1 `terminal_margin` (spaces.py)

Using the same absolute-progress arithmetic as `features._track_block`
(`abs_pos = player.lap * length + player.position`; a finished player counts as
`laps * length + length`, i.e. strictly ahead of any unfinished player):

```
remaining(p) = 0.0                          if p.finished
             = laps*length - (p.lap*length + p.position)   otherwise (clipped >= 0)
margin(state, pid) = clip((min over opponents of remaining(opp) - remaining(me))
                          / track.length, -1.0, +1.0)
```

Positive = ahead of the whole field; in an unclipped 2-seat game
`margin(s, 0) == -margin(s, 1)` (assert in a test). Solo (`n <= 1`) returns 0.0.
Docstring must state the normalization: a one-full-lap lead saturates at ±1.

### 4.2 Collector plumbing (multiseat.py)

`MultiSeatCollector.__init__` gains `margin_coef: float = 0.0` (keyword-only). In
`_play_one_game`'s game-end completion loop, after the existing `step_reward` (and
before the §4.6 truncation fold), add:

```
if self.margin_coef != 0.0:
    reward += self.margin_coef * terminal_margin(state, seat)
```

Applied on **both** terminated and truncated ends (decision §2.2). Intermediate
transitions are untouched. Update the class docstring's reward description.

### 4.3 Recipe plumbing (recipe.py, CLI)

`A5Config.margin_coef: float = 0.0`; `train_selfplay_a5` passes it to the training
collector(s). **Eval collectors are never given the margin** — eval scores from the
terminal state directly (`_placement_reward`), so the yardstick is identical across
arms; state this in a comment. `--margin-coef` (default 0.0) on the A5 CLI.

### 4.4 Gate experiment (`scripts/a6_gate.py`)

A/B over the A5 pool recipe (the committed default arm), identical except
`margin_coef`: **baseline (0.0) vs margin (0.5)** × seeds {0,1,2} × 150k steps,
Tiny-Heat 2p, masked head — the baseline cells are re-runs of the A5 gate's pool
arm, so results should be consistent with doc §8 there. Per run report (from the
eval records): winrate vs weak at the ~20k / ~40k / ~80k checkpoints and final,
steps-to-first-eval-≥70%, min entropy, wall time; then per-arm mean ± std, chunked
invocation supported (`--arms baseline margin --seeds ...`), rows print immediately.

## 5. Acceptance gate (exit criteria)

- **G1 — margin function correct.** Unit tests (§6.1) pass, including the 2-seat
  antisymmetry, clipping, finished-player, and solo cases.
- **G2 — sample efficiency (the point of A6).** The margin arm shows a measurable
  gain over baseline: **mean winrate-vs-weak at the ~40k checkpoint is at least 3
  points above baseline**, or **mean steps-to-≥70% is strictly lower on every
  seed**. And **no final regression**: margin arm final vs-weak within 5 points of
  baseline. Anything else = FAIL, reported honestly (no coef tuning; note the known
  ceiling-effect risk on this bed — the baseline already reaches ~85%+ — in the
  verdict either way).
- **G3 — default-off safety.** With `margin_coef=0.0` the collector's stored rewards
  are byte-identical to pre-A6 (regression test); full suite green; `ruff`/`mypy`
  per repo practice.
- **G4 — decision recorded.** On PASS: flip `A5Config.margin_coef` default to 0.5 in
  the same commit and note it in §8. On FAIL: default stays 0.0.

## 6. Tests (`tests/test_a6_margin.py`)

1. `terminal_margin`: hand-built 2-seat states (leader ahead by k spaces → expected
   clipped value; antisymmetry; both-finished → 0 vs 0 handling; finished-vs-not →
   positive for the finisher; solo → 0; margin > full lap → clipped to 1).
2. Collector applies margin only at game end and only when `margin_coef != 0`
   (drive one seeded game, compare stored final rewards with/without coef; earlier
   rewards identical).
3. Truncated game pays margin (monkeypatch `MAX_ROUNDS` low, assert the final
   reward includes the margin term as well as the truncation fold).
4. `margin_coef=0.0` byte-identity: stored reward stream equals the A2-era
   collector's for the same seed.
5. Config/CLI plumbing: `A5Config.margin_coef` reaches the training collector;
   eval path never sees it.
6. `train_selfplay_a5(margin_coef=0.5)` smoke: 2–3 tiny iterations, finite losses.

## 7. Notes for the implementer

- Match conventions (`from __future__ import annotations`, house docstrings, full
  hints, `PYTHONPATH=src`). Only edits to existing files: `spaces.py` (one new pure
  function), `multiseat.py` (§4.2), `recipe.py` (config + pass-through), the A5 CLI.
- The gate is 6 runs × ~2.5–4 min; chunk `a6_gate.py` invocations and assemble the
  table yourself.
- Run before committing: new tests, full suite, the gate; append `## 8. Gate
  results` (table + G1–G4 verdicts + the PASS/FAIL default decision) to this doc.
- Commit protocol (as A3/A5): commit 1 = this doc alone (`Design Sprint A6: dense
  terminal-margin value target`); commit 2 = implementation + tests + gate results
  (`Implement Sprint A6: ...` with the verdict in the subject), only after full
  suite green + gate run. `git add` specific paths; do not push. End both messages
  with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## 8. Gate results (2026-07-10) — verdict: G2 FAIL, default stays 0.0

Setup: `scripts/a6_gate.py` defaults — Tiny-Heat, 2 players, masked head, pool
recipe (`pool_prob=0.5`), 150k steps (~73 iterations), seeds {0,1,2}. Baseline =
`margin_coef=0.0` (a re-run of the A5 pool arm; consistent with §8 there); margin =
`margin_coef=0.5`. Checkpoint columns read the vs-weak winrate off the eval record
nearest 20k / 40k / 80k steps; `steps->=70%` is the first eval whose vs-weak winrate
reaches 0.70.

| arm | seed | wr@20k | wr@40k | wr@80k | final | steps->=70% | min entropy | stage1 | wall(s) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0.750 | 0.775 | 0.825 | 0.750 | 20588 | 0.567 | pass | 145 |
| baseline | 1 | 0.675 | 0.825 | 0.875 | 0.950 | 41209 | 0.576 | pass | 131 |
| baseline | 2 | 0.700 | 0.850 | 0.750 | 0.775 | 20615 | 0.537 | pass | 171 |
| margin | 0 | 0.675 | 0.850 | 0.850 | 0.875 | 41140 | 0.445 | pass | 138 |
| margin | 1 | 0.625 | 0.850 | 0.850 | 0.900 | 41152 | 0.467 | pass | 135 |
| margin | 2 | 0.725 | 0.750 | 0.850 | 0.875 | 20567 | 0.495 | pass | 160 |

| arm | wr@20k | wr@40k | wr@80k | final | mean steps->=70% |
|---|---|---|---|---|---|
| baseline | 0.708 ± 0.031 | 0.817 ± 0.031 | 0.817 ± 0.051 | 0.825 ± 0.089 | 27471 ± 9714 |
| margin | 0.675 ± 0.041 | 0.817 ± 0.047 | 0.850 ± 0.000 | 0.883 ± 0.012 | 34286 ± 9701 |

### The decisive finding: the margin term is identically zero on this bed

The margin is **provably 0 for every transition on the 1-lap Tiny-Heat bed**, so the
two arms train on byte-identical rewards and the dense target contributes *nothing*.
Faithfully mirroring `features._track_block` (design §4.1), each player's remaining
distance is `remaining = max(0, laps*length - (lap*length + position))`. Tiny-Heat
has `laps=1` and the engine races at `lap=1`, so for every unfinished player
`lap*length (14) >= laps*length (14)` and `remaining` clips to 0; finished players
are 0 by definition. Hence `terminal_margin ≡ 0` (verified empirically: 86/86 calls
during a smoke run return 0.0; 20/20 games at a forced truncation return 0.0). This
is the same degeneracy that makes `_track_block.dist_to_finish` always 0 on a 1-lap
track — a property of the frozen feature convention, not of this implementation.

`terminal_margin` is pure (no RNG/state side effects, checked), so `reward += 0.5*0`
is an exact no-op: a single collect with `margin_coef=0.5` produces byte-identical
reward streams to `0.0` (0.0 max-diff). The table's apparent between-arm differences
are therefore **not attributable to the margin**. They are run-to-run nondeterminism
of the A5 self-play loop: two identical `margin_coef=0.0` runs at the same seed
diverge by ~0.27 in max parameter and ~0.12 in mean entropy. That nondeterminism is
pre-existing A5 behavior (the `coef=0.0` path is byte-identical to pre-A6), not
introduced by A6, and it dominates any per-seat comparison at this budget.

### Verdicts (per §5)

- **G1 — margin function correct: PASS.** `tests/test_a6_margin.py` (6 tests) —
  antisymmetry, ±1 clipping, both-finished, finished-vs-not, solo, and the collector
  plumbing (margin only at game end, only when `coef != 0`, on truncation too) —
  all green. The unit tests exercise a *live* margin by driving a multi-lap track;
  the arithmetic is correct where the bed is non-degenerate.
- **G2 — sample efficiency: FAIL.** By the letter of §5: mean wr@40k is **0.817 for
  both arms** (not baseline+3), and `steps->=70%` is **not strictly lower on every
  seed** (margin is worse on seed 0: 41140 vs 20588). The "no final regression"
  clause is met (margin final 0.883 ≥ baseline 0.825), but the primary criterion is
  not. More fundamentally, the margin is identically 0 on this bed, so **no gain is
  even possible** — this is not a ceiling effect but an *inert-signal* effect. FAIL,
  reported as measured, no coefficient tuning.
- **G3 — default-off safety: PASS.** With `margin_coef=0.0` the branch is never
  entered (0 calls) and the stored reward stream is byte-identical to the pre-A6
  default-constructor collector (`test_margin_coef_zero_byte_identity`). Full suite:
  **1050 passed / 1 skipped**. `ruff check` clean; `mypy --strict` clean on the
  changed `src` files.
- **G4 — decision: default stays 0.0.** Gate did not PASS, so `A5Config.margin_coef`
  remains `0.0` (opt-in, off). The mechanism is committed and correct, ready to be
  re-gated on a bed where it can carry signal.

### Follow-up (not part of A6; out of scope here)

The result is a **bed/target mismatch**, not a broken mechanism. To actually test the
dense target, a future sprint should either (a) run the gate on a **multi-lap** bed
where `remaining` is non-degenerate, and/or (b) make the terminal reward
**finish-order-aware** so a dominant win is graded above a squeaker even at full
termination (where all `remaining` are 0). A prerequisite for any real A/B on this
loop is to **make A5 training reproducible** (same seed → same weights); the current
run-to-run nondeterminism is large enough to swamp a small dense-reward effect.
