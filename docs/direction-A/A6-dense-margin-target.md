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
