# Sprint A1 — Tiny-Heat proving ground

> **Status:** **implemented and complete** (2026-07-09). The Tiny-Heat proving
> ground, sanity harness, and supporting tests landed with the A0–A2 implementation.
> Retained because later Direction A gates use this bed and its assumptions.

## 1. Purpose

C6's #1 lesson: **prove the method moves where iteration is cheap, before paying for
scale.** A1 builds the "Connect4-equivalent" bed for Direction A — a deliberately small
Heat variant where a full method experiment (A2 multi-seat, A3 head, A5 anti-collapse) is a
minutes-loop, not a day. It must stay **honest on the hard axes** (stochastic card draws,
hidden information, multiplayer) so results transfer to full Heat; it just shrinks the
*size* of the game, not its *nature*.

A1 is infrastructure: a tiny game configuration + a sanity/throughput harness. No learning
result is expected here — that's A5. The A0 custom-PPO loop must run on Tiny-Heat
**unchanged**.

## 2. Key finding that shapes the design: shrink the track, not the engine

The engine is already data-driven where it matters and hard-coded where it doesn't:

- **Track is pure data** — `Track(spaces, corners, start_positions, laps)`. A tiny track is
  just a small `Track`. `HeatEnv` already accepts a `Track` or a sampler, and A0's `train()`
  already takes a `track=` arg and `num_players` via `A0Config`. So the tiny **track + seat
  count** needs essentially no new engine code.
- **The real USA track is already small** (30 spaces, 3 corners, 1 lap) and is the one that
  reached 94%. So Tiny-Heat must go meaningfully *below* it (~12–16 spaces, 1–2 corners,
  2–3 cars) to be a genuinely cheaper bed.
- **Deck/hand are hard-coded** (`cards.py` builders, `PlayerState.create`,
  `GameState.create`, hand size 7). Shrinking the *deck* would mean forking three engine
  modules.

**Decision: A1 is data-only — a tiny track + fewer seats, full deck.** The three hard axes
survive a short track untouched (draws are still stochastic, opponents' hands still hidden,
2–3 seats is still multiplayer). A short, strategically shallow but *valid* game is exactly
what a Connect4-class proving ground is. **The deck/hand fork is explicitly deferred** (see
§6) — we only do it if A1's sanity sims show the full-deck tiny game is *degenerate* (card
and heat decisions never bite). This trades the sprint-plan's literal "small deck" wording
for the same goal — a cheap, faithful bed — without a speculative engine rewrite. Flag for
sign-off.

## 3. Scope

**In scope**
- A `tiny_heat` track factory (fixed, hand-built, deterministic) + a tiny `TrackGenParams`
  for an optional tiny *distribution*.
- A Tiny-Heat `A0Config` preset (2–3 seats, rollout sizing fit to short episodes) and a
  convenience env builder.
- A sanity + throughput harness: runs self-play games, asserts legality + termination,
  reports games/sec, episode length, and non-degeneracy diagnostics; measures Tiny-Heat
  vs USA so the speedup is quantified.
- Tests.

**Out of scope (named so they aren't built here)**
- Deck/hand parametrization (engine fork) → deferred, §6.
- Multi-seat self-play / per-seat trajectories → **A2** (A1 runs the existing single-seat
  A0 loop on the tiny bed).
- The dot-product head → **A3**; structured encoder → **A4**; anti-collapse / a learning
  result → **A5**. A1 proves the bed is cheap and valid, not that anything learns.
- Track-distribution generalization as a *goal* → **A8** (A1 ships tiny params as a
  convenience, but the fixed tiny track is the A1 bed).

## 4. Design

### 4.1 Module layout (additive)

```
src/heat/ml/selfplay/tiny_heat.py   # tiny track + params + config presets + env builder
scripts/tiny_heat_sanity.py         # sanity + throughput/calibration harness
tests/test_tiny_heat.py
```
(Keeping it under `selfplay/` keeps the Direction-A substrate together; nothing in the
engine or existing ml modules is edited.)

### 4.2 `tiny_heat.py` API

- `tiny_heat_track() -> Track` — a **fixed, hand-built** tiny track, built directly as a
  `Track` and passed through `heat.tracks.validate.validate_track` (assert valid at import
  or in a test). Target geometry (developer calibrates against §4.4 sanity output so the
  game is non-degenerate, not a 2-round sprint):
  - **~12–16 spaces**, **1–2 corners** (corner length 1–2, speed limits in {2,3,4}),
    **1 lap**, **≥4 start positions** on straights (so 2–4 seats fit; include the
    start-line straight, mirroring the static-track grid convention).
  - Lanes: mostly 1; a single 2-lane space is fine (keeps slipstream/overtake reachable).
  - The resulting game should run **~6–12 rounds** with a couple of meaningful gear/corner
    decisions — long enough that skill separates from random, short enough to be cheap.
- `tiny_heat_params() -> TrackGenParams` — a tiny generation distribution
  (`length_range≈(12,16)`, `num_corners_range=(1,2)`, `laps=1`, `corner_len_range=(1,2)`,
  `speed_limit_choices=(2,3,4)`). Must satisfy `TrackGenParams` validation
  (`length_range[0] ≥ MIN_TRACK_LENGTH=8`; note `min_start_positions` defaults to
  `MAX_PLAYERS=6`, so confirm tiny lengths still admit 6 start straights, or set
  `min_start_positions` to the tiny seat count if the generator allows — verify against the
  validator). Used later for randomized tiny beds; **not** required by the A1 gate.
- `tiny_heat_config(**overrides) -> A0Config` — an `A0Config` preset: `num_players=2`
  (default; allow 3), `randomize_seat=True`, and rollout sizing sensible for short episodes
  (e.g. a smaller `n_steps` is fine since episodes are short; keep `batch_size`/`n_epochs`
  defaults). Overridable.
- `make_tiny_env(num_players=2, *, distribution=False, seed=None) -> HeatEnv` — convenience:
  `HeatEnv(track=tiny_heat_track() | track_sampler(tiny_heat_params()), num_players=...,
  randomize_seat=True)`. `distribution=True` selects the sampler (per-episode tiny track);
  default is the fixed tiny track (the deterministic A1 bed).

### 4.3 Running the A0 loop on Tiny-Heat (no new training code)

`train(tiny_heat_config(), track=tiny_heat_track(), opponents=...)` must run unchanged.
A1 adds *no* training logic — it only supplies the config/track. Confirm via a test that a
2-iteration `train()` on Tiny-Heat completes without error.

### 4.4 Sanity + throughput harness (`scripts/tiny_heat_sanity.py`)

Runs `--games N` self-play games on Tiny-Heat (default opponents `HeuristicAgent`, plus a
`--random` mode) by driving the env, and reports — for **both Tiny-Heat and USA** so the
ratio is explicit:

- **Correctness (asserts):** every chosen action is in the decision's
  `legal_action_mask`; every game **terminates** (`is_game_over`) and **none truncates** at
  `MAX_ROUNDS`.
- **Cost:** wall-clock **games/sec** and **env-steps/sec** (single process, CPU).
- **Non-degeneracy diagnostics (the calibration signal):** mean rounds/game, mean learner
  **decisions/episode**, mean corners encountered/game, fraction of episodes with ≥1
  heat-payment / ≥1 non-trivial CARDS choice, finish rate. These tell us whether the tiny
  game still exercises the hard axes or has collapsed into a trivial sprint (→ §6 trigger).

Print a small table; exit non-zero if any correctness assert fails.

## 5. Acceptance gate (exit criteria)

- **G1 — valid & terminating.** `tiny_heat_track()` passes `validate_track`; the sanity
  harness runs ≥200 games with **zero** illegal actions and **zero** truncations; every
  game finishes.
- **G2 — cheaper bed (the point of A1).** Tiny-Heat games/sec is **≥3× the USA-track rate**
  in the same single-process harness, **and** a 100k-step A0 `train()` probe on Tiny-Heat
  (2 seats, `RandomAgent`) completes in **< ~10 min** single-env CPU. (Ratio is the primary
  gate; the absolute is a sanity floor — record both.)
- **G3 — non-degenerate.** Sanity diagnostics show the tiny game still exercises the hard
  axes: mean ≥ ~6 rounds and ≥ ~1 corner encounter/game, and a non-trivial fraction of
  episodes involve a real CARDS/heat decision. If not, calibrate the track (lengthen
  slightly / adjust corners) — or, if calibration can't fix it without re-lengthening to
  USA scale, that is the §6 deck-fork trigger; **stop and report** rather than shipping a
  degenerate or not-actually-tiny bed.
- **G4 — runs A0 unchanged + non-regression.** A `train()` smoke on Tiny-Heat passes; full
  existing suite green; `ruff`/`mypy` clean on new files. Additive only — no edits to
  engine or existing ml modules.

## 6. Deferred: deck/hand shrink (only if A1 shows degeneracy)

If G3 shows the full-deck tiny game is degenerate (cards/heat never matter), the fix is a
small, backward-compatible deck parametrization — a `GameConfig`/`DeckConfig` threaded
through `GameState.create` → `PlayerState.create` → the `cards.py` builders, defaulting to
today's exact values (so existing behavior is byte-for-byte unchanged). That is a separate
mini-sprint (call it A1.5), designed when/if triggered — not built speculatively here.

## 7. Notes for the implementer

- **Additive only.** Do not modify `track.py`, `cards.py`, `player_state.py`,
  `game_state.py`, `env.py`, or the existing `selfplay/*` loop. Reuse them.
- Build the fixed tiny track by hand as a `Track` and **validate it** — don't hand-wave
  geometry; the validator (`heat.tracks.validate.validate_track`) is the contract.
- Match conventions: `from __future__ import annotations`, house-style module docstrings,
  full type hints, `PYTHONPATH=src` layout.
- Run `scripts/tiny_heat_sanity.py` and paste the table (Tiny vs USA) in your report — the
  G2/G3 numbers are the deliverable, not just "it runs."
- Do **not** commit; leave changes on the working tree and report results. If G3 calibration
  hits the §6 wall, stop and report rather than forcing it.
