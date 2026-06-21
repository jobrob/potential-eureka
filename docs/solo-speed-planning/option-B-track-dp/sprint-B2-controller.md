# Sprint B2 — Online plan-following controller

> The cheap online driver. A `BaseAgent` that solves the track once at race start
> (B1), then at each turn reads `π*`/`V*`/the heat budget and commits a move,
> resolving the **actual drawn hand** with one-ply lookahead so it never drives
> into a spin the plan averaged away.

## Goal

Implement `TrackDPAgent(BaseAgent)`: precompute the `PlannedTrack` for the game's
track on first decision, then drive each turn by following `π*` for gear and
selecting the real card play whose one-turn-forward outcome best matches the
planned heat/position trajectory, penalizing spins and overspeed. Expose a
`SearchProfile`-style cost report so the "fast" claim is measured.

## Scope

**In**
- A `BaseAgent` subclass implementing `choose_gear`, `choose_cards`,
  `choose_react`, `choose_slipstream`, `choose_discard`.
- Lazy per-track precompute (solve once, cache by track identity).
- 1-ply lookahead over the **real** `legal_card_plays` to pick the card play that
  matches the plan and avoids spins.
- Budget-aware React (take cooldowns where the plan banks heat) and discard (keep a
  low card if a tight corner is imminent — partial recovery of dropped hand-memory).
- Profiling (clones/move, ms/move) via the existing `SearchProfile`.

**Out**
- Solving the DP (B1) — B2 *uses* `solve_track`.
- The full shared-eval gate and spine export (B3).
- Opponents / determinization / multiplayer (solo scope).

## Deliverables (concrete)

- `src/heat/agents/track_dp_agent.py`
  - `class TrackDPAgent(BaseAgent)`:
    - `__init__(self, *, model_mode="banded", rollout_policy=None,
      own_spin_penalty=..., seed=None)` — mirrors `LookaheadAgent`'s
      constructor conventions; default `rollout_policy = HeuristicAgent`.
    - `_ensure_plan(state)` — on first call (or new track) calls
      `dp_solver.solve_track(state.track, ...)` and caches the `PlannedTrack`,
      keyed by `id(track)`/track name (the track is shared & immutable per game).
    - `choose_gear(...)` — map current `(pos, heat, gear)` to `ReducedState`, read
      `π*`, pick the legal gear nearest `gear + Δgear*` (re-validate against
      `legal_gear_shifts`), cache the plan signature like `LookaheadAgent`.
    - `choose_cards(...)` — **1-ply lookahead**: for each candidate in
      `legal_card_plays(hand, gear)` (deduped by resulting speed, reusing the S1
      prune idea), clone the state, force `(gear, play)`, run **one** turn via
      `run_round_driver`, and score by closeness of realized `(end_pos, end_heat)`
      to the plan target — with a large penalty for a spin or for exceeding the next
      corner's safe speed (the corner-skill lesson). Return the best legal play.
    - `choose_react(...)` — delegate to `rollout_policy`, but **override cooldown**
      to take the max when the plan's `cool_window` covers the current position
      (bank heat where the budget says to). Never boost into a tight corner.
    - `choose_slipstream` / `choose_discard` — delegate to `rollout_policy`, with a
      discard override that **keeps one low Speed card** when the next corner within
      reach is tight (so the hand can execute the conserve intent the model assumed).
    - `self.profile = SearchProfile()` — record clones/move + ms/move (1-ply, so a
      small constant).
- `tests/test_track_dp_agent.py`
  - Plays a full solo game on a generated track to completion without error.
  - On a hand-built single-tight-corner track, the agent **conserves into the
    corner** (plays the low cards / drops gear) and does **not spin** when it holds
    a safe play — and falls back to the least-overspeed play when it does not.
  - Determinism: same state + seed ⇒ same move (byte-stable, as S1 requires).
  - Cost: asserts clones/move is a small constant (≪ S1 at horizon 2).

## Success criteria (measurable, shared eval — solo, held-out generated)

- **Finishes 100% solo** on the held-out band.
- **Worst-case (max) spins/limit-1-pass ≤ HeuristicAgent's** (the S1 bar) — the
  controller must convert the plan into clean tight corners.
- **ms/move ≪ S1/S2 `LookaheadAgent`** at the same eval (the "plan once, drive
  cheaply" payoff), reported via `SearchProfile`. Per-track precompute counted and
  reported separately (amortized over the whole race).
- The agent's realized heat trajectory **tracks the planned budget** (a debug check
  that planned vs realized carry-in heat at each corner are close) — the proof the
  controller is actually *following* the plan, not accidentally driving like the
  heuristic.

## Risks & mitigations

- **R1 — Plan/reality drift mid-race** (clog and unlucky draws push the real state
  off the planned trajectory). *Mitigation:* the agent reads `π*` from the
  **current** `(pos, heat, gear)` every turn (closed-loop, not an open-loop script),
  so it re-enters the optimal policy from wherever it actually is. Optional cheap
  re-solve from the current state if drift exceeds a threshold (a B3 tuning knob).
- **R2 — 1-ply lookahead can still pick a spin when the hand is all-high into a
  tight corner.** *Mitigation:* the card scorer treats a spin as near-terminal
  (huge penalty, mirroring `LookaheadAgent`'s own-spin handling) and prefers the
  minimum-overspeed play; the discard override keeps a low card to make a safe play
  available next time. B3 measures the residual.
- **R3 — Precompute cost on first move** could spike latency. *Mitigation:* B1
  guarantees ≤ ~50 ms/track; precompute is a one-time per-race cost, reported
  separately so it does not contaminate the ms/move claim.
- **R4 — Over-conservative driving** (the plan slows more than needed and loses on
  rounds-to-finish). *Mitigation:* the `intent` bands include `push`; B3 tunes the
  spin penalty / band thresholds against rounds-to-finish so safety does not cost
  speed beyond the S1 baseline.

## Dependencies

- **B1 (`dp_solver.solve_track` + `PlannedTrack`)** and transitively B0.
- `GameState.clone`, `run_round_driver`, `legal_card_plays`, `legal_gear_shifts`,
  `SearchProfile`, `HeuristicAgent` (all present).
- `BaseAgent` protocol (present) — B2 produces a drop-in agent for `eval_search`.

## Rough effort

**~1 sprint.** The controller is light (1-ply, no training), but the card-scoring
rule, the budget-aware React/discard overrides, and the closed-loop plan lookup
need care and the determinism/no-spin tests to be trustworthy.
