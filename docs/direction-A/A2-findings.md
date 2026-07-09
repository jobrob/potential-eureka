# A2 findings — multi-seat harness green; first non-collapsing self-play

> **Status:** findings (2026-07-09). Results of implementing + validating Sprint A2
> ([design](A2-multiseat-selfplay.md)), plus an exploratory pure-self-play probe that
> pulls the core A5 question forward.

## 1. Gate results (all PASS)

- **G1 — self-play completes legally.** Fresh-policy sweeps over Tiny-Heat 2p/4p and
  USA 2p: zero illegal actions (every stored action cross-checked against
  `legal_action_mask`), every game terminates or truncates, every non-empty seat
  stream ends `done=True` (no pending-transition leaks).
- **G2 — per-seat credit assignment correct.** Stream isolation proven (`_store` only
  ever writes `buffers[seat]`); terminal placement rewards equal `_placement_reward`
  of the actual finish order and sum to ~0 for 2 seats; intermediate sparse rewards
  are 0; truncated games fold `gamma * V(s_next)` into the final reward (§4.6 fix,
  applied to both the A2 collector and A0's `collect_rollout`).
- **G3 — throughput.** Collection-only, matched net (256×256) on Tiny-Heat 2p:
  **A2 self-play ≈ 2,305 recorded transitions/s vs A0 single-seat ≈ 2,245/s**. The
  bottleneck is the per-decision CPU network forward, not the engine. End-to-end
  (collect + 10-epoch PPO update): ~1,200 transitions/s.
- **G4 — non-regression.** Full suite **1029 passed / 1 skipped** (7 new tests). The
  env→codec extraction (`decode_legal_action` / `forced_action` / `NO_FORCED`) is
  behavior-identical; a test asserts `HeatEnv` delegates to the shared functions so
  the two paths cannot drift.

**Learning cross-check** (beyond the gates): the A2 harness in its A0-equivalent
configuration (1 policy seat + 1 scripted `HeuristicAgent`, Tiny-Heat, 60k steps)
climbs to a **+0.95–1.00** return plateau — wrong per-seat rewards, GAE streams, or
masking could not produce this, so credit assignment is validated end to end.

## 2. Exploratory probe — naive pure self-play does NOT collapse

The recurring failure this project has never gotten past is self-play collapse
(Sprint 5 Phase 2, 6C/6D leagues, 8C BC→PPO). With the harness green we ran the
deliberately *naive* recipe — current-policy 2p self-play on Tiny-Heat, **no entropy
floor, no snapshot pool** — 250k steps, seed 0, evaluating vs the weak
`HeuristicAgent` (both seat orders; the policy never trains against it):

| self-play steps | winrate vs heuristic | entropy |
|---|---|---|
| 0 (untrained) | ~41% | 1.12 |
| 20k | ~70% | 1.12 |
| 41k | ~86% | 0.92 |
| 144k | ~92% | 0.69 |
| 250k (final, 231 games) | **~85% (mean return +0.70, se 0.047)** | ~0.73, stable |

- **No collapse signature:** entropy decays gently then holds flat for 130k+ steps;
  value loss falls monotonically (0.36 → 0.12); skill never regresses.
- **Transfer:** self-play alone reaches ~85–92% vs an opponent it never saw —
  the Big-2 premise (self-play produces general play) working in miniature.
- **Expected gap:** direct A0 training *against* the heuristic reaches ~96–98% on the
  same bed; self-play lands a few points lower vs that specific opponent, with mild
  84–93% oscillation characteristic of chasing one's current self — exactly what
  A5's snapshot pool is meant to damp.

**Implication for A5:** the question shifts from *"does self-play work at all?"* to
*"stabilize something already working"* (multi-seed robustness, entropy floor,
snapshot pool, Stage-1 validation). Combined with the A0 result, the historical
collapses increasingly look like SB3 / BC-warm-start substrate artifacts rather than
task hardness.

**Caveats:** tiny bed, 2 players, weak-heuristic yardstick, single seed. Needs
repeating across seeds (the real A5 gate), at 3+ seats, and on the generated-track
distribution (A8) before any stronger claim.

## 3. Implementation notes / deviations from the design

1. Per-seat buffer capacity is `n_steps + 5*MAX_ROUNDS` (not literally `n_steps`):
   the always-finish-the-game rule can overshoot, and 5 decisions/seat/round
   (GEAR, CARDS, REACT, SLIPSTREAM, DISCARD) is a verified driver upper bound.
2. `collect` / `train_multiseat` take a `gamma` keyword (needed by the §4.6
   truncation fold inside the collector).
3. A `_on_game_end` no-op seam exists so tests can capture terminal states.
4. Probe script: `selfplay_probe.py` (session scratchpad, not committed) — trains via
   `MultiSeatCollector` + `ppo_update` directly and evals via `scripted_seats`.
