# Sprint A2 — N-agent self-play harness

> **Status:** **implemented and complete** (2026-07-09); all A2 gates passed.
> Retained as the rollout and per-seat credit-assignment contract used by later
> Direction A work. See [A2 findings](A2-findings.md) for measured results.

## 1. Purpose

A0 proved the custom PPO loop is correct, but it trains a **single seat against
scripted opponents** — `HeatEnv` auto-advances every non-learner seat internally, so
it structurally cannot produce per-seat trajectories. A2 replaces that asymmetry with
**true current-policy self-play**: 2–N seats all driven by the *same live policy*, each
acting on its **own per-seat partial observation**, each producing its **own trajectory
stream** for PPO. This is the sprint that introduces the mechanism every prior
incarnation of this project collapsed on — so A2 is judged on *correctness of per-seat
collection*, *not* on learning stability (that is A5's gate; A5 builds directly on this
harness).

A2 also fixes one known A0 bug while the buffer contract is still cheap to touch: the
rollout treats **truncation as termination** (`ppo.py` `collect_rollout`, `done =
terminated or truncated`), which wrongly zeroes the value bootstrap on time-limit
cutoffs (see §4.6).

## 2. Key design decisions

1. **Drive `run_round_driver` directly; do not extend `HeatEnv`.**
   `run_round_driver(state)` already yields `Decision(kind, player_id, legal)` at
   *every* seat's decision point and resumes via `send` — exactly the hook a shared-
   policy collector needs. `HeatEnv` stays untouched as the A0/SB3 baseline.
2. **Extract, don't duplicate, the decision→engine-action logic.** `HeatEnv._decode_legal`
   (decode + CARDS snap-to-legal-tuple) and `HeatEnv._forced_action` (auto-resolve
   degenerate ≤1-legal decisions, including the all-False empty-hand CARDS mask) are
   the subtle legality plumbing. Move their bodies to **module-level functions in
   `action_codec.py`** — `decode_legal_action(decision, state, flat_index)` and
   `forced_action(decision, state)` (returning a module-level `NO_FORCED` sentinel) —
   and have `HeatEnv` delegate to them. Behavior must be **byte-identical** (the 1022-
   test suite is the guard). This is the one permitted edit to existing modules; it
   exists so the env path and the self-play path can never drift apart.
3. **One `RolloutBuffer` per seat, never interleaved.** GAE assumes a single
   time-ordered stream (`values[t+1]` follows `values[t]` within one seat's
   experience). Each seat gets its own buffer; GAE is computed per stream; streams are
   concatenated only *after* `compute_gae`, for the (seat-agnostic) PPO update.
   A test must assert this invariant (§6).
4. **Whole-game collection.** The collector runs complete games (create state → loop
   `run_round_driver` rounds → `is_game_over` or `round_num > MAX_ROUNDS`) until the
   total recorded transitions across seats reaches `n_steps`, always **finishing the
   in-flight game**. Every seat stream therefore ends at an episode boundary
   (`done=True` on its last step), so the GAE bootstrap value is always 0 and there is
   no carry-over obs/mask between rollouts. This trades a little rollout-size jitter
   for a much simpler correctness story.
5. **Reward reuses `step_reward` per seat over that seat's decision span.** A seat's
   transition runs from its decision to its *next* decision (or game end). Reward for
   that transition = `step_reward(prev, curr, seat_id, done, terminated=...)` where
   `prev` is the state snapshot (`state.clone(reseed=0)`) taken when the seat acted and
   `curr` is the state at the seat's next decision / terminal. Under the default sparse
   placement mode this reduces to terminal-only placement reward, but the span hook is
   what A6's dense targets will plug into. Truncated games pay no placement reward
   (existing `step_reward` semantics — do not change them).

## 3. Scope

**In scope**
- `src/heat/ml/selfplay/multiseat.py`: the multi-seat collector + `train_multiseat`.
- The `action_codec.py` extraction of §2.2 (+ `env.py` delegation, byte-identical).
- The truncation-bootstrap fix, in both the new collector and A0's `collect_rollout`.
- Optional scripted seats (`scripted_seats: dict[seat_id, BaseAgent]`), default none —
  scripted seats are advanced but **not recorded**. This is deliberately minimal: it is
  the hook A5's snapshot-pool opponents and A7's eval will reuse, and it lets a test
  cross-check the collector against the A0 env setup.
- A minimal CLI `scripts/train_selfplay_a2.py` and tests.

**Out of scope (named so they aren't built here)**
- Dot-product head → **A3**; structured/hidden-info encoder → **A4** (A2 keeps the flat
  obs + fixed mask; per-seat obs via the existing `encode_observation(state, seat_id,
  decision)`, which is already honest about hidden information).
- Entropy schedules, snapshot pools, any anti-collapse tuning, any learning *result* →
  **A5**. A2's smoke checks finiteness, not skill.
- Dense/margin rewards → **A6** (the span hook of §2.5 is where they will land).
- Vectorization/batched acting. Decisions arrive sequentially from the generator; A2
  acts one obs at a time exactly like A0.

## 4. Design

### 4.1 Module layout

```
src/heat/ml/selfplay/multiseat.py   # collector + train_multiseat (new)
scripts/train_selfplay_a2.py        # CLI (new)
tests/test_multiseat_selfplay.py    # tests (new)
src/heat/ml/action_codec.py         # + decode_legal_action, forced_action, NO_FORCED
src/heat/ml/env.py                  # _decode_legal/_forced_action delegate (behavior identical)
src/heat/ml/selfplay/ppo.py         # collect_rollout truncation fix only
```

### 4.2 Collector

```python
class MultiSeatCollector:
    def __init__(self, track: Track | TrackSource, num_players: int,
                 *, scripted_seats: dict[int, BaseAgent] | None = None): ...
    def collect(self, policy: HeatPolicy, n_steps: int, device, rng
                ) -> tuple[list[RolloutBuffer], list[float]]:
        """Run whole games until sum(len(buf) for policy seats) >= n_steps.
        Returns per-seat buffers (GAE not yet computed) + per-seat-episode returns."""
```

Game loop per episode (mirrors `HeatEnv.reset`/`_advance_to_learner`, via the shared
helpers):

```
seed  = rng.integers(...)             # same-seed -> same-episode determinism
state = GameState.create(track, num_players, seed=seed); all players lap = 1
gen   = run_round_driver(state); send_value = None
loop:
    decision = gen.send(send_value)   # StopIteration -> new round generator, continue
    if game over / truncated: break
    seat = decision.player_id
    if seat in scripted_seats:  send_value = opponent_action(...); continue
    forced = forced_action(decision, state)
    if forced is not NO_FORCED: send_value = forced; continue
    # a real policy decision:
    complete seat's pending transition (reward over span, done=False)  # §4.4
    obs  = encode_observation(state, seat, decision)
    mask = legal_action_mask(decision, state)
    action, logp, value, _ = policy.act(obs, mask)          # single-row batch, like A0
    pending[seat] = (obs, action, logp, value, mask, state.clone(reseed=0))
    send_value = decode_legal_action(decision, state, int(action))
at game end: complete every pending transition (done=True, terminal/truncation reward §4.6)
```

Per-seat buffers are sized `capacity=n_steps` (partial fill is native — `get()` and
`compute_gae` already slice by `_pos`). Seats with `randomize_seat` semantics are moot:
**every policy seat is the learner**, which is itself the strongest seat-symmetry
randomization. (`A0Config.randomize_seat` is ignored here; document it.)

### 4.3 Per-seat observation & masks

`encode_observation(state, seat_id, decision)` and
`legal_action_mask(decision, state)` are already pure per-seat functions — reuse them
directly. No encoder or codec changes.

### 4.4 Pending-transition bookkeeping (the credit-assignment core)

A seat's PPO step spans from its decision to its next decision. The collector keeps at
most one *pending* transition per seat: `(obs, action, logp, value, mask, prev_state)`.
When that seat's next real decision arrives, emit
`buffer[seat].add(obs, action, logp, value, reward=step_reward(prev_state, state,
seat, done=False, terminated=False), done=False, mask)`. When the game ends, complete
all pending transitions with `done=True` and `step_reward(prev_state, state, seat,
done=True, terminated=terminated)`. A seat whose pending slot is empty at game end
(e.g. every decision degenerate) simply contributes nothing for that game.

### 4.5 GAE + update

After `collect`: for each non-empty seat buffer, `compute_gae(last_value=0.0, gamma,
gae_lambda)` (always 0 — whole-game collection, §2.4). Concatenate the per-seat
`get()` dicts along dim 0 into one batch and run the existing `ppo_update` unchanged
(advantage normalization then happens over the full concatenated batch, which is
correct for a shared policy). `train_multiseat(config: A0Config, *, track=None,
scripted_seats=None, on_iteration=None) -> HeatPolicy` mirrors A0's `train`
(same device/seed handling, same `on_iteration` info dict + `n_episodes`).

Log honestly: **in pure self-play with sparse placement reward the mean per-seat
return is ≈ 0 by construction** (zero-sum-ish); it is *not* a learning signal. The
smoke gate checks finiteness and mechanics, nothing more.

### 4.6 Truncation bootstrap fix (A0 + A2)

SB3-style time-limit handling, no buffer/GAE change: when an episode ends by
**truncation** (`round_num > MAX_ROUNDS`, `terminated=False`), fold the bootstrap into
that final step's reward at collection time:

```
reward += gamma * V(s_next)    # V from policy.act on the post-step obs, decision=None
```

- In `multiseat.py`: `s_next` for each seat is `encode_observation(state, seat, None)`.
- In `ppo.py:collect_rollout`: apply the same fold when `truncated and not terminated`
  (the env already returns the two flags separately), keeping `done=True` in the buffer.
- Add a `# truncation != termination:` comment at both sites referencing this section.

This corrects the known A0 bias (value bootstrap wrongly zeroed on time-limit cutoffs)
without widening the frozen buffer layout.

### 4.7 CLI

`scripts/train_selfplay_a2.py`, mirroring `train_selfplay_a0.py` flags (`--timesteps
--n-steps --batch-size --n-epochs --players --lr --device --seed --hidden`) plus
`--track {tiny,usa}` (default `tiny`) and `--scripted-opponents N` (fill the last N
seats with `HeuristicAgent`s; default 0 = pure self-play). Per-iteration print matches
A0's format.

## 5. Acceptance gate (exit criteria)

- **G1 — self-play games complete, legally.** ≥200 pure-self-play games (2 and 4
  seats, Tiny-Heat and USA) with an untrained policy: zero illegal actions (every
  sampled action cross-checked against `legal_action_mask`), every game terminates or
  truncates, no pending-transition leaks (all buffers end on `done=True`).
- **G2 — per-seat credit assignment is correct (the point of A2).** Tests (§6) pass:
  per-seat streams isolated; terminal placement rewards per seat match
  `_placement_reward` of the actual finish order (winner positive, loser negative,
  2-seat rewards sum to ~0); intermediate sparse rewards are 0; truncated games pay no
  placement but do get the §4.6 bootstrap fold.
- **G3 — throughput acceptable in Tiny-Heat.** Recorded transitions/sec in pure
  2-seat self-play on Tiny-Heat is at least the A0 single-seat rate (self-play records
  ~N seats' transitions per game, so this should hold with margin) — report the
  measured number; a `train_multiseat` smoke of a few iterations completes in seconds.
- **G4 — non-regression + byte-identical refactor.** Full suite green (the env
  delegation of §2.2 changes no behavior); A0's `train` still runs; `ruff`/`mypy`
  clean on new/changed files.

## 6. Tests (`tests/test_multiseat_selfplay.py`)

1. **Legality/termination sweep** (G1): drive the collector with a fresh policy on
   Tiny-Heat 2p and 4p (+ one USA case); assert masks, termination, buffer `done`
   invariants.
2. **Stream isolation**: seat buffers have independent lengths; a marker test that
   interleaving would fail — e.g. assert every stored transition of seat *i*'s buffer
   was produced by seat *i* (thread a debug seat-id array through the test's collector
   subclass or assert via episode-return bookkeeping).
3. **Terminal credit**: run one seeded 2p game to completion; recompute
   `_placement_reward` from the final state per seat; assert each seat's last stored
   reward equals it (and earlier rewards are all 0 under default sparse mode).
4. **Truncation bootstrap** (unit): a tiny synthetic check that a truncated episode's
   final reward includes `gamma * V(s_next)` — e.g. run a game with `MAX_ROUNDS`
   monkeypatched low (or a state forced past the cap) and compare against
   `policy.act`'s value on the post-game obs.
5. **Scripted-seat cross-check**: 2 seats, seat 1 scripted `HeuristicAgent` — only
   seat 0 records transitions; behavior matches the A0 env's semantics (this is the
   regression bridge between the two paths).
6. **`train_multiseat` smoke**: 2 iterations, tiny net, Tiny-Heat; all logged losses
   finite; returns list non-empty.
7. **Refactor identity**: existing suite covers `HeatEnv`; additionally assert
   `decode_legal_action`/`forced_action` are the functions `HeatEnv` now calls (e.g.
   the env methods are thin delegations), so drift is structurally impossible.

## 7. Notes for the implementer

- Match conventions: `from __future__ import annotations`, house-style module
  docstrings (see `selfplay/ppo.py`), full type hints, `PYTHONPATH=src` layout,
  dataclass-style config reuse (`A0Config` as-is; no new config class).
- The **only** edits to existing files are: `action_codec.py` (+2 module functions +
  sentinel, bodies moved from env), `env.py` (delegation), `ppo.py` (§4.6 fix).
  Everything else is additive.
- `state.clone(reseed=0)` for every snapshot (never perturb the live RNG).
- Keep `policy.act` single-row batching like A0 — no premature vectorization.
- Run: full suite, the new tests, and a short `scripts/train_selfplay_a2.py --timesteps
  10000` on Tiny-Heat; paste the G3 throughput numbers in your report.
- Do **not** commit; leave changes on the working tree and report results.
