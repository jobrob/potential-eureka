# Sprint C10 — incremental round advance (kill the replay re-simulation)

> **The engine-side throughput sprint, and an honest reframing.** C7 cashed in the
> CPU cores (~8× parallel workers); C8 took the determinism-safe per-leaf neural
> wins and found the residual wall is `net-fwd`. This sprint looks at the *other*
> half of per-move time — the engine — and finds that the cost is **not cloning**
> (a common assumption) but **re-simulating the round from its start on every tree
> edge**. It proposes converting that O(depth²) replay into O(depth) single-step
> advance. The ceiling is **modest and honestly reported up front**; read the
> "Is this worth doing?" section before committing.

## Profiling motivation (measured, not assumed)

Serial `bench_selfplay.py` on the Tier-0 config (`c5_main_best.zip`, two-player,
16 sims, 4 games, codec v3) — cProfile `tottime`, 52.3s wall:

| phase | share | what it is |
|---|---|---|
| `net-fwd` | ~34% | torch forward (`linear` 8.2s + module call machinery) — the C8 residual / C9 territory |
| `encode` | ~21% | `features.py` obs build (per leaf) |
| **engine replay** | **~19%** | `_advance` → `run_round_driver` re-simulation |
| `clone` (pure copy) | **~3.5%** | `GameState`/`PlayerState`/`Deck.clone` list copies |
| other plumbing | ~22% | MCTS bookkeeping, SB3 wrappers |

**The headline correction:** the per-move profile reports `clones/move ≈ 15.9`, which
*looks* like the engine cost is copying. It is not. Pure cloning is **~3.5%**
(`game_state.py:137` cumtime 1.17s over 53,761 calls). The real engine cost is the
**replay**: `_advance` (`mcts_agent.py:437`) clones the round-start state and then
**re-drives `run_round_driver` from the top of the round**, forcing the whole
recorded `action_path` plus one new action. `run_round_driver` is therefore entered
**397,542 times across 53,761 `_advance` calls (~7.4 re-drives per edge)**, and each
re-drive re-runs the genuine game rules (`step_reveal_and_move` 1.36s,
`phase_play_cards` 0.85s, `step_check_corner` 0.72s, `phase_shift_gears` 0.68s,
`step_replenish` 0.34s, the `legal_*` generators, …).

### Why replay exists (the obstacle the fix must clear)

`run_round_driver` (`driver.py:76`) is a **generator**. Its position within the round
— the `for player in active_players` loop index, the collected `gear_decisions` /
`card_decisions` dicts, `pre_play_hands`, the accumulated `events` — lives in the
**generator frame, not in `GameState`**. A live generator cannot be cloned or pickled
in CPython, so a tree node cannot store "the round, paused here." The current design's
only way to realize a child transition is to **start a fresh generator from round-start
and replay**. That is correct and deterministic — but it pays the full round prefix on
every edge. (See the `EngineTransitionModel` docstring at `mcts_agent.py:356`.)

## The win this sprint takes

Replace replay-from-round-start with **single-decision advance**. Lift the driver's
in-round position out of the generator frame into an explicit, clonable **round
cursor**, so a node can store `(GameState, RoundCursor)` and a `step` does:

1. clone the **parent** node's `(state, cursor)` (one fork, same copy cost as today —
   clone is already cheap), and
2. **advance exactly one decision** (apply the forced action, run intervening
   non-decision phases, stop at the next decision),

instead of cloning round-start and re-driving `action_path + [next_action]` from the
top. Total driver-step work drops from `Σ depth(edge)` to `Σ 1` — the measured ~7.4×
fewer driver entries on this config.

### Folded-in cheap win (do this regardless, it is free and safe)

`_advance` forces `clone.logging_enabled = True` on **every** throwaway replay
(`mcts_agent.py:461`) so leaf spin-accounting can read `spin_out` events. That makes
the logging-gated work — `pre_play_hands` (`driver.py:108` listcomp, 0.33s), the
per-turn `hand_repr` + `distance_to_next_corner` + `turn_start_data` block, `log_event`
(0.07s) — run on all ~398k re-drives and then get discarded. **Detect spins directly**
(read `player.spun_out` after the move, or a lightweight counter) and leave
`logging_enabled` **off** during search clones. Worth ~1–1.5% on its own, and it
**does not depend on the cursor rewrite** — ship it first as a standalone, low-risk
change.

## Scope

### In scope
1. **Round cursor**: a small dataclass capturing the driver's in-round position
   (phase + sub-step, partial `gear_decisions`/`card_decisions`, `turn_order` index,
   `events` accumulator). Clonable and a pure function of progress.
2. **Re-entrant driver / single-step API**: an `advance_one(state, cursor, action) ->
   (next_decision | terminal, new_cursor)` that runs intervening auto-phases and stops
   at the next decision — the same transitions `run_round_driver` makes, decomposed
   into steps. Keep `run_round_driver` as the canonical reference (the equivalence
   oracle) and ideally **express it as a thin loop over `advance_one`** so there is one
   source of truth, not two engines to keep in sync.
3. **Re-point `EngineTransitionModel`** to store `(state, cursor)` per node and `step`
   via `advance_one`, removing the replay loop. Preserve the chance-edge semantics
   (round-boundary crossing = replenish draw) and the `reseed` RNG contract **exactly**.
4. **Drop forced replay-logging** (the cheap win above).

### Out of scope
- Batched-GPU `net-fwd` (the larger 34% slice — the conditional GPU sprint C8 pencilled
  as C9; separate).
- Lean observation `encode` (~20%) — taken first as **Sprint C9**
  ([`sprint-C9-lean-observation-encode.md`](sprint-C9-lean-observation-encode.md));
  larger slice, lower risk than this engine rewrite.
- `encode` optimization (the 21% slice — a real and *higher-ceiling* target, but a
  different subsystem; see recommendation).
- Any change to the rules themselves, the codec, or the learning targets.

## The determinism / equivalence contract (first-class risk)

This touches the engine, so it is the highest-risk change in the C-series. The bar:

- **Byte-identical transitions.** For every `(round_start, action_path, next_action,
  reseed)`, the new single-step path must produce a `StepResult` **equal** to the
  current replay path — same `state`, same `decision`, same `is_chance`, same terminal
  flag. Add a **differential test** that drives a battery of real search states through
  BOTH the replay engine and the cursor engine and asserts equality (including RNG
  stream position, so chance edges match).
- **Reproducibility unchanged.** Same `(state, seed)` ⇒ identical move; all existing
  "same seed twice → byte-identical" tests stay green.
- **`run_round_driver` parity.** If it is refactored to loop over `advance_one`, the
  full existing engine/driver test suite is the regression oracle — it must stay green
  with zero golden re-baselines (a pure refactor, not a behaviour change).
- **Behaviour re-validation.** A 1–2-generation Tier-0 slice via `az_loop_1v1` produces
  an equivalent read-(b)/ECE curve to the pre-C10 path.

## Success criteria

- **Primary (measured):** engine-replay phase shrinks from ~19% toward the clone floor
  (~3.5%) on the same `bench_selfplay` config; driver entries per `_advance` drop from
  ~7.4 to ~1. Realized end-to-end speedup **reported honestly** (expected ~1.1–1.15×;
  see below).
- **Equivalence:** differential replay-vs-cursor test green to exact equality; all
  C0–C8 + engine tests green with no golden re-baseline.
- **Honest reporting:** updated phase breakdown recorded; explicit statement of the new
  residual wall.

## Is this worth doing? (read before committing)

**The ceiling is modest and the risk is real — this section is the deliverable as much
as the design is.** Engine replay is ~19% of wall; driving it to the clone floor saves
at most ~15 points → a **~1.1–1.15× end-to-end** speedup, in the same ballpark as C8.
Against that: it is an **engine rewrite under a byte-identical contract** — the most
delicate change in the series.

Three honest options:

1. **Ship only the free win** (drop forced replay-logging), skip the cursor rewrite.
   ~1–1.5×%, near-zero risk. **Recommended floor.**
2. **Do the full cursor rewrite** for the ~1.1–1.15×. Justified only if engine becomes
   the binding constraint after C9, and only with the differential test as a gate.
3. **Redirect to `encode` (21%) instead.** Larger slice than engine replay, lower risk
   than an engine rewrite: the per-leaf obs build re-counts deck composition and
   re-walks track blocks every leaf (`features.py:86`/`:206`/`:303`); much is invariant
   within a search and cacheable. **This is likely the better next bet than the engine
   cursor** if the goal is throughput-per-unit-risk.

**Recommendation:** take option 1 now (free, safe — and now folded into **Sprint C9**'s
scope), and **prefer the `encode` sprint (C9) over the full cursor rewrite** unless
profiling after C9 shows the engine is the wall. The cursor rewrite is documented here so
the option is on the table with its true cost.

## Dependencies
- C0–C8 built and green; `bench_selfplay.py` is the measurement instrument.
- No GPU needed (engine + search are CPU; this is orthogonal to C9).

## Effort
- Free win (drop replay-logging + direct spin detect): small, low risk.
- Round cursor + re-entrant driver + differential equivalence test: **moderate–large**
  and delicate (the byte-identical contract is the hard part, not the code volume).
- Benchmark + Tier-0 re-validation: ~½ day compute/analysis.
