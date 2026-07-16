# Design: Sprint 6E — Strong Heuristic Agent (the strength bar)

> **Status:** planning / design only — no code. Split out of Sprint 6B (see
> `docs/sprint6b-benchmark-eval-design.md`, which now owns only the evaluation
> overhaul). Part of Sprint 6 (`docs/sprint6-roadmap.md`). Buildable independently
> and in parallel with 6A/6B/6C; it is a standalone `BaseAgent` testable in a plain
> `Game` with no ML and no eval harness.

## Why this is its own sub-sprint

The strong heuristic is the single piece that decides whether "strong agent" means
anything: it *is* the strength bar the capstone is measured against. It is the
hardest, most design-heavy, most correctness-risky work unit in the original 6B,
and it has a different shape from the eval overhaul:

- **Different risk.** More logic = more edge cases that can emit an illegal move
  or spin the car out. The eval overhaul is plumbing over existing primitives
  (`run_batch`, `aggregate_stats`); this is novel game-playing logic.
- **Self-contained gate.** "Beats `HeuristicAgent` head-to-head by a margin" is a
  pass/fail that needs no ELO, no CIs, no cross-track — just a seeded `run_batch`.
- **Soft, one-directional coupling.** 6B's eval *wants* this agent as a rung on
  its ladder, but can rank `{Random, Heuristic, MLAgent}` without it
  (`sprint6-roadmap.md §5.2` already lists the two as separate parallelizable
  rows). There is a mild *back*-coupling — tuning this agent's handful of
  constants benefits from 6B's round-robin — but the agent is built to beat the
  old heuristic with principled defaults **first**, then refined once 6B lands.
  Neither blocks the other.

## Goal

A genuinely strong, **opponent-aware, economy-planning** scripted agent, shipped as
a drop-in `BaseAgent` (the same five pull-methods as `HeuristicAgent`,
`heuristic_agent.py:79-334`), exposing a **difficulty ladder** via a single
`strength` parameter so 6B's ELO has a spread of rungs to rank against.

It must clear two bars:
1. **Correctness:** every choice is in the legal set it was handed, across full
   `Game` runs (the engine guards in `driver.py` are only a backstop).
2. **Strength:** beats `HeuristicAgent` head-to-head with a decisive margin, and
   the ladder is monotone (`strength` k+1 beats k over a seeded batch).

## Motivation — what's actually wrong with `HeuristicAgent`

`HeuristicAgent` (`heuristic_agent.py:13-334`) loses to itself the moment you think
about HEAT as an *economy over a whole race* rather than a sequence of independent
turns. Four concrete defects, each mapped to the fix below:

1. **Gear and cards are optimized independently and inconsistently.**
   `choose_gear` (`heuristic_agent.py:79`) scores a gear by an *estimate* of hand
   speed (top-N card values, stress hard-coded at `2.5`, `heuristic_agent.py:93`),
   then `choose_cards` (`heuristic_agent.py:143`) re-derives speed from scratch with
   a *different* scoring formula (`speed * 3 + stress*5`, vs the gear method's
   `gear*10 - corner_cost*10`). The two can disagree: a gear is chosen for an
   estimated card play that `choose_cards` then declines to make. → **Fix: joint
   gear+cards planning through one evaluator.**
2. **Totally opponent-blind.** None of the five methods reads another player's
   state. `choose_slipstream` (`heuristic_agent.py:281`) recomputes its own corner
   cost and never asks *who* it is slipstreaming off (the engine already gates
   eligibility on a rival being 1-2 ahead, `rules.slipstream_eligible:323`). There
   is no blocking (`rules.resolve_blocked_position:515` exists and is never
   exploited), no overtaking intent, no leader/chaser behaviour. → **Fix: read
   `state.players`; target slipstream, block, and modulate risk by race position.**
3. **Hand-tuned magic numbers with no common currency.** `score -= corner_cost*10`
   (`:118`), `score -= 100` spinout (`:121`), `score -= heat_cost*15` (`:128`),
   `score -= corner_cost*12` (`:181`), `score -= 200` spinout (`:185`). These live
   in different methods on different scales and are not comparable. → **Fix: a
   single objective in one unit — expected race progress — so every term is
   derived from the same currency, not invented per call site.**
4. **No race-long heat/deck economy.** Heat is the central resource: you start with
   `HEAT_POOL_SIZE = 6` (`rules.py:24`), spend it on ±2 gear shifts
   (`rules.legal_gear_shifts:31`), corner overage (`rules.corner_heat_cost:184`),
   and boost; you only get it back via cooldown (gear 1 → 3, gear 2 → 1,
   `rules.cooldown_amount:379`, plus +1 adrenaline). Crucially, paid heat goes into
   your *deck* (`PlayerState.pay_heat:163`) and returns later as hand clog that can
   make you *cluttered* and unable to move (`rules.is_cluttered_hand:110`). The old
   `choose_discard` (`heuristic_agent.py:308`) uses a local hand-size rule and there
   is no heat plan across the race. → **Fix: project the heat pool + hand clog over
   the next few corners and pick gears/discards that stay solvent.**

So beating `HeuristicAgent` proves little, and overfitting to exploit it is exactly
why the Sprint-5 policy shattered in self-play (`sprint6-roadmap.md §2`).

## Files to create / modify

| File | Action | What |
|---|---|---|
| `src/heat/agents/strong_heuristic.py` | **new** | `StrongHeuristicAgent(BaseAgent)` + the shared move-evaluator and turn-plan cache. |
| `src/heat/agents/_move_eval.py` | **new (optional split)** | The pure `evaluate_move(...)` value function + heat/corner projection, importable by a future `LookaheadAgent`. May live inside `strong_heuristic.py` if it stays small. |
| `src/heat/simulation/runner.py` | **modify (small)** | Add `strong_heuristic_agent_factory(strength=...)` mirroring `heuristic_agent_factory` (`runner.py:133`) — top-level/`functools.partial`, picklable for `run_batch(parallel=True)`. |
| `src/heat/agents/__init__.py` | **modify** | Export `StrongHeuristicAgent`. |
| `tests/test_strong_heuristic.py` | **new** | Legality, strength vs old heuristic, ladder monotonicity, opponent-awareness behaviours, determinism, latency budget. |

No engine, codec, obs, or action-space changes (`sprint6-roadmap.md §3`).

## Contract

```python
# agents/strong_heuristic.py
class StrongHeuristicAgent(BaseAgent):
    def __init__(
        self,
        name: str = "StrongHeuristic",
        *,
        strength: int = 2,          # 0..3 difficulty ladder (see §"Difficulty ladder")
        heat_price: float | None = None,   # shadow price of 1 heat in spaces; None -> derived
        seed: int | None = None,    # only used for expected-value tie-breaks; agent is otherwise deterministic
    ) -> None: ...
    # choose_gear / choose_cards / choose_react / choose_slipstream / choose_discard
```

- **Drop-in `BaseAgent`** (`base.py:12`): no engine change; the engine pulls the
  five decisions in order (gear → cards → react → slipstream → discard).
- **Deterministic given state.** No global RNG. The optional `seed` only seeds a
  local `random.Random` for tie-breaking between equal-valued plans, so a seeded
  `run_batch` is reproducible (matches the determinism invariant,
  `sprint6-roadmap.md §3`).
- **Picklable** via a top-level factory (`runner.py:122-135` pattern) so it runs in
  parallel batches.

## Core idea — one evaluator, one currency, plan once

Two architectural commitments make this agent *coherent* where the old one is
piecemeal:

### (A) A single objective measured in **spaces of progress**

The race is won by completing all laps first; the natural common currency for
every decision is **expected net spaces of progress over the rest of the race**.
Every term in the evaluator is expressed in that unit, so the weights are *derived*
rather than invented:

```
value(plan) ≈  spaces_gained_this_turn
             − heat_price · heat_spent_this_turn
             − P(spinout) · spinout_loss_in_spaces
             + future_solvency_bonus            # can I still clear the next corners?
             + positional_value                 # slipstream/overtake/block/finish effects
```

- **`spaces_gained_this_turn`** = realized move distance: `calculate_speed(cards)`
  (`rules.py:124`) + expected stress-flip value + planned boost/adrenaline +
  expected slipstream, after `resolve_blocked_position` (`rules.py:515`) is taken
  into account (you don't gain spaces you get bumped back out of).
- **`heat_price`** = the **shadow price of one heat, in spaces.** This is the only
  real free parameter and it is *interpretable*: "how many future spaces is one
  heat worth?" A principled default derives it from the cooldown rate and average
  corner overage (see §"Setting `heat_price`"); it is *not* a bare magic number,
  and 6B's round-robin can refine it empirically later.
- **`P(spinout) · spinout_loss_in_spaces`** replaces the old flat `-100`/`-200`
  (`heuristic_agent.py:121,185`). A spin (`rules.check_spin_out:295`) stops the car
  at the corner, adds stress (`rules.spin_out_stress_count:307`), and burns the
  turn — a quantifiable loss in spaces, weighted by the probability the plan
  actually can't pay (probability because stress flips are random).
- **`future_solvency_bonus`** is the lookahead term (§ next).
- **`positional_value`** is the opponent-aware term (§ "Opponent awareness").

### (B) Plan once at `choose_gear`, execute consistently

The engine asks for gear, then cards, then react, then slipstream as *separate*
callbacks, but a strong move is a *joint* decision. So:

1. At **`choose_gear`**, enumerate the cross-product: for each legal gear `g`
   (`rules.legal_gear_shifts:31`), enumerate `rules.legal_card_plays(hand, g)`
   (`rules.py:70`), score each resulting move with `evaluate_move`, and keep the
   best card-play per gear. Pick the gear whose best plan scores highest. **Cache**
   the whole intended turn — `(gear, card_play, react_intent, slipstream_intent)` —
   keyed by a turn signature (`player_id`, `turn_start_position`, `turn_start_lap`,
   `len(hand)`).
2. At **`choose_cards`**, return the cached `card_play` *after re-validating* it is
   in the `legal_plays` the engine hands in (state can differ; if it's gone,
   recompute against the new legal set). This removes the gear/cards disagreement at
   the root (`heuristic_agent.py:79` vs `:143`).
3. At **`choose_react`** and **`choose_slipstream`**, refine the cached intent
   against the *now-realized* state (actual cards revealed, actual
   `speed_from_cards`, whether eligible). React/slipstream are where the random
   reveal has resolved, so they re-evaluate with real numbers rather than expected
   ones.

This is the structural fix for defect #1 and the substrate for #2–#4.

## Difficulty ladder (`strength`)

A single integer knob produces a spread of rungs so 6B's ELO/TrueSkill has
something to rank — and so 6C can train against a *cheaper* rung if the strongest is
too slow:

| `strength` | Behaviour | Purpose |
|---|---|---|
| **0** | Myopic, single-corner, opponent-blind. Roughly reproduces `HeuristicAgent` through the new evaluator (no lookahead, no positional term). | Sanity floor; should ≈ tie the old heuristic. |
| **1** | Joint gear+cards (B) + single-corner solvency + heat economy. Still opponent-blind. | The "is the *plan* coherent" rung. |
| **2** *(default)* | Adds **two-corner lookahead** and **opponent awareness** (slipstream targeting, risk-by-position, end-game push). | The real strength bar. |
| **3** | Adds **blocking** and **deeper heat projection** (3 corners / deck-quality model). | Top rung; may be slower — fine, it's an opponent, not a trainer. |

The ladder must be **monotone** under head-to-head (test gate). `strength` maps to
the optional `LookaheadAgent` cleanly: that agent reuses the *same* `evaluate_move`
but does explicit engine-clone rollouts (`GameState.clone:137`) instead of the
structured projection — so it is "the same brain with real search," a natural rung
above `strength=3`.

## Decision-by-decision design

### `choose_gear` (joint planner; defect #1, #3, #4)

For each `(new_gear, heat_cost)` in `legal_gears`:
- Skip nothing pre-emptively (the old method's `len(playable) < new_gear` skip,
  `heuristic_agent.py:103`, is wrong when heat-fill is legal — let
  `legal_card_plays` decide, `rules.py:91-103`).
- For each `card_play` in `legal_card_plays(hand, new_gear)`, compute
  `evaluate_move`, subtracting `heat_price · heat_cost` for the shift itself.
- Track the global best `(gear, card_play)`; cache it.

Because the objective already prices heat and corner risk in one currency, the old
ad-hoc penalties (`gear*10`, `-corner_cost*10`, the `heat_available<=2 and gear>=3`
special-case at `heuristic_agent.py:124`) all collapse into the evaluator.

### `choose_cards` (honor the plan)

Return the cached `card_play` if present in `legal_plays`; else recompute the best
play at the *current* gear via the evaluator. (Single-element `legal_plays` short-
circuits exactly as today, `heuristic_agent.py:149`.)

### `choose_react` (boost / cooldown / adrenaline; defect #2, #4)

Now that cards are revealed, decide with real `speed_from_cards`:
- **Cooldown** is almost free value: it un-clogs the hand *and* refills the heat
  pool (`PlayerState.cooldown:180`). Default to cooling the max the gear allows
  (`max_cooldown`), and take the +1 adrenaline cooldown whenever `heat_in_hand`
  exceeds it — same as the old base rule (`heuristic_agent.py:214-221`) but now
  also folded into the heat projection (a planned gear-1 turn is *worth more* when
  the hand is clogged, because it recovers heat).
- **Boost** uses the realized corners-crossed (`rules.corners_crossed:199`) and
  `corner_speed_for_check` (`rules.py:496`): boost only when the extra expected
  spaces beat `heat_price · 1` **and** the boost flip won't push `speed` over a
  crossed corner's limit into unaffordable overage. The old "no corner + heat ≥ 4 →
  boost" rule (`heuristic_agent.py:241`) becomes a special case of the value
  comparison. **End-game override:** on the final lap, heat unspent at the finish is
  *wasted*, so `heat_price → ~0` near the line and boost becomes near-free (modulo
  spinout) — see §"End-game".
- **Adrenaline speed**: take it unless the +1 pushes a crossed corner into overage
  the player can't afford (the old check, `heuristic_agent.py:254-272`, is already
  reasonable; keep it, expressed in the evaluator's currency).

### `choose_slipstream` (target a rival; defect #2)

The engine only calls this when eligible — i.e. a non-finished rival is 1-2 ahead
(`rules.slipstream_eligible:323`), and never when it would cross the finish
(`rules.slipstream_would_cross_finish:358`). The old method ignores all of that and
just recomputes its own corner cost (`heuristic_agent.py:281-306`). Strong version:
- Compute the +2 move's corners-crossed and their cost **at the player's
  card-speed** — note slipstream movement is *excluded* from the corner speed check
  (`rules.corner_heat_cost` docstring + `corner_speed_for_check:496`), so a
  slipstream can shove you *through* a corner while the check uses your (lower) card
  speed. This is a genuinely good deal the old agent under-uses.
- Value it as `+2 spaces − heat_price·cost − P(spinout)·loss`, plus a
  **positional bonus** if the +2 lands the player ahead of / alongside a rival
  (overtake) or closes a gap that sets up next turn's slipstream. Take it when
  positive. This naturally declines a slipstream that would dump you into an
  unaffordable corner — without the old hard `cost <= 2` cap (`:304`).

### `choose_discard` (deck cycling; defect #4)

Replace the local hand-size rule (`heuristic_agent.py:320`) with a deck-quality
view over `rules.legal_discards` (`rules.py:615`, Speed/Upgrade only):
- Estimate **expected future hand speed** from the multiset of cards remaining in
  deck + hand (the agent knows its own deck *contents*, just not the shuffled
  order). Discarding a low value-1 speed card raises the *mean* of what you'll
  redraw — a derivable improvement, not a fixed "discard all value-1" rule.
- Cycle more aggressively when the hand is rich and the next corners are far
  (room to redraw), and **never** discard down to a cluttered hand
  (`rules.is_cluttered_hand:110`) for the upcoming gear. Never discard upgrades
  (kept from the old logic, `heuristic_agent.py:328`).

## Multi-corner lookahead & heat projection (`strength` 2–3)

The "future_solvency_bonus" is a cheap **structured projection**, not a game-tree
search:

1. From the candidate end position, find the next 1–2 corners
   (`rules.distance_to_next_corner:260`, iterated).
2. Estimate the speed the agent will *want* to arrive at each (expected hand speed
   from the deck-quality model), and the resulting overage cost
   (`rules.corner_heat_cost:184`).
3. Project the **heat trajectory**: start from `heat_available`, subtract this
   turn's spend, add expected cooldown for the planned gears
   (`rules.cooldown_amount:379`), subtract projected corner overage. If the
   trajectory goes negative *before* a corner is cleared, that plan risks a future
   spin → large `future_solvency` penalty. This is the term that stops the agent
   from dumping heat now and stalling at the next corner (the old agent's blind
   spot).

This is O(corners × plays) per turn with tiny constants — bounded and fast. No
engine cloning is needed at `strength ≤ 3`; cloning is reserved for the optional
`LookaheadAgent`.

## Opponent awareness (`strength` 2–3; defect #2)

Read `state.players` (positions, laps, gears, finished flags). Derive the agent's
**rank** (sort by `(lap, position)`) and the gaps to the car ahead/behind.

- **Slipstream targeting** — already covered in `choose_slipstream`; uses rival
  positions the engine's eligibility check guarantees exist.
- **Risk modulation by position.** Behind (especially trailing → adrenaline-
  eligible, `rules.adrenaline_eligible:397`): raise tolerance for heat spend and
  small spinout risk — push. Leading comfortably: lower `heat_price` is *wrong* (you
  want to conserve), so *raise* effective spinout aversion and bank heat; manage the
  gap to the chaser.
- **Blocking (`strength` 3).** Near a low-lane space or a corner, prefer a move that
  lands on the space that fills the lane(s) ahead of a *trailing* rival, forcing
  them back via `rules.resolve_blocked_position:515`. This is purely a positional
  bonus in the evaluator; it never makes an illegal move.
- **Avoid self-blocking.** Conversely, deprioritize a move whose target space is
  already full (you'd get bumped back, losing the spaces you paid heat for).

## End-game

Two finish-line facts change the math on the final lap:
- **Heat is use-it-or-lose-it.** Unspent heat at the finish buys nothing, so
  `heat_price` decays toward ~0 as `distance_to_finish` (the old
  `_distance_to_finish:54` helper, kept) shrinks on the final lap — making boost
  and ±2 shifts near-free to spend down the pool.
- **Slipstream can't cross the line** (`rules.slipstream_would_cross_finish:358`),
  so the end-game push relies on cards + boost + adrenaline, not slipstream.

## Setting `heat_price` (the one real parameter)

Principled default, then optional empirical refinement:
- **Derivation.** One heat buys, on average, the ability to exceed a corner limit by
  1 (≈ keeps ~1–2 spaces of speed you'd otherwise have to shed) or a ±2 shift
  (tempo). One gear-1 turn recovers up to 3 heat at the cost of a slow turn. A
  defensible starting value is `heat_price ≈ 1.0–1.5 spaces/heat`, scaled down near
  the finish. The point is it's an *interpretable* quantity, unlike `*10`/`*15`.
- **Refinement (optional, post-6B).** Sweep `heat_price ∈ {0.75, 1.0, 1.25, 1.5}`
  through 6B's `round_robin_elo` and pick the rung with the highest rating. This is
  the soft back-coupling to 6B — *not* a blocker; the derived default already beats
  the old heuristic.

## Test gates (`tests/test_strong_heuristic.py`)

- **Legality (hard).** Across many turns of full `Game` runs at every `strength`,
  assert each `choose_*` return is in the legal set handed in (gear ∈ `legal_gears`;
  card play ∈ `legal_plays`; discard ⊆ `legal_discards`; cooldown ≤ allowed). Assert
  directly — engine guards are a backstop only (mirrors the `MLAgent` legality
  approach, `sprint5-ml-roadmap.md:489`).
- **No spinout regression.** Over a seeded batch, `StrongHeuristicAgent` spins out
  strictly less often than `HeuristicAgent` on the same seeds.
- **Strength (the headline gate).** `StrongHeuristicAgent(strength=2)` beats
  `HeuristicAgent` head-to-head over N seeded games with win-rate well above 50% by
  a margin (concrete pass/fail). Use `run_batch` with a fixed `seed`.
- **Ladder monotonicity.** `strength` k+1 beats k head-to-head (k = 0,1,2) over a
  seeded batch; `strength=0` ≈ ties the old heuristic (within a tolerance band).
- **Joint plan consistency.** The gear chosen at `choose_gear` and the play returned
  at `choose_cards` correspond to the *same* cached plan (no internal disagreement);
  on a forced state change the recompute path still returns a legal play.
- **Opponent-awareness behaviours (targeted unit assertions, not aggregate).**
  In a constructed state where a rival offers a clearly-good slipstream, the agent
  takes it; where a boost/slipstream would overshoot an unaffordable corner, it
  declines; (`strength=3`) where blocking a trailing rival is available and cheap,
  it chooses the blocking move.
- **Determinism.** Same state → same decision (no global RNG); a seeded `run_batch`
  is byte-reproducible across runs.
- **Latency budget.** Per-decision cost is benchmarked and bounded (e.g. assert a
  full game at `strength=2` completes within an order of magnitude of
  `HeuristicAgent`); guards against the lookahead blowing up eval/training time.
- **Picklability.** `strong_heuristic_agent_factory(strength=k)` pickles and runs
  under `run_batch(parallel=True)` (top-level/`partial`, no lambdas —
  `runner.py:150-167` pre-flight check).
- **Full suite green:** `PYTHONPATH=src python -m pytest tests/ -q`.

## Build sequence

1. **Evaluator core** — `evaluate_move(...)` + heat/corner projection as a pure
   function over `(state, player_id, candidate move)`, unit-tested in isolation on
   hand-built `GameState`s.
2. **`StrongHeuristicAgent` `strength` 0–1** — joint gear+cards plan cache,
   single-corner solvency, heat economy, deck-cycling discard. Gate: `strength=1`
   beats `HeuristicAgent`; `strength=0` ≈ ties it.
3. **`strength` 2** — two-corner lookahead + opponent awareness (slipstream
   targeting, risk-by-position, end-game). Gate: the headline strength gate +
   opponent-behaviour units.
4. **`strength` 3** — blocking + deeper projection. Gate: ladder monotonicity.
5. **Factory + exports + picklability**; benchmark latency.
6. **(Hand-off to 6B)** the agent is now a rung in the ELO ladder; optional
   `heat_price` sweep once 6B's `round_robin_elo` exists.
7. Full suite.

## Risks + de-risking

| Risk | De-risking |
|---|---|
| More logic → illegal move or accidental spinout. | Build strictly on `rules.legal_*` enumerations; never construct moves raw. The legality + no-spinout-regression gates run full games and assert directly. |
| Lookahead is slow → drags eval and 6C training. | Structured projection (no engine cloning) at `strength ≤ 3`; latency-budget gate; the `strength` knob lets training pick a cheaper rung. |
| "Stronger" asserted but not real / not monotone. | The head-to-head margin gate and ladder-monotonicity gate are concrete pass/fail, not vibes. |
| `heat_price` is just a new magic number. | It's a single *interpretable* quantity (spaces per heat) with a derived default; everything else flows from the one currency; optional empirical sweep via 6B. |
| Gear/cards plan desync if state changes between callbacks. | Cache keyed by a turn signature; `choose_cards` re-validates against the handed legal set and recomputes on miss (tested). |
| Non-determinism creeps in. | No global RNG; optional local `Random(seed)` only for equal-value tie-breaks; determinism + reproducible-batch gates. |

## Non-goals

- **No engine rule changes.** Uses only existing `rules.*` and public `state`
  (`sprint6-roadmap.md §4`).
- **No obs/action/codec changes**, no `CODEC_VERSION` bump (`sprint6-roadmap.md §3`).
- **No full game-tree search / MCTS.** The `strength` ladder tops out at a
  structured 2–3-corner projection; explicit engine-clone rollout is deferred to the
  *optional* `LookaheadAgent` (owned by 6B), which reuses this evaluator.
- **No "ML beats StrongHeuristic" as a unit gate.** This sub-sprint builds the
  *bar*; measuring the trained agent against it is the capstone's manual metric.
- **No new dependency.** Pure Python over the existing engine.

## Open questions / assumptions

- **Naming/numbering.** Shipped as **6E** to avoid renumbering 6A/6B/6C/6D;
  `sprint6-roadmap.md` and `sprint6b-...` are updated to point here. (Alternative:
  fold back as 6B-i — rejected per the structure decision.)
- **Where the evaluator lives.** Assumed inline in `strong_heuristic.py` unless it
  grows enough to warrant `agents/_move_eval.py`; the only hard requirement is that
  it stay a *pure* function so the optional `LookaheadAgent` can import it.
- **`heat_price` default + final-lap decay curve.** Assumed `~1.0–1.5` spaces/heat
  decaying to ~0 over the final lap; exact curve is a tuning detail, surfaced as a
  parameter with a fixed default for reproducibility.
- **Expected stress-flip value.** The old agent hard-codes `2.5`
  (`heuristic_agent.py:93`). Assumed replaced by the mean basic-card value of the
  agent's *own* remaining deck (derivable from `PlayerState.deck` contents); verify
  the deck multiset is queryable without consuming draw order.
- **Lookahead corner count at `strength=3`.** Assumed 3; capped if the latency
  budget is threatened.
