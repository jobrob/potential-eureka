# Design: Sprint 6E (redesign) — Strong Heuristic Agent, relative-objective rebuild

> **Status:** planning / design only — no code. **Supersedes the implementation
> approach in `docs/sprint6e-strong-heuristic-design.md`.** That doc's *intent*
> still stands (a strong, opponent-aware, economy-planning scripted `BaseAgent`
> exposing a difficulty ladder for 6B's ELO); this doc replaces *how* it gets
> there, because a code review of the WIP found the goals are not met for an
> **architectural** reason, not a tuning miss.
>
> Read the original first — this doc reuses its file table, contract, and
> validation-gate format, and only re-specifies the parts the review flagged.
> Part of Sprint 6 (`docs/sprint6-roadmap.md`); buildable independently of
> 6A/6B/6C as a standalone `BaseAgent` testable in a plain `Game`.

## Summary

The WIP `StrongHeuristicAgent` is correct, deterministic, and legal, but it does
not meet either headline goal: it does not decisively beat the weak
`HeuristicAgent`, and its `strength` rungs do not measurably separate. The root
cause is the **objective**. The evaluator
(`_move_eval.evaluate_move`) maximizes the agent's *own absolute expected net
spaces of progress*. That quantity is already near-optimal at the margin for a
sensible card/gear choice, so the strength-2/3 features (opponent awareness,
lookahead, blocking) are **collinear with own-progress** and add ~0 measurable
strength (seat-neutral diagnostic over ~900 games: every feature 0.500–0.509 vs
strength 1). HEAT is a **relative, zero-sum finishing-order race**; an agent that
only maximizes its own progress has no lever to pull once its own move is locally
optimal.

This redesign keeps everything the review called solid (the one-currency
evaluator structure, plan-once cache, the deliberately-crippled strength-0 floor,
correct slipstream/corner-check handling, the worst-case-solvent boost fix) and
replaces the **objective** with a genuinely relative one, fixes a lap-accounting
bug that silently disabled opponent-awareness on finish-crossing turns, replaces
an inert solvency term, re-specifies spin-out aversion, and replaces the fragile
strict-adjacent-monotonicity ladder gate with a robust margin/floor/ELO gate.

## Motivation — why the current design cannot separate

### P0 (root cause): the objective is absolute, not relative

`evaluate_move` (`_move_eval.py:342-463`) returns

```
value = spaces_gained                       # own progress this turn
      - heat_price * heat_spent
      - p_spinout * SPINOUT_LOSS_SPACES
      + project_solvency(...)               # currently inert (P2)
      + _overtake_bonus(...)                 # tiny, capped 0.6, and buggy (P1)
      + block_bonus(...)                     # tiny, capped 0.15 (P3)
```

The dominant term is `spaces_gained` (typically 4–16 spaces). The positional
terms are bounded at **0.6** (`_overtake_bonus`, `_move_eval.py:489`) and **0.15**
(`block_bonus`, `_move_eval.py:322`). A bonus that small can only break a tie
between two plans whose own-progress is *already within 0.6 spaces of equal* — and
the agent already picks the highest own-progress plan, so those plans are rarely
the ones in contention. The features are, in effect, never decisive. Worse, the
"opponent-aware" bonus is **monotone in own forward progress** (passing a rival is
mechanically the same event as moving far), so it does not encode anything the
absolute objective wasn't already rewarding. That is why every strength rung lands
at ~0.50.

**The fix is to make the objective relative**: score a plan by how it changes the
agent's *finishing-order prospects against the actual field*, not by its own
spaces in isolation. Concretely, value a plan by **expected gap-to-rivals change**
and a **lead-dependent risk posture**, in a way that is *not* collinear with
own-progress. Spending heat to deny a rival a slipstream, conceding a space to
bank heat while leading, or burning the pool to overtake while trailing are all
moves an own-progress maximizer rates identically to their no-opponent
counterparts — a relative objective rates them differently. Details in
[Proposed Design](#proposed-design).

### P1: lap-accounting bug disables opponent-awareness on finish crossings

In `evaluate_move` (`_move_eval.py:417-421`):

```python
new_pos, _ = rules.calculate_move_position(
    from_position, int(round(expected_speed)), track, from_lap
)
effect = resolve_landing(state, player_id, new_pos)
spaces_gained = expected_speed - effect.spaces_lost_to_block
```

`calculate_move_position` returns `new_pos` **modulo `track.length`** and discards
the `crossed_finish` boolean. `evaluate_move` then passes the *original*
`from_lap` to `resolve_landing` and to `_overtake_bonus` (`:448-450`).
`_overtake_bonus` computes `end_prog = landed_lap * length + landed_pos`
(`_move_eval.py:480`) with the stale lap, so on any finish-crossing move
`end_prog` *drops below* `start_prog` and the overtake test
`start_prog <= op < end_prog` (`:487`) is vacuously false. Opponent-awareness
silently switches off precisely on the high-value laps. The same stale-lap bug is
in `choose_slipstream` (`strong_heuristic.py:565-584`, which never recomputes the
landed lap) and `block_bonus` (`_move_eval.py:314`).

**Fix:** compute the landed lap from the move and thread it through every
positional helper (precise spec in [Detailed Changes](#detailed-changes)).

### P2: inert solvency, and spin-out tuning that regresses

`project_solvency` (`_move_eval.py:178-230`) credits `cooldown_amount(planned_gear)`
**every** corner in the horizon and subtracts only a fixed `_EXPECTED_CORNER_OVERAGE = 0.5`
(`:58`) per corner. For any planned gear 1–2 the per-corner cooldown (3 or 1)
swamps the 0.5 overage, so the projected pool monotonically *rises* and
`worst_shortfall` is essentially never negative — the term returns 0.0 in normal
play. It is dead weight. The redesign replaces it with a solvency model that
actually projects the heat trajectory against realistic per-corner cost.

Separately, the no-spinout-regression gate is currently failing the wrong way
(strong ~116 spins vs weak ~103 over the seed batch). The spin-out aversion model
(`p_spinout` step function at `_move_eval.py:396-414` + `SPINOUT_LOSS_SPACES = 8.0`)
must be re-specified so strong spins **strictly less** than weak with margin. Note:
the dominant spin-out source — boost ignoring boost-induced extra movement and
flip variance — was already diagnosed and fixed in the WIP (worst-case-solvent
boost using the max owned Basic, `strong_heuristic.py:456-500`). **Keep that
approach; document it as settled.** Remaining spins come from card plays whose
realized stress/adrenaline flips overshoot a corner the mean did not.

### P3: block heuristic and react planning are proxies, not projections

`block_bonus` (`_move_eval.py:291-323`) awards a flat 0.15 when a single-lane
landing space sits 1–2 ahead of a trailing rival. It is a proximity proxy: it
does not check that the rival would actually be bounced, and at 0.15 it cannot
flip a card choice against any real progress difference. The redesign specifies a
**real block projection** (does landing here actually fill the lane(s) the rival
needs, and what does `resolve_blocked_position` do to them) priced in the relative
currency. Optionally, extend the plan-once cache into the react phase.

## Current Behavior (the WIP, as reviewed)

- **Beats weak heuristic:** marginally / not decisively (target was a decisive
  margin; `test_strength2_beats_heuristic` asserts `> 0.62` win-share and this is
  not reliably met).
- **Ladder:** rungs do not separate (every seat-neutral per-feature delta is
  0.500–0.509 vs strength 1; `test_ladder_monotone` is fragile and effectively
  measures noise).
- **Spin-outs:** strong regresses vs weak (~116 vs ~103).
- **Legality / determinism / picklability / latency:** all good. Keep.

## Proposed Design

### Overview

Replace the absolute objective with a **relative finishing-order objective** built
from two opponent-coupled components that are provably *not* collinear with
own-progress: **(1) expected gap-change against the nearest rivals**, priced in
spaces, and **(2) a lead-dependent risk/heat posture** that shifts the *effective
heat price and spin-out aversion* by race rank. Fix the lap bug so these
components see correct positions on finish-crossing turns. Replace `project_solvency`
with a real forward-heat projection. Re-tier the ladder so each rung adds a
component that *demonstrably* moves the relative objective, and replace the
strict-adjacent-monotonicity gate with a robust margin/floor/ELO gate.

### The new relative objective (precise definition)

Let the field be the non-finished players plus the focal player. Define
lap-aware progress (already in code as `race_progress`, `_move_eval.py:238-240`):

```
prog(p) = p.lap * track.length + p.position
```

For the focal agent considering a plan that lands it at `(end_lap, end_pos)` after
blocking, define `prog_self_after = end_lap * track.length + end_pos`. For each
rival `r`, define the **signed gap** before and after *this agent's* move (rivals
have not moved yet this turn, so their `prog(r)` is fixed within the decision):

```
gap_before(r) = prog_self_before - prog(r)
gap_after(r)  = prog_self_after  - prog(r)
```

The objective is:

```
value(plan) =  alpha * own_progress_term          # keep, but down-weighted
             + RELATIVE_WEIGHT * relative_term     # the new lever (P0)
             - heat_price_eff * heat_spent          # heat_price_eff is rank-adjusted
             - p_spinout * spinout_loss_eff         # spinout_loss_eff is rank-adjusted
             + solvency_term                         # real projection (P2)
             + block_term                            # real projection (P3, strength 3)
```

where:

- **`own_progress_term` = `spaces_gained`** (`expected_speed - spaces_lost_to_block`),
  exactly as today. Kept because raw progress is genuinely most of what wins a
  race; `alpha` defaults to **1.0**.

- **`relative_term`** is the part that is *deliberately decoupled* from own
  progress. It is **not** "did I pass someone" (that is collinear). It is the
  **expected change in the closest competitive gaps, weighted toward the rivals
  that decide my finishing position**:

  ```
  relative_term = sum over the K nearest rivals r of
                    w(r) * f( gap_after(r) ) - w(r) * f( gap_before(r) )
  ```

  with a **saturating** gap-value function `f` so that the marginal value of
  pulling away from a rival you are already crushing is near zero, and the
  marginal value of closing/holding a *contested* gap (|gap| small) is high:

  ```
  f(g) = SCALE * tanh( g / GAP_SOFTNESS )
  ```

  `f` is monotone but **concave for g>0 and convex for g<0**, so `f(gap_after) -
  f(gap_before)` is *largest in magnitude when the gap is near zero* — i.e. when
  the agent is wheel-to-wheel with a rival, which is exactly where own-progress is
  a poor proxy for finishing prospects. Two plans with identical own-progress but
  different opponent geometry (one closes a contested 1-space gap, one extends a
  10-space lead) now score differently. **This is the structural break from the
  absolute objective.**

  - `K` = number of rivals considered (default **2**: the car directly ahead and
    the car directly behind by `prog`). Considering only the nearest contestants
    keeps the term from being dominated by far-off cars and keeps it cheap.
  - `w(r)` weights the car *ahead of me* (the one I must pass to gain a place) and
    the car *behind me* (the one that can pass me) more than others; default
    `w = 1.0` for both nearest neighbours, `0` beyond `K`.
  - `GAP_SOFTNESS` (spaces): the gap scale over which `f` saturates. Default
    **3.0** (≈ a typical single-turn move), so contests within a move's reach are
    the steep region.
  - `SCALE` (spaces): the max value of holding/winning one contest. Default
    **2.0** so a fully-won contest is worth ~2 spaces of progress — comparable to
    a gear shift, large enough to *flip* a close card choice (unlike the old
    capped-0.6 bonus) but not so large it buys progress with heat irrationally.
  - `RELATIVE_WEIGHT` default **1.0**; surfaced as a constructor-derived scalar so
    6B can sweep it.

  **Why this is not collinear with own-progress:** holding a defensive line
  (small own-progress, but `gap_after(behind rival)` preserved) and over-extending
  a lead (large own-progress, but `f` already saturated so `relative_term ≈ 0`)
  are scored *oppositely* to the absolute objective. The diagnostic that found
  every feature at ~0.50 was measuring additive bonuses on top of an unchanged
  argmax; here the argmax itself changes.

- **`heat_price_eff`** and **`spinout_loss_eff`** are the **lead-dependent risk
  posture** (component 2). These replace the WIP's tiny ±10% price nudges
  (`strong_heuristic.py:123-129`) with a posture that actually changes behaviour:

  ```
  rank, total = race_rank(state, player_id)        # 0 = leader
  lead_frac = 1 - rank / max(1, total - 1)          # 1.0 leader, 0.0 last
  heat_price_eff   = base_heat_price * (1 + LEAD_PRICE_SWING * (lead_frac - 0.5) * 2)
  spinout_loss_eff = SPINOUT_LOSS_SPACES * (1 + LEAD_RISK_SWING * (lead_frac - 0.5) * 2)
  ```

  - Leading (`lead_frac→1`): heat is *dearer* and spinning is *costlier* →
    conserve, bank heat, refuse marginal risk. A leader has more to lose from a
    spin than a trailer.
  - Trailing (`lead_frac→0`): heat is *cheaper* and spin-loss is *discounted* →
    push, spend the pool, accept variance to make up places. A trailer that
    finishes "safely last" gains nothing; variance is its friend.
  - `LEAD_PRICE_SWING` default **0.4** (±40% around base), `LEAD_RISK_SWING`
    default **0.3**. These are *posture* swings, deliberately larger than the
    WIP's ±10% so they bite. End-game decay (heat→worthless near the line on the
    final lap) is applied **after** the rank adjustment, multiplicatively, keeping
    the existing decay-over-final-8-spaces curve (`strong_heuristic.py:115-120`).

  This component is also non-collinear with own-progress: it changes *which* plan
  is optimal as a function of standing, not of distance.

### Why the two components together separate the ladder

| Component | Moves the argmax when… | Collinear with own-progress? |
|---|---|---|
| `relative_term` (gap, saturating) | wheel-to-wheel with a contested rival | No — saturates on uncontested gaps |
| risk posture (`*_eff`) | leading vs trailing | No — depends on rank, not distance |
| real solvency (P2) | a future corner would force a spin | Partly, but corrects a *failure* |
| real block (P3) | landing actually bounces a rival | No |

Each rung turns on a component that changes *decisions*, not just adds a tiebreak,
so the rungs separate measurably. The strength-0 floor stays absolute and myopic
(it must ≈ tie the weak heuristic), giving the ladder room to climb.

### Detailed Changes

#### `src/heat/agents/_move_eval.py`

- **New tunables** (top of file, "spaces" currency, replacing/adding to
  `:33-62`):
  - `RELATIVE_WEIGHT: float = 1.0`
  - `GAP_SOFTNESS: float = 3.0`
  - `GAP_SCALE: float = 2.0`
  - `NEAREST_RIVALS_K: int = 2`
  - `LEAD_PRICE_SWING: float = 0.4`
  - `LEAD_RISK_SWING: float = 0.3`
  - Keep `DEFAULT_HEAT_PRICE = 1.25`, `SPINOUT_LOSS_SPACES` (retune below).
  - **Remove** `_EXPECTED_CORNER_OVERAGE` and `SOLVENCY_PENALTY_PER_HEAT` if the
    solvency rewrite no longer uses them (see below); otherwise re-document.

- **Fix the lap bug (P1).** Add a helper and use it everywhere a landed position
  is computed:

  ```python
  def landed_lap_and_pos(
      track: Track, from_position: int, from_lap: int, spaces: int
  ) -> tuple[int, int]:
      """Lap-aware landing: returns (end_lap, end_pos) after moving `spaces`."""
      laps = rules.laps_completed_by_move(from_position, spaces, track)
      new_pos, _ = rules.calculate_move_position(from_position, spaces, track, from_lap)
      return from_lap + laps, new_pos
  ```

  Use `rules.laps_completed_by_move` (`rules.py:465-480`) — it already handles a
  single move spanning ≥2 laps. In `evaluate_move` replace `:417-421` so
  `effect`, `_overtake_bonus`/`relative_term`, `block_term`, and
  `project_solvency` all receive `end_lap` not `from_lap`. `prog_self_after` then
  uses `end_lap`.

- **New `relative_term` function.** A pure function over `(state, player_id,
  prog_self_before, prog_self_after)` implementing the gap-saturation formula
  above. It iterates `state.players`, computes signed gaps to each non-finished
  rival, keeps the `K` whose `|gap_before|` is smallest, and sums
  `w(r) * (f(gap_after) - f(gap_before))`. `f(g) = GAP_SCALE * tanh(g / GAP_SOFTNESS)`.
  **Replaces** `_overtake_bonus` (`:466-489`) — delete it; it is collinear and
  buggy.

- **New risk-posture helper.** `effective_risk(state, player_id, base_heat_price)`
  returning `(heat_price_eff, spinout_loss_eff)` per the rank formula. The agent's
  `_effective_heat_price` (`strong_heuristic.py:102-131`) folds into this; the
  end-game decay stays.

- **Rewrite `project_solvency` (P2).** Model the heat trajectory honestly:

  ```
  heat = heat_after_turn
  pos, lap = end_pos, end_lap
  for each of the next `horizon_corners` corners ahead:
      corner, dist = distance_to_next_corner(track, pos)
      # cooldown is credited only for the gears we will PLAUSIBLY be in before
      # this corner: approximate as one low-gear turn's cooldown IF dist is large
      # enough to take a slow turn, else 0. Do NOT credit full cooldown every corner.
      turns_available = max(0, dist // EXPECTED_TURN_DISTANCE)
      heat += min(turns_available, 1) * cooldown_amount(planned_gear) ... capped at pool
      # cost is the overage at the speed we must arrive carrying, using the
      # deck-quality expected arrival speed, NOT a flat 0.5:
      arrival_speed = expected_arrival_speed(player, planned_gear)
      heat -= corner_heat_cost(int(arrival_speed), corner)
      track worst (most-negative) heat
  return min(0, worst) * SOLVENCY_PENALTY_PER_HEAT
  ```

  The key change vs the WIP: cooldown is **not** credited once per corner
  unconditionally, and corner cost is the **real overage at projected arrival
  speed**, so the projection can actually go negative and the penalty fires when a
  plan genuinely cannot stay solvent. Tune `SOLVENCY_PENALTY_PER_HEAT` so the term
  is a real deterrent (≈ `SPINOUT_LOSS_SPACES / 2` per heat of shortfall) but does
  not dominate `relative_term`. If a faithful projection proves too noisy, the
  fallback is to **replace** it with a single-step check: "after this turn's spend,
  is `heat_after_turn` ≥ the expected overage of the *next* corner?" — a cheaper,
  always-firing guard. Pick the projection if it passes the no-spinout gate; else
  the single-step guard.

- **Re-specify spin-out aversion (P2).** Keep the structure (`p_spinout` priced by
  buffer thinness against the worst-case stress flip, `:388-414`) but recalibrate:
  - Keep `risk_speed = expected_speed + speed_variance` and the worst-case crossing.
  - Replace the hard step function with a smoother schedule on `risk_slack =
    heat_available - total_heat_risk`:
    `p_spinout = clamp01( (1 - risk_slack) / SPINOUT_SLACK_SCALE )` for
    `risk_slack < SPINOUT_SLACK_SCALE`, else 0; `p_spinout = 1.0` when even the
    *expected* case cannot pay (`total_heat_needed > heat_available`). Default
    `SPINOUT_SLACK_SCALE = 2.0` (two heat of buffer ⇒ ~0 risk).
  - Retune `SPINOUT_LOSS_SPACES` upward to **~10–12** so the *expected* spin
    penalty `p_spinout * spinout_loss_eff` reliably exceeds the marginal spaces a
    risky play buys. This, combined with the worst-case-solvent boost (kept), must
    push strong spins **strictly below** weak with margin (gate below).

- **Real block projection (P3).** Replace `block_bonus` (`:291-323`) with a
  function that, for the focal agent's landing `(end_lap, end_pos)`, checks each
  trailing rival within reach next turn and asks: *if I occupy this space, does the
  rival's natural next move land on a space whose lanes are now full (counting
  me), forcing `resolve_blocked_position` to bounce it backward?* Use
  `rules.resolve_blocked_position` (`rules.py:515-553`) on the rival's projected
  target with the field including the agent at its new position. Value the block
  as `BLOCK_WEIGHT * (rival spaces denied)` in the relative currency (it widens
  `gap_after(behind rival)`), so it composes with `relative_term` rather than
  being a flat constant. `BLOCK_WEIGHT` default **1.0** (denied spaces are real
  gap). Only enabled at `strength == 3`.

#### `src/heat/agents/strong_heuristic.py`

- **Thread the relative objective through the planner.** `_plan_turn`
  (`:153-205`) and `_best_play` (`:316-351`) call `evaluate_move` with the new
  signature; `evaluate_move` internally computes `prog_self_before/after` and calls
  `relative_term` when `opponent_aware` is set. No structural change to the
  plan-once cache (`:256-273`) — keep it.
- **`_effective_heat_price` → risk posture.** Replace the ±10% nudge
  (`:122-129`) with `ME.effective_risk(...)`; pass both `heat_price_eff` and
  `spinout_loss_eff` into `evaluate_move`. Keep the end-game decay (`:115-120`).
- **`choose_slipstream` lap fix + relative value.** At `:565-584`, compute the
  landed lap via `ME.landed_lap_and_pos` and value the slipstream with the same
  `relative_term` (slipstream that pulls level with / ahead of the car directly
  ahead is the canonical contested-gap win). Keep the corner-cost-at-card-speed
  handling (`:564-573`) — it is correct per `corner_speed_for_check`
  (`rules.py:496-508`) and is called out as solid.
- **Keep settled items unchanged:** the worst-case-solvent boost
  (`_should_boost`, `:456-500`), the myopic strength-0 boost/adrenaline
  (`_myopic_*`, `:502-552`), the deck-quality discard (`choose_discard`,
  `:586-625`), the turn-signature cache, `_select_best` tie-break (`:133-147`),
  the strength-0 decoupled `_myopic_gear` floor (`:207-254`).

#### Ladder feature flags (`strong_heuristic.py:78-86`)

Re-tier so each rung turns on a *decision-moving* component:

| `strength` | Adds (cumulative) | Why it separates |
|---|---|---|
| **0** | myopic, absolute, decoupled gear/cards, no solvency, no positional | floor: ≈ ties weak heuristic |
| **1** | joint gear+cards plan + **real forward solvency** + spin-out aversion | coherent plan that stays heat-solvent and spins less → beats 0 on fewer wasted turns |
| **2** *(default)* | **relative_term** (gap saturation) + **risk posture** (rank-adjusted price/spin) + two-corner horizon + end-game decay | the real strength bar: changes the argmax in contested situations and by standing |
| **3** | **real block projection** + three-corner horizon | denies rivals spaces; composes additively in the relative currency |

Flags map cleanly: `_opponent_aware = s>=2` now gates `relative_term` (not the
deleted overtake bonus); `_position_price = s>=2` gates the risk posture;
`_blocking = s>=3` gates the real block projection.

#### `src/heat/simulation/runner.py`

No behavioural change. `strong_heuristic_agent_factory` (`:164-190`) already
takes `strength` and `heat_price` and is a picklable top-level `functools.partial`
(`:184-190`). **Confirm only** that the new tunables are class/module constants
(not constructor args), so the factory signature is unchanged and stays picklable.

#### `src/heat/agents/__init__.py`

No change (already exports `StrongHeuristicAgent`).

### Data Model / Interface Changes

- `evaluate_move` signature gains nothing required from callers except passing
  `spinout_loss` (the rank-adjusted loss) alongside `heat_price`; everything else
  is internal. Keep it keyword-only and additive so the optional future
  `LookaheadAgent` (which reuses this evaluator) is unaffected.
- New pure helpers in `_move_eval.py`: `landed_lap_and_pos`, `relative_term`,
  `effective_risk`, rewritten `project_solvency`, rewritten block projection.
  Deleted: `_overtake_bonus`, old `block_bonus`, possibly `_EXPECTED_CORNER_OVERAGE`.
- `MoveEval` dataclass (`:331-339`) gains an optional `relative_value: float = 0.0`
  field for test introspection (additive, frozen-dataclass-safe with a default).
- **No engine/rules changes. No ML-contract changes** (`spaces.py` OBS_DIM=72 /
  ACTION_DIM=516 untouched). Factories stay picklable.

## Alternatives Considered

- **Just enlarge the existing bonus caps.** Rejected (P0): the bonus is collinear
  with own-progress, so scaling it up either does nothing (when it can't flip the
  argmax) or double-counts progress (when it can). The problem is the *shape* of
  the objective, not the magnitude of an additive term.
- **Full minimax / engine-clone rollouts at strength 3.** Rejected for scope and
  latency; reserved for the optional `LookaheadAgent` (owned by 6B), which reuses
  the same `evaluate_move`. The structured relative objective is the right altitude
  for a scripted bar.
- **Pure gap-difference `relative_term = gap_after - gap_before` (linear).**
  Rejected: linear gap-difference *is* collinear with own-progress
  (`gap_after - gap_before = own_progress` exactly when rivals don't move). The
  `tanh` saturation is what decouples it — it down-weights uncontested gaps and
  up-weights contested ones.
- **Keep strict adjacent monotonicity, retune constants until it passes.**
  Rejected (P1): the review found this gate is fragile by construction — adjacent
  rungs differ by one marginal feature and noise over any feasible game count can
  invert k+1 vs k. Replaced by the robust gate below.

## Testing Strategy

Reuse the existing `tests/test_strong_heuristic.py` structure. Keep the
legality, determinism, picklability, latency, and joint-plan-consistency gates
unchanged (the review says these pass and are sound). Replace/retune the three
strength/ladder/spin gates and add targeted relative-objective units.

### Revised validation gates (concrete pass/fail)

- **Legality (hard, unchanged).** Every `choose_*` return ∈ the legal set handed
  in, across full `Game` runs at every strength (`TestLegality`, 6 games × 4
  strengths). Assert directly; engine guards are backstop only.

- **Strength headline (retuned).** `strength=2` vs weak `HeuristicAgent`, 2v2
  win-share over **`H2H_GAMES = 200`** seeded games (up from 120 for a tighter CI),
  `seed = 12345`. **Pass: share ≥ 0.62.** (Same metric as `test_strength2_beats_heuristic`,
  more games.)

- **Strength-0 floor (unchanged).** `strength=0` vs weak heuristic: **0.40 ≤ share ≤
  0.60** (≈ tie within a band).

- **Robust ladder gate (replaces strict adjacent monotonicity).** Run a single
  round-robin among `{s0, s1, s2, s3}` (the same seam 6B will use), 2v2 per pair,
  `N = 200` games per pair, fixed seed. Pass **all** of:
  1. **Top beats floor decisively:** `s3` vs `s0` share ≥ **0.62**.
  2. **Monotone non-adjacent vs floor with increasing margin:** share(`sk` vs `s0`)
     is non-decreasing in `k` and `share(s2 vs s0) - share(s1 vs s0) ≥ 0.03` and
     `share(s3 vs s0) ≥ share(s2 vs s0)` (each higher rung beats the floor by a
     wider margin). This tolerates adjacent rungs being within noise of each other
     while still proving the ladder climbs.
  3. **Top beats every lower rung:** `s3` vs `s1` share > 0.50 and `s3` vs `s2`
     share > 0.50 (allowing a small noise band: require `> 0.50` with the
     observed share also `> 0.50 + noise` where `noise = 0.5/sqrt(N) ≈ 0.035`,
     i.e. effectively **≥ 0.535**).
  4. **ELO sanity (optional, computed inline from the round-robin):** fit simple
     pairwise ELO from the win-shares; assert ratings strictly increasing in
     strength with `s3 - s0 ≥ 100` ELO. (This is the same `dict[label, factory]`
     round-robin 6B will consume; computing ELO here de-risks the hand-off.)

- **No-spinout regression (retuned to pass with margin).** Over `seeds =
  range(2000, 2040)` (40 games, up from 30), `strength=2` total spin-outs **≤
  0.85 ×** weak-heuristic spin-outs (strictly fewer, with a 15% margin), replacing
  the bare `<=` that currently fails. Drives the P2 spin-out retune.

- **Relative-objective units (new, targeted — the P0 proof).** Construct states
  where own-progress is *tied* between two plans but opponent geometry differs,
  and assert the relative objective picks the contested one:
  1. **Defends a contested gap:** leading by 1 space over the chaser, two plans of
     equal own-progress, one preserves the 1-space lead and one concedes it →
     agent picks the lead-preserving plan (risk posture + relative_term).
  2. **Closes a contested gap when trailing:** 1 behind the leader, equal-progress
     plans, one pulls level → agent picks it.
  3. **Saturation:** 10 spaces ahead of all rivals → `relative_term ≈ 0`, agent
     reverts to pure own-progress (no irrational heat spend to extend a won lead).
  4. **Risk posture bites:** same state evaluated as leader vs as last place yields
     different boost/heat decisions (leader conserves, trailer pushes).

- **Lap-bug regression (new).** On a constructed finish-crossing move, assert
  `relative_term` / overtake value is non-zero where a rival is passed across the
  line (would be silently zero under the old stale-lap code). Directly pins P1.

- **Slipstream behaviours (kept).** Good slipstream taken; overshoot-into-
  unaffordable-corner declined (`test_takes_good_slipstream`,
  `test_declines_overshoot_slipstream`, `test_declines_boost_that_overshoots`).

- **Block behaviour (retargeted for P3).** `strength=3` on a single-lane track
  prefers the play that *actually* bounces a trailing rival (verified via
  `resolve_blocked_position`), not merely lands near it (replaces the
  proximity-only `test_strength3_takes_cheap_block`).

- **Full suite green:** `PYTHONPATH=src python -m pytest tests/ -q`.

### Edge cases to cover

- Finish-crossing moves (lap bug), single-lap tracks, no-corner tracks (block
  track), heat pool empty, hand cluttered, sole survivor (no rivals → `relative_term
  = 0`, risk posture neutral), rivals all finished.

## Implementation Plan

Each step is a single commit, gated before moving on.

1. **Lap-bug fix (P1) + helper.** Add `landed_lap_and_pos`; thread `end_lap`
   through `evaluate_move`, `choose_slipstream`, block. Add the lap-bug regression
   unit. *(Smallest, highest-confidence fix; unblocks correct opponent geometry.)*
2. **Relative objective core (P0).** Add tunables, `relative_term`, delete
   `_overtake_bonus`; wire into `evaluate_move` behind `opponent_aware`. Add the
   relative-objective units (defend/close/saturation). Re-run strength gate.
3. **Risk posture (P0 cont.).** Replace the ±10% nudge with `effective_risk`;
   thread `heat_price_eff` + `spinout_loss_eff`. Add the risk-posture unit.
4. **Real solvency + spin-out retune (P2).** Rewrite `project_solvency`;
   recalibrate `p_spinout`/`SPINOUT_LOSS_SPACES`. Gate: no-spinout regression with
   margin.
5. **Real block projection (P3).** Replace `block_bonus`; retarget the block unit.
6. **Ladder re-tier + robust gate.** Set the new flag→component mapping; implement
   the round-robin + ELO ladder gate. Gate: robust ladder gate passes.
7. **Latency + picklability re-check; full suite.** Confirm `strength=2` stays
   within the existing latency budget (~15× weak heuristic) and factories still
   pickle. Run the whole suite.
8. **(Hand-off to 6B.)** Expose rungs to 6B's `round_robin_elo` as
   `{"StrongHeuristic-s1": factory(strength=1), "StrongHeuristic-s2": ...,
   "StrongHeuristic-s3": ...}` (and optionally `-s0` as a weak anchor). Optional
   `RELATIVE_WEIGHT` / `heat_price` / posture-swing sweep once 6B lands.

## Keep vs Replace (explicit)

**Keep (review called solid):**
- One-currency evaluator structure (`evaluate_move` as the single value function).
- Plan-once cache keyed by turn signature (`strong_heuristic.py:256-273`,
  `choose_gear`/`choose_cards` consistency).
- Deliberately-crippled strength-0 floor (`_myopic_gear`, `_myopic_boost`,
  `_myopic_adrenaline`, `:207-254`, `:502-552`).
- Correct slipstream / corner-check handling (card-speed corner check excluding
  slipstream movement, `choose_slipstream:564-573`).
- Worst-case-solvent boost using max owned Basic + flip variance (`_should_boost`,
  `:456-500`) — **settled fix, document and keep.**
- Deck-quality discard (`choose_discard:586-625`), `_select_best` tie-break,
  determinism/picklability/latency machinery, the `runner.py` factory.

**Replace:**
- Objective: absolute own-progress + tiny capped bonuses → relative gap-saturation
  + lead-dependent risk posture (P0).
- `_overtake_bonus` (`:466-489`) → `relative_term` (deleted; collinear + buggy).
- Stale `from_lap` on landing/positional terms → `landed_lap_and_pos` (P1).
- Inert `project_solvency` (`:178-230`) → real forward-heat projection (or
  single-step guard fallback) (P2).
- `p_spinout` step function + `SPINOUT_LOSS_SPACES` value → recalibrated smooth
  schedule + larger loss (P2).
- Flat-0.15 `block_bonus` (`:291-323`) → real block projection in relative
  currency (P3).
- ±10% rank price nudge (`:122-129`) → `effective_risk` posture (P0).
- Strict-adjacent-monotonicity ladder gate → robust margin/floor/ELO gate.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `relative_term` over-weights opponents → buys progress with heat irrationally / regresses raw progress. | `tanh` saturation caps per-contest value at `GAP_SCALE=2`; `alpha=1.0` keeps own-progress dominant; strength gate (vs weak) and no-spinout gate both guard against pathological aggression. |
| Risk posture makes the leader too passive (gets caught) or the trailer too reckless (spins). | Swings are bounded (±40% price, ±30% spin); no-spinout gate bounds recklessness; round-robin ladder gate bounds passivity (s2/s3 must still beat lower rungs and the floor). |
| Real solvency projection is noisy / over-penalizes. | Single-step-guard fallback specified; gate on no-spinout regression decides which ships. |
| Spin-out retune trades fewer spins for weaker racing. | The retune must pass *both* the no-spinout gate *and* the strength gate simultaneously; if they conflict, prefer the smooth schedule with `SPINOUT_SLACK_SCALE` tuned rather than inflating `SPINOUT_LOSS_SPACES` without bound. |
| Ladder still doesn't separate after the objective change. | Robust gate is designed to *measure* separation honestly (margin-over-floor, top-beats-all, ELO spread ≥100) rather than demand brittle adjacency; if a rung genuinely adds nothing, that is surfaced as a real failure to redesign that rung, not noise. |
| New constants are "just magic numbers." | All are in *spaces* or as interpretable swings/fractions, with stated defaults and a documented derivation; 6B's round-robin can sweep `RELATIVE_WEIGHT`, `heat_price`, and the swings empirically. |
| Latency grows from per-rival gap loops. | `K=2` nearest rivals only; horizon capped at 3 corners; latency gate (~15× weak) unchanged and re-checked at step 7. |

## Open Questions / Assumptions

- **`GAP_SOFTNESS` / `GAP_SCALE` / swing defaults** (3.0 / 2.0 / 0.4 / 0.3) are
  principled first guesses; final values come from the strength + ladder gates and
  optional 6B sweep. Surfaced as module constants for reproducibility.
- **`K` nearest rivals = 2** (car directly ahead + directly behind). Assumed
  sufficient for 4-player USA-track races; revisit if 6-player fields need more.
- **Solvency: projection vs single-step guard.** Assumed the faithful projection
  ships; the single-step guard is the documented fallback if the projection fails
  the no-spinout gate or proves too noisy. Decision made empirically at step 4.
- **`alpha` (own-progress weight) = 1.0.** Assumed; if the relative objective
  proves too aggressive, `alpha` can be raised rather than shrinking
  `RELATIVE_WEIGHT`, keeping the relative *shape* intact.
- **ELO sub-gate** is computed inline from the round-robin for de-risking the 6B
  hand-off; if 6B's `round_robin_elo` is not yet available it is computed with a
  minimal local pairwise fit (no dependency on 6B).
- **Game counts** raised to 200 (head-to-head) / 40 (spin) for tighter CIs;
  confirm the latency budget tolerates the larger batches in CI (they run
  sequentially in-test; parallel is available if needed).
