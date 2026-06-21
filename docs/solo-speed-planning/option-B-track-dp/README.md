# Option B — Offline whole-track Dynamic Programming

> **Status:** design-only, pre-sprint. The standalone realization of the "plan the
> lap, drive it cheaply" instinct. Read the options landscape
> ([`../../solo-speed-whole-track-planning-options.md`](../../solo-speed-whole-track-planning-options.md))
> and the plans index ([`../README.md`](../README.md)) first for the shared scope,
> the value-function spine, the reuse assets, and the eval methodology this plan
> inherits.

---

## 1. The idea, in one paragraph

The track is **known before the race** (`generate_track(seed, params)` returns the
full `Track` — corners, speed limits, lengths, laps — up front). So instead of
re-deriving "spend heat here / cool down there" every move with a 2-ply search that
cannot see four corners ahead, we **precompute the whole-lap plan once**: build a
reduced solo MDP over `(track position × heat × gear)`, solve it offline by value
iteration / backward DP to get a **value-over-position map** `V*(pos, heat, gear)`
(≈ expected rounds-to-finish, the shared spine) plus a derived **per-sector heat
budget** (target heat to carry into each corner, and where to bank cooldowns to
afford it). Online, a cheap controller reads the plan: at each turn it picks the
`(gear, cards)` whose realized outcome best matches the planned heat trajectory,
using one-ply lookahead over the **actual** hand drawn to absorb draw variance.

This is the eco-driving / EV-energy-management / fuel-strategy pattern (DP over a
known route with an energy budget) adapted to HEAT, where **heat is the energy**:
finite pool, spent on overspeed and big gear shifts, replenished only by slowing
down (cooldown in gears 1–2).

---

## 2. Why this option needs a feasibility spike *first*

This option **lives or dies on the state abstraction**. Exact DP over the true
state — full hand (7 cards) × draw-pile order × discard × heat × position × gear
— is astronomically large and stochastic in the draw, so we *must* compress. The
compression is lossy, and if it is *too* lossy the DP will optimize a fantasy and
the controller will drive into spins the plan said were safe. We therefore make
**Sprint B0 a modeling/feasibility spike** whose only job is to measure the gap
between the reduced model's predicted per-turn outcomes and the **true engine**
(`run_round_driver`), before we build a solver on top of it. If the gap is too
large to close with the mitigations in B0, we stop and fold B's findings into
Option A instead of shipping a solver nobody can trust. This directly applies the
8C/S3 lesson: never trust a clean internal metric that was never gated against the
real engine on held-out generated tracks.

---

## 3. The abstraction (resolved, grounded in the engine)

### 3.1 What the real engine actually does (verified in source)

From `heat.engine.rules` / `heat.engine.driver` / `heat.models.player_state`:

- **Speed** = sum of `gear` played card values (`calculate_speed`); a **stress**
  card flips to a random *owned Basic* (mean ≈ `_move_eval.expected_basic_value`,
  ~2.5 for the 1–4 deck); **boost**/**adrenaline** add speed at heat-relevant
  cost. You must play exactly `gear` cards (`cards_to_play_count`); a **cluttered**
  hand (fewer playable than `gear`) means the car does not move.
- **Heat is the energy.** `heat_available = len(heat_pool)`. It is **spent** on
  (a) a ±2 gear shift (1 heat, `legal_gear_shifts`) and (b) **corner overspeed**:
  `corner_heat_cost(speed, corner) = max(0, speed - speed_limit)` per corner
  crossed (`corners_crossed`). Paying heat moves those cards from the pool into the
  **discard** (`PlayerState.pay_heat`), so they re-enter the deck and clog later.
- **Cooldown is the only refill.** `cooldown_amount(gear)` returns 3 in gear 1,
  1 in gear 2, 0 in gear ≥3 — and it only recovers heat **cards that are sitting
  in hand** (`PlayerState.cooldown`). So refilling heat requires both a low gear
  *and* heat cards clogging the hand.
- **Spin-out** (`check_spin_out`): if a crossed corner's `heat_cost >
  heat_available`, the car **spins** — reset to `corner.start - 1`, take 1–2 stress
  (`spin_out_stress_count`), forced to gear 1 next round. This is the catastrophe
  the whole-lap budget exists to prevent.
- **Gear shifting** (`legal_gear_shifts`): ±1 or stay for free, ±2 for 1 heat,
  clamped to `[MIN_GEAR=1, MAX_GEAR=4]`.

### 3.2 The reduced MDP

**Reduced state** `s = (pos, heat, gear)`:

- `pos ∈ [0, track.length × laps)` — absolute progress (so corner identity and
  remaining distance are both implicit). Position is the DP's "stage" axis.
- `heat ∈ [0, HEAT_POOL_SIZE=6]` — integer heat available. Small, exact.
- `gear ∈ [1, 4]` — current gear (controls how many cards we commit and the
  cooldown rate; cheap to keep exact).

Everything else (the 7-card hand, the draw-pile order, the discard, stress/heat
clog) is **abstracted into a stochastic resource model**, *not* enumerated.

**Action** `a = (Δgear, target_speed_band)`: pick the next gear (legal shift) and
an *intended* speed band for the turn. The action does **not** name specific cards
— the controller resolves that online against the real hand (§5).

**Stochastic resource model — the heart of B.** Given `(gear, deck-composition)`
we model the turn's realized speed `S` as a **distribution** `P(S | gear, deck)`,
derived from the deck's Basic-card value multiset rather than the live hand:

- For a given gear, the achievable speed is (approximately) the sum of `gear`
  draws from the player's Basic-value distribution, with stress/boost folded in as
  their expected flip value (reusing `expected_basic_value` / `play_speed`). The
  spike (B0) chooses between three fidelities and measures which is good enough:
  1. **Expected-only** — collapse `S` to its mean `E[S | gear]` (cheapest;
     ignores draw variance; likely too optimistic at tight corners).
  2. **Few-moment / banded** — a small discrete distribution (e.g. low / mean /
     high speed bands per gear) so the DP can reason about an *above-mean* draw
     forcing overspeed — the variance that actually causes limit-1 spins.
  3. **Empirical** — estimate `P(S | gear)` by sampling real hands from a freshly
     dealt deck (Monte-Carlo over `legal_card_plays`) and the controller's own
     card-choice rule; the most faithful, used as the B0 ground-truth reference.
- **Hand controllability.** Because the player *chooses* which `gear` cards to
  play, the realized speed is not a blind draw — a good controller plays *low*
  cards into a tight corner when it holds them. The model captures this as a
  **conditional-on-intent** distribution: `P(S | gear, intent=conserve)` uses the
  lower order statistics of the draw, `P(S | gear, intent=push)` the upper. This is
  the single most important fidelity knob and B0 must validate it.

**Transition.** From `(pos, heat, gear)` under action `(Δgear, intent)`:
1. Pay the gear-shift heat (`1` if `|Δgear| = 2`).
2. Sample/marginalize `S ~ P(S | gear', intent)`.
3. `corners = corners_crossed(pos, pos+S)`; `cost = Σ corner_heat_cost(S, c)`.
4. If `cost > heat`: **spin** — go to `(corner.start−1, heat, gear=1)`, add a
   round penalty (and the modeled cost of the stress clog). Else: advance to
   `pos+S`, `heat ← heat − cost`, then apply **cooldown refill** for the resulting
   gear, modeled as expected heat returned given the modeled hand clog.

**Reward / cost-to-go.** Each transition costs **1 round** (so `V` ≈
expected-rounds-to-finish, the spine). A spin adds the rounds lost plus a shaped
penalty matching the corner-skill lessons (own-spin dominates). Terminal: `pos ≥
length × laps` ⇒ `V = 0`.

**Solve.** Position is monotone-increasing within a lap (you only move forward or
spin *backward to a fixed corner-relative spot*), so the MDP is **near-acyclic**
in `pos`; we solve by **backward DP / value iteration** over `(pos, heat, gear)`,
sweeping `pos` from finish to start, with a few value-iteration passes to settle
the spin back-edges. State-space size ≈ `length·laps × 7 × 4` (e.g. 90×2×7×4 ≈
5,000 states) — trivially solvable in milliseconds per track.

### 3.3 Where this can diverge from the engine (stated honestly)

- **Hand memory is dropped.** The real hand persists across turns (you keep cards
  you did not play); the resource model treats each turn's draw as fresh from the
  deck distribution. This **understates** the value of *holding* a known low card
  for a known upcoming corner — exactly the skill that survives limit-1. B0
  measures this; the conditional-on-intent model (§3.2) is the first mitigation,
  and the online 1-ply controller (§5) recovers much of it at execution time.
- **Clog dynamics are approximate.** Heat paid → discard → reshuffles into hand
  later, throttling future cooldown and cluttering. The DP models clog only as an
  expected drag on cooldown yield; the true timing is stochastic.
- **Stress flips and boost** are modeled at expected value; their variance is in
  the banded distribution but their exact draw is not.
- **Slipstream / blocking / opponents** are out of scope (solo), so their absence
  is correct here, not a divergence.

The spike's pass/fail bar (§B0) is the explicit gate on whether these divergences
are tolerable.

---

## 4. Design decisions — resolved

**Per-track vs track-agnostic — recommend PER-TRACK.** Solve the DP **once at race
start** for the actual generated track. The state space is tiny and the solve is
milliseconds (§3.2), and a per-track solve is *exact for that track's corner
layout* — no feature-generalization error. A track-agnostic value (DP features
shared across tracks) is **deferred**: it trades the (negligible) precompute cost
for generalization risk we do not need, and a per-track `V*` is itself the cleanest
**training target** for a future track-agnostic net (it can warm-start Option A/C).
We pay the precompute once per race; the `precompute ms/track` metric keeps us
honest that it stays cheap.

**Online controller — plan as a soft guide, not a hard script.** The value map +
heat budget give, for every `(pos, heat, gear)`, the DP-optimal `(Δgear, intent)`.
The controller (§5) follows it but resolves the *actual* card play with one-ply
lookahead over the real hand, so it never drives into a spin the plan assumed away.

**Own-draw uncertainty — handled at two levels.** (a) In the *model*: the
stochastic resource distribution `P(S | gear, intent)` is the in-expectation
treatment the spine asks for. (b) At *execution*: the 1-ply controller sees the
real drawn hand and picks the safe play, collapsing the residual variance the DP
averaged over. This mirrors the S1 finding that 2 draw-samples sufficed.

**How B feeds the spine (A / C).** B produces a concrete `V*(pos, heat, gear)` and
a per-sector heat budget with **zero training**. That object is:
- a **warm-start / sanity oracle** for Option A's learned leaf value (regress the
  net toward `V*`, or compare the learned leaf against the DP value to localize
  where the net is wrong);
- a **model-based prior / shaping signal** for Option C's value head;
- an **interpretable baseline plan** to debug both ("the DP says cool down before
  corner 4; why doesn't the net?").

---

## 5. The online controller (how the plan becomes a move)

At each `choose_gear` / `choose_cards` decision for the real `GameState`:

1. Look up the current reduced state `s = (pos, heat, gear)` (read straight off
   `PlayerState`).
2. Read the DP policy `π*(s) = (Δgear*, intent*)` and the **planned heat target**
   for the next corner (the budget).
3. **Gear:** choose the legal gear nearest `gear + Δgear*` (re-validated against
   `legal_gear_shifts`).
4. **Cards (1-ply over the real hand):** among `legal_card_plays(hand, gear)`,
   score each candidate by *simulating the single turn forward* (the same
   `clone + run_round_driver` trick S1 uses, but **horizon 1**) and selecting the
   play whose realized `(end-position, end-heat)` best matches the plan's target
   trajectory — heavily penalizing any candidate that **spins** or that overshoots
   the next corner's safe speed. This is where the controller beats the DP's
   averaging: if the hand is unlucky (all high cards into a tight corner), it picks
   the least-bad real play rather than the fantasy average.
5. **React/slipstream/discard:** delegate to `HeuristicAgent` (the validated
   rollout policy), or a thin budget-aware override that takes cooldowns where the
   plan says to bank heat. (Discard low cards the plan will not need; keep a low
   card if the plan flags a tight corner within reach — the partial recovery of the
   dropped hand-memory.)

The controller exposes a `SearchProfile`-style cost report (clones/move, ms/move)
so the "fast online" claim is measured. With horizon-1 lookahead over a deduped
play set, clones/move is a small constant — far below S1/S2.

---

## 6. Sprint list (ordered)

| Sprint | Title | One-line goal |
|---|---|---|
| **B0** | [Feasibility spike — reduced-model fidelity](sprint-B0-feasibility-spike.md) | Build the stochastic-resource model + a per-turn simulator and **measure the gap vs the true engine**; gate go/no-go. |
| **B1** | [Offline DP solver + value/budget map](sprint-B1-dp-solver.md) | Backward DP / value iteration over `(pos, heat, gear)` → `V*` and per-sector heat budget, per track. |
| **B2** | [Online plan-following controller](sprint-B2-controller.md) | A `BaseAgent` that reads `V*`/budget and drives with 1-ply lookahead over the real hand. |
| **B3** | [Eval, tuning & spine hand-off](sprint-B3-eval-and-spine.md) | Gate on the shared eval vs Heuristic/S1-S2; tune the abstraction; export `V*` for Option A/C. |

**Why four (a spike + three).** B0 is non-negotiable (the abstraction is the whole
risk). B1 (solver) and B2 (controller) are genuinely separate deliverables with
separate failure modes — a correct solver with a sloppy controller, or vice versa,
fail differently and must be gated independently. B3 is the shared-eval gate plus
the spine export, kept separate so the agent is measured the way 8C/S3 taught us
(held-out generated, worst-case by corner limit) and so the `V*` hand-off to A/C is
an explicit, testable artifact rather than an afterthought. Three "build" sprints
matches the options-doc estimate (2–3) with the spike making the modeling risk
explicit and front-loaded.

---

## 7. Measurable success ladder (shared eval)

All gates on **held-out generated tracks** (disjoint seed band; reuse
`eval_search.py`'s `_HELDOUT_BASE` + `_TIGHT_PARAMS`), **solo**, reported as
**worst-case (p90/max), bucketed by corner speed-limit** — never finish-rate or
mean alone. Baselines: `HeuristicAgent` (the bar), the S1/S2 `LookaheadAgent` (to
match or beat on heat budgeting), and — where relevant — a model-free control.

1. **Rung 0 (B0 gate — feasibility):** the reduced model predicts true-engine
   per-turn outcomes within tolerance — **median absolute error in end-of-turn
   heat ≤ 1** and **predicted-spin vs actual-spin agreement ≥ ~90%** across a
   sample of states drawn from real games, *especially at limit-1 corners*. Miss ⇒
   tighten the model (banded/conditional) or abort B and feed findings to A.
2. **Rung 1 (B2 — finishes & is safe):** the controller finishes **100% solo** and
   its **worst-case spins/limit-1-pass ≤ HeuristicAgent's** (the S1 bar).
3. **Rung 2 (B2/B3 — heat budgeting works):** a **heat-efficiency** win — fewer
   heat-units spent per space and/or fewer spins than HeuristicAgent for the same
   or better rounds-to-finish — demonstrating the *budget* (not just survival)
   improved.
4. **Rung 3 (B3 — competitive on speed):** **rounds-to-finish ≤ S1/S2
   `LookaheadAgent`** on the held-out set, at **a fraction of its ms/move**
   (the "plan once, drive cheaply" payoff).
5. **Rung 4 (B3 — feeds the spine):** `V*` exported in a codec-compatible,
   reusable form and shown to correlate with realized rounds-to-finish (a usable
   warm-start / leaf prior for Option A/C).

A sprint that produces an agent **must** print the shared eval block
(rounds-to-finish, spins/pass by limit mean/p90/max, heat-efficiency, ms/move).
