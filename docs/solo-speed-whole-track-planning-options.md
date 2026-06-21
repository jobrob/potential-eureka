# Solo Speed — Whole-Track Planning: Design Options

> **Status:** Options landscape, pre-sprint. This formalizes the candidate
> algorithm families for the "make our car go fast" feature so we can pick which
> to expand into detailed, sprint-broken plans. It is the deeper successor to the
> one-paragraph **S5** stub in `docs/sprint-search-imitation-design.md`.
>
> **Input:** the 2026-06-21 distillation runs (memory
> `project-s4-distillation-vs-strong`), the solo-driving investigation
> (`project-solo-driving-decision-trace`), and the S1/S2 search outcomes.

---

## 0. Scope & objective (decided)

**In scope — "go fast, solo."**
- One car, minimize **rounds-to-finish** (≈ lap time) on the held-out *generated*
  track distribution, while keeping tight (limit-1) corners clean.
- **Whole-lap heat budgeting** — the long-horizon resource decision ("spend heat
  here, cool down there") that a 2-ply search cannot see.
- **Our own card-draw uncertainty** — we must "guess" our future draws (chance
  over our own deck). Solo is *nearly* deterministic (only our draws), so this is
  a modest, well-bounded stochasticity (the S1 prototype found 2 draw-samples
  sufficient).

**Out of scope (for now).**
- Opponent modeling, hidden-info determinization of opponents, multiplayer
  win-rate, slipstream/blocking tactics. The value/search "spine" below is built
  so opponents can be layered back in later (S2 already showed the determinization
  belief composes cleanly), but we do **not** pay for it now.

**Why solo first.** It isolates the actual skill (drive the track optimally),
removes the single hardest piece (hidden information), and matches the corner gate
we already measure. "Go as fast as we can" is a clean, well-defined optimization
target.

---

## 1. What the runs already taught us (hard constraints on any design)

1. **Depth alone is inert-to-harmful.** Horizons 2/3/4 performed about the same,
   and naive deeper search *selected spinning lines* until we added the pre-spin
   progress floor. → The lever is **not** "search deeper," it is **a better
   estimate of how good a position is** (the leaf value).
2. **Imitation caps out (~0.6 corner-card accuracy).** Pure copying of the search
   agent cannot exceed it and stalls below it. → To get *better* than today's
   search we must **learn from outcomes (value), not just labels.**
3. **Heat is the blind spot.** The current search optimizes near-term
   progress−spins; it has no representation of "I will run out of heat in 4
   corners." → Whole-track planning's whole job is to **add the long-horizon heat
   budget.**
4. **The track is known in advance.** `generate_track(seed, params)` is available
   before the race. → We can **precompute** a per-position strategy/value, instead
   of rediscovering it every move.

**Reusable assets (every option builds on these):**
`GameState.clone()` + `run_round_driver` (perfect simulator), the frozen codec
(`features.encode_observation` / `action_codec` / `spaces.CODEC_VERSION`),
`LookaheadAgent`'s pluggable **`leaf_value`** hook, the **separate critic/value
head** already in the model (designed as "how's the race going"), the
`TrackGenParams` generator, the `eval_search.py` harness (rounds + spins-by-corner
-limit), and the corner-skill scoring lessons (own-spin penalty, pre-spin floor,
`HeuristicAgent` rollout).

---

## 2. The shared spine: a "rest-of-lap" value function

Every option below is, at bottom, a different way to **produce** and **use** one
object:

> **V(position, heat, speed/gear, hand-summary) ≈ expected rounds to finish**
> (a cost-to-go), with our own future draws taken in expectation.

This single function *is* the compressed whole-track plan: if V is good, "spend
heat here / cool there" falls out of acting greedily (or with shallow search) with
respect to it, because V already encodes the consequences for the rest of the lap.
The options differ along three axes:

- **State abstraction** — how we compress the huge (hand × deck × heat × position)
  state enough to be learnable/solvable.
- **Training signal** — Monte-Carlo returns, TD bootstrapping, exact DP backups,
  or self-play search targets.
- **Inference use** — act greedily on V, do a shallow search guided by V, or
  follow a precomputed plan and only re-plan occasionally.

Keep this spine in mind: several "options" are really *components* that compose.

---

## 3. The options

For each: the idea, how it maps to HEAT-solo, what we'd build here, pros, cons /
risks, and a rough sprint count.

### Option A — Learned leaf value in the existing search *(incremental)*
- **Idea.** Keep the S1/S2 `LookaheadAgent`, but replace the weak leaf evaluator
  (`progress` / reckless rollout) with a **learned V** estimating rest-of-lap
  cost-to-go. Shallow search + good value = the cheapest attack on constraint #1.
- **HEAT mapping.** Train V by regressing on simulated rounds-to-finish (or TD);
  plug it in via the existing `leaf_value` knob. Own-draw handled by the existing
  `n_determinizations` averaging.
- **Build.** A value-net trainer (can reuse the model's critic architecture + the
  codec), a `leaf_value="learned"` path in `LookaheadAgent`, eval vs the `progress`
  leaf.
- **Pros.** Lowest risk; reuses everything; directly tests "is the leaf the
  bottleneck?"; a clean win feeds every later option.
- **Cons/risks.** Still search at inference (not the fast net). A value trained on
  rollout data inherits rollout bias; can't exceed the policy that generated it
  unless iterated.
- **Rough size:** 1–2 sprints.

### Option B — Offline whole-track Dynamic Programming *(your "plan the lap" idea)*
- **Idea.** Exploit the known track: build a **simplified solo MDP over (track
  position × heat × speed)** and solve it by value iteration / DP **once per track
  (or once, track-agnostic via features)**. Output: a value-over-position map and
  an explicit per-sector heat budget. A cheap online controller then follows it
  (greedy + 1-ply for the draw).
- **HEAT mapping.** This is literally "in this section spend heat, here cool down."
  It's the eco-driving / EV-energy-management / fuel-strategy approach (DP over a
  route), adapted to HEAT's heat-as-energy.
- **Build.** A reduced state model of the engine's speed/heat/corner dynamics, a DP
  solver over position×heat, and a controller that maps the value map to per-turn
  card choices (handling the actual hand on the fly).
- **Pros.** Directly solves long-horizon heat budgeting; **interpretable** (you can
  read the plan); fast online; no training instability. Strong fit for "go fast."
- **Cons/risks.** The hand/deck makes *exact* DP intractable → needs a defensible
  abstraction (treat cards as a stochastic resource / assume average availability);
  the reduced model may diverge from the true engine; per-track precompute cost.
- **Rough size:** 2–3 sprints.

### Option C — Single-agent AlphaZero / Stochastic-MuZero *(highest ceiling)*
- **Idea.** Search **and** network co-improve: MCTS guided by a policy+value net,
  solo time-trial "self-play" generates better-than-net targets, the net trains
  toward them, the better net guides better search; repeat. Solo ⇒ **no opponent
  node** — just decision nodes + **chance nodes for our own draws** (the
  Stochastic-MuZero / expectimax-MCTS variant).
- **HEAT mapping.** The value head learns whole-lap consequences (incl. heat); the
  policy head prunes the 494-wide CARDS branch; PUCT search picks moves. This is
  the only option that can *surpass* today's search.
- **Build.** A (stochastic) MCTS over the engine, self-play target generation,
  policy+value training loop, reusing the codec + critic. Largest new machinery.
- **Pros.** Highest ceiling; the value net *is* the whole-track plan; can drive
  inference search shallow (fast + strong); proven recipe for perfect-simulator
  games.
- **Cons/risks.** Most complex and compute-hungry; MCTS-with-chance is subtle;
  needs the most careful eval to avoid the collapse pathologies we've seen.
- **Rough size:** 3–5 sprints.

### Option D — Hierarchical RL: manager + worker *(your intuition, learned)*
- **Idea.** A high-level **manager** picks a strategy/sub-goal per sector
  ("push" / "conserve heat" / "set up the next corner"); a low-level **worker**
  executes turns toward it with light lookahead. Temporal abstraction handles the
  long horizon; the worker handles control.
- **HEAT mapping.** Exactly "set a strategy over the lap, then play each turn
  cheaply relying on it" — the options framework / Feudal-Networks pattern.
- **Build.** Define the option/sub-goal space (sector-level heat intents), a manager
  policy over sectors, a worker policy/controller conditioned on the current intent.
- **Pros.** Structurally matches the problem; the manager re-plans rarely (your
  "don't recalculate constantly"); interpretable intents.
- **Cons/risks.** HRL is notoriously finicky to train; designing the option space is
  an art; less turn-key than AlphaZero; reward attribution between levels is tricky.
- **Rough size:** 3–4 sprints.

### Option E — MPC / plan-and-cache *(your "rely on the strategy, re-plan rarely")*
- **Idea.** Receding-horizon control: each *re-plan* computes a short trajectory
  (a few moves) to a learned **terminal value** (the spine V), commits to it, and
  **only re-plans every K turns or at sector boundaries** — not every turn. Between
  re-plans, execute the cached plan cheaply.
- **HEAT mapping.** The control-theory version of "plan, then follow the plan":
  short online search + a value that summarizes the rest of the lap, with explicit
  plan caching so we don't recompute constantly.
- **Build.** A short-horizon planner (can reuse the search infra), a terminal value
  (Option A's V), and a re-plan trigger + plan cache.
- **Pros.** Principled and fast; directly encodes "re-plan rarely"; small delta over
  A once V exists; robust to model error via periodic re-planning.
- **Cons/risks.** Discrete combinatorial card actions make the "trajectory" rougher
  than continuous MPC; overlaps heavily with A and C (it's mostly an *inference*
  strategy on top of a value).
- **Rough size:** 2–3 sprints (mostly after A).

### Option F — Model-free RL with whole-track observation *(baseline / control)*
- **Idea.** No search at all: a strong policy net trained with PPO on a time/heat
  reward, given a richer **whole-track observation** (what's coming up the road) and
  the corner-skill lessons baked in.
- **HEAT mapping.** The "fast game with no search" endpoint, retried with better
  eyesight than the earlier sprints had.
- **Pros.** Simplest inference; reuses the existing RL stack.
- **Cons/risks.** The solo-driving investigation **refuted** six model-free levers
  on exactly the limit-1 hard-exploration problem; without search/value-bootstrapping
  it is likely to stall again. Best treated as a **control/baseline**, not the main
  bet.
- **Rough size:** 1–2 sprints.

---

## 4. Comparison matrix

| Option | Ceiling | Inference speed | Build cost | Risk | Exploits known track | Can surpass current search? |
|---|---|---|---|---|---|---|
| A · learned leaf value | Med | Slow (search) | 1–2 | Low | No | Marginally (better leaf) |
| B · offline track DP | Med–High | Fast | 2–3 | Med | **Yes (core)** | Yes, on heat budgeting |
| C · AlphaZero / S-MuZero | **High** | Tunable (shallow+net) | 3–5 | High | Indirect | **Yes** |
| D · hierarchical RL | Med–High | Fast | 3–4 | Med–High | Indirect | Yes, on long horizon |
| E · MPC / plan-and-cache | Med–High | Fast | 2–3 | Med | Optional | Depends on V |
| F · model-free + obs | Low–Med | **Fastest** | 1–2 | Med (refuted before) | No | Unlikely |

---

## 5. How they relate (not mutually exclusive)

The **value function V is the shared spine.** That implies a natural ordering:

```
A (learned V plugged into search)  ── the cheap, de-risking first win
   ├─ E (MPC/plan-cache)            ── inference layer on top of V
   ├─ B (offline track DP)          ── a model-based way to PRODUCE V / a prior,
   │                                   and the most literal "plan the lap"
   └─ C (AlphaZero/MuZero)          ── trains V + policy to SURPASS the teacher
        └─ D (hierarchy)            ── adds temporal abstraction if flat V plateaus
F  ── runs alongside as the no-search baseline/control
```

Suggested spine-first read: **A** proves the value lever cheaply; **B** is the
standalone realization of your lap-planning intuition and can also seed a prior;
**C** is the high-ceiling bet that can beat the current search; **E/D** are
inference- and structure-level refinements layered on the spine.

---

## 6. Shared evaluation methodology

Whatever we build, gate it the way we already trust (don't relearn the 8C/S3
lessons):
- **Primary:** rounds-to-finish (≈ lap time) **and** worst-case (p90/max)
  spins/pass **bucketed by corner speed-limit**, on **held-out generated tracks**
  (disjoint seed band). Never finish-rate alone.
- **Add a heat-efficiency metric:** heat spent vs. distance / time, and
  cool-downs taken — to confirm the heat *budgeting* (not just corner survival)
  improved.
- **Cost report:** ms/move and clones/move (the `SearchProfile` we already print),
  so "fast" claims are measured.
- Compare against: `HeuristicAgent` (the honest bar), the S1/S2 `LookaheadAgent`
  (the current best, to be surpassed), and Option F (the model-free control).

---

## 7. Decision framework — which to expand into sprint plans

Pick along three questions:
1. **Do we want a fast (no/low-search) bot, or is inference search acceptable?**
   Fast → B / D / F or C-with-shallow-search. Search OK → A / E first.
2. **Appetite for the high-ceiling bet vs. a sure incremental win?**
   Sure win → A (then E). High ceiling → C.
3. **How much do we want to lean on "the track is known"?**
   A lot → B is the standout and most directly matches your intuition.

**My recommendation to write up first (in order):**
- **A** — 1–2 sprints, de-risks the whole spine, almost certainly a win, and every
  other option benefits from it. Write this up regardless.
- **B** — the cleanest expression of your "plan the lap, execute cheaply" instinct,
  standalone-useful, and exploits the known track. Strong candidate for the first
  *substantial* multi-sprint plan.
- **C** — the one that can actually *beat* today's search and approach "ideal."
  Highest effort; write it up if we want the ceiling, and let A/B de-risk its value
  function first.

D, E, F are best kept as **refinement/baseline** plans we expand only if A–C leave
a specific gap (D for long-horizon plateaus, E for inference speed, F as the
control).
