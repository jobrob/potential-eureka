# Search + Imitation Sprints — beating the limit-1 corner via the simulator we own

> **Status:** **Sprints S1 (a.k.a. "9a") and S2 are BUILT + VALIDATED + green**
> (see the "S1 — Outcome" and "S2 — Outcome" subsections in §4); everything
> downstream of S2 is design-only.
> Correction: the Tier-1.1 prototype `experiments/proto_search.py` referenced
> below **was never committed and did not exist** — S1's `LookaheadAgent` was
> built from scratch by generalizing `StrongHeuristicAgent`'s 1-ply joint
> planning. The prototype *numbers* quoted in the TL;DR are kept only as the
> original motivation; the **validated** S1 numbers live in §4.
> **Input:** the solo-driving investigation (memory: `project-solo-driving-decision-trace`)
> and the Tier-1 idea triage from the 2026-06-19 review.
> **Goal (unchanged):** an agent that takes tight (speed-limit-1) corners cleanly
> and **generalizes across the procedurally generated track distribution**, not
> one memorized track.

---

## TL;DR

The solo-driving investigation refuted six model-free levers (budget, time
penalty, tight-corner oversampling, reset-state curriculum, recurrent PPO,
obs representation) and concluded the limit-1 death-spiral is a
**hard-exploration / credit-assignment** problem: from a post-spin 0-heat
state almost every action re-spins, so the precise recovery is *essentially
never sampled* by PPO.

Two families of technique are matched to that diagnosis and are **not built
yet** anywhere in the codebase:

- **Tier 1.1 — decision-time search.** We own a perfect forward simulator
  (`run_round_driver` + `GameState.clone()`). The agent can *try moves before
  committing* and simply **see** the spin coming. The strong heuristic already
  does 1-ply lookahead via `_move_eval`; we generalize that to N-ply.
- **Tier 1.2 — imitation.** A working search agent is the best *demonstrator*:
  distil its clean-recovery play into the existing `MaskablePPO` policy net so
  the recovery behaviour is in-distribution, then RL-fine-tune. This composes
  with 1.1 (1.1 generates the data 1.2 trains on) and is the path to a *fast*
  net that needs no search at inference.

**Prototype evidence (zero training, `experiments/proto_search.py`, synthetic
limit-1 loop, 20 races each):**

| Agent | Finish | Rounds (finishers) | **Spins / limit-1 pass** |
|---|---|---|---|
| `NaiveFast` (reckless baseline) | 35% | 50.0 | **0.98** |
| `HeuristicAgent` (hand-coded ref) | 100% | 18.7 | 0.28 |
| Search, naive rollout, h=2 | 100% | 15.4 | **0.15** |
| **Search, heuristic rollout, h=2** | **100%** | **12.2** | **0.00** |

The *same* reckless policy that spins on 98% of passes, wrapped in 2-turn
engine lookahead, finishes 100% of races and **beats the hand-coded heuristic
on both spins and speed** — with no learning. Two levers emerged: **search
depth** (h≥2 suffices to anticipate the next-turn corner) and **rollout/leaf
policy quality** (naive→heuristic took spins 0.15→0.00). That is the "worth
developing" signal this plan builds on.

**Sprint sequence (cheap, proven wins first):**

1. **S1 — Productionize the rollout `SearchAgent`** (the proven prototype → a
   real agent), validated on *generated* tracks. Highest ROI, lowest risk.
2. **S2 — Harden & scale search**: branching control, chance nodes,
   multiplayer determinization, depth/budget tuning, learned leaf value.
3. **S3 — Distil search into the net (Behavioral Cloning).** A fast net with
   the recovery behaviour baked in, no search at inference.
4. **S4 — BC → RL fine-tune + DAgger** (+ GAIL/POfD as a fallback). Recover
   off-distribution states the expert never visits.
5. **S5 — AlphaZero/MuZero loop (stretch).** Policy/value-guided MCTS where
   search and learning co-improve. Highest ceiling, highest effort.

---

## 1. Background: why search/imitation and not "more PPO"

From `project-solo-driving-decision-trace` (definitive findings):

- **Root cause is the limit-1 death-spiral**, training-invariant, concentrated
  on the tightest corners (limit-1 ≈ 11 spins/pass for a reckless policy;
  heuristic ~1; strong heuristic ~0). It is a *policy* failure, not an engine
  bug (`diag_validate_spinout`): a scripted recovery agent crawls through with
  0 spins, so the car is **not** trapped.
- **Refuted model-free fixes:** budget, per-step time penalty, tight-corner
  oversampling, reset-state/backward curriculum, recurrent PPO, and obs
  representation (absolute vs relative limit encoding helped the *recoverable*
  range but did nothing for limit-1).
- **Reframed as hard exploration:** from a 0-heat pre-corner state the recovery
  (gear-1 + a single 1-card + cool) is never sampled, so flooding training with
  tight corners just fills the buffer with unrecoverable spin-loops.

A perfect simulator + an existing competent expert is *exactly* the setting
where search and imitation dominate model-free RL. Neither exists in the repo:
no MCTS/search agent (the `_move_eval.py` docstring even refers to "a future
`LookaheadAgent`" never written), no behavioral cloning, no off-policy/replay.

---

## 2. The technique spectrum (and where each variation lands)

Decision-time search runs from cheap-and-static to full-AlphaZero. Every
variation I floated in the review is listed here and assigned a sprint.

| Variation | What it is | Sprint |
|---|---|---|
| **L1 — 1-ply greedy** | score each legal action with a static value (`_move_eval`), pick best | exists (`StrongHeuristicAgent`) — the baseline to beat |
| **L3 — Monte-Carlo rollouts** | from each candidate, play forward with a default policy, average outcome | **S1** (the prototype) |
| **L2 — depth-limited expectimax** | branch on own choices, take expectation over chance, value the leaf | **S2** |
| **Leaf value = `_move_eval`** | use the heuristic's spaces-currency value at the search leaf | **S1/S2** |
| **Leaf value = learned critic** | use the PPO value head at the leaf instead | **S2** (option), **S5** |
| **Branching control** | dedup by resulting speed; top-k by heuristic / policy prior to tame the 494-wide CARDS space | **S2** |
| **Chance nodes / determinization** | sample/fix hidden cards & draws, average over N determinizations | **S1** (own draws), **S2** (multiplayer hidden hands) |
| **Search over more decisions** | extend search past gear+cards to REACT/boost & slipstream | **S2** |
| **Depth / sim budget tuning** | horizon, # rollouts, time budget per move | **S1** baseline, **S2** tuned |
| **L4 — policy/value-guided MCTS** | net proposes actions (prunes CARDS) + values leaves; UCT search | **S5** |
| **Search-as-trainer (AZ targets)** | search output becomes the net's training target; iterate | **S5** |
| **BC from heuristic / from search** | supervised (obs→action) on expert trajectories | **S3** |
| **DAgger** | query the expert on states the *learner* visits (esp. post-spin) | **S4** |
| **BC → PPO fine-tune** | warm-start RL from the cloned net (AlphaStar/OpenAI-Five recipe) | **S4** |
| **GAIL / POfD** | adversarial / demonstration-shaped imitation reward | **S4** (fallback) |

---

## 3. Cross-cutting concerns (apply to every sprint)

These are the lessons the investigation paid for; bake them into every eval.

- **Measure on GENERATED tracks, not USA.** Every "good" number in the 8C saga
  was USA-only and collapsed to ~0% on generated tracks. The prototype used a
  synthetic limit-1 loop; S1 onward must gate on `TrackSampler` / held-out
  generated tracks with limit-1 corners present.
- **Report p90 / worst-case spins-per-pass, not the mean.** Mean speed *hid*
  the death-spiral (6.7 vs 7.1 spaces/round looked fine while limit-1 spun 11×).
  Bucket spins by the speed-limit of the corner spun at.
- **Determinization & hidden information.** Solo is nearly deterministic (only
  own draws); the prototype used 2 seeds and it was enough. Multiplayer hides
  opponents' hands and deck order — S2 must sample determinizations and average,
  or accept solo/“open-hand” search first.
- **Clone cost is the budget.** Search clones many states per move. `clone()`
  already skips the event log by default; profile it (S2) before deep search.
- **Don't regress the contract.** BC/RL reuse the frozen obs/action codec
  (`features.encode_observation`, `action_codec`, `spaces.CODEC_VERSION`) so a
  cloned/fine-tuned net drops into `MLAgent`, `evaluate_ml`, and the league
  unchanged.

**Success bar (shared):** on held-out generated tracks with limit-1 corners,
**worst-case spins/limit-1-pass ≤ heuristic** and **finish rate = 100% solo**;
in 4p, **win-rate vs the weak heuristic ≥ parity and trending up**.

---

## 4. Sprints

### Sprint S1 — Productionize the rollout `SearchAgent`  (HIGH ROI, low risk)

**Goal.** Turn `experiments/proto_search.py` into a real agent and confirm the
prototype result holds on the actual objective (generated tracks, multiplayer
context), not just a synthetic loop.

**Scope (variations: L3 rollouts, L2-lite, leaf=`_move_eval`/heuristic rollout,
own-draw determinization, depth baseline).**

- New `src/heat/agents/search_agent.py` (`LookaheadAgent`) implementing
  `BaseAgent`: at the gear decision, fork the live state, simulate each
  candidate `(gear, cards)` plan to a horizon, score `progress − λ·spins`,
  pick the best; cache the play for the following CARDS decision; delegate
  REACT/SLIPSTREAM/DISCARD to a configurable default policy. (This is the
  prototype, cleaned up.)
- Config: `rollout_policy` (`HeuristicAgent` default — it gave 0 spins),
  `horizon` (default 2), `n_determinizations` (default 2), `spin_penalty`,
  and `leaf_value` (`progress`/`_move_eval`).
- Reuse `ml.opponents.opponent_action` for the rollout dispatch.
- Eval harness `experiments/eval_search.py`: spins-per-pass **bucketed by
  corner limit** + p90, finish rate, rounds-to-finish, on held-out generated
  tracks (`TrackSampler` seeds). Wire `LookaheadAgent` into `evaluate_ml` /
  the league as a first-class agent.

**Deliverables.** The agent, its config, the generated-track eval, unit tests
(search never returns an illegal move; horizon-0 == greedy; determinism under
fixed seed).

**Success criteria.** On generated tracks with limit-1 corners: worst-case
spins/limit-1-pass **≤ `HeuristicAgent`**, solo finish **100%**, and rounds
≤ heuristic. (Prototype already shows ≤ on a synthetic loop; this is the
generalization check.)

**Risks.** (1) Multiplayer determinization not yet handled → start by validating
**solo** (the corner-learning objective) and 4p with open-hand search, defer
proper hidden-info sampling to S2. (2) Branching blow-up on rich hands → S1 keeps
the dedup-by-speed prune from the prototype.

#### Sprint S1 — Outcome (BUILT + VALIDATED, 2026-06-20)

Delivered: `src/heat/agents/search_agent.py` (`LookaheadAgent`), registered in
`heat.agents`; picklable `lookahead_agent_factory` in **both**
`simulation/runner.py` and `ml/evaluate.py`; `experiments/eval_search.py`
(generated tracks, spins bucketed by corner limit, p90/max); 15 unit tests in
`tests/test_search_agent.py`. All on branch `worktree-sprint-9a-search-agent`.

**Validated result (correct spin accounting, held-out generated tracks):**
limit-1 spins/pass pooled **~0.06–0.22 vs HeuristicAgent ~0.49 (~3× fewer)**;
worst-case (max) spins/limit-1-pass **≤ heuristic in every config tested**;
**100% solo finish**; **rounds ≤ heuristic**. Robust across 24/32 tracks,
seeds {0,5}, and horizons {2,3,4}. All three S1 success criteria PASS.

**Three things the next sprint MUST inherit (hard-won — see memory
`project-sprint-9a-search-agent`):**

1. **The naive `progress − λ·spins` objective in this doc is INERT.** Because S1
   search only *forces* the learner's first-round gear+cards and delegates
   everything else to the rollout policy, the limit-1 spins that remain occur in
   rollout-policy-driven later steps that are ~identical across candidates, so
   `λ·spins` is a constant offset that cancels from the argmax (sweeping λ from
   11→60 changed *nothing*). **The working objective splits spins by round:**
   `progress − own_spin_penalty·own_spins − spin_penalty·later_spins`, where
   `own_spins` are first-round spins (the forced move's own spin — the only spin
   the search controls; `own_spin_penalty = 1000`, sized to dominate any banked
   progress) and `later_spins` is a mild tie-break. S2/S5 reusing the leaf value
   must keep this split (or an equivalent that makes the controllable spin
   dominate), not the flat penalty.
2. **End-of-horizon `progress` rewards reckless lines at depth.** A spin resets
   the car to `corner.start−1`; the rollout policy then banks recovery progress
   over the remaining `horizon−1` rounds, so deeper horizons *select* spinning
   lines (h=3 blew up to ~370 spins across the held-out set). Fix shipped: a
   **depth-invariant pre-spin progress floor** — when the forced move spins,
   credit progress only up to `corner.start−1` (lap 0), never post-spin recovery
   (`_pre_spin_progress`). Any deeper search (S2 expectimax, S5 MCTS value
   targets) inherits this hazard; value the leaf at/just-before the failed corner,
   not after recovery.
3. **`StrongHeuristicAgent` is the WRONG rollout/leaf policy on this
   distribution.** At strength 2 it spins *more* than the plain `HeuristicAgent`
   on tight generated tracks (its 1-ply solvency is USA-fit). S1's default
   rollout policy is `HeuristicAgent`; keep it (or build a tighter one) — do not
   assume "stronger scripted = better leaf."

**Also note for S2:** spin accounting reads `spin_out` events from the *clone's*
event log (logging force-enabled on the throwaway clone) and is counted **once**
after the rollout — counting the cumulative log per round multiply-counts early
spins (the bug that originally inflated S1's headline). Determinism for the unit
test comes from a per-(candidate, determinization) explicit `reseed=` derived
from a hand-folded turn seed (NOT Python `hash()`, which is `PYTHONHASHSEED`
-salted); `GameState.clone(reseed=None)` advances the parent RNG, so never clone
candidates in a bare loop from the live state.

---

### Sprint S2 — Harden & scale the search  (MED, enables real multiplayer)

**Goal.** Make search fast and correct enough for full 4p play and deeper
horizons.

**Scope (variations: L2 expectimax, branching control, multiplayer chance
nodes, search over REACT/boost, depth/budget tuning, learned leaf value).**

- **Proper expectimax / chance nodes:** model the card draw at end-of-turn as a
  chance node; average over determinizations rather than a single fixed draw.
- **Multiplayer hidden information:** sample opponents' hidden hands/decks from
  the public belief (uniform over unseen cards) and average — "determinized
  search". Validate spins/pass and win-rate vs heuristic in 4p.
- **Branching control:** top-k candidate plays by a fast prior (heuristic score
  or, later, the policy net) so the 494-wide CARDS space stays tractable at
  depth; profile `clone()` and cap a per-move time/sim budget.
- **Extend search past gear+cards** to REACT (boost/adrenaline) and slipstream,
  which also move speed across a corner.
- **Leaf value option:** plug `_move_eval.evaluate_move` (spaces currency,
  relative objective) as the leaf evaluator for shallow high-quality search.

**Deliverables.** Configurable depth/budget; multiplayer determinized search;
profiling numbers (clones/move, ms/move); league entry for the tuned agent.

**Success criteria.** 4p win-rate vs weak heuristic **> parity** and vs strong
heuristic **competitive**, at an inference cost the eval harness tolerates.

**Risks.** Determinized search can be over-optimistic in adversarial hidden-info
play; keep solo/limit-1 discipline as the primary metric, treat multiplayer
win-rate as secondary until S4/S5.

#### Sprint S2 — Outcome (BUILT + VALIDATED, 2026-06-20)

Delivered on branch `worktree-sprint-9b-search-s2` (stacked on S1), all as
**opt-in config on `LookaheadAgent`** so the S1 default path — and all 15 S1
tests — are byte-for-byte unchanged:

- **Multiplayer hidden-info determinization** (`determinize_hidden=True`):
  before each rollout clone, each opponent's hidden `hand + draw_pile` is pooled,
  id-sorted (canonical), shuffled with a det-specific RNG derived from the same
  deterministic reseed, and re-dealt to a same-size hand + shuffled draw pile;
  the discard pile (public) is untouched. The belief is **multiset-preserving**
  (never invents/loses an opponent card), leaves the learner's own hand alone,
  and is a no-op in solo. Averaging over `n_determinizations` such re-deals is
  the chance-node expectation over the opponents' hidden state.
- **Expectimax / own-draw chance node:** the existing `n_determinizations`
  averaging is the expectation over the learner's own future draws (each clone
  re-seeds the draw order); S2 layers the opponent-hand chance node on top of it.
- **Branching control:** `top_k` keeps only the best-`k` candidates ranked by a
  fast `_move_eval` prior; candidates are scored **prior-descending**, so a
  `sim_budget` (per-move clone cap) only ever drops the least-promising lines.
  Both default to off (= S1 full search), and a unit test proves
  top-k-keeps-all == full-search argmax and unbudgeted == raw S1 argmax.
- **Profiling:** every `LookaheadAgent` owns a `SearchProfile` (clones/move,
  ms/move), printed by the eval harness — always on, no flag.
- **Leaf value:** `leaf_value="move_eval"` already plugged in S1; carried.
- **REACT/slipstream:** already executed inside the rollout via the rollout
  policy on the clone (they are part of every rolled-out line), so deeper search
  already values them; S2 did not add a *separate* own-REACT branch (the rollout
  policy's REACT is competent and branching it would multiply cost for little
  measured gain — flagged for S5 if a gap appears).
- **League entry:** both `lookahead_agent_factory`s (runner + evaluate) gained
  `determinize_hidden` / `top_k` / `sim_budget` as plain primitives, so the tuned
  determinized agent pickles into `evaluate_league` / `run_batch(parallel=True)`
  unchanged. The eval harness registers it as the `LookaheadDet` contender.

15 new tests in `tests/test_search_agent_s2.py` (determinization multiset
invariant / own-hand+discard untouched / hand-size kept / actually-resamples /
determinism-under-determinization / 4p legal completion / solo no-op; top-k keeps
full-search best; budget caps clones/move and still scores ≥1 candidate;
profiling records). **All 15 S1 + 15 S2 + 28 strong-heuristic tests green.**

**Validated numbers (24 held-out generated tracks, tight/limit-1-weighted,
horizon=2, dets=2):**

| Metric (limit-1) | Heuristic | StrongHeur | Lookahead (S1) | **LookaheadDet (S2)** |
|---|---|---|---|---|
| solo worst-case spins/L1-pass | 1.50 | 1.86 | 0.50 | **0.50** |
| solo finish | 100% | 100% | 100% | **100%** |
| solo rounds | 26.2 | 42.1 | 19.3 | **19.3** |
| solo pooled L1 spins/passes | 64/130 | 298/190 | 7/93 | **7/93** |
| 4p p90 spins/L1-pass | 1.24 | 2.20 | 0.43 | **0.25** |
| 4p pooled L2 / L3 spins | 14/5 | 30/19 | 5/0 | **0/0** |

- **Primary gate (solo/limit-1): PASS** — worst-case 0.50 ≤ heuristic 1.50,
  100% finish, rounds 19.3 ≤ 26.2. Solo is identical to S1 (determinization is a
  no-op without opponents), exactly as intended.
- **Secondary gate (seat-neutral 4p win-rate): PASS both.** Focal agent rotated
  through all four seats (front-seat positional bias cancelled): **vs 3× weak
  heuristic 63.5%** (parity 25%), **vs 3× strong heuristic 76.0%**. Both well
  above parity — the determinized agent is the strongest contender in the field.
- **In 4p, determinization is a net positive on the corner metric too:**
  `LookaheadDet` drives L2/L3 to **0 spins** and a tighter L1 p90 (0.25 vs the
  open-hand `Lookahead`'s 0.43), i.e. sampling opponent hands made it *less*
  reckless, not more.
- **Profiling / cost:** full search ~25 clones/move, ~4.4 ms/move solo /
  ~30 ms/move in 4p (the 4p cost is the rollout simulating 3 opponents per round,
  not the search breadth). `top_k=6` cuts this to **~11 clones/move (2.4×
  cheaper)** while **still PASSING all three S1 criteria** (worst-case L1 0.667 ≤
  heuristic 1.50) — the branching control is free accuracy-wise on this
  distribution. Clone cost itself profiled at **~16 µs/clone (~64k/s)**.

**Things S3+ MUST inherit:**

1. **Determinization belief = pool `hand+draw_pile`, keep discard fixed.** The
   opponent's *total card multiset* is public (standard deck + observable
   stress/heat); only the hand/draw partition + order are hidden. Re-dealing the
   pooled hidden cards (NOT touching the discard) is the max-entropy belief and
   is multiset-preserving — a BC/DAgger expert that queries the search in 4p must
   use this same belief, or its targets will be inconsistent with the env's real
   info set. Canonicalize (id-sort) before the det shuffle, exactly like
   `Deck.attach_rng`, or determinism breaks.
2. **The S1 controllable-spin-dominates leaf split survived multiplayer
   unchanged** — determinization adds variance to *later_spins* (opponent-driven
   traffic) but `own_spins` is still the lever, so the `own_spin_penalty=1000` +
   pre-spin-progress-floor combination is still load-bearing. Do not flatten it.
3. **Branching control is prior-ordered, not prior-filtered-then-arbitrary.**
   `top_k` and `sim_budget` are only safe because candidates are scored in
   descending-prior order, so a cut drops the *worst* lines. A learned-prior
   variant (S5) must preserve that ordering guarantee.
4. **4p search cost scales with opponent count in the rollout, not breadth.**
   ~30 ms/move in 4p vs ~4 ms solo is the 3 opponents being simulated each round,
   so a faster rollout policy (or a learned leaf that shortens the horizon) is
   the lever for S5 throughput — not a smaller `top_k`.

---

### Sprint S3 — Distil search into the net (Behavioral Cloning)  (the fast policy)

**Goal.** A *fast* `MaskablePPO` policy that reproduces the search agent's
clean-corner driving with **no search at inference** — i.e., bake the recovery
behaviour into the net so it is in-distribution.

**Scope (variations: BC from search, BC from strong heuristic).**

- Generate a demonstration dataset by running the S1/S2 `LookaheadAgent` (and/or
  `StrongHeuristicAgent`) over many generated tracks, logging
  `(encode_observation(state, pid, decision), chosen_flat_action,
  action_mask)` at every learner decision — reusing the exact frozen codec so
  the data is policy-compatible.
- Supervised cross-entropy training of `HeatMLPExtractor` + policy head on the
  masked action targets (a standalone trainer, not PPO). Hold out tracks for a
  validation split.
- Load the BC weights into a `MaskablePPO` policy and evaluate as an ordinary
  `MLAgent` (contract sidecar + `CODEC_VERSION`).

**Deliverables.** `experiments/gen_demos.py`, `experiments/train_bc.py`, a BC
checkpoint, and an `evaluate_ml` report on held-out generated tracks.

**Success criteria.** The BC net (no search) achieves spins/limit-1-pass within
a small margin of the search agent and **finishes generated tracks** — the first
time a *learned* policy clears the limit-1 corner. Confirms the corner skill is
learnable once it is in the data.

**Risks.** BC inherits the expert's blind spots and **drifts off-distribution**
(compounding errors) — exactly what S4 (DAgger) fixes. Action-target snapping
must match the env's value-multiset rule (`env._decode_legal`).

#### Sprint S3 — Outcome (BUILT + VALIDATED, 2026-06-21)

Delivered on branch `worktree-sprint-9c-bc` (stacked on S1+S2), with the frozen
obs/action codec reused unchanged so the BC net drops into `MLAgent` /
`evaluate_ml` / the league with no contract change:

- **`experiments/gen_demos.py`** — runs the S1/S2 `LookaheadAgent` over many
  GENERATED tracks (the eval harness's tight/limit-1-weighted params), driving
  `run_round_driver` exactly as `HeatEnv` does, and logs
  `(encode_observation, chosen_flat_action, action_mask)` at every **real-choice**
  learner decision. Output is a compressed `.npz` (obs/action/mask/kind/track_seed/
  split) + a `.demos.json` provenance sidecar (codec version, expert config).
- **`experiments/train_bc.py`** — standalone supervised **masked** cross-entropy
  trainer (NOT PPO): builds a real `MaskablePPO` via `build_model` (so the weights
  live in the exact architecture `MLAgent` expects), then optimizes
  `-log_prob(target)` from `policy.get_distribution(obs, action_masks=mask)` — the
  same masked distribution `MLAgent` arg-maxes over at inference. Saves via
  `save_checkpoint` (zip + `.meta.json`). Only the actor trunk+head are trained;
  the critic is left at init (`MLAgent` never reads the value head).
- **`experiments/eval_bc.py`** — behavioral gate; reuses `eval_search.py`'s
  spins-by-corner-limit / finish / rounds reconstruction verbatim and adds the BC
  net (loaded as a contract-checked `MLAgent`) as a contender vs the Lookahead
  expert and the Heuristic, on the **held-out** 900_000+ track band (disjoint
  from the train 100_000+ / val 500_000+ bands `gen_demos` samples).
- 11 unit tests in `tests/test_bc.py` (demo tuples codec-valid + mask-consistent +
  only-real-choices-logged + track-disjoint split; CARDS target snaps like
  `env._decode_legal`; off-table REACT dropped not crashed; BC checkpoint loads as
  MaskablePPO + has the contract sidecar + round-trips through `MLAgent` returning
  only legal moves). **All 30 S1+S2 + 11 S3 tests green; full suite 770 passed.**

**Dataset (solo, 150 train / 40 val tracks):** 17,079 tuples (13,930 train /
3,149 val), kind balance GEAR 27% / CARDS 25% / DISCARD 23% / REACT 24% (solo has
no SLIPSTREAM). 4p set (150/40): 16,342 tuples incl. 6% SLIPSTREAM. **432–592
expert decisions per set were dropped as unencodable** — see learning #2.

**Supervised metrics (solo, 30 epochs, GPU):** train CE 0.43 / acc 0.84 /
CARDS-acc 0.55; **val CE 0.61 / acc 0.78 / CARDS-acc 0.48**. Val CE bottoms ~epoch
30 then overfits. So the net reproduces the expert's *label* on ~78% of unseen
states — but only ~48% on the high-stakes CARDS (corner-entry) decision.

**Behavioral gate (held-out generated, solo, 24 tracks) — the honest result:**

| Metric (limit-1) | Heuristic | Lookahead (expert) | **BC (no search)** |
|---|---|---|---|
| solo finish (engine) | 100% | 100% | 100%* |
| rounds-to-finish | 26.2 | 20.8 | **113 (11/24 hit MAX_ROUNDS=200)** |
| L1 spins/pass mean | 0.47 | 0.06 | **15.0** |
| L1 spins/pass p90 / max | 1.50 / 1.50 | 0.20 / 0.67 | **31.2 / 63.7** |
| pooled L1 spins/passes | 64/130 | 6/83 | **2366/170** |

\* "finish" is the engine's lap-counter sense; in racing terms the BC car
**crawl-and-spins** (113 rounds, ~11× the expert), so the finish is degenerate,
not skilled. The 4p net is the same (L1 mean 23.9, max 194, ~102 rounds).

**Gate verdict: NOT MET — and this is the expected, designed S3 outcome.** BC
alone does **not** learn the limit-1 corner. It is ~250× worse than the expert
and ~30× worse than the plain heuristic on worst-case L1 spins/pass. The §4 risk
note ("BC drifts off-distribution; compounding errors; S4's job, not yours")
materialized in full: the net clones the expert's labels well on the states the
**expert visits** (which almost never include a spin — pooled expert L1 spins are
6/83), so the post-spin 0-heat **recovery** states are essentially absent from the
data. The ~22% of CARDS the net gets wrong put it into exactly those unseen
states, where it has no learned recovery and re-spins, compounding into the
death-spiral. (Decision trace: the net correctly sits in gear 1–2 near corners
but plays a too-high card — e.g. an Upgrade-5 in gear 1 — into a limit-1 corner.)
Sampling vs arg-max made no material difference over the gate (det L1 15.0 vs
sampled 11.5), so it is not an arg-max-mode artifact — it is genuine drift.

**Things S4 MUST inherit:**

1. **BC ≠ corner skill; you MUST close the loop on the learner's OWN states
   (DAgger), not just the expert's.** The expert's near-zero spin rate means its
   trajectories contain almost no recovery demonstrations, so a pure
   obs→action clone has no data for the post-spin states it will actually visit.
   Roll out the *current BC learner*, collect the 0-heat post-spin states it
   lands in, query the `LookaheadAgent` for the correct action **on those
   states**, aggregate, retrain. This is the whole reason S3's gate fails and S4
   exists — do not expect more demos / a bigger net to fix it (we already saw
   val-acc plateau at ~78% with severe behavioral failure).
2. **The flat action space is a behavior-covering SUBSET of the engine's true
   action space — the scripted/search expert emits actions the codec can't
   represent.** ~3–4% of real-choice expert decisions (concentrated in REACT:
   the fixed 8-slot `_REACT_TABLE` does not enumerate every legal
   `(cooldown, boost, adrenaline_*)` combo, e.g. `cooldown_count=1` +
   adrenaline_speed) have **no `encode_action_index`**. `gen_demos` drops these
   (and counts them) rather than crashing or fabricating a wrong target. A
   `MaskablePPO` policy can only ever *emit* an in-table action, so this is the
   right call — but S4/DAgger querying the expert must apply the same drop, and
   any future codec expansion (S5) must re-check this REACT coverage.
3. **Action-target snapping via the value-multiset rule held up.**
   `encode_action_index` collapses a CARDS play to its value-multiset index — the
   same collapse `env._decode_legal` round-trips through — so a logged target is
   always set in its own mask (asserted in `gen_demos` and unit-tested against
   `env._decode_legal`). The BC failure is NOT a snapping bug; the contract is
   clean. Keep this invariant when generating DAgger labels.
4. **Train the actor only; the critic is uninitialized in the BC checkpoint.**
   S4's BC→PPO warm-start must not assume a calibrated value head — let PPO learn
   it from reward (a KL-to-BC regularizer on the *actor* is the right anti-
   forgetting lever, per the §4 S4 risk note, not a frozen critic).
5. **Measurement footgun: "finish rate" is near-useless solo.** The engine's
   lap counter eventually completes even for a crawl-and-spin policy, so 100%
   "finish" hid a 113-round degenerate race. Gate S4 on **worst-case L1
   spins/pass and rounds-to-finish**, never finish-rate alone (same lesson as the
   8C USA-mean saga, one layer deeper).

---

### Sprint S4 — BC → RL fine-tune + DAgger  (close the distribution gap)

**Goal.** Lift the BC net above the expert and fix the states the expert never
visits (post-spin recovery), then push win-rate via the existing opponent
curriculum.

**Scope (variations: BC→PPO fine-tune, DAgger, GAIL/POfD fallback).**

- **BC → PPO fine-tune:** warm-start `train_self_play` from the BC checkpoint
  (the `warm_start_path` hook already exists) and run the opponent ramp
  (solo → weak → mixed → strong). The AlphaStar/OpenAI-Five recipe.
- **DAgger:** roll out the *current learner*, collect the states it actually
  visits (especially post-spin 0-heat states), query the search agent for the
  correct action on those states, aggregate into the dataset, retrain. Directly
  targets the off-distribution recovery BC alone misses.
- **Fallback — GAIL / POfD:** if BC→PPO drifts, add a demonstration-shaped
  reward (discriminator or potential-based imitation term in `step_reward`)
  rather than pure cross-entropy.

**Deliverables.** A DAgger loop (`experiments/dagger.py`), a fine-tuned
checkpoint, league + Wilson-LB gate results vs the existing baselines.

**Success criteria.** Fine-tuned net **≥ search agent** on spins/pass and
**> weak-heuristic win-rate** on generated 4p, while running at full net speed
(no search). This is the headline "learned to drive corners" result.

**Risks.** Catastrophic forgetting of corner discipline during the strong phase
(the 8C bug-2 pattern) — keep the Wilson-LB best-checkpoint gate and consider a
behavioral-cloning regularizer (KL-to-BC) during fine-tune.

---

### Sprint S5 — AlphaZero / MuZero loop  (stretch, highest ceiling)

**Goal.** Co-improve search and the net: the net proposes which actions to
search and values leaves; search produces better-than-net targets that train
the net; repeat.

**Scope (variations: L4 policy/value-guided MCTS, search-as-trainer).**

- MCTS with the policy head as the prior (prunes the 494 CARDS branch) and the
  value head at leaves; UCT/PUCT selection; determinized for hidden info.
- Training loop: self-play with MCTS, store `(obs, search_policy, outcome)`,
  train net toward the search policy + value. Reuse the league for opponents.

**Deliverables.** Design spike first (is the per-move MCTS cost affordable at our
throughput?), then a prototype if S1–S4 plateau below target.

**Success criteria.** Beats the S4 fine-tuned net on generated 4p win-rate.

**Risks.** Large effort; only justified if S1–S4 leave a gap. Hidden-info MCTS
is subtle; keep it gated behind a measured need.

---

## 5. Dependency order & rationale

```
S1 (proven rollout search)
  └─ S2 (deeper/faster/multiplayer search)
        ├─ S3 (BC distil search → fast net)
        │     └─ S4 (BC→PPO + DAgger → fast net that beats the expert)
        └─ S5 (AZ loop: net+search co-improve)   [stretch, escalate-only]
```

Cheap proven win first (S1), harden it (S2), then *two* payoffs branch off:
distil to a fast net (S3→S4) for production inference, or escalate to the full
AZ loop (S5) only if distillation plateaus. S1 alone already yields an agent
that beats the hand-coded heuristic; everything after buys either speed
(no-search inference) or ceiling.

## 6. What this plan explicitly does NOT do

- It does not retry the six refuted model-free levers (budget, time penalty,
  tight-corner oversampling, reset-state curriculum, recurrent PPO, obs
  representation).
- It does not gate on USA. All success criteria are on generated tracks with
  limit-1 corners, worst-case (p90) spins/pass.
- It does not build S5 up front. Search-as-inference (S1) and distillation
  (S3/S4) must be measured first; the AZ loop is escalate-only.
