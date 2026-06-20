# Search + Imitation Sprints — beating the limit-1 corner via the simulator we own

> **Status:** design. The Tier-1.1 prototype (`experiments/proto_search.py`) is
> built and run; everything downstream of Sprint S1 is design-only.
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
