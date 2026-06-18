# ML Performance Improvements — Ideas & Decisions (pre-implementation)

> **Status:** ideas / decisions only — **no code yet**. Captures the plan for the
> next ML training effort, focused on **cross-track generalization and skill**.
> Supersedes "just train longer" — we identified concrete structural levers first.

## Why this exists (context)

Two training runs (2026-06-17) established the picture:

- The USA-only run (`heat_ppo_strong`) is a **track specialist**: ~0.96 win-rate vs
  heuristics **on USA**, but only ~0.145 on held-out generated tracks.
- The first generated-track run (`heat_ppo_gen`, 1.6M steps) produced a **weak
  generalist**: ~0.175 vs weak / ~0.092 vs strong on held-out generated tracks,
  ~0.06 on USA. (Opponents are heuristics, so a random policy ≈ 0, not 0.25.)

Root causes identified:
1. **The agent is nearly blind to the track.** `encode_observation`
   (`features.py`) is 72 dims, of which **only 4** describe the track, and only
   the **next corner** (`_track_lookahead`: dist-to-next-corner, its speed limit,
   current lanes, in-corner flag). A policy cannot plan multi-corner heat economy
   — or transfer it across layouts — when it can only see the next corner. On a
   fixed track it memorizes; across random tracks it's crippled. The docs already
   conceded cross-track generalization was "unverified and not a goal"
   (`sprint5-ml-roadmap.md:581`).
2. **Generated tracks are a much harder learning problem** than one fixed track
   (which hit 0.92 in Phase 1), so 1.6M steps was ~5–10× too few, and with no
   curriculum the agent faces the full distribution from step 0.
3. **Phase-2 self-play has never been additive** here — it collapsed in both runs;
   all the strength came from Phase 1. Half the budget was wasted on a phase that
   degrades the policy (the 6C gate prevents the *best* checkpoint being lost, but
   the compute is still spent).

The human-player insight that drove the obs decision: a skilled driver **looks at
the whole track** — banking heat through corner-dense sections to spend on long
straights, and managing the pool to the line. The agent needs that global view,
not a myopic next-corner peek.

---

## Idea 1 (DECIDED: option B) — Whole-track observation + spatial/sequence feature extractor

**Goal:** give the policy the *entire* track each step so it can plan global heat
economy (cool here, spend there, manage to the finish), and transfer that skill to
any track.

The obs must stay a **fixed-length vector** (SB3 `Box`, MLP/codec contract), so
"whole track" is encoded at fixed length and reshaped inside a custom extractor.

### Option A (considered, NOT chosen) — all-corners fixed slots + plain MLP
Encode every corner ego-centrically into `MAX_CORNERS` (~10–12) padded slots
`(distance-ahead/len, speed_limit/max, corner_len/len)` + globals (laps remaining,
dist-to-finish, heat pool, pos-in-lap). With ≤7 corners on real/generated tracks
this *is* the whole track, and inter-corner distances encode the straights.
Cheapest (stays an MLP), but caps at `MAX_CORNERS` and gives no fine spatial
detail. **Rejected** in favour of B for headroom and "true skill".

### Option B (CHOSEN) — full track-map + CNN/attention extractor
Represent the whole track richly and process it with a dedicated encoder:

- **Representation (two candidate encodings, decide at build time):**
  - **Spatial grid (recommended starting point):** a fixed-length 1D grid over the
    track (e.g. `GRID = 96` cells, ego-centric: cell 0 = current space, wrapping
    forward), each cell carrying `(is_corner, speed_limit_norm, corner_len_flag,
    lanes_norm, is_finish_line)`. Processed by a small **1D-CNN** → pooled
    embedding. Naturally "looks at everything", scale-free over track length,
    straights/clusters are spatially explicit.
  - **Corner sequence:** variable-length corner tokens (ego-centric, ordered),
    each `(distance-ahead, speed_limit, corner_len, lanes)`, processed by a small
    **self-attention / GRU** encoder with a padding mask. More compact; attention
    pools "which corners matter".
- **Architecture:** a custom `BaseFeaturesExtractor` that splits the fixed obs into
  (a) the **ego/scalar block** (hand histogram, gear, kinematics, deck, opponents,
  phase context — the existing 72-dim content minus the old 4-dim lookahead) → small
  MLP, and (b) the **track block** → CNN (grid) or attention (sequence) → pooled
  track embedding. Concatenate → `features_dim` latent → existing policy/value
  heads under `MaskablePPO`. Action masking unchanged.
- **Globals to include regardless of encoding:** laps remaining, distance-to-finish
  (end-game: heat is use-it-or-lose-it on the final lap), heat pool size,
  position-in-lap.

**Contract impact:** `OBS_DIM` grows substantially (e.g. 72 → ~170+ depending on
GRID/MAX_CORNERS), **`CODEC_VERSION` 1 → 2** (invalidates existing checkpoints —
acceptable; they're throwaway experiments). The `MLAgent` meta-tripwire already
catches version mismatches. `ACTION_DIM` unchanged.

**Files:** `features.py` (new whole-track block + reshape contract),
`spaces.py` (`OBS_DIM`, `CODEC_VERSION`), `model.py` (new custom feature
extractor; `net_profile` for the CNN/attention sizes), plus a generated-track
feature test (shape/bounds, ego-centric correctness, padding-mask correctness).

**Effort:** moderate–large (architecture change). **Risk:** moderate — the biggest
surface area of the bundle, but well-isolated behind the extractor + codec bump.

**Honest ceiling note:** B is the scalable end-state. If it proves heavy to tune,
A is the low-risk fallback that still captures global planning for bounded tracks.

---

## Idea 2 (DECIDED) — Track-difficulty curriculum

Start training on a **narrow `TrackGenParams`** (shorter tracks, fewer corners,
maybe 1 lap, gentler speed limits) and **widen toward the full distribution over
Phase 1**, so the agent learns fundamentals before facing every layout. Far more
sample-efficient than brute-forcing the full distribution from step 0.

**Implementation sketch:** a params schedule threaded into `train_self_play` /
`TrackSampler` (e.g. interpolate `num_corners_range`, `length_range`, `laps` from
"easy" to default across the first N steps). Reuses existing `TrackGenParams` knobs.
**Effort:** small–moderate. **Risk:** low.

---

## Idea 3 (DECIDED) — Stronger / longer dense reward shaping

Current shaping is progress-only, weight `0.02` annealed to `0` over Phase 2. On a
much harder sparse problem, hold a **stronger progress signal longer** (and
optionally add an explicit **anti-spinout** penalty and/or **corner-clear** reward)
to densify the gradient. The original design explicitly sanctions this: default 0,
"turn it on if learning stalls" (`sprint5-ml-roadmap.md §6.3`) — it has stalled.

**Implementation sketch:** raise `shaping_weight_start`, anneal more slowly (or to a
small non-zero floor); optionally extend `step_reward` (`spaces.py`) with a bounded
anti-spinout term. Keep it modest so it never dominates the win signal.
**Effort:** tiny (config) + small (optional new term). **Risk:** low–med (over-shaping
can teach degenerate play — keep bounded, anneal).

---

## Idea 4 (DECIDED) — Phase-1-heavy budget (minimal / no Phase 2)

Phase-2 self-play collapsed and added ~nothing in both runs; the strength came from
Phase 1 (vs the strong-heuristic curriculum). For the generalization run, pour the
(much larger) budget into **Phase 1 on generated tracks vs strong heuristics**, and
either **skip Phase 2** or keep it short. Self-play across random tracks is even more
unstable, and there's no point spending compute on a phase that degrades the policy.

**Implementation sketch:** set `phase1_steps ≈ total_timesteps` (or a small Phase 2).
**Effort:** trivial (config). **Risk:** low. Revisit self-play once a strong
generalist base exists.

---

## Idea 5 (DECIDED: OK to grow) — Network size

User accepts trading **training time** for performance. With the richer whole-track
obs and a CNN/attention extractor, bump the net (a new "xlarge" `net_profile`:
larger `features_dim`, deeper policy/value MLP, CNN/attention width). Cost is wall-
clock only; GPU has headroom. Tune width when wiring the extractor.

---

## Idea 6 (minor / cheap insurance)

- **Longer entropy warm-up** and/or `reload_best_on_regression=True` if any Phase 2
  is kept — guards the collapse cliff.
- **Broader Phase-1 opponent mix** — StrongHeuristic strengths 2 **and** 3, plus a
  Heuristic and a Random, for curriculum variety.
- **Longer run overall** — the dominant lever once 1–4 are in. Size after a short
  throughput probe with the new (heavier) extractor.

---

## Code-audit findings (2026-06-18) — issues & additional levers

> A read-through of the ML stack (`features.py`, `spaces.py`, `env.py`,
> `model.py`, `training.py`, `vec.py`, `action_codec.py`, `evaluate.py`,
> `generator.py`) surfaced issues the six ideas above don't touch. They sit in
> the **training loop, action space, network conditioning, and eval
> methodology**, and several interact badly with decisions already made —
> especially **Idea 4** (Phase-1-heavy budget). Numbered 7+ to continue the
> decisions log. Tier 1 are not optional polish: under Idea 4 they decide
> whether the big run can be trusted or could silently discard its best policy.

### Tier 1 — interacts with decisions already made (do alongside Idea 1B)

#### Idea 7 (DECIDED) — In-Phase-1 periodic eval + best-checkpoint preservation
**Problem.** The 6C eval-gated best-checkpoint machinery only runs *in Phase 2*
(`training.py:911-919`). Phase 1 scores the gate exactly once, at its end
(`training.py:778-779`). **Idea 4 makes Phase 1 ≈ the whole run**, so we would
train millions of steps and keep whatever the *final* weights are — even if the
policy peaked partway and regressed. The collapse protection built in 6C is
effectively inactive for the run that matters.
**Fix.** Add a periodic eval during Phase 1 (held-out gate every N steps) that
preserves the best-scoring checkpoint, and log the score to TensorBoard so
plateaus/regressions are visible live (folds in the old "Idea 6 minor — periodic
eval" point). **Effort:** small (an SB3 `EvalCallback`-style hook on the existing
`_gate`). **Risk:** low.

#### Idea 8 (DECIDED) — Gate against the opponent we actually train against
**Problem.** The inline gate calls `evaluate_ml` with its default
`opponent_factory=heuristic_agent_factory()` (`evaluate.py:131-132`) — the *weak*
`HeuristicAgent`. When `use_strong_heuristic_opponents=True` we train vs
`StrongHeuristicAgent` but select the "best" checkpoint by win-rate vs the easy
heuristic. That metric saturates near 1.0 and loses discriminating power exactly
where we need it, and "best" is chosen on a different distribution than we
optimize.
**Fix.** Gate vs the strong opponent (report both weak and strong). **Effort:**
tiny (pass the strong factory into the gate). **Risk:** low.

#### Idea 9 (DECIDED) — Make the gate statistically trustworthy
**Problem.** `gate_games=20` split across the 3 held-out tracks is ≈ **6
games/track** (`training.py:648`). A 6-game win-rate has a ±20%+ CI. Both
best-checkpoint selection (Idea 7) and the 6D PFSP weights ride on that noise, so
a lucky-but-worse checkpoint can displace a genuinely better one.
**Fix.** Raise gate games for the long run and gate on the **Wilson lower bound**
rather than the point estimate (`wilson_interval` is already imported in
`evaluate.py`), so we only promote a checkpoint when it is *confidently* better.
**Effort:** small. **Risk:** low.

#### Idea 10 (DECIDED) — Randomize the learner's seat / start position
**Problem.** The learner is hard-pinned to seat 0 everywhere (`HeatEnv`
`learner_id=0`; `evaluate_ml` always seats the MLAgent at 0, `evaluate.py:139`).
Training never sees a mid-pack start, yet HEAT has real grid / first-mover
effects. We may be generalizing across *tracks* while overfitting a single
*start position*. (The 6B `head_to_head`/`round_robin` evals cancel seat order;
training does not — an inconsistency that hides the gap.)
**Fix.** Randomize `learner_id` (hence start slot) per episode. The observation
already encodes opponents *relatively* (`features._opponent_slots`), so the
problem is well-posed — this is a cheap generalization lever, not an architecture
change. **Effort:** small. **Risk:** low.

### Tier 2 — structural levers (new features beyond the obs work)

#### Idea 11 (DECIDED) — Audit / widen the REACT action table
**Problem.** REACT is a fixed 8-slot table (`action_codec.py:149-158`) that omits
many legal heat plays — e.g. cool-1+boost, cool-2+adrenaline-speed,
boost+adrenaline-cooldown. In a game whose entire skill ceiling *is* heat
economy, this caps how finely the policy can manage the engine, and **no amount
of whole-track observation (Idea 1) fixes an action the policy cannot emit.**
**Fix.** Audit which legal react combinations are ever optimal and expand the
table to cover them (a contract change: bumps `REACT_SIZE`/`ACTION_DIM`, rides
the same codec-v2 bump as Idea 1). **Effort:** small–moderate. **Risk:** low–med
(larger action space to mask correctly — covered by the codec tests).

#### Idea 12 (DECIDED) — Slipstream-context observation features
**Problem.** Slipstream is decided nearly blind: the obs carries only a single
"slipstream available" bit (`features.py:213`), not the gap to the car ahead or
its speed — which is exactly what determines whether taking it is worth it. The
whole-track obs (Idea 1) is about *track* geometry and won't add this *opponent*
context.
**Fix.** Add slipstream-relevant features (gap to the car ahead, its
speed/whether a boost is pending) to the ego/scalar block. Folds naturally into
the Idea 1 obs rework + codec v2. **Effort:** small. **Risk:** low.

### Tier 3 — network conditioning (expanded from original "issue 7")

#### Idea 13 (DECIDED) — Decision-kind conditioning + unshared extractor
**Problem.** One shared features extractor (`HeatMLPExtractor`) feeds both heads
(`share_features_extractor` defaults to `True`, `model.py:232-239`), and the only
signal telling the net *which* of the five decision kinds is pending is the 5-dim
one-hot buried in the 72-dim obs (`features.py:201-202`). The **actor** must
behave completely differently per kind, yet a wide trunk can dilute or ignore 5
inputs out of 72 — and **this gets worse after Idea 1**, when `OBS_DIM` grows to
~170+ and the one-hot drops to ~3% of the input. (The *critic* concern is milder:
V(s) across the micro-decisions within one round is nearly identical — the
terminal placement reward doesn't care which sub-decision is pending — so this is
primarily an actor-conditioning problem, **not** a reason to build a multi-head
critic.)
**Fix (cheap, do now, in the same extractor rewrite):**
1. **Skip-connect the kind one-hot (+ react/slipstream context bits) past the
   trunk** — concatenate onto the latent before the heads, or FiLM-modulate the
   latent with it — so the actor always sees an undiluted mode signal. Assert the
   one-hot's obs indices in the feature test (the slice becomes part of the frozen
   contract).
2. **Set `share_features_extractor=False`** so the actor (organized around "what
   action given this kind") and critic (organized around "how's the race going")
   stop competing for one trunk — a standard PPO friction that worsens as the net
   grows (Idea 5). One-line `policy_kwargs` change; costs params, already accepted.

**Effort:** small (rides on the Idea 1 extractor work). **Risk:** low.

**Contingency (NOT default):** a per-kind value/policy head (gather the head
indexed by the one-hot) is the heavy option — it needs a custom
`MaskableActorCriticPolicy` subclass, not just `policy_kwargs`. Gate it behind a
concrete diagnostic: only build it if, *after* 1+2, the value loss fails to
converge or poor behavior stays concentrated in one decision kind (e.g. react
play lags while gear/cards improve).

### Tier 4 — performance & minor / cheap insurance

- **Idea 14 (DECIDED) — Cache the static track encoding for Option B.** The track
  is constant within an episode, so the expensive whole-track grid/corner block
  must be computed once per `reset()` and only ego-shifted per step.
  `encode_observation` is pure-Python list-building (`features.py:238-248`) and
  the engine is *already* the throughput bottleneck (~1,080 steps/s,
  `vec.py:4-6`); recomputing the heavy Option-B block every step would tank it.
  Build this into the Idea 1 design from the start. **Effort:** small (design-time).
  **Risk:** low.
- **Idea 15 (DECIDED) — Keep `normalize_obs` OFF for Option B.** The obs is already
  bounded to [-1, 1]; `VecNormalize` obs-standardization would fight both the
  bounded design and the CNN's grid channels. (The gate already uses the real,
  un-normalized win-rate, so reward-norm doesn't corrupt selection — keep it that
  way.) **Effort:** trivial (config). **Risk:** low.
- **Idea 16 (opportunistic) — Partial-progress reward on truncation.** A truncated
  episode (`round_num > MAX_ROUNDS = 200`) gives *everyone* reward 0
  (`spaces.py:152-153`). Rare at current track lengths, but if the curriculum /
  longer runs push episodes out, a *winning-but-unfinished* learner gets no
  signal. A bounded partial-progress grade on truncation is cheap insurance.
- **Idea 17 (opportunistic) — Evaluate sampled *and* deterministic.** MLAgent eval
  is argmax (`deterministic=True`); a stochastically-trained policy in a
  hidden-information game can under-report and be exploitable under argmax. Report
  both.
- **Idea 18 (opportunistic) — A north-star reference opponent.** Every opponent is
  a heuristic or a past self, so we never measure the *true* skill ceiling. An
  eval-only lookahead / shallow-search opponent would tell us how much headroom
  remains beyond "beats the strong heuristic."

## Decisions log

| # | Idea | Decision |
|---|---|---|
| 1 | Whole-track observation | **Option B**: full track-map + CNN/attention extractor; codec bump to v2 |
| 2 | Difficulty curriculum | **Yes** — narrow→wide `TrackGenParams` over Phase 1 |
| 3 | Reward shaping | **Yes** — stronger/longer; optional anti-spinout term |
| 4 | Phase budget | **Yes** — Phase-1-heavy, minimal/no Phase 2 |
| 5 | Net size | **OK to grow** (training-time cost acceptable) |
| 6 | Minor levers | Opportunistic |
| 7 | In-Phase-1 best-checkpoint + periodic eval | **Yes** — restores the collapse net under Idea 4 |
| 8 | Gate vs the trained-against opponent | **Yes** — gate vs strong, report both |
| 9 | Gate statistical robustness | **Yes** — more games + Wilson lower-bound gating |
| 10 | Randomize learner seat / start | **Yes** — cheap generalization lever |
| 11 | Widen REACT action table | **Yes** — audit + expand (rides codec v2) |
| 12 | Slipstream-context obs features | **Yes** — fold into Idea 1 obs |
| 13 | Decision-kind conditioning + unshared extractor | **Yes** (skip-connect + `share_features_extractor=False`); multi-head head only on diagnostic |
| 14 | Cache static track encoding | **Yes** — design Option B around it |
| 15 | `normalize_obs` off for Option B | **Yes** |
| 16–18 | Truncation reward / dual eval / north-star opponent | Opportunistic |

## Open questions

- **B encoding:** spatial 1D-CNN grid (recommended start) vs corner-sequence
  attention — decide at build time; could prototype both cheaply on the feature
  test before committing the long run.
- **`GRID` / `MAX_CORNERS` / globals** exact sizes → final `OBS_DIM`.
- **Curriculum schedule shape** (linear vs staged) and over how many steps.
- **Shaping magnitude / anti-spinout weight** — tune small, gate against degenerate
  play.
- **Run scale** for the real run — set after a throughput probe with the new
  extractor (heavier per-step than the MLP).
- **Phase-1 eval cadence** (Idea 7) — how often to run the held-out gate vs the
  per-eval cost (the gate plays real games); tie to Idea 9's game count.
- **REACT table scope** (Idea 11) — which missing combos are actually worth the
  added action-space width; confirm against the engine's legal-react rules.
- **Conditioning form** (Idea 13) — plain skip-concat vs FiLM modulation; decide
  on the feature test before the long run.

## Suggested build order (when we proceed)

1. **Idea 1B** — whole-track obs + custom extractor + codec v2 + feature tests
   (the foundational change; everything else rides on it). Fold the obs/codec-v2
   companions in here: **Idea 12** (slipstream features), **Idea 11** (wider REACT
   table), **Idea 13** (decision-kind conditioning + unshared extractor), and
   **Idea 14** (static-track caching) — they all share the same codec bump and
   extractor rewrite, so doing them together avoids a second contract change.
2. **Idea 4** (phase budget) + **Idea 3** (shaping) + **Idea 15** (`normalize_obs`
   off) — config-level, fold in.
3. **Tier-1 training-loop fixes** — **Idea 7** (in-Phase-1 best-checkpoint +
   periodic eval), **Idea 8** (gate vs strong), **Idea 9** (Wilson-LCB gate),
   **Idea 10** (seat randomization). Land these *before* the long run: under Idea 4
   they decide whether the run is trustworthy and recoverable.
4. **Idea 2** (curriculum) — params schedule.
5. **Idea 5** (net profile) — size the extractor/net.
6. Throughput probe → size and launch the long run; evaluate on held-out generated
   tracks (and USA) exactly as in `tmp/eval_generalization.py`, reporting **both**
   weak- and strong-heuristic win-rates and **both** sampled and deterministic
   policies (Idea 17). Treat **Idea 16/18** as opportunistic.

## Validation (unchanged methodology)

Train on generated tracks, **gate and evaluate on held-out generated tracks** (seeds
disjoint from training/gate), vs weak and strong heuristics, plus a USA reference.
Target: a model that wins decisively on *unseen* generated tracks (not just one
track) — the real generalization bar.
