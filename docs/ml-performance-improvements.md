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

## Decisions log

| # | Idea | Decision |
|---|---|---|
| 1 | Whole-track observation | **Option B**: full track-map + CNN/attention extractor; codec bump to v2 |
| 2 | Difficulty curriculum | **Yes** — narrow→wide `TrackGenParams` over Phase 1 |
| 3 | Reward shaping | **Yes** — stronger/longer; optional anti-spinout term |
| 4 | Phase budget | **Yes** — Phase-1-heavy, minimal/no Phase 2 |
| 5 | Net size | **OK to grow** (training-time cost acceptable) |
| 6 | Minor levers | Opportunistic |

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

## Suggested build order (when we proceed)

1. **Idea 1B** — whole-track obs + custom extractor + codec v2 + feature tests
   (the foundational change; everything else rides on it).
2. **Idea 4** (phase budget) + **Idea 3** (shaping) — config-level, fold in.
3. **Idea 2** (curriculum) — params schedule.
4. **Idea 5** (net profile) — size the extractor/net.
5. Throughput probe → size and launch the long run; evaluate on held-out generated
   tracks (and USA) exactly as in `tmp/eval_generalization.py`.

## Validation (unchanged methodology)

Train on generated tracks, **gate and evaluate on held-out generated tracks** (seeds
disjoint from training/gate), vs weak and strong heuristics, plus a USA reference.
Target: a model that wins decisively on *unseen* generated tracks (not just one
track) — the real generalization bar.
