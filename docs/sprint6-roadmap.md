# Design: Sprint 6 — Strong Agent (Roadmap)

> **Status:** planning / roadmap only. This document does **not** implement any
> code. It splits Sprint 6 into **three mutually-parallel sub-sprints (6A, 6B,
> 6C) plus a fourth (6D) that depends on 6C** and serializes after it, locks the
> contract invariants that must hold across them, gives the dependency graph, and
> ends with an honest verdict on parallelism. It mirrors the structure of
> `docs/sprint5-ml-roadmap.md` (§1 executive summary, §4 dependency graph +
> parallelism verdict) and cites real `file:line` anchors.
>
> **Prerequisite (already met & verified):** the Sprint 5 ML layer is landed and
> tested on `master` (507 tests green). The frozen ML contract lives in
> `src/heat/ml/spaces.py` (`OBS_DIM=72` at `spaces.py:42`, `ACTION_DIM=516` at
> `spaces.py:125`, `MAX_PLAYERS=6` at `spaces.py:47`, `CODEC_VERSION=1` at
> `spaces.py:35`, `SHAPING_WEIGHT=0.0` at `spaces.py:134`). The env, model,
> training, agent, and eval modules all build on it.

---

## 1. Executive summary

Sprint 5 produced a working RL pipeline but **not a strong agent**. A real 2M-step
training run + code review surfaced three independent weaknesses, each of which
this sprint addresses with a dedicated, separately shippable sub-sprint:

1. **The training target is too narrow.** `HeatEnv` trains on a single fixed
   track (USA, hard-coded at `env.py:75` → `env.py:112`, sampled once at
   `env.py:185`), and only **two** hand-authored tracks exist (`tracks/usa.json`,
   `tracks/silverstone.json`). A policy trained on one track cannot be shown to
   generalize. → **Sprint 6A: procedural track generation + multi-track
   training/eval.**
2. **The benchmark is weak.** "94% vs heuristic" measures the *heuristic's*
   weakness, not the policy's strength: `HeuristicAgent` is one-turn greedy with
   zero opponent awareness (`heuristic_agent.py:79-334`). Win-rate vs one weak
   scripted opponent is not a meaningful strength bar, and there is no
   head-to-head, no ELO, no cross-track eval. This splits into two sub-sprints:
   → **Sprint 6E: a genuinely strong heuristic agent (opponent-aware, joint
   gear+cards, multi-corner lookahead, race-long heat/deck economy) on a difficulty
   ladder — the strength bar itself.** → **Sprint 6B: richer evaluation
   (head-to-head, pool ELO/TrueSkill with CIs, cross-track) — the honest
   scoreboard that ranks the ladder.** 6B consumes 6E's agent but falls back to
   `{Random, Heuristic, MLAgent}` if 6E has not landed (soft coupling).
3. **Self-play collapses, and the throughput ceiling blocks recovery.** Phase 2
   self-play craters reward to a sustained −1 and the *final saved checkpoint
   overwrites the good Phase-1 model* (`train_self_play` at `training.py:376-379`
   always writes the final, collapsed model to the canonical path). Meanwhile
   training runs a single non-vectorized env (`HeatEnv` is one `gym.Env`,
   `env.py:78`) at ~1,080 steps/s, bottlenecked on the pure-Python engine, with
   `device="cpu"` hard-set (`model.py:87`, `training.py:139`). → **Sprint 6C:
   training throughput (vectorized envs, optional CUDA) + self-play stability
   (eval-gated best-checkpoint preservation, stochastic snapshots, shaping
   schedule, critic warm-up).**

**The capstone** — actually training a genuinely strong agent — depends on 6A, 6C,
and the 6E+6B pair landing. 6A widens what it learns on, **6E is the real bar** and
6B is the honest scoreboard that ranks it, 6C lets it train fast without throwing
away the good model.

**A fourth sub-sprint, 6D (opponent league + PFSP), is a stretch / strengthening
extension built on top of 6C.** 6C makes self-play *survivable* (a collapse can no
longer destroy the good model); 6D attacks self-play *fragility at the root* by
replacing the FIFO snapshot pool (`training.py:356-358`) with a persistent,
diverse opponent league sampled by **prioritized fictitious self-play (PFSP)**.
Because it extends 6C's vectorized training loop and its opponent-sampling
machinery, **6D depends on 6C and is NOT parallel with it.** The **base capstone
is not blocked on 6D** — it can train on 6A+6B+6C+6E alone; 6D, if it lands, feeds an
*even-stronger* capstone. See `docs/sprint6d-league-selfplay-design.md`.

### Verdict on parallelism (detail in §5)

**6A, 6B, 6C, and 6E are each independently buildable and can run in PARALLEL.**
None of the four depends on another's *runtime behavior* to begin — unlike the
Sprint 5 `env → training → agent` pipeline, these are orthogonal extensions of an
already-complete pipeline. 6E is a standalone `BaseAgent` testable in a plain
`Game`; 6B is eval plumbing over existing primitives. The only couplings are
**soft** (a fallback always exists):

- **(i)** 6C's eval-gated snapshot promotion *benefits from* 6B's richer eval but
  can fall back to a minimal win-rate gate over `evaluate_ml` (which already
  exists, `evaluate.py:82`).
- **(ii)** 6B's cross-track evaluation *benefits from* 6A's track generator but
  can fall back to the two static tracks.
- **(iii)** 6B's rating ladder *benefits from* 6E's `StrongHeuristicAgent` as a
  rung but can fall back to ranking `{Random, Heuristic, MLAgent}`. (6E in turn
  *benefits from* 6B's round-robin to tune its `heat_price`, but ships a derived
  default first — a soft *back*-edge, not a blocker.)
- **(iv)** the **capstone strong-agent training run depends on 6A + 6B + 6C + 6E**.

So the honest verdict is: **4-way parallel (6A/6B/6C/6E) for the bulk of the
sprint**, then **6D serializes after 6C** (it extends 6C's loop), with a short
serial tail where the capstone consumes 6A+6B+6C+6E (and *optionally* benefits from
6D). This is genuinely more parallel than Sprint 5 (which was front-loaded and
collapsed to one track by 5b).

### Sub-sprint breakdown

| Sub-sprint | Goal | Key files (new/modified) | Gating tests | Depends on |
|---|---|---|---|---|
| **6A — Track generation** | Seedable procedural `Track` generator producing valid raceable tracks; per-episode track sampling in `HeatEnv`; multi-track eval sweep | `tracks/generator.py` (new); `ml/env.py` (track sampling); maybe `tracks/validate.py` | `test_track_generator.py` (validity, determinism), `test_ml_env.py` additions (obs still `(72,)` ∈ [−1,1] on generated tracks) | spaces contract (frozen) |
| **6B — Eval overhaul** | Head-to-head + pool ELO/TrueSkill (with CIs) + cross-track eval; new stats types | `ml/evaluate.py` (head-to-head, ELO); `simulation/stats.py` (ELO/round-robin types) | `test_ml_eval.py` additions (ELO determinism, head-to-head symmetry, CI sanity) | spaces contract; reuses `run_batch`/`aggregate_stats`; *soft* on 6E |
| **6E — Strong heuristic** | Genuinely strong, opponent-aware scripted agent: joint gear+cards, multi-corner lookahead, race-long heat/deck economy, on a difficulty ladder — the strength bar | `agents/strong_heuristic.py` (new); `simulation/runner.py` (`strong_heuristic_agent_factory`) | `test_strong_heuristic.py` (legality, beats `HeuristicAgent` head-to-head, ladder monotonicity, opponent-awareness behaviours) | `BaseAgent`, `rules.*` (all exist) |
| **6C — Training overhaul** | Vectorized/parallel envs + optional CUDA + larger net; self-play that does not collapse; best-checkpoint never overwritten; return/reward normalization; committed training CLI + TensorBoard; reproducible checkpoints | `ml/training.py` (vec envs, gated promotion, schedules, `VecNormalize`, repro meta); `ml/model.py` (`device` selection, larger net); `agents/ml_agent.py` (apply norm stats); `scripts/train_ml.py` (new); maybe `ml/vec.py` (new) | `test_ml_training.py` (vec determinism, best-checkpoint preservation, norm-stats round-trip, meta back-compat), `test_train_cli.py`, `test_ml_training_smoke.py` (`slow`: self-play smoke does not collapse) | spaces contract; `HeatEnv` (exists) |
| **6D — League / PFSP self-play** | Persistent diverse opponent **league** + **prioritized fictitious self-play** replacing FIFO snapshots; per-opponent win-rate bookkeeping | `ml/league.py` (new — league + PFSP sampler); `ml/training.py` (use league in place of `_mixed_opponent_pool` `training.py:383-399`) | `test_league.py` (retention determinism, PFSP weights sum to 1 & correct), `test_ml_training_smoke.py` (`slow`: league self-play not below the 6C gated baseline) | **6C** (extends its vec loop + opponent sampling) |
| **Capstone** | Train a genuinely strong agent on generated tracks vs the strong heuristic + ELO scoreboard | — (a run, not code) | manual / CI-optional | **6A + 6B + 6C + 6E** (6D optional — strengthens it) |

Each sub-sprint ends by writing tests and running the full suite —
`PYTHONPATH=src python -m pytest tests/ -q` — the hard project convention.

---

## 2. Motivation

The Sprint 5 success metric ("MLAgent win-rate ≥ HeuristicAgent",
`sprint5-ml-roadmap.md:496`) was *met* (94.3% vs 1.9% at the 1.5M snapshot) yet
the agent is **not strong**. Three verified findings explain the gap:

1. **Self-play collapse (confirmed).** Phase 1 vs the scripted pool reached
   `ep_rew_mean ≈ 0.9`; the 1.5M snapshot beats the heuristic 94.3% vs 1.9%
   (4-player). The instant Phase 2 started, reward cratered to a sustained **−1**
   and never recovered; the final saved checkpoint scores 10.3% (below the 25%
   random baseline). Mechanisms diagnosed (and visible in the code):
   - **(a) Value-function shock on `model.set_env`** (`training.py:365`). The
     critic is trained to expect returns ≈ +0.9 against the scripted pool; when
     the opponent pool is swapped to strong self-copies, realized returns become
     ≈ −1, so the advantage estimates invert and PPO takes large destructive
     updates.
   - **(b) Entropy / exploration collapse under pure-sparse reward**
     (`SHAPING_WEIGHT=0.0`, `spaces.py:134`). When every game is a loss (all −1),
     there is no gradient signal differentiating actions, so the policy cannot
     climb back.
   - **(c) Deterministic identical opponents.** `_mixed_opponent_pool`
     (`training.py:383-399`) fills snapshot seats with
     `FrozenSnapshotAgent(p)`, whose default is `deterministic=True`
     (`training.py:128-130`). With `snapshot_mix=0.5` over a 4-player game that is
     ~1-2 identical, deterministic, strong copies — zero variance, uniform losing.
   - **(d) Phase-1 brittleness.** The policy overfit to exploit the weak
     heuristic; it has no robust general racing skill to fall back on.
2. **The heuristic is weak** (`heuristic_agent.py`). One-turn greedy scoring
   (`choose_gear:79`, `choose_cards:143`), zero opponent awareness (slipstream
   `choose_slipstream:281` and the missing block logic ignore rivals), hand-tuned
   magic numbers (e.g. `score -= corner_cost * 10` at `heuristic_agent.py:117`),
   gear/cards not jointly optimized (gear estimates speed, cards re-derive it
   independently), and no race-long deck/heat economy. So beating it is not a
   meaningful strength bar — and is *why* the policy shattered in self-play.
3. **GPU / throughput.** The machine has an RTX 4080 SUPER (16 GB), but installed
   torch is **CPU-only** (`2.12.0+cpu`) and training is a single non-vectorized
   env at ~1,080 steps/s. The bottleneck is pure-Python engine stepping, not the
   net (`sprint5-ml-roadmap.md:523` "Training is CPU-bound … the existing
   pure-Python tests"). **A naive GPU switch on a tiny MLP single-env gives ~no
   speedup or a regression** (per-batch host↔device transfer dominates). The real
   lever is **vectorized envs** across CPU cores to beat the Python bottleneck,
   which then justifies a larger net that benefits from the GPU.

---

## 3. Contract invariants (what Sprint 6 must NOT break)

Sprint 6 adds breadth (tracks, opponents, throughput, stability); it must keep
the Sprint 5 frozen contract intact. The single most important invariant:

> **The `OBS_DIM=72` observation contract is track-agnostic and must stay so.**
> `encode_observation` (`features.py`) is already **track-relative**: it
> normalizes by `track.length`, `track.laps`, and corner geometry (confirmed in
> `features.py`; called out in `sprint5-ml-roadmap.md:579-582` and the
> `spaces.py:122-125` track-lookahead block). Therefore a *generated* track
> produces a `(72,)` float32 vector in `[−1, 1]` with **no codec change** — as
> long as the generated track satisfies the same structural assumptions the
> features make (non-zero length, ≥1 lap, corners with valid `start ≤ end` and a
> positive `speed_limit`, see `track.py:8-21`). **6A owns proving this
> invariant** with a test that runs `encode_observation` on generated tracks and
> asserts shape `(72,)` and bounds `[−1, 1]`.

Other invariants:

- **No `CODEC_VERSION` bump.** None of 6A/6B/6C changes `OBS_DIM`, `ACTION_DIM`,
  or the codec layout. If any of them *did* (it should not), the checkpoint
  sidecar tripwire (`MLAgent` meta check, `ml_agent.py:21-27`; written by
  `save_checkpoint` at `training.py:73-96`) would correctly reject old models —
  treat any need to bump `CODEC_VERSION` as a contract change requiring all three
  sub-sprints to re-sync, not a local edit.
- **`MAX_PLAYERS=6` and the action layout are untouched.** Track generation must
  produce `len(start_positions) ≥ MAX_PLAYERS` (or at least ≥ the configured
  `num_players`) so all seats can be placed (`game_state.py:225` places player
  `i` at `track.start_positions[i]`).
- **Determinism remains seed-reproducible.** `GameState.create(track,
  num_players, seed=)` (`game_state.py:184`) is a deterministic function of its
  seed; 6A's generator and 6C's vec envs must each be seedable so the whole
  pipeline stays reproducible (see each sub-sprint's determinism test gate).
- **The `BaseAgent` pull-protocol is the only agent interface.** 6B's strong
  heuristic is a drop-in `BaseAgent` (like `HeuristicAgent`,
  `heuristic_agent.py:13`); it must not require engine changes.

---

## 4. Non-goals (sprint-wide)

- **No engine rule changes.** Track generation uses the existing `Track`/`Space`/
  `Corner` schema (`track.py:8-52`); no new board mechanics.
- **No `CODEC_VERSION` / obs / action-space changes** (§3).
- **No true multi-head policy / new RL algorithm.** Still `MaskablePPO` over the
  flattened masked `Discrete(ACTION_DIM)` (`model.py:173`,
  `sprint5-ml-roadmap.md:561`).
- **No distributed / multi-machine training.** 6C is single-machine: multi-core
  CPU vec envs + one optional local GPU.
- **No learned track generator / GAN tracks.** 6A is rule-based procedural
  generation, not ML-generated tracks.
- **The capstone "strong agent" is a *run*, not a unit gate.** Like
  `sprint5-ml-roadmap.md:496`, strength is a manual/CI-optional metric.

---

## 5. Dependency graph + parallelism verdict

### 5.1 Dependency graph

```
        ┌──────────────────────────────────────────────────────────┐
        │  FROZEN CONTRACT  (ml/spaces.py: OBS_DIM=72, ACTION_DIM=   │
        │  516, MAX_PLAYERS=6, CODEC_VERSION=1)  — UNCHANGED         │
        └───────┬───────────────────┬───────────────────┬───────────┘
                │                   │                   │
        ┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
        │ 6A  track gen  │  │ 6B  benchmark  │  │ 6C  training    │   ── 6A/6B/6C
        │ + multi-track  │  │ strong heur +  │  │ throughput +    │      PARALLEL
        │ env sampling   │  │ ELO/h2h/x-trk  │  │ self-play stab. │
        └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
                │  (generated         │ (richer eval     │ (gated promotion,
                │   tracks)           │  + strong heur)  │  vec envs)
                │   soft (ii)─────────┤                  │ HARD edge 6C → 6D
                │                     └──── soft (i) ────┤  (eval gate         │
                │                                        │   benefits from 6B) │
                │                                        │             ┌───────▼────────┐
                │                                        │             │ 6D  league /   │ SEQUENTIAL
                │                                        │             │ PFSP self-play │ after 6C
                │                                        │             │ (extends 6C)   │ (stretch)
                │                                        │             └───────┬────────┘
                └────────────────┬───────────────────────┘                    │
                                 ▼                          (optional, stronger │ capstone)
                    ┌────────────────────────────┐ ◄─────────────────────────┘
                    │  CAPSTONE: strong-agent run │  needs 6A + 6B + 6C
                    │  (generated tracks, strong  │  (6D optional: strengthens
                    │   heuristic, ELO scoreboard)│   it, does not block it)
                    └────────────────────────────┘
```

> **Not drawn above:** **6E (strong heuristic)** is a fourth parallel strand. It is
> the `agents/strong_heuristic.py` agent that the graph's "6B benchmark" box and the
> capstone both reference. It feeds 6B's rating ladder by a *soft* edge (vi) and the
> capstone by a *hard* edge — see (iv) below. Treat the "6B benchmark" box as
> the **6B+6E** pair.

Soft edges (dashed) each have an explicit fallback so the consumer is never
*blocked*:

- **(i) 6C ← 6B:** eval-gated promotion is best with 6B's ELO/cross-track eval;
  fallback = a minimal win-rate gate via the existing `evaluate_ml`
  (`evaluate.py:82`).
- **(ii) 6B ← 6A:** cross-track eval is best with 6A's generator; fallback =
  sweep the two static tracks (`tracks/usa.json`, `tracks/silverstone.json`).
- **(iii) capstone ← {6A,6B,6C,6E}:** hard — the capstone consumes all four.
- **(iv) 6B ← 6E:** the rating ladder is best with 6E's `StrongHeuristicAgent` rung;
  fallback = rank `{Random, Heuristic, MLAgent}`. A soft *back*-edge runs the other
  way (6E's `heat_price` tuning benefits from 6B's round-robin), resolved by 6E
  shipping a derived default first — so the two are not mutually blocking.

The one **hard intra-sprint edge** is **6C → 6D** (solid in the graph): 6D extends
6C's vectorized training loop and its opponent-sampling code, so it cannot start
until 6C's stable loop exists. 6D → capstone is a *soft* edge: a 6D-strengthened
loop produces a better capstone, but the base capstone needs only 6A+6B+6C+6E.

### 5.2 Work-unit table: PARALLELIZABLE vs SEQUENTIAL

| Work unit | Mode | Codes against | Why / what it waits on |
|---|---|---|---|
| **6A** generator + env sampling | **PARALLELIZABLE** | `Track` schema (`track.py`), frozen `spaces` consts | Pure data generation + a one-line-ish env reset change; no dependency on 6B/6C. Tests run on the generator and obs bounds alone. |
| **6E** strong heuristic | **PARALLELIZABLE** | `BaseAgent` (`base.py`), `rules.*` | A new drop-in agent; testable in a plain `Game` with no ML at all. Its own sub-sprint (`docs/sprint6e-strong-heuristic-design.md`). |
| **6B** ELO / h2h / cross-track eval | **PARALLELIZABLE** | `run_batch`/`aggregate_stats` (exist), `Track` | New stats + harness; cross-track *prefers* 6A but falls back to static tracks. |
| **6C** vec envs + device selection + larger net | **PARALLELIZABLE** | `HeatEnv` (exists), SB3 VecEnv API | Rewrites env construction + model device; needs no 6A/6B runtime. |
| **6C** self-play stability (gated promotion, schedules, stochastic snapshots, return normalization, repro meta) | **PARALLELIZABLE** (after 6C throughput internally) | `HeatEnv`, `evaluate_ml` (exists), `ml_agent.py` | The promotion gate *prefers* 6B's richer eval but falls back to win-rate. Internally ordered after vec-env foundation (§6C). |
| **6C** training CLI + TensorBoard | **PARALLELIZABLE** | `PPOConfig`/`CurriculumConfig` (exist) | New `scripts/train_ml.py`; replaces the throwaway `train_and_eval_run.py`. No 6A/6B/6D runtime. |
| **6D** opponent league + PFSP sampler | **SEQUENTIAL** (after 6C) | 6C's vec loop + `_mixed_opponent_pool` (`training.py:383-399`), `FrozenSnapshotAgent` | Extends 6C's Phase-2 opponent sampling — **cannot run in parallel with 6C**; needs the stable vec-env training loop to build on. |
| **6E** strong heuristic agent | **PARALLELIZABLE** | `BaseAgent`, `rules.*` (all exist) | A new drop-in agent + value evaluator; tested in a plain `Game`, no ML. Soft-feeds 6B's ladder; the capstone uses it as the strength bar. |
| **Capstone** strong-agent training run | **SEQUENTIAL** | 6A + 6B + 6C + 6E landed (6D optional) | Consumes generated tracks (6A), the strong heuristic (6E) + ELO scoreboard (6B), and the stable vec-env training loop (6C). A landed 6D makes it stronger but is not required. |

### 5.3 Honest parallelism verdict

**Genuinely parallel: all four sub-sprints (6A/6B/6C/6E), for the bulk of the
sprint.** Unlike Sprint 5's `env → training → agent → eval` pipeline (where each
stage consumed the previous stage's *behavior*), these are orthogonal extensions of
an already-complete pipeline. Each compiles and tests against the **frozen contract
+ existing modules**, not against another sub-sprint's runtime. Four agents can own
one sub-sprint each with near-zero integration friction (6B and 6E are the closest
pair, coupled only by the soft ladder edge with a named fallback).

**The only serial point is the capstone**, which is a *run*, not code — it waits
for all four to land, then consumes them together. The soft edges (6C's gate wanting
6B's eval; 6B's cross-track wanting 6A's generator; 6B's ladder wanting 6E's agent)
are convenience, not blockers: each has a named fallback, so none forces an ordering.

**Bottom line:** plan for **4-way parallelism** across the sub-sprints, then a
short serial capstone. Expected realistic speedup over fully-sequential is real
this time (~4 strands), not the front-loaded "1.5 agents" of Sprint 5 — because
the hard data-dependency pipeline already exists and is being widened, not built.

### 5.4 Suggested internal ordering within each sub-sprint

- **6A:** generator + validator first (unit-testable in isolation) → then env
  per-episode sampling → then the obs-invariant test.
- **6E:** evaluator core first (pure value function, unit-tested) → strength
  ladder `0→1→2→3` → gate: beats old heuristic head-to-head + ladder monotone.
  (Full detail in `docs/sprint6e-strong-heuristic-design.md`.)
- **6B:** stats types + ELO/CIs first → then `head_to_head`/`round_robin_elo`
  harness → then cross-track sweep. Runs against `{Random, Heuristic, MLAgent}`
  until 6E's strong rung lands.
- **6C:** **throughput/vectorization foundation first** (vec envs, device
  selection, larger net, measurement plan) → **then self-play stability on top**
  (gated promotion, stochastic snapshots, schedules, critic warm-up). The
  stability work assumes the vec-env loop exists.

---

## 6. Where the brief / a finding could be wrong (evidence)

I verified every claim in the brief against the code. All held. Two clarifying
corrections for the per-sprint docs:

- **`sb3-contrib` is already a dependency.** The brief's Sprint-5-era note about
  "add `sb3-contrib`" is done: `pyproject.toml:16` already lists it in the `ml`
  extra. 6C's only dep change is **optional CUDA torch**, not `sb3-contrib`.
- **The "final checkpoint overwrites the good model" failure is precise and
  code-visible.** `train_self_play` writes the post-collapse final model to the
  canonical `run_name` path unconditionally at `training.py:376-379`, *after* the
  Phase-2 loop has already overwritten the in-memory model. There is no
  best-so-far tracking anywhere in `training.py`. This is exactly the failure 6C
  must fix; the snapshot files (`{run_name}_snap{idx}`, `training.py:349-351`) are
  the only surviving Phase-1-ish artifacts and they are FIFO-capped at 3
  (`training.py:356-358`), so a long Phase 2 can evict the good one too.

See the per-sprint docs for full detail:

- `docs/sprint6a-track-generation-design.md`
- `docs/sprint6b-benchmark-eval-design.md`
- `docs/sprint6e-strong-heuristic-design.md`
- `docs/sprint6c-training-overhaul-design.md`
