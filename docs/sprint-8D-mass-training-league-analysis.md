# Sprint 8D — Mass training sweep + large round-robin league analysis (implementation design)

> **Status:** design only — **no code yet**. Builds **directly on Sprint 8C**
> (`docs/sprint-8C-solo-pretrain-opponent-curriculum.md`), which is *itself*
> design-only at time of writing (verified below — `TrainingPhase`,
> `OpponentSchedule`, `sprint_8c_curriculum`, `evaluate_league`, `LeagueLadder`
> do **not** yet exist in the tree). 8D **consumes** 8C's parameterized recipe
> and league evaluator as a stable API surface and does **not** redesign them.
> **Scope:** (1) a declarative **sweep** over 8C's recipe knobs → many gated
> checkpoints + a run manifest; (2) a **large round-robin league** with *matchup
> sampling* that ranks every swept checkpoint + reference heuristics + baseline
> anchors, persisting per-game outcomes so ratings are recomputable; (3) an
> **attribution** layer that estimates "what each factor brings" with honest
> uncertainty; (4) explicit **compute budgeting** with a default campaign that
> fits a few GPU-hours and a scale knob; (5) **reproducibility** (seeded runs,
> pinned held-out track set, git-SHA-per-run via the existing checkpoint sidecar).
> **No obs/action/codec change** (`OBS_DIM=104`, `ACTION_DIM=516`,
> `CODEC_VERSION=2` — confirmed `spaces.py:38/46/150` via 8C §2). 8D is
> **orchestration + analysis ON TOP of 8C**; the only core-API touches it may
> require are listed explicitly in **§12 (8C API additions required)**.

---

## 1. Goal

Turn the single validated recipe 8C productionizes into a **reproducible factor
sweep**, then rank everything it produces in one large league so we can attribute
rating differences to the factors we varied:

1. **Sweep.** Declaratively define a grid (or sampled set) of configs over 8C's
   knobs — solo step count, per-stage step ratios/budgets, per-phase gamma,
   opponent-pool composition, shaping weight, `randomize_seat`, net-size profile,
   seed. Each config → one gated `train_self_play(...)` run → one best checkpoint
   + a manifest row.
2. **League.** Scale 8C's `evaluate_league` to *all* swept checkpoints + the
   reference heuristics (weak/strong) + baseline/anchor checkpoints. Free-for-all
   is C(n,4) fields, which explodes for tens of models, so design **matchup
   sampling** (each model in ≥K random fields) with seat-rotation fairness and
   bootstrap CIs, and **persist every per-game outcome** so ratings recompute
   without replaying.
3. **Attribution.** Over the manifest + ladder, estimate the marginal effect of
   each factor on rating (grouped diffs and/or a simple OLS of rating on factors),
   with uncertainty, emitting both a human-readable markdown report and
   machine-readable CSV/JSON. Stay honest: the sweep is **observational**, ratings
   have CIs, factors interact — say so in the report.
4. **Compute budget.** Make N configs × per-run cost (the gate adds time) + league
   games explicit; ship a small default campaign + a scale knob.
5. **Reproducibility.** Seeded runs, a **pinned held-out track set** for the league
   (distinct from training tracks), and a manifest that records the git SHA per run
   (reuse `save_checkpoint`'s sidecar — confirmed to record `git_sha`/`seed`/
   `ppo_config`/`curriculum_config`/`track_config`/`normalize`, `training.py:171-184`).

This is **pure orchestration + analysis**: a thin, unit-tested `src/` module for
the declarative spec, manifest, matchup sampling, and attribution math; the heavy
compute (actual training runs, actual league games) lives in `experiments/`
scripts that call it. No training/eval *core* change beyond the §12 list.

---

## 2. Verified code baseline (file:line)

Every reference below was opened and confirmed against the **current** tree.
Where the symbol is an **8C deliverable that does not yet exist**, it is marked
**(8C, not yet in tree)** — 8D depends on 8C landing first (§14). Drift from the
user's framing / the 8C doc is called out in **§13**.

### 2.1 Exists today (8D builds on these directly)

| Symbol | Location | Current behavior (confirmed) |
|---|---|---|
| `round_robin_elo` | `evaluate.py:398-497` | round-robins every unordered **pair**; extra seats filled by **cycling the pair** (`evaluate.py:358-364`) → never 4 distinct models; computes ELO (+ optional TrueSkill) + pairwise table + optional per-track. Takes `tracks: list[Track]`, `seed`, `rating`, `bootstrap`, `parallel=False`. |
| `_round_robin_outcomes` | `evaluate.py:325-395` | the seating + relabel + global-`game_index`-restamp core that `round_robin_elo` wraps. **The template for a free-for-all seater.** |
| `_relabel_outcomes` | `evaluate.py:217-235` | replaces each seat's `agent_type` with a **pool label** so two distinct checkpoints sharing class `MLAgent` stay separate for rating. **The mechanism that makes a mixed-model field rateable.** |
| `head_to_head` | `evaluate.py:238-322` | seat-order-cancelled A-vs-B → `HeadToHeadStats` (`a_win_rate` + Wilson CI). |
| `evaluate_ml` | `evaluate.py:132-…` | single-model eval vs an opponent factory; `learner_seat` (Sprint A). Not the league path, but the per-checkpoint smoke check. |
| `ml_agent_factory(path)` | `evaluate.py:74-110` | picklable factory loading an SB3 checkpoint for a league seat. |
| `strong_heuristic_agent_factory(strength=2)` | `evaluate.py:112-129` | picklable strong-heuristic factory (Sprint A). |
| `heuristic_agent_factory()` | `evaluate.py` runner import | picklable weak-heuristic factory. |
| `run_batch(track, seat_factories, *, num_games, parallel, seed)` | `runner.py:371-…` | runs N games on one track with fixed seat factories; returns `list[GameOutcome]` ordered by `game_index`; per-game seed derived from `(base_seed, game_index)` → deterministic, order-independent (`runner.py:198-202`). |
| `GameOutcome` | `runner.py:88-109` | `game_index`, `seed`, `num_players`, `winner_id`, `finish_order: tuple[int,…]` (best→worst), `players: tuple[PlayerOutcome,…]`. **Plain serializable dataclass — the per-game persistence record.** |
| `PlayerOutcome` | `runner.py:65-…` | `player_id`, `agent_type`, `finish_position`, … — carries the pool label after `_relabel_outcomes`. |
| `compute_elo` | `stats.py:421-486` | replays each game's full `finish_order` as pairwise wins; **seeded bootstrap CI** (`bootstrap`, `bootstrap_seed`); sorts by `game_index` → deterministic. Already N-way / rank-based. |
| `compute_trueskill` | `stats.py:528-…` | consumes full `finish_order` as a ranked multiplayer outcome; mu/sigma. Already rank-based multiplayer. |
| `wilson_interval(wins, games)` | `stats.py:319-344` | Wilson CI; `games==0 -> (0,1)`. |
| `EloRating` / `TrueSkillRating` / `RoundRobinStats` / `HeadToHeadStats` | `stats.py:134/151/176/111` | rating result dataclasses (`rating`+`rating_ci`+`games`; `mu`+`sigma`; tables). |
| `save_checkpoint(...)` | `training.py:139-193` | SB3 zip + `.meta.json` sidecar recording `obs_dim`/`action_dim`/`codec_version` **plus** `git_sha` (`_git_sha()`), `seed`, `ppo_config`, `curriculum_config`, `track_config`, `normalize` (`training.py:171-184`). **This sidecar IS the per-run manifest backbone.** |
| `load_meta(path)` | `training.py:196-…` | reads a checkpoint sidecar back. **The manifest reader for the league + attribution.** |
| `PPOConfig` | `model.py:47-110` | `net_arch`/`features_extractor_hidden`/`features_dim` (net-size knobs), `share_features_extractor=False`, `gamma=0.999`, `shaping_weight=0.0`, `seed`, `device`, `n_envs`. **The per-run hyperparameter object the sweep varies.** |
| `CurriculumConfig` | `training.py:314-373+` | gate/curriculum knobs: `total_timesteps`/`phase1_steps`, `randomize_seat` (Sprint A), `use_strong_heuristic_opponents`, `broaden_phase1_mix`, `gate_use_wilson_lb`, `gate_games`, `normalize_obs/normalize_reward`, etc. **The per-run recipe object the sweep varies.** |
| `sprint_a_curriculum` / `sprint_b_curriculum` | `training.py:508/556` | existing presets the sweep can branch from. |
| `League` / `LeagueEntry` | `league.py:44-370` | 6D PFSP *training* league (opponent sampler). **Not** the evaluator; not used by 8D. |
| `experiments/` scripts | `proto_solo.py`, `proto_finetune.py`, `diag_ablate.py`, `diag_canlearn.py` | the existing pattern: heavy-compute throwaway harnesses live here. `diag_ablate.py` = the closest prior art (toggle one factor, eval). 8D's campaign runner is a sibling. |
| Top-level run script | `train_and_eval_run.py` | the existing "train then eval" entrypoint pattern the campaign launcher mirrors. |

### 2.2 8C deliverables 8D consumes (verified **not yet in tree**)

| Symbol | 8C section | Status (confirmed by grep against current tree) |
|---|---|---|
| `TrainingPhase` (dataclass: `name`, `num_players`, `reward_mode`, `gamma`, `shaping_weight`, `steps`, `pool_kind`) | 8C §6.1 | **not present** in `training.py`. 8D parameterizes a *list* of these. |
| `OpponentSchedule` / `OpponentStage` | 8C §5.2 | **not present**. 8D varies the pool composition through it. |
| `train_self_play(..., warm_start_path=None, phases=None)` | 8C §6.2 | `train_self_play` exists (`training.py:1029`); the `warm_start_path`/`phases` params are 8C additions **not yet present**. |
| `sprint_8c_curriculum(...)` preset | 8C §8.1 | **not present** (`grep` returns only `sprint_a_curriculum`/`sprint_b_curriculum`). |
| `evaluate_league(contenders, *, tracks, num_players=4, games_per_matchup=50, mode, rating, seed) -> LeagueLadder` | 8C §7.5 | **not present** in `evaluate.py`. **8D requires a matchup-sampling extension + per-game-outcome export — see §12.** |
| `LeagueLadder` (dataclass: `ratings`, `trueskill`, `pairwise`) | 8C §7.5 | **not present** in `stats.py`/`evaluate.py`. |

**Implication:** 8D cannot be implemented until 8C lands. The 8D module is written
against 8C's *interfaces*; this doc treats them as fixed and only proposes
*additions* (§12), never redesigns.

---

## 3. Architecture decisions up front (what I chose vs left open)

**Chosen (baked into this design):**

- **Orchestration is a thin, tested `src/heat/ml/sweep.py` module; heavy compute
  is `experiments/` scripts.** The declarative spec → config expansion, the run
  manifest read/write, the matchup-sampling plan, and the attribution math are
  pure/deterministic and **unit-tested**. The two scripts that actually burn GPU
  (`experiments/run_campaign.py`, `experiments/run_league.py`) are thin shells that
  call the tested module + `train_self_play` / `evaluate_league`. This matches the
  repo's existing split (`experiments/` for compute, `src/` + `tests/` for logic)
  and the user's explicit recommendation.
- **The run manifest is the checkpoint sidecar + a campaign index.** `save_checkpoint`
  already writes per-run `git_sha`/`seed`/`ppo_config`/`curriculum_config`/`track_config`
  (`training.py:171-184`). 8D adds a campaign-level `manifest.jsonl` (one row per
  run: `run_id`, the swept factor values, checkpoint path, sidecar path, status,
  wall-clock, final gate score) that *indexes* those sidecars. No new metadata
  store — the sidecar is the source of truth; the manifest is a join key + the
  factor vector for attribution.
- **Sequential run execution on the single-GPU box** (one `train_self_play` at a
  time), relying on the **existing intra-run `n_envs` CPU parallelism** (`PPOConfig.n_envs`,
  `model.py:108-110`) for throughput. Two SB3 models contending for one GPU is net
  slower and risks OOM; the campaign is embarrassingly parallel only across *runs*,
  which we serialize. (Multi-GPU / multi-box is an Open Question, §11, not this
  campaign.)
- **Failure isolation per run.** Each run is a try/except around `train_self_play`;
  a crash marks that manifest row `failed` (with the exception text) and the
  campaign continues. The manifest is append-only (`jsonl`), flushed after every
  run, so a mid-campaign kill loses at most the in-flight run. **Resumability:** on
  restart, skip `run_id`s already `done` in the manifest.
- **League = free-for-all with matchup *sampling* (not exhaustive), rank-based
  rating with bootstrap CIs, per-game outcomes persisted.** Exhaustive C(n,4)
  enumeration is infeasible for tens of models (§7.1); we sample fields so each
  contender appears in ≥K fields, rotate seats for fairness, and store every
  `GameOutcome` to `league_games.jsonl` so `compute_elo`/`compute_trueskill` can be
  re-run with different ratings/bootstrap settings **without replaying a single
  race**.
- **Attribution = grouped diffs *and* a simple OLS, both reported with caveats.**
  Headline is per-factor grouped mean-rating-with-CI (robust, assumption-light);
  the OLS of rating on one-hot/numeric factors is a secondary view that estimates
  marginal effects *holding others fixed* — explicitly flagged as observational and
  confounded by interactions and the non-orthogonality of a sampled grid.

**Left open (flagged in §11, decided at build time):**

- Grid vs random/Latin-hypercube sampling of the factor space, and which factors
  are *primary* (swept) vs *fixed* for the first campaign.
- Exhaustive vs sampled matchups already chosen (sampled); the open part is K
  (fields per contender) and the target CI width.
- Whether the **headline** league is cross-checkpoint (swept models vs each other,
  heuristics as anchors only) vs purely-vs-fixed-references.
- Attribution statistical method weighting (grouped diffs as headline vs OLS) and
  how aggressively to caveat confounding.

---

## 4. The sweep specification

### 4.1 What gets varied (the factor set, grounded in the learnings doc)

`docs/ml-learnings-solo-pretrain.md` §5-7 names the factors worth sweeping. Mapped
to the **current** config surface + 8C's additions:

| Factor | Knob (where it lives) | 8C/A/B status | Default-campaign treatment |
|---|---|---|---|
| Solo step count | `phases[0].steps` (`TrainingPhase`, 8C §6.1) | 8C | **primary** (2 levels) |
| Per-stage step ratios / budgets | `phases[i].steps` across the weak/mixed/strong phases | 8C | **primary** (2 ratio profiles) |
| Per-phase gamma | `phases[i].gamma` (8C §6.3 gamma handoff) | 8C | **fixed** at the recipe default (0.99 solo / 0.999 race) for campaign 1 — the gamma handoff is the riskiest 8C mechanism; don't sweep it until it's proven |
| Opponent-pool composition | `OpponentSchedule` / `pool_kind` (8C §5.2) | 8C | **primary** (2: weak→mixed→strong vs weak→strong) |
| Shaping weight | `PPOConfig.shaping_weight` (`model.py:90`) / `phases[i].shaping_weight` | exists | **secondary** (2 levels) |
| `randomize_seat` | `CurriculumConfig.randomize_seat` (Sprint A) | exists | **fixed ON** (proven win, `ml-learnings §3`) — kept as a *control* knob, swept only to re-confirm if budget allows |
| Net-size profile | `PPOConfig.net_arch` / `features_extractor_hidden` / `features_dim` (`model.py:59-65`) | exists | **secondary** (2: default vs larger) |
| Seed | `PPOConfig.seed` (`model.py:101`) | exists | **replication** (≥2 seeds per cell so within-cell variance is estimable) |

**Primary-vs-secondary** matters because a full grid over all eight factors is far
too large (§6 budget). Campaign 1 sweeps ~3 primary factors × ~2 secondary × ≥2
seeds; the rest are pinned at the validated recipe.

### 4.2 The declarative spec (a `SweepSpec` in `src/heat/ml/sweep.py`)

A frozen dataclass describing the factor grid + sampling policy. **Data only** — no
model, no env — so it is trivially serializable, diffable, and unit-testable.

```python
# src/heat/ml/sweep.py  (NEW, thin + tested)

@dataclass(frozen=True)
class FactorAxis:
    """One swept factor: a name + the discrete levels to try.

    `name` is a dotted path into the per-run config the expander knows how to
    apply (e.g. "phases.0.steps", "ppo.shaping_weight", "schedule.pool_kinds").
    Continuous factors are pre-discretized to levels so the grid is finite and
    the attribution (§9) has clean groups.
    """
    name: str
    levels: tuple  # e.g. (200_000, 400_000)

@dataclass(frozen=True)
class SweepSpec:
    """Declarative factor grid + sampling policy for a campaign."""
    axes: tuple[FactorAxis, ...]
    seeds: tuple[int, ...]              # replication within every cell
    method: str = "grid"               # "grid" | "random" | "lhs"
    max_configs: int | None = None     # cap for "random"/"lhs"
    base_preset: str = "sprint_8c"     # which curriculum preset to branch from

    def expand(self) -> "list[RunConfig]":
        """Materialize the (possibly sampled) list of RunConfigs.

        grid: Cartesian product of all axis levels x seeds.
        random/lhs: draw `max_configs` cells (seeded) so a huge factorial space
        is covered cheaply; LHS spreads draws across each axis's marginal.
        Deterministic given the spec (the sampler is seeded by a spec hash).
        """
```

```python
@dataclass(frozen=True)
class RunConfig:
    """One fully-resolved run: the factor vector + the concrete configs to train.

    `factors` is the {axis_name: level} dict (the attribution design matrix row).
    `ppo` / `curriculum` / `phases` / `schedule` are the resolved objects passed
    straight to `train_self_play`. `run_id` is a deterministic hash of `factors`
    (stable across campaign restarts -> resumability).
    """
    run_id: str
    factors: dict           # {axis_name: level}  -> attribution row
    ppo: "PPOConfig"
    curriculum: "CurriculumConfig"
    phases: "list[TrainingPhase]"      # 8C
    schedule: "OpponentSchedule"       # 8C
```

**Grid vs random/LHS (Open Question §11, but a recommendation):** start **grid**
for campaign 1 — a small, fully-crossed design makes the grouped-diff attribution
clean (every level appears with every other, so marginal means are unconfounded by
*design imbalance*). Move to **LHS** only when the factorial blows past the compute
budget; LHS covers a high-dimensional space with few runs but yields an unbalanced
design that complicates grouped diffs and pushes attribution toward the regression
(§9.2).

### 4.3 Config expansion (applying a factor vector to the 8C recipe)

`expand()` starts from the `base_preset` (`sprint_8c_curriculum()`, 8C §8.1) and
overlays each axis level onto a copy via `dataclasses.replace` (for `PPOConfig`/
`CurriculumConfig`) and list edits (for `phases`). The dotted-path applier is a
small tested switch — the *only* place 8D reaches into the 8C config shape, so if
8C renames a field, exactly one function changes.

> **No core change.** `expand()` produces standard `PPOConfig`/`CurriculumConfig`/
> `TrainingPhase`/`OpponentSchedule` objects. `train_self_play` already accepts
> them (8C). 8D adds no field to any of them.

---

## 5. The campaign harness (orchestration + manifest)

### 5.1 The run loop (sequential, isolated, resumable)

`experiments/run_campaign.py` (thin shell) drives the tested
`run_campaign(spec, *, out_dir, ...)` in `sweep.py`:

```python
def run_campaign(spec: SweepSpec, *, out_dir: str, held_out_seeds, resume=True):
    manifest = ManifestWriter(os.path.join(out_dir, "manifest.jsonl"))  # append-only
    done = manifest.completed_run_ids() if resume else set()
    for rc in spec.expand():
        if rc.run_id in done:
            continue                          # resumability (§3)
        manifest.mark_started(rc)
        try:
            t0 = time.time()
            model, best_path = train_self_play(       # 8C entrypoint, unchanged
                rc.ppo_as_config(), rc.curriculum,
                phases=rc.phases, warm_start_path=None,
            )
            # best_path's sidecar already has git_sha/seed/ppo/curriculum (training.py:171)
            manifest.mark_done(rc, checkpoint=best_path,
                               wall_clock=time.time() - t0,
                               gate_score=_read_gate_from_meta(best_path))
        except Exception as exc:                       # failure isolation (§3)
            manifest.mark_failed(rc, error=repr(exc))
            continue
```

- **Sequential**: exactly one `train_self_play` in flight; intra-run throughput is
  the existing `n_envs` SubprocVecEnv parallelism (`model.py:108-110`).
- **Isolation**: one run's exception never aborts the campaign; it is recorded and
  skipped.
- **Resumable**: `run_id` is a deterministic hash of the factor vector, so a
  restart with the same `SweepSpec` skips completed runs by reading the manifest.
- **Per-run checkpoint dir**: set `CurriculumConfig.run_name = f"campaign_{run_id}"`
  and a per-run `checkpoint_dir` so runs don't clobber each other's `best`/`final`
  checkpoints (`training.py:343-345` names them off `run_name`).

### 5.2 The manifest schema (`manifest.jsonl`, one row per run)

```json
{
  "run_id": "a3f9c1",
  "status": "done",                 // started | done | failed
  "factors": {"phases.0.steps": 400000, "schedule.pool_kinds": "weak,mixed,strong",
              "ppo.shaping_weight": 0.05, "seed": 0},
  "checkpoint": "checkpoints/campaign_a3f9c1/heat_ppo.zip",
  "meta": "checkpoints/campaign_a3f9c1/heat_ppo.meta.json",  // save_checkpoint sidecar
  "git_sha": "291d5c2",             // mirrored from the sidecar for convenience
  "gate_score": 0.84,               // final promote_score (Wilson LB) from the run
  "wall_clock_s": 612.4,
  "error": null
}
```

The `meta` sidecar (`save_checkpoint`, `training.py:171-184`) is the authoritative
record of *how* the run was configured (full `ppo_config`/`curriculum_config`/
`git_sha`/`seed`); the manifest row is the **join key + the factor vector** the
attribution consumes, plus campaign-level status/timing the sidecar doesn't carry.

> **Reproducibility (req 5):** every run's git SHA, seed, and full config are
> captured by the existing sidecar — **no new code** to record provenance, only the
> campaign-level index. Confirmed `_git_sha()` is called in `save_checkpoint`
> (`training.py:178`).

---

## 6. Compute budgeting (made explicit)

Per `ml-learnings-solo-pretrain.md` §5, the validated 1.2M-step recipe ran in
**~16 min** *without* the gate; the §6 caveat warns the in-training Wilson-LB gate
adds wall-clock (the ~54-min baselines spent ~20 min on gate games). So budget a
gated run at **~25-40 min** (gate cadence tunable, 8C §10).

**Default campaign (fits a few GPU-hours):**

| Quantity | Value | Note |
|---|---:|---|
| Primary factors | 3 × 2 levels = 8 cells | solo-steps, step-ratio, pool-composition |
| Secondary | fold into 2 cells (not full-crossed) | shaping-weight, net-size paired |
| Seeds / cell | 2 | within-cell variance for attribution |
| **Runs** | **~16-24** | `8 × 2 (seed)` primary + a few secondary cells |
| Per-run cost | ~30 min gated | from learnings §5-6 |
| **Training total** | **~8-12 GPU-hours** | sequential on one GPU |
| League contenders | ~16-24 ckpts + 3 refs + 2 anchors ≈ 25 | |
| League games | K=30 fields/contender × ~25 ÷ 4 seats × seat-rotations ≈ **~5-9k games** | §7.3 sizing |
| League cost | ~0.5-1.5 GPU-hours | games are cheap vs training; `ml_agent_factory` reloads per worker so `parallel=False` (sequential) is the safe default (`evaluate.py:426-429`) |

**Scale knob:** `SweepSpec.max_configs` (and `seeds`) scales the training cost
linearly; `games_per_contender` (K, §7) scales the league cost. A 2× campaign is
~20-24 GPU-hours. The doc ships the small default and the knob.

---

## 7. The large round-robin league with matchup sampling

### 7.1 The combinatorics problem (why exhaustive fails)

8C's `evaluate_league` (free-for-all, 8C §7.3) plays size-`num_players` fields. For
`n` contenders and 4 seats, exhaustive enumeration is **C(n,4)** distinct fields:
n=25 → **12 650** fields × `games_per_matchup` × seat-rotations. At even 10 games ×
12 rotations that is ~1.5M games — infeasible. The cost is super-linear in `n`
exactly where 8D wants tens of models. **Exhaustive enumeration does not scale; we
sample fields.**

### 7.2 Matchup sampling design (each contender in ≥K fields, seat-rotated)

Replace exhaustive C(n,4) with a **sampled, balanced** field set:

- **Coverage target:** every contender appears in **≥K** distinct fields (K a knob;
  default 30). Build fields by repeatedly drawing 4 distinct contenders without
  replacement-bias, tracking per-contender appearance counts and preferring
  under-represented contenders, until all reach K. This is a balanced-incomplete-
  block-style sampler — deterministic given the seed.
- **Seat-rotation fairness:** for each sampled field of 4 models, run the game once
  per cyclic seat rotation (4 rotations) — or a fixed Latin-square subset — so every
  model occupies every grid slot equally (8C §7.4). This cancels the first-mover
  advantage `head_to_head` already corrects for, but in the 4-way setting.
- **Track sweep:** play each field-rotation across the **pinned held-out track set**
  (§8) so ratings aggregate across tracks, not one lucky layout.
- **Rating convergence / CI:** the per-contender game count is `K × 4 (rotations) ×
  |held_out_tracks| × games_per_field`. Pick K and `games_per_field` so the
  bootstrap CI on the headline rating (`compute_elo`'s seeded bootstrap,
  `stats.py:438-441`, or TrueSkill sigma) is below a target width (Open Question
  §11 — e.g. ELO CI half-width < ~30, or TrueSkill sigma < ~1.0). The store-and-
  recompute design (§7.4) lets us *measure* the CI and add games only where it's
  still wide, instead of guessing up front.

### 7.3 Default sizing (the small campaign)

n≈25, K=30 fields/contender → ~`25×30/4 ≈ 190` unique fields × 4 seat-rotations ×
~4 held-out tracks × ~2 games/field ≈ **~6k games** (the §6 estimate). TrueSkill
sigma shrinks with games; the bootstrap ELO CI is read off directly. If a
contender's CI is still wide after the first pass, draw more fields containing it
(the sampler is incremental).

### 7.4 Persist per-game outcomes → recompute ratings without replaying

The headline reproducibility/efficiency move. `evaluate_league` (8C) currently
returns only the `LeagueLadder`. 8D needs the **raw per-game outcomes** persisted so
ratings recompute under different settings (ELO vs TrueSkill, different bootstrap
seed, dropping a contender) for free:

- After relabelling (`_relabel_outcomes`, `evaluate.py:217`), serialize every
  `GameOutcome` (a plain dataclass: `game_index`, `seed`, `num_players`,
  `winner_id`, `finish_order`, `players[].agent_type/finish_position`,
  `runner.py:88-109`) to `league_games.jsonl` — one line per game, plus the field's
  track name and the contender labels.
- **Recompute path:** `compute_elo(outcomes, ...)` / `compute_trueskill(outcomes,
  ...)` are **pure functions of the outcome list** (sorted by `game_index`,
  `stats.py:452/555`). So re-rating is `load jsonl -> compute_elo` with zero races
  replayed. This is what makes "tens of models, several rating views, CI tuning"
  cheap.

> **§12 API addition required:** `evaluate_league` must (a) accept a
> matchup-sampling spec (K / `games_per_field` / sampler seed) rather than only
> exhaustive enumeration, and (b) **export the per-game `GameOutcome` list** (return
> it or write the jsonl), not just the `LeagueLadder`. These are additive — see §12.

### 7.5 Contenders: swept checkpoints + references + anchors

```python
contenders = {
    # swept models -- the headline (cross-checkpoint free-for-all)
    **{rc.run_id: ml_agent_factory(row.checkpoint)
       for rc, row in manifest.done_rows()},
    # reference yardsticks (anchors, NOT the headline)
    "weak_heuristic":   heuristic_agent_factory(),
    "strong_heuristic": strong_heuristic_agent_factory(strength=2),
    "strong_heuristic3":strong_heuristic_agent_factory(strength=3),
    # baseline/prototype anchors so ratings are interpretable across campaigns
    "baseline_sprintB": ml_agent_factory(BASELINE_B_CKPT),   # if available
}
```

All factories are picklable (`ml_agent_factory`/`*_heuristic_agent_factory`,
`evaluate.py:74/112`). **Headline question (Open §11):** include the swept models
*against each other* (free-for-all fields of distinct checkpoints) as the headline,
with heuristics only as **anchors** — recommended, because the inter-checkpoint
ranking is exactly "what each factor brings," and anchoring to fixed references
keeps ratings comparable across campaigns. The pure-vs-fixed-reference view is a
secondary, cheaper read.

---

## 8. The pinned held-out track set (reproducibility)

The league must run on tracks **distinct from any used in training** so a swept
model can't be rewarded for memorizing its training distribution. Reuse the
existing held-out machinery:

- Sprint A pins held-out generated tracks via `_HOLDOUT_TRACK_SEEDS` for the gate
  (referenced in `sprint-A` §2, `training.py` `_gate_tracks`). 8D pins a **separate,
  larger** league track set (e.g. a fixed list of `generate_track(seed)` for a
  frozen seed list, plus 1-2 named tracks like `usa` as human-legible anchors).
- The track set is a **frozen constant** in the campaign config (its seed list is
  recorded in the league artifact) so any rerun — or a future campaign — ranks on
  the identical tracks. `evaluate_league` already takes `tracks: list[Track]` (8C
  §7.5, mirroring `round_robin_elo`'s `tracks` param, `evaluate.py:403`).

> **Confounding caveat:** if the *training* track distribution is itself a swept
> factor (it isn't in campaign 1 — `use_track_curriculum` is off, 8C §8.1), the
> held-out set must be held *truly* out for every arm. Flagged in §11.

---

## 9. Attribution — "what each factor brings"

### 9.1 Inputs

A tidy table joining the manifest's factor vectors to the ladder's ratings:

| run_id | solo_steps | step_ratio | pool_comp | shaping_w | net_size | seed | rating | rating_ci_lo | rating_ci_hi |
|---|---|---|---|---|---|---|---|---|---|

`rating` is the headline TrueSkill mu (or ELO point) for that run's checkpoint label
from the `LeagueLadder`; the CI is the bootstrap/sigma band.

### 9.2 Two estimators (both reported, with caveats)

**(a) Grouped marginal diffs (headline — assumption-light).** For each factor, group
runs by its level and report the mean rating per level **with a CI** (across the runs
in that group, which span all *other* factor combinations + seeds). The "effect" of
moving a factor from level L0→L1 is the difference of group means; its uncertainty
combines the per-group spread and the per-run rating CI. In a **balanced grid**
(§4.2) every level co-occurs equally with every other level, so these marginal means
are not confounded *by design imbalance* (they still average over interactions —
caveat below).

**(b) OLS of rating on factors (secondary — estimates "holding others fixed").** Fit
`rating ~ C(solo_steps) + C(step_ratio) + C(pool_comp) + shaping_w + net_size + seed`
(one-hot the categoricals). Coefficients estimate each factor's marginal effect
holding the others fixed; report coefficient ± standard error. **Weighted** by
inverse rating-variance if CIs vary a lot. This is the cleaner "marginal effect"
read but leans on linearity/additivity.

### 9.3 Honesty (baked into the report text, per the user's explicit instruction)

The generated report **must** state, in prose:

- **Observational, not causal.** We swept a grid and *observed* ratings; we did not
  randomize at the unit level beyond seeds. Factor effects are associations under
  *this* recipe and track set.
- **Ratings have CIs.** Every effect is reported with uncertainty; differences inside
  overlapping CIs are "not distinguishable," not "zero."
- **Factors interact.** A marginal effect averages over the other factors' levels; the
  grouped diff and the OLS can disagree when interactions are strong — when they do,
  the report flags it and shows the relevant 2-way cell means rather than a single
  marginal number.
- **Sampled-grid imbalance.** If LHS/random sampling (§4.2) was used, the design is
  unbalanced and the grouped diffs inherit that imbalance — the OLS becomes the
  primary read and the report says so.

### 9.4 Outputs (human- + machine-readable)

- `attribution.md` — the ranked ladder table (rating + CI), a per-factor
  effects table (grouped diff + OLS coef, both with uncertainty), the 2-way cells for
  any flagged interaction, and the honesty prose (§9.3).
- `ladder.csv` — every contender's rating/CI/games.
- `effects.json` — `{factor: {level: mean_rating_ci, ols_coef_se}}`, machine-readable
  for any downstream plotting.
- `league_games.jsonl` — the raw per-game outcomes (§7.4), the recompute substrate.
- `manifest.jsonl` — the run index (§5.2).

These are **artifacts written by the `experiments/` script**, not committed; the
`src/sweep.py` functions that *compute* the tables (grouped diffs, OLS, ladder join)
are pure and unit-tested.

---

## 10. Test plan

`PYTHONPATH=src python -m pytest tests/ -q`. New `tests/test_sweep.py` (+ a small
extension to `tests/test_ml_eval.py`/`test_league.py` for the sampling + export).
All tests are **fast** — they exercise the pure orchestration/analysis logic with
cheap scripted agents and tiny synthetic ladders; **no SB3 training, no real league
runs** (heavy compute stays in `experiments/`).

**Sweep spec + expansion (`test_sweep.py`):**
- `test_grid_expand_cardinality`: a `SweepSpec` with axes of sizes (2,3) and 2 seeds
  expands to exactly `2*3*2 = 12` `RunConfig`s; `run_id`s are unique.
- `test_run_id_deterministic`: same factor vector → same `run_id` across two
  `expand()` calls (resumability invariant).
- `test_random_lhs_respects_max_configs`: `method="random"`/`"lhs"` with
  `max_configs=N` yields exactly N configs, deterministically given the spec hash.
- `test_factor_applied_to_config`: a factor `"ppo.shaping_weight": 0.05` produces a
  `RunConfig.ppo` with `shaping_weight == 0.05` and **every other field equal** to the
  base preset (the dotted-path applier touches only its target — `dataclasses.replace`
  guard).
- `test_phases_factor_applied`: a `"phases.0.steps"` level edits only phase 0's
  `steps` in the resolved `phases` list (8C `TrainingPhase` shape).

**Manifest (`test_sweep.py`):**
- `test_manifest_roundtrip`: write started/done/failed rows → `completed_run_ids()`
  returns exactly the `done` ids; `failed` rows carry the error string.
- `test_resume_skips_done`: a `run_campaign` driven with a stubbed trainer (monkeypatched
  `train_self_play`) skips `run_id`s already `done` in a pre-seeded manifest.
- `test_failure_isolation`: a stubbed trainer that raises on one config marks that row
  `failed` and still processes the rest (campaign not aborted).

**Matchup sampling (`test_sweep.py` / `test_ml_eval.py`):**
- `test_each_contender_meets_K`: the field sampler gives every contender ≥K fields for
  n in {5, 12, 25}; fields are size-4 with distinct members.
- `test_sampling_deterministic`: same `(contenders, K, seed)` → identical field list.
- `test_seat_rotation_balanced`: across a field's rotations, every contender occupies
  every seat index equally (the fairness invariant).
- `test_far_fewer_than_exhaustive`: for n=25, the sampled field count `≪ C(25,4)`
  (asserts the scaling win is real).

**Per-game persistence + recompute (`test_ml_eval.py`):**
- `test_outcomes_jsonl_roundtrip`: serialize a list of `GameOutcome` to jsonl and
  reload; `compute_elo(reloaded) == compute_elo(original)` (byte-identical ratings —
  the "recompute without replay" guarantee).
- `test_recompute_changes_rating_system_only`: from one stored outcome set, `compute_elo`
  and `compute_trueskill` both run with no game replay (assert no `run_batch` call via
  a spy).

**Attribution (`test_sweep.py`):**
- `test_grouped_diff_on_synthetic`: a synthetic ladder where one factor adds a constant
  +R to every run at level L1 → the grouped diff recovers ~+R within its CI; an
  irrelevant factor's diff straddles 0.
- `test_ols_recovers_known_effect`: a synthetic additive design → OLS coefficients
  recover the planted per-factor effects within standard error.
- `test_report_states_caveats`: the generated `attribution.md` text contains the
  observational/interaction/CI caveats (§9.3) — a regression guard that honesty prose
  is not silently dropped.

**Contract guard:**
- `test_codec_unchanged_8d`: `OBS_DIM==104`, `ACTION_DIM==516`, `CODEC_VERSION==2`
  (8D must not touch the contract — it never edits `spaces.py`).

---

## 11. Open questions (decide at build time)

- **Grid vs sampled sweep; primary vs fixed factors.** Recommended: **grid** for
  campaign 1 (clean grouped-diff attribution), ~3 primary × ~2 secondary × ≥2 seeds;
  gamma + `randomize_seat` pinned. Switch to LHS only when the factorial exceeds the
  budget (then attribution leans on the OLS, §9.2). Confirm the primary set before
  launch.
- **Matchup K and CI-width target.** Sampling is chosen; the open part is K
  (fields/contender, default 30) and `games_per_field`, set so the headline rating CI
  is below a target width (e.g. ELO bootstrap half-width < ~30 or TrueSkill sigma <
  ~1.0). The store-and-recompute design (§7.4) lets us *measure* and top up
  incrementally rather than guess.
- **Headline league composition.** Cross-checkpoint free-for-all (swept models vs each
  other, heuristics as anchors) — recommended — vs purely-vs-fixed-references. The
  former directly answers "what each factor brings"; the latter is cheaper but less
  faithful.
- **Sequential vs parallel run execution.** Sequential on the single GPU is chosen
  (GPU contention/OOM risk). Multi-GPU / multi-box campaign parallelism is a future
  knob (the manifest's `run_id` partitioning already supports sharding runs across
  boxes), out of scope here.
- **Where orchestration lives.** Thin tested `src/heat/ml/sweep.py` (spec, manifest,
  sampler, attribution math) + heavy `experiments/run_campaign.py` /
  `run_league.py` — recommended and matches the repo split. Confirm the module name.
- **Attribution method weighting.** Grouped diffs as headline + OLS as secondary
  (recommended); how hard to caveat confounding (the report always carries the §9.3
  prose). Whether to add a rank-based effect estimate (e.g. effect on league *rank*
  rather than rating) for robustness.
- **Held-out track confounding.** If a later campaign sweeps the *training* track
  distribution, ensure the league set stays held-out for every arm (§8). Latent in
  campaign 1 (track curriculum off).
- **TrueSkill vs ELO as the attribution `rating`.** Both are produced; TrueSkill mu is
  the natural multiplayer headline, ELO gives the bootstrap CI directly. Pick one as
  the attribution response and report the other alongside.

---

## 12. 8C API additions required (fold back into the 8C doc)

8D is orchestration + analysis on top of 8C, but the league scale-up needs **two
additive** changes to 8C's `evaluate_league` (8C §7.5). Neither redesigns 8C's
behavior; both are backward-compatible defaults:

1. **Matchup sampling parameters on `evaluate_league`.** Today's 8C sketch enumerates
   fields (small-N). Add a sampling spec so it scales to tens of contenders:
   ```python
   def evaluate_league(
       contenders, *, tracks, num_players=4,
       games_per_matchup=50, mode="free_for_all", rating="trueskill", seed=0,
       # --- 8D additions (additive, defaults preserve 8C small-N behavior) ---
       matchup_sampling: str = "exhaustive",     # "exhaustive" | "sampled"
       fields_per_contender: int | None = None,  # K when sampled
       games_per_field: int | None = None,
   ) -> LeagueLadder: ...
   ```
   `matchup_sampling="exhaustive"` (default) = exactly 8C's behavior; `"sampled"`
   activates §7.2. **Recommendation:** implement the balanced field sampler in 8D's
   `sweep.py` and pass the resulting field list into a thin `evaluate_league` seam,
   so 8C's evaluator stays simple and the sampler is unit-tested in 8D.

2. **Per-game outcome export from `evaluate_league`.** 8C returns only `LeagueLadder`.
   8D needs the raw `GameOutcome` list to persist + recompute (§7.4). Add either a
   return field or a sink:
   ```python
   # option A: richer return
   @dataclass
   class LeagueLadder:
       ratings: dict[str, EloRating]
       trueskill: dict[str, TrueSkillRating] | None
       pairwise: dict[tuple[str, str], HeadToHeadStats] | None
       outcomes: list[GameOutcome] | None = None   # 8D addition (default None)
   # option B: evaluate_league(..., outcomes_path: str | None = None)  # writes jsonl
   ```
   Option A (carry the relabelled outcomes on the ladder) is cleaner and keeps
   `evaluate_league` pure; 8D's script writes the jsonl. **Recommended: A.**

If 8C is implemented before these are folded in, 8D can still work by **bypassing
`evaluate_league`** and calling its primitives directly (`run_batch` →
`_relabel_outcomes` → `compute_elo`/`compute_trueskill`, all of which exist today,
§2.1) — but that duplicates seating/relabel logic. Folding the two additions into
8C is the clean path.

**No other core change.** The sweep, manifest, sampler, attribution, and
persistence all live in 8D's new `sweep.py` + `experiments/` scripts; `train_self_play`
(8C), `save_checkpoint`, the rating math, and `run_batch` are consumed unchanged.

---

## 13. Drift corrected vs the user's framing / the 8C doc

- **8C is not yet in the tree.** The user framed 8D as building on 8C's delivered
  `TrainingPhase`/`OpponentSchedule`/`sprint_8c_curriculum`/`evaluate_league`/
  `LeagueLadder`. Verified by grep: **none of these exist yet** (`evaluate.py`,
  `training.py`, `stats.py` have only the pre-8C primitives). 8C is design-only
  (its own header says "no code yet"). **8D therefore hard-depends on 8C landing
  first** (§14) and is written against 8C's *interfaces*, treating them as fixed.
- **`evaluate_league` as sketched is exhaustive/small-N.** The user's scope says
  "scale 8C's `evaluate_league`," but 8C's §7.5 sketch enumerates fields and is
  explicitly "small-N this sprint." Scaling to tens of models is **not** a parameter
  turn — it needs matchup *sampling* + per-game export, which are genuine API
  additions (§12), not a config bump. Flagged rather than silently assumed.
- **The manifest already half-exists.** "manifest with git SHA per run" reads like
  new infrastructure, but `save_checkpoint` already writes `git_sha`/`seed`/full
  configs to the `.meta.json` sidecar (`training.py:171-184`, `_git_sha()` at
  `:178`). 8D's manifest is a thin *index* over those sidecars + campaign status,
  not a new provenance store.
- **Rating math is already N-way / rank-based and pure.** "Free-for-all … ELO vs
  TrueSkill … bootstrap CIs" reads as open design, but `compute_elo` (with seeded
  bootstrap CI, `stats.py:421-486`) and `compute_trueskill` (`stats.py:528+`) already
  ingest full `finish_order` and are pure functions of the outcome list — so the
  "persist outcomes, recompute ratings without replay" design (§7.4) is essentially
  free, and the only genuinely new league code is the **balanced field sampler +
  seat rotation** (§7.2), not the rating engine.
- **`round_robin_elo` cannot be the league as-is.** It fills >2 seats by **cycling
  the pair** (`evaluate.py:358-364`), so it never produces a 4-distinct-model field —
  confirming 8C's choice of a new free-for-all seater, and 8D's matchup sampler must
  build *distinct-4* fields, not call `round_robin_elo`.
- **`run_batch`/`GameOutcome` are directly persistable.** `GameOutcome`
  (`runner.py:88-109`) is a flat dataclass (`finish_order`, `players[].agent_type`),
  so the jsonl per-game store (§7.4) is a plain serialize — no new outcome type.
- **Sequential, not parallel, on one GPU.** The user lists "sequential vs parallel"
  as open; given one GPU + `parallel=False` being the *documented* safe default for
  an MLAgent pool (`evaluate.py:426-429`, "reloads its SB3 model per worker"), 8D
  bakes **sequential runs + sequential league** and leaves cross-box sharding as the
  future knob.
- **All other references verified against the current tree:** `evaluate.py:74/112/
  217-235/238-322/325-395/398-497`; `runner.py:65/88-109/198-202/371`; `stats.py:111/
  134/151/176/319-344/421-486/528`; `training.py:139-193/196/314-373/343-345/508/556/
  1029`; `model.py:47-110/59-65/90/101/108-110`; `league.py:44-370`. The 8C symbols in
  §2.2 are confirmed **absent** (8C deliverables).

---

## 14. Task checklist (in order)

> **Prerequisite:** Sprint 8C is implemented and merged (its `TrainingPhase`,
> `OpponentSchedule`, `train_self_play(phases=, warm_start_path=)`,
> `sprint_8c_curriculum`, `evaluate_league`, `LeagueLadder` exist). 8D **blocks on
> this** (§13).

1. **`evaluate.py` (8C seam, §12)** — add `matchup_sampling`/`fields_per_contender`/
   `games_per_field` params (default = exhaustive 8C behavior) and per-game outcome
   export on `LeagueLadder` (`outcomes` field). Fold these into the 8C doc.
2. **`src/heat/ml/sweep.py` (NEW)** — `FactorAxis`, `SweepSpec.expand()` (grid/random/
   LHS), `RunConfig`, the dotted-path config applier over `PPOConfig`/`CurriculumConfig`/
   `phases`/`OpponentSchedule`. Pure + tested. (§4)
3. **`src/heat/ml/sweep.py`** — `ManifestWriter`/`ManifestReader` (`jsonl`, started/done/
   failed, `completed_run_ids`), and `run_campaign(spec, ...)` (sequential, isolated,
   resumable) wrapping `train_self_play`. (§5)
4. **`src/heat/ml/sweep.py`** — the **balanced field sampler** (each contender ≥K,
   distinct-4 fields, seeded) + seat-rotation helper; the per-game `GameOutcome`
   jsonl writer/reader. (§7.2/§7.4)
5. **`src/heat/ml/sweep.py`** — the attribution layer: ladder×manifest join, grouped
   diffs, OLS, the `attribution.md`/`ladder.csv`/`effects.json` emitters with the
   §9.3 honesty prose. (§9)
6. **`experiments/run_campaign.py` (NEW, thin)** — define the default `SweepSpec`
   (§4.1/§6), the per-run checkpoint dirs, and call `run_campaign`. Heavy compute.
7. **`experiments/run_league.py` (NEW, thin)** — read the manifest, build the contender
   map (swept ckpts + references + anchors), pin the held-out track set (§8), call
   `evaluate_league(..., matchup_sampling="sampled", ...)`, persist `league_games.jsonl`,
   and run the attribution emitters. Heavy compute.
8. **Tests** — all of §10 (`tests/test_sweep.py` + extensions to `test_ml_eval.py`/
   `test_league.py`); fast, no training, no real league.
9. **Full suite green:** `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 15. Definition of done

- (a) **Declarative sweep.** A `SweepSpec` expands (grid or sampled) to a
  deterministic list of `RunConfig`s over 8C's recipe knobs; each `run_id` is stable
  (resumability). Expansion + applier unit-tested; no core-config field added.
- (b) **Campaign harness.** `run_campaign` runs configs **sequentially**, isolates
  per-run failures (one crash never aborts the campaign), and **resumes** by skipping
  `done` rows. Failure-isolation + resume tests green.
- (c) **Run manifest.** `manifest.jsonl` indexes every run's factor vector +
  checkpoint + status + wall-clock, joined to the existing `save_checkpoint` sidecar
  (git SHA / seed / full config — no new provenance code).
- (d) **Large league with matchup sampling.** Every contender plays ≥K distinct-4
  fields, seat-rotated, on the **pinned held-out track set**, producing rank-based
  ratings (TrueSkill/ELO) with bootstrap CIs; the field count is `≪ C(n,4)` (sampling
  win tested). Sampler is deterministic given the seed.
- (e) **Per-game persistence + recompute.** Every `GameOutcome` is stored to
  `league_games.jsonl`; ratings recompute from the store with **no race replayed**
  (`compute_elo(reloaded) == compute_elo(original)` test green).
- (f) **Attribution.** Grouped-diff + OLS per-factor effects with uncertainty, emitted
  as `attribution.md` (with the observational/interaction/CI honesty prose) +
  `ladder.csv` + `effects.json`. Synthetic-effect recovery tests green; honesty-prose
  guard green.
- (g) **Compute budget.** A documented default campaign fitting a few GPU-hours (~16-24
  runs) + scale knobs (`max_configs`, `seeds`, K, `games_per_field`).
- (h) **Contract unchanged:** `OBS_DIM==104`, `ACTION_DIM==516`, `CODEC_VERSION==2`;
  the only 8C-core touch is the two additive `evaluate_league` changes (§12), which
  default to 8C's existing behavior.
- (i) **Orchestration is thin + tested in `src/`; heavy compute is `experiments/`.**
- (j) Full suite green: `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 16. Ordering & dependencies

- **Hard dependency on Sprint 8C** (§13/§14): 8D consumes 8C's `TrainingPhase`/
  `OpponentSchedule`/`train_self_play(phases=, warm_start_path=)`/`sprint_8c_curriculum`/
  `evaluate_league`/`LeagueLadder`, none of which exist yet. 8D is unbuildable until
  8C lands.
- **Two additive 8C-API changes** (matchup sampling + per-game export on
  `evaluate_league`, §12) should be folded into the 8C doc/implementation; if 8C ships
  without them, 8D can fall back to calling `run_batch`/`compute_elo` directly (all
  present today) at the cost of duplicating seating/relabel logic.
- **No obs/action/codec dependency.** 8D never edits `spaces.py`/`features.py`/the
  action codec; it is pure orchestration + analysis over the existing rating
  primitives and the 8C recipe.
