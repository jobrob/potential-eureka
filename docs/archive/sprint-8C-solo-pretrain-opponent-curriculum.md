# Sprint 8C — Solo pretrain + opponent curriculum (production) + round-robin league evaluator (implementation design)

> **Status:** design only — **no code yet**. Productionizes the validated recipe
> in `docs/ml-learnings-solo-pretrain.md` §7 ("Seed for the next sprint") and the
> two throwaway prototypes `experiments/proto_solo.py` /
> `experiments/proto_finetune.py`.
> **Scope:** (1) solo reward as a first-class reward *mode* in `spaces.step_reward`;
> (2) an `OpponentSchedule` that ramps the *opponent* pool weak→strong via staged
> vec-env rebuilds (generalizing Sprint B's track-curriculum machinery to the
> opponent axis); (3) multi-phase chaining + warm-start in `train_self_play`, with
> Sprint A's Wilson-LB gate flowing across all phases; (4) a small-N **round-robin
> league evaluator** producing ratings; (5) one preset + launch script replacing
> `proto_*`.
> **No obs/action contract change:** `OBS_DIM` (104), `ACTION_DIM` (516), and
> `CODEC_VERSION` (2) are untouched — confirmed `spaces.py:38/46/150`. This sprint
> changes only the *reward* (a new solo terminal mode) and the *training schedule*.
> **Depends on nothing new:** Sprint A + B are merged (verified below). Sprint 8D
> (mass sweep + league analysis) builds ON TOP of 8C's parameterized recipe + the
> league evaluator.

---

## 1. Goal

Turn the prototype recipe into first-class machinery so the same run can be
launched, gated honestly, and compared via a league:

1. **Dense solo pretrain** as a real reward mode (`num_players=1`, dense progress
   + terminal finish bonus, `gamma<1`), not a Gym wrapper + monkey-patched globals
   (`proto_solo.py:43-75`).
2. **Opponent curriculum** weak→mixed→strong on the axis the learnings doc proved
   matters (`ml-learnings-solo-pretrain.md` §4), reusing Sprint B's process-safe
   staged vec-env-rebuild mechanism (`training.py:1177-1201`) but ramping the
   *opponent pool* instead of track difficulty.
3. **Warm-start + multi-phase chaining** in `train_self_play`, with Sprint A's
   chunked Wilson-LB gate (`training.py:1164-1216`) + best-checkpoint preservation
   carried across **all** phases.
4. **A round-robin league evaluator** (the comparison backbone) producing ratings
   over a fixed held-out track set, seat-order-cancelled, reusing the existing
   `round_robin_elo` / `head_to_head` / `compute_elo` / `compute_trueskill`
   machinery (`evaluate.py:398`, `evaluate.py:238`, `stats.py:421/528`).

Per the learnings doc: keep `randomize_seat` throughout (the one Sprint-A win,
`ml-learnings-solo-pretrain.md` §3); **demote the track curriculum to
optional/off** (net-negative at every horizon, same doc); keep the machinery.

---

## 2. Verified code baseline (file:line)

Every reference below was opened and confirmed against the current tree. Drift
from the learnings doc (written against the prototypes) is called out in **§11**.

> **Path note (review):** the rating primitives cited as `stats.py` live at
> `src/heat/simulation/stats.py` (NOT `src/heat/ml/stats.py`); the league/ELO
> code is `src/heat/ml/evaluate.py`. All cited line numbers are correct.

| Symbol | Location | Current behavior (confirmed) |
|---|---|---|
| `CODEC_VERSION` / `OBS_DIM` / `ACTION_DIM` | `spaces.py:38/46/150` | `2` / `104` / `516`. **Unchanged this sprint.** |
| `SHAPING_WEIGHT` / `SHAPING_PROGRESS_COEF` | `spaces.py:159/162` | module globals; set by `apply_shaping_config` (`model.py:184-185`). |
| `_placement_reward` | `spaces.py:175-189` | `n = starting_player_count or num_players`; **`if n <= 1: return 0.0`** (`:182-183`) — the solo slot the §7 spec calls out. Else graded `1 - 2*(rank-1)/(n-1)`. |
| `_spinout_penalty` | `spaces.py:192-206` | rising-edge `player.spun_out` flag → `1.0`/`0.0`. |
| `step_reward(prev, curr, learner_id, done)` | `spaces.py:209-249` | terminal `_placement_reward` when `done`; optional dense progress when `SHAPING_WEIGHT != 0`; optional bounded spinout. **Receives `done` (bool), NOT a `terminated`/`truncated` split, and NOT `gamma`.** This is the key signature gap (§4.1). |
| `HeatEnv.__init__` | `env.py:113-155` | `num_players` guard is **`[1, MAX_PLAYERS]`** (`:123-126`) — solo already allowed (Sprint-learnings engine change, kept). `randomize_seat` (`:119`). |
| `HeatEnv.step` | `env.py:249-282` | computes `terminated, truncated = self._episode_flags()`, `self._done = terminated or truncated`, then `step_reward(prev, self.state, self.learner_id, self._done)` (`:275-278`). **`terminated`/`truncated` are in scope here but collapsed to `done` before the reward call.** |
| `HeatEnv._episode_flags` | `env.py:406-411` | `terminated = state.is_game_over`; `truncated = (not terminated) and round_num > MAX_ROUNDS`. |
| `GameState.is_game_over` | `game_state.py:101-103` | `all(p.finished for p in players)` — works for a 1-player game (the lone car finishing terminates). |
| `GameState.create(track, num_players, seed)` | `game_state.py:184`, `starting_player_count=num_players` `:233` | solo `create(track, 1)` constructs (the learnings-doc engine change). |
| `make_vec_env` | `vec.py:101-166` | threads `shaping_weight`/`shaping_progress_coef`/`shaping_spinout_*`/`randomize_seat` into each worker; re-applies shaping globals inside the worker (`vec.py:68-71`). **No `gamma` plumbing (gamma lives on the model, not the env).** |
| `heat_env_factory` | `vec.py:44-83` | top-level picklable; re-applies shaping globals per worker. |
| `PPOConfig` | `model.py:47-110` | has `gamma=0.999` (`:79`), shaping knobs, `share_features_extractor=False`. **`gamma` is read once in `build_model` (`model.py:268`) and baked into the SB3 model.** |
| `build_model` | `model.py:233-280` | `MaskablePPO(..., gamma=config.gamma, ...)`; `apply_shaping_config(config)` first (`:247`). |
| `apply_shaping_config` | `model.py:174-187` | pushes the 4 shaping knobs into `spaces` globals. |
| `train_self_play` | `training.py:1029-1421` | Phase-1 chunked learn/gate loop (`:1164-1216`) + Phase-2 self-play (`:1248-1404`). Builds env once from `_resolve_track_source` (`:1072`). Returns `(model, best_path)`. |
| Phase-1 chunk loop | `training.py:1164-1209` | `model.learn(chunk, reset_num_timesteps=(p1_chunk_idx==0))`; per-chunk `_gate`; `_save_best` on strict `promote_score` improvement. |
| Curriculum stage rebuild | `training.py:1177-1201` | on stage-boundary crossing, builds a fresh step-pinned `StepAwareTrackSampler`, `_build_vec_env`, `_swap_vec_env`. **This is the exact mechanism to generalize to the opponent axis.** |
| `_swap_vec_env` | `training.py:1424-1445` | `model.set_env(new)`; carries `VecNormalize` stats; closes old. Picklable-agnostic. |
| `_gate` / `_gate_score` / `GateResult` | `training.py:1081-1101` / `924-1026` / `854-882` | Wilson-LB gate; `gate_fn` float shim (`:1082-1092`). Gates on a **fixed held-out generated set** (`_gate_tracks`, `:842-851`) regardless of training stage. |
| `_scripted_opponents` | `training.py:595-622` | `[base]*n_opp` + trailing `RandomAgent`; `base` = `StrongHeuristicAgent` if `use_strong` else `HeuristicAgent`. `broaden_mix` → `_broadened_strong_pool`. |
| `_broadened_strong_pool` / `_StrongHeuristicFactory` | `training.py:491-505` / `474-488` | cycles `[Strong(3), Strong(2), Heuristic, Random]`; picklable factory for the strength-3 seat. |
| `_build_vec_env` | `training.py:654-693` | wraps `make_vec_env` (+ optional `VecNormalize`). Takes `opponents`, `shaping_weight`, `seed`. |
| `sprint_a_curriculum` / `sprint_b_curriculum` | `training.py:508-553` / `556-592` | presets; B = A + `use_track_curriculum=True`. |
| `CurriculumConfig` | `training.py:313-471` | all gate/curriculum/league knobs. `gate_games`, `phase1_eval_every`, `gate_use_wilson_lb`, `randomize_seat`, `broaden_phase1_mix`, `use_strong_heuristic_opponents`, `use_track_curriculum`, etc. |
| `save_checkpoint` | `training.py:139-193` | SB3 zip + sidecar (`obs_dim`/`action_dim`/`codec_version` + repro metadata). |
| `StepAwareTrackSampler` / `CurriculumSchedule` | `generator.py:396+` / `329-383` | picklable staged-difficulty samplers (Sprint B). The schedule pattern (frozen dataclass of endpoints + `*_at(step)`) is the template for `OpponentSchedule`. |
| `round_robin_elo` | `evaluate.py:398-497` | every unordered pair; fills extra seats by **cycling the pair** (`evaluate.py:358-364`) → pairwise-with-duplicates, NOT 4 distinct models. Computes ELO (+ optional TrueSkill) + pairwise table + per-track. |
| `head_to_head` | `evaluate.py:238-322` | seat-order-cancelled A-vs-B; returns `HeadToHeadStats` (`a_win_rate` + Wilson CI). |
| `compute_elo` | `stats.py:421-486` | replays each game's **full `finish_order`** as pairwise wins (winner beats each lower finisher); seeded bootstrap CI. **Already N-way / rank-based.** |
| `compute_trueskill` | `stats.py:528+` | consumes the **full `finish_order`** as a ranked multiplayer outcome. **Already rank-based multiplayer.** |
| `_relabel_outcomes` | `evaluate.py:217-235` | relabels each seat's `agent_type` to a pool key, so distinct models in one game stay distinct labels for rating. **This is what makes a free-for-all rating possible with the existing rating math.** |
| `RoundRobinStats` | `stats.py:176-191` | `ratings` / `trueskill` / `pairwise` / `per_track`. |
| `ml_agent_factory` / `strong_heuristic_agent_factory` / `heuristic_agent_factory` | `evaluate.py:74-92` / `112-129` / runner | picklable path/strength factories for league seats. |
| `League` / `LeagueEntry` | `league.py:44-370` | 6D PFSP league (retention/sampling/bookkeeping). **Path-only entries.** Reusable as the *training* opponent pool; the league *evaluator* (§7) is a separate, simpler round-robin and does not need PFSP. |

---

## 3. Architecture decisions up front (what I chose vs left open)

**Chosen (baked into this design):**
- **Solo is a phase *inside* `train_self_play`, sequenced by a phase list** —
  not a separate `pretrain_solo()` entrypoint. Rationale: the gate, best-checkpoint
  preservation, seat handling, and `_swap_vec_env` are already in
  `train_self_play`; a second entrypoint would duplicate all of it. (The warm-start
  *path* is still first-class, so an externally-pretrained solo checkpoint can be
  fed in — §6.) **Open sub-question** in §10 only on the per-phase-gamma wrinkle.
- **Opponent curriculum = staged vec-env rebuilds** (Option ii from Sprint B
  §5.3), reusing `_swap_vec_env`. The process-safe choice; the same one Sprint B
  settled on (`training.py:1148-1155` comment). A "live opponent pool counter in
  workers" is rejected for the same `SubprocVecEnv`-pickling reason.
- **Solo terminal reward slots into `_placement_reward`'s `n <= 1` branch**
  (`spaces.py:182-183`), gated by a new reward-mode flag — exactly as §7 item 1
  directs.
- **League evaluator: free-for-all (4 distinct models/race) + rank-based rating,
  reusing `compute_elo`/`compute_trueskill`.** The rating math already consumes
  full `finish_order` (§2), so this is the natural fit. The pairwise
  `round_robin_elo` is kept available as the fallback/secondary view. (Tradeoff
  laid out in §7.3.)

**Left open (flagged in §10, decided at build time):**
- Per-phase `gamma` mechanism (SB3 bakes gamma at construction — §4.3).
- Gate yardstick across phases (current-stage opponent vs fixed final target).
- Per-stage step-budget defaults (prototype's 400k were arbitrary).
- Free-for-all sample size + whether to add a Bradley-Terry/rank-logit rating
  beyond ELO/TrueSkill.

---

## 4. Solo reward as a first-class reward mode

### 4.1 Problem (file:line)

`proto_solo.py` bolts the finish bonus on with a `gym.Wrapper`
(`proto_solo.py:43-57`) and monkey-sets `spaces.SHAPING_WEIGHT = 1.0` /
`SHAPING_PROGRESS_COEF = 1.0` inside the env-fn closure (`proto_solo.py:67-68`).
That is throwaway: it bypasses `apply_shaping_config`, is not picklable through
`make_vec_env`'s worker re-application, and the bonus lives outside `step_reward`
so the gate/eval path never sees the same reward. `_placement_reward` already
returns `0.0` for `n <= 1` (`spaces.py:182-183`), so the env currently gives a
solo car **zero terminal reward** — the prototype had to add it externally.

### 4.2 Design — a `reward_mode` and a solo finish bonus inside `step_reward`

Add a module-global `REWARD_MODE` to `spaces.py` (mirroring the existing
shaping-global pattern, set by `apply_shaping_config`), plus a solo finish-bonus
constant, and route the solo terminal reward through `_placement_reward`'s
`n <= 1` branch.

```python
# spaces.py — new module globals (mirroring SHAPING_WEIGHT et al.)

#: Reward mode. "race" (default) == the existing terminal placement reward;
#: "solo" == dense progress (already on via SHAPING_WEIGHT) + a terminal finish
#: bonus paid only when the lone car FINISHES (terminated), not when the episode
#: truncates. Set from PPOConfig.reward_mode by apply_shaping_config.
REWARD_MODE: str = "race"

#: Terminal finish bonus for solo mode, paid once when the lone car completes its
#: laps. Positive-only (no negative rewards anywhere in solo) so there is no
#: "crash to end the episode early" exploit. Default matches the prototype's 5.0.
SOLO_FINISH_BONUS: float = 5.0
```

`step_reward` gains a `terminated` argument so the bonus can be paid only on a
genuine finish (the §7 / `proto_solo.py:54` requirement). **This is a signature
change** — see §4.4 for the call-site plumbing.

```python
def step_reward(prev, curr, learner_id, done, *, terminated=False):
    reward = 0.0

    if done:
        if REWARD_MODE == "solo":
            # Solo terminal: finish bonus ONLY on a real finish (terminated),
            # never on truncation. _placement_reward returns 0 for n<=1, so the
            # placement term contributes nothing here — the bonus is the whole
            # terminal signal. No negative branch => no crash-to-end exploit.
            if terminated:
                reward += SOLO_FINISH_BONUS
        else:
            reward += _placement_reward(curr, learner_id)

    # Dense progress shaping (unchanged); solo mode runs with SHAPING_WEIGHT>0.
    if SHAPING_WEIGHT != 0.0:
        ...  # existing progress term, spaces.py:232-240

    # Optional bounded anti-spinout (unchanged, default-off).
    ...
    return reward
```

Notes:
- The dense progress term is the *existing* shaping (`spaces.py:232-240`); solo
  mode simply runs with `SHAPING_WEIGHT > 0` (the prototype used `1.0`). No new
  progress code.
- `_placement_reward`'s `n <= 1` early return stays as-is; solo mode never relies
  on it for signal, it relies on the explicit `SOLO_FINISH_BONUS`. (Alternative
  considered: put the bonus *inside* `_placement_reward`'s `n<=1` branch. Rejected
  — it would fire on `done` regardless of terminated/truncated, reintroducing the
  truncation-pays-the-bonus bug `proto_solo.py:54` was careful to avoid. Keeping
  the bonus in `step_reward` is where the `terminated` flag is in scope.)

### 4.3 Why solo induces *speed* (the discount argument — for the doc + a test)

Summed **undiscounted** progress over a solo episode is constant:
`Σ progress_t = track.length × track.laps` regardless of how many steps it took
(every space is traversed exactly once per lap). So progress alone cannot reward
*speed*. Under `gamma < 1`, the same progress and the same finish bonus are worth
more the earlier they arrive (`Σ γ^t r_t`), so the policy minimizes
steps-to-finish. With `gamma = 0.999` (the race default, `model.py:79`) the
horizon is so long that the discount gradient is nearly flat — the prototype used
`gamma = 0.99` (`proto_solo.py:39`) precisely to make "finish sooner" pay. Hence
**solo needs `gamma < 1` (≈0.99), distinct from the racing 0.999.** No negative
rewards anywhere in solo, so there is no incentive to crash/spin to end the
episode and stop accruing — there is nothing bad to stop accruing.

### 4.4 Plumbing the new `terminated` arg + `reward_mode`/`gamma` config

Threaded through the existing shaping-config plumbing:

1. **`spaces.step_reward`** — add `terminated` kwarg (default `False` so any
   existing caller is unchanged); add the `REWARD_MODE`/`SOLO_FINISH_BONUS`
   branch.
2. **`HeatEnv.step`** (`env.py:275-278`) — already computes `terminated,
   truncated`; pass `terminated=terminated` into `step_reward`:
   ```python
   terminated, truncated = self._episode_flags()
   self._done = terminated or truncated
   reward = step_reward(prev, self.state, self.learner_id, self._done,
                        terminated=terminated)
   ```
   (`truncated` is already separate; the only change is forwarding `terminated`.)
3. **`PPOConfig`** (`model.py:47-110`) — add `reward_mode: str = "race"` and
   `solo_finish_bonus: float = 5.0`. **`gamma` already exists** (`model.py:79`) —
   the solo phase needs a *different* gamma than the race phases, which is the
   open question in §10 (SB3 bakes gamma at construction).
4. **`apply_shaping_config`** (`model.py:174-187`) — also set
   `spaces.REWARD_MODE = config.reward_mode` and
   `spaces.SOLO_FINISH_BONUS = config.solo_finish_bonus`.
5. **`make_vec_env` / `heat_env_factory`** (`vec.py:44-83`, `:101-166`) — add
   `reward_mode` + `solo_finish_bonus` to the worker re-application block
   (`vec.py:68-71`) exactly like the existing shaping globals, since a spawned
   worker starts from pristine module globals. **Picklable:** they are plain
   `str`/`float` carried on `_EnvBuilder._kwargs` — no closures.
6. **`_build_vec_env`** (`training.py:654-693`) — forward `config.reward_mode` /
   `config.solo_finish_bonus` to `make_vec_env`.

### 4.5 No obs/action/codec change — confirmed

`step_reward`'s signature and `REWARD_MODE` live entirely in the reward path.
`OBS_DIM`/`ACTION_DIM`/`CODEC_VERSION` (`spaces.py:38/46/150`) and
`observation_space()` are untouched. Solo `num_players=1` was already permitted
(`env.py:123-126`) and the opponent-slot obs block zero-fills absent seats, so a
solo observation is a valid v2 vector with no codec edit. **State explicitly in
the DoD.**

---

## 5. `OpponentSchedule` — ramp the opponent pool, not the track

### 5.1 Problem (file:line)

The learnings doc's central finding: the curriculum must ramp the **opponent**
axis (weak→strong), and the *track* curriculum is net-negative
(`ml-learnings-solo-pretrain.md` §3-4). Sprint B built a step-aware
*track*-difficulty schedule + staged vec-env rebuilds
(`training.py:1177-1201`). 8C reuses that exact rebuild mechanism but swaps what
each stage changes: the **opponent pool**, via `_scripted_opponents`
(`training.py:595-622`) and `_broadened_strong_pool` (`:491-505`).

### 5.2 The schedule object (mirrors `CurriculumSchedule`)

A new picklable `OpponentSchedule` in `training.py` (next to `_scripted_opponents`)
maps a stage index to the opponent pool for that stage. It is **data**, not a live
counter, so it pickles into `SubprocVecEnv` workers cleanly (each worker is built
from the pool of *its* stage; the pool itself is a list of picklable zero-arg
factories — classes or `_StrongHeuristicFactory`).

```python
@dataclass(frozen=True)
class OpponentStage:
    """One opponent-curriculum stage: a pool spec + a step budget."""
    pool_kind: str          # "weak" | "mixed" | "strong"
    steps: int              # learn budget for this stage

@dataclass(frozen=True)
class OpponentSchedule:
    """Weak -> mixed -> strong opponent ramp across staged vec-env rebuilds.

    Generalizes Sprint B's CurriculumSchedule from the track axis to the OPPONENT
    axis (the axis ml-learnings-solo-pretrain.md proved matters). Each stage names
    a pool kind, resolved to picklable opponent factories by `pool_for`. Frozen +
    hashable; the pools it produces are lists of zero-arg picklable factories so
    they survive SubprocVecEnv spawn.
    """
    stages: tuple[OpponentStage, ...]

    def pool_for(self, num_players: int, stage: int) -> list:
        kind = self.stages[stage].pool_kind
        if kind == "weak":
            # 2x HeuristicAgent + Random (the proto's "weak", proto_finetune.py:40-44)
            return _scripted_opponents(num_players, use_strong=False)
        if kind == "strong":
            # broadened strong pool (Strong3/Strong2/Heuristic/Random)
            return _scripted_opponents(num_players, use_strong=True, broaden_mix=True)
        if kind == "mixed":
            # half-strong / half-weak rung between weak and strong
            return _mixed_strength_pool(num_players)
        raise ValueError(f"unknown opponent stage kind {kind!r}")
```

`_mixed_strength_pool` cycles e.g. `[Strong(2), Heuristic, Random]` to fill the
seats — a rung between weak and the broadened strong pool. All entries are the
same picklable callables `_scripted_opponents` already returns, so nothing new is
needed for `SubprocVecEnv`.

The prototype validated exactly two opponent stages (vs-weak then vs-strong,
`proto_finetune.py`); the production default adds an explicit "mixed" rung the
spec calls for (weak → mixed → strong). The 3-stage default is in §8.

### 5.3 Driving it from `train_self_play` (reuse the staged-rebuild loop)

The Phase-1 chunk loop already rebuilds + `_swap_vec_env`s on a stage boundary
(`training.py:1177-1201`). 8C generalizes this from "track curriculum stage" to a
unified **phase list** (§6) where each phase carries its own opponent pool, gamma,
shaping weight, and step budget. At each phase boundary:

```python
new_pool = opponent_schedule.pool_for(num_players, phase_idx)
new_venv = _build_vec_env(track=track_source, num_players=num_players,
                          opponents=new_pool, learner_id=learner_id,
                          config=phase_cfg, curriculum=curriculum,
                          shaping_weight=phase_cfg.shaping_weight,
                          seed=seed + 1000 + phase_idx)
venv = _swap_vec_env(model, venv, new_venv)   # carries VecNormalize stats
```

This **replaces** Sprint A's train-from-scratch-vs-strong (one fixed pool for the
whole run) and Sprint B's track-difficulty ramp (`use_track_curriculum` defaults
**off** in the 8C preset; the machinery stays for ablation — documented net-
negative).

### 5.4 Picklability (the `SubprocVecEnv` requirement)

- `OpponentSchedule` / `OpponentStage` are frozen dataclasses of `str`/`int` —
  picklable.
- `pool_for` returns lists of the *same* picklable factories `_scripted_opponents`
  already ships (`HeuristicAgent`, `RandomAgent`, `StrongHeuristicAgent`,
  `_StrongHeuristicFactory`). Each stage's pool is materialized in the **main**
  process and shipped to workers via `_EnvBuilder._kwargs` (`vec.py:94-98`),
  exactly as today. No live schedule object crosses the boundary — only the
  resolved pool list does.

---

## 6. Multi-phase chaining + warm-start in `train_self_play`

### 6.1 The phase list

Today `train_self_play` is hard-wired Phase-1 (scripted) → Phase-2 (self-play)
(`training.py:1117/1220`). 8C introduces an explicit **ordered phase list** that
generalizes both and adds the solo phase at the front:

```python
@dataclass(frozen=True)
class TrainingPhase:
    """One stage of the 8C recipe (solo pretrain or an opponent-curriculum rung)."""
    name: str                 # "solo" | "weak" | "mixed" | "strong"
    num_players: int          # 1 for solo, num_players for the rest
    reward_mode: str          # "solo" | "race"
    gamma: float              # solo ~0.99; race ~0.999 (see §10 gamma wrinkle)
    shaping_weight: float     # solo 1.0; race 0.05 (proto values)
    steps: int                # per-phase learn budget
    pool_kind: str | None     # None for solo; else weak/mixed/strong
```

The default 8C phase list (from the validated recipe,
`ml-learnings-solo-pretrain.md` §5, with the explicit mixed rung):

| # | name | players | reward | gamma | shaping | pool | steps (default §8) |
|---|---|---:|---|---:|---:|---|---:|
| 0 | solo | 1 | solo | 0.99 | 1.0 | — | 300k |
| 1 | weak | 4 | race | 0.999 | 0.05 | weak | 300k |
| 2 | mixed | 4 | race | 0.999 | 0.05 | mixed | 400k |
| 3 | strong | 4 | race | 0.999 | 0.05 | strong | 600k |

Each phase runs the existing **chunked learn/gate loop** (`training.py:1164-1209`)
so Sprint A's Wilson-LB gate + `_save_best` fire after every `phase1_eval_every`
chunk *within* every phase, not just once per phase. `best_score` is a single
running scalar across all phases, so the canonical best checkpoint is preserved
end-to-end (a collapsing strong phase cannot overwrite a better mixed-phase
checkpoint).

### 6.2 Warm-start as a first-class path

The prototype warm-starts via `MaskablePPO.load(SOLO_CKPT, env=venv)`
(`proto_finetune.py:65`). Production exposes this as a parameter:

```python
def train_self_play(config=None, curriculum=None, *, num_players=4,
                    track=None, learner_id=0, gate_fn=None,
                    warm_start_path: str | None = None,   # NEW
                    phases: "list[TrainingPhase] | None" = None):  # NEW
```

- `warm_start_path` set → instead of `build_model(venv, config)`
  (`training.py:1134`), call `MaskablePPO.load(warm_start_path, env=venv,
  device=resolve_device(config.device))` and re-apply shaping/`reward_mode`
  globals. This lets the main run resume from an externally-produced solo
  checkpoint *and* lets the in-process solo phase hand off to the opponent phases
  (the solo phase saves a checkpoint; the next phase warm-starts from it, exactly
  the prototype's two-script handoff but inside one call).
- `phases=None` → default 8C phase list (§6.1). Passing an explicit list (or the
  legacy 1-phase `[strong]`) keeps old behavior reproducible.

### 6.3 The gamma handoff (the load-bearing wrinkle)

Because SB3 bakes `gamma` at construction (`model.py:268`), a phase that needs a
*different* gamma than the one the model was built/loaded with must **rebuild or
reload the model with the new gamma**, carrying the weights forward
(`model.set_parameters` / save+`load`). The solo→weak boundary is exactly such a
boundary (0.99 → 0.999). Mechanism options + recommendation are in §10 (this is
the single most important open question to resolve before building).

### 6.4 Gate target across phases

`_gate_score` gates on a **fixed held-out generated set** (`_gate_tracks`,
`training.py:842-851`) vs the *trained-against* opponent when
`use_strong_heuristic_opponents` (`training.py:968-972`). Across an opponent ramp,
"the trained-against opponent" changes per phase, which makes `promote_score`
non-comparable across phases (a checkpoint that looks great vs weak should not
outrank one that's merely good vs strong). **Recommended:** gate every phase
against the **final strong target** (a fixed yardstick) so `best_score` is a
single coherent scale; report the current-stage win-rate separately for live
visibility. Flagged in §10 as the call to confirm.

---

## 7. Round-robin league EVALUATOR

### 7.1 What it is (and what it is not)

A standalone evaluator (`evaluate.py`, near `round_robin_elo`) that takes a set of
**checkpoints + reference heuristics**, plays them against each other on a fixed
held-out track set, seat-order-cancelled, and produces ratings. Small-N this
sprint (a handful of checkpoints + 2-3 reference agents); 8D scales it up. It is
**not** the training-time PFSP `League` (`league.py`) — that stays the *opponent
sampler* for self-play. The evaluator is read-only comparison.

### 7.2 Reuse, not rebuild

The rating backbone already exists and **already handles N-way rank-based
ratings**:
- `compute_elo` (`stats.py:421`) replays each game's full `finish_order` as
  pairwise wins → an N-player ELO.
- `compute_trueskill` (`stats.py:528`) consumes the full `finish_order` as a
  ranked multiplayer outcome.
- `_relabel_outcomes` (`evaluate.py:217`) stamps each seat with its pool label, so
  distinct models in one race stay distinct labels for rating.
- `head_to_head` (`evaluate.py:238`) gives seat-cancelled pairwise win-rates +
  Wilson CIs.

So the evaluator is mostly **orchestration** over existing primitives.

### 7.3 The key decision: pairwise vs free-for-all (recommended: free-for-all)

`round_robin_elo` today is **pairwise**: each unordered pair plays, and extra
seats are filled by *cycling the same two contenders* (`evaluate.py:358-364`). For
a 4-player game that means seats = `[A, B, A, B]` — never 4 distinct models. That
is fine for a 2-player ladder but does not measure how a model does in a real
4-way field of *different* opponents.

| | Pairwise (existing `round_robin_elo`) | Free-for-all (4 distinct/race) |
|---|---|---|
| What it measures | head-to-head skill, isolated | performance in a mixed field (the real task) |
| Seats | A,B cycled to fill | 4 distinct pool members per race |
| Rating input | full finish_order (already works) | full finish_order (already works) |
| Reuse | `round_robin_elo` as-is | new seating fn + reuse `compute_elo`/`compute_trueskill` |
| Variance/cost | more games (every pair) | fewer games cover more comparisons; needs seat rotation for fairness |
| Seat cancellation | via pair swap | via Latin-square / rotated seatings across the pool |

**Recommendation: free-for-all, rank-based, reusing `compute_elo` + `compute_trueskill`.**
The HEAT task *is* a 4-way race, and the rating math already ingests full
finish-order, so free-for-all is both more faithful and nearly free to add. Keep
pairwise (`round_robin_elo`) available as a secondary view (it gives clean
per-pair Wilson CIs the free-for-all does not).

**Rating system:** prefer **TrueSkill** (or the existing ELO) as the headline —
both are already rank-based and N-way (`stats.py:528`/`421`). A Bradley-Terry /
rank-logit model is a possible 8D addition but is *not* needed here; ELO's seeded
bootstrap CI (`stats.py:464-485`) already gives an uncertainty band. Flagged in §10.

### 7.4 Seat-order cancellation in free-for-all

Pairwise cancels seat order by swapping the two seats. Free-for-all cancels it by
**rotating the seating across games**: for a fixed set of 4 models in a race, run
the game once per cyclic rotation of the 4 seats (or a fixed Latin-square subset),
so every model occupies every grid slot equally. This is deterministic given the
seed and removes the first-mover advantage `head_to_head` already corrects for.
The evaluator's seating helper produces these rotations; `compute_elo` then
ingests the rotated outcomes unchanged.

### 7.5 API sketch

```python
@dataclass
class LeagueLadder:
    ratings: dict[str, EloRating]               # reuse compute_elo
    trueskill: dict[str, TrueSkillRating] | None  # reuse compute_trueskill
    pairwise: dict[tuple[str, str], HeadToHeadStats] | None  # optional 2p view
    outcomes: list[GameOutcome] | None = None   # 8D: raw per-game records (default None)

def evaluate_league(
    contenders: Mapping[str, AgentFactory],   # label -> picklable factory
    *,
    tracks: list[Track],                       # fixed held-out set
    num_players: int = 4,
    games_per_matchup: int = 50,
    mode: str = "free_for_all",                # or "pairwise"
    rating: str = "trueskill",                 # or "elo"
    seed: int = 0,
    # --- 8D additions (additive; defaults preserve 8C small-N behavior) ---
    matchup_sampling: str = "exhaustive",      # "exhaustive" | "sampled"
    fields_per_contender: int | None = None,   # K when sampled
    games_per_field: int | None = None,
) -> LeagueLadder:
    """Round-robin a pool of checkpoints + reference heuristics on a held-out
    track set; seat-order-cancelled; returns rank-based ratings.

    free_for_all: enumerate size-`num_players` combinations of the contender
    labels, play each on rotated seatings (§7.4), relabel by pool key
    (_relabel_outcomes), and feed the combined outcomes to compute_elo /
    compute_trueskill. pairwise: delegate to round_robin_elo.
    """
```

> **8D forward-compatibility (added in review).** Sprint 8D (mass sweep + large
> league) scales this evaluator to tens of contenders, where exhaustive C(n,4)
> field enumeration is infeasible. To avoid an 8C→8D rewrite, build §7.5 with two
> **additive** seams from the start (both default to today's small-N behavior):
> (1) `matchup_sampling="sampled"` + `fields_per_contender`/`games_per_field` so a
> balanced field sampler (each contender in ≥K distinct-4 fields) can be passed in;
> (2) the `LeagueLadder.outcomes` field exposing the relabelled per-game
> `GameOutcome` list, so 8D can persist outcomes and recompute ratings (ELO vs
> TrueSkill, CI tuning) **without replaying races** (`compute_elo`/`compute_trueskill`
> are pure functions of the outcome list). 8C may leave `matchup_sampling="sampled"`
> unimplemented (raise `NotImplementedError`) and just expose the param + the
> `outcomes` field; 8D fills in the sampler. See `docs/sprint-8D-...md` §12.

- `contenders` mixes `ml_agent_factory(path)` for checkpoints and
  `heuristic_agent_factory()` / `strong_heuristic_agent_factory()` for reference
  yardsticks — all picklable (`evaluate.py:74/112`, runner).
- **Determinism:** every `run_batch` call gets a derived seed; ratings are a pure
  function of the seeded outcomes (the existing `compute_elo` sorts by
  `game_index` first, `stats.py:452`), so a fixed seed → identical ratings (test
  in §9).
- Small-N this sprint: a few checkpoints + 2-3 references, `games_per_matchup`
  modest (§10 sizing). 8D turns the knobs up and adds the analysis layer.

---

## 8. The preset + launch script (replacing `proto_*`)

### 8.1 `sprint_8c_curriculum` preset (`training.py`)

A single `CurriculumConfig` + default `TrainingPhase` list + `OpponentSchedule`,
built on `sprint_a_curriculum` (so the trustworthy gate is inherited) with the
track curriculum **off**:

```python
def sprint_8c_curriculum(run_name="heat_ppo_sprint8C", checkpoint_dir="checkpoints"):
    cfg = sprint_a_curriculum(total_timesteps=1_600_000, run_name=run_name,
                              checkpoint_dir=checkpoint_dir)
    return dataclasses.replace(
        cfg,
        use_track_curriculum=False,        # demoted (net-negative per learnings)
        randomize_seat=True,               # the proven Sprint-A win, kept
        use_strong_heuristic_opponents=True,
        broaden_phase1_mix=True,
        gate_use_wilson_lb=True,
        normalize_obs=False,               # Idea 15 recorded constraint
    )
```

The default phase list (§6.1) and `OpponentSchedule` are attached by the launch
script (or returned alongside the config). Per-stage budgets are the §6.1 table;
**they are placeholders** the run will tune (the prototype's 400k/stage were
arbitrary — solo likely needs less, the strong ramp more; §10).

### 8.2 Launch script `train_8c.py` (replaces `proto_solo.py` + `proto_finetune.py`)

A thin top-level script (sibling of the existing `train_and_eval_run.py`) that:
1. builds `PPOConfig(n_envs=8, n_steps=1024, batch_size=256, ...)` (the proto
   hyperparams, `proto_solo.py:109`),
2. calls `train_self_play(config, sprint_8c_curriculum(), phases=DEFAULT_8C_PHASES)`
   — which runs solo → weak → mixed → strong with the gate across all phases,
3. runs `evaluate_league({...best checkpoint, references...}, tracks=HELDOUT)`
   and prints the ladder.

This collapses the two-script prototype handoff (`proto_finetune.py:65` reloading
`proto_solo.py`'s checkpoint) into one gated, league-scored run.

---

## 9. Test plan

`PYTHONPATH=src python -m pytest tests/ -q`. New/extended files alongside the
existing suite (`tests/test_ml_training.py`, `test_ml_env.py`,
`test_track_curriculum.py`, `test_league.py`).

**Solo reward (`tests/test_ml_features.py` reward section or new `test_solo_reward.py`):**
- `test_solo_finish_bonus_only_on_terminated`: with `REWARD_MODE="solo"`,
  `step_reward(prev, curr, lid, done=True, terminated=True)` returns
  `SOLO_FINISH_BONUS` (+ progress); `done=True, terminated=False` (truncation)
  returns **no bonus**. The `proto_solo.py:54` invariant, now unit-tested.
- `test_solo_reward_no_negative`: over a solo episode, no `step_reward` value is
  negative (no crash-to-end exploit), even with the spinout term off.
- `test_solo_reward_bounded`: per-step reward ≤ `SHAPING_WEIGHT*progress_max +
  SOLO_FINISH_BONUS`; finite, no NaN.
- `test_solo_speed_gradient_sign`: discounted return of a *fast* finish exceeds a
  *slow* finish for the same total progress under `gamma<1` (the §4.3 argument):
  construct two reward streams (bonus at t=10 vs t=50), assert
  `Σ γ^t r_fast > Σ γ^t r_slow` for `gamma=0.99` and **equal** for `gamma=1.0`
  (proves discounting, not progress, drives speed).
- `test_reward_mode_default_unchanged`: with `REWARD_MODE="race"` (default),
  `step_reward` equals the current placement reward exactly (regression guard —
  the new `terminated` kwarg defaults must not perturb race mode).

**`OpponentSchedule` (`test_track_curriculum.py` analogue / new `test_opponent_schedule.py`):**
- `test_stage_boundaries`: a 3-stage schedule returns weak/mixed/strong pools at
  stages 0/1/2; out-of-range clamps to the last.
- `test_pool_composition`: weak pool has no `StrongHeuristicAgent`; strong pool
  contains `StrongHeuristicAgent` (strengths 2 & 3) + a `RandomAgent`; every pool
  has exactly `num_players-1` entries.
- `test_schedule_picklable`: `pickle.loads(pickle.dumps(schedule))` round-trips;
  every factory in every stage's `pool_for(...)` is picklable (the
  `SubprocVecEnv` requirement — assert via `pickle.dumps` over each entry).

**Warm-start / multi-phase chaining (`test_ml_training.py`, reuse `gate_fn` shim `:1082`):**
- `test_phase_list_runs_in_order`: with a tiny 2-phase `[solo, weak]` list and an
  injected `gate_fn`, assert both phases call `learn` (spy) and the env is rebuilt
  (opponent pool changes) at the boundary.
- `test_best_checkpoint_preserved_across_phases`: a **descending** injected score
  sequence spanning a phase boundary → the canonical `best_path` holds the first
  (highest) model (the cross-phase analogue of the existing Phase-1 descending-
  score test).
- `test_warm_start_loads_not_builds`: with `warm_start_path` set, monkeypatch
  `MaskablePPO.load` / `build_model` and assert `load` is called and `build_model`
  is not.
- `test_solo_phase_uses_one_player`: the solo phase builds its vec env with
  `num_players=1` (spy on `_build_vec_env`/`make_vec_env` kwargs).

**League evaluator (`new test_league_evaluator.py`):**
- `test_seat_order_cancelled`: a free-for-all of 4 identical agents yields ~equal
  ratings (within a tolerance) — no seat gets a systematic edge after rotation.
- `test_deterministic_ratings_fixed_seed`: same `(contenders, tracks, seed)` twice
  → byte-identical `ratings` (and `trueskill`). Uses cheap scripted agents (no SB3
  load) so it is fast.
- `test_stronger_agent_outranks_weaker`: `StrongHeuristicAgent` outranks
  `RandomAgent` in both ELO and TrueSkill (sanity that the rank decomposition is
  wired correctly through the new seating).
- `test_pairwise_mode_delegates`: `mode="pairwise"` produces a `pairwise` table
  consistent with `round_robin_elo`.

**Contract guard (extend an existing features/contract test):**
- `test_codec_unchanged_8c`: `OBS_DIM==104`, `ACTION_DIM==516`,
  `CODEC_VERSION==2` — fails fast if 8C accidentally touches the contract.

---

## 10. Open questions (decide at build time)

- **Per-phase gamma mechanism (the load-bearing one).** SB3 bakes `gamma` at
  `MaskablePPO` construction (`model.py:268`); `set_env`/`learn` do not change it.
  Solo needs `gamma≈0.99`, the race phases `0.999` (§4.3). Options: (a) at the
  solo→race boundary, **save the solo checkpoint and reload it** with a fresh
  `PPOConfig(gamma=0.999)` (`MaskablePPO.load` reads gamma from the saved model
  unless overridden — *verify whether `load(..., gamma=...)`/`custom_objects` can
  override it, or whether you must build fresh and `set_parameters`* ); (b) run the
  whole thing at one compromise gamma (simplest, but loses the solo speed gradient
  the recipe depends on); (c) build a fresh model per phase with the phase gamma
  and `set_parameters(prev)` to carry weights. **Recommended:** (a) or (c) —
  confirm SB3's `load`/`custom_objects` gamma-override behavior first; this is the
  one place the design must be verified empirically before coding.
- **Gate yardstick across phases (§6.4).** Gate every phase vs the **final strong
  target** (fixed yardstick, recommended) so `best_score` is one coherent scale,
  vs gating vs the current stage's opponent (cheaper, but `promote_score` is not
  comparable across the ramp). Confirm before wiring `_gate_score`'s per-phase
  opponent factory.
- **Per-stage step budgets.** §6.1's 300k/300k/400k/600k are placeholders (the
  prototype's 400k/stage were arbitrary, `ml-learnings-solo-pretrain.md` §6). Solo
  likely needs *less* (it converged in 3.3 min / 400k at 98% finish); the strong
  ramp likely *more*. Tune against the gate + league ladder.
- **Free-for-all sample size + rating choice (§7.3).** `games_per_matchup` and
  whether to add a Bradley-Terry/rank-logit rating beyond ELO/TrueSkill. Small-N
  this sprint; 8D sizes it for significance. The number of size-4 combinations
  grows as C(n,4), so cap contenders or sample matchups when the pool is large.
- **Solo as in-process phase vs separate `pretrain_solo()` (resolved to
  in-process, §3) — revisit only if the gamma handoff (a) forces a checkpoint
  boundary anyway**, in which case a thin `pretrain_solo()` wrapper that produces
  the checkpoint the main run warm-starts from becomes essentially free. Note for
  the implementer.
- **VecNormalize across the gamma reload.** `_swap_vec_env` carries `VecNormalize`
  stats (`training.py:1433-1441`), but a model *reload* for a new gamma may reset
  the return normalization (returns are gamma-dependent). If `normalize_reward` is
  ever enabled with per-phase gamma, confirm the stats handoff is still valid (the
  8C preset keeps `normalize_reward=False`, so this is latent, not active).
- **Solo-phase gating is ill-posed against the race yardstick (added in review).**
  §6.1 runs the chunked Wilson-LB gate within *every* phase, and §6.4 gates vs the
  final strong 4-player target. But the **solo phase** trains a 1-player time-trial
  policy; scoring it in a 4-player strong race is both OOD (it never saw an
  opponent) and uninformative — `best_score` stays ~0 through solo and the
  preserved "best" is meaningless until the first opponent phase. **Recommend:**
  during the solo phase either skip the gate entirely, or gate on a solo metric
  (mean steps-to-finish, as `proto_solo.py:_solo_steps_to_finish`); only start the
  race-yardstick gate at the first opponent phase. Decide when wiring §6.1.

---

## 11. Drift corrected vs the learnings doc / prototypes

- **`step_reward` has no `terminated`/`truncated` split today.** The prototype's
  finish-bonus-only-on-terminated rule (`proto_solo.py:54`) lives in a Gym wrapper
  *outside* `step_reward`. `HeatEnv.step` computes `terminated`/`truncated`
  (`env.py:275`) but collapses them to `done` before calling `step_reward`
  (`env.py:278`). Productionizing the rule therefore requires a **signature
  change** (`step_reward(..., terminated=...)`) + forwarding `terminated` at the
  one call site — not just a globals tweak. The learnings doc's "add the solo
  terminal reward to `spaces.step_reward`" glosses this; §4.4 makes it explicit.
- **`_placement_reward` already returns 0 for `n<=1` — but adding the bonus
  *inside* it is wrong.** It fires on `done` regardless of terminated/truncated,
  which would pay the bonus on truncation (the bug `proto_solo.py:54` avoids). The
  bonus belongs in `step_reward` where `terminated` is in scope (§4.2).
- **`gamma` is NOT an env/reward knob — it's baked into the SB3 model at
  construction** (`model.py:268`). The recipe's "speed via `gamma<1`" and the
  per-phase 0.99/0.999 split therefore cannot be a per-phase `step_reward`
  argument; it forces a model rebuild/reload at the solo→race boundary (§6.3,
  §10). The learnings doc treats gamma as a stage setting without noting SB3's
  construction-time binding.
- **`make_vec_env` already exists with the shaping + `randomize_seat` plumbing
  the prototype hand-rolled.** `proto_solo.py:59-75` builds raw `HeatEnv`s in a
  closure and monkey-sets globals because "reusing `make_vec_env`'s shaping
  plumbing one env at a time is awkward" (its comment). In the merged tree
  `make_vec_env`/`heat_env_factory` re-apply shaping globals per worker
  (`vec.py:68-71`) and thread `randomize_seat` (`vec.py:78`) — so the production
  solo env goes through `make_vec_env` with two added kwargs (`reward_mode`,
  `solo_finish_bonus`), not a bespoke closure.
- **The rating math is already N-way / rank-based.** The learnings doc's "league
  evaluator … pairwise vs free-for-all … Elo vs TrueSkill" frames this as open
  design, but `compute_elo` (`stats.py:432`, "replays each game's `finish_order`")
  and `compute_trueskill` (`stats.py:540`, "consumes the full `finish_order`")
  **already** ingest full multiplayer finish order, and `_relabel_outcomes`
  (`evaluate.py:217`) already keeps distinct models distinct. So free-for-all
  rank-based rating is mostly orchestration over existing primitives — only the
  *seating* (4 distinct models + rotation) is new (§7.4). `round_robin_elo`'s
  extra-seat handling **cycles the pair** (`evaluate.py:358-364`), which is why a
  new free-for-all seating fn is needed rather than calling it directly.
- **`proto_finetune.py` used `reset_num_timesteps=True` per stage**
  (`proto_finetune.py:68/81`). The production multi-phase loop should use
  `reset_num_timesteps=False` after the first phase (matching the Sprint-A
  chunked loop, `training.py:1169`) so the TensorBoard step axis and the gate
  cadence stay continuous across phases.
- **All other references verified accurate against the current tree:**
  `spaces.py:38/46/150/182-183`; `env.py:123-126/275-278/406-411`;
  `model.py:79/174-187/268`; `vec.py:68-71/78`; `training.py:491-505/595-622/
  654-693/842-851/968-972/1164-1209/1177-1201/1424-1445`; `evaluate.py:217-235/
  238-322/358-364/398-497`; `stats.py:176-191/421-486/528+`; `league.py:44-370`;
  `generator.py:329-383/396+`.

---

## 12. Task checklist (in order)

1. **`spaces.py`** — add `REWARD_MODE`/`SOLO_FINISH_BONUS` globals; add
   `terminated` kwarg + the solo branch to `step_reward`. Keep `_placement_reward`
   untouched. (§4.2)
2. **`env.py`** — forward `terminated=terminated` into `step_reward`
   (`env.py:278`). (§4.4)
3. **`model.py`** — add `PPOConfig.reward_mode`/`solo_finish_bonus`; set both in
   `apply_shaping_config`. (§4.4)
4. **`vec.py`** — thread `reward_mode`/`solo_finish_bonus` through
   `make_vec_env`/`heat_env_factory`/`_EnvBuilder` (worker re-application block).
   (§4.4)
5. **`training.py`** — add `OpponentStage`/`OpponentSchedule` (+ `_mixed_strength_pool`)
   and `TrainingPhase`; refactor `train_self_play` to drive an ordered phase list
   via the existing staged-rebuild + chunked-gate loop; add `warm_start_path` +
   `phases` params; resolve the per-phase gamma handoff (§6.3/§10).
   (§5, §6)
6. **`training.py`** — `_build_vec_env` forwards `reward_mode`/`solo_finish_bonus`;
   `sprint_8c_curriculum` preset (track curriculum off, seat randomization on).
   (§8.1)
7. **`evaluate.py`** — `evaluate_league` (+ free-for-all seating/rotation helper +
   `LeagueLadder`), reusing `compute_elo`/`compute_trueskill`/`_relabel_outcomes`/
   `head_to_head`. (§7)
8. **`train_8c.py`** — launch script replacing `proto_solo.py` + `proto_finetune.py`.
   (§8.2)
9. **Tests** — all of §9.
10. Full suite green: `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 13. Definition of done

- (a) **Solo reward is a first-class mode** in `spaces.step_reward`: dense progress
  + terminal finish bonus paid **only on `terminated`**, parameterized
  (`reward_mode`, `solo_finish_bonus`), no negative rewards, no Gym wrapper / no
  monkey-set globals. Speed-gradient and terminated-only tests green.
- (b) **`OpponentSchedule`** ramps weak→mixed→strong via process-safe staged
  vec-env rebuilds (reusing `_swap_vec_env`), is picklable for `SubprocVecEnv`, and
  the track curriculum is demoted to default-off (machinery retained).
- (c) **`train_self_play` chains phases** (solo → opponent curriculum) with
  first-class warm-start, and Sprint A's Wilson-LB gate + best-checkpoint
  preservation flow across **all** phases (descending-score test green across a
  phase boundary). Per-phase gamma handoff resolved (§10).
- (d) **The league evaluator** plays checkpoints + reference heuristics
  free-for-all on a fixed held-out track set, seat-order-cancelled, and produces
  **deterministic** rank-based ratings on a fixed seed (reusing
  `compute_elo`/`compute_trueskill`).
- (e) **One preset + launch script** replace the `proto_*` scripts.
- (f) **Contract unchanged:** `OBS_DIM==104`, `ACTION_DIM==516`, `CODEC_VERSION==2`;
  all new behavior opt-in / default-off where it would otherwise change existing
  runs.
- (g) Full suite green: `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 14. Ordering & dependencies

- **No new upstream dependency.** Sprint A + B are merged and verified (§2): the
  v2 obs, the Wilson-LB chunked gate, `randomize_seat`, the staged-rebuild
  machinery, the strong-opponent factories, and the round-robin/ELO/TrueSkill
  rating primitives all already exist. 8C is reward + schedule + orchestration on
  top of them.
- **Sprint 8D builds on 8C.** The mass training sweep + league analysis consume
  8C's parameterized recipe (the `TrainingPhase` list + `OpponentSchedule` + the
  tuned per-stage budgets) and the `evaluate_league` evaluator (scaled up with more
  contenders, larger `games_per_matchup`, and the analysis layer). Keep the
  `evaluate_league` API and the phase/schedule dataclasses stable enough for 8D to
  parameterize without a rewrite.
