# Design: Sprint 6C — Training Throughput + Self-Play Stability

> **Status:** planning / design only — no code. Part of Sprint 6 (see
> `docs/sprint6-roadmap.md`). Buildable independently and in parallel with 6A/6B.
> Two parts are merged here because both rewrite the training loop / env
> construction in `ml/training.py` (and `ml/model.py`).

## Goal

Make training **fast enough and stable enough to produce a strong agent**:

- **Part 1 — throughput & hardware:** vectorized envs (SubprocVecEnv /
  DummyVecEnv) to beat the pure-Python engine bottleneck; optional CUDA torch +
  graceful `device` selection; a larger net that actually benefits from the GPU;
  and a measurement plan (env-step vs net time; steps/s vs `n_envs`).
- **Part 2 — self-play stability:** stochastic snapshot opponents, a dense
  reward-shaping schedule, a gradual opponent-mix ramp, a Phase-2 LR/entropy
  schedule + critic warm-up after the env swap, and **evaluation-gated snapshot
  promotion that never overwrites the best checkpoint** — the exact failure that
  lost the good model.

**Internal ordering (important):** build the **throughput/vectorization
foundation first**, then layer self-play stability on top — the stability work
(gated promotion, schedules) assumes the vec-env training loop exists.

## Motivation (verified)

- **Single non-vectorized env, engine-bound.** `HeatEnv` is one `gym.Env`
  (`env.py:78`), built once per phase in `_make_env` (`training.py:248-259`) and
  passed straight to `build_model` (`training.py:335`). Throughput ~1,080
  steps/s; the bottleneck is pure-Python engine stepping in
  `run_round_driver` (`env.py:194`), not the MLP. `device="cpu"` is hard-set
  (`model.py:87`) and `MaskablePPO.load(..., device="cpu")` is hardcoded
  (`training.py:139`).
- **A naive GPU switch will not help** a tiny `[256,256]` MLP (`model.py:58`) on a
  single env: per-rollout batches are small and host↔device transfer dominates.
  The lever is **more envs in parallel**, then a **bigger net** to make the GPU
  worth using.
- **Self-play collapses and overwrites the good model.** `train_self_play`
  (`training.py:308-380`): Phase 1 trains vs the scripted pool; Phase 2 loops
  freezing snapshots and calling `model.set_env(env)` (`training.py:365`) on a
  pool rebuilt by `_mixed_opponent_pool` (`training.py:383-399`). Then it writes
  the **final, collapsed** model to the canonical `run_name` path
  unconditionally (`training.py:376-379`). There is **no best-so-far tracking**.
  Diagnosed causes: value-function shock on `set_env` (a), entropy collapse under
  `SHAPING_WEIGHT=0.0` (b, `spaces.py:134`), deterministic-identical snapshot
  opponents (c, `FrozenSnapshotAgent` default `deterministic=True`,
  `training.py:128-130`, used at `training.py:397`), and Phase-1 brittleness (d).

## Files to create / modify

| File | Action | What |
|---|---|---|
| `src/heat/ml/vec.py` | **new** | `make_vec_env(env_factory, n_envs, vec_cls, seed)`; per-worker opponent/track construction; deterministic per-worker seeding. |
| `src/heat/ml/model.py` | **modify** | `device` selection (`auto`/`cpu`/`cuda` with graceful CPU fallback); a larger net profile; `resolve_device()` helper. Anchors: `PPOConfig.device` (`model.py:87`), `net_arch`/`features_*` (`model.py:58-65`), `build_model` device wiring (`model.py:189`). |
| `src/heat/ml/training.py` | **modify (substantial)** | Vec-env construction; **return/reward normalization** (`VecNormalize` + save/load of stats); eval-gated **best-checkpoint preservation**; stochastic snapshots; opponent-mix ramp; shaping schedule; LR/entropy schedule + critic warm-up; **repro metadata in the sidecar** (hyperparams, git SHA, seed, track/curriculum config). Anchors: `train_self_play` (`training.py:308-380`), `_make_env` (`training.py:248`), `_mixed_opponent_pool` (`training.py:383`), `save_checkpoint` (`training.py:73-96`), `meta_path_for` (`training.py:62`), `FrozenSnapshotAgent` (`training.py:110`), `MaskablePPO.load(device=...)` (`training.py:139`). |
| `src/heat/agents/ml_agent.py` | **modify** | At inference, load the saved `VecNormalize` stats sidecar and apply the **same observation/return normalization** the policy was trained under, so a normalized checkpoint acts correctly. Keep the §3.4 meta tripwire intact (`ml_agent.py:21-27`, `_validate_meta` `ml_agent.py:93-123`). |
| `<checkpoint>.vecnorm.pkl` | **new (sidecar artifact)** | Running `VecNormalize` statistics saved next to each checkpoint (alongside the existing `.meta.json`, `training.py:62-70`). Additive — **not** a codec change. |
| `scripts/train_ml.py` | **new** | Committed training CLI (argparse over `CurriculumConfig`/`PPOConfig`) + TensorBoard logging of losses **and** eval-gate scores. Replaces the uncommitted throwaway `train_and_eval_run.py` at repo root. |
| `pyproject.toml` | **modify (optional dep)** | Document optional **CUDA torch** install; keep `[ml]` working on CPU-only machines. `tensorboard` is already in `[ml]` (`pyproject.toml:16`). Anchor: `ml` extra (`pyproject.toml:16`). |
| `tests/test_ml_training.py` | **new / extend** | Vec-env determinism; best-checkpoint preservation; gated promotion; **VecNormalize stats round-trip**; **sidecar repro-metadata round-trip + back-compat**. |
| `tests/test_ml_agent.py` | **extend** | MLAgent applies the saved normalization at inference; missing stats sidecar falls back cleanly for un-normalized checkpoints. |
| `tests/test_train_cli.py` | **new** | `scripts/train_ml.py` arg-parse → config build smoke; TB log dir is created. |
| `tests/test_ml_training_smoke.py` | **modify** | `slow`: self-play smoke that does **not** collapse; vec smoke. |

## Part 1 — Throughput & hardware

### Vectorized envs

`HeatEnv` already anticipates this: opponents may be **factories** so env copies
hold independent state (`env.py:62-65`, "factories let env copies hold
independent RNG state — useful for vectorized envs"), and `_build_opponents`
re-instantiates per reset (`env.py:151-164`). So vectorization is mostly a
construction concern.

```python
# ml/vec.py
def heat_env_factory(*, track_source, num_players, opponents, learner_id, seed):
    """Top-level (picklable) zero-arg-ish builder returning a fresh HeatEnv.
    Each subprocess calls this to build its own env with its own opponent
    instances and its own track sampler stream."""

def make_vec_env(env_factory, n_envs: int, *, vec_cls="subproc", seed: int = 0):
    """Build a SubprocVecEnv (n_envs across processes) or DummyVecEnv
    (in-process, n_envs==1 or debugging). Seeds each sub-env deterministically
    as seed + worker_index so the whole vec is reproducible."""
```

Key requirements:

- **Per-subprocess opponent construction.** Opponent **specs** must be picklable
  (`env.py:62-65`). Scripted agents are fine; **`FrozenSnapshotAgent` already
  pickles by path** and nulls the model in `__getstate__` (`training.py:142-146`)
  — reused as-is. Never pass a live SB3 model across the process boundary.
- **Per-subprocess track sampling (6A interplay).** When 6A's `track_sampler`
  (a top-level callable) is used, each worker derives its own track from its own
  seed stream — picklable and independent. Falls back to a fixed track (a `Track`
  is picklable) when 6A is absent. This is soft edge (ii)/(b) in the roadmap.
- **`SubprocVecEnv` vs `DummyVecEnv`.** Subproc gives true CPU parallelism (the
  point — beat the Python bottleneck); Dummy is for `n_envs==1` / debugging /
  Windows-spawn troubleshooting. `n_envs` is a `PPOConfig`/`CurriculumConfig`
  field.
- **`MaskablePPO` + vec + masking.** sb3-contrib's `MaskablePPO` supports vec
  envs; the `action_masks()` hook (`env.py:239`) is collected per sub-env. Verify
  the mask plumbing under `SubprocVecEnv` in a smoke test (it is the one place
  vectorization could silently break masking).

### Device selection (optional CUDA, graceful fallback)

```python
# ml/model.py
def resolve_device(requested: str) -> str:
    """requested in {"auto","cpu","cuda"}.
      auto -> "cuda" if torch.cuda.is_available() else "cpu"
      cuda -> "cuda" if available else "cpu" (warn, do not crash)
      cpu  -> "cpu"
    Keeps [ml] usable on CPU-only machines (the installed torch is 2.12.0+cpu)."""
```

- Change `PPOConfig.device` default from the hard `"cpu"` (`model.py:87`) to
  `"auto"`, resolved in `build_model` (`model.py:189`).
- Replace the hardcoded `MaskablePPO.load(..., device="cpu")`
  (`training.py:139`) with the resolved device (snapshots can load on CPU even
  when the learner trains on GPU — opponent inference is tiny; keep snapshots on
  CPU to avoid contending for GPU memory, and document that choice).
- **pyproject:** keep the default `[ml]` extra CPU-only (`pyproject.toml:16`); add
  an *optional* documented CUDA install path (a separate extra or a README note
  pinning a `+cu12x` torch wheel). Do **not** force CUDA into `[ml]` — it would
  break CPU-only machines.

> **Honest note (from the brief, confirmed):** GPU is justified **only together
> with** more envs + a bigger net. On a single env with the current tiny MLP,
> `device="cuda"` will give ~no speedup or a regression. Design GPU as part of
> the throughput story, never in isolation.

### Larger net (to justify the GPU)

Once `n_envs` raises rollout throughput, a bigger policy becomes worth it. Add a
config profile (e.g. `net_arch=[512,512]`, `features_extractor_hidden=[512,512]`,
`features_dim=256`) selectable via `PPOConfig` (`model.py:58-65`). Keep the small
profile as default for tests; the capstone uses the large profile.

### Measurement plan

A throughput measurement (a script or a `slow`-marked benchmark, not a unit gate):

- **env-step time vs net time:** time `gen.send`/engine advance per `HeatEnv.step`
  (`env.py:254-297`) vs the PPO update, to confirm the engine is the bottleneck
  (validates the "CPU is fine, engine-bound" claim, `sprint6-roadmap.md §2`).
- **steps/s vs `n_envs`:** sweep `n_envs ∈ {1,2,4,8,16}` on CPU; expect near-linear
  scaling until cores saturate. This is the evidence that vectorization (not GPU)
  is the real lever.
- **steps/s CPU vs GPU** at the large net + high `n_envs`, to decide if GPU pays
  off for the capstone.

## Part 2 — Self-play stability

Each item targets one diagnosed collapse mechanism (labels match
`sprint6-roadmap.md §2`).

### 2.1 Eval-gated best-checkpoint preservation (fixes the lost model)

The single most important fix. Today `train_self_play` overwrites the canonical
path with the final (collapsed) model (`training.py:376-379`) and FIFO-evicts
snapshots (`training.py:356-358`).

```python
# training.py (new behavior in the Phase-2 loop)
# After each Phase-2 chunk:
score = evaluate_gate(model)          # see below
if score > best_score:
    best_score = score
    save_checkpoint(model, best_path, ...)   # write to a PRESERVED best_path
# The collapsed final model goes to a SEPARATE `*_final` path, never best_path.
```

- **`best_path` is never overwritten by a worse model.** The canonical
  `run_name` checkpoint always holds the best-evaluated policy.
- **The eval gate** (soft edge (i)): prefer 6B's `round_robin_elo` / cross-track
  win-rate; **fallback** = a minimal win-rate over `evaluate_ml` (`evaluate.py:82`)
  vs a fixed opponent on a few seeds. The gate runs every `snapshot_every`
  (`training.py:343`), reusing the eval-cadence loop already present.
- Keep the best model in memory too, so a collapsed run can optionally **reload
  the best** before continuing (a recovery option, config-gated).

### 2.2 Stochastic snapshot opponents (fixes mechanism c)

`_mixed_opponent_pool` builds `FrozenSnapshotAgent(p)` (`training.py:397`) with
the default `deterministic=True` (`training.py:128-130`) → identical, zero-variance
opponents. Change Phase-2 snapshots to **`FrozenSnapshotAgent(p,
deterministic=False)`** and draw from **multiple distinct snapshots** so seats
differ. This restores opponent variance so the learner sees diverse losses (a
gradient to learn from), not uniform −1.

### 2.3 Dense reward-shaping schedule (fixes mechanism b)

`SHAPING_WEIGHT` defaults to 0 (`spaces.py:134`); under all-loss self-play there
is no gradient. Enable dense progress shaping (`step_reward` already supports it,
`spaces.py:180-188`; tuned via `PPOConfig.shaping_weight` →
`apply_shaping_config`, `model.py:91-102`) on a **schedule**: a small positive
weight during the Phase-1→Phase-2 transition and early Phase 2 (to keep a
learning signal while everyone is losing), annealed back toward sparse as the
policy stabilizes. The schedule sets `spaces.SHAPING_WEIGHT` between
`model.learn` chunks (the same mutate-globals pattern as `model.py:91-102`;
process-global is correct — one process trains one policy).

### 2.4 Gradual opponent-mix ramp (fixes mechanisms a + d)

`_mixed_opponent_pool` jumps to `snapshot_mix=0.5` immediately
(`training.py:226`, `training.py:392`). Instead **ramp** the snapshot fraction
from ~0 upward across Phase 2 so the opponent distribution shifts gradually,
limiting the value-function shock and giving the brittle Phase-1 policy time to
generalize. Mix also includes scripted agents (and, post-6B, the **strong
heuristic**) so the pool never becomes all-self-copies.

### 2.5 Phase-2 LR / entropy schedule + critic warm-up (fixes mechanism a)

The value-function shock at `model.set_env` (`training.py:365`) is the core of
the collapse: the critic expects ≈ +0.9, reality becomes ≈ −1, advantages invert.
Mitigations, applied right after the env swap:

- **Critic warm-up:** after `set_env`, run a short low-LR (or value-only-emphasis,
  higher `vf_coef`) window so the critic re-fits the new return distribution
  before the policy takes large steps.
- **LR schedule:** drop `learning_rate` (`model.py:71`) for the first Phase-2
  chunks, then restore — smaller updates while the distribution shifts.
- **Entropy schedule:** keep `ent_coef` (`model.py:70`) up early in Phase 2 to
  preserve exploration (prevents the entropy collapse that locks in all-loss).

> SB3 supports LR schedules via a callable `learning_rate`; entropy/`vf_coef` are
> set at construction, so a clean approach is to **rebuild the model's optimizer
> settings per phase** or pass schedule callables. Specify the chosen mechanism
> when implementing; the contract is "LR/entropy/vf are phase-dependent, not
> fixed".

### 2.6 Return / reward normalization (hardens the critic against the value shock)

§2.5 *softens* the value shock with critic warm-up + an LR schedule; this item
*removes its scale-sensitivity at the root*. The collapse begins because the
critic is trained to expect returns ≈ **+0.9** vs the scripted pool and then sees
≈ **−1** after the opponent-pool swap (`training.py:365`, mechanism (a)). The
standard robust fix is **return/reward normalization** so the critic targets are
**scale-invariant** to that distribution shift: wrap the (vec) env in SB3's
`VecNormalize(norm_reward=True)` (and optionally `norm_obs`), which maintains a
running estimate of the return scale and divides rewards by it, so the value
target stays roughly stationary even when the realized win/loss mix flips.

```python
# ml/training.py (sketch) — normalization wraps the VecEnv from ml/vec.py
from stable_baselines3.common.vec_env import VecNormalize
venv = make_vec_env(env_factory, n_envs, vec_cls=..., seed=seed)   # §Part 1
venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                    gamma=config.gamma, clip_reward=10.0)
# ... train; then persist the running stats alongside the checkpoint:
venv.save(vecnorm_path_for(best_path))   # "<best_path>.vecnorm.pkl"
```

**The cross-cutting caveat (must be designed in, not bolted on):** the running
normalization statistics are *part of the model* — a checkpoint saved with
normalized targets is **wrong if loaded without its stats**. So:

- **Save with the checkpoint.** `save_checkpoint` (`training.py:73-96`) gains a
  companion artifact `"<path>.vecnorm.pkl"` (a `vecnorm_path_for(path)` helper
  mirroring `meta_path_for`, `training.py:62-70`). Both the **best** and
  `*_final` paths get their own stats file. Record in the `.meta.json` sidecar a
  `"normalize": {...}` block (e.g. `norm_obs`/`norm_reward`/`clip_reward`) so a
  reader knows whether a stats file is expected — **additive metadata, not a
  codec change** (`sprint6-roadmap.md §3`).
- **Load at inference (`MLAgent`).** `MLAgent` (`ml_agent.py:65-129`) must, on lazy
  load, also load `"<model_path>.vecnorm.pkl"` and apply the **same**
  normalization to observations/returns it was trained under. If `norm_obs=True`
  was used, the obs passed to `model.predict` (`ml_agent.py:138-147`) must be
  normalized with the saved `obs_rms` (the policy never saw raw obs); reward
  normalization affects only the critic so it does not change action selection,
  but obs normalization does. The §3.4 meta tripwire stays first
  (`_validate_meta`, `ml_agent.py:93-123`).
- **Back-compat.** A checkpoint with **no** stats file (the default for an
  un-normalized run, or any Sprint-5 model) must still load and act exactly as
  today. The stats file and the `"normalize"` meta block are **optional**; their
  absence means "no normalization", not an error.
- **Interaction with vec + snapshots.** `VecNormalize` wraps the `VecEnv`
  (`make_vec_env`, §Part 1), so it lives *outside* the per-worker envs — one set
  of running stats for the learner. `FrozenSnapshotAgent` opponents
  (`training.py:110`, loaded by path) run their **own** inference; if snapshots
  were trained with normalization they must load their own stats file the same
  way `MLAgent` does (reuse the same loader).

> **Why this and §2.5, not one or the other:** §2.5 controls *how fast* the policy
> reacts during the shift; §2.6 keeps the critic's *target scale* stable so there
> is less shift to react to. Together with the §2.1 best-checkpoint gate (the
> safety net), they attack the value shock at three layers.

## Part 3 — Committed training CLI, TensorBoard, and reproducible checkpoints

Cheap, high-utility infrastructure that turns "a run happened on someone's
machine" into "a run that is repeatable, observable, and committed".

### 3.1 Committed training CLI (`scripts/train_ml.py`)

There is a committed **eval** CLI (`scripts/evaluate_ml.py`) but **no committed
training CLI** — the current runner `train_and_eval_run.py` is an uncommitted
throwaway at repo root. Add `scripts/train_ml.py`: an argparse front-end over
`CurriculumConfig`/`PPOConfig` that calls `train_self_play`.

```python
# scripts/train_ml.py (sketch)
#   --timesteps / --phase1-steps / --snapshot-every / --snapshot-mix
#   --n-envs / --vec {subproc,dummy} / --device {auto,cpu,cuda}
#   --net-profile {small,large}              # §Part 1 larger net
#   --shaping-weight / --shaping-schedule    # §2.3
#   --normalize-reward / --normalize-obs     # §2.6
#   --run-name --seed --checkpoint-dir --tensorboard-log
# Builds PPOConfig + CurriculumConfig from args, calls train_self_play(...),
# prints the best/final checkpoint paths. Replaces train_and_eval_run.py.
```

This **replaces the throwaway `train_and_eval_run.py`** (delete it from the repo
root as part of this work).

### 3.2 TensorBoard logging (losses **and** the eval-gate score)

`tensorboard` is already in the `[ml]` extra (`pyproject.toml:16`) and
`PPOConfig.tensorboard_log` already exists (`model.py:88`) and is already passed
into `MaskablePPO` (`model.py:190`) — but it is **never set** by any committed
code, so it is effectively unused. Wiring:

- The CLI sets `PPOConfig.tensorboard_log` (a run dir), so PPO's built-in scalars
  (policy/value loss, entropy, `approx_kl`, clip fraction) are logged for free.
- **Crucially, also log the eval-gate score** (§2.1) to the same TensorBoard run
  each time the gate runs (every `snapshot_every`, `training.py:343`), e.g. via
  the model's logger (`model.logger.record("eval/gate_score", score)`) or an SB3
  callback. **This makes a collapse like the observed one visible live** — the
  gate score flattening/dropping at the Phase-1→Phase-2 boundary is exactly the
  signal that was invisible in the Sprint-5 run.

### 3.3 Reproducibility metadata in the checkpoint (addition #8)

The `.meta.json` sidecar today holds only the obs/action/codec **contract**
(`training.py:86-92`). Extend it with everything needed to **reproduce** a
checkpoint:

```jsonc
// <checkpoint>.meta.json  (NEW additive fields in italics; existing fields kept)
{
  "obs_dim": 72, "action_dim": 516, "codec_version": 1,   // EXISTING tripwire
  "track_name": "usa", "num_players": 4,                   // EXISTING
  "git_sha": "30a0f19...",                                 // NEW
  "seed": 12345,                                           // NEW
  "ppo_config": { /* PPOConfig fields */ },                // NEW
  "curriculum_config": { /* CurriculumConfig fields */ },  // NEW
  "track_config": { /* TrackGenParams or track name(s) */},// NEW (6A interplay)
  "normalize": { "norm_obs": false, "norm_reward": true }  // NEW (§2.6)
}
```

- **Keep the contract tripwire intact.** `MLAgent._validate_meta`
  (`ml_agent.py:93-123`) checks only `obs_dim`/`action_dim`/`codec_version`; the
  new fields are *additive metadata* the loader **ignores for validation**. The
  `CODEC_VERSION` check must not change, and a checkpoint that **lacks** the new
  fields must still load (back-compat: existing Sprint-5 sidecars stay valid).
- **New saves include them.** `save_checkpoint` gains the extra fields (git SHA
  resolved via `git rev-parse HEAD`, tolerating a non-git checkout by recording
  `null`; configs serialized from the `@dataclass`es). This is purely additive to
  `save_checkpoint` (`training.py:73-96`).

## Build sequence

1. **Throughput foundation first:**
   1. `resolve_device` + `PPOConfig.device="auto"` + load-device fix
      (`training.py:139`); CPU fallback test.
   2. `ml/vec.py` (`make_vec_env`, picklable `heat_env_factory`); wire
      `train_self_play`/`smoke_train` to build a vec env.
   3. Larger-net config profile.
   4. Measurement script (`slow`/manual).
2. **Self-play stability on top:**
   5. Eval-gated **best-checkpoint preservation** (separate `best_path` vs
      `*_final`).
   6. Stochastic + multi-snapshot opponents.
   7. Shaping schedule, opponent-mix ramp.
   8. Critic warm-up + LR/entropy schedule.
   9. **Return/reward normalization** (`VecNormalize` around the vec env);
      `vecnorm_path_for` + save the stats with both `best_path` and `*_final`;
      `MLAgent` loads + applies the stats at inference (back-compat: no stats →
      no normalization). (§2.6)
3. **Infrastructure (Part 3):**
   10. **Reproducibility metadata** added to `save_checkpoint` (git SHA, seed,
       configs, `normalize` block); keep the tripwire intact, stay back-compat.
   11. **`scripts/train_ml.py`** CLI over `PPOConfig`/`CurriculumConfig`; delete
       the throwaway `train_and_eval_run.py`.
   12. **TensorBoard** wiring: CLI sets `tensorboard_log`; log the eval-gate score
       each gate run so a collapse is visible live.
4. Full suite: `PYTHONPATH=src python -m pytest tests/ -q` (with `slow`
   deselected by default, `pyproject.toml:24-25`).

## Test gates

### `test_ml_training.py` (fast, default-run)

- **Device fallback:** `resolve_device("cuda")` returns `"cpu"` without raising
  when CUDA is unavailable; `resolve_device("cpu")` is `"cpu"`; `"auto"` matches
  `torch.cuda.is_available()`.
- **Vec-env determinism:** a `make_vec_env(..., seed=s)` with `n_envs>1` produces
  the same rollout/results on two builds with the same seed (per-worker seeding is
  reproducible). Use a tiny config.
- **Best-checkpoint preservation (the headline gate):** simulate a sequence where
  a later model evaluates *worse* than an earlier one and assert the canonical
  `best_path` still holds the better model (e.g. monkeypatch the eval gate to
  return a descending score and assert the saved best is the first/high one).
  This is the direct regression test for the lost-model failure
  (`training.py:376-379`).
- **Gated promotion:** the best checkpoint is only rewritten when the gate score
  strictly improves; `*_final` is always written separately.
- **VecNormalize stats round-trip (§2.6):** saving a checkpoint with
  normalization writes a `"<path>.vecnorm.pkl"`; loading it restores the same
  running stats (mean/var/count), and the `.meta.json` records the `normalize`
  block. Saving an **un-normalized** run writes **no** stats file.
- **Repro-metadata round-trip + back-compat (addition #8):** `save_checkpoint`
  writes `git_sha`/`seed`/`ppo_config`/`curriculum_config`/`track_config` and
  they read back identically; a **legacy sidecar without** the new fields still
  loads and still passes `_validate_meta` (the `CODEC_VERSION`/`obs_dim`/
  `action_dim` tripwire is unchanged and is the only thing validated).

### `test_ml_agent.py` (fast)

- **MLAgent applies normalization at inference (§2.6):** an `MLAgent` over a
  normalized checkpoint loads its `vecnorm.pkl` and normalizes obs with the saved
  `obs_rms` before `predict`; a checkpoint with **no** stats file behaves exactly
  as today (raw obs). The §3.4 meta tripwire still fires first on a mismatched
  contract.

### `test_train_cli.py` (fast)

- **CLI config build:** `scripts/train_ml.py` parses a representative arg vector
  into a valid `PPOConfig` + `CurriculumConfig` (no training run — assert the
  built config, mock/short-circuit `train_self_play`).
- **TB log dir:** passing `--tensorboard-log <dir>` results in the dir being set
  on `PPOConfig.tensorboard_log` (and created by the run).

### `test_ml_training_smoke.py` (`slow`, deselected by default)

- **Vec smoke:** `smoke_train` over a small `SubprocVecEnv`/`DummyVecEnv` runs,
  saves, and `MaskablePPO.load` round-trips a model that predicts a legal action
  (mirrors the Sprint-5 smoke, `sprint5-ml-roadmap.md:443-447`); assert masking
  works under the vec env.
- **Self-play does NOT collapse:** a short Phase-1→Phase-2 self-play run with the
  new stability features, asserting the **best-checkpoint eval score does not
  regress below the Phase-1 score** (i.e. the gate preserves Phase-1 strength
  even if Phase 2 wobbles). This is the behavioral guard against the exact
  collapse — kept fast/small (assert "best ≥ phase-1 baseline", not "learns to
  superhuman").
- Keep each `slow` test bounded (a few seconds to low tens of seconds).

## Risks + de-risking

| Risk | De-risking |
|---|---|
| `SubprocVecEnv` masking breaks silently (illegal actions reach the driver). | A vec smoke test asserts no illegal action is sent (the driver's own guards are a backstop, `env.py` `_decode_legal`); verify `action_masks()` is collected per sub-env. |
| Windows `spawn` pickling failures for env/opponent specs. | Reuse the proven path: top-level factories + `FrozenSnapshotAgent` pickles by path (`training.py:142-146`); a pre-build pickle check like `runner.py:150-167`. |
| GPU gives no speedup / regresses (single small net). | Designed only with vec + larger net; the measurement plan makes the decision data-driven, not assumed. CPU fallback is always available. |
| Stability fixes individually help but the run still collapses. | The **best-checkpoint gate is the safety net**: even a collapsing Phase 2 cannot destroy the good model. Each other fix is additive; the gate alone already prevents the *catastrophic* outcome (lost model). |
| Shaping schedule teaches degenerate behavior. | Keep weights small and annealed; shaping is config-driven (`model.py:91-102`); the eval gate (real win-rate/ELO, 6B) catches a policy that games the shaping. |
| Optional CUDA torch breaks CPU-only installs. | CUDA stays an *optional* documented install; `[ml]` default remains CPU-only (`pyproject.toml:16`); `resolve_device` never crashes on a CPU box. |
| **`VecNormalize` stats not shipped with the checkpoint → silent garbage at inference** (the policy saw normalized obs; `MLAgent` feeds raw obs). | The stats file is saved next to *every* checkpoint and `MLAgent` loads it on lazy load; the `normalize` meta block declares whether stats are expected; a round-trip test (`test_ml_agent.py`) asserts inference applies them. Absent stats ⇒ explicitly "no normalization", not a crash. |
| **Repro-metadata additions accidentally break the contract tripwire / old checkpoints.** | New fields are additive and ignored by `_validate_meta` (`ml_agent.py:93-123`); a back-compat test loads a legacy sidecar lacking them. `CODEC_VERSION` is untouched (`sprint6-roadmap.md §3`). |
| **`norm_reward` interacts badly with the shaping schedule (§2.3) — moving reward scale + moving shaping weight.** | Apply normalization to the *final* reward (shaping included) so the running scale tracks whatever the reward currently is; keep shaping weights small/annealed; the eval gate uses **un-normalized** real win-rate/ELO so it is immune to the normalization scale. |

## Non-goals

- **No distributed / multi-machine training** — single machine, multi-core CPU +
  one optional local GPU.
- **No new RL algorithm / true multi-head** — still `MaskablePPO` over the
  flattened masked `Discrete(ACTION_DIM)` (`model.py:173`).
- **No obs/action/codec changes** (`sprint6-roadmap.md §3`); the vec/throughput
  work must keep `OBS_DIM=72`/`ACTION_DIM=516` and the checkpoint sidecar
  tripwire (`training.py:86-92`) intact. The `VecNormalize` stats file (§2.6) and
  the repro-metadata fields (§3.3) are **additive sidecar/metadata, not a codec
  change** — they never alter the obs/action layout and never bump
  `CODEC_VERSION`.
- **No hyperparameter-search infrastructure** — schedules are hand-specified, not
  auto-tuned.
- **No change to `simulation/`/`stats/`** — the eval gate *consumes* 6B/`evaluate_ml`,
  it does not modify the eval layer.
- **Forcing CUDA** — it stays optional.

## Open questions / assumptions

- **`n_envs` default.** Assumed configurable, default modest (e.g. 4-8) for
  training, 1 (`DummyVecEnv`) for tests. Capstone tunes to core count.
- **Snapshots on CPU while learner on GPU.** Assumed (opponent inference is tiny;
  avoids GPU-memory contention). Confirm `MaskablePPO.load(device="cpu")` on a
  GPU-trained checkpoint is fine (it is, SB3 maps tensors on load).
- **Exact SB3 mechanism for LR/entropy schedules + critic warm-up.** Assumed via
  schedule callables and/or per-phase optimizer reconfiguration; the contract is
  "phase-dependent", the mechanism is an implementation choice to verify against
  the installed SB3/sb3-contrib version.
- **Eval-gate cost vs cadence.** The gate runs every `snapshot_every`
  (`training.py:343`); a full ELO round-robin may be too slow inline — assumed the
  inline gate is a cheap win-rate vs a fixed opponent, with the full 6B
  round-robin run *after* training. Tune cadence to keep training throughput.
- **`norm_obs` on/off by default (§2.6).** Assumed `norm_reward=True` (the direct
  fix for the value shock) with `norm_obs=False` by default — the obs codec is
  already bounded to `[−1, 1]` (`sprint6-roadmap.md §3`), so obs normalization is
  optional. If `norm_obs` is enabled, `MLAgent` *must* apply the saved `obs_rms`;
  decide per capstone and record it in the `normalize` meta block.
- **Git SHA in a dirty/non-git tree.** Assumed `save_checkpoint` records `null`
  when `git rev-parse` fails (e.g. an exported tree); optionally a `dirty` flag if
  the working tree has uncommitted changes. Non-fatal either way.
- **Where `scripts/train_ml.py` logs the eval-gate score.** Assumed via
  `model.logger.record(...)` or an SB3 callback into the same `tensorboard_log`
  run; the contract is "gate score is a TB scalar", the exact hook is an
  implementation choice to verify against the installed SB3.
