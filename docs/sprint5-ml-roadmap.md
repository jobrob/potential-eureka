# Design: Sprint 5 — Machine Learning Layer (Roadmap)

> **Status:** planning / roadmap only. This document does **not** implement any
> ML code. It splits Sprint 5 (PLAN.md items 15–19) into independently
> shippable, test-gated sub-sprints, locks the interface contracts that must be
> agreed before any parallel work, and gives an honest verdict on where
> multiple agents can work concurrently.
>
> **Prerequisite (already met):** the Sprint 5 *engine-readiness substrate* is
> landed and tested — `engine/driver.py` (`run_round_driver`), `GameState.clone`
> / `GameState.create(seed=)` / injectable `rng`, `Deck.draw_pile` /
> `discard_pile` accessors, and the unified legal-action API
> (`rules.legal_react_options`, `legal_slipstream`, `legal_discards`,
> `ReactOptions`). See `docs/sprint5-engine-readiness-design.md`. This roadmap
> builds **only** on those real APIs (cited below as `file:line`).

---

## 1. Executive summary

Sprint 5 turns the pure-function HEAT engine into a reinforcement-learning
target and trains a PPO policy that should match or beat the heuristic agent.
The work is large and tightly coupled through one contract: the
**observation → action(+mask) → reward** triple that the env, the model, and the
agent all must agree on. The realistic plan is therefore:

1. **Lock the interface contracts first** (§3). These are small, cheap to agree,
   and the single biggest de-risker for parallel work.
2. **Split into four sub-sprints** (§2, §5), each shippable behind the per-step
   test gate (`PYTHONPATH=src python -m pytest tests/ -q`).
3. **Parallelize only the two units that are genuinely decoupled by the
   contracts** — `ml/features.py` (5a) and `ml/model.py` head/extractor scaffold
   (part of 5c) — and keep the env → training → agent chain sequential, because
   it is a true data-dependency pipeline (§4).

**Honest verdict on parallelism (detail in §4):** modest and front-loaded. Once
the contracts in §3 are frozen, **exactly two** work units can run truly
concurrently against stub interfaces — the feature extractor and the model's
policy/extractor scaffold — plus the always-parallel dependency/tooling chore
(5a-0). Everything downstream (`HeatEnv` → `training.py` → `ml_agent.py` →
evaluation) is a sequential pipeline where each stage consumes the previous
stage's *runtime behavior*, not just its type signature, so splitting it across
agents would create integration pain that outweighs the wall-clock saving. Net:
plan for ~1.5 agents' worth of parallelism at the start of the sprint,
converging to a single sequential track by 5b.

### Sub-sprint breakdown

| Sub-sprint | Goal | Key files | Gating tests | Depends on |
|---|---|---|---|---|
| **5a — Features & tooling** | Deterministic `GameState → np.ndarray` feature vector of fixed dim; ML deps installed | `ml/features.py`, `pyproject.toml`, `ml/spaces.py` (contract consts) | `test_ml_features.py` (shape, determinism, bounds, clone-invariance, player-count invariance) | substrate only |
| **5b — Gym environment** | `HeatEnv(gym.Env)` driving `run_round_driver`, one RL step per `Decision`, with action masking + reward | `ml/env.py`, `ml/action_codec.py` | `test_ml_env.py` (reset/step contract, mask legality, episode termination, determinism) | 5a (obs), §3 contracts |
| **5c — Model & training** | SB3 PPO with masked discrete policy; self-play smoke training | `ml/model.py`, `ml/training.py` | `test_ml_model.py` (forward/shape, mask application), `test_ml_training_smoke.py` (`slow`: 100-step loss decreases, checkpoint round-trips) | 5b; model scaffold parallelizable vs 5a |
| **5d — Agent & evaluation** | `MLAgent(BaseAgent)` loads a checkpoint; win-rate eval vs heuristic via the simulation/stats layer | `agents/ml_agent.py`, `ml/evaluate.py`, `scripts/evaluate_ml.py` | `test_ml_agent.py` (protocol conformance, legal-only choices), `test_ml_eval.py` (eval harness on a tiny model) | 5b (action codec), 5c (checkpoint format) |

Each sub-sprint ends by writing tests and running the full suite — the
hard project convention (`docs/sprint5-engine-readiness-design.md:31`).

---

## 2. Motivation

PLAN.md items 15–19 specify the ML layer: `ml/features.py`, `ml/model.py`, a
Gymnasium `HeatEnv`, `ml/training.py` (PPO/SB3 self-play), and
`agents/ml_agent.py`. The substrate that makes this feasible is now in place:

- `run_round_driver(state)` yields a `Decision` at every agent decision point and
  resumes via `gen.send(action)` (`driver.py:76`), the exact push-model backbone
  for `HeatEnv.step()`.
- `GameState.create(..., seed=)` (`game_state.py:183`) and `clone(reseed=)`
  (`game_state.py:137`) give reproducible `env.reset()`.
- `rules.legal_*` (`rules.py:580–626`) enumerate the legal action set at any
  decision point without running the loop — the source of truth for action
  masking.
- `BaseAgent` (`agents/base.py:12`) / `Agent` Protocol (`game.py:22`) is the
  drop-in point so `MLAgent` works in `Game`, `simulation/`, and the viewer
  unchanged.

The remaining risk is not the engine; it is the **ML contract design** — fitting
HEAT's heterogeneous, variable-size, partially-observable decision space onto
SB3's PPO assumptions. That is what §3 and §6 address.

---

## 3. Interface contracts to lock FIRST

These five contracts are the linchpin. They are small, and freezing them up
front is what lets 5a and the 5c model scaffold proceed in parallel without
integration pain. **Recommendation: land these as a thin `ml/spaces.py` +
`ml/action_codec.py` skeleton (constants, dataclasses, stub function signatures
with `raise NotImplementedError`) in the first hour of 5a, reviewed and frozen
before any other unit starts.**

### 3.1 Observation-space spec (consumed by 5a producer, 5b/5c/5d consumers)

A fixed-length `float32` vector. Proposed dimensionality **`OBS_DIM = 72`**
(within PLAN.md's "~60–80"). Variable player count is handled by a **fixed
opponent-slot encoding**: `MAX_PLAYERS = 6` seats; absent/finished opponents are
zero-filled with a presence flag. Hidden opponent hands are **not** in the
vector (partial observability is honest — see §3.1 note).

```python
# ml/spaces.py  (contract; frozen first)
import numpy as np
import gymnasium as gym

OBS_DIM: int = 72
MAX_PLAYERS: int = 6

def observation_space() -> gym.spaces.Box:
    # Bounded, normalized features. low/high chosen so values land in [0, 1]
    # (or [-1, 1] for signed relative positions) after scaling in features.py.
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
```

Proposed feature blocks (the 5a implementer owns the exact packing; the **order
and count are the contract**):

| Block | Floats | Source API | Notes |
|---|---:|---|---|
| Hand histogram | 8 | `player.hand` (`player_state.py:45`) | counts of Speed values 1–4 (4), Heat (1), Stress (1), Upgrade-0/Upgrade-5 (2), each scaled (e.g. `/7`) |
| Own gear | 4 | `player.gear` | one-hot gear 1–4 |
| Own kinematics | 4 | `player.position`, `.lap`, `.heat_available`, `.finished` | position `/track.length`, lap `/track.laps`, heat `/6`, finished 0/1 |
| Deck composition | 6 | `deck.draw_pile` + `discard_pile` (`cards.py:91,96`) | fraction of {Speed, Heat, Stress, Upgrade} in draw+discard; draw-pile size `/total`; discard size `/total` |
| Track lookahead | 4 | `rules.distance_to_next_corner` (`rules.py:260`) | dist-to-next-corner `/track.length`, next speed_limit `/`maxlimit, current-space lanes, in-corner flag |
| Adrenaline / position context | 2 | `rules.adrenaline_eligible` (`rules.py:397`) | adrenaline-eligible flag; own rank `/(num_players-1)` |
| Opponent slots (5 × MAX_PLAYERS-1) | 5×5 = 25 | other `PlayerState` | per opponent: presence flag, relative position (signed, wrap-aware via `(other.position-own)%length`), gear`/4`, lap-delta, finished flag |
| Phase / decision context | ~13 | the `Decision.kind` at the step | one-hot decision kind (5), plus react-context bits (can_boost, has_adrenaline, max_cooldown`/3`), slipstream-available flag, cluttered flag, round_num`/`cap, padding to 72 |

> **Note on partial observability.** Opponent *hands* and *deck order* are hidden
> in the real game; the observation exposes only opponent **public** state
> (position, gear, lap, finished) plus the learning seat's own private state.
> This is correct for HEAT and means the policy is necessarily a partial-info
> policy (one more reason PPO over AlphaZero — PLAN.md:82). Deck-composition
> features use the learning seat's **own** deck only.

> **Padding rule (contract):** the exact float count per block is fixed; any
> spare slots are explicit zero-padding at the tail so `OBS_DIM` is stable. The
> 5a implementer may rebalance block sizes **only** by also editing the frozen
> constant and notifying 5b/5c — i.e. `OBS_DIM` changes are a contract change,
> not a local edit.

```python
# ml/features.py  (signature is the contract; body is 5a's work)
def encode_observation(state: GameState, player_id: int,
                       decision: "Decision | None") -> np.ndarray:
    """Return a float32 vector of shape (OBS_DIM,). Pure; no mutation of state.
    `decision` supplies the phase/decision-context block; None at episode reset
    boundaries (zero-filled context)."""
```

### 3.2 Action-space encoding + masking scheme (the hardest contract)

HEAT has five heterogeneous decision kinds (`DecisionKind`, `driver.py:42`):
`GEAR`, `CARDS`, `REACT`, `SLIPSTREAM`, `DISCARD`. **Recommendation: a single
flattened masked `Discrete(N)` action space, not true multi-head.** This is the
SB3-friendly choice (see §6.1). Each environment step is *one* `Decision`, so the
env only ever needs the mask for the *current* decision kind — the union space is
masked down to that kind's legal actions.

Proposed flattened layout (the **offsets are the contract**):

| Sub-range | Size | Decision kind | Decoding |
|---|---:|---|---|
| `[0,4)` | 4 | GEAR | index → target gear 1–4; map to the `(new_gear, heat_cost)` tuple in `legal_gear_shifts` output (`rules.py:31`) |
| `[4,4+C)` | `C` | CARDS | index over **value-multiset combinations** (see redundancy note) for gears 1–4 |
| `[…)` | 8 | REACT | discretized `ReactDecision`: `cooldown_count ∈ {0..3}` (4) × {boost?} folded, plus adrenaline bits — encode as a small fixed enumeration of legal `(cooldown, boost, adr_speed, adr_cooldown)` combos |
| `[…)` | 2 | SLIPSTREAM | {take, decline} |
| `[…)` | up to `D` | DISCARD | choose-subset over discardable Speed/Upgrade cards; cap to a bounded discrete set (e.g. "discard none / discard lowest-k") |
| **Total** | `ACTION_DIM` | — | fixed constant in `ml/spaces.py` |

```python
# ml/action_codec.py  (contract: these four signatures + ACTION_DIM)
ACTION_DIM: int  # frozen constant

def legal_action_mask(decision: Decision, state: GameState) -> np.ndarray:
    """Bool array shape (ACTION_DIM,): True where the flat action is legal for
    THIS decision. Uses rules.legal_* (rules.py:580-626) + legal_gear_shifts /
    legal_card_plays (rules.py:31, 69). All-False is impossible: every decision
    has >=1 legal action."""

def encode_action_index(decision: Decision, engine_action: object) -> int:
    """Map a concrete engine action (gear tuple / card tuple / ReactDecision /
    bool / card list) to its flat index — used by MLAgent's optional imitation
    and by tests."""

def decode_action(decision: Decision, flat_index: int, state: GameState) -> object:
    """Map a flat index back to the concrete object run_round_driver.send()
    expects for decision.kind. Inverse of encode within the legal set."""
```

> **Value-redundant card plays (carried over from the substrate design,
> `rules.py:70`).** `legal_card_plays` returns one tuple per distinct *Card
> object* combination, so two value-3 Speed cards yield two equivalent tuples.
> The action codec **must** collapse these: enumerate over the **multiset of
> card values** (e.g. "play {3,3}"), and in `decode_action` realize a chosen
> value-multiset to concrete cards by picking the lowest-`id` representatives.
> This keeps `ACTION_DIM` bounded and stops PPO from wasting probability mass on
> duplicate actions. `CARDS` sub-range size `C` = number of distinct
> value-multisets reachable across gears 1–4 (bounded and enumerable once).

### 3.3 Reward function signature (consumed by 5b; tuned in 5c)

```python
# ml/spaces.py (signature frozen; shaping coefficients tunable in 5c)
def step_reward(prev: GameState, curr: GameState, learner_id: int,
                done: bool) -> float:
    """Reward for the learning seat between two driver steps.
    Default: sparse terminal (win/loss/placement) + small dense shaping.
    Shaping weights are config (PPOConfig), not hard-coded, so 5c can tune
    without touching 5b."""
```

Recommended default (see §5b):
- **Terminal (sparse):** `+1` win, graded placement reward for mid-ranks (e.g.
  `1 - 2*(rank-1)/(num_players-1)` ∈ [-1, +1]), computed from
  `state.finished_players` (`game_state.py:93`).
- **Dense shaping (small, optional, default ~0.01 weight):** per-step progress
  (`Δposition` lap-aware) minus a heat-spend penalty proxy. Kept tiny so it does
  not dominate the win signal. **Default the shaping weight to 0 for the first
  training run** (pure sparse) and only enable it if learning stalls.

### 3.4 Model checkpoint save/load format (5c produces, 5d consumes)

- **Format:** SB3's native `model.save(path)` / `PPO.load(path)` (a `.zip`).
  The `MLAgent` ↔ model boundary is "a path to an SB3 `.zip` + the frozen
  `OBS_DIM`/`ACTION_DIM`/codec version." No custom serialization.
- **Versioning (contract):** write a sidecar `path.meta.json` with
  `{obs_dim, action_dim, codec_version, track_name, num_players}`. `MLAgent`
  asserts these match its runtime codec on load — this is the tripwire that
  catches a contract drift between 5a/5b and a stale checkpoint.

### 3.5 `MLAgent ↔ model` interface (5d)

```python
# agents/ml_agent.py  (BaseAgent subclass — drops into Game/simulation/viewer)
class MLAgent(BaseAgent):
    def __init__(self, model_path: str, *, deterministic: bool = True,
                 name: str = "MLAgent") -> None: ...
    # implements choose_gear/choose_cards/choose_react/choose_slipstream/
    # choose_discard by: build Decision-equivalent context -> encode_observation
    # -> model.predict(obs, action_masks=mask) -> decode_action.
```

> **Key constraint:** `BaseAgent`'s methods are *pull* (`choose_gear(state,
> player_id, legal_gears)` etc., `agents/base.py:22-74`) and receive **already
> enumerated legal options**, not a `Decision`. So `MLAgent` reconstructs the
> mask from the `legal_*` arguments it is handed (it does **not** need the driver
> to run). This is why `MLAgent` works inside the existing pull-driven `Game`
> loop *and* the env's push loop can use the same codec. The codec functions in
> §3.2 must therefore accept the raw legal lists too (overload or a thin
> `Decision`-from-args constructor) — flag this for the 5d implementer.

---

## 4. Dependency graph + parallelism verdict

### 4.1 Dependency graph

```
            ┌─────────────────────────────────────────────┐
            │  §3 CONTRACTS (ml/spaces.py + action_codec    │
            │  skeleton, stubs)  — FREEZE FIRST             │
            └───────────────┬─────────────────┬─────────────┘
                            │                 │
         ┌──────────────────▼───┐     ┌───────▼───────────────┐
         │ 5a features.py       │     │ 5c-model scaffold      │   ── PARALLEL ──
         │ (encode_observation) │     │ (SB3 MaskablePolicy /  │   (against stubs)
         │ + deps install       │     │  feature extractor     │
         │ + action_codec impl  │     │  wiring, no training)  │
         └──────────┬───────────┘     └───────────┬───────────┘
                    │  (real obs + codec)          │ (real policy)
                    ▼                              │
         ┌──────────────────────┐                  │
         │ 5b env.py            │◄─────────────────┘  needs real obs+codec
         │ (HeatEnv.step/reset) │                     to be USEFUL; scaffold can
         └──────────┬───────────┘                     compile against stubs only
                    │  (working env)
                    ▼
         ┌──────────────────────┐
         │ 5c training.py       │   (PPO config, self-play loop, smoke test)
         └──────────┬───────────┘
                    │  (checkpoint .zip + meta)
                    ▼
         ┌──────────────────────┐
         │ 5d ml_agent.py +     │   (load checkpoint, BaseAgent conformance,
         │ evaluate.py          │    win-rate eval via simulation/stats)
         └──────────────────────┘
```

### 4.2 Work-unit table: PARALLELIZABLE vs SEQUENTIAL

| Work unit | Mode | Codes against | Why / what it waits on |
|---|---|---|---|
| **5a-0** deps in `pyproject.toml` (`ml` extra already exists, `cards.py`/`pyproject.toml:16`); add `sb3-contrib` for maskable PPO | **PARALLELIZABLE** | nothing | Pure chore; the `[ml]` optional group exists (`pyproject.toml:16`) but lacks `sb3-contrib`. Can land anytime. |
| **§3 contract skeleton** (`ml/spaces.py`, `ml/action_codec.py` stubs) | **SEQUENTIAL — must be first** | substrate APIs | Everything else codes against these. ~1 hour. Gate all parallel work behind its merge. |
| **5a** `features.py` `encode_observation` | **PARALLELIZABLE** | §3.1 obs spec + raw model APIs | Pure function of `GameState`; needs only the frozen `OBS_DIM` and field accessors. No dependency on env/model runtime. |
| **5a** `action_codec.py` impl | **PARALLELIZABLE (with 5a features)** | §3.2 + `rules.legal_*` | Pure mapping functions; testable in isolation against `legal_*` outputs. Best owned by the same agent as 5b later, but the **encode/decode/mask bodies** can be written and unit-tested independently. |
| **5c-model scaffold** (custom features-extractor / Maskable policy wiring) | **PARALLELIZABLE** | §3.1 `OBS_DIM`, §3.2 `ACTION_DIM` | Can be built and unit-tested (forward pass, mask application) against a **stub obs of the right shape** and a random mask — it does not need a real env. |
| **5b** `HeatEnv` | **SEQUENTIAL** | real 5a obs + real codec | Its `step()`/`reset()` behavior depends on the *runtime* of `encode_observation` + `decode_action`, not just signatures. Building it before 5a is real means testing against a stub that will be thrown away — net negative. |
| **5c** `training.py` (PPO loop, self-play, smoke) | **SEQUENTIAL** | working 5b env | Cannot train without a real env. Self-play opponent wiring needs `MLAgent`-style snapshots → also depends on the codec. |
| **5d** `ml_agent.py` + `evaluate.py` | **SEQUENTIAL** | codec (3.2) + checkpoint (3.4) | Needs a real checkpoint to load and a real codec to act; eval needs the simulation/stats layer (already exists). |

### 4.3 Honest parallelism verdict

**Genuinely parallel:** after the §3 skeleton merges, **three** strands can run
at once — (a) 5a features+codec, (b) 5c model scaffold against stubs, (c) the
5a-0 deps chore. That is the real ceiling, and strand (b) only saves time if the
model-scaffold author resists the temptation to also wire training (which needs
the env).

**Not worth parallelizing:** `env → training → ml_agent → evaluation` is a true
pipeline. Each stage consumes the *behavior* of the previous one (a working env,
then a real checkpoint), not merely its type signature. Splitting these across
agents forces stub-then-rewrite churn and integration debugging that costs more
than the sequential wall-clock. **Do not** spawn an agent to "start the env"
before features is real, or to "start training" before the env passes its gate.

**Bottom line:** lock §3, fan out to ~2–3 parallel strands for the first
sub-sprint, then collapse to a single sequential track from 5b onward. Expected
realistic speedup over fully-sequential: small (the front third of the sprint),
not a 4× from "four sub-sprints, four agents."

---

## 5. Per-sub-sprint detail

### 5a — Features & tooling

**Goal.** A pure, deterministic `encode_observation(state, player_id, decision)
-> np.ndarray(OBS_DIM,)` and the frozen contract skeleton; ML deps installed.

**Files.** `ml/spaces.py` (new, contract consts + spaces), `ml/features.py`
(new), `ml/action_codec.py` (new; impl of §3.2), `pyproject.toml` (add
`sb3-contrib` to the existing `[ml]` extra, `pyproject.toml:16`).

**Deliverables.**
- `OBS_DIM`, `MAX_PLAYERS`, `ACTION_DIM`, `observation_space()`,
  `step_reward()` signature, codec stubs → then real bodies.
- `encode_observation` reading only public APIs: `player.hand`, `.gear`,
  `.position`, `.lap`, `.heat_available`, `.finished` (`player_state.py`),
  `deck.draw_pile`/`discard_pile` (`cards.py:91,96`),
  `rules.distance_to_next_corner` (`rules.py:260`),
  `rules.adrenaline_eligible` (`rules.py:397`), `track.length`/`track.laps`
  (`track.py:54`).
- `legal_action_mask`/`encode_action_index`/`decode_action` over `rules.legal_*`.

**Test gates (`test_ml_features.py`, `test_ml_action_codec.py`).**
- Shape always `(OBS_DIM,)`, dtype `float32`, all values in `[-1, 1]`.
- **Determinism:** same `(state, player_id, decision)` → identical vector.
- **Clone invariance:** `encode_observation(state, …)` ==
  `encode_observation(state.clone(), …)` (`game_state.py:137`) — features don't
  depend on RNG identity.
- **Player-count invariance:** 2-player and 6-player states both produce
  `OBS_DIM`; absent opponent slots are zero + presence flag 0.
- Codec: for many random `Decision`s, `legal_action_mask` is True for exactly
  the actions `rules.legal_*` allow; `decode_action(encode_action_index(x)) == x`
  within the legal set; card-value redundancy collapsed (two equal-value Speed
  cards → one action index).
- Full suite green.

**Deps.** Substrate only.

### 5b — Gymnasium environment

**Goal.** `HeatEnv(gym.Env)`: one RL step per `Decision`, masked action space,
reward shaping, reproducible reset, single learning seat vs supplied opponents.

**Files.** `ml/env.py` (new). May add `ml/opponents.py` for the opponent-policy
adapter.

**Mechanics (grounded in the real driver).**
- `reset(seed=…)` builds a fresh state via `GameState.create(track,
  num_players, seed=seed)` (`game_state.py:183`) — or `template.clone(reseed=…)`
  (`game_state.py:137`) if a fixed starting config is wanted — sets `lap=1` per
  player as `Game.__init__` does (`game.py:143`), and creates the round
  generator `gen = run_round_driver(state)` (`driver.py:76`).
- `step(action)`: the env **auto-advances** every `Decision` whose `player_id`
  is **not** the learning seat by querying the opponent policy and
  `gen.send(opp_action)`, until the generator yields a `Decision` for the
  learning seat (or the round ends). Then it decodes the agent's `action` via
  `decode_action` and `gen.send(...)`. When a round generator `StopIteration`s,
  the env starts a new round generator (the round counter advanced inside the
  generator, `driver.py:228`). Episode ends when `state.is_game_over`
  (`game_state.py:101`) or `round_num > MAX_ROUNDS` (`game.py:19`).
- **Observation/mask exposure:** at each learning-seat pause, obs =
  `encode_observation(state, learner_id, decision)`, mask =
  `legal_action_mask(decision, state)` (returned via the SB3-maskable
  `action_masks()` hook — see §6.1).
- **Reward:** `step_reward(prev, curr, learner_id, done)` (§3.3); snapshot
  `prev = state.clone()` (cheap; `game_state.py:137`) before advancing, or track
  the scalar deltas directly to avoid clone cost in the hot loop (preferred).
- **Opponents:** supplied at construction as a list of policies — start with the
  existing `HeuristicAgent`/`RandomAgent` (`runner.py:43-44`); during self-play
  (5c) swap in frozen `MLAgent` snapshots. The opponent adapter calls the same
  `decode_action`/legal logic so opponents only ever make legal moves.

**Test gates (`test_ml_env.py`).**
- Gym API conformance: `reset()` returns `(obs, info)` with
  `obs.shape == (OBS_DIM,)`; `step()` returns 5-tuple; `action_masks()` returns
  `(ACTION_DIM,)` bool.
- **Mask legality:** a masked-illegal action is never sent to the driver
  (sample only from masked-legal indices; assert the driver never raises the
  `ValueError("illegal …")` guards at `driver.py:98,119`).
- **Termination:** every episode terminates with exactly one winner; episode
  length bounded by `MAX_ROUNDS`.
- **Determinism:** same `reset(seed)` + same action sequence → identical reward
  trajectory and finish order.
- Cluttered/finish-early/spun-out edge paths (`driver.py:170,177,194`) advance
  without requesting a learning-seat action when not applicable.
- Full suite green.

**Deps.** 5a (real obs + codec), §3 contracts.

### 5c — Model & training

**Goal.** SB3 PPO with a masked discrete policy; a self-play smoke training run.

**Files.** `ml/model.py` (new; policy/feature-extractor config), `ml/training.py`
(new; PPO setup + self-play curriculum + smoke entry point).

**Model.**
- **Recommendation: `sb3_contrib.MaskablePPO` with `MaskableActorCriticPolicy`
  over the single flattened `Discrete(ACTION_DIM)` space (§3.2), plus a custom
  `BaseFeaturesExtractor` = MLP (3 layers, 128–256 units, per PLAN.md:62).** This
  sidesteps SB3's lack of native multi-head support (§6.1). The "separate heads
  per decision type" idea from PLAN.md:62 is **reinterpreted** as: a single head
  over the union action space, with the decision-kind one-hot in the observation
  (§3.1) telling the net which sub-range is active, and the mask zeroing the
  rest. This is simpler and trains more stably than true multi-head under SB3.
- `model.py` exposes a `build_model(env, config) -> MaskablePPO` and the
  `PPOConfig` dataclass (net arch, shaping weights, n_steps, lr, etc.).

**Training.**
- PPO config defaults: `n_steps=2048`, `batch_size=256`, `gamma=0.999` (long
  episodes), `ent_coef≈0.01`, MLP `[256,256]`. CPU is fine (engine is the
  bottleneck, not the net).
- **Self-play curriculum:** Phase 1 train vs `HeuristicAgent` (and some
  `RandomAgent`) opponents; Phase 2 periodically **freeze** the current policy to
  a checkpoint and mix frozen snapshots into the opponent pool (the env's
  opponent list, 5b). Snapshot cadence is config.
- **Smoke test** (`test_ml_training_smoke.py`, marked `slow` per
  `pyproject.toml:24`): train ~100–500 steps on a tiny net, assert (a) it runs
  without error, (b) `model.save`/`PPO.load` round-trips and the loaded model
  predicts a legal action, (c) explained-variance/loss is finite and the policy
  loss is not NaN. **Do not** assert "ML beats heuristic" here (too slow/flaky) —
  that is 5d's eval, run manually/CI-optional.

**Test gates.**
- `test_ml_model.py`: forward pass shape; mask application zeroes illegal logits;
  builds against the real env's spaces.
- `test_ml_training_smoke.py` (`slow`): the smoke run above. Keep < ~30 s.
- Full suite green (with the `slow` test deselected by default).

**Deps.** 5b (working env). Model scaffold parallelizable vs 5a (§4.2).

### 5d — Agent & evaluation

**Goal.** `MLAgent(BaseAgent)` that loads a checkpoint and plays inside the
existing `Game`/simulation/viewer paths; a win-rate evaluation vs heuristic
reusing the simulation/stats layer.

**Files.** `agents/ml_agent.py` (new), `ml/evaluate.py` (new),
`scripts/evaluate_ml.py` (new CLI).

**MLAgent.**
- Subclass `BaseAgent` (`agents/base.py:12`); implement the five `choose_*`
  methods by reconstructing the mask from the legal lists handed in (§3.5),
  encoding the obs, calling `model.predict(obs, action_masks=mask,
  deterministic=…)`, decoding to the engine action.
- Loads via `PPO.load(path)` and validates `path.meta.json` against runtime
  `OBS_DIM`/`ACTION_DIM`/codec version (§3.4) — fail fast on drift.

**Evaluation (reuse, don't rebuild).**
- Add an `ml_agent_factory(model_path)` mirroring `random_agent_factory` /
  `heuristic_agent_factory` (`runner.py:128-135`) so `MLAgent` slots into
  `run_batch` (`runner.py:316`) unchanged. **Caveat:** `run_batch` parallel mode
  pickles factories to worker processes (`runner.py:25-30`); an SB3 model may be
  heavy/awkward to pickle — default `MLAgent` eval to `parallel=False`, or have
  the factory load the model from `model_path` inside the worker (picklable
  path, not the model object).
- Compute win-rate with `aggregate_stats(outcomes, by="agent_type")`
  (`stats.py`) — the headline "MLAgent vs HeuristicAgent" comparison the success
  metric needs (PLAN.md:96).

**Test gates (`test_ml_agent.py`, `test_ml_eval.py`).**
- Protocol conformance: `MLAgent` satisfies `BaseAgent` and runs a full `Game`
  to completion without crashing.
- **Legal-only:** across many turns, every `choose_*` return is in the legal
  set it was handed (the engine's own guards at `driver.py:98,119` are a
  backstop, but assert it directly too).
- Eval harness runs `run_batch` with an `MLAgent` factory on a tiny throwaway
  model and produces `AgentStats` with a sane `win_rate`.
- Full suite green.

**Success metric (manual / CI-optional, not a unit gate).** After a real
training run, `MLAgent` win-rate ≥ `HeuristicAgent` win-rate over N games via the
eval harness (PLAN.md:96).

**Deps.** 5b (codec), 5c (checkpoint format).

---

## 6. Risks & how the gates de-risk them

### 6.1 SB3 ↔ action-space fit (highest risk)

- **Tension:** PLAN.md:62 imagines "separate heads per decision type" + variable
  card-combination sizes. SB3's `PPO` assumes a **single** fixed action space and
  has **no** native action masking; true multi-head custom policies are possible
  but fiddly and poorly supported.
- **Recommendation (simplest viable):** use **`sb3_contrib.MaskablePPO`** over a
  **single flattened `Discrete(ACTION_DIM)`** (§3.2), decision-kind signaled via
  the observation and irrelevant sub-ranges masked out. Avoid true multi-head.
  Add `sb3-contrib` to deps (5a-0).
- **De-risk:** the §3.2 codec is unit-tested in 5a *before* any model exists; the
  5c model scaffold is tested for mask application against stub obs; 5b asserts
  no illegal action ever reaches the driver. A contract drift surfaces at the
  earliest gate, not in a training run.

### 6.2 Training time / flakiness

- Training is CPU-bound and far heavier than the existing pure-Python tests.
- **De-risk:** keep all *gating* ML tests fast/smoke-level; mark the training
  smoke `slow` (`pyproject.toml:24` already defines the marker) and deselect by
  default. The "beats heuristic" metric is a manual/optional eval, never a unit
  gate. Assert *finite, non-NaN* losses rather than a learning threshold.

### 6.3 Reward shaping

- Dense shaping can dominate the win signal and teach degenerate behavior.
- **De-risk:** `step_reward` is a frozen signature with **config-driven** weights
  (§3.3); default the dense weight to **0** (pure sparse terminal) for the first
  run; turn it on only if learning stalls. Because the weight is config, 5c tunes
  it without touching 5b.

### 6.4 Hidden-info / observation correctness

- Leaking opponent hands would make the policy unrealistic and unfair.
- **De-risk:** §3.1 explicitly excludes opponent hands; a 5a test asserts the
  vector is unchanged when an opponent's *hand* is permuted (only public state
  matters).

### 6.5 Self-play opponent pickling for parallel eval

- SB3 models are not cleanly picklable into `ProcessPoolExecutor` workers
  (`runner.py:25-30` warns about non-picklable factories and falls back to
  sequential).
- **De-risk:** the `MLAgent` factory pickles a **model path**, loading the model
  inside the worker; default eval to `parallel=False`. Documented in 5d.

---

## 7. Non-goals / out of scope

- **No engine rule changes.** `legal_card_plays` redundancy is handled in the ML
  codec (§3.2), not by changing `rules.py`.
- **No true multi-head policy / custom PPO algorithm.** Masked flattened
  `Discrete` only (§6.1). Multi-head is recorded as a fallback if masking proves
  insufficient.
- **No MCTS / AlphaZero.** Ruled out by hidden info + randomness (PLAN.md:82).
- **No new track authoring**, no GPU/distributed training, no hyperparameter
  search infrastructure — single-machine CPU PPO.
- **No mid-round serialization** of the driver generator (the substrate design
  already excluded this; the env holds the live generator within a step only).
- **"ML beats heuristic" is a success *metric*, not a unit-test gate** — it is a
  manual/CI-optional evaluation after real training.
- **No change to `simulation/` or `stats/`** beyond adding an `MLAgent` factory;
  the eval reuses `run_batch` + `aggregate_stats` as-is.

---

## 8. Open questions / assumptions

- **Single learning seat vs multi-seat self-play in one env.** Assumed: one
  learning seat per env, opponents supplied externally (simplest for SB3). True
  simultaneous multi-agent self-play (all seats learning) is a larger design and
  is out of scope for this sprint.
- **Fixed track for training.** Assumed a single track (e.g. USA/Silverstone)
  for the first training runs; `OBS_DIM`'s track features are track-relative
  (normalized by `track.length`/`track.laps`) so the obs *shape* generalizes, but
  cross-track generalization is unverified and not a goal here.
- **`reset()` source:** assumed `GameState.create(seed=)` per episode (fresh
  shuffle) rather than `clone(reseed=)` of a fixed template; the latter is
  available if a held-fixed starting grid is later wanted (both supported by the
  substrate).
- **`CARDS` action-space size `C`.** Needs a one-time enumeration of distinct
  value-multisets across gears 1–4 to fix `ACTION_DIM`; the 5a implementer
  computes and freezes it. Assumed bounded and small.
- **Adrenaline/react discretization granularity.** The 8-slot REACT encoding
  (§3.2) assumes the legal `(cooldown, boost, adr_speed, adr_cooldown)` combos
  collapse to a small fixed enumeration; verify against `ReactDecision`
  (`phases.py:28`) and `legal_react_options` (`rules.py:580`) bounds when fixing
  `ACTION_DIM`.
```
