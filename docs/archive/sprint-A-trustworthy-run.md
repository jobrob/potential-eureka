# Sprint A — Trustworthy & measurable run (implementation design)

> **Status:** design only — **no code yet**. Expands the Sprint A summary row of
> `docs/ml-improvement-sprints.md` into an implementation-ready spec.
> **Scope:** Ideas 7, 8, 9, 10, 4, 3, the `share_features_extractor=False` half
> of Idea 13, and the broader Phase-1 opponent mix from Idea 6. Idea 15
> (`normalize_obs` off) is a recorded constraint, **not** a task.
> **No obs/action contract change** in this sprint: `OBS_DIM`, `ACTION_DIM`, and
> `CODEC_VERSION` are untouched.

---

## 1. Goal

Make the training loop *honest* so that every number produced by Sprint B (and
beyond) can be believed:

1. Pick the genuinely-best checkpoint during a Phase-1-heavy run, not the final
   (possibly-regressed) weights.
2. Gate against the opponent we actually train against (strong heuristic), and
   report both weak and strong win-rates.
3. Gate with enough games and on the Wilson **lower bound**, so a lucky
   checkpoint cannot displace a genuinely-better one.
4. Remove the seat-0 overfit by randomizing the learner's start seat per
   episode, and measure that it helped via a seat-swept eval.
5. Cheap config levers: Phase-1-heavy budget (4), stronger/longer shaping (3,
   optional bounded anti-spinout term), unshared features extractor (13-part-1),
   broader Phase-1 opponent mix (6).

This is pure training-loop / config / `policy_kwargs` plumbing. **It must land
before any long run and before Sprint B**, because Sprint B's generalization
number is only trustworthy once the gate is.

---

## 2. Verified code baseline (file:line)

Every reference below was opened and confirmed against the current tree. Drift
from the master doc is called out in **§9**.

| Symbol | Location | Current behavior |
|---|---|---|
| `CurriculumConfig` | `training.py:305-412` | dataclass holding all curriculum + gate knobs. `gate_games=20` (`:372`), `use_strong_heuristic_opponents=False` (`:412`). |
| `_scripted_opponents` | `training.py:415-433` | builds Phase-1 pool: `[base]*n_opp` with `pool[-1]=RandomAgent`; `base` is `StrongHeuristicAgent` if `use_strong` else `HeuristicAgent`. |
| `_gate_score` | `training.py:619-677` | saves model to a temp checkpoint, calls `evaluate_ml(...)` per gate track, averages MLAgent win-rate. `per_track_games = max(1, gate_games // len(gate_tracks))` (`:648`). `evaluate_ml` is called with **no `opponent_factory`** (`:666`). |
| `_gate_tracks` | `training.py:607-616` | fixed track → itself; sampler → held-out generated set from `_HOLDOUT_TRACK_SEEDS` (`:586`). |
| Phase-1 gate (once) | `training.py:778-779` | `best_score = _gate(model, venv)` then `_save_best(model, venv)` — fires **exactly once**, at the end of Phase 1. |
| Phase-2 gate (periodic) | `training.py:914-919` | `score = _gate(...)`; logs `eval/gate_score`/`eval/best_score`; promotes only `if score > best_score`. **Only in the Phase-2 `while` loop.** |
| `_save_best` | `training.py:743-755` | writes the canonical `best_path` checkpoint + sidecar. |
| `evaluate_ml` | `evaluate.py:94-152` | `opponent_factory=None` defaults to `heuristic_agent_factory()` (`:131-132`); seat 0 is `ml_agent_factory(model_path)`, seats 1..n-1 are the opponent (`:139`). |
| `head_to_head` | `evaluate.py:190-274` | seat-order-cancelled A-vs-B; returns `HeadToHeadStats` with `a_win_rate` + `a_win_rate_ci` = `wilson_interval(a_wins, games)`. |
| `wilson_interval` | `stats.py:319-344` | `wilson_interval(wins, games, *, z=_Z_95) -> (lo, hi)`; `games==0 -> (0.0, 1.0)`. Already imported in `evaluate.py:52`. |
| `HeatEnv.__init__` | `env.py:106-146` | `learner_id` pinned (default 0); opponents normalized once at construction (`_opponent_specs`). |
| `HeatEnv.reset` | `env.py:191-230` | samples track (if sampler), creates state, `_build_opponents()`, advances to first learner decision. **Does not re-pick `learner_id`.** |
| `_opponent_slots` | `features.py:154-181` | opponents encoded **relatively** (wrap-aware signed rel-position, gear, lap-delta) → obs is seat-agnostic, so seat randomization is well-posed with no obs change. |
| `step_reward` | `spaces.py:157-190` | terminal placement reward + optional dense progress shaping gated on `SHAPING_WEIGHT != 0`. |
| `SHAPING_WEIGHT` / `SHAPING_PROGRESS_COEF` | `spaces.py:134/137` | module globals; set from `PPOConfig` by `apply_shaping_config` (`model.py:159-170`). |
| `PPOConfig` | `model.py:47-95` | hyperparams + `shaping_weight`/`shaping_progress_coef`. **No `share_features_extractor` field.** |
| `build_model` `policy_kwargs` | `model.py:232-239` | sets `net_arch`, `features_extractor_class`, `features_extractor_kwargs`. **Does not set `share_features_extractor`** → inherits SB3 default (`True`). |

---

## 3. Idea-by-idea design

### Idea 7 — In-Phase-1 periodic eval + best-checkpoint preservation

**Problem (file:line).** Under Idea 4, Phase 1 ≈ the whole run. But the Phase-1
gate fires **once** (`training.py:778-779`): `best_score` is set and `_save_best`
is called a single time, *after the entire Phase-1 `model.learn(...)`*. The
periodic, best-preserving gate loop lives only in the Phase-2 `while remaining >
0` block (`training.py:811-919`, promotion at `:917-919`). A Phase-1-heavy run
that peaks at, say, 60% of the way through and then regresses keeps the
**final** weights — the collapse protection is effectively inactive for the run
that matters.

**Fix.** Chunk Phase 1 into `phase1_eval_every`-sized `learn` calls (with
`reset_num_timesteps=False`), and after each chunk run the existing `_gate` and
the existing `score > best_score → _save_best` logic. Log `eval/gate_score`
(and the strong/weak split from Idea 8) to TensorBoard each chunk so plateaus
and regressions are visible live.

**Concrete change (`training.py`, in `train_self_play`).** Replace the single
Phase-1 `learn` + one-shot gate (`:773-779`) with a chunked loop. Sketch:

```python
# --- Phase 1: scripted opponents (now chunked + periodically gated) ---
apply_shaping_config(config)
scripted = _scripted_opponents(
    num_players, use_strong=curriculum.use_strong_heuristic_opponents
)
venv = _build_vec_env(track=track_source, num_players=num_players,
                      opponents=scripted, learner_id=learner_id,
                      config=config, curriculum=curriculum,
                      shaping_weight=config.shaping_weight, seed=seed)
model = build_model(venv, config)

best_score = None
p1_remaining = curriculum.phase1_steps
p1_step = max(1, curriculum.phase1_eval_every)
p1_chunk_idx = 0
while p1_remaining > 0:
    chunk = min(p1_step, p1_remaining)
    model.learn(total_timesteps=chunk, progress_bar=False,
                reset_num_timesteps=(p1_chunk_idx == 0))
    p1_remaining -= chunk
    p1_chunk_idx += 1
    score = _gate(model, venv)                 # GateResult (Idea 8/9)
    model.logger.record("eval/gate_score", score.promote_score)
    model.logger.record("eval/win_rate_strong", score.win_rate_strong)
    model.logger.record("eval/win_rate_weak", score.win_rate_weak)
    if best_score is None or score.promote_score > best_score:
        best_score = score.promote_score
        _save_best(model, venv)
# Guarantee at least one best save even if phase1_steps < phase1_eval_every:
if best_score is None:
    best_score = _gate(model, venv).promote_score
    _save_best(model, venv)
```

The Phase-2 loop (`:811-919`) is unchanged except that its `_gate` now returns
the richer `GateResult` (Idea 8/9) and promotion compares `promote_score`.

**New config fields (`CurriculumConfig`).**

```python
#: Steps per Phase-1 learn/gate chunk (Idea 7). The held-out gate runs after
#: each chunk; set so the periodic eval does not dominate wall-clock (tie to
#: gate_games — see Open Questions). 0 / >= phase1_steps reverts to one gate.
phase1_eval_every: int = 50_000
```

**Risk / cost.** Low. The only cost is the per-chunk gate (real games). The
cadence knob bounds it. Tests inject `gate_fn` so they pay nothing.

---

### Idea 8 — Gate against the opponent we actually train against

**Problem (file:line).** `_gate_score` calls `evaluate_ml(gate_path, ...)`
**without** an `opponent_factory` (`training.py:666`). `evaluate_ml` then
defaults to `heuristic_agent_factory()` (`evaluate.py:131-132`) — the *weak*
`HeuristicAgent`. When `use_strong_heuristic_opponents=True` we train vs
`StrongHeuristicAgent` but select "best" by win-rate vs the easy heuristic,
which saturates near 1.0 and loses discriminating power exactly where we need
it.

**Fix.** Gate vs the opponent matching the training pool, and report **both**
weak and strong win-rates. The promotion score is the strong (or, more
precisely, the trained-against) win-rate's Wilson LB (Idea 9).

**Concrete changes.**

1. New picklable strong-opponent factory in `evaluate.py` mirroring
   `heuristic_agent_factory`:

   ```python
   def strong_heuristic_agent_factory(strength: int = 2) -> AgentFactory:
       """Picklable factory producing StrongHeuristicAgent(strength=...)."""
       return functools.partial(_make_strong_heuristic, strength=strength)
   ```

   (Verify the constructor signature against
   `heat/agents/strong_heuristic.py`; `_scripted_opponents` instantiates it
   zero-arg, so `strength` has a default there — match it, default `strength=2`.)

2. `_gate_score` builds the opponent factory from the curriculum and passes it
   to `evaluate_ml`, and also runs a weak-opponent pass for reporting. Return a
   small `GateResult` rather than a bare float:

   ```python
   @dataclass
   class GateResult:
       win_rate_strong: float   # vs the trained-against pool
       win_rate_weak: float     # vs HeuristicAgent (reporting)
       wilson_lb_strong: float  # promotion criterion (Idea 9)
       games_strong: int

       @property
       def promote_score(self) -> float:
           return self.wilson_lb_strong
   ```

   `_gate_score(...)` then, per gate track, calls `evaluate_ml(..., opponent_factory=strong_factory)` and (cheaper) `evaluate_ml(..., opponent_factory=None)` for the weak number, aggregating wins/games across tracks (see Idea 9 for the aggregation that feeds the Wilson LB).

3. `gate_fn` test override stays a `model -> float`; wrap a float into a
   `GateResult` so existing tests (`test_ml_training.py:181/219`, which inject
   `gate_fn=lambda m: <float>`) keep working:

   ```python
   def _gate(m, venv=None):
       if gate_fn is not None:
           s = float(gate_fn(m))
           return GateResult(s, s, s, games_strong=0)
       return _gate_score(m, curriculum=curriculum, ...)
   ```

**Risk / cost.** Low. Tiny extra eval cost (weak pass is optional and can be
gated behind a `gate_report_weak: bool = True` knob). Promotion now keys off the
right distribution.

---

### Idea 9 — Make the gate statistically trustworthy

**Problem (file:line).** `gate_games=20` (`training.py:372`) split across the 3
held-out tracks is `per_track_games = max(1, 20 // 3) = 6` (`training.py:648`).
A 6-game win-rate has a ±20%+ CI; both best-checkpoint selection (Idea 7) and
the 6D PFSP weights ride on that noise.

**Fix.** (a) Raise `gate_games` for the long run. (b) Aggregate **wins and
games across the held-out tracks** (not a mean of per-track rates) and promote
on the **Wilson lower bound** of the pooled (wins, games), using the already-
imported `wilson_interval` (`evaluate.py:52`, `stats.py:319`).

**Why pooled wins/games, not a mean of rates.** `wilson_interval` needs integer
`(wins, games)`. The current `_gate_score` discards counts and averages rates
(`training.py:664-677`). `evaluate_ml` returns `AgentStats`; confirm it exposes
`wins`/`games` (it aggregates via `aggregate_stats`) — if `AgentStats` carries
only `win_rate`, recover counts as `round(win_rate * num_games)` per track, or
(cleaner) switch the gate to `head_to_head` which already returns
`a_wins`/`games`/`a_win_rate_ci`. Decide at build time (Open Questions).

**Concrete change (`_gate_score`).**

```python
total_wins = 0
total_games = 0
for i, gt in enumerate(gate_tracks):
    per_agent = evaluate_ml(gate_path, num_games=per_track_games,
                            num_players=num_players, track=gt,
                            seed=seed + i, parallel=False,
                            opponent_factory=strong_factory)
    ml = per_agent.get("MLAgent")
    wins = int(round(ml.win_rate * per_track_games)) if ml else 0
    total_wins += wins
    total_games += per_track_games
lb_strong, _ = wilson_interval(total_wins, total_games)
```

**New config fields.**

```python
#: When True, the promotion criterion is the Wilson LOWER bound of the pooled
#: held-out (wins, games) vs the trained-against opponent, not the point
#: estimate (Idea 9). Off keeps the legacy point-estimate behavior for old runs.
gate_use_wilson_lb: bool = True
```

Keep `gate_games` as the knob; the long run sets it high (e.g. `gate_games=120`
→ 40/track). The default stays 20 so existing tests are unaffected.

**Risk / cost.** Low. More games = more gate wall-clock; bounded by cadence
(Idea 7) and the `gate_games` knob.

---

### Idea 10 — Randomize the learner's seat / start position

**Problem (file:line).** `learner_id` is pinned everywhere: `HeatEnv` default 0
(`env.py:111`), never re-picked in `reset` (`env.py:191-230`); `evaluate_ml`
always seats the MLAgent at seat 0 (`evaluate.py:139`). Training never sees a
mid-pack start, yet HEAT has real grid / first-mover effects, so we may be
overfitting the start position even as we generalize across tracks.

**Why it's well-posed.** The observation already encodes opponents *relatively*
(`features._opponent_slots`, `features.py:154-181`) and own state absolutely but
seat-agnostically — nothing in the obs hard-codes seat 0. So changing the
learner's seat changes only *which* engine seat the policy drives, not the obs
contract. No `features.py`/`spaces.py` change.

**Fix.**

1. `HeatEnv`: add `randomize_seat: bool = False`. When set, `reset` re-picks
   `learner_id` from the episode RNG **before** building opponents and advancing.
   Opponents already rebuild per reset (`_build_opponents`, `env.py:172-185`),
   so the only addition is re-seating and re-iterating opponent specs over the
   non-learner seats.

   ```python
   def reset(self, *, seed=None, options=None):
       super().reset(seed=seed)
       if self._track_source is not None:
           self.track = self._track_source(seed)
       if self.randomize_seat:
           # self.np_random is seeded by super().reset(seed=...)
           self.learner_id = int(self.np_random.integers(self.num_players))
       self.state = GameState.create(self.track, self.num_players, seed=seed)
       ...
       self._opponents = self._build_opponents()  # already keyed off learner_id
   ```

   `_build_opponents` (`env.py:172-185`) iterates `self._opponent_specs` over
   all `pid != self.learner_id` — already correct once `learner_id` moves, as
   long as `len(_opponent_specs) == num_players - 1` (it is). Determinism is
   preserved: the seat is drawn from the seeded `np_random`, so "same seed →
   same episode" still holds.

2. `make_vec_env` (`vec.py`): thread a `randomize_seat` flag through to each
   `HeatEnv`. Confirm the constructor wiring in `vec.py` and add the kwarg.

3. `train_self_play`: pass `curriculum.randomize_seat` into the env build
   (`_build_vec_env` → `make_vec_env`). Add the config field.

4. **Eval harness must be able to vary seat to *measure* the benefit.**
   `evaluate_ml` hard-codes the MLAgent at seat 0 (`evaluate.py:139`). Add a
   `learner_seat: int = 0` parameter that places `ml_agent_factory(model_path)`
   at that seat and the opponents elsewhere, so a seat-swept eval can confirm
   the win-rate is seat-robust:

   ```python
   def evaluate_ml(model_path, *, ..., learner_seat: int = 0):
       ...
       factories = [opponent_factory] * num_players
       factories[learner_seat] = ml_agent_factory(model_path)
   ```

   (The DoD seat-sweep loops `learner_seat` over `range(num_players)` and asserts
   the spread is small.)

**New config field (`CurriculumConfig`).**

```python
#: Randomize the learner's start seat each episode (Idea 10). Off by default so
#: existing fixed-seat runs are byte-for-byte unchanged.
randomize_seat: bool = False
```

**Risk / cost.** Low–moderate; the only mildly cross-cutting piece in the
sprint. Covered by an env test (§4) asserting obs shape/bounds and opponent
count stay consistent across every drawn seat, and that "same seed → same seat".

---

### Idea 4 — Phase-1-heavy budget (minimal / no Phase 2)

**Problem.** Phase 2 self-play collapsed and added ~nothing in both runs
(documented in `ml-performance-improvements.md` §"Why this exists"). The
`while remaining > 0` Phase-2 loop (`training.py:811`) runs only when
`total_timesteps - phase1_steps > 0` (`:785`).

**Fix.** No code change — a config setting. For the generalization run, set
`phase1_steps == total_timesteps` (Phase 2 skipped entirely) or a small Phase 2.
This is *why* Ideas 7/8/9 are mandatory: with Phase 1 = the whole run, the
periodic Phase-1 gate (Idea 7) is the only thing preserving the best model.

**Deliverable.** A documented launch config (e.g. in the run script / a
`CurriculumConfig` preset) with `phase1_steps = total_timesteps`,
`phase1_eval_every` set to a sensible cadence, `gate_games` raised. No new field.

**Risk / cost.** Trivial. Frees ~half the compute.

---

### Idea 3 — Stronger / longer reward shaping (+ optional bounded anti-spinout)

**Problem.** Shaping is progress-only, weight defaults to 0 (`spaces.SHAPING_WEIGHT
= 0.0`, `spaces.py:134`; `PPOConfig.shaping_weight = 0.0`, `model.py:81`). On the
much harder sparse generated-track problem, the gradient is too thin early.

**Fix (cheap, config).** Raise `shaping_weight` and anneal slowly to a small
non-zero floor rather than to 0. The existing schedule (`shaping_weight_start`/
`shaping_weight_end`, `training.py:348-349`) already anneals across **Phase 2**;
under Idea 4 there is little/no Phase 2, so for Phase 1 use a non-zero
`PPOConfig.shaping_weight` (applied by `apply_shaping_config` at
`training.py:758`). Document the recommended starting magnitude (Open Questions).

**Optional bounded anti-spinout term (MED, default-off).** Extend `step_reward`
(`spaces.py:157-190`) with a small, **bounded** penalty when the learner spins
out / overshoots a corner speed limit, gated behind a new default-0 weight so
existing behavior is unchanged:

```python
#: Optional bounded anti-spinout penalty weight (Idea 3). Default 0 == off.
SHAPING_SPINOUT_WEIGHT: float = 0.0
#: Hard cap on the per-step spinout penalty magnitude (keep it from dominating).
SHAPING_SPINOUT_CAP: float = 0.05
```

```python
if SHAPING_SPINOUT_WEIGHT != 0.0:
    penalty = _spinout_penalty(prev, curr, learner_id)   # in [0, 1]
    reward -= min(SHAPING_SPINOUT_CAP,
                  SHAPING_SPINOUT_WEIGHT * penalty)
```

`_spinout_penalty` reads the engine's spinout/heat-overpay signal between
`prev` and `curr` (confirm the exact field on `PlayerState`/`GameState` that
records a forced cooldown or spinout this step). Add `shaping_spinout_weight` /
`shaping_spinout_cap` to `PPOConfig` and push them in `apply_shaping_config`
(`model.py:159-170`) alongside the existing two globals.

**Risk / cost.** Tiny (config) + small (optional term). Keep bounded and ablate
against degenerate play (Open Questions).

---

### Idea 13-part-1 — `share_features_extractor=False`

**Problem (and a drift correction — see §9).** The master doc and
`ml-performance-improvements.md` Idea 13 cite "`share_features_extractor`
defaults to `True`, `model.py:232-239`." **`share_features_extractor` does not
appear in `model.py` at all.** Lines 232-239 are the `policy_kwargs` dict, which
sets `net_arch` / `features_extractor_class` / `features_extractor_kwargs` and
**never sets `share_features_extractor`** — so the value is inherited from SB3's
`ActorCriticPolicy` default, which **is** `True`. The *conclusion* (a single
shared trunk feeds both heads) is correct; the *citation* (an explicit setting at
those lines) is not. The fix is to add the key.

**Fix.** One line in `policy_kwargs` (`model.py:232-239`):

```python
policy_kwargs: dict = {
    "net_arch": list(config.net_arch),
    "share_features_extractor": config.share_features_extractor,  # Idea 13-1
    "features_extractor_class": HeatMLPExtractor,
    "features_extractor_kwargs": {...},
}
```

**New config field (`PPOConfig`, `model.py:47-95`).**

```python
#: Give the actor and critic separate feature extractors (Idea 13). False here
#: == SB3 default (one shared trunk). True splits them so the actor ("what
#: action given this kind") and critic ("how's the race going") stop competing
#: for one trunk — friction that worsens as the net grows.
share_features_extractor: bool = False
```

Naming caution: the field reads `share_features_extractor` to mirror SB3, so
**default `False` means *unshared*** (the new behavior). Document this inversion
in the field docstring to avoid a footgun.

**Risk / cost.** Low. Costs params (already accepted). No codec dependency, so
it lands in Sprint A.

---

### Idea 6 — Broader Phase-1 opponent mix

**Problem (file:line).** `_scripted_opponents` (`training.py:415-433`) fills all
opponent seats with one base class (`StrongHeuristicAgent` or `HeuristicAgent`)
plus a single trailing `RandomAgent`. That is thin curriculum variety.

**Fix.** Build a richer Phase-1 mix: `StrongHeuristicAgent` at strengths 2 **and**
3, a plain `HeuristicAgent`, and a `RandomAgent`, distributed across the
available opponent seats. Keep it behind the existing `use_strong` switch so
non-strong runs are unchanged, and behind a new `broaden_phase1_mix` flag so the
exact composition is opt-in.

```python
def _scripted_opponents(num_players, *, use_strong=False, broaden_mix=False):
    n_opp = num_players - 1
    if use_strong and broaden_mix:
        # e.g. [Strong(3), Strong(2), Heuristic, Random] truncated/cycled to n_opp
        pool = _broadened_strong_pool(n_opp)
    else:
        base = StrongHeuristicAgent if use_strong else HeuristicAgent
        pool = [base] * n_opp
        if n_opp >= 1:
            pool[-1] = RandomAgent
    return pool
```

`_broadened_strong_pool` returns zero-arg callables (classes or
`functools.partial` for the strength-3 variant — confirm
`StrongHeuristicAgent(strength=3)` is constructible and picklable for
`SubprocVecEnv`). New `CurriculumConfig` field:

```python
#: Broaden the Phase-1 strong-opponent pool (strengths 2 & 3 + heuristic +
#: random) for curriculum variety (Idea 6). Off by default.
broaden_phase1_mix: bool = False
```

**Risk / cost.** Low (cheap, opt-in). LOW-MED value per the ROI scorecard.

---

### Idea 15 — `normalize_obs` OFF (recorded constraint, not a task)

Already the default: `CurriculumConfig.normalize_obs = False`
(`training.py:365`). The obs is bounded to `[-1, 1]` by design
(`features.py` `_clip*` + the final `np.clip`, `features.py:253`), and the gate
uses the real un-normalized win-rate. **Do not turn it on.** No build item —
record it in the launch config and in `CurriculumConfig`'s docstring.

---

## 4. Test plan

Add to `tests/test_ml_training.py` (gate/seat) and `tests/test_ml_env.py`
(seat-randomization env behavior). Reuse the `_tiny_ppo` / `_tiny_curriculum`
helpers already in `test_ml_training.py` and the `gate_fn` injection pattern
(`test_ml_training.py:192`).

**Gate (Ideas 7/8/9):**

- `test_phase1_periodic_gate_preserves_best`: with `phase1_steps`
  spanning several `phase1_eval_every` chunks and a **descending** injected
  `gate_fn` score sequence, assert the canonical `best_path` checkpoint holds
  the first (highest) model — the Phase-1 analogue of the existing
  `test_best_checkpoint_preserved_under_descending_scores`
  (`test_ml_training.py:181`).
- `test_phase1_gate_promotes_on_strict_improvement`: ascending-then-flat scores;
  assert promotion happens only on strict improvement.
- `test_gate_uses_strong_opponent_factory`: monkeypatch `evaluate_ml` to record
  the `opponent_factory` it received; assert that with
  `use_strong_heuristic_opponents=True` the gate passes a strong factory (not
  `None`).
- `test_gate_promotes_on_wilson_lb`: a unit test of the pooled-(wins,games) →
  `wilson_interval` promotion: two checkpoints with equal point estimates but
  different game counts → the higher-`n` one has the higher LB and is promoted.
- `test_gate_result_reports_weak_and_strong`: assert `GateResult` carries both
  win-rates.

**Seat randomization (Idea 10):**

- `test_reset_repicks_seat_when_enabled`: with `randomize_seat=True`, drive many
  resets with distinct seeds and assert `env.learner_id` takes >1 distinct value
  across `range(num_players)`.
- `test_seat_randomization_is_seed_deterministic`: same `reset(seed=k)` twice →
  same `learner_id` and identical first obs.
- `test_obs_consistent_across_seats`: for every drawn seat, `reset` returns an
  obs of shape `(OBS_DIM,)`, bounds `[-1, 1]`, and `len(env._opponents) ==
  num_players - 1` (opponents fill exactly the non-learner seats).
- `test_evaluate_ml_learner_seat`: `evaluate_ml(..., learner_seat=k)` places the
  MLAgent at seat `k`; a small seat sweep returns finite win-rates for every
  seat.

**Shaping (Idea 3):**

- `test_spinout_penalty_bounded`: with `SHAPING_SPINOUT_WEIGHT` set high, the
  per-step penalty magnitude never exceeds `SHAPING_SPINOUT_CAP`.
- `test_shaping_default_off_unchanged`: with all shaping weights 0, `step_reward`
  equals the pure-sparse value (regression guard).

**Model (Idea 13-1):**

- `test_unshared_features_extractor_built`: `build_model(env, PPOConfig(
  share_features_extractor=False))` produces a policy whose actor and critic
  feature extractors are distinct objects (assert `policy.pi_features_extractor
  is not policy.vf_features_extractor`, per the SB3 attribute names — confirm at
  build time).

**Opponent mix (Idea 6):**

- `test_broadened_phase1_pool_composition`: `_scripted_opponents(num_players,
  use_strong=True, broaden_mix=True)` yields the expected class mix and exactly
  `num_players - 1` entries.

**Suite command:** `PYTHONPATH=src python -m pytest tests/ -q`

---

## 5. Task checklist (in order)

1. `evaluate.py`: add `strong_heuristic_agent_factory` (+ top-level
   `_make_strong_heuristic`); add `learner_seat: int = 0` to `evaluate_ml`.
2. `training.py`: add `GateResult` dataclass; rewrite `_gate_score` to take an
   opponent factory, pool wins/games across gate tracks, and return a
   `GateResult` with the Wilson LB. Adapt `_gate` (and the `gate_fn` float
   shim).
3. `training.py`: chunk Phase 1 into `phase1_eval_every` learn/gate iterations
   with best-checkpoint preservation + TensorBoard logging (Idea 7).
4. `training.py` `CurriculumConfig`: add `phase1_eval_every`,
   `gate_use_wilson_lb`, `randomize_seat`, `broaden_phase1_mix`. Update
   `_scripted_opponents` for the broadened mix (Idea 6).
5. `env.py`: add `randomize_seat`; re-pick `learner_id` in `reset` from the
   seeded RNG before building opponents. `vec.py`: thread the flag through
   `make_vec_env`; `training.py`: pass it in `_build_vec_env`.
6. `model.py`: add `PPOConfig.share_features_extractor=False`; set it in
   `policy_kwargs` (Idea 13-1).
7. `spaces.py`: add `SHAPING_SPINOUT_WEIGHT`/`SHAPING_SPINOUT_CAP` (default-off)
   + `_spinout_penalty` in `step_reward`. `PPOConfig` + `apply_shaping_config`:
   thread the new shaping knobs (Idea 3).
8. Write the launch config / preset: `phase1_steps == total_timesteps`,
   raised `gate_games`, sensible `phase1_eval_every`, `use_strong_heuristic_
   opponents=True`, `randomize_seat=True`, non-zero `shaping_weight`,
   `normalize_obs=False` (Ideas 4/15).
9. Add all tests in §4; run `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 6. Definition of done

- (a) A Phase-1-only run preserves and reports the **best** held-out checkpoint
  (not the final one), with `eval/gate_score` on TensorBoard.
- (b) The gate reports **both** weak- and strong-heuristic win-rates and promotes
  on the **Wilson lower bound** of the pooled held-out (wins, games) vs the
  trained-against opponent.
- (c) The gate plays enough games (e.g. ≥40/track) that the LB is usable.
- (d) Training episodes see randomized learner seats; a seat-swept `evaluate_ml`
  confirms no seat-0 overfit (small win-rate spread across `learner_seat`).
- (e) `share_features_extractor=False` is wired and tested; broadened Phase-1
  mix available; bounded shaping available and default-off-safe.
- (f) Full suite green: `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 7. Ordering & dependencies

- **No upstream dependency.** Sprint A is pure training-loop/config/`policy_kwargs`.
- **Sprint A must land before Sprint B.** Sprint B's held-out generalization
  number is only believable once the gate selects the genuinely-best checkpoint
  (Idea 7), scores against the right opponent (Idea 8), and uses the Wilson LB
  over enough games (Idea 9). Sprint B explicitly depends on A.

---

## 8. Open questions (decide at build time)

- **Gate cadence vs cost.** `phase1_eval_every` × `gate_games`: the gate plays
  real games, so a tight cadence with a high game count can dominate wall-clock.
  Tie `phase1_eval_every` to the raised `gate_games`.
- **Wins/games recovery for the Wilson LB.** Confirm whether `AgentStats`
  exposes integer `wins`/`games`; if only `win_rate`, either reconstruct counts
  (`round(win_rate * n)`) or switch the gate to `head_to_head` (which already
  returns `a_wins`/`games`/`a_win_rate_ci`). Prefer `head_to_head` for honest
  seat-cancellation.
- **Shaping magnitude / anti-spinout weight.** Pick a small starting
  `shaping_weight` and a low `SHAPING_SPINOUT_CAP`; ablate against degenerate
  (over-cautious) play.
- **`StrongHeuristicAgent` strength API.** Confirm the `strength=2`/`strength=3`
  constructor and picklability before wiring the broadened mix and the strong
  gate factory.
- **SB3 unshared-extractor attribute names.** Confirm `pi_features_extractor` /
  `vf_features_extractor` (vs `features_extractor`) on the built policy for the
  Idea-13-1 test.

---

## 9. Drift corrected vs the master doc / source doc

- **`share_features_extractor` citation is wrong.** Both
  `ml-improvement-sprints.md` (Idea 13 row) and `ml-performance-improvements.md`
  (Idea 13) state it "defaults to `True`, `model.py:232-239`." The symbol does
  **not** exist anywhere in `model.py`; lines 232-239 are the `policy_kwargs`
  dict that omits it, so it inherits SB3's default (`True`). The *behavioral*
  claim (shared trunk) holds; the fix is to **add** the key, not flip an
  existing one. The `policy_kwargs` block is the right edit site.
- **Gate "averages rates," doesn't pool counts.** The master doc's Idea-9 fix
  ("gate on the Wilson lower bound") needs integer `(wins, games)`, but
  `_gate_score` currently averages per-track *win-rates* and discards counts
  (`training.py:664-677`). The design therefore re-pools wins/games (or moves to
  `head_to_head`) — a detail the one-line summary glossed.
- **`gate_fn` is a `model -> float`.** Tests inject `gate_fn=lambda m: <float>`
  (`test_ml_training.py:192/246`). Returning a richer `GateResult` from the real
  gate requires a float-shim in `_gate` so those tests still pass — noted so the
  implementer doesn't break the existing suite.
- **All other line references verified accurate:** `training.py` `:648`/`:666`/
  `:778-779`/`:914-919`; `evaluate.py` `:131-132`/`:139`; `features.py`
  `_track_lookahead`/`_opponent_slots`; `spaces.py` `step_reward`;
  `wilson_interval` is indeed already imported in `evaluate.py:52`.
</content>
</invoke>
