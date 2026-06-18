# Sprint B — Cheap whole-track obs (Option A) + curriculum baseline (implementation design)

> **Status:** design only — **no code yet**. Expands the Sprint B summary row of
> `docs/ml-improvement-sprints.md`.
> **Scope:** Idea 1 **Option A** (all-corners fixed-slot ego-centric obs + plain
> MLP; small `OBS_DIM` bump; `CODEC_VERSION` 1→2) and Idea 2 (step-aware
> track-difficulty curriculum).
> **Goal:** establish the held-out generalization baseline that Option B
> (Sprint C) would have to beat — the cheapest test of the core hypothesis
> ("the agent is blind to the track").
> **Depends on Sprint A** (`docs/sprint-A-trustworthy-run.md`): the trustworthy
> gate is what makes B's number believable.

---

## 1. Goal

Give the policy the **whole track** every step at the cheapest possible cost: a
fixed-slot, ego-centric, all-corners observation block processed by the *existing
MLP* — no CNN, no attention, no per-step caching problem. The generator caps
corners at 7 (`generator.py:46`, `num_corners_range=(3, 7)`), so with
`MAX_CORNERS=8` the padded slots encode **every corner of every generated
track**, and the inter-corner distances encode the straights. This is the
controlled experiment for root-cause #1; its held-out win-rate becomes the
**Option-A baseline** that decides whether Sprint C (Option B) is worth building.

Pair it with a **step-aware difficulty curriculum** (Idea 2): start narrow
(short tracks, few corners, 1 lap, gentle limits) and widen to the full
distribution over the first N steps. The master doc's key finding — verified
below — is that `TrackSampler` **freezes its `TrackGenParams` at construction**
(`generator.py:270-283`), so this needs a *stateful, step-aware* schedule, not a
config tweak.

---

## 2. Verified code baseline (file:line)

| Symbol | Location | Current state |
|---|---|---|
| `CODEC_VERSION` | `spaces.py:35` | `= 1`. Bump to `2` this sprint. |
| `OBS_DIM` | `spaces.py:42` | `= 72`. |
| `BLOCK_TRACK_LOOKAHEAD` | `spaces.py:56` | `= 4` (the 4-dim next-corner block being **replaced**). |
| `BLOCK_PHASE_CONTEXT` | `spaces.py:63-72` | **derived** as `OBS_DIM - (sum of all other blocks)` → currently 19. This auto-absorbs any tail; see §3.1 caveat. |
| `observation_space()` | `spaces.py:77-84` | `Box(-1, 1, (OBS_DIM,), float32)`. |
| `_track_lookahead` | `features.py:108-128` | the 4-dim block to replace: dist-to-next-corner/len, next speed_limit/max, current-space lanes, in-corner flag. |
| `encode_observation` | `features.py:224-254` | packs blocks in frozen order (`:239-246`), asserts `(OBS_DIM,)`, clips to `[-1, 1]`. |
| `distance_to_next_corner` | `rules.py:260-288` | `(track, position) -> (Corner|None, dist)`; dist wraps, a corner at `position` counts as a full lap away. |
| `Corner` | `track.py:8-20` | `start`, `end`, `speed_limit` (frozen). `len = end - start + 1`. |
| `Track.length` / `.laps` / `.corners` / `.get_corner_at` | `track.py:54-64` | track geometry the new block reads. |
| `TrackGenParams` | `generator.py:34-92` | **frozen** dataclass: `length_range=(50,90)`, `num_corners_range=(3,7)`, `laps=2`, `corner_len_range=(1,3)`, `speed_limit_choices=(1..5)`, etc. |
| `generate_track` | `generator.py:203-248` | `(seed, params, *, name, max_retries) -> Track`; private `random.Random(seed)`. |
| `TrackSampler` | `generator.py:251-283` | `__init__(self, params=None, *, base_seed=0)` stores `self.params` **once**; `__call__(seed)` builds with that frozen `params`. **No step awareness.** |
| `_resolve_track_source` | `training.py:589-599` | `None -> TrackSampler(base_seed=seed)`. |
| `HeatEnv` track source | `env.py:73`, `env.py:128-133`, `env.py:210-211` | `TrackSource = Track | Callable[[int|None], Track]`; `reset` calls `self._track_source(seed)`. |
| `HeatMLPExtractor` | `model.py:178-208` | flat-obs MLP; `in_dim = observation_space.shape[0]` — **auto-adapts to a larger `OBS_DIM`**, no architecture edit needed for Option A. |

---

## 3. Idea 1 Option A — all-corners ego-centric obs

### 3.1 New observation layout

Replace `BLOCK_TRACK_LOOKAHEAD` (4 dims) with a **track block** =
`MAX_CORNERS` ego-centric corner slots + a small globals sub-block.

**Per-corner slot (4 floats each), ego-centric (ordered by distance ahead from
the learner's current position, wrapping forward):**

| Field | Normalization | Notes |
|---|---|---|
| `dist_ahead` | `dist / track.length` → `[0,1]` | distance from current position to the corner start, forward/wrapping (same convention as `distance_to_next_corner`). |
| `speed_limit` | `corner.speed_limit / max_limit` → `[0,1]` | `max_limit = max(c.speed_limit for c in corners)`. |
| `corner_len` | `(end - start + 1) / max_corner_len` → `[0,1]` | or `/ track.length`; pick one and freeze (Open Questions). |
| `lanes` | `lanes_at(corner.start) / max_lanes` → `[0,1]` | lanes at the corner entry. |

Absent slots (tracks with < `MAX_CORNERS` corners) are **zero-filled**, with the
`dist_ahead` field doubling as the padding/presence signal (0 → "no corner
here"). Because every real corner has `dist_ahead > 0` (a corner at the current
position reads as a full lap away, `rules.py:281-282`), zero is an unambiguous
padding marker. *Decision (Open Questions): add an explicit per-slot presence
bit (5th float) vs. rely on the `dist_ahead==0` convention.* The §4 padding test
freezes whichever choice is made.

**Globals sub-block (4 floats), included regardless:**

| Field | Normalization |
|---|---|
| `laps_remaining` | `(track.laps - player.lap) / track.laps` (or `+1` lap-aware) → `[0,1]` |
| `dist_to_finish` | lap-aware spaces remaining `/ (track.length * track.laps)` → `[0,1]` |
| `heat_pool` | `player.heat_available / rules.HEAT_POOL_SIZE` → `[0,1]` |
| `pos_in_lap` | `player.position / track.length` → `[0,1]` |

> Note `heat_pool` and `pos_in_lap` partly duplicate `_own_kinematics`
> (`features.py:74-83`). Keep them in the track block anyway (the master doc's
> globals list is explicit) **or** drop the duplicates to save 2 dims — decide at
> build time and freeze. The block-size constants make either choice a one-line
> edit.

**New block size:** `BLOCK_TRACK = MAX_CORNERS * CORNER_SLOT_FLOATS + TRACK_GLOBALS`.
With `MAX_CORNERS=8`, `CORNER_SLOT_FLOATS=4`, `TRACK_GLOBALS=4`:
`BLOCK_TRACK = 8*4 + 4 = 36` (vs. the old 4).

**New `OBS_DIM`:** `72 - 4 + 36 = 104`. (If a per-slot presence bit is added →
`CORNER_SLOT_FLOATS=5` → `BLOCK_TRACK = 44` → `OBS_DIM = 112`.) Pick `MAX_CORNERS`
and the slot width at build time; the resulting `OBS_DIM` is the frozen v2
contract.

**Critical caveat — `BLOCK_PHASE_CONTEXT` is derived, not literal.** In
`spaces.py:63-72`, `BLOCK_PHASE_CONTEXT = OBS_DIM - (all other blocks)`. If we
bump `OBS_DIM` and add `BLOCK_TRACK` but *forget* to subtract `BLOCK_TRACK` in
that derivation, the phase-context block silently absorbs the difference and the
obs layout corrupts. The edit **must**:

1. Replace `BLOCK_TRACK_LOOKAHEAD` with `BLOCK_TRACK` (and the corner/globals
   sub-constants).
2. Subtract `BLOCK_TRACK` in the `BLOCK_PHASE_CONTEXT` derivation.
3. Set `OBS_DIM` to the new total so `BLOCK_PHASE_CONTEXT` stays at its intended
   value (e.g. 19), not a number that quietly drifts.

Keep the existing `assert BLOCK_PHASE_CONTEXT >= 0` and add an assert that
`BLOCK_PHASE_CONTEXT` equals the intended literal, so a miscount fails fast.

### 3.2 `spaces.py` changes

```python
CODEC_VERSION: int = 2                 # was 1  (§3.1)

MAX_CORNERS: int = 8                   # generator caps at 7; +1 slack
CORNER_SLOT_FLOATS: int = 4            # (dist_ahead, speed_limit, corner_len, lanes)
TRACK_GLOBALS: int = 4                 # laps_remaining, dist_to_finish, heat_pool, pos_in_lap
BLOCK_TRACK: int = MAX_CORNERS * CORNER_SLOT_FLOATS + TRACK_GLOBALS   # 36

OBS_DIM: int = 104                     # 72 - BLOCK_TRACK_LOOKAHEAD(4) + BLOCK_TRACK(36)

# Remove BLOCK_TRACK_LOOKAHEAD; in the BLOCK_PHASE_CONTEXT derivation,
# subtract BLOCK_TRACK instead of BLOCK_TRACK_LOOKAHEAD.
BLOCK_PHASE_CONTEXT: int = (
    OBS_DIM
    - BLOCK_HAND_HISTOGRAM
    - BLOCK_OWN_GEAR
    - BLOCK_OWN_KINEMATICS
    - BLOCK_DECK_COMPOSITION
    - BLOCK_TRACK
    - BLOCK_ADRENALINE_CONTEXT
    - BLOCK_OPPONENT_SLOTS
)
assert BLOCK_PHASE_CONTEXT == 19, BLOCK_PHASE_CONTEXT   # freeze the intended size
```

`ACTION_DIM` and the action layout are **unchanged**. `observation_space()`
auto-picks up the new `OBS_DIM`.

### 3.3 `features.py` changes

Replace `_track_lookahead` (`features.py:108-128`) with `_track_block`, and swap
the call in `encode_observation` (`features.py:243`).

```python
def _track_block(player: PlayerState, track) -> list[float]:
    """MAX_CORNERS ego-centric corner slots + globals (replaces the old 4-dim
    next-corner lookahead). Ego-centric: corners ordered by forward distance from
    the learner's current position, wrapping; absent slots zero-filled."""
    length = track.length or 1
    pos = player.position
    corners = list(track.corners)
    max_limit = max((c.speed_limit for c in corners), default=1) or 1
    max_clen = max(((c.end - c.start + 1) for c in corners), default=1) or 1
    max_lanes = max((s.lanes for s in track.spaces), default=1) or 1

    # Forward distance to each corner start (wrap-aware), matching
    # rules.distance_to_next_corner's convention (a corner at pos == one lap).
    def fwd_dist(c):
        d = (c.start - pos) % length
        return length if d == 0 else d
    ordered = sorted(corners, key=fwd_dist)

    slots: list[float] = []
    for i in range(spaces.MAX_CORNERS):
        if i < len(ordered):
            c = ordered[i]
            d = fwd_dist(c)
            entry_lanes = track.spaces[c.start].lanes if 0 <= c.start < length else 1
            slots += [
                _clip01(d / length),
                _clip01(c.speed_limit / max_limit),
                _clip01((c.end - c.start + 1) / max_clen),
                _clip01(entry_lanes / max_lanes),
            ]
        else:
            slots += [0.0, 0.0, 0.0, 0.0]   # padding

    laps = track.laps or 1
    laps_remaining = _clip01((laps - player.lap) / laps)
    total_len = length * laps
    abs_pos = player.lap * length + pos
    dist_to_finish = _clip01((total_len - abs_pos) / total_len)
    heat = _clip01(player.heat_available / rules.HEAT_POOL_SIZE)
    pos_in_lap = _clip01(pos / length)
    globals_ = [laps_remaining, dist_to_finish, heat, pos_in_lap]

    return slots + globals_
```

`encode_observation` (`features.py:238-246`):

```python
values += _deck_composition(player)            # 6
values += _track_block(player, track)          # BLOCK_TRACK  (was _track_lookahead, 4)
values += _adrenaline_context(state, player)   # 2
```

All fields are pre-clipped to `[0,1]`; the function's tail `np.clip(vec, -1, 1)`
(`features.py:253`) remains the safety net. The block is pure-Python list-building
exactly like the existing blocks, so it adds negligible per-step cost — **no
caching needed** (that is Idea 14, a Sprint-C concern; Option A's block is cheap
enough to recompute every step).

### 3.4 Why the MLP needs no change

`HeatMLPExtractor.__init__` reads `in_dim = int(observation_space.shape[0])`
(`model.py:196`), so a larger `OBS_DIM` flows through automatically. `NET_PROFILES`
sizes (`model.py:103-106`) are unaffected. Sprint B stays an MLP-only experiment
by construction.

---

## 4. Idea 1 test plan (new feature test)

Add `tests/test_ml_features_track_block.py` (or extend `test_ml_features.py`)
covering the new block specifically on **generated** tracks:

- **Shape/dtype/bounds (codec v2):** for many `generate_track(seed)` tracks and
  random player states, `encode_observation` returns `(OBS_DIM,)`, `float32`,
  all values in `[-1, 1]`, no NaN — the v2 analogue of
  `test_ml_features.py:67` (`TestShapeAndBounds`).
- **Ego-centric ordering:** the corner slots are sorted by forward distance from
  the player's position; advancing the player past the nearest corner rotates the
  slots (slot 0's `dist_ahead` jumps to the *next* corner). Assert monotonic
  non-decreasing `dist_ahead` across populated slots.
- **Padding correctness:** a track with `k < MAX_CORNERS` corners zero-fills
  slots `k..MAX_CORNERS-1` (every field 0.0 / presence bit 0); a 7-corner track
  (the generator max) fills 7 slots and pads 1.
- **All-corners coverage:** for a generated track, every corner appears in
  exactly one populated slot (no corner dropped, none duplicated) — the
  whole-track guarantee Option A is bought for.
- **Globals correctness:** `laps_remaining` decreases as `player.lap`
  increases; `dist_to_finish` hits ~0 near the final-lap finish; `heat_pool`
  matches `heat_available / HEAT_POOL_SIZE`.
- **Determinism / no-mutation:** same `(state, pid, decision)` → identical
  vector; `encode_observation` does not mutate `state` (mirror
  `test_ml_features.py:82-90`).
- **Block-accounting guard:** assert `BLOCK_PHASE_CONTEXT == 19` and
  `OBS_DIM == BLOCK_HAND_HISTOGRAM + ... + BLOCK_TRACK + ... + BLOCK_PHASE_CONTEXT`
  so a future miscount fails in CI (guards the §3.1 derived-block footgun).

---

## 5. Idea 2 — step-aware difficulty curriculum

### 5.1 Problem (verified)

`TrackSampler` stores `self.params` once in `__init__` (`generator.py:273`) and
every `__call__(seed)` builds with that **frozen** `params` (`generator.py:283`).
There is no step counter and no way to widen the distribution over training. A
config tweak cannot express "easy → full over N steps"; it needs a **stateful,
step-aware sampler** whose effective `TrackGenParams` interpolate with the global
training step.

### 5.2 The schedule object

A new `CurriculumSchedule` (in `generator.py` next to `TrackSampler`, or a small
`heat/tracks/curriculum.py`) interpolates the tunable bounds from an "easy"
endpoint to the full default across a step horizon:

```python
@dataclass(frozen=True)
class CurriculumSchedule:
    """Interpolate TrackGenParams from `easy` to `full` over `horizon_steps`.

    `frac = clip(step / horizon_steps, 0, 1)` (linear; see Open Questions for
    staged). Integer range bounds are rounded; `laps` steps up at thresholds.
    """
    easy: TrackGenParams
    full: TrackGenParams
    horizon_steps: int

    def params_at(self, step: int) -> TrackGenParams:
        frac = 0.0 if self.horizon_steps <= 0 else min(1.0, max(0.0, step / self.horizon_steps))
        def lerp_range(a, b):
            return (round(a[0] + (b[0]-a[0])*frac), round(a[1] + (b[1]-a[1])*frac))
        return dataclasses.replace(
            self.full,
            length_range=lerp_range(self.easy.length_range, self.full.length_range),
            num_corners_range=lerp_range(self.easy.num_corners_range, self.full.num_corners_range),
            corner_len_range=lerp_range(self.easy.corner_len_range, self.full.corner_len_range),
            laps=round(self.easy.laps + (self.full.laps - self.easy.laps) * frac),
        )
```

Default endpoints (tunable, Open Questions):
- `easy`: `length_range=(30,40)` *(must stay ≥ `MIN_TRACK_LENGTH`; verify —
  `TrackGenParams.__post_init__` enforces `length_range[0] >= MIN_TRACK_LENGTH`,
  `generator.py:64-68`)*, `num_corners_range=(2,3)`, `laps=1`,
  `corner_len_range=(1,1)`, gentler `speed_limit_choices` (e.g. `(2,3,4)`).
- `full`: the default `TrackGenParams()`.

### 5.3 Threading the step counter into sampling

`TrackSampler.__call__(seed)` (`generator.py:277-283`) takes only a seed — it has
no idea what training step it is. Two viable wirings; pick one at build time:

**Option (i) — stateful sampler with a shared step counter (recommended).**
Add a `StepAwareTrackSampler` that holds a `CurriculumSchedule` and a mutable
current-step value, and rebuilds `params` per call:

```python
class StepAwareTrackSampler:
    def __init__(self, schedule, *, base_seed=0):
        self.schedule = schedule
        self.base_seed = base_seed
        self._step = 0                 # updated by the training loop
        self._fallback_rng = random.Random(base_seed)
    def set_step(self, step: int) -> None:
        self._step = int(step)
    def __call__(self, seed):
        params = self.schedule.params_at(self._step)
        track_seed = (self._fallback_rng.randrange(2**31) if seed is None
                      else (int(seed) + self.base_seed) % (2**31))
        return generate_track(track_seed, params)
```

The cross-process catch (the master doc's noted risk): under `SubprocVecEnv` the
sampler is pickled into workers, so a `set_step` on the main-process object does
**not** reach them. Mitigations (decide at build time):
- Drive the schedule by *episode seed bands* instead of a live counter: derive
  `frac` from the seed the env passes (monotonic-ish with wall progress) — fully
  stateless, picklable, but only approximately step-aligned.
- Or rebuild the vec env at curriculum stage boundaries (like the Phase-2 env
  swap, `training.py:882-892`): finite stages, each a fresh sampler pinned to that
  stage's `params`. Coarser but robust and process-safe.

`StepAwareTrackSampler` must remain **picklable** (top-level class, no closures)
exactly like `TrackSampler` (`generator.py:251-261` documents this constraint).

**Option (ii) — staged env rebuilds.** Define K stages; at each stage boundary
rebuild the training vec env with `TrackSampler(params=schedule.params_at(stage))`.
Reuses the existing `_swap_vec_env` machinery and sidesteps the cross-process
counter entirely. Simpler and process-safe; coarser granularity.

### 5.4 `CurriculumConfig` + `train_self_play` wiring

New `CurriculumConfig` fields (`training.py:305-412`):

```python
#: Enable the step-aware track-difficulty curriculum (Idea 2). Off by default so
#: existing generated-track runs are unchanged.
use_track_curriculum: bool = False
#: Steps over which difficulty ramps easy -> full (the schedule horizon).
curriculum_horizon_steps: int = 1_000_000
#: Curriculum shape: "linear" (default) or "staged".
curriculum_shape: str = "linear"
#: Number of stages when curriculum_shape == "staged".
curriculum_stages: int = 4
```

`_resolve_track_source` (`training.py:589-599`) gains a curriculum branch: when
`use_track_curriculum` and `track is None`, return a `StepAwareTrackSampler`
(Option i) or build the staged rebuild plan (Option ii) instead of a plain
`TrackSampler`. The Phase-1 chunked learn loop from **Sprint A** (Idea 7) is the
natural place to call `sampler.set_step(total_steps_so_far)` between chunks
(Option i) or to trigger a stage rebuild (Option ii) — another reason A lands
first.

### 5.5 Idea 2 test plan

Add `tests/test_track_curriculum.py`:

- **Endpoint correctness:** `schedule.params_at(0) == easy`-equivalent bounds;
  `params_at(horizon_steps) == full` (the default `TrackGenParams`);
  `params_at(2*horizon)` clamps to `full`.
- **Monotone interpolation:** `num_corners_range[1]` and `length_range[1]` are
  non-decreasing in `step`; `laps` non-decreasing.
- **Validity at every fraction:** for `frac` in `{0, .25, .5, .75, 1}`, the
  interpolated `TrackGenParams` constructs without raising
  (`__post_init__` guards, `generator.py:61-92`) and `generate_track` yields a
  valid track — guards against an interpolated `length_range[0] <
  MIN_TRACK_LENGTH` or an inverted range.
- **Picklability:** `pickle.loads(pickle.dumps(StepAwareTrackSampler(...)))`
  round-trips and still samples (the `SubprocVecEnv` requirement).
- **Step plumbing:** after `set_step(k)`, `__call__` draws from `params_at(k)`
  (assert via corner-count distribution over many seeds, or by spying
  `params_at`).

**Suite command:** `PYTHONPATH=src python -m pytest tests/ -q`

---

## 6. Task checklist (in order)

1. `spaces.py`: add `MAX_CORNERS`/`CORNER_SLOT_FLOATS`/`TRACK_GLOBALS`/
   `BLOCK_TRACK`; remove `BLOCK_TRACK_LOOKAHEAD`; fix the `BLOCK_PHASE_CONTEXT`
   derivation (subtract `BLOCK_TRACK`); set the new `OBS_DIM`; bump
   `CODEC_VERSION` to 2; add the `BLOCK_PHASE_CONTEXT == 19` assert.
2. `features.py`: implement `_track_block`; replace the `_track_lookahead` call
   in `encode_observation`; delete `_track_lookahead`.
3. Add the Idea-1 feature test (§4); run the suite — the existing
   `test_ml_features.py` shape/bounds tests should pass against the new `OBS_DIM`.
4. `generator.py` (or `heat/tracks/curriculum.py`): add `CurriculumSchedule` and
   `StepAwareTrackSampler` (Option i) **or** the staged-rebuild plan (Option ii),
   keeping everything top-level/picklable.
5. `training.py`: add the four curriculum `CurriculumConfig` fields; branch
   `_resolve_track_source` on `use_track_curriculum`; call `set_step` /
   stage-rebuild from the Sprint-A Phase-1 chunk loop.
6. Add the Idea-2 curriculum tests (§5.5).
7. Write the Sprint-B launch config: Option-A obs (automatic via v2),
   curriculum on, Sprint-A trustworthy gate on, `normalize_obs=False`,
   `phase1_steps == total_timesteps`. Run a medium-length training job; record
   the held-out generalist win-rate vs strong heuristics (Wilson-LB, from
   Sprint A's gate) as the **Option-A baseline**.
8. Full suite green: `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 7. Definition of done

- The v2 obs encodes **all** corners of every generated track (≤7) ego-centric +
  padded, processed by the existing MLP; feature tests (shape/bounds/ego-centric/
  padding/coverage/block-accounting) green.
- `CODEC_VERSION == 2`; the `MLAgent` version tripwire rejects v1 checkpoints
  (intended — they're throwaway).
- The step-aware curriculum ramps easy→full, is picklable, produces valid tracks
  at every fraction, and is wired into `train_self_play`; curriculum tests green.
- A medium-length run (curriculum on, **Sprint-A gate on**) reports a held-out
  generalist win-rate vs strong heuristics, **recorded as the Option-A baseline.**
  Decision gate: clears target decisively → Sprint C may be unnecessary; plateaus
  below target → Sprint C is justified *with a baseline to beat.*
- Full suite green.

---

## 8. Ordering & dependencies

- **Sprint A must land first.** Sprint B's headline number is the held-out
  win-rate under Sprint A's Wilson-LB gate vs the strong opponent; without A the
  number is selected on noise vs the wrong opponent and is uninterpretable. The
  curriculum step plumbing (§5.4) also hangs off A's Phase-1 chunk loop.
- **Single codec bump.** This is the only `CODEC_VERSION` change in the cheap
  path. If Sprint C is **not** needed (A clears target), the obs/codec companions
  (slipstream features, etc.) were never paid for. If C *is* escalated, those
  ride C's bump (v2 if C immediately follows, else v3) — never a second bump in
  this sprint.

---

## 9. Open questions (decide at build time)

- **`MAX_CORNERS` and slot width → final `OBS_DIM`.** `MAX_CORNERS=8` (gen caps
  at 7, +1 slack); 4 vs 5 floats/slot (explicit presence bit?) → `OBS_DIM` 104
  vs 112. Freeze before training.
- **`corner_len` normalization:** `/ max_corner_len` (relative) vs `/ track.length`
  (absolute-ish). Pick one and freeze in the test.
- **Globals duplication:** keep `heat_pool`/`pos_in_lap` in both the kinematics
  and track blocks, or drop the duplicates to save 2 dims.
- **Curriculum schedule shape & horizon:** linear vs staged; `horizon_steps`
  relative to `phase1_steps`. Ablate against no-curriculum once Sprint A's gate
  is trustworthy.
- **Curriculum step-plumbing across `SubprocVecEnv`:** live `set_step` (Option i,
  needs a process-safe step signal) vs staged env rebuilds (Option ii, coarser
  but robust). Recommended start: staged rebuilds for correctness, revisit if
  granularity matters.
- **Easy endpoint floor:** ensure `easy.length_range[0] >= MIN_TRACK_LENGTH`
  (`generator.py:64-68`) and that easy params keep the generate-then-validate
  accept rate high.

---

## 10. Drift corrected vs the master doc / source doc

- **`BLOCK_PHASE_CONTEXT` is derived, not a literal.** The master doc says "bump
  `OBS_DIM`, block sizes." Verified: `BLOCK_PHASE_CONTEXT = OBS_DIM - Σ(other
  blocks)` (`spaces.py:63-72`). Bumping `OBS_DIM`/adding the track block without
  subtracting it in that derivation silently corrupts the layout. The design
  makes this explicit and adds a CI assert — a footgun the summary row omitted.
- **`TrackSampler` freeze confirmed and located precisely.** `self.params` is
  set once in `__init__` (`generator.py:273`); `__call__` uses it unchanged
  (`generator.py:283`). The master doc's claim holds; the design also surfaces
  the **`SubprocVecEnv` pickling** consequence (a live step counter doesn't reach
  workers), which the master doc only hinted at — hence the two wiring options.
- **No MLP/extractor edit needed for Option A.** `HeatMLPExtractor` reads
  `observation_space.shape[0]` (`model.py:196`), so the larger `OBS_DIM` flows
  through with zero `model.py` changes — confirming Option A is genuinely
  MLP-only (no CNN, no Idea-14 caching).
- **All other references verified:** `generator.py:46` `num_corners_range=(3,7)`;
  `spaces.py:35/42/56`; `features.py:108-128`/`:224-254`; `rules.py:260-288`
  distance convention; `track.py` `Corner`/`length`.
</content>
</invoke>
