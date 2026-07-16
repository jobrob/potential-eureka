# Design: Sprint 6A — Procedural Track Generation + Multi-Track Training/Eval

> **Status:** planning / design only — no code. Part of Sprint 6 (see
> `docs/sprint6-roadmap.md`). Buildable independently and in parallel with 6B/6C.

## Goal

Give the RL pipeline **many valid tracks** instead of the two hand-authored ones
(`tracks/usa.json`, `tracks/silverstone.json`). Deliver:

1. A **seedable procedural `Track` generator** that produces valid, raceable
   tracks within the existing `models/track.py` schema, with no engine changes.
2. **Per-episode track sampling** in `HeatEnv.reset` so training sees a
   distribution of tracks (the substrate for a track-general policy).
3. A **multi-track eval sweep** so evaluation can report per-track and aggregate
   results (consumed by 6B's cross-track eval).
4. A test that proves the **`OBS_DIM=72` invariant holds on generated tracks**
   (the contract called out in `sprint6-roadmap.md §3`).

## Motivation

`HeatEnv` hard-codes the USA track: `_default_track()` (`env.py:73-75`) loads
`"usa"`, `__init__` stores a single `self.track` (`env.py:112`), and `reset`
re-uses that same fixed track every episode (`env.py:185`,
`GameState.create(self.track, ...)`). With only two static tracks, a policy
cannot be shown to generalize, and "strong on USA" is indistinguishable from
"overfit to USA". `encode_observation` is already track-relative
(`sprint6-roadmap.md §3`; the track-lookahead block at `spaces.py:122-125`
normalizes by corner geometry), so the obs *shape* is track-agnostic by design —
the missing piece is a supply of varied tracks and a way to feed them per
episode.

## Files to create / modify

| File | Action | What |
|---|---|---|
| `src/heat/tracks/generator.py` | **new** | `generate_track(seed, params) -> Track`; `TrackGenParams` dataclass; internal helpers (corner placement, lane assignment, start grid). |
| `src/heat/tracks/validate.py` | **new** (or a `validate_track` fn inside `generator.py`) | `validate_track(track) -> None`/raises; the explicit validity contract, reusable by tests and by the generator's accept/reject loop. |
| `src/heat/ml/env.py` | **modify** | Add a track *source* (sampler) so `reset(seed=)` can draw a track per episode; keep the fixed-track path as the default for back-compat. Anchor: `__init__` `self.track` (`env.py:112`), `reset` (`env.py:170-202`). |
| `tests/test_track_generator.py` | **new** | Validity, determinism, distribution sanity. |
| `tests/test_track_validate.py` | **new** (or fold into the above) | The validator accepts the two static tracks and rejects degenerate ones. |
| `tests/test_ml_env.py` | **modify** | Add cases: env with a generated-track sampler still returns `(72,)` obs ∈ [−1, 1], terminates, is seed-deterministic. |

> No change to `models/track.py`, `tracks/loader.py`, or the `spaces`/codec
> modules. Generation targets the existing schema only.

## The `Track` schema being targeted (from `track.py`)

```python
@dataclass(frozen=True)
class Corner:   # track.py:8
    start: int; end: int; speed_limit: int          # start <= end, speed_limit > 0
@dataclass(frozen=True)
class Space:    # track.py:23
    index: int; lanes: int = 1
@dataclass
class Track:    # track.py:36
    name: str
    spaces: list[Space]
    corners: list[Corner]
    start_positions: list[int]
    laps: int = 1
    # length == len(spaces) (property, track.py:54)
    # get_corner_at(pos), spaces_in_corner(corner)  (track.py:59-68)
```

The track is a **closed loop** (the engine wraps position modulo `track.length`,
e.g. `heuristic_agent.py:46`, `rules.calculate_move_position` used at
`heuristic_agent.py:70`). The generator must respect that loop topology.

## Contracts / interfaces

### Generator

```python
# tracks/generator.py
from dataclasses import dataclass, field
from heat.models.track import Track

@dataclass(frozen=True)
class TrackGenParams:
    """Tunable bounds for procedural generation. All ranges inclusive."""
    length_range: tuple[int, int] = (50, 90)      # total spaces (cf. usa/silverstone sizes)
    num_corners_range: tuple[int, int] = (3, 7)
    speed_limit_choices: tuple[int, ...] = (1, 2, 3, 4, 5)
    laps: int = 2
    min_corner_gap: int = 4        # straight spaces required between corner ends/starts
    corner_len_range: tuple[int, int] = (1, 3)
    multi_lane_fraction: float = 0.3   # fraction of spaces with lanes >= 2
    max_lanes: int = 2
    min_start_positions: int = 6        # == MAX_PLAYERS (spaces.py:47)

def generate_track(seed: int, params: TrackGenParams | None = None,
                   *, name: str | None = None) -> Track:
    """Return a valid Track deterministically from (seed, params).

    Uses a private ``random.Random(seed)`` only — never the global RNG — so it is
    reproducible and parallel-safe (mirrors GameState.create's RNG policy,
    game_state.py:209-214). Internally generate-then-validate with bounded
    retries (re-draw on validation failure); raise if it cannot produce a valid
    track within the retry budget (signals params are over-constrained)."""
```

### Validator (the validity contract)

```python
# tracks/validate.py
def validate_track(track: Track) -> None:
    """Raise TrackValidationError if `track` is not raceable. Checks:
      1. length >= some floor and == len(spaces); space indices are 0..length-1.
      2. laps >= 1.
      3. corners: 0 <= start <= end < length; speed_limit > 0; no overlap;
         gap between consecutive corners (wrap-aware) >= min straight; corners
         do not cover the whole loop (a raceable straight must exist).
      4. start_positions: count >= MAX_PLAYERS; all in [0, length); distinct;
         none placed inside a corner (so the start grid is on a straight).
      5. lanes: every space lanes >= 1.
    Pure; no I/O. Reused by the generator's accept/reject loop and by tests."""
```

> **Validity constraints rationale (why these, not arbitrary).**
> - *Corner spacing / no full-loop corner* → guarantees a "straight" exists so
>   cars are not perpetually paying corner heat (would make every game a heat
>   death). The heuristic and the engine both assume corners are discrete
>   features (`get_corner_at`, `track.py:59`).
> - *`start_positions` ≥ `MAX_PLAYERS` and on a straight* → all seats can be
>   placed (`game_state.py:225`) without two cars colliding at a corner on turn 0.
> - *`speed_limit > 0`* → matches `Corner` semantics (`track.py:16`) and keeps
>   the obs's `speed_limit / maxlimit` normalization (`spaces.py:122-125`) finite.
> - *`length` floor / `laps ≥ 1`* → keeps the obs's `position/length` and
>   `lap/laps` normalizations (`spaces.py:118-121`, see roadmap §3) in `[−1, 1]`
>   and non-degenerate (no divide-by-zero).

### Env track sampling

The minimal, back-compat-preserving change: let `HeatEnv` accept an optional
**track source** alongside the existing fixed `track` arg.

```python
# ml/env.py (sketch — extends the existing __init__/reset, env.py:94-202)
TrackSource = Track | Callable[[int | None], Track]   # fixed track OR a sampler(seed)->Track

class HeatEnv(gym.Env):
    def __init__(self, track: TrackSource | None = None, ...):
        # if track is callable -> store as self._track_source (sampler)
        # else -> keep current fixed-track behavior (env.py:112), self._track_source = None
        ...
    def reset(self, *, seed=None, options=None):
        if self._track_source is not None:
            self.track = self._track_source(seed)   # per-episode track
        # else: self.track stays fixed (current behavior)
        self.state = GameState.create(self.track, self.num_players, seed=seed)
        ...   # unchanged from env.py:185-202
```

A convenience sampler bridges the generator to the env:

```python
# tracks/generator.py
def track_sampler(params: TrackGenParams | None = None, *, base_seed: int = 0
                  ) -> Callable[[int | None], Track]:
    """Return a sampler(seed) -> generate_track(derived_seed). When the env
    passes its episode seed, the track is a deterministic function of it, so
    reset(seed) reproduces both the track AND the deck shuffle."""
```

> **Why a sampler, not a list of pre-generated tracks:** the env's determinism
> contract keys everything on `reset(seed)` (`env.py:183-187`). A sampler that
> derives the track from the same seed keeps "same seed → same episode" intact,
> including for 6C's vectorized envs (each worker derives its own track from its
> own seed stream). A fixed pre-generated *set* is still supported trivially by a
> sampler that indexes into it by `seed % len(set)`.

## Build sequence

1. **`validate_track`** + `TrackValidationError`. Unit-test it against the two
   static tracks (must pass) and hand-built degenerate tracks (must raise).
2. **`generate_track`** (generate-then-validate with bounded retries) +
   `TrackGenParams`. Test determinism and validity over many seeds.
3. **`track_sampler`** convenience wrapper.
4. **`HeatEnv` track-source** support (back-compat: callable → sample, else
   fixed). Extend `test_ml_env.py`.
5. **Obs-invariant test**: generated tracks → `encode_observation` shape `(72,)`,
   dtype float32, values ∈ [−1, 1].
6. Run the full suite: `PYTHONPATH=src python -m pytest tests/ -q`.

## Test gates (`test_track_generator.py`, `test_track_validate.py`, `test_ml_env.py`)

- **Validity:** `validate_track(generate_track(s))` never raises for many seeds
  `s`; the two static tracks pass the validator.
- **Determinism:** `generate_track(s)` == `generate_track(s)` (same `name`,
  `spaces`, `corners`, `start_positions`, `laps`) — and uses only its private
  RNG (seed the global `random` differently between the two calls and assert the
  result is unchanged, proving no global-RNG leak).
- **Distribution sanity:** over N seeds, generated lengths/corner-counts span
  their configured ranges (not all identical); no track violates `min_corner_gap`.
- **Schema conformance:** space indices are `0..length-1`; corners satisfy
  `0 <= start <= end < length`; `len(start_positions) >= MAX_PLAYERS`.
- **Obs invariant (the contract gate):** for many generated tracks, build a
  `HeatEnv` (or call `encode_observation` directly) and assert obs shape `(72,)`,
  dtype float32, all values in `[−1, 1]` — at reset *and* mid-episode (after a
  corner-lookahead is non-trivial).
- **Env integration:** an env with a `track_sampler` source resets and steps to
  termination; `reset(seed)` twice with the same seed yields the same track and
  the same first obs (sampler determinism end-to-end).
- **Full suite green** (the per-step convention).

## Risks + de-risking

| Risk | De-risking |
|---|---|
| Generated track breaks an unstated engine assumption (e.g. corner at the start line, zero-straight loop) → silent bad training or a driver crash. | The validator encodes those assumptions explicitly and is run *inside* the generator's accept/reject loop; the static-track-passes test pins the validator to known-good geometry. |
| Generator over-constrained → infinite retry / raise. | Bounded retry budget with a clear error; `TrackGenParams` defaults are chosen with slack (length 50-90, 3-7 corners, gap 4) so the accept rate is high; a test asserts the default params produce a valid track for a range of seeds. |
| Obs normalization divides by a generated zero (e.g. `maxlimit`, `track.laps`). | Validator guarantees `laps >= 1`, `speed_limit > 0`, `length >= floor`; the obs-invariant test is the backstop. If a normalization constant (e.g. "maxlimit") is a hard-coded literal in `features.py`, 6A must confirm generated `speed_limit` choices stay within it (note in Open Questions). |
| Per-episode track sampling changes env determinism for existing callers. | The sampler path is **opt-in**: a non-callable `track` keeps the exact current fixed-track behavior (`env.py:112`, `env.py:185`); existing tests are unaffected. |
| Multi-lane / `start_positions` interaction with engine blocking/slipstream. | Keep `max_lanes` small (default 2) and start positions on a straight; generation does not invent new mechanics, only new layouts within the proven schema. |

## Non-goals

- **No learned / GAN track generation** — rule-based procedural only.
- **No new `Track` fields or engine mechanics** — strictly the existing schema.
- **No persistence format change** — generated tracks live in memory; optionally
  a helper could dump one to JSON via the existing `loader.py` shape, but that is
  not required and not a gate.
- **No curriculum over track difficulty** (e.g. easy→hard scheduling). 6A
  supplies the *distribution*; any curriculum is a 6C/capstone concern.
- **No change to the obs/action contract** — proving the invariant holds is the
  job; changing it is explicitly out of scope (`sprint6-roadmap.md §3`).

## Open questions / assumptions

- **Normalization constants in `features.py`.** The roadmap establishes the obs
  is track-relative; 6A must confirm the *exact* normalizers (e.g. whether
  "next speed_limit" is divided by a literal max or by a track-derived max) so
  `speed_limit_choices` stay in-range. If a literal cap exists, clamp
  `speed_limit_choices` to it. (Verify against `features.py` track-lookahead
  block when implementing.)
- **Default `laps` for generated tracks.** Assumed 2 (longer episodes → more
  reward signal). Static tracks use their JSON `laps`; the generator default is
  independent and tunable via `TrackGenParams.laps`.
- **Track variety vs `num_players`.** Generation guarantees ≥ `MAX_PLAYERS` start
  positions so any 2-6 player game fits; whether to *vary* `num_players` per
  episode is a 6C training-config decision, not 6A's.
