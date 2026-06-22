"""Seedable procedural :class:`Track` generation (Sprint 6A).

:func:`generate_track` produces a valid, raceable track deterministically from a
seed, targeting only the existing :mod:`heat.models.track` schema -- no engine
or codec changes. It uses a *private* ``random.Random(seed)`` (never the global
RNG), mirroring ``GameState.create``'s RNG policy, so generation is reproducible
and parallel-safe. Generation is generate-then-validate with bounded retries:
each attempt re-draws from the seeded stream; the result is checked against
:func:`heat.tracks.validate.validate_track` and re-drawn on failure, raising if
no valid track is produced within the retry budget (a signal that the params are
over-constrained).

:func:`track_sampler` bridges the generator to :class:`~heat.ml.env.HeatEnv`:
it returns a ``sampler(seed) -> Track`` so the env's ``reset(seed)`` draws a
track that is a deterministic function of the episode seed, keeping
"same seed -> same episode" intact (including for vectorized envs, where each
worker derives its own track from its own seed stream).
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Callable

from heat.tracks.validate import MIN_TRACK_LENGTH, validate_track
from heat.models.track import Corner, Space, Track
from heat.ml.spaces import MAX_PLAYERS

#: Default bounded retry budget for generate-then-validate.
_MAX_RETRIES: int = 200

#: Track-seed space modulus. Generated track seeds live in ``[0, 2**31)`` to
#: match ``random.Random``'s comfortable integer range.
_TRACK_SEED_MOD: int = 2**31

#: Boundary that partitions the track-seed space into two structurally-disjoint
#: regions (code-review 2026-06-22 #6, "seed-band leakage"):
#:
#:   * the LOW region ``[0, _NAMESPACE_SPLIT)`` is the ``base_seed == 0``
#:     *identity / eval namespace*: ``track_seed = seed % _NAMESPACE_SPLIT``.
#:     The held-out eval band (``eval_search._HELDOUT_BASE = 900_000`` + offsets,
#:     used as direct ``generate_track`` seeds) lives here and is reachable ONLY
#:     from ``base_seed == 0``.
#:   * the HIGH region ``[_NAMESPACE_SPLIT, _TRACK_SEED_MOD)`` is where EVERY
#:     non-zero ``base_seed`` (training / self-play namespaces) lands, via a
#:     stable hash fold of ``(base_seed, seed)``. Because every training seed is
#:     ``>= _NAMESPACE_SPLIT`` and the held-out band is far below it
#:     (``900_000 << 2**30``), no training campaign can EVER regenerate a
#:     held-out eval track -- the disjointness is by construction, not by hoping
#:     additive seeds don't collide.
#:
#: 2**30 leaves ~1.07e9 identity seeds (the eval/base-0 stream) and ~1.07e9
#: training seeds, both far larger than any campaign needs.
_NAMESPACE_SPLIT: int = 2**30


def _namespaced_track_seed(base_seed: int, seed: int) -> int:
    """Map ``(base_seed, episode_seed)`` to a track seed with namespace disjointness.

    The disjointness guarantee (code-review 2026-06-22 #6):

      * ``base_seed == 0`` is the *identity / eval* namespace and returns
        ``seed % _NAMESPACE_SPLIT`` -- so ``base_seed=0`` reproduces the legacy
        "track is a deterministic function of the episode seed" contract for
        every ``seed < _NAMESPACE_SPLIT`` (which includes the held-out eval band
        at ``900_000+``). This is the ONLY namespace that can reach the LOW
        region, hence the only one that can reach the held-out band.
      * any non-zero ``base_seed`` (a training / self-play namespace) folds
        ``(base_seed, seed)`` into the HIGH region ``[_NAMESPACE_SPLIT,
        _TRACK_SEED_MOD)`` with a stable, platform-independent integer hash (the
        ``LookaheadAgent._turn_seed`` hand-fold style -- NOT Python's
        ``PYTHONHASHSEED``-salted ``hash()``). Distinct ``base_seed`` values
        therefore produce structurally separated (and, for the same episode
        seed, distinct) track-seed streams, and none of them can collide with
        the eval band.

    Determinism: a pure function of ``(base_seed, seed)`` -- same inputs always
    yield the same track seed, in-process or across a ``SubprocVecEnv`` worker.
    """
    base = int(base_seed)
    s = int(seed)
    if base == 0:
        # Identity / eval namespace: legacy "track == f(episode seed)" contract.
        return s % _NAMESPACE_SPLIT

    # Training namespace: stable hand-folded hash of (base_seed, seed), mapped
    # into the HIGH region so it can never reach the eval band. The fold mirrors
    # search_agent._turn_seed (survives PYTHONHASHSEED; platform-independent).
    acc = (base & 0x7FFFFFFF) * 2654435761
    acc = (acc * 1000003 + (s & 0x7FFFFFFF)) & 0x7FFFFFFF
    high_span = _TRACK_SEED_MOD - _NAMESPACE_SPLIT
    return _NAMESPACE_SPLIT + (acc % high_span)


@dataclass(frozen=True)
class TrackGenParams:
    """Tunable bounds for procedural generation. All ranges inclusive.

    Defaults are chosen with slack (lengths 50-90, 3-7 corners, gap 4) so the
    accept rate of the generate-then-validate loop is high. ``speed_limit_choices``
    are positive and bounded; the obs codec normalizes the speed limit by the
    track-derived max (``features._track_lookahead``), so any positive choice
    keeps the observation in ``[-1, 1]`` -- there is no literal cap to clamp to.
    """

    length_range: tuple[int, int] = (50, 90)
    num_corners_range: tuple[int, int] = (3, 7)
    speed_limit_choices: tuple[int, ...] = (1, 2, 3, 4, 5)
    laps: int = 2
    #: Straight spaces required between a corner's end and the next corner's
    #: start (wrap-aware). A generation preference, not a universal validity
    #: rule (see :mod:`heat.tracks.validate`).
    min_corner_gap: int = 4
    corner_len_range: tuple[int, int] = (1, 3)
    #: Fraction of spaces given >= 2 lanes (approximate; per-space coin flip).
    multi_lane_fraction: float = 0.3
    max_lanes: int = 2
    #: Minimum number of start-grid positions. Must be >= MAX_PLAYERS so every
    #: 2-6 player game fits.
    min_start_positions: int = MAX_PLAYERS

    def __post_init__(self) -> None:
        if self.length_range[0] > self.length_range[1]:
            raise ValueError(f"invalid length_range {self.length_range}")
        if self.length_range[0] < MIN_TRACK_LENGTH:
            raise ValueError(
                f"length_range low {self.length_range[0]} < MIN_TRACK_LENGTH "
                f"{MIN_TRACK_LENGTH}"
            )
        if self.num_corners_range[0] > self.num_corners_range[1]:
            raise ValueError(f"invalid num_corners_range {self.num_corners_range}")
        if self.num_corners_range[0] < 0:
            raise ValueError("num_corners_range low must be >= 0")
        if not self.speed_limit_choices or any(
            s <= 0 for s in self.speed_limit_choices
        ):
            raise ValueError("speed_limit_choices must be non-empty and positive")
        if self.laps < 1:
            raise ValueError("laps must be >= 1")
        if self.min_corner_gap < 1:
            raise ValueError("min_corner_gap must be >= 1")
        if self.corner_len_range[0] < 1 or (
            self.corner_len_range[0] > self.corner_len_range[1]
        ):
            raise ValueError(f"invalid corner_len_range {self.corner_len_range}")
        if not (0.0 <= self.multi_lane_fraction <= 1.0):
            raise ValueError("multi_lane_fraction must be in [0, 1]")
        if self.max_lanes < 1:
            raise ValueError("max_lanes must be >= 1")
        if self.min_start_positions < MAX_PLAYERS:
            raise ValueError(
                f"min_start_positions must be >= MAX_PLAYERS ({MAX_PLAYERS})"
            )


def _place_corners(
    rng: random.Random, length: int, params: TrackGenParams
) -> list[Corner]:
    """Lay corners around the loop, each separated from the previous by at least
    ``min_corner_gap`` straight spaces, leaving a trailing straight before the
    wrap to space[0] (the start line). Returns the corners (may be fewer than
    requested if the loop runs out of room -- the validator still accepts any
    non-overlapping, non-full-loop layout)."""
    n_corners = rng.randint(*params.num_corners_range)
    corners: list[Corner] = []
    if n_corners == 0:
        return corners

    gap = params.min_corner_gap
    clen_lo, clen_hi = params.corner_len_range
    # Walk a cursor forward, alternating gap-then-corner. Reserve a trailing gap
    # so the wrap-around straight (covering the start line at space 0) exists.
    cursor = gap  # leading straight before the first corner
    for _ in range(n_corners):
        clen = rng.randint(clen_lo, clen_hi)
        start = cursor
        end = start + clen - 1
        # Need room for this corner AND a trailing straight (gap) before wrap.
        if end + gap >= length:
            break
        speed_limit = rng.choice(params.speed_limit_choices)
        corners.append(Corner(start=start, end=end, speed_limit=speed_limit))
        cursor = end + 1 + gap
    return corners


def _assign_lanes(
    rng: random.Random, length: int, params: TrackGenParams
) -> list[Space]:
    """Build ``length`` spaces, each with 1 lane or up to ``max_lanes`` with
    probability ``multi_lane_fraction``."""
    spaces: list[Space] = []
    for i in range(length):
        if params.max_lanes > 1 and rng.random() < params.multi_lane_fraction:
            lanes = rng.randint(2, params.max_lanes)
        else:
            lanes = 1
        spaces.append(Space(index=i, lanes=lanes))
    return spaces


def _start_grid(
    rng: random.Random,
    length: int,
    corners: list[Corner],
    params: TrackGenParams,
) -> list[int]:
    """Pick ``>= min_start_positions`` distinct straight (non-corner) spaces for
    the start grid. Prefers a contiguous block on the wrap-around straight that
    includes the start line (space 0), the way the static tracks lay out their
    grid (``[1, 2, 3, 4, 5, 0]``)."""
    corner_spaces: set[int] = set()
    for c in corners:
        corner_spaces.update(range(c.start, c.end + 1))
    straights = [i for i in range(length) if i not in corner_spaces]
    n = params.min_start_positions
    if len(straights) < n:
        return []  # not enough room; caller re-draws

    # Prefer a contiguous run of straights so the grid sits together. Find the
    # longest contiguous run; if it is long enough, take its first `n` spaces.
    runs: list[list[int]] = []
    run: list[int] = []
    prev = None
    for s in straights:
        if prev is not None and s == prev + 1:
            run.append(s)
        else:
            if run:
                runs.append(run)
            run = [s]
        prev = s
    if run:
        runs.append(run)
    runs.sort(key=len, reverse=True)

    if runs and len(runs[0]) >= n:
        block = runs[0][:n]
        # Mirror the static-track grid convention: rotate so the lowest index
        # (the start line, typically) comes last -> e.g. [1,2,3,4,5,0].
        if block[0] == 0 and len(block) > 1:
            block = block[1:] + [block[0]]
        return block

    # Fallback: take the first `n` straight spaces anywhere on the loop.
    return straights[:n]


def _build_attempt(rng: random.Random, params: TrackGenParams, name: str) -> Track:
    """Draw one candidate track from the RNG (may be invalid; caller validates)."""
    length = rng.randint(*params.length_range)
    corners = _place_corners(rng, length, params)
    spaces = _assign_lanes(rng, length, params)
    start_positions = _start_grid(rng, length, corners, params)
    return Track(
        name=name,
        spaces=spaces,
        corners=corners,
        start_positions=start_positions,
        laps=params.laps,
    )


def generate_track(
    seed: int,
    params: TrackGenParams | None = None,
    *,
    name: str | None = None,
    max_retries: int = _MAX_RETRIES,
) -> Track:
    """Return a valid :class:`Track` deterministically from ``(seed, params)``.

    Uses a private ``random.Random(seed)`` only -- never the global RNG -- so the
    result is reproducible and independent of any surrounding global-random
    state. Generate-then-validate with bounded retries: re-draws on validation
    failure, raising :class:`RuntimeError` if it cannot produce a valid track
    within ``max_retries`` (signalling over-constrained ``params``).

    Args:
        seed: integer seed for the private RNG.
        params: generation bounds; defaults to :class:`TrackGenParams`.
        name: track name; defaults to ``f"gen-{seed}"``.
        max_retries: bounded attempt budget.

    Returns:
        A validated :class:`Track`.

    Raises:
        RuntimeError: if no valid track is produced within ``max_retries``.
    """
    params = params or TrackGenParams()
    name = name if name is not None else f"gen-{seed}"
    rng = random.Random(seed)

    last_error: Exception | None = None
    for _ in range(max_retries):
        track = _build_attempt(rng, params, name)
        try:
            validate_track(track)
        except Exception as exc:  # noqa: BLE001 - re-raised below if budget spent
            last_error = exc
            continue
        return track

    raise RuntimeError(
        f"generate_track(seed={seed}) failed to produce a valid track within "
        f"{max_retries} retries; params are likely over-constrained "
        f"(last validation error: {last_error})"
    )


class TrackSampler:
    """Picklable ``sampler(seed) -> Track`` for per-episode track generation.

    A top-level class (not a closure) so it survives ``SubprocVecEnv`` ``spawn``
    pickling on Windows -- the env's ``track`` source is shipped to each worker,
    and a closure cannot be pickled. A generated track is a deterministic
    function of ``(base_seed, episode seed)`` via :func:`_namespaced_track_seed`,
    so ``reset(seed)`` reproduces BOTH the track AND the deck shuffle. When the
    env passes ``seed=None`` (unseeded reset), a private RNG draws a fresh track
    seed so successive resets vary. ``random.Random`` pickles its state, so a
    pickled sampler resumes its fallback stream consistently in each worker.

    Namespace disjointness (code-review 2026-06-22 #6): track seeds are NOT a
    plain additive ``seed + base_seed`` mix (which could let a long training
    campaign silently regenerate the held-out eval tracks). Instead
    :func:`_namespaced_track_seed` partitions the seed space so the held-out eval
    band (``base_seed == 0``, the identity namespace, seeds ``900_000+``) and any
    non-zero training ``base_seed`` land in structurally disjoint regions -- a
    training namespace can never collide with the eval band, and two different
    ``base_seed`` values never share a track for the same episode seed.

    Args:
        params: generation bounds, forwarded to :func:`generate_track`.
        base_seed: the *namespace* selector for seed derivation. ``0`` is the
            identity / eval namespace (``track_seed = seed`` for seeds in the low
            region, including the held-out ``900_000+`` band); any non-zero value
            selects a disjoint training namespace in the high seed region. Lets
            distinct vec-env workers / samplers cover provably non-overlapping
            track streams, and keeps self-play streams disjoint from the eval
            band BY CONSTRUCTION (not by hoping additive seeds don't overlap).
    """

    def __init__(
        self, params: TrackGenParams | None = None, *, base_seed: int = 0
    ) -> None:
        self.params = params or TrackGenParams()
        self.base_seed = base_seed
        self._fallback_rng = random.Random(base_seed)

    def __call__(self, seed: int | None) -> Track:
        if seed is None:
            track_seed = self._fallback_rng.randrange(2**31)
        else:
            # Namespaced derivation: base_seed selects a disjoint seed region so
            # a training namespace can never regenerate the held-out eval band.
            track_seed = _namespaced_track_seed(self.base_seed, seed)
        return generate_track(track_seed, self.params)


def track_sampler(
    params: TrackGenParams | None = None,
    *,
    base_seed: int = 0,
) -> Callable[[int | None], Track]:
    """Return a picklable ``sampler(seed) -> Track`` for per-episode sampling.

    Thin constructor for :class:`TrackSampler` (kept for call-site stability).
    The returned object is callable exactly like the old closure form but also
    pickles across process boundaries.

    Args:
        params: generation bounds, forwarded to :func:`generate_track`.
        base_seed: the *namespace* selector for seed derivation (see
            :class:`TrackSampler`). ``0`` is the identity / eval namespace
            (reaches the held-out ``900_000+`` band); any non-zero value selects
            a disjoint training namespace, so self-play streams cannot collide
            with the held-out eval band by construction.
    """
    return TrackSampler(params, base_seed=base_seed)


# ---------------------------------------------------------------------------
# Step-aware difficulty curriculum (Sprint B, Idea 2)
# ---------------------------------------------------------------------------


def easy_track_params() -> TrackGenParams:
    """The "easy" curriculum endpoint: short, gentle, single-lap tracks.

    Narrow distribution the curriculum starts from before widening to the full
    default :class:`TrackGenParams` over the schedule horizon. ``length_range``
    stays comfortably above ``MIN_TRACK_LENGTH`` (= 8) so every interpolated
    range constructs and the generate-then-validate accept rate stays high.
    """
    return TrackGenParams(
        length_range=(30, 40),
        num_corners_range=(2, 3),
        laps=1,
        corner_len_range=(1, 1),
        speed_limit_choices=(2, 3, 4),
    )


@dataclass(frozen=True)
class CurriculumSchedule:
    """Interpolate :class:`TrackGenParams` from ``easy`` to ``full`` over a step
    horizon (Sprint B Idea 2).

    ``frac = clip(step / horizon_steps, 0, 1)`` drives a linear interpolation of
    the tunable bounds; integer range endpoints are rounded and ``laps`` steps up
    at the rounded threshold. The non-interpolated fields are taken from
    ``full`` (so e.g. ``speed_limit_choices`` snap to the full set once ramping
    starts past 0 -- the easy gentler choices only apply exactly at step 0). The
    schedule is a frozen dataclass of two frozen :class:`TrackGenParams`, so it
    is hashable and picklable.

    Each ``params_at`` result is a fresh ``TrackGenParams`` whose
    ``__post_init__`` validates the interpolated bounds; the design keeps the
    easy endpoint above ``MIN_TRACK_LENGTH`` and never inverts a range, so every
    fraction yields a constructible, generatable params object.
    """

    easy: TrackGenParams
    full: TrackGenParams
    horizon_steps: int

    def fraction_at(self, step: int) -> float:
        """The clipped ramp fraction ``frac in [0, 1]`` for a global step."""
        if self.horizon_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, step / self.horizon_steps))

    def params_at(self, step: int) -> TrackGenParams:
        """The interpolated :class:`TrackGenParams` for a global training step.

        At ``step <= 0`` returns the easy-equivalent bounds; at
        ``step >= horizon_steps`` returns the full default; clamps beyond the
        horizon. Integer ranges are rounded; ``laps`` steps up at its threshold.
        """
        frac = self.fraction_at(step)

        def lerp_range(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
            lo = round(a[0] + (b[0] - a[0]) * frac)
            hi = round(a[1] + (b[1] - a[1]) * frac)
            return (lo, hi)

        laps = round(self.easy.laps + (self.full.laps - self.easy.laps) * frac)
        return dataclasses.replace(
            self.full,
            length_range=lerp_range(self.easy.length_range, self.full.length_range),
            num_corners_range=lerp_range(
                self.easy.num_corners_range, self.full.num_corners_range
            ),
            corner_len_range=lerp_range(
                self.easy.corner_len_range, self.full.corner_len_range
            ),
            laps=laps,
        )


def default_curriculum_schedule(horizon_steps: int) -> CurriculumSchedule:
    """A :class:`CurriculumSchedule` from :func:`easy_track_params` to the full
    default :class:`TrackGenParams` over ``horizon_steps``."""
    return CurriculumSchedule(
        easy=easy_track_params(),
        full=TrackGenParams(),
        horizon_steps=horizon_steps,
    )


class StepAwareTrackSampler:
    """Picklable ``sampler(seed) -> Track`` whose difficulty follows a
    :class:`CurriculumSchedule` and a mutable current-step counter (Sprint B
    Idea 2).

    A top-level class (no closures) so it survives ``SubprocVecEnv`` ``spawn``
    pickling exactly like :class:`TrackSampler`. Each ``__call__`` rebuilds the
    effective :class:`TrackGenParams` from ``schedule.params_at(self._step)``,
    then derives a track seed identically to :class:`TrackSampler` (so
    "same seed -> same track" holds *for a fixed step*).

    Cross-process caveat: under ``SubprocVecEnv`` the sampler is pickled into the
    workers, so :meth:`set_step` on the main-process object does NOT reach them.
    The training loop therefore drives the curriculum by **rebuilding the vec env
    at stage boundaries** with a fresh, step-pinned sampler (process-safe staged
    rebuilds), rather than relying on a live counter crossing the process
    boundary. ``set_step`` remains available for the single-process
    (``DummyVecEnv``) path and for tests.

    Args:
        schedule: the easy->full difficulty schedule.
        base_seed: namespace selector for the derived track seed (as in
            :class:`TrackSampler`); ``0`` is the identity / eval namespace, any
            non-zero value a disjoint training namespace.
        step: the initial global training step (pins the difficulty for staged
            rebuilds; a fresh sampler per stage carries that stage's step).
    """

    def __init__(
        self,
        schedule: CurriculumSchedule,
        *,
        base_seed: int = 0,
        step: int = 0,
    ) -> None:
        self.schedule = schedule
        self.base_seed = base_seed
        self._step = int(step)
        self._fallback_rng = random.Random(base_seed)

    def set_step(self, step: int) -> None:
        """Update the current global training step (single-process path / tests).

        Under ``SubprocVecEnv`` this does not reach pickled workers; the training
        loop uses staged env rebuilds for process-safe curriculum progression.
        """
        self._step = int(step)

    def params_at_current(self) -> TrackGenParams:
        """The effective :class:`TrackGenParams` at the current step."""
        return self.schedule.params_at(self._step)

    def __call__(self, seed: int | None) -> Track:
        params = self.schedule.params_at(self._step)
        if seed is None:
            track_seed = self._fallback_rng.randrange(2**31)
        else:
            # Same namespaced derivation as TrackSampler (code-review #6): the
            # base_seed selects a disjoint seed region, so a training namespace
            # can never regenerate the held-out eval band.
            track_seed = _namespaced_track_seed(self.base_seed, seed)
        return generate_track(track_seed, params)
