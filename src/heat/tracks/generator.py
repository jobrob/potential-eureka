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

import random
from dataclasses import dataclass
from typing import Callable

from heat.tracks.validate import MIN_TRACK_LENGTH, validate_track
from heat.models.track import Corner, Space, Track
from heat.ml.spaces import MAX_PLAYERS

#: Default bounded retry budget for generate-then-validate.
_MAX_RETRIES: int = 200


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


def track_sampler(
    params: TrackGenParams | None = None,
    *,
    base_seed: int = 0,
) -> Callable[[int | None], Track]:
    """Return a ``sampler(seed) -> Track`` for per-episode track sampling.

    The sampler derives the track seed from the episode ``seed`` (and
    ``base_seed``) so a generated track is a deterministic function of the
    episode seed: ``reset(seed)`` reproduces BOTH the track AND the deck shuffle.
    When the env passes ``seed=None`` (unseeded reset), the sampler draws a fresh
    track seed from its own private RNG so successive resets vary.

    Args:
        params: generation bounds, forwarded to :func:`generate_track`.
        base_seed: offset mixed into the derived seed (lets distinct vec-env
            workers / samplers cover disjoint track streams from the same
            episode seeds).
    """
    params = params or TrackGenParams()
    fallback_rng = random.Random(base_seed)

    def sampler(seed: int | None) -> Track:
        if seed is None:
            track_seed = fallback_rng.randrange(2**31)
        else:
            # Combine episode seed with base_seed deterministically.
            track_seed = (int(seed) + base_seed) % (2**31)
        return generate_track(track_seed, params)

    return sampler
