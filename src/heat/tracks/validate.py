"""Track validity contract for procedural generation (Sprint 6A).

:func:`validate_track` is the single, explicit definition of a *raceable* track:
a pure predicate (raises on failure) reused both by the generator's
accept/reject loop and by tests. It encodes the structural assumptions the
engine and the ML feature extractor make about a :class:`~heat.models.track.Track`
so that a *generated* track can never silently break training (a corner on the
start line, a zero-straight loop, a divide-by-zero in obs normalization, etc.).

Design note on corner spacing
------------------------------
The hand-authored static tracks (``tracks/usa.json``, ``tracks/silverstone.json``)
are the known-good ground truth and MUST pass this validator. Silverstone has
*adjacent* corners (end=10, start=11) with no straight between them, so a hard
"minimum straight between every pair of consecutive corners" rule would reject a
known-good track. The validator therefore enforces only the genuinely-raceable
invariants here (no corner overlap; at least one non-corner / straight space
exists on the loop). The richer ``min_corner_gap`` (straight spaces between
corners) is a *generation* preference, enforced inside the generator, not a
universal validity rule -- see :mod:`heat.tracks.generator`.
"""

from __future__ import annotations

from heat.ml.spaces import MAX_PLAYERS
from heat.models.track import Track

#: Lower bound on total spaces. Short enough to admit the static tracks (USA has
#: 30 spaces) and the in-memory test tracks, long enough that the obs
#: ``position / length`` and corner-distance normalizations are non-degenerate.
MIN_TRACK_LENGTH: int = 8


class TrackValidationError(ValueError):
    """Raised when a :class:`Track` is not a valid, raceable track."""


def validate_track(track: Track) -> None:
    """Raise :class:`TrackValidationError` if ``track`` is not raceable.

    Checks (all required for a track to be safe for the engine + ML obs codec):

    1. ``length >= MIN_TRACK_LENGTH`` and ``length == len(spaces)``; space
       indices are exactly ``0..length-1`` in order; every space has
       ``lanes >= 1``.
    2. ``laps >= 1``.
    3. Corners: ``0 <= start <= end < length``; ``speed_limit > 0``; no two
       corners overlap; the corners do not cover the whole loop (at least one
       straight / non-corner space must exist so cars are not perpetually
       paying corner heat).
    4. ``start_positions``: count ``>= MAX_PLAYERS``; every entry in
       ``[0, length)``; all distinct; none placed inside a corner (the start
       grid sits on a straight so seats can be placed without a turn-0 corner
       collision).

    Pure: no I/O, no mutation of ``track``.
    """
    length = track.length

    # --- 1. length / spaces / lanes -------------------------------------
    if length < MIN_TRACK_LENGTH:
        raise TrackValidationError(
            f"track length {length} < minimum {MIN_TRACK_LENGTH}"
        )
    if length != len(track.spaces):
        raise TrackValidationError(
            f"length {length} != len(spaces) {len(track.spaces)}"
        )
    for expected_index, space in enumerate(track.spaces):
        if space.index != expected_index:
            raise TrackValidationError(
                f"space at position {expected_index} has index {space.index}; "
                "space indices must be 0..length-1 in order"
            )
        if space.lanes < 1:
            raise TrackValidationError(
                f"space {space.index} has lanes {space.lanes} < 1"
            )

    # --- 2. laps --------------------------------------------------------
    if track.laps < 1:
        raise TrackValidationError(f"laps {track.laps} < 1")

    # --- 3. corners -----------------------------------------------------
    corner_spaces: set[int] = set()
    for corner in track.corners:
        if not (0 <= corner.start <= corner.end < length):
            raise TrackValidationError(
                f"corner {corner} violates 0 <= start <= end < length ({length})"
            )
        if corner.speed_limit <= 0:
            raise TrackValidationError(
                f"corner {corner} has non-positive speed_limit"
            )
        covered = set(range(corner.start, corner.end + 1))
        overlap = corner_spaces & covered
        if overlap:
            raise TrackValidationError(
                f"corner {corner} overlaps existing corner spaces {sorted(overlap)}"
            )
        corner_spaces |= covered

    if len(corner_spaces) >= length:
        raise TrackValidationError(
            "corners cover the whole loop; no straight (non-corner) space exists"
        )

    # --- 4. start positions ---------------------------------------------
    starts = track.start_positions
    if len(starts) < MAX_PLAYERS:
        raise TrackValidationError(
            f"need >= {MAX_PLAYERS} start positions, got {len(starts)}"
        )
    if len(set(starts)) != len(starts):
        raise TrackValidationError(
            f"start positions must be distinct, got {starts}"
        )
    for pos in starts:
        if not (0 <= pos < length):
            raise TrackValidationError(
                f"start position {pos} out of range [0, {length})"
            )
        if pos in corner_spaces:
            raise TrackValidationError(
                f"start position {pos} lies inside a corner; start grid must "
                "be on a straight"
            )
