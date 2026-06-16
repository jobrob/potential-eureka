"""Tests for the track validity contract (Sprint 6A, tracks/validate.py).

Gates:
  * The two hand-authored static tracks pass the validator (known-good pins).
  * Hand-built degenerate tracks are rejected with TrackValidationError.
"""

from __future__ import annotations

import pytest

from heat.ml.spaces import MAX_PLAYERS
from heat.models.track import Corner, Space, Track
from heat.tracks.loader import load_track_by_name
from heat.tracks.validate import (
    MIN_TRACK_LENGTH,
    TrackValidationError,
    validate_track,
)


def _good_track(length: int = 20) -> Track:
    """A minimal valid track: one short corner, a straight start grid."""
    spaces = [Space(index=i, lanes=1) for i in range(length)]
    corners = [Corner(start=10, end=11, speed_limit=3)]
    starts = [0, 1, 2, 3, 4, 5]
    return Track(name="good", spaces=spaces, corners=corners,
                 start_positions=starts, laps=2)


class TestStaticTracksPass:
    @pytest.mark.parametrize("name", ["usa", "silverstone"])
    def test_static_track_is_valid(self, name: str) -> None:
        # Known-good ground truth: must never raise.
        validate_track(load_track_by_name(name))


class TestAcceptsGoodTrack:
    def test_minimal_good_track_passes(self) -> None:
        validate_track(_good_track())


class TestRejectsDegenerate:
    def test_too_short(self) -> None:
        length = MIN_TRACK_LENGTH - 1
        t = Track(
            name="short",
            spaces=[Space(index=i) for i in range(length)],
            corners=[],
            start_positions=list(range(MAX_PLAYERS)),
            laps=1,
        )
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_noncontiguous_indices_after_drop(self) -> None:
        t = _good_track()
        # Drop a middle space: indices become 0,1,2,4,5,... -> the consecutive
        # 0..length-1 index check trips at the gap.
        del t.spaces[3]
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_nonconsecutive_space_indices(self) -> None:
        t = _good_track()
        t.spaces[5] = Space(index=999, lanes=1)
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_zero_lane_space(self) -> None:
        t = _good_track()
        t.spaces[3] = Space(index=3, lanes=0)
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_zero_laps(self) -> None:
        t = _good_track()
        t.laps = 0
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_corner_out_of_range(self) -> None:
        t = _good_track()
        t.corners = [Corner(start=18, end=25, speed_limit=3)]  # end >= length
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_corner_start_after_end(self) -> None:
        t = _good_track()
        t.corners = [Corner(start=12, end=10, speed_limit=3)]
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_nonpositive_speed_limit(self) -> None:
        t = _good_track()
        t.corners = [Corner(start=10, end=11, speed_limit=0)]
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_overlapping_corners(self) -> None:
        t = _good_track()
        t.corners = [
            Corner(start=10, end=12, speed_limit=3),
            Corner(start=11, end=13, speed_limit=2),
        ]
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_corners_cover_whole_loop(self) -> None:
        length = 10
        t = Track(
            name="all-corner",
            spaces=[Space(index=i) for i in range(length)],
            corners=[Corner(start=0, end=length - 1, speed_limit=3)],
            start_positions=list(range(MAX_PLAYERS)),
            laps=1,
        )
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_too_few_start_positions(self) -> None:
        t = _good_track()
        t.start_positions = [0, 1, 2]  # < MAX_PLAYERS
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_duplicate_start_positions(self) -> None:
        t = _good_track()
        t.start_positions = [0, 0, 1, 2, 3, 4]
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_start_position_out_of_range(self) -> None:
        t = _good_track()
        t.start_positions = [0, 1, 2, 3, 4, 999]
        with pytest.raises(TrackValidationError):
            validate_track(t)

    def test_start_position_inside_corner(self) -> None:
        t = _good_track()  # corner spans 10..11
        t.start_positions = [10, 1, 2, 3, 4, 5]
        with pytest.raises(TrackValidationError):
            validate_track(t)
