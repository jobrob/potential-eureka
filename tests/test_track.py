"""Tests for track models and loader."""

import json
import tempfile
from pathlib import Path

import pytest

from heat.models.track import Corner, Space, Track
from heat.tracks.loader import load_track, load_track_by_name


class TestTrackModel:
    def test_track_length(self):
        spaces = [Space(i) for i in range(10)]
        track = Track("Test", spaces, [], [0, 1])
        assert track.length == 10

    def test_get_corner_at(self):
        spaces = [Space(i) for i in range(20)]
        corner = Corner(start=5, end=8, speed_limit=3)
        track = Track("Test", spaces, [corner], [0])

        assert track.get_corner_at(5) == corner
        assert track.get_corner_at(7) == corner
        assert track.get_corner_at(8) == corner
        assert track.get_corner_at(4) is None
        assert track.get_corner_at(9) is None

    def test_spaces_in_corner(self):
        spaces = [Space(i, lanes=2 if i < 5 else 1) for i in range(10)]
        corner = Corner(start=5, end=7, speed_limit=4)
        track = Track("Test", spaces, [corner], [0])

        corner_spaces = track.spaces_in_corner(corner)
        assert len(corner_spaces) == 3
        assert all(s.lanes == 1 for s in corner_spaces)

    def test_corner_frozen(self):
        corner = Corner(start=0, end=3, speed_limit=4)
        with pytest.raises(AttributeError):
            corner.speed_limit = 5  # type: ignore[misc]


class TestTrackLoader:
    def test_load_track_from_json(self, tmp_path: Path):
        data = {
            "name": "TestTrack",
            "laps": 2,
            "spaces": [{"index": i, "lanes": 1 + (i % 2)} for i in range(5)],
            "corners": [{"start": 2, "end": 3, "speed_limit": 3}],
            "start_positions": [0, 1],
        }
        track_file = tmp_path / "test.json"
        track_file.write_text(json.dumps(data))

        track = load_track(track_file)
        assert track.name == "TestTrack"
        assert track.laps == 2
        assert track.length == 5
        assert len(track.corners) == 1
        assert track.corners[0].speed_limit == 3
        assert track.start_positions == [0, 1]

    def test_load_usa_track(self):
        track = load_track_by_name("usa")
        assert track.name == "USA"
        assert track.length == 30
        assert len(track.corners) == 3

    def test_load_nonexistent_track(self):
        with pytest.raises(FileNotFoundError):
            load_track_by_name("nonexistent_track")
