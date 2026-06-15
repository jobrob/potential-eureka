"""Tests for the game viewer, focused on the track visualisation."""

from __future__ import annotations

import pytest

from heat.models.game_state import GameState
from heat.models.player_state import PlayerState
from heat.models.track import Corner, Space, Track
from heat.tracks.loader import load_track_by_name
from heat.viewer import (
    _assign_markers,
    _build_edges,
    _distribute_spaces,
    render_track,
)


def _make_player(player_id: int, name: str, position: int, lap: int = 1) -> PlayerState:
    p = PlayerState.create(player_id, name)
    p.position = position
    p.lap = lap
    return p


def _state(track: Track, players: list[PlayerState]) -> GameState:
    return GameState(track=track, players=players)


def _tiny_track(n: int = 8, name: str = "Tiny") -> Track:
    spaces = [Space(i) for i in range(n)]
    corners = [Corner(start=2, end=3, speed_limit=3)]
    return Track(name=name, spaces=spaces, corners=corners, start_positions=[0], laps=1)


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestDistributeSpaces:
    @pytest.mark.parametrize("total", [4, 8, 10, 30, 50, 80, 123])
    def test_counts_sum_to_total(self, total: int) -> None:
        top, right, bottom, left = _distribute_spaces(total)
        assert top + right + bottom + left == total

    @pytest.mark.parametrize("total", [4, 8, 10, 30, 50, 80])
    def test_all_edges_non_empty_for_real_rectangles(self, total: int) -> None:
        top, right, bottom, left = _distribute_spaces(total)
        assert top >= 1 and right >= 1 and bottom >= 1 and left >= 1

    def test_zero(self) -> None:
        assert _distribute_spaces(0) == (0, 0, 0, 0)

    def test_below_four_goes_on_top(self) -> None:
        assert _distribute_spaces(3) == (3, 0, 0, 0)


class TestBuildEdges:
    @pytest.mark.parametrize("n", [8, 10, 30, 50, 80])
    def test_edges_partition_all_indices(self, n: int) -> None:
        track = Track("T", [Space(i) for i in range(n)], [], [0])
        top, right, bottom, left = _build_edges(track)
        combined = sorted(top + right + bottom + left)
        assert combined == list(range(n))

    def test_bottom_and_left_render_order(self) -> None:
        # Bottom is right-to-left (descending); left is bottom-to-top (descending).
        track = Track("T", [Space(i) for i in range(12)], [], [0])
        top, right, bottom, left = _build_edges(track)
        assert top == sorted(top)  # ascending
        assert right == sorted(right)  # ascending (top-to-bottom)
        assert bottom == sorted(bottom, reverse=True)
        assert left == sorted(left, reverse=True)


class TestAssignMarkers:
    def test_unique_markers(self) -> None:
        players = [_make_player(i, f"P{i}", 0) for i in range(5)]
        markers = _assign_markers(players)
        assert len(set(markers.values())) == len(players)

    def test_keyed_by_player_id(self) -> None:
        players = [_make_player(3, "Z", 0), _make_player(1, "A", 0)]
        markers = _assign_markers(players)
        assert set(markers.keys()) == {1, 3}


# ---------------------------------------------------------------------------
# render_track tests
# ---------------------------------------------------------------------------


class TestRenderTrack:
    def test_header_contains_track_name_and_lap_info(self) -> None:
        track = load_track_by_name("usa")
        out = render_track(_state(track, [_make_player(0, "Max", 5)]))
        first = out.splitlines()[0]
        assert "USA" in first
        assert "Lap" in first
        assert "/1" in first  # USA has 1 lap

    def test_corner_labels_present(self) -> None:
        track = load_track_by_name("usa")
        out = render_track(_state(track, []))
        # USA has 3 corners -> labels C1, C2, C3 should appear somewhere.
        assert "C1" in out
        assert "C2" in out
        assert "C3" in out

    def test_corner_notation_includes_speed_limit(self) -> None:
        track = load_track_by_name("usa")
        out = render_track(_state(track, []))
        # Corner 0 is spaces 6-8, speed limit 4 -> "[C1:4 ...]"
        assert "[C1:4" in out

    def test_player_marker_and_legend_present(self) -> None:
        track = load_track_by_name("usa")
        out = render_track(_state(track, [_make_player(0, "Max", 5)]))
        assert "Legend" in out
        assert "Max" in out
        markers = _assign_markers([_make_player(0, "Max", 5)])
        assert markers[0] in out

    def test_no_legend_without_players(self) -> None:
        track = load_track_by_name("usa")
        out = render_track(_state(track, []))
        assert "Legend" not in out

    def test_zero_players_does_not_crash(self) -> None:
        track = load_track_by_name("usa")
        out = render_track(_state(track, []))
        assert "USA" in out

    def test_multiple_players_same_space(self) -> None:
        track = load_track_by_name("usa")
        # Two players on the same space on the top edge.
        players = [_make_player(0, "Ann", 2), _make_player(1, "Bob", 2)]
        out = render_track(_state(track, players))
        markers = _assign_markers(players)
        # Both markers should be present and adjacent (side by side).
        assert markers[0] in out
        assert markers[1] in out
        assert (markers[0] + markers[1]) in out or (markers[1] + markers[0]) in out

    def test_lapped_players_show_lap_number(self) -> None:
        track = load_track_by_name("silverstone")  # 2 laps
        players = [
            _make_player(0, "Max", 5, lap=1),
            _make_player(1, "Lewis", 30, lap=2),
        ]
        out = render_track(_state(track, players))
        # Different laps -> lap annotation like "L1"/"L2" should appear.
        assert "L1" in out
        assert "L2" in out

    def test_same_lap_no_lap_annotation(self) -> None:
        track = load_track_by_name("usa")
        players = [_make_player(0, "Max", 5, lap=1), _make_player(1, "Lewis", 20, lap=1)]
        out = render_track(_state(track, players))
        assert "L1" not in out

    @pytest.mark.parametrize("name", ["usa", "silverstone"])
    def test_real_tracks_render(self, name: str) -> None:
        track = load_track_by_name(name)
        players = [_make_player(0, "Max", 3), _make_player(1, "Lewis", 12)]
        out = render_track(_state(track, players))
        assert track.name in out
        assert out.count("\n") > 4  # multi-line rectangle

    def test_tiny_track_does_not_collapse(self) -> None:
        track = _tiny_track(8)
        out = render_track(_state(track, [_make_player(0, "A", 0)]))
        lines = out.splitlines()
        # Header + (legend) + blank + top + at least one vertical row + bottom.
        assert len(lines) >= 5
        assert "Tiny" in out
        # Every index 0..7 should appear in the rendering.
        for i in range(8):
            assert str(i) in out

    def test_long_track_renders_all_spaces(self) -> None:
        spaces = [Space(i) for i in range(80)]
        corners = [Corner(10, 12, 3), Corner(40, 41, 2)]
        track = Track("Big", spaces, corners, [0], laps=3)
        players = [_make_player(0, "Ann", 5), _make_player(1, "Cy", 45, lap=2)]
        out = render_track(_state(track, players))
        assert "Big" in out
        assert "C1" in out and "C2" in out

    def test_deterministic(self) -> None:
        track = load_track_by_name("silverstone")
        players = [_make_player(0, "Max", 9, 1), _make_player(1, "Lewis", 33, 2)]
        s1 = render_track(_state(track, players))
        s2 = render_track(_state(track, players))
        assert s1 == s2

    @pytest.mark.parametrize("n", [4, 8, 10, 15, 30, 50, 80])
    def test_render_never_crashes_across_sizes(self, n: int) -> None:
        spaces = [Space(i) for i in range(n)]
        corners = [Corner(2, 3, 3)] if n > 4 else []
        track = Track(f"T{n}", spaces, corners, [0])
        players = [_make_player(0, "A", 0), _make_player(1, "B", min(2, n - 1))]
        out = render_track(_state(track, players))
        assert isinstance(out, str) and out

    def test_snapshot_tiny_track_no_players(self) -> None:
        # Small, fixed track with a stable layout for a snapshot-style check.
        track = _tiny_track(8, name="Mini")
        out = render_track(_state(track, []))
        lines = out.splitlines()
        assert lines[0] == "    Mini (Lap 1/1)"
        # Corner C1 covers spaces 2-3. Here space 2 lands on the top edge and
        # space 3 wraps onto the right edge, so the corner bracket on the top
        # edge shows just the space(s) that fall there.
        assert "[C1:3 2]" in out
        assert "──0──1──[C1:3 2]──" in out

    def test_horizontal_corner_groups_consecutive_spaces(self) -> None:
        # A corner fully contained on the top edge groups its spaces with "·".
        spaces = [Space(i) for i in range(30)]
        corners = [Corner(3, 5, 4)]
        track = Track("Grp", spaces, corners, [0])
        out = render_track(_state(track, []))
        assert "[C1:4 3·4·5]" in out
