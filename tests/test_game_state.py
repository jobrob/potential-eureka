"""Tests for the GameState model."""

import pytest

from heat.models.game_state import GameState, Phase
from heat.models.track import Corner, Space, Track


def _make_track(num_spaces: int = 20) -> Track:
    spaces = [Space(i, lanes=2) for i in range(num_spaces)]
    corners = [Corner(start=5, end=7, speed_limit=3)]
    start_pos = list(range(min(6, num_spaces)))
    return Track("Test", spaces, corners, start_pos)


class TestGameState:
    def test_create(self):
        track = _make_track()
        state = GameState.create(track, 4)
        assert state.num_players == 4
        assert state.round_num == 1
        assert state.current_phase == Phase.SHIFT_GEARS
        assert len(state.turn_order) == 4

    def test_create_with_names(self):
        track = _make_track()
        names = ["Alice", "Bob", "Carol"]
        state = GameState.create(track, 3, player_names=names)
        assert state.get_player(0).name == "Alice"
        assert state.get_player(2).name == "Carol"

    def test_create_mismatched_names(self):
        track = _make_track()
        with pytest.raises(ValueError):
            GameState.create(track, 3, player_names=["Alice"])

    def test_start_positions(self):
        track = _make_track()
        state = GameState.create(track, 3)
        for i in range(3):
            assert state.get_player(i).position == track.start_positions[i]

    def test_active_players(self):
        track = _make_track()
        state = GameState.create(track, 3)
        assert len(state.active_players) == 3

        state.players[0].finished = True
        state.players[0].finish_order = 1
        assert len(state.active_players) == 2
        assert len(state.finished_players) == 1

    def test_is_game_over(self):
        track = _make_track()
        state = GameState.create(track, 2)
        assert not state.is_game_over

        for i, p in enumerate(state.players):
            p.finished = True
            p.finish_order = i + 1
        assert state.is_game_over

    def test_log_event(self):
        track = _make_track()
        state = GameState.create(track, 2)
        state.log_event("test_event", player_id=0, data={"key": "value"})
        assert len(state.event_log) == 1
        assert state.event_log[0].event_type == "test_event"

    def test_logging_disabled(self):
        track = _make_track()
        state = GameState.create(track, 2, logging_enabled=False)
        state.log_event("test_event")
        assert len(state.event_log) == 0

    def test_compute_turn_order(self):
        track = _make_track()
        state = GameState.create(track, 3)
        # Move player 2 ahead
        state.players[2].position = 15
        state.players[1].position = 10
        state.players[0].position = 5
        state.compute_turn_order()
        # Furthest ahead first
        assert state.turn_order == [2, 1, 0]

    def test_phase_enum(self):
        phases = list(Phase)
        assert len(phases) == 9
        assert Phase.SHIFT_GEARS in phases
        assert Phase.REPLENISH in phases
