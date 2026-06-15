"""Tests for GameState.clone (Step B3)."""

from __future__ import annotations

import random

from heat.models.game_state import GameState, Phase
from heat.models.track import Space, Track


def _track(laps: int = 1) -> Track:
    spaces = [Space(index=i, lanes=2) for i in range(12)]
    return Track(
        name="clone-test",
        spaces=spaces,
        corners=[],
        start_positions=[0, 1, 2, 3],
        laps=laps,
    )


def _state() -> GameState:
    return GameState.create(_track(), 4, seed=42)


class TestCloneEquality:
    def test_full_state_equality(self) -> None:
        s = _state()
        s.round_num = 3
        s.current_phase = Phase.REACT
        s.log_event("foo", player_id=0)
        c = s.clone(copy_event_log=True)
        assert c.round_num == s.round_num
        assert c.current_phase == s.current_phase
        assert c.turn_order == s.turn_order
        assert c.starting_player_count == s.starting_player_count
        assert c.num_players == s.num_players
        for cp, sp in zip(c.players, s.players):
            assert cp.player_id == sp.player_id
            assert cp.hand == sp.hand
            assert cp.heat_pool == sp.heat_pool
            assert list(cp.deck.draw_pile) == list(sp.deck.draw_pile)

    def test_stress_counter_preserved(self) -> None:
        s = _state()
        s.next_stress_id()
        s.next_stress_id()
        c = s.clone()
        assert c._stress_counter == s._stress_counter
        # And it keeps counting from there.
        assert c.next_stress_id() == 3


class TestCloneIsolation:
    def test_player_mutation_isolated(self) -> None:
        s = _state()
        c = s.clone()
        c.players[0].gear = 4
        c.players[0].deck.draw(5)
        assert s.players[0].gear == 1
        assert s.players[0].deck.draw_pile_size != c.players[0].deck.draw_pile_size

    def test_track_shared(self) -> None:
        s = _state()
        c = s.clone()
        assert c.track is s.track


class TestCloneRng:
    def test_default_fork_is_independent(self) -> None:
        s = _state()
        c = s.clone()
        before = s.rng.getstate()
        # Advancing the clone's stream must not touch the source.
        for _ in range(10):
            c.rng.random()
        assert s.rng.getstate() == before

    def test_default_fork_is_reproducible(self) -> None:
        s1 = GameState.create(_track(), 4, seed=7)
        s2 = GameState.create(_track(), 4, seed=7)
        c1 = s1.clone()
        c2 = s2.clone()
        seq1 = [c1.rng.random() for _ in range(5)]
        seq2 = [c2.rng.random() for _ in range(5)]
        assert seq1 == seq2

    def test_reseed_is_deterministic(self) -> None:
        s = _state()
        c1 = s.clone(reseed=123)
        c2 = s.clone(reseed=123)
        assert [c1.rng.random() for _ in range(5)] == [
            c2.rng.random() for _ in range(5)
        ]

    def test_clone_deck_bound_to_clone_rng(self) -> None:
        s = _state()
        c = s.clone()
        for p in c.players:
            assert p.deck._rng is c.rng


class TestEventLogPolicy:
    def test_event_log_not_copied_by_default(self) -> None:
        s = _state()
        s.log_event("e1", player_id=0)
        c = s.clone()
        assert c.event_log == []

    def test_event_log_copied_when_requested(self) -> None:
        s = _state()
        s.log_event("e1", player_id=0)
        c = s.clone(copy_event_log=True)
        assert len(c.event_log) == 1
        # Shallow copy: appending to clone does not affect source.
        c.log_event("e2", player_id=1)
        assert len(s.event_log) == 1
