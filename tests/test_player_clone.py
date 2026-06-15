"""Tests for PlayerState.clone (Step B2)."""

from __future__ import annotations

import random

from heat.models.cards import Card, CardType
from heat.models.player_state import PlayerState


def _player(seed: int = 1) -> PlayerState:
    return PlayerState.create(0, rng=random.Random(seed))


class TestPlayerClone:
    def test_field_by_field_equality(self) -> None:
        p = _player()
        p.gear = 3
        p.position = 7
        p.lap = 2
        p.spun_out = True
        p.boost_used_this_turn = True
        p.speed_from_cards = 5
        p.cluttered = True
        c = p.clone()
        for fld in (
            "player_id", "name", "gear", "position", "lap", "spun_out",
            "finished", "finish_order", "boost_used_this_turn",
            "speed_from_cards", "speed_from_boost", "speed_from_adrenaline",
            "slipstream_moved", "cluttered", "turn_start_position",
            "turn_start_lap",
        ):
            assert getattr(c, fld) == getattr(p, fld)
        assert c.hand == p.hand
        assert c.heat_pool == p.heat_pool

    def test_hand_mutation_isolated(self) -> None:
        p = _player()
        c = p.clone()
        c.hand.pop()
        assert len(c.hand) == len(p.hand) - 1

    def test_heat_pool_mutation_isolated(self) -> None:
        p = _player()
        c = p.clone()
        c.pay_heat(2)
        assert c.heat_available == p.heat_available - 2
        assert p.heat_available == 6

    def test_deck_mutation_isolated(self) -> None:
        p = _player()
        c = p.clone()
        c.deck.draw(3)
        assert c.deck.draw_pile_size != p.deck.draw_pile_size

    def test_gear_change_isolated(self) -> None:
        p = _player()
        c = p.clone()
        c.gear = 4
        assert p.gear == 1

    def test_deck_bound_to_supplied_rng(self) -> None:
        p = _player()
        shared = random.Random(5)
        c = p.clone(shared)
        assert c.deck._rng is shared

    def test_cards_shared_by_identity(self) -> None:
        p = _player()
        c = p.clone()
        for original, copied in zip(p.hand, c.hand):
            assert original is copied
