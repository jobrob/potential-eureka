"""Tests for the PlayerState model."""

import pytest

from heat.models.cards import Card, CardType
from heat.models.player_state import PlayerState


class TestPlayerState:
    def test_create_default(self):
        player = PlayerState.create(0)
        assert player.player_id == 0
        assert player.name == "Player 0"
        assert player.gear == 1
        assert len(player.hand) == 7
        assert player.deck.draw_pile_size == 11  # 18 - 7 drawn (12 speed + 3 upgrade + 3 stress)
        assert player.heat_available == 6
        assert not player.spun_out
        assert not player.finished

    def test_create_with_name(self):
        player = PlayerState.create(1, "Alice")
        assert player.name == "Alice"

    def test_speed_cards_in_hand(self):
        player = PlayerState.create(0)
        speed_cards = player.speed_cards_in_hand
        assert all(c.card_type == CardType.SPEED for c in speed_cards)

    def test_pay_heat(self):
        player = PlayerState.create(0)
        initial_heat = player.heat_available
        paid = player.pay_heat(2)
        assert len(paid) == 2
        assert player.heat_available == initial_heat - 2
        assert all(c.card_type == CardType.HEAT for c in paid)
        # Heat cards go to discard
        assert player.deck.discard_pile_size == 2

    def test_pay_too_much_heat(self):
        player = PlayerState.create(0)
        with pytest.raises(ValueError):
            player.pay_heat(100)

    def test_cooldown(self):
        player = PlayerState.create(0)
        # Remove any heat cards that were randomly drawn, then add a known one
        player.hand = [c for c in player.hand if c.card_type != CardType.HEAT]
        heat_card = Card(CardType.HEAT, 0, "test_heat")
        player.hand.append(heat_card)
        initial_heat_pool = player.heat_available

        cooled = player.cooldown(1)
        assert len(cooled) == 1
        assert cooled[0] == heat_card
        assert heat_card not in player.hand
        assert player.heat_available == initial_heat_pool + 1

    def test_cooldown_no_heat_in_hand(self):
        player = PlayerState.create(0)
        # Remove any heat cards that may have been drawn into hand
        player.hand = [c for c in player.hand if c.card_type != CardType.HEAT]
        cooled = player.cooldown(3)
        assert len(cooled) == 0
