"""Tests for the cards module."""

import pytest

from heat.models.cards import Card, CardType, Deck, create_heat_cards, create_starting_deck


class TestCard:
    def test_frozen(self):
        card = Card(CardType.SPEED, 3, "test_1")
        with pytest.raises(AttributeError):
            card.value = 5  # type: ignore[misc]

    def test_hashable(self):
        card = Card(CardType.SPEED, 3, "test_1")
        s = {card}
        assert card in s

    def test_repr_speed(self):
        card = Card(CardType.SPEED, 4, "s1")
        assert "Speed(4)" in repr(card)

    def test_repr_heat(self):
        card = Card(CardType.HEAT, 0, "h1")
        assert "Heat" in repr(card)


class TestDeck:
    def test_initial_shuffle(self):
        cards = [Card(CardType.SPEED, i, f"c{i}") for i in range(10)]
        deck = Deck(cards)
        assert deck.draw_pile_size == 10
        assert deck.discard_pile_size == 0

    def test_draw(self):
        cards = [Card(CardType.SPEED, i, f"c{i}") for i in range(5)]
        deck = Deck(cards)
        drawn = deck.draw(3)
        assert len(drawn) == 3
        assert deck.draw_pile_size == 2

    def test_draw_with_reshuffle(self):
        cards = [Card(CardType.SPEED, i, f"c{i}") for i in range(3)]
        deck = Deck(cards)
        drawn = deck.draw(2)
        deck.discard(drawn)
        # Draw pile has 1, discard has 2 — drawing 3 should trigger reshuffle
        drawn2 = deck.draw(3)
        assert len(drawn2) == 3
        assert deck.total_size == 3  # All cards still accounted for

    def test_draw_empty_deck(self):
        deck = Deck([])
        drawn = deck.draw(5)
        assert drawn == []

    def test_discard(self):
        cards = [Card(CardType.SPEED, 1, "c1")]
        deck = Deck(cards)
        drawn = deck.draw(1)
        deck.discard(drawn)
        assert deck.discard_pile_size == 1
        assert deck.draw_pile_size == 0

    def test_total_size(self):
        cards = [Card(CardType.SPEED, i, f"c{i}") for i in range(7)]
        deck = Deck(cards)
        deck.draw(3)
        deck.discard([Card(CardType.SPEED, 99, "extra")])
        # 4 in draw + 1 in discard; 3 drawn cards are gone from deck
        assert deck.total_size == 5

    def test_iter(self):
        cards = [Card(CardType.SPEED, i, f"c{i}") for i in range(5)]
        deck = Deck(cards)
        all_cards = list(deck)
        assert len(all_cards) == 5

    def test_len(self):
        cards = [Card(CardType.SPEED, i, f"c{i}") for i in range(5)]
        deck = Deck(cards)
        assert len(deck) == 5


class TestCardFactories:
    def test_starting_deck_count(self):
        cards = create_starting_deck(0)
        assert len(cards) == 12
        assert all(c.card_type == CardType.SPEED for c in cards)

    def test_starting_deck_unique_ids(self):
        cards = create_starting_deck(0)
        ids = [c.id for c in cards]
        assert len(set(ids)) == len(ids)

    def test_different_players_different_ids(self):
        cards0 = create_starting_deck(0)
        cards1 = create_starting_deck(1)
        ids0 = {c.id for c in cards0}
        ids1 = {c.id for c in cards1}
        assert ids0.isdisjoint(ids1)

    def test_heat_cards(self):
        heat = create_heat_cards(0, 6)
        assert len(heat) == 6
        assert all(c.card_type == CardType.HEAT for c in heat)
        assert all(c.value == 0 for c in heat)
