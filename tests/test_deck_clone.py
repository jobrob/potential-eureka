"""Tests for Deck read-only accessors and clone (Step B1)."""

from __future__ import annotations

import random

from heat.models.cards import Card, CardType, Deck


def _make_cards(n: int) -> list[Card]:
    return [Card(CardType.SPEED, i, f"c{i}") for i in range(n)]


class TestDeckAccessors:
    def test_draw_pile_snapshot_matches_iteration(self) -> None:
        deck = Deck(_make_cards(8), rng=random.Random(1))
        assert list(deck.draw_pile) == list(deck)

    def test_draw_pile_is_tuple_and_immutable(self) -> None:
        deck = Deck(_make_cards(5), rng=random.Random(1))
        snap = deck.draw_pile
        assert isinstance(snap, tuple)
        # Mutating the deck does not retroactively change the snapshot.
        deck.draw(1)
        assert len(snap) == 5
        assert deck.draw_pile_size == 4

    def test_discard_pile_accessor(self) -> None:
        deck = Deck(_make_cards(4), rng=random.Random(1))
        drawn = deck.draw(2)
        deck.discard(drawn)
        assert set(deck.discard_pile) == set(drawn)
        assert isinstance(deck.discard_pile, tuple)


class TestDeckClone:
    def test_clone_equal_contents(self) -> None:
        deck = Deck(_make_cards(10), rng=random.Random(3))
        deck.discard(deck.draw(3))
        clone = deck.clone()
        assert list(clone.draw_pile) == list(deck.draw_pile)
        assert list(clone.discard_pile) == list(deck.discard_pile)

    def test_clone_independent_lists(self) -> None:
        deck = Deck(_make_cards(6), rng=random.Random(3))
        clone = deck.clone()
        clone.draw(6)
        # Draining the clone must not touch the source.
        assert deck.draw_pile_size == 6
        assert clone.draw_pile_size == 0

    def test_clone_shares_card_identity(self) -> None:
        deck = Deck(_make_cards(5), rng=random.Random(3))
        clone = deck.clone()
        for original, copied in zip(deck.draw_pile, clone.draw_pile):
            assert original is copied

    def test_clone_rng_independent(self) -> None:
        deck = Deck(_make_cards(5), rng=random.Random(3))
        clone = deck.clone(rng=random.Random(99))
        assert clone._rng is not deck._rng

    def test_clone_default_rng_is_fresh(self) -> None:
        deck = Deck(_make_cards(5), rng=random.Random(3))
        clone = deck.clone()
        assert clone._rng is not deck._rng
