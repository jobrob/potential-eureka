"""Tests for injectable RNG on Deck (Sprint 5, Step A1)."""

from __future__ import annotations

import random

from heat.models.cards import Card, CardType, Deck


def _make_cards(n: int) -> list[Card]:
    return [Card(CardType.SPEED, i, f"c{i}") for i in range(n)]


class TestDeckInjectedRng:
    def test_same_seed_same_order(self) -> None:
        """Two decks built with equal-seeded RNGs shuffle identically."""
        cards1 = _make_cards(20)
        cards2 = _make_cards(20)
        deck1 = Deck(cards1, rng=random.Random(42))
        deck2 = Deck(cards2, rng=random.Random(42))
        assert [c.id for c in deck1] == [c.id for c in deck2]

    def test_different_seed_different_order(self) -> None:
        """Different seeds (very likely) produce different orders."""
        deck1 = Deck(_make_cards(30), rng=random.Random(1))
        deck2 = Deck(_make_cards(30), rng=random.Random(2))
        assert [c.id for c in deck1] != [c.id for c in deck2]

    def test_default_rng_still_shuffles(self) -> None:
        """With no rng supplied, a deck is still constructed and usable."""
        deck = Deck(_make_cards(12))
        assert deck.total_size == 12
        # All cards still present regardless of order.
        assert {c.id for c in deck} == {f"c{i}" for i in range(12)}

    def test_injected_rng_used_for_reshuffle(self) -> None:
        """Draw + discard + reshuffle consumes the injected stream
        reproducibly."""
        def run() -> list[str]:
            deck = Deck(_make_cards(5), rng=random.Random(7))
            drawn = deck.draw(5)
            deck.discard(drawn)
            # Force a reshuffle by drawing again from an empty draw pile.
            redrawn = deck.draw(5)
            return [c.id for c in redrawn]

        assert run() == run()

    def test_injected_rng_used_for_add_to_draw_pile(self) -> None:
        """add_to_draw_pile shuffles via the injected stream reproducibly."""
        def run() -> list[str]:
            deck = Deck(_make_cards(4), rng=random.Random(11))
            deck.add_to_draw_pile([Card(CardType.HEAT, 0, "h0")])
            return [c.id for c in deck.draw(5)]

        assert run() == run()


class TestAttachRng:
    def test_attach_reshuffle_is_seed_determined(self) -> None:
        """After attach_rng with the same seed, deck order matches
        regardless of the throwaway construction RNG."""
        deck1 = Deck(_make_cards(25), rng=random.Random(123))
        deck2 = Deck(_make_cards(25), rng=random.Random(456))
        # Different construction seeds -> (almost surely) different order.
        assert [c.id for c in deck1] != [c.id for c in deck2]

        deck1.attach_rng(random.Random(999))
        deck2.attach_rng(random.Random(999))
        assert [c.id for c in deck1] == [c.id for c in deck2]

    def test_attach_no_reshuffle_preserves_order(self) -> None:
        """attach_rng(reshuffle=False) rebinds the stream but keeps order."""
        deck = Deck(_make_cards(10), rng=random.Random(5))
        before = [c.id for c in deck]
        deck.attach_rng(random.Random(5), reshuffle=False)
        assert [c.id for c in deck] == before

    def test_attach_binds_stream_for_future_shuffles(self) -> None:
        """After attach, subsequent reshuffles use the attached stream."""
        def run() -> list[str]:
            deck = Deck(_make_cards(6), rng=random.Random(1))
            deck.attach_rng(random.Random(77))
            drawn = deck.draw(6)
            deck.discard(drawn)
            return [c.id for c in deck.draw(6)]

        assert run() == run()
