"""Card, CardType, and Deck models for the HEAT board game."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class CardType(Enum):
    """Types of cards in the game."""

    SPEED = "speed"
    HEAT = "heat"
    STRESS = "stress"
    UPGRADE = "upgrade"


@dataclass(frozen=True)
class Card:
    """A single card in the game.

    Frozen so cards can be used as dict keys and in sets.
    """

    card_type: CardType
    value: int
    id: str

    @property
    def display_name(self) -> str:
        if self.card_type == CardType.SPEED:
            return str(self.value)
        elif self.card_type == CardType.HEAT:
            return "Heat"
        elif self.card_type == CardType.STRESS:
            return "Stress"
        elif self.card_type == CardType.UPGRADE:
            return f"Upgrade({self.value})"
        return self.card_type.value

    def __repr__(self) -> str:
        return self.display_name


class Deck:
    """A deck with draw pile, discard pile, and automatic reshuffling."""

    def __init__(
        self,
        cards: list[Card] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._draw_pile: list[Card] = list(cards) if cards else []
        self._discard_pile: list[Card] = []
        self._rng.shuffle(self._draw_pile)

    def attach_rng(self, rng: random.Random, reshuffle: bool = True) -> None:
        """Re-bind this deck to a shared RNG.

        If ``reshuffle`` is True, re-shuffle the draw pile from the new
        stream so deck order is a deterministic function of the supplied
        RNG (e.g. the game seed) rather than the pre-attach local stream.

        Because ``random.shuffle`` permutes the *current* list order, the
        draw pile is first restored to a canonical (id-sorted) order before
        shuffling. This guarantees the post-attach order depends only on the
        attached RNG and the deck's contents -- never on the throwaway local
        stream used when the deck was first constructed.
        """
        self._rng = rng
        if reshuffle:
            self._draw_pile.sort(key=lambda c: c.id)
            rng.shuffle(self._draw_pile)

    @property
    def draw_pile_size(self) -> int:
        return len(self._draw_pile)

    @property
    def discard_pile_size(self) -> int:
        return len(self._discard_pile)

    @property
    def total_size(self) -> int:
        return len(self._draw_pile) + len(self._discard_pile)

    @property
    def draw_pile(self) -> tuple[Card, ...]:
        """Read-only snapshot of the draw pile (bottom..top)."""
        return tuple(self._draw_pile)

    @property
    def discard_pile(self) -> tuple[Card, ...]:
        """Read-only snapshot of the discard pile."""
        return tuple(self._discard_pile)

    def clone(self, rng: random.Random | None = None) -> Deck:
        """Return a deep-ish copy of this deck.

        New pile lists are created, but the ``Card`` objects are frozen and
        therefore shared safely (no need to copy them).

        RNG policy: the caller supplies the clone's RNG (normally the cloned
        ``GameState``'s rng). If None, the clone gets a fresh, independent
        ``random.Random()`` so the two decks never share a stream.
        """
        new: Deck = Deck.__new__(Deck)
        new._draw_pile = list(self._draw_pile)
        new._discard_pile = list(self._discard_pile)
        new._rng = rng if rng is not None else random.Random()
        return new

    def draw(self, count: int = 1) -> list[Card]:
        """Draw cards from the draw pile, reshuffling discard if needed."""
        drawn: list[Card] = []
        for _ in range(count):
            if not self._draw_pile:
                if not self._discard_pile:
                    break  # No cards left anywhere
                self._reshuffle()
            if self._draw_pile:
                drawn.append(self._draw_pile.pop())
        return drawn

    def discard(self, cards: list[Card]) -> None:
        """Add cards to the discard pile."""
        self._discard_pile.extend(cards)

    def add_to_draw_pile(self, cards: list[Card]) -> None:
        """Add cards directly to the draw pile (e.g., heat cards entering the deck)."""
        self._draw_pile.extend(cards)
        self._rng.shuffle(self._draw_pile)

    def _reshuffle(self) -> None:
        """Shuffle the discard pile into the draw pile."""
        self._draw_pile.extend(self._discard_pile)
        self._discard_pile.clear()
        self._rng.shuffle(self._draw_pile)

    def __iter__(self) -> Iterator[Card]:
        """Iterate over all cards (draw pile + discard pile)."""
        yield from self._draw_pile
        yield from self._discard_pile

    def __len__(self) -> int:
        return self.total_size


def create_starting_deck(player_id: int) -> list[Card]:
    """Create the standard 12-card starting deck for a player.

    Starting deck: 3 copies each of values 1, 2, 3, 4 (12 cards total).
    """
    prefix = f"p{player_id}"
    cards: list[Card] = []
    card_id = 0
    for value in [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]:
        cards.append(Card(CardType.SPEED, value, f"{prefix}_spd_{card_id}"))
        card_id += 1
    return cards


def create_heat_cards(player_id: int, count: int = 6) -> list[Card]:
    """Create heat cards for a player's heat pool."""
    return [
        Card(CardType.HEAT, 0, f"p{player_id}_heat_{i}")
        for i in range(count)
    ]


def create_starting_upgrade_cards(player_id: int) -> list[Card]:
    """Create the 3 Starting Upgrade cards per the official rules.

    Returns a list of 3 cards to be shuffled into the player's draw deck:
    1. A 0-value upgrade card
    2. A 5-value upgrade card
    3. An extra Heat card

    During stress/boost resolution, Upgrade cards are NOT Basic cards.
    When flipped, they are discarded and flipping continues until a
    Basic (SPEED type) card is found.
    """
    prefix = f"p{player_id}"
    return [
        Card(CardType.UPGRADE, 0, f"{prefix}_upg_0"),
        Card(CardType.UPGRADE, 5, f"{prefix}_upg_5"),
        Card(CardType.HEAT, 0, f"{prefix}_upg_heat"),
    ]


def create_stress_cards(player_id: int, count: int = 3) -> list[Card]:
    """Create stress cards to be shuffled into a player's deck."""
    return [
        Card(CardType.STRESS, 0, f"p{player_id}_stress_{i}")
        for i in range(count)
    ]
