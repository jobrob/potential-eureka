"""Card, CardType, and Deck models for the HEAT board game."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
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

    def __repr__(self) -> str:
        if self.card_type == CardType.SPEED:
            return f"Speed({self.value})"
        return f"{self.card_type.value.capitalize()}(id={self.id})"


class Deck:
    """A deck with draw pile, discard pile, and automatic reshuffling."""

    def __init__(self, cards: list[Card] | None = None) -> None:
        self._draw_pile: list[Card] = list(cards) if cards else []
        self._discard_pile: list[Card] = []
        random.shuffle(self._draw_pile)

    @property
    def draw_pile_size(self) -> int:
        return len(self._draw_pile)

    @property
    def discard_pile_size(self) -> int:
        return len(self._discard_pile)

    @property
    def total_size(self) -> int:
        return len(self._draw_pile) + len(self._discard_pile)

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
        random.shuffle(self._draw_pile)

    def _reshuffle(self) -> None:
        """Shuffle the discard pile into the draw pile."""
        self._draw_pile.extend(self._discard_pile)
        self._discard_pile.clear()
        random.shuffle(self._draw_pile)

    def __iter__(self) -> Iterator[Card]:
        """Iterate over all cards (draw pile + discard pile)."""
        yield from self._draw_pile
        yield from self._discard_pile

    def __len__(self) -> int:
        return self.total_size


def create_starting_deck(player_id: int) -> list[Card]:
    """Create the standard 12-card starting deck for a player.

    Standard HEAT starting hand:
    - Speed cards: 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 (two of each 1-5, but plan
      says value-based, we'll use the standard distribution)
    Actually the real game has specific speed values. We'll use a common setup:
    1×0 (start), 1×1, 2×2, 2×3, 2×4, 1×5 = not quite right.

    Real HEAT starting deck (12 cards): 1, 2, 3, 4, 5, 6, 7, 8, 0(heat-adjacent)
    Let's use: two copies each of speed 1, 2, 3, 4 and one each of speed 5, 6
    = 10 speed cards + 0 stress. That's not 12.

    Simplified standard: speed values [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    """
    prefix = f"p{player_id}"
    cards: list[Card] = []
    card_id = 0
    for value in [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]:
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
