"""Per-player state model for the HEAT board game."""

from __future__ import annotations

from dataclasses import dataclass, field

from heat.models.cards import (
    Card,
    CardType,
    Deck,
    create_heat_cards,
    create_starting_deck,
    create_starting_upgrade_cards,
    create_stress_cards,
)


@dataclass
class PlayerState:
    """Mutable state for a single player during a game.

    Attributes:
        player_id: Unique player identifier (0-based).
        name: Display name.
        deck: The player's personal draw/discard deck.
        hand: Cards currently in hand.
        gear: Current gear (1-4).
        position: Current space index on the track.
        lap: Current lap number (0 = not started, 1 = first lap, etc.).
        heat_pool: Heat cards not yet in the deck (available to pay costs).
        cooldown_pool: Heat cards removed via cooldown (returned to heat_pool at round end).
        spun_out: Whether the player spun out this round.
        finished: Whether the player has crossed the finish line.
        finish_order: Position the player finished in (0 = not finished).
        turn_start_position: Space index at the start of this turn (used
            for corner checking).
        turn_start_lap: Lap number at the start of this turn (used with
            turn_start_position to compute lap-aware spaces moved).
    """

    player_id: int
    name: str
    deck: Deck
    hand: list[Card] = field(default_factory=list)
    gear: int = 1
    position: int = 0
    lap: int = 0
    heat_pool: list[Card] = field(default_factory=list)
    cooldown_pool: list[Card] = field(default_factory=list)
    spun_out: bool = False
    finished: bool = False
    finish_order: int = 0
    # Per-turn transient fields (cleared each round)
    cards_played: list[Card] = field(default_factory=list)
    boost_used_this_turn: bool = False
    speed_from_cards: int = 0
    speed_from_boost: int = 0
    speed_from_adrenaline: int = 0
    slipstream_moved: int = 0
    cluttered: bool = False
    turn_start_position: int = 0
    turn_start_lap: int = 0

    @classmethod
    def create(cls, player_id: int, name: str | None = None) -> PlayerState:
        """Create a new player with a standard starting deck and heat pool.

        Draws an initial hand of 7 cards.
        """
        if name is None:
            name = f"Player {player_id}"

        speed_cards = create_starting_deck(player_id)
        upgrade_cards = create_starting_upgrade_cards(player_id)
        stress_cards = create_stress_cards(player_id)
        all_deck_cards = speed_cards + upgrade_cards + stress_cards
        deck = Deck(all_deck_cards)
        heat_pool = create_heat_cards(player_id)

        player = cls(
            player_id=player_id,
            name=name,
            deck=deck,
            heat_pool=heat_pool,
        )
        # Draw initial hand of 7 cards
        player.hand = player.deck.draw(7)
        return player

    @property
    def speed_cards_in_hand(self) -> list[Card]:
        """Return only speed cards from the hand."""
        return [c for c in self.hand if c.card_type == CardType.SPEED]

    @property
    def heat_in_hand(self) -> list[Card]:
        """Return heat cards currently in hand (clogging)."""
        return [c for c in self.hand if c.card_type == CardType.HEAT]

    @property
    def stress_in_hand(self) -> list[Card]:
        """Return stress cards currently in hand."""
        return [c for c in self.hand if c.card_type == CardType.STRESS]

    @property
    def heat_available(self) -> int:
        """Number of heat cards available to pay costs."""
        return len(self.heat_pool)

    def pay_heat(self, amount: int) -> list[Card]:
        """Remove heat cards from the pool and add them to the deck.

        Returns the heat cards that were paid.
        Raises ValueError if not enough heat available.
        """
        if amount > len(self.heat_pool):
            raise ValueError(
                f"Player {self.player_id} cannot pay {amount} heat "
                f"(only {len(self.heat_pool)} available)"
            )
        paid = self.heat_pool[:amount]
        self.heat_pool = self.heat_pool[amount:]
        # Heat cards go into the discard pile (will eventually clog the hand)
        self.deck.discard(paid)
        return paid

    def cooldown(self, amount: int) -> list[Card]:
        """Remove heat cards from the hand back to the heat pool.

        Returns the heat cards that were cooled down.
        """
        heat_cards = self.heat_in_hand
        to_cool = min(amount, len(heat_cards))
        cooled: list[Card] = []
        for card in heat_cards[:to_cool]:
            self.hand.remove(card)
            self.heat_pool.append(card)
            cooled.append(card)
        return cooled
