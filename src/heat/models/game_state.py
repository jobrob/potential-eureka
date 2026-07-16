"""Full game state model for the HEAT board game."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from heat.models.player_state import PlayerState
from heat.models.track import Track


class Phase(Enum):
    """Phases within a single round."""

    SHIFT_GEARS = "shift_gears"
    PLAY_CARDS = "play_cards"
    REVEAL_AND_MOVE = "reveal_and_move"
    ADRENALINE = "adrenaline"
    REACT = "react"
    SLIPSTREAM = "slipstream"
    CHECK_CORNER = "check_corner"
    DISCARD = "discard"
    REPLENISH = "replenish"


@dataclass
class GameEvent:
    """A logged event for replay and training data.

    Attributes:
        round_num: The round in which this event occurred.
        phase: The phase during which this event occurred.
        player_id: The player involved (None for global events).
        event_type: Short identifier for the event kind.
        data: Arbitrary event payload.
    """

    round_num: int
    phase: Phase
    player_id: int | None
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    """Complete state of a HEAT game in progress.

    Attributes:
        track: The race track being played on.
        players: List of player states, indexed by player_id.
        round_num: Current round number (1-based).
        current_phase: Current phase within the round.
        turn_order: Player IDs in current turn order (leader last for slipstream).
        event_log: Full history of game events (toggleable for performance).
        logging_enabled: Whether to record events in the log.
        rng: The canonical RNG for this game. Owned by GameState and bound
            into each player's Deck so shuffles are reproducible and isolated
            per game (rather than relying on the module-global random).
    """

    track: Track
    players: list[PlayerState]
    round_num: int = 1
    current_phase: Phase = Phase.SHIFT_GEARS
    turn_order: list[int] = field(default_factory=list)
    event_log: list[GameEvent] = field(default_factory=list)
    logging_enabled: bool = True
    starting_player_count: int = 0
    rng: random.Random = field(default_factory=random.Random)
    _stress_counter: int = 0

    def next_stress_id(self) -> int:
        """Return a unique counter value for stress card IDs."""
        self._stress_counter += 1
        return self._stress_counter

    def __post_init__(self) -> None:
        if not self.turn_order:
            self.turn_order = [p.player_id for p in self.players]

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def active_players(self) -> list[PlayerState]:
        """Players who haven't finished yet."""
        return [p for p in self.players if not p.finished]

    @property
    def finished_players(self) -> list[PlayerState]:
        """Players who have crossed the finish line, in finish order."""
        return sorted(
            [p for p in self.players if p.finished],
            key=lambda p: p.finish_order,
        )

    @property
    def is_game_over(self) -> bool:
        """Game ends only when every player has finished the race."""
        return all(p.finished for p in self.players)

    def get_player(self, player_id: int) -> PlayerState:
        """Get a player by ID."""
        return self.players[player_id]

    def log_event(
        self,
        event_type: str,
        player_id: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Record a game event if logging is enabled."""
        if self.logging_enabled:
            self.event_log.append(
                GameEvent(
                    round_num=self.round_num,
                    phase=self.current_phase,
                    player_id=player_id,
                    event_type=event_type,
                    data=data or {},
                )
            )

    def compute_turn_order(self) -> list[int]:
        """Recompute turn order based on position (leader = last index).

        Players further ahead go first. Ties broken by player_id (lower first).
        """
        active = self.active_players
        ordered = sorted(active, key=lambda p: (-p.position, -p.lap, p.player_id))
        self.turn_order = [p.player_id for p in ordered]
        return self.turn_order

    def clone(
        self,
        *,
        copy_event_log: bool = False,
        reseed: int | None = None,
    ) -> GameState:
        """Return a faithful deep-ish copy of this game state.

        RNG policy (the explicit interaction with item A):
          - ``reseed`` is None (default): the clone gets its OWN rng forked
            deterministically from this state's rng via
            ``random.Random(self.rng.random())``. This makes the clone
            independent (mutating one stream never touches the other) AND
            reproducible (forking the same parent rng twice yields the same
            child stream). The parent's rng is advanced by one draw.
          - ``reseed`` is an int: the clone's rng is ``random.Random(reseed)``,
            for callers (e.g. ``env.reset()``) that want an explicit fresh seed.

        Cloned player decks are bound to the clone's rng.

        ``Track`` is immutable race configuration and is SHARED (not copied).

        ``event_log`` is NOT copied by default (it is large, append-only replay
        data a lookahead/rollout clone does not need). When ``copy_event_log``
        is True the list is shallow-copied (GameEvents are treated as immutable
        records and shared).
        """
        new_rng = (
            random.Random(reseed)
            if reseed is not None
            else random.Random(self.rng.random())
        )
        new = GameState(
            track=self.track,
            players=[p.clone(new_rng) for p in self.players],
            round_num=self.round_num,
            current_phase=self.current_phase,
            turn_order=list(self.turn_order),
            event_log=list(self.event_log) if copy_event_log else [],
            logging_enabled=self.logging_enabled,
            starting_player_count=self.starting_player_count,
            rng=new_rng,
        )
        new._stress_counter = self._stress_counter
        return new

    @classmethod
    def create(
        cls,
        track: Track,
        num_players: int,
        player_names: list[str] | None = None,
        logging_enabled: bool = True,
        seed: int | None = None,
    ) -> GameState:
        """Create a new game with players placed at start positions.

        RNG policy:
          - ``seed`` is an int: the game RNG is ``random.Random(seed)`` and
            the whole game (deck order, initial hands, in-game reshuffles)
            is a deterministic function of that seed alone.
          - ``seed`` is None (default): the game RNG is forked from the
            CURRENT global ``random`` state via ``random.Random(
            random.random())``. This preserves backward compatibility with
            callers/tests that seed the global ``random`` module before
            constructing a game (e.g. ``random.seed(999); Game(...)``): the
            global stream deterministically seeds the game RNG. New code
            should pass an explicit ``seed`` and never rely on global state.
        """
        if player_names and len(player_names) != num_players:
            raise ValueError("player_names length must match num_players")

        if seed is not None:
            rng = random.Random(seed)
        else:
            # Fork from the global random state for back-compat with the
            # legacy "seed the global module then build a Game" pattern.
            rng = random.Random(random.random())

        players: list[PlayerState] = []
        for i in range(num_players):
            name = player_names[i] if player_names else None
            # Build the deck without drawing the hand yet; attach the game
            # RNG (re-shuffling) so deck order is seed-determined, THEN draw.
            player = PlayerState.create(i, name, draw_hand=False)
            player.deck.attach_rng(rng)
            player.hand = player.deck.draw(7)
            # Place at starting position
            if i < len(track.start_positions):
                player.position = track.start_positions[i]
            players.append(player)

        return cls(
            track=track,
            players=players,
            logging_enabled=logging_enabled,
            starting_player_count=num_players,
            rng=rng,
        )
