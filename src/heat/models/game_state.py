"""Full game state model for the HEAT board game."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
    data: dict = field(default_factory=dict)


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
    """

    track: Track
    players: list[PlayerState]
    round_num: int = 1
    current_phase: Phase = Phase.SHIFT_GEARS
    turn_order: list[int] = field(default_factory=list)
    event_log: list[GameEvent] = field(default_factory=list)
    logging_enabled: bool = True

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
        """Game ends when all players finish or only one remains."""
        return all(p.finished for p in self.players)

    def get_player(self, player_id: int) -> PlayerState:
        """Get a player by ID."""
        return self.players[player_id]

    def log_event(
        self,
        event_type: str,
        player_id: int | None = None,
        data: dict | None = None,
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

    @classmethod
    def create(
        cls,
        track: Track,
        num_players: int,
        player_names: list[str] | None = None,
        logging_enabled: bool = True,
    ) -> GameState:
        """Create a new game with players placed at start positions."""
        if player_names and len(player_names) != num_players:
            raise ValueError("player_names length must match num_players")

        players: list[PlayerState] = []
        for i in range(num_players):
            name = player_names[i] if player_names else None
            player = PlayerState.create(i, name)
            # Place at starting position
            if i < len(track.start_positions):
                player.position = track.start_positions[i]
            players.append(player)

        return cls(
            track=track,
            players=players,
            logging_enabled=logging_enabled,
        )
