"""Abstract base class for HEAT board game agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.engine.phases import ReactDecision


class BaseAgent(ABC):
    """Abstract base agent that all concrete agents must subclass.

    Defines the five decision methods matching the Agent Protocol
    in engine/game.py exactly.
    """

    def __init__(self, name: str = "Agent") -> None:
        self.name = name

    @abstractmethod
    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Choose a gear shift.

        legal_gears: list of (new_gear, heat_cost) tuples.
        Returns the chosen (new_gear, heat_cost).
        """
        ...

    @abstractmethod
    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        """Choose which cards to play."""
        ...

    @abstractmethod
    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        """Choose React actions: cooldown, boost, and adrenaline usage."""
        ...

    @abstractmethod
    def choose_slipstream(
        self,
        state: GameState,
        player_id: int,
    ) -> bool:
        """Choose whether to take slipstream. Only called if eligible."""
        ...

    @abstractmethod
    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        """Choose which hand cards to voluntarily discard."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"
