"""Random agent for the HEAT board game."""

from __future__ import annotations

import random

from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.engine.phases import ReactDecision
from heat.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that makes uniformly random legal choices.

    Uses a seeded RNG for reproducibility.
    """

    def __init__(self, seed: int | None = None, name: str = "RandomAgent") -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        return self._rng.choice(legal_gears)

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        return self._rng.choice(legal_plays)

    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        cooldown_count = self._rng.randint(0, max_cooldown)
        use_boost = can_boost and self._rng.choice([True, False])
        use_adrenaline_speed = has_adrenaline and self._rng.choice([True, False])
        use_adrenaline_cooldown = has_adrenaline and self._rng.choice([True, False])
        return ReactDecision(
            cooldown_count=cooldown_count,
            use_boost=use_boost,
            use_adrenaline_speed=use_adrenaline_speed,
            use_adrenaline_cooldown=use_adrenaline_cooldown,
        )

    def choose_slipstream(
        self,
        state: GameState,
        player_id: int,
    ) -> bool:
        return self._rng.choice([True, False])

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        if not discardable:
            return []
        # Randomly choose a subset: for each card, flip a coin
        return [c for c in discardable if self._rng.choice([True, False])]
