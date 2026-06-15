"""Game orchestrator for the HEAT board game engine.

Provides the Agent protocol, GameResult, and Game class that runs
a complete HEAT race by coordinating agents, phases, and rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from heat.models.cards import Card
from heat.models.game_state import GameEvent, GameState
from heat.models.track import Track
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.phases import ReactDecision

MAX_ROUNDS: int = 200


class Agent(Protocol):
    """Protocol that agents must satisfy.

    Defined as a Protocol so the engine has zero dependency on the
    agents package. Concrete agent classes in later sprints will
    implement this protocol.
    """

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

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        """Choose which cards to play."""
        ...

    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        """Choose React actions: cooldown, boost, and adrenaline usage.

        max_cooldown: maximum heat cards that can be cooled (gear-based +
                      adrenaline cooldown if applicable).
        can_boost: True if the player has heat to pay and hasn't boosted.
        has_adrenaline: True if the player is eligible for adrenaline.
        Returns a ReactDecision.
        """
        ...

    def choose_slipstream(
        self,
        state: GameState,
        player_id: int,
    ) -> bool:
        """Choose whether to take slipstream. Only called if eligible."""
        ...

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        """Choose which hand cards to voluntarily discard.

        discardable: cards eligible for discard (Speed and Upgrade only,
                     NOT Heat or Stress).
        Returns a subset of discardable (may be empty = keep all).
        """
        ...


@dataclass
class GameResult:
    """Final result of a completed game.

    Attributes:
        finish_order: Player IDs in the order they finished.
        total_rounds: Number of rounds played.
        event_log: Complete event log (empty if logging was disabled).
    """

    finish_order: list[int]
    total_rounds: int
    event_log: list[GameEvent]


class Game:
    """Orchestrates a complete HEAT race.

    Responsibilities:
    - Initialize game state from track + player count.
    - Run the round loop until the game is over.
    - Collect decisions from agents at each decision point.
    - Apply simultaneous phases (steps 1-2) for all players.
    - Run sequential per-player loop (steps 3-9) in turn order.
    - Handle cluttered hand (skip to replenish).
    - Track finish order.
    - Return final results.
    """

    def __init__(
        self,
        track: Track,
        agents: list[Agent],
        player_names: list[str] | None = None,
        logging_enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        num_players = len(agents)
        self._state = GameState.create(
            track,
            num_players,
            player_names=player_names,
            logging_enabled=logging_enabled,
            seed=seed,
        )
        # Map player_id -> Agent for O(1) lookup
        self._agents: dict[int, Agent] = {
            i: agent for i, agent in enumerate(agents)
        }
        # Set initial lap to 1 for all players (race has started)
        for player in self._state.players:
            player.lap = 1

    @property
    def state(self) -> GameState:
        """The current game state."""
        return self._state

    @property
    def is_over(self) -> bool:
        """Whether the game has ended."""
        return self._state.is_game_over

    def run(self) -> GameResult:
        """Run the complete game to termination.

        Returns GameResult with finish order and event log.
        """
        while not self.is_over:
            self.run_round()

            # Safety valve: prevent infinite games
            if self._state.round_num > MAX_ROUNDS:
                self._force_finish_remaining()
                break

        return GameResult(
            finish_order=[p.player_id for p in self._state.finished_players],
            total_rounds=self._state.round_num - 1,
            event_log=self._state.event_log,
        )

    def run_round(self) -> list[GameEvent]:
        """Run a single round of the game.

        Reimplemented as a thin pump over ``run_round_driver``: the driver
        runs the round and pauses at each agent decision point, and
        ``_answer`` dispatches to the matching ``Agent.choose_*`` method.
        Behavior is identical to the previous monolithic loop.

        Returns the list of events generated during this round.
        """
        gen = run_round_driver(self._state)
        try:
            decision = next(gen)
            while True:
                action = self._answer(decision)
                decision = gen.send(action)
        except StopIteration as stop:
            return stop.value or []

    # ------------------------------------------------------------------
    # Decision dispatch (push -> pull bridge to the Agent protocol)
    # ------------------------------------------------------------------

    def _answer(self, decision: Decision) -> object:
        """Dispatch a driver Decision to the owning agent's choose_* method.

        Each branch passes exactly the arguments the pre-refactor inline loop
        passed, so agent behavior is unchanged.
        """
        pid = decision.player_id
        agent = self._agents[pid]
        kind = decision.kind

        if kind == DecisionKind.GEAR:
            legal_gears: list[tuple[int, int]] = decision.legal  # type: ignore[assignment]
            chosen = agent.choose_gear(self._state, pid, legal_gears)
            if chosen not in legal_gears:
                raise ValueError(
                    f"Agent {pid} chose illegal gear shift {chosen}"
                )
            return chosen

        if kind == DecisionKind.CARDS:
            legal_plays: list[tuple[Card, ...]] = decision.legal  # type: ignore[assignment]
            chosen_cards = agent.choose_cards(self._state, pid, legal_plays)
            if chosen_cards not in legal_plays:
                raise ValueError(f"Agent {pid} chose illegal card play")
            return chosen_cards

        if kind == DecisionKind.REACT:
            options: rules.ReactOptions = decision.legal  # type: ignore[assignment]
            # max_cooldown is gear-based; the +1 adrenaline cooldown (if used)
            # is still applied later inside step_react.
            return agent.choose_react(
                self._state,
                pid,
                max_cooldown=options.max_cooldown,
                can_boost=options.can_boost,
                has_adrenaline=options.has_adrenaline,
            )

        if kind == DecisionKind.SLIPSTREAM:
            return agent.choose_slipstream(self._state, pid)

        if kind == DecisionKind.DISCARD:
            discardable: list[Card] = decision.legal  # type: ignore[assignment]
            return agent.choose_discard(self._state, pid, discardable)

        raise ValueError(f"Unknown decision kind {kind}")

    def _force_finish_remaining(self) -> None:
        """Force-finish all remaining players for MAX_ROUNDS safety."""
        active = [p for p in self._state.players if not p.finished]
        # Rank remaining players by position (further ahead = better)
        ranked = sorted(
            active, key=lambda p: (p.lap, p.position), reverse=True
        )
        finished_count = len(
            [p for p in self._state.players if p.finished]
        )
        for player in ranked:
            finished_count += 1
            player.finished = True
            player.finish_order = finished_count
