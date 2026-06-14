"""Game orchestrator for the HEAT board game engine.

Provides the Agent protocol, GameResult, and Game class that runs
a complete HEAT race by coordinating agents, phases, and rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from heat.models.cards import Card, CardType
from heat.models.game_state import GameEvent, GameState, Phase
from heat.models.player_state import PlayerState
from heat.models.track import Track
from heat.engine import rules
from heat.engine.phases import (
    ReactDecision,
    phase_play_cards,
    phase_shift_gears,
    step_adrenaline,
    step_check_corner,
    step_discard,
    step_react,
    step_replenish,
    step_reveal_and_move,
    step_slipstream,
)

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
    ) -> None:
        num_players = len(agents)
        self._state = GameState.create(
            track,
            num_players,
            player_names=player_names,
            logging_enabled=logging_enabled,
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

        Returns the list of events generated during this round.
        """
        events: list[GameEvent] = []

        # === SIMULTANEOUS STEPS (all players at once) ===

        # 0. Recompute turn order based on current positions
        self._state.compute_turn_order()

        # 1. SHIFT GEARS (simultaneous)
        gear_decisions = self._collect_gear_decisions()
        events += phase_shift_gears(self._state, gear_decisions)

        # 2. PLAY CARDS (simultaneous)
        card_decisions = self._collect_card_decisions()
        events += phase_play_cards(self._state, card_decisions)

        # === PER-PLAYER SEQUENTIAL STEPS (front-to-back) ===

        for pid in list(self._state.turn_order):  # copy since order is stable
            player = self._state.get_player(pid)
            if player.finished:
                continue

            # Log turn start context
            if self._state.logging_enabled:
                next_corner_dist = None
                next_corner_limit = None
                for corner in self._state.track.corners:
                    dist = (corner.start - player.position) % self._state.track.length
                    if dist == 0:
                        dist = self._state.track.length  # already at/past this corner
                    if next_corner_dist is None or dist < next_corner_dist:
                        next_corner_dist = dist
                        next_corner_limit = corner.speed_limit

                hand_repr = [repr(c) for c in player.hand]
                turn_start_data = {
                    "hand": hand_repr,
                    "hand_size": len(player.hand),
                    "gear": player.gear,
                    "heat_available": player.heat_available,
                    "position": player.position,
                    "next_corner_dist": next_corner_dist,
                    "next_corner_speed_limit": next_corner_limit,
                }
                self._state.log_event(
                    "turn_start",
                    player_id=player.player_id,
                    data=turn_start_data,
                )
                events.append(GameEvent(
                    self._state.round_num, Phase.REVEAL_AND_MOVE,
                    player.player_id, "turn_start",
                    turn_start_data,
                ))

            # CLUTTERED HAND CHECK: If the player had a cluttered hand,
            # their car does not move. Set gear to 1, skip steps 3-8,
            # go straight to replenish.
            if player.cluttered:
                player.gear = 1
                events += step_replenish(self._state, player)
                continue

            # Step 3: REVEAL & MOVE
            events += step_reveal_and_move(self._state, player)
            if player.finished:
                events += step_replenish(self._state, player)
                continue

            # Step 4: ADRENALINE (automatic)
            events += step_adrenaline(self._state, player)

            # Step 5: REACT (agent decision)
            react_decision = self._collect_react_decision(player)
            events += step_react(self._state, player, react_decision)
            if player.finished:
                events += step_replenish(self._state, player)
                continue

            # Step 6: SLIPSTREAM (agent decision if eligible)
            if rules.slipstream_eligible(
                player,
                list(self._state.active_players),
                self._state.track,
            ):
                take_slip = self._agents[pid].choose_slipstream(
                    self._state, pid
                )
                events += step_slipstream(self._state, player, take_slip)

            # Step 7: CHECK CORNER
            events += step_check_corner(self._state, player)

            # Step 8: DISCARD (agent decision)
            discardable = [
                c
                for c in player.hand
                if c.card_type in (CardType.SPEED, CardType.UPGRADE)
            ]
            if discardable:
                to_discard = self._agents[pid].choose_discard(
                    self._state, pid, discardable
                )
                events += step_discard(self._state, player, to_discard)

            # Step 9: REPLENISH
            events += step_replenish(self._state, player)

        # === END OF ROUND ===

        # Clear spun_out flags for next round
        for player in self._state.active_players:
            player.spun_out = False

        # Advance round counter
        self._state.round_num += 1

        return events

    # ------------------------------------------------------------------
    # Decision collection methods
    # ------------------------------------------------------------------

    def _collect_gear_decisions(self) -> dict[int, tuple[int, int]]:
        """Query each active agent for their gear choice."""
        decisions: dict[int, tuple[int, int]] = {}
        for player in self._state.active_players:
            pid = player.player_id
            if player.spun_out:
                decisions[pid] = (1, 0)  # Forced to gear 1, no cost
            else:
                legal = rules.legal_gear_shifts(
                    player.gear, player.heat_available
                )
                chosen = self._agents[pid].choose_gear(
                    self._state, pid, legal
                )
                if chosen not in legal:
                    raise ValueError(
                        f"Agent {pid} chose illegal gear shift {chosen}"
                    )
                decisions[pid] = chosen
        return decisions

    def _collect_card_decisions(self) -> dict[int, tuple[Card, ...]]:
        """Query each active agent for their card play choice."""
        decisions: dict[int, tuple[Card, ...]] = {}
        for player in self._state.active_players:
            pid = player.player_id
            legal = rules.legal_card_plays(player.hand, player.gear)
            chosen = self._agents[pid].choose_cards(
                self._state, pid, legal
            )
            if chosen not in legal:
                raise ValueError(
                    f"Agent {pid} chose illegal card play"
                )
            # Detect cluttered hand
            if rules.is_cluttered_hand(player.hand, player.gear):
                player.cluttered = True
            decisions[pid] = chosen
        return decisions

    def _collect_react_decision(self, player: PlayerState) -> ReactDecision:
        """Query an agent for their React decision."""
        pid = player.player_id
        gear_cooldown = rules.cooldown_amount(player.gear)
        has_adrenaline = rules.adrenaline_eligible(
            player,
            list(self._state.active_players),
            self._state.starting_player_count,
        )

        # Max cooldown = gear-based + 1 if adrenaline cooldown is available
        max_cooldown = gear_cooldown  # adrenaline adds to this if used

        can_boost = (
            player.heat_available > 0 and not player.boost_used_this_turn
        )

        return self._agents[pid].choose_react(
            self._state,
            pid,
            max_cooldown=max_cooldown,
            can_boost=can_boost,
            has_adrenaline=has_adrenaline,
        )

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
