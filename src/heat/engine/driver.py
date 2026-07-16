"""Step-wise (pausable) round driver for the HEAT engine.

``run_round_driver`` is a generator-based state machine that runs ONE round,
yielding a :class:`Decision` at each agent decision point and resuming with
the action supplied via ``generator.send(...)``. Non-decision phases
auto-advance inside the generator.

This is a mechanical transcription of ``Game.run_round`` (and its
``_collect_gear_decisions`` / ``_collect_card_decisions`` helpers). Where the
pull-driven ``run_round`` calls ``agent.choose_*(...)``, the driver instead
``yield``s a ``Decision`` and waits for the caller to ``send`` the chosen
action. ``Game.run_round`` is reimplemented as a thin pump over this driver,
and a future ``HeatEnv.step`` drives the same generator directly.

The generator RETURNS (via ``StopIteration.value``) the list of events
produced during the round, matching ``run_round``'s return value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generator, cast

from heat.models.cards import Card
from heat.models.game_state import GameEvent, GameState, Phase
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


class DecisionKind(Enum):
    """The kind of agent decision a :class:`Decision` is requesting."""

    GEAR = "gear"            # simultaneous
    CARDS = "cards"          # simultaneous
    REACT = "react"          # sequential, per player
    SLIPSTREAM = "slipstream"
    DISCARD = "discard"


@dataclass
class Decision:
    """A pause point yielded by the driver.

    The caller inspects ``kind`` / ``player_id`` / ``legal`` and sends back an
    action of the matching type:

        GEAR        legal: list[tuple[int, int]]      action: tuple[int, int]
        CARDS       legal: list[tuple[Card, ...]]     action: tuple[Card, ...]
        REACT       legal: rules.ReactOptions         action: ReactDecision
        SLIPSTREAM  legal: bool (always True asked)   action: bool
        DISCARD     legal: list[Card]                 action: list[Card]
    """

    kind: DecisionKind
    player_id: int
    legal: object


# Type alias for clarity: the driver yields Decision, receives an action
# object, and finally returns list[GameEvent].
RoundDriver = Generator[Decision, object, list[GameEvent]]


def simultaneous_decisions(
    state: GameState, kind: DecisionKind
) -> list[Decision]:
    """Return the yielded choices for one unapplied simultaneous phase.

    The helper is intentionally read-only.  Both the round driver and batched
    rollout collectors use it so active-player filtering, seat order, and legal
    payload construction cannot drift between the scalar and batched paths.
    Spun-out gear choices are omitted because the driver resolves them without
    yielding a decision.
    """
    if kind is DecisionKind.GEAR:
        return [
            Decision(
                DecisionKind.GEAR,
                player.player_id,
                rules.legal_gear_shifts(player.gear, player.heat_available),
            )
            for player in state.active_players
            if not player.spun_out
        ]
    if kind is DecisionKind.CARDS:
        return [
            Decision(
                DecisionKind.CARDS,
                player.player_id,
                rules.legal_card_plays(player.hand, player.gear),
            )
            for player in state.active_players
        ]
    raise ValueError(f"{kind.value} is not a simultaneous decision kind")


def run_round_driver(state: GameState) -> RoundDriver:
    """Run ONE round as a pausable generator (see module docstring)."""
    events: list[GameEvent] = []

    # === SIMULTANEOUS STEPS (all players at once) ===

    # 0. Recompute turn order based on current positions
    state.compute_turn_order()

    # 1. SHIFT GEARS (simultaneous): collect every active player's decision
    #    BEFORE applying the phase (no peeking at others' moves).
    gear_decisions: dict[int, tuple[int, int]] = {}
    for player in state.active_players:
        if player.spun_out:
            gear_decisions[player.player_id] = (1, 0)
    for decision in simultaneous_decisions(state, DecisionKind.GEAR):
        legal_gears = cast(list[tuple[int, int]], decision.legal)
        chosen = yield decision
        if chosen not in legal_gears:
            raise ValueError(
                f"Agent {decision.player_id} chose illegal gear shift {chosen}"
            )
        gear_decisions[decision.player_id] = chosen
    events += phase_shift_gears(state, gear_decisions)

    # Capture each active player's hand BEFORE playing cards
    pre_play_hands: dict[int, list[str]] = {}
    if state.logging_enabled:
        for player in state.active_players:
            pre_play_hands[player.player_id] = [
                c.display_name for c in player.hand
            ]

    # 2. PLAY CARDS (simultaneous): collect all, then apply.
    card_decisions: dict[int, tuple[Card, ...]] = {}
    for decision in simultaneous_decisions(state, DecisionKind.CARDS):
        legal_plays = cast(list[tuple[Card, ...]], decision.legal)
        chosen_cards = yield decision
        if chosen_cards not in legal_plays:
            raise ValueError(f"Agent {decision.player_id} chose illegal card play")
        card_decisions[decision.player_id] = cast(tuple[Card, ...], chosen_cards)
    events += phase_play_cards(state, card_decisions)

    # === PER-PLAYER SEQUENTIAL STEPS (front-to-back) ===

    for pid in list(state.turn_order):  # copy since order is stable
        player = state.get_player(pid)
        if player.finished:
            continue

        # Log turn start context
        if state.logging_enabled:
            next_corner, dist = rules.distance_to_next_corner(
                state.track, player.position,
            )
            # Preserve original semantics: no corners -> None distance.
            if next_corner is None:
                next_corner_dist = None
                next_corner_limit = None
            else:
                next_corner_dist = dist
                next_corner_limit = next_corner.speed_limit

            hand_repr = pre_play_hands.get(
                pid, [c.display_name for c in player.hand]
            )
            turn_start_data = {
                "hand": hand_repr,
                "hand_size": len(hand_repr),
                "gear": player.gear,
                "heat_available": player.heat_available,
                "position": player.position,
                "next_corner_dist": next_corner_dist,
                "next_corner_speed_limit": next_corner_limit,
            }
            state.log_event(
                "turn_start",
                player_id=player.player_id,
                data=turn_start_data,
            )
            events.append(GameEvent(
                state.round_num, Phase.REVEAL_AND_MOVE,
                player.player_id, "turn_start",
                turn_start_data,
            ))

        # CLUTTERED HAND CHECK: car does not move. Gear -> 1, skip 3-8.
        if player.cluttered:
            player.gear = 1
            events += step_replenish(state, player)
            continue

        # Step 3: REVEAL & MOVE
        events += step_reveal_and_move(state, player)
        if player.finished:
            events += step_replenish(state, player)
            continue

        # Step 4: ADRENALINE (automatic)
        events += step_adrenaline(state, player)

        # Step 5: REACT (agent decision)
        react_options = rules.legal_react_options(
            player,
            list(state.active_players),
            state.starting_player_count,
        )
        react_decision = cast(
            ReactDecision,
            (yield Decision(DecisionKind.REACT, pid, react_options)),
        )
        events += step_react(state, player, react_decision)
        if player.finished:
            events += step_replenish(state, player)
            continue

        # Step 6: SLIPSTREAM (agent decision if eligible)
        if rules.legal_slipstream(
            player,
            list(state.active_players),
            state.track,
        ):
            take_slip = cast(
                bool,
                (yield Decision(DecisionKind.SLIPSTREAM, pid, True)),
            )
            events += step_slipstream(state, player, take_slip)

        # Step 7: CHECK CORNER
        events += step_check_corner(state, player)

        # Step 8: DISCARD (agent decision)
        discardable = rules.legal_discards(player)
        if discardable:
            to_discard = cast(
                list[Card],
                (yield Decision(DecisionKind.DISCARD, pid, discardable)),
            )
            events += step_discard(state, player, to_discard)

        # Step 9: REPLENISH
        events += step_replenish(state, player)

    # === END OF ROUND ===

    # Clear spun_out flags for next round
    for player in state.active_players:
        player.spun_out = False

    # Advance round counter
    state.round_num += 1

    return events
