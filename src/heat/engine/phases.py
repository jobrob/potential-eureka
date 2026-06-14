"""Phase and step functions for the HEAT board game engine.

Simultaneous phases process all players at once (called once per round):
- phase_shift_gears
- phase_play_cards

Per-player steps process one player at a time (called per player in turn order):
- step_reveal_and_move
- step_adrenaline
- step_react
- step_slipstream
- step_check_corner
- step_discard
- step_replenish
"""

from __future__ import annotations

from dataclasses import dataclass

from heat.models.cards import Card, CardType
from heat.models.game_state import GameEvent, GameState, Phase
from heat.models.player_state import PlayerState
from heat.engine import rules



@dataclass
class ReactDecision:
    """A player's decision for the React step.

    Attributes:
        cooldown_count: How many heat cards to cool from hand (0 to max).
        use_boost: Whether to pay 1 heat for a boost flip.
        use_adrenaline_speed: Whether to use adrenaline +1 speed.
        use_adrenaline_cooldown: Whether to use adrenaline +1 cooldown.
    """

    cooldown_count: int = 0
    use_boost: bool = False
    use_adrenaline_speed: bool = False
    use_adrenaline_cooldown: bool = False


# ---------------------------------------------------------------------------
# Simultaneous phases
# ---------------------------------------------------------------------------


def phase_shift_gears(
    state: GameState,
    decisions: dict[int, tuple[int, int]],
) -> list[GameEvent]:
    """All players simultaneously choose a new gear.

    decisions: {player_id: (new_gear, heat_cost)}

    Validates each choice against legal_gear_shifts(). Raises ValueError
    for illegal shifts. Updates player.gear and pays heat if needed.
    Skips finished players. Spun-out players are forced to gear 1.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.SHIFT_GEARS

    for pid, (new_gear, heat_cost) in decisions.items():
        player = state.get_player(pid)

        if player.finished:
            continue

        # Spun-out players are forced to gear 1
        if player.spun_out:
            if new_gear != 1 or heat_cost != 0:
                raise ValueError(
                    f"Player {pid} is spun out and must shift to gear 1 "
                    f"with 0 heat cost, got ({new_gear}, {heat_cost})"
                )
            player.gear = 1
            state.log_event(
                "gear_shift",
                player_id=pid,
                data={"new_gear": 1, "heat_cost": 0, "spun_out": True},
            )
            events.append(
                GameEvent(
                    state.round_num, Phase.SHIFT_GEARS, pid,
                    "gear_shift",
                    {"new_gear": 1, "heat_cost": 0, "spun_out": True},
                )
            )
            continue

        # Validate the choice
        legal = rules.legal_gear_shifts(player.gear, player.heat_available)
        if (new_gear, heat_cost) not in legal:
            raise ValueError(
                f"Player {pid} chose illegal gear shift "
                f"({new_gear}, {heat_cost}). Legal options: {legal}"
            )

        old_gear = player.gear
        player.gear = new_gear
        if heat_cost > 0:
            player.pay_heat(heat_cost)

        state.log_event(
            "gear_shift",
            player_id=pid,
            data={
                "old_gear": old_gear,
                "new_gear": new_gear,
                "heat_cost": heat_cost,
            },
        )
        events.append(
            GameEvent(
                state.round_num, Phase.SHIFT_GEARS, pid,
                "gear_shift",
                {"old_gear": old_gear, "new_gear": new_gear, "heat_cost": heat_cost},
            )
        )

    return events


def phase_play_cards(
    state: GameState,
    decisions: dict[int, tuple[Card, ...]],
) -> list[GameEvent]:
    """All players simultaneously choose which cards to play.

    decisions: {player_id: tuple_of_cards_to_play}

    Validates against legal_card_plays(). Removes played cards from hand.
    Stores the played cards on player.cards_played.
    Detects cluttered hand and sets player.cluttered flag.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.PLAY_CARDS

    for pid, cards in decisions.items():
        player = state.get_player(pid)

        if player.finished:
            continue

        # Detect cluttered hand BEFORE removing cards
        if rules.is_cluttered_hand(player.hand, player.gear):
            player.cluttered = True

        # Validate the choice
        legal = rules.legal_card_plays(player.hand, player.gear)
        if cards not in legal:
            raise ValueError(
                f"Player {pid} chose illegal card play. "
                f"Cards: {cards}, Legal count: {len(legal)}"
            )

        # Remove played cards from hand, store in cards_played
        player.cards_played = list(cards)
        for card in cards:
            player.hand.remove(card)

        state.log_event(
            "play_cards",
            player_id=pid,
            data={
                "cards": [c.id for c in cards],
                "cluttered": player.cluttered,
            },
        )
        events.append(
            GameEvent(
                state.round_num, Phase.PLAY_CARDS, pid,
                "play_cards",
                {"cards": [c.id for c in cards], "cluttered": player.cluttered},
            )
        )

    return events


# ---------------------------------------------------------------------------
# Per-player step functions
# ---------------------------------------------------------------------------


def step_reveal_and_move(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Reveal played cards and move the player.

    1. Save current position as turn_start_position.
    2. Reveal cards_played.
    3. Resolve any Stress cards (flip until Basic card found for each).
    4. Calculate total speed = sum of card values + stress resolution values.
    5. Store speed_from_cards on player.
    6. Calculate new position (handle lap wrap).
    7. Update position and lap.
    8. Check if player crossed finish line on final lap.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.REVEAL_AND_MOVE

    # Save start-of-turn position for corner checking later
    player.turn_start_position = player.position

    # Calculate base speed from non-stress cards
    total_speed = rules.calculate_speed(tuple(player.cards_played))

    # Resolve stress cards
    for card in player.cards_played:
        if card.card_type == CardType.STRESS:
            value, flipped = rules.resolve_stress_card(player.deck)
            total_speed += value

            # The basic card found (last in flipped) goes to played area
            # Non-basic cards were already discarded by resolve_stress_card
            if flipped:
                basic_card = flipped[-1]
                if basic_card.card_type == CardType.SPEED:
                    player.cards_played.append(basic_card)

            # Collect IDs of non-SPEED cards that were discarded during flipping
            discarded = [
                c.id for c in flipped if c.card_type != CardType.SPEED
            ]

            state.log_event(
                "stress_resolved",
                player_id=player.player_id,
                data={
                    "value": value,
                    "flipped_count": len(flipped),
                    "discarded": discarded,
                },
            )

    player.speed_from_cards = total_speed

    # Calculate new position
    new_pos, crossed = rules.calculate_move_position(
        player.position, total_speed, state.track, player.lap,
    )
    # Resolve blocking (space full)
    new_pos = rules.resolve_blocked_position(
        new_pos, state.track, state.players, player.player_id,
    )
    player.position = new_pos

    if crossed:
        player.lap += 1
        if rules.check_finished(player, state.track):
            player.finished = True
            # Assign finish order
            finished_count = len([p for p in state.players if p.finished])
            player.finish_order = finished_count

    state.log_event(
        "reveal_and_move",
        player_id=player.player_id,
        data={
            "speed": total_speed,
            "new_position": player.position,
            "lap": player.lap,
            "finished": player.finished,
        },
    )
    events.append(
        GameEvent(
            state.round_num, Phase.REVEAL_AND_MOVE, player.player_id,
            "reveal_and_move",
            {
                "speed": total_speed,
                "new_position": player.position,
                "lap": player.lap,
                "finished": player.finished,
            },
        )
    )

    return events


def step_adrenaline(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Grant adrenaline symbols to trailing player.

    Adrenaline grants symbols usable in React:
    - +1 speed (move 1 extra space AND adds 1 to Speed for corner check)
    - +1 cooldown (can cool 1 additional heat card)
    - The player can use BOTH, EITHER, or NEITHER in React.

    This step sets eligibility flags only. React consumes them.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.ADRENALINE

    eligible = rules.adrenaline_eligible(
        player,
        list(state.active_players) + list(state.finished_players),
        state.starting_player_count,
    )

    if eligible:
        state.log_event(
            "adrenaline_granted",
            player_id=player.player_id,
            data={"eligible": True},
        )
        events.append(
            GameEvent(
                state.round_num, Phase.ADRENALINE, player.player_id,
                "adrenaline_granted",
                {"eligible": True},
            )
        )

    # We store eligibility as an event. The React step will check
    # adrenaline_eligible again (or the caller passes has_adrenaline).
    # The key thing is we do NOT move the player or change state here.

    return events


def step_react(
    state: GameState,
    player: PlayerState,
    decision: ReactDecision,
) -> list[GameEvent]:
    """Activate React symbols: Cooldown, Boost, and Adrenaline bonuses.

    1. COOLDOWN: Move heat cards from hand to heat pool.
    2. BOOST: Pay 1 heat, flip until Basic card found (once per turn).
    3. ADRENALINE SPEED: +1 movement and +1 to corner speed.
    4. ADRENALINE COOLDOWN: +1 to cooldown max (already factored into
       cooldown_count by the caller).

    After React, move the player forward by boost + adrenaline speed.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.REACT

    # --- Adrenaline cooldown (increases max cooldown by 1) ---
    adrenaline_cooldown_bonus = 1 if decision.use_adrenaline_cooldown else 0

    # --- Cooldown ---
    max_cooldown = rules.cooldown_amount(player.gear) + adrenaline_cooldown_bonus
    actual_cooldown = min(decision.cooldown_count, max_cooldown)

    if actual_cooldown > 0:
        cooled = player.cooldown(actual_cooldown)
        state.log_event(
            "cooldown",
            player_id=player.player_id,
            data={"count": len(cooled)},
        )
        events.append(
            GameEvent(
                state.round_num, Phase.REACT, player.player_id,
                "cooldown",
                {"count": len(cooled)},
            )
        )

    # --- Boost ---
    boost_value = 0
    if decision.use_boost:
        if player.boost_used_this_turn:
            raise ValueError(
                f"Player {player.player_id} already used boost this turn"
            )
        if player.heat_available <= 0:
            raise ValueError(
                f"Player {player.player_id} has no heat to pay for boost"
            )

        player.pay_heat(1)
        boost_value, flipped = rules.resolve_boost(player.deck)
        player.boost_used_this_turn = True

        # The basic card found goes to played area (counts for corner speed)
        if flipped:
            basic_card = flipped[-1]
            if basic_card.card_type == CardType.SPEED:
                player.cards_played.append(basic_card)

        state.log_event(
            "boost",
            player_id=player.player_id,
            data={"value": boost_value, "flipped_count": len(flipped)},
        )
        events.append(
            GameEvent(
                state.round_num, Phase.REACT, player.player_id,
                "boost",
                {"value": boost_value, "flipped_count": len(flipped)},
            )
        )

    player.speed_from_boost = boost_value

    # --- Adrenaline speed ---
    adrenaline_speed = 0
    if decision.use_adrenaline_speed:
        adrenaline_speed = 1
        state.log_event(
            "adrenaline_speed",
            player_id=player.player_id,
            data={"speed_bonus": 1},
        )
        events.append(
            GameEvent(
                state.round_num, Phase.REACT, player.player_id,
                "adrenaline_speed",
                {"speed_bonus": 1},
            )
        )

    player.speed_from_adrenaline = adrenaline_speed

    # --- Move forward by boost + adrenaline speed ---
    extra_movement = boost_value + adrenaline_speed
    if extra_movement > 0:
        new_pos, crossed = rules.calculate_move_position(
            player.position, extra_movement, state.track, player.lap,
        )
        new_pos = rules.resolve_blocked_position(
            new_pos, state.track, state.players, player.player_id,
        )
        player.position = new_pos

        if crossed:
            player.lap += 1
            if rules.check_finished(player, state.track):
                player.finished = True
                finished_count = len([p for p in state.players if p.finished])
                player.finish_order = finished_count

    return events


def step_slipstream(
    state: GameState,
    player: PlayerState,
    decision: bool,
) -> list[GameEvent]:
    """Slipstreaming (drafting) for a single player.

    If eligible and decision is True, move +2 spaces.
    Slipstream cannot cross the finish line.
    Slipstream movement does NOT add to corner speed.

    Sets player.slipstream_moved = 2 if taken, 0 otherwise.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.SLIPSTREAM

    player.slipstream_moved = 0

    if not decision:
        return events

    # Verify eligibility
    all_players = list(state.active_players) + list(state.finished_players)
    if not rules.slipstream_eligible(player, all_players, state.track):
        # Not eligible -- silently skip (no error, just don't move)
        return events

    # Cannot cross finish line
    if rules.slipstream_would_cross_finish(player, state.track):
        return events

    # Move +2
    new_pos, crossed = rules.calculate_move_position(
        player.position, 2, state.track, player.lap,
    )
    new_pos = rules.resolve_blocked_position(
        new_pos, state.track, state.players, player.player_id,
    )
    player.position = new_pos
    player.slipstream_moved = 2

    if crossed:
        player.lap += 1
        if rules.check_finished(player, state.track):
            player.finished = True
            finished_count = len([p for p in state.players if p.finished])
            player.finish_order = finished_count

    state.log_event(
        "slipstream",
        player_id=player.player_id,
        data={"moved": 2, "new_position": player.position},
    )
    events.append(
        GameEvent(
            state.round_num, Phase.SLIPSTREAM, player.player_id,
            "slipstream",
            {"moved": 2, "new_position": player.position},
        )
    )

    return events


def step_check_corner(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Check all corners crossed during this player's total movement.

    Uses turn_start_position (set in step_reveal_and_move) and the
    player's current position to find all corners crossed.

    Speed for corner check = corner_speed_for_check (excludes slipstream).

    If the player crossed the finish line, corners are ignored.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.CHECK_CORNER

    # After finish line, ignore corner speed limits
    if player.finished:
        return events

    # Find all corners crossed from turn start to current position
    crossed_corners = rules.corners_crossed(
        player.turn_start_position, player.position, state.track,
    )

    if not crossed_corners:
        return events

    # Calculate speed for corner check (excludes slipstream)
    speed = rules.corner_speed_for_check(player)

    # Calculate total heat cost across all corners
    total_heat_cost = 0
    for corner in crossed_corners:
        cost = rules.corner_heat_cost(speed, corner)
        total_heat_cost += cost

    if total_heat_cost == 0:
        # Under all speed limits, no cost
        state.log_event(
            "corner_check",
            player_id=player.player_id,
            data={
                "corners": len(crossed_corners),
                "speed": speed,
                "heat_cost": 0,
            },
        )
        events.append(
            GameEvent(
                state.round_num, Phase.CHECK_CORNER, player.player_id,
                "corner_check",
                {"corners": len(crossed_corners), "speed": speed, "heat_cost": 0},
            )
        )
        return events

    # Check for spin out
    if rules.check_spin_out(player, total_heat_cost):
        # SPIN OUT
        # Pay all remaining heat
        remaining_heat = player.heat_available
        if remaining_heat > 0:
            player.pay_heat(remaining_heat)

        # Find the first corner that caused the spin out and place
        # the player BEFORE it (corner.start - 1)
        # Use the first corner in the list (earliest encountered)
        spin_corner = crossed_corners[0]
        spin_position = max(0, spin_corner.start - 1)
        player.position = spin_position

        # Add stress cards to hand
        stress_count = rules.spin_out_stress_count(player.gear)
        for _ in range(stress_count):
            stress_id = state.next_stress_id()
            stress_card = Card(
                CardType.STRESS, 0, f"stress_penalty_{stress_id}"
            )
            player.hand.append(stress_card)

        # Set gear to 1
        player.gear = 1
        player.spun_out = True

        state.log_event(
            "spin_out",
            player_id=player.player_id,
            data={
                "corner_start": spin_corner.start,
                "new_position": spin_position,
                "stress_added": stress_count,
                "heat_paid": remaining_heat,
            },
        )
        events.append(
            GameEvent(
                state.round_num, Phase.CHECK_CORNER, player.player_id,
                "spin_out",
                {
                    "corner_start": spin_corner.start,
                    "new_position": spin_position,
                    "stress_added": stress_count,
                    "heat_paid": remaining_heat,
                },
            )
        )
    else:
        # Can pay the heat cost
        player.pay_heat(total_heat_cost)

        state.log_event(
            "corner_check",
            player_id=player.player_id,
            data={
                "corners": len(crossed_corners),
                "speed": speed,
                "heat_cost": total_heat_cost,
            },
        )
        events.append(
            GameEvent(
                state.round_num, Phase.CHECK_CORNER, player.player_id,
                "corner_check",
                {
                    "corners": len(crossed_corners),
                    "speed": speed,
                    "heat_cost": total_heat_cost,
                },
            )
        )

    return events


def step_discard(
    state: GameState,
    player: PlayerState,
    decision: list[Card],
) -> list[GameEvent]:
    """Player may optionally discard Speed/Upgrade cards from hand.

    Only Speed and Upgrade cards can be discarded (not Heat or Stress).
    This is a PLAYER CHOICE about which hand cards to voluntarily discard.

    decision: list of Card objects the player wants to discard from hand.
              Empty list means keep everything.
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.DISCARD

    if not decision:
        return events

    # Validate: only Speed and Upgrade can be discarded
    for card in decision:
        if card.card_type not in (CardType.SPEED, CardType.UPGRADE):
            raise ValueError(
                f"Player {player.player_id} tried to discard a "
                f"{card.card_type.value} card. Only Speed and Upgrade "
                f"cards can be discarded."
            )
        if card not in player.hand:
            raise ValueError(
                f"Player {player.player_id} tried to discard card "
                f"{card.id} which is not in their hand."
            )

    # Remove from hand and add to discard pile
    for card in decision:
        player.hand.remove(card)
    player.deck.discard(decision)

    state.log_event(
        "discard",
        player_id=player.player_id,
        data={"count": len(decision), "cards": [c.id for c in decision]},
    )
    events.append(
        GameEvent(
            state.round_num, Phase.DISCARD, player.player_id,
            "discard",
            {"count": len(decision), "cards": [c.id for c in decision]},
        )
    )

    return events


def step_replenish(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Move played cards to discard and draw back up to hand size.

    1. Move all cards from cards_played to discard pile.
    2. Draw cards until hand size reaches HAND_SIZE (7).
    3. Clear per-turn transient state.

    Cooldown does NOT happen here (it's in React).
    No draw penalty for spin-out (penalty is stress cards).
    """
    events: list[GameEvent] = []
    state.current_phase = Phase.REPLENISH

    # Move played cards to discard pile
    if player.cards_played:
        player.deck.discard(player.cards_played)
        player.cards_played = []

    # Draw to hand size
    cards_needed = rules.HAND_SIZE - len(player.hand)
    drawn: list[Card] = []
    if cards_needed > 0:
        drawn = player.deck.draw(cards_needed)
        player.hand.extend(drawn)

    # Clear transient fields
    player.boost_used_this_turn = False
    player.speed_from_cards = 0
    player.speed_from_boost = 0
    player.speed_from_adrenaline = 0
    player.slipstream_moved = 0
    player.cluttered = False
    player.turn_start_position = 0

    drawn_repr = [repr(c) for c in drawn]
    state.log_event(
        "replenish",
        player_id=player.player_id,
        data={
            "hand_size": len(player.hand),
            "drawn": drawn_repr,
        },
    )
    events.append(
        GameEvent(
            state.round_num, Phase.REPLENISH, player.player_id,
            "replenish",
            {
                "hand_size": len(player.hand),
                "drawn": drawn_repr,
            },
        )
    )

    return events
