"""Pure rule functions for the HEAT board game engine.

Every function is pure: it takes state as input and returns computed results
without mutating anything (except resolve_stress_card/resolve_boost which
draw from the deck). The engine and agents both depend on this module.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from heat.models.cards import Card, CardType, Deck
from heat.models.player_state import PlayerState
from heat.models.track import Corner, Track

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_GEAR: int = 1
MAX_GEAR: int = 4
HAND_SIZE: int = 7
HEAT_POOL_SIZE: int = 6


# ---------------------------------------------------------------------------
# 1. Gear shifting
# ---------------------------------------------------------------------------

def legal_gear_shifts(
    current_gear: int,
    heat_available: int,
) -> list[tuple[int, int]]:
    """Return all (new_gear, heat_cost) pairs a player can shift to.

    Rules:
    - Can shift +1, -1, or stay (0 heat cost).
    - Can shift +2 or -2 by paying 1 heat from the heat pool.
    - Gear must remain in [MIN_GEAR, MAX_GEAR].
    - Returns a list of (new_gear, heat_cost) sorted by new_gear.
    """
    seen: dict[int, int] = {}
    for delta in (-2, -1, 0, 1, 2):
        new_gear = current_gear + delta
        if not (MIN_GEAR <= new_gear <= MAX_GEAR):
            continue
        heat_cost = 1 if abs(delta) == 2 else 0
        if heat_cost > heat_available:
            continue
        # Keep the cheaper option if the same gear is reachable by two deltas
        if new_gear not in seen or heat_cost < seen[new_gear]:
            seen[new_gear] = heat_cost
    return sorted(seen.items(), key=lambda x: x[0])


# ---------------------------------------------------------------------------
# 2. Cards to play
# ---------------------------------------------------------------------------

def cards_to_play_count(gear: int) -> int:
    """Number of cards a player must play for the given gear."""
    return gear


# ---------------------------------------------------------------------------
# 3. Legal card plays
# ---------------------------------------------------------------------------

def legal_card_plays(
    hand: list[Card],
    gear: int,
) -> list[tuple[Card, ...]]:
    """Return all legal combinations of cards to play.

    Rules:
    - Must play exactly ``gear`` cards.
    - Speed cards and Upgrade cards can always be played.
    - Stress cards can be played (resolved at reveal time).
    - Heat cards in hand CANNOT be played voluntarily.
    - If a player does not have enough playable cards, they must fill
      remaining slots with heat cards from their hand.

    Returns a list of tuples, each tuple being a valid card combination.
    """
    count = cards_to_play_count(gear)

    playable = [c for c in hand if c.card_type != CardType.HEAT]
    heat_cards = [c for c in hand if c.card_type == CardType.HEAT]

    if len(playable) >= count:
        # Enough playable cards -- enumerate all combinations
        return list(itertools.combinations(playable, count))

    # Not enough playable cards -- must use all playable + fill with heat
    heat_needed = count - len(playable)
    if heat_needed > len(heat_cards):
        # Not enough cards at all (should not happen in a normal game)
        # Return the best we can do
        return [tuple(playable + heat_cards)]

    heat_combos = list(itertools.combinations(heat_cards, heat_needed))
    return [tuple(playable) + hc for hc in heat_combos]


# ---------------------------------------------------------------------------
# 4. Cluttered hand detection
# ---------------------------------------------------------------------------

def is_cluttered_hand(hand: list[Card], gear: int) -> bool:
    """Return True if the player must use Heat cards to fill their play count.

    Official rule: if you cannot play enough non-Heat cards for your gear,
    your car does not move this turn.
    """
    playable = [c for c in hand if c.card_type != CardType.HEAT]
    return len(playable) < cards_to_play_count(gear)


# ---------------------------------------------------------------------------
# 5. Speed calculation
# ---------------------------------------------------------------------------

def calculate_speed(cards: tuple[Card, ...]) -> int:
    """Sum the values of played cards.

    Stress cards count as 0 here (resolved separately during reveal phase).
    Upgrade cards use their face value.
    """
    return sum(c.value for c in cards if c.card_type != CardType.STRESS)


# ---------------------------------------------------------------------------
# 6. Stress resolution
# ---------------------------------------------------------------------------

def resolve_stress_card(deck: Deck) -> tuple[int, list[Card]]:
    """Flip cards from the deck until a Basic (SPEED type) card is found.

    A "Basic card" is a SPEED-type card. Heat, Stress, and Upgrade cards
    are NOT Basic -- they are discarded and flipping continues.

    Returns:
        (value, flipped_cards) where value is the Basic card's value
        and flipped_cards is the list of ALL cards flipped (including the
        Basic card at the end).

    If the deck is completely empty (draw + discard both empty), returns
    (0, []).
    """
    flipped: list[Card] = []
    while True:
        drawn = deck.draw(1)
        if not drawn:
            return 0, flipped  # Deck exhausted
        card = drawn[0]
        flipped.append(card)
        if card.card_type == CardType.SPEED:
            return card.value, flipped
        # Non-Basic card: discard it and keep flipping
        deck.discard([card])


# ---------------------------------------------------------------------------
# 7. Boost resolution
# ---------------------------------------------------------------------------

def resolve_boost(deck: Deck) -> tuple[int, list[Card]]:
    """Resolve a boost action by flipping cards until a Basic card is found.

    Uses the same mechanic as stress card resolution. The Basic card found
    is added to the Play Area (counts toward corner speed check). Boost is
    limited to ONCE per turn.

    Returns (value, flipped_cards) -- same semantics as resolve_stress_card.
    """
    return resolve_stress_card(deck)


# ---------------------------------------------------------------------------
# 8. Corner heat cost
# ---------------------------------------------------------------------------

def corner_heat_cost(speed: int, corner: Corner) -> int:
    """Heat cost for passing through a corner at the given speed.

    Cost = max(0, speed - speed_limit).

    The 'speed' parameter should INCLUDE card values + stress flip values +
    boost flip value + adrenaline speed, but EXCLUDE slipstream movement.
    """
    return max(0, speed - corner.speed_limit)


# ---------------------------------------------------------------------------
# 9. Corners crossed
# ---------------------------------------------------------------------------

def corners_crossed(
    start_pos: int,
    end_pos: int,
    track: Track,
    spaces_moved: int | None = None,
) -> list[Corner]:
    """Return all corners the player moved through or into.

    A corner applies if the player's path [start_pos+1 .. end_pos]
    intersects with the corner's [start .. end] range.
    The start_pos itself is excluded (the player was already there).

    Handles wrap-around for multi-lap tracks.

    ``spaces_moved`` is the actual number of spaces the player advanced
    this turn (lap-aware). It is required to disambiguate the case where
    ``end_pos == start_pos``: that can mean either "did not move" (0
    spaces) or "moved exactly one or more full laps" (>= track.length
    spaces), which lands back on the same space but DOES cross every
    corner on the lap. When ``spaces_moved`` is None it is inferred from
    the shortest forward path (legacy behaviour, which cannot detect a
    full-lap landing on the start space).
    """
    length = track.length

    if length == 0:
        return []

    # Infer spaces moved from positions if not supplied. This cannot
    # distinguish a full-lap move (end == start) from no movement.
    if spaces_moved is None:
        spaces_moved = (end_pos - start_pos) % length

    if spaces_moved <= 0:
        # No movement
        return []

    # Build the set of positions traversed (excluding start). Walk the
    # path forward space-by-space so a full-lap (or multi-lap) move
    # correctly includes every corner, even when it lands on start_pos.
    if spaces_moved >= length:
        # One or more full laps: every space on the track is traversed.
        traversed_positions = range(length)
    elif end_pos > start_pos:
        # Normal forward movement (no wrap)
        traversed_positions = range(start_pos + 1, end_pos + 1)
    else:
        # Wrap-around: crossed the finish line
        traversed_positions = list(range(start_pos + 1, length)) + list(
            range(0, end_pos + 1)
        )

    traversed = set(traversed_positions)
    result: list[Corner] = []
    for corner in track.corners:
        corner_positions = set(range(corner.start, corner.end + 1))
        if corner_positions.intersection(traversed):
            result.append(corner)
    return result


def distance_to_next_corner(
    track: Track,
    position: int,
) -> tuple[Corner | None, int]:
    """Return the nearest corner ahead and the distance to its start.

    Distance is measured from ``position`` to ``corner.start`` going
    forward, wrapping around the track. A corner whose start coincides
    with ``position`` is treated as a FULL lap away (distance ==
    ``track.length``), since the player is already standing on it; the
    next time they reach that corner is one lap later. This matches the
    convention used in turn-start logging and the standings display.

    Returns ``(corner, distance)``. If the track has no corners, returns
    ``(None, track.length)``.
    """
    best_corner: Corner | None = None
    best_dist: int | None = None
    length = track.length
    for corner in track.corners:
        dist = (corner.start - position) % length
        if dist == 0:
            dist = length  # already at/past this corner this lap
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_corner = corner
    if best_dist is None:
        return None, length
    return best_corner, best_dist


# ---------------------------------------------------------------------------
# 10. Spin-out check
# ---------------------------------------------------------------------------

def check_spin_out(player: PlayerState, heat_cost: int) -> bool:
    """A player spins out if they cannot pay the required heat cost.

    Returns True if heat_cost > player.heat_available.
    """
    return heat_cost > player.heat_available


# ---------------------------------------------------------------------------
# 11. Spin-out stress count
# ---------------------------------------------------------------------------

def spin_out_stress_count(gear: int) -> int:
    """Number of Stress cards added to hand during a spin-out.

    Official rule:
    - Gear 1 or 2: take 1 Stress card
    - Gear 3 or 4: take 2 Stress cards
    """
    if gear <= 2:
        return 1
    return 2


# ---------------------------------------------------------------------------
# 12. Slipstream eligibility
# ---------------------------------------------------------------------------

def slipstream_eligible(
    player: PlayerState,
    all_players: list[PlayerState],
    track: Track,
) -> bool:
    """A player can slipstream if there is another (non-finished) player
    exactly 1 or 2 spaces ahead of them.

    RESTRICTION: Slipstream can never be used to cross the finish line,
    nor after having crossed the finish line.
    """
    if player.finished:
        return False

    if slipstream_would_cross_finish(player, track):
        return False

    # Check for another non-finished car 1 or 2 spaces ahead
    for other in all_players:
        if other.player_id == player.player_id:
            continue
        if other.finished:
            continue
        # Calculate distance ahead (handling wrap-around)
        dist = (other.position - player.position) % track.length
        if dist in (1, 2):
            return True

    return False


# ---------------------------------------------------------------------------
# 13. Slipstream finish check
# ---------------------------------------------------------------------------

def slipstream_would_cross_finish(
    player: PlayerState,
    track: Track,
) -> bool:
    """Return True if slipstreaming 2 spaces would cross the finish line.

    Official rule: 'Slipstreaming can never be used to cross the finish
    line nor after having crossed the finish line.'
    """
    if player.finished:
        return True
    if player.lap >= track.laps:
        # On final lap -- check if +2 would cross
        return (player.position + 2) >= track.length
    return False


# ---------------------------------------------------------------------------
# 14. Cooldown amount
# ---------------------------------------------------------------------------

def cooldown_amount(gear: int) -> int:
    """Cooldown based on current gear.

    Gear 1: cool 3 heat cards (move from hand back to heat pool)
    Gear 2: cool 1 heat card
    Gear 3+: no cooldown
    """
    if gear == 1:
        return 3
    elif gear == 2:
        return 1
    return 0


# ---------------------------------------------------------------------------
# 15. Adrenaline eligibility
# ---------------------------------------------------------------------------

def adrenaline_eligible(
    player: PlayerState,
    all_players: list[PlayerState],
    starting_player_count: int,
) -> bool:
    """A player gets adrenaline if they are trailing.

    Eligibility is based on the number of cars that STARTED the race,
    not the remaining cars in play.

    - 2-4 players started: last place gets adrenaline
    - 5-6 players started: last 2 places get adrenaline

    Only active (non-finished) players can receive adrenaline. Finished
    players are excluded from the position ranking but the threshold is
    still based on starting_player_count.
    """
    if player.finished:
        return False

    active = [p for p in all_players if not p.finished]
    if len(active) <= 1:
        return False

    # Sort active players: worst position first (lowest lap, then lowest
    # position). player_id is a final tiebreak so the recipient is
    # deterministic when trailing players are tied, matching the ordering
    # convention used by compute_turn_order.
    ranked = sorted(active, key=lambda p: (p.lap, p.position, p.player_id))

    # Determine how many trailing players get adrenaline
    if starting_player_count >= 5:
        adrenaline_count = 2
    else:
        adrenaline_count = 1

    trailing_ids = {p.player_id for p in ranked[:adrenaline_count]}
    return player.player_id in trailing_ids


# ---------------------------------------------------------------------------
# 16. Move position calculation
# ---------------------------------------------------------------------------

def calculate_move_position(
    current_pos: int,
    speed: int,
    track: Track,
    current_lap: int,
) -> tuple[int, bool]:
    """Calculate new position and whether a lap was completed.

    Returns (new_position, crossed_finish_line).
    Position wraps modulo track.length for multi-lap tracks.

    Note: ``crossed_finish_line`` is a boolean and does NOT capture how
    many laps a single large move completes. Callers that credit laps
    should use :func:`laps_completed_by_move` to handle a single move
    that spans two or more laps (speed >= 2 * track.length).
    """
    raw_pos = current_pos + speed
    if raw_pos >= track.length:
        # Crossed the finish line
        new_pos = raw_pos % track.length
        return new_pos, True
    return raw_pos, False


def laps_completed_by_move(
    current_pos: int,
    speed: int,
    track: Track,
) -> int:
    """Return how many laps a single move of ``speed`` spaces completes.

    A lap is completed each time the player's path crosses the finish
    line (position wraps past ``track.length``). Equals
    ``(current_pos + speed) // track.length``. A single move of
    ``speed >= 2 * track.length`` therefore credits two or more laps,
    where naive ``+= 1`` logic would credit only one.
    """
    if track.length == 0:
        return 0
    return (current_pos + speed) // track.length


# ---------------------------------------------------------------------------
# 17. Finish check
# ---------------------------------------------------------------------------

def check_finished(player: PlayerState, track: Track) -> bool:
    """Player finishes if lap > track.laps (they have completed all laps)."""
    return player.lap > track.laps


# ---------------------------------------------------------------------------
# 18. Corner speed for check
# ---------------------------------------------------------------------------

def corner_speed_for_check(player: PlayerState) -> int:
    """Calculate the effective speed for corner checking purposes.

    INCLUDES: card values (speed_from_cards) + boost flip value
    (speed_from_boost) + adrenaline +1 speed (speed_from_adrenaline).

    EXCLUDES: slipstream movement (slipstream_moved).
    """
    return (
        player.speed_from_cards
        + player.speed_from_boost
        + player.speed_from_adrenaline
    )


# ---------------------------------------------------------------------------
# 19. Collision / blocking resolution
# ---------------------------------------------------------------------------

def resolve_blocked_position(
    target_pos: int,
    track: Track,
    all_players: list[PlayerState],
    moving_player_id: int,
) -> int:
    """Resolve blocking when a space is full.

    Official rule: 'if you would end your move in a Space where there are
    cars on all Spots, then you are blocked and must put your car in the
    first Space with a free Spot behind the cars that blocked you.'

    Returns the actual position the player should land on.
    """
    pos = target_pos % track.length

    # Count cars at target position (excluding the moving player)
    def cars_at(position: int) -> int:
        return sum(
            1 for p in all_players
            if p.player_id != moving_player_id
            and not p.finished
            and p.position == position
        )

    # Check if target space has room
    lanes = track.spaces[pos].lanes if pos < len(track.spaces) else 1
    if cars_at(pos) < lanes:
        return pos

    # Space is full — find first available space behind
    for offset in range(1, track.length):
        check_pos = (pos - offset) % track.length
        check_lanes = track.spaces[check_pos].lanes if check_pos < len(track.spaces) else 1
        if cars_at(check_pos) < check_lanes:
            return check_pos

    # Shouldn't happen, but fallback to target
    return pos


# ---------------------------------------------------------------------------
# 20. Unified legal-action enumeration (decision-point API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactOptions:
    """Legal React action envelope for one player at the React decision.

    Mirrors exactly what ``Game._collect_react_decision`` computes inline so
    a Gym env and the agent protocol agree on the React action space.

    Attributes:
        max_cooldown: Gear-based cooldown amount. The +1 adrenaline cooldown
            is NOT included here; it is applied later inside ``step_react``.
        can_boost: True if the player has heat to pay and hasn't boosted.
        has_adrenaline: True if the player is eligible for adrenaline.
    """

    max_cooldown: int
    can_boost: bool
    has_adrenaline: bool


def legal_react_options(
    player: PlayerState,
    active_players: list[PlayerState],
    starting_player_count: int,
) -> ReactOptions:
    """Return the legal React options for ``player`` at the React decision.

    Pure extraction of the inline logic in ``Game._collect_react_decision``;
    no behavior change. ``max_cooldown`` is gear-only (the adrenaline +1 is
    applied later in ``step_react``).
    """
    return ReactOptions(
        max_cooldown=cooldown_amount(player.gear),
        can_boost=(
            player.heat_available > 0 and not player.boost_used_this_turn
        ),
        has_adrenaline=adrenaline_eligible(
            player, active_players, starting_player_count
        ),
    )


def legal_slipstream(
    player: PlayerState,
    active_players: list[PlayerState],
    track: Track,
) -> bool:
    """Return whether ``player`` may take slipstream at this decision point.

    Thin pass-through to :func:`slipstream_eligible`, provided as the single
    named decision-point entry the env queries.
    """
    return slipstream_eligible(player, active_players, track)


def legal_discards(player: PlayerState) -> list[Card]:
    """Return the Speed/Upgrade cards eligible for voluntary discard.

    Pure extraction of the inline filter in ``Game.run_round`` (and the rule
    ``step_discard`` re-validates the same set). Heat and Stress cards are
    never voluntarily discardable.
    """
    return [
        c
        for c in player.hand
        if c.card_type in (CardType.SPEED, CardType.UPGRADE)
    ]
