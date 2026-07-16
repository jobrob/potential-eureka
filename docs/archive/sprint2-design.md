# Sprint 2 Design: Game Engine

## Summary

Sprint 2 transforms the HEAT data models from Sprint 1 into a playable game engine. It delivers four modules -- `engine/rules.py`, `engine/events.py`, `engine/phases.py`, and `engine/game.py` -- that together implement the full round loop as defined by the official HEAT: Pedal to the Metal rulebook. The engine enforces all rules, enumerates legal moves for agents, and produces a stream of `GameEvent` records for replay and ML training. After Sprint 2 is complete, a caller can instantiate a `Game`, supply agent callbacks, and run a complete race to termination.

**Key architectural principle**: Steps 1-2 (Gear Shift + Card Play) are resolved simultaneously for all players. Steps 3-9 (Reveal & Move through Replenish) are resolved sequentially per player in front-to-back turn order -- each player completes ALL of steps 3-9 before the next player begins step 3. This matches the official HEAT rulebook.

## Motivation

- Sprint 1 delivered static data containers. Without an engine, nothing moves.
- Sprint 3 (Agents) depends on the engine providing `legal_*()` functions and a phase-driven game loop.
- Sprint 5 (ML) depends on deterministic, event-sourced game traces that only the engine can produce.
- The performance target of <10ms per game demands that the engine be lean -- pure functions, minimal allocation, no unnecessary copying of state.

---

## Existing Foundation (Sprint 1 Recap)

Before detailing the new modules, here is a precise summary of what already exists and what the engine will depend on.

### `models/cards.py`
- `CardType` enum: `SPEED`, `HEAT`, `STRESS`
- `Card` frozen dataclass: `card_type`, `value`, `id`
- `Deck` class with `draw(count)`, `discard(cards)`, `add_to_draw_pile(cards)`, auto-reshuffle
- `create_starting_deck(player_id)` -- 12 speed cards: two each of values 1-6
- `create_heat_cards(player_id, count=6)` -- heat cards with value 0

### `models/track.py`
- `Space(index, lanes=1)` -- frozen
- `Corner(start, end, speed_limit)` -- frozen
- `Track(name, spaces, corners, start_positions, laps=1)` with `length`, `get_corner_at(position)`, `spaces_in_corner(corner)`

### `models/player_state.py`
- `PlayerState` mutable dataclass: `player_id`, `name`, `deck`, `hand`, `gear` (1-4), `position`, `lap`, `heat_pool`, `cooldown_pool`, `spun_out`, `finished`, `finish_order`
- `PlayerState.create(player_id, name)` -- builds deck, draws 7-card hand, initializes 6 heat cards
- Properties: `speed_cards_in_hand`, `heat_in_hand`, `stress_in_hand`, `heat_available`
- Methods: `pay_heat(amount)` -- moves heat from pool to deck discard; `cooldown(amount)` -- moves heat from hand back to pool

### `models/game_state.py`
- `Phase` enum: 9 phases (SHIFT_GEARS through REPLENISH)
- `GameEvent(round_num, phase, player_id, event_type, data)` -- already defined here
- `GameState(track, players, round_num, current_phase, turn_order, event_log, logging_enabled)` with `active_players`, `finished_players`, `is_game_over`, `get_player(id)`, `log_event(...)`, `compute_turn_order()`

### Track data (`tracks/usa.json`)
- 30 spaces, 3 corners (speed limits 4, 3, 5), 6 start positions, 1 lap
- Lanes: 2 on straights, 1 in corners

---

## Design Decision: Where Does `GameEvent` Live?

**Decision: Keep `GameEvent` in `models/game_state.py`. Do not create `engine/events.py`.**

Rationale:
1. `GameEvent` is already defined in `models/game_state.py` and exported from `models/__init__.py`. Moving it would break the existing public API and tests.
2. `GameEvent` is a pure data container -- it belongs in the models layer, not the engine layer.
3. The PLAN.md item "engine/events.py -- GameEvent dataclass" was written before Sprint 1 placed `GameEvent` in models. The intent is already satisfied.
4. If engine-specific event utilities are needed later (e.g., event filtering, replay serialization), those can go in `engine/events.py` at that time, importing `GameEvent` from models.

The `engine/events.py` file should still be created as an empty module with a docstring noting this decision, so the package structure matches the plan and future code has a natural home.

```python
# engine/events.py
"""Game event utilities.

GameEvent is defined in models.game_state and re-exported from models.
This module is reserved for event filtering, replay serialization,
and other engine-level event processing added in later sprints.
"""

from heat.models.game_state import GameEvent

__all__ = ["GameEvent"]
```

---

## Required Model Changes

Sprint 2 requires several additions to the existing models.

### `models/cards.py` -- New CardType and Starting Upgrade Cards

Add a new `CardType` member and a factory function for Starting Upgrade cards:

```python
class CardType(Enum):
    SPEED = "speed"
    HEAT = "heat"
    STRESS = "stress"
    UPGRADE = "upgrade"  # NEW: Starting Upgrade cards


def create_starting_upgrade_cards(player_id: int) -> dict[str, list[Card]]:
    """Create the 3 Starting Upgrade cards per the official rules.

    Returns a dict with two keys:
    - "deck": cards shuffled into the player's draw deck [the extra Heat card]
    - "hand_eligible": cards that are part of the regular card pool
      [the 0-value and 5-value speed-type upgrades]

    The official HEAT rules give each player 3 Starting Upgrade cards:
    1. A 0-value speed card (UPGRADE type, value 0) -- shuffled into deck
    2. A 5-value speed card (UPGRADE type, value 5) -- shuffled into deck
    3. An extra Heat card (HEAT type, value 0) -- shuffled into deck

    IMPORTANT: For stress/boost resolution, Upgrade cards are NOT Basic cards.
    When flipped during stress or boost resolution, they are discarded and
    flipping continues until a Basic (SPEED type) card is found.
    """
    prefix = f"p{player_id}"
    upgrade_0 = Card(CardType.UPGRADE, 0, f"{prefix}_upg_0")
    upgrade_5 = Card(CardType.UPGRADE, 5, f"{prefix}_upg_5")
    extra_heat = Card(CardType.HEAT, 0, f"{prefix}_upg_heat")
    return {
        "deck": [upgrade_0, upgrade_5, extra_heat],
    }
```

**Key rule**: During stress resolution and boost resolution, UPGRADE cards are NOT Basic cards. When flipped, they are immediately discarded and flipping continues. Only SPEED-type cards are "Basic" and stop the flipping loop.

### `models/player_state.py` -- New Fields

```python
# Add to PlayerState fields:
cards_played: list[Card] = field(default_factory=list)
boost_used_this_turn: bool = False        # Boost is limited to once per turn
speed_from_cards: int = 0                 # Card speed (for corner check tracking)
speed_from_boost: int = 0                 # Boost speed (counts for corners)
speed_from_adrenaline: int = 0            # Adrenaline +1 speed (counts for corners)
slipstream_moved: int = 0                 # Spaces moved by slipstream (does NOT count for corners)
```

Update `PlayerState.create()` to include Starting Upgrade cards in the deck:

```python
@classmethod
def create(cls, player_id: int, name: str | None = None) -> PlayerState:
    if name is None:
        name = f"Player {player_id}"

    speed_cards = create_starting_deck(player_id)
    upgrade_cards = create_starting_upgrade_cards(player_id)
    all_deck_cards = speed_cards + upgrade_cards["deck"]
    deck = Deck(all_deck_cards)
    heat_pool = create_heat_cards(player_id)

    player = cls(
        player_id=player_id,
        name=name,
        deck=deck,
        heat_pool=heat_pool,
    )
    player.hand = player.deck.draw(7)
    return player
```

### `models/game_state.py` -- Track Starting Player Count

```python
# Add to GameState fields:
starting_player_count: int = 0  # Number of players that started the race (for adrenaline)
```

Set this during `GameState.create()`:

```python
@classmethod
def create(cls, track, num_players, ...) -> GameState:
    ...
    state = cls(track=track, players=players, logging_enabled=logging_enabled)
    state.starting_player_count = num_players
    return state
```

---

## Module 1: `engine/rules.py` -- Pure Rule Functions

This is the foundation module. Every function is pure: it takes state as input and returns computed results without mutating anything. The engine and agents both depend on it.

### Constants

```python
MIN_GEAR: int = 1
MAX_GEAR: int = 4
HAND_SIZE: int = 7
HEAT_POOL_SIZE: int = 6
```

### Function Catalog

#### 1. `legal_gear_shifts(current_gear: int, heat_available: int) -> list[tuple[int, int]]`

Returns the list of (new_gear, heat_cost) tuples a player can shift to. In HEAT, a player can shift +/-1 for free, or +/-2 by paying 1 Heat.

```python
def legal_gear_shifts(
    current_gear: int,
    heat_available: int,
) -> list[tuple[int, int]]:
    """Return all (new_gear, heat_cost) pairs a player can shift to.

    Rules:
    - Can shift +1, -1, or stay (0 heat cost).
    - Can shift +2 or -2 by paying 1 heat from the heat pool.
    - Gear must remain in [1, 4].
    - Returns a list of (new_gear, heat_cost) sorted by new_gear.

    Examples:
        legal_gear_shifts(2, 3) -> [(1, 1), (1, 0), (2, 0), (3, 0), (4, 1)]
        -- wait, -2 from 2 would be 0 which is below MIN_GEAR, so:
        legal_gear_shifts(2, 3) -> [(1, 0), (2, 0), (3, 0), (4, 1)]
        legal_gear_shifts(2, 0) -> [(1, 0), (2, 0), (3, 0)]
          (no +/-2 options because no heat to pay)
        legal_gear_shifts(1, 0) -> [(1, 0), (2, 0)]
        legal_gear_shifts(3, 2) -> [(1, 1), (2, 0), (3, 0), (4, 0)]
    """
    shifts = []
    for delta in [-2, -1, 0, 1, 2]:
        new_gear = current_gear + delta
        if not (MIN_GEAR <= new_gear <= MAX_GEAR):
            continue
        heat_cost = 1 if abs(delta) == 2 else 0
        if heat_cost > heat_available:
            continue
        shifts.append((new_gear, heat_cost))

    # Deduplicate: if both delta=-1 and delta=+1 reach the same gear
    # (impossible given the math), keep the cheaper one.
    # In practice, no dedup needed, but sort by gear for determinism.
    seen = {}
    for gear, cost in shifts:
        if gear not in seen or cost < seen[gear]:
            seen[gear] = cost
    return sorted(seen.items(), key=lambda x: x[0])
```

**Edge cases**:
- Gear 1: cannot shift to 0 or -1. Free shifts: [1, 2]. With heat: also [3] (shift +2, cost 1).
- Gear 4: cannot shift to 5 or 6. Free shifts: [3, 4]. With heat: also [2] (shift -2, cost 1).
- Gear 3 with heat: [1 (cost 1), 2, 3, 4]. Shift -2 to gear 1 costs 1 heat.
- No heat available: only +/-1 and stay.

#### 2. `cards_to_play_count(gear: int) -> int`

Returns the number of cards a player must play in the given gear. In HEAT, the gear number equals the number of cards played.

```python
def cards_to_play_count(gear: int) -> int:
    """Number of cards a player must play for the given gear."""
    return gear
```

#### 3. `legal_card_plays(hand: list[Card], gear: int) -> list[tuple[Card, ...]]`

The most complex rule function. Returns all legal combinations of cards a player can play.

```python
def legal_card_plays(
    hand: list[Card],
    gear: int,
) -> list[tuple[Card, ...]]:
    """Return all legal combinations of cards to play.

    Rules:
    - Must play exactly `gear` cards.
    - Speed cards and Upgrade cards (value > 0) can always be played.
    - Heat cards in hand CANNOT be played voluntarily (they are dead weight).
    - Stress cards CAN be played -- their value is resolved at reveal time
      (flip from deck until a Basic card is found, use that value).
    - Upgrade cards with value 0 CAN be played (they contribute 0 speed).
    - If a player does not have enough playable cards (speed + stress + upgrade),
      they must fill remaining slots with heat cards from their hand.
    - CLUTTERED HAND: If the player has so many Heat cards they cannot play
      enough non-Heat cards for their gear, this function still returns valid
      combos (using heat cards to fill). The Game orchestrator handles the
      "car does not move" consequence separately.

    Returns a list of tuples, each tuple being a valid card combination.
    Uses itertools.combinations for enumeration.
    """
```

**Implementation approach**:
1. Separate hand into playable cards (SPEED + STRESS + UPGRADE) and heat cards in hand.
2. `count = cards_to_play_count(gear)`
3. If the player has enough playable cards (>= count), enumerate all `combinations(playable_cards, count)`.
4. If not enough playable cards, the player must play all playable cards plus enough heat cards from hand to fill. This produces fewer (possibly one) valid combination.

**Key subtlety**: Heat cards in the _hand_ are different from heat cards in the _heat pool_. Hand heat cards are drawn-into-hand clogging cards. They cannot be voluntarily played but must be used if the player has too few other cards.

**Performance note**: With a 7-card hand and gear 4, worst case is C(7,4) = 35 combinations. This is trivially fast.

#### 4. `is_cluttered_hand(hand: list[Card], gear: int) -> bool`

Detects whether the player has too many Heat cards to play enough non-Heat cards for their gear.

```python
def is_cluttered_hand(hand: list[Card], gear: int) -> bool:
    """Return True if the player must use Heat cards to fill their play count.

    Official rule: if you cannot play enough non-Heat cards for your gear,
    your car does not move this turn. You use as many playable cards as
    possible, cover the difference with Heat cards, set gear to 1, place
    all those cards in your discard pile, and skip straight to Replenish.

    This function only detects the condition. The Game orchestrator applies
    the consequence.
    """
    playable = [c for c in hand if c.card_type != CardType.HEAT]
    return len(playable) < cards_to_play_count(gear)
```

#### 5. `calculate_speed(cards: tuple[Card, ...]) -> int`

Sums the speed values of the played cards. Stress cards contribute 0 at this point (their real value is resolved during reveal). Upgrade cards contribute their face value.

```python
def calculate_speed(cards: tuple[Card, ...]) -> int:
    """Sum the values of played cards. Stress cards count as 0 here
    (resolved separately during reveal phase). Upgrade cards use face value."""
    return sum(c.value for c in cards if c.card_type != CardType.STRESS)
```

#### 6. `resolve_stress_card(deck: Deck) -> tuple[int, list[Card]]`

When a stress card is revealed, keep flipping cards from the draw pile until a Basic (SPEED type) card is found. All non-Basic cards flipped along the way are discarded.

```python
def resolve_stress_card(deck: Deck) -> tuple[int, list[Card]]:
    """Flip cards from the deck until a Basic (SPEED type) card is found.

    Official rule: "flipping the top card of your draw deck: If it is a
    Basic card, add it to your Play Area. Otherwise, immediately put it
    in the discard pile and keep flipping until you find a Basic card."

    A "Basic card" is a SPEED-type card. Heat, Stress, and Upgrade cards
    are NOT Basic -- they are discarded and flipping continues.

    Returns:
        (value, flipped_cards) where value is the Basic card's value
        and flipped_cards is the list of ALL cards flipped (including the
        Basic card at the end). The caller is responsible for placing the
        Basic card in the play area and the non-Basic cards in the discard.

    If the deck is completely empty (draw + discard both empty), returns
    (0, []) -- the stress card contributes 0 speed.
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
```

**Key difference from old design**: The old design drew exactly 1 card. The official rule requires looping until a SPEED card is found, discarding all Heat/Stress/Upgrade cards encountered along the way.

#### 7. `resolve_boost(deck: Deck) -> tuple[int, list[Card]]`

Boost uses the same mechanic as stress resolution: flip until a Basic card.

```python
def resolve_boost(deck: Deck) -> tuple[int, list[Card]]:
    """Resolve a boost action by flipping cards until a Basic card is found.

    Official rule: "Boosting gives you a [random flip] symbol" which works
    identically to stress card resolution -- flip until a Basic (SPEED)
    card is found, discard non-Basic cards along the way.

    The Basic card found is added to the Play Area (counts toward corner
    speed check). Boost is limited to ONCE per turn.

    Returns (value, flipped_cards) -- same semantics as resolve_stress_card.
    """
    return resolve_stress_card(deck)  # Same mechanic
```

#### 8. `corner_heat_cost(speed: int, corner: Corner) -> int`

Calculate how much heat a player must pay for exceeding a corner's speed limit.

```python
def corner_heat_cost(speed: int, corner: Corner) -> int:
    """Heat cost for passing through a corner at the given speed.

    Cost = max(0, speed - speed_limit).

    IMPORTANT: The 'speed' parameter must be calculated correctly:
    - INCLUDES: card values + stress flip values + boost flip value +
      adrenaline +1 speed (if speed bonus was used)
    - EXCLUDES: slipstream movement (slipstream does NOT increase speed
      for corner check purposes, even if it moves you through a corner)
    """
    return max(0, speed - corner.speed_limit)
```

#### 9. `corners_crossed(start_pos: int, end_pos: int, track: Track) -> list[Corner]`

Determine which corners a player passed through (or stopped in) during their move.

```python
def corners_crossed(
    start_pos: int,
    end_pos: int,
    track: Track,
) -> list[Corner]:
    """Return all corners the player moved through or into.

    A corner applies if the player's path [start_pos+1 .. end_pos]
    intersects with the corner's [start .. end] range.
    The start_pos itself is excluded (the player was already there).

    Handles wrap-around for multi-lap tracks.
    """
```

**Wrap-around logic for multi-lap tracks**: If `end_pos < start_pos` (player crossed the finish line), the path wraps: check `[start_pos+1 .. track.length-1]` and `[0 .. end_pos]`.

#### 10. `check_spin_out(player: PlayerState, heat_cost: int) -> bool`

Determine whether a player spins out due to insufficient heat.

```python
def check_spin_out(player: PlayerState, heat_cost: int) -> bool:
    """A player spins out if they cannot pay the required heat cost.

    Returns True if heat_cost > player.heat_available.
    """
    return heat_cost > player.heat_available
```

#### 11. `spin_out_effects(gear: int) -> int`

Calculate the number of Stress cards a player receives during a spin-out, based on their current gear.

```python
def spin_out_stress_count(gear: int) -> int:
    """Number of Stress cards added to hand during a spin-out.

    Official rule:
    - Gear 1 or 2: take 1 Stress card
    - Gear 3 or 4: take 2 Stress cards

    These Stress cards are added directly to the player's hand.
    """
    if gear <= 2:
        return 1
    return 2
```

**Spin-out full effects** (applied in `phase_check_corner`):
1. Pay all remaining heat: `player.pay_heat(player.heat_available)`
2. Move back to the first available space BEFORE the Corner Line (the corner's `start - 1`, or the last available space before the corner start if that space is occupied)
3. Take Stress cards into hand: 1 if gear 1-2, 2 if gear 3-4
4. Set gear to 1
5. Set `spun_out = True`

**CRITICAL difference from old design**: The old design placed the player at `corner.end` (the end of the corner). The official rule places them BEFORE the corner line (`corner.start - 1`). The old design also mentioned "draw 1 fewer card" which does not exist in the official rules. The penalty is Stress cards added to hand.

#### 12. `slipstream_eligible(player: PlayerState, all_players: list[PlayerState], track: Track) -> bool`

Check if a player can slipstream (draft behind another car).

```python
def slipstream_eligible(
    player: PlayerState,
    all_players: list[PlayerState],
    track: Track,
) -> bool:
    """A player can slipstream if there is another (non-finished) player
    exactly 1 or 2 spaces ahead of them.

    RESTRICTION: Slipstream can never be used to cross the finish line,
    nor after having crossed the finish line. If the player is on their
    final lap and slipstreaming would move them past the finish, they
    are NOT eligible.
    """
```

#### 13. `slipstream_would_cross_finish(player: PlayerState, track: Track) -> bool`

Check whether slipstreaming would cross the finish line.

```python
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
```

#### 14. `cooldown_amount(gear: int) -> int`

How many heat cards a player can move from hand back to their heat pool. This is used during the React step.

```python
def cooldown_amount(gear: int) -> int:
    """Cooldown based on current gear.

    Gear 1: cool 3 heat cards (move from hand back to heat pool)
    Gear 2: cool 1 heat card
    Gear 3+: no cooldown

    IMPORTANT: Cooldown happens during React (step 5), NOT during Replenish.
    The Cooldown symbol appears on gear positions 1 and 2. The player
    activates it during the React step along with Boost and Adrenaline.
    """
    if gear == 1:
        return 3
    elif gear == 2:
        return 1
    return 0
```

#### 15. `adrenaline_eligible(player: PlayerState, all_players: list[PlayerState], starting_player_count: int) -> bool`

Adrenaline is available to trailing players, based on the number of players that **started** the race.

```python
def adrenaline_eligible(
    player: PlayerState,
    all_players: list[PlayerState],
    starting_player_count: int,
) -> bool:
    """A player gets adrenaline if they are trailing.

    Official rule: Adrenaline eligibility is based on the number of cars
    that STARTED the race, not the remaining cars in play.

    - 2 players started: last place gets adrenaline
    - 3-4 players started: last place gets adrenaline
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

    # Sort active players: worst position first
    ranked = sorted(active, key=lambda p: (p.lap, p.position))

    # Determine how many trailing players get adrenaline
    if starting_player_count >= 5:
        adrenaline_count = 2
    else:
        adrenaline_count = 1

    trailing_ids = {p.player_id for p in ranked[:adrenaline_count]}
    return player.player_id in trailing_ids
```

#### 16. `calculate_move_position(current_pos: int, speed: int, track: Track, current_lap: int) -> tuple[int, bool]`

Calculate the new position after moving, including lap completion.

```python
def calculate_move_position(
    current_pos: int,
    speed: int,
    track: Track,
    current_lap: int,
) -> tuple[int, bool]:
    """Calculate new position and whether a lap was completed.

    Returns (new_position, crossed_finish_line).
    Position wraps modulo track.length for multi-lap.
    If the player completes their final lap, they finish.
    """
```

#### 17. `check_finished(player: PlayerState, track: Track) -> bool`

Determine whether a player has crossed the finish line on their final lap.

```python
def check_finished(player: PlayerState, track: Track) -> bool:
    """Player finishes if lap > track.laps (they have completed all laps)."""
    return player.lap > track.laps
```

#### 18. `corner_speed_for_check(player: PlayerState) -> int`

Calculate the speed value used for the corner speed check.

```python
def corner_speed_for_check(player: PlayerState) -> int:
    """Calculate the effective speed for corner checking purposes.

    Official rules:
    - INCLUDES: card values (speed_from_cards) + boost flip value
      (speed_from_boost) + adrenaline +1 speed (speed_from_adrenaline)
    - EXCLUDES: slipstream movement (slipstream_moved)

    'Boost symbols always increase your Speed value for the purpose of
    the Check Corner step.'

    'Slipstreaming does NOT increase your Speed value for the purpose
    of the Check Corner step.'

    Even if slipstream moves a player through a corner, the speed for
    that corner check is the card speed (not 2). The player still pays
    heat based on their actual speed.
    """
    return (
        player.speed_from_cards
        + player.speed_from_boost
        + player.speed_from_adrenaline
    )
```

---

## Module 2: `engine/events.py` -- Event Re-export

As discussed above, this module re-exports `GameEvent` from models and serves as a future home for event utilities.

```python
"""Game event utilities.

GameEvent is defined in models.game_state and re-exported from models.
This module is reserved for event filtering, replay serialization,
and other engine-level event processing added in later sprints.
"""

from heat.models.game_state import GameEvent

__all__ = ["GameEvent"]
```

---

## Module 3: `engine/phases.py` -- Per-Player Turn Processing

### Architectural Change: Per-Player Sequential Processing

The official HEAT rules define the round structure as:

> "All players complete steps 1 and 2 at the same time. Then, starting with the frontmost car, complete steps 3 through 9."

This means:
- **Steps 1-2** (Gear Shift + Play Cards): All players choose simultaneously, then all choices are revealed/applied.
- **Steps 3-9** (Reveal through Replenish): Each player completes ALL of steps 3-9 before the next player begins. Processing order is front-to-back (leader first).

This is a fundamental departure from the "one phase for all players, then next phase for all players" approach. Instead, the engine runs a per-player loop for steps 3-9.

### Simultaneous Phase Functions

These process all players at once, called once per round.

#### Phase 1: `phase_shift_gears`

```python
def phase_shift_gears(
    state: GameState,
    decisions: dict[int, tuple[int, int]],
) -> list[GameEvent]:
    """All players simultaneously choose a new gear.

    decisions: {player_id: (new_gear, heat_cost)}

    The heat_cost is 0 for +/-1 shifts and 1 for +/-2 shifts.
    Validates each choice against legal_gear_shifts(). Raises ValueError
    for illegal shifts. Updates player.gear and pays heat if needed.
    Skips finished players and spun-out players (who are forced to gear 1).
    """
```

**Simultaneous**: All decisions are collected before any are applied. The orchestrator gathers decisions from all agents first, then calls this phase once.

**Spin-out override**: A player who spun out in the previous round has their gear forced to 1 and cannot choose.

**+/-2 gear shift heat cost**: When a player shifts +/-2, 1 heat is paid via `player.pay_heat(1)` during this phase.

#### Phase 2: `phase_play_cards`

```python
def phase_play_cards(
    state: GameState,
    decisions: dict[int, tuple[Card, ...]],
) -> list[GameEvent]:
    """All players simultaneously choose which cards to play.

    decisions: {player_id: tuple_of_cards_to_play}

    Validates against legal_card_plays(). Removes played cards from hand.
    Stores the played cards on player.cards_played for later phases.

    CLUTTERED HAND CHECK: If is_cluttered_hand() returns True for a player,
    this function still stores their cards, but marks them. The Game
    orchestrator will detect this and skip steps 3-8 for that player,
    moving them straight to Replenish with no movement and gear set to 1.
    """
```

**Storage for reveal**: The played cards are stored on `player.cards_played`.

### Per-Player Step Functions

These are called once per player, in turn order, as part of the per-player loop in steps 3-9. Each function operates on a single player.

#### Step 3: `step_reveal_and_move`

```python
def step_reveal_and_move(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Reveal played cards and move the player.

    1. Reveal cards_played.
    2. Resolve any Stress cards (flip until Basic card found for each).
    3. Calculate total speed = sum of card values + stress resolution values.
    4. Store speed_from_cards on player (for corner check).
    5. Calculate new position (handle lap wrap).
    6. Update position and lap.
    7. Check if player crossed finish line.

    No agent decisions -- deterministic given the played cards.
    """
```

**Stress resolution**: For each stress card in `cards_played`, call `resolve_stress_card(player.deck)`. This loops through the deck until a SPEED card is found, discarding non-SPEED cards. The SPEED card's value is added to the total speed, and the SPEED card itself goes to the play area (discard with other played cards later).

#### Step 4: `step_adrenaline`

```python
def step_adrenaline(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Grant adrenaline symbols to trailing player.

    Official rule: 'Adrenaline gives that player extra symbols they may
    use in the next step (React). They may move 1 extra Space while adding
    1 to their Speed value and/or gain 1 extra Cooldown.'

    Adrenaline grants SYMBOLS usable in React:
    - +1 speed (move 1 extra space AND adds 1 to Speed for corner check)
    - +1 cooldown (can cool 1 additional heat card)
    - The player can use BOTH, EITHER, or NEITHER

    This step does NOT move the player directly. It sets flags/values
    that the React step will consume.

    No decision here -- eligibility is automatic. The choice of how to
    use the symbols happens in React.
    """
```

**Key difference from old design**: Adrenaline does NOT directly move the player. It grants symbols (+1 speed, +1 cooldown) that are consumed during the React step. The player chooses how to use them in React.

#### Step 5: `step_react`

```python
def step_react(
    state: GameState,
    player: PlayerState,
    decision: ReactDecision,
) -> list[GameEvent]:
    """Activate React symbols: Cooldown, Boost, and Adrenaline bonuses.

    Official rule: 'activate symbols you have access to in ANY order:
    symbols from your current gear (Boost and/or Cooldown) and Adrenaline
    if last.'

    React encompasses THREE things:

    1. COOLDOWN (from gear symbol):
       - Gear 1: cool 3 heat cards (move from hand to heat pool)
       - Gear 2: cool 1 heat card
       - Gear 3-4: no cooldown
       The player MAY choose to cool fewer than the maximum.

    2. BOOST (from gear symbol, costs 1 heat, once per turn):
       - Pay 1 heat to flip cards from deck until a Basic card is found.
       - The Basic card's value is added to speed (and counts for corner
         check).
       - Limited to ONCE per turn.
       - The player may choose NOT to boost.

    3. ADRENALINE BONUSES (if eligible):
       - +1 speed: move 1 extra space, adds 1 to Speed value for corners
       - +1 cooldown: cool 1 additional heat card from hand
       - Can use both, either, or neither.

    decisions structure:
        ReactDecision(
            cooldown_count: int,     # 0 to cooldown_amount(gear) + adrenaline_cooldown
            use_boost: bool,         # whether to pay 1 heat for boost
            use_adrenaline_speed: bool,  # +1 speed from adrenaline
            use_adrenaline_cooldown: bool,  # +1 cooldown from adrenaline
        )

    After React, move the player forward by boost_value + adrenaline_speed.
    Update speed_from_boost and speed_from_adrenaline on the player for
    corner check purposes.
    """
```

**CRITICAL difference from old design**: The old design had React as "pay N heat for +N speed" with no limit, and Cooldown was in Replenish. In the official rules:
- Cooldown is part of React, not Replenish.
- Boost is part of React, limited to once per turn, and uses the flip-until-Basic mechanic.
- Adrenaline symbols are consumed in React.
- React is a single combined step where all three types of symbols are activated.

#### Step 6: `step_slipstream`

```python
def step_slipstream(
    state: GameState,
    player: PlayerState,
    decision: bool,
) -> list[GameEvent]:
    """Slipstreaming (drafting) for a single player.

    If eligible (another car 1-2 spaces ahead), gain +2 spaces.

    RESTRICTIONS:
    - Slipstream can NEVER be used to cross the finish line.
    - Slipstream can NEVER be used after having crossed the finish line.
    - Slipstream movement does NOT add to Speed for corner check purposes.

    Sets player.slipstream_moved = 2 if slipstream was taken (for tracking).
    The corner check will use corner_speed_for_check() which excludes
    slipstream from the speed value.

    decision: True to take slipstream, False to decline.
    """
```

**Key difference from old design**:
1. Slipstream cannot cross the finish line (restriction was missing).
2. Slipstream speed does NOT count for corner checks. The old design incorrectly said "slipstream corners use speed = 2". The official rule says slipstream does NOT increase Speed for corner check at all. The player's corner speed is their card speed + boost + adrenaline speed, regardless of whether slipstream moved them through a corner.

**Processing order**: In the per-player sequential model, each player's slipstream is processed during their own steps 3-9 turn. However, the official rules process slipstream back-to-front. Since steps 3-9 are processed front-to-back, slipstream within each player's turn naturally handles this: by the time a trailing player gets to their slipstream step, the leader has already finished their full turn (including slipstream). The orchestrator processes players front-to-back, so the trailing player sees updated positions.

**NOTE**: The official rules specifically say slipstream runs back-to-front within the turn order. Since steps 3-9 run front-to-back (leader first), we have a conflict. The resolution: each player resolves their own slipstream during their step 6. Since leaders go first and trailing players go last, a trailing player's slipstream eligibility check sees the leader's final position (post-slipstream). This effectively achieves the same result as processing slipstream back-to-front in a separate pass, because a trailing player's slipstream cannot affect a leader's slipstream (leaders are ahead, not behind).

#### Step 7: `step_check_corner`

```python
def step_check_corner(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Check all corners crossed during this player's total movement.

    For this player:
    1. Determine all corners crossed from their start-of-turn position
       to their current position (after all movement including slipstream).
    2. For each corner, calculate heat cost using corner_speed_for_check():
       speed = card values + boost value + adrenaline speed.
       EXCLUDES slipstream (official rule).
    3. If player can pay total cost, call player.pay_heat().
    4. If player cannot pay (spin out):
       a. Pay all remaining heat.
       b. Move player back to first available space BEFORE the Corner Line
          (corner.start - 1, or earlier if occupied).
       c. Add Stress cards to hand: 1 if gear 1-2, 2 if gear 3-4.
       d. Set gear to 1.
       e. Set spun_out = True.

    AFTER FINISH LINE: If the player has already crossed the finish line,
    disregard any speed limits in corners. Simply move as far as possible.
    No heat cost for corners after finishing.

    No agent decisions -- this is deterministic.
    """
```

**CRITICAL differences from old design**:
1. **Speed calculation**: Uses `corner_speed_for_check()` which includes card speed + boost + adrenaline speed but EXCLUDES slipstream. The old design was ambiguous and in places said "slipstream corners use speed = 2" which is wrong.
2. **Spin-out placement**: Move to first available space BEFORE the corner start (`corner.start - 1`), NOT to `corner.end`. The old design incorrectly placed players at `corner.end`.
3. **Spin-out penalty**: Player receives 1 Stress card (gear 1-2) or 2 Stress cards (gear 3-4) added directly to hand. The old design incorrectly said "draw 1 fewer card" which is not in the official rules.
4. **After finish line**: Corners after the finish line are ignored -- no speed limits apply. The old design did not mention this.

#### Step 8: `step_discard`

```python
def step_discard(
    state: GameState,
    player: PlayerState,
    decision: list[Card],
) -> list[GameEvent]:
    """Player may optionally discard cards from their hand.

    Official rule: 'You may discard cards from your hand if you do not
    want to save them for future rounds... you can never choose to discard
    Stress or Heat cards.'

    This is a PLAYER CHOICE about which hand cards to voluntarily discard.
    Only Speed and Upgrade cards can be discarded (not Heat or Stress).
    This is optional -- the player may discard zero cards.

    decision: list of Card objects the player wants to discard from hand.
              Empty list means keep everything.

    NOTE: Moving played cards to discard pile happens in Replenish, not here.
    """
```

**CRITICAL difference from old design**: The old design treated Discard as automatic cleanup of played cards. The official rule is that Discard is an optional player decision about hand cards. The player can choose to throw away Speed/Upgrade cards they don't want for future rounds. They CANNOT discard Heat or Stress cards. Moving played cards (`cards_played`) to the discard pile is part of Replenish.

#### Step 9: `step_replenish`

```python
def step_replenish(
    state: GameState,
    player: PlayerState,
) -> list[GameEvent]:
    """Move played cards to discard and draw back up to hand size.

    Official rule for Replenish:
    1. Move all cards from Play Area (cards_played) to discard pile.
    2. Draw cards until hand size reaches HAND_SIZE (7).

    Also clears per-turn transient state:
    - player.cards_played = []
    - player.boost_used_this_turn = False
    - player.speed_from_cards = 0
    - player.speed_from_boost = 0
    - player.speed_from_adrenaline = 0
    - player.slipstream_moved = 0

    NOTE: Cooldown does NOT happen here. Cooldown is part of React (step 5).
    NOTE: There is NO "draw 1 fewer card" penalty for spin-out. The spin-out
    penalty is Stress cards added to hand (handled in step 7).
    """
```

**CRITICAL differences from old design**:
1. **No cooldown here**: Cooldown was incorrectly placed in Replenish. It belongs in React.
2. **No spin-out draw penalty**: The old design said spun-out players draw 1 fewer card. This is not in the official rules. The spin-out penalty is Stress cards.
3. **Moving played cards to discard**: This happens here, not in the Discard step. The Discard step is for voluntary hand card disposal.

### Phase Summary Table

| # | Step | Agent Decision? | Processing | Notes |
|---|------|----------------|------------|-------|
| 1 | Shift Gears | Yes (new gear + heat cost) | Simultaneous, all players | Collected before revealing |
| 2 | Play Cards | Yes (card combo) | Simultaneous, all players | Collected before revealing |
| 3 | Reveal & Move | No | Sequential, per player (front-to-back) | Part of per-player loop |
| 4 | Adrenaline | No (automatic grant of symbols) | Sequential, per player | Symbols used in React |
| 5 | React | Yes (cooldown count, boost?, adrenaline use) | Sequential, per player | Cooldown + Boost + Adrenaline |
| 6 | Slipstream | Yes (take it?) | Sequential, per player | Cannot cross finish line |
| 7 | Check Corner | No | Sequential, per player | Speed excludes slipstream |
| 8 | Discard | Yes (which hand cards to discard?) | Sequential, per player | Optional, no Heat/Stress |
| 9 | Replenish | No | Sequential, per player | Move played cards to discard, draw to 7 |

---

## Module 4: `engine/game.py` -- Game Orchestrator

### `ReactDecision` Dataclass

```python
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
```

### Agent Protocol

```python
from typing import Protocol

class Agent(Protocol):
    """Protocol that agents must satisfy.

    Defined here as a Protocol so the engine has zero dependency on the
    agents package. The concrete Agent base class in Sprint 3 will
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
```

**Changes from old Agent protocol**:
- `choose_gear` now takes `list[tuple[int, int]]` (gear, heat_cost pairs) instead of `list[int]`.
- `choose_adrenaline` is REMOVED -- adrenaline is automatic; the player's choice of how to use the symbols is part of `choose_react`.
- `choose_react` replaces the old heat-spend decision with a full `ReactDecision` covering cooldown + boost + adrenaline.
- `choose_discard` takes a filtered list of discardable cards (Speed/Upgrade only).

### Class: `Game`

```python
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
    ) -> None: ...

    @property
    def state(self) -> GameState: ...

    @property
    def is_over(self) -> bool: ...

    def run(self) -> GameResult: ...

    def run_round(self) -> list[GameEvent]: ...
```

### `GameResult` Dataclass

```python
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
```

### The Round Loop: `run_round()`

This is the heart of the engine. Each call to `run_round()` advances the game by one complete round.

```
def run_round(self) -> list[GameEvent]:
    events: list[GameEvent] = []

    # === SIMULTANEOUS STEPS (all players at once) ===

    # 0. Recompute turn order based on current positions
    self._state.compute_turn_order()

    # 1. SHIFT GEARS (simultaneous)
    #    - For each active player: query agent.choose_gear()
    #    - Validate against rules.legal_gear_shifts(current_gear, heat_available)
    #    - Spun-out players are forced to gear 1 (no choice)
    #    - +/-2 shifts pay 1 heat
    gear_decisions = self._collect_gear_decisions()
    events += phase_shift_gears(self._state, gear_decisions)

    # 2. PLAY CARDS (simultaneous)
    #    - For each active player: query agent.choose_cards()
    #    - Validate against rules.legal_card_plays()
    #    - Detect cluttered hands
    card_decisions = self._collect_card_decisions()
    events += phase_play_cards(self._state, card_decisions)

    # === PER-PLAYER SEQUENTIAL STEPS (front-to-back) ===

    # Recompute turn order after gear/card decisions
    # (positions haven't changed, but this ensures consistency)
    self._state.compute_turn_order()

    for pid in self._state.turn_order:
        player = self._state.get_player(pid)
        if player.finished:
            continue

        # CLUTTERED HAND CHECK: If the player had a cluttered hand,
        # their car does not move. Set gear to 1, place cards in
        # discard, skip steps 3-8, go straight to replenish.
        if rules.is_cluttered_hand(player.hand, player.gear):
            # Actually, cards were already removed from hand in step 2.
            # The check was done during card play. Handle the consequence:
            player.gear = 1
            events += step_replenish(self._state, player)
            continue

        # 3. REVEAL & MOVE
        events += step_reveal_and_move(self._state, player)
        if player.finished:
            continue

        # 4. ADRENALINE (automatic symbol grant)
        events += step_adrenaline(self._state, player)

        # 5. REACT (agent decision: cooldown + boost + adrenaline)
        react_decision = self._collect_react_decision(player)
        events += step_react(self._state, player, react_decision)
        if player.finished:
            continue

        # 6. SLIPSTREAM (agent decision: take it?)
        if rules.slipstream_eligible(
            player,
            list(self._state.active_players),
            self._state.track,
        ):
            take_slip = self._agents[pid].choose_slipstream(
                self._state, pid
            )
            events += step_slipstream(self._state, player, take_slip)
            if player.finished:
                continue

        # 7. CHECK CORNER
        events += step_check_corner(self._state, player)

        # 8. DISCARD (agent decision: which hand cards to toss?)
        discardable = [
            c for c in player.hand
            if c.card_type in (CardType.SPEED, CardType.UPGRADE)
        ]
        if discardable:
            to_discard = self._agents[pid].choose_discard(
                self._state, pid, discardable
            )
            events += step_discard(self._state, player, to_discard)

        # 9. REPLENISH
        events += step_replenish(self._state, player)

    # === END OF ROUND ===

    # Clear spun_out flags (penalty was applied this round, cleared for next)
    for player in self._state.active_players:
        player.spun_out = False

    # Advance round counter
    self._state.round_num += 1

    return events
```

### The Game Loop: `run()`

```python
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
```

`MAX_ROUNDS` constant (e.g., 200) prevents infinite loops in degenerate cases.

### Cluttered Hand Handling

The cluttered hand rule is checked at the start of each player's per-player loop:

```python
# In run_round(), inside the per-player loop:
if is_cluttered_hand_situation(player):
    # The player already "played" cards in step 2 (all playable + heat fill).
    # Those cards are in cards_played. The car does not move.
    player.gear = 1
    # Skip steps 3-8, go straight to replenish:
    events += step_replenish(self._state, player)
    continue
```

The `is_cluttered_hand` check uses the state from BEFORE cards were removed from hand. Since `phase_play_cards` removes cards from hand, we need to track this condition during card play. Add a `cluttered: bool = False` field to `PlayerState`, set during `phase_play_cards` if the player had to use heat cards to fill their play count.

### Decision Collection Methods

These private methods on `Game` bridge agents and phases:

```python
def _collect_gear_decisions(self) -> dict[int, tuple[int, int]]:
    """Query each active agent for their gear choice."""
    decisions: dict[int, tuple[int, int]] = {}
    for player in self._state.active_players:
        pid = player.player_id
        if player.spun_out:
            decisions[pid] = (1, 0)  # Forced to gear 1, no cost
        else:
            legal = rules.legal_gear_shifts(player.gear, player.heat_available)
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
        player.heat_available > 0
        and not player.boost_used_this_turn
    )

    return self._agents[pid].choose_react(
        self._state,
        pid,
        max_cooldown=max_cooldown,
        can_boost=can_boost,
        has_adrenaline=has_adrenaline,
    )
```

---

## Engine `__init__.py` Exports

```python
# engine/__init__.py
"""HEAT game engine."""

from heat.engine.game import Game, GameResult, ReactDecision
from heat.engine.phases import (
    phase_shift_gears,
    phase_play_cards,
    step_reveal_and_move,
    step_adrenaline,
    step_react,
    step_slipstream,
    step_check_corner,
    step_discard,
    step_replenish,
)
from heat.engine import rules

__all__ = [
    "Game",
    "GameResult",
    "ReactDecision",
    "phase_shift_gears",
    "phase_play_cards",
    "step_reveal_and_move",
    "step_adrenaline",
    "step_react",
    "step_slipstream",
    "step_check_corner",
    "step_discard",
    "step_replenish",
    "rules",
]
```

Note the naming change: simultaneous phases keep the `phase_` prefix; per-player sequential steps use the `step_` prefix. This makes the distinction clear at the call site.

---

## Alternatives Considered

### 1. Immutable State with Copy-on-Write

Instead of mutating `GameState` in place, each phase could return a new `GameState`. This would be cleaner functionally but kills performance: copying 4-6 `PlayerState` objects (each with a `Deck` containing two lists) 9 times per round would dominate the <10ms budget. **Rejected** in favor of in-place mutation.

### 2. Event Sourcing as Primary State

Instead of mutating state and also logging events, derive all state from events. Elegant but adds complexity and slows everything down. The PLAN.md already chose "toggleable logging" which implies state is primary. **Rejected**.

### 3. Moving GameEvent to engine/events.py

Would require updating `models/game_state.py`, `models/__init__.py`, and all existing tests. Gains nothing since `GameEvent` is a pure data container. **Rejected** -- keep in models, re-export from engine.

### 4. Separate Move Resolution for Collisions

A full collision-resolution system where players block each other in single-lane spaces. This adds significant complexity. The base HEAT game handles this with lane rules. **Deferred** -- implement a simplified version where single-lane spaces have capacity limits, and excess players stop 1 space behind. Can be refined later.

### 5. All-Phase-Then-All-Players vs Per-Player Sequential

The old design processed each phase across all players before moving to the next phase. The official rules require steps 3-9 to be processed per-player (each player completes all 7 steps before the next). **Per-player sequential adopted** to match official rules. This means one player's slipstream movement is visible to the next player's reveal, which affects game dynamics.

---

## Edge Cases and Tricky Rules

### 1. Cluttered Hand (Heat Card Clogging)
If a player has so many Heat cards in hand that they cannot play enough non-Heat cards for their gear: use as many playable cards as possible, cover the difference with Heat cards, the car does NOT move, gear immediately set to 1, all played cards placed in discard pile, skip steps 3-8 and go straight to Replenish (step 9). This is detected via `is_cluttered_hand()` and handled in the per-player loop.

### 2. Stress Resolution Loops Through Non-Basic Cards
When resolving a Stress card (or Boost), flip cards from the draw pile one at a time. If the flipped card is NOT a Basic (SPEED) card -- i.e., it is Heat, Stress, or Upgrade -- discard it immediately and keep flipping. Only stop when a SPEED card is found (or the deck is exhausted). This can potentially flip many cards.

### 3. Boost is Once Per Turn, Not Unlimited
The old design allowed paying N heat for +N speed. The official rule: pay 1 heat for a single boost flip (same flip-until-Basic mechanic as stress). Limited to once per turn. Tracked via `player.boost_used_this_turn`.

### 4. Adrenaline Grants Symbols, Not Movement
Adrenaline does NOT move the player directly. It grants +1 speed and/or +1 cooldown symbols that are consumed during the React step. The player can choose to use both, either, or neither.

### 5. Adrenaline Threshold Uses Starting Player Count
The number of trailing players who get adrenaline is based on how many players started the race, NOT how many are still active. If 5 players started and 3 have finished, the threshold is still "last 2 places" (because 5 started), applied to the 2 remaining active players.

### 6. Corner Speed Excludes Slipstream, Includes Boost
Speed for corner checking = card values + stress flip values + boost flip value + adrenaline +1 speed (if used). Slipstream movement (2 spaces) does NOT count. Even if slipstream moves a player through a corner, the speed checked is the card+boost+adrenaline speed. The old design was wrong in saying "slipstream corners use speed = 2."

### 7. Slipstream Cannot Cross Finish Line
A player on their final lap cannot slipstream past the finish line. `slipstream_eligible()` must check this. Finished players also cannot slipstream.

### 8. Spin-Out: Back to Before Corner, Not Corner End
On spin-out, the player moves back to the first available space BEFORE the corner line (`corner.start - 1`), not to `corner.end` as the old design stated. If that space is occupied (single-lane), move further back.

### 9. Spin-Out: Stress Cards, Not Draw Penalty
Spin-out penalty: take 1 Stress card into hand if gear 1-2, or 2 Stress cards if gear 3-4. There is NO "draw 1 fewer card" penalty. The old design was wrong.

### 10. Corners After Finish Line Are Ignored
Official rule: "Disregard any Speed Limits in corners after the finish line, simply move as far as you can." Once a player crosses the finish line, corner speed limits no longer apply. `step_check_corner` must check whether the player has already finished and skip corner cost assessment.

### 11. Discard Is Player Choice, Not Automatic Cleanup
The Discard step (step 8) is an optional player decision to throw away unwanted hand cards (Speed/Upgrade only, cannot discard Heat or Stress). Moving played cards (`cards_played`) to the discard pile is part of Replenish (step 9), not the Discard step.

### 12. Cooldown Is In React, Not Replenish
Cooldown (Gear 1 = 3, Gear 2 = 1) happens during React (step 5), not Replenish (step 9). Replenish is just: move played cards to discard, draw to 7.

### 13. Starting Upgrade Cards Are Not Basic
The three Starting Upgrade cards per player include UPGRADE-type cards (value 0 and value 5). During stress/boost resolution, these are NOT Basic cards -- if flipped, they are discarded and flipping continues.

### 14. Multi-Corner Moves
A high-speed move can cross multiple corners. Each corner's heat cost is calculated independently based on the same speed value. The player pays the sum of all corner costs, not just the worst one.

### 15. Finishing Mid-Round
A player can finish during step 3 (reveal & move), step 5 (react -- from boost or adrenaline speed), or step 6 (slipstream -- wait, no, slipstream cannot cross finish line). Once finished, remaining steps are skipped. The `finish_order` counter increments.

### 16. All Players Spin Out
If all players spin out in the same corner, the game continues (they are placed before the corner, gear 1, and resume next round). The game only ends when all players finish.

### 17. Wrap-Around on Multi-Lap Tracks
Position arithmetic must use `% track.length`. A player at position 28 on a 30-space track moving 5 spaces ends at position 3 and completes a lap.

### 18. Turn Order Ties
Two players at the same position and lap: broken by lower `player_id` first. This is implemented in `GameState.compute_turn_order()` already.

### 19. Simultaneous Finish
If multiple players cross the finish line in the same reveal-and-move step, the one who moved further past the finish line ranks higher. If still tied, the one with higher original position (before this round) ranks higher.

### 20. Not Enough Cards to Play
If a player's hand has fewer playable cards (speed + stress + upgrade) than their gear requires, they must play heat cards from their hand to fill the gap. Heat cards contribute 0 speed. `legal_card_plays()` must handle this.

### 21. Empty Deck During Stress/Boost Resolution
If the deck is empty (draw pile and discard pile both empty) when resolving a stress card or boost, the flip contributes 0 speed. This is unlikely but possible if many heat cards have clogged the deck.

---

## Testing Strategy

### `tests/test_rules.py`

| Test | What it Validates |
|------|-------------------|
| `test_legal_gear_shifts_from_1_no_heat` | Returns [(1, 0), (2, 0)] |
| `test_legal_gear_shifts_from_1_with_heat` | Returns [(1, 0), (2, 0), (3, 1)] |
| `test_legal_gear_shifts_from_4_no_heat` | Returns [(3, 0), (4, 0)] |
| `test_legal_gear_shifts_from_4_with_heat` | Returns [(2, 1), (3, 0), (4, 0)] |
| `test_legal_gear_shifts_from_2_no_heat` | Returns [(1, 0), (2, 0), (3, 0)] |
| `test_legal_gear_shifts_from_2_with_heat` | Returns [(1, 0), (2, 0), (3, 0), (4, 1)] |
| `test_legal_gear_shifts_from_3_with_heat` | Returns [(1, 1), (2, 0), (3, 0), (4, 0)] |
| `test_cards_to_play_count` | gear 1 -> 1, gear 4 -> 4 |
| `test_legal_card_plays_basic` | 7 speed cards, gear 2 -> C(7,2) = 21 combos |
| `test_legal_card_plays_with_heat_in_hand` | Heat cards excluded unless forced |
| `test_legal_card_plays_insufficient_playable` | Must play heat from hand |
| `test_legal_card_plays_stress_cards` | Stress cards are playable |
| `test_legal_card_plays_upgrade_cards` | Upgrade cards are playable |
| `test_is_cluttered_hand_true` | 5 heat + 2 speed in hand, gear 3 -> True |
| `test_is_cluttered_hand_false` | 2 heat + 5 speed in hand, gear 3 -> False |
| `test_corner_heat_cost_under_limit` | Returns 0 |
| `test_corner_heat_cost_over_limit` | Returns speed - limit |
| `test_corner_heat_cost_at_limit` | Returns 0 |
| `test_corners_crossed_none` | Move within straight |
| `test_corners_crossed_one` | Move through one corner |
| `test_corners_crossed_multiple` | Long move crosses two corners |
| `test_corners_crossed_wraparound` | Multi-lap crossing finish line |
| `test_spin_out_check` | True when cost > available heat |
| `test_no_spin_out` | False when cost <= available heat |
| `test_spin_out_stress_count_low_gear` | Gear 1 or 2 -> 1 stress card |
| `test_spin_out_stress_count_high_gear` | Gear 3 or 4 -> 2 stress cards |
| `test_slipstream_eligible_car_ahead` | True with car 1-2 ahead |
| `test_slipstream_eligible_no_car` | False with no car nearby |
| `test_slipstream_cannot_cross_finish` | False on final lap if +2 would cross |
| `test_cooldown_gear_1` | Returns 3 |
| `test_cooldown_gear_2` | Returns 1 |
| `test_cooldown_gear_3_4` | Returns 0 |
| `test_resolve_stress_card_basic_on_top` | Draws 1 SPEED card, returns value |
| `test_resolve_stress_card_skips_heat` | Skips Heat cards, finds SPEED |
| `test_resolve_stress_card_skips_upgrade` | Skips Upgrade cards, finds SPEED |
| `test_resolve_stress_card_empty_deck` | Returns (0, []) |
| `test_resolve_boost_same_as_stress` | Same flip-until-basic mechanic |
| `test_corner_speed_for_check_excludes_slipstream` | Slipstream not in speed |
| `test_corner_speed_for_check_includes_boost` | Boost value included |
| `test_corner_speed_for_check_includes_adrenaline` | Adrenaline +1 included |
| `test_calculate_move_position_normal` | Simple forward movement |
| `test_calculate_move_position_wrap` | Lap completion |
| `test_adrenaline_eligible_last_2p` | Last place gets adrenaline (2 started) |
| `test_adrenaline_eligible_last_5p` | Last 2 places get adrenaline (5 started) |
| `test_adrenaline_uses_starting_count` | Even if players finished, starting count matters |
| `test_adrenaline_not_eligible_leader` | Leader does not get adrenaline |

### `tests/test_phases.py`

Each step/phase gets its own test class. Tests use a small helper track (5-10 spaces, 1 corner) and manually constructed `GameState` objects.

| Test Class | Key Tests |
|------------|-----------|
| `TestPhaseShiftGears` | Valid shift applied; illegal shift raises; spun-out forced to 1; +/-2 shift pays 1 heat |
| `TestPhasePlayCards` | Cards removed from hand; stored in cards_played; illegal combo raises; cluttered hand detected |
| `TestStepRevealAndMove` | Position updated correctly; stress resolved with loop-until-basic; lap incremented; finish detected |
| `TestStepAdrenaline` | Eligible player gets symbol flags set; non-eligible skipped; symbols not consumed yet |
| `TestStepReact` | Cooldown moves heat from hand to pool; boost flips until basic + pays 1 heat; boost once per turn enforced; adrenaline speed adds to movement and corner speed; adrenaline cooldown adds to cooldown max |
| `TestStepSlipstream` | Eligible moves +2; not eligible stays; cannot cross finish line; slipstream_moved tracked |
| `TestStepCheckCorner` | Under limit = no cost; over limit = heat paid; speed excludes slipstream; speed includes boost; spin-out: placed before corner start, gets stress cards, gear 1; corners ignored after finish line |
| `TestStepDiscard` | Only Speed/Upgrade can be discarded; Heat/Stress cannot; optional (empty list = keep all) |
| `TestStepReplenish` | Played cards moved to discard; hand refilled to 7; cooldown NOT here; transient fields cleared; no draw penalty for spin-out |

### `tests/test_game.py`

Integration-level tests using a mock agent.

| Test | What it Validates |
|------|-------------------|
| `test_game_creation` | Game initializes with correct state; starting_player_count set |
| `test_single_round` | One round completes without error |
| `test_per_player_sequential_order` | Steps 3-9 run per-player, front-to-back |
| `test_cluttered_hand_skips_to_replenish` | Player with cluttered hand doesn't move, gear set to 1 |
| `test_game_terminates` | Full game with random-like agent finishes |
| `test_finish_order_correct` | Winner finishes first |
| `test_slipstream_no_cross_finish` | Slipstream blocked at finish line |
| `test_spin_out_placement_before_corner` | Spun out player placed before corner.start |
| `test_spin_out_stress_penalty` | Spun out player gets 1 or 2 stress cards |
| `test_gear_shift_plus_minus_2` | +/-2 gear shift costs 1 heat |
| `test_boost_once_per_turn` | Second boost attempt rejected |
| `test_adrenaline_uses_starting_count` | Adrenaline threshold uses starting players |
| `test_corners_ignored_after_finish` | No corner cost after crossing finish |
| `test_max_rounds_safety` | Game terminates at MAX_ROUNDS |
| `test_event_log_populated` | Events logged when enabled |
| `test_event_log_disabled` | No events when disabled |

### Test Helpers

Create `tests/conftest.py` with shared fixtures:

```python
import pytest
from heat.models.cards import Card, CardType
from heat.models.track import Track, Space, Corner
from heat.models.game_state import GameState

@pytest.fixture
def small_track() -> Track:
    """A minimal track for testing: 10 spaces, 1 corner."""
    spaces = [Space(i, lanes=2) for i in range(10)]
    corners = [Corner(start=4, end=6, speed_limit=3)]
    return Track("Test", spaces, corners, [0, 1, 2, 3], laps=1)

@pytest.fixture
def game_state(small_track: Track) -> GameState:
    """A 2-player game on the small track."""
    return GameState.create(small_track, 2)

def make_stress_card(player_id: int, idx: int = 0) -> Card:
    """Helper to create a stress card for testing."""
    return Card(CardType.STRESS, 0, f"p{player_id}_stress_{idx}")

def make_upgrade_card(player_id: int, value: int, idx: int = 0) -> Card:
    """Helper to create an upgrade card for testing."""
    return Card(CardType.UPGRADE, value, f"p{player_id}_upg_{idx}")
```

---

## Implementation Order

The modules have clear dependencies. Build bottom-up:

### Step 1: Model additions
- Add `UPGRADE` to `CardType` enum in `models/cards.py`
- Add `create_starting_upgrade_cards()` to `models/cards.py`
- Add transient fields to `PlayerState`: `cards_played`, `boost_used_this_turn`, `speed_from_cards`, `speed_from_boost`, `speed_from_adrenaline`, `slipstream_moved`, `cluttered`
- Update `PlayerState.create()` to include Starting Upgrade cards in the deck
- Add `starting_player_count` field to `GameState`
- Update `GameState.create()` to set `starting_player_count`
- Update any tests that check field counts (unlikely to break)
- **Commit**: "Add UPGRADE card type, Starting Upgrade cards, and per-turn tracking fields"

### Step 2: `engine/rules.py` -- Core rules (no phase dependencies)
- Implement all pure functions listed above, including:
  - `legal_gear_shifts()` with +/-2 and heat cost
  - `resolve_stress_card()` with loop-until-Basic
  - `resolve_boost()` with same mechanic
  - `is_cluttered_hand()`
  - `corner_speed_for_check()` excluding slipstream
  - `spin_out_stress_count()`
  - `slipstream_would_cross_finish()`
  - `adrenaline_eligible()` with starting_player_count
- Write `tests/test_rules.py` with full coverage
- This module has zero dependency on the rest of the engine
- **Commit**: "Implement engine/rules.py with all game rule functions"

### Step 3: `engine/events.py` -- Trivial re-export
- Create the module with docstring and re-export
- **Commit**: "Add engine/events.py as GameEvent re-export point"

### Step 4: `engine/phases.py` -- Simultaneous phases + per-player steps
- Implement simultaneous phases first:
  1. `phase_shift_gears` -- applies gear changes, pays heat for +/-2 shifts
  2. `phase_play_cards` -- removes cards from hand, stores in cards_played, detects cluttered
- Then implement per-player steps (simplest to most complex):
  3. `step_replenish` -- moves played cards to discard, draws to 7, clears transients
  4. `step_discard` -- optional hand card disposal (Speed/Upgrade only)
  5. `step_reveal_and_move` -- resolves stress (loop-until-Basic), calculates movement
  6. `step_adrenaline` -- grants symbol flags to eligible trailing players
  7. `step_react` -- cooldown + boost (once, flip-until-Basic) + adrenaline symbols
  8. `step_slipstream` -- +2 movement, cannot cross finish, tracks slipstream_moved
  9. `step_check_corner` -- speed excludes slipstream, spin-out places before corner, stress penalty
- Write `tests/test_phases.py` as each step is implemented
- **Commit**: "Implement engine/phases.py with simultaneous phases and per-player steps"

### Step 5: `engine/game.py` -- Orchestrator
- Implement `ReactDecision` dataclass
- Implement `Agent` protocol with updated method signatures
- Implement `GameResult` dataclass
- Implement `Game` class with:
  - Simultaneous steps 1-2 for all players
  - Per-player sequential loop for steps 3-9 (front-to-back)
  - Cluttered hand handling (skip to replenish)
  - Finish detection at multiple points
- Write `tests/test_game.py` with a mock agent (random choices from legal options)
- **Commit**: "Implement engine/game.py with Game orchestrator and per-player round loop"

### Step 6: `engine/__init__.py` -- Package exports
- Wire up all exports with `phase_` and `step_` naming convention
- **Commit**: "Configure engine package exports"

### Step 7: Integration smoke test
- Write a test that runs a complete game with 4 random-choice agents on the USA track
- Verify: game terminates, no crashes, exactly one winner per finishing position
- Test cluttered hand, spin-out, slipstream restrictions in integration
- **Commit**: "Add integration smoke test for full game simulation"

---

## Dependency Graph

```
models/cards.py ──────┐
models/track.py ──────┤
models/player_state.py┤
models/game_state.py ─┴──> engine/rules.py ──> engine/phases.py ──> engine/game.py
                           engine/events.py ─────────────────────┘
```

The engine package depends on models but never the reverse. Agents (Sprint 3) will depend on `engine/game.py`'s `Agent` protocol and `engine/rules.py`'s legal-move functions.

---

## Open Questions

1. **Lane blocking / collision resolution**: The physical HEAT game has detailed rules about how cars occupy lanes. Should Sprint 2 implement full lane mechanics, or simplify to "spaces have a max capacity and excess cars stop behind"? **Recommendation**: Simplify for now. Full lane mechanics can be added in a follow-up without changing the engine API.

2. **Stress card pool**: Where do the Stress cards for spin-out penalties come from? The official rules have a shared supply of Stress cards. We need a shared pool in `GameState` or a factory function that creates Stress cards on demand. **Recommendation**: Create Stress cards on demand with a global counter for unique IDs. No shared pool needed in the data model.

3. **Spin-out placement with occupied spaces**: When placing a spun-out player before the corner start (`corner.start - 1`), what if that space is occupied by another car in a single-lane section? **Recommendation**: Find the first available space moving backwards from `corner.start - 1`. If no space is available (extremely unlikely), place at `corner.start - 1` regardless (allow stacking as a simplification).

4. **Simultaneous finish tie-breaking**: If two players cross the finish line in the same phase, what determines rank? **Recommendation**: The player who ends further past the finish line ranks higher. If still tied, use their pre-move position (further back = lower rank, since they moved faster).

5. **Max rounds safety valve**: What should `MAX_ROUNDS` be? With a 30-space track, 1 lap, and even the slowest possible play (gear 1, playing a value-1 card each round), a player moves 1 space per round and finishes in 30 rounds. 200 is a generous upper bound. For multi-lap tracks, scale with `track.laps`.

6. **Boost gear availability**: The official rules show Boost symbols on specific gear positions. For simplicity, should boost be available in all gears, or only certain gears? The rules indicate boost is available in gear 2+ (gears 2, 3, 4 have the boost symbol; gear 1 does not). **Recommendation**: Allow boost only in gears 2-4 for now, with a constant `BOOST_MIN_GEAR = 2` that can be adjusted.

---

## Summary of All 15 Discrepancy Fixes

For traceability, here is how each of the 15 identified discrepancies is addressed in this design:

| # | Discrepancy | Fix Applied |
|---|-------------|-------------|
| 1 | Turn structure: steps 3-9 per player | Architectural change to per-player sequential loop in `run_round()` |
| 2 | Gear shifting +/-2 costs 1 heat | `legal_gear_shifts()` accepts `heat_available`, returns `(gear, cost)` tuples |
| 3 | Stress resolution: loop until Basic | `resolve_stress_card()` loops, discarding non-SPEED cards |
| 4 | Boost: once per turn, flip-until-Basic | `resolve_boost()` uses same loop; `boost_used_this_turn` flag; part of React |
| 5 | Adrenaline: symbols not movement | Adrenaline grants symbols consumed in React, not direct movement |
| 6 | React: Cooldown + Boost + Adrenaline | `step_react()` handles all three; `ReactDecision` dataclass |
| 7 | Cooldown in React, not Replenish | Moved from `step_replenish` to `step_react` |
| 8 | Discard: player choice | `step_discard()` is optional hand card disposal (Speed/Upgrade only) |
| 9 | Spin-out: before corner, stress cards | Placed at `corner.start - 1`; `spin_out_stress_count()` for 1-2 stress cards |
| 10 | Corner speed: includes boost, excludes slipstream | `corner_speed_for_check()` sums cards + boost + adrenaline, not slipstream |
| 11 | Slipstream cannot cross finish | `slipstream_would_cross_finish()` check in `slipstream_eligible()` |
| 12 | Cluttered hand: no movement, gear 1 | `is_cluttered_hand()` detection; skip steps 3-8 in orchestrator |
| 13 | Starting Upgrade cards | `UPGRADE` CardType; `create_starting_upgrade_cards()`; not Basic for flips |
| 14 | Ignore corners after finish | `step_check_corner()` skips corner costs for finished players |
| 15 | Adrenaline: starting player count | `adrenaline_eligible()` takes `starting_player_count`; `GameState` tracks it |
