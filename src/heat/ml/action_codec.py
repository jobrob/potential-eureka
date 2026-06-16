"""Flattened, masked action codec for the HEAT RL layer (Sprint 5a).

Implements the three §3.2 contract functions over a single flattened
``Discrete(ACTION_DIM)`` space whose sub-range offsets live in
:mod:`heat.ml.spaces`:

* :func:`legal_action_mask` -- bool array, True exactly where the flat action is
  legal for the *current* decision.
* :func:`encode_action_index` -- concrete engine action -> flat index.
* :func:`decode_action` -- flat index -> concrete object the driver's
  ``send(...)`` expects for ``decision.kind``.

Value-redundant card plays are collapsed: the CARDS sub-range enumerates over
*value-multisets* (e.g. "play {3, 3}"), not concrete ``Card`` objects, so two
equal-value Speed cards map to one action index. ``decode_action`` realizes a
chosen value-multiset to concrete cards by picking the lowest-``id``
representatives from the current hand. No engine rule is changed (§7).
"""

from __future__ import annotations

import itertools

import numpy as np

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.phases import ReactDecision
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.ml import spaces
from heat.ml.spaces import (
    ACTION_DIM,
    CARDS_OFFSET,
    DISCARD_OFFSET,
    DISCARD_SIZE,
    GEAR_OFFSET,
    GEAR_SIZE,
    REACT_OFFSET,
    REACT_SIZE,
    SLIPSTREAM_OFFSET,
)

# ---------------------------------------------------------------------------
# CARDS sub-range: value-multiset enumeration
# ---------------------------------------------------------------------------
#
# Each playable/forced card maps to a canonical "value token". Two cards with
# the same token are interchangeable for the action space (the engine keeps them
# distinct by id; PPO must not waste probability mass on the duplicates).


def card_token(card: Card) -> str:
    """Return the canonical value-token for a card.

    Tokens (the 8-symbol alphabet): ``S1..S4`` (Speed by value), ``U0``/``U5``
    (Upgrade by value), ``ST`` (Stress), ``H`` (Heat, forced filler only).
    """
    if card.card_type == CardType.SPEED:
        return f"S{card.value}"
    if card.card_type == CardType.UPGRADE:
        return f"U{card.value}"
    if card.card_type == CardType.STRESS:
        return "ST"
    return "H"  # HEAT


#: The fixed token alphabet (sorted for a stable enumeration order).
_TOKEN_ALPHABET: tuple[str, ...] = ("H", "S1", "S2", "S3", "S4", "ST", "U0", "U5")


def _enumerate_card_multisets() -> list[tuple[str, ...]]:
    """Enumerate every distinct value-multiset reachable across gears 1-4.

    A play is exactly ``gear`` cards, so a play's value-multiset has size in
    1..4. The reachable set is all size-g multisets over the 8-token alphabet
    for g in 1..4. Each multiset is stored as a sorted token tuple; the list
    order is the canonical, frozen index order for the CARDS sub-range.
    """
    seen: set[tuple[str, ...]] = set()
    ordered: list[tuple[str, ...]] = []
    for g in range(1, rules.MAX_GEAR + 1):
        for combo in itertools.combinations_with_replacement(_TOKEN_ALPHABET, g):
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


#: Canonical ordered list of card value-multisets; index in this list + the
#: CARDS offset is the flat action index.
_CARD_MULTISETS: list[tuple[str, ...]] = _enumerate_card_multisets()
_CARD_MULTISET_INDEX: dict[tuple[str, ...], int] = {
    ms: i for i, ms in enumerate(_CARD_MULTISETS)
}

# Self-check the frozen contract: enumeration must match spaces.CARDS_SIZE.
assert len(_CARD_MULTISETS) == spaces.CARDS_SIZE, (
    f"CARDS enumeration produced {len(_CARD_MULTISETS)} multisets but "
    f"spaces.CARDS_SIZE == {spaces.CARDS_SIZE}"
)


def _play_to_multiset(cards: tuple[Card, ...]) -> tuple[str, ...]:
    """Canonical value-multiset (sorted token tuple) for a concrete card play."""
    return tuple(sorted(card_token(c) for c in cards))


def _realize_multiset(
    multiset: tuple[str, ...], hand: list[Card]
) -> tuple[Card, ...]:
    """Realize a token multiset to concrete cards from ``hand``.

    For each token, pick unused cards matching that token by lowest ``id``
    first. Raises ValueError if the hand cannot satisfy the multiset (the
    caller should only decode masked-legal indices, so this is a guard).
    """
    # Bucket hand cards by token, each bucket sorted by id (lowest first).
    buckets: dict[str, list[Card]] = {}
    for card in hand:
        buckets.setdefault(card_token(card), []).append(card)
    for cards in buckets.values():
        cards.sort(key=lambda c: c.id)

    chosen: list[Card] = []
    cursor: dict[str, int] = {}
    for token in multiset:
        bucket = buckets.get(token, [])
        idx = cursor.get(token, 0)
        if idx >= len(bucket):
            raise ValueError(
                f"Cannot realize card multiset {multiset}: hand lacks token "
                f"{token!r}"
            )
        chosen.append(bucket[idx])
        cursor[token] = idx + 1
    return tuple(chosen)


# ---------------------------------------------------------------------------
# REACT sub-range: fixed 8-slot enumeration
# ---------------------------------------------------------------------------
#
# (cooldown_count, use_boost, use_adrenaline_speed, use_adrenaline_cooldown).
# A behavior-covering, fixed table. Slots whose preconditions are not met for a
# given ReactOptions are masked off.

_REACT_TABLE: tuple[tuple[int, bool, bool, bool], ...] = (
    (0, False, False, False),  # 0: do nothing
    (1, False, False, False),  # 1: cool 1
    (2, False, False, False),  # 2: cool 2
    (3, False, False, False),  # 3: cool 3 (gear-1 max)
    (0, True, False, False),   # 4: boost only
    (0, False, True, False),   # 5: adrenaline speed
    (3, False, False, True),   # 6: max cool + adrenaline cooldown (up to 4)
    (0, True, True, False),    # 7: boost + adrenaline speed
)
assert len(_REACT_TABLE) == REACT_SIZE


def _react_slot_legal(
    slot: tuple[int, bool, bool, bool], opts: rules.ReactOptions
) -> bool:
    """Whether a REACT table slot is legal under the given ReactOptions."""
    cooldown_count, use_boost, use_adr_speed, use_adr_cooldown = slot
    max_cd = opts.max_cooldown + (1 if use_adr_cooldown else 0)
    if cooldown_count > max_cd:
        return False
    if use_boost and not opts.can_boost:
        return False
    if (use_adr_speed or use_adr_cooldown) and not opts.has_adrenaline:
        return False
    return True


def _react_to_slot_index(decision_obj: ReactDecision) -> int:
    """Map a concrete ReactDecision to its REACT table slot index (exact)."""
    key = (
        decision_obj.cooldown_count,
        decision_obj.use_boost,
        decision_obj.use_adrenaline_speed,
        decision_obj.use_adrenaline_cooldown,
    )
    for i, slot in enumerate(_REACT_TABLE):
        if slot == key:
            return i
    raise ValueError(f"ReactDecision {decision_obj} is not in the fixed table")


# ---------------------------------------------------------------------------
# DISCARD sub-range
# ---------------------------------------------------------------------------
#
# index 0 == discard none; index k == discard the k lowest discardable cards
# (sorted by value then id). Masked to the available number of discardable cards.


def _discard_order(discardable: list[Card]) -> list[Card]:
    """Discardable cards sorted lowest-first (value then id) -- the canonical
    order the "discard lowest-k" actions slice from."""
    return sorted(discardable, key=lambda c: (c.value, c.id))


# ---------------------------------------------------------------------------
# Public contract functions (§3.2)
# ---------------------------------------------------------------------------


def legal_action_mask(decision: Decision, state: GameState) -> np.ndarray:
    """Bool array of shape ``(ACTION_DIM,)``: True where the flat action is legal
    for THIS decision. Never all-False (every decision has >= 1 legal action)."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    kind = decision.kind

    if kind == DecisionKind.GEAR:
        # decision.legal is list[(new_gear, heat_cost)]; gear in 1..4.
        for new_gear, _heat_cost in decision.legal:
            mask[GEAR_OFFSET + (new_gear - rules.MIN_GEAR)] = True

    elif kind == DecisionKind.CARDS:
        # decision.legal is list[tuple[Card, ...]]; collapse to value-multisets.
        for play in decision.legal:
            idx = _CARD_MULTISET_INDEX.get(_play_to_multiset(play))
            if idx is not None:
                mask[CARDS_OFFSET + idx] = True

    elif kind == DecisionKind.REACT:
        opts: rules.ReactOptions = decision.legal
        for i, slot in enumerate(_REACT_TABLE):
            if _react_slot_legal(slot, opts):
                mask[REACT_OFFSET + i] = True

    elif kind == DecisionKind.SLIPSTREAM:
        # Both take and decline are legal whenever a slipstream decision is asked.
        mask[SLIPSTREAM_OFFSET + 0] = True  # take
        mask[SLIPSTREAM_OFFSET + 1] = True  # decline

    elif kind == DecisionKind.DISCARD:
        # decision.legal is list[Card] (discardable). index 0 = discard none;
        # index k = discard k lowest, for k in 1..min(len, DISCARD_SIZE-1).
        n = len(decision.legal)
        mask[DISCARD_OFFSET + 0] = True
        for k in range(1, DISCARD_SIZE):
            if k <= n:
                mask[DISCARD_OFFSET + k] = True

    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown decision kind {kind!r}")

    return mask


def encode_action_index(decision: Decision, engine_action: object) -> int:
    """Map a concrete engine action to its flat index (inverse of decode within
    the legal set). Used by tests and optional imitation."""
    kind = decision.kind

    if kind == DecisionKind.GEAR:
        new_gear, _heat_cost = engine_action  # type: ignore[misc]
        return GEAR_OFFSET + (new_gear - rules.MIN_GEAR)

    if kind == DecisionKind.CARDS:
        multiset = _play_to_multiset(tuple(engine_action))  # type: ignore[arg-type]
        idx = _CARD_MULTISET_INDEX.get(multiset)
        if idx is None:
            raise ValueError(f"Card play {engine_action} has no encoding")
        return CARDS_OFFSET + idx

    if kind == DecisionKind.REACT:
        return REACT_OFFSET + _react_to_slot_index(engine_action)  # type: ignore[arg-type]

    if kind == DecisionKind.SLIPSTREAM:
        return SLIPSTREAM_OFFSET + (0 if engine_action else 1)

    if kind == DecisionKind.DISCARD:
        discardable: list[Card] = list(decision.legal)
        cards = list(engine_action)  # type: ignore[arg-type]
        k = len(cards)
        if k >= DISCARD_SIZE:
            raise ValueError(f"Discard of {k} exceeds DISCARD_SIZE")
        return DISCARD_OFFSET + k

    raise ValueError(f"Unknown decision kind {kind!r}")


def decode_action(
    decision: Decision, flat_index: int, state: GameState
) -> object:
    """Map a flat index back to the concrete object the driver expects for
    ``decision.kind``. Inverse of :func:`encode_action_index` within the legal
    set."""
    kind = decision.kind

    if kind == DecisionKind.GEAR:
        new_gear = (flat_index - GEAR_OFFSET) + rules.MIN_GEAR
        for option in decision.legal:  # (new_gear, heat_cost)
            if option[0] == new_gear:
                return option
        raise ValueError(f"Gear index {flat_index} not in legal set")

    if kind == DecisionKind.CARDS:
        multiset = _CARD_MULTISETS[flat_index - CARDS_OFFSET]
        # Return the matching tuple straight from ``decision.legal`` so the
        # result is directly sendable to ``run_round_driver`` -- the driver's
        # guard checks ``chosen in legal_plays``, and ``rules.legal_card_plays``
        # yields tuples in HAND order, whereas ``_realize_multiset`` would
        # re-order to token/id order and fail that identity check for multi-card
        # plays. Only fall back to realizing from the hand when no legal play
        # matches (e.g. a codec round-trip with a hand-only context).
        for play in decision.legal:
            if _play_to_multiset(play) == multiset:
                return play
        player = state.get_player(decision.player_id)
        return _realize_multiset(multiset, player.hand)

    if kind == DecisionKind.REACT:
        cooldown_count, use_boost, use_adr_speed, use_adr_cooldown = _REACT_TABLE[
            flat_index - REACT_OFFSET
        ]
        return ReactDecision(
            cooldown_count=cooldown_count,
            use_boost=use_boost,
            use_adrenaline_speed=use_adr_speed,
            use_adrenaline_cooldown=use_adr_cooldown,
        )

    if kind == DecisionKind.SLIPSTREAM:
        return (flat_index - SLIPSTREAM_OFFSET) == 0  # 0 -> take, 1 -> decline

    if kind == DecisionKind.DISCARD:
        k = flat_index - DISCARD_OFFSET
        if k == 0:
            return []
        ordered = _discard_order(list(decision.legal))
        return ordered[:k]

    raise ValueError(f"Unknown decision kind {kind!r}")
