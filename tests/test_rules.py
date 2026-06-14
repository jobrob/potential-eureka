"""Comprehensive tests for heat.engine.rules module."""

from __future__ import annotations

import pytest

from heat.models.cards import Card, CardType, Deck
from heat.models.player_state import PlayerState
from heat.models.track import Corner, Space, Track
from heat.engine.rules import (
    MIN_GEAR,
    MAX_GEAR,
    HAND_SIZE,
    HEAT_POOL_SIZE,
    legal_gear_shifts,
    cards_to_play_count,
    legal_card_plays,
    is_cluttered_hand,
    calculate_speed,
    resolve_stress_card,
    resolve_boost,
    corner_heat_cost,
    corners_crossed,
    check_spin_out,
    spin_out_stress_count,
    slipstream_eligible,
    slipstream_would_cross_finish,
    cooldown_amount,
    adrenaline_eligible,
    calculate_move_position,
    check_finished,
    corner_speed_for_check,
    resolve_blocked_position,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _speed(value: int, idx: int = 0) -> Card:
    """Create a speed card for testing."""
    return Card(CardType.SPEED, value, f"test_spd_{value}_{idx}")


def _heat(idx: int = 0) -> Card:
    """Create a heat card for testing."""
    return Card(CardType.HEAT, 0, f"test_heat_{idx}")


def _stress(idx: int = 0) -> Card:
    """Create a stress card for testing."""
    return Card(CardType.STRESS, 0, f"test_stress_{idx}")


def _upgrade(value: int, idx: int = 0) -> Card:
    """Create an upgrade card for testing."""
    return Card(CardType.UPGRADE, value, f"test_upg_{value}_{idx}")


def _make_track(
    length: int = 10,
    corners: list[Corner] | None = None,
    laps: int = 1,
) -> Track:
    """Create a simple test track."""
    spaces = [Space(i, lanes=2) for i in range(length)]
    if corners is None:
        corners = []
    starts = list(range(min(4, length)))
    return Track("Test", spaces, corners, starts, laps=laps)


def _make_player(
    player_id: int = 0,
    position: int = 0,
    lap: int = 1,
    gear: int = 2,
    heat_pool_size: int = 6,
    finished: bool = False,
) -> PlayerState:
    """Create a player with controlled state for testing."""
    deck = Deck([])
    heat_pool = [
        Card(CardType.HEAT, 0, f"p{player_id}_heat_{i}")
        for i in range(heat_pool_size)
    ]
    player = PlayerState(
        player_id=player_id,
        name=f"Player {player_id}",
        deck=deck,
        heat_pool=heat_pool,
        gear=gear,
        position=position,
        lap=lap,
        finished=finished,
    )
    return player


def _make_deck_with_known_cards(cards: list[Card]) -> Deck:
    """Create a deck with a known draw pile order (last card drawn first)."""
    deck = Deck()
    # Directly set the draw pile to control order
    # Deck.draw() pops from the end, so the last card in the list is drawn first
    deck._draw_pile = list(cards)
    return deck


# ===========================================================================
# Tests: Constants
# ===========================================================================

class TestConstants:
    def test_min_gear(self) -> None:
        assert MIN_GEAR == 1

    def test_max_gear(self) -> None:
        assert MAX_GEAR == 4

    def test_hand_size(self) -> None:
        assert HAND_SIZE == 7

    def test_heat_pool_size(self) -> None:
        assert HEAT_POOL_SIZE == 6


# ===========================================================================
# Tests: legal_gear_shifts
# ===========================================================================

class TestLegalGearShifts:
    def test_from_1_no_heat(self) -> None:
        result = legal_gear_shifts(1, 0)
        assert result == [(1, 0), (2, 0)]

    def test_from_1_with_heat(self) -> None:
        result = legal_gear_shifts(1, 3)
        assert result == [(1, 0), (2, 0), (3, 1)]

    def test_from_2_no_heat(self) -> None:
        result = legal_gear_shifts(2, 0)
        assert result == [(1, 0), (2, 0), (3, 0)]

    def test_from_2_with_heat(self) -> None:
        result = legal_gear_shifts(2, 3)
        assert result == [(1, 0), (2, 0), (3, 0), (4, 1)]

    def test_from_3_with_heat(self) -> None:
        result = legal_gear_shifts(3, 2)
        assert result == [(1, 1), (2, 0), (3, 0), (4, 0)]

    def test_from_3_no_heat(self) -> None:
        result = legal_gear_shifts(3, 0)
        assert result == [(2, 0), (3, 0), (4, 0)]

    def test_from_4_no_heat(self) -> None:
        result = legal_gear_shifts(4, 0)
        assert result == [(3, 0), (4, 0)]

    def test_from_4_with_heat(self) -> None:
        result = legal_gear_shifts(4, 1)
        assert result == [(2, 1), (3, 0), (4, 0)]

    def test_sorted_by_gear(self) -> None:
        result = legal_gear_shifts(2, 5)
        gears = [g for g, _ in result]
        assert gears == sorted(gears)

    def test_always_includes_stay(self) -> None:
        for gear in range(MIN_GEAR, MAX_GEAR + 1):
            result = legal_gear_shifts(gear, 0)
            assert (gear, 0) in result


# ===========================================================================
# Tests: cards_to_play_count
# ===========================================================================

class TestCardsToPlayCount:
    def test_gear_1(self) -> None:
        assert cards_to_play_count(1) == 1

    def test_gear_2(self) -> None:
        assert cards_to_play_count(2) == 2

    def test_gear_3(self) -> None:
        assert cards_to_play_count(3) == 3

    def test_gear_4(self) -> None:
        assert cards_to_play_count(4) == 4


# ===========================================================================
# Tests: legal_card_plays
# ===========================================================================

class TestLegalCardPlays:
    def test_basic_speed_cards_gear_2(self) -> None:
        hand = [_speed(v, i) for i, v in enumerate([1, 2, 3, 4, 5, 6, 1])]
        result = legal_card_plays(hand, 2)
        # C(7, 2) = 21 combinations
        assert len(result) == 21
        # Each combo has exactly 2 cards
        for combo in result:
            assert len(combo) == 2

    def test_basic_speed_cards_gear_1(self) -> None:
        hand = [_speed(v) for v in [1, 2, 3, 4, 5, 6, 1]]
        result = legal_card_plays(hand, 1)
        assert len(result) == 7

    def test_with_heat_in_hand_enough_playable(self) -> None:
        # 5 speed + 2 heat, gear 3 => C(5, 3) = 10 combos (heat excluded)
        hand = [_speed(v, i) for i, v in enumerate([1, 2, 3, 4, 5])] + [
            _heat(0),
            _heat(1),
        ]
        result = legal_card_plays(hand, 3)
        assert len(result) == 10
        # No combo should include a heat card
        for combo in result:
            for card in combo:
                assert card.card_type != CardType.HEAT

    def test_insufficient_playable_forces_heat(self) -> None:
        # 2 speed + 5 heat, gear 3 => must play 2 speed + 1 heat
        hand = [_speed(1, 0), _speed(2, 1)] + [_heat(i) for i in range(5)]
        result = legal_card_plays(hand, 3)
        # All playable (2 speed) + choose 1 from 5 heat = C(5, 1) = 5 combos
        assert len(result) == 5
        for combo in result:
            assert len(combo) == 3
            speed_count = sum(1 for c in combo if c.card_type == CardType.SPEED)
            heat_count = sum(1 for c in combo if c.card_type == CardType.HEAT)
            assert speed_count == 2
            assert heat_count == 1

    def test_stress_cards_are_playable(self) -> None:
        # 3 speed + 2 stress + 2 heat, gear 2 => playable = 5
        hand = (
            [_speed(v, i) for i, v in enumerate([1, 2, 3])]
            + [_stress(0), _stress(1)]
            + [_heat(0), _heat(1)]
        )
        result = legal_card_plays(hand, 2)
        # C(5, 2) = 10 combos (5 playable cards: 3 speed + 2 stress)
        assert len(result) == 10
        # No heat cards should appear
        for combo in result:
            for card in combo:
                assert card.card_type != CardType.HEAT

    def test_upgrade_cards_are_playable(self) -> None:
        # 3 speed + 1 upgrade + 3 heat, gear 2 => 4 playable
        hand = (
            [_speed(v, i) for i, v in enumerate([1, 2, 3])]
            + [_upgrade(5)]
            + [_heat(i) for i in range(3)]
        )
        result = legal_card_plays(hand, 2)
        # C(4, 2) = 6 combos
        assert len(result) == 6

    def test_all_heat_in_hand(self) -> None:
        # 7 heat cards, gear 2 => 0 playable, must use 2 heat
        hand = [_heat(i) for i in range(7)]
        result = legal_card_plays(hand, 2)
        # C(7, 2) = 21 combos (all heat to fill)
        assert len(result) == 21
        for combo in result:
            assert len(combo) == 2
            assert all(c.card_type == CardType.HEAT for c in combo)

    def test_exact_playable_count(self) -> None:
        # Exactly 3 playable cards, gear 3 => 1 combo
        hand = [_speed(1, 0), _speed(2, 1), _stress(0)] + [_heat(i) for i in range(4)]
        result = legal_card_plays(hand, 3)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_upgrade_zero_value_playable(self) -> None:
        # Upgrade with value 0 should still be playable
        hand = [_upgrade(0, 0)] + [_speed(v, i) for i, v in enumerate([1, 2, 3, 4, 5, 6])]
        result = legal_card_plays(hand, 1)
        # 7 playable cards, gear 1 => C(7, 1) = 7 combos
        assert len(result) == 7
        # The upgrade(0) card should appear in exactly one combo
        upg_combos = [c for c in result if any(card.card_type == CardType.UPGRADE for card in c)]
        assert len(upg_combos) == 1


# ===========================================================================
# Tests: is_cluttered_hand
# ===========================================================================

class TestIsClutteredHand:
    def test_cluttered_true(self) -> None:
        # 5 heat + 2 speed in hand, gear 3 => 2 playable < 3 needed => True
        hand = [_heat(i) for i in range(5)] + [_speed(1, 0), _speed(2, 1)]
        assert is_cluttered_hand(hand, 3) is True

    def test_cluttered_false(self) -> None:
        # 2 heat + 5 speed in hand, gear 3 => 5 playable >= 3 needed => False
        hand = [_heat(0), _heat(1)] + [_speed(v, i) for i, v in enumerate([1, 2, 3, 4, 5])]
        assert is_cluttered_hand(hand, 3) is False

    def test_exact_boundary(self) -> None:
        # Exactly enough playable cards => not cluttered
        hand = [_heat(i) for i in range(4)] + [_speed(v, i) for i, v in enumerate([1, 2, 3])]
        assert is_cluttered_hand(hand, 3) is False

    def test_one_short(self) -> None:
        # One fewer playable card than needed
        hand = [_heat(i) for i in range(5)] + [_speed(1, 0), _speed(2, 1)]
        assert is_cluttered_hand(hand, 3) is True

    def test_gear_1_with_all_heat(self) -> None:
        # All heat cards, gear 1 => 0 playable < 1 needed => True
        hand = [_heat(i) for i in range(7)]
        assert is_cluttered_hand(hand, 1) is True

    def test_stress_counts_as_playable(self) -> None:
        # 5 heat + 1 speed + 1 stress, gear 2 => 2 playable >= 2 needed => False
        hand = [_heat(i) for i in range(5)] + [_speed(1), _stress(0)]
        assert is_cluttered_hand(hand, 2) is False


# ===========================================================================
# Tests: calculate_speed
# ===========================================================================

class TestCalculateSpeed:
    def test_speed_cards_only(self) -> None:
        cards = (_speed(3), _speed(5))
        assert calculate_speed(cards) == 8

    def test_stress_cards_count_as_zero(self) -> None:
        cards = (_speed(4), _stress(0))
        assert calculate_speed(cards) == 4

    def test_upgrade_cards_use_face_value(self) -> None:
        cards = (_speed(3), _upgrade(5))
        assert calculate_speed(cards) == 8

    def test_upgrade_zero_value(self) -> None:
        cards = (_speed(2), _upgrade(0))
        assert calculate_speed(cards) == 2

    def test_empty_cards(self) -> None:
        assert calculate_speed(()) == 0

    def test_all_stress(self) -> None:
        cards = (_stress(0), _stress(1))
        assert calculate_speed(cards) == 0

    def test_mixed(self) -> None:
        cards = (_speed(1), _speed(4), _stress(0), _upgrade(5))
        assert calculate_speed(cards) == 10


# ===========================================================================
# Tests: resolve_stress_card
# ===========================================================================

class TestResolveStressCard:
    def test_basic_on_top(self) -> None:
        # Speed card is on top of deck (drawn first)
        speed = _speed(4)
        deck = _make_deck_with_known_cards([speed])
        value, flipped = resolve_stress_card(deck)
        assert value == 4
        assert flipped == [speed]

    def test_skips_heat_cards(self) -> None:
        # Heat on top, then speed underneath
        heat = _heat(0)
        speed = _speed(3)
        # draw() pops from end, so speed is drawn first... we need heat first
        # [heat, speed] -> pop() gives speed first. We want heat first.
        # So: [speed, heat] -> pop() gives heat, then speed
        deck = _make_deck_with_known_cards([speed, heat])
        value, flipped = resolve_stress_card(deck)
        assert value == 3
        assert len(flipped) == 2
        assert flipped[0] == heat  # heat drawn first and discarded
        assert flipped[1] == speed  # speed drawn second, stops flipping

    def test_skips_upgrade_cards(self) -> None:
        upgrade = _upgrade(5)
        speed = _speed(2)
        deck = _make_deck_with_known_cards([speed, upgrade])
        value, flipped = resolve_stress_card(deck)
        assert value == 2
        assert len(flipped) == 2
        assert flipped[0] == upgrade
        assert flipped[1] == speed

    def test_skips_stress_cards(self) -> None:
        stress = _stress(0)
        speed = _speed(6)
        deck = _make_deck_with_known_cards([speed, stress])
        value, flipped = resolve_stress_card(deck)
        assert value == 6
        assert len(flipped) == 2
        assert flipped[0] == stress
        assert flipped[1] == speed

    def test_skips_multiple_non_basic(self) -> None:
        heat = _heat(0)
        upgrade = _upgrade(0, 1)
        stress = _stress(2)
        speed = _speed(1)
        deck = _make_deck_with_known_cards([speed, stress, upgrade, heat])
        value, flipped = resolve_stress_card(deck)
        assert value == 1
        assert len(flipped) == 4

    def test_empty_deck(self) -> None:
        deck = _make_deck_with_known_cards([])
        value, flipped = resolve_stress_card(deck)
        assert value == 0
        assert flipped == []

    def test_finds_speed_in_discard_after_draw_exhausted(self) -> None:
        # Speed card is the only card in the discard pile, draw pile is empty.
        # draw(1) will trigger reshuffle and find the speed card.
        speed = _speed(5)
        deck = _make_deck_with_known_cards([])
        deck._discard_pile = [speed]
        value, flipped = resolve_stress_card(deck)
        assert value == 5
        assert flipped == [speed]


# ===========================================================================
# Tests: resolve_boost
# ===========================================================================

class TestResolveBoost:
    def test_same_as_stress_resolution(self) -> None:
        speed = _speed(3)
        deck = _make_deck_with_known_cards([speed])
        value, flipped = resolve_boost(deck)
        assert value == 3
        assert flipped == [speed]

    def test_skips_non_basic(self) -> None:
        heat = _heat(0)
        speed = _speed(4)
        deck = _make_deck_with_known_cards([speed, heat])
        value, flipped = resolve_boost(deck)
        assert value == 4
        assert len(flipped) == 2

    def test_empty_deck(self) -> None:
        deck = _make_deck_with_known_cards([])
        value, flipped = resolve_boost(deck)
        assert value == 0
        assert flipped == []


# ===========================================================================
# Tests: corner_heat_cost
# ===========================================================================

class TestCornerHeatCost:
    def test_under_limit(self) -> None:
        corner = Corner(start=4, end=6, speed_limit=5)
        assert corner_heat_cost(3, corner) == 0

    def test_at_limit(self) -> None:
        corner = Corner(start=4, end=6, speed_limit=5)
        assert corner_heat_cost(5, corner) == 0

    def test_over_limit_by_1(self) -> None:
        corner = Corner(start=4, end=6, speed_limit=5)
        assert corner_heat_cost(6, corner) == 1

    def test_over_limit_by_3(self) -> None:
        corner = Corner(start=4, end=6, speed_limit=3)
        assert corner_heat_cost(6, corner) == 3

    def test_speed_zero(self) -> None:
        corner = Corner(start=0, end=2, speed_limit=3)
        assert corner_heat_cost(0, corner) == 0


# ===========================================================================
# Tests: corners_crossed
# ===========================================================================

class TestCornersCrossed:
    def test_no_corners_on_track(self) -> None:
        track = _make_track(10, corners=[])
        result = corners_crossed(0, 5, track)
        assert result == []

    def test_no_corners_crossed(self) -> None:
        # Corner at 4-6, move from 0 to 3 (doesn't reach corner)
        track = _make_track(10, corners=[Corner(4, 6, 3)])
        result = corners_crossed(0, 3, track)
        assert result == []

    def test_one_corner_crossed(self) -> None:
        track = _make_track(10, corners=[Corner(4, 6, 3)])
        result = corners_crossed(2, 7, track)
        assert len(result) == 1
        assert result[0] == Corner(4, 6, 3)

    def test_move_into_corner(self) -> None:
        # End position is inside the corner
        track = _make_track(10, corners=[Corner(4, 6, 3)])
        result = corners_crossed(2, 5, track)
        assert len(result) == 1

    def test_move_starting_in_corner(self) -> None:
        # Start inside corner, end outside. start_pos excluded, so
        # positions 5, 6, 7 are traversed. Corner 4-6 intersects at 5, 6.
        track = _make_track(10, corners=[Corner(4, 6, 3)])
        result = corners_crossed(4, 7, track)
        assert len(result) == 1

    def test_multiple_corners(self) -> None:
        track = _make_track(
            20,
            corners=[Corner(3, 5, 3), Corner(10, 12, 4)],
        )
        result = corners_crossed(1, 15, track)
        assert len(result) == 2

    def test_wraparound(self) -> None:
        # Track length 10, corner at 8-9, move from 7 to 1 (wraps)
        track = _make_track(10, corners=[Corner(8, 9, 3)])
        result = corners_crossed(7, 1, track)
        assert len(result) == 1
        assert result[0] == Corner(8, 9, 3)

    def test_wraparound_corner_at_start(self) -> None:
        # Track length 10, corner at 0-2, move from 8 to 3 (wraps)
        track = _make_track(10, corners=[Corner(0, 2, 4)])
        result = corners_crossed(8, 3, track)
        assert len(result) == 1
        assert result[0] == Corner(0, 2, 4)

    def test_no_movement(self) -> None:
        track = _make_track(10, corners=[Corner(4, 6, 3)])
        result = corners_crossed(5, 5, track)
        assert result == []

    def test_start_pos_excluded(self) -> None:
        # Player is at position 4 (corner start). Move to 4 = no movement.
        # But if they "move" from 3 to 4, position 4 IS traversed.
        track = _make_track(10, corners=[Corner(4, 6, 3)])
        result = corners_crossed(3, 4, track)
        assert len(result) == 1

    def test_start_pos_in_corner_not_counted_if_no_movement(self) -> None:
        # Already at corner position, no movement
        track = _make_track(10, corners=[Corner(4, 6, 3)])
        result = corners_crossed(4, 4, track)
        assert result == []


# ===========================================================================
# Tests: check_spin_out
# ===========================================================================

class TestCheckSpinOut:
    def test_spin_out_when_insufficient_heat(self) -> None:
        player = _make_player(heat_pool_size=2)
        assert check_spin_out(player, 3) is True

    def test_no_spin_out_when_enough_heat(self) -> None:
        player = _make_player(heat_pool_size=5)
        assert check_spin_out(player, 3) is False

    def test_no_spin_out_at_exact_heat(self) -> None:
        player = _make_player(heat_pool_size=3)
        assert check_spin_out(player, 3) is False

    def test_spin_out_with_zero_heat(self) -> None:
        player = _make_player(heat_pool_size=0)
        assert check_spin_out(player, 1) is True

    def test_no_spin_out_with_zero_cost(self) -> None:
        player = _make_player(heat_pool_size=0)
        assert check_spin_out(player, 0) is False


# ===========================================================================
# Tests: spin_out_stress_count
# ===========================================================================

class TestSpinOutStressCount:
    def test_gear_1(self) -> None:
        assert spin_out_stress_count(1) == 1

    def test_gear_2(self) -> None:
        assert spin_out_stress_count(2) == 1

    def test_gear_3(self) -> None:
        assert spin_out_stress_count(3) == 2

    def test_gear_4(self) -> None:
        assert spin_out_stress_count(4) == 2


# ===========================================================================
# Tests: slipstream_eligible
# ===========================================================================

class TestSlipstreamEligible:
    def test_car_1_ahead(self) -> None:
        track = _make_track(30)
        player = _make_player(0, position=5, lap=1)
        other = _make_player(1, position=6, lap=1)
        assert slipstream_eligible(player, [player, other], track) is True

    def test_car_2_ahead(self) -> None:
        track = _make_track(30)
        player = _make_player(0, position=5, lap=1)
        other = _make_player(1, position=7, lap=1)
        assert slipstream_eligible(player, [player, other], track) is True

    def test_car_3_ahead_not_eligible(self) -> None:
        track = _make_track(30)
        player = _make_player(0, position=5, lap=1)
        other = _make_player(1, position=8, lap=1)
        assert slipstream_eligible(player, [player, other], track) is False

    def test_no_car_ahead(self) -> None:
        track = _make_track(30)
        player = _make_player(0, position=10, lap=1)
        other = _make_player(1, position=5, lap=1)  # Behind, not ahead
        assert slipstream_eligible(player, [player, other], track) is False

    def test_cannot_cross_finish(self) -> None:
        track = _make_track(30, laps=1)
        # On final lap, position 28, +2 would be 30 >= 30 => crosses finish
        player = _make_player(0, position=28, lap=1)
        other = _make_player(1, position=29, lap=1)
        assert slipstream_eligible(player, [player, other], track) is False

    def test_finished_player_not_eligible(self) -> None:
        track = _make_track(30)
        player = _make_player(0, position=5, lap=1, finished=True)
        other = _make_player(1, position=6, lap=1)
        assert slipstream_eligible(player, [player, other], track) is False

    def test_finished_other_not_counted(self) -> None:
        track = _make_track(30)
        player = _make_player(0, position=5, lap=1)
        other = _make_player(1, position=6, lap=1, finished=True)
        assert slipstream_eligible(player, [player, other], track) is False

    def test_wrap_around_eligibility(self) -> None:
        track = _make_track(30, laps=2)
        # Player at position 29, other at position 0 (1 ahead with wrap)
        player = _make_player(0, position=29, lap=1)
        other = _make_player(1, position=0, lap=1)
        # Distance = (0 - 29) % 30 = 1
        assert slipstream_eligible(player, [player, other], track) is True


# ===========================================================================
# Tests: slipstream_would_cross_finish
# ===========================================================================

class TestSlipstreamWouldCrossFinish:
    def test_would_cross_on_final_lap(self) -> None:
        track = _make_track(30, laps=1)
        player = _make_player(position=28, lap=1)
        assert slipstream_would_cross_finish(player, track) is True

    def test_would_not_cross_not_final_lap(self) -> None:
        track = _make_track(30, laps=2)
        player = _make_player(position=28, lap=1)
        assert slipstream_would_cross_finish(player, track) is False

    def test_already_finished(self) -> None:
        track = _make_track(30, laps=1)
        player = _make_player(position=5, lap=1, finished=True)
        assert slipstream_would_cross_finish(player, track) is True

    def test_safe_position_final_lap(self) -> None:
        track = _make_track(30, laps=1)
        player = _make_player(position=20, lap=1)
        assert slipstream_would_cross_finish(player, track) is False

    def test_exact_boundary(self) -> None:
        # position + 2 == track.length => crosses
        track = _make_track(30, laps=1)
        player = _make_player(position=28, lap=1)
        assert slipstream_would_cross_finish(player, track) is True

    def test_one_before_boundary(self) -> None:
        # position + 2 == track.length - 1 => does not cross
        track = _make_track(30, laps=1)
        player = _make_player(position=27, lap=1)
        assert slipstream_would_cross_finish(player, track) is False


# ===========================================================================
# Tests: cooldown_amount
# ===========================================================================

class TestCooldownAmount:
    def test_gear_1(self) -> None:
        assert cooldown_amount(1) == 3

    def test_gear_2(self) -> None:
        assert cooldown_amount(2) == 1

    def test_gear_3(self) -> None:
        assert cooldown_amount(3) == 0

    def test_gear_4(self) -> None:
        assert cooldown_amount(4) == 0


# ===========================================================================
# Tests: adrenaline_eligible
# ===========================================================================

class TestAdrenalineEligible:
    def test_last_place_2_players(self) -> None:
        p1 = _make_player(0, position=10, lap=1)
        p2 = _make_player(1, position=5, lap=1)
        players = [p1, p2]
        # p2 is in last place (position 5 < position 10)
        assert adrenaline_eligible(p2, players, starting_player_count=2) is True
        assert adrenaline_eligible(p1, players, starting_player_count=2) is False

    def test_last_2_places_5_players(self) -> None:
        players = [
            _make_player(0, position=20, lap=1),
            _make_player(1, position=15, lap=1),
            _make_player(2, position=10, lap=1),
            _make_player(3, position=5, lap=1),
            _make_player(4, position=2, lap=1),
        ]
        # With 5 starters, last 2 get adrenaline (positions 2 and 5)
        assert adrenaline_eligible(players[4], players, 5) is True  # pos 2
        assert adrenaline_eligible(players[3], players, 5) is True  # pos 5
        assert adrenaline_eligible(players[2], players, 5) is False  # pos 10
        assert adrenaline_eligible(players[0], players, 5) is False  # pos 20

    def test_starting_count_matters_not_current(self) -> None:
        # 5 started, 3 finished, 2 remain active. Last 2 should get adrenaline
        # since starting count is 5 (>= 5 threshold).
        p1 = _make_player(0, position=10, lap=1)
        p2 = _make_player(1, position=5, lap=1)
        p3 = _make_player(2, position=0, lap=1, finished=True)
        p4 = _make_player(3, position=0, lap=1, finished=True)
        p5 = _make_player(4, position=0, lap=1, finished=True)
        players = [p1, p2, p3, p4, p5]
        # Both remaining active players are in "last 2" since only 2 active
        assert adrenaline_eligible(p1, players, starting_player_count=5) is True
        assert adrenaline_eligible(p2, players, starting_player_count=5) is True

    def test_finished_player_not_eligible(self) -> None:
        p1 = _make_player(0, position=10, lap=1, finished=True)
        p2 = _make_player(1, position=5, lap=1)
        p3 = _make_player(2, position=3, lap=1)
        assert adrenaline_eligible(p1, [p1, p2, p3], 3) is False

    def test_leader_not_eligible(self) -> None:
        p1 = _make_player(0, position=20, lap=1)
        p2 = _make_player(1, position=5, lap=1)
        assert adrenaline_eligible(p1, [p1, p2], 2) is False

    def test_single_active_player_not_eligible(self) -> None:
        p1 = _make_player(0, position=5, lap=1)
        p2 = _make_player(1, position=0, finished=True)
        assert adrenaline_eligible(p1, [p1, p2], 2) is False

    def test_lap_difference_matters(self) -> None:
        # p1 on lap 2, p2 on lap 1 -- p2 is behind
        p1 = _make_player(0, position=5, lap=2)
        p2 = _make_player(1, position=25, lap=1)
        assert adrenaline_eligible(p2, [p1, p2], 2) is True
        assert adrenaline_eligible(p1, [p1, p2], 2) is False

    def test_3_players_started_only_last_gets_adrenaline(self) -> None:
        p1 = _make_player(0, position=15, lap=1)
        p2 = _make_player(1, position=10, lap=1)
        p3 = _make_player(2, position=5, lap=1)
        players = [p1, p2, p3]
        assert adrenaline_eligible(p3, players, 3) is True   # last place
        assert adrenaline_eligible(p2, players, 3) is False   # middle
        assert adrenaline_eligible(p1, players, 3) is False   # leader


# ===========================================================================
# Tests: calculate_move_position
# ===========================================================================

class TestCalculateMovePosition:
    def test_normal_move(self) -> None:
        track = _make_track(30)
        new_pos, crossed = calculate_move_position(5, 3, track, 1)
        assert new_pos == 8
        assert crossed is False

    def test_wrap_around(self) -> None:
        track = _make_track(30)
        new_pos, crossed = calculate_move_position(28, 5, track, 1)
        assert new_pos == 3
        assert crossed is True

    def test_exact_finish(self) -> None:
        track = _make_track(30)
        new_pos, crossed = calculate_move_position(25, 5, track, 1)
        assert new_pos == 0
        assert crossed is True

    def test_no_movement(self) -> None:
        track = _make_track(30)
        new_pos, crossed = calculate_move_position(10, 0, track, 1)
        assert new_pos == 10
        assert crossed is False

    def test_large_speed(self) -> None:
        track = _make_track(10)
        new_pos, crossed = calculate_move_position(5, 12, track, 1)
        assert new_pos == 7  # (5 + 12) % 10 = 7
        assert crossed is True


# ===========================================================================
# Tests: check_finished
# ===========================================================================

class TestCheckFinished:
    def test_finished_when_lap_exceeds_total(self) -> None:
        track = _make_track(30, laps=2)
        player = _make_player(lap=3)
        assert check_finished(player, track) is True

    def test_not_finished_on_final_lap(self) -> None:
        track = _make_track(30, laps=2)
        player = _make_player(lap=2)
        assert check_finished(player, track) is False

    def test_not_finished_first_lap(self) -> None:
        track = _make_track(30, laps=1)
        player = _make_player(lap=1)
        assert check_finished(player, track) is False

    def test_finished_single_lap_track(self) -> None:
        track = _make_track(30, laps=1)
        player = _make_player(lap=2)
        assert check_finished(player, track) is True


# ===========================================================================
# Tests: corner_speed_for_check
# ===========================================================================

class TestCornerSpeedForCheck:
    def test_cards_only(self) -> None:
        player = _make_player()
        player.speed_from_cards = 6
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        assert corner_speed_for_check(player) == 6

    def test_includes_boost(self) -> None:
        player = _make_player()
        player.speed_from_cards = 4
        player.speed_from_boost = 3
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        assert corner_speed_for_check(player) == 7

    def test_includes_adrenaline(self) -> None:
        player = _make_player()
        player.speed_from_cards = 4
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 1
        player.slipstream_moved = 0
        assert corner_speed_for_check(player) == 5

    def test_excludes_slipstream(self) -> None:
        player = _make_player()
        player.speed_from_cards = 4
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 2
        assert corner_speed_for_check(player) == 4

    def test_all_sources(self) -> None:
        player = _make_player()
        player.speed_from_cards = 5
        player.speed_from_boost = 3
        player.speed_from_adrenaline = 1
        player.slipstream_moved = 2
        # Should be 5 + 3 + 1 = 9 (slipstream excluded)
        assert corner_speed_for_check(player) == 9

    def test_zero_speed(self) -> None:
        player = _make_player()
        player.speed_from_cards = 0
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        assert corner_speed_for_check(player) == 0


# ===========================================================================
# 19. Collision / blocking resolution
# ===========================================================================

class TestResolveBlockedPosition:
    def test_no_blocking_space_has_room(self) -> None:
        """Target space has room — player lands there."""
        track = _make_track(length=10)  # all spaces have 2 lanes
        p0 = _make_player(0, position=3)
        p1 = _make_player(1, position=5)
        # p0 moves to position 5 where p1 is, but 2 lanes available
        result = resolve_blocked_position(5, track, [p0, p1], 0)
        assert result == 5

    def test_blocked_single_lane(self) -> None:
        """Target space is full (1 lane, 1 car) — pushed back."""
        spaces = [Space(i, lanes=1) for i in range(10)]
        track = Track("Test", spaces, [], [0, 1], laps=1)
        p0 = _make_player(0, position=2)
        p1 = _make_player(1, position=5)
        # p0 tries to land on 5 where p1 is, single lane
        result = resolve_blocked_position(5, track, [p0, p1], 0)
        assert result == 4  # pushed back 1 space

    def test_blocked_multiple_cars(self) -> None:
        """Target space full, space behind also full — pushed further back."""
        spaces = [Space(i, lanes=1) for i in range(10)]
        track = Track("Test", spaces, [], [0, 1, 2], laps=1)
        p0 = _make_player(0, position=2)
        p1 = _make_player(1, position=5)
        p2 = _make_player(2, position=4)
        # p0 tries position 5 (full), position 4 (full), lands at 3
        result = resolve_blocked_position(5, track, [p0, p1, p2], 0)
        assert result == 3

    def test_two_lanes_fits_two_cars(self) -> None:
        """Two-lane space can hold 2 cars."""
        track = _make_track(length=10)  # 2 lanes each
        p0 = _make_player(0, position=0)
        p1 = _make_player(1, position=5)
        p2 = _make_player(2, position=5)
        # p0 tries position 5 where two cars are, but only 2 lanes
        result = resolve_blocked_position(5, track, [p0, p1, p2], 0)
        assert result == 4  # full, pushed back

    def test_finished_players_dont_block(self) -> None:
        """Finished players are removed from track — don't count."""
        spaces = [Space(i, lanes=1) for i in range(10)]
        track = Track("Test", spaces, [], [0, 1], laps=1)
        p0 = _make_player(0, position=2)
        p1 = _make_player(1, position=5, finished=True)
        result = resolve_blocked_position(5, track, [p0, p1], 0)
        assert result == 5  # finished player doesn't block

    def test_self_not_counted(self) -> None:
        """Moving player's current position doesn't block themselves."""
        spaces = [Space(i, lanes=1) for i in range(10)]
        track = Track("Test", spaces, [], [0], laps=1)
        p0 = _make_player(0, position=5)
        result = resolve_blocked_position(5, track, [p0], 0)
        assert result == 5
