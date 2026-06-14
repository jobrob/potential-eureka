"""Comprehensive tests for heat.engine.phases module."""

from __future__ import annotations

import pytest

from heat.models.cards import Card, CardType, Deck
from heat.models.player_state import PlayerState
from heat.models.track import Corner, Space, Track
from heat.models.game_state import GameState, Phase
from heat.engine.phases import (
    ReactDecision,
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


def _make_deck_with_known_cards(cards: list[Card]) -> Deck:
    """Create a deck with a known draw pile order (last card drawn first)."""
    deck = Deck()
    deck._draw_pile = list(cards)
    return deck


def _make_player(
    player_id: int = 0,
    position: int = 0,
    lap: int = 1,
    gear: int = 2,
    heat_pool_size: int = 6,
    finished: bool = False,
    hand: list[Card] | None = None,
    deck: Deck | None = None,
    spun_out: bool = False,
) -> PlayerState:
    """Create a player with controlled state for testing."""
    if deck is None:
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
        spun_out=spun_out,
    )
    if hand is not None:
        player.hand = list(hand)
    return player


def _make_game_state(
    track: Track | None = None,
    players: list[PlayerState] | None = None,
    num_players: int = 2,
    logging_enabled: bool = True,
) -> GameState:
    """Create a GameState with manually constructed players."""
    if track is None:
        track = _make_track()
    if players is None:
        players = [_make_player(i) for i in range(num_players)]
    state = GameState(
        track=track,
        players=players,
        logging_enabled=logging_enabled,
        starting_player_count=len(players),
    )
    return state


# ===========================================================================
# Tests: ReactDecision
# ===========================================================================


class TestReactDecision:
    def test_defaults(self) -> None:
        rd = ReactDecision()
        assert rd.cooldown_count == 0
        assert rd.use_boost is False
        assert rd.use_adrenaline_speed is False
        assert rd.use_adrenaline_cooldown is False

    def test_custom_values(self) -> None:
        rd = ReactDecision(
            cooldown_count=2,
            use_boost=True,
            use_adrenaline_speed=True,
            use_adrenaline_cooldown=True,
        )
        assert rd.cooldown_count == 2
        assert rd.use_boost is True


# ===========================================================================
# Tests: phase_shift_gears
# ===========================================================================


class TestPhaseShiftGears:
    def test_valid_shift_applied(self) -> None:
        """Player shifts from gear 2 to gear 3 (free)."""
        track = _make_track()
        player = _make_player(0, gear=2)
        state = _make_game_state(track, [player])

        events = phase_shift_gears(state, {0: (3, 0)})
        assert player.gear == 3
        assert len(events) == 1

    def test_stay_in_current_gear(self) -> None:
        """Player stays in gear 2."""
        player = _make_player(0, gear=2)
        state = _make_game_state(players=[player])

        phase_shift_gears(state, {0: (2, 0)})
        assert player.gear == 2

    def test_illegal_shift_raises(self) -> None:
        """Shifting from gear 2 to gear 4 with no heat should raise."""
        player = _make_player(0, gear=2, heat_pool_size=0)
        state = _make_game_state(players=[player])

        with pytest.raises(ValueError):
            phase_shift_gears(state, {0: (4, 1)})

    def test_illegal_gear_value_raises(self) -> None:
        """Shifting to gear 5 should raise."""
        player = _make_player(0, gear=4)
        state = _make_game_state(players=[player])

        with pytest.raises(ValueError):
            phase_shift_gears(state, {0: (5, 0)})

    def test_spun_out_forced_to_gear_1(self) -> None:
        """Spun out player must shift to gear 1."""
        player = _make_player(0, gear=3, spun_out=True)
        state = _make_game_state(players=[player])

        events = phase_shift_gears(state, {0: (1, 0)})
        assert player.gear == 1
        assert len(events) == 1

    def test_spun_out_wrong_gear_raises(self) -> None:
        """Spun out player trying to shift to non-1 gear should raise."""
        player = _make_player(0, gear=3, spun_out=True)
        state = _make_game_state(players=[player])

        with pytest.raises(ValueError):
            phase_shift_gears(state, {0: (2, 0)})

    def test_plus_2_shift_pays_heat(self) -> None:
        """Shifting +2 gears costs 1 heat."""
        player = _make_player(0, gear=2, heat_pool_size=6)
        state = _make_game_state(players=[player])

        phase_shift_gears(state, {0: (4, 1)})
        assert player.gear == 4
        assert player.heat_available == 5  # Paid 1 heat

    def test_minus_2_shift_pays_heat(self) -> None:
        """Shifting -2 gears costs 1 heat."""
        player = _make_player(0, gear=3, heat_pool_size=6)
        state = _make_game_state(players=[player])

        phase_shift_gears(state, {0: (1, 1)})
        assert player.gear == 1
        assert player.heat_available == 5  # Paid 1 heat

    def test_finished_player_skipped(self) -> None:
        """Finished players should be skipped."""
        player = _make_player(0, gear=2, finished=True)
        state = _make_game_state(players=[player])

        events = phase_shift_gears(state, {0: (3, 0)})
        # Gear should not change
        assert player.gear == 2
        assert len(events) == 0

    def test_multiple_players(self) -> None:
        """Multiple players shift gears simultaneously."""
        p1 = _make_player(0, gear=1)
        p2 = _make_player(1, gear=3)
        state = _make_game_state(players=[p1, p2])

        phase_shift_gears(state, {0: (2, 0), 1: (4, 0)})
        assert p1.gear == 2
        assert p2.gear == 4


# ===========================================================================
# Tests: phase_play_cards
# ===========================================================================


class TestPhasePlayCards:
    def test_cards_removed_from_hand(self) -> None:
        """Played cards should be removed from hand."""
        card1 = _speed(3, 0)
        card2 = _speed(4, 1)
        hand = [card1, card2, _speed(1, 2), _speed(2, 3),
                _speed(5, 4), _speed(6, 5), _speed(1, 6)]
        player = _make_player(0, gear=2, hand=hand)
        state = _make_game_state(players=[player])

        phase_play_cards(state, {0: (card1, card2)})
        assert card1 not in player.hand
        assert card2 not in player.hand
        assert len(player.hand) == 5

    def test_stored_in_cards_played(self) -> None:
        """Played cards should be stored in cards_played."""
        card1 = _speed(3, 0)
        card2 = _speed(4, 1)
        hand = [card1, card2, _speed(1, 2), _speed(2, 3),
                _speed(5, 4), _speed(6, 5), _speed(1, 6)]
        player = _make_player(0, gear=2, hand=hand)
        state = _make_game_state(players=[player])

        phase_play_cards(state, {0: (card1, card2)})
        assert card1 in player.cards_played
        assert card2 in player.cards_played

    def test_illegal_combo_raises(self) -> None:
        """Playing a card not in hand should raise."""
        card1 = _speed(3, 0)
        card2 = _speed(4, 1)
        fake_card = _speed(99, 99)
        hand = [card1, card2, _speed(1, 2), _speed(2, 3),
                _speed(5, 4), _speed(6, 5), _speed(1, 6)]
        player = _make_player(0, gear=2, hand=hand)
        state = _make_game_state(players=[player])

        with pytest.raises(ValueError):
            phase_play_cards(state, {0: (card1, fake_card)})

    def test_cluttered_hand_detected(self) -> None:
        """Cluttered hand should be flagged."""
        # 5 heat + 2 speed, gear 3 => cluttered
        hand = [_heat(i) for i in range(5)] + [_speed(1, 0), _speed(2, 1)]
        player = _make_player(0, gear=3, hand=hand)
        state = _make_game_state(players=[player])

        # Must play 2 speed + 1 heat (forced)
        cards_to_play = (hand[5], hand[6], hand[0])  # 2 speed + 1 heat
        phase_play_cards(state, {0: cards_to_play})
        assert player.cluttered is True

    def test_not_cluttered(self) -> None:
        """Non-cluttered hand should not be flagged."""
        hand = [_speed(v, i) for i, v in enumerate([1, 2, 3, 4, 5, 6, 1])]
        player = _make_player(0, gear=2, hand=hand)
        state = _make_game_state(players=[player])

        phase_play_cards(state, {0: (hand[0], hand[1])})
        assert player.cluttered is False

    def test_finished_player_skipped(self) -> None:
        """Finished players should be skipped."""
        hand = [_speed(v, i) for i, v in enumerate([1, 2, 3, 4, 5, 6, 1])]
        player = _make_player(0, gear=2, hand=hand, finished=True)
        state = _make_game_state(players=[player])

        events = phase_play_cards(state, {0: (hand[0], hand[1])})
        assert len(player.cards_played) == 0
        assert len(events) == 0


# ===========================================================================
# Tests: step_reveal_and_move
# ===========================================================================


class TestStepRevealAndMove:
    def test_position_updated(self) -> None:
        """Player should move forward by the sum of card values."""
        track = _make_track(length=20)
        player = _make_player(0, position=2, lap=1, gear=2)
        player.cards_played = [_speed(3, 0), _speed(4, 1)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.position == 9  # 2 + 3 + 4 = 9
        assert player.speed_from_cards == 7

    def test_speed_from_cards_set(self) -> None:
        """speed_from_cards should be set to total card speed."""
        track = _make_track(length=30)
        player = _make_player(0, position=0, lap=1)
        player.cards_played = [_speed(2, 0), _speed(5, 1)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.speed_from_cards == 7

    def test_stress_resolved_with_loop(self) -> None:
        """Stress card should flip until a SPEED card is found."""
        track = _make_track(length=30)
        # Deck has: heat on top (drawn first), then speed 4
        speed_card = _speed(4, 99)
        heat_card = _heat(99)
        deck = _make_deck_with_known_cards([speed_card, heat_card])
        player = _make_player(0, position=0, lap=1, deck=deck)
        player.cards_played = [_stress(0)]  # stress resolves to 4
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        # Stress card value 0, but resolved to speed 4
        assert player.speed_from_cards == 4
        assert player.position == 4

    def test_stress_with_speed_card_on_top(self) -> None:
        """Stress resolves immediately if SPEED card is on top."""
        track = _make_track(length=30)
        speed_card = _speed(3, 99)
        deck = _make_deck_with_known_cards([speed_card])
        player = _make_player(0, position=0, lap=1, deck=deck)
        player.cards_played = [_stress(0)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.speed_from_cards == 3
        assert player.position == 3

    def test_multiple_stress_cards(self) -> None:
        """Multiple stress cards should each be resolved."""
        track = _make_track(length=30)
        s1 = _speed(2, 91)
        s2 = _speed(3, 92)
        # Two stress cards, each resolves to a speed card
        deck = _make_deck_with_known_cards([s2, s1])
        player = _make_player(0, position=0, lap=1, deck=deck)
        player.cards_played = [_stress(0), _stress(1)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        # Both stress resolved: 2 + 3 = 5
        assert player.speed_from_cards == 5
        assert player.position == 5

    def test_lap_incremented(self) -> None:
        """Crossing the finish line should increment lap."""
        track = _make_track(length=10, laps=2)
        player = _make_player(0, position=8, lap=1)
        player.cards_played = [_speed(4, 0)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.lap == 2
        assert player.position == 2  # (8 + 4) % 10 = 2

    def test_finish_detected(self) -> None:
        """Player should be marked as finished on final lap crossing."""
        track = _make_track(length=10, laps=1)
        player = _make_player(0, position=7, lap=1)
        player.cards_played = [_speed(5, 0)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.finished is True
        assert player.finish_order == 1
        assert player.lap == 2

    def test_turn_start_position_set(self) -> None:
        """turn_start_position should be set before movement."""
        track = _make_track(length=20)
        player = _make_player(0, position=5, lap=1)
        player.cards_played = [_speed(3, 0)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.turn_start_position == 5

    def test_upgrade_card_value_counted(self) -> None:
        """Upgrade card values should be counted for speed."""
        track = _make_track(length=30)
        player = _make_player(0, position=0, lap=1)
        player.cards_played = [_speed(2, 0), _upgrade(5, 0)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.speed_from_cards == 7
        assert player.position == 7

    def test_empty_deck_stress_resolution(self) -> None:
        """Stress resolution with empty deck yields 0."""
        track = _make_track(length=30)
        deck = _make_deck_with_known_cards([])
        player = _make_player(0, position=0, lap=1, deck=deck)
        player.cards_played = [_stress(0)]
        state = _make_game_state(track, [player])

        step_reveal_and_move(state, player)
        assert player.speed_from_cards == 0
        assert player.position == 0


# ===========================================================================
# Tests: step_adrenaline
# ===========================================================================


class TestStepAdrenaline:
    def test_eligible_player_gets_event(self) -> None:
        """Trailing player should get adrenaline event."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=15, lap=1)
        p2 = _make_player(1, position=5, lap=1)
        state = _make_game_state(track, [p1, p2])

        events = step_adrenaline(state, p2)
        # p2 is in last place, should get adrenaline
        assert len(events) == 1
        assert events[0].event_type == "adrenaline_granted"

    def test_non_eligible_skipped(self) -> None:
        """Leading player should not get adrenaline."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=15, lap=1)
        p2 = _make_player(1, position=5, lap=1)
        state = _make_game_state(track, [p1, p2])

        events = step_adrenaline(state, p1)
        assert len(events) == 0

    def test_adrenaline_does_not_move(self) -> None:
        """Adrenaline should NOT change position."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=15, lap=1)
        p2 = _make_player(1, position=5, lap=1)
        state = _make_game_state(track, [p1, p2])

        original_pos = p2.position
        step_adrenaline(state, p2)
        assert p2.position == original_pos

    def test_adrenaline_does_not_set_speed(self) -> None:
        """Adrenaline should not set speed fields directly."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=15, lap=1)
        p2 = _make_player(1, position=5, lap=1)
        state = _make_game_state(track, [p1, p2])

        step_adrenaline(state, p2)
        assert p2.speed_from_adrenaline == 0  # Not set until React


# ===========================================================================
# Tests: step_react
# ===========================================================================


class TestStepReact:
    def test_cooldown_moves_heat_to_pool(self) -> None:
        """Cooldown should move heat cards from hand to heat pool."""
        track = _make_track()
        heat_in_hand = [_heat(10), _heat(11), _heat(12)]
        hand = [_speed(1, 0), _speed(2, 1)] + heat_in_hand
        player = _make_player(0, gear=1, hand=hand, heat_pool_size=3)
        state = _make_game_state(track, [player])

        # Gear 1 = cooldown 3, we cool 2
        decision = ReactDecision(cooldown_count=2)
        step_react(state, player, decision)

        # 2 heat cards moved from hand to pool
        heat_remaining_in_hand = [c for c in player.hand if c.card_type == CardType.HEAT]
        assert len(heat_remaining_in_hand) == 1
        assert player.heat_available == 5  # 3 original + 2 cooled

    def test_cooldown_capped_at_max(self) -> None:
        """Cooldown should not exceed max allowed."""
        track = _make_track()
        hand = [_heat(10), _heat(11), _heat(12)] + [_speed(1, 0)]
        player = _make_player(0, gear=2, hand=hand, heat_pool_size=3)
        state = _make_game_state(track, [player])

        # Gear 2 = cooldown 1, request 3 => capped to 1
        decision = ReactDecision(cooldown_count=3)
        step_react(state, player, decision)

        heat_in_hand = [c for c in player.hand if c.card_type == CardType.HEAT]
        assert len(heat_in_hand) == 2  # Only 1 cooled

    def test_boost_flips_and_pays_heat(self) -> None:
        """Boost should pay 1 heat and flip cards until basic."""
        track = _make_track(length=30)
        speed_card = _speed(4, 99)
        deck = _make_deck_with_known_cards([speed_card])
        player = _make_player(0, position=5, lap=1, gear=2,
                              deck=deck, heat_pool_size=6)
        state = _make_game_state(track, [player])

        decision = ReactDecision(use_boost=True)
        step_react(state, player, decision)

        assert player.speed_from_boost == 4
        assert player.heat_available == 5  # Paid 1
        assert player.boost_used_this_turn is True
        assert player.position == 9  # 5 + 4 = 9

    def test_boost_once_per_turn_enforced(self) -> None:
        """Using boost twice should raise."""
        track = _make_track(length=30)
        deck = _make_deck_with_known_cards([_speed(2, 99)])
        player = _make_player(0, position=0, gear=2, deck=deck, heat_pool_size=6)
        player.boost_used_this_turn = True
        state = _make_game_state(track, [player])

        decision = ReactDecision(use_boost=True)
        with pytest.raises(ValueError, match="already used boost"):
            step_react(state, player, decision)

    def test_boost_no_heat_raises(self) -> None:
        """Boost with no heat available should raise."""
        track = _make_track()
        deck = _make_deck_with_known_cards([_speed(2, 99)])
        player = _make_player(0, gear=2, deck=deck, heat_pool_size=0)
        state = _make_game_state(track, [player])

        decision = ReactDecision(use_boost=True)
        with pytest.raises(ValueError, match="no heat"):
            step_react(state, player, decision)

    def test_adrenaline_speed_adds_movement(self) -> None:
        """Adrenaline speed should move player 1 extra space."""
        track = _make_track(length=30)
        player = _make_player(0, position=5, lap=1, gear=3)
        state = _make_game_state(track, [player])

        decision = ReactDecision(use_adrenaline_speed=True)
        step_react(state, player, decision)

        assert player.speed_from_adrenaline == 1
        assert player.position == 6  # 5 + 1

    def test_adrenaline_cooldown_increases_max(self) -> None:
        """Adrenaline cooldown should allow 1 extra heat cooled."""
        track = _make_track()
        hand = [_heat(10), _heat(11), _heat(12)] + [_speed(1, 0)]
        player = _make_player(0, gear=2, hand=hand, heat_pool_size=3)
        state = _make_game_state(track, [player])

        # Gear 2 = cooldown 1, with adrenaline = 2
        decision = ReactDecision(cooldown_count=2, use_adrenaline_cooldown=True)
        step_react(state, player, decision)

        heat_in_hand = [c for c in player.hand if c.card_type == CardType.HEAT]
        assert len(heat_in_hand) == 1  # 3 - 2 = 1

    def test_no_react_actions(self) -> None:
        """No actions should not change anything."""
        track = _make_track()
        player = _make_player(0, position=5, gear=3)
        state = _make_game_state(track, [player])

        decision = ReactDecision()
        step_react(state, player, decision)

        assert player.position == 5
        assert player.speed_from_boost == 0
        assert player.speed_from_adrenaline == 0

    def test_boost_and_adrenaline_combined(self) -> None:
        """Boost and adrenaline speed should both contribute to movement."""
        track = _make_track(length=30)
        speed_card = _speed(3, 99)
        deck = _make_deck_with_known_cards([speed_card])
        player = _make_player(0, position=5, lap=1, gear=2,
                              deck=deck, heat_pool_size=6)
        state = _make_game_state(track, [player])

        decision = ReactDecision(use_boost=True, use_adrenaline_speed=True)
        step_react(state, player, decision)

        assert player.speed_from_boost == 3
        assert player.speed_from_adrenaline == 1
        assert player.position == 9  # 5 + 3 + 1 = 9

    def test_react_finish_detection(self) -> None:
        """Player can finish during React (boost/adrenaline movement)."""
        track = _make_track(length=10, laps=1)
        speed_card = _speed(5, 99)
        deck = _make_deck_with_known_cards([speed_card])
        player = _make_player(0, position=8, lap=1, gear=2,
                              deck=deck, heat_pool_size=6)
        state = _make_game_state(track, [player])

        decision = ReactDecision(use_boost=True)
        step_react(state, player, decision)

        assert player.finished is True


# ===========================================================================
# Tests: step_slipstream
# ===========================================================================


class TestStepSlipstream:
    def test_eligible_moves_plus_2(self) -> None:
        """Eligible player should move +2 when decision is True."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=5, lap=1)
        p2 = _make_player(1, position=7, lap=1)  # 2 ahead
        state = _make_game_state(track, [p1, p2])

        events = step_slipstream(state, p1, True)
        assert p1.position == 7  # 5 + 2
        assert p1.slipstream_moved == 2
        assert len(events) == 1

    def test_decline_slipstream(self) -> None:
        """Player declines slipstream."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=5, lap=1)
        p2 = _make_player(1, position=7, lap=1)
        state = _make_game_state(track, [p1, p2])

        events = step_slipstream(state, p1, False)
        assert p1.position == 5
        assert p1.slipstream_moved == 0
        assert len(events) == 0

    def test_not_eligible_stays(self) -> None:
        """Not eligible player stays put even if decision is True."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=5, lap=1)
        p2 = _make_player(1, position=10, lap=1)  # 5 ahead, not eligible
        state = _make_game_state(track, [p1, p2])

        events = step_slipstream(state, p1, True)
        assert p1.position == 5
        assert p1.slipstream_moved == 0
        assert len(events) == 0

    def test_cannot_cross_finish(self) -> None:
        """Slipstream should not move player across finish line."""
        track = _make_track(length=10, laps=1)
        p1 = _make_player(0, position=8, lap=1)
        p2 = _make_player(1, position=9, lap=1)  # 1 ahead
        state = _make_game_state(track, [p1, p2])

        events = step_slipstream(state, p1, True)
        assert p1.position == 8  # Should not move (would cross finish)
        assert p1.slipstream_moved == 0

    def test_slipstream_moved_tracked(self) -> None:
        """slipstream_moved should be 2 when taken, 0 when not."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=5, lap=1)
        p2 = _make_player(1, position=6, lap=1)
        state = _make_game_state(track, [p1, p2])

        step_slipstream(state, p1, True)
        assert p1.slipstream_moved == 2

    def test_finished_player_cannot_slipstream(self) -> None:
        """Finished player should not be able to slipstream."""
        track = _make_track(length=30)
        p1 = _make_player(0, position=5, lap=1, finished=True)
        p2 = _make_player(1, position=6, lap=1)
        state = _make_game_state(track, [p1, p2])

        events = step_slipstream(state, p1, True)
        assert p1.slipstream_moved == 0
        assert len(events) == 0


# ===========================================================================
# Tests: step_check_corner
# ===========================================================================


class TestStepCheckCorner:
    def test_under_limit_no_cost(self) -> None:
        """Speed under corner limit should cost nothing."""
        corner = Corner(start=4, end=6, speed_limit=5)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=7, lap=1, gear=2, heat_pool_size=6)
        player.turn_start_position = 2
        player.speed_from_cards = 3  # Under limit of 5
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        state = _make_game_state(track, [player])

        events = step_check_corner(state, player)
        assert player.heat_available == 6  # No heat paid
        assert len(events) == 1
        assert events[0].data["heat_cost"] == 0

    def test_over_limit_pays_heat(self) -> None:
        """Speed over corner limit should pay heat."""
        corner = Corner(start=4, end=6, speed_limit=3)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=7, lap=1, gear=2, heat_pool_size=6)
        player.turn_start_position = 2
        player.speed_from_cards = 5  # 5 - 3 = 2 heat
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        state = _make_game_state(track, [player])

        step_check_corner(state, player)
        assert player.heat_available == 4  # Paid 2 heat

    def test_spin_out_before_corner(self) -> None:
        """Spin out should place player before corner start."""
        corner = Corner(start=4, end=6, speed_limit=2)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=7, lap=1, gear=2, heat_pool_size=1)
        player.turn_start_position = 2
        player.speed_from_cards = 5  # 5 - 2 = 3 heat needed, only 1 available
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        state = _make_game_state(track, [player])

        step_check_corner(state, player)

        assert player.spun_out is True
        assert player.position == 3  # corner.start - 1 = 4 - 1 = 3
        assert player.gear == 1
        assert player.heat_available == 0  # All heat paid

    def test_spin_out_stress_cards_added(self) -> None:
        """Spin out should add stress cards to hand."""
        corner = Corner(start=4, end=6, speed_limit=2)
        track = _make_track(length=10, corners=[corner])
        # Gear 2 => 1 stress card
        player = _make_player(0, position=7, lap=1, gear=2, heat_pool_size=0)
        player.turn_start_position = 2
        player.speed_from_cards = 5
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        state = _make_game_state(track, [player])

        initial_hand_len = len(player.hand)
        step_check_corner(state, player)

        # Gear 2 => 1 stress card added
        stress_in_hand = [c for c in player.hand if c.card_type == CardType.STRESS]
        assert len(stress_in_hand) >= 1
        assert len(player.hand) == initial_hand_len + 1

    def test_spin_out_high_gear_more_stress(self) -> None:
        """Gear 3-4 spin out should add 2 stress cards."""
        corner = Corner(start=4, end=6, speed_limit=2)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=7, lap=1, gear=4, heat_pool_size=0)
        player.turn_start_position = 2
        player.speed_from_cards = 10
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        state = _make_game_state(track, [player])

        initial_hand_len = len(player.hand)
        step_check_corner(state, player)

        # Gear 4 => 2 stress cards
        assert len(player.hand) == initial_hand_len + 2

    def test_corners_ignored_after_finish(self) -> None:
        """Finished player should not pay corner costs."""
        corner = Corner(start=4, end=6, speed_limit=2)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=5, lap=1, gear=2,
                              heat_pool_size=6, finished=True)
        player.turn_start_position = 0
        player.speed_from_cards = 10
        state = _make_game_state(track, [player])

        events = step_check_corner(state, player)
        assert player.heat_available == 6  # No heat paid
        assert len(events) == 0

    def test_speed_excludes_slipstream(self) -> None:
        """Corner speed check should exclude slipstream movement."""
        corner = Corner(start=4, end=6, speed_limit=3)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=8, lap=1, gear=2, heat_pool_size=6)
        player.turn_start_position = 2
        player.speed_from_cards = 3  # At limit
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 2  # This should NOT count
        state = _make_game_state(track, [player])

        step_check_corner(state, player)
        assert player.heat_available == 6  # No heat paid (speed = 3, limit = 3)

    def test_speed_includes_boost(self) -> None:
        """Corner speed check should include boost value."""
        corner = Corner(start=4, end=6, speed_limit=3)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=7, lap=1, gear=2, heat_pool_size=6)
        player.turn_start_position = 2
        player.speed_from_cards = 3
        player.speed_from_boost = 2  # 3 + 2 = 5, over limit by 2
        player.speed_from_adrenaline = 0
        player.slipstream_moved = 0
        state = _make_game_state(track, [player])

        step_check_corner(state, player)
        assert player.heat_available == 4  # Paid 2 heat

    def test_no_corners_crossed(self) -> None:
        """No corners crossed should produce no events."""
        corner = Corner(start=8, end=9, speed_limit=3)
        track = _make_track(length=10, corners=[corner])
        player = _make_player(0, position=3, lap=1, gear=2)
        player.turn_start_position = 0
        player.speed_from_cards = 3
        state = _make_game_state(track, [player])

        events = step_check_corner(state, player)
        assert len(events) == 0

    def test_multiple_corners(self) -> None:
        """Multiple corners crossed should sum heat costs."""
        c1 = Corner(start=3, end=5, speed_limit=3)
        c2 = Corner(start=10, end=12, speed_limit=4)
        track = _make_track(length=20, corners=[c1, c2])
        player = _make_player(0, position=15, lap=1, gear=2, heat_pool_size=6)
        player.turn_start_position = 1
        player.speed_from_cards = 7
        player.speed_from_boost = 0
        player.speed_from_adrenaline = 0
        state = _make_game_state(track, [player])

        step_check_corner(state, player)
        # Corner 1: 7 - 3 = 4 heat
        # Corner 2: 7 - 4 = 3 heat
        # Total: 7 heat, but only 6 available => spin out
        assert player.spun_out is True


# ===========================================================================
# Tests: step_discard
# ===========================================================================


class TestStepDiscard:
    def test_speed_can_be_discarded(self) -> None:
        """Speed cards should be discardable."""
        track = _make_track()
        card = _speed(3, 0)
        hand = [card, _speed(1, 1), _heat(0)]
        player = _make_player(0, hand=hand)
        state = _make_game_state(track, [player])

        step_discard(state, player, [card])
        assert card not in player.hand
        assert len(player.hand) == 2

    def test_upgrade_can_be_discarded(self) -> None:
        """Upgrade cards should be discardable."""
        track = _make_track()
        card = _upgrade(5, 0)
        hand = [card, _speed(1, 1), _heat(0)]
        player = _make_player(0, hand=hand)
        state = _make_game_state(track, [player])

        step_discard(state, player, [card])
        assert card not in player.hand

    def test_heat_cannot_be_discarded(self) -> None:
        """Heat cards should not be discardable."""
        track = _make_track()
        card = _heat(0)
        hand = [card, _speed(1, 1)]
        player = _make_player(0, hand=hand)
        state = _make_game_state(track, [player])

        with pytest.raises(ValueError, match="heat"):
            step_discard(state, player, [card])

    def test_stress_cannot_be_discarded(self) -> None:
        """Stress cards should not be discardable."""
        track = _make_track()
        card = _stress(0)
        hand = [card, _speed(1, 1)]
        player = _make_player(0, hand=hand)
        state = _make_game_state(track, [player])

        with pytest.raises(ValueError, match="stress"):
            step_discard(state, player, [card])

    def test_empty_discard_keeps_all(self) -> None:
        """Empty discard list should keep all cards."""
        track = _make_track()
        hand = [_speed(1, 0), _speed(2, 1), _heat(0)]
        player = _make_player(0, hand=hand)
        state = _make_game_state(track, [player])

        events = step_discard(state, player, [])
        assert len(player.hand) == 3
        assert len(events) == 0

    def test_card_not_in_hand_raises(self) -> None:
        """Discarding a card not in hand should raise."""
        track = _make_track()
        card = _speed(99, 99)
        hand = [_speed(1, 0)]
        player = _make_player(0, hand=hand)
        state = _make_game_state(track, [player])

        with pytest.raises(ValueError, match="not in their hand"):
            step_discard(state, player, [card])

    def test_discarded_cards_go_to_deck_discard(self) -> None:
        """Discarded cards should go to the deck's discard pile."""
        track = _make_track()
        card = _speed(3, 0)
        deck = Deck([])
        hand = [card, _speed(1, 1)]
        player = _make_player(0, hand=hand, deck=deck)
        state = _make_game_state(track, [player])

        step_discard(state, player, [card])
        assert deck.discard_pile_size == 1


# ===========================================================================
# Tests: step_replenish
# ===========================================================================


class TestStepReplenish:
    def test_played_cards_moved_to_discard(self) -> None:
        """cards_played should be moved to the deck's discard pile."""
        track = _make_track()
        card1 = _speed(3, 0)
        card2 = _speed(4, 1)
        # Pre-populate draw pile with enough cards so the discard pile
        # is not reshuffled during the draw step
        draw_cards = [_speed(v, i + 50) for i, v in enumerate([1, 2, 3, 4, 5])]
        deck = _make_deck_with_known_cards(draw_cards)
        player = _make_player(0, deck=deck)
        player.cards_played = [card1, card2]
        player.hand = [_speed(1, 2), _speed(2, 3)]
        state = _make_game_state(track, [player])

        step_replenish(state, player)
        assert player.cards_played == []
        assert deck.discard_pile_size >= 2  # At least the 2 played cards

    def test_hand_refilled_to_7(self) -> None:
        """Hand should be refilled to HAND_SIZE (7)."""
        track = _make_track()
        # Put some cards in draw pile for drawing
        draw_cards = [_speed(v, i + 20) for i, v in enumerate([1, 2, 3, 4, 5])]
        deck = _make_deck_with_known_cards(draw_cards)
        player = _make_player(0, deck=deck)
        player.hand = [_speed(1, 10), _speed(2, 11)]  # 2 cards in hand
        player.cards_played = []
        state = _make_game_state(track, [player])

        step_replenish(state, player)
        assert len(player.hand) == 7  # 2 + 5 drawn

    def test_transient_fields_cleared(self) -> None:
        """All transient fields should be cleared."""
        track = _make_track()
        deck = _make_deck_with_known_cards(
            [_speed(v, i + 50) for i, v in enumerate([1, 2, 3, 4, 5, 6, 1])]
        )
        player = _make_player(0, deck=deck)
        player.hand = []
        player.cards_played = [_speed(1, 30)]
        player.boost_used_this_turn = True
        player.speed_from_cards = 5
        player.speed_from_boost = 3
        player.speed_from_adrenaline = 1
        player.slipstream_moved = 2
        player.cluttered = True
        player.turn_start_position = 5
        state = _make_game_state(track, [player])

        step_replenish(state, player)

        assert player.cards_played == []
        assert player.boost_used_this_turn is False
        assert player.speed_from_cards == 0
        assert player.speed_from_boost == 0
        assert player.speed_from_adrenaline == 0
        assert player.slipstream_moved == 0
        assert player.cluttered is False
        assert player.turn_start_position == 0

    def test_no_cooldown_in_replenish(self) -> None:
        """Cooldown should NOT happen during replenish."""
        track = _make_track()
        heat_in_hand = [_heat(10), _heat(11)]
        deck = _make_deck_with_known_cards(
            [_speed(v, i + 50) for i, v in enumerate([1, 2, 3, 4, 5])]
        )
        player = _make_player(0, gear=1, hand=heat_in_hand, deck=deck,
                              heat_pool_size=4)
        player.cards_played = []
        state = _make_game_state(track, [player])

        original_heat_pool = player.heat_available
        step_replenish(state, player)

        # Heat cards should still be in hand (drawn to 7, heat cards remain)
        assert player.heat_available == original_heat_pool  # No cooldown

    def test_no_draw_penalty_for_spin_out(self) -> None:
        """Spun-out player should draw normally to 7 cards."""
        track = _make_track()
        draw_cards = [_speed(v, i + 50) for i, v in enumerate([1, 2, 3, 4, 5, 6, 1])]
        deck = _make_deck_with_known_cards(draw_cards)
        player = _make_player(0, deck=deck, spun_out=True)
        player.hand = []
        player.cards_played = []
        state = _make_game_state(track, [player])

        step_replenish(state, player)
        assert len(player.hand) == 7

    def test_not_enough_cards_to_draw(self) -> None:
        """If deck has fewer cards than needed, draw what's available."""
        track = _make_track()
        deck = _make_deck_with_known_cards([_speed(1, 50)])
        player = _make_player(0, deck=deck)
        player.hand = [_speed(2, 10)]
        player.cards_played = []
        state = _make_game_state(track, [player])

        step_replenish(state, player)
        # 1 in hand + 1 drawn = 2 (not enough to reach 7)
        assert len(player.hand) == 2


class TestStressCounterPerGame:
    """Verify stress card IDs use per-game counter, not module-level."""

    def test_separate_games_have_independent_counters(self) -> None:
        """Two separate GameState instances should have independent stress counters."""
        track = _make_track(corners=[Corner(start=4, end=6, speed_limit=2)])

        # Game 1: trigger a spin-out
        player1 = _make_player(0, position=7, gear=3, heat_pool_size=0)
        player1.turn_start_position = 0
        player1.speed_from_cards = 8
        state1 = _make_game_state(track, [player1])
        step_check_corner(state1, player1)
        stress_ids_1 = [c.id for c in player1.hand if c.card_type == CardType.STRESS]

        # Game 2: trigger a spin-out
        player2 = _make_player(0, position=7, gear=3, heat_pool_size=0)
        player2.turn_start_position = 0
        player2.speed_from_cards = 8
        state2 = _make_game_state(track, [player2])
        step_check_corner(state2, player2)
        stress_ids_2 = [c.id for c in player2.hand if c.card_type == CardType.STRESS]

        # Each game should have its own counter starting from 1
        assert stress_ids_1 == stress_ids_2  # same IDs since independent counters
