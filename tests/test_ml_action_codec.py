"""Tests for the ML action codec (Sprint 5a, ml/action_codec.py).

Gates: mask legality matches rules.legal_* exactly and is never all-False;
decode(encode(x)) == x within the legal set; value-redundant card plays are
collapsed to a single action index.
"""

from __future__ import annotations

import random

import numpy as np

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.phases import ReactDecision
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.ml import action_codec
from heat.ml.action_codec import (
    decode_action,
    encode_action_index,
    legal_action_mask,
)
from heat.ml.spaces import (
    ACTION_DIM,
    CARDS_OFFSET,
    DISCARD_OFFSET,
    GEAR_OFFSET,
    REACT_OFFSET,
)


def test_published_codec_contract() -> None:
    """Own the published dimensions/version; intentional codec changes update this guard."""
    from heat.ml import spaces

    assert (spaces.OBS_DIM, spaces.ACTION_DIM, spaces.CODEC_VERSION) == (104, 516, 3)


def _track(laps: int = 2) -> Track:
    spaces_ = [Space(index=i, lanes=2) for i in range(20)]
    corners = [Corner(start=5, end=6, speed_limit=3)]
    return Track(
        name="codec-test",
        spaces=spaces_,
        corners=corners,
        start_positions=[0, 1, 2, 3],
        laps=laps,
    )


def _randomize(state: GameState, rng: random.Random) -> None:
    for p in state.players:
        p.gear = rng.randint(1, 4)
        p.position = rng.randint(0, state.track.length - 1)
        p.lap = rng.randint(1, state.track.laps)
        p.boost_used_this_turn = rng.random() < 0.5
        pay = rng.randint(0, p.heat_available)
        if pay:
            p.pay_heat(pay)


def _all_decisions(state: GameState) -> list[Decision]:
    """Build one Decision of each kind per player from the real legal_* API."""
    decisions: list[Decision] = []
    active = list(state.active_players)
    for p in state.players:
        pid = p.player_id
        decisions.append(
            Decision(
                DecisionKind.GEAR,
                pid,
                rules.legal_gear_shifts(p.gear, p.heat_available),
            )
        )
        decisions.append(
            Decision(
                DecisionKind.CARDS,
                pid,
                rules.legal_card_plays(p.hand, p.gear),
            )
        )
        decisions.append(
            Decision(
                DecisionKind.REACT,
                pid,
                rules.legal_react_options(
                    p, active, state.starting_player_count
                ),
            )
        )
        decisions.append(Decision(DecisionKind.SLIPSTREAM, pid, True))
        decisions.append(
            Decision(DecisionKind.DISCARD, pid, rules.legal_discards(p))
        )
    return decisions


class TestMaskLegality:
    def test_mask_shape_and_never_all_false(self) -> None:
        rng = random.Random(1)
        for trial in range(100):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            for decision in _all_decisions(state):
                mask = legal_action_mask(decision, state)
                assert mask.shape == (ACTION_DIM,)
                assert mask.dtype == bool
                assert mask.any(), f"all-False mask for {decision.kind}"

    def test_gear_mask_matches_legal(self) -> None:
        rng = random.Random(2)
        for trial in range(50):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            for p in state.players:
                legal = rules.legal_gear_shifts(p.gear, p.heat_available)
                decision = Decision(DecisionKind.GEAR, p.player_id, legal)
                mask = legal_action_mask(decision, state)
                expected = {
                    GEAR_OFFSET + (g - rules.MIN_GEAR) for g, _ in legal
                }
                got = set(np.flatnonzero(mask).tolist())
                assert got == expected

    def test_cards_mask_collapses_to_distinct_value_multisets(self) -> None:
        rng = random.Random(3)
        for trial in range(50):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            for p in state.players:
                legal = rules.legal_card_plays(p.hand, p.gear)
                decision = Decision(DecisionKind.CARDS, p.player_id, legal)
                mask = legal_action_mask(decision, state)
                # Expected set = distinct value-multisets across all legal plays.
                distinct = {
                    action_codec._play_to_multiset(play) for play in legal
                }
                expected = {
                    CARDS_OFFSET + action_codec._CARD_MULTISET_INDEX[ms]
                    for ms in distinct
                }
                got = set(np.flatnonzero(mask).tolist())
                assert got == expected
                # Collapse really happened: many concrete plays, fewer indices.
                assert mask.sum() == len(distinct)

    def test_react_mask_matches_options(self) -> None:
        rng = random.Random(4)
        for trial in range(80):
            state = GameState.create(_track(), 5, seed=trial)
            _randomize(state, rng)
            active = list(state.active_players)
            for p in state.players:
                opts = rules.legal_react_options(
                    p, active, state.starting_player_count
                )
                decision = Decision(DecisionKind.REACT, p.player_id, opts)
                mask = legal_action_mask(decision, state)
                for i, slot in enumerate(action_codec._REACT_TABLE):
                    expected = action_codec._react_slot_legal(slot, opts)
                    assert bool(mask[REACT_OFFSET + i]) == expected

    def test_discard_mask_matches_count(self) -> None:
        rng = random.Random(5)
        from heat.ml.spaces import DISCARD_SIZE

        for trial in range(50):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            for p in state.players:
                discardable = rules.legal_discards(p)
                decision = Decision(
                    DecisionKind.DISCARD, p.player_id, discardable
                )
                mask = legal_action_mask(decision, state)
                n = len(discardable)
                assert bool(mask[DISCARD_OFFSET + 0])  # discard none always legal
                for k in range(1, DISCARD_SIZE):
                    assert bool(mask[DISCARD_OFFSET + k]) == (k <= n)


class TestEncodeDecodeRoundTrip:
    def test_roundtrip_over_legal_actions(self) -> None:
        rng = random.Random(6)
        for trial in range(80):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            active = list(state.active_players)
            for p in state.players:
                pid = p.player_id
                # GEAR
                for option in rules.legal_gear_shifts(p.gear, p.heat_available):
                    dec = Decision(DecisionKind.GEAR, pid, [option])
                    idx = encode_action_index(dec, option)
                    assert decode_action(dec, idx, state) == option

                # CARDS (compare by value-multiset due to redundancy collapse).
                legal_plays = rules.legal_card_plays(p.hand, p.gear)
                dec = Decision(DecisionKind.CARDS, pid, legal_plays)
                for play in legal_plays:
                    idx = encode_action_index(dec, play)
                    decoded = decode_action(dec, idx, state)
                    assert action_codec._play_to_multiset(
                        decoded
                    ) == action_codec._play_to_multiset(play)
                    # Decoded cards really come from the hand.
                    assert all(c in p.hand for c in decoded)

                # REACT (round-trip over the legal fixed slots).
                opts = rules.legal_react_options(
                    p, active, state.starting_player_count
                )
                dec = Decision(DecisionKind.REACT, pid, opts)
                mask = legal_action_mask(dec, state)
                for flat in np.flatnonzero(mask):
                    react = decode_action(dec, int(flat), state)
                    assert isinstance(react, ReactDecision)
                    assert encode_action_index(dec, react) == int(flat)

                # SLIPSTREAM
                dec = Decision(DecisionKind.SLIPSTREAM, pid, True)
                for choice in (True, False):
                    idx = encode_action_index(dec, choice)
                    assert decode_action(dec, idx, state) == choice

                # DISCARD
                discardable = rules.legal_discards(p)
                dec = Decision(DecisionKind.DISCARD, pid, discardable)
                mask = legal_action_mask(dec, state)
                for flat in np.flatnonzero(mask):
                    cards = decode_action(dec, int(flat), state)
                    assert encode_action_index(dec, cards) == int(flat)
                    # decoded discards are a subset of discardable, lowest-first.
                    assert all(c in discardable for c in cards)


class TestDecodedCardPlayIsDriverSendable:
    def test_decoded_cards_are_in_legal_plays(self) -> None:
        """decode_action for CARDS must return a tuple that is identity-equal to
        one in decision.legal, so it passes the driver's `chosen in legal_plays`
        guard (rules.legal_card_plays yields tuples in HAND order, not
        token/id order)."""
        rng = random.Random(11)
        for trial in range(80):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            for p in state.players:
                legal_plays = rules.legal_card_plays(p.hand, p.gear)
                if not legal_plays:
                    continue
                dec = Decision(DecisionKind.CARDS, p.player_id, legal_plays)
                mask = legal_action_mask(dec, state)
                for flat in np.flatnonzero(mask):
                    decoded = decode_action(dec, int(flat), state)
                    assert decoded in legal_plays


class TestCardValueRedundancyCollapsed:
    def test_two_equal_speed_cards_one_index(self) -> None:
        # Craft a hand with two equal-value (value 3) Speed cards at gear 1.
        c_a = Card(CardType.SPEED, 3, "x_spd_a")
        c_b = Card(CardType.SPEED, 3, "x_spd_b")
        c_c = Card(CardType.SPEED, 1, "x_spd_c")
        hand = [c_a, c_b, c_c]
        legal = rules.legal_card_plays(hand, gear=1)
        # Engine yields one tuple per distinct Card object: (3a,), (3b,), (1,).
        assert (c_a,) in legal and (c_b,) in legal
        idx_a = encode_action_index(
            Decision(DecisionKind.CARDS, 0, legal), (c_a,)
        )
        idx_b = encode_action_index(
            Decision(DecisionKind.CARDS, 0, legal), (c_b,)
        )
        assert idx_a == idx_b  # collapsed to one action index
