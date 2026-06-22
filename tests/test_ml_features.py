"""Tests for the ML observation encoder (Sprint 5a, ml/features.py).

Gates: shape/dtype/bounds, determinism, clone-invariance, player-count
invariance, and hidden-info correctness (opponent-hand permutation invariance).
"""

from __future__ import annotations

import random

import numpy as np

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.ml import features
from heat.ml.spaces import OBS_DIM


def _track(laps: int = 2) -> Track:
    spaces_ = [Space(index=i, lanes=2) for i in range(20)]
    corners = [
        Corner(start=5, end=6, speed_limit=3),
        Corner(start=14, end=15, speed_limit=2),
    ]
    return Track(
        name="feat-test",
        spaces=spaces_,
        corners=corners,
        start_positions=[0, 1, 2, 3, 4, 5],
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
        if rng.random() < 0.15:
            p.finished = True
            p.finish_order = p.player_id + 1


def _sample_decisions(state: GameState, player_id: int) -> list[Decision | None]:
    """A representative set of Decisions for the phase-context block."""
    player = state.get_player(player_id)
    opts = rules.legal_react_options(
        player, list(state.active_players), state.starting_player_count
    )
    return [
        None,
        Decision(DecisionKind.GEAR, player_id, [(player.gear, 0)]),
        Decision(DecisionKind.CARDS, player_id, [tuple(player.hand[:1])]),
        Decision(DecisionKind.REACT, player_id, opts),
        Decision(DecisionKind.SLIPSTREAM, player_id, True),
        Decision(DecisionKind.DISCARD, player_id, rules.legal_discards(player)),
    ]


class TestShapeAndBounds:
    def test_shape_dtype_bounds(self) -> None:
        rng = random.Random(0)
        for trial in range(50):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            for pid in range(state.num_players):
                for decision in _sample_decisions(state, pid):
                    vec = features.encode_observation(state, pid, decision)
                    assert vec.shape == (OBS_DIM,)
                    assert vec.dtype == np.float32
                    assert np.all(vec >= -1.0)
                    assert np.all(vec <= 1.0)
                    assert not np.any(np.isnan(vec))


class TestDeterminism:
    def test_same_inputs_same_vector(self) -> None:
        state = GameState.create(_track(), 4, seed=7)
        decision = Decision(DecisionKind.GEAR, 0, [(1, 0), (2, 0)])
        a = features.encode_observation(state, 0, decision)
        b = features.encode_observation(state, 0, decision)
        assert np.array_equal(a, b)

    def test_no_mutation_of_state(self) -> None:
        state = GameState.create(_track(), 4, seed=7)
        before_hand = list(state.get_player(0).hand)
        before_pos = state.get_player(0).position
        features.encode_observation(state, 0, None)
        assert state.get_player(0).hand == before_hand
        assert state.get_player(0).position == before_pos


class TestCloneInvariance:
    def test_clone_gives_identical_vector(self) -> None:
        rng = random.Random(3)
        for trial in range(30):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            clone = state.clone()
            for pid in range(state.num_players):
                decision = Decision(
                    DecisionKind.REACT,
                    pid,
                    rules.legal_react_options(
                        state.get_player(pid),
                        list(state.active_players),
                        state.starting_player_count,
                    ),
                )
                clone_decision = Decision(
                    DecisionKind.REACT,
                    pid,
                    rules.legal_react_options(
                        clone.get_player(pid),
                        list(clone.active_players),
                        clone.starting_player_count,
                    ),
                )
                a = features.encode_observation(state, pid, decision)
                b = features.encode_observation(clone, pid, clone_decision)
                assert np.array_equal(a, b)


class TestPlayerCountInvariance:
    def test_two_and_six_players_same_dim(self) -> None:
        for num in (2, 6):
            state = GameState.create(_track(), num, seed=num)
            vec = features.encode_observation(state, 0, None)
            assert vec.shape == (OBS_DIM,)

    def test_absent_opponent_slots_zero_with_presence_flag(self) -> None:
        from heat.ml import spaces as sp

        state = GameState.create(_track(), 2, seed=11)
        vec = features.encode_observation(state, 0, None)
        # Opponent block start index = sum of preceding blocks.
        start = (
            sp.BLOCK_HAND_HISTOGRAM
            + sp.BLOCK_OWN_GEAR
            + sp.BLOCK_OWN_KINEMATICS
            + sp.BLOCK_DECK_COMPOSITION
            + sp.BLOCK_TRACK
            + sp.BLOCK_ADRENALINE_CONTEXT
        )
        opp_block = vec[start : start + sp.BLOCK_OPPONENT_SLOTS]
        # 2 players -> 1 opponent present, 4 absent slots (each 5 floats, all 0).
        present_slot = opp_block[: sp.OPP_SLOT_FLOATS]
        assert present_slot[0] == 1.0  # presence flag of the one opponent
        absent = opp_block[sp.OPP_SLOT_FLOATS :]
        assert np.all(absent == 0.0)


class TestStatePurity:
    """The observation must be a pure function of game state: encoding the same
    `state` with `decision=None` (Option C's value path) and with a real decision
    (the policy/prior path) must be identical on EVERY index except the genuine
    decision-context bits (indices 0..8 of the phase block). In particular
    `round_num` (phase-block index 9) is pure game state and must be encoded
    consistently regardless of `decision` (code-review 2026-06-22 #1)."""

    def test_round_num_encoded_when_decision_is_none(self) -> None:
        from heat.ml import features as feat
        from heat.ml import spaces as sp

        state = GameState.create(_track(), 4, seed=5)
        # Force a non-zero round_num so the index is observably set.
        state.round_num = 7
        vec = feat.encode_observation(state, 0, None)
        round_idx = OBS_DIM - sp.BLOCK_PHASE_CONTEXT + feat._PHASE_ROUND_NUM_INDEX
        assert vec[round_idx] == feat._clip01(7 / feat._ROUND_CAP)
        assert vec[round_idx] > 0.0  # actually populated, not zero-padding

    def test_decision_none_matches_real_decision_except_context_bits(self) -> None:
        from heat.ml import spaces as sp

        rng = random.Random(99)
        phase_start = OBS_DIM - sp.BLOCK_PHASE_CONTEXT
        # Indices 0..8 of the phase block are the only legitimate decision-context
        # bits (kind one-hot 0..4, react 5..7, slipstream 8). Everything else --
        # including round_num at index 9 -- must be decision-invariant.
        context_abs = set(range(phase_start, phase_start + 9))

        for trial in range(20):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            state.round_num = rng.randint(0, 60)
            for pid in range(state.num_players):
                base = features.encode_observation(state, pid, None)
                for decision in _sample_decisions(state, pid):
                    if decision is None:
                        continue
                    other = features.encode_observation(state, pid, decision)
                    for i in range(OBS_DIM):
                        if i in context_abs:
                            continue
                        assert other[i] == base[i], (
                            f"trial {trial} pid {pid} kind {decision.kind} "
                            f"index {i}: {other[i]} != {base[i]} -- observation "
                            f"is not a pure function of game state"
                        )


class TestHiddenInfo:
    def test_opponent_hand_permutation_invariant(self) -> None:
        state = GameState.create(_track(), 4, seed=21)
        base = features.encode_observation(state, 0, None)
        # Permute (and even replace) opponent 1's hand: the learner's vector
        # must not change, because opponent hands are hidden.
        opp = state.get_player(1)
        opp.hand = list(reversed(opp.hand))
        after_reverse = features.encode_observation(state, 0, None)
        assert np.array_equal(base, after_reverse)

        # Mutate opponent hand contents entirely (still hidden public-wise).
        opp.hand = []
        after_clear = features.encode_observation(state, 0, None)
        assert np.array_equal(base, after_clear)
