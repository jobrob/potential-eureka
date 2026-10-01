"""Codec v4 tells apart states that codec v3 encoded as the same vector.

The live encoder is v4. ``codec_version=3`` must still reproduce the old
vector so a historical checkpoint sees the inputs it was trained on.
"""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from heat.engine import phases, rules
from heat.engine.driver import Decision, DecisionKind
from heat.ml.action_codec import legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.native_env.bridge import legacy_state_to_native
from heat.ml.selfplay.checkpoint import (
    CheckpointMismatchError,
    load_policy,
    save_policy,
)
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.spaces import BLOCK_PHASE_CONTEXT, OBS_DIM
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.tracks.generator import TrackGenParams, generate_track


def _track(length: int, laps: int, corners: list[Corner]) -> Track:
    return Track(
        name=f"sufficiency-{length}",
        spaces=[Space(index=i, lanes=1) for i in range(length)],
        corners=corners,
        start_positions=[0, 1, 2, 3, 4, 5],
        laps=laps,
    )


def test_absolute_corner_limits_split_v4() -> None:
    """Same geometry with every limit 1 vs 2 is one vector under v3.

    Gear legality uses gear and heat only, so the legal mask stays equal.
    The observation is what changes, and that makes the (observation, mask)
    pair differ.
    """
    tracks = [
        generate_track(17, TrackGenParams(speed_limit_choices=(limit,)))
        for limit in (1, 2)
    ]
    geometry = [
        [(corner.start, corner.end) for corner in track.corners] for track in tracks
    ]
    assert geometry[0] == geometry[1]
    assert tracks[0].length == tracks[1].length
    assert [corner.speed_limit for corner in tracks[0].corners] == [1] * len(
        tracks[0].corners
    )
    assert [corner.speed_limit for corner in tracks[1].corners] == [2] * len(
        tracks[1].corners
    )
    assert [
        rules.corner_heat_cost(3, track.corners[0]) for track in tracks
    ] == [2, 1]

    observations_v4 = []
    observations_v3 = []
    masks = []
    native_observations = []
    for track in tracks:
        state = GameState.create(track, 2, seed=5)
        for player in state.players:
            player.lap = 1
        seat = state.players[0]
        decision = Decision(
            DecisionKind.GEAR,
            0,
            rules.legal_gear_shifts(seat.gear, seat.heat_available),
        )
        observations_v4.append(encode_observation(state, 0, decision))
        observations_v3.append(
            encode_observation(state, 0, decision, codec_version=3)
        )
        masks.append(legal_action_mask(decision, state))
        native_observations.append(
            legacy_state_to_native(state, game_id=0).observation(0, decision)
        )

    assert not np.array_equal(observations_v4[0], observations_v4[1])
    assert np.array_equal(observations_v3[0], observations_v3[1])
    assert not np.array_equal(native_observations[0], native_observations[1])
    np.testing.assert_array_equal(native_observations[0], observations_v4[0])
    np.testing.assert_array_equal(native_observations[1], observations_v4[1])
    assert np.array_equal(masks[0], masks[1])
    assert observations_v4[0].tobytes() != observations_v4[1].tobytes()


def test_final_lap_start_keeps_one_lap_of_distance() -> None:
    """Position 0 on the last lap of an 83-space 2-lap track is half the race."""
    track = _track(83, 2, [Corner(start=10, end=12, speed_limit=2)])
    state = GameState.create(track, 2, seed=0)
    player = state.players[0]
    player.lap = track.laps
    player.position = 0
    # Second track global: hand + gear + kinematics + deck + 8 corner slots.
    dist_index = 8 + 4 + 4 + 6 + 8 * 4 + 1
    current = encode_observation(state, 0, None)
    legacy = encode_observation(state, 0, None, codec_version=3)
    assert abs(float(current[dist_index]) - 0.5) < 1e-6
    assert float(legacy[dist_index]) == 0.0


def test_react_speed_and_turn_origin_split_v4() -> None:
    """Same square and the same react menu, but one line spins and the other does not."""
    track = _track(40, 2, [Corner(start=10, end=12, speed_limit=3)])
    fast = GameState.create(track, 2, seed=3)
    for player in fast.players:
        player.lap = 1
    seat = fast.players[0]
    seat.position = 14
    seat.turn_start_position = 8
    seat.turn_start_lap = seat.lap
    seat.speed_from_cards = 5
    seat.speed_from_boost = 0
    seat.speed_from_adrenaline = 0
    seat.pay_heat(seat.heat_available - 1)

    slow = copy.deepcopy(fast)
    other = slow.players[0]
    other.turn_start_position = 13
    other.speed_from_cards = 1

    def react_decision(state: GameState) -> Decision:
        player = state.players[0]
        options = rules.legal_react_options(
            player, state.active_players, state.starting_player_count
        )
        return Decision(DecisionKind.REACT, 0, options)

    fast_decision = react_decision(fast)
    slow_decision = react_decision(slow)
    assert np.array_equal(
        legal_action_mask(fast_decision, fast),
        legal_action_mask(slow_decision, slow),
    )
    fast_v4 = encode_observation(fast, 0, fast_decision)
    slow_v4 = encode_observation(slow, 0, slow_decision)
    fast_v3 = encode_observation(fast, 0, fast_decision, codec_version=3)
    slow_v3 = encode_observation(slow, 0, slow_decision, codec_version=3)
    fast_native = legacy_state_to_native(fast, game_id=0).observation(0, fast_decision)
    slow_native = legacy_state_to_native(slow, game_id=0).observation(0, slow_decision)
    assert not np.array_equal(fast_v4, slow_v4)
    assert np.array_equal(fast_v3, slow_v3)
    assert not np.array_equal(fast_native, slow_native)
    np.testing.assert_array_equal(fast_native, fast_v4)
    np.testing.assert_array_equal(slow_native, slow_v4)

    phase = OBS_DIM - BLOCK_PHASE_CONTEXT
    value_obs = encode_observation(fast, 0, None)
    assert np.array_equal(value_obs[phase + 9 : phase + 17], fast_v4[phase + 9 : phase + 17])
    assert fast_v4[phase + 10] > 0.0

    phases.step_check_corner(fast, fast.players[0])
    phases.step_check_corner(slow, slow.players[0])
    assert fast.players[0].spun_out
    assert not slow.players[0].spun_out


def test_unknown_codec_is_rejected() -> None:
    track = _track(20, 1, [Corner(start=4, end=5, speed_limit=2)])
    state = GameState.create(track, 2, seed=0)
    with pytest.raises(ValueError, match="codec_version"):
        encode_observation(state, 0, None, codec_version=2)


def test_loaded_v3_policy_keeps_its_codec() -> None:
    """A checkpoint labeled codec 3 must be shown v3 inputs after it loads."""
    policy = HeatPolicy(hidden_sizes=(8,))
    assert policy.codec_version == 4
    track = _track(83, 2, [Corner(start=10, end=12, speed_limit=2)])
    state = GameState.create(track, 2, seed=1)
    state.players[0].lap = track.laps
    state.players[0].position = 0
    with TemporaryDirectory() as folder:
        path = Path(folder) / "policy.pt"
        save_policy(policy, A0Config(hidden_sizes=(8,)), path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob["codec_version"] = 3
        torch.save(blob, path)
        loaded = load_policy(path)
        assert loaded.codec_version == 3
        shown = encode_observation(
            state, 0, None, codec_version=loaded.codec_version
        )
        assert np.array_equal(
            shown, encode_observation(state, 0, None, codec_version=3)
        )
        assert not np.array_equal(shown, encode_observation(state, 0, None))
        blob["codec_version"] = 2
        torch.save(blob, path)
        with pytest.raises(CheckpointMismatchError, match="codec_version"):
            load_policy(path)
