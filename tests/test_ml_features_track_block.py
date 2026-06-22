"""Tests for the Sprint B Option-A all-corners ego-centric track block.

Covers the new ``features._track_block`` (codec v2) per the Sprint B design §4:
shape/dtype/bounds on generated tracks, ego-centric ordering, padding
correctness, all-corners coverage, globals correctness, determinism/no-mutation,
and the block-accounting guard (the §3.1 derived-block footgun).
"""

from __future__ import annotations

import random

import numpy as np

from heat.engine import rules
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.ml import features
from heat.ml import spaces as sp
from heat.ml.spaces import OBS_DIM
from heat.tracks.generator import generate_track


# --- Block geometry (symbolic; the test must follow the contract, not hardcode) -

#: Index where the track block starts (sum of the preceding blocks, in packing
#: order: hand, gear, kinematics, deck-composition).
_TRACK_START = (
    sp.BLOCK_HAND_HISTOGRAM
    + sp.BLOCK_OWN_GEAR
    + sp.BLOCK_OWN_KINEMATICS
    + sp.BLOCK_DECK_COMPOSITION
)
_SLOTS_FLOATS = sp.MAX_CORNERS * sp.CORNER_SLOT_FLOATS
_GLOBALS_START = _TRACK_START + _SLOTS_FLOATS


def _track_block_of(vec: np.ndarray) -> np.ndarray:
    return vec[_TRACK_START : _TRACK_START + sp.BLOCK_TRACK]


def _slot(block: np.ndarray, i: int) -> np.ndarray:
    """Corner slot ``i`` (CORNER_SLOT_FLOATS floats) of a track block."""
    lo = i * sp.CORNER_SLOT_FLOATS
    return block[lo : lo + sp.CORNER_SLOT_FLOATS]


def _dist_ahead_col(block: np.ndarray) -> list[float]:
    return [float(_slot(block, i)[0]) for i in range(sp.MAX_CORNERS)]


def _synthetic_track(n_corners: int, laps: int = 2) -> Track:
    """A small deterministic track with exactly ``n_corners`` corners."""
    length = 40
    spaces_ = [Space(index=i, lanes=(2 if i % 5 == 0 else 1)) for i in range(length)]
    corners = []
    # Place non-overlapping single-space corners spread around the loop.
    for k in range(n_corners):
        start = 4 + k * 5
        corners.append(Corner(start=start, end=start, speed_limit=1 + (k % 4) + 1))
    return Track(
        name=f"synth-{n_corners}",
        spaces=spaces_,
        corners=corners,
        start_positions=[0, 1, 2, 3, 4, 5],
        laps=laps,
    )


class TestShapeAndBoundsV2:
    def test_generated_tracks_shape_dtype_bounds(self) -> None:
        rng = random.Random(0)
        for seed in range(40):
            track = generate_track(seed)
            state = GameState.create(track, 4, seed=seed)
            for p in state.players:
                p.position = rng.randint(0, track.length - 1)
                p.lap = rng.randint(1, track.laps)
            for pid in range(state.num_players):
                vec = features.encode_observation(state, pid, None)
                assert vec.shape == (OBS_DIM,)
                assert vec.dtype == np.float32
                assert np.all(vec >= -1.0)
                assert np.all(vec <= 1.0)
                assert not np.any(np.isnan(vec))


class TestEgoCentricOrdering:
    def test_dist_ahead_monotone_across_populated_slots(self) -> None:
        track = generate_track(7)
        n = len(track.corners)
        state = GameState.create(track, 2, seed=7)
        player = state.get_player(0)
        for pos in (0, 3, track.length // 2, track.length - 1):
            player.position = pos
            vec = features.encode_observation(state, 0, None)
            block = _track_block_of(vec)
            dists = _dist_ahead_col(block)
            populated = dists[:n]
            # Non-decreasing forward distance across the populated corner slots.
            for a, b in zip(populated, populated[1:]):
                assert a <= b + 1e-6
            # Padding slots (if any) are exactly zero in dist_ahead.
            for d in dists[n:]:
                assert d == 0.0

    def test_advancing_past_nearest_corner_rotates_slots(self) -> None:
        track = _synthetic_track(3)
        state = GameState.create(track, 2, seed=1)
        player = state.get_player(0)

        player.position = 0
        before = _dist_ahead_col(_track_block_of(
            features.encode_observation(state, 0, None)
        ))
        # Nearest corner is at start=4 -> dist 4. Advance just past it.
        player.position = 6
        after = _dist_ahead_col(_track_block_of(
            features.encode_observation(state, 0, None)
        ))
        # Slot 0's dist_ahead must change (the previous nearest corner is now the
        # farthest, ~a full lap away), i.e. the ego-centric window rotated.
        assert before[0] != after[0]


class TestPaddingAndCoverage:
    def test_fewer_corners_zero_pads_tail(self) -> None:
        track = _synthetic_track(3)
        state = GameState.create(track, 2, seed=2)
        block = _track_block_of(features.encode_observation(state, 0, None))
        # Slots 0..2 populated (dist_ahead > 0), slots 3..MAX-1 fully zero.
        for i in range(3):
            assert _slot(block, i)[0] > 0.0
        for i in range(3, sp.MAX_CORNERS):
            assert np.all(_slot(block, i) == 0.0)

    def test_seven_corner_track_fills_seven_pads_one(self) -> None:
        # The generator caps at 7 corners; find a generated 7-corner track.
        seven = None
        for seed in range(200):
            t = generate_track(seed)
            if len(t.corners) == 7:
                seven = t
                break
        assert seven is not None, "expected a 7-corner generated track"
        state = GameState.create(seven, 2, seed=99)
        block = _track_block_of(features.encode_observation(state, 0, None))
        for i in range(7):
            assert _slot(block, i)[0] > 0.0
        # MAX_CORNERS == 8, so exactly one padding slot remains.
        assert np.all(_slot(block, 7) == 0.0)

    def test_all_corners_appear_exactly_once(self) -> None:
        track = generate_track(11)
        n = len(track.corners)
        state = GameState.create(track, 2, seed=11)
        player = state.get_player(0)
        player.position = 0
        block = _track_block_of(features.encode_observation(state, 0, None))

        # The dist_ahead of each populated slot, de-normalized back to a forward
        # space count, must match the multiset of forward distances of the real
        # corners (every corner covered once, none dropped/duplicated).
        length = track.length
        expected = sorted(
            (length if (c.start - 0) % length == 0 else (c.start - 0) % length)
            for c in track.corners
        )
        got = sorted(
            round(float(_slot(block, i)[0]) * length) for i in range(n)
        )
        assert got == expected


class TestGlobals:
    def test_globals_layout(self) -> None:
        track = _synthetic_track(3, laps=3)
        state = GameState.create(track, 2, seed=5)
        player = state.get_player(0)
        player.lap = 1
        player.position = 10
        vec = features.encode_observation(state, 0, None)
        g = vec[_GLOBALS_START : _GLOBALS_START + sp.TRACK_GLOBALS]
        laps_remaining, dist_to_finish, heat_pool, pos_in_lap = (float(x) for x in g)

        length = track.length
        laps = track.laps
        assert abs(laps_remaining - (laps - player.lap) / laps) < 1e-6
        total = length * laps
        abs_pos = player.lap * length + player.position
        assert abs(dist_to_finish - (total - abs_pos) / total) < 1e-6
        assert abs(heat_pool - player.heat_available / rules.HEAT_POOL_SIZE) < 1e-6
        assert abs(pos_in_lap - player.position / length) < 1e-6

    def test_laps_remaining_decreases_with_lap(self) -> None:
        track = _synthetic_track(3, laps=3)
        state = GameState.create(track, 2, seed=6)
        player = state.get_player(0)

        def laps_remaining_at(lap: int) -> float:
            player.lap = lap
            vec = features.encode_observation(state, 0, None)
            return float(vec[_GLOBALS_START])

        assert laps_remaining_at(1) > laps_remaining_at(2) >= laps_remaining_at(3)

    def test_dist_to_finish_near_zero_at_final_lap_end(self) -> None:
        track = _synthetic_track(3, laps=2)
        state = GameState.create(track, 2, seed=8)
        player = state.get_player(0)
        player.lap = track.laps  # on the final lap
        player.position = track.length - 1
        vec = features.encode_observation(state, 0, None)
        dist_to_finish = float(vec[_GLOBALS_START + 1])
        assert dist_to_finish < 1.0 / track.length + 1e-6


class TestDeterminismAndNoMutation:
    def test_deterministic_and_pure(self) -> None:
        track = generate_track(3)
        state = GameState.create(track, 4, seed=3)
        import copy

        snapshot = copy.deepcopy(state)
        a = features.encode_observation(state, 0, None)
        b = features.encode_observation(state, 0, None)
        assert np.array_equal(a, b)
        # No mutation: every player's public state is unchanged.
        for before, after in zip(snapshot.players, state.players):
            assert before.position == after.position
            assert before.lap == after.lap
            assert before.heat_available == after.heat_available


class TestBlockAccounting:
    def test_phase_context_is_nineteen(self) -> None:
        assert sp.BLOCK_PHASE_CONTEXT == 19

    def test_obs_dim_equals_block_sum(self) -> None:
        total = (
            sp.BLOCK_HAND_HISTOGRAM
            + sp.BLOCK_OWN_GEAR
            + sp.BLOCK_OWN_KINEMATICS
            + sp.BLOCK_DECK_COMPOSITION
            + sp.BLOCK_TRACK
            + sp.BLOCK_ADRENALINE_CONTEXT
            + sp.BLOCK_OPPONENT_SLOTS
            + sp.BLOCK_PHASE_CONTEXT
        )
        assert total == OBS_DIM == 104

    def test_block_track_constants(self) -> None:
        assert sp.MAX_CORNERS == 8
        assert sp.CORNER_SLOT_FLOATS == 4
        assert sp.TRACK_GLOBALS == 4
        assert sp.BLOCK_TRACK == 36

    def test_codec_version_bumped(self) -> None:
        # v3 (Option C prep, code-review 2026-06-22 #1): round_num is now encoded
        # unconditionally so the observation is a pure function of game state.
        assert sp.CODEC_VERSION == 3

    def test_corner_len_normalized_by_max_corner_len(self) -> None:
        # Freeze the corner_len normalization choice (/ max_corner_len, relative).
        # A track whose longest corner spans 3 spaces: that corner's corner_len
        # field reads 1.0; a 1-space corner reads 1/3.
        length = 40
        spaces_ = [Space(index=i, lanes=1) for i in range(length)]
        corners = [
            Corner(start=4, end=4, speed_limit=2),   # len 1
            Corner(start=12, end=14, speed_limit=2),  # len 3 (the max)
        ]
        track = Track(
            name="clen", spaces=spaces_, corners=corners,
            start_positions=[0, 1, 2, 3, 4, 5], laps=1,
        )
        state = GameState.create(track, 2, seed=0)
        player = state.get_player(0)
        player.position = 0
        block = _track_block_of(features.encode_observation(state, 0, None))
        # Ordered by forward distance from pos 0: corner@4 (len 1) then corner@12
        # (len 3). corner_len is the 3rd float (index 2) of each slot.
        assert abs(float(_slot(block, 0)[2]) - 1.0 / 3.0) < 1e-6
        assert abs(float(_slot(block, 1)[2]) - 1.0) < 1e-6
