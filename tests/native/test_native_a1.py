"""Compact-state and CPython-RNG gates for Direction D3 chunk A1."""

from __future__ import annotations

import random

import pytest

from heat.ml.native_env.bridge import legacy_state_to_native
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.models.game_state import GameState, Phase
from heat.tracks.generator import generate_track
from heat_native._core import _PythonMt19937


def _generated_state(seed: int, seats: int) -> GameState:
    """Create a reproducible full-rules state within the frozen capacities."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 10_000)
    for player in state.players:
        player.lap = 1
    return state


@pytest.mark.parametrize("seats", range(2, 7))
def test_generated_states_round_trip_to_exact_d0_snapshots(seats: int) -> None:
    """Every supported seat count survives the compact native representation."""
    state = _generated_state(3_000 + seats, seats)
    bridge = legacy_state_to_native(state, game_id=40_000 + seats)
    assert bridge.semantic_snapshot() == canonical_game_state(state)
    receipt = bridge.receipt()
    assert receipt["card_token_bytes"] == 4
    assert receipt["players_capacity"] == 6
    assert receipt["track_space_capacity"] == 90


def test_transient_fields_ordered_cards_and_gauss_cache_round_trip() -> None:
    """A mid-turn state retains ordered zones, flags, spin history, and RNG cache."""
    state = _generated_state(3_100, 3)
    state.round_num = 73
    state.current_phase = Phase.REACT
    state.turn_order = [2, 0, 1]
    state._stress_counter = 123
    state.rng.gauss(0.0, 1.0)
    player = state.players[0]
    player.gear = 4
    player.position = 38
    player.lap = 2
    player.spun_out = True
    player.spin_log = [(4, 12), (70, 31)]
    player.boost_used_this_turn = True
    player.speed_from_cards = 7
    player.speed_from_boost = 3
    player.speed_from_adrenaline = 1
    player.slipstream_moved = 2
    player.cluttered = True
    player.turn_start_position = 30
    player.turn_start_lap = 2
    player.cooldown_pool.append(player.heat_pool.pop())
    drawn = player.deck.draw(10)
    player.hand.extend(drawn[:3])
    player.deck.discard(drawn[3:])
    player.cards_played = [player.hand.pop()]

    bridge = legacy_state_to_native(state, game_id=41_000)
    assert bridge.semantic_snapshot() == canonical_game_state(state)


@pytest.mark.parametrize("seed", [0, 1, 811, 2**32 - 1])
def test_cpython_mt19937_words_floats_widths_shuffle_and_state(seed: int) -> None:
    """Native chance consumes the exact CPython 3.12 MT19937 stream."""
    python_rng = random.Random(seed)
    native_rng = _PythonMt19937(python_rng.getstate())

    for width in (0, 1, 5, 31, 32, 33, 64, 65, 127):
        assert native_rng.getrandbits(width) == python_rng.getrandbits(width)
    for _ in range(8):
        assert native_rng.random() == python_rng.random()
    for bound in (1, 2, 3, 7, 16, 31, 90, 2**32 + 15):
        assert native_rng.randbelow(bound) == python_rng._randbelow(bound)
    for length in (0, 1, 2, 3, 7, 16, 24, 90):
        expected = list(range(length))
        python_rng.shuffle(expected)
        assert native_rng.shuffled(length) == expected
    assert native_rng.getstate() == python_rng.getstate()
