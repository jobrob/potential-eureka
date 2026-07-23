"""Chunk-1 shape, bridge, round-trip, and capacity gates for Direction D2."""

from __future__ import annotations

import pytest

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState, Phase
from heat.models.track import Space, Track
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.ml.vector_env.bridge import (
    legacy_states_to_tensor,
    tensor_semantic_snapshot,
)
from heat.ml.vector_env.state import (
    MAX_CARDS_PER_ZONE,
    MAX_TRACK_SPACES,
    TensorStateCapacityError,
)
from heat.tracks.generator import generate_track


def _generated_state(seed: int, seats: int) -> GameState:
    """Build a started full-rules state on a reproducible generated track."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 10_000)
    for player in state.players:
        player.lap = 1
    return state


def test_generated_tracks_round_trip_exactly_across_two_to_six_seats() -> None:
    """The bridge preserves every D0 oracle field across the supported field sizes."""
    states = [_generated_state(700 + seats, seats) for seats in range(2, 7)]
    game_ids = [20_000 + seats for seats in range(2, 7)]

    tensor_state = legacy_states_to_tensor(states, game_ids=game_ids)

    assert tensor_state.batch_size == 5
    for game_id, legacy_state in zip(game_ids, states, strict=True):
        assert tensor_semantic_snapshot(tensor_state, game_id) == canonical_game_state(
            legacy_state
        )


def test_round_trip_preserves_transient_state_ordered_cards_and_rng_cache() -> None:
    """Mid-turn fields needed by later kernels and resume survive without loss."""
    state = _generated_state(811, 2)
    state.round_num = 17
    state.current_phase = Phase.REACT
    state.turn_order = [1, 0]
    state._stress_counter = 9
    state.rng.gauss(0.0, 1.0)  # Populate Python Random's optional cached value.

    player = state.players[0]
    player.gear = 4
    player.position = 38
    player.lap = 2
    player.spun_out = True
    player.spin_log = [(4, 12), (16, 31)]
    player.boost_used_this_turn = True
    player.speed_from_cards = 7
    player.speed_from_boost = 3
    player.speed_from_adrenaline = 1
    player.slipstream_moved = 2
    player.cluttered = True
    player.turn_start_position = 30
    player.turn_start_lap = 2
    player.cooldown_pool.append(player.heat_pool.pop())
    drawn = player.deck.draw(100)
    player.hand.extend(drawn[:4])
    player.deck.discard(drawn[4:])
    player.cards_played = [player.hand.pop(), player.hand.pop()]

    state.players[1].finished = True
    state.players[1].finish_order = 1

    tensor_state = legacy_states_to_tensor([state], game_ids=[811])

    assert tensor_semantic_snapshot(tensor_state, 811) == canonical_game_state(state)
    assert tensor_state.game_active.tolist() == [True]
    assert tensor_state.player_active.tolist() == [[True, False, False, False, False, False]]


@pytest.mark.parametrize("overflow", ["track", "hand"])
def test_bridge_rejects_capacity_overflow_instead_of_clipping(overflow: str) -> None:
    """Frozen state shapes fail clearly when an input cannot fit."""
    if overflow == "track":
        length = MAX_TRACK_SPACES + 1
        track = Track(
            "too-wide",
            [Space(index) for index in range(length)],
            [],
            list(range(6)),
            laps=2,
        )
        state = GameState.create(track, 2, seed=1)
        expected = "track spaces"
    else:
        state = _generated_state(812, 2)
        state.players[0].hand = [
            Card(CardType.SPEED, 1, f"overflow-{index}")
            for index in range(MAX_CARDS_PER_ZONE + 1)
        ]
        expected = "hand"

    with pytest.raises(TensorStateCapacityError, match=expected):
        legacy_states_to_tensor([state], game_ids=[1])


def test_bridge_requires_unique_stable_game_identities() -> None:
    """Ambiguous lane identities are rejected before any tensor is built."""
    state = _generated_state(813, 2)
    with pytest.raises(ValueError, match="unique stable identities"):
        legacy_states_to_tensor([state, state], game_ids=[4, 4])
