"""Determinism checks available at Direction D2's tensor-state boundary."""

from __future__ import annotations

from io import BytesIO

import torch

from heat.models.game_state import GameState
from heat.ml.vector_env.bridge import (
    legacy_states_to_tensor,
    tensor_semantic_snapshot,
)
from heat.ml.vector_env.state import TensorGameState
from heat.tracks.generator import generate_track


def _state(seed: int, seats: int = 3) -> GameState:
    """Build a deterministic started state for batch-independence checks."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 50_000)
    for player in state.players:
        player.lap = 1
    return state


def _assert_tensor_state_equal(left: TensorGameState, right: TensorGameState) -> None:
    """Compare all tensor fields and immutable identity metadata exactly."""
    assert left.track_names == right.track_names
    assert left.player_names == right.player_names
    assert left.card_id_vocabulary == right.card_id_vocabulary
    left_fields = dict(left.tensor_fields())
    right_fields = dict(right.tensor_fields())
    assert left_fields.keys() == right_fields.keys()
    for name in left_fields:
        assert torch.equal(left_fields[name], right_fields[name]), name


def test_same_seed_packs_to_identical_tensor_state() -> None:
    """Repeated construction has no hidden global or unseeded bridge input."""
    first = legacy_states_to_tensor([_state(901)], game_ids=[91])
    second = legacy_states_to_tensor([_state(901)], game_ids=[91])
    _assert_tensor_state_equal(first, second)


def test_game_snapshot_is_independent_of_batch_size_and_lane_order() -> None:
    """Padding width, neighboring games, and lane position cannot alter a game."""
    focal = _state(902)
    expected = tensor_semantic_snapshot(
        legacy_states_to_tensor([focal], game_ids=[7]), 7
    )

    for batch_size in (1, 8, 32, 256):
        ids = list(range(1_000, 1_000 + batch_size))
        packed = legacy_states_to_tensor([focal] * batch_size, game_ids=ids)
        assert tensor_semantic_snapshot(packed, ids[-1]) == expected

    neighbors = [_state(903, 2), focal, _state(904, 6)]
    ordered = legacy_states_to_tensor(neighbors, game_ids=[30, 7, 90])
    reordered = legacy_states_to_tensor(
        [neighbors[2], neighbors[0], neighbors[1]], game_ids=[90, 30, 7]
    )
    assert tensor_semantic_snapshot(ordered, 7) == expected
    assert tensor_semantic_snapshot(reordered, 7) == expected


def test_finished_lane_does_not_change_an_active_neighbor() -> None:
    """An inactive lane changes only its own active masks and stored state."""
    focal = _state(905)
    finished = _state(906, 2)
    for order, player in enumerate(finished.players, start=1):
        player.finished = True
        player.finish_order = order

    alone = legacy_states_to_tensor([focal], game_ids=[5])
    together = legacy_states_to_tensor([finished, focal], game_ids=[6, 5])

    assert together.game_active.tolist() == [False, True]
    assert tensor_semantic_snapshot(together, 5) == tensor_semantic_snapshot(alone, 5)


def test_saved_tensor_state_resumes_with_identical_snapshot() -> None:
    """All Chunk-1 state, including Python RNG words, survives save and reload."""
    original = legacy_states_to_tensor([_state(907, 4)], game_ids=[907])
    payload = BytesIO()
    torch.save(original, payload)
    payload.seek(0)
    restored = torch.load(payload, weights_only=False)

    assert isinstance(restored, TensorGameState)
    restored.validate()
    _assert_tensor_state_equal(original, restored)
    assert tensor_semantic_snapshot(restored, 907) == tensor_semantic_snapshot(
        original, 907
    )
