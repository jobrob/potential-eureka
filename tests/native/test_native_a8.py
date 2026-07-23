"""Compact trajectory and canonical GAE gates for Direction D3 A8."""

from __future__ import annotations

import numpy as np
import pytest

from heat.ml.selfplay.buffer import RolloutBuffer
from heat.ml.spaces import ACTION_DIM, OBS_DIM
from heat_native import NativeTrajectoryArena


def _append(
    arena: NativeTrajectoryArena,
    *,
    game_ids: list[int],
    seat_ids: list[int],
    sequences: list[int],
    policy_ids: list[int],
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    recorded: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Append deterministic real-shape rows and return their obs/masks."""
    rows = len(game_ids)
    observations = np.arange(rows * OBS_DIM, dtype=np.float32).reshape(
        rows, OBS_DIM
    )
    masks = np.zeros((rows, ACTION_DIM), dtype=np.bool_)
    actions = (np.arange(rows) * 7 % ACTION_DIM).astype(np.int64)
    masks[np.arange(rows), actions] = True
    arena.append(
        observations,
        masks,
        np.asarray(game_ids, dtype=np.uint64),
        np.asarray(seat_ids, dtype=np.int16),
        np.asarray(sequences, dtype=np.uint32),
        np.asarray(policy_ids, dtype=np.uint16),
        actions,
        np.linspace(-0.2, -0.1, rows, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(dones, dtype=np.bool_),
        np.asarray(recorded, dtype=np.bool_),
    )
    return observations, masks


def _reference_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run the established Python RolloutBuffer oracle for one seat stream."""
    buffer = RolloutBuffer(len(rewards), OBS_DIM, ACTION_DIM)
    for index in range(len(rewards)):
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        mask[0] = True
        buffer.add(
            np.zeros(OBS_DIM, dtype=np.float32),
            0,
            0.0,
            float(values[index]),
            float(rewards[index]),
            bool(dones[index]),
            mask,
        )
    buffer.compute_gae(0.0, 0.99, 0.95)
    return buffer.advantages.copy(), buffer.returns.copy()


def test_canonical_finalization_bootstrap_snapshots_and_gae_are_exact() -> None:
    """Scrambled current/snapshot rows finalize into exact per-seat PPO streams."""
    arena = NativeTrajectoryArena(4, 3)
    observations, masks = _append(
        arena,
        game_ids=[2, 1, 2, 1, 2, 1, 1],
        seat_ids=[0, 1, 0, 0, 0, 1, 0],
        sequences=[1, 0, 0, 0, 2, 1, 1],
        policy_ids=[3, 3, 3, 9, 3, 3, 9],
        rewards=[0.2, 0.2, 0.1, -5.0, 0.3, 0.3, -6.0],
        values=[0.2, 0.5, 0.1, 9.0, 0.3, 0.6, 9.0],
        dones=[False] * 7,
        recorded=[True, True, True, False, True, True, False],
    )
    arena.fold_bootstrap(1, 1, 0.7, 0.99)
    arena.close_stream(2, 0, 1.0, 0.4, 0.25)
    # Frozen snapshot/anchor rows are supported but deliberately not PPO data.
    arena.close_stream(1, 0, -2.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="already closed"):
        arena.fold_bootstrap(1, 1, 0.7, 0.99)

    result = arena.finalize(0.99, 0.95)
    np.testing.assert_array_equal(result["game_ids"], [1, 1, 2, 2, 2])
    np.testing.assert_array_equal(result["seat_ids"], [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(result["decision_sequences"], [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(result["policy_ids"], [3, 3, 3, 3, 3])
    np.testing.assert_array_equal(
        result["obs"], observations[[1, 5, 2, 0, 4]]
    )
    np.testing.assert_array_equal(result["masks"], masks[[1, 5, 2, 0, 4]])

    first_rewards = np.asarray([0.2, 0.3 + 0.99 * 0.7], dtype=np.float32)
    first_values = np.asarray([0.5, 0.6], dtype=np.float32)
    first_dones = np.asarray([0.0, 1.0], dtype=np.float32)
    second_rewards = np.asarray([0.1, 0.2, 0.3 + 1.0 + 0.25 * 0.4], dtype=np.float32)
    second_values = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    second_dones = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    expected_advantages = np.concatenate(
        [
            _reference_gae(first_rewards, first_values, first_dones)[0],
            _reference_gae(second_rewards, second_values, second_dones)[0],
        ]
    )
    expected_returns = np.concatenate(
        [
            _reference_gae(first_rewards, first_values, first_dones)[1],
            _reference_gae(second_rewards, second_values, second_dones)[1],
        ]
    )
    np.testing.assert_array_equal(result["advantages"], expected_advantages)
    np.testing.assert_array_equal(result["returns"], expected_returns)
    assert arena.stats() == {
        "rows": 7,
        "capacity": 12,
        "blocks_used": 2,
        "blocks_total": 3,
        "bootstrap_folds": 1,
        "closed_streams": 2,
    }


def test_arena_capacity_and_incomplete_streams_fail_closed() -> None:
    """The bounded arena never truncates and cannot emit open PPO streams."""
    arena = NativeTrajectoryArena(1, 1)
    _append(
        arena,
        game_ids=[10],
        seat_ids=[0],
        sequences=[0],
        policy_ids=[3],
        rewards=[0.0],
        values=[0.0],
        dones=[False],
        recorded=[True],
    )
    with pytest.raises(ValueError, match="not terminal or bootstrapped"):
        arena.finalize(0.99, 0.95)
    with pytest.raises(ValueError, match="capacity"):
        _append(
            arena,
            game_ids=[11],
            seat_ids=[0],
            sequences=[0],
            policy_ids=[3],
            rewards=[0.0],
            values=[0.0],
            dones=[True],
            recorded=[True],
        )
    arena.reset()
    assert arena.stats()["rows"] == 0
