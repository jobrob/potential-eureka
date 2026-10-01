"""CPU batched policy and keyed-sampling gates for Direction D3 A7."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from heat.ml.native_env.coordinator import (
    CpuPolicyCoordinator,
    PolicySubmission,
    ReadyBatch,
    RegisteredPolicy,
    keyed_sampling_word,
)
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.spaces import ACTION_DIM, OBS_DIM


def _policies() -> tuple[RegisteredPolicy, ...]:
    """Create deterministic current and snapshot policy entries."""
    with torch.random.fork_rng():
        torch.manual_seed(700)
        current = HeatPolicy(hidden_sizes=(16,))
        torch.manual_seed(701)
        snapshot = HeatPolicy(hidden_sizes=(16,))
    return (
        RegisteredPolicy(3, 17, current),
        RegisteredPolicy(9, 4, snapshot),
    )


def _ready(
    order: np.ndarray | None = None, *, value_only: bool = False
) -> ReadyBatch:
    """Build real-shape rows with stable identities and contiguous groups."""
    rows = 12
    rng = np.random.default_rng(702)
    observations = rng.normal(size=(rows, OBS_DIM)).astype(np.float32)
    masks = rng.random((rows, ACTION_DIM)) < 0.15
    masks[:, 0] = True
    arrays: tuple[np.ndarray, ...] = (
        observations,
        masks,
        np.repeat(np.arange(6, dtype=np.uint64), 2),
        np.full(rows, 2, dtype=np.uint16),
        np.arange(100, 100 + rows, dtype=np.uint64),
        (np.arange(rows) % 6).astype(np.int16),
        np.arange(20, 20 + rows, dtype=np.uint32),
        np.where(np.arange(rows) % 3, 3, 9).astype(np.uint16),
        np.where(np.arange(rows) % 3, 17, 4).astype(np.uint32),
        np.zeros(rows, dtype=np.bool_),
    )
    if value_only:
        arrays[-1][-1] = True
        arrays[1][-1] = False
    if order is not None:
        arrays = tuple(np.ascontiguousarray(array[order]) for array in arrays)
    return ReadyBatch(*arrays, lease_token=55)  # type: ignore[arg-type]


def _by_identity(
    ready: ReadyBatch, submission: PolicySubmission
) -> dict[tuple[int, int, int], tuple[int, float, float, int]]:
    """Map aligned outputs by immutable decision identity."""
    return {
        (
            int(ready.game_ids[index]),
            int(ready.seat_ids[index]),
            int(ready.decision_sequences[index]),
        ): (
            int(submission.actions[index]),
            float(submission.logps[index]),
            float(submission.values[index]),
            int(submission.sampling_words[index]),
        )
        for index in range(ready.row_count)
    }


def test_keyed_words_have_a_frozen_cross_platform_receipt() -> None:
    """The named unsigned-overflow hash cannot drift silently."""
    assert keyed_sampling_word(123, 3, 17, 100, 4, 20) == 1_925_020_195
    assert keyed_sampling_word(123, 9, 4, 999, 1, 88) == 706_065_512


def test_actions_are_independent_of_batch_order_and_policy_partition() -> None:
    """Reordering groups changes neither actions nor actor-critic outputs."""
    ready = _ready()
    first = CpuPolicyCoordinator(_policies(), sampling_seed=123).dispatch(ready)
    order = np.arange(12).reshape(6, 2)[::-1].reshape(-1)
    reordered = _ready(order)
    second = CpuPolicyCoordinator(_policies(), sampling_seed=123).dispatch(reordered)
    assert _by_identity(ready, first) == _by_identity(reordered, second)
    assert np.all(ready.legal_masks[np.arange(12), first.actions])


def test_value_only_rows_and_real_batch_telemetry() -> None:
    """Bootstrap rows skip sampling while sharing one value batch."""
    ready = _ready(value_only=True)
    original_masks = ready.legal_masks.copy()
    coordinator = CpuPolicyCoordinator(_policies(), sampling_seed=456)
    submission = coordinator.dispatch(ready)
    np.testing.assert_array_equal(ready.legal_masks, original_masks)
    assert submission.actions[-1] == -1
    assert submission.logps[-1] == 0.0
    assert submission.sampling_words[-1] == 0
    assert submission.lease_token == 55
    assert np.isfinite(submission.values).all()
    stats = coordinator.stats()
    assert stats["dispatch_calls"] == 1
    assert stats["inference_calls"] == 2
    assert stats["rows"] == 12
    assert stats["rows_by_policy"] == {3: 8, 9: 4}


def test_mutated_policy_version_is_rejected() -> None:
    """An admitted iteration cannot cross a policy-version barrier."""
    ready = _ready()
    versions = ready.policy_versions.copy()
    versions[0] += 1
    mutated = ReadyBatch(
        ready.observations,
        ready.legal_masks,
        ready.group_tokens,
        ready.group_sizes,
        ready.game_ids,
        ready.seat_ids,
        ready.decision_sequences,
        ready.policy_ids,
        versions,
        ready.value_only,
        ready.lease_token,
    )
    with pytest.raises(ValueError, match="policy version mismatch"):
        CpuPolicyCoordinator(_policies(), sampling_seed=1).dispatch(mutated)
