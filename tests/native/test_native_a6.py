"""Bounded worker-pool gates for Direction D3 chunk A6."""

from __future__ import annotations

import time

import numpy as np
import pytest

from heat.ml.native_env.bridge import legacy_state_to_native
from heat.models.game_state import GameState
from heat.tracks.generator import generate_track
from heat_native import NativePool


def _native_state(game_id: int) -> object:
    """Create one real compact game state for pool lifecycle tests."""
    state = GameState.create(
        generate_track(20_000 + game_id),
        2 + game_id % 5,
        seed=30_000 + game_id,
    )
    return legacy_state_to_native(state, game_id=game_id).native_handle


def _pool(workers: int, slots: int = 16) -> NativePool:
    """Build a bounded pool with the A0 observation-buffer contract."""
    return NativePool(
        {
            "row_capacity": 8,
            "worker_count": workers,
            "slot_capacity": slots,
            "queue_capacity": slots,
        },
        np.zeros((8, 104), dtype=np.float32),
    )


def _run(workers: int, order: list[int]) -> dict[int, int]:
    """Admit a perturbed manifest and return canonical per-game digests."""
    pool = _pool(workers)
    try:
        pool.admit(
            [
                {
                    "state": _native_state(game_id),
                    "work_units": (game_id * 17) % 20_000,
                }
                for game_id in order
            ]
        )
        rows: list[dict[str, int]] = []
        deadline = time.monotonic() + 10.0
        while len(rows) < len(order) and time.monotonic() < deadline:
            rows.extend(pool.wait_completed(len(order), 100))
        assert len(rows) == len(order)
        return {row["game_id"]: row["digest"] for row in rows}
    finally:
        pool.close(timeout_ms=2_000)


@pytest.mark.parametrize("workers", [1, 2, 4, 8, 12])
def test_worker_count_and_manifest_order_do_not_change_game_digests(
    workers: int,
) -> None:
    """Thread timing, padding-like work, and admission order are semantic noise."""
    game_ids = list(range(200, 212))
    expected = _run(1, game_ids)
    assert _run(workers, list(reversed(game_ids))) == expected


def test_capacity_backpressure_is_atomic_and_slots_are_reused() -> None:
    """Over-admission rejects the whole manifest and finalization frees slots."""
    pool = _pool(2, slots=2)
    try:
        with pytest.raises(ValueError, match="slot capacity"):
            pool.admit([{"state": _native_state(game_id)} for game_id in range(3)])
        assert pool.stats()["admitted"] == 0

        pool.admit([{"state": _native_state(300)}, {"state": _native_state(301)}])
        rows = pool.wait_completed(2, 5_000)
        assert len(rows) == 2
        pool.admit([{"state": _native_state(302)}])
        assert pool.wait_completed(1, 5_000)[0]["game_id"] == 302
    finally:
        pool.close(timeout_ms=2_000)


def test_injected_worker_fault_stops_pool_and_is_propagated() -> None:
    """A native worker failure becomes a pool-wide trainer-visible error."""
    pool = _pool(2)
    pool.admit(
        [
            {"state": _native_state(400), "inject_fault": True},
            {"state": _native_state(401), "work_units": 10_000},
        ]
    )
    with pytest.raises(RuntimeError, match="injected native worker fault"):
        pool.wait_completed(2, 5_000)
    assert pool.stats()["faulted"] is True
    pool.close(timeout_ms=2_000)
    assert pool.closed


def test_stop_interrupts_bounded_work_and_joins_all_workers() -> None:
    """Close cooperatively cancels long jobs instead of waiting for completion."""
    pool = _pool(4)
    pool.admit(
        [
            {"state": _native_state(500 + index), "work_units": 2**32 - 1}
            for index in range(4)
        ]
    )
    started = time.monotonic()
    pool.close(timeout_ms=2_000)
    assert time.monotonic() - started < 2.0
    assert pool.closed
