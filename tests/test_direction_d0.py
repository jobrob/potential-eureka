"""Focused correctness gates for Direction D0's contracts and resume path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heat.engine.driver import run_round_driver
from heat.models.game_state import GameState
from heat.ml.selfplay.phase1 import (
    A8Config,
    A8ResumeConfig,
    SeatCountSchedule,
    train_selfplay_a8,
)
from heat.ml.selfplay.semantic_contract import canonical_decision_row
from heat.ml.selfplay.tiny_heat import tiny_heat_track
from heat.ml.selfplay.training_state import (
    TrainingStateError,
    load_training_state,
    recipe_sha256,
    resolved_recipe,
    training_state_digest,
)
from heat.tracks.generator import TrackGenParams


def _resume_recipe(iterations: int = 3) -> A8Config:
    """Return a tiny full-rules recipe that still exercises pool and schedule state."""
    return A8Config(
        total_timesteps=iterations * 32,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2, 3),
        snapshot_every=1,
        pool_capacity=2,
        pool_prob=0.5,
        stage1_enabled=False,
        device="cpu",
        seed=23,
    )


def _load(path: Path, config: A8Config, *, source: str = "test-source") -> dict:
    """Load a test checkpoint through the production compatibility contract."""
    recipe = resolved_recipe(config, TrackGenParams())
    return load_training_state(
        path,
        expected_recipe_sha256=recipe_sha256(recipe),
        campaign_id="dev-d0-test",
        source_identity=source,
        anchor_identity=None,
    )


def test_seat_schedule_round_trip_preserves_next_work() -> None:
    """A saved partial balance block resumes with the exact next seat count."""
    original = SeatCountSchedule((2, 3, 4), np.random.default_rng(7))
    assert original.next() in {2, 3, 4}
    state = original.export_state()
    expected = [original.next() for _ in range(8)]

    restored = SeatCountSchedule((2, 3, 4), np.random.default_rng(999))
    restored.restore_state(state)
    assert [restored.next() for _ in range(8)] == expected


def test_semantic_decision_receipt_is_seed_exact() -> None:
    """Identical scalar states yield byte-equivalent decision receipts."""
    rows = []
    for _ in range(2):
        state = GameState.create(tiny_heat_track(), 2, seed=11)
        for player in state.players:
            player.lap = 1
        decision = next(run_round_driver(state))
        rows.append(
            canonical_decision_row(
                state, decision, game_id=0, decision_index=0
            )
        )
    assert rows[0] == rows[1]
    assert len(rows[0]["observation"]) > 0
    assert any(rows[0]["legal_mask"])


def test_a8_safe_boundary_resume_matches_uninterrupted_complete_state(
    tmp_path: Path,
) -> None:
    """One stop restores policy, optimizer, pool, diagnostics, RNG, and schedule."""
    config = _resume_recipe()
    control_path = tmp_path / "control.pt"
    segmented_path = tmp_path / "segmented.pt"
    common = {
        "campaign_id": "dev-d0-test",
        "source_identity": "test-source",
    }

    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(**common, save_path=control_path),
    )
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            **common,
            save_path=segmented_path,
            max_iterations=1,
        ),
    )
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            **common,
            load_path=segmented_path,
            save_path=segmented_path,
        ),
    )

    control = _load(control_path, config)
    segmented = _load(segmented_path, config)
    assert training_state_digest(control) == training_state_digest(segmented)
    assert control["progress"]["iteration"] == 3
    assert control["progress"]["next_iteration"] == 4
    assert len(control["snapshot_pool"]["entries"]) == 2


def test_training_state_rejects_damaged_bytes_and_source_mismatch(
    tmp_path: Path,
) -> None:
    """Receipts and provenance tripwires reject invalid state before continuation."""
    config = _resume_recipe(iterations=1)
    path = tmp_path / "state.pt"
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            campaign_id="dev-d0-test",
            source_identity="test-source",
            save_path=path,
        ),
    )
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(TrainingStateError, match="SHA-256 mismatch"):
        _load(path, config)

    # Re-create a valid checkpoint for the independent provenance tripwire.
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            campaign_id="dev-d0-test",
            source_identity="test-source",
            save_path=path,
        ),
    )
    with pytest.raises(TrainingStateError, match="source_identity"):
        _load(path, config, source="different-source")


def test_graceful_stop_saves_the_requested_safe_boundary(tmp_path: Path) -> None:
    """A bounded run saves the exact iteration that requested the stop."""
    config = _resume_recipe(iterations=4)
    path = tmp_path / "stopped.pt"

    _policy, records = train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            campaign_id="dev-d0-test",
            source_identity="test-source",
            save_path=path,
        ),
        should_stop=lambda iteration, _record: iteration == 2,
    )

    state = _load(path, config)
    assert len(records) == 2
    assert state["progress"]["iteration"] == 2
    assert state["progress"]["next_iteration"] == 3
