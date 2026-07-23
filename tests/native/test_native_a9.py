"""End-to-end native collector integration gates for Direction D3 A9."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from heat.ml.native_env.collector import NativeCollector, NativeCollectorState
from heat.ml.selfplay.phase1 import A8Config, A8ResumeConfig, train_selfplay_a8
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.snapshots import SnapshotAgent
from heat.ml.selfplay.tiny_heat import tiny_heat_track
from heat.ml.selfplay.training_state import training_state_digest


def _policy(seed: int = 901) -> tuple[A8Config, object]:
    """Build one deterministic small actor-critic for native gates."""
    config = A8Config(hidden_sizes=(16,), device="cpu", seed=seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = build_policy(config)
    return config, policy


def test_native_collection_is_exact_under_repeated_manifest_execution() -> None:
    """Identical keyed manifests produce byte-identical canonical PPO tensors."""
    config, policy = _policy()

    def collect() -> tuple[dict[str, torch.Tensor], NativeCollectorState]:
        state = NativeCollectorState(rows_per_game_ema=10.0)
        collector = NativeCollector(
            tiny_heat_track(),
            3,
            state=state,
            worker_count=2,
            ready_capacity=12,
            sampling_seed=12345,
            rows_ema_alpha=0.25,
        )
        batch, _returns = collector.collect(
            policy, 64, torch.device("cpu"), np.random.default_rng(77),
            gamma=config.gamma, gae_lambda=config.gae_lambda, policy_version=9,
        )
        assert collector.last_games_collected >= 2
        assert collector.last_pool_stats["faulted"] is False
        assert collector.last_coordinator_stats["rows"] >= len(batch["actions"])
        return batch, state

    first, first_state = collect()
    second, second_state = collect()
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)
    assert first_state.export_state() == second_state.export_state()
    assert torch.isfinite(first["advantages"]).all()
    assert torch.isfinite(first["returns"]).all()


def test_native_collector_supports_frozen_snapshot_rows_without_recording_them() -> None:
    """Snapshot actions share batches but only current-policy seats reach PPO."""
    config, policy = _policy(902)
    snapshot = SnapshotAgent(policy, name="native-test-snapshot")
    state = NativeCollectorState(rows_per_game_ema=10.0)
    collector = NativeCollector(
        tiny_heat_track(),
        2,
        state=state,
        worker_count=2,
        ready_capacity=8,
        sampling_seed=54321,
        rows_ema_alpha=0.25,
        scripted_seats={1: snapshot},
    )
    batch, _returns = collector.collect(
        policy, 32, torch.device("cpu"), np.random.default_rng(78),
        gamma=config.gamma, gae_lambda=config.gae_lambda, policy_version=3,
    )
    rows_by_policy = collector.last_coordinator_stats["rows_by_policy"]
    assert rows_by_policy[0] == len(batch["actions"])
    assert rows_by_policy[1] > 0


def test_native_collector_refills_finished_slots_before_the_policy_barrier() -> None:
    """Manifest-v2 replaces early finishers while a larger rollout is open."""
    config, policy = _policy(905)
    state = NativeCollectorState(rows_per_game_ema=100.0)
    collector = NativeCollector(
        tiny_heat_track(),
        3,
        state=state,
        worker_count=4,
        ready_capacity=24,
        sampling_seed=67890,
        rows_ema_alpha=0.25,
    )
    batch, _returns = collector.collect(
        policy,
        2_048,
        torch.device("cpu"),
        np.random.default_rng(79),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        policy_version=4,
    )
    assert len(batch["actions"]) >= 2_048
    assert collector.last_admission_batches > 1
    assert collector.last_refill_games > 0
    assert collector.last_peak_active_slots == 4


def test_native_training_smoke_updates_after_draining_every_admitted_game() -> None:
    """A complete native rollout/update preserves snapshot-controller integration."""
    config = A8Config(
        total_timesteps=64,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2,),
        snapshot_every=1,
        pool_capacity=2,
        pool_prob=1.0,
        collector_mode="native",
        native_workers=2,
        native_ready_capacity=12,
        stage1_enabled=False,
        device="cpu",
        seed=903,
    )
    _policy_result, records = train_selfplay_a8(config, profile=True)
    assert len(records) == 2
    assert records[-1]["opponent_snapshot_iterations"] == 1.0
    assert records[-1]["opponent_current_iterations"] == 1.0
    assert records[-1]["n_recorded"] >= config.n_steps
    assert records[-1]["native_next_game_id"] == records[-1]["total_games"]
    assert records[-1]["mean_action_batch"] >= 1.0


def test_native_safe_boundary_resume_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    """A10 native continuation reproduces the complete checkpoint exactly."""
    config = A8Config(
        total_timesteps=96,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2, 3),
        snapshot_every=1,
        pool_capacity=2,
        pool_prob=1.0,
        collector_mode="native",
        native_workers=2,
        native_ready_capacity=12,
        stage1_enabled=False,
        device="cpu",
        seed=904,
    )
    complete_path = tmp_path / "complete.pt"
    segmented_path = tmp_path / "segmented.pt"
    identity = {
        "campaign_id": "dev-native-a10-resume",
        "source_identity": "test-tree",
    }
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(**identity, save_path=complete_path),
    )
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            **identity,
            save_path=segmented_path,
            max_iterations=1,
        ),
    )
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            **identity,
            load_path=segmented_path,
            save_path=segmented_path,
        ),
    )
    complete = torch.load(complete_path, map_location="cpu", weights_only=False)
    segmented = torch.load(segmented_path, map_location="cpu", weights_only=False)
    assert training_state_digest(complete) == training_state_digest(segmented)
    assert complete["native_collector"] == segmented["native_collector"]
    assert complete["native_runtime_receipt"] == segmented["native_runtime_receipt"]
