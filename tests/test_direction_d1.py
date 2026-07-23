"""Focused correctness gates for Direction D1's exact legacy lane substrate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from heat.engine.driver import Decision, RoundDriver, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.ml.action_codec import (
    NO_FORCED,
    decode_legal_action,
    forced_action,
    legal_action_mask,
)
from heat.ml.features import encode_observation
from heat.ml.selfplay.multiseat import CollectorTiming, MultiSeatCollector
from heat.ml.selfplay.phase1 import (
    A8Config,
    A8ResumeConfig,
    train_selfplay_a8,
)
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.ml.selfplay.tiny_heat import tiny_heat_track
from heat.ml.selfplay.training_state import (
    load_training_state,
    recipe_sha256,
    resolved_recipe,
    training_state_digest,
)
from heat.ml.selfplay.vector_collector import ExactLaneCollector
from heat.ml.spaces import ACTION_DIM, OBS_DIM
from heat.ml.vector_env import LegacyLaneEnvironment, ReadyDecision
from heat.tracks.generator import TrackGenParams


@dataclass
class _ScalarOracle:
    """Minimal scalar driver used to compare every D1 policy boundary."""

    state: GameState
    driver: RoundDriver
    send_value: object = None

    @classmethod
    def create(cls, *, seed: int, seats: int) -> _ScalarOracle:
        """Create the same scalar game state used by a legacy lane."""
        state = GameState.create(tiny_heat_track(), seats, seed=seed)
        for player in state.players:
            player.lap = 1
        return cls(state=state, driver=run_round_driver(state))

    def advance(self) -> Decision | None:
        """Advance forced work and return the next live scalar decision."""
        while True:
            terminated = self.state.is_game_over
            truncated = not terminated and self.state.round_num > MAX_ROUNDS
            if terminated or truncated:
                return None
            try:
                decision = self.driver.send(self.send_value)
            except StopIteration:
                terminated = self.state.is_game_over
                truncated = not terminated and self.state.round_num > MAX_ROUNDS
                if terminated or truncated:
                    return None
                self.driver = run_round_driver(self.state)
                self.send_value = None
                continue
            forced = forced_action(decision, self.state)
            if forced is not NO_FORCED:
                self.send_value = forced
                continue
            self.send_value = None
            return decision

    def apply(self, decision: Decision, action: int) -> None:
        """Decode one legal flat action for the next scalar advance call."""
        self.send_value = decode_legal_action(decision, self.state, action)


def _first_legal_actions(masks: np.ndarray) -> np.ndarray:
    """Choose a deterministic legal flat action from each policy row."""
    return np.asarray(
        [int(np.flatnonzero(mask)[0]) for mask in masks], dtype=np.int64
    )


def test_legacy_lanes_batch_mixed_seat_games_in_stable_order() -> None:
    """Mixed lane shapes expose one isolated, deterministic row per game."""
    environment = LegacyLaneEnvironment(tiny_heat_track())
    ready = environment.reset(
        game_ids=(30, 10, 20),
        seeds=(303, 101, 202),
        seat_counts=(6, 2, 4),
    )

    assert [identity.game_id for identity in ready] == [10, 20, 30]
    batch = environment.observe(ready)
    assert batch.observations.shape == (3, OBS_DIM)
    assert batch.legal_masks.shape == (3, ACTION_DIM)
    assert np.all(batch.legal_masks.any(axis=1))

    completed = environment.step(
        ready,
        _first_legal_actions(batch.legal_masks),
        policy_version=7,
    )
    assert all(transition.policy_version == 7 for transition in completed)
    assert [identity.game_id for identity in environment.ready()] == [10, 20, 30]


def test_legacy_lane_matches_scalar_oracle_and_completes_every_action() -> None:
    """Fixed actions preserve scalar state, observations, and transition identity."""
    seed = 9173
    game_id = 41
    environment = LegacyLaneEnvironment(tiny_heat_track())
    environment.reset((game_id,), (seed,), (2,))
    oracle = _ScalarOracle.create(seed=seed, seats=2)
    applied: dict[ReadyDecision, int] = {}
    completed = {}

    while True:
        scalar_decision = oracle.advance()
        ready = environment.ready()
        if scalar_decision is None:
            assert ready == ()
            break

        assert len(ready) == 1
        identity = ready[0]
        assert identity.seat_id == scalar_decision.player_id
        assert environment.semantic_snapshot(game_id) == canonical_game_state(
            oracle.state
        )

        batch = environment.observe(ready)
        expected_observation = encode_observation(
            oracle.state, scalar_decision.player_id, scalar_decision
        )
        expected_mask = legal_action_mask(scalar_decision, oracle.state)
        np.testing.assert_array_equal(batch.observations[0], expected_observation)
        np.testing.assert_array_equal(batch.legal_masks[0], expected_mask)

        action = int(np.flatnonzero(expected_mask)[0])
        applied[identity] = action
        oracle.apply(scalar_decision, action)
        for transition in environment.step(
            ready, np.asarray([action], dtype=np.int64), policy_version=3
        ):
            assert transition.decision not in completed
            completed[transition.decision] = transition

    assert environment.semantic_snapshot(game_id) == canonical_game_state(oracle.state)
    assert completed.keys() == applied.keys()
    for identity, transition in completed.items():
        assert transition.action == applied[identity]
        assert transition.policy_version == 3
        if transition.done:
            assert transition.successor_legal_mask is None
        else:
            assert transition.successor_observation is not None
            assert transition.successor_legal_mask is not None

    assert environment.finished_game_ids() == (game_id,)
    replacement = environment.replace_finished((42,), (seed + 1,), (2,))
    assert len(replacement) == 1
    assert replacement[0].game_id == 42


def test_legacy_lane_rejects_policy_version_crossing_before_mutation() -> None:
    """Pending old-policy spans block a new policy version at the lane boundary."""
    environment = LegacyLaneEnvironment(tiny_heat_track())
    ready = environment.reset((5,), (55,), (2,))
    batch = environment.observe(ready)
    environment.step(
        ready,
        _first_legal_actions(batch.legal_masks),
        policy_version=1,
    )
    next_ready = environment.ready()
    before = environment.semantic_snapshot(5)

    with pytest.raises(ValueError, match="cannot cross policy versions"):
        environment.step(
            next_ready,
            _first_legal_actions(environment.observe(next_ready).legal_masks),
            policy_version=2,
        )

    assert environment.semantic_snapshot(5) == before


def test_lane_collector_records_exact_target_and_computes_fragment_gae() -> None:
    """The bounded cut has no retained overshoot and every buffer is GAE-ready."""
    config = A8Config(hidden_sizes=(16,), device="cpu", seed=13)
    torch.manual_seed(13)
    policy = build_policy(config)
    collector = ExactLaneCollector(tiny_heat_track(), 2, lane_count=4)
    timing = CollectorTiming()

    buffers, returns = collector.collect(
        policy,
        96,
        torch.device("cpu"),
        np.random.default_rng(13),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        policy_version=0,
        timing=timing,
    )

    assert returns == []
    assert sum(len(buffer) for buffer in buffers) == 96
    assert timing.recorded_action_rows == 96
    assert timing.drain_action_rows < 4 * 2 * 5 * 200
    assert timing.action_inference_rows >= 96
    assert max(timing.action_batch_histogram) == 4
    for buffer in buffers:
        batch = buffer.get()
        assert torch.isfinite(batch["advantages"]).all()
        assert torch.isfinite(batch["returns"]).all()


def test_single_lane_complete_game_matches_scalar_trajectory_arrays() -> None:
    """One D1 lane reproduces scalar actions, rewards, and per-seat GAE exactly."""
    config = A8Config(hidden_sizes=(16,), device="cpu", seed=17)
    torch.manual_seed(17)
    policy = build_policy(config)
    scalar = MultiSeatCollector(tiny_heat_track(), 2)

    torch.manual_seed(101)
    scalar_buffers, _returns = scalar.collect(
        policy,
        1,
        torch.device("cpu"),
        np.random.default_rng(101),
        gamma=config.gamma,
    )
    expected_count = sum(len(buffer) for buffer in scalar_buffers)
    for buffer in scalar_buffers:
        if len(buffer):
            buffer.compute_gae(0.0, config.gamma, config.gae_lambda)

    lane = ExactLaneCollector(tiny_heat_track(), 2, lane_count=1)
    torch.manual_seed(101)
    lane_buffers, _returns = lane.collect(
        policy,
        expected_count,
        torch.device("cpu"),
        np.random.default_rng(101),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        policy_version=0,
    )

    expected = [buffer for buffer in scalar_buffers if len(buffer)]
    assert len(lane_buffers) == len(expected)
    for scalar_buffer, lane_buffer in zip(expected, lane_buffers, strict=True):
        scalar_batch = scalar_buffer.get()
        lane_batch = lane_buffer.get()
        assert scalar_batch.keys() == lane_batch.keys()
        for key in scalar_batch:
            torch.testing.assert_close(
                lane_batch[key], scalar_batch[key], rtol=0.0, atol=0.0
            )


def _lane_resume_config() -> A8Config:
    """Return a short D1 recipe that exercises two safe PPO boundaries."""
    return A8Config(
        total_timesteps=64,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2, 3),
        snapshot_every=1,
        pool_capacity=2,
        pool_prob=0.5,
        collector_mode="lanes",
        lane_count=3,
        stage1_enabled=False,
        device="cpu",
        seed=29,
    )


def _load_lane_state(path: Path, config: A8Config) -> dict:
    """Load a D1 checkpoint through the production recipe tripwire."""
    recipe = resolved_recipe(config, TrackGenParams())
    return load_training_state(
        path,
        expected_recipe_sha256=recipe_sha256(recipe),
        campaign_id="dev-d1-test",
        source_identity="d1-test-source",
        anchor_identity=None,
    )


def test_lane_training_resume_matches_uninterrupted_complete_state(
    tmp_path: Path,
) -> None:
    """D1 lanes remain exact across a fresh-process-style safe boundary."""
    config = _lane_resume_config()
    control_path = tmp_path / "control.pt"
    segmented_path = tmp_path / "segmented.pt"
    common = {
        "campaign_id": "dev-d1-test",
        "source_identity": "d1-test-source",
    }
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(**common, save_path=control_path),
    )
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            **common, save_path=segmented_path, max_iterations=1
        ),
    )
    train_selfplay_a8(
        config,
        resume=A8ResumeConfig(
            **common, load_path=segmented_path, save_path=segmented_path
        ),
    )

    control = _load_lane_state(control_path, config)
    segmented = _load_lane_state(segmented_path, config)
    assert training_state_digest(control) == training_state_digest(segmented)
