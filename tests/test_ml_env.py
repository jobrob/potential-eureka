"""Tests for the HEAT Gymnasium environment (Sprint 5b, ml/env.py).

Gates (per docs/sprint5-ml-roadmap.md §5b):
  * Gym API conformance: reset() -> (obs, info); step() -> 5-tuple;
    action_masks() -> (ACTION_DIM,) bool.
  * Mask legality: drive full episodes sampling only masked-legal indices and
    assert the driver never raises its illegal-action ValueError guards.
  * Termination: every episode terminates (full finish order) or truncates at
    MAX_ROUNDS; episode length is bounded.
  * Determinism: same reset(seed) + same action sequence -> identical reward
    trajectory and finish order.
  * Edge paths: cluttered / finished-player / spun-out decisions advance; the
    empty-hand forced-CARDS auto-resolution (5a review carry-over #1) is covered.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.engine.driver import Decision, DecisionKind
from heat.engine.game import MAX_ROUNDS
from heat.models.track import Corner, Space, Track
from heat.ml import spaces
from heat.ml.action_codec import legal_action_mask
from heat.ml.env import HeatEnv
from heat.ml.spaces import ACTION_DIM, OBS_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track(laps: int = 1) -> Track:
    """A short track so episodes finish quickly in tests."""
    spaces_ = [Space(index=i, lanes=2) for i in range(12)]
    corners = [Corner(start=4, end=5, speed_limit=3)]
    return Track(
        name="env-test",
        spaces=spaces_,
        corners=corners,
        start_positions=[0, 1, 2, 3, 4, 5],
        laps=laps,
    )


def _make_env(num_players: int = 3, **kwargs) -> HeatEnv:
    return HeatEnv(track=_track(), num_players=num_players, **kwargs)


def _rollout(
    env: HeatEnv, seed: int, action_rng_seed: int = 0
) -> tuple[list[float], list[int], int, bool, bool]:
    """Run one full episode picking a deterministic masked-legal action.

    Returns (reward_trajectory, finish_order, n_steps, terminated, truncated).
    The action choice is a deterministic function of (action_rng_seed, step)
    over the currently-legal indices, so two identical-seed rollouts match.
    """
    obs, info = env.reset(seed=seed)
    assert obs.shape == (OBS_DIM,)
    rng = random.Random(action_rng_seed)
    rewards: list[float] = []
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        action = legal[rng.randrange(len(legal))]
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(reward)
        steps += 1
        assert steps <= 5000, "episode did not terminate within step bound"
    finish_order = [p.player_id for p in env.state.finished_players]
    return rewards, finish_order, steps, terminated, truncated


# ---------------------------------------------------------------------------
# Gym API conformance
# ---------------------------------------------------------------------------


class TestGymConformance:
    def test_reset_returns_obs_and_info(self) -> None:
        env = _make_env()
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        assert "action_mask" in info
        assert info["action_mask"].shape == (ACTION_DIM,)
        assert info["action_mask"].dtype == bool

    def test_step_returns_five_tuple(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_masks_shape_dtype(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        mask = env.action_masks()
        assert mask.shape == (ACTION_DIM,)
        assert mask.dtype == bool
        assert mask.any(), "a live decision must never expose an all-False mask"

    def test_spaces_match_contract(self) -> None:
        env = _make_env()
        assert env.observation_space.shape == (OBS_DIM,)
        assert env.action_space.n == ACTION_DIM

    def test_info_mask_matches_codec(self) -> None:
        env = _make_env()
        _, info = env.reset(seed=3)
        expected = legal_action_mask(env._decision, env.state)
        np.testing.assert_array_equal(info["action_mask"], expected)


# ---------------------------------------------------------------------------
# Mask legality: never send an illegal action to the driver
# ---------------------------------------------------------------------------


class TestMaskLegality:
    @pytest.mark.parametrize("num_players", [2, 3, 4, 6])
    def test_full_episodes_never_raise_illegal(self, num_players: int) -> None:
        """Driving episodes by sampling only masked-legal indices must never
        trip the driver's illegal-gear / illegal-card ValueError guards."""
        for seed in range(6):
            env = _make_env(num_players=num_players)
            # _rollout raises if the driver rejects an action.
            _rewards, finish_order, _steps, term, trunc = _rollout(env, seed)
            assert term or trunc

    def test_exposed_decisions_have_multiple_legal_actions(self) -> None:
        """Every decision surfaced to the policy is a real choice (>= 2 legal
        actions); forced/degenerate decisions are auto-resolved internally."""
        env = _make_env(num_players=3)
        env.reset(seed=1)
        terminated = truncated = False
        rng = random.Random(0)
        while not (terminated or truncated):
            mask = env.action_masks()
            assert int(mask.sum()) >= 2
            legal = np.flatnonzero(mask)
            action = int(legal[rng.randrange(len(legal))])
            _, _, terminated, truncated, _ = env.step(action)

    def test_only_current_kind_subrange_is_masked(self) -> None:
        """The mask's True entries always lie inside the current decision kind's
        contiguous sub-range."""
        ranges = {
            DecisionKind.GEAR: (spaces.GEAR_OFFSET, spaces.GEAR_OFFSET + spaces.GEAR_SIZE),
            DecisionKind.CARDS: (spaces.CARDS_OFFSET, spaces.CARDS_OFFSET + spaces.CARDS_SIZE),
            DecisionKind.REACT: (spaces.REACT_OFFSET, spaces.REACT_OFFSET + spaces.REACT_SIZE),
            DecisionKind.SLIPSTREAM: (
                spaces.SLIPSTREAM_OFFSET,
                spaces.SLIPSTREAM_OFFSET + spaces.SLIPSTREAM_SIZE,
            ),
            DecisionKind.DISCARD: (
                spaces.DISCARD_OFFSET,
                spaces.DISCARD_OFFSET + spaces.DISCARD_SIZE,
            ),
        }
        env = _make_env(num_players=3)
        env.reset(seed=2)
        rng = random.Random(0)
        terminated = truncated = False
        while not (terminated or truncated):
            lo, hi = ranges[env._decision.kind]
            legal = np.flatnonzero(env.action_masks())
            assert legal.min() >= lo and legal.max() < hi
            action = int(legal[rng.randrange(len(legal))])
            _, _, terminated, truncated, _ = env.step(action)


# ---------------------------------------------------------------------------
# Termination / bounds
# ---------------------------------------------------------------------------


class TestTermination:
    @pytest.mark.parametrize("num_players", [2, 3, 4, 6])
    def test_every_episode_terminates_with_full_finish_order(
        self, num_players: int
    ) -> None:
        for seed in range(6):
            env = _make_env(num_players=num_players)
            _rewards, finish_order, steps, term, trunc = _rollout(env, seed)
            assert term and not trunc
            # All seats finished, exactly once each, with a total order.
            assert sorted(finish_order) == list(range(num_players))
            assert env.state.round_num <= MAX_ROUNDS
            assert steps > 0

    def test_terminal_reward_is_placement(self) -> None:
        """On termination the learner receives a placement reward in [-1, 1];
        all pre-terminal rewards are 0 (sparse, SHAPING_WEIGHT default 0)."""
        env = _make_env(num_players=4)
        rewards, _finish, _steps, term, _trunc = _rollout(env, seed=0)
        assert term
        assert all(r == 0.0 for r in rewards[:-1])
        assert -1.0 <= rewards[-1] <= 1.0

    def test_step_after_done_raises(self) -> None:
        env = _make_env(num_players=2)
        _rollout(env, seed=0)
        with pytest.raises(RuntimeError):
            env.step(0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_trajectory(self) -> None:
        env_a = _make_env(num_players=4)
        env_b = _make_env(num_players=4)
        rew_a, fin_a, steps_a, _, _ = _rollout(env_a, seed=7, action_rng_seed=3)
        rew_b, fin_b, steps_b, _, _ = _rollout(env_b, seed=7, action_rng_seed=3)
        assert steps_a == steps_b
        assert rew_a == rew_b
        assert fin_a == fin_b

    def test_different_seed_can_differ(self) -> None:
        """Sanity: distinct game seeds generally yield distinct finish orders /
        trajectories (guards against a constant/ignored seed)."""
        env = _make_env(num_players=4)
        results = set()
        for seed in range(8):
            _rew, fin, _steps, _t, _tr = _rollout(env, seed=seed, action_rng_seed=1)
            results.add(tuple(fin))
        assert len(results) > 1

    def test_reset_same_seed_same_initial_obs(self) -> None:
        env = _make_env(num_players=3)
        obs1, _ = env.reset(seed=11)
        obs2, _ = env.reset(seed=11)
        np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Opponent wiring
# ---------------------------------------------------------------------------


class TestOpponents:
    def test_random_opponents_complete_episode(self) -> None:
        env = _make_env(num_players=3, opponents=lambda: RandomAgent(seed=0))
        _rew, finish, _steps, term, _trunc = _rollout(env, seed=0)
        assert term
        assert sorted(finish) == [0, 1, 2]

    def test_mixed_opponent_list(self) -> None:
        env = HeatEnv(
            track=_track(),
            num_players=3,
            opponents=[HeuristicAgent(), RandomAgent(seed=1)],
        )
        _rew, finish, _steps, term, _trunc = _rollout(env, seed=0)
        assert term and sorted(finish) == [0, 1, 2]

    def test_wrong_opponent_count_raises(self) -> None:
        with pytest.raises(ValueError):
            HeatEnv(track=_track(), num_players=3, opponents=[HeuristicAgent()])

    def test_learner_id_other_than_zero(self) -> None:
        env = HeatEnv(track=_track(), num_players=3, learner_id=2)
        _rew, finish, _steps, term, _trunc = _rollout(env, seed=0)
        assert term and sorted(finish) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Edge path: empty-hand forced-CARDS auto-resolution (carry-over #1)
# ---------------------------------------------------------------------------


class TestForcedDecisions:
    def test_empty_hand_cards_mask_is_all_false(self) -> None:
        """Precondition for the edge case: an empty hand yields a CARDS decision
        whose legal set is ``[()]`` and whose codec mask is all-False."""
        env = _make_env(num_players=2)
        env.reset(seed=0)
        # An empty-hand CARDS decision: legal == [()] (rules.legal_card_plays).
        empty_decision = Decision(DecisionKind.CARDS, env.learner_id, [()])
        mask = legal_action_mask(empty_decision, env.state)
        assert not mask.any(), "empty-hand CARDS must produce an all-False mask"

    def test_forced_action_resolves_all_false_mask(self) -> None:
        """The env auto-resolves an all-False CARDS decision to the engine's
        single forced play ``()`` rather than exposing the unusable mask."""
        env = _make_env(num_players=2)
        env.reset(seed=0)
        empty_decision = Decision(DecisionKind.CARDS, env.learner_id, [()])
        forced = env._forced_action(empty_decision)
        assert forced == ()

    def test_forced_action_resolves_single_legal_index(self) -> None:
        """A one-hot mask (single legal action) is auto-resolved by decoding the
        single legal flat index, not exposed as an RL step."""
        env = _make_env(num_players=2)
        env.reset(seed=0)
        # GEAR decision with a single legal option -> forced.
        single_gear = Decision(DecisionKind.GEAR, env.learner_id, [(1, 0)])
        forced = env._forced_action(single_gear)
        assert forced == (1, 0)

    def test_forced_action_returns_sentinel_for_real_choice(self) -> None:
        """A decision with >= 2 legal actions is NOT auto-resolved."""
        from heat.ml.env import _NO_FORCED

        env = _make_env(num_players=2)
        env.reset(seed=0)
        multi_gear = Decision(DecisionKind.GEAR, env.learner_id, [(1, 0), (2, 0)])
        assert env._forced_action(multi_gear) is _NO_FORCED

    def test_empty_learner_hand_episode_still_progresses(self) -> None:
        """Integration: forcibly empty the learner's hand mid-episode; the env
        must auto-resolve the resulting forced CARDS decisions and never surface
        an all-False mask to the policy, and the episode still terminates."""
        env = _make_env(num_players=2)
        obs, info = env.reset(seed=0)
        rng = random.Random(0)
        terminated = truncated = False
        emptied_once = False
        steps = 0
        while not (terminated or truncated):
            # Periodically empty the learner's hand to force the edge case.
            learner = env.state.get_player(env.learner_id)
            if not emptied_once and learner.hand:
                learner.hand = []
                emptied_once = True
            mask = env.action_masks()
            assert mask.any(), "forced/empty decision leaked an all-False mask"
            legal = np.flatnonzero(mask)
            action = int(legal[rng.randrange(len(legal))])
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            assert steps <= 5000
        assert terminated or truncated
        assert emptied_once
