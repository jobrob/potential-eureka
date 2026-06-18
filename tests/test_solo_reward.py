"""Sprint 8C: solo reward mode unit tests (§9).

Covers the solo finish bonus (paid only on ``terminated``), the no-negative
invariant, boundedness, the gamma speed-gradient argument (§4.3), and the
race-mode default-unchanged regression guard.
"""

from __future__ import annotations

import pytest

import heat.ml.spaces as spaces
from heat.ml.spaces import step_reward


@pytest.fixture(autouse=True)
def _reset_reward_globals():
    """Snapshot + restore the reward-mode module globals around each test."""
    saved = (
        spaces.REWARD_MODE,
        spaces.SOLO_FINISH_BONUS,
        spaces.SHAPING_WEIGHT,
        spaces.SHAPING_PROGRESS_COEF,
        spaces.SHAPING_SPINOUT_WEIGHT,
    )
    yield
    (
        spaces.REWARD_MODE,
        spaces.SOLO_FINISH_BONUS,
        spaces.SHAPING_WEIGHT,
        spaces.SHAPING_PROGRESS_COEF,
        spaces.SHAPING_SPINOUT_WEIGHT,
    ) = saved


def _solo_state(seed: int = 0):
    """A 1-player game state advanced one step, for reward-delta inputs."""
    from heat.ml.env import HeatEnv

    env = HeatEnv(num_players=1, opponents=None, learner_id=0)
    env.reset(seed=seed)
    prev = env.state.clone(reseed=0)
    return prev, env.state, env.learner_id


def test_solo_finish_bonus_only_on_terminated():
    """The bonus is paid on a real finish (terminated), never on truncation."""
    spaces.REWARD_MODE = "solo"
    spaces.SOLO_FINISH_BONUS = 5.0
    spaces.SHAPING_WEIGHT = 0.0  # isolate the terminal signal
    prev, curr, lid = _solo_state()

    r_term = step_reward(prev, curr, lid, done=True, terminated=True)
    r_trunc = step_reward(prev, curr, lid, done=True, terminated=False)

    assert r_term == pytest.approx(5.0)
    assert r_trunc == pytest.approx(0.0)


def test_solo_reward_no_negative():
    """No solo step reward is negative (no crash-to-end exploit), spinout off."""
    spaces.REWARD_MODE = "solo"
    spaces.SOLO_FINISH_BONUS = 5.0
    spaces.SHAPING_WEIGHT = 1.0
    spaces.SHAPING_PROGRESS_COEF = 1.0
    spaces.SHAPING_SPINOUT_WEIGHT = 0.0

    from heat.ml.env import HeatEnv

    env = HeatEnv(num_players=1, opponents=None, learner_id=0)
    obs, info = env.reset(seed=3)
    import numpy as np

    for _ in range(2000):
        mask = env.action_masks()
        a = int(np.random.default_rng(0).choice(np.flatnonzero(mask)))
        prev = env.state.clone(reseed=0)
        obs, r, term, trunc, info = env.step(a)
        assert r >= 0.0, f"negative solo reward {r}"
        if term or trunc:
            break


def test_solo_reward_bounded():
    """Per-step solo reward is finite and <= shaping_max + bonus."""
    import math

    spaces.REWARD_MODE = "solo"
    spaces.SOLO_FINISH_BONUS = 5.0
    spaces.SHAPING_WEIGHT = 1.0
    spaces.SHAPING_PROGRESS_COEF = 1.0
    prev, curr, lid = _solo_state(seed=1)
    r = step_reward(prev, curr, lid, done=True, terminated=True)
    assert math.isfinite(r)
    # progress over one step is <= a few spaces / length << 1; bonus is 5.0.
    assert r <= spaces.SHAPING_WEIGHT * 5.0 + spaces.SOLO_FINISH_BONUS


def test_solo_speed_gradient_sign():
    """Discounted return of a fast finish beats a slow one under gamma<1 (§4.3).

    Same total progress + bonus, paid at t=10 vs t=50. Under gamma=0.99 the fast
    stream's discounted return is strictly larger; under gamma=1.0 they are equal
    (proving DISCOUNTING, not progress, drives speed).
    """
    bonus = 5.0

    def discounted(t_finish: int, gamma: float) -> float:
        # constant progress per step until finish, then the bonus at t_finish.
        prog = 1.0 / t_finish  # same total progress (== 1.0) either way
        ret = sum((gamma**t) * prog for t in range(t_finish))
        ret += (gamma**t_finish) * bonus
        return ret

    fast = discounted(10, 0.99)
    slow = discounted(50, 0.99)
    assert fast > slow

    fast_eq = discounted(10, 1.0)
    slow_eq = discounted(50, 1.0)
    assert fast_eq == pytest.approx(slow_eq)


def test_reward_mode_default_unchanged():
    """REWARD_MODE='race' (default) == the existing placement reward exactly.

    The new ``terminated`` kwarg + the solo branch must not perturb race mode.
    """
    assert spaces.REWARD_MODE == "race"  # module default
    spaces.SHAPING_WEIGHT = 0.0

    # A 4-player terminal state: race mode pays the placement reward; the new
    # terminated kwarg (default + explicit) must not change it.
    from heat.ml.env import HeatEnv

    env = HeatEnv(num_players=4, learner_id=0)
    env.reset(seed=5)
    prev = env.state.clone(reseed=0)
    curr = env.state

    base = step_reward(prev, curr, 0, done=True)
    with_kw = step_reward(prev, curr, 0, done=True, terminated=True)
    placement = spaces._placement_reward(curr, 0)
    assert base == pytest.approx(placement)
    assert with_kw == pytest.approx(placement)
