"""Fast smoke test for the A0 custom-PPO self-play skeleton (Sprint A0, G1).

Mirrors the style of ``tests/test_ml_training_smoke.py`` but exercises the NEW
custom loop (``heat.ml.selfplay``), not the SB3 path. It trains a tiny net for a
few hundred steps and asserts:

(a) an end-to-end rollout + >= 1 PPO update runs without error,
(b) every reported loss term is finite (no NaN/Inf -- catches GAE-sign /
    normalization bugs),
(c) the masked-categorical policy NEVER samples an illegal action, cross-checked
    independently against :func:`heat.ml.action_codec.legal_action_mask`,
(d) the masked distribution places exactly zero probability on illegal actions.

These are the A0 G1 exit criteria (the longer G2 learning-curve probe is a
manual run, not part of CI). Kept fast via a tiny net and a short rollout, so it
runs on the default (non-slow) gate.
"""

from __future__ import annotations

import numpy as np
import torch

from heat.ml.action_codec import legal_action_mask
from heat.ml.env import HeatEnv
from heat.ml.selfplay import A0Config, HeatPolicy
from heat.ml.selfplay.ppo import train
from heat.ml.spaces import ACTION_DIM, OBS_DIM


def _tiny_config(**overrides: object) -> A0Config:
    base = dict(
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        total_timesteps=256,
        hidden_sizes=(32, 32),
        num_players=2,
        device="cpu",
        seed=0,
    )
    base.update(overrides)
    return A0Config(**base)  # type: ignore[arg-type]


def test_a0_train_runs_and_losses_finite() -> None:
    """(a) + (b): a tiny end-to-end run takes >= 1 PPO update with finite losses."""
    seen: list[dict[str, float]] = []

    def _capture(_iteration: int, info: dict[str, float]) -> None:
        seen.append(info)

    policy = train(_tiny_config(), on_iteration=_capture)

    assert isinstance(policy, HeatPolicy)
    assert len(seen) >= 1, "expected at least one PPO update"
    for info in seen:
        for key in (
            "policy_loss",
            "value_loss",
            "update_entropy",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
        ):
            assert np.isfinite(info[key]), f"non-finite {key}={info[key]}"
        assert "entropy" not in info


def test_a0_policy_never_samples_illegal_action() -> None:
    """(c) + (d): the masked-categorical policy never puts mass on / samples an
    illegal action, cross-checked against ``legal_action_mask`` directly.

    Drives a real env exactly as the rollout collector does, but recomputes the
    legal mask straight from the engine decision (an independent source of truth)
    and asserts the sampled action is legal under it -- and that the policy's own
    distribution assigns the illegal set zero probability.
    """
    policy = HeatPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=(32, 32))
    env = HeatEnv(num_players=2)
    obs, info = env.reset(seed=7)
    mask = info["action_mask"]

    for _ in range(200):
        # Independent ground-truth mask straight from the engine decision.
        if env._decision is not None and not env._done:  # noqa: SLF001
            truth = legal_action_mask(env._decision, env.state)  # noqa: SLF001
            assert np.array_equal(truth, mask), "env mask diverged from codec mask"

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)

        # The policy's distribution must assign zero probability to illegal actions.
        dist, _value = policy._distribution_and_value(obs_t, mask_t)  # noqa: SLF001
        probs = dist.probs.detach().squeeze(0).numpy()
        illegal = ~np.asarray(mask)
        assert np.allclose(probs[illegal], 0.0), "illegal action received mass"

        action_t, _logp, _value, _entropy = policy.act(obs_t, mask_t)
        action = int(action_t.item())
        assert mask[action], f"sampled illegal action {action}"

        obs, _reward, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]
        if terminated or truncated:
            obs, info = env.reset(seed=8)
            mask = info["action_mask"]


def test_a0_buffer_gae_matches_manual_reference() -> None:
    """A focused GAE check: the buffer's advantages/returns match a hand-computed
    reference on a tiny, fully-specified rollout (guards the sign + done masking).
    """
    from heat.ml.selfplay.buffer import RolloutBuffer

    gamma, lam = 0.99, 0.95
    buf = RolloutBuffer(capacity=3, obs_dim=1, action_dim=1, device="cpu")
    # rewards, values, dones for 3 steps; episode ends on the last step.
    rewards = [1.0, 0.0, 2.0]
    values = [0.5, 0.4, 0.3]
    dones = [False, False, True]
    for r, v, d in zip(rewards, values, dones):
        buf.add(
            obs=np.zeros(1, dtype=np.float32),
            action=0,
            logp=0.0,
            value=v,
            reward=r,
            done=d,
            mask=np.ones(1, dtype=bool),
        )
    last_value = 0.0  # last step ended the episode
    buf.compute_gae(last_value, gamma, lam)

    # Manual GAE (reverse): nonterminal cuts the bootstrap on the terminal step.
    adv = [0.0, 0.0, 0.0]
    next_gae = 0.0
    for t in reversed(range(3)):
        nonterm = 0.0 if dones[t] else 1.0
        next_v = last_value if t == 2 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        next_gae = delta + gamma * lam * nonterm * next_gae
        adv[t] = next_gae
    expected_returns = [adv[t] + values[t] for t in range(3)]

    np.testing.assert_allclose(buf.advantages, adv, rtol=1e-6)
    np.testing.assert_allclose(buf.returns, expected_returns, rtol=1e-6)
