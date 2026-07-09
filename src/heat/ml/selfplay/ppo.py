"""Minimal custom PPO update + train loop for HEAT self-play (Sprint A0).

This is the Direction-A substrate spike: a small, transparent PPO we own end to
end. :func:`train` collects a rollout by driving the existing single-seat
:class:`heat.ml.env.HeatEnv` (per §4.3 of the A0 design), computes GAE in
:class:`heat.ml.selfplay.buffer.RolloutBuffer`, then runs clipped-surrogate
minibatch SGD via :func:`ppo_update`.

Every loss term is explicit and logged so the classic from-scratch PPO bugs
(GAE sign, advantage normalization, masking, the clip) are visible and testable
(the A0 G2 sanity probe). The loop owns the rollout boundary, which is exactly
what SB3's ``.learn()`` hides and what A2 (per-seat trajectories) / Direction D
(GPU leaf-batching) need.

A0 stays single-seat with scripted opponents and reuses the flat observation and
the existing mask; it is correctness, not skill (skill is A5+). It is purely
additive: it does not touch ``env.py`` / ``model.py`` / ``training.py`` /
``action_codec.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from heat.agents.heuristic_agent import HeuristicAgent
from heat.ml.env import HeatEnv, OpponentSpec, TrackSource
from heat.ml.model import resolve_device
from heat.ml.selfplay.buffer import RolloutBuffer
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.spaces import ACTION_DIM, OBS_DIM


@dataclass
class A0Config:
    """Hyperparameters for the A0 custom PPO loop.

    Names/defaults mirror :class:`heat.ml.model.PPOConfig` where sensible so the
    config stays familiar across the SB3 and custom paths, but this dataclass is
    deliberately standalone -- it does NOT import or subclass ``PPOConfig`` (the
    A0 design forbids entangling the two substrates).
    """

    # --- rollout / optimization ---
    #: Steps collected per rollout before each PPO update (one "iteration").
    n_steps: int = 2048
    #: Minibatch size for the inner SGD.
    batch_size: int = 256
    #: Optimization passes over each collected rollout.
    n_epochs: int = 10
    #: Total environment steps to train for (rounded up to whole rollouts).
    total_timesteps: int = 100_000

    # --- PPO / GAE (defaults shared with PPOConfig) ---
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

    # --- network ---
    #: Policy/value MLP trunk widths.
    hidden_sizes: tuple[int, ...] = (256, 256)

    # --- env ---
    #: Total seats (learner + scripted opponents).
    num_players: int = 4
    #: Re-pick the learner seat each episode (A0 design default).
    randomize_seat: bool = True

    # --- bookkeeping ---
    seed: int | None = None
    #: Device request, resolved via :func:`heat.ml.model.resolve_device`.
    device: str = "auto"


def _default_opponents() -> HeuristicAgent:
    """A scripted :class:`HeuristicAgent` opponent (broadcast to all opp seats)."""
    return HeuristicAgent()


def collect_rollout(
    env: HeatEnv,
    policy: HeatPolicy,
    buffer: RolloutBuffer,
    device: torch.device,
    *,
    obs: NDArray[np.float32],
    mask: NDArray[np.bool_],
    rng: np.random.Generator,
    gamma: float,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], list[float]]:
    """Fill ``buffer`` with one rollout by driving ``env`` (A0 design §4.3).

    Drives the existing single-seat env directly: it already returns the obs,
    the reward, and ``info["action_mask"]``. On episode end the env is reset
    (with a fresh seed drawn from ``rng`` so episodes vary) and collection
    continues, so a rollout can span several episodes.

    Args:
        env: the single-seat :class:`HeatEnv` to drive.
        policy: the acting policy (sampled under ``torch.no_grad`` via ``act``).
        buffer: a freshly :meth:`~RolloutBuffer.reset` buffer to fill.
        device: device the per-step obs/mask tensors are placed on.
        obs: the current observation carried in from the previous rollout/reset.
        mask: the current legal-action mask matching ``obs``.
        rng: numpy RNG used to draw per-episode reset seeds.
        gamma: discount factor, used for the truncation value-bootstrap fold
            (A2 design §4.6).

    Returns:
        ``(obs, mask, episode_returns)`` -- the carry-over obs/mask for the next
        rollout, and the list of finished-episode returns observed this rollout
        (used by the caller for the G2 learning-curve probe).
    """
    buffer.reset()
    episode_returns: list[float] = []
    running_return = 0.0

    while not buffer.full:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        action_t, logp_t, value_t, _entropy_t = policy.act(obs_t, mask_t)
        action = int(action_t.item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        reward = float(reward)

        # truncation != termination (A2 design §4.6): a time-limit cutoff
        # (truncated, not a real terminal) must NOT zero the value bootstrap the
        # way ``done=True`` does in GAE. Fold ``gamma * V(s_next)`` into this
        # final reward (SB3-style), keeping ``done=True`` in the buffer so the
        # frozen layout is untouched.
        if truncated and not terminated:
            reward += gamma * _bootstrap_value(
                policy, next_obs, info["action_mask"], device, last_done=False
            )

        running_return += reward

        buffer.add(
            obs=obs,
            action=action,
            logp=float(logp_t.item()),
            value=float(value_t.item()),
            reward=reward,
            done=done,
            mask=mask,
        )

        if done:
            episode_returns.append(running_return)
            running_return = 0.0
            reset_seed = int(rng.integers(0, 2**31 - 1))
            obs, info = env.reset(seed=reset_seed)
            mask = info["action_mask"]
        else:
            obs = next_obs
            mask = info["action_mask"]

    return obs, mask, episode_returns


def _bootstrap_value(
    policy: HeatPolicy,
    obs: NDArray[np.float32],
    mask: NDArray[np.bool_],
    device: torch.device,
    last_done: bool,
) -> float:
    """Value of the state following the last stored step (0 if that step ended
    an episode), used to bootstrap GAE."""
    if last_done:
        return 0.0
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    _action, _logp, value_t, _entropy = policy.act(obs_t, mask_t)
    return float(value_t.item())


def ppo_update(
    policy: HeatPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: A0Config,
) -> dict[str, float]:
    """Run ``n_epochs`` of clipped-surrogate minibatch SGD over one rollout.

    Loss = ``policy_loss + vf_coef * value_loss - ent_coef * entropy``:

    * **policy_loss** -- the PPO clipped surrogate, ``-min(r * A, clip(r, 1±eps)
      * A)`` averaged over the minibatch, where ``r = exp(logp_new - logp_old)``
      and ``A`` is the (batch-)normalized advantage.
    * **value_loss** -- MSE of the predicted value against the GAE return target.
    * **entropy** -- mean policy entropy, *subtracted* (a bonus) to discourage
      premature collapse.

    Advantages are normalized across the whole rollout (mean 0 / std 1) for
    scale-stable updates, gradients are clipped to ``max_grad_norm``, and the
    same legal-action mask used at collection time is re-applied via
    :meth:`HeatPolicy.evaluate` so the update is consistent with the rollout.

    Returns a dict of mean loss terms (for logging / the smoke test's finiteness
    check).
    """
    obs = batch["obs"]
    actions = batch["actions"]
    old_logps = batch["logps"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    masks = batch["masks"]

    n = obs.shape[0]
    batch_size = min(config.batch_size, n)

    # Normalize advantages over the whole rollout (standard PPO; scale-stable).
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    for _epoch in range(config.n_epochs):
        perm = torch.randperm(n, device=obs.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            new_logps, values, entropy = policy.evaluate(
                obs[idx], actions[idx], masks[idx]
            )

            ratio = torch.exp(new_logps - old_logps[idx])
            mb_adv = advantages[idx]
            unclipped = ratio * mb_adv
            clipped = (
                torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                * mb_adv
            )
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = nn.functional.mse_loss(values, returns[idx])
            entropy_mean = entropy.mean()

            loss = (
                policy_loss
                + config.vf_coef * value_loss
                - config.ent_coef * entropy_mean
            )

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_mean.item()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


def train(
    config: A0Config | None = None,
    *,
    env: HeatEnv | None = None,
    opponents: OpponentSpec | None = None,
    track: TrackSource | None = None,
    on_iteration: object = None,
) -> HeatPolicy:
    """Run the A0 custom PPO loop and return the trained policy.

    Collects rollouts of ``n_steps`` by driving a single-seat :class:`HeatEnv`
    (scripted opponents, ``randomize_seat`` per config), then applies a PPO
    update per rollout until ``total_timesteps`` steps have been collected.

    Args:
        config: hyperparameters; a default :class:`A0Config` if omitted.
        env: an optional pre-built env (the smoke test injects a tiny-field one).
            When omitted, a :class:`HeatEnv` is built from ``config`` /
            ``opponents`` / ``track``.
        opponents: opponent spec for the built env (defaults to
            :class:`HeuristicAgent`). Ignored if ``env`` is supplied.
        track: track source for the built env. Ignored if ``env`` is supplied.
        on_iteration: optional callback ``fn(iteration, info)`` invoked after
            each PPO update with the loss terms + mean episode return -- the hook
            the G2 learning-curve probe uses. Typed ``object`` so callers may
            pass any callable without import gymnastics.

    Returns:
        The trained :class:`HeatPolicy`.
    """
    if config is None:
        config = A0Config()

    device = torch.device(resolve_device(config.device))

    if env is None:
        opp: OpponentSpec = opponents if opponents is not None else _default_opponents()
        env = HeatEnv(
            track=track,
            num_players=config.num_players,
            opponents=opp,
            randomize_seat=config.randomize_seat,
        )

    policy = HeatPolicy(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=config.hidden_sizes
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    # Seed torch + numpy for reproducibility; the env reset seeds come from rng.
    if config.seed is not None:
        torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    buffer = RolloutBuffer(
        capacity=config.n_steps,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        device=device,
    )

    reset_seed = int(rng.integers(0, 2**31 - 1))
    obs, info = env.reset(seed=reset_seed)
    mask = info["action_mask"]

    n_iterations = max(1, config.total_timesteps // config.n_steps)
    for iteration in range(n_iterations):
        obs, mask, episode_returns = collect_rollout(
            env, policy, buffer, device, obs=obs, mask=mask, rng=rng,
            gamma=config.gamma,
        )
        last_done = bool(buffer.dones[buffer.capacity - 1])
        last_value = _bootstrap_value(policy, obs, mask, device, last_done)
        buffer.compute_gae(last_value, config.gamma, config.gae_lambda)

        batch = buffer.get()
        losses = ppo_update(policy, optimizer, batch, config)

        if callable(on_iteration):
            mean_return = (
                float(np.mean(episode_returns)) if episode_returns else float("nan")
            )
            on_iteration(
                iteration,
                {
                    **losses,
                    "mean_episode_return": mean_return,
                    "n_episodes": len(episode_returns),
                },
            )

    return policy
