"""PPO model + config for the HEAT RL layer (Sprint 5c).

This module builds the policy that learns to play HEAT: a
:class:`sb3_contrib.MaskablePPO` over the single flattened, masked
``Discrete(ACTION_DIM)`` action space defined by :mod:`heat.ml.spaces` (§3.2,
§6.1 of ``docs/sprint5-ml-roadmap.md``). Action masking comes from the env's
``action_masks()`` hook (the sb3-contrib MaskablePPO convention); the decision
kind is signalled to the net via the one-hot block already inside the
observation (§3.1), so a *single* head over the union action space is correct --
we deliberately do NOT build a true multi-head policy (§7 non-goals).

Design notes
------------
* **Features extractor.** A small MLP (:class:`HeatMLPExtractor`) that maps the
  72-float observation to a learned latent. Per §5c / PLAN.md it is a 3-layer
  MLP, 128-256 units. The policy/value MLP (``net_arch``) sits on top of this
  latent, as in standard SB3.
* **Reward-shaping config flow (no 5b edits).** ``step_reward`` reads the
  module-level :data:`heat.ml.spaces.SHAPING_WEIGHT` /
  :data:`SHAPING_PROGRESS_COEF` at call time. :func:`apply_shaping_config` sets
  those globals from a :class:`PPOConfig`, so config flows to the env purely by
  mutating the frozen-default knobs -- the 5b env is untouched. :func:`build_model`
  applies it automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import torch
from torch import nn

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from heat.ml import spaces


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Hyperparameters for HEAT MaskablePPO training and the reward-shaping knobs.

    The shaping fields default to the pure-sparse setting (§6.3): the first
    training run uses only the terminal placement reward. They are wired to the
    env via :func:`apply_shaping_config` (which sets the module-level
    :mod:`heat.ml.spaces` constants ``step_reward`` reads) -- no 5b edit needed.
    """

    # --- network architecture ---
    #: Policy/value MLP head sizes (on top of the features extractor latent).
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    #: Hidden sizes of the custom features-extractor MLP (§5c: 3-layer, 128-256).
    features_extractor_hidden: list[int] = field(
        default_factory=lambda: [256, 256]
    )
    #: Dimensionality of the latent the features extractor emits.
    features_dim: int = 128

    # --- PPO hyperparameters (§5c defaults) ---
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.999  # long episodes
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- reward shaping (default OFF -> pure sparse first run, §6.3) ---
    #: Weight of the dense per-step progress shaping. 0.0 == pure sparse.
    shaping_weight: float = 0.0
    #: Per-space progress coefficient inside the dense shaping term.
    shaping_progress_coef: float = 1.0

    # --- bookkeeping ---
    seed: int | None = None
    verbose: int = 0
    device: str = "cpu"  # engine is the bottleneck, not the net (§5c)
    tensorboard_log: str | None = None


def apply_shaping_config(config: PPOConfig) -> None:
    """Push the reward-shaping knobs from ``config`` into :mod:`heat.ml.spaces`.

    ``spaces.step_reward`` reads ``spaces.SHAPING_WEIGHT`` /
    ``spaces.SHAPING_PROGRESS_COEF`` at call time, and :class:`heat.ml.env.HeatEnv`
    calls ``step_reward`` each step. Mutating these module globals is therefore
    how 5c tunes shaping without editing the frozen 5b env (§3.3, §6.3). This is
    process-global state, which is correct here: a training process trains one
    policy with one reward definition.
    """
    spaces.SHAPING_WEIGHT = config.shaping_weight
    spaces.SHAPING_PROGRESS_COEF = config.shaping_progress_coef


# ---------------------------------------------------------------------------
# Features extractor
# ---------------------------------------------------------------------------


class HeatMLPExtractor(BaseFeaturesExtractor):
    """MLP features extractor over the flat HEAT observation (§5c / PLAN.md).

    Maps ``Box(OBS_DIM,)`` -> a learned ``features_dim`` latent through a small
    MLP (default 3 linear layers with ReLU). The decision-kind one-hot is part
    of the observation, so this single shared trunk has the context it needs to
    specialise per decision kind; the action mask handles legality downstream.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim)
        hidden_sizes = hidden_sizes if hidden_sizes is not None else [256, 256]

        in_dim = int(observation_space.shape[0])
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, features_dim))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(env: gym.Env, config: PPOConfig | None = None) -> MaskablePPO:
    """Build a :class:`MaskablePPO` for ``env`` from ``config``.

    Uses :class:`MaskableActorCriticPolicy` over the env's ``Discrete(ACTION_DIM)``
    space with the custom :class:`HeatMLPExtractor`. The policy reads the env's
    ``action_masks()`` automatically during rollout collection (sb3-contrib
    convention), so illegal actions never receive probability mass.

    Side effect: applies ``config``'s reward-shaping knobs to :mod:`heat.ml.spaces`
    via :func:`apply_shaping_config` so shaping flows to the env.
    """
    if config is None:
        config = PPOConfig()

    apply_shaping_config(config)

    policy_kwargs: dict = {
        "net_arch": list(config.net_arch),
        "features_extractor_class": HeatMLPExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_dim,
            "hidden_sizes": list(config.features_extractor_hidden),
        },
    }

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        verbose=config.verbose,
        device=config.device,
        tensorboard_log=config.tensorboard_log,
    )
    return model
