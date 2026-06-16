"""Fast tests for the HEAT PPO model scaffold (Sprint 5c, §5c).

These are gating (NOT slow): a single tiny model build against the real env's
spaces, a forward pass producing ``(.., ACTION_DIM)`` logits, and -- the key
property -- that applying an action mask drives the masked-out actions'
probabilities to ~0. No training loop here.
"""

from __future__ import annotations

import numpy as np
import torch

from heat.ml.env import HeatEnv
from heat.ml.model import HeatMLPExtractor, PPOConfig, build_model
from heat.ml.spaces import ACTION_DIM, OBS_DIM


def _tiny_config() -> PPOConfig:
    return PPOConfig(
        net_arch=[16, 16],
        features_extractor_hidden=[16],
        features_dim=16,
        n_steps=64,
        batch_size=32,
        seed=0,
        verbose=0,
    )


def test_build_model_against_real_env_spaces() -> None:
    env = HeatEnv(num_players=2)
    model = build_model(env, _tiny_config())

    # Policy is wired to the env's real spaces and the custom extractor.
    assert model.observation_space.shape == (OBS_DIM,)
    assert model.action_space.n == ACTION_DIM
    assert isinstance(model.policy.features_extractor, HeatMLPExtractor)


def test_forward_pass_logits_shape() -> None:
    env = HeatEnv(num_players=2)
    model = build_model(env, _tiny_config())
    policy = model.policy

    batch = 5
    obs = torch.zeros((batch, OBS_DIM), dtype=torch.float32)

    features = policy.extract_features(obs)
    latent_pi, _ = policy.mlp_extractor(features)
    logits = policy.action_net(latent_pi)

    assert logits.shape == (batch, ACTION_DIM)
    assert torch.isfinite(logits).all()


def test_mask_application_zeros_illegal_probabilities() -> None:
    """A mask with a few legal entries must zero every masked-out action's prob."""
    env = HeatEnv(num_players=2)
    model = build_model(env, _tiny_config())
    policy = model.policy

    obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)

    # Build the (unmasked) categorical distribution, then apply a sparse mask.
    features = policy.extract_features(obs)
    latent_pi, _ = policy.mlp_extractor(features)
    distribution = policy._get_action_dist_from_latent(latent_pi)

    legal = [0, 1, 4, 100, ACTION_DIM - 1]
    mask = np.zeros((1, ACTION_DIM), dtype=bool)
    mask[0, legal] = True

    distribution.apply_masking(mask)
    probs = distribution.distribution.probs.detach().numpy()[0]

    legal_set = set(legal)
    illegal = [i for i in range(ACTION_DIM) if i not in legal_set]

    # Masked-out actions get ~0 probability; legal ones keep all the mass.
    assert np.allclose(probs[illegal], 0.0, atol=1e-6)
    assert probs[legal].sum() > 1.0 - 1e-5
    # Sampling must only ever return a legal action.
    for _ in range(50):
        sampled = int(distribution.sample().item())
        assert sampled in legal_set
