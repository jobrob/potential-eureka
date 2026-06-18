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


def _actor_latent(policy, obs):
    """Return the actor (policy) latent for ``obs``, for shared OR unshared trunks.

    SB3's ``ActorCriticPolicy`` returns a single feature tensor when the features
    extractor is shared and a ``(pi_features, vf_features)`` tuple when it is not.
    Mirror SB3's own ``forward`` branch so model tests probe the logits/masking
    contract independent of the (Sprint-A) share-features topology.
    """
    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, _ = policy.mlp_extractor(features)
    else:
        pi_features, _ = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    return latent_pi


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
    obs = torch.zeros((batch, OBS_DIM), dtype=torch.float32, device=policy.device)

    # With an unshared features extractor (the Sprint-A default), SB3's
    # ``extract_features`` returns a (pi_features, vf_features) tuple and the
    # MlpExtractor exposes per-head forward methods; with a shared trunk it
    # returns one tensor consumed by ``mlp_extractor``. Drive the actor head
    # either way so this test checks the logits contract, not the share topology.
    pi_latent = _actor_latent(policy, obs)
    logits = policy.action_net(pi_latent)

    assert logits.shape == (batch, ACTION_DIM)
    assert torch.isfinite(logits).all()


def test_mask_application_zeros_illegal_probabilities() -> None:
    """A mask with a few legal entries must zero every masked-out action's prob."""
    env = HeatEnv(num_players=2)
    model = build_model(env, _tiny_config())
    policy = model.policy

    obs = torch.zeros((1, OBS_DIM), dtype=torch.float32, device=policy.device)

    # Build the (unmasked) categorical distribution, then apply a sparse mask.
    # ``_actor_latent`` handles both shared and unshared features extractors.
    latent_pi = _actor_latent(policy, obs)
    distribution = policy._get_action_dist_from_latent(latent_pi)

    legal = [0, 1, 4, 100, ACTION_DIM - 1]
    mask = np.zeros((1, ACTION_DIM), dtype=bool)
    mask[0, legal] = True

    distribution.apply_masking(mask)
    probs = distribution.distribution.probs.detach().cpu().numpy()[0]

    legal_set = set(legal)
    illegal = [i for i in range(ACTION_DIM) if i not in legal_set]

    # Masked-out actions get ~0 probability; legal ones keep all the mass.
    assert np.allclose(probs[illegal], 0.0, atol=1e-6)
    assert probs[legal].sum() > 1.0 - 1e-5
    # Sampling must only ever return a legal action.
    for _ in range(50):
        sampled = int(distribution.sample().item())
        assert sampled in legal_set


# ---------------------------------------------------------------------------
# Sprint A: unshared features extractor (Idea 13-part-1)
# ---------------------------------------------------------------------------


def test_unshared_features_extractor_built() -> None:
    """share_features_extractor=False -> actor and critic get distinct trunks."""
    import dataclasses

    cfg = dataclasses.replace(_tiny_config(), device="cpu")
    cfg = dataclasses.replace(cfg, share_features_extractor=False)
    model = build_model(HeatEnv(num_players=2), cfg)
    policy = model.policy
    # Confirmed SB3 attribute names: pi_/vf_features_extractor. Unshared -> the
    # two are distinct objects.
    assert policy.pi_features_extractor is not policy.vf_features_extractor


def test_shared_features_extractor_when_enabled() -> None:
    """share_features_extractor=True -> one shared trunk (SB3 legacy default)."""
    import dataclasses

    cfg = dataclasses.replace(_tiny_config(), device="cpu")
    cfg = dataclasses.replace(cfg, share_features_extractor=True)
    model = build_model(HeatEnv(num_players=2), cfg)
    policy = model.policy
    assert policy.pi_features_extractor is policy.vf_features_extractor


def test_share_features_extractor_defaults_false() -> None:
    """The PPOConfig default is unshared (the new behavior; note the inversion)."""
    assert PPOConfig().share_features_extractor is False
