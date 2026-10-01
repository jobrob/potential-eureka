"""Tests for Sprint A4's structured hidden-information encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from heat.models.game_state import GameState
from heat.ml.features import encode_observation
from heat.ml.selfplay import (
    A0Config,
    DotProductPolicy,
    HeatPolicy,
    StructuredObservationEncoder,
    build_policy,
    load_policy,
    save_policy,
)
from heat.ml.selfplay.multiseat import train_multiseat
from heat.ml.selfplay.tiny_heat import tiny_heat_track
from heat.ml.spaces import ACTION_DIM, BLOCK_HAND_HISTOGRAM, OBS_DIM


def _legal_masks(batch: int) -> torch.Tensor:
    mask = torch.zeros((batch, ACTION_DIM), dtype=torch.bool)
    mask[:, :4] = True
    return mask


def test_structured_encoder_shape_empty_hand_and_count_sensitivity() -> None:
    torch.manual_seed(0)
    encoder = StructuredObservationEncoder(hidden_sizes=(24, 16))

    obs = torch.zeros((3, OBS_DIM), dtype=torch.float32)
    obs[1, 0] = 1.0 / 7.0  # one S1
    obs[2, 0] = 2.0 / 7.0  # two S1: raw histogram must retain multiplicity
    latent = encoder(obs)

    assert latent.shape == (3, 16)
    assert torch.isfinite(latent).all()
    assert not torch.allclose(latent[0], latent[1])
    assert not torch.allclose(latent[1], latent[2])


@pytest.mark.parametrize("head", ["masked", "dotprod"])
def test_structured_policy_interface_legality_and_gradients(head: str) -> None:
    torch.manual_seed(1)
    policy = build_policy(
        A0Config(head=head, encoder="structured", hidden_sizes=(24, 16))
    )
    obs = torch.rand((8, OBS_DIM), dtype=torch.float32)
    # Make the hand block valid count/7 values rather than arbitrary fractions.
    obs[:, :BLOCK_HAND_HISTOGRAM] = (
        torch.randint(0, 3, (8, BLOCK_HAND_HISTOGRAM)).float() / 7.0
    )
    mask = _legal_masks(8)

    dist, value = policy._distribution_and_value(obs, mask)  # noqa: SLF001
    assert torch.all(dist.probs[:, 4:] == 0)
    assert value.shape == (8,)

    actions, _old_logp, _old_value, _old_entropy = policy.act(obs, mask)
    assert torch.all(actions < 4)
    logp, values, entropy = policy.evaluate(obs, actions, mask)
    loss = -logp.mean() + values.square().mean() - 0.01 * entropy.mean()
    loss.backward()  # type: ignore[no-untyped-call]

    trunk = policy.trunk
    assert isinstance(trunk, StructuredObservationEncoder)
    assert trunk.card_embeddings.weight.grad is not None
    assert float(trunk.card_embeddings.weight.grad.abs().sum()) > 0.0
    assert all(p.grad is not None for p in trunk.public_encoder.parameters())
    assert all(p.grad is not None for p in trunk.fusion.parameters())


def test_factory_supports_encoder_head_matrix_and_rejects_unknown() -> None:
    for encoder in ("flat", "structured"):
        masked = build_policy(
            A0Config(head="masked", encoder=encoder, hidden_sizes=(16,))
        )
        dotprod = build_policy(
            A0Config(head="dotprod", encoder=encoder, hidden_sizes=(16,))
        )
        assert isinstance(masked, HeatPolicy)
        assert isinstance(dotprod, DotProductPolicy)
        assert masked.encoder == encoder
        assert dotprod.encoder == encoder

    with pytest.raises(ValueError, match="unknown encoder"):
        build_policy(A0Config(encoder="private-peek", hidden_sizes=(16,)))


def test_opponent_private_state_does_not_leak_but_own_hand_matters() -> None:
    state = GameState.create(tiny_heat_track(), 2, seed=7)
    changed_opponent = state.clone(reseed=99)
    changed_opponent.get_player(1).hand = []
    changed_opponent.get_player(1).deck.draw(100)

    obs = encode_observation(state, 0, None)
    opponent_obs = encode_observation(changed_opponent, 0, None)
    assert np.array_equal(obs, opponent_obs), (
        "acting-seat observation leaked opponent private hand/deck state"
    )

    torch.manual_seed(2)
    encoder = StructuredObservationEncoder(hidden_sizes=(16,))
    with torch.no_grad():
        latent = encoder(torch.from_numpy(obs).unsqueeze(0))
        opponent_latent = encoder(torch.from_numpy(opponent_obs).unsqueeze(0))
    assert torch.equal(latent, opponent_latent)

    changed_own = state.clone(reseed=100)
    changed_own.get_player(0).hand = []
    own_obs = encode_observation(changed_own, 0, None)
    assert not np.array_equal(
        obs[:BLOCK_HAND_HISTOGRAM], own_obs[:BLOCK_HAND_HISTOGRAM]
    )
    with torch.no_grad():
        own_latent = encoder(torch.from_numpy(own_obs).unsqueeze(0))
    assert not torch.equal(latent, own_latent)


def test_structured_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = A0Config(
        head="masked", encoder="structured", hidden_sizes=(24, 16)
    )
    policy = build_policy(config)
    path = tmp_path / "structured.pt"
    save_policy(policy, config, path)
    loaded = load_policy(path)

    assert isinstance(loaded.trunk, StructuredObservationEncoder)
    assert loaded.encoder == "structured"
    obs = torch.rand((4, OBS_DIM))
    mask = _legal_masks(4)
    actions = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        before = policy.evaluate(obs, actions, mask)
        after = loaded.evaluate(obs, actions, mask)
    for expected, actual in zip(before, after, strict=True):
        assert torch.equal(expected, actual)


def test_structured_multiseat_training_smoke() -> None:
    seen: list[dict[str, float]] = []
    config = A0Config(
        encoder="structured",
        hidden_sizes=(16,),
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        total_timesteps=64,
        num_players=2,
        device="cpu",
        seed=0,
    )
    policy = train_multiseat(
        config,
        on_iteration=lambda _iteration, info: seen.append(info),
    )
    assert isinstance(policy.trunk, StructuredObservationEncoder)
    assert seen
    assert "entropy" not in seen[-1]
    for key in (
        "policy_loss",
        "value_loss",
        "update_entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    ):
        assert np.isfinite(seen[-1][key])
