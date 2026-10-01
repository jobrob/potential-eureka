"""Tests for the A3 legal-action dot-product head (Sprint A3, §6).

Covers the four exit gates:

* G2 (feature table correct): static-table spot checks for a gear, a card
  value-multiset, a react slot, and a discard index, plus kind-block mutual
  exclusivity.
* G1 (interface + legality): :class:`DotProductPolicy` matches the frozen
  ``act`` / ``evaluate`` shapes and puts **zero probability on illegal actions**
  (the A0 smoke-test pattern, with random masks); ``act`` / ``evaluate`` agree
  on log-probabilities; PPO-style gradients flow into every learnable block.
* The ``build_policy`` factory maps ``config.head`` to the right class.
* Both heads train (single-seat A0 + multi-seat A2 self-play smokes) with finite
  losses under ``head="dotprod"``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from heat.engine import rules
from heat.ml.action_codec import _CARD_MULTISETS, _REACT_TABLE, _TOKEN_ALPHABET
from heat.ml.selfplay import A0Config, DotProductPolicy, HeatPolicy, build_policy
from heat.ml.selfplay.action_features import (
    ACTION_FEAT_DIM,
    action_feature_table,
)
from heat.ml.selfplay.multiseat import train_multiseat
from heat.ml.selfplay.ppo import train
from heat.ml.spaces import (
    ACTION_DIM,
    CARDS_OFFSET,
    DISCARD_OFFSET,
    GEAR_OFFSET,
    OBS_DIM,
    REACT_OFFSET,
)

# Hardcoded feature-row column layout (independent of the module's own
# constants, so a layout drift is actually caught rather than silently mirrored).
_KIND0 = 0        # kind one-hot: GEAR/CARDS/REACT/SLIPSTREAM/DISCARD in 0..4
_GEAR0 = 5        # gear one-hot: gear 1..4 in 5..8
_HIST0 = 9        # histogram: 8 tokens in 9..16 (_TOKEN_ALPHABET order)
_SIZE = 17
_VALSUM = 18
_REACT0 = 19      # cooldown/4, boost, adr_speed, adr_cooldown in 19..22
_SLIP = 23
_DISC = 24


# ---------------------------------------------------------------------------
# 1. Feature-table spot checks (G2)
# ---------------------------------------------------------------------------


def test_feature_table_spot_checks() -> None:
    table = action_feature_table()
    assert table.shape == (ACTION_DIM, ACTION_FEAT_DIM)
    assert table.dtype == np.float32

    # --- gear-3 row: kind==GEAR, one-hot gear 3, nothing else set. ---
    gear3_row = GEAR_OFFSET + (3 - rules.MIN_GEAR)
    assert table[gear3_row, _KIND0 + 0] == 1.0
    assert table[gear3_row, _GEAR0 + (3 - rules.MIN_GEAR)] == 1.0
    assert table[gear3_row].sum() == pytest.approx(2.0)  # kind + gear one-hots

    # --- ("S3","S3") multiset: histogram 2xS3, size 2/4, value sum 6/20. ---
    s3_row = CARDS_OFFSET + _CARD_MULTISETS.index(("S3", "S3"))
    assert table[s3_row, _KIND0 + 1] == 1.0
    s3_col = _HIST0 + _TOKEN_ALPHABET.index("S3")
    assert table[s3_row, s3_col] == pytest.approx(0.5)  # count 2 / 4
    # No other histogram token set.
    hist = table[s3_row, _HIST0 : _HIST0 + 8]
    assert hist.sum() == pytest.approx(0.5)
    assert table[s3_row, _SIZE] == pytest.approx(2 / 4)
    assert table[s3_row, _VALSUM] == pytest.approx(6 / 20)

    # --- react slot 6: (3, False, False, True) -> 0.75, 0, 0, 1. ---
    assert _REACT_TABLE[6] == (3, False, False, True)
    react6_row = REACT_OFFSET + 6
    assert table[react6_row, _KIND0 + 2] == 1.0
    assert table[react6_row, _REACT0 + 0] == pytest.approx(3 / 4)
    assert table[react6_row, _REACT0 + 1] == 0.0
    assert table[react6_row, _REACT0 + 2] == 0.0
    assert table[react6_row, _REACT0 + 3] == 1.0

    # --- discard k=5: kind==DISCARD, k/7 feature. ---
    disc5_row = DISCARD_OFFSET + 5
    assert table[disc5_row, _KIND0 + 4] == 1.0
    assert table[disc5_row, _DISC] == pytest.approx(5 / 7)

    # --- kind blocks mutually exclusive: exactly one kind one-hot per row. ---
    kind_sums = table[:, _KIND0 : _KIND0 + 5].sum(axis=1)
    assert np.all(kind_sums == 1.0)
    # Range + finiteness.
    assert not np.isnan(table).any()
    assert table.min() >= 0.0 and table.max() <= 1.0


# ---------------------------------------------------------------------------
# Helpers for the policy tests
# ---------------------------------------------------------------------------


def _random_masks(batch: int, rng: np.random.Generator) -> torch.Tensor:
    """Random ``(batch, ACTION_DIM)`` bool masks, each row >= 1 legal action."""
    mask = rng.random((batch, ACTION_DIM)) < 0.4
    for b in range(batch):
        if not mask[b].any():
            mask[b, int(rng.integers(0, ACTION_DIM))] = True
        else:
            # Guarantee at least one legal even if the random draw was all False.
            mask[b, int(np.flatnonzero(mask[b])[0])] = True
    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# 2. No-illegal-mass + shape/dtype conformance (G1)
# ---------------------------------------------------------------------------


def test_dotprod_policy_no_illegal_mass_and_shapes() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    batch = 16
    policy = DotProductPolicy(hidden_sizes=(16, 16))
    obs = torch.as_tensor(
        rng.standard_normal((batch, OBS_DIM)), dtype=torch.float32
    )
    mask = _random_masks(batch, rng)

    # Distribution places exactly zero probability on illegal actions.
    dist, value = policy._distribution_and_value(obs, mask)  # noqa: SLF001
    probs = dist.probs.detach().numpy()
    illegal = ~mask.numpy()
    assert np.allclose(probs[illegal], 0.0)

    # act: shapes/dtypes + only-legal sampling.
    action, logp, value_a, entropy = policy.act(obs, mask)
    assert action.shape == (batch,) and action.dtype == torch.long
    for t in (logp, value_a, entropy):
        assert t.shape == (batch,) and t.dtype == torch.float32
    for b in range(batch):
        assert bool(mask[b, int(action[b])]), "sampled illegal action"

    # evaluate: shapes + finiteness on the same actions.
    logp_e, value_e, entropy_e = policy.evaluate(obs, action, mask)
    for t in (logp_e, value_e, entropy_e):
        assert t.shape == (batch,)
        assert torch.isfinite(t).all()


# ---------------------------------------------------------------------------
# 3. act / evaluate log-prob consistency (G1)
# ---------------------------------------------------------------------------


def test_dotprod_act_evaluate_logp_consistent() -> None:
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    batch = 12
    policy = DotProductPolicy(hidden_sizes=(16, 16))
    obs = torch.as_tensor(
        rng.standard_normal((batch, OBS_DIM)), dtype=torch.float32
    )
    mask = _random_masks(batch, rng)

    action, logp_act, _value, _entropy = policy.act(obs, mask)
    logp_eval, _v, _e = policy.evaluate(obs, action, mask)
    assert torch.allclose(logp_act, logp_eval, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Gradient flow into every learnable block (G1)
# ---------------------------------------------------------------------------


def test_dotprod_gradients_flow() -> None:
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    batch = 24
    policy = DotProductPolicy(hidden_sizes=(16, 16))
    obs = torch.as_tensor(
        rng.standard_normal((batch, OBS_DIM)), dtype=torch.float32
    )
    mask = _random_masks(batch, rng)
    action, _logp, _value, _entropy = policy.act(obs, mask)
    advantages = torch.as_tensor(
        rng.standard_normal(batch), dtype=torch.float32
    )
    returns = torch.as_tensor(rng.standard_normal(batch), dtype=torch.float32)

    logp, value, entropy = policy.evaluate(obs, action, mask)
    loss = -(logp * advantages).mean() + (value - returns).pow(2).mean()
    loss = loss - 0.01 * entropy.mean()
    loss.backward()  # type: ignore[no-untyped-call]

    def _grad_norm(module: torch.nn.Module) -> float:
        total = 0.0
        for p in module.parameters():
            assert p.grad is not None, "missing gradient"
            total += float(p.grad.abs().sum().item())
        return total

    assert _grad_norm(policy.action_mlp) > 0.0
    assert _grad_norm(policy.state_proj) > 0.0
    assert _grad_norm(policy.trunk) > 0.0


# ---------------------------------------------------------------------------
# 5. build_policy factory (correct class + unknown raises)
# ---------------------------------------------------------------------------


def test_build_policy_factory() -> None:
    masked = build_policy(A0Config(head="masked", hidden_sizes=(16, 16)))
    assert isinstance(masked, HeatPolicy)
    assert not isinstance(masked, DotProductPolicy)

    dotprod = build_policy(A0Config(head="dotprod", hidden_sizes=(16, 16)))
    assert isinstance(dotprod, DotProductPolicy)

    with pytest.raises(ValueError, match="unknown head"):
        build_policy(A0Config(head="bogus"))


# ---------------------------------------------------------------------------
# 6. Train / train_multiseat smokes with the dot-product head (finite losses)
# ---------------------------------------------------------------------------


def _tiny_config(**overrides: object) -> A0Config:
    base: dict[str, object] = dict(
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        total_timesteps=256,
        hidden_sizes=(16, 16),
        num_players=2,
        device="cpu",
        seed=0,
        head="dotprod",
    )
    base.update(overrides)
    return A0Config(**base)  # type: ignore[arg-type]


def test_dotprod_train_a0_smoke() -> None:
    seen: list[dict[str, float]] = []
    policy = train(_tiny_config(), on_iteration=lambda _i, info: seen.append(info))

    assert isinstance(policy, DotProductPolicy)
    assert len(seen) >= 1
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


def test_dotprod_train_multiseat_smoke() -> None:
    seen: list[dict[str, float]] = []
    policy = train_multiseat(
        _tiny_config(), on_iteration=lambda _i, info: seen.append(info)
    )

    assert isinstance(policy, DotProductPolicy)
    assert len(seen) >= 1
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
