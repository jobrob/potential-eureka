"""Rollout entropy must stay distinct from optimizer diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import heat.ml.selfplay.phase1 as phase1
from heat.ml.selfplay.phase1 import A8Config, _phase_rollout_diagnostics, train_selfplay_a8
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.selfplay.ppo import A0Config, _explained_variance, ppo_update
from heat.ml.spaces import (
    ACTION_DIM,
    CARDS_OFFSET,
    DISCARD_OFFSET,
    GEAR_OFFSET,
    GEAR_SIZE,
    OBS_DIM,
    REACT_OFFSET,
    REACT_SIZE,
    SLIPSTREAM_OFFSET,
    SLIPSTREAM_SIZE,
)

_PHASES = ("gear", "cards", "react", "slipstream", "discard")


def test_explained_variance_matches_population_definition() -> None:
    """Variance is the population figure; a flat return has nothing to explain."""
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    returns = np.array([1.0, 2.0, 5.0, 4.0], dtype=np.float64)
    expected = 1.0 - float(np.var(returns - values) / np.var(returns))
    actual = _explained_variance(torch.tensor(values), torch.tensor(returns))
    assert actual == pytest.approx(expected)
    assert _explained_variance(torch.ones(4), torch.full((4,), 2.0)) == 0.0
    assert _explained_variance(torch.zeros(1), torch.zeros(1)) == 0.0


def test_ppo_update_explained_variance_uses_paired_minibatch_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Last-epoch values stay paired with their returns across a reversed split."""

    def _reversed(n: int, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.arange(n - 1, -1, -1, device=device)

    monkeypatch.setattr(torch, "randperm", _reversed)
    torch.manual_seed(1)
    n = 8
    policy = HeatPolicy(hidden_sizes=(8,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0)
    obs = torch.randn(n, OBS_DIM)
    actions = torch.zeros(n, dtype=torch.long)
    masks = torch.zeros(n, ACTION_DIM, dtype=torch.bool)
    masks[:, :2] = True
    returns = torch.tensor([0.0, 1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 6.0])
    batch = {
        "obs": obs,
        "actions": actions,
        "logps": torch.zeros(n),
        "advantages": torch.zeros(n),
        "returns": returns,
        "masks": masks,
    }
    with torch.no_grad():
        _logp, values, _entropy = policy.evaluate(obs, actions, masks)
    expected = _explained_variance(values, returns)
    losses = ppo_update(
        policy,
        optimizer,
        batch,
        A0Config(batch_size=3, n_epochs=1, ent_coef=0.0),
    )
    assert losses["explained_variance"] == pytest.approx(expected)
    assert expected != pytest.approx(0.0)


def test_ppo_update_logs_diagnostics_without_stopping() -> None:
    """A large KL is recorded, and every epoch still updates the weights."""
    torch.manual_seed(0)
    n = 8
    policy = HeatPolicy(hidden_sizes=(16,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = A0Config(
        batch_size=4,
        n_epochs=3,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    obs = torch.randn(n, OBS_DIM)
    actions = torch.zeros(n, dtype=torch.long)
    masks = torch.zeros(n, ACTION_DIM, dtype=torch.bool)
    masks[:, :3] = True
    with torch.no_grad():
        current_logp, _values, _entropy = policy.evaluate(obs, actions, masks)
    batch = {
        "obs": obs,
        "actions": actions,
        "logps": current_logp - 3.0,
        "advantages": torch.zeros(n),
        "returns": torch.full((n,), 10.0),
        "masks": masks,
    }
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    steps = 0
    original_step = optimizer.step

    def _counting_step(*args: object, **kwargs: object) -> object:
        nonlocal steps
        steps += 1
        return original_step(*args, **kwargs)  # type: ignore[no-any-return, arg-type]

    optimizer.step = _counting_step  # type: ignore[method-assign]
    losses = ppo_update(policy, optimizer, batch, config)

    assert steps == config.n_epochs * 2
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters(), strict=True)
    )
    assert "entropy" not in losses
    # Value loss still moves the shared trunk, so the gap is not exactly 3.
    # It stays far outside the clip range, and the update does not stop.
    assert losses["approx_kl"] < -1.0
    assert losses["clip_fraction"] == pytest.approx(1.0)
    assert losses["explained_variance"] == 0.0
    for key in ("policy_loss", "value_loss", "update_entropy"):
        assert np.isfinite(losses[key])


def test_phase_rollout_diagnostics_group_by_taken_action() -> None:
    """Phase stats follow the taken action, and an empty phase stays zero."""
    actions = torch.tensor(
        [
            GEAR_OFFSET + GEAR_SIZE - 1,
            GEAR_OFFSET,
            CARDS_OFFSET,
            REACT_OFFSET + REACT_SIZE - 1,
            SLIPSTREAM_OFFSET + SLIPSTREAM_SIZE - 1,
        ],
        dtype=torch.long,
    )
    entropy = torch.tensor([1.0, 3.0, 4.0, 2.0, 0.5])
    masks = torch.zeros((5, ACTION_DIM), dtype=torch.bool)
    masks[0, GEAR_OFFSET : GEAR_OFFSET + 2] = True
    masks[1, GEAR_OFFSET : GEAR_OFFSET + 4] = True
    masks[2, CARDS_OFFSET : CARDS_OFFSET + 10] = True
    masks[3, REACT_OFFSET : REACT_OFFSET + 1] = True
    masks[4, SLIPSTREAM_OFFSET : SLIPSTREAM_OFFSET + 2] = True

    diagnostics = _phase_rollout_diagnostics(entropy, actions, masks)

    assert diagnostics["rollout_rows_gear"] == 2.0
    assert diagnostics["rollout_entropy_gear"] == pytest.approx(2.0)
    assert diagnostics["rollout_legal_count_gear"] == pytest.approx(3.0)
    assert diagnostics["rollout_rows_cards"] == 1.0
    assert diagnostics["rollout_entropy_cards"] == pytest.approx(4.0)
    assert diagnostics["rollout_legal_count_cards"] == pytest.approx(10.0)
    assert diagnostics["rollout_rows_react"] == 1.0
    assert diagnostics["rollout_entropy_react"] == pytest.approx(2.0)
    assert diagnostics["rollout_legal_count_react"] == pytest.approx(1.0)
    assert diagnostics["rollout_rows_slipstream"] == 1.0
    assert diagnostics["rollout_entropy_slipstream"] == pytest.approx(0.5)
    assert diagnostics["rollout_legal_count_slipstream"] == pytest.approx(2.0)
    assert diagnostics["rollout_rows_discard"] == 0.0
    assert diagnostics["rollout_entropy_discard"] == 0.0
    assert diagnostics["rollout_legal_count_discard"] == 0.0
    assert DISCARD_OFFSET == SLIPSTREAM_OFFSET + SLIPSTREAM_SIZE
    assert sum(diagnostics[f"rollout_rows_{name}"] for name in _PHASES) == 5.0


def test_phase1_record_keeps_rollout_entropy_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller sees pre-update entropy even when the optimizer reports 9.876."""
    received: list[float] = []
    original_update = phase1.EntropyController.update

    def _spy(self: phase1.EntropyController, mean_entropy: float) -> float:
        received.append(mean_entropy)
        return original_update(self, mean_entropy)

    def _fake_ppo_update(
        _policy: object,
        _optimizer: object,
        _batch: object,
        _config: object,
    ) -> dict[str, float]:
        return {
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "update_entropy": 9.876,
            "approx_kl": 0.01,
            "clip_fraction": 0.25,
            "explained_variance": 0.5,
        }

    monkeypatch.setattr(phase1.EntropyController, "update", _spy)
    monkeypatch.setattr(phase1, "ppo_update", _fake_ppo_update)
    _policy, records = train_selfplay_a8(
        A8Config(
            seat_counts=(2,),
            total_timesteps=16,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            hidden_sizes=(16,),
            snapshot_every=10,
            pool_prob=0.0,
            collector_mode="phase",
            device="cpu",
            seed=0,
        )
    )

    assert len(records) == 1
    assert received
    record = records[0]
    assert record["rollout_entropy"] == received[0]
    assert record["min_entropy"] == received[0]
    assert record["min_entropy"] == min(received)
    assert record["update_entropy"] == pytest.approx(9.876)
    assert record["rollout_entropy"] != pytest.approx(9.876)
    assert "entropy" not in record
    assert record["approx_kl"] == pytest.approx(0.01)
    assert record["clip_fraction"] == pytest.approx(0.25)
    assert record["explained_variance"] == pytest.approx(0.5)
    assert sum(record[f"rollout_rows_{name}"] for name in _PHASES) == record["n_recorded"]
    for name in _PHASES:
        assert np.isfinite(record[f"rollout_entropy_{name}"])
        assert np.isfinite(record[f"rollout_legal_count_{name}"])
        if record[f"rollout_rows_{name}"] == 0.0:
            assert record[f"rollout_entropy_{name}"] == 0.0
            assert record[f"rollout_legal_count_{name}"] == 0.0
        else:
            assert record[f"rollout_legal_count_{name}"] >= 1.0
