"""Tests for the HEAT RL evaluation harness (Sprint 5d).

Covers :mod:`heat.ml.evaluate`:
* ``ml_agent_factory`` is picklable (a top-level ``functools.partial``, not a
  lambda/closure) and carries only the model *path* (§6.5).
* ``evaluate_ml`` runs ``run_batch`` on a tiny throwaway model and produces
  ``AgentStats`` with a sane ``win_rate`` in [0, 1] and ``games_played`` matching
  the batch size.

The "MLAgent beats heuristic" success metric is a MANUAL eval (§6.2/§7), NOT
asserted here.
"""

from __future__ import annotations

import functools
import pickle

import pytest

from heat.agents.ml_agent import MLAgent
from heat.ml.evaluate import evaluate_ml, ml_agent_factory


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """Train and save a tiny throwaway model once for this module."""
    from heat.ml.model import PPOConfig
    from heat.ml.training import smoke_train

    ckpt = str(tmp_path_factory.mktemp("ckpt") / "tiny_model")
    config = PPOConfig(
        net_arch=[16, 16],
        features_extractor_hidden=[16],
        features_dim=16,
        n_steps=64,
        batch_size=32,
        seed=0,
        verbose=0,
    )
    _model, saved = smoke_train(
        total_timesteps=64,
        num_players=2,
        checkpoint_path=ckpt,
        config=config,
    )
    assert saved == ckpt
    return ckpt


def test_ml_agent_factory_is_picklable_and_path_only() -> None:
    """The factory must pickle (top-level partial) and carry only the path."""
    factory = ml_agent_factory("some/path", name="ML")
    assert isinstance(factory, functools.partial)

    restored = pickle.loads(pickle.dumps(factory))
    agent = restored(0, None)
    assert isinstance(agent, MLAgent)
    assert agent.model_path == "some/path"
    assert agent._model is None  # lazy: nothing loaded just by constructing


@pytest.mark.slow
def test_evaluate_ml_produces_sane_stats(tiny_checkpoint) -> None:
    """The eval harness runs run_batch and returns AgentStats with a sane
    win_rate in [0, 1] and games_played matching the batch size."""
    num_games = 6
    per_agent = evaluate_ml(
        tiny_checkpoint,
        num_games=num_games,
        num_players=2,
        track="usa",
        seed=0,
        parallel=False,
    )

    assert "MLAgent" in per_agent
    assert "HeuristicAgent" in per_agent

    ml_stats = per_agent["MLAgent"]
    assert ml_stats.games_played == num_games
    assert 0.0 <= ml_stats.win_rate <= 1.0
    assert ml_stats.wins == sum(
        c for pos, c in ml_stats.finish_position_counts.items() if pos == 1
    )

    # Two seats, one MLAgent + one HeuristicAgent: wins partition across them.
    total_wins = sum(s.wins for s in per_agent.values())
    assert total_wins == num_games
