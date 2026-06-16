"""Tests for :class:`heat.agents.ml_agent.MLAgent` (Sprint 5d).

Covers:
* **Protocol conformance** -- ``MLAgent`` is a ``BaseAgent`` and plays a full
  ``Game`` to completion without crashing (``slow``: needs a tiny trained model).
* **Legal-only** -- across a full game, every ``choose_*`` return maps to a
  flat index that is True in the legal mask it was handed (asserted directly via
  the codec, not just relying on the driver's guards).
* **Meta tripwire (§3.4)** -- an ``MLAgent`` pointed at a checkpoint whose
  sidecar reports a wrong ``obs_dim`` / ``action_dim`` / ``codec_version`` raises
  on load.
"""

from __future__ import annotations

import json

import pytest

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.ml_agent import CheckpointMismatchError, MLAgent
from heat.engine.driver import Decision, DecisionKind
from heat.engine.game import Game
from heat.engine import rules
from heat.ml import spaces
from heat.ml.action_codec import encode_action_index, legal_action_mask
from heat.ml.training import meta_path_for
from heat.tracks.loader import load_track_by_name


# ---------------------------------------------------------------------------
# Shared tiny-checkpoint fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """Train a tiny throwaway model and save it; return the checkpoint path.

    Module-scoped so the (relatively) expensive train happens once for all
    tests in this file that need a real, loadable checkpoint.
    """
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


# ---------------------------------------------------------------------------
# Meta tripwire (§3.4) -- fast, no model load needed
# ---------------------------------------------------------------------------


def _write_meta(path: str, meta: dict) -> None:
    with open(meta_path_for(path), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def _good_meta() -> dict:
    return {
        "obs_dim": spaces.OBS_DIM,
        "action_dim": spaces.ACTION_DIM,
        "codec_version": spaces.CODEC_VERSION,
        "track_name": "USA",
        "num_players": 2,
    }


def test_meta_missing_sidecar_raises(tmp_path) -> None:
    """No sidecar at all -> fail fast with a clear error (no SB3 load attempted)."""
    agent = MLAgent(str(tmp_path / "no_such_model"))
    with pytest.raises(CheckpointMismatchError):
        agent._get_model()


@pytest.mark.parametrize("bad_key", ["obs_dim", "action_dim", "codec_version"])
def test_meta_tripwire_rejects_drifted_contract(tmp_path, bad_key) -> None:
    """A sidecar whose obs/action/codec disagrees with the live contract raises.

    The (fake) zip is never reached: ``_validate_meta`` runs before the SB3 load,
    so a deliberately bogus value on each of the three contract keys must raise.
    """
    ckpt = str(tmp_path / "drifted_model")
    meta = _good_meta()
    meta[bad_key] = meta[bad_key] + 999  # drift this contract field
    _write_meta(ckpt, meta)

    agent = MLAgent(ckpt)
    with pytest.raises(CheckpointMismatchError) as excinfo:
        agent._get_model()
    assert bad_key in str(excinfo.value)


def test_meta_validation_passes_for_matching_contract(tmp_path) -> None:
    """A matching sidecar passes validation (failure then comes only from the
    absent SB3 zip, i.e. validation itself does not raise)."""
    ckpt = str(tmp_path / "ok_meta_no_zip")
    _write_meta(ckpt, _good_meta())
    agent = MLAgent(ckpt)
    # _validate_meta must NOT raise; the subsequent zip load will fail instead.
    agent._validate_meta()


# ---------------------------------------------------------------------------
# Conformance + legal-only (need a real checkpoint -> slow)
# ---------------------------------------------------------------------------


def test_mlagent_is_base_agent(tiny_checkpoint) -> None:
    agent = MLAgent(tiny_checkpoint)
    assert isinstance(agent, BaseAgent)


def test_mlagent_is_picklable_by_path(tiny_checkpoint) -> None:
    """Loading then pickling must drop the heavy model (§6.5)."""
    import pickle

    agent = MLAgent(tiny_checkpoint)
    agent._get_model()  # populate the cache
    assert agent._model is not None

    restored = pickle.loads(pickle.dumps(agent))
    assert restored._model is None
    assert restored.model_path == tiny_checkpoint


class _LegalitySpy(MLAgent):
    """An MLAgent that asserts every chosen action is legal under its own mask.

    For each ``choose_*`` call it rebuilds the same ``Decision`` the agent uses,
    encodes the returned engine action to a flat index, and asserts that index
    is True in ``legal_action_mask`` -- a direct legal-only check independent of
    the driver's own guards.
    """

    def _assert_legal(self, decision: Decision, engine_action: object, state) -> None:
        mask = legal_action_mask(decision, state)
        idx = encode_action_index(decision, engine_action)
        assert mask[idx], (
            f"chosen action {engine_action!r} -> index {idx} is not legal for "
            f"{decision.kind}"
        )

    def choose_gear(self, state, player_id, legal_gears):
        result = super().choose_gear(state, player_id, legal_gears)
        assert result in legal_gears
        self._assert_legal(
            Decision(DecisionKind.GEAR, player_id, legal_gears), result, state
        )
        return result

    def choose_cards(self, state, player_id, legal_plays):
        result = super().choose_cards(state, player_id, legal_plays)
        self._assert_legal(
            Decision(DecisionKind.CARDS, player_id, legal_plays), result, state
        )
        return result

    def choose_react(self, state, player_id, max_cooldown, can_boost, has_adrenaline):
        result = super().choose_react(
            state, player_id, max_cooldown, can_boost, has_adrenaline
        )
        opts = rules.ReactOptions(
            max_cooldown=max_cooldown,
            can_boost=can_boost,
            has_adrenaline=has_adrenaline,
        )
        self._assert_legal(
            Decision(DecisionKind.REACT, player_id, opts), result, state
        )
        return result

    def choose_slipstream(self, state, player_id):
        result = super().choose_slipstream(state, player_id)
        assert isinstance(result, bool)
        self._assert_legal(
            Decision(DecisionKind.SLIPSTREAM, player_id, True), result, state
        )
        return result

    def choose_discard(self, state, player_id, discardable):
        result = super().choose_discard(state, player_id, discardable)
        self._assert_legal(
            Decision(DecisionKind.DISCARD, player_id, discardable), result, state
        )
        return result


# ---------------------------------------------------------------------------
# VecNormalize at inference (§2.6) + back-compat
# ---------------------------------------------------------------------------


def test_mlagent_no_vecnorm_sidecar_uses_raw_obs(tmp_path) -> None:
    """A checkpoint with no .vecnorm.pkl behaves exactly as Sprint 5 (raw obs)."""
    ckpt = str(tmp_path / "plain_model")
    _write_meta(ckpt, _good_meta())  # legacy meta: no "normalize" block

    agent = MLAgent(ckpt)
    agent._load_vecnorm()  # called inside _get_model; safe to call directly
    assert agent._vecnorm is None
    assert agent._norm_obs is False


def test_mlagent_loads_and_applies_obs_normalization(tmp_path) -> None:
    """When norm_obs=True and a stats file exists, obs are normalized pre-predict.

    Builds the artifacts directly (a real obs-normalized training run is slow):
    a matching meta with ``normalize.norm_obs=True`` and a saved ``VecNormalize``
    whose obs_rms has a known mean/var. ``_load_vecnorm`` must load it and
    ``normalize_obs`` must shift the observation by that mean/var.
    """
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from heat.ml.env import HeatEnv
    from heat.ml.training import vecnorm_path_for

    ckpt = str(tmp_path / "norm_model")
    meta = _good_meta()
    meta["normalize"] = {"norm_obs": True, "norm_reward": True, "clip_reward": 10.0}
    _write_meta(ckpt, meta)

    # Build + warm a VecNormalize so obs_rms holds non-trivial running stats.
    venv = VecNormalize(
        DummyVecEnv([lambda: HeatEnv(num_players=2)]),
        norm_obs=True,
        norm_reward=True,
    )
    venv.reset()
    for _ in range(5):
        masks = venv.env_method("action_masks")
        actions = [int(np.flatnonzero(m)[0]) for m in masks]
        venv.step(np.asarray(actions))
    venv.save(vecnorm_path_for(ckpt))
    venv.close()

    agent = MLAgent(ckpt)
    agent._load_vecnorm()
    assert agent._norm_obs is True
    assert agent._vecnorm is not None
    # Inference stats are frozen and reward-norm disabled.
    assert agent._vecnorm.training is False
    assert agent._vecnorm.norm_reward is False

    raw = np.zeros(spaces.OBS_DIM, dtype=np.float32)
    normed = agent._vecnorm.normalize_obs(raw)
    # Normalization shifts by the (non-zero) running mean -> output differs.
    assert not np.allclose(normed, raw)


@pytest.mark.slow
def test_mlagent_plays_full_game_and_only_legal(tiny_checkpoint) -> None:
    """A full game with MLAgent in seat 0 vs HeuristicAgent completes, and every
    MLAgent choice is legal under its own mask."""
    track = load_track_by_name("usa")
    agents = [
        _LegalitySpy(tiny_checkpoint, name="ML"),
        HeuristicAgent(name="Heur"),
    ]
    game = Game(track, agents, logging_enabled=False, seed=7)
    result = game.run()

    assert game.is_over
    assert result.total_rounds > 0
    assert len(result.finish_order) == 2
