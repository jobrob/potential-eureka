"""Slow smoke test for HEAT PPO training (Sprint 5c, §5c / §6.2).

Marked ``slow`` so it is deselected from the fast gate. It trains a tiny net
for a few hundred steps and asserts:

(a) training runs without error,
(b) ``model.save`` + ``MaskablePPO.load`` round-trip AND the ``.meta.json``
    sidecar is written and reloads with the frozen contract values,
(c) the loaded model predicts a LEGAL action (respects ``action_masks()``),
(d) losses / values are finite (not NaN).

It deliberately does NOT assert "ML beats heuristic" -- that is 5d's manual eval.
Kept under ~30s via a tiny net and short rollout.
"""

from __future__ import annotations

import numpy as np
import pytest

from sb3_contrib import MaskablePPO

from heat.ml.env import HeatEnv
from heat.ml.model import PPOConfig
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM
from heat.ml.training import load_meta, meta_path_for, smoke_train

pytestmark = pytest.mark.slow


def test_training_smoke_runs_saves_and_predicts_legal(tmp_path) -> None:
    ckpt = str(tmp_path / "smoke_model")

    config = PPOConfig(
        net_arch=[32, 32],
        features_extractor_hidden=[32],
        features_dim=32,
        n_steps=128,
        batch_size=64,
        seed=0,
        verbose=0,
    )

    # (a) trains without error and saves a checkpoint + sidecar.
    model, saved = smoke_train(
        total_timesteps=256,
        num_players=2,
        checkpoint_path=ckpt,
        config=config,
    )
    assert saved == ckpt

    # (d) losses / training metrics are finite (not NaN).
    logger_vals = model.logger.name_to_value
    for key, val in logger_vals.items():
        if key.startswith("train/") and isinstance(val, (int, float)):
            assert np.isfinite(val), f"non-finite training metric {key}={val}"

    # (b) sidecar exists and reloads with the frozen contract values.
    meta = load_meta(ckpt)
    assert meta == {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "codec_version": CODEC_VERSION,
        "track_name": "USA",
        "num_players": 2,
    }
    assert meta_path_for(ckpt).endswith(".meta.json")

    # (b) SB3 archive round-trips.
    loaded = MaskablePPO.load(ckpt, device="cpu")
    assert loaded.observation_space.shape == (OBS_DIM,)
    assert loaded.action_space.n == ACTION_DIM

    # (c) the loaded model predicts a LEGAL action under the env's mask.
    env = HeatEnv(num_players=2)
    obs, _info = env.reset(seed=123)
    for _ in range(20):
        mask = env.action_masks()
        action, _ = loaded.predict(obs, action_masks=mask, deterministic=True)
        flat = int(np.asarray(action).reshape(-1)[0])
        assert mask[flat], f"predicted illegal action {flat}"
        obs, _reward, terminated, truncated, _info = env.step(flat)
        if terminated or truncated:
            break
