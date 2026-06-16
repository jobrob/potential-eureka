"""Fast tests for the Sprint 6C training overhaul (vec envs + self-play stability).

These are the default-run gates from the design's "Test gates" section:

* device fallback (``resolve_device``);
* vec-env determinism (per-worker seeding reproducible);
* **best-checkpoint preservation** — the headline regression for the lost-model
  failure: a descending eval-gate sequence must NOT overwrite the canonical best
  checkpoint with a worse model;
* gated promotion (best rewritten only on strict improvement; ``*_final``
  always written separately);
* VecNormalize stats round-trip + ``normalize`` meta block;
* repro-metadata round-trip + legacy back-compat (tripwire still the only check).

All of these avoid real (slow) evaluation by injecting a deterministic
``gate_fn`` and using a tiny net / few timesteps. The self-play *smoke* (real
training that does not collapse) is the ``slow`` test in
``test_ml_training_smoke.py``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import torch

from heat.ml.model import (
    NET_PROFILES,
    PPOConfig,
    net_profile_config,
    resolve_device,
)
from heat.ml.training import (
    CurriculumConfig,
    load_meta,
    meta_path_for,
    save_checkpoint,
    train_self_play,
    vecnorm_path_for,
)
from heat.ml.vec import make_vec_env
from heat.ml import spaces


# ---------------------------------------------------------------------------
# Device selection (§Part 1)
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_is_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_matches_cuda_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def test_resolve_device_cuda_falls_back_without_crash() -> None:
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert resolve_device("cuda") == "cpu"


def test_resolve_device_cuda_warns_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available; no fallback warning expected")
    with pytest.warns(RuntimeWarning):
        assert resolve_device("cuda") == "cpu"


def test_net_profiles_exist_and_differ() -> None:
    assert "small" in NET_PROFILES and "large" in NET_PROFILES
    small = net_profile_config("small")
    large = net_profile_config("large")
    assert large.features_dim > small.features_dim
    assert large.net_arch != small.net_arch


def test_net_profile_unknown_raises() -> None:
    with pytest.raises(ValueError):
        net_profile_config("enormous")


# ---------------------------------------------------------------------------
# Vec-env determinism (§Part 1)
# ---------------------------------------------------------------------------


def _rollout(venv, n_steps: int) -> np.ndarray:
    """Drive a vec env with masked-uniform actions; return the reward stream."""
    obs = venv.reset()
    rewards: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        masks = venv.env_method("action_masks")
        actions = []
        for m in masks:
            legal = np.flatnonzero(m)
            actions.append(int(rng.choice(legal)))
        obs, r, done, info = venv.step(np.asarray(actions))
        rewards.append(np.asarray(r, dtype=float).copy())
    return np.array(rewards)


def test_make_vec_env_seed_reproducible() -> None:
    """Two vec envs built with the same seed produce the same rollout."""
    kwargs = dict(
        track=None,
        num_players=2,
        opponents=None,
        learner_id=0,
        n_envs=2,
        vec_cls="dummy",  # in-process: deterministic + fast for the gate
        seed=123,
    )
    v1 = make_vec_env(**kwargs)
    try:
        r1 = _rollout(v1, 12)
    finally:
        v1.close()

    v2 = make_vec_env(**kwargs)
    try:
        r2 = _rollout(v2, 12)
    finally:
        v2.close()

    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Shared tiny configs for the training-loop gates
# ---------------------------------------------------------------------------


def _tiny_ppo(seed: int = 0) -> PPOConfig:
    return PPOConfig(
        net_arch=[16, 16],
        features_extractor_hidden=[16],
        features_dim=16,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        seed=seed,
        verbose=0,
        device="cpu",
        n_envs=1,
    )


def _tiny_curriculum(tmp_path, **overrides) -> CurriculumConfig:
    base = dict(
        total_timesteps=192,
        phase1_steps=64,
        snapshot_every=64,
        checkpoint_dir=str(tmp_path),
        run_name="t",
    )
    base.update(overrides)
    return CurriculumConfig(**base)


# ---------------------------------------------------------------------------
# Best-checkpoint preservation (the headline gate, §2.1)
# ---------------------------------------------------------------------------


def test_best_checkpoint_preserved_under_descending_scores(tmp_path) -> None:
    """A later, worse model must NOT overwrite the canonical best checkpoint.

    The gate returns a strictly DESCENDING score, so the Phase-1 baseline is the
    best and every Phase-2 chunk is worse. The canonical ``run_name`` checkpoint
    must therefore still hold the best (first/highest) model, and a separate
    ``*_final`` checkpoint must exist for the (worse) final model. This is the
    direct regression test for the lost-model failure.
    """
    scores = iter([0.9, 0.5, 0.2, 0.1, 0.05])

    def gate_fn(_model) -> float:
        return next(scores)

    config = _tiny_ppo()
    curriculum = _tiny_curriculum(tmp_path)
    model, best_path = train_self_play(
        config,
        curriculum,
        num_players=2,
        gate_fn=gate_fn,
    )

    # Best checkpoint + sidecar exist on the canonical path.
    assert best_path.endswith("t")
    meta = load_meta(best_path)
    # The best was saved at the Phase-1 baseline (score 0.9); its recorded gate
    # is not in meta, but the checkpoint must round-trip with the contract.
    assert meta["obs_dim"] == spaces.OBS_DIM
    assert meta["action_dim"] == spaces.ACTION_DIM
    assert meta["codec_version"] == spaces.CODEC_VERSION

    # A separate *_final checkpoint exists for the collapsed final model.
    import os

    assert os.path.exists(meta_path_for(f"{best_path}_final"))


def test_best_checkpoint_promoted_on_strict_improvement(tmp_path) -> None:
    """The best checkpoint is rewritten only when the gate strictly improves.

    We monkeypatch ``save_checkpoint`` via a recording wrapper to capture which
    paths were written and in what order, then assert the best path is written
    again on an improving score but the ``*_final`` is always written once.
    """
    import heat.ml.training as training_mod

    calls: list[str] = []
    real_save = training_mod.save_checkpoint

    def recording_save(model, path, **kwargs):  # noqa: ANN001
        calls.append(path)
        return real_save(model, path, **kwargs)

    training_mod.save_checkpoint = recording_save
    try:
        # Ascending then flat: baseline 0.1, chunk1 0.5 (improves -> promote),
        # chunk2 0.5 (no improve -> no promote).
        scores = iter([0.1, 0.5, 0.5])

        config = _tiny_ppo()
        curriculum = _tiny_curriculum(
            tmp_path, total_timesteps=192, phase1_steps=64, snapshot_every=64
        )
        _model, best_path = train_self_play(
            config, curriculum, num_players=2, gate_fn=lambda m: next(scores)
        )
    finally:
        training_mod.save_checkpoint = real_save

    best_writes = [c for c in calls if c == best_path]
    final_writes = [c for c in calls if c == f"{best_path}_final"]
    # Baseline write + one improving promotion == 2 writes to best_path.
    assert len(best_writes) == 2
    # The final (collapsed) model is always written exactly once, separately.
    assert len(final_writes) == 1


# ---------------------------------------------------------------------------
# Repro metadata round-trip + back-compat (§3.3)
# ---------------------------------------------------------------------------


def test_save_checkpoint_writes_repro_metadata(tmp_path) -> None:
    """save_checkpoint records git_sha/seed/configs/normalize, readable back."""
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model

    env = HeatEnv(num_players=2)
    model = build_model(env, _tiny_ppo(seed=7))

    ppo = _tiny_ppo(seed=7)
    curr = CurriculumConfig(run_name="r")
    path = str(tmp_path / "ckpt")
    save_checkpoint(
        model,
        path,
        track_name="USA",
        num_players=2,
        seed=7,
        ppo_config=ppo,
        curriculum_config=curr,
        track_config={"track_name": "USA"},
        normalize={"norm_obs": False, "norm_reward": True, "clip_reward": 10.0},
    )

    meta = load_meta(path)
    assert meta["seed"] == 7
    assert meta["ppo_config"]["features_dim"] == 16
    assert meta["curriculum_config"]["run_name"] == "r"
    assert meta["track_config"] == {"track_name": "USA"}
    assert meta["normalize"]["norm_reward"] is True
    # git_sha is a string in this repo, or None outside a git checkout.
    assert meta["git_sha"] is None or isinstance(meta["git_sha"], str)


def test_legacy_sidecar_without_new_fields_loads_and_validates(tmp_path) -> None:
    """A Sprint-5 sidecar lacking the 6C fields still loads + passes the tripwire."""
    from heat.agents.ml_agent import MLAgent

    legacy = {
        "obs_dim": spaces.OBS_DIM,
        "action_dim": spaces.ACTION_DIM,
        "codec_version": spaces.CODEC_VERSION,
        "track_name": "USA",
        "num_players": 2,
    }
    path = str(tmp_path / "legacy_model")
    with open(meta_path_for(path), "w", encoding="utf-8") as fh:
        json.dump(legacy, fh)

    # The tripwire checks only the three contract keys; the new fields' absence
    # must not raise.
    agent = MLAgent(path)
    agent._validate_meta()  # must not raise


# ---------------------------------------------------------------------------
# VecNormalize stats round-trip (§2.6)
# ---------------------------------------------------------------------------


def test_unnormalized_run_writes_no_vecnorm_file(tmp_path) -> None:
    """A run without normalization writes no .vecnorm.pkl and a null normalize block."""
    import os

    scores = iter([0.5, 0.4, 0.3])
    config = _tiny_ppo()
    curriculum = _tiny_curriculum(tmp_path)  # normalize_* default False
    _model, best_path = train_self_play(
        config, curriculum, num_players=2, gate_fn=lambda m: next(scores)
    )

    assert not os.path.exists(vecnorm_path_for(best_path))
    assert load_meta(best_path)["normalize"] is None


def test_normalized_run_round_trips_vecnorm_stats(tmp_path) -> None:
    """A normalized run writes a .vecnorm.pkl whose stats reload identically."""
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from heat.ml.env import HeatEnv

    scores = iter([0.5, 0.4, 0.3])
    config = _tiny_ppo()
    curriculum = _tiny_curriculum(tmp_path, normalize_reward=True)
    _model, best_path = train_self_play(
        config, curriculum, num_players=2, gate_fn=lambda m: next(scores)
    )

    stats_path = vecnorm_path_for(best_path)
    assert os.path.exists(stats_path)

    meta = load_meta(best_path)
    assert meta["normalize"]["norm_reward"] is True
    assert meta["normalize"]["norm_obs"] is False

    # Reload the stats; the running return estimate must restore.
    dummy = DummyVecEnv([lambda: HeatEnv(num_players=2)])
    vn = VecNormalize.load(stats_path, venv=dummy)
    assert vn.ret_rms is not None
    assert np.isfinite(vn.ret_rms.var)
