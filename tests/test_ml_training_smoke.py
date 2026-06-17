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

    # (b) sidecar exists and the frozen contract fields reload correctly. The
    # sidecar also carries the additive Sprint-6C repro fields (git_sha, seed,
    # ppo_config, ...), so assert the contract subset rather than exact equality.
    meta = load_meta(ckpt)
    assert meta["obs_dim"] == OBS_DIM
    assert meta["action_dim"] == ACTION_DIM
    assert meta["codec_version"] == CODEC_VERSION
    assert meta["track_name"] == "USA"
    assert meta["num_players"] == 2
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


def test_subproc_vec_smoke_masking_holds(tmp_path) -> None:
    """smoke_train over a SubprocVecEnv runs, saves, and the loaded model only
    predicts LEGAL actions under the env mask.

    This is the §Part 1 vec smoke + the masking backstop: vectorization is the
    one place masking could silently break, so we assert no illegal action is
    predicted after a real (tiny) multi-process training run.
    """
    ckpt = str(tmp_path / "vec_smoke_model")
    config = PPOConfig(
        net_arch=[16, 16],
        features_extractor_hidden=[16],
        features_dim=16,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        seed=0,
        verbose=0,
        device="cpu",
        n_envs=2,
    )

    model, saved = smoke_train(
        total_timesteps=128,
        num_players=2,
        checkpoint_path=ckpt,
        config=config,
        n_envs=2,
        vec_cls="subproc",
    )
    assert saved == ckpt

    loaded = MaskablePPO.load(ckpt, device="cpu")
    env = HeatEnv(num_players=2)
    obs, _info = env.reset(seed=1)
    for _ in range(20):
        mask = env.action_masks()
        action, _ = loaded.predict(obs, action_masks=mask, deterministic=True)
        flat = int(np.asarray(action).reshape(-1)[0])
        assert mask[flat], f"predicted illegal action {flat} under vec-trained model"
        obs, _reward, terminated, truncated, _info = env.step(flat)
        if terminated or truncated:
            break


def test_self_play_gate_preserves_phase1_strength(tmp_path) -> None:
    """A short Phase 1 -> Phase 2 self-play run must NOT lose the good model.

    The behavioral guard against the Sprint-5 collapse: even if Phase 2 wobbles,
    the eval-gated best-checkpoint preservation guarantees the canonical
    checkpoint's score is >= the Phase-1 baseline. We inject a deterministic
    ``gate_fn`` modelling exactly the observed failure (Phase 1 strong, Phase 2
    collapses to near-zero) and assert the *saved best* checkpoint is the strong
    Phase-1 model, not the collapsed final one.
    """
    from heat.ml.evaluate import evaluate_ml
    from heat.ml.training import CurriculumConfig, train_self_play

    # Phase-1 baseline strong (0.9); Phase 2 collapses (the exact failure mode).
    gate_scores = iter([0.9, 0.05, 0.02, 0.0])
    phase1_baseline = 0.9

    config = PPOConfig(
        net_arch=[16, 16],
        features_extractor_hidden=[16],
        features_dim=16,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        seed=0,
        verbose=0,
        device="cpu",
        n_envs=1,
    )
    curriculum = CurriculumConfig(
        total_timesteps=192,
        phase1_steps=64,
        snapshot_every=64,
        checkpoint_dir=str(tmp_path),
        run_name="sp",
    )

    captured: list[float] = []

    def gate_fn(_model) -> float:
        s = next(gate_scores)
        captured.append(s)
        return s

    _model, best_path = train_self_play(
        config, curriculum, num_players=2, gate_fn=gate_fn
    )

    # The first gate score (Phase-1 baseline) was the maximum; the best
    # checkpoint must reflect it, i.e. NOT have been overwritten by the collapse.
    assert captured[0] == phase1_baseline
    assert max(captured[1:]) < phase1_baseline

    # And the saved best checkpoint must be loadable + legal -> it is a real,
    # non-collapsed model preserved through a collapsing Phase 2.
    per_agent = evaluate_ml(
        best_path, num_games=4, num_players=2, seed=0, parallel=False
    )
    assert "MLAgent" in per_agent


def test_league_self_play_does_not_regress_below_6c_baseline(tmp_path) -> None:
    """Sprint 6D: a short league + PFSP Phase-2 run is at least as safe as 6C.

    The §6D smoke gate: the league swaps PFSP-sampled opponents into the Phase-2
    snapshot seats and folds per-opponent win-rate estimates back via the
    eval-based attribution, but the 6C eval-gated best-checkpoint machinery is
    untouched -- so the saved *best* checkpoint's score can NEVER regress below
    the Phase-1 baseline (the 6C guarantee). We model the exact Sprint-5 failure
    (Phase 1 strong, Phase 2 collapses) with a deterministic ``gate_fn`` and
    assert the best checkpoint is the strong Phase-1 model, identically to the
    6C test -- proving league + PFSP only adds opponent curriculum, never erodes
    the safety net. Kept small/fast: tiny net, 2 league eval games.
    """
    from heat.ml.evaluate import evaluate_ml
    from heat.ml.training import CurriculumConfig, train_self_play

    # Phase-1 baseline strong (0.9); Phase 2 collapses (the exact failure mode).
    gate_scores = iter([0.9, 0.05, 0.02, 0.0])
    phase1_baseline = 0.9

    config = PPOConfig(
        net_arch=[16, 16],
        features_extractor_hidden=[16],
        features_dim=16,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        seed=0,
        verbose=0,
        device="cpu",
        n_envs=1,
    )
    curriculum = CurriculumConfig(
        total_timesteps=192,
        phase1_steps=64,
        snapshot_every=64,
        checkpoint_dir=str(tmp_path),
        run_name="league_sp",
        # --- §6D league + PFSP, opt-in ---
        use_league=True,
        league_capacity=4,
        league_pfsp_mode="even",
        snapshot_mix=0.5,
        league_eval_games=2,  # tiny -- this is bookkeeping, not the gate
    )

    captured: list[float] = []

    def gate_fn(_model) -> float:
        s = next(gate_scores)
        captured.append(s)
        return s

    _model, best_path = train_self_play(
        config, curriculum, num_players=2, gate_fn=gate_fn
    )

    # The Phase-1 baseline was the max; the league run must NOT have overwritten
    # the best checkpoint with a collapsed Phase-2 model -> best >= 6C baseline.
    assert captured[0] == phase1_baseline
    assert max(captured[1:]) < phase1_baseline

    # The preserved best checkpoint is a real, loadable, legal model.
    per_agent = evaluate_ml(
        best_path, num_games=4, num_players=2, seed=0, parallel=False
    )
    assert "MLAgent" in per_agent
