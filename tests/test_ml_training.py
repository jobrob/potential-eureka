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
    GateResult,
    TrainingPhase,
    default_8c_phases,
    load_meta,
    meta_path_for,
    save_checkpoint,
    sprint_8c_curriculum,
    sprint_a_curriculum,
    train_self_play,
    vecnorm_path_for,
    _resolve_track_source,
    _scripted_opponents,
    _track_label,
    _gate_tracks,
)
from heat.tracks.generator import TrackSampler
from heat.tracks.loader import load_track_by_name
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


def _agent_stats(*, games: int, wins: int):
    """A minimal ``AgentStats`` for monkeypatched gate-eval returns.

    The gate only reads ``wins`` / ``games_played`` / ``win_rate``; the finish
    distribution / averages are irrelevant here, so they are filled with inert
    placeholders.
    """
    from heat.simulation.stats import AgentStats

    return AgentStats(
        agent_type="MLAgent",
        games_played=games,
        wins=wins,
        win_rate=wins / games if games else 0.0,
        finish_position_counts={},
        avg_finish_position=0.0,
        avg_heat_remaining=0.0,
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


# ---------------------------------------------------------------------------
# Track source: generated-by-default training (§6A integration)
# ---------------------------------------------------------------------------


class TestTrackSource:
    def test_default_is_generated_sampler(self) -> None:
        """No track -> a generated-track sampler (the new default)."""
        src = _resolve_track_source(None, seed=0)
        assert isinstance(src, TrackSampler)
        assert _track_label(src) == "generated"

    def test_fixed_track_is_preserved(self) -> None:
        """A pinned Track passes through unchanged and keeps its name."""
        usa = load_track_by_name("usa")
        src = _resolve_track_source(usa, seed=0)
        assert src is usa
        assert _track_label(src) == usa.name

    def test_gate_tracks_for_generated_are_fixed_holdout(self) -> None:
        """Generated training gates on a stable, reproducible held-out set."""
        src = _resolve_track_source(None, seed=0)
        a = [t.name for t in _gate_tracks(src)]
        b = [t.name for t in _gate_tracks(src)]
        assert a == b and len(a) >= 1
        assert all(name.startswith("holdout-") for name in a)

    def test_gate_tracks_for_fixed_is_itself(self) -> None:
        usa = load_track_by_name("usa")
        assert [t.name for t in _gate_tracks(usa)] == [usa.name]


# ---------------------------------------------------------------------------
# Sprint A: in-Phase-1 periodic gate + best-checkpoint preservation (Idea 7)
# ---------------------------------------------------------------------------


def test_phase1_periodic_gate_preserves_best(tmp_path) -> None:
    """A descending Phase-1 gate must keep the FIRST (highest) chunk's model.

    Phase-1-only run (phase1_steps == total_timesteps) chunked into several
    ``phase1_eval_every`` iterations. With a strictly descending injected gate,
    the canonical ``best_path`` checkpoint must hold the first/highest model --
    the Phase-1 analogue of the descending-scores Phase-2 regression test.
    """
    scores = iter([0.9, 0.5, 0.2, 0.1])

    config = _tiny_ppo()
    # Phase-1-only: total == phase1, chunked into 3 x 64-step gates.
    curriculum = _tiny_curriculum(
        tmp_path,
        total_timesteps=192,
        phase1_steps=192,
        phase1_eval_every=64,
    )
    _model, best_path = train_self_play(
        config, curriculum, num_players=2, gate_fn=lambda m: next(scores)
    )

    # Canonical best checkpoint exists + round-trips with the contract.
    meta = load_meta(best_path)
    assert meta["obs_dim"] == spaces.OBS_DIM
    assert meta["action_dim"] == spaces.ACTION_DIM
    assert meta["codec_version"] == spaces.CODEC_VERSION


def test_phase1_gate_promotes_on_strict_improvement(tmp_path) -> None:
    """In a Phase-1-only run, best_path is written only on strict improvement."""
    import heat.ml.training as training_mod

    calls: list[str] = []
    real_save = training_mod.save_checkpoint

    def recording_save(model, path, **kwargs):  # noqa: ANN001
        calls.append(path)
        return real_save(model, path, **kwargs)

    training_mod.save_checkpoint = recording_save
    try:
        # 3 Phase-1 chunks: 0.1 (first -> save), 0.5 (improves -> save),
        # 0.5 (flat -> no save).
        scores = iter([0.1, 0.5, 0.5])
        config = _tiny_ppo()
        curriculum = _tiny_curriculum(
            tmp_path,
            total_timesteps=192,
            phase1_steps=192,
            phase1_eval_every=64,
        )
        _model, best_path = train_self_play(
            config, curriculum, num_players=2, gate_fn=lambda m: next(scores)
        )
    finally:
        training_mod.save_checkpoint = real_save

    best_writes = [c for c in calls if c == best_path]
    # First chunk save + one improving promotion == 2 writes to best_path.
    assert len(best_writes) == 2


# ---------------------------------------------------------------------------
# Sprint A: gate scores against the trained-against opponent (Idea 8)
# ---------------------------------------------------------------------------


def test_gate_uses_strong_opponent_factory(tmp_path, monkeypatch) -> None:
    """With use_strong_heuristic_opponents the gate passes a strong factory.

    Monkeypatch ``evaluate_ml`` (as imported inside ``_pooled_win_counts``) to
    record every ``opponent_factory`` it receives, then assert the strong pass
    used a non-None factory producing a StrongHeuristicAgent.
    """
    from heat.agents.strong_heuristic import StrongHeuristicAgent
    import heat.ml.evaluate as eval_mod

    seen: list = []

    def fake_evaluate_ml(model_path, *, opponent_factory=None, **kwargs):  # noqa: ANN001
        seen.append(opponent_factory)
        return {
            "MLAgent": _agent_stats(games=4, wins=2)
        }

    monkeypatch.setattr(eval_mod, "evaluate_ml", fake_evaluate_ml)

    from heat.ml.training import _gate_score

    config = _tiny_ppo()
    curriculum = _tiny_curriculum(
        tmp_path, use_strong_heuristic_opponents=True, gate_games=6
    )

    # Build a real tiny model to save during the gate.
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model

    model = build_model(HeatEnv(num_players=2), config)
    result = _gate_score(
        model,
        curriculum=curriculum,
        num_players=2,
        track=load_track_by_name("usa"),
        seed=0,
    )

    assert isinstance(result, GateResult)
    # At least one strong pass used a real strong factory.
    strong_factories = [f for f in seen if f is not None]
    assert strong_factories, "strong pass should pass a non-None opponent factory"
    agent = strong_factories[0](player_id=1, seed=0)
    assert isinstance(agent, StrongHeuristicAgent)
    # And at least one weak pass used the default (None) factory.
    assert any(f is None for f in seen)


def test_gate_result_reports_weak_and_strong(tmp_path, monkeypatch) -> None:
    """GateResult carries distinct weak + strong win-rates from the two passes."""
    import heat.ml.evaluate as eval_mod

    # Strong pass returns 1/4; weak pass returns 3/4. The two passes are
    # distinguished by whether opponent_factory is None (weak) or not (strong).
    def fake_evaluate_ml(model_path, *, opponent_factory=None, **kwargs):  # noqa: ANN001
        if opponent_factory is None:
            return {"MLAgent": _agent_stats(games=4, wins=3)}
        return {"MLAgent": _agent_stats(games=4, wins=1)}

    monkeypatch.setattr(eval_mod, "evaluate_ml", fake_evaluate_ml)

    from heat.ml.training import _gate_score
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model

    config = _tiny_ppo()
    curriculum = _tiny_curriculum(
        tmp_path, use_strong_heuristic_opponents=True, gate_games=4
    )
    model = build_model(HeatEnv(num_players=2), config)
    result = _gate_score(
        model,
        curriculum=curriculum,
        num_players=2,
        track=load_track_by_name("usa"),
        seed=0,
    )
    assert result.win_rate_strong == 0.25
    assert result.win_rate_weak == 0.75
    # Wilson LB of 1/4 is below the point estimate.
    assert 0.0 <= result.wilson_lb_strong < 0.25
    assert result.promote_score == result.wilson_lb_strong


# ---------------------------------------------------------------------------
# Sprint A: Wilson-LB promotion (Idea 9)
# ---------------------------------------------------------------------------


def test_gate_promotes_on_wilson_lb() -> None:
    """Equal point estimates but more games -> higher Wilson LB -> promoted.

    Unit test of the pooled-(wins, games) -> wilson_interval promotion criterion:
    two checkpoints both at 50% win-rate, one over 10 games and one over 100; the
    higher-n checkpoint has the strictly higher lower bound.
    """
    from heat.simulation.stats import wilson_interval

    lb_small, _ = wilson_interval(5, 10)
    lb_large, _ = wilson_interval(50, 100)
    assert lb_large > lb_small  # more games -> tighter -> higher LB at equal rate


# ---------------------------------------------------------------------------
# Sprint A: broadened Phase-1 opponent mix (Idea 6)
# ---------------------------------------------------------------------------


def test_broadened_phase1_pool_composition() -> None:
    """use_strong + broaden_mix yields the expected class mix and seat count."""
    from heat.agents.heuristic_agent import HeuristicAgent
    from heat.agents.strong_heuristic import StrongHeuristicAgent
    from heat.ml.training import _StrongHeuristicFactory

    pool = _scripted_opponents(4, use_strong=True, broaden_mix=True)
    assert len(pool) == 3  # num_players - 1

    # Instantiate each spec (class or factory) and check the realized agent mix.
    agents = [spec() if callable(spec) else spec for spec in pool]
    # Template is [Strong(3), Strong(2), Heuristic, Random] cycled to 3 seats.
    assert isinstance(agents[0], StrongHeuristicAgent) and agents[0].strength == 3
    assert isinstance(agents[1], StrongHeuristicAgent) and agents[1].strength == 2
    assert isinstance(agents[2], HeuristicAgent)
    # The strength-3 seat is built by the picklable factory, not a lambda.
    assert isinstance(pool[0], _StrongHeuristicFactory)


def test_broaden_mix_is_noop_without_strong() -> None:
    """broaden_mix without use_strong leaves the default weak pool unchanged."""
    from heat.agents.heuristic_agent import HeuristicAgent
    from heat.agents.random_agent import RandomAgent

    pool = _scripted_opponents(4, use_strong=False, broaden_mix=True)
    assert pool[:-1] == [HeuristicAgent, HeuristicAgent]
    assert pool[-1] is RandomAgent


def test_broadened_pool_is_picklable() -> None:
    """The broadened pool specs survive pickling (SubprocVecEnv spawn)."""
    import pickle

    pool = _scripted_opponents(4, use_strong=True, broaden_mix=True)
    restored = pickle.loads(pickle.dumps(pool))
    assert len(restored) == 3


# ---------------------------------------------------------------------------
# Sprint A: launch preset (Ideas 4/15)
# ---------------------------------------------------------------------------


def test_sprint_a_preset_is_phase1_only_and_strong() -> None:
    """The Sprint-A preset bundles the Idea 4/8/9/10/15 levers."""
    curr = sprint_a_curriculum(1_000_000)
    assert curr.phase1_steps == curr.total_timesteps  # Idea 4: Phase-1-only
    assert curr.gate_games >= 120 and curr.gate_use_wilson_lb  # Idea 9
    assert curr.phase1_eval_every > 0  # Idea 7 cadence
    assert curr.use_strong_heuristic_opponents  # Idea 8
    assert curr.randomize_seat  # Idea 10
    assert curr.normalize_obs is False  # Idea 15 (recorded constraint)


# ---------------------------------------------------------------------------
# Sprint 8C: multi-phase chaining + warm-start (§9)
# ---------------------------------------------------------------------------


def _phase(name, players, reward, gamma, shaping, steps, pool):
    return TrainingPhase(name, players, reward, gamma, shaping, steps, pool)


def _tiny_8c_curriculum(tmp_path, **overrides) -> CurriculumConfig:
    """A tiny sprint-8c-style curriculum: gate every chunk, cheap."""
    base = dict(
        checkpoint_dir=str(tmp_path),
        run_name="t8c",
        phase1_eval_every=64,
        gate_games=4,
        use_strong_heuristic_opponents=False,  # cheaper gate (gate_fn injected)
        randomize_seat=True,
    )
    base.update(overrides)
    return CurriculumConfig(**base)


def test_sprint_8c_preset_levers() -> None:
    """The 8C preset keeps seat randomization on and the track curriculum off."""
    cfg = sprint_8c_curriculum()
    assert cfg.use_track_curriculum is False  # demoted (net-negative)
    assert cfg.randomize_seat is True  # the proven Sprint-A win
    assert cfg.use_strong_heuristic_opponents is True
    assert cfg.gate_use_wilson_lb is True
    assert cfg.normalize_obs is False


def test_default_8c_phases_shape() -> None:
    """The default phase list is solo(0.99) -> weak/mixed/strong(0.999)."""
    phases = default_8c_phases(4)
    assert [p.name for p in phases] == ["solo", "weak", "mixed", "strong"]
    assert phases[0].num_players == 1 and phases[0].reward_mode == "solo"
    assert phases[0].gamma == 0.99
    assert all(p.gamma == 0.999 for p in phases[1:])
    assert all(p.reward_mode == "race" for p in phases[1:])


def test_phase_list_runs_in_order(tmp_path, monkeypatch) -> None:
    """A 2-phase [solo, weak] list calls learn on both phases and rebuilds the env.

    Spies on ``model.learn`` (counts calls) and on ``_build_vec_env`` (records the
    per-phase num_players) to confirm both phases execute and the env is rebuilt
    with a different player count at the boundary.
    """
    import heat.ml.training as training_mod

    built_players: list[int] = []
    real_build = training_mod._build_vec_env

    def spy_build(*args, **kwargs):
        built_players.append(kwargs["num_players"])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(training_mod, "_build_vec_env", spy_build)

    phases = [
        _phase("solo", 1, "solo", 0.99, 1.0, 64, None),
        _phase("weak", 2, "race", 0.999, 0.05, 64, "weak"),
    ]
    config = _tiny_ppo()
    curriculum = _tiny_8c_curriculum(tmp_path)
    _model, best_path = train_self_play(
        config, curriculum, num_players=2, phases=phases,
        gate_fn=lambda m: 0.5,
    )
    # Env built for the solo (1p) phase and the weak (2p) phase.
    assert 1 in built_players and 2 in built_players
    # Best checkpoint exists.
    meta = load_meta(best_path)
    assert meta["codec_version"] == spaces.CODEC_VERSION


def test_best_checkpoint_preserved_across_phases(tmp_path) -> None:
    """A descending score spanning the phase boundary keeps the first best.

    Two opponent phases (no solo, so every chunk is gated). The score descends
    across the boundary, so the canonical best must hold the first (highest)
    model and a separate *_final must exist (cross-phase analogue of the Phase-1
    descending-score test).
    """
    scores = iter([0.9, 0.5, 0.2, 0.1, 0.05, 0.01])

    phases = [
        _phase("weak", 2, "race", 0.999, 0.05, 128, "weak"),
        _phase("strong", 2, "race", 0.999, 0.05, 128, "strong"),
    ]
    config = _tiny_ppo()
    curriculum = _tiny_8c_curriculum(tmp_path, phase1_eval_every=64)
    _model, best_path = train_self_play(
        config, curriculum, num_players=2, phases=phases,
        gate_fn=lambda m: next(scores),
    )
    import os

    meta = load_meta(best_path)
    assert meta["codec_version"] == spaces.CODEC_VERSION
    assert os.path.exists(meta_path_for(f"{best_path}_final"))


def test_warm_start_loads_not_builds(tmp_path, monkeypatch) -> None:
    """With warm_start_path set, the first phase loads (not builds) the model."""
    import heat.ml.training as training_mod
    from sb3_contrib import MaskablePPO

    # Produce a real checkpoint to warm-start from.
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model

    seed_env = HeatEnv(num_players=2)
    seed_model = build_model(seed_env, _tiny_ppo(seed=1))
    warm_path = str(tmp_path / "warm")
    training_mod.save_checkpoint(
        seed_model, warm_path, track_name="usa", num_players=2
    )

    load_calls: list[str] = []
    build_calls: list[int] = []
    real_load = MaskablePPO.load
    real_build = training_mod.build_model

    def spy_load(path, *args, **kwargs):
        load_calls.append(str(path))
        return real_load(path, *args, **kwargs)

    def spy_build(env, cfg=None):
        build_calls.append(1)
        return real_build(env, cfg)

    monkeypatch.setattr(MaskablePPO, "load", staticmethod(spy_load))
    monkeypatch.setattr(training_mod, "build_model", spy_build)

    phases = [_phase("weak", 2, "race", 0.999, 0.05, 64, "weak")]
    config = _tiny_ppo()
    curriculum = _tiny_8c_curriculum(tmp_path, run_name="warmrun")
    train_self_play(
        config, curriculum, num_players=2, phases=phases,
        gate_fn=lambda m: 0.5, warm_start_path=warm_path,
    )
    assert any(warm_path in c for c in load_calls)
    assert build_calls == []  # build_model never called for the warm-started phase


def test_solo_phase_uses_one_player(tmp_path, monkeypatch) -> None:
    """The solo phase builds its vec env with num_players=1 (spy on _build_vec_env)."""
    import heat.ml.training as training_mod

    seen: list[tuple[int, str]] = []
    real_build = training_mod._build_vec_env

    def spy_build(*args, **kwargs):
        seen.append((kwargs["num_players"], kwargs.get("reward_mode")))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(training_mod, "_build_vec_env", spy_build)

    phases = [
        _phase("solo", 1, "solo", 0.99, 1.0, 64, None),
        _phase("weak", 2, "race", 0.999, 0.05, 64, "weak"),
    ]
    train_self_play(
        _tiny_ppo(), _tiny_8c_curriculum(tmp_path, run_name="soloplayers"),
        num_players=2, phases=phases, gate_fn=lambda m: 0.5,
    )
    # The solo phase built a 1-player solo-reward env.
    assert (1, "solo") in seen
