"""Tests for Sprint S4: DAgger loop + BC->PPO warm-start + KL-to-BC regularizer.

Covers the contract-critical guarantees the design flags for S4:

  * DAgger labels are codec-valid and mask-consistent, logged ONLY on real-choice
    (mask.sum() > 1) LEARNER-visited states, with the target set in its own mask
    -- the same contract gen_demos enforces, now on the learner's distribution.
  * Unencodable EXPERT actions (off-table REACT) are dropped + counted, never
    crashed or fabricated (the same ~3-4% drop gen_demos applies).
  * The DAgger trajectory follows the LEARNER's distribution (the learner's action
    advances the game; the expert is queried only for the label) -- the property
    that distinguishes DAgger from gen_demos and fixes the off-distribution gap.
  * Dataset aggregation concatenates parts on-contract and preserves row/split
    alignment (DAgger's D <- D u D_i).
  * The warm-start path loads a BC checkpoint and the produced policy plays only
    legal moves through MLAgent (the fine-tune entry point's core contract).
  * The KL-to-BC regularizer: zero against its own reference, positive against a
    drifted policy, a true no-op when disabled, and differentiable.

These run a *tiny* dataset + 1-epoch trains so they stay fast.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

# The experiments scripts are not a package; add the dir to the path so the
# Sprint-S3/S4 deliverables can be imported by module name.
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import dagger  # noqa: E402
import gen_demos  # noqa: E402
import train_bc  # noqa: E402

from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM


# ---------------------------------------------------------------------------
# Fixtures: a tiny seed dataset + a 1-epoch BC checkpoint to roll out
# ---------------------------------------------------------------------------


def _demo_args(out, *, players, train_tracks, val_tracks):
    return argparse.Namespace(
        out=out, players=players, train_tracks=train_tracks,
        val_tracks=val_tracks, horizon=2, dets=2, top_k=6, sim_budget=None,
        open_hand=False, expert_seed=0, game_seed=7000,
    )


def _expert_args(players=1):
    return argparse.Namespace(
        players=players, horizon=2, dets=2, top_k=6, sim_budget=None,
        open_hand=False, expert_seed=0,
    )


@pytest.fixture(scope="module")
def seed_dataset(tmp_path_factory) -> str:
    """A tiny solo seed demonstration dataset (gen_demos), once for the module."""
    out = str(tmp_path_factory.mktemp("s4") / "seed.npz")
    gen_demos.generate_dataset(_demo_args(out, players=1, train_tracks=2,
                                          val_tracks=1))
    return out


@pytest.fixture(scope="module")
def bc_checkpoint(seed_dataset, tmp_path_factory) -> str:
    """A 1-epoch BC checkpoint trained on the seed dataset (the iter-0 learner)."""
    out = str(tmp_path_factory.mktemp("s4_bc") / "bc.zip")
    bc_args = argparse.Namespace(
        data=seed_dataset, out=out, epochs=1, batch=64, lr=3e-4, eval_every=1,
        net="default", device="cpu", seed=0, players_meta=1,
    )
    train_bc.train_bc(bc_args)
    return out


@pytest.fixture(scope="module")
def dagger_data(bc_checkpoint) -> dict:
    """One DAgger collection over the learner's own states (tiny solo)."""
    arrays, stats = dagger.collect_dagger_dataset(
        bc_checkpoint, num_players=1, train_tracks=2, val_tracks=1,
        expert_args=_expert_args(1), game_seed_base=8000,
    )
    return {"arrays": arrays, "stats": stats}


# ---------------------------------------------------------------------------
# DAgger label contract (mirrors gen_demos, on the learner's distribution)
# ---------------------------------------------------------------------------


def test_dagger_labels_have_contract_shapes(dagger_data):
    a = dagger_data["arrays"]
    assert a["obs"].ndim == 2 and a["obs"].shape[1] == OBS_DIM
    assert a["mask"].shape == (a["obs"].shape[0], ACTION_DIM)
    assert a["action"].shape == (a["obs"].shape[0],)
    assert a["obs"].dtype == np.float32
    # Some tuples must have been collected (the learner visits real choices).
    assert a["obs"].shape[0] > 0


def test_dagger_targets_are_set_in_their_own_mask(dagger_data):
    """Every EXPERT label is legal under the LEARNER state's mask."""
    a = dagger_data["arrays"]
    chosen_legal = a["mask"][np.arange(len(a["action"])), a["action"]]
    assert chosen_legal.all(), "a DAgger expert target is not set in its mask"


def test_dagger_only_logs_real_choices(dagger_data):
    """Only learner decisions with > 1 legal action are logged (env auto-resolve)."""
    a = dagger_data["arrays"]
    n_legal = a["mask"].sum(axis=1)
    assert (n_legal > 1).all(), "a degenerate (<=1 legal) decision was logged"


def test_dagger_tracks_come_from_train_val_bands(dagger_data):
    """DAgger rollouts draw from the SAME train/val bands as the seed dataset.

    They must stay disjoint from the held-out gate band (900_000+), so a DAgger
    track is never one the behavioral gate is run on.
    """
    a = dagger_data["arrays"]
    seeds = set(a["track_seed"].tolist())
    assert seeds, "no track seeds recorded"
    # Every seed is in the train band [TRAIN_BASE, TRAIN_BASE+N) or the val band.
    for s in seeds:
        in_train = gen_demos._TRAIN_SEED_BASE <= s < gen_demos._TRAIN_SEED_BASE + 1000
        in_val = gen_demos._VAL_SEED_BASE <= s < gen_demos._VAL_SEED_BASE + 1000
        assert in_train or in_val, f"seed {s} outside the train/val bands"
        assert s < 900_000, f"seed {s} collides with the held-out gate band"


def test_dagger_reports_dropped_unencodable(dagger_data):
    """Unencodable expert actions are counted (dropped, not crashed).

    The collection completed without raising, and the dropped count is a
    non-negative integer (the off-table REACT drop gen_demos also applies).
    """
    st = dagger_data["stats"]
    assert isinstance(st["unencodable_dropped"], int)
    assert st["unencodable_dropped"] >= 0


def test_unencodable_react_is_dropped_not_crashed():
    """An off-table REACT expert action returns None from the shared encoder."""
    from heat.engine.driver import Decision, DecisionKind
    from heat.engine.phases import ReactDecision
    from heat.engine import rules

    opts = rules.ReactOptions(max_cooldown=3, can_boost=True, has_adrenaline=True)
    decision = Decision(DecisionKind.REACT, 0, opts)
    off_table = ReactDecision(
        cooldown_count=1, use_boost=False,
        use_adrenaline_speed=True, use_adrenaline_cooldown=False,
    )
    # DAgger reuses gen_demos._encode_target (single source of truth).
    assert gen_demos._encode_target(decision, off_table) is None


# ---------------------------------------------------------------------------
# DAgger trajectory follows the LEARNER's distribution (the key property)
# ---------------------------------------------------------------------------


def test_dagger_trajectory_is_the_learners_not_the_experts(bc_checkpoint):
    """The game advances with the LEARNER's action, not the expert's.

    We roll out twice on the same track/seed with the SAME expert config but two
    different learners (the BC net vs an always-gear-1 stub). If the trajectory
    were driven by the expert (as gen_demos is), the two runs would visit
    identical states and log identical observations. Because DAgger drives the
    game with the LEARNER's action, the two learners visit DIFFERENT states, so
    the logged observation sets differ. This is the property that makes DAgger
    collect the recovery states BC misses.
    """
    from heat.agents.base import BaseAgent
    from heat.engine.driver import DecisionKind
    from heat.agents.heuristic_agent import HeuristicAgent
    from heat.agents.ml_agent import MLAgent

    class _GearOneAgent(HeuristicAgent):
        """A degenerate learner that always crawls in the lowest gear/play."""

        def choose_gear(self, state, player_id, legal_gears):
            # Pick the lowest available gear (crawl).
            return min(legal_gears, key=lambda g: g[0])

    # Roll out the BC learner.
    buf_bc = gen_demos._DemoBuffer()
    dagger._rollout_one_track(
        gen_demos._TRAIN_SEED_BASE, "train", num_players=1, game_seed=8000,
        learner=MLAgent(bc_checkpoint, deterministic=True),
        expert=gen_demos._make_expert(1, _expert_args(1)),
        opponent=HeuristicAgent(), buf=buf_bc,
    )
    # Roll out the crawl learner on the identical track/seed.
    buf_crawl = gen_demos._DemoBuffer()
    dagger._rollout_one_track(
        gen_demos._TRAIN_SEED_BASE, "train", num_players=1, game_seed=8000,
        learner=_GearOneAgent(),
        expert=gen_demos._make_expert(1, _expert_args(1)),
        opponent=HeuristicAgent(), buf=buf_crawl,
    )

    # Different learners drive to different states -> different logged obs.
    assert len(buf_bc) > 0 and len(buf_crawl) > 0
    obs_bc = np.asarray(buf_bc.obs, dtype=np.float32)
    obs_crawl = np.asarray(buf_crawl.obs, dtype=np.float32)
    # The trajectories diverge: either a different number of decisions, or (if the
    # same count) at least one differing observation row.
    different = obs_bc.shape != obs_crawl.shape or not np.allclose(
        obs_bc, obs_crawl
    )
    assert different, (
        "BC and crawl learners produced identical state sets -- the trajectory is "
        "NOT following the learner's distribution (DAgger property violated)"
    )


# ---------------------------------------------------------------------------
# Aggregation (DAgger's D <- D u D_i)
# ---------------------------------------------------------------------------


def test_aggregate_concatenates_on_contract(seed_dataset, dagger_data):
    seed = dagger._load_npz(seed_dataset)
    new = dagger_data["arrays"]
    agg = dagger.aggregate_datasets([seed, new])
    assert agg["obs"].shape[0] == seed["obs"].shape[0] + new["obs"].shape[0]
    assert agg["obs"].shape[1] == OBS_DIM
    assert agg["mask"].shape[1] == ACTION_DIM
    # Every aggregated target is still in its own mask.
    legal = agg["mask"][np.arange(len(agg["action"])), agg["action"]]
    assert legal.all()
    # Split column stays aligned: it has exactly the seed + new rows of each kind.
    assert agg["split"].shape[0] == agg["obs"].shape[0]


def test_aggregate_skips_empty_parts(seed_dataset):
    seed = dagger._load_npz(seed_dataset)
    empty = {
        "obs": np.zeros((0, OBS_DIM), np.float32),
        "action": np.zeros((0,), np.int64),
        "mask": np.zeros((0, ACTION_DIM), bool),
        "kind": np.zeros((0,), np.int8),
        "track_seed": np.zeros((0,), np.int64),
        "split": np.zeros((0,), "S5"),
    }
    agg = dagger.aggregate_datasets([seed, empty])
    assert agg["obs"].shape[0] == seed["obs"].shape[0]


# ---------------------------------------------------------------------------
# Full DAgger loop end-to-end (from a seed checkpoint)
# ---------------------------------------------------------------------------


def test_run_dagger_loop_produces_loadable_checkpoint(
    bc_checkpoint, seed_dataset, tmp_path
):
    """One DAgger iteration from a seed checkpoint yields a contract-valid net."""
    from heat.agents.ml_agent import MLAgent
    from heat.engine.game import Game
    from heat.tracks.generator import generate_track, TrackGenParams

    prefix = str(tmp_path / "dag")
    args = argparse.Namespace(
        seed_data=seed_dataset, seed_ckpt=bc_checkpoint, out_prefix=prefix,
        iterations=1, rollout_train_tracks=2, rollout_val_tracks=1,
        players=1, horizon=2, dets=2, top_k=6, sim_budget=None, open_hand=False,
        expert_seed=0, game_seed=8000,
        epochs=1, batch=64, lr=3e-4, net="default", device="cpu", seed=0,
    )
    summary = dagger.run_dagger(args)
    final = summary["final_ckpt"]
    assert os.path.exists(final), "DAgger did not write a final checkpoint"

    # The produced net loads as MLAgent and plays only legal moves (the engine
    # raises on an illegal move, so a completed game proves legality).
    params = TrackGenParams(
        num_corners_range=(4, 7), speed_limit_choices=(1, 1, 2, 3), laps=2
    )
    track = generate_track(900_222, params)
    agent = MLAgent(final, deterministic=True, name="DAgger")
    result = Game(track, [agent], logging_enabled=False, seed=0).run()
    assert result.total_rounds >= 1


# ---------------------------------------------------------------------------
# BC -> PPO warm-start (the fine-tune entry point's core contract)
# ---------------------------------------------------------------------------


def test_warm_start_loads_bc_and_plays_legal(bc_checkpoint):
    """train_self_play warm-starts from the BC checkpoint and produces a model
    whose policy plays only legal moves through MLAgent.

    Exercises the warm_start_path hook directly (a single tiny phase) so the
    fine-tune entry point's core -- "load the BC weights, keep playing legally" --
    is covered without a full ramp.
    """
    import dataclasses

    from heat.agents.ml_agent import MLAgent
    from heat.engine.game import Game
    from heat.ml.model import PPOConfig
    from heat.ml.training import (
        TrainingPhase, sprint_8c_curriculum, train_self_play,
    )
    from heat.tracks.generator import generate_track, TrackGenParams

    cfg = PPOConfig(device="cpu", seed=0, n_envs=1, n_steps=128, batch_size=32)
    cur = sprint_8c_curriculum(
        total_timesteps=128, run_name="test_s4_warm",
        checkpoint_dir=str(_tmp_ckpt_dir()),
    )
    cur = dataclasses.replace(cur, phase1_eval_every=128, gate_games=1)
    phases = [TrainingPhase("weak", 2, "race", 0.999, 0.05, 128, "weak")]

    model, best = train_self_play(
        config=cfg, curriculum=cur, num_players=2,
        warm_start_path=bc_checkpoint, phases=phases,
    )
    assert os.path.exists(best) or os.path.exists(best + ".zip")

    params = TrackGenParams(
        num_corners_range=(4, 7), speed_limit_choices=(1, 1, 2, 3), laps=2
    )
    track = generate_track(900_333, params)
    agent = MLAgent(best, deterministic=True, name="Warm")
    result = Game(track, [agent], logging_enabled=False, seed=0).run()
    assert result.total_rounds >= 1


def _tmp_ckpt_dir():
    import tempfile
    return tempfile.mkdtemp(prefix="s4_warm_")


# ---------------------------------------------------------------------------
# KL-to-BC anti-forgetting regularizer
# ---------------------------------------------------------------------------


def test_kl_to_bc_zero_against_self_positive_against_drift(bc_checkpoint):
    """KL(ref||ref) == 0; KL(drifted||ref) > 0; differentiable; clears cleanly."""
    import torch
    from sb3_contrib import MaskablePPO

    from heat.ml import kl_regularizer as klr

    klr.set_kl_to_bc(reference_path=bc_checkpoint, coef=0.5, device="cpu")
    try:
        ref = klr._STATE.reference
        assert ref is not None and klr._STATE.coef == 0.5

        obs = torch.zeros(4, OBS_DIM)
        mask = np.zeros((4, ACTION_DIM), dtype=bool)
        mask[:, 0:5] = True

        # KL of a fresh (non-frozen) copy of the reference against the reference
        # is ~0, and -- because the CURRENT policy's params carry grad -- the KL
        # is differentiable w.r.t. the actor (the anti-forgetting lever).
        current = MaskablePPO.load(bc_checkpoint, device="cpu")
        kl_self = klr._kl_to_reference(current.policy, obs, mask)
        assert kl_self.requires_grad
        assert abs(float(kl_self.detach())) < 1e-4

        other = MaskablePPO.load(bc_checkpoint, device="cpu")
        with torch.no_grad():
            for p in other.policy.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        kl_drift = klr._kl_to_reference(other.policy, obs, mask)
        assert float(kl_drift.detach()) > 0.0
    finally:
        klr.clear_kl_to_bc()
    assert klr._STATE.reference is None and klr._STATE.coef == 0.0


def test_kl_to_bc_disabled_is_noop_train(bc_checkpoint):
    """With coef<=0 the class-level train wrapper delegates to stock MaskablePPO.

    set_kl_to_bc(coef=0) must NOT install an active penalty: the reference is
    cleared and a subsequent stock-MaskablePPO train path is unchanged. We assert
    the disabled state directly (the patched train is a pass-through when
    reference is None).
    """
    from heat.ml import kl_regularizer as klr

    klr.set_kl_to_bc(reference_path=bc_checkpoint, coef=0.0, device="cpu")
    assert klr._STATE.reference is None
    assert klr._STATE.coef == 0.0
