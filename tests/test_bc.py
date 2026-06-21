"""Tests for Sprint S3 (Behavioral Cloning): gen_demos + train_bc.

Covers the contract-critical guarantees the design flags:

  * Demo tuples are codec-valid and mask-consistent: every logged
    ``(obs, action, mask)`` has the right shapes, the target action is set in
    its own mask, and only real-choice decisions (mask.sum() > 1) are logged
    (matching ``HeatEnv``'s auto-resolve of degenerate decisions).
  * Action-target snapping matches ``env._decode_legal``'s value-multiset rule:
    decoding a logged CARDS target through the SAME path the env uses yields a
    legal play whose value-multiset equals the expert's chosen play.
  * Unencodable expert actions (off-table REACT) are dropped, not crashed.
  * The BC checkpoint loads as an ordinary ``MaskablePPO`` policy and round-trips
    through ``MLAgent`` (the contract sidecar + CODEC_VERSION tripwire passes,
    and the agent returns legal moves).

These run a *tiny* dataset + a 1-epoch train so they stay fast.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# The experiments scripts are not a package; add the dir to the path so the
# Sprint-S3 deliverables can be imported by their module name.
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gen_demos  # noqa: E402
import train_bc  # noqa: E402

from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.search_agent import LookaheadAgent
from heat.ml.action_codec import _play_to_multiset, decode_action, legal_action_mask
from heat.ml.env import HeatEnv
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM


# ---------------------------------------------------------------------------
# gen_demos: dataset generation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_demos(tmp_path_factory) -> dict:
    """Generate a tiny solo demonstration dataset once for the module."""
    out = str(tmp_path_factory.mktemp("bc") / "demos.npz")
    args = _demo_args(out, players=1, train_tracks=2, val_tracks=1)
    gen_demos.generate_dataset(args)
    data = np.load(out)
    return {k: data[k] for k in data.files} | {"path": out}


def _demo_args(out, *, players, train_tracks, val_tracks):
    import argparse

    return argparse.Namespace(
        out=out,
        players=players,
        train_tracks=train_tracks,
        val_tracks=val_tracks,
        horizon=2,
        dets=2,
        top_k=6,
        sim_budget=None,
        open_hand=False,
        expert_seed=0,
        game_seed=7000,
    )


def test_demo_tuples_have_contract_shapes(tiny_demos):
    obs = tiny_demos["obs"]
    mask = tiny_demos["mask"]
    action = tiny_demos["action"]
    assert obs.ndim == 2 and obs.shape[1] == OBS_DIM
    assert mask.shape == (obs.shape[0], ACTION_DIM)
    assert action.shape == (obs.shape[0],)
    assert obs.dtype == np.float32
    # All observations are within the [-1, 1] contract bound.
    assert obs.min() >= -1.0 - 1e-6 and obs.max() <= 1.0 + 1e-6


def test_demo_targets_are_set_in_their_own_mask(tiny_demos):
    """Every logged target action must be legal under its own mask."""
    mask = tiny_demos["mask"]
    action = tiny_demos["action"]
    chosen_legal = mask[np.arange(len(action)), action]
    assert chosen_legal.all(), "a logged target is not set in its mask"


def test_only_real_choices_are_logged(tiny_demos):
    """Only decisions with > 1 legal action are logged (env auto-resolve rule)."""
    mask = tiny_demos["mask"]
    n_legal = mask.sum(axis=1)
    assert (n_legal > 1).all(), "a degenerate (<=1 legal) decision was logged"


def test_train_val_split_is_track_disjoint(tiny_demos):
    """Train and val tuples come from disjoint track-seed bands."""
    split = tiny_demos["split"]
    track_seed = tiny_demos["track_seed"]
    train_seeds = set(track_seed[split == b"train"].tolist())
    val_seeds = set(track_seed[split == b"val"].tolist())
    assert train_seeds and val_seeds
    assert train_seeds.isdisjoint(val_seeds)


def test_sidecar_records_codec_version(tiny_demos):
    import json

    meta_path = os.path.splitext(tiny_demos["path"])[0] + ".demos.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["codec_version"] == CODEC_VERSION
    assert meta["obs_dim"] == OBS_DIM
    assert meta["action_dim"] == ACTION_DIM


# ---------------------------------------------------------------------------
# Action-target snapping matches env._decode_legal
# ---------------------------------------------------------------------------


def test_cards_target_snaps_like_env_decode_legal():
    """A logged CARDS target, decoded the way the env does, is a legal play with
    the SAME value-multiset as the expert's chosen play.

    This is the design's flagged BC risk: the target must round-trip through the
    value-multiset rule, exactly as ``HeatEnv._decode_legal`` snaps a decoded
    play to a legal tuple. We reconstruct a real CARDS decision from a live game,
    have the expert choose, encode the target, then verify
    ``env._decode_legal(decision, target)`` returns a legal play matching the
    expert's value-multiset.
    """
    from heat.tracks.generator import generate_track, TrackGenParams

    params = TrackGenParams(
        num_corners_range=(4, 7), speed_limit_choices=(1, 1, 2, 3), laps=2
    )
    track = generate_track(123_456, params)
    state = GameState.create(track, 1, seed=3)
    for p in state.players:
        p.lap = 1

    expert = LookaheadAgent(horizon=2, n_determinizations=2, top_k=6, seed=0)
    env = HeatEnv(num_players=2)  # only for its _decode_legal method
    env.state = state

    checked = 0
    gen = run_round_driver(state)
    send = None
    while checked < 3:
        if state.is_game_over or state.round_num > MAX_ROUNDS:
            break
        try:
            decision = gen.send(send)
        except StopIteration:
            if state.is_game_over or state.round_num > MAX_ROUNDS:
                break
            gen = run_round_driver(state)
            send = None
            continue

        action = expert.choose_cards(state, decision.player_id, decision.legal) \
            if decision.kind == DecisionKind.CARDS else None
        if decision.kind == DecisionKind.CARDS:
            mask = legal_action_mask(decision, state)
            if mask.sum() > 1:
                target = gen_demos._encode_target(decision, action)
                assert target is not None
                # Decode the target the SAME way the env does and confirm it
                # snaps to a legal play with the expert's value-multiset.
                env.state = state
                snapped = env._decode_legal(decision, int(target))
                assert snapped in decision.legal
                assert _play_to_multiset(snapped) == _play_to_multiset(action)
                checked += 1
            send = action
        else:
            from heat.ml.opponents import opponent_action

            send = opponent_action(expert, decision, state)

    assert checked >= 1, "no multi-choice CARDS decision encountered"


def test_unencodable_react_is_dropped_not_crashed():
    """An off-table REACT action returns None from _encode_target (dropped)."""
    from heat.engine.phases import ReactDecision
    from heat.engine import rules

    opts = rules.ReactOptions(max_cooldown=3, can_boost=True, has_adrenaline=True)
    decision = Decision(DecisionKind.REACT, 0, opts)
    # cooldown_count=1 + adrenaline_speed is legal but NOT in the codec's
    # fixed 8-slot REACT table.
    off_table = ReactDecision(
        cooldown_count=1,
        use_boost=False,
        use_adrenaline_speed=True,
        use_adrenaline_cooldown=False,
    )
    assert gen_demos._encode_target(decision, off_table) is None


# ---------------------------------------------------------------------------
# train_bc: checkpoint loads + round-trips through MLAgent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bc_checkpoint(tiny_demos, tmp_path_factory) -> str:
    """Train a 1-epoch BC checkpoint on the tiny dataset."""
    import argparse

    out = str(tmp_path_factory.mktemp("bc_ckpt") / "bc.zip")
    args = argparse.Namespace(
        data=tiny_demos["path"],
        out=out,
        epochs=1,
        batch=64,
        lr=3e-4,
        eval_every=1,
        net="default",
        device="cpu",
        seed=0,
        players_meta=1,
    )
    train_bc.train_bc(args)
    return out


def test_bc_checkpoint_loads_as_maskable_ppo(bc_checkpoint):
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(bc_checkpoint, device="cpu")
    assert model.observation_space.shape == (OBS_DIM,)
    assert model.action_space.n == ACTION_DIM


def test_bc_checkpoint_has_contract_sidecar(bc_checkpoint):
    from heat.ml.training import load_meta

    meta = load_meta(bc_checkpoint)
    assert meta["obs_dim"] == OBS_DIM
    assert meta["action_dim"] == ACTION_DIM
    assert meta["codec_version"] == CODEC_VERSION


def test_bc_roundtrips_through_ml_agent(bc_checkpoint):
    """The BC checkpoint drives an MLAgent that returns only legal moves.

    Exercises the full contract path: MLAgent lazy-loads the model, the §3.4
    sidecar tripwire passes, and a short game completes with the cloned policy
    choosing legal actions throughout (the engine raises on an illegal move, so
    a completed game proves legality).
    """
    from heat.agents.ml_agent import MLAgent
    from heat.engine.game import Game
    from heat.tracks.generator import generate_track, TrackGenParams

    params = TrackGenParams(
        num_corners_range=(4, 7), speed_limit_choices=(1, 1, 2, 3), laps=2
    )
    track = generate_track(900_111, params)
    agent = MLAgent(bc_checkpoint, deterministic=True, name="BC")
    game = Game(track, [agent], logging_enabled=False, seed=0)
    result = game.run()  # raises if the agent ever returns an illegal move
    assert result.total_rounds >= 1


def test_bc_predicts_legal_action_under_mask(bc_checkpoint):
    """A direct MaskablePPO.predict under a mask returns an in-mask action."""
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(bc_checkpoint, device="cpu")
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, size=(OBS_DIM,)).astype(np.float32)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    # Make GEAR 1..3 legal.
    mask[0:3] = True
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    assert bool(mask[int(np.asarray(action).reshape(-1)[0])])
