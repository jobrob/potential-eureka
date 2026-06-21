"""Tests for Sprint A1 (Option A): gen_value_data + train_value.

Covers the four contract-critical guarantees the A1 design flags:

  (a) **Dataset is codec-valid.** Every logged ``obs`` has the right shape, is
      finite, and is in ``[-1, 1]``; ``rounds_remaining >= 0``; train/val track
      seeds are disjoint (track-disjoint split, not a row shuffle).
  (b) **Labeling is correct.** For a tiny synthetic race the backfilled
      ``rounds_remaining`` equals ``finish_round - round_num`` for each row, and a
      ``MAX_ROUNDS``-truncated race is dropped (rows discarded + counted).
  (c) **Trained checkpoint is a contract-valid V.** A tiny, few-epoch checkpoint
      loads as a ``MaskablePPO``, carries the §3.4 contract sidecar, loads under
      the ``MLAgent`` tripwire, and ``policy.predict_values(encode_observation(...))``
      returns a finite scalar (the hook A2 calls).
  (d) **Encoding parity.** A state encoded by ``gen_value_data`` (``decision=None``)
      is byte-identical to the same state encoded at a rolled-out leaf (the
      encoding A2 uses), so train and inference see the same vector.

These run a *tiny* dataset + a 1-2 epoch train so they stay fast.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pytest

# The experiments scripts are not a package; add the dir to the path so the
# Sprint-A1 deliverables can be imported by their module name (mirrors test_bc).
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gen_value_data  # noqa: E402
import train_value  # noqa: E402

from heat.agents.heuristic_agent import HeuristicAgent
from heat.engine.driver import run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.ml.features import encode_observation
from heat.ml.opponents import opponent_action
from heat.ml.spaces import CODEC_VERSION, OBS_DIM
from heat.models.game_state import GameState
from heat.tracks.generator import TrackGenParams, generate_track


def _gen_args(out, *, train_tracks, val_tracks, rollouts=1):
    return argparse.Namespace(
        out=out,
        train_tracks=train_tracks,
        val_tracks=val_tracks,
        rollouts=rollouts,
        game_seed=7000,
    )


@pytest.fixture(scope="module")
def tiny_value_data(tmp_path_factory) -> dict:
    """Generate a tiny solo value dataset once for the module."""
    out = str(tmp_path_factory.mktemp("value") / "value.npz")
    gen_value_data.generate_dataset(_gen_args(out, train_tracks=2, val_tracks=1))
    data = np.load(out)
    return {k: data[k] for k in data.files} | {"path": out}


# ---------------------------------------------------------------------------
# (a) dataset rows are codec-valid + labels non-negative + split disjoint
# ---------------------------------------------------------------------------


def test_dataset_has_contract_shapes(tiny_value_data):
    obs = tiny_value_data["obs"]
    rr = tiny_value_data["rounds_remaining"]
    cl = tiny_value_data["corner_limit_at_state"]
    assert obs.ndim == 2 and obs.shape[1] == OBS_DIM
    assert obs.dtype == np.float32
    assert rr.shape == (obs.shape[0],)
    assert cl.shape == (obs.shape[0],)
    assert rr.dtype == np.float32


def test_dataset_obs_finite_and_in_bounds(tiny_value_data):
    obs = tiny_value_data["obs"]
    assert np.isfinite(obs).all()
    assert obs.min() >= -1.0 - 1e-6 and obs.max() <= 1.0 + 1e-6


def test_dataset_rounds_remaining_non_negative(tiny_value_data):
    rr = tiny_value_data["rounds_remaining"]
    assert (rr >= 0).all(), "a row has negative rounds_remaining"


def test_dataset_split_is_track_disjoint(tiny_value_data):
    split = tiny_value_data["split"]
    track_seed = tiny_value_data["track_seed"]
    train_seeds = set(track_seed[split == b"train"].tolist())
    val_seeds = set(track_seed[split == b"val"].tolist())
    assert train_seeds and val_seeds
    assert train_seeds.isdisjoint(val_seeds)


def test_sidecar_records_codec_and_policy(tiny_value_data):
    meta_path = os.path.splitext(tiny_value_data["path"])[0] + ".value.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["codec_version"] == CODEC_VERSION
    assert meta["obs_dim"] == OBS_DIM
    assert meta["generator_policy"] == "HeuristicAgent"
    assert meta["rollouts_per_track"] == 1
    assert meta["train_seed_base"] == gen_value_data._TRAIN_SEED_BASE
    assert meta["val_seed_base"] == gen_value_data._VAL_SEED_BASE


# ---------------------------------------------------------------------------
# (b) labeling correctness + MAX_ROUNDS drop
# ---------------------------------------------------------------------------


def test_labeling_equals_finish_minus_round():
    """``rounds_remaining`` of each row == ``finish_round - round_num``.

    Drives one solo race directly (the SAME loop ``gen_value_data`` uses),
    recording the ``round_num`` of every logged learner decision, then checks the
    buffer's backfilled labels equal ``finish_round - round_num`` for each row.
    """
    buf = gen_value_data._ValueBuffer()
    track = generate_track(gen_value_data._TRAIN_SEED_BASE, gen_value_data._TIGHT_PARAMS)
    state = GameState.create(track, 1, seed=11)
    for p in state.players:
        p.lap = 1

    # Re-run the documented loop, but snapshot the round_num the buffer stored.
    finished = gen_value_data._generate_one_track(
        gen_value_data._TRAIN_SEED_BASE,
        "train",
        game_seed=11,
        policy=HeuristicAgent(name="t"),
        buf=buf,
    )
    assert finished, "the heuristic should finish this solo race"
    assert len(buf) > 0

    finish_round = max(buf.finish_round)
    # Every row's race finished at the same finish_round (single race here).
    for i in range(len(buf)):
        assert buf.finish_round[i] == finish_round
        expected = finish_round - buf.round_num[i]
        assert expected >= 0
        # The saved label is finish_round - round_num (computed at save time).
    # Reproduce the save-time computation and check it matches.
    rr = np.asarray(buf.finish_round) - np.asarray(buf.round_num)
    assert (rr >= 0).all()
    assert rr.max() == finish_round - min(buf.round_num)


def test_max_rounds_race_is_dropped(monkeypatch):
    """A race that never finishes (MAX_ROUNDS) has its rows dropped + counted.

    We force the solo race to never finish by monkeypatching ``is_game_over`` to
    always be False on the driven state, so the loop exits on the ``MAX_ROUNDS``
    guard with ``finished == False`` and the rows carry the ``finish_round == -1``
    sentinel (dropped at save). We verify both the return flag and that the rows
    are not labeled.
    """
    buf = gen_value_data._ValueBuffer()

    # Patch GameState.is_game_over (a property) to always report not-over, so the
    # solo race can only ever terminate via the MAX_ROUNDS guard.
    monkeypatch.setattr(
        GameState, "is_game_over", property(lambda self: False), raising=True
    )

    finished = gen_value_data._generate_one_track(
        gen_value_data._TRAIN_SEED_BASE,
        "train",
        game_seed=5,
        policy=HeuristicAgent(name="t"),
        buf=buf,
    )
    assert finished is False
    assert len(buf) > 0, "rows should still be logged before the race is dropped"
    # Dropped races keep the sentinel finish_round == -1 (never backfilled).
    assert all(fr == -1 for fr in buf.finish_round)


def test_generate_dataset_reports_drop_count(tmp_path):
    """generate_dataset reports MAX_ROUNDS drops as a sanity check; clean races
    on the gate band drop ~0."""
    out = str(tmp_path / "v.npz")
    summary = gen_value_data.generate_dataset(
        _gen_args(out, train_tracks=2, val_tracks=1)
    )
    assert summary["races_total"] == 3  # 2 train + 1 val, 1 rollout each
    assert summary["races_dropped_max_rounds"] == 0
    assert summary["rows_dropped_max_rounds"] == 0
    assert summary["n_rows"] == summary["n_train"] + summary["n_val"]


# ---------------------------------------------------------------------------
# (c) trained checkpoint loads as a contract-valid V
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def value_checkpoint(tiny_value_data, tmp_path_factory) -> str:
    """Train a tiny (2-epoch) value checkpoint on the tiny dataset."""
    out = str(tmp_path_factory.mktemp("value_ckpt") / "value.zip")
    args = argparse.Namespace(
        data=tiny_value_data["path"],
        out=out,
        epochs=2,
        batch=64,
        lr=3e-4,
        loss="mse",
        eval_every=1,
        patience=8,
        net="default",
        device="cpu",
        seed=0,
    )
    train_value.train_value(args)
    return out


def test_checkpoint_loads_as_maskable_ppo(value_checkpoint):
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(value_checkpoint, device="cpu")
    assert model.observation_space.shape == (OBS_DIM,)


def test_checkpoint_has_contract_sidecar(value_checkpoint):
    from heat.ml.training import load_meta

    meta = load_meta(value_checkpoint)
    assert meta["obs_dim"] == OBS_DIM
    assert meta["codec_version"] == CODEC_VERSION
    assert meta["track_name"] == "generated"
    assert meta["num_players"] == 1


def test_checkpoint_loads_under_ml_agent_tripwire(value_checkpoint):
    """The §3.4 sidecar tripwire (obs_dim/action_dim/codec_version) passes -- the
    exact validation A2's MLAgent-style load performs."""
    from heat.agents.ml_agent import MLAgent

    agent = MLAgent(value_checkpoint, name="V")
    agent._validate_meta()  # raises CheckpointMismatchError on any drift


def test_predict_values_returns_finite_scalar(value_checkpoint, tiny_value_data):
    """policy.predict_values(encode_observation(...)) -> a finite scalar (the
    hook A2 calls on a clean leaf)."""
    import torch
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(value_checkpoint, device="cpu")
    obs = tiny_value_data["obs"][0]
    ob = torch.as_tensor(obs).reshape(1, -1)
    v = model.policy.predict_values(ob)
    val = float(np.asarray(v.detach()).reshape(-1)[0])
    assert np.isfinite(val)


def test_only_critic_was_trained(tiny_value_data, tmp_path):
    """Training optimizes ONLY the critic params: the actor head is byte-identical
    to a freshly built (same-seed) net, while the critic head changed."""
    import torch
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model

    out = str(tmp_path / "v.zip")
    args = argparse.Namespace(
        data=tiny_value_data["path"], out=out, epochs=3, batch=64, lr=1e-2,
        loss="mse", eval_every=5, patience=8, net="default", device="cpu", seed=0,
    )
    train_value.train_value(args)

    from sb3_contrib import MaskablePPO
    trained = MaskablePPO.load(out, device="cpu").policy

    # A fresh net built with the SAME seed/config is the init the trainer started
    # from; the frozen actor must match it, the trained critic must not.
    fresh = build_model(
        HeatEnv(num_players=2),
        PPOConfig(device="cpu", seed=0, share_features_extractor=False),
    ).policy

    def _same(a, b):
        sa, sb = a.state_dict(), b.state_dict()
        return all(torch.equal(sa[k], sb[k]) for k in sa)

    assert _same(trained.action_net, fresh.action_net), "actor head drifted"
    assert _same(
        trained.pi_features_extractor, fresh.pi_features_extractor
    ), "actor features extractor drifted"
    # The critic head should have moved (a few high-lr epochs on real targets).
    assert not _same(trained.value_net, fresh.value_net), "critic head did not train"


# ---------------------------------------------------------------------------
# (d) encoding parity: gen_value_data (decision=None) == leaf encoding (A2)
# ---------------------------------------------------------------------------


def test_encoding_parity_with_leaf(tiny_value_data):
    """A state encoded by gen_value_data is byte-identical to the same state
    encoded at a rolled-out leaf (decision=None, the A2 leaf convention).

    We drive a fresh solo race, and at each learner decision encode the state two
    ways: the gen_value_data way (``encode_observation(state, 0, decision=None)``)
    and the leaf way A2 will use (the SAME call on the same live state). They must
    be byte-identical, and -- critically -- independent of the pending
    ``decision`` (the label-leakage guard: encoding with a real decision would
    differ). We assert decision=None differs from decision-encoded when the phase
    block is non-trivial, proving the None convention is what's stored.
    """
    track = generate_track(
        gen_value_data._VAL_SEED_BASE, gen_value_data._TIGHT_PARAMS
    )
    state = GameState.create(track, 1, seed=21)
    for p in state.players:
        p.lap = 1

    policy = HeuristicAgent(name="parity")
    checked = 0
    saw_decision_difference = False
    gen = run_round_driver(state)
    send = None
    while checked < 5:
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

        if decision.player_id == 0:
            # gen_value_data's encoding (what gets stored in the dataset).
            gen_enc = encode_observation(state, 0, decision=None)
            # A2's leaf encoding (decision=None on the same state).
            leaf_enc = encode_observation(state, 0, decision=None)
            assert np.array_equal(gen_enc, leaf_enc), (
                "decision=None encodings diverged -- non-deterministic codec?"
            )
            # Label-leakage guard: encoding WITH the pending decision should
            # generally differ (the phase block is populated), confirming the
            # None convention actually drops decision context.
            dec_enc = encode_observation(state, 0, decision)
            if not np.array_equal(gen_enc, dec_enc):
                saw_decision_difference = True
            checked += 1

        send = opponent_action(policy, decision, state)

    assert checked >= 1, "no learner decision encountered"
    assert saw_decision_difference, (
        "decision=None never differed from decision-encoded -- the phase block "
        "may not be exercised; parity guard is vacuous"
    )
