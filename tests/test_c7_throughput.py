"""Tests for the Sprint C7 self-play throughput sprint -- the DETERMINISM contract.

C7 is a PURE PERFORMANCE sprint: model reuse + parallel workers + an encoder
cache, none of which may change a single logged value. The whole sprint's risk is
a determinism regression, so these tests pin the three hard contracts the design
("Determinism -- the hard contract") makes load-bearing:

  1. **byte-identical dataset INDEPENDENT of --workers** -- ``generate_dataset``
     with ``workers=4`` produces arrays EQUAL to ``workers=1`` after the
     deterministic seed-sorted merge (the parallel-aggregation contract);
  2. **the cached encoder is bit-identical** -- the per-track precompute in
     ``features._track_block`` produces the exact same observation vector as a
     freshly-computed reference, over a battery of decision states (a wrong obs
     silently poisons training);
  3. **model caching is read-only** -- the process-level model cache returns the
     SAME module object for a repeated path (loaded once) and does not change the
     search output: a fixed ``(state, seed)`` plan is byte-stable whether or not a
     cached model is reused.

All runnable WITHOUT a trained checkpoint (a fresh random-weight codec-v3 prior is
enough -- C7 is net-agnostic). Mirrors the experiments-dir-on-sys.path + cold-prior
fixture style of ``tests/test_az_targets.py`` and ``tests/test_c6_winloss_1v1.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# experiments/ is not a package; add it so the C7 deliverables import by name.
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gen_selfplay as G  # noqa: E402

from heat.agents import mcts_agent  # noqa: E402
from heat.agents.mcts_agent import MCTSAgent, MCTSConfig, NetAdapter  # noqa: E402
from heat.engine import rules  # noqa: E402
from heat.models.game_state import GameState  # noqa: E402
from heat.ml import features as F  # noqa: E402
from heat.ml.action_codec import legal_action_mask  # noqa: E402
from heat.ml.features import encode_observation  # noqa: E402
from heat.tracks.generator import generate_track  # noqa: E402


# ---------------------------------------------------------------------------
# Shared cold prior (a random-weight live-codec checkpoint; C7 is net-agnostic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cold_prior(tmp_path_factory) -> str:
    """Mint a fresh random-weight ``MaskablePPO`` checkpoint once.

    C7 verifies determinism (model reuse / workers / encoder cache), which is
    net-agnostic, so random weights are enough; CPU matches the NetAdapter path.
    """
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model
    from heat.ml.training import save_checkpoint

    out = str(tmp_path_factory.mktemp("c7prior") / "cold.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=0)
    return out


def _gen_args(out, *, model, workers, tracks=4, val_tracks=2, sims=8, two_player=False):
    """A tiny generation Namespace at a fixed config + seed (the only knob varied
    across the determinism test is ``workers``)."""
    return argparse.Namespace(
        out=out,
        model=model,
        tracks=tracks,
        val_tracks=val_tracks,
        sims=sims,
        dirichlet_eps=0.25,
        dirichlet_alpha=0.5,
        temperature_moves=10,
        root_selector="puct",
        traj_greedy=False,
        two_player=two_player,
        opponent_snapshot=None,
        workers=workers,
        seed=0,
        game_seed=8000,
    )


# ---------------------------------------------------------------------------
# 1. The dataset is byte-identical INDEPENDENT of --workers (the merge contract)
# ---------------------------------------------------------------------------


_DATASET_KEYS = ("obs", "pi", "mask", "z", "kind", "track_seed", "split")


class TestWorkersByteIdentical:
    def _generate(self, tmp_path, cold_prior, workers, **kw):
        out = str(tmp_path / f"sp_w{workers}.npz")
        G.generate_dataset(_gen_args(out, model=cold_prior, workers=workers, **kw))
        data = np.load(out)
        return {k: data[k] for k in data.files}

    @pytest.mark.slow
    def test_workers_4_equals_workers_1_solo(self, cold_prior, tmp_path) -> None:
        """The C2-C5 solo generator: ``workers=4`` == ``workers=1`` byte-for-byte
        on every array, after the deterministic seed-sorted merge."""
        serial = self._generate(tmp_path, cold_prior, workers=1)
        parallel = self._generate(tmp_path, cold_prior, workers=4)
        for k in _DATASET_KEYS:
            assert np.array_equal(serial[k], parallel[k]), (
                f"array {k!r} differs between workers=1 and workers=4 "
                "(the deterministic merge is broken)"
            )

    @pytest.mark.slow
    def test_workers_4_equals_workers_1_two_player(self, cold_prior, tmp_path) -> None:
        """The C6 1v1 win/loss generator: also byte-identical across --workers."""
        serial = self._generate(tmp_path, cold_prior, workers=1, two_player=True)
        parallel = self._generate(tmp_path, cold_prior, workers=4, two_player=True)
        for k in _DATASET_KEYS:
            assert np.array_equal(serial[k], parallel[k]), (
                f"array {k!r} differs between workers=1 and workers=4 (1v1)"
            )

    def test_workers_default_is_serial(self, tmp_path, monkeypatch) -> None:
        """Missing workers uses the serial dispatcher, without loading a model."""
        import multiprocessing

        args = _gen_args(str(tmp_path / "unused.npz"), model="unused.zip", workers=1)
        delattr(args, "workers")
        calls = []

        def run_one(spec):
            calls.append(spec)
            return object()

        class DispatchComplete(Exception):
            """Stop before dataset serialization after checking dispatch results."""

        def merge(buf, results):
            assert len(results) == args.tracks + args.val_tracks
            assert len(calls) == len(results)
            raise DispatchComplete

        def pool_forbidden(*args, **kwargs):
            pytest.fail("missing workers must not create a multiprocessing pool")

        monkeypatch.setattr(G, "_run_one_game", run_one)
        monkeypatch.setattr(G, "_merge_into", merge)
        monkeypatch.setattr(multiprocessing, "get_context", pool_forbidden)
        with pytest.raises(DispatchComplete):
            G.generate_dataset(args)


# ---------------------------------------------------------------------------
# 2. The cached encoder is bit-identical to a fresh reference (encoder cache)
# ---------------------------------------------------------------------------


def _reference_track_block(player, track) -> list[float]:
    """Uncached v4 ``_track_block``. The cached encoder must match it bit for bit."""
    from heat.ml import spaces

    length = track.length or 1
    pos = player.position
    corners = list(track.corners)

    def fwd_dist(c) -> int:
        d = (c.start - pos) % length
        return length if d == 0 else d

    ordered = sorted(corners, key=fwd_dist)

    def clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    slots: list[float] = []
    for i in range(spaces.MAX_CORNERS):
        if i < len(ordered):
            c = ordered[i]
            d = fwd_dist(c)
            if 0 <= c.start < len(track.spaces):
                entry_lanes = track.spaces[c.start].lanes
            else:
                entry_lanes = 1
            slots += [
                clip01(d / length),
                clip01(c.speed_limit / spaces.SPEED_LIMIT_SCALE),
                clip01((c.end - c.start + 1) / spaces.CORNER_LENGTH_SCALE),
                clip01(entry_lanes / spaces.LANE_SCALE),
            ]
        else:
            slots += [0.0, 0.0, 0.0, 0.0]

    laps = track.laps or 1
    total_len = length * laps
    if player.finished or player.lap > laps:
        laps_remaining = 0.0
        dist_to_finish = 0.0
    else:
        current_lap = max(player.lap, 1)
        laps_remaining = clip01((laps - current_lap + 1) / laps)
        completed = (current_lap - 1) * length + pos
        dist_to_finish = clip01((total_len - completed) / total_len)
    heat = clip01(player.heat_available / rules.HEAT_POOL_SIZE)
    pos_in_lap = clip01(pos / length)
    return slots + [laps_remaining, dist_to_finish, heat, pos_in_lap]


class TestEncoderCacheBitIdentical:
    def test_track_block_matches_reference_over_battery(self) -> None:
        """The cached ``_track_block`` is bit-identical to the un-cached reference
        across many tracks x every position x both laps."""
        for tseed in range(25):
            track = generate_track(tseed)
            state = GameState.create(track, 2, logging_enabled=False, seed=0)
            player = state.players[0]
            for lap in (1, 2):
                for pos in range(track.length):
                    player.position = pos
                    player.lap = lap
                    got = np.asarray(F._track_block(player, track), dtype=np.float32)
                    exp = np.asarray(
                        _reference_track_block(player, track), dtype=np.float32
                    )
                    assert np.array_equal(got, exp), (tseed, lap, pos)

    def test_full_observation_bit_identical_over_decision_battery(self, cold_prior) -> None:
        """``encode_observation`` (which calls the cached ``_track_block``) is
        bit-identical to the SAME state encoded after clearing the precompute cache,
        over a battery of real decision states and decision kinds.

        Encodes once (warming the cache), clears the per-track precompute, encodes
        again (cold), and asserts byte-equality -- so the cache can never drift the
        observation a leaf eval consumes.
        """
        from heat.engine.driver import run_round_driver
        from heat.engine.game import MAX_ROUNDS

        seen_kinds: set = set()
        n_checked = 0
        for tseed in range(6):
            track = generate_track(tseed)
            state = GameState.create(track, 2, logging_enabled=False, seed=tseed)
            for p in state.players:
                p.lap = 1
            gen = run_round_driver(state)
            send = None
            guard = 0
            while guard < 400:
                guard += 1
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
                pid = decision.player_id
                # Warm encode, then cold encode (cache cleared) -- must match
                # for both the historical codec and the live one.
                for codec in (3, 4):
                    warm = encode_observation(
                        state, pid, decision, codec_version=codec
                    )
                    F._track_precompute.clear()
                    cold = encode_observation(
                        state, pid, decision, codec_version=codec
                    )
                    assert np.array_equal(warm, cold), (tseed, decision.kind, codec)
                    warm_v = encode_observation(state, pid, None, codec_version=codec)
                    F._track_precompute.clear()
                    cold_v = encode_observation(state, pid, None, codec_version=codec)
                    assert np.array_equal(warm_v, cold_v)
                # A v4 cache entry stores raw geometry, so a later v3 read matches
                # a cold v3 encode.
                encode_observation(state, pid, decision, codec_version=4)
                cross = encode_observation(state, pid, decision, codec_version=3)
                F._track_precompute.clear()
                cold_v3 = encode_observation(state, pid, decision, codec_version=3)
                assert np.array_equal(cross, cold_v3)
                seen_kinds.add(decision.kind)
                n_checked += 1
                mask = legal_action_mask(decision, state)
                flat = int(np.argmax(mask)) if mask.any() else -1
                from heat.ml.action_codec import decode_action

                send = decode_action(decision, flat, state) if flat >= 0 else None
        assert n_checked > 0
        # The battery exercised more than one decision kind (real coverage).
        assert len(seen_kinds) >= 2


# ---------------------------------------------------------------------------
# 3. Model caching is read-only (load once, search output unchanged)
# ---------------------------------------------------------------------------


class TestModelCacheReadOnly:
    def test_cache_returns_same_module_object(self, cold_prior) -> None:
        """``_load_cached_model`` loads ONCE per (abspath, mtime): a repeated call
        returns the SAME object (the per-game reload tax is gone)."""
        mcts_agent._MODEL_CACHE.clear()
        m1 = mcts_agent._load_cached_model(cold_prior)
        m2 = mcts_agent._load_cached_model(cold_prior)
        assert m1 is m2, "the cache must return the identical loaded module"
        # Two adapters at the same path share the one cached module.
        a1, a2 = NetAdapter(cold_prior), NetAdapter(cold_prior)
        assert a1._get_model() is a2._get_model()

    def test_cache_key_includes_mtime(self, cold_prior, tmp_path) -> None:
        """The cache key is ``(abspath, mtime)`` so a re-written checkpoint at the
        same path is reloaded (a different module object)."""
        import shutil
        import time as _time

        p = str(tmp_path / "ckpt.zip")
        shutil.copyfile(cold_prior, p)
        meta_src = os.path.splitext(cold_prior)[0] + ".meta.json"
        if os.path.exists(meta_src):
            shutil.copyfile(meta_src, os.path.splitext(p)[0] + ".meta.json")
        mcts_agent._MODEL_CACHE.clear()
        m1 = mcts_agent._load_cached_model(p)
        # Re-write with a newer mtime; the key changes => a fresh load.
        _time.sleep(0.01)
        shutil.copyfile(cold_prior, p)
        os.utime(p, None)
        m2 = mcts_agent._load_cached_model(p)
        assert m1 is not m2

    def test_search_byte_stable_with_cached_model(self, cold_prior) -> None:
        """A fixed ``(state, seed)`` plan is byte-identical whether the model is
        freshly loaded or served from the cache -- caching does not change targets.
        """
        track = generate_track(7)

        def plan(use_cache: bool):
            if not use_cache:
                mcts_agent._MODEL_CACHE.clear()
            agent = MCTSAgent(
                model_path=cold_prior,
                config=MCTSConfig(n_simulations=16),
                seed=0,
            )
            state = GameState.create(track, 1, logging_enabled=False, seed=123)
            state.players[0].lap = 1
            state.players[0].position = 2
            lg = rules.legal_gear_shifts(
                state.players[0].gear, state.players[0].heat_available
            )
            gear = agent.choose_gear(state, 0, lg)
            state.players[0].gear = gear[0]
            lp = rules.legal_card_plays(state.players[0].hand, gear[0])
            cards = agent.choose_cards(state, 0, lp)
            return gear, tuple(c.display_name for c in cards)

        mcts_agent._MODEL_CACHE.clear()
        first = plan(use_cache=False)   # cold load
        cached = plan(use_cache=True)   # served from cache
        assert first == cached
