"""Tests for Sprint A2 (Option A): ``leaf_value="learned"`` in the LookaheadAgent.

The learned leaf plugs A1's trained value net ``V`` into ``_leaf_score`` on
**clean** leaves only (``own_spins == 0``); spinning leaves keep the unchanged
``_pre_spin_progress`` floor + the dominant own-spin penalty. These tests pin the
six guarantees the A2 doc lists:

  (a) **spin dominance:** with a *stub* V returning an arbitrarily huge value, a
      candidate whose forced move spins still scores strictly below a clean one --
      V can never out-rank the own-spin penalty (the §2.4 scale guard).
  (b) **clean-leaf-only:** V is invoked only when ``own_spins == 0`` (counting
      stub) -- spinning leaves go through ``_pre_spin_progress``.
  (c) **determinism:** fixed state + seed + V -> byte-stable plan selection.
  (d) **legality:** the learned-leaf agent never returns an illegal move.
  (e) **horizon-0 equivalence preserved** for the non-learned paths (the S1
      ``progress``/``move_eval`` greedy equivalence still holds -- learned is a
      purely additive branch).
  (f) **contract tripwire:** a V whose sidecar ``codec_version`` mismatches raises
      the ``MLAgent``-style ``CheckpointMismatchError``.

Tests (a)-(e) inject a stub V by subclassing/overriding ``_value_leaf`` (or
``_get_value_model``), so no real checkpoint is needed; (f) writes a real (tiny)
checkpoint and corrupts its sidecar.

Mirrors the structure/style of ``tests/test_search_agent.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.engine import rules
from heat.engine.game import Game
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.search_agent import LookaheadAgent
from heat.tracks.generator import generate_track


# ---------------------------------------------------------------------------
# Helpers + stub-V agents
# ---------------------------------------------------------------------------

#: A throwaway path string for ``leaf_value="learned"`` construction. The stub
#: agents below override the leaf so the model is never actually loaded from it.
_DUMMY_V = "checkpoints/_nonexistent_stub_v.zip"


def _gen_track(seed: int = 7) -> Track:
    return generate_track(seed=seed)


def _limit1_loop() -> Track:
    """Short synthetic loop with a single tight (limit-1) corner (S1 stress track)."""
    spaces = [Space(i, lanes=2) for i in range(24)]
    corners = [Corner(start=10, end=11, speed_limit=1)]
    return Track("Limit1Loop", spaces, corners, [0, 1, 2, 3, 4, 5], laps=2)


class _ConstVAgent(LookaheadAgent):
    """Learned-leaf agent whose V returns a fixed (configurable) value.

    Overrides only ``_value_leaf`` so no checkpoint is loaded; everything else --
    the spin floor, the penalties, the search machinery -- is the real code path.
    """

    def __init__(self, *args, v_value: float = 0.0, **kwargs) -> None:
        super().__init__(*args, leaf_value="learned",
                         value_model_path=_DUMMY_V, **kwargs)
        self._v_value = v_value
        self.v_calls = 0

    def _value_leaf(self, clone: GameState, player_id: int) -> float:
        self.v_calls += 1
        return self._v_value


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_learned_requires_value_model_path(self) -> None:
        with pytest.raises(ValueError):
            LookaheadAgent(leaf_value="learned")  # missing value_model_path

    def test_learned_accepts_path(self) -> None:
        agent = LookaheadAgent(leaf_value="learned", value_model_path=_DUMMY_V)
        assert agent.leaf_value == "learned"
        assert agent.value_model_path == _DUMMY_V
        assert agent._value_model is None  # lazy: not loaded at construction

    def test_unknown_leaf_value_still_rejected(self) -> None:
        with pytest.raises(ValueError):
            LookaheadAgent(leaf_value="bogus")


# ---------------------------------------------------------------------------
# (a) spin dominance: V can never out-rank the own-spin penalty
# ---------------------------------------------------------------------------


class TestSpinDominance:
    def test_spinning_leaf_scores_below_clean_leaf_with_huge_v(self) -> None:
        """A leaf whose forced move spun (own_spins=1) scores strictly below a
        clean leaf (own_spins=0) even when the stub V returns a gigantic value.

        ``_leaf_score`` is the unit under test: on the spinning branch V is never
        called (the ``_pre_spin_progress`` floor + 1000-space own-spin penalty
        apply), so no V output -- however large -- can lift a spinning candidate
        above a clean one.
        """
        track = _limit1_loop()
        agent = _ConstVAgent(horizon=1, n_determinizations=1, v_value=1e9)

        clone = GameState.create(track, 1, logging_enabled=True, seed=5)
        clone.players[0].lap = 1
        clone.players[0].position = 8
        player = clone.players[0]
        gear = (player.gear, 0)
        cards = tuple(player.hand[:1])

        clean = agent._leaf_score(clone, 0, gear, cards, own_spins=0, later_spins=0)
        spun = agent._leaf_score(clone, 0, gear, cards, own_spins=1, later_spins=0)

        assert clean > spun, (
            f"clean leaf ({clean}) must beat a spinning leaf ({spun}) even with "
            f"V=1e9 -- the own-spin penalty must dominate V by construction"
        )
        # The clean leaf used V; the spinning leaf must NOT have (floor only).
        assert agent.v_calls == 1, "V should be called exactly once (clean only)"


# ---------------------------------------------------------------------------
# (b) clean-leaf-only: V invoked iff own_spins == 0
# ---------------------------------------------------------------------------


class TestCleanLeafOnly:
    def test_v_called_only_on_clean_leaves(self) -> None:
        track = _limit1_loop()
        agent = _ConstVAgent(horizon=1, n_determinizations=1, v_value=0.0)
        clone = GameState.create(track, 1, logging_enabled=True, seed=5)
        clone.players[0].lap = 1
        player = clone.players[0]
        gear = (player.gear, 0)
        cards = tuple(player.hand[:1])

        # Spinning leaf: V must NOT be consulted.
        agent._leaf_score(clone, 0, gear, cards, own_spins=2, later_spins=1)
        assert agent.v_calls == 0, "V called on a spinning leaf (own_spins>0)"

        # Clean leaf: V IS consulted exactly once.
        agent._leaf_score(clone, 0, gear, cards, own_spins=0, later_spins=0)
        assert agent.v_calls == 1, "V not called on a clean leaf (own_spins==0)"

    def test_full_solo_game_invokes_v_at_least_once(self) -> None:
        """Driving a real solo game with the learned leaf calls V on clean leaves
        (the search reaches clean leaves on this track)."""
        track = _gen_track(7)
        agent = _ConstVAgent(horizon=1, n_determinizations=1, v_value=0.0, seed=0)
        game = Game(track, [agent], logging_enabled=False, seed=42)
        result = game.run()
        assert result.finish_order == [0]
        assert agent.v_calls > 0, "the learned leaf never scored a clean leaf"


# ---------------------------------------------------------------------------
# (c) determinism: same state + seed + V -> byte-stable plan
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_state_same_plan_learned(self) -> None:
        track = _gen_track(7)

        def plan():
            a = _ConstVAgent(horizon=2, n_determinizations=2, v_value=3.5, seed=0)
            state = GameState.create(track, 1, logging_enabled=False, seed=123)
            state.players[0].lap = 1
            state.players[0].position = 2
            lg = rules.legal_gear_shifts(
                state.players[0].gear, state.players[0].heat_available
            )
            gear = a.choose_gear(state, 0, lg)
            state.players[0].gear = gear[0]
            lp = rules.legal_card_plays(state.players[0].hand, gear[0])
            cards = a.choose_cards(state, 0, lp)
            return gear, tuple(c.display_name for c in cards)

        first = plan()
        for _ in range(5):
            assert plan() == first


# ---------------------------------------------------------------------------
# (d) legality: never returns an illegal move
# ---------------------------------------------------------------------------


class _LegalityWrapper:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name

    def choose_gear(self, state, player_id, legal_gears):
        choice = self.inner.choose_gear(state, player_id, legal_gears)
        assert choice in legal_gears, f"illegal gear {choice}"
        return choice

    def choose_cards(self, state, player_id, legal_plays):
        choice = self.inner.choose_cards(state, player_id, legal_plays)
        assert choice in legal_plays, "illegal card play"
        return choice

    def choose_react(self, state, player_id, max_cooldown, can_boost, has_adrenaline):
        return self.inner.choose_react(
            state, player_id, max_cooldown, can_boost, has_adrenaline
        )

    def choose_slipstream(self, state, player_id):
        return self.inner.choose_slipstream(state, player_id)

    def choose_discard(self, state, player_id, discardable):
        choice = self.inner.choose_discard(state, player_id, discardable)
        assert all(c in discardable for c in choice)
        return choice


class TestLegality:
    def test_all_decisions_legal_solo_learned(self) -> None:
        track = _gen_track(7)
        for game_idx in range(3):
            agent = _LegalityWrapper(
                _ConstVAgent(horizon=2, n_determinizations=1, v_value=0.0)
            )
            game = Game(track, [agent], logging_enabled=False, seed=1000 + game_idx)
            result = game.run()
            assert len(result.finish_order) == 1

    def test_never_illegal_on_limit1_loop_learned(self) -> None:
        track = _limit1_loop()
        agent = _LegalityWrapper(
            _ConstVAgent(horizon=2, n_determinizations=2, v_value=2.0)
        )
        game = Game(track, [agent], logging_enabled=False, seed=99)
        result = game.run()
        assert len(result.finish_order) == 1


# ---------------------------------------------------------------------------
# (e) horizon-0 equivalence preserved for the non-learned paths
# ---------------------------------------------------------------------------


class TestNonLearnedPathsUnchanged:
    """The learned branch is purely additive: the ``progress``/``move_eval`` leaf
    paths still reduce to the single-round argmax at horizon 0 (the S1 contract).
    """

    def _greedy_reference(self, agent, state, player_id):
        legal_gears = rules.legal_gear_shifts(
            state.get_player(player_id).gear,
            state.get_player(player_id).heat_available,
        )
        candidates = agent._candidate_plans(state, player_id, legal_gears)
        sig = agent._turn_signature(state, player_id)
        turn_seed = agent._turn_seed(sig)
        best = candidates[0]
        best_v = float("-inf")
        for idx, (gear, cards) in enumerate(candidates):
            v = agent._score_plan(state, player_id, gear, cards, turn_seed, idx)
            if v > best_v:
                best_v = v
                best = (gear, cards)
        return best

    @pytest.mark.parametrize("leaf", ["progress", "move_eval"])
    def test_horizon0_matches_single_round_argmax(self, leaf: str) -> None:
        track = _gen_track(7)
        agent = LookaheadAgent(horizon=0, n_determinizations=1, leaf_value=leaf)
        state = GameState.create(track, 1, logging_enabled=False, seed=321)
        state.players[0].lap = 1
        state.players[0].position = 2

        ref_gear, ref_cards = self._greedy_reference(agent, state, 0)

        legal_gears = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )
        chosen_gear = agent.choose_gear(state, 0, legal_gears)
        state.players[0].gear = chosen_gear[0]
        legal_plays = rules.legal_card_plays(state.players[0].hand, chosen_gear[0])
        chosen_cards = agent.choose_cards(state, 0, legal_plays)

        assert chosen_gear == ref_gear
        assert chosen_cards == ref_cards


# ---------------------------------------------------------------------------
# (f) contract tripwire: mismatched sidecar codec_version raises
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_value_checkpoint(tmp_path_factory) -> str:
    """Train a tiny real value checkpoint (reuses the A1 deliverables).

    Mirrors ``tests/test_value_net.py``'s tiny train: a 2-track dataset + 2-epoch
    critic-only fit, fast enough for a unit test.
    """
    exp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    import gen_value_data  # noqa: E402
    import train_value  # noqa: E402

    data_out = str(tmp_path_factory.mktemp("a2_value_data") / "value.npz")
    gen_value_data.generate_dataset(
        argparse.Namespace(
            out=data_out, train_tracks=2, val_tracks=1, rollouts=1, game_seed=7000
        )
    )
    ckpt_out = str(tmp_path_factory.mktemp("a2_value_ckpt") / "value.zip")
    train_value.train_value(
        argparse.Namespace(
            data=data_out, out=ckpt_out, epochs=2, batch=64, lr=3e-4, loss="mse",
            eval_every=1, patience=8, net="default", device="cpu", seed=0,
        )
    )
    return ckpt_out


class TestContractTripwire:
    def test_valid_checkpoint_loads_under_tripwire(self, real_value_checkpoint) -> None:
        """A contract-matching V passes the A2 sidecar validation (sanity)."""
        agent = LookaheadAgent(
            leaf_value="learned", value_model_path=real_value_checkpoint
        )
        agent._validate_value_meta()  # must not raise

    def test_mismatched_codec_version_raises(
        self, real_value_checkpoint, tmp_path
    ) -> None:
        """A V whose sidecar codec_version is corrupted fails fast with the
        MLAgent-style mismatch error (a stale model vs drifted codec).

        Copies the checkpoint + sidecar to a private temp path before corrupting
        so the shared module-scoped fixture is never mutated (test isolation).
        """
        import shutil

        from heat.agents.ml_agent import CheckpointMismatchError
        from heat.ml.training import meta_path_for

        # Copy the zip + sidecar to a fresh path we own, then corrupt the copy.
        copy_zip = str(tmp_path / "value.zip")
        shutil.copy(real_value_checkpoint, copy_zip)
        shutil.copy(meta_path_for(real_value_checkpoint), meta_path_for(copy_zip))

        meta_path = meta_path_for(copy_zip)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["codec_version"] = meta.get("codec_version", 0) + 999
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

        agent = LookaheadAgent(leaf_value="learned", value_model_path=copy_zip)
        with pytest.raises(CheckpointMismatchError):
            agent._validate_value_meta()

    def test_missing_sidecar_raises(self, tmp_path) -> None:
        """A path with no sidecar at all fails fast (mirrors MLAgent)."""
        from heat.agents.ml_agent import CheckpointMismatchError

        bogus = str(tmp_path / "no_such_model.zip")
        agent = LookaheadAgent(leaf_value="learned", value_model_path=bogus)
        with pytest.raises(CheckpointMismatchError):
            agent._validate_value_meta()

    def test_end_to_end_learned_leaf_with_real_v(self, real_value_checkpoint) -> None:
        """A real V drives a solo game to completion via the learned leaf (the
        load + predict_values + leaf path all wire together)."""
        track = _gen_track(7)
        agent = LookaheadAgent(
            horizon=1,
            n_determinizations=1,
            leaf_value="learned",
            value_model_path=real_value_checkpoint,
            seed=0,
        )
        game = Game(track, [agent], logging_enabled=False, seed=42)
        result = game.run()
        assert result.finish_order == [0]
        assert agent._value_model is not None  # lazy-loaded on first clean leaf


# ---------------------------------------------------------------------------
# Picklability (path-only __getstate__)
# ---------------------------------------------------------------------------


class TestPicklability:
    def test_learned_agent_pickles_by_path(self) -> None:
        import pickle

        from heat.simulation.runner import lookahead_agent_factory

        fac = lookahead_agent_factory(
            horizon=1, n_determinizations=1, leaf_value="learned",
            value_model_path=_DUMMY_V,
        )
        restored = pickle.loads(pickle.dumps(fac))
        agent = restored(0, 123)
        assert isinstance(agent, LookaheadAgent)
        assert agent.leaf_value == "learned"
        assert agent.value_model_path == _DUMMY_V
        assert agent._value_model is None

    def test_loaded_model_is_nulled_on_pickle(self, tmp_path) -> None:
        """Even after a model is loaded, __getstate__ nulls it so the agent
        pickles by path (the MLAgent contract)."""
        import pickle

        agent = LookaheadAgent(leaf_value="learned", value_model_path=_DUMMY_V)
        agent._value_model = object()  # pretend a heavy model is loaded
        restored = pickle.loads(pickle.dumps(agent))
        assert restored._value_model is None
        assert restored.value_model_path == _DUMMY_V
