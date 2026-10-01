"""Tests for Sprint A2 (Option A): ``leaf_value="learned"`` in the LookaheadAgent.

The learned leaf plugs A1's trained value net ``V`` into ``_leaf_score`` on
**clean** leaves only (``own_spins == 0``); spinning leaves keep the unchanged
``_pre_spin_progress`` floor + the dominant own-spin penalty. These tests pin the
learned-path guarantees from the A2 design:

  (a) **spin dominance:** with a *stub* V returning an arbitrarily huge value, a
      candidate whose forced move spins still scores strictly below a clean one --
      V can never out-rank the own-spin penalty (the §2.4 scale guard).
  (b) **clean-leaf-only:** V is invoked only when ``own_spins == 0`` (counting
      stub) -- spinning leaves go through ``_pre_spin_progress``.
  (c) **determinism:** fixed state + seed + V -> byte-stable plan selection.
  (d) **legality:** the learned-leaf agent never returns an illegal move.
  (f) **contract tripwire:** a V whose sidecar ``codec_version`` mismatches raises
      the ``MLAgent``-style ``CheckpointMismatchError``.

Tests (a)-(d) inject a stub value function. Metadata tests write only a sidecar;
one integration test saves and loads a random-weight checkpoint. Non-learned
horizon-zero behavior is covered in test_search_agent.py.

Mirrors the structure/style of ``tests/test_search_agent.py``.
"""

from __future__ import annotations

import json

import pytest

from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.engine import rules
from heat.engine.game import Game
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
# (f) contract tripwire: mismatched sidecar codec_version raises
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_value_checkpoint(tmp_path_factory) -> str:
    """Save a random-weight model for the real loading/inference integration test."""
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model
    from heat.ml.training import save_checkpoint

    path = str(tmp_path_factory.mktemp("a2_value_ckpt") / "value.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, path, track_name="generated", num_players=1, seed=0)
    return path


@pytest.fixture
def value_sidecar(tmp_path) -> str:
    """Write only the metadata consumed by the compatibility check."""
    from heat.ml import spaces
    from heat.ml.training import meta_path_for

    path = str(tmp_path / "value.zip")
    with open(meta_path_for(path), "w", encoding="utf-8") as output:
        json.dump({"obs_dim": spaces.OBS_DIM, "action_dim": spaces.ACTION_DIM,
                   "codec_version": spaces.CODEC_VERSION}, output)
    return path


class TestContractTripwire:
    def test_valid_checkpoint_loads_under_tripwire(self, value_sidecar) -> None:
        """A contract-matching V passes the A2 sidecar validation (sanity)."""
        agent = LookaheadAgent(
            leaf_value="learned", value_model_path=value_sidecar
        )
        agent._validate_value_meta()  # must not raise

    def test_mismatched_codec_version_raises(self, value_sidecar) -> None:
        """Reject an incompatible sidecar without constructing a model."""
        from heat.agents.ml_agent import CheckpointMismatchError
        from heat.ml.training import meta_path_for

        meta_path = meta_path_for(value_sidecar)
        with open(meta_path, encoding="utf-8") as source:
            meta = json.load(source)
        meta["codec_version"] += 999
        with open(meta_path, "w", encoding="utf-8") as output:
            json.dump(meta, output)
        agent = LookaheadAgent(leaf_value="learned", value_model_path=value_sidecar)
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
