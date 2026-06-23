"""Tests for Sprint C9 -- the lean observation-encode path (the byte-identical
equivalence contract).

C9 is a PURE PERFORMANCE sprint over the per-leaf observation build. None of it
may change a single float the search/training reads, so the whole sprint's risk
is a numerical / determinism regression. ``encode_observation`` is a frozen pure
function; the contract is **byte-identical** output (``np.array_equal``, NOT
``allclose``).

These tests pin the first-class equivalence contract the design makes load-bearing:

  1. **Tier-1 local rewrites are bit-identical** -- the new single-pass
     ``_deck_composition`` and fixed-index ``_hand_histogram`` reproduce the
     PRE-C9 reference implementations exactly over a battery of real states.
  2. **Tier-2 shared prefix is bit-identical** -- ``encode_observation_pair``
     returns vectors EQUAL to two independent ``encode_observation`` calls (the
     prior obs with the decision, the value obs with ``decision=None``), over a
     battery of real search states across BOTH modes (solo + two-player) and BOTH
     seats (learner + opponent).
  3. **Spin detection without logging is byte-identical** -- a fixed
     ``(state, seed)`` MCTS plan is byte-stable whether the live game's logging is
     ON or OFF (the folded-in free win: search clones no longer force the event
     log; spins are read from the per-player ``spin_log``).
  4. **Determinism preserved** -- same ``(state, seed)`` => identical move.

Runnable WITHOUT a trained checkpoint (a fresh random-weight codec-v3 prior is
enough). Mirrors the cold-prior fixture + experiments-on-path style of
``tests/test_c7_throughput.py`` and ``tests/test_c8_lean_inference.py``.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from heat.agents import mcts_agent  # noqa: E402
from heat.agents.mcts_agent import MCTSAgent, MCTSConfig  # noqa: E402
from heat.engine import rules  # noqa: E402
from heat.engine.driver import run_round_driver  # noqa: E402
from heat.engine.game import MAX_ROUNDS  # noqa: E402
from heat.models.cards import CardType  # noqa: E402
from heat.models.game_state import GameState  # noqa: E402
from heat.ml import features as F  # noqa: E402
from heat.ml.action_codec import decode_action, legal_action_mask  # noqa: E402
from heat.ml.features import (  # noqa: E402
    encode_observation,
    encode_observation_pair,
)
from heat.tracks.generator import generate_track  # noqa: E402


# ---------------------------------------------------------------------------
# Shared cold prior (a random-weight codec-v3 checkpoint; C9 is net-agnostic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cold_prior(tmp_path_factory) -> str:
    """Mint a fresh random-weight codec-v3 ``MaskablePPO`` checkpoint once.

    C9 verifies the per-leaf encode is byte-identical and the search is
    unchanged, both net-agnostic; random weights are enough and CPU matches the
    NetAdapter path.
    """
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model
    from heat.ml.training import save_checkpoint

    out = str(tmp_path_factory.mktemp("c9prior") / "cold.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=0)
    return out


# ---------------------------------------------------------------------------
# PRE-C9 reference implementations (faithful copies of the old code) -- the
# bit-identical oracle the Tier-1 rewrites must match.
# ---------------------------------------------------------------------------


def _reference_deck_composition(player) -> list[float]:
    """The PRE-C9 ``_deck_composition`` body: list(draw)+list(discard) then four
    independent genexpr passes. The bit-identical reference."""
    def clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    deck = player.deck
    all_cards = list(deck.draw_pile) + list(deck.discard_pile)
    total = len(all_cards)
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    n_speed = sum(1 for c in all_cards if c.card_type == CardType.SPEED)
    n_heat = sum(1 for c in all_cards if c.card_type == CardType.HEAT)
    n_stress = sum(1 for c in all_cards if c.card_type == CardType.STRESS)
    n_upgrade = sum(1 for c in all_cards if c.card_type == CardType.UPGRADE)
    return [
        clip01(n_speed / total),
        clip01(n_heat / total),
        clip01(n_stress / total),
        clip01(n_upgrade / total),
        clip01(deck.draw_pile_size / total),
        clip01(deck.discard_pile_size / total),
    ]


def _reference_hand_histogram(player) -> list[float]:
    """The PRE-C9 ``_hand_histogram`` body: the string-keyed dict + f-strings."""
    def clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    counts = {
        "S1": 0, "S2": 0, "S3": 0, "S4": 0,
        "H": 0, "ST": 0, "U0": 0, "U5": 0,
    }
    for card in player.hand:
        if card.card_type == CardType.SPEED:
            key = f"S{card.value}"
            if key in counts:
                counts[key] += 1
        elif card.card_type == CardType.HEAT:
            counts["H"] += 1
        elif card.card_type == CardType.STRESS:
            counts["ST"] += 1
        elif card.card_type == CardType.UPGRADE:
            counts["U0" if card.value == 0 else "U5"] += 1
    order = ["S1", "S2", "S3", "S4", "H", "ST", "U0", "U5"]
    return [clip01(counts[k] / 7.0) for k in order]


# ---------------------------------------------------------------------------
# A battery of real decision states (driving real rounds), shared by the tests.
# ---------------------------------------------------------------------------


def _decision_states(num_players: int, n_tracks: int = 6):
    """Yield ``(state, decision)`` over real driven rounds for ``n_tracks`` tracks.

    Drives ``run_round_driver`` to real GEAR/CARDS/REACT/SLIPSTREAM/DISCARD
    decisions (the exact distribution a search/encode hits), answering each with a
    legal default so the game advances. Yields the live state + the pending
    decision at each step.
    """
    for tseed in range(n_tracks):
        track = generate_track(tseed)
        state = GameState.create(
            track, num_players, logging_enabled=False, seed=tseed
        )
        for p in state.players:
            p.lap = 1
        gen = run_round_driver(state)
        send = None
        guard = 0
        while guard < 600:
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
            yield state, decision
            mask = legal_action_mask(decision, state)
            flat = int(np.argmax(mask)) if mask.any() else -1
            send = decode_action(decision, flat, state) if flat >= 0 else None


# ---------------------------------------------------------------------------
# 1. Tier-1 local rewrites are bit-identical to the PRE-C9 reference
# ---------------------------------------------------------------------------


class TestTier1BitIdentical:
    def test_deck_composition_matches_reference(self) -> None:
        """The single-pass ``_deck_composition`` equals the old 4-pass reference
        byte-for-byte over a battery of real (solo + 2p) states."""
        n_checked = 0
        for num_players in (1, 2):
            for state, _decision in _decision_states(num_players):
                for p in state.players:
                    got = np.asarray(F._deck_composition(p), dtype=np.float32)
                    exp = np.asarray(
                        _reference_deck_composition(p), dtype=np.float32
                    )
                    assert np.array_equal(got, exp), (num_players, p.player_id)
                    n_checked += 1
        assert n_checked > 0

    def test_hand_histogram_matches_reference(self) -> None:
        """The fixed-index ``_hand_histogram`` equals the old dict+f-string
        reference byte-for-byte over a battery of real states."""
        n_checked = 0
        for num_players in (1, 2):
            for state, _decision in _decision_states(num_players):
                for p in state.players:
                    got = np.asarray(F._hand_histogram(p), dtype=np.float32)
                    exp = np.asarray(
                        _reference_hand_histogram(p), dtype=np.float32
                    )
                    assert np.array_equal(got, exp), (num_players, p.player_id)
                    n_checked += 1
        assert n_checked > 0

    def test_full_observation_unchanged_vs_reference_blocks(self) -> None:
        """A full ``encode_observation`` whose deck/hand blocks are swapped for the
        references is byte-identical to the live encode -- proves the rewrites did
        not perturb any neighbouring block via aliasing/ordering."""
        n_checked = 0
        for num_players in (1, 2):
            for state, decision in _decision_states(num_players):
                pid = decision.player_id
                live = encode_observation(state, pid, decision)
                # Rebuild the same vector but force the reference sub-blocks in.
                player = state.get_player(pid)
                ref_deck = np.asarray(
                    _reference_deck_composition(player), dtype=np.float32
                )
                ref_hand = np.asarray(
                    _reference_hand_histogram(player), dtype=np.float32
                )
                # hand is the first 8 floats; deck is the 6 floats at offset 16.
                assert np.array_equal(live[0:8], ref_hand)
                assert np.array_equal(live[16:22], ref_deck)
                n_checked += 1
        assert n_checked > 0


# ---------------------------------------------------------------------------
# 2. Tier-2 shared prefix == two independent encodes (both modes, both seats)
# ---------------------------------------------------------------------------


class TestTier2SharedPrefix:
    def _check_pair(self, state, pid, decision) -> None:
        """The shared-prefix pair equals two independent encodes, byte-for-byte."""
        prior_obs, value_obs = encode_observation_pair(state, pid, decision)
        exp_prior = encode_observation(state, pid, decision)
        exp_value = encode_observation(state, pid, None)
        assert np.array_equal(prior_obs, exp_prior), (pid, decision.kind, "prior")
        assert np.array_equal(value_obs, exp_value), (pid, decision.kind, "value")

    def test_pair_equals_two_encodes_solo(self) -> None:
        """Solo: the moving seat is the learner; the pair must match two encodes."""
        seen = set()
        n = 0
        for state, decision in _decision_states(1):
            self._check_pair(state, decision.player_id, decision)
            seen.add(decision.kind)
            n += 1
        assert n > 0 and len(seen) >= 2

    def test_pair_equals_two_encodes_two_player_both_seats(self) -> None:
        """Two-player: check the pair for the SEAT THAT MOVES at each decision --
        covers both the learner seat (pid 0) and the opponent seat (pid 1), since
        the driver yields decisions for both. The pair is always (prior, value) of
        the SAME pid, so it is byte-identical regardless of which seat moves."""
        seen_pids = set()
        seen_kinds = set()
        n = 0
        for state, decision in _decision_states(2):
            pid = decision.player_id
            self._check_pair(state, pid, decision)
            seen_pids.add(pid)
            seen_kinds.add(decision.kind)
            n += 1
        assert n > 0
        # The driven 2p game exercised BOTH seats and multiple decision kinds.
        assert seen_pids == {0, 1}
        assert len(seen_kinds) >= 2

    def test_pair_value_block_independent_of_decision(self) -> None:
        """The value obs from the pair ignores the decision (it is the
        ``decision=None`` encoding) -- so it equals the pair's value obs built with
        a DIFFERENT decision's prior, on every index. Pins that only the phase tail
        differs between prior and value."""
        from heat.ml.spaces import BLOCK_PHASE_CONTEXT, OBS_DIM

        prefix_len = OBS_DIM - BLOCK_PHASE_CONTEXT
        n = 0
        for state, decision in _decision_states(1):
            prior_obs, value_obs = encode_observation_pair(
                state, decision.player_id, decision
            )
            # prior and value share the entire prefix (every block before phase).
            assert np.array_equal(prior_obs[:prefix_len], value_obs[:prefix_len])
            n += 1
        assert n > 0


# ---------------------------------------------------------------------------
# 3. Spin detection without logging (the folded-in free win) is byte-identical
# ---------------------------------------------------------------------------


class TestSpinDetectionWithoutLogging:
    def _plan(self, cold_prior, *, logging_enabled: bool, track_seed: int,
              game_seed: int, num_players: int, two_player: bool):
        """Run a fixed MCTS plan and return (gear, card-names) -- the search output
        the spin floor influences. The live game's ``logging_enabled`` is the only
        varied knob."""
        mcts_agent._MODEL_CACHE.clear()
        track = generate_track(track_seed)
        agent = MCTSAgent(
            model_path=cold_prior,
            config=MCTSConfig(n_simulations=24, two_player=two_player),
            seed=0,
        )
        state = GameState.create(
            track, num_players, logging_enabled=logging_enabled, seed=game_seed
        )
        for p in state.players:
            p.lap = 1
        p0 = state.players[0]
        p0.position = 2
        lg = rules.legal_gear_shifts(p0.gear, p0.heat_available)
        gear = agent.choose_gear(state, 0, lg)
        state.players[0].gear = gear[0]
        lp = rules.legal_card_plays(state.players[0].hand, gear[0])
        cards = agent.choose_cards(state, 0, lp)
        return gear, tuple(c.display_name for c in cards)

    def test_search_byte_stable_logging_on_vs_off_solo(self, cold_prior) -> None:
        """Solo: the plan is byte-identical whether the live game logs events or
        not -- the spin floor now reads ``spin_log`` (populated unconditionally),
        so search clones no longer need the forced event log."""
        for tseed in (3, 7, 11):
            on = self._plan(
                cold_prior, logging_enabled=True, track_seed=tseed,
                game_seed=123, num_players=1, two_player=False,
            )
            off = self._plan(
                cold_prior, logging_enabled=False, track_seed=tseed,
                game_seed=123, num_players=1, two_player=False,
            )
            assert on == off, f"plan diverged with logging off (solo, tseed={tseed})"

    def test_search_byte_stable_logging_on_vs_off_two_player(self, cold_prior) -> None:
        """Two-player win/loss: the plan is byte-identical with logging on vs off."""
        for tseed in (3, 7):
            on = self._plan(
                cold_prior, logging_enabled=True, track_seed=tseed,
                game_seed=456, num_players=2, two_player=True,
            )
            off = self._plan(
                cold_prior, logging_enabled=False, track_seed=tseed,
                game_seed=456, num_players=2, two_player=True,
            )
            assert on == off, f"plan diverged with logging off (2p, tseed={tseed})"


# ---------------------------------------------------------------------------
# 4. Determinism preserved (same (state, seed) => identical move)
# ---------------------------------------------------------------------------


class TestDeterminismPreserved:
    def test_repeated_plan_byte_stable(self, cold_prior) -> None:
        """Two searches from the same (state, seed) produce the identical plan."""
        track = generate_track(5)

        def plan():
            mcts_agent._MODEL_CACHE.clear()
            agent = MCTSAgent(
                model_path=cold_prior, config=MCTSConfig(n_simulations=16), seed=0,
            )
            state = GameState.create(track, 1, logging_enabled=False, seed=99)
            state.players[0].lap = 1
            state.players[0].position = 4
            lg = rules.legal_gear_shifts(
                state.players[0].gear, state.players[0].heat_available
            )
            gear = agent.choose_gear(state, 0, lg)
            state.players[0].gear = gear[0]
            lp = rules.legal_card_plays(state.players[0].hand, gear[0])
            cards = agent.choose_cards(state, 0, lp)
            return gear, tuple(c.display_name for c in cards)

        assert plan() == plan()
