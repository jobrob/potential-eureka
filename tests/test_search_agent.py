"""Tests for the Sprint S1 LookaheadAgent (forward-rollout search).

Covers the design's three named gates plus the cross-cutting checks:

  * **Legality** -- across full solo and 4p games, every gear/cards/react/
    slipstream/discard the agent returns is in the legal set the engine hands in.
  * **horizon=0 == greedy** -- with no rollout, the chosen first move equals the
    one-ply choice scored by the same leaf value.
  * **Determinism** -- same state + same seed => byte-stable chosen plan.
  * A full solo game completes with the agent (finishes, no crash).
  * The picklable factory survives ``pickle.dumps`` and a ``parallel=True`` batch.

Mirrors the structure/style of ``tests/test_strong_heuristic.py``.
"""

from __future__ import annotations

import pickle

import pytest

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.engine import rules
from heat.engine.game import Game
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.search_agent import LookaheadAgent
from heat.agents import LookaheadAgent as ExportedLookahead
from heat.agents import _move_eval as ME
from heat.tracks.generator import generate_track
from heat.simulation.runner import (
    heuristic_agent_factory,
    lookahead_agent_factory,
    run_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_track(seed: int = 7) -> Track:
    """A generated track that contains at least one limit-1 corner (seed 7)."""
    return generate_track(seed=seed)


def _limit1_loop() -> Track:
    """A short synthetic loop with a single tight (limit-1) corner.

    Mirrors the prototype's synthetic stress track: a straight run-up into a
    limit-1 corner, lanes=2 so slipstream eligibility is reachable.
    """
    spaces = [Space(i, lanes=2) for i in range(24)]
    corners = [Corner(start=10, end=11, speed_limit=1)]
    return Track("Limit1Loop", spaces, corners, [0, 1, 2, 3, 4, 5], laps=2)


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        agent = LookaheadAgent()
        assert agent.name == "Lookahead"
        assert agent.horizon == 2
        assert agent.n_determinizations == 2
        assert agent.leaf_value == "progress"
        assert isinstance(agent.rollout_policy, HeuristicAgent)

    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ValueError):
            LookaheadAgent(horizon=-1)
        with pytest.raises(ValueError):
            LookaheadAgent(n_determinizations=0)
        with pytest.raises(ValueError):
            LookaheadAgent(leaf_value="bogus")

    def test_rollout_policy_instance_and_factory(self) -> None:
        inst = HeuristicAgent(name="Custom")
        assert LookaheadAgent(rollout_policy=inst).rollout_policy is inst
        made = LookaheadAgent(rollout_policy=lambda: HeuristicAgent(name="F"))
        assert isinstance(made.rollout_policy, HeuristicAgent)
        assert made.rollout_policy.name == "F"

    def test_exported_from_package(self) -> None:
        assert ExportedLookahead is LookaheadAgent


# ---------------------------------------------------------------------------
# Legality (hard gate) -- full games
# ---------------------------------------------------------------------------


class _LegalityWrapper:
    """Wraps an agent and asserts every decision is in the legal set handed in."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name

    def choose_gear(self, state, player_id, legal_gears):
        choice = self.inner.choose_gear(state, player_id, legal_gears)
        assert choice in legal_gears, f"illegal gear {choice} not in {legal_gears}"
        return choice

    def choose_cards(self, state, player_id, legal_plays):
        choice = self.inner.choose_cards(state, player_id, legal_plays)
        assert choice in legal_plays, "illegal card play"
        return choice

    def choose_react(self, state, player_id, max_cooldown, can_boost, has_adrenaline):
        d = self.inner.choose_react(
            state, player_id, max_cooldown, can_boost, has_adrenaline
        )
        assert 0 <= d.cooldown_count <= max_cooldown + (
            1 if d.use_adrenaline_cooldown else 0
        )
        if not can_boost:
            assert d.use_boost is False
        if not has_adrenaline:
            assert d.use_adrenaline_speed is False
            assert d.use_adrenaline_cooldown is False
        return d

    def choose_slipstream(self, state, player_id):
        choice = self.inner.choose_slipstream(state, player_id)
        assert isinstance(choice, bool)
        return choice

    def choose_discard(self, state, player_id, discardable):
        choice = self.inner.choose_discard(state, player_id, discardable)
        assert all(c in discardable for c in choice)
        return choice


class TestLegality:
    def test_all_decisions_legal_solo(self) -> None:
        track = _gen_track(7)
        for game_idx in range(4):
            agent = _LegalityWrapper(
                LookaheadAgent(horizon=2, n_determinizations=2)
            )
            game = Game(track, [agent], logging_enabled=False, seed=1000 + game_idx)
            result = game.run()
            assert len(result.finish_order) == 1

    def test_all_decisions_legal_4p(self) -> None:
        track = _gen_track(11)
        for game_idx in range(3):
            agents = [
                _LegalityWrapper(LookaheadAgent(horizon=2, n_determinizations=1)),
                _LegalityWrapper(HeuristicAgent()),
                _LegalityWrapper(HeuristicAgent()),
                _LegalityWrapper(HeuristicAgent()),
            ]
            game = Game(track, agents, logging_enabled=False, seed=2000 + game_idx)
            result = game.run()
            assert len(result.finish_order) == 4

    def test_never_illegal_on_limit1_loop(self) -> None:
        """The tight-corner stress track is the case S1 is designed for."""
        track = _limit1_loop()
        agent = _LegalityWrapper(LookaheadAgent(horizon=2, n_determinizations=2))
        game = Game(track, [agent], logging_enabled=False, seed=99)
        result = game.run()
        assert len(result.finish_order) == 1


# ---------------------------------------------------------------------------
# horizon=0 == greedy one-ply
# ---------------------------------------------------------------------------


class TestHorizonZeroIsGreedy:
    """``horizon=0`` must equal scoring each candidate's first move at the leaf.

    With no rollout depth, the agent forces the candidate move for a single
    round and scores the resulting state by the leaf value. A *reference* greedy
    that simulates only that one forced round and reads the same leaf value must
    pick the identical plan. We assert the agent's chosen plan equals the argmax
    of an independent single-round simulation -- i.e. there is no hidden rollout.
    """

    def _greedy_reference(
        self, agent: LookaheadAgent, state: GameState, player_id: int
    ) -> tuple[tuple[int, int], tuple[Card, ...]]:
        """Brute-force the best candidate by simulating exactly one forced round."""
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
        state.players[0].position = 2  # run-up to the limit-1 corner at 4-5

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

    def test_horizon0_differs_from_deeper_search_somewhere(self) -> None:
        """Sanity: depth actually changes the chosen plan on the tight track.

        If horizon 0 and horizon 2 always agreed, the rollout would be inert.
        Over a sweep of pre-corner states and run-up distances at least one
        full ``(gear, cards)`` plan must differ. We compare the full plan (not
        just the gear) because depth most often shifts the *card* commitment in
        front of a limit-1 corner while the gear is unchanged.
        """
        track = _limit1_loop()

        def full_plan(horizon: int, seed: int, position: int):
            a = LookaheadAgent(horizon=horizon, n_determinizations=1)
            st = GameState.create(track, 1, logging_enabled=False, seed=seed)
            st.players[0].lap = 1
            st.players[0].position = position
            lg = rules.legal_gear_shifts(
                st.players[0].gear, st.players[0].heat_available
            )
            gear = a.choose_gear(st, 0, lg)
            st.players[0].gear = gear[0]
            lp = rules.legal_card_plays(st.players[0].hand, gear[0])
            cards = a.choose_cards(st, 0, lp)
            return gear, tuple(c.display_name for c in cards)

        differed = False
        for seed in range(20):
            for position in (7, 8, 9):  # varying run-up to the limit-1 corner
                if full_plan(0, seed, position) != full_plan(2, seed, position):
                    differed = True
                    break
            if differed:
                break
        assert differed, "horizon=2 never differed from horizon=0 on the tight track"


# ---------------------------------------------------------------------------
# Determinism under a fixed seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_state_same_plan(self) -> None:
        track = _gen_track(7)

        def plan(seed_agent: int):
            a = LookaheadAgent(horizon=2, n_determinizations=2, seed=seed_agent)
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

        first = plan(0)
        for _ in range(5):
            assert plan(0) == first

    def test_batch_reproducible(self) -> None:
        track_a = _gen_track(9)
        track_b = _gen_track(9)
        facs = [
            lookahead_agent_factory(horizon=2, n_determinizations=1),
            heuristic_agent_factory(),
        ]
        a = run_batch(track_a, facs, num_games=4, seed=321, parallel=False, laps=2)
        b = run_batch(track_b, facs, num_games=4, seed=321, parallel=False, laps=2)
        for oa, ob in zip(a, b):
            assert oa.finish_order == ob.finish_order
            assert oa.total_rounds == ob.total_rounds


# ---------------------------------------------------------------------------
# Full game completes
# ---------------------------------------------------------------------------


def _solo_spins(agent, track, seed: int) -> int:
    """Run one solo game and return the seat-0 spin count."""
    game = Game(track, [agent], logging_enabled=True, seed=seed)
    result = game.run()
    assert result.finish_order == [0]
    return sum(1 for e in result.event_log if e.event_type == "spin_out")


class TestFullGame:
    def test_solo_game_finishes_and_spins_at_most_heuristic(self) -> None:
        """The agent finishes and clears corners no worse than the heuristic.

        The previous ``spins == 0`` assertion was an artifact of a spin
        double-counting bug (it read the cumulative event log once per round, so
        an early spin was multiply-counted; the "0" was luck on this seed, not a
        guarantee). The agent's real, *correctly accounted* contract is the S1
        success bar: spins-per-tight-corner are no worse than ``HeuristicAgent``
        on the same track/seed -- so we assert exactly that against a live
        baseline rather than a brittle magic constant. On track seed 7 / game
        seed 42 the corrected agent clears all corners (0 spins) while the
        heuristic spins once; the assertion is ``<=`` so it stays valid if the
        engine or agent shifts the line slightly, as long as the agent never
        regresses below the reference it is designed to beat.
        """
        track = _gen_track(7)
        agent = LookaheadAgent(horizon=2, n_determinizations=2, seed=0)
        look_spins = _solo_spins(agent, track, seed=42)

        baseline_spins = _solo_spins(HeuristicAgent(), track, seed=42)

        assert look_spins <= baseline_spins, (
            f"Lookahead spun {look_spins} times vs heuristic {baseline_spins} "
            f"on track 7 / seed 42 -- it must clear tight corners no worse than "
            f"the reference it is built to beat."
        )


# ---------------------------------------------------------------------------
# Picklability
# ---------------------------------------------------------------------------


class TestPicklability:
    def test_factory_pickles(self) -> None:
        fac = lookahead_agent_factory(horizon=2, n_determinizations=2)
        restored = pickle.loads(pickle.dumps(fac))
        agent = restored(0, 123)
        assert isinstance(agent, LookaheadAgent)
        assert agent.horizon == 2
        assert agent.n_determinizations == 2

    def test_parallel_batch_runs(self) -> None:
        track = _gen_track(13)
        facs = [
            lookahead_agent_factory(horizon=1, n_determinizations=1),
            heuristic_agent_factory(),
        ]
        outcomes = run_batch(track, facs, num_games=4, seed=9, parallel=True, laps=2)
        assert len(outcomes) == 4
