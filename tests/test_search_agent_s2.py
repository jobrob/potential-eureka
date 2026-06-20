"""Tests for the Sprint S2 hardening of the LookaheadAgent.

S2 adds, on top of the S1 forward-rollout search (all opt-in, so the S1 tests
in ``test_search_agent.py`` exercise the unchanged default path):

  * **Hidden-information determinization** -- opponents' hidden hands/decks are
    re-sampled from the public belief (uniform over the cards known to be in
    their hand+draw pile) before each rollout. Tests: the belief is
    multiset-preserving (never invents/loses an opponent card, never touches the
    learner's own hand or any discard pile), determinism is preserved under a
    fixed seed, and a 4p game still completes legally with it on.
  * **Top-k branching control** -- candidates ranked by the fast ``_move_eval``
    prior, only the best ``top_k`` rolled out. Test: on a small case where the
    full (un-pruned) search is affordable, a generous top-k never drops the
    plan the full search would have chosen.
  * **Per-move sim budget** -- a cap on rollout clones per move. Test: the
    profiler never records more clones in a single move than the budget.
  * **Profiling** -- clones/move and ms/move are recorded.

Mirrors the style of ``tests/test_search_agent.py``.
"""

from __future__ import annotations

import random
from collections import Counter

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
# Helpers (shared with the S1 suite's style)
# ---------------------------------------------------------------------------


def _gen_track(seed: int = 7) -> Track:
    return generate_track(seed=seed)


def _limit1_loop() -> Track:
    spaces = [Space(i, lanes=2) for i in range(24)]
    corners = [Corner(start=10, end=11, speed_limit=1)]
    return Track("Limit1Loop", spaces, corners, [0, 1, 2, 3, 4, 5], laps=2)


def _pre_corner_4p_state(seed: int = 123) -> GameState:
    """A 4p state with seat 0 run-up to the limit-1 corner, ready to plan."""
    track = _gen_track(7)
    state = GameState.create(track, 4, logging_enabled=False, seed=seed)
    state.players[0].lap = 1
    state.players[0].position = 2
    return state


def _plan(agent: LookaheadAgent, state: GameState, player_id: int = 0):
    """Drive the agent through one gear+cards decision; return the plan tuple."""
    p = state.get_player(player_id)
    legal_gears = rules.legal_gear_shifts(p.gear, p.heat_available)
    gear = agent.choose_gear(state, player_id, legal_gears)
    state.get_player(player_id).gear = gear[0]
    legal_plays = rules.legal_card_plays(
        state.get_player(player_id).hand, gear[0]
    )
    cards = agent.choose_cards(state, player_id, legal_plays)
    return gear, tuple(c.display_name for c in cards)


def _pool_multiset(player) -> Counter:
    """All cards a player owns, as a (type, value, id) multiset."""
    cards = list(player.hand) + list(player.deck.draw_pile) + list(
        player.deck.discard_pile
    )
    return Counter((c.card_type, c.value, c.id) for c in cards)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestS2Config:
    def test_new_defaults_match_s1(self) -> None:
        a = LookaheadAgent()
        assert a.determinize_hidden is False
        assert a.top_k is None
        assert a.sim_budget is None

    def test_rejects_bad_s2_config(self) -> None:
        with pytest.raises(ValueError):
            LookaheadAgent(top_k=0)
        with pytest.raises(ValueError):
            LookaheadAgent(sim_budget=0)


# ---------------------------------------------------------------------------
# Hidden-information determinization
# ---------------------------------------------------------------------------


class TestDeterminization:
    def test_preserves_each_opponent_card_multiset(self) -> None:
        """Resampling must never invent or lose an opponent's card."""
        state = _pre_corner_4p_state()
        agent = LookaheadAgent(determinize_hidden=True)
        before = {i: _pool_multiset(state.get_player(i)) for i in range(4)}

        clone = state.clone(reseed=123)
        agent._determinize_opponents(clone, 0, random.Random(7))

        after = {i: _pool_multiset(clone.get_player(i)) for i in range(4)}
        for i in range(4):
            assert before[i] == after[i], f"opponent {i} card multiset changed"

    def test_leaves_own_hand_and_discards_untouched(self) -> None:
        state = _pre_corner_4p_state()
        agent = LookaheadAgent(determinize_hidden=True)
        clone = state.clone(reseed=123)
        agent._determinize_opponents(clone, 0, random.Random(7))

        # The learner's own hand is not hidden from itself -> unchanged order.
        assert [c.id for c in state.get_player(0).hand] == [
            c.id for c in clone.get_player(0).hand
        ]
        # Discards are public -> unchanged for every opponent.
        for i in range(1, 4):
            assert list(state.get_player(i).deck.discard_pile) == list(
                clone.get_player(i).deck.discard_pile
            )

    def test_keeps_opponent_hand_size(self) -> None:
        state = _pre_corner_4p_state()
        agent = LookaheadAgent(determinize_hidden=True)
        clone = state.clone(reseed=123)
        agent._determinize_opponents(clone, 0, random.Random(7))
        for i in range(1, 4):
            assert len(clone.get_player(i).hand) == len(
                state.get_player(i).hand
            )

    def test_actually_resamples_hidden_hands(self) -> None:
        """The belief must move at least one opponent's hand (else it is inert)."""
        state = _pre_corner_4p_state()
        agent = LookaheadAgent(determinize_hidden=True)
        clone = state.clone(reseed=123)
        agent._determinize_opponents(clone, 0, random.Random(7))
        moved = sum(
            1
            for i in range(1, 4)
            if [c.id for c in state.get_player(i).hand]
            != [c.id for c in clone.get_player(i).hand]
        )
        assert moved >= 1

    def test_determinized_search_is_deterministic(self) -> None:
        """Same state + seed => byte-stable plan even with determinization on."""

        def plan_once() -> tuple:
            return _plan(
                LookaheadAgent(
                    horizon=2,
                    n_determinizations=2,
                    determinize_hidden=True,
                    seed=0,
                ),
                _pre_corner_4p_state(),
            )

        first = plan_once()
        for _ in range(5):
            assert plan_once() == first

    def test_4p_game_completes_legally_with_determinization(self) -> None:
        track = _gen_track(11)
        agents = [
            LookaheadAgent(horizon=2, n_determinizations=2, determinize_hidden=True),
            HeuristicAgent(),
            HeuristicAgent(),
            HeuristicAgent(),
        ]
        game = Game(track, agents, logging_enabled=False, seed=2024)
        result = game.run()
        assert len(result.finish_order) == 4

    def test_determinization_noop_in_solo(self) -> None:
        """With no opponents, determinize_hidden must change nothing."""
        track = _gen_track(7)
        state = GameState.create(track, 1, logging_enabled=False, seed=5)
        agent = LookaheadAgent(determinize_hidden=True)
        clone = state.clone(reseed=99)
        before = [c.id for c in clone.get_player(0).hand]
        agent._determinize_opponents(clone, 0, random.Random(1))
        assert [c.id for c in clone.get_player(0).hand] == before


# ---------------------------------------------------------------------------
# Top-k branching control
# ---------------------------------------------------------------------------


class TestTopK:
    def test_topk_preserves_full_search_best_when_k_generous(self) -> None:
        """A top-k that keeps every candidate must pick the full-search plan.

        On a small pre-corner state the full (un-pruned) search is cheap. We run
        it once unpruned and once with ``top_k`` equal to the candidate count;
        the chosen plan must be identical -- pruning to "all of them" is a no-op,
        which proves the prior ranking never reorders the eventual argmax away.
        """
        track = _limit1_loop()

        # Count the candidates so we can set top_k = full breadth.
        probe = LookaheadAgent(horizon=2, n_determinizations=1)
        st = GameState.create(track, 1, logging_enabled=False, seed=3)
        st.players[0].lap = 1
        st.players[0].position = 8
        legal_gears = rules.legal_gear_shifts(
            st.players[0].gear, st.players[0].heat_available
        )
        n_candidates = len(probe._candidate_plans(st, 0, legal_gears))

        def plan_with(top_k):
            a = LookaheadAgent(
                horizon=2, n_determinizations=1, top_k=top_k, seed=0
            )
            s = GameState.create(track, 1, logging_enabled=False, seed=3)
            s.players[0].lap = 1
            s.players[0].position = 8
            return _plan(a, s)

        full = plan_with(None)
        kept_all = plan_with(n_candidates)
        assert kept_all == full

    def test_topk_one_still_returns_a_legal_plan(self) -> None:
        """An aggressive top_k=1 must still drive a full game legally."""
        track = _gen_track(7)
        agent = LookaheadAgent(horizon=2, n_determinizations=1, top_k=1)
        game = Game(track, [agent], logging_enabled=False, seed=7)
        result = game.run()
        assert result.finish_order == [0]


# ---------------------------------------------------------------------------
# Per-move sim budget + profiling
# ---------------------------------------------------------------------------


class TestBudgetAndProfile:
    def test_budget_caps_clones_per_move(self) -> None:
        budget = 4
        agent = LookaheadAgent(
            horizon=2, n_determinizations=2, sim_budget=budget, seed=0
        )
        state = _pre_corner_4p_state()
        p = state.get_player(0)
        legal_gears = rules.legal_gear_shifts(p.gear, p.heat_available)
        agent.choose_gear(state, 0, legal_gears)  # exactly one planning call
        assert agent.profile.moves == 1
        assert agent.profile.clones <= budget

    def test_budget_scores_at_least_one_candidate(self) -> None:
        """A budget smaller than one candidate's cost still returns a real plan."""
        agent = LookaheadAgent(
            horizon=2, n_determinizations=4, sim_budget=1, seed=0
        )
        state = _pre_corner_4p_state()
        gear, cards = _plan(agent, state)
        assert gear is not None and len(cards) >= 1
        # One candidate (4 clones) was scored despite the budget of 1.
        assert agent.profile.clones >= 1

    def test_profile_records_clones_and_time(self) -> None:
        agent = LookaheadAgent(horizon=2, n_determinizations=2, seed=0)
        state = _pre_corner_4p_state()
        _plan(agent, state)
        assert agent.profile.moves >= 1
        assert agent.profile.clones > 0
        assert agent.profile.clones_per_move() > 0
        assert agent.profile.ms_per_move() >= 0.0

    def test_unbudgeted_unpruned_matches_s1_argmax(self) -> None:
        """Default S2 config (no prune, no budget) must equal the raw argmax.

        This is the contract that keeps every S1 test valid: with ``top_k=None``
        and ``sim_budget=None`` the planner scores every candidate, so the chosen
        plan is exactly the S1 best-by-rollout plan regardless of the prior
        ordering introduced in S2.
        """
        track = _limit1_loop()
        agent = LookaheadAgent(horizon=2, n_determinizations=1, seed=0)
        st = GameState.create(track, 1, logging_enabled=False, seed=3)
        st.players[0].lap = 1
        st.players[0].position = 8

        # Raw argmax over all candidates in enumeration order (S1 algorithm).
        legal_gears = rules.legal_gear_shifts(
            st.players[0].gear, st.players[0].heat_available
        )
        candidates = agent._candidate_plans(st, 0, legal_gears)
        sig = agent._turn_signature(st, 0)
        turn_seed = agent._turn_seed(sig)
        best = candidates[0]
        best_v = float("-inf")
        for idx, (gear, cards) in enumerate(candidates):
            v = agent._score_plan(st, 0, gear, cards, turn_seed, idx)
            if v > best_v:
                best_v = v
                best = (gear, cards)
        raw = (best[0], tuple(c.display_name for c in best[1]))

        chosen = _plan(agent, st)
        assert chosen == raw
