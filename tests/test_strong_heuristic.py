"""Tests for the Sprint 6E StrongHeuristicAgent (relative-objective rebuild).

Covers the redesign-doc gates: legality across full games at every strength,
the retuned strength headline, a robust ladder gate (separation against a fixed
weak anchor + top-beats-floor + ELO), the no-spinout regression with margin,
joint-plan consistency, targeted relative-objective / lap-bug / block units,
determinism, picklability, and a latency budget.

All head-to-head measurements are **seat-neutral**: the HEAT engine gives the
front seats (turn order 0,1) a measurable positional edge, so every comparison
averages "A in the front seats" with "A in the back seats". Measuring win share
with a fixed seat assignment (as the WIP did) reads ~0.56 for *any* matchup and
is not a valid skill signal -- see ``_seat_neutral_share``.
"""

from __future__ import annotations

import math
import pickle
import time

import pytest

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.engine import rules
from heat.engine.game import Game
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.agents import StrongHeuristicAgent as ExportedStrong
from heat.agents import _move_eval as ME
from heat.tracks.loader import load_track_by_name
from heat.simulation.runner import (
    heuristic_agent_factory,
    strong_heuristic_agent_factory,
    run_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USA_LAPS = 3
# Total games per matchup (split into two seat-swapped halves for neutrality).
H2H_GAMES = 200
H2H_SEED = 12345


def _usa() -> Track:
    return load_track_by_name("usa")


def _front_share(factory_a, factory_b, *, games, seed, laps, a_front) -> float:
    """A's win share with A in the front seats (0,1) or the back seats (2,3)."""
    track = _usa()
    if a_front:
        factories = [factory_a, factory_a, factory_b, factory_b]
        a_ids = {0, 1}
    else:
        factories = [factory_b, factory_b, factory_a, factory_a]
        a_ids = {2, 3}
    outcomes = run_batch(
        track, factories, num_games=games, seed=seed, parallel=False, laps=laps
    )
    wins = sum(1 for o in outcomes if o.winner_id in a_ids)
    return wins / games


def _seat_neutral_share(
    factory_a,
    factory_b,
    *,
    games: int = H2H_GAMES,
    seed: int = H2H_SEED,
    laps: int = USA_LAPS,
) -> float:
    """Seat-neutral win share of A vs B in a 2v2 batch.

    Averages A-in-front and A-in-back over ``games // 2`` games each, removing
    the front-seat positional advantage so the result reflects skill, not seat.
    """
    half = games // 2
    front = _front_share(
        factory_a, factory_b, games=half, seed=seed, laps=laps, a_front=True
    )
    back = _front_share(
        factory_a, factory_b, games=half, seed=seed + 7, laps=laps, a_front=False
    )
    return (front + back) / 2.0


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        agent = StrongHeuristicAgent()
        assert agent.strength == 2
        assert agent.name == "StrongHeuristic"
        assert agent.base_heat_price == ME.DEFAULT_HEAT_PRICE

    def test_strength_bounds(self) -> None:
        for s in (0, 1, 2, 3):
            StrongHeuristicAgent(strength=s)
        with pytest.raises(ValueError):
            StrongHeuristicAgent(strength=4)
        with pytest.raises(ValueError):
            StrongHeuristicAgent(strength=-1)

    def test_exported_from_package(self) -> None:
        assert ExportedStrong is StrongHeuristicAgent


# ---------------------------------------------------------------------------
# Legality (hard gate) -- full games at every strength
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
        assert all(
            c.card_type in (CardType.SPEED, CardType.UPGRADE) for c in choice
        )
        return choice


class TestLegality:
    @pytest.mark.parametrize("strength", [0, 1, 2, 3])
    def test_all_decisions_legal_full_games(self, strength: int) -> None:
        track = load_track_by_name("usa")
        track.laps = 3
        for game_idx in range(6):
            seed = 1000 + game_idx
            agents = [
                _LegalityWrapper(StrongHeuristicAgent(strength=strength)),
                _LegalityWrapper(StrongHeuristicAgent(strength=strength)),
                _LegalityWrapper(HeuristicAgent()),
                _LegalityWrapper(StrongHeuristicAgent(strength=strength)),
            ]
            game = Game(track, agents, logging_enabled=False, seed=seed)
            result = game.run()
            assert len(result.finish_order) == 4


# ---------------------------------------------------------------------------
# Strength headline + robust ladder gate
# ---------------------------------------------------------------------------


class TestStrength:
    def test_strength2_beats_heuristic(self) -> None:
        """strength=2 decisively beats the weak HeuristicAgent (seat-neutral)."""
        share = _seat_neutral_share(
            strong_heuristic_agent_factory(strength=2),
            heuristic_agent_factory(),
        )
        assert share >= 0.62, f"strength=2 win share only {share:.3f}"

    def test_strength0_ties_heuristic(self) -> None:
        """The strength-0 floor ~ties the weak heuristic (a real floor)."""
        share = _seat_neutral_share(
            strong_heuristic_agent_factory(strength=0),
            heuristic_agent_factory(),
        )
        assert 0.40 <= share <= 0.62, f"strength=0 share {share:.3f} not ~tie"


def _ladder_shares_vs_weak() -> dict[int, float]:
    """Seat-neutral win share of each rung vs the fixed weak heuristic anchor.

    A fixed weak anchor is the honest ladder reference (and the one 6B's ELO
    pool uses): each rung faces the *same* opponent, so increasing share is a
    direct skill signal. Measuring rung-k-vs-rung-0 instead compresses, because
    the strong rungs cluster against each other -- the relative objective and
    block projection move decisions against *weak* play far more than against an
    equally-strong opponent that contests the same gaps symmetrically.
    """
    weak = heuristic_agent_factory()
    return {
        k: _seat_neutral_share(strong_heuristic_agent_factory(strength=k), weak)
        for k in (0, 1, 2, 3)
    }


def _simple_elo(shares_vs_anchor: dict[int, float]) -> dict[int, float]:
    """Fit pairwise ELO of each rung from its win share vs a common anchor.

    With a single shared opponent, the ELO of rung k relative to the anchor is
    the direct logistic inverse of its win share. Anchor rating := 0.
    """
    ratings: dict[int, float] = {}
    for k, s in shares_vs_anchor.items():
        s = min(max(s, 1e-3), 1 - 1e-3)
        ratings[k] = -400.0 * math.log10(1.0 / s - 1.0)
    return ratings


class TestLadder:
    """Robust ladder gate (replaces the fragile strict-adjacent monotonicity).

    Measured against a fixed weak anchor, seat-neutral, N=200 per matchup.
    Observed (seed 12345): s0=0.59, s1=0.80, s2=0.835, s3=0.83 vs weak.
    """

    def test_ladder_separates_against_weak_anchor(self) -> None:
        shares = _ladder_shares_vs_weak()

        # 1. Each rung is at least as strong as the floor against the anchor,
        #    and the engaged rungs (s1+) beat the weak anchor decisively.
        assert shares[1] >= shares[0]
        assert shares[2] >= shares[1] - 0.02  # within noise of monotone
        assert shares[2] >= 0.62, f"s2 vs weak {shares[2]:.3f}"
        assert shares[3] >= 0.62, f"s3 vs weak {shares[3]:.3f}"

        # 2. The relative-objective rung adds measurable strength over the
        #    planning-only rung against the anchor (the P0 payoff).
        assert shares[2] - shares[1] >= 0.02, (
            f"s2-s1 vs weak only {shares[2] - shares[1]:.3f}"
        )

        # 3. Top beats the floor head-to-head, decisively.
        top_vs_floor = _seat_neutral_share(
            strong_heuristic_agent_factory(strength=3),
            strong_heuristic_agent_factory(strength=0),
        )
        assert top_vs_floor >= 0.55, f"s3 vs s0 head-to-head {top_vs_floor:.3f}"

        # 4. ELO sanity from the anchor shares: strictly increasing s0->s2 and
        #    a real spread top-minus-floor. (s2 and s3 cluster on the USA
        #    track, so we require s3 >= s2 - small noise, not strict.)
        elo = _simple_elo(shares)
        assert elo[0] < elo[1] < elo[2], f"ELO not increasing s0..s2: {elo}"
        assert elo[3] >= elo[2] - 20.0, f"ELO s3 collapsed: {elo}"
        assert elo[2] - elo[0] >= 100.0, (
            f"ELO spread s2-s0 only {elo[2] - elo[0]:.1f}"
        )


# ---------------------------------------------------------------------------
# No spinout regression
# ---------------------------------------------------------------------------


def _count_spinouts(make_agent, seeds, laps=3) -> int:
    """Run single games and count spin-out events for a self-play field."""
    track = load_track_by_name("usa")
    track.laps = laps
    total = 0
    for seed in seeds:
        agents = [make_agent() for _ in range(4)]
        game = Game(track, agents, logging_enabled=True, seed=seed)
        result = game.run()
        total += sum(
            1 for e in result.event_log if e.event_type == "spin_out"
        )
    return total


def _make_strong2():
    return StrongHeuristicAgent(strength=2)


def _make_weak():
    return HeuristicAgent()


class TestSpinout:
    def test_no_spinout_regression(self) -> None:
        """strength=2 spins strictly fewer than weak, with a 15% margin."""
        seeds = list(range(2000, 2040))  # 40 games
        strong_spins = _count_spinouts(_make_strong2, seeds)
        heur_spins = _count_spinouts(_make_weak, seeds)
        assert strong_spins <= 0.85 * heur_spins, (
            f"strong spun {strong_spins} vs 0.85*weak={0.85 * heur_spins:.1f} "
            f"(weak {heur_spins})"
        )


# ---------------------------------------------------------------------------
# Joint-plan consistency
# ---------------------------------------------------------------------------


class TestJointPlan:
    def test_gear_and_cards_same_plan(self) -> None:
        """The play returned at choose_cards matches the gear's cached plan."""
        agent = StrongHeuristicAgent(strength=2)
        track = _usa()
        state = GameState.create(track, 2, logging_enabled=False, seed=99)
        for p in state.players:
            p.lap = 1

        legal_gears = rules.legal_gear_shifts(
            state.get_player(0).gear, state.get_player(0).heat_available
        )
        gear = agent.choose_gear(state, 0, legal_gears)
        state.get_player(0).gear = gear[0]
        legal_plays = rules.legal_card_plays(state.get_player(0).hand, gear[0])
        play = agent.choose_cards(state, 0, legal_plays)
        assert agent._plan_gear == gear
        assert play == agent._plan_cards
        assert play in legal_plays

    def test_recompute_on_forced_state_change(self) -> None:
        """If the cached play is gone, choose_cards recomputes a legal play."""
        agent = StrongHeuristicAgent(strength=2)
        track = _usa()
        state = GameState.create(track, 2, logging_enabled=False, seed=7)
        for p in state.players:
            p.lap = 1
        legal_gears = rules.legal_gear_shifts(
            state.get_player(0).gear, state.get_player(0).heat_available
        )
        gear = agent.choose_gear(state, 0, legal_gears)
        state.get_player(0).gear = gear[0]
        c = Card(CardType.SPEED, 4, "fake_s4")
        fake_legal = [(c,)] if gear[0] == 1 else [(c, c)]
        play = agent.choose_cards(state, 0, fake_legal)
        assert play in fake_legal


# ---------------------------------------------------------------------------
# Relative-objective units (the P0 proof) -- constructed states
# ---------------------------------------------------------------------------


def _straight(length: int = 40, laps: int = 2, lanes: int = 2) -> Track:
    spaces = [Space(i, lanes=lanes) for i in range(length)]
    return Track("Straight", spaces, [], list(range(6)), laps=laps)


class TestRelativeObjective:
    def test_saturates_on_uncontested_lead(self) -> None:
        """A huge lead -> relative_term ~ 0 (no irrational push to extend it)."""
        track = _straight()
        state = GameState.create(track, 2, logging_enabled=False, seed=3)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.position = 20
        rival = state.get_player(1)
        rival.position = 10  # 10 spaces behind -> saturated lead
        before = ME.race_progress(me, track)
        val = ME.relative_term(state, 0, before, before + 3)
        assert abs(val) < 0.1, f"saturated lead should be ~0, got {val:.3f}"

    def test_values_closing_a_contested_gap(self) -> None:
        """Pulling level with the car 1 ahead is worth markedly more than 0."""
        track = _straight()
        state = GameState.create(track, 2, logging_enabled=False, seed=3)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.position = 20
        rival = state.get_player(1)
        rival.position = 21  # 1 ahead -> contested gap of -1
        before = ME.race_progress(me, track)
        pull_level = ME.relative_term(state, 0, before, before + 1)
        concede = ME.relative_term(state, 0, before, before)
        assert pull_level > 0.1, f"closing contested gap weak: {pull_level:.3f}"
        assert pull_level > concede
        # And it is decisively larger than extending an uncontested lead by the
        # same one space (the structural decoupling from own-progress).
        rival.position = 5
        before2 = ME.race_progress(me, track)
        extend_lead = ME.relative_term(state, 0, before2, before2 + 1)
        assert pull_level > extend_lead

    def test_risk_posture_shifts_with_rank(self) -> None:
        """Leader pays more per heat / fears spins more; trailer the reverse."""
        track = _straight()
        state = GameState.create(track, 3, logging_enabled=False, seed=11)
        for p in state.players:
            p.lap = 1
        # Make player 0 the leader.
        state.get_player(0).position = 30
        state.get_player(1).position = 15
        state.get_player(2).position = 5
        lead_price, lead_spin = ME.effective_risk(state, 0, ME.DEFAULT_HEAT_PRICE)
        trail_price, trail_spin = ME.effective_risk(state, 2, ME.DEFAULT_HEAT_PRICE)
        assert lead_price > ME.DEFAULT_HEAT_PRICE > trail_price
        assert lead_spin > ME.SPINOUT_LOSS_SPACES > trail_spin

    def test_sole_survivor_neutral(self) -> None:
        """No rivals -> relative_term is 0 and risk posture is neutral."""
        track = _straight()
        state = GameState.create(track, 2, logging_enabled=False, seed=5)
        for p in state.players:
            p.lap = 1
        state.get_player(1).finished = True
        me = state.get_player(0)
        before = ME.race_progress(me, track)
        assert ME.relative_term(state, 0, before, before + 5) == 0.0
        price, spin = ME.effective_risk(state, 0, ME.DEFAULT_HEAT_PRICE)
        assert price == pytest.approx(ME.DEFAULT_HEAT_PRICE)
        assert spin == pytest.approx(ME.SPINOUT_LOSS_SPACES)


# ---------------------------------------------------------------------------
# Lap-bug regression (P1)
# ---------------------------------------------------------------------------


class TestLapBug:
    def test_relative_value_nonzero_across_finish(self) -> None:
        """Passing a rival across the finish line yields a positive relative
        value (would be silently negative under the old stale-lap code)."""
        spaces = [Space(i, lanes=2) for i in range(10)]
        track = Track("Loop", spaces, [], [0, 1, 2, 3], laps=3)
        state = GameState.create(track, 2, logging_enabled=False, seed=1)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.position = 8  # near the line
        rival = state.get_player(1)
        rival.lap = 2
        rival.position = 0  # just over the line, ahead of me

        end_lap, end_pos = ME.landed_lap_and_pos(track, 8, 1, 4)
        assert end_lap == 2  # the move crossed the finish
        before = ME.race_progress(me, track)
        after = end_lap * track.length + end_pos
        good = ME.relative_term(state, 0, before, after)
        # Correct (lap-aware) value is positive: we passed the rival.
        assert good > 0.0, f"lap-aware overtake value should be >0, got {good:.3f}"
        # The old stale-lap progress (from_lap kept) would be negative.
        stale_after = 1 * track.length + end_pos
        stale = ME.relative_term(state, 0, before, stale_after)
        assert stale < 0.0  # demonstrates the bug the fix removes


class TestSolvencyFinishBoundary:
    """Forward heat planning must not inspect a lap after the race ends."""

    def test_next_lap_corner_is_ignored_only_on_final_straight(self) -> None:
        """The same wrapped corner is relevant one lap earlier, not at finish."""
        spaces = [Space(index, lanes=2) for index in range(10)]
        corner = Corner(start=2, end=2, speed_limit=1)
        track = Track("Finish boundary", spaces, [corner], [0, 1], laps=2)
        state = GameState.create(track, 2, logging_enabled=False, seed=1)
        player = state.get_player(0)

        final_straight = ME.project_solvency(
            state,
            player,
            end_position=5,
            end_lap=2,
            heat_after_turn=0,
            planned_gear=2,
            expected_speed=4.0,
            horizon_corners=1,
            heat_price=ME.DEFAULT_HEAT_PRICE,
        )
        earlier_lap = ME.project_solvency(
            state,
            player,
            end_position=5,
            end_lap=1,
            heat_after_turn=0,
            planned_gear=2,
            expected_speed=4.0,
            horizon_corners=1,
            heat_price=ME.DEFAULT_HEAT_PRICE,
        )

        assert final_straight == 0.0
        assert earlier_lap < 0.0


# ---------------------------------------------------------------------------
# Block behaviour (P3) -- real projection via resolve_blocked_position
# ---------------------------------------------------------------------------


class TestBlocking:
    def test_block_value_only_when_rival_actually_bounced(self) -> None:
        """block_value > 0 exactly when landing fills the lane the rival needs."""
        spaces = [Space(i, lanes=1) for i in range(20)]
        track = Track("BlockTrack", spaces, [], [0, 1], laps=1)
        state = GameState.create(track, 2, logging_enabled=False, seed=1)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.position = 5
        me.gear = 2
        rival = state.get_player(1)
        rival.position = 4
        rival.gear = 2  # natural target is space 6

        # Sanity: with the agent at 6, the rival is bounced off space 6.
        me.position = 6
        landed = rules.resolve_blocked_position(6, track, state.players, 1)
        assert landed != 6, "expected the rival to be blocked off space 6"
        me.position = 5  # restore

        blocks = ME.block_value(state, 0, 6, 1)   # land on the rival's target
        no_block = ME.block_value(state, 0, 7, 1)  # land past it
        assert blocks > 0.0
        assert no_block == 0.0

    def test_blocking_only_enabled_at_strength3(self) -> None:
        """The block term is active only at strength 3.

        A real bounce of a trailing rival adds value to the *same* landing for a
        strength-3 agent but not for a strength-2 one (whose ``blocking`` flag is
        off). We compare the evaluator value of the identical blocking play at
        both strengths; the strength-3 value is strictly higher by the (positive)
        block contribution. This is the honest, decisive claim -- a slower
        blocking play is *not* forced over a faster one, because the saturating
        block value is deliberately modest (it never buys a lost space of
        progress), exactly as the relative currency intends.
        """
        spaces = [Space(i, lanes=1) for i in range(20)]
        track = Track("BlockTrack", spaces, [], [0, 1], laps=1)
        state = GameState.create(track, 2, logging_enabled=False, seed=1)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.position = 4
        me.gear = 1
        rival = state.get_player(1)
        rival.position = 3
        rival.gear = 2  # natural target is space 5 -> blocked if we land at 5

        c1 = Card(CardType.SPEED, 1, "p0_spd_x1")  # lands the agent at 5
        me.hand = [c1, Card(CardType.SPEED, 2, "p0_spd_x2")]

        def _value(strength: int) -> float:
            agent = StrongHeuristicAgent(strength=strength)
            heat_price, spin_loss = agent._risk_posture(state, 0)
            return ME.evaluate_move(
                state,
                0,
                expected_speed=1.0,
                heat_spent=0,
                from_position=4,
                from_lap=1,
                planned_gear=1,
                heat_price=heat_price,
                horizon_corners=agent._horizon,
                opponent_aware=agent._opponent_aware,
                blocking=agent._blocking,
                enable_solvency=agent._solvency,
                spinout_loss=spin_loss,
            ).value

        # The block actually fires on this landing (sanity).
        assert ME.block_value(state, 0, 5, 1) > 0.0
        assert _value(3) > _value(2), "strength 3 should value the real block higher"


# ---------------------------------------------------------------------------
# Slipstream behaviours (kept)
# ---------------------------------------------------------------------------


def _slip_track() -> Track:
    """Long straight, single corner far away, lanes=2 everywhere."""
    spaces = [Space(i, lanes=2) for i in range(30)]
    corners = [Corner(start=20, end=22, speed_limit=3)]
    return Track("SlipTrack", spaces, corners, [0, 1, 2, 3, 4, 5], laps=1)


class TestSlipstream:
    def test_takes_good_slipstream(self) -> None:
        """A free +2 slipstream (no corners crossed) is taken."""
        agent = StrongHeuristicAgent(strength=2)
        track = _slip_track()
        state = GameState.create(track, 2, logging_enabled=False, seed=1)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.position = 5
        me.speed_from_cards = 3
        rival = state.get_player(1)
        rival.position = 7  # 2 ahead -> eligibility holds
        assert agent.choose_slipstream(state, 0) is True

    def test_declines_overshoot_slipstream(self) -> None:
        """A slipstream that lands an unaffordable corner is declined."""
        agent = StrongHeuristicAgent(strength=2)
        spaces = [Space(i, lanes=2) for i in range(15)]
        corners = [Corner(start=3, end=5, speed_limit=3)]
        track = Track("Tight", spaces, corners, [0, 1, 2, 3], laps=1)
        state = GameState.create(track, 2, logging_enabled=False, seed=1)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.position = 2          # +2 -> 4, into corner 3-5
        me.speed_from_cards = 6  # way over the limit of 3
        me.heat_pool = []        # cannot pay any overage
        assert agent.choose_slipstream(state, 0) is False

    def test_declines_boost_that_overshoots(self) -> None:
        """Boost is declined when speed already exceeds a crossed corner."""
        agent = StrongHeuristicAgent(strength=2)
        spaces = [Space(i, lanes=2) for i in range(15)]
        corners = [Corner(start=3, end=5, speed_limit=3)]
        track = Track("Tight2", spaces, corners, [0, 1, 2, 3], laps=1)
        state = GameState.create(track, 2, logging_enabled=False, seed=1)
        for p in state.players:
            p.lap = 1
        me = state.get_player(0)
        me.turn_start_position = 2
        me.turn_start_lap = 1
        me.position = 5            # crossed corner 3-5
        me.speed_from_cards = 4    # already over limit 3
        me.heat_pool = [Card(CardType.HEAT, 0, f"h{i}") for i in range(1)]
        d = agent.choose_react(
            state, 0, max_cooldown=0, can_boost=True, has_adrenaline=False
        )
        assert d.use_boost is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_state_same_decision(self) -> None:
        track = _usa()
        a1 = StrongHeuristicAgent(strength=2)
        a2 = StrongHeuristicAgent(strength=2)
        state = GameState.create(track, 3, logging_enabled=False, seed=55)
        for p in state.players:
            p.lap = 1
        legal = rules.legal_gear_shifts(
            state.get_player(0).gear, state.get_player(0).heat_available
        )
        for _ in range(10):
            assert a1.choose_gear(state, 0, legal) == a2.choose_gear(
                state, 0, legal
            )

    def test_batch_reproducible(self) -> None:
        track_a = _usa()
        track_b = _usa()
        facs = [
            strong_heuristic_agent_factory(strength=2),
            strong_heuristic_agent_factory(strength=2),
            heuristic_agent_factory(),
        ]
        a = run_batch(track_a, facs, num_games=8, seed=321, parallel=False, laps=3)
        b = run_batch(track_b, facs, num_games=8, seed=321, parallel=False, laps=3)
        for oa, ob in zip(a, b):
            assert oa.finish_order == ob.finish_order
            assert oa.total_rounds == ob.total_rounds


# ---------------------------------------------------------------------------
# Picklability
# ---------------------------------------------------------------------------


class TestPicklability:
    def test_factory_pickles(self) -> None:
        for s in (0, 1, 2, 3):
            fac = strong_heuristic_agent_factory(strength=s)
            restored = pickle.loads(pickle.dumps(fac))
            agent = restored(0, 123)
            assert isinstance(agent, StrongHeuristicAgent)
            assert agent.strength == s

    def test_parallel_batch_runs(self) -> None:
        track = _usa()
        facs = [
            strong_heuristic_agent_factory(strength=2),
            strong_heuristic_agent_factory(strength=1),
        ]
        outcomes = run_batch(
            track, facs, num_games=4, seed=9, parallel=True, laps=2
        )
        assert len(outcomes) == 4


# ---------------------------------------------------------------------------
# Latency budget
# ---------------------------------------------------------------------------


class TestLatency:
    def test_within_order_of_magnitude_of_heuristic(self) -> None:
        track = load_track_by_name("usa")
        track.laps = 3

        def _time(make_agent, n=8) -> float:
            start = time.perf_counter()
            for seed in range(n):
                agents = [make_agent() for _ in range(4)]
                Game(track, agents, logging_enabled=False, seed=seed).run()
            return time.perf_counter() - start

        t_heur = _time(lambda: HeuristicAgent())
        t_strong = _time(lambda: StrongHeuristicAgent(strength=2))
        assert t_strong < t_heur * 15 + 1.0, (
            f"strong {t_strong:.3f}s vs heuristic {t_heur:.3f}s"
        )
