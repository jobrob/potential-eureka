"""Tests for the batch simulation runner."""

from __future__ import annotations

import pickle

import pytest

from heat.tracks.loader import load_track_by_name
from heat.simulation.runner import (
    GameOutcome,
    PlayerOutcome,
    heuristic_agent_factory,
    random_agent_factory,
    run_batch,
    run_single_game,
)


@pytest.fixture
def track():
    return load_track_by_name("usa")


def _assert_valid_outcome(outcome: GameOutcome, num_players: int) -> None:
    """Shared structural invariants for a single GameOutcome."""
    assert isinstance(outcome, GameOutcome)
    assert outcome.num_players == num_players
    assert len(outcome.players) == num_players
    # Exactly one winner, consistent with finish_order.
    assert outcome.winner_id == outcome.finish_order[0]
    assert set(outcome.finish_order) == set(range(num_players))
    assert len(outcome.finish_order) == num_players
    assert outcome.total_rounds > 0
    # Players ordered by player_id.
    assert [p.player_id for p in outcome.players] == list(range(num_players))
    for po in outcome.players:
        assert isinstance(po, PlayerOutcome)
        assert 0 <= po.heat_remaining <= 6
        assert 1 <= po.finish_position <= num_players


class TestRunBatchStructure:
    def test_returns_exact_count_ordered_by_index(self, track) -> None:
        factories = [random_agent_factory(), random_agent_factory()]
        outcomes = run_batch(track, factories, num_games=8, seed=1, parallel=False)
        assert len(outcomes) == 8
        assert [o.game_index for o in outcomes] == list(range(8))

    def test_each_outcome_has_one_winner(self, track) -> None:
        factories = [random_agent_factory() for _ in range(3)]
        outcomes = run_batch(track, factories, num_games=5, seed=2, parallel=False)
        for outcome in outcomes:
            _assert_valid_outcome(outcome, num_players=3)

    def test_heat_remaining_in_range(self, track) -> None:
        factories = [random_agent_factory() for _ in range(4)]
        outcomes = run_batch(track, factories, num_games=4, seed=3, parallel=False)
        for outcome in outcomes:
            for po in outcome.players:
                assert 0 <= po.heat_remaining <= 6


class TestDeterminism:
    def test_sequential_determinism(self, track) -> None:
        factories = [random_agent_factory() for _ in range(3)]
        track2 = load_track_by_name("usa")
        a = run_batch(track, factories, num_games=6, seed=42, parallel=False)
        b = run_batch(track2, factories, num_games=6, seed=42, parallel=False)
        for oa, ob in zip(a, b):
            assert oa.finish_order == ob.finish_order
            assert oa.total_rounds == ob.total_rounds
            assert oa.players == ob.players


class TestEdgeCases:
    def test_single_game_returns_one(self, track) -> None:
        factories = [random_agent_factory(), random_agent_factory()]
        outcomes = run_batch(track, factories, num_games=1, seed=5, parallel=False)
        assert len(outcomes) == 1
        _assert_valid_outcome(outcomes[0], num_players=2)

    def test_single_game_with_parallel_flag(self, track) -> None:
        # num_games == 1 must short-circuit to sequential, never spawn a pool.
        factories = [random_agent_factory(), random_agent_factory()]
        outcomes = run_batch(track, factories, num_games=1, seed=5, parallel=True)
        assert len(outcomes) == 1
        _assert_valid_outcome(outcomes[0], num_players=2)

    def test_single_player_wins(self, track) -> None:
        factories = [heuristic_agent_factory()]
        outcomes = run_batch(track, factories, num_games=2, seed=9, parallel=False)
        for outcome in outcomes:
            assert outcome.num_players == 1
            assert outcome.winner_id == 0
            assert outcome.finish_order == (0,)

    def test_invalid_factory_count_raises(self, track) -> None:
        with pytest.raises(ValueError):
            run_batch(track, [], num_games=1, parallel=False)
        with pytest.raises(ValueError):
            run_batch(
                track,
                [random_agent_factory() for _ in range(7)],
                num_games=1,
                parallel=False,
            )

    def test_invalid_num_games_raises(self, track) -> None:
        factories = [random_agent_factory()]
        with pytest.raises(ValueError):
            run_batch(track, factories, num_games=0, parallel=False)

    def test_unseeded_records_none_seed(self, track) -> None:
        factories = [random_agent_factory(), random_agent_factory()]
        outcomes = run_batch(track, factories, num_games=2, seed=None, parallel=False)
        assert all(o.seed is None for o in outcomes)


class TestPicklableFactories:
    def test_random_factory_roundtrip(self) -> None:
        factory = random_agent_factory(name="R")
        restored = pickle.loads(pickle.dumps(factory))
        agent = restored(0, 123)
        assert type(agent).__name__ == "RandomAgent"

    def test_heuristic_factory_roundtrip(self) -> None:
        factory = heuristic_agent_factory(name="H")
        restored = pickle.loads(pickle.dumps(factory))
        agent = restored(0, None)
        assert type(agent).__name__ == "HeuristicAgent"

    def test_outcome_is_picklable(self, track) -> None:
        factories = [random_agent_factory(), random_agent_factory()]
        outcome = run_single_game(track, factories, 0, base_seed=1)
        restored = pickle.loads(pickle.dumps(outcome))
        assert restored == outcome


class TestAgentType:
    def test_agent_type_recorded(self, track) -> None:
        factories = [random_agent_factory(), heuristic_agent_factory()]
        outcome = run_single_game(track, factories, 0, base_seed=1)
        types = {po.player_id: po.agent_type for po in outcome.players}
        assert types[0] == "RandomAgent"
        assert types[1] == "HeuristicAgent"


# Keep parallel game counts small so the spawn/pickle overhead stays cheap.
_PAR_GAMES = 6


class TestParallel:
    def test_parallel_determinism(self, track) -> None:
        factories = [random_agent_factory() for _ in range(3)]
        track2 = load_track_by_name("usa")
        a = run_batch(track, factories, num_games=_PAR_GAMES, seed=42, parallel=True)
        b = run_batch(track2, factories, num_games=_PAR_GAMES, seed=42, parallel=True)
        for oa, ob in zip(a, b):
            assert oa.finish_order == ob.finish_order
            assert oa.total_rounds == ob.total_rounds

    def test_parallel_returns_ordered(self, track) -> None:
        factories = [random_agent_factory(), random_agent_factory()]
        outcomes = run_batch(
            track, factories, num_games=_PAR_GAMES, seed=1, parallel=True
        )
        assert [o.game_index for o in outcomes] == list(range(_PAR_GAMES))
        for outcome in outcomes:
            _assert_valid_outcome(outcome, num_players=2)


class TestSequentialEqualsParallel:
    """Core guarantee: same seed -> identical per-game results either way."""

    def test_equivalence(self, track) -> None:
        factories = [random_agent_factory() for _ in range(3)]
        track2 = load_track_by_name("usa")
        seq = run_batch(track, factories, num_games=_PAR_GAMES, seed=7, parallel=False)
        par = run_batch(track2, factories, num_games=_PAR_GAMES, seed=7, parallel=True)
        assert len(seq) == len(par) == _PAR_GAMES
        for s, p in zip(seq, par):
            assert s.game_index == p.game_index
            assert s.finish_order == p.finish_order
            assert s.total_rounds == p.total_rounds


def _bad_lambda_factory():
    """A lambda factory: not picklable, must trigger sequential fallback."""
    return lambda player_id, seed: random_agent_factory()(player_id, seed)


class TestFallback:
    def test_lambda_factory_falls_back(self, track) -> None:
        # Lambdas cannot be pickled; parallel=True must transparently fall
        # back to the sequential path and still return valid outcomes.
        factories = [_bad_lambda_factory(), _bad_lambda_factory()]
        outcomes = run_batch(track, factories, num_games=4, seed=3, parallel=True)
        assert len(outcomes) == 4
        for outcome in outcomes:
            _assert_valid_outcome(outcome, num_players=2)
