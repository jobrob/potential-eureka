"""Tests for the simulation stats aggregation layer."""

from __future__ import annotations

import pytest

from heat.tracks.loader import load_track_by_name
from heat.simulation.runner import (
    GameOutcome,
    PlayerOutcome,
    heuristic_agent_factory,
    random_agent_factory,
    run_batch,
)
from heat.simulation.stats import aggregate_stats, format_summary


def _player(
    pid: int,
    *,
    agent_type: str = "RandomAgent",
    finish_position: int = 1,
    heat_remaining: int = 0,
) -> PlayerOutcome:
    return PlayerOutcome(
        player_id=pid,
        name=f"P{pid}",
        agent_type=agent_type,
        finish_position=finish_position,
        final_lap=1,
        final_position=0,
        heat_remaining=heat_remaining,
    )


def _game(game_index: int, players: list[PlayerOutcome]) -> GameOutcome:
    """Build a GameOutcome from players. Winner = the rank-1 player."""
    ranked = sorted(players, key=lambda p: p.finish_position)
    finish_order = tuple(p.player_id for p in ranked)
    winner = ranked[0]
    return GameOutcome(
        game_index=game_index,
        seed=None,
        num_players=len(players),
        winner_id=winner.player_id,
        winner_name=winner.name,
        finish_order=finish_order,
        total_rounds=10,
        players=tuple(sorted(players, key=lambda p: p.player_id)),
    )


class TestWinRateMath:
    def test_player0_wins_7_of_10(self) -> None:
        games = []
        for i in range(10):
            if i < 7:
                p0 = _player(0, finish_position=1)
                p1 = _player(1, finish_position=2)
            else:
                p0 = _player(0, finish_position=2)
                p1 = _player(1, finish_position=1)
            games.append(_game(i, [p0, p1]))

        stats = aggregate_stats(games, by="player_id")
        assert stats.per_agent["player_0"].win_rate == pytest.approx(0.7)
        assert stats.per_agent["player_1"].win_rate == pytest.approx(0.3)
        assert stats.win_counts_by_player_id[0] == 7
        assert stats.win_counts_by_player_id[1] == 3


class TestFinishDistribution:
    def test_distribution_and_avg_position(self) -> None:
        # player 0: ranks 1,1,2 -> avg 1.333; player 1: ranks 2,2,1 -> avg 1.667
        games = [
            _game(0, [_player(0, finish_position=1), _player(1, finish_position=2)]),
            _game(1, [_player(0, finish_position=1), _player(1, finish_position=2)]),
            _game(2, [_player(0, finish_position=2), _player(1, finish_position=1)]),
        ]
        stats = aggregate_stats(games, by="player_id")
        p0 = stats.per_agent["player_0"]
        assert p0.finish_position_counts == {1: 2, 2: 1}
        assert p0.avg_finish_position == pytest.approx(4 / 3)

        p1 = stats.per_agent["player_1"]
        assert p1.finish_position_counts == {1: 1, 2: 2}
        assert p1.avg_finish_position == pytest.approx(5 / 3)

    def test_missing_rank_defaults_to_zero(self) -> None:
        games = [_game(0, [_player(0, finish_position=1), _player(1, finish_position=2)])]
        stats = aggregate_stats(games, by="player_id")
        counts = stats.per_agent["player_0"].finish_position_counts
        # Rank 2 never occurred for player 0 -> .get defaults to 0.
        assert counts.get(2, 0) == 0


class TestRoundStats:
    def test_round_aggregates(self) -> None:
        games = []
        for i, rounds in enumerate([5, 10, 15]):
            g = _game(i, [_player(0, finish_position=1), _player(1, finish_position=2)])
            g = GameOutcome(
                game_index=g.game_index,
                seed=g.seed,
                num_players=g.num_players,
                winner_id=g.winner_id,
                winner_name=g.winner_name,
                finish_order=g.finish_order,
                total_rounds=rounds,
                players=g.players,
            )
            games.append(g)
        stats = aggregate_stats(games)
        assert stats.min_rounds == 5
        assert stats.max_rounds == 15
        assert stats.avg_rounds == pytest.approx(10.0)


class TestHeatRemaining:
    def test_avg_heat_remaining(self) -> None:
        games = [
            _game(0, [_player(0, heat_remaining=2), _player(1, heat_remaining=4)]),
            _game(1, [_player(0, heat_remaining=4), _player(1, heat_remaining=0)]),
        ]
        stats = aggregate_stats(games, by="player_id")
        assert stats.per_agent["player_0"].avg_heat_remaining == pytest.approx(3.0)
        assert stats.per_agent["player_1"].avg_heat_remaining == pytest.approx(2.0)


class TestGrouping:
    def test_agent_type_aggregates_seats(self) -> None:
        # Two heuristic seats (0,1) + two random seats (2,3).
        games = [
            _game(
                0,
                [
                    _player(0, agent_type="HeuristicAgent", finish_position=1),
                    _player(1, agent_type="HeuristicAgent", finish_position=2),
                    _player(2, agent_type="RandomAgent", finish_position=3),
                    _player(3, agent_type="RandomAgent", finish_position=4),
                ],
            )
        ]
        by_type = aggregate_stats(games, by="agent_type")
        assert set(by_type.per_agent) == {"HeuristicAgent", "RandomAgent"}
        assert by_type.per_agent["HeuristicAgent"].games_played == 2
        assert by_type.per_agent["RandomAgent"].games_played == 2

        by_seat = aggregate_stats(games, by="player_id")
        assert set(by_seat.per_agent) == {"player_0", "player_1", "player_2", "player_3"}
        assert by_seat.per_agent["player_0"].games_played == 1


class TestEdgeCases:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregate_stats([])

    def test_invalid_grouping_raises(self) -> None:
        games = [_game(0, [_player(0, finish_position=1)])]
        with pytest.raises(ValueError):
            aggregate_stats(games, by="nonsense")

    def test_single_outcome(self) -> None:
        games = [_game(0, [_player(0, finish_position=1), _player(1, finish_position=2)])]
        stats = aggregate_stats(games)
        assert stats.min_rounds == stats.max_rounds == int(stats.avg_rounds)
        assert stats.num_games == 1

    def test_single_player_win_rate_one(self) -> None:
        games = [_game(i, [_player(0, finish_position=1)]) for i in range(3)]
        stats = aggregate_stats(games, by="player_id")
        assert stats.per_agent["player_0"].win_rate == pytest.approx(1.0)


class TestFormatSummary:
    def test_smoke(self) -> None:
        games = [
            _game(
                0,
                [
                    _player(0, agent_type="HeuristicAgent", finish_position=1),
                    _player(1, agent_type="RandomAgent", finish_position=2),
                ],
            )
        ]
        text = format_summary(aggregate_stats(games))
        assert isinstance(text, str)
        assert text
        assert "HEAT Simulation Summary" in text
        assert "HeuristicAgent" in text
        assert "RandomAgent" in text


class TestIntegration:
    def test_real_batch_aggregates_consistently(self) -> None:
        track = load_track_by_name("usa")
        factories = [random_agent_factory() for _ in range(3)]
        outcomes = run_batch(track, factories, num_games=20, seed=11, parallel=False)
        stats = aggregate_stats(outcomes, by="agent_type")
        # Every game has exactly one winner.
        assert sum(s.wins for s in stats.per_agent.values()) == 20
        # Total player-games = games * players.
        assert sum(s.games_played for s in stats.per_agent.values()) == 20 * 3

    def test_heuristic_beats_random(self) -> None:
        track = load_track_by_name("usa")
        factories = [
            heuristic_agent_factory(),
            heuristic_agent_factory(),
            random_agent_factory(),
            random_agent_factory(),
        ]
        outcomes = run_batch(track, factories, num_games=60, seed=123, parallel=False)
        stats = aggregate_stats(outcomes, by="agent_type")
        heuristic = stats.per_agent["HeuristicAgent"]
        random_grp = stats.per_agent["RandomAgent"]

        # win_rate is per player-game; with 2 heuristic + 2 random seats the
        # theoretical max for any group is 0.5. The robust skill signal is that
        # heuristics take the large majority of games (share of wins > 0.55)
        # and clearly outperform random on win rate and avg finish position.
        heuristic_share = heuristic.wins / (heuristic.wins + random_grp.wins)
        assert heuristic_share > 0.55, (
            f"Heuristic share of wins {heuristic_share:.2f} not > 0.55"
        )
        assert heuristic.win_rate > random_grp.win_rate
        assert heuristic.avg_finish_position < random_grp.avg_finish_position
