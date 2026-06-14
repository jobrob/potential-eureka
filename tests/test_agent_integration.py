"""Integration tests: agents playing full games on the USA track."""

from __future__ import annotations

import random

import pytest

from heat.tracks.loader import load_track_by_name
from heat.engine.game import Game
from heat.agents.random_agent import RandomAgent
from heat.agents.heuristic_agent import HeuristicAgent


class TestRandomAgentFullGame:
    """Run multiple full games with RandomAgents to verify no crashes."""

    def test_50_games_all_terminate(self) -> None:
        """50 games with 4 RandomAgents on USA track should all finish."""
        for i in range(50):
            random.seed(i * 100)
            track = load_track_by_name("usa")
            agents = [RandomAgent(seed=i * 10 + j) for j in range(4)]
            game = Game(track, agents, logging_enabled=False)
            result = game.run()

            assert game.is_over, f"Game {i} did not terminate"
            assert result.total_rounds > 0, f"Game {i} had 0 rounds"
            assert len(result.finish_order) == 4, (
                f"Game {i}: expected 4 finishers, got {len(result.finish_order)}"
            )
            assert set(result.finish_order) == {0, 1, 2, 3}, (
                f"Game {i}: not all players finished"
            )


class TestHeuristicAgentFullGame:
    """Run multiple full games with HeuristicAgents to verify no crashes."""

    def test_50_games_all_terminate(self) -> None:
        """50 games with 4 HeuristicAgents on USA track should all finish."""
        for i in range(50):
            random.seed(i * 200)
            track = load_track_by_name("usa")
            agents = [HeuristicAgent(name=f"H{j}") for j in range(4)]
            game = Game(track, agents, logging_enabled=False)
            result = game.run()

            assert game.is_over, f"Game {i} did not terminate"
            assert result.total_rounds > 0, f"Game {i} had 0 rounds"
            assert len(result.finish_order) == 4, (
                f"Game {i}: expected 4 finishers, got {len(result.finish_order)}"
            )
            assert set(result.finish_order) == {0, 1, 2, 3}, (
                f"Game {i}: not all players finished"
            )


class TestHeuristicVsRandom:
    """Heuristic agents should outperform random agents."""

    def test_heuristic_wins_majority(self) -> None:
        """Over 200 games, heuristic agents (players 0,1) should win >55%."""
        heuristic_wins = 0
        total = 200

        for i in range(total):
            random.seed(i * 1000)
            track = load_track_by_name("usa")
            agents = [
                HeuristicAgent(name="H0"),
                HeuristicAgent(name="H1"),
                RandomAgent(seed=i * 10, name="R0"),
                RandomAgent(seed=i * 10 + 1, name="R1"),
            ]
            game = Game(track, agents, logging_enabled=False)
            result = game.run()

            winner = result.finish_order[0]
            if winner in (0, 1):  # heuristic agents
                heuristic_wins += 1

        win_rate = heuristic_wins / total
        assert win_rate > 0.60, (
            f"Heuristic win rate {win_rate:.1%} is below 60% threshold "
            f"({heuristic_wins}/{total})"
        )
