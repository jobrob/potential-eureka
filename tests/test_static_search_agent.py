"""Focused contract tests for the frozen StaticSearchV1 benchmark."""

from __future__ import annotations

import pickle

from heat.agents import StaticSearchAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.engine.game import Game
from heat.simulation.runner import static_search_agent_factory
from heat.tracks.generator import generate_track


def test_static_search_v1_configuration_is_frozen() -> None:
    """The versioned ruler always builds the H2-selected configuration."""
    agent = StaticSearchAgent()
    assert agent.name == "StaticSearchV1"
    assert agent.horizon == 1
    assert agent.n_determinizations == 2
    assert agent.determinize_hidden is True
    assert agent.top_k == 6
    assert agent.leaf_value == "progress"
    assert isinstance(agent.rollout_policy, HeuristicAgent)


def test_static_search_factory_pickles_with_stable_name() -> None:
    """The factory can cross process boundaries without configuration drift."""
    restored = pickle.loads(pickle.dumps(static_search_agent_factory()))
    agent = restored(3, 99)
    assert isinstance(agent, StaticSearchAgent)
    assert agent.name == "StaticSearchV1-3"
    assert agent.seed is None


def test_static_search_replays_a_full_game_deterministically() -> None:
    """Equal track and game seeds produce an identical legal race trace."""
    track = generate_track(713_999)

    def run() -> tuple[list[int], list[object]]:
        game = Game(
            track,
            [StaticSearchAgent(), HeuristicAgent()],
            logging_enabled=True,
            seed=123,
        )
        result = game.run()
        return result.finish_order, list(result.event_log)

    first = run()
    second = run()
    assert first == second
    assert len(first[0]) == 2
