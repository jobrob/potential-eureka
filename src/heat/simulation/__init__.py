"""Batch simulation layer for the HEAT board game.

Provides a headless batch runner (:func:`run_batch`) that plays many games
with ``logging_enabled=False`` and returns picklable :class:`GameOutcome`
records, plus a pure aggregation layer (:func:`aggregate_stats`) that turns a
list of outcomes into summary statistics.

Usage:
    >>> from heat.tracks.loader import load_track_by_name
    >>> from heat.simulation import run_batch, random_agent_factory, aggregate_stats
    >>> track = load_track_by_name("usa")
    >>> factories = [random_agent_factory(), random_agent_factory()]
    >>> outcomes = run_batch(track, factories, num_games=10, seed=42)
    >>> stats = aggregate_stats(outcomes)
"""

from __future__ import annotations

from heat.simulation.runner import (
    AgentFactory,
    GameOutcome,
    PlayerOutcome,
    heuristic_agent_factory,
    random_agent_factory,
    static_search_agent_factory,
    run_batch,
    run_single_game,
)
from heat.simulation.stats import (
    AgentStats,
    SimulationStats,
    aggregate_stats,
    format_summary,
)

__all__ = [
    "AgentFactory",
    "GameOutcome",
    "PlayerOutcome",
    "run_batch",
    "run_single_game",
    "random_agent_factory",
    "heuristic_agent_factory",
    "static_search_agent_factory",
    "AgentStats",
    "SimulationStats",
    "aggregate_stats",
    "format_summary",
]
