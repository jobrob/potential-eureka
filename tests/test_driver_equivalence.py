"""Equivalence test: Game.run_round (pump over driver) vs golden logs (Step D2).

The golden logs in tests/data/golden_event_logs.json were captured from the
pre-refactor monolithic run_round. After reimplementing run_round as a thin
pump over run_round_driver, a seeded game must reproduce byte-identical event
logs, finish orders, and round counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from heat.engine.game import Game
from heat.agents.random_agent import RandomAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.models.track import Corner, Space, Track

_GOLDEN_PATH = Path(__file__).parent / "data" / "golden_event_logs.json"


def _track(laps: int = 2) -> Track:
    spaces = [Space(i, lanes=2) for i in range(16)]
    corners = [Corner(start=5, end=6, speed_limit=3),
               Corner(start=12, end=13, speed_limit=2)]
    return Track("GoldTrack", spaces, corners, [0, 1, 2, 3], laps=laps)


def _sig(events) -> list[list]:
    return [
        [e.round_num, e.phase.value, e.player_id, e.event_type, repr(e.data)]
        for e in events
    ]


def _load_golden() -> dict:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


_RANDOM_SEEDS = [1, 7, 42, 100, 2024]
_HEUR_SEEDS = [3, 55, 999]


@pytest.mark.parametrize("seed", _RANDOM_SEEDS)
def test_random_game_matches_golden(seed: int) -> None:
    golden = _load_golden()["random_" + str(seed)]
    agents = [RandomAgent(seed=seed * 10 + i) for i in range(4)]
    game = Game(_track(), agents, seed=seed)
    result = game.run()
    assert result.finish_order == golden["finish_order"]
    assert result.total_rounds == golden["total_rounds"]
    assert _sig(result.event_log) == golden["events"]


@pytest.mark.parametrize("seed", _HEUR_SEEDS)
def test_heuristic_game_matches_golden(seed: int) -> None:
    golden = _load_golden()["heur_" + str(seed)]
    agents = [HeuristicAgent() for _ in range(4)]
    game = Game(_track(), agents, seed=seed)
    result = game.run()
    assert result.finish_order == golden["finish_order"]
    assert result.total_rounds == golden["total_rounds"]
    assert _sig(result.event_log) == golden["events"]
