"""Parity / determinism tests for the game.py legal-action refactor (Step C2).

After lifting React / slipstream / discard legality into rules.legal_*,
game behavior must be unchanged. We assert that a seeded game produces an
identical event log and finish order across two independent runs (the
refactored decision-collection path is fully deterministic given a seed).
"""

from __future__ import annotations

from heat.engine.game import Game
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.models.track import Corner, Space, Track


def _track(laps: int = 2) -> Track:
    spaces = [Space(i, lanes=2) for i in range(16)]
    corners = [Corner(start=5, end=6, speed_limit=3),
               Corner(start=12, end=13, speed_limit=2)]
    return Track("ParityTrack", spaces, corners, [0, 1, 2, 3], laps=laps)


def _event_signature(events) -> list[tuple]:
    return [
        (e.round_num, e.phase, e.player_id, e.event_type, repr(e.data))
        for e in events
    ]


class TestSeededParity:
    def test_random_agents_event_log_identical(self) -> None:
        def run() -> tuple[list[int], int, list[tuple]]:
            agents = [RandomAgent(seed=10 + i) for i in range(4)]
            game = Game(_track(), agents, seed=2024)
            result = game.run()
            return (
                result.finish_order,
                result.total_rounds,
                _event_signature(result.event_log),
            )

        a = run()
        b = run()
        assert a[0] == b[0]
        assert a[1] == b[1]
        assert a[2] == b[2]

    def test_heuristic_agents_finish_order_identical(self) -> None:
        def run() -> tuple[list[int], int]:
            agents = [HeuristicAgent() for _ in range(4)]
            game = Game(_track(), agents, seed=555)
            result = game.run()
            return result.finish_order, result.total_rounds

        assert run() == run()
