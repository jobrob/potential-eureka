#!/usr/bin/env python
"""CLI to evaluate a trained HEAT RL checkpoint vs an opponent (Sprint 5d).

Runs a batch of games with an :class:`~heat.agents.ml_agent.MLAgent` in seat 0
against a chosen opponent pool and prints a win-rate table.

Usage (from the repo root, with the package importable)::

    PYTHONPATH=src python scripts/evaluate_ml.py \\
        --model checkpoints/heat_ppo --games 200 --track usa \\
        --opponents heuristic --players 2

Add ``--parallel`` to fan games across worker processes (off by default -- each
worker reloads the full SB3 model, which usually outweighs the speedup; see
``heat.ml.evaluate`` §6.5).
"""

from __future__ import annotations

import argparse
import sys

from heat.ml.evaluate import evaluate_ml
from heat.simulation.runner import (
    heuristic_agent_factory,
    random_agent_factory,
)

_OPPONENT_FACTORIES = {
    "heuristic": heuristic_agent_factory,
    "random": random_agent_factory,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HEAT RL checkpoint vs an opponent.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the SB3 checkpoint (.zip, with .meta.json sidecar).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to run (default: 100).",
    )
    parser.add_argument(
        "--track",
        default="usa",
        help="Track name to race on (default: usa).",
    )
    parser.add_argument(
        "--opponents",
        choices=sorted(_OPPONENT_FACTORIES),
        default="heuristic",
        help="Opponent agent type for the non-learner seats (default: heuristic).",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of seats per game, 2..6 (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic runs (default: 0).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run games across worker processes (default: sequential).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar if installed.",
    )
    return parser.parse_args(argv)


def _format_table(per_agent: dict, num_games: int, num_players: int) -> str:
    """Render the per-agent win-rate table, learner-first then by win rate."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("HEAT ML Evaluation")
    lines.append("=" * 60)
    lines.append(f"Games:   {num_games}")
    lines.append(f"Players: {num_players}")
    lines.append("-" * 60)
    lines.append(
        f"{'Agent':<18}{'Games':>7}{'Wins':>7}{'Win%':>8}"
        f"{'AvgPos':>8}{'AvgHeat':>9}"
    )
    lines.append("-" * 60)

    # Learner ("MLAgent") first for a readable headline, then by win rate desc.
    def sort_key(key: str) -> tuple[int, float]:
        is_ml = 0 if key == "MLAgent" else 1
        return (is_ml, -per_agent[key].win_rate)

    for key in sorted(per_agent, key=sort_key):
        s = per_agent[key]
        lines.append(
            f"{s.agent_type:<18}{s.games_played:>7}{s.wins:>7}"
            f"{s.win_rate * 100:>7.1f}%"
            f"{s.avg_finish_position:>8.2f}{s.avg_heat_remaining:>9.2f}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    per_agent = evaluate_ml(
        args.model,
        opponent_factory=_OPPONENT_FACTORIES[args.opponents](),
        num_games=args.games,
        num_players=args.players,
        track=args.track,
        seed=args.seed,
        parallel=args.parallel,
        progress=args.progress,
    )

    print(_format_table(per_agent, args.games, args.players))
    return 0


if __name__ == "__main__":
    sys.exit(main())
