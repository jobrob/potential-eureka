#!/usr/bin/env python3
"""Run a batch of HEAT games and print aggregate statistics.

Usage:
    python scripts/run_simulation.py                              # 100 games, 2 heuristic, USA
    python scripts/run_simulation.py --games 1000 --random 4      # 1000 games, 4 random
    python scripts/run_simulation.py --heuristic 2 --random 2     # 2 heuristic vs 2 random
    python scripts/run_simulation.py --players 4                  # 4 heuristic agents
    python scripts/run_simulation.py --games 500 --seed 0         # reproducible batch
    python scripts/run_simulation.py --no-parallel                # force sequential
    python scripts/run_simulation.py --group player_id            # group stats by seat
    python scripts/run_simulation.py --progress                   # show a progress bar
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from repo root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heat.tracks.loader import load_track_by_name
from heat.simulation.runner import (
    heuristic_agent_factory,
    random_agent_factory,
    run_batch,
)
from heat.simulation.stats import aggregate_stats, format_summary

_DEFAULT_NAMES = ["Max", "Lewis", "Charles", "Lando", "Carlos", "Oscar"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a batch of HEAT games and print aggregate stats."
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games to simulate (default: 100)",
    )
    parser.add_argument(
        "--track", default="usa",
        help="Track name to race on (default: usa)",
    )
    parser.add_argument(
        "--laps", type=int, default=None,
        help="Number of laps (default: track default)",
    )
    parser.add_argument(
        "--players", type=int, default=None,
        help="Total players, all heuristic (shortcut for --heuristic N)",
    )
    parser.add_argument(
        "--heuristic", type=int, default=None,
        help="Number of heuristic agents",
    )
    parser.add_argument(
        "--random", type=int, default=0,
        help="Number of random agents (default: 0)",
    )
    parser.add_argument(
        "--names",
        help="Comma-separated player names (e.g. Max,Lewis,Charles)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base seed for deterministic, order-independent runs",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Force sequential execution (no process pool)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max worker processes for the parallel path (default: auto)",
    )
    parser.add_argument(
        "--group", choices=("agent_type", "player_id"), default="agent_type",
        help="Grouping key for stats (default: agent_type)",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Show a tqdm progress bar (if tqdm is installed)",
    )
    return parser


def _resolve_seat_counts(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve (n_heuristic, n_random), mirroring scripts/run_race.py logic.

    Exits with code 1 (stderr message) if the total is outside 1..6.
    """
    n_heuristic = args.heuristic
    n_random = args.random

    if args.players is not None and n_heuristic is None:
        n_heuristic = args.players - n_random
    elif n_heuristic is None:
        n_heuristic = 2  # default

    total = n_heuristic + n_random
    if total < 1:
        print("Error: need at least 1 player.", file=sys.stderr)
        sys.exit(1)
    if total > 6:
        print("Error: max 6 players.", file=sys.stderr)
        sys.exit(1)
    if n_heuristic < 0 or n_random < 0:
        print("Error: agent counts cannot be negative.", file=sys.stderr)
        sys.exit(1)

    return n_heuristic, n_random


def _build_factories(n_heuristic: int, n_random: int, names_arg: str | None):
    """Build a picklable agent factory per seat, applying optional names."""
    names: list[str] = []
    for i in range(n_heuristic + n_random):
        names.append(_DEFAULT_NAMES[i] if i < len(_DEFAULT_NAMES) else f"P{i + 1}")

    if names_arg:
        for i, n in enumerate(names_arg.split(",")):
            if i < len(names):
                names[i] = n.strip()

    factories = []
    for i in range(n_heuristic):
        factories.append(heuristic_agent_factory(name=names[i]))
    for i in range(n_random):
        idx = n_heuristic + i
        factories.append(random_agent_factory(name=names[idx]))
    return factories


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.games < 1:
        print("Error: --games must be >= 1.", file=sys.stderr)
        sys.exit(1)

    n_heuristic, n_random = _resolve_seat_counts(args)

    try:
        track = load_track_by_name(args.track)
    except FileNotFoundError:
        print(f"Error: track '{args.track}' not found.", file=sys.stderr)
        sys.exit(1)

    factories = _build_factories(n_heuristic, n_random, args.names)

    start = time.perf_counter()
    outcomes = run_batch(
        track,
        factories,
        args.games,
        parallel=not args.no_parallel,
        max_workers=args.workers,
        seed=args.seed,
        laps=args.laps,
        progress=args.progress,
    )
    elapsed = time.perf_counter() - start

    stats = aggregate_stats(outcomes, by=args.group)
    print(format_summary(stats))

    games_per_sec = args.games / elapsed if elapsed > 0 else float("inf")
    ms_per_game = (elapsed / args.games) * 1000 if args.games else 0.0
    print(
        f"Ran {args.games} games in {elapsed:.2f}s "
        f"({games_per_sec:.0f} games/s, {ms_per_game:.2f} ms/game)"
    )


if __name__ == "__main__":
    main()
