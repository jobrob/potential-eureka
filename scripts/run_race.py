#!/usr/bin/env python3
"""Run a HEAT race with live play-by-play output.

Usage:
    python scripts/run_race.py                          # defaults: 2 heuristic agents, USA, 1 lap
    python scripts/run_race.py --players 4              # 4 heuristic agents
    python scripts/run_race.py --players 2 --laps 2     # 2-lap race
    python scripts/run_race.py --random 2 --heuristic 2 # 2 random vs 2 heuristic
    python scripts/run_race.py --log race.log           # also save to file
    python scripts/run_race.py --seed 42                # reproducible run
    python scripts/run_race.py --no-standings           # hide standings between rounds
    python scripts/run_race.py --names Max,Lewis        # custom player names
"""

from __future__ import annotations

import argparse
import random
import sys
import os

# Allow running from repo root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heat.tracks.loader import load_track_by_name
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.viewer import watch_game


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a HEAT board game race with play-by-play output."
    )
    parser.add_argument(
        "--track", default="silverstone",
        help="Track name to race on (default: silverstone)",
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
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log", default=None,
        help="Also save output to this file",
    )
    parser.add_argument(
        "--no-standings", action="store_true",
        help="Hide standings between rounds",
    )

    args = parser.parse_args()

    # Seed
    if args.seed is not None:
        random.seed(args.seed)

    # Load track
    try:
        track = load_track_by_name(args.track)
    except FileNotFoundError:
        print(f"Error: track '{args.track}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.laps is not None:
        track.laps = args.laps

    # Build agents
    n_heuristic = args.heuristic
    n_random = args.random

    if args.players is not None and n_heuristic is None:
        n_heuristic = args.players - n_random
    elif n_heuristic is None:
        n_heuristic = 2  # default

    if n_heuristic + n_random < 1:
        print("Error: need at least 1 player.", file=sys.stderr)
        sys.exit(1)
    if n_heuristic + n_random > 6:
        print("Error: max 6 players.", file=sys.stderr)
        sys.exit(1)

    default_names = ["Max", "Lewis", "Charles", "Lando", "Carlos", "Oscar"]
    agents = []
    names = []

    for i in range(n_heuristic):
        name = default_names[i] if i < len(default_names) else f"Heuristic-{i+1}"
        agents.append(HeuristicAgent(name=name))
        names.append(name)

    for i in range(n_random):
        idx = n_heuristic + i
        name = default_names[idx] if idx < len(default_names) else f"Random-{i+1}"
        seed = (args.seed or 0) + 100 + i
        agents.append(RandomAgent(seed=seed, name=name))
        names.append(name)

    # Override names if provided
    if args.names:
        custom_names = args.names.split(",")
        for i, n in enumerate(custom_names):
            if i < len(names):
                names[i] = n.strip()

    # Run
    result = watch_game(
        track,
        agents,
        player_names=names,
        show_standings=not args.no_standings,
        log_file=args.log,
    )


if __name__ == "__main__":
    main()
