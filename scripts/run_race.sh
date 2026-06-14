#!/bin/bash
# Run a HEAT board game race with play-by-play output.
#
# Usage:
#   ./scripts/run_race.sh                                  # defaults: 2 heuristic, USA, 1 lap
#   ./scripts/run_race.sh --players 4 --laps 2             # 4-player 2-lap race
#   ./scripts/run_race.sh --heuristic 2 --random 2         # heuristic vs random
#   ./scripts/run_race.sh --seed 42 --log race.log         # reproducible, saved to file
#   ./scripts/run_race.sh --help                           # full options

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHONPATH="$REPO_ROOT/src" python "$SCRIPT_DIR/run_race.py" "$@"
