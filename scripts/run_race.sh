#!/bin/bash
# Run a HEAT board game race with play-by-play output.
# Double-click from file explorer or run from terminal.
# Logs are saved to scripts/logs/ with a timestamp.
#
# Usage:
#   ./scripts/run_race.sh                                  # defaults: 2 heuristic, USA, 1 lap
#   ./scripts/run_race.sh --players 4 --laps 2             # 4-player 2-lap race
#   ./scripts/run_race.sh --heuristic 2 --random 2         # heuristic vs random
#   ./scripts/run_race.sh --seed 42                        # reproducible run
#   ./scripts/run_race.sh --help                           # full options

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/race_${TIMESTAMP}.log"

echo "================================================"
echo "  HEAT Race Runner"
echo "  Log will be saved to: $LOG_FILE"
echo "================================================"
echo ""

PYTHONPATH="$REPO_ROOT/src" python "$SCRIPT_DIR/run_race.py" --log "$LOG_FILE" "$@"

echo ""
echo "================================================"
echo "  Log saved to: $LOG_FILE"
echo "================================================"
echo ""
echo "Press any key to exit..."
read -n 1 -s
