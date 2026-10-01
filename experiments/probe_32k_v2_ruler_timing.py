#!/usr/bin/env python
"""Short V2 ruler timing probe for the 32K learning-quality campaign.

Design Q07: the old 75-minute figure was never timed on V2 rulers. Recalibrate
expected wall-clock with one short timing run before a real campaign.

This probe races tiny cells with HeuristicWeakV2 / HeuristicRepairedV2 /
StaticSearchV2 only (no historical G0002 load required). It does not allocate
G0005 and is not a skill measurement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from experiments.eval_32k_learning_quality import TRACK_BASE
from experiments.run_32k_learning_quality import QUESTION_ID
from heat.agents.static_v2 import (
    HeuristicV2Agent,
    RepairedHeuristicV2Agent,
    StaticSearchV2Agent,
)
from heat.ml.selfplay.eval_harness import _play_race, race_coordinates
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.ppo import A0Config
from heat.tracks.generator import generate_track

RULERS = {
    "heuristic_weak_v2": HeuristicV2Agent,
    "heuristic_repaired_v2": RepairedHeuristicV2Agent,
    "static_search_v2": StaticSearchV2Agent,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a bounded V2 timing probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--seats", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--track-base", type=int, default=TRACK_BASE)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/32k_learning_quality/v2_ruler_timing.json"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned probe without racing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Time one tiny V2 ruler grid and extrapolate promotion-scale cost."""
    args = _parse_args(argv)
    seats = tuple(args.seats)
    plan = {
        "question_id": QUESTION_ID,
        "g0005_allocated": False,
        "games": args.games,
        "tracks": args.tracks,
        "seats": list(seats),
        "rulers": list(RULERS),
        "track_base": args.track_base,
        "note": "Timing probe only; not a skill result.",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return 0

    tracks = [generate_track(args.track_base + index) for index in range(args.tracks)]
    # Tiny random policy: measures ruler planning cost under a learned opponent
    # seat without depending on registered G0002 checkpoint bytes.
    policy = build_policy(A0Config(hidden_sizes=(16,), device="cpu"))
    device = torch.device("cpu")
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for ruler_id, factory in RULERS.items():
        for seat_count in seats:
            coordinates = race_coordinates(args.games, len(tracks), seat_count)
            cell_started = perf_counter()
            for race_index, (track_index, focal_seat, _repeat) in enumerate(coordinates):
                _play_race(
                    policy,
                    factory,
                    tracks[track_index],
                    focal_seat,
                    seat_count,
                    7_000_000 + race_index,
                    device,
                )
            elapsed = perf_counter() - cell_started
            rows.append(
                {
                    "ruler": ruler_id,
                    "seats": seat_count,
                    "games": args.games,
                    "elapsed_seconds": elapsed,
                    "seconds_per_race": elapsed / args.games,
                }
            )
            print(
                f"{ruler_id} seats={seat_count} "
                f"{elapsed / args.games:.3f}s/race ({args.games} games)",
                flush=True,
            )

    seconds_per_race = [row["seconds_per_race"] for row in rows]
    mean_spr = sum(seconds_per_race) / len(seconds_per_race)
    # Promotion: 6 contenders × 4 rulers × 3 seats × 200 races = 14,400 races
    # when candidate+G0002 are both present (design Q04).
    promotion_races = 6 * 4 * 3 * 200
    milestone_races = 12 * 2 * 3 * 40
    report = {
        **plan,
        "cells": rows,
        "mean_seconds_per_race": mean_spr,
        "extrapolation": {
            "promotion_races_design": promotion_races,
            "milestone_races_design": milestone_races,
            "promotion_hours_estimate": promotion_races * mean_spr / 3600.0,
            "milestone_hours_estimate": milestone_races * mean_spr / 3600.0,
            "caveat": (
                "Estimate from a tiny random-policy probe; StaticSearchV2 dominates. "
                "Re-time on the campaign machine before trusting the hard caps."
            ),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    _atomic_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    from _runlog import run_main

    run_main("probe_32k_v2_ruler_timing", main)
