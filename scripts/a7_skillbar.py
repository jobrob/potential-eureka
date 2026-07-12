#!/usr/bin/env python
"""A7 skill-bar report for a Direction-A policy or a heuristic baseline (§4.3).

Runs the :func:`heat.ml.selfplay.eval_harness.evaluate_policy` grid (opponents x
seat counts x track splits) for a saved checkpoint, or for a heuristic-only
baseline (``strong`` / ``weak`` / ``random`` in the policy seat -- the §5 G1
baseline path), and prints the markdown report. Optionally adds a fixed-anchor
self-improvement probe row (``--anchor``).

Usage (from the repo root)::

    # End-to-end policy report on a saved checkpoint
    PYTHONPATH=src python scripts/a7_skillbar.py --checkpoint final.pt

    # Heuristic-only baseline grid (G1): strong in the policy seat vs weak field
    PYTHONPATH=src python scripts/a7_skillbar.py --baseline strong --games 200

    # Add the fixed-anchor probe (G3) against an early checkpoint
    PYTHONPATH=src python scripts/a7_skillbar.py --checkpoint final.pt \\
        --anchor anchor.pt --anchor-games 200

``--bar OPP[:MARGIN]`` (repeatable) asserts the report clears the skill bar for
that opponent (Wilson-LB above chance by MARGIN at every seat count); the script
exits non-zero if any requested bar fails. With no ``--bar`` the script is
report-only (always exit 0 on a clean run).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Callable

import torch

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.models.track import Track
from heat.ml.selfplay.checkpoint import load_policy
from heat.ml.selfplay.eval_harness import (
    EvalReport,
    evaluate_policy,
    evaluate_vs_anchor,
    held_out_tracks,
)
from heat.ml.selfplay.policy import PPOPolicy
from heat.ml.selfplay.snapshots import SnapshotAgent
from heat.ml.selfplay.tiny_heat import tiny_heat_track

_OPP_FACTORIES: dict[str, Callable[[], BaseAgent]] = {
    "weak": HeuristicAgent,
    "strong": StrongHeuristicAgent,
    "random": RandomAgent,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A7 skill-bar report.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                     help="Saved policy checkpoint to score (save_policy blob).")
    src.add_argument("--baseline", choices=["strong", "weak", "random"],
                     default=None,
                     help="Score a heuristic in the policy seat instead (G1).")
    parser.add_argument("--opponents", nargs="+", choices=["weak", "strong"],
                        default=["weak", "strong"],
                        help="Opponent fields to grid (default: weak strong).")
    parser.add_argument("--seats", type=int, nargs="+", default=[2, 3, 4, 6],
                        help="Seat counts to sweep (default: 2 3 4 6).")
    parser.add_argument("--splits", nargs="+", choices=["tiny", "heldout"],
                        default=["tiny", "heldout"],
                        help="Track splits (default: tiny heldout).")
    parser.add_argument("--heldout-n", type=int, default=20,
                        help="Held-out tracks to cycle (default: 20).")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per grid cell (default: 50).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed (default: 0).")
    parser.add_argument("--anchor", type=str, default=None, metavar="PATH",
                        help="Optional early checkpoint: add the fixed-anchor "
                             "self-improvement probe row.")
    parser.add_argument("--anchor-players", type=int, default=2,
                        help="Seats for the anchor probe (default: 2).")
    parser.add_argument("--anchor-games", type=int, default=200,
                        help="Games for the anchor probe (default: 200).")
    parser.add_argument("--bar", action="append", default=[], metavar="OPP[:M]",
                        help="Assert clears_bar(OPP, M) at every seat; repeatable. "
                             "Exit non-zero if it fails.")
    return parser.parse_args(argv)


def _resolve_policy(args: argparse.Namespace) -> PPOPolicy | BaseAgent:
    if args.checkpoint is not None:
        return load_policy(args.checkpoint)
    return _OPP_FACTORIES[args.baseline]()


def _resolve_splits(args: argparse.Namespace) -> dict[str, Track | list[Track]]:
    splits: dict[str, Track | list[Track]] = {}
    for name in args.splits:
        if name == "tiny":
            splits["tiny"] = tiny_heat_track()
        else:
            splits["heldout"] = held_out_tracks(n=args.heldout_n)
    return splits


def _check_bars(report: EvalReport, bars: list[str]) -> bool:
    """Print each requested bar's verdict; return True iff all pass."""
    all_pass = True
    print()
    print("Skill-bar predicates:")
    for spec in bars:
        opp, _, margin_s = spec.partition(":")
        margin = float(margin_s) if margin_s else 0.0
        ok = report.clears_bar(opp, margin)
        all_pass = all_pass and ok
        print(f"  clears_bar({opp!r}, {margin}) = {'PASS' if ok else 'FAIL'}")
    return all_pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    policy = _resolve_policy(args)
    opponents = {name: _OPP_FACTORIES[name] for name in args.opponents}
    splits = _resolve_splits(args)

    report = evaluate_policy(
        policy,
        opponents=opponents,
        seat_counts=tuple(args.seats),
        splits=splits,
        games_per_cell=args.games,
        seed=args.seed,
        device=torch.device("cpu"),
    )

    if args.anchor is not None:
        anchor_policy = load_policy(args.anchor)
        anchor = SnapshotAgent(anchor_policy, name="anchor")
        cell = evaluate_vs_anchor(
            policy,
            anchor,
            num_players=args.anchor_players,
            track=tiny_heat_track(),
            games=args.anchor_games,
            seed=args.seed,
            device=torch.device("cpu"),
        )
        report = dataclasses.replace(report, anchor=cell)

    print(report.to_markdown())

    if args.bar:
        return 0 if _check_bars(report, args.bar) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
