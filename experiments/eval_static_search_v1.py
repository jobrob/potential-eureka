"""B1 confirmation gate for the frozen StaticSearchV1 benchmark.

The experiment separates four claims: strength against fixed scripted fields,
tight-corner safety, decision latency, and usefulness against representative A8
checkpoints. It writes raw JSON evidence; the human decision belongs in the H2
HTML investigation document.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from diag_h1_heuristic import GameSpec
from eval_search import _run_field
from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.engine.game import Game
from heat.ml.policy_registry import find_registered_run
from heat.ml.selfplay.checkpoint import load_policy
from heat.ml.selfplay.eval_harness import EvalCell, evaluate_policy
from heat.tracks.generator import generate_track


DEFAULT_AGENT_IDS = ("G0002-R00", "G0002-R01", "G0002-R02")
DEFAULT_RUNS = tuple(find_registered_run(agent_id) for agent_id in DEFAULT_AGENT_IDS)
DEFAULT_CHECKPOINTS = tuple(str(run["selected_checkpoint"]) for run in DEFAULT_RUNS)
CHECKPOINT_IDS_BY_PATH = {
    str(run["selected_checkpoint"]): str(run["selected_checkpoint_id"])
    for run in DEFAULT_RUNS
}
KNOWN_LOCKS = (
    GameSpec(4, 700_004, 750_012, 0),
    GameSpec(2, 700_004, 730_012, 0),
    GameSpec(2, 700_004, 730_013, 1),
    GameSpec(6, 700_002, 770_008, 2),
    GameSpec(6, 700_001, 770_004, 4),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the reproducible B1 experiment controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength-games", type=int, default=200)
    parser.add_argument("--challenge-games", type=int, default=50)
    parser.add_argument("--tracks", type=int, default=20)
    parser.add_argument("--tight-tracks", type=int, default=24)
    parser.add_argument("--latency-tracks", type=int, default=12)
    parser.add_argument("--track-base", type=int, default=713_000)
    parser.add_argument("--tight-track-base", type=int, default=913_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--checkpoints", nargs="*", default=list(DEFAULT_CHECKPOINTS)
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/a8_b1_static_search_v1.json")
    )
    args = parser.parse_args(argv)
    for field in (
        "strength_games",
        "challenge_games",
        "tracks",
        "tight_tracks",
        "latency_tracks",
    ):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def _run_cell(
    policy: Any,
    opponent: Callable[[], BaseAgent],
    opponent_label: str,
    seats: int,
    tracks: list[Any],
    games: int,
    seed: int,
) -> EvalCell:
    """Run and print one cell so a long campaign always shows progress."""
    report = evaluate_policy(
        policy,
        opponents={opponent_label: opponent},
        seat_counts=(seats,),
        splits={"b1_frozen": tracks},
        games_per_cell=games,
        seed=seed,
        device=torch.device("cpu"),
    )
    cell = report.cells[0]
    print(
        f"cell opponent={opponent_label} seats={seats} games={cell.games} "
        f"wins={cell.wins} win={cell.win_rate:.3f} "
        f"wilson=[{cell.wilson_lb:.3f},{cell.wilson_ub:.3f}] "
        f"chance={cell.chance:.3f} reward={cell.mean_placement_reward:+.3f}",
        flush=True,
    )
    return cell


def _strength_gate(
    tracks: list[Any], games: int, seed: int
) -> list[EvalCell]:
    """Evaluate the candidate against weak and repaired homogeneous fields."""
    cells: list[EvalCell] = []
    opponents: tuple[tuple[str, Callable[[], BaseAgent]], ...] = (
        ("weak", HeuristicAgent),
        ("heuristic_repaired", StrongHeuristicAgent),
    )
    for oi, (label, factory) in enumerate(opponents):
        for si, seats in enumerate((2, 4, 6)):
            cells.append(
                _run_cell(
                    StaticSearchAgent(),
                    factory,
                    label,
                    seats,
                    tracks,
                    games,
                    seed + oi * 100 + si * 10,
                )
            )
    return cells


def _tight_safety(track_base: int, tracks: int, seed: int) -> dict[str, Any]:
    """Compare per-corner spin tails on the existing tight-track discipline."""
    seeds = [track_base + i for i in range(tracks)]
    labels = {
        "weak": HeuristicAgent,
        "heuristic_repaired": StrongHeuristicAgent,
        "static_search_v1": StaticSearchAgent,
    }
    aggregates, profiles = _run_field(
        labels, track_seeds=seeds, num_players=4, game_seed_base=seed
    )
    result: dict[str, Any] = {}
    for label, aggregate in aggregates.items():
        limits: dict[str, Any] = {}
        for limit in (1, 2, 3, 4, 5):
            mean, p90, maximum = aggregate.spin_stats(limit)
            passes = aggregate.total_passes_by_limit.get(limit, 0)
            spins = aggregate.total_spins_by_limit.get(limit, 0)
            if passes or spins:
                limits[str(limit)] = {
                    "mean": mean,
                    "p90": p90,
                    "max": maximum,
                    "passes": passes,
                    "spins": spins,
                }
        result[label] = {
            "games": aggregate.games,
            "finish_rate": aggregate.finish_rate(),
            "mean_rounds": aggregate.mean_rounds(),
            "limits": limits,
            "profile": profiles[label],
        }
        l1 = limits.get("1", {})
        print(
            f"safety agent={label} finish={aggregate.finish_rate():.3f} "
            f"rounds={aggregate.mean_rounds():.1f} "
            f"L1={l1.get('spins', 0)}/{l1.get('passes', 0)} "
            f"p90={l1.get('p90', math.nan):.3f} "
            f"max={l1.get('max', math.nan):.3f} {profiles[label]}",
            flush=True,
        )
    return result


def _known_lock_gate() -> list[dict[str, Any]]:
    """Replay the five historical recovery locks with StaticSearchV1."""
    rows: list[dict[str, Any]] = []
    for spec in KNOWN_LOCKS:
        agents: list[BaseAgent] = [HeuristicAgent() for _ in range(spec.seats)]
        agents[spec.strong_seat] = StaticSearchAgent()
        game = Game(
            generate_track(spec.track_seed),
            agents,
            logging_enabled=True,
            seed=spec.game_seed,
        )
        race = game.run()
        spin_log = game.state.get_player(spec.strong_seat).spin_log
        by_corner = Counter(corner for _round, corner in spin_log)
        row = {
            "spec": asdict(spec),
            "spins": len(spin_log),
            "max_same_corner": max(by_corner.values(), default=0),
            "rounds": race.total_rounds,
            "place": race.finish_order.index(spec.strong_seat) + 1,
        }
        rows.append(row)
        print(
            f"lock seats={spec.seats} track={spec.track_seed} "
            f"game={spec.game_seed} spins={row['spins']} "
            f"max_same_corner={row['max_same_corner']}",
            flush=True,
        )
    return rows


def _latency_gate(track_base: int, tracks: int, seed: int) -> dict[str, float]:
    """Measure the distribution of per-game average focal latency at six seats."""
    samples: list[float] = []
    for ti in range(tracks):
        track = generate_track(track_base + ti)
        for focal_seat in range(6):
            candidate = StaticSearchAgent()
            agents: list[BaseAgent] = [HeuristicAgent() for _ in range(6)]
            agents[focal_seat] = candidate
            Game(
                track,
                agents,
                logging_enabled=False,
                seed=seed + ti * 10 + focal_seat,
            ).run()
            if candidate.profile.moves:
                samples.append(
                    1000.0 * candidate.profile.seconds / candidate.profile.moves
                )
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    result = {
        "games": float(len(samples)),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
    }
    print(
        f"latency six-seat games={len(samples)} median={result['median_ms']:.2f}ms "
        f"p95={result['p95_ms']:.2f}ms max={result['max_ms']:.2f}ms",
        flush=True,
    )
    return result


def _checkpoint_challenge(
    checkpoints: list[str], tracks: list[Any], games: int, seed: int
) -> dict[str, list[EvalCell]]:
    """Challenge representative A8 policies against the frozen search field."""
    results: dict[str, list[EvalCell]] = {}
    for ci, checkpoint in enumerate(checkpoints):
        policy = load_policy(checkpoint)
        cells: list[EvalCell] = []
        checkpoint_id = CHECKPOINT_IDS_BY_PATH.get(checkpoint, checkpoint)
        print(f"challenge checkpoint={checkpoint_id} path={checkpoint}", flush=True)
        for si, seats in enumerate((2, 4, 6)):
            cells.append(
                _run_cell(
                    policy,
                    StaticSearchAgent,
                    "static_search_v1",
                    seats,
                    tracks,
                    games,
                    seed + ci * 100 + si * 10,
                )
            )
        results[checkpoint_id] = cells
    return results


def main(argv: list[str] | None = None) -> int:
    """Run all B1 gates and write one machine-readable evidence file."""
    args = _parse_args(argv)
    started = time.perf_counter()
    tracks = [generate_track(args.track_base + i) for i in range(args.tracks)]
    print(
        "B1 StaticSearchV1 gate "
        f"strength_games={args.strength_games} challenge_games={args.challenge_games} "
        f"tracks={args.tracks} tight_tracks={args.tight_tracks}",
        flush=True,
    )
    strength_started = time.perf_counter()
    strength = _strength_gate(tracks, args.strength_games, args.seed)
    strength_seconds = time.perf_counter() - strength_started

    safety_started = time.perf_counter()
    safety = _tight_safety(
        args.tight_track_base, args.tight_tracks, args.seed + 10_000
    )
    known_locks = _known_lock_gate()
    safety_seconds = time.perf_counter() - safety_started

    latency_started = time.perf_counter()
    latency = _latency_gate(
        args.track_base + 10_000, args.latency_tracks, args.seed + 20_000
    )
    latency_seconds = time.perf_counter() - latency_started

    challenge_started = time.perf_counter()
    challenge = _checkpoint_challenge(
        args.checkpoints, tracks, args.challenge_games, args.seed + 30_000
    )
    challenge_seconds = time.perf_counter() - challenge_started
    total_seconds = time.perf_counter() - started

    payload = {
        "config": {
            "agent": StaticSearchAgent.VERSION,
            "strength_games": args.strength_games,
            "challenge_games": args.challenge_games,
            "tracks": args.tracks,
            "tight_tracks": args.tight_tracks,
            "latency_tracks": args.latency_tracks,
            "track_base": args.track_base,
            "tight_track_base": args.tight_track_base,
            "seed": args.seed,
            "checkpoints": args.checkpoints,
            "checkpoint_ids": [
                CHECKPOINT_IDS_BY_PATH.get(checkpoint, checkpoint)
                for checkpoint in args.checkpoints
            ],
        },
        "strength": [asdict(cell) for cell in strength],
        "safety": safety,
        "known_locks": known_locks,
        "latency": latency,
        "challenge": {
            checkpoint: [asdict(cell) for cell in cells]
            for checkpoint, cells in challenge.items()
        },
        "timing_seconds": {
            "strength": strength_seconds,
            "safety": safety_seconds,
            "latency": latency_seconds,
            "challenge": challenge_seconds,
            "total": total_seconds,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"B1 complete wall={total_seconds:.1f}s out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
