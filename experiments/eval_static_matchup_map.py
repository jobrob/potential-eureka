"""T0 matchup map for the three frozen strategic static agents.

The screen crosses every directed focal-versus-homogeneous-field pairing with
two generated-track families, 2/4/6 seats, and every starting seat. Results are
persisted after each cell so the bounded campaign can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.agents.static_v2 import (
    HeuristicV2Agent,
    RepairedHeuristicV2Agent,
    StaticSearchV2Agent,
)
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.ml.selfplay.eval_harness import _first_place, _play_scripted_game
from heat.ml.spaces import _placement_reward
from heat.simulation.stats import wilson_interval
from heat.tracks.generator import TrackGenParams, generate_track


SEAT_COUNTS = (2, 4, 6)
DEFAULT_TRACK_BASE = 715_000
TIGHT_TRACK_BASE = 915_000
TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),
    laps=2,
)


def _weak_agent() -> BaseAgent:
    """Build the frozen fast heuristic ruler."""
    return HeuristicAgent(name="HeuristicWeakV1")


def _repaired_agent() -> BaseAgent:
    """Build the repaired strength-2 heuristic with its safety guard enabled."""
    return StrongHeuristicAgent(
        name="HeuristicRepairedV1",
        strength=2,
        avoid_certain_spins=True,
    )


def _search_agent() -> BaseAgent:
    """Build the frozen B1 search ruler."""
    return StaticSearchAgent()


def _weak_v2_agent() -> BaseAgent:
    """Build the T2-repaired weak heuristic."""
    return HeuristicV2Agent()


def _repaired_v2_agent() -> BaseAgent:
    """Build the T2-repaired strength-2 heuristic."""
    return RepairedHeuristicV2Agent()


def _search_v2_agent() -> BaseAgent:
    """Build the T2-repaired static search policy."""
    return StaticSearchV2Agent()


@dataclass(frozen=True)
class FrozenAgentSpec:
    """Versioned identity and factory for one T0 strategic configuration."""

    agent_id: str
    label: str
    factory: Callable[[], BaseAgent]


FROZEN_AGENTS = (
    FrozenAgentSpec("heuristic_weak_v1", "Weak heuristic", _weak_agent),
    FrozenAgentSpec(
        "heuristic_repaired_v1", "Repaired heuristic", _repaired_agent
    ),
    FrozenAgentSpec("static_search_v1", "StaticSearchV1", _search_agent),
)
V2_AGENTS = (
    FrozenAgentSpec("heuristic_weak_v2", "Weak heuristic V2", _weak_v2_agent),
    FrozenAgentSpec(
        "heuristic_repaired_v2", "Repaired heuristic V2", _repaired_v2_agent
    ),
    FrozenAgentSpec("static_search_v2", "StaticSearchV2", _search_v2_agent),
)
POPULATIONS = {"v1": FROZEN_AGENTS, "v2": V2_AGENTS}
AGENTS_BY_ID = {spec.agent_id: spec for spec in FROZEN_AGENTS}


@dataclass(frozen=True)
class RaceSpec:
    """One matched track, starting-seat, and game-seed coordinate."""

    family: str
    track_seed: int
    seat_count: int
    focal_seat: int
    repeat: int
    game_seed: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded T0 matrix and persistence controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--default-track-base", type=int, default=DEFAULT_TRACK_BASE)
    parser.add_argument("--tight-track-base", type=int, default=TIGHT_TRACK_BASE)
    parser.add_argument("--population", choices=tuple(POPULATIONS), default="v1")
    parser.add_argument(
        "--opponent-population",
        choices=tuple(POPULATIONS),
        default=None,
        help="Optional unchanged field population for an isolated focal comparison.",
    )
    parser.add_argument(
        "--focal-style",
        choices=("heuristic_weak", "heuristic_repaired", "static_search"),
        default=None,
        help="Optional single focal style for a bounded timing replay.",
    )
    parser.add_argument("--max-seconds", type=float, default=1_800.0)
    parser.add_argument(
        "--out", type=Path, default=None
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.tracks <= 100:
        parser.error("--tracks must be in 1..100")
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be in 1..10")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.out is None:
        if args.opponent_population is not None:
            suffix = f"_{args.population}_vs_{args.opponent_population}"
        else:
            suffix = "" if args.population == "v1" else "_v2"
        args.out = Path(f"runs/t0_static_matchup_map{suffix}.json")
    return args


def _game_seed(
    base_seed: int,
    family_index: int,
    seat_count: int,
    track_index: int,
    focal_seat: int,
    repeat: int,
) -> int:
    """Return a matchup-independent seed for a matched race coordinate."""
    return (
        base_seed * 10_000_000
        + family_index * 1_000_000
        + seat_count * 100_000
        + track_index * 1_000
        + focal_seat * 10
        + repeat
    )


def race_specs(
    *,
    family: str,
    family_index: int,
    track_base: int,
    tracks: int,
    seat_count: int,
    repeats: int,
    seed: int,
) -> list[RaceSpec]:
    """Cross every track with every focal starting seat and repeat."""
    return [
        RaceSpec(
            family=family,
            track_seed=track_base + track_index,
            seat_count=seat_count,
            focal_seat=focal_seat,
            repeat=repeat,
            game_seed=_game_seed(
                seed,
                family_index,
                seat_count,
                track_index,
                focal_seat,
                repeat,
            ),
        )
        for track_index in range(tracks)
        for focal_seat in range(seat_count)
        for repeat in range(repeats)
    ]


def _cell_key(focal: str, opponent: str, family: str, seats: int) -> str:
    """Return the stable identity used for partial persistence and resume."""
    return f"{focal}|{opponent}|{family}|{seats}"


def _agent_style(agent_id: str) -> str:
    """Return the version-independent static style identity."""
    return agent_id.rsplit("_v", 1)[0]


def _search_profile(agents: Sequence[BaseAgent]) -> dict[str, float | int]:
    """Aggregate StaticSearchV1 planning cost from a group of agents."""
    search_agents = [agent for agent in agents if isinstance(agent, StaticSearchAgent)]
    moves = sum(agent.profile.moves for agent in search_agents)
    seconds = sum(agent.profile.seconds for agent in search_agents)
    clones = sum(agent.profile.clones for agent in search_agents)
    return {"moves": moves, "seconds": seconds, "clones": clones}


def _run_race(
    spec: RaceSpec,
    focal: FrozenAgentSpec,
    opponent: FrozenAgentSpec,
) -> dict[str, Any]:
    """Play one all-scripted matched coordinate and retain auditable raw data."""
    params = TIGHT_PARAMS if spec.family == "tight_generated" else None
    track = generate_track(spec.track_seed, params)
    agents = [opponent.factory() for _ in range(spec.seat_count)]
    agents[spec.focal_seat] = focal.factory()
    started = time.perf_counter()
    state = _play_scripted_game(track, spec.seat_count, dict(enumerate(agents)), spec.game_seed)
    elapsed = time.perf_counter() - started
    focal_player = state.get_player(spec.focal_seat)
    focal_profile = _search_profile([agents[spec.focal_seat]])
    field_profile = _search_profile(
        [agent for seat, agent in enumerate(agents) if seat != spec.focal_seat]
    )
    return {
        **asdict(spec),
        "won": _first_place(state, spec.focal_seat),
        "place": focal_player.finish_order if focal_player.finished else None,
        "placement_reward": _placement_reward(state, spec.focal_seat),
        "rounds": state.round_num,
        "elapsed_seconds": elapsed,
        "focal_search": focal_profile,
        "field_search": field_profile,
    }


def summarize_races(races: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one cell while preserving starting-seat slices and latency."""
    if not races:
        raise ValueError("cannot summarize an empty T0 cell")
    games = len(races)
    wins = sum(bool(row["won"]) for row in races)
    lower, upper = wilson_interval(wins, games)
    seat_rows: dict[str, dict[str, float | int]] = {}
    for focal_seat in sorted({int(row["focal_seat"]) for row in races}):
        selected = [row for row in races if int(row["focal_seat"]) == focal_seat]
        seat_wins = sum(bool(row["won"]) for row in selected)
        seat_rows[str(focal_seat)] = {
            "games": len(selected),
            "wins": seat_wins,
            "win_rate": seat_wins / len(selected),
            "mean_placement_reward": sum(
                float(row["placement_reward"]) for row in selected
            )
            / len(selected),
        }
    focal_moves = sum(int(row["focal_search"]["moves"]) for row in races)
    focal_seconds = sum(float(row["focal_search"]["seconds"]) for row in races)
    field_moves = sum(int(row["field_search"]["moves"]) for row in races)
    field_seconds = sum(float(row["field_search"]["seconds"]) for row in races)
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "wilson_lb": lower,
        "wilson_ub": upper,
        "chance": 1.0 / int(races[0]["seat_count"]),
        "mean_placement_reward": sum(
            float(row["placement_reward"]) for row in races
        )
        / games,
        "mean_rounds": sum(int(row["rounds"]) for row in races) / games,
        "wall_seconds": sum(float(row["elapsed_seconds"]) for row in races),
        "mean_wall_ms_per_game": 1000.0
        * sum(float(row["elapsed_seconds"]) for row in races)
        / games,
        "focal_search_ms_per_move": (
            1000.0 * focal_seconds / focal_moves if focal_moves else None
        ),
        "field_search_ms_per_move": (
            1000.0 * field_seconds / field_moves if field_moves else None
        ),
        "starting_seats": seat_rows,
    }


def _config(
    args: argparse.Namespace,
    agents: Sequence[FrozenAgentSpec] = FROZEN_AGENTS,
    opponent_agents: Sequence[FrozenAgentSpec] | None = None,
) -> dict[str, Any]:
    """Return the complete resolved T0 recipe used for resume validation."""
    config = {
        "schema": 1,
        "agents": [
            {"agent_id": spec.agent_id, "label": spec.label} for spec in agents
        ],
        "seat_counts": list(SEAT_COUNTS),
        "tracks_per_family": args.tracks,
        "repeats": args.repeats,
        "seed": args.seed,
        "track_families": {
            "default_generated": {
                "track_base": args.default_track_base,
                "params": "TrackGenParams defaults",
            },
            "tight_generated": {
                "track_base": args.tight_track_base,
                "params": {
                    "num_corners_range": list(TIGHT_PARAMS.num_corners_range),
                    "speed_limit_choices": list(TIGHT_PARAMS.speed_limit_choices),
                    "laps": TIGHT_PARAMS.laps,
                },
            },
        },
    }
    if opponent_agents is not None:
        config["opponent_agents"] = [
            {"agent_id": spec.agent_id, "label": spec.label}
            for spec in opponent_agents
        ]
    return config


def _persist(
    path: Path,
    *,
    config: dict[str, Any],
    cells: dict[str, dict[str, Any]],
    started: float,
    complete: bool,
    stop_reason: str | None = None,
) -> None:
    """Atomically persist completed cells and their raw race coordinates."""
    payload = {
        "complete": complete,
        "stop_reason": stop_reason,
        "config": config,
        "cells": list(cells.values()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_existing(
    path: Path, *, resume: bool, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Load compatible completed cells only for an explicit resume."""
    if not resume or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config") != config:
        raise ValueError("existing T0 artifact does not match the resolved recipe")
    return {str(row["key"]): row for row in payload.get("cells", [])}


def main(argv: list[str] | None = None) -> int:
    """Run or resume T0 until complete or the wall-time stop rule fires."""
    args = _parse_args(argv)
    population_agents = POPULATIONS[args.population]
    agents = population_agents
    if args.focal_style is not None:
        agents = tuple(
            spec
            for spec in agents
            if _agent_style(spec.agent_id) == args.focal_style
        )
    opponent_agents = (
        POPULATIONS[args.opponent_population]
        if args.opponent_population is not None
        else population_agents
    )
    isolated = args.opponent_population is not None or args.focal_style is not None
    config = _config(
        args,
        agents,
        opponent_agents if isolated else None,
    )
    cells = _load_existing(args.out, resume=args.resume, config=config)
    started = time.perf_counter()
    pair_count = sum(
        _agent_style(focal.agent_id) != _agent_style(opponent.agent_id)
        for focal in agents
        for opponent in opponent_agents
    )
    expected = pair_count * 2 * len(SEAT_COUNTS)
    print(
        f"T0 static matchup map cells={expected} completed={len(cells)} "
        f"tracks/family={args.tracks} repeats={args.repeats} "
        f"max_seconds={args.max_seconds:.0f}",
        flush=True,
    )
    families = (
        ("default_generated", 0, args.default_track_base),
        ("tight_generated", 1, args.tight_track_base),
    )
    for family, family_index, track_base in families:
        for focal in agents:
            for opponent in opponent_agents:
                if _agent_style(focal.agent_id) == _agent_style(opponent.agent_id):
                    continue
                for seat_count in SEAT_COUNTS:
                    key = _cell_key(focal.agent_id, opponent.agent_id, family, seat_count)
                    if key in cells:
                        print(f"skip completed {key}", flush=True)
                        continue
                    elapsed = time.perf_counter() - started
                    if elapsed >= args.max_seconds:
                        reason = f"wall-time cap reached before {key}"
                        _persist(
                            args.out,
                            config=config,
                            cells=cells,
                            started=started,
                            complete=False,
                            stop_reason=reason,
                        )
                        print(f"T0 stopped: {reason}; out={args.out}", flush=True)
                        return 0
                    specs = race_specs(
                        family=family,
                        family_index=family_index,
                        track_base=track_base,
                        tracks=args.tracks,
                        seat_count=seat_count,
                        repeats=args.repeats,
                        seed=args.seed,
                    )
                    cell_started = time.perf_counter()
                    races = [
                        _run_race(spec, focal, opponent)
                        for spec in specs
                    ]
                    summary = summarize_races(races)
                    cells[key] = {
                        "key": key,
                        "focal": focal.agent_id,
                        "opponent": opponent.agent_id,
                        "family": family,
                        "seat_count": seat_count,
                        "summary": summary,
                        "races": races,
                    }
                    _persist(
                        args.out,
                        config=config,
                        cells=cells,
                        started=started,
                        complete=False,
                    )
                    print(
                        f"cell {key} games={summary['games']} wins={summary['wins']} "
                        f"win={summary['win_rate']:.3f} "
                        f"reward={summary['mean_placement_reward']:+.3f} "
                        f"wall={time.perf_counter() - cell_started:.1f}s",
                        flush=True,
                    )
    _persist(
        args.out,
        config=config,
        cells=cells,
        started=started,
        complete=True,
    )
    print(
        f"T0 complete cells={len(cells)} wall={time.perf_counter() - started:.1f}s "
        f"out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
