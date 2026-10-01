#!/usr/bin/env python
"""Question B evaluation matrices for the 32K learning-quality campaign.

Builds the promotion and milestone screens from the revised design (Q04/Q05):
fresh default tracks ``1_415_000..1_415_039``, V2 rulers plus G0002-R00@1000K,
matched race coordinates via :func:`race_coordinates`, and one persisted raw
row per race. The library harness already rotates focal seats; this script
does not replace it -- it calls ``_play_race`` once per coordinate and stores
the auditable row the aggregate summary needs.

Default mode is scaffolding / dry-run. Real racing requires ``--execute``.
G0005 is never allocated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.run_32k_learning_quality import (
    CODEC_PIN,
    QUESTION_ID,
)
from heat.agents.base import BaseAgent
from heat.agents.static_v2 import (
    HeuristicV2Agent,
    RepairedHeuristicV2Agent,
    StaticSearchV2Agent,
)
from heat.ml.policy_registry import find_registered_run
from heat.ml.selfplay.checkpoint import load_policy
from heat.ml.selfplay.eval_harness import _play_race, race_coordinates
from heat.ml.selfplay.snapshots import SnapshotAgent
from heat.ml.spaces import CODEC_VERSION
from heat.models.track import Track
from heat.tracks.generator import generate_track

SCHEMA_VERSION = 1
TRACK_BASE = 1_415_000
TRACK_COUNT = 40
SEAT_COUNTS = (2, 4, 6)
PROMOTION_GAMES = 200
MILESTONE_GAMES = 40
PROMOTION_GAME_SEED_BASE = 3_200_000
MILESTONE_GAME_SEED_BASE = 3_210_000

PROMOTION_RULERS = (
    "heuristic_weak_v2",
    "heuristic_repaired_v2",
    "static_search_v2",
    "historical_g0002_r00_1000k",
)
MILESTONE_RULERS = (
    "heuristic_weak_v2",
    "historical_g0002_r00_1000k",
)

DEFAULT_BASELINE_AGENTS = ("G0002-R00", "G0002-R01", "G0002-R02")
DEFAULT_OUT = Path("runs/32k_learning_quality/eval")


@dataclass(frozen=True)
class ContenderSpec:
    """One learned policy identity used as a matrix contender."""

    agent_id: str
    generation_id: str
    training_seed: int
    checkpoint_id: str
    checkpoint: str
    role: str  # "baseline" | "candidate"


def track_sha256(track: Track) -> str:
    """Hash the realized track geometry used in a race row."""
    payload = {
        "name": track.name,
        "laps": track.laps,
        "start_positions": list(track.start_positions),
        "spaces": [(space.index, space.lanes) for space in track.spaces],
        "corners": [
            (corner.start, corner.end, corner.speed_limit) for corner in track.corners
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def track_band(track_base: int = TRACK_BASE, count: int = TRACK_COUNT) -> list[Track]:
    """Return the frozen fresh-development default track band."""
    return [generate_track(track_base + index) for index in range(count)]


def _baseline_contender(agent_id: str) -> ContenderSpec:
    """Resolve one immutable G0002 selected checkpoint from the registry."""
    run = find_registered_run(agent_id)
    return ContenderSpec(
        agent_id=agent_id,
        generation_id=agent_id.split("-", 1)[0],
        training_seed=int(run["seed"]),
        checkpoint_id=str(run["selected_checkpoint_id"]),
        checkpoint=str(run["selected_checkpoint"]),
        role="baseline",
    )


def candidate_contender(
    *,
    agent_id: str,
    training_seed: int,
    checkpoint: str,
    checkpoint_id: str | None = None,
) -> ContenderSpec:
    """Build a candidate contender from an explicit checkpoint path.

    Until G0005 is registered, candidate IDs should be ``dev-32k-qb-*`` labels
    or provisional ``G0005-R##@1000K`` strings that are NOT written to the
    registry by this script.
    """
    generation_id = agent_id.split("-", 1)[0]
    return ContenderSpec(
        agent_id=agent_id,
        generation_id=generation_id,
        training_seed=training_seed,
        checkpoint_id=checkpoint_id or f"{agent_id}@1000K",
        checkpoint=checkpoint,
        role="candidate",
    )


def ruler_factories(
    *,
    include_historical: bool = True,
) -> dict[str, Callable[[], BaseAgent]]:
    """Build V2 ruler factories; optionally include the frozen G0002-R00 ruler."""
    factories: dict[str, Callable[[], BaseAgent]] = {
        "heuristic_weak_v2": HeuristicV2Agent,
        "heuristic_repaired_v2": RepairedHeuristicV2Agent,
        "static_search_v2": StaticSearchV2Agent,
    }
    if include_historical:
        # Loading may fail on machines without the checkpoint bytes; callers that
        # only need the V2 timing probe can pass include_historical=False.
        anchor_path = find_registered_run("G0002-R00")["selected_checkpoint"]
        anchor_policy = load_policy(anchor_path)
        anchor = SnapshotAgent(anchor_policy, name="G0002-R00@1000K")
        factories["historical_g0002_r00_1000k"] = lambda: anchor
    return factories


def matrix_manifest(
    *,
    matrix_id: str,
    games: int,
    seats: tuple[int, ...] = SEAT_COUNTS,
    track_count: int = TRACK_COUNT,
) -> dict[str, Any]:
    """Return the matched coordinate manifest shared by every contender."""
    cells = {}
    for seat_count in seats:
        coords = race_coordinates(games, track_count, seat_count)
        seat_totals = [0] * seat_count
        for _track_index, focal_seat, _repeat in coords:
            seat_totals[focal_seat] += 1
        cells[str(seat_count)] = {
            "games": games,
            "coordinates": [
                {
                    "race_index": index,
                    "track_index": track_index,
                    "focal_seat": focal_seat,
                    "repeat_index": repeat_index,
                }
                for index, (track_index, focal_seat, repeat_index) in enumerate(coords)
            ],
            "focal_seat_counts": seat_totals,
            "omitted_note": (
                "Five repeats cannot cover six seats on a 200/40/6 cell; "
                "omission is recorded, not filled."
                if seat_count == 6 and games == 200 and track_count == 40
                else None
            ),
        }
    return {
        "matrix_id": matrix_id,
        "track_base": TRACK_BASE,
        "track_count": track_count,
        "seat_counts": list(seats),
        "games_per_cell": games,
        "cells": cells,
    }


def cell_key(matrix_id: str, contender: str, ruler: str, seats: int) -> str:
    """Stable resume key for one matrix cell."""
    return f"{matrix_id}|{contender}|{ruler}|{seats}"


def race_row_id(
    matrix_id: str, contender: str, ruler: str, seats: int, race_index: int
) -> str:
    """Stable identity for one race coordinate under one contender/ruler."""
    return f"{cell_key(matrix_id, contender, ruler, seats)}|{race_index}"


def build_race_row(
    *,
    matrix_id: str,
    contender: ContenderSpec,
    ruler_id: str,
    seat_count: int,
    race_index: int,
    track_index: int,
    focal_seat: int,
    repeat_index: int,
    track: Track,
    game_seed: int,
    outcome_win: bool,
    placement_reward: float,
    terminal_round: int,
    completion_state: str,
    source_identity: str,
    config_hash: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Assemble one auditable raw race row (design Q05 field groups)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "race_coordinate_id": race_row_id(
            matrix_id, contender.agent_id, ruler_id, seat_count, race_index
        ),
        "contender_agent_id": contender.agent_id,
        "contender_generation_id": contender.generation_id,
        "contender_checkpoint_id": contender.checkpoint_id,
        "contender_training_seed": contender.training_seed,
        "contender_checkpoint": contender.checkpoint,
        "contender_role": contender.role,
        "ruler_id": ruler_id,
        "seat_count": seat_count,
        "focal_starting_seat": focal_seat,
        "track_family": "default_generated",
        "track_seed": TRACK_BASE + track_index,
        "track_index": track_index,
        "track_sha256": track_sha256(track),
        "game_seed": game_seed,
        "repeat_index": repeat_index,
        "race_index": race_index,
        "realized_track": {
            "length": track.length,
            "laps": track.laps,
            "corner_count": len(track.corners),
            "corners": [
                {
                    "start": corner.start,
                    "end": corner.end,
                    "speed_limit": corner.speed_limit,
                }
                for corner in track.corners
            ],
            "lane_counts": [space.lanes for space in track.spaces],
            "start_grid": list(track.start_positions),
        },
        "finish_position_first": outcome_win,
        "placement_reward": placement_reward,
        "completion_state": completion_state,
        "terminal_round_count": terminal_round,
        "receipt": {
            "source_identity": source_identity,
            "resolved_evaluation_config_hash": config_hash,
            "ruler_implementation_identity": ruler_id,
            "elapsed_seconds": elapsed_seconds,
            "codec_version": CODEC_VERSION,
            "question_id": QUESTION_ID,
        },
    }


def config_hash(payload: dict[str, Any]) -> str:
    """Hash the evaluation config used for resume identity checks."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist JSON through a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_resumable_payload(
    path: Path,
    *,
    expected_config_hash: str,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    """Load completed cells only when identities match exactly."""
    if not resume or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.get("config", {}).get("config_hash")
    if stored_hash != expected_config_hash:
        raise ValueError(
            "resume rejected: evaluation config hash mismatch "
            f"(stored={stored_hash!r} expected={expected_config_hash!r})"
        )
    if payload.get("question_id") != QUESTION_ID:
        raise ValueError("resume rejected: question_id mismatch")
    if int(payload.get("codec_pin", -1)) != CODEC_PIN:
        raise ValueError("resume rejected: codec pin mismatch")
    return {row["key"]: row for row in payload.get("cells", [])}


def _load_contender_policy(spec: ContenderSpec) -> Any:
    """Load one contender checkpoint as a playable policy or snapshot agent."""
    policy = load_policy(spec.checkpoint)
    return SnapshotAgent(policy, name=spec.checkpoint_id)


def run_matrix(
    *,
    matrix_id: str,
    games: int,
    game_seed_base: int,
    rulers: tuple[str, ...],
    contenders: tuple[ContenderSpec, ...],
    tracks: list[Track],
    out_path: Path,
    resume: bool,
    source_identity: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run or resume one matrix, persisting raw rows after every cell."""
    factories = ruler_factories(include_historical="historical_g0002_r00_1000k" in rulers)
    missing = [ruler for ruler in rulers if ruler not in factories]
    if missing:
        raise ValueError(f"unknown rulers: {missing}")
    manifest = matrix_manifest(matrix_id=matrix_id, games=games)
    config = {
        "matrix_id": matrix_id,
        "games_per_cell": games,
        "track_base": TRACK_BASE,
        "track_count": len(tracks),
        "seat_counts": list(SEAT_COUNTS),
        "rulers": list(rulers),
        "contenders": [asdict(spec) for spec in contenders],
        "game_seed_base": game_seed_base,
        "question_id": QUESTION_ID,
        "codec_pin": CODEC_PIN,
        "manifest": manifest,
    }
    digest = config_hash(config)
    config["config_hash"] = digest
    cells = load_resumable_payload(out_path, expected_config_hash=digest, resume=resume)
    started = time.perf_counter()
    expected = len(contenders) * len(rulers) * len(SEAT_COUNTS)
    print(
        f"{matrix_id}: {expected} cells, games/cell={games}, resume={resume}",
        flush=True,
    )

    for contender in contenders:
        policy = _load_contender_policy(contender)
        for ruler_id in rulers:
            factory = factories[ruler_id]
            for seat_count in SEAT_COUNTS:
                key = cell_key(matrix_id, contender.agent_id, ruler_id, seat_count)
                if key in cells:
                    continue
                cell_started = time.perf_counter()
                coordinates = race_coordinates(games, len(tracks), seat_count)
                rows: list[dict[str, Any]] = []
                wins = 0
                reward_sum = 0.0
                for race_index, (track_index, focal_seat, repeat_index) in enumerate(
                    coordinates
                ):
                    track = tracks[track_index]
                    game_seed = game_seed_base + race_index
                    outcome = _play_race(
                        policy,
                        factory,
                        track,
                        focal_seat,
                        seat_count,
                        game_seed,
                        device,
                    )
                    # RaceOutcome does not expose terminal state; record reward-level
                    # completion only. Full terminal rounds need a harness extension.
                    completion_state = "scored"
                    terminal_round = -1
                    rows.append(
                        build_race_row(
                            matrix_id=matrix_id,
                            contender=contender,
                            ruler_id=ruler_id,
                            seat_count=seat_count,
                            race_index=race_index,
                            track_index=track_index,
                            focal_seat=focal_seat,
                            repeat_index=repeat_index,
                            track=track,
                            game_seed=game_seed,
                            outcome_win=bool(outcome.win),
                            placement_reward=float(outcome.placement_reward),
                            terminal_round=terminal_round,
                            completion_state=completion_state,
                            source_identity=source_identity,
                            config_hash=digest,
                            elapsed_seconds=time.perf_counter() - cell_started,
                        )
                    )
                    wins += int(outcome.win)
                    reward_sum += float(outcome.placement_reward)
                cells[key] = {
                    "key": key,
                    "matrix_id": matrix_id,
                    "contender": contender.agent_id,
                    "generation_id": contender.generation_id,
                    "checkpoint_id": contender.checkpoint_id,
                    "ruler": ruler_id,
                    "seats": seat_count,
                    "games": games,
                    "wins": wins,
                    "mean_placement_reward": reward_sum / games,
                    "raw_rows": rows,
                }
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "question_id": QUESTION_ID,
                    "codec_pin": CODEC_PIN,
                    "g0005_allocated": False,
                    "complete": len(cells) == expected,
                    "config": config,
                    "source_identity": source_identity,
                    "cells": list(cells.values()),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                _atomic_json(out_path, payload)
                print(
                    f"  wrote {key} reward={reward_sum / games:.4f} "
                    f"wins={wins}/{games}",
                    flush=True,
                )
    return {
        "complete": len(cells) == expected,
        "cells": len(cells),
        "out": str(out_path),
    }


def plan_payload(
    *,
    candidate_specs: tuple[ContenderSpec, ...],
    include_baselines: bool = True,
) -> dict[str, Any]:
    """Build the dry-run evaluation plan without racing or loading checkpoints."""
    baselines = (
        tuple(_baseline_contender(agent_id) for agent_id in DEFAULT_BASELINE_AGENTS)
        if include_baselines
        else ()
    )
    # Avoid loading G0002 weights during dry-run; only record intended rulers.
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": QUESTION_ID,
        "codec_pin": CODEC_PIN,
        "g0005_allocated": False,
        "track_band": {
            "base": TRACK_BASE,
            "count": TRACK_COUNT,
            "seeds": list(range(TRACK_BASE, TRACK_BASE + TRACK_COUNT)),
        },
        "matrices": {
            "promotion": {
                "games_per_cell": PROMOTION_GAMES,
                "rulers": list(PROMOTION_RULERS),
                "seat_counts": list(SEAT_COUNTS),
                "game_seed_namespace": PROMOTION_GAME_SEED_BASE,
                "expected_races": (
                    (len(baselines) + len(candidate_specs))
                    * len(PROMOTION_RULERS)
                    * len(SEAT_COUNTS)
                    * PROMOTION_GAMES
                ),
            },
            "milestone": {
                "games_per_cell": MILESTONE_GAMES,
                "rulers": list(MILESTONE_RULERS),
                "seat_counts": list(SEAT_COUNTS),
                "game_seed_namespace": MILESTONE_GAME_SEED_BASE,
                "note": "Diagnostic only; run after promotion.",
            },
        },
        "baselines": [asdict(spec) for spec in baselines],
        "candidates": [asdict(spec) for spec in candidate_specs],
        "manifest_preview": matrix_manifest(
            matrix_id="promotion", games=PROMOTION_GAMES
        ),
    }


def _parse_candidate_args(raw: list[str]) -> tuple[ContenderSpec, ...]:
    """Parse ``agent_id:seed:checkpoint`` candidate triples from the CLI."""
    specs: list[ContenderSpec] = []
    for item in raw:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "candidate must be agent_id:seed:checkpoint, "
                f"got {item!r}"
            )
        agent_id, seed_text, checkpoint = parts
        specs.append(
            candidate_contender(
                agent_id=agent_id,
                training_seed=int(seed_text),
                checkpoint=checkpoint,
            )
        )
    return tuple(specs)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse dry-run / execute controls for the Question B eval matrices."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--matrix",
        choices=("promotion", "milestone", "both"),
        default="both",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=[],
        help="Optional agent_id:seed:checkpoint triples for candidate policies.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Scaffolding only: omit G0002 baselines (not for a real pilot).",
    )
    parser.add_argument("--plan-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.execute:
        args.dry_run = False
    return args


def main(argv: list[str] | None = None) -> int:
    """Dry-run the eval plan, or execute matrices with raw-row persistence."""
    args = _parse_args(argv)
    candidates = _parse_candidate_args(args.candidates)
    plan = plan_payload(
        candidate_specs=candidates,
        include_baselines=not args.skip_baselines,
    )
    plan_path = args.plan_out or (args.out_dir / "eval_plan.json")
    if args.dry_run:
        _atomic_json(plan_path, plan)
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        print(
            "\nDRY-RUN only. No races started. G0005 was NOT allocated.\n"
            "Promotion should be scheduled before the milestone screen.\n"
            "Time the V2 rulers with experiments/probe_32k_v2_ruler_timing.py "
            "before treating wall-clock estimates as real.",
            flush=True,
        )
        return 0

    if not candidates and not args.skip_baselines:
        # Baselines alone are a valid G0002 re-measure; candidates optional.
        pass
    if CODEC_VERSION != CODEC_PIN:
        raise RuntimeError(
            f"live CODEC_VERSION={CODEC_VERSION} does not match Question B pin "
            f"{CODEC_PIN}"
        )

    from experiments.verify_direction_d3_native import native_source_tree_identity

    source_identity = native_source_tree_identity()
    tracks = track_band()
    baselines = (
        ()
        if args.skip_baselines
        else tuple(_baseline_contender(agent_id) for agent_id in DEFAULT_BASELINE_AGENTS)
    )
    contenders = baselines + candidates
    if not contenders:
        raise ValueError("no contenders: pass --candidates or keep baselines")
    device = torch.device("cpu")
    matrices: list[tuple[str, int, int, tuple[str, ...]]] = []
    if args.matrix in {"promotion", "both"}:
        matrices.append(
            ("promotion", PROMOTION_GAMES, PROMOTION_GAME_SEED_BASE, PROMOTION_RULERS)
        )
    if args.matrix in {"milestone", "both"}:
        matrices.append(
            ("milestone", MILESTONE_GAMES, MILESTONE_GAME_SEED_BASE, MILESTONE_RULERS)
        )
    # Design: schedule promotion before milestone.
    results = []
    for matrix_id, games, seed_base, rulers in matrices:
        out_path = args.out_dir / f"{matrix_id}.json"
        results.append(
            run_matrix(
                matrix_id=matrix_id,
                games=games,
                game_seed_base=seed_base,
                rulers=rulers,
                contenders=contenders,
                tracks=tracks,
                out_path=out_path,
                resume=args.resume,
                source_identity=source_identity,
                device=device,
            )
        )
    _atomic_json(plan_path, plan)
    print(json.dumps({"results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    from _runlog import run_main

    run_main("eval_32k_learning_quality", main)
