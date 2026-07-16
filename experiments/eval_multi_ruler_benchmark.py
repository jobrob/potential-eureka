"""Reliable A8 benchmark across several fixed, non-transitive ruler styles."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.ml.policy_registry import find_registered_run
from heat.ml.selfplay.checkpoint import load_policy
from heat.ml.selfplay.eval_harness import EvalCell, evaluate_policy
from heat.ml.selfplay.snapshots import SnapshotAgent
from heat.tracks.generator import generate_track


@dataclass(frozen=True)
class ContenderSpec:
    """One selected learned policy in the benchmark matrix."""

    agent_id: str
    generation_id: str
    seed: int
    selected_checkpoint_id: str
    checkpoint: str


def _registered_contender(agent_id: str) -> ContenderSpec:
    """Resolve one immutable selected checkpoint from the central registry."""
    run = find_registered_run(agent_id)
    return ContenderSpec(
        agent_id=agent_id,
        generation_id=agent_id.split("-", 1)[0],
        seed=int(run["seed"]),
        selected_checkpoint_id=str(run["selected_checkpoint_id"]),
        checkpoint=str(run["selected_checkpoint"]),
    )


DEFAULT_AGENT_IDS = (
    "G0002-R00",
    "G0002-R01",
    "G0002-R02",
    "G0003-R00",
    "G0003-R01",
    "G0003-R02",
)
CONTENDERS = tuple(_registered_contender(agent_id) for agent_id in DEFAULT_AGENT_IDS)
SEAT_COUNTS = (2, 4, 6)
RULER_LABELS = (
    "weak",
    "heuristic_repaired",
    "search_candidate",
    "historical_g0002_r00_1000k",
)
LEGACY_AGENT_IDS = {
    "s1_seed0": "G0002-R00",
    "s1_seed1": "G0002-R01",
    "s1_seed2": "G0002-R02",
    "s2_seed0": "G0003-R00",
    "s2_seed1": "G0003-R01",
    "s2_seed2": "G0003-R02",
}
LEGACY_RULER_IDS = {
    "historical_s1_seed0": "historical_g0002_r00_1000k",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse matrix size, frozen track band, persistence, and resume controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--tracks", type=int, default=40)
    parser.add_argument("--track-base", type=int, default=714_000)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(DEFAULT_AGENT_IDS),
        help="Registered G####-R## contenders (default: B2 G0002/G0003).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/a8_b2_multi_ruler.json")
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed cells already present in --out.",
    )
    parser.add_argument(
        "--migrate-identities-only",
        action="store_true",
        help="Rewrite legacy B2 labels to registered policy IDs without racing.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild the deterministic summary for --out without racing.",
    )
    args = parser.parse_args(argv)
    if args.games < 1 or args.tracks < 1:
        parser.error("--games and --tracks must be positive")
    return args


def _migrate_payload_identities(payload: dict[str, Any]) -> bool:
    """Rewrite old S1/S2 aliases in one B2 payload to immutable registry IDs."""
    changed = False
    specs = {spec.agent_id: spec for spec in CONTENDERS}
    config = payload.get("config")
    if isinstance(config, dict):
        old_contenders = config.get("contenders")
        new_contenders = [asdict(spec) for spec in CONTENDERS]
        if old_contenders != new_contenders:
            config["contenders"] = new_contenders
            changed = True
        old_rulers = config.get("rulers", [])
        new_rulers = [LEGACY_RULER_IDS.get(str(ruler), ruler) for ruler in old_rulers]
        if old_rulers != new_rulers:
            config["rulers"] = new_rulers
            changed = True

    for row in payload.get("cells", []):
        old_agent = str(row.get("contender"))
        agent_id = LEGACY_AGENT_IDS.get(old_agent, old_agent)
        old_ruler = str(row.get("ruler"))
        ruler = LEGACY_RULER_IDS.get(old_ruler, old_ruler)
        spec = specs.get(agent_id)
        if spec is None:
            continue
        replacements = {
            "contender": agent_id,
            "recipe": spec.generation_id,
            "selected_checkpoint_id": spec.selected_checkpoint_id,
            "ruler": ruler,
            "key": _cell_key(agent_id, ruler, int(row["seats"])),
        }
        for key, value in replacements.items():
            if row.get(key) != value:
                row[key] = value
                changed = True
    return changed


def _migrate_result_file(path: Path) -> bool:
    """Atomically migrate an existing B2 JSON artifact without rerunning cells."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = _migrate_payload_identities(payload)
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    return changed


def _rulers() -> dict[str, Callable[[], BaseAgent]]:
    """Build the four fixed ruler factories, including one historical policy."""
    anchor_policy = load_policy(_registered_contender("G0002-R00").checkpoint)
    anchor = SnapshotAgent(anchor_policy, name="G0002-R00@1000K")
    return {
        "weak": HeuristicAgent,
        "heuristic_repaired": StrongHeuristicAgent,
        "search_candidate": StaticSearchAgent,
        # SnapshotAgent has no mutable turn cache; sharing the frozen instance
        # avoids deep-copying a neural network for every opponent seat and game.
        "historical_g0002_r00_1000k": lambda: anchor,
    }


def _cell_key(contender: str, ruler: str, seats: int) -> str:
    """Return the stable identity used for partial persistence and resume."""
    return f"{contender}|{ruler}|{seats}"


def _load_existing(path: Path, resume: bool) -> dict[str, dict[str, Any]]:
    """Load previously completed cells only when explicit resume is requested."""
    if not resume or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["key"]: row for row in payload.get("cells", [])}


def _persist(
    path: Path,
    *,
    args: argparse.Namespace,
    contenders: tuple[ContenderSpec, ...],
    cells: dict[str, dict[str, Any]],
    started: float,
    complete: bool,
) -> None:
    """Atomically enough for this local run, persist every completed matrix cell."""
    payload = {
        "complete": complete,
        "config": {
            "games_per_cell": args.games,
            "tracks": args.tracks,
            "track_base": args.track_base,
            "seed": args.seed,
            "seat_counts": list(SEAT_COUNTS),
            "rulers": list(RULER_LABELS),
            "contenders": [asdict(spec) for spec in contenders],
        },
        "cells": list(cells.values()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-run and recipe placement metrics and frozen A8 gates."""
    rows = payload.get("cells", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark payload contains no cells")

    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_agent.setdefault(str(row["contender"]), []).append(row)

    run_rows: list[dict[str, Any]] = []
    for agent_id, agent_cells in sorted(by_agent.items()):
        rewards = [float(row["cell"]["mean_placement_reward"]) for row in agent_cells]
        ruler_averages = {
            ruler: statistics.fmean(
                float(row["cell"]["mean_placement_reward"])
                for row in agent_cells
                if row["ruler"] == ruler
            )
            for ruler in sorted({str(row["ruler"]) for row in agent_cells})
        }
        run_rows.append(
            {
                "agent_id": agent_id,
                "generation_id": str(agent_cells[0]["recipe"]),
                "aggregate_reward": statistics.fmean(rewards),
                "ruler_rewards": ruler_averages,
            }
        )

    generations: dict[str, dict[str, Any]] = {}
    for generation_id in sorted({row["generation_id"] for row in run_rows}):
        generation_runs = [
            row for row in run_rows if row["generation_id"] == generation_id
        ]
        aggregates = [float(row["aggregate_reward"]) for row in generation_runs]
        rulers = sorted(generation_runs[0]["ruler_rewards"])
        ruler_medians = {
            ruler: statistics.median(
                float(row["ruler_rewards"][ruler]) for row in generation_runs
            )
            for ruler in rulers
        }
        ruler_seat_medians: dict[str, dict[str, float]] = {}
        for ruler in rulers:
            seat_values: dict[str, float] = {}
            seats = sorted(
                {
                    int(row["seats"])
                    for row in rows
                    if row["recipe"] == generation_id and row["ruler"] == ruler
                }
            )
            for seat_count in seats:
                values = [
                    float(row["cell"]["mean_placement_reward"])
                    for row in rows
                    if row["recipe"] == generation_id
                    and row["ruler"] == ruler
                    and int(row["seats"]) == seat_count
                ]
                seat_values[str(seat_count)] = statistics.median(values)
            ruler_seat_medians[ruler] = seat_values
        generations[generation_id] = {
            "run_aggregates": {
                str(row["agent_id"]): row["aggregate_reward"]
                for row in generation_runs
            },
            "median_aggregate_reward": statistics.median(aggregates),
            "population_sd_aggregate_reward": statistics.pstdev(aggregates),
            "ruler_median_rewards": ruler_medians,
            "ruler_seat_median_rewards": ruler_seat_medians,
        }

    gates: dict[str, Any] = {}
    if "G0002" in generations and "G0004" in generations:
        baseline = generations["G0002"]
        candidate = generations["G0004"]
        ruler_differences = {
            ruler: candidate["ruler_median_rewards"][ruler]
            - baseline["ruler_median_rewards"][ruler]
            for ruler in baseline["ruler_median_rewards"]
        }
        baseline_sd = float(baseline["population_sd_aggregate_reward"])
        candidate_sd = float(candidate["population_sd_aggregate_reward"])
        skill_difference = float(candidate["median_aggregate_reward"]) - float(
            baseline["median_aggregate_reward"]
        )
        breadth_nonnegative = sum(value >= 0.0 for value in ruler_differences.values())
        stabilization = {
            "stability_pass": candidate_sd <= 0.8 * baseline_sd,
            "skill_guard_pass": skill_difference >= -0.030,
            "breadth_guard_pass": breadth_nonnegative >= 3,
            "candidate_to_baseline_sd_ratio": (
                candidate_sd / baseline_sd if baseline_sd > 0.0 else None
            ),
            "median_skill_difference": skill_difference,
            "ruler_median_differences": ruler_differences,
            "breadth_nonnegative_rulers": breadth_nonnegative,
        }
        stabilization["pass"] = all(
            bool(stabilization[key])
            for key in ("stability_pass", "skill_guard_pass", "breadth_guard_pass")
        )
        gates["stabilization"] = stabilization

    for generation_id, generation in generations.items():
        aggregates = [float(value) for value in generation["run_aggregates"].values()]
        weak_by_seat = generation["ruler_seat_median_rewards"].get("weak", {})
        final_gate = {
            "positive_median_pass": generation["median_aggregate_reward"] > 0.0,
            "two_positive_runs_pass": sum(value > 0.0 for value in aggregates) >= 2,
            "worst_run_pass": min(aggregates) > -0.050,
            "weak_all_seats_pass": bool(weak_by_seat)
            and all(value > 0.0 for value in weak_by_seat.values()),
        }
        final_gate["pass"] = all(final_gate.values())
        gates[f"final_{generation_id.lower()}"] = final_gate

    return {"runs": run_rows, "generations": generations, "gates": gates}


def _write_summary(path: Path) -> dict[str, Any]:
    """Read a raw matrix and write its deterministic adjacent summary JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize_payload(payload)
    summary_path = path.with_suffix(".summary.json")
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    print(f"summary={summary_path}", flush=True)
    return summary


def _run_cell(
    policy: Any,
    factory: Callable[[], BaseAgent],
    ruler: str,
    seats: int,
    tracks: list[Any],
    games: int,
    seed: int,
) -> EvalCell:
    """Evaluate one matched cell and return its raw-count report."""
    report = evaluate_policy(
        policy,
        opponents={ruler: factory},
        seat_counts=(seats,),
        splits={"b2_frozen": tracks},
        games_per_cell=games,
        seed=seed,
        device=torch.device("cpu"),
    )
    return report.cells[0]


def main(argv: list[str] | None = None) -> int:
    """Run or resume the full contender × ruler × seat benchmark matrix."""
    args = _parse_args(argv)
    if args.migrate_identities_only:
        if not args.out.exists():
            raise FileNotFoundError(args.out)
        changed = _migrate_result_file(args.out)
        print(f"identity migration changed={changed} out={args.out}", flush=True)
        return 0
    if args.summarize_only:
        _write_summary(args.out)
        return 0
    contenders = tuple(_registered_contender(agent_id) for agent_id in args.agents)
    if len({spec.agent_id for spec in contenders}) != len(contenders):
        raise ValueError("--agents contains duplicates")
    started = time.perf_counter()
    tracks = [generate_track(args.track_base + i) for i in range(args.tracks)]
    rulers = _rulers()
    cells = _load_existing(args.out, args.resume)
    expected = len(contenders) * len(rulers) * len(SEAT_COUNTS)
    print(
        f"B2 multi-ruler games/cell={args.games} tracks={args.tracks} "
        f"cells={expected} completed={len(cells)}",
        flush=True,
    )
    for contender_index, contender in enumerate(contenders):
        policy = load_policy(contender.checkpoint)
        for ruler_index, (ruler, factory) in enumerate(rulers.items()):
            for seat_index, seats in enumerate(SEAT_COUNTS):
                key = _cell_key(contender.agent_id, ruler, seats)
                if key in cells:
                    print(f"skip completed {key}", flush=True)
                    continue
                cell_started = time.perf_counter()
                # Equal seeds across contenders make every ruler/seat comparison
                # matched; offsets keep ruler and seat cells independent.
                cell = _run_cell(
                    policy,
                    factory,
                    ruler,
                    seats,
                    tracks,
                    args.games,
                    args.seed + ruler_index * 100 + seat_index * 10,
                )
                seconds = time.perf_counter() - cell_started
                row = {
                    "key": key,
                    "contender": contender.agent_id,
                    "recipe": contender.generation_id,
                    "training_seed": contender.seed,
                    "selected_checkpoint_id": contender.selected_checkpoint_id,
                    "checkpoint": contender.checkpoint,
                    "ruler": ruler,
                    "seats": seats,
                    "cell": asdict(cell),
                    "seconds": seconds,
                }
                cells[key] = row
                _persist(
                    args.out,
                    args=args,
                    contenders=contenders,
                    cells=cells,
                    started=started,
                    complete=False,
                )
                remaining = expected - len(cells)
                print(
                    f"cell {len(cells)}/{expected} {key} wins={cell.wins}/{cell.games} "
                    f"win={cell.win_rate:.3f} reward={cell.mean_placement_reward:+.3f} "
                    f"wilson=[{cell.wilson_lb:.3f},{cell.wilson_ub:.3f}] "
                    f"wall={seconds:.1f}s remaining={remaining}",
                    flush=True,
                )
    _persist(
        args.out,
        args=args,
        contenders=contenders,
        cells=cells,
        started=started,
        complete=len(cells) == expected,
    )
    print(
        f"B2 complete cells={len(cells)}/{expected} "
        f"wall={time.perf_counter() - started:.1f}s out={args.out}",
        flush=True,
    )
    _write_summary(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
