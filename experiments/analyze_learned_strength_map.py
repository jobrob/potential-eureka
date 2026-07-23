"""Consolidate the existing learned-policy evidence for investigation L0.

L0 is deliberately evidence-only.  It reads registered A8 artifacts, keeps
development and untouched-final results separate, and records which planned
stratifications cannot be reconstructed from the historical result schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatrixSpec:
    """Describe one frozen multi-ruler artifact and its evidence role."""

    artifact_id: str
    path: Path
    evidence_role: str
    expected_generations: tuple[str, ...]


MATRIX_SPECS = (
    MatrixSpec(
        "b2_development",
        Path("runs/a8_b2_multi_ruler.json"),
        "consumed_development",
        ("G0002", "G0003"),
    ),
    MatrixSpec(
        "s3_development",
        Path("runs/a8_s3_validation.json"),
        "consumed_development",
        ("G0002", "G0004"),
    ),
    MatrixSpec(
        "a8_untouched_final",
        Path("runs/a8_final_test.json"),
        "historical_untouched_final_do_not_tune",
        ("G0002",),
    ),
)

MILESTONE_DIRECTORIES = {
    "G0002": "a8_s1_5ep_seed{seed}",
    "G0003": "a8_s2_anchor_seed{seed}",
}
MILESTONES = (300_000, 500_000, 750_000, 1_000_000)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the repository root and deterministic output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/l0_learned_strength_map.json"),
    )
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and fail clearly on a different top-level type."""

    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 receipt for one consumed source artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_mapping(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Compute an equal-cell mean reward for every value of one dimension."""

    values: dict[str, list[float]] = {}
    for row in rows:
        label = str(row[key])
        values.setdefault(label, []).append(float(row["cell"]["mean_placement_reward"]))
    return {
        label: statistics.fmean(rewards)
        for label, rewards in sorted(values.items())
    }


def _summarize_matrix(root: Path, spec: MatrixSpec) -> dict[str, Any]:
    """Validate and summarize one historical multi-ruler matrix."""

    path = root / spec.path
    payload = _read_json_object(path)
    if payload.get("complete") is not True:
        raise ValueError(f"{path} is not marked complete")

    config = payload.get("config")
    cells = payload.get("cells")
    if not isinstance(config, dict) or not isinstance(cells, list):
        raise ValueError(f"{path} must contain object config and list cells")

    expected_cells = (
        len(config["contenders"])
        * len(config["rulers"])
        * len(config["seat_counts"])
    )
    if len(cells) != expected_cells:
        raise ValueError(f"{path} has {len(cells)} cells; expected {expected_cells}")
    keys = [str(row["key"]) for row in cells]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate cell keys")

    games_per_cell = int(config["games_per_cell"])
    if any(int(row["cell"]["games"]) != games_per_cell for row in cells):
        raise ValueError(f"{path} contains an incomplete game cell")

    generations = sorted({str(row["recipe"]) for row in cells})
    if tuple(generations) != spec.expected_generations:
        raise ValueError(
            f"{path} generations {generations} do not match {spec.expected_generations}"
        )

    run_rows: list[dict[str, Any]] = []
    for agent_id in sorted({str(row["contender"]) for row in cells}):
        agent_cells = [row for row in cells if str(row["contender"]) == agent_id]
        rewards = [float(row["cell"]["mean_placement_reward"]) for row in agent_cells]
        run_rows.append(
            {
                "agent_id": agent_id,
                "generation_id": str(agent_cells[0]["recipe"]),
                "training_seed": int(agent_cells[0]["training_seed"]),
                "selected_checkpoint_id": str(
                    agent_cells[0]["selected_checkpoint_id"]
                ),
                "aggregate_reward": statistics.fmean(rewards),
                "ruler_rewards": _mean_mapping(agent_cells, "ruler"),
                "seat_rewards": _mean_mapping(agent_cells, "seats"),
            }
        )

    generation_rows: dict[str, dict[str, Any]] = {}
    for generation_id in generations:
        generation_runs = [
            row for row in run_rows if row["generation_id"] == generation_id
        ]
        generation_rows[generation_id] = {
            "run_aggregates": {
                str(row["agent_id"]): float(row["aggregate_reward"])
                for row in generation_runs
            },
            "median_aggregate_reward": statistics.median(
                float(row["aggregate_reward"]) for row in generation_runs
            ),
            "population_sd_aggregate_reward": statistics.pstdev(
                float(row["aggregate_reward"]) for row in generation_runs
            ),
            "ruler_median_rewards": {
                ruler: statistics.median(
                    float(row["ruler_rewards"][ruler]) for row in generation_runs
                )
                for ruler in sorted(generation_runs[0]["ruler_rewards"])
            },
            "seat_median_rewards": {
                seat: statistics.median(
                    float(row["seat_rewards"][seat]) for row in generation_runs
                )
                for seat in sorted(generation_runs[0]["seat_rewards"], key=int)
            },
        }

    return {
        "artifact_id": spec.artifact_id,
        "path": spec.path.as_posix(),
        "sha256": _sha256(path),
        "evidence_role": spec.evidence_role,
        "complete": True,
        "track_seed_range": [
            int(config["track_base"]),
            int(config["track_base"]) + int(config["tracks"]) - 1,
        ],
        "game_seed": int(config["seed"]),
        "games_per_cell": games_per_cell,
        "cell_count": len(cells),
        "race_count": len(cells) * games_per_cell,
        "rulers": [str(value) for value in config["rulers"]],
        "seat_counts": [int(value) for value in config["seat_counts"]],
        "runs": run_rows,
        "generations": generation_rows,
    }


def _parse_milestone_markdown(path: Path) -> list[dict[str, Any]]:
    """Parse the compact A8 evaluation table stored beside each checkpoint."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 11 or cells[0] in {"opponent", "---"}:
            continue
        rows.append(
            {
                "opponent": cells[0],
                "split": cells[1],
                "seats": int(cells[2]),
                "games": int(cells[3]),
                "wins": int(cells[4]),
                "mean_placement_reward": float(cells[9]),
            }
        )
    opponents = {str(row["opponent"]) for row in rows}
    expected_rows = len(opponents) * 3
    if (
        len(rows) != expected_rows
        or {int(row["seats"]) for row in rows} != {2, 4, 6}
        or any(
            sum(row["opponent"] == opponent for row in rows) != 3
            for opponent in opponents
        )
    ):
        raise ValueError(
            f"{path} has an incomplete opponent-by-seat evaluation grid"
        )
    return rows


def _summarize_milestones(root: Path) -> dict[str, Any]:
    """Consolidate the registered G0002/G0003 checkpoint curves."""

    run_rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, str]] = []
    for generation_id, directory_pattern in MILESTONE_DIRECTORIES.items():
        for seed in range(3):
            for step in MILESTONES:
                relative = Path("runs") / directory_pattern.format(seed=seed)
                relative /= f"step_{step}.eval.md"
                path = root / relative
                cells = _parse_milestone_markdown(path)
                rewards = [float(row["mean_placement_reward"]) for row in cells]
                run_rows.append(
                    {
                        "agent_id": f"{generation_id}-R{seed:02d}@{step // 1000}K",
                        "generation_id": generation_id,
                        "training_seed": seed,
                        "step": step,
                        "aggregate_reward": statistics.fmean(rewards),
                        "opponent_rewards": {
                            opponent: statistics.fmean(
                                float(row["mean_placement_reward"])
                                for row in cells
                                if row["opponent"] == opponent
                            )
                            for opponent in sorted(
                                {str(row["opponent"]) for row in cells}
                            )
                        },
                        "seat_rewards": {
                            str(seats): statistics.fmean(
                                float(row["mean_placement_reward"])
                                for row in cells
                                if int(row["seats"]) == seats
                            )
                            for seats in sorted({int(row["seats"]) for row in cells})
                        },
                    }
                )
                source_receipts.append(
                    {"path": relative.as_posix(), "sha256": _sha256(path)}
                )

    generation_curves: dict[str, Any] = {}
    for generation_id in MILESTONE_DIRECTORIES:
        generation_runs = [
            row for row in run_rows if row["generation_id"] == generation_id
        ]
        per_step = {
            str(step): {
                "run_aggregates": {
                    str(row["training_seed"]): float(row["aggregate_reward"])
                    for row in generation_runs
                    if int(row["step"]) == step
                },
                "median_aggregate_reward": statistics.median(
                    float(row["aggregate_reward"])
                    for row in generation_runs
                    if int(row["step"]) == step
                ),
            }
            for step in MILESTONES
        }
        monotonic_runs = 0
        for seed in range(3):
            curve = [
                float(row["aggregate_reward"])
                for row in generation_runs
                if int(row["training_seed"]) == seed
            ]
            if all(after >= before for before, after in zip(curve, curve[1:])):
                monotonic_runs += 1
        generation_curves[generation_id] = {
            "steps": per_step,
            "monotonic_run_count": monotonic_runs,
            "run_count": 3,
        }

    return {
        "evidence_role": "legacy_milestone_diagnostic",
        "limitations": [
            "The G0002 and G0003 milestone grids use different ruler sets.",
            "The rulers use historical pre-V2 implementations.",
            "Each ruler-seat cell contains 50 races.",
            "G0003 selected final policies for R01 and R02 do not exactly equal @1000K.",
        ],
        "source_receipts": source_receipts,
        "runs": run_rows,
        "generation_curves": generation_curves,
    }


def _matched_generation_comparison(
    matrix: dict[str, Any], baseline: str, candidate: str
) -> dict[str, Any]:
    """Compare two generations by matching their registered training seed."""

    runs = matrix["runs"]
    paired_rows: list[dict[str, Any]] = []
    for seed in range(3):
        baseline_run = next(
            row
            for row in runs
            if row["generation_id"] == baseline and row["training_seed"] == seed
        )
        candidate_run = next(
            row
            for row in runs
            if row["generation_id"] == candidate and row["training_seed"] == seed
        )
        ruler_differences = {
            ruler: float(candidate_run["ruler_rewards"][ruler])
            - float(baseline_run["ruler_rewards"][ruler])
            for ruler in sorted(baseline_run["ruler_rewards"])
        }
        directions = {
            1 if difference > 0 else -1 if difference < 0 else 0
            for difference in ruler_differences.values()
        }
        paired_rows.append(
            {
                "training_seed": seed,
                "aggregate_difference": float(candidate_run["aggregate_reward"])
                - float(baseline_run["aggregate_reward"]),
                "ruler_differences": ruler_differences,
                "all_rulers_same_direction": len(directions) == 1,
            }
        )
    return {
        "artifact_id": matrix["artifact_id"],
        "baseline_generation_id": baseline,
        "candidate_generation_id": candidate,
        "paired_runs": paired_rows,
        "positive_pair_count": sum(
            row["aggregate_difference"] > 0 for row in paired_rows
        ),
    }


def build_l0_report(root: Path) -> dict[str, Any]:
    """Build the complete deterministic L0 evidence map."""

    matrices = [_summarize_matrix(root, spec) for spec in MATRIX_SPECS]
    milestones = _summarize_milestones(root)
    b2 = next(row for row in matrices if row["artifact_id"] == "b2_development")
    s3 = next(row for row in matrices if row["artifact_id"] == "s3_development")
    b2_comparison = _matched_generation_comparison(b2, "G0002", "G0003")
    s3_comparison = _matched_generation_comparison(s3, "G0002", "G0004")

    return {
        "schema_version": 1,
        "investigation_id": "L0",
        "status": "complete_existing_evidence_only",
        "policy_registry": {
            "index_path": "experiments/policy_registry/index.json",
            "index_sha256": _sha256(root / "experiments/policy_registry/index.json"),
            "generation_ids": ["G0002", "G0003", "G0004"],
            "generation_allocated": False,
        },
        "matrices": matrices,
        "milestones": milestones,
        "matched_comparisons": [b2_comparison, s3_comparison],
        "coverage": {
            "available": [
                "generation",
                "training_seed",
                "selected_checkpoint",
                "legacy_milestone",
                "ruler",
                "seat_count",
                "track_seed_band",
            ],
            "not_recoverable_from_existing_artifacts": [
                "per_race_starting_position",
                "per_track_outcomes",
                "track_family_outcomes",
            ],
            "reason": (
                "Historical multi-ruler artifacts persist one aggregate per "
                "contender-ruler-seat cell rather than per-race records."
            ),
        },
        "findings": {
            "total_existing_races": sum(
                int(matrix["race_count"]) for matrix in matrices
            ),
            "selected_policy_order_is_not_stable_across_seeds": (
                0 < b2_comparison["positive_pair_count"] < 3
            ),
            "checkpoint_progress_is_not_monotonic": any(
                curve["monotonic_run_count"] < curve["run_count"]
                for curve in milestones["generation_curves"].values()
            ),
            "g0003_b2_positive_matched_seeds": b2_comparison[
                "positive_pair_count"
            ],
            "g0004_s3_positive_matched_seeds": s3_comparison[
                "positive_pair_count"
            ],
            "untouched_final_remains_historical_only": True,
        },
        "minimum_live_benchmark": {
            "screen": {
                "rulers": [
                    "heuristic_weak_v2",
                    "static_search_v2",
                    "historical_g0002_r00_1000k",
                ],
                "seat_counts": [2, 4, 6],
                "games_per_cell": 100,
                "purpose": "routine milestone screen",
            },
            "promotion": {
                "rulers": [
                    "heuristic_weak_v2",
                    "heuristic_repaired_v2",
                    "static_search_v2",
                    "historical_g0002_r00_1000k",
                ],
                "seat_counts": [2, 4, 6],
                "games_per_cell": 200,
                "purpose": "recipe or substrate adoption gate",
            },
            "required_new_fields": [
                "track_seed",
                "track_family",
                "game_seed",
                "focal_starting_seat",
                "finish_position",
                "placement_reward",
            ],
            "data_role": (
                "Declare a fresh development band for the 32K comparison; "
                "do not reopen the A8 untouched final band."
            ),
            "rationale": [
                "Weak V2 preserves the basic competence floor.",
                "StaticSearchV2 supplies the most informative fixed challenge.",
                "The historical G0002 policy detects learned-style regression.",
                "Repaired V2 remains in promotion because static matchups are non-transitive.",
            ],
        },
        "next_decision": {
            "action": "design_registered_32k_learning_quality_comparison",
            "freeze_runtime_recipe": {
                "collector_mode": "native",
                "active_slots": 48,
                "ready_capacity": 288,
                "n_steps": 32_768,
                "refill_reserve": 1.30,
                "torch_threads": [8, 2],
            },
            "do_not_start": [
                "new training before the comparison design and registry allocation",
                "Option C CUDA work",
                "reuse of the A8 untouched final band",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write the deterministic L0 evidence artifact atomically."""

    args = _parse_args(argv)
    root = args.root.resolve()
    output = args.out if args.out.is_absolute() else root / args.out
    report = build_l0_report(root)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        f"L0 complete: {report['findings']['total_existing_races']} existing races; "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
