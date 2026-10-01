#!/usr/bin/env python
"""Analyze Question B 32K learning-quality evaluation artifacts.

Validates matrix completeness, computes the frozen adoption gates (Q06),
reports descriptive paired bootstrap intervals, and emits failure-localization
tables. Never invents race results: missing cells yield ``NO DECISION``.

G0005 is not allocated or mutated here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from experiments.run_32k_learning_quality import QUESTION_ID

AGGREGATE_TOLERANCE = -0.030
MATCHED_TOLERANCE = -0.030
MATCHED_FLOOR = -0.100
BREADTH_TOLERANCE = -0.050
COLLAPSE_ENTROPY_FLOOR = 0.40
COLLAPSE_ENT_COEF_MAX = 0.10
COLLAPSE_STREAK = 3
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 32_005


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically through a temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_matrix(path: Path) -> dict[str, Any]:
    """Load one evaluation matrix JSON artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"matrix root must be an object: {path}")
    return payload


def validate_completeness(payload: dict[str, Any]) -> dict[str, Any]:
    """Check cell count, unique race coordinates, and required row fields."""
    cells = payload.get("cells", [])
    config = payload.get("config", {})
    contenders = config.get("contenders", [])
    rulers = config.get("rulers", [])
    seats = config.get("seat_counts", [])
    games = int(config.get("games_per_cell", 0))
    expected_cells = len(contenders) * len(rulers) * len(seats)
    errors: list[str] = []
    if len(cells) != expected_cells:
        errors.append(
            f"cell count {len(cells)} != expected {expected_cells}"
        )
    if not payload.get("complete", False):
        errors.append("payload.complete is not true")
    if payload.get("question_id") != QUESTION_ID:
        errors.append(f"question_id mismatch: {payload.get('question_id')!r}")

    seen_keys: set[str] = set()
    for cell in cells:
        key = str(cell.get("key"))
        if key in seen_keys:
            errors.append(f"duplicate cell key {key}")
        seen_keys.add(key)
        rows = cell.get("raw_rows", [])
        if len(rows) != games:
            errors.append(f"{key}: raw_rows={len(rows)} != games={games}")
        coord_ids = [row.get("race_coordinate_id") for row in rows]
        if len(coord_ids) != len(set(coord_ids)):
            errors.append(f"{key}: duplicate race_coordinate_id values")
        for row in rows:
            for field in (
                "contender_agent_id",
                "ruler_id",
                "seat_count",
                "track_seed",
                "game_seed",
                "focal_starting_seat",
                "placement_reward",
                "track_sha256",
            ):
                if field not in row:
                    errors.append(f"{key}: missing raw-row field {field}")
                    break

    return {
        "ok": not errors,
        "expected_cells": expected_cells,
        "actual_cells": len(cells),
        "errors": errors,
    }


def _cell_reward(cell: dict[str, Any]) -> float:
    """Return mean placement reward, preferring the cell summary field."""
    if "mean_placement_reward" in cell:
        return float(cell["mean_placement_reward"])
    rows = cell.get("raw_rows", [])
    if not rows:
        raise ValueError("cell has no reward evidence")
    return statistics.fmean(float(row["placement_reward"]) for row in rows)


def summarize_runs(payload: dict[str, Any]) -> dict[str, Any]:
    """Equal-weight means across ruler×seat cells, then recipe medians."""
    cells = payload.get("cells", [])
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_agent.setdefault(str(cell["contender"]), []).append(cell)

    run_rows: list[dict[str, Any]] = []
    for agent_id, agent_cells in sorted(by_agent.items()):
        rewards = [_cell_reward(cell) for cell in agent_cells]
        ruler_ids = sorted({str(cell["ruler"]) for cell in agent_cells})
        seat_ids = sorted({int(cell["seats"]) for cell in agent_cells})
        ruler_means = {
            ruler: statistics.fmean(
                _cell_reward(cell) for cell in agent_cells if cell["ruler"] == ruler
            )
            for ruler in ruler_ids
        }
        seat_means = {
            str(seats): statistics.fmean(
                _cell_reward(cell) for cell in agent_cells if int(cell["seats"]) == seats
            )
            for seats in seat_ids
        }
        generation_id = str(
            agent_cells[0].get("generation_id")
            or agent_id.split("-", 1)[0]
        )
        run_rows.append(
            {
                "agent_id": agent_id,
                "generation_id": generation_id,
                "training_seed": _training_seed(agent_id, agent_cells),
                "aggregate_reward": statistics.fmean(rewards),
                "ruler_rewards": ruler_means,
                "seat_rewards": seat_means,
            }
        )

    generations: dict[str, dict[str, Any]] = {}
    for generation_id in sorted({row["generation_id"] for row in run_rows}):
        generation_runs = [
            row for row in run_rows if row["generation_id"] == generation_id
        ]
        aggregates = [float(row["aggregate_reward"]) for row in generation_runs]
        ruler_ids = sorted(generation_runs[0]["ruler_rewards"])
        seat_ids = sorted(generation_runs[0]["seat_rewards"])
        generations[generation_id] = {
            "run_aggregates": {
                str(row["agent_id"]): float(row["aggregate_reward"])
                for row in generation_runs
            },
            "run_by_seed": {
                str(row["training_seed"]): float(row["aggregate_reward"])
                for row in generation_runs
            },
            "median_aggregate_reward": statistics.median(aggregates),
            "ruler_median_rewards": {
                ruler: statistics.median(
                    float(row["ruler_rewards"][ruler]) for row in generation_runs
                )
                for ruler in ruler_ids
            },
            "seat_median_rewards": {
                seat: statistics.median(
                    float(row["seat_rewards"][seat]) for row in generation_runs
                )
                for seat in seat_ids
            },
        }
    return {"runs": run_rows, "generations": generations}


def _training_seed(agent_id: str, cells: list[dict[str, Any]]) -> int:
    """Recover the training seed from raw rows or the agent id suffix."""
    for cell in cells:
        rows = cell.get("raw_rows") or []
        if rows and "contender_training_seed" in rows[0]:
            return int(rows[0]["contender_training_seed"])
    if "-R" in agent_id:
        try:
            return int(agent_id.rsplit("-R", 1)[1][:2])
        except ValueError:
            return -1
    return -1


def paired_seed_deltas(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    """Matched-seed aggregate deltas between candidate and baseline generations."""
    deltas = []
    for seed, candidate_value in sorted(
        candidate["run_by_seed"].items(), key=lambda item: int(item[0])
    ):
        if seed not in baseline["run_by_seed"]:
            continue
        baseline_value = float(baseline["run_by_seed"][seed])
        deltas.append(
            {
                "training_seed": int(seed),
                "candidate": float(candidate_value),
                "baseline": baseline_value,
                "delta": float(candidate_value) - baseline_value,
            }
        )
    return deltas


def evaluate_gates(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    collapse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen Q06 adoption guards to recipe-level summaries."""
    aggregate_delta = float(candidate["median_aggregate_reward"]) - float(
        baseline["median_aggregate_reward"]
    )
    matched = paired_seed_deltas(candidate, baseline)
    matched_ok = (
        len(matched) >= 2
        and sum(row["delta"] >= MATCHED_TOLERANCE for row in matched) >= 2
        and all(row["delta"] >= MATCHED_FLOOR for row in matched)
    )
    ruler_deltas = {
        ruler: float(candidate["ruler_median_rewards"][ruler])
        - float(baseline["ruler_median_rewards"][ruler])
        for ruler in baseline["ruler_median_rewards"]
        if ruler in candidate["ruler_median_rewards"]
    }
    seat_deltas = {
        seat: float(candidate["seat_median_rewards"][seat])
        - float(baseline["seat_median_rewards"][seat])
        for seat in baseline["seat_median_rewards"]
        if seat in candidate["seat_median_rewards"]
    }
    breadth_ok = bool(ruler_deltas) and bool(seat_deltas) and all(
        value >= BREADTH_TOLERANCE for value in ruler_deltas.values()
    ) and all(value >= BREADTH_TOLERANCE for value in seat_deltas.values())

    collapse_ok = True
    collapse_detail: dict[str, Any] = {"applied": False}
    if collapse is not None:
        collapse_ok = bool(collapse.get("pass", False))
        collapse_detail = collapse

    finite_ok = all(
        math.isfinite(value) for value in candidate["run_aggregates"].values()
    ) and all(math.isfinite(value) for value in baseline["run_aggregates"].values())

    gates = {
        "aggregate": {
            "pass": aggregate_delta >= AGGREGATE_TOLERANCE,
            "delta": aggregate_delta,
            "tolerance": AGGREGATE_TOLERANCE,
        },
        "matched_runs": {
            "pass": matched_ok,
            "deltas": matched,
            "tolerance": MATCHED_TOLERANCE,
            "floor": MATCHED_FLOOR,
        },
        "breadth": {
            "pass": breadth_ok,
            "ruler_deltas": ruler_deltas,
            "seat_deltas": seat_deltas,
            "tolerance": BREADTH_TOLERANCE,
        },
        "collapse": {
            "pass": collapse_ok and finite_ok,
            "finite_values": finite_ok,
            "detail": collapse_detail,
        },
    }
    gates["all_pass"] = all(bool(gates[name]["pass"]) for name in gates)
    return gates


def collapse_guard_from_records(
    records_by_run: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Fail when rollout_entropy stays below floor while ent_coef is maxed.

    Looks for three consecutive completed updates with
    ``rollout_entropy < 0.40`` and ``ent_coef`` (or ``entropy_coef``) already
    at 0.10. ``update_entropy`` does not count.
    """
    failures: list[dict[str, Any]] = []
    for run_id, records in records_by_run.items():
        streak = 0
        for record in records:
            entropy = float(record.get("rollout_entropy", float("nan")))
            ent_coef = float(
                record.get(
                    "ent_coef",
                    record.get("entropy_coef", record.get("current_ent_coef", 0.0)),
                )
            )
            if (
                math.isfinite(entropy)
                and entropy < COLLAPSE_ENTROPY_FLOOR
                and ent_coef >= COLLAPSE_ENT_COEF_MAX - 1e-12
            ):
                streak += 1
            else:
                streak = 0
            if streak >= COLLAPSE_STREAK:
                failures.append(
                    {
                        "run_id": run_id,
                        "streak": streak,
                        "iteration": int(record.get("iteration", -1)),
                        "rollout_entropy": entropy,
                        "ent_coef": ent_coef,
                    }
                )
                break
    return {
        "pass": not failures,
        "applied": True,
        "failures": failures,
        "rule": (
            f"{COLLAPSE_STREAK} consecutive updates with "
            f"rollout_entropy < {COLLAPSE_ENTROPY_FLOOR} while "
            f"ent_coef >= {COLLAPSE_ENT_COEF_MAX}"
        ),
    }


def paired_bootstrap_interval(
    paired_deltas: list[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Deterministic 95% bootstrap interval over paired race/run deltas."""
    if not paired_deltas:
        return {"low": float("nan"), "high": float("nan"), "mean": float("nan")}
    import numpy as np

    rng = np.random.default_rng(seed)
    values = np.asarray(paired_deltas, dtype=np.float64)
    draws = []
    n = len(values)
    for _ in range(samples):
        index = rng.integers(0, n, size=n)
        draws.append(float(values[index].mean()))
    draws.sort()
    low = draws[int(0.025 * (samples - 1))]
    high = draws[int(0.975 * (samples - 1))]
    return {
        "mean": float(values.mean()),
        "low": low,
        "high": high,
        "samples": float(samples),
    }


def failure_localization(gates: dict[str, Any]) -> dict[str, Any]:
    """Rank investigation branches for an evidence-valid FAIL (Q08)."""
    if gates.get("all_pass"):
        return {"status": "PASS", "branches": []}
    branches: list[str] = []
    matched = gates["matched_runs"]["deltas"]
    if not gates["aggregate"]["pass"]:
        branches.append(
            "All seeds/rulers weak: inspect update cadence, first-update behavior, "
            "advantage accounting, opponent availability."
        )
    if matched and sum(row["delta"] < MATCHED_TOLERANCE for row in matched) == 1:
        branches.append(
            "One coherent seed regression: inspect training seed, checkpoint path, "
            "opponent-source sequence, trajectory divergence."
        )
    if not gates["breadth"]["pass"]:
        weak_rulers = [
            ruler
            for ruler, delta in gates["breadth"]["ruler_deltas"].items()
            if delta < BREADTH_TOLERANCE
        ]
        weak_seats = [
            seat
            for seat, delta in gates["breadth"]["seat_deltas"].items()
            if delta < BREADTH_TOLERANCE
        ]
        branches.append(
            "Ruler/seat slice weak: "
            f"rulers={weak_rulers} seats={weak_seats}. "
            "Replay decisions on the failing style and start positions."
        )
    if not gates["collapse"]["pass"]:
        branches.append(
            "Collapse guard failed: inspect rollout_entropy timeline, controller "
            "saturation, value/policy loss, and skill consequence."
        )
    return {"status": "FAIL", "branches": branches}


def analyze_promotion(
    payload: dict[str, Any],
    *,
    candidate_generation: str,
    baseline_generation: str = "G0002",
    collapse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full promotion analysis with disposition PASS / FAIL / NO DECISION."""
    completeness = validate_completeness(payload)
    if not completeness["ok"]:
        return {
            "disposition": "NO DECISION",
            "question_id": QUESTION_ID,
            "g0005_allocated": False,
            "completeness": completeness,
            "note": "Repair evidence production before making a learning claim.",
        }
    summary = summarize_runs(payload)
    if (
        candidate_generation not in summary["generations"]
        or baseline_generation not in summary["generations"]
    ):
        return {
            "disposition": "NO DECISION",
            "question_id": QUESTION_ID,
            "completeness": completeness,
            "summary": summary,
            "note": "Candidate or baseline generation missing from matrix.",
        }
    candidate = summary["generations"][candidate_generation]
    baseline = summary["generations"][baseline_generation]
    gates = evaluate_gates(
        candidate=candidate, baseline=baseline, collapse=collapse
    )
    matched_deltas = [row["delta"] for row in gates["matched_runs"]["deltas"]]
    interval = paired_bootstrap_interval(matched_deltas)
    localization = failure_localization(gates)
    disposition = "PASS" if gates["all_pass"] else "FAIL"
    return {
        "disposition": disposition,
        "question_id": QUESTION_ID,
        "g0005_allocated": False,
        "completeness": completeness,
        "summary": summary,
        "gates": gates,
        "descriptive_bootstrap_95": interval,
        "failure_localization": localization,
        "note": (
            "Confidence intervals are descriptive and do not override point gates. "
            "Three seeds make PASS a pilot, not a tight statistical claim."
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse analysis inputs for the Question B promotion artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promotion",
        type=Path,
        default=Path("runs/32k_learning_quality/eval/promotion.json"),
    )
    parser.add_argument(
        "--candidate-generation",
        default="G0005",
        help=(
            "Candidate generation id in the matrix (default G0005). "
            "Does not allocate or register that id."
        ),
    )
    parser.add_argument("--baseline-generation", default="G0002")
    parser.add_argument(
        "--training-records",
        type=Path,
        default=None,
        help="Optional JSON map of run_id -> iteration records for collapse guard.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/32k_learning_quality/analysis.json"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gate constants and exit without reading race artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Analyze a promotion matrix or print the frozen gate contract."""
    args = _parse_args(argv)
    if args.dry_run:
        plan = {
            "question_id": QUESTION_ID,
            "g0005_allocated": False,
            "gates": {
                "aggregate_tolerance": AGGREGATE_TOLERANCE,
                "matched_tolerance": MATCHED_TOLERANCE,
                "matched_floor": MATCHED_FLOOR,
                "breadth_tolerance": BREADTH_TOLERANCE,
                "collapse": {
                    "rollout_entropy_floor": COLLAPSE_ENTROPY_FLOOR,
                    "ent_coef_max": COLLAPSE_ENT_COEF_MAX,
                    "streak": COLLAPSE_STREAK,
                },
            },
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "note": "No artifacts read. G0005 not allocated.",
        }
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return 0
    if not args.promotion.exists():
        raise FileNotFoundError(
            f"promotion artifact missing: {args.promotion}. "
            "Run eval_32k_learning_quality.py --execute after training, or pass "
            "--dry-run to inspect gate constants only."
        )
    payload = load_matrix(args.promotion)
    collapse = None
    if args.training_records is not None:
        records = json.loads(args.training_records.read_text(encoding="utf-8"))
        collapse = collapse_guard_from_records(records)
    report = analyze_promotion(
        payload,
        candidate_generation=args.candidate_generation,
        baseline_generation=args.baseline_generation,
        collapse=collapse,
    )
    _atomic_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    from _runlog import run_main

    run_main("analyze_32k_learning_quality", main)
