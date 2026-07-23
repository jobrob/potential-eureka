#!/usr/bin/env python
"""Analyze the completed A10 soak without running more training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
import torch

from heat.ml.selfplay.training_state import load_training_state


def _parse_args() -> argparse.Namespace:
    """Parse the immutable checkpoint and diagnostic receipt paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/direction_d/d3_a10_native_soak_state.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a10_soak_diagnostic.json"),
    )
    return parser.parse_args()


def _load_verified(path: Path) -> dict[str, Any]:
    """Load the historical run through its stored production identity gates."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint root must be a dictionary")
    return load_training_state(
        path,
        expected_recipe_sha256=str(raw["recipe_sha256"]),
        campaign_id=str(raw["campaign_id"]),
        source_identity=str(raw["source_identity"]),
        anchor_identity=raw.get("anchor_identity"),
    )


def _opponent_sources(rows: list[dict[str, float]]) -> list[str]:
    """Recover each iteration's opponent source from cumulative counters."""
    previous = {"current": 0, "snapshot": 0, "anchor": 0}
    sources: list[str] = []
    for row in rows:
        changed = []
        for source in previous:
            key = f"opponent_{source}_iterations"
            current = int(row[key])
            if current > previous[source]:
                changed.append(source)
            previous[source] = current
        if len(changed) != 1:
            raise ValueError("each iteration must advance one opponent counter")
        sources.append(changed[0])
    return sources


def _aggregate(rows: Iterable[dict[str, float]]) -> dict[str, Any]:
    """Aggregate one diagnostic group using actual recorded rows."""
    selected = list(rows)
    trained_rows = sum(int(row["n_recorded"]) for row in selected)
    iteration_seconds = sum(float(row["iteration_seconds"]) for row in selected)
    rollout_seconds = sum(float(row["rollout_seconds"]) for row in selected)
    update_seconds = sum(float(row["update_seconds"]) for row in selected)
    inference_seconds = sum(float(row["inference_seconds"]) for row in selected)
    calls = sum(float(row["action_inference_calls"]) for row in selected)
    inference_rows = sum(float(row["action_inference_rows"]) for row in selected)
    calls_1 = sum(float(row["action_batch_1_calls"]) for row in selected)
    calls_le_4 = sum(
        sum(float(row[f"action_batch_{size}_calls"]) for size in range(1, 5))
        for row in selected
    )
    return {
        "iterations": len(selected),
        "first_iteration": int(selected[0]["iteration"]),
        "last_iteration": int(selected[-1]["iteration"]),
        "trained_rows": trained_rows,
        "iteration_seconds": iteration_seconds,
        "trained_rows_per_second": trained_rows / iteration_seconds,
        "rollout_microseconds_per_row": rollout_seconds * 1e6 / trained_rows,
        "update_microseconds_per_row": update_seconds * 1e6 / trained_rows,
        "inference_microseconds_per_row": inference_seconds * 1e6 / trained_rows,
        "mean_action_batch": inference_rows / calls,
        "batch_1_call_share": calls_1 / calls,
        "batch_le_4_call_share": calls_le_4 / calls,
        "mean_rollout_overshoot": statistics.mean(
            float(row["rollout_overshoot"]) for row in selected
        ),
        "mean_entropy": statistics.mean(float(row["entropy"]) for row in selected),
        "seat_counts": {
            str(seats): sum(int(row["seat_count"]) == seats for row in selected)
            for seats in range(2, 7)
        },
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    """Return Pearson correlation when both inputs have non-zero variance."""
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> int:
    """Write a compact time-, seat-, and batching-correlation receipt."""
    args = _parse_args()
    checkpoint = _load_verified(args.checkpoint)
    raw_rows = checkpoint["records"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("checkpoint has no iteration records")
    rows = [{str(key): float(value) for key, value in row.items()} for row in raw_rows]
    sources = _opponent_sources(rows)

    index_blocks = np.array_split(np.arange(len(rows)), 6)
    six_blocks = [
        _aggregate([rows[int(index)] for index in indices])
        for indices in index_blocks
    ]
    source_breakdown_by_block = []
    for indices in index_blocks:
        block: dict[str, Any] = {}
        for source in ("current", "snapshot"):
            selected = [
                rows[int(index)]
                for index in indices
                if sources[int(index)] == source
            ]
            if selected:
                block[source] = _aggregate(selected)
        source_breakdown_by_block.append(block)
    best = max(six_blocks, key=lambda block: block["trained_rows_per_second"])
    worst = min(six_blocks, key=lambda block: block["trained_rows_per_second"])
    comparison = {
        "rate_change": (
            worst["trained_rows_per_second"] / best["trained_rows_per_second"] - 1.0
        ),
        "rollout_cost_change": (
            worst["rollout_microseconds_per_row"]
            / best["rollout_microseconds_per_row"]
            - 1.0
        ),
        "update_cost_change": (
            worst["update_microseconds_per_row"]
            / best["update_microseconds_per_row"]
            - 1.0
        ),
        "inference_cost_change": (
            worst["inference_microseconds_per_row"]
            / best["inference_microseconds_per_row"]
            - 1.0
        ),
        "mean_action_batch_change": (
            worst["mean_action_batch"] / best["mean_action_batch"] - 1.0
        ),
        "batch_1_share_change": (
            worst["batch_1_call_share"] / best["batch_1_call_share"] - 1.0
        ),
    }

    by_seat: dict[str, Any] = {}
    for seats in range(2, 7):
        by_seat[str(seats)] = _aggregate(
            row for row in rows if int(row["seat_count"]) == seats
        )
    by_source_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row, source in zip(rows, sources, strict=True):
        by_source_rows[source].append(row)
    by_source = {
        source: _aggregate(group) for source, group in sorted(by_source_rows.items())
    }

    per_iteration_rate = [
        row["n_recorded"] / row["iteration_seconds"] for row in rows
    ]
    metrics = {
        "iteration": [row["iteration"] for row in rows],
        "seat_count": [row["seat_count"] for row in rows],
        "recorded_rows": [row["n_recorded"] for row in rows],
        "mean_action_batch": [row["mean_action_batch"] for row in rows],
        "entropy": [row["entropy"] for row in rows],
        "snapshot_opponent": [1.0 if source == "snapshot" else 0.0 for source in sources],
    }
    correlations = {
        name: _correlation(values, per_iteration_rate)
        for name, values in metrics.items()
    }
    slowest = sorted(
        (
            {
                "iteration": int(row["iteration"]),
                "seat_count": int(row["seat_count"]),
                "opponent_source": source,
                "trained_rows_per_second": rate,
                "iteration_seconds": row["iteration_seconds"],
                "mean_action_batch": row["mean_action_batch"],
                "entropy": row["entropy"],
            }
            for row, source, rate in zip(rows, sources, per_iteration_rate, strict=True)
        ),
        key=lambda item: item["trained_rows_per_second"],
    )[:20]

    system_wide = (
        comparison["rollout_cost_change"] > 0.15
        and comparison["update_cost_change"] > 0.15
    )
    batching_explains = abs(comparison["mean_action_batch_change"]) > 0.15
    decision = (
        "test_system_contention_and_thermal_hypothesis"
        if system_wide and not batching_explains
        else "investigate_collector_workload_change"
    )
    artifact = {
        "schema_version": 1,
        "stage": "D3_A10_soak_posthoc_diagnostic",
        "status": "complete",
        "decision": decision,
        "hypothesis": (
            "The sustained slowdown is a system-wide compute effect rather than "
            "a change in native policy-batch shape or seat/opponent mix."
        ),
        "confidence_before_analysis": 0.80,
        "checkpoint": str(args.checkpoint),
        "overall": _aggregate(rows),
        "six_equal_iteration_blocks": six_blocks,
        "opponent_source_by_block": source_breakdown_by_block,
        "best_to_worst_comparison": comparison,
        "by_seat_count": by_seat,
        "by_opponent_source": by_source,
        "correlation_with_iteration_rate": correlations,
        "slowest_iterations": slowest,
        "interpretation": {
            "system_wide_cost_increase": system_wide,
            "policy_batch_shape_explains_slowdown": batching_explains,
            "note": (
                "A causal diagnosis still requires a bounded continuous replay "
                "with durable CPU, memory, and clock telemetry."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": decision,
                "best_to_worst_comparison": comparison,
                "correlations": correlations,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
