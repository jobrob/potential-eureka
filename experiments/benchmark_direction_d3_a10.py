#!/usr/bin/env python
"""Bounded A10 native full-training worker/capacity scaling gate.

Hypothesis (0.65 confidence before measurement): the integrated native CPU
collector sustains at least 4,000 trained transitions/s, with a best stable
configuration reaching the 5,000/s adoption target.  This disposable dev
benchmark does not save a learned policy or allocate a policy generation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter
from typing import Any

import numpy as np
import torch

from heat.ml.selfplay.phase1 import A8Config, train_selfplay_a8

if __package__:
    from experiments.verify_direction_d3_native import native_source_tree_identity
else:
    from verify_direction_d3_native import native_source_tree_identity


def _parse_args() -> argparse.Namespace:
    """Parse the fixed worker grid and bounded runtime controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--torch-interop-threads", type=int, default=8)
    parser.add_argument("--refill-reserve-factor", type=float, default=1.3)
    parser.add_argument("--timeout-seconds", type=float, default=7_200.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a10_native_benchmark.json"),
    )
    args = parser.parse_args()
    if any(worker < 1 for worker in args.workers):
        parser.error("workers must all be positive")
    if args.repeats < 1 or args.warmups < 0 or args.transitions < 1 or args.n_steps < 1:
        parser.error("repeats, transitions, and n-steps must be positive")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        parser.error("Torch thread counts must be positive")
    if not 1.0 <= args.refill_reserve_factor <= 2.0:
        parser.error("refill reserve factor must be in [1, 2]")
    if args.timeout_seconds <= 0.0 or args.heartbeat_seconds <= 0.0:
        parser.error("timeouts must be positive")
    return args


def _action_histogram(records: list[dict[str, float]]) -> dict[int, int]:
    """Aggregate dynamic action-batch histogram fields."""
    histogram: dict[int, int] = {}
    for record in records:
        for key, value in record.items():
            if key.startswith("action_batch_") and key.endswith("_calls"):
                size = int(key[len("action_batch_") : -len("_calls")])
                histogram[size] = histogram.get(size, 0) + int(value)
    return {size: count for size, count in sorted(histogram.items()) if count}


def _percentile(histogram: dict[int, int], quantile: float) -> int:
    """Return the nearest-rank batch size for a call-count histogram."""
    target = max(1, int(np.ceil(sum(histogram.values()) * quantile)))
    cumulative = 0
    for size, count in histogram.items():
        cumulative += count
        if cumulative >= target:
            return size
    return 0


def _run_once(
    *,
    workers: int,
    repeat: int,
    args: argparse.Namespace,
    deadline: float,
) -> dict[str, Any]:
    """Measure one matched production-recipe repeat with progress heartbeats."""
    ready_capacity = workers * 6
    config = A8Config(
        total_timesteps=args.transitions,
        n_steps=args.n_steps,
        batch_size=256,
        n_epochs=5,
        hidden_sizes=(256, 256),
        seat_counts=(2, 3, 4, 5, 6),
        seed=args.seed,
        device="cpu",
        collector_mode="native",
        native_workers=workers,
        native_ready_capacity=ready_capacity,
        native_refill_reserve_factor=args.refill_reserve_factor,
        head="masked",
        encoder="flat",
        margin_coef=0.0,
        anchor_share=0.0,
        stage1_enabled=False,
    )
    started = perf_counter()
    last_heartbeat = started
    print(
        f"A10 workers={workers} repeat={repeat + 1}/{args.repeats} started",
        flush=True,
    )

    def progress(iteration: int, record: dict[str, float]) -> None:
        """Enforce the campaign deadline and expose forward progress."""
        nonlocal last_heartbeat
        now = perf_counter()
        if now > deadline:
            raise TimeoutError("A10 scaling benchmark exceeded its two-hour cap")
        if now - last_heartbeat >= args.heartbeat_seconds:
            print(
                f"A10 workers={workers} repeat={repeat + 1}/{args.repeats} "
                f"iteration={iteration} trained={int(record['steps'])} "
                f"elapsed={now - started:.1f}s",
                flush=True,
            )
            last_heartbeat = now

    _policy, records = train_selfplay_a8(
        config, on_iteration=progress, profile=True
    )
    wall_seconds = perf_counter() - started
    recorded = int(records[-1]["steps"])
    rollout_seconds = sum(row["rollout_seconds"] for row in records)
    histogram = _action_histogram(records)
    result: dict[str, Any] = {
        "workers": workers,
        "ready_capacity": ready_capacity,
        "repeat": repeat,
        "seed": args.seed,
        "requested_transitions": args.transitions,
        "recorded_transitions": recorded,
        "iterations": len(records),
        "games": int(records[-1]["total_games"]),
        "wall_seconds": wall_seconds,
        "rollout_seconds": rollout_seconds,
        "inference_seconds": sum(row["inference_seconds"] for row in records),
        "update_seconds": sum(row["update_seconds"] for row in records),
        "prepare_seconds": sum(row["prepare_seconds"] for row in records),
        "native_admission_batches": int(
            sum(row.get("native_admission_batches", 0.0) for row in records)
        ),
        "native_refill_games": int(
            sum(row.get("native_refill_games", 0.0) for row in records)
        ),
        "native_peak_active_slots": int(
            max(row.get("native_peak_active_slots", 0.0) for row in records)
        ),
        "trained_transitions_per_second": recorded / wall_seconds,
        "rollout_transitions_per_second": recorded / rollout_seconds,
        "mean_action_batch": (
            sum(row["action_inference_rows"] for row in records)
            / sum(row["action_inference_calls"] for row in records)
        ),
        "p50_action_batch": _percentile(histogram, 0.50),
        "p95_action_batch": _percentile(histogram, 0.95),
        "max_action_batch": max(histogram, default=0),
        "action_batch_histogram": histogram,
        "manifest_overshoot": recorded - args.transitions,
    }
    print(
        f"A10 workers={workers} repeat={repeat + 1}/{args.repeats} "
        f"trained={result['trained_transitions_per_second']:.1f}/s "
        f"mean_batch={result['mean_action_batch']:.2f}",
        flush=True,
    )
    return result


def main() -> int:
    """Run the bounded grid, write its receipt, and apply A10's stop rules."""
    args = _parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    started = perf_counter()
    deadline = started + args.timeout_seconds
    repeats: list[dict[str, Any]] = []
    for workers in args.workers:
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.transitions = args.n_steps
        for warmup in range(args.warmups):
            _run_once(
                workers=workers,
                repeat=-(warmup + 1),
                args=warmup_args,
                deadline=deadline,
            )
        for repeat in range(args.repeats):
            repeats.append(
                _run_once(
                    workers=workers,
                    repeat=repeat,
                    args=args,
                    deadline=deadline,
                )
            )

    summaries: list[dict[str, Any]] = []
    for workers in args.workers:
        rows = [row for row in repeats if row["workers"] == workers]
        rates = [float(row["trained_transitions_per_second"]) for row in rows]
        median_rate = statistics.median(rates)
        spread = (max(rates) - min(rates)) / median_rate
        summaries.append(
            {
                "workers": workers,
                "ready_capacity": workers * 6,
                "median_trained_transitions_per_second": median_rate,
                "relative_repeat_spread": spread,
                "median_rollout_transitions_per_second": statistics.median(
                    float(row["rollout_transitions_per_second"]) for row in rows
                ),
                "median_mean_action_batch": statistics.median(
                    float(row["mean_action_batch"]) for row in rows
                ),
            }
        )
    best = max(
        summaries,
        key=lambda row: float(row["median_trained_transitions_per_second"]),
    )
    rate = float(best["median_trained_transitions_per_second"])
    stable = float(best["relative_repeat_spread"]) < 0.05
    decision = (
        "adopt_candidate"
        if rate >= 5_000.0 and stable
        else "continue_or_redesign"
        if rate >= 4_000.0
        else "stop"
    )
    artifact = {
        "schema_version": 1,
        "stage": "D3_A10_native_scaling_benchmark",
        "status": "pass",
        "decision": decision,
        "hypothesis": (
            "The integrated native CPU collector sustains at least 4,000 trained "
            "transitions/s and can reach 5,000/s with under 5% repeat spread."
        ),
        "confidence_before_run": 0.65,
        "scope": {
            "worker_grid": args.workers,
            "repeats_per_configuration": args.repeats,
            "discarded_warmups_per_configuration": args.warmups,
            "requested_transitions_per_repeat": args.transitions,
            "n_steps": args.n_steps,
            "refill_reserve_factor": args.refill_reserve_factor,
            "timeout_seconds": args.timeout_seconds,
            "recipe": "A8 flat/masked, hidden 256x256, PPO 5 epochs, batch 256",
        },
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "source_identity": native_source_tree_identity(),
        },
        "repeats": repeats,
        "summaries": summaries,
        "best": best,
        "continue_gate_pass": rate >= 4_000.0,
        "adoption_rate_pass": rate >= 5_000.0,
        "repeatability_pass": stable,
        "elapsed_seconds": perf_counter() - started,
        "known_architecture_limit": (
            "NativePool workers currently validate manifests; active native states "
            "are advanced by the Python coordinator thread between batched policy calls."
        ),
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
                "best": best,
                "elapsed_seconds": artifact["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if decision != "stop" else 2


if __name__ == "__main__":
    raise SystemExit(main())
