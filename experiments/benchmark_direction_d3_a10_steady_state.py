#!/usr/bin/env python
"""Compare A10 slot widths after the normal snapshot pool reaches steady state.

Hypothesis (0.55 confidence): 48 active slots recover enough live/snapshot
policy batch width to improve sustained trained throughput by at least 5% over
the selected 32-slot arm and reach the 5,000/s adoption line. The first 50 PPO
updates fill and cycle the five-entry snapshot pool; only later updates count.
This is a disposable ``dev-*`` benchmark and saves no learned policy.
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
    """Parse the fixed steady-state comparison and runtime cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[32, 48])
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--torch-interop-threads", type=int, default=2)
    parser.add_argument("--refill-reserve-factor", type=float, default=1.30)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a10_steady_state_slots.json"),
    )
    args = parser.parse_args()
    if any(worker < 1 for worker in args.workers):
        parser.error("workers must be positive")
    if args.warmup_iterations < 50:
        parser.error("warmup must include at least five snapshot cycles")
    if args.measured_iterations < 10:
        parser.error("measure at least ten steady-state iterations")
    if args.timeout_seconds <= 0.0 or args.heartbeat_seconds <= 0.0:
        parser.error("timeouts must be positive")
    return args


def _opponent_sources(rows: list[dict[str, float]]) -> list[str]:
    """Recover per-iteration current/snapshot choice from cumulative counters."""
    previous = {"current": 0, "snapshot": 0, "anchor": 0}
    sources: list[str] = []
    for row in rows:
        changed: list[str] = []
        for source in previous:
            current = int(row[f"opponent_{source}_iterations"])
            if current > previous[source]:
                changed.append(source)
            previous[source] = current
        if len(changed) != 1:
            raise ValueError("each iteration must select exactly one opponent source")
        sources.append(changed[0])
    return sources


def _aggregate(
    rows: list[dict[str, float]], sources: list[str], wall_seconds: float
) -> dict[str, Any]:
    """Aggregate actual rows and costs for the measured steady-state window."""
    trained_rows = sum(int(row["n_recorded"]) for row in rows)
    calls = sum(float(row["action_inference_calls"]) for row in rows)
    inference_rows = sum(float(row["action_inference_rows"]) for row in rows)
    grouped: dict[str, dict[str, float | int]] = {}
    for source in ("current", "snapshot"):
        selected = [row for row, value in zip(rows, sources, strict=True) if value == source]
        if selected:
            selected_rows = sum(int(row["n_recorded"]) for row in selected)
            selected_seconds = sum(float(row["iteration_seconds"]) for row in selected)
            grouped[source] = {
                "iterations": len(selected),
                "trained_rows": selected_rows,
                "trained_rows_per_iteration_second": selected_rows / selected_seconds,
                "mean_action_batch": sum(
                    float(row["action_inference_rows"]) for row in selected
                )
                / sum(float(row["action_inference_calls"]) for row in selected),
            }
    return {
        "iterations": len(rows),
        "trained_rows": trained_rows,
        "wall_seconds": wall_seconds,
        "trained_rows_per_second": trained_rows / wall_seconds,
        "rollout_microseconds_per_row": sum(
            float(row["rollout_seconds"]) for row in rows
        )
        * 1e6
        / trained_rows,
        "update_microseconds_per_row": sum(
            float(row["update_seconds"]) for row in rows
        )
        * 1e6
        / trained_rows,
        "inference_microseconds_per_row": sum(
            float(row["inference_seconds"]) for row in rows
        )
        * 1e6
        / trained_rows,
        "mean_action_batch": inference_rows / calls,
        "mean_rollout_overshoot": statistics.mean(
            float(row["rollout_overshoot"]) for row in rows
        ),
        "opponent_sources": grouped,
        "seat_counts": {
            str(seats): sum(int(row["seat_count"]) == seats for row in rows)
            for seats in range(2, 7)
        },
    }


def _run_arm(
    workers: int, args: argparse.Namespace, deadline: float
) -> dict[str, Any]:
    """Train through pool warmup and measure only the later fixed window."""
    total_iterations = args.warmup_iterations + args.measured_iterations
    config = A8Config(
        total_timesteps=total_iterations * args.n_steps,
        n_steps=args.n_steps,
        batch_size=256,
        n_epochs=5,
        hidden_sizes=(256, 256),
        seat_counts=(2, 3, 4, 5, 6),
        seed=args.seed,
        device="cpu",
        collector_mode="native",
        native_workers=workers,
        native_ready_capacity=workers * 6,
        native_refill_reserve_factor=args.refill_reserve_factor,
        head="masked",
        encoder="flat",
        margin_coef=0.0,
        anchor_share=0.0,
        stage1_enabled=False,
    )
    arm_started = perf_counter()
    last_heartbeat = arm_started
    measured_started: float | None = None
    measured_finished: float | None = None

    def progress(iteration: int, record: dict[str, float]) -> None:
        """Capture exact measurement boundaries and enforce the global cap."""
        nonlocal last_heartbeat, measured_started, measured_finished
        now = perf_counter()
        if now > deadline:
            raise TimeoutError("steady-state slot benchmark exceeded its cap")
        if iteration == args.warmup_iterations:
            measured_started = now
        if iteration == total_iterations:
            measured_finished = now
        if now - last_heartbeat >= args.heartbeat_seconds:
            phase = "measure" if iteration > args.warmup_iterations else "warmup"
            print(
                f"A10 steady workers={workers} phase={phase} "
                f"iteration={iteration}/{total_iterations} "
                f"steps={int(record['steps'])}",
                flush=True,
            )
            last_heartbeat = now

    _policy, records = train_selfplay_a8(config, profile=True, on_iteration=progress)
    if measured_started is None or measured_finished is None:
        raise RuntimeError("measurement boundaries were not observed")
    sources = _opponent_sources(records)
    measured_rows = records[args.warmup_iterations :]
    measured_sources = sources[args.warmup_iterations :]
    result = {
        "workers": workers,
        "ready_capacity": workers * 6,
        "warmup_iterations": args.warmup_iterations,
        "measured": _aggregate(
            measured_rows,
            measured_sources,
            measured_finished - measured_started,
        ),
        "total_wall_seconds": perf_counter() - arm_started,
    }
    print(
        f"A10 steady workers={workers} "
        f"rate={result['measured']['trained_rows_per_second']:.1f}/s",
        flush=True,
    )
    return result


def main() -> int:
    """Run the matched arms and choose whether wider slots merit confirmation."""
    args = _parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    started = perf_counter()
    deadline = started + args.timeout_seconds
    arms = [_run_arm(workers, args, deadline) for workers in args.workers]
    baseline = next((arm for arm in arms if arm["workers"] == 32), None)
    candidate = next((arm for arm in arms if arm["workers"] == 48), None)
    matched_comparison = baseline is not None and candidate is not None
    baseline_rate = (
        float(baseline["measured"]["trained_rows_per_second"])
        if baseline is not None
        else None
    )
    candidate_rate = (
        float(candidate["measured"]["trained_rows_per_second"])
        if candidate is not None
        else None
    )
    relative_gain = (
        candidate_rate / baseline_rate - 1.0
        if candidate_rate is not None and baseline_rate is not None
        else None
    )
    candidate_pass = bool(
        candidate_rate is not None
        and relative_gain is not None
        and candidate_rate >= 5_000.0
        and relative_gain >= 0.05
    )
    artifact = {
        "schema_version": 1,
        "stage": "D3_A10_snapshot_steady_state_slot_width",
        "status": "pass" if candidate_pass else "complete",
        "decision": (
            "confirm_48_slots"
            if candidate_pass
            else "keep_32_slots"
            if matched_comparison
            else "diagnostic_only"
        ),
        "hypothesis": (
            "After the five-snapshot pool reaches steady state, 48 active slots "
            "improve trained throughput by at least 5% and reach 5,000/s."
        ),
        "confidence_before_run": 0.55,
        "scope": {
            "workers": args.workers,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "n_steps": args.n_steps,
            "refill_reserve_factor": args.refill_reserve_factor,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
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
            "source_identity": native_source_tree_identity(),
        },
        "arms": arms,
        "candidate_relative_gain": relative_gain,
        "candidate_rate_pass": (
            candidate_rate >= 5_000.0 if candidate_rate is not None else None
        ),
        "candidate_gain_pass": (
            relative_gain >= 0.05 if relative_gain is not None else None
        ),
        "elapsed_seconds": perf_counter() - started,
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
                "decision": artifact["decision"],
                "baseline_rate": baseline_rate,
                "candidate_rate": candidate_rate,
                "candidate_relative_gain": relative_gain,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
