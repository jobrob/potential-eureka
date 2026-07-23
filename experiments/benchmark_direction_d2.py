#!/usr/bin/env python
"""Bounded CPU kernel benchmark and D1 projection for Direction D2 Chunk 6."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np

from heat.ml.action_codec import legal_action_mask
from heat.ml.features import encode_observation
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.ml.vector_env.bridge import legacy_states_to_tensor
from heat.ml.vector_env.kernels import apply_cards_to_react
from heat.ml.vector_env.observations import (
    tensor_legal_action_masks,
    tensor_observations,
)

if __package__:
    from .verify_direction_d2_differential import (
        PreparedBatch,
        _source_tree_identity,
        prepare_batch,
        rebuild_scalar_replays,
        run_scalar_to_next_react,
    )
else:
    from verify_direction_d2_differential import (
        PreparedBatch,
        _source_tree_identity,
        prepare_batch,
        rebuild_scalar_replays,
        run_scalar_to_next_react,
    )


_REPO_ROOT = Path(__file__).resolve().parents[1]


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS layout for passive RSS telemetry."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _process_memory_bytes() -> tuple[int | None, int | None]:
    """Return current and peak working set on Windows when available."""
    if sys.platform != "win32":
        return None, None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not success:
        return None, None
    return int(counters.working_set_size), int(counters.peak_working_set_size)


def _write_json(path: Path, value: object) -> None:
    """Write one stable human-readable benchmark artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_passed_correctness(path: Path) -> dict[str, Any]:
    """Refuse timing unless the frozen Chunk-5 gate has actually passed."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("stage") != "D2_chunk_5_random_differential"
        or value.get("status") != "pass"
        or int(value.get("completed_transitions", 0)) < 100_000
        or int(value.get("semantic_mismatches", -1)) != 0
    ):
        raise ValueError("Chunk-6 timing requires a passed 100k Chunk-5 artifact")
    return value


def _tensor_state_bytes(batch: PreparedBatch) -> int:
    """Count fixed tensor storage without Python metadata or allocator overhead."""
    return sum(
        value.numel() * value.element_size()
        for _name, value in batch.tensor_state.tensor_fields()
    )


def _scalar_once(
    states: list[GameState],
    choices_by_game: list[dict[int, tuple[Card, ...]]],
) -> None:
    """Run the matched legacy slice, including next observation and mask."""
    checksum = 0.0
    for state, choices in zip(states, choices_by_game, strict=True):
        _events, decision, _stress, _replenish = run_scalar_to_next_react(
            state, choices, record_draws=False
        )
        observation = encode_observation(state, decision.player_id, decision)
        mask = legal_action_mask(decision, state)
        checksum += float(observation[0]) + float(mask.sum())
    if not np.isfinite(checksum):  # pragma: no cover - timing guard
        raise RuntimeError("non-finite scalar benchmark checksum")


def _measure_scalar(
    batch: PreparedBatch,
    seed: int,
    *,
    warmups: int,
    repeats: int,
    deadline: float,
) -> list[float]:
    """Time only scalar rule/observation work, excluding fixture rebuilding."""
    for _ in range(warmups):
        if perf_counter() > deadline:
            raise TimeoutError("Chunk-6 benchmark exceeded its wall-clock cap")
        states, choices = rebuild_scalar_replays(batch, seed)
        _scalar_once(states, choices)
    samples: list[float] = []
    for _ in range(repeats):
        if perf_counter() > deadline:
            raise TimeoutError("Chunk-6 benchmark exceeded its wall-clock cap")
        states, choices = rebuild_scalar_replays(batch, seed)
        started = perf_counter()
        _scalar_once(states, choices)
        samples.append(perf_counter() - started)
    return samples


def _tensor_once(batch: PreparedBatch) -> None:
    """Run the matched tensor slice, including next observation and mask."""
    result = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )
    observations = tensor_observations(result.state, result.next_decisions)
    masks = tensor_legal_action_masks(result.state, result.next_decisions)
    checksum = float(observations[0, 0].item()) + float(masks.sum().item())
    if not np.isfinite(checksum):  # pragma: no cover - timing guard
        raise RuntimeError("non-finite tensor benchmark checksum")


def _conversion_once(batch: PreparedBatch) -> None:
    """Measure the test-only legacy-to-tensor bridge separately."""
    converted = legacy_states_to_tensor(
        batch.source_states,
        game_ids=batch.game_ids,
    )
    if converted.batch_size != len(batch.game_ids):  # pragma: no cover
        raise RuntimeError("conversion benchmark lost a lane")


def _measure(
    operation: Callable[[], None],
    *,
    warmups: int,
    repeats: int,
    deadline: float,
) -> list[float]:
    """Measure one operation with untimed warmups and a shared deadline."""
    for _ in range(warmups):
        if perf_counter() > deadline:
            raise TimeoutError("Chunk-6 benchmark exceeded its wall-clock cap")
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        if perf_counter() > deadline:
            raise TimeoutError("Chunk-6 benchmark exceeded its wall-clock cap")
        started = perf_counter()
        operation()
        samples.append(perf_counter() - started)
    return samples


def _timing_summary(samples: list[float], batch_size: int) -> dict[str, object]:
    """Summarize seconds and per-game throughput without hiding repeat spread."""
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "relative_spread": (max(samples) - min(samples)) / median,
        "median_games_per_second": batch_size / median,
    }


def project_d1_end_to_end(
    d1_artifact: dict[str, Any], kernel_speedup: float
) -> dict[str, object]:
    """Apply measured kernel gain to D1 scalar rollout work via Amdahl's law.

    The estimate treats scalar rollout time other than policy inference as
    replaceable by a complete tensor engine at the measured slice speed. It
    leaves policy inference, PPO update, and preparation unchanged.
    """
    if kernel_speedup <= 0:
        raise ValueError("kernel_speedup must be positive")
    projections: list[dict[str, float | int]] = []
    for run in d1_artifact.get("runs", []):
        if run.get("collector_mode") != "scalar":
            continue
        rollout = float(run["rollout_seconds"])
        inference = float(run["inference_seconds"])
        replaceable = max(0.0, rollout - inference)
        projected_rollout = inference + replaceable / kernel_speedup
        projected_iteration = (
            projected_rollout
            + float(run["update_seconds"])
            + float(run["prepare_seconds"])
        )
        baseline_iteration = float(run["iteration_seconds"])
        projections.append(
            {
                "repeat": int(run["repeat"]),
                "baseline_iteration_seconds": baseline_iteration,
                "replaceable_rollout_seconds": replaceable,
                "fixed_inference_seconds": inference,
                "fixed_update_seconds": float(run["update_seconds"]),
                "fixed_prepare_seconds": float(run["prepare_seconds"]),
                "projected_iteration_seconds": projected_iteration,
                "projected_end_to_end_speedup": (
                    baseline_iteration / projected_iteration
                ),
            }
        )
    if not projections:
        raise ValueError("D1 artifact has no scalar timing runs")
    speedups = [
        float(item["projected_end_to_end_speedup"])
        for item in projections
    ]
    return {
        "assumption": (
            "All non-inference scalar rollout time is replaced at the measured "
            "D2 slice speed; inference, PPO update, and preparation stay fixed."
        ),
        "runs": projections,
        "median_projected_end_to_end_speedup": statistics.median(speedups),
    }


def run_benchmark(args: argparse.Namespace) -> int:
    """Run the frozen batch curve and write the Chunk-6 decision artifact."""
    correctness = _load_passed_correctness(args.correctness_artifact)
    d1_artifact = json.loads(args.d1_artifact.read_text(encoding="utf-8"))
    started = perf_counter()
    deadline = started + args.timeout_seconds
    arms: list[dict[str, object]] = []
    for arm_index, batch_size in enumerate(args.batch_sizes):
        batch = prepare_batch(
            args.case_start + arm_index * 1_000,
            batch_size,
            args.seed,
        )
        scalar_samples = _measure_scalar(
            batch,
            args.seed,
            warmups=args.warmups,
            repeats=args.repeats,
            deadline=deadline,
        )
        tensor_samples = _measure(
            lambda: _tensor_once(batch),
            warmups=args.warmups,
            repeats=args.repeats,
            deadline=deadline,
        )
        conversion_samples = _measure(
            lambda: _conversion_once(batch),
            warmups=args.warmups,
            repeats=args.repeats,
            deadline=deadline,
        )
        scalar_summary = _timing_summary(scalar_samples, batch_size)
        tensor_summary = _timing_summary(tensor_samples, batch_size)
        conversion_summary = _timing_summary(conversion_samples, batch_size)
        scalar_median = float(scalar_summary["median_seconds"])
        tensor_median = float(tensor_summary["median_seconds"])
        conversion_median = float(conversion_summary["median_seconds"])
        kernel_speedup = scalar_median / tensor_median
        conversion_inclusive_speedup = scalar_median / (
            tensor_median + conversion_median
        )
        projection = project_d1_end_to_end(d1_artifact, kernel_speedup)
        current_memory, peak_memory = _process_memory_bytes()
        arm = {
            "batch_size": batch_size,
            "scalar": scalar_summary,
            "tensor": tensor_summary,
            "conversion": conversion_summary,
            "kernel_speedup": kernel_speedup,
            "conversion_inclusive_speedup": conversion_inclusive_speedup,
            "tensor_state_bytes": _tensor_state_bytes(batch),
            "tensor_state_bytes_per_game": _tensor_state_bytes(batch) / batch_size,
            "process_working_set_bytes": current_memory,
            "process_peak_working_set_bytes": peak_memory,
            "d1_projection": projection,
        }
        arms.append(arm)
        print(
            f"D2 Chunk 6 batch={batch_size}: kernel={kernel_speedup:.3f}x, "
            f"with_conversion={conversion_inclusive_speedup:.3f}x, "
            "projected_end_to_end="
            f"{float(projection['median_projected_end_to_end_speedup']):.3f}x",
            flush=True,
        )

    useful_arms = [arm for arm in arms if int(arm["batch_size"]) >= 32]
    best = max(useful_arms, key=lambda arm: float(arm["kernel_speedup"]))
    best_kernel = float(best["kernel_speedup"])
    best_projection = float(
        dict(best["d1_projection"])["median_projected_end_to_end_speedup"]
    )
    exact_gate = (
        correctness["status"] == "pass"
        and int(correctness["semantic_mismatches"]) == 0
    )
    continuation_gate = exact_gate and best_kernel >= 3.0 and best_projection >= 1.5
    decision = "go" if continuation_gate else "redesign"
    artifact = {
        "schema_version": 1,
        "stage": "D2_chunk_6_kernel_benchmark",
        "status": "pass",
        "hypothesis": (
            "The exact tensor slice is at least 5x faster at a useful batch and "
            "supports at least a 1.5x D1-based end-to-end projection."
        ),
        "confidence_before_run": 0.6,
        "device": "cpu",
        "batch_sizes": args.batch_sizes,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "case_start": args.case_start,
        "arms": arms,
        "best_useful_batch": int(best["batch_size"]),
        "best_kernel_speedup": best_kernel,
        "best_projected_end_to_end_speedup": best_projection,
        "target_5x_kernel_pass": best_kernel >= 5.0,
        "minimum_3x_kernel_pass": best_kernel >= 3.0,
        "projected_1_5x_end_to_end_pass": best_projection >= 1.5,
        "correctness_gate_pass": exact_gate,
        "decision": decision,
        "elapsed_seconds": perf_counter() - started,
        "timeout_seconds": args.timeout_seconds,
        "source_identity": _source_tree_identity(),
        "correctness_artifact": str(args.correctness_artifact),
        "correctness_source_identity": correctness["source_identity"],
        "d1_artifact": str(args.d1_artifact),
        "projection_boundary": (
            "This is an Amdahl-law estimate for a future complete tensor rules "
            "path, not a measured end-to-end training result."
        ),
    }
    _write_json(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the frozen CPU curve and bounded runtime controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256]
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6_006)
    parser.add_argument("--case-start", type=int, default=200_000)
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument(
        "--correctness-artifact",
        type=Path,
        default=Path("runs/direction_d/d2_chunk5_differential.json"),
    )
    parser.add_argument(
        "--d1-artifact",
        type=Path,
        default=Path("runs/direction_d/d1_lane_benchmark.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d2_chunk6_benchmark.json"),
    )
    args = parser.parse_args(argv)
    if not args.batch_sizes or any(
        size < 1 or size > 256 for size in args.batch_sizes
    ):
        parser.error("batch sizes must all be in 1..256")
    if not any(size >= 32 for size in args.batch_sizes):
        parser.error("at least one useful batch size (>=32) is required")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run_benchmark(_parse_args()))
