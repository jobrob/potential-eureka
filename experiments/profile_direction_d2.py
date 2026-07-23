#!/usr/bin/env python
"""Profile the exact Direction D2 CPU slice without changing its semantics."""

from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import statistics
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np
from torch.profiler import ProfilerActivity, profile, record_function

from heat.ml.vector_env.bridge import legacy_states_to_tensor
from heat.ml.vector_env.kernels import apply_cards_to_react
from heat.ml.vector_env.kernels.cards_to_react import TensorKernelResult
from heat.ml.vector_env.observations import (
    tensor_legal_action_masks,
    tensor_observations,
)

if __package__:
    from .benchmark_direction_d2 import _measure_scalar
    from .verify_direction_d2_differential import (
        PreparedBatch,
        _source_tree_identity,
        prepare_batch,
    )
else:
    from benchmark_direction_d2 import _measure_scalar
    from verify_direction_d2_differential import (
        PreparedBatch,
        _source_tree_identity,
        prepare_batch,
    )


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required object-shaped JSON artifact."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    """Write one stable, human-readable profiling artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _check_deadline(deadline: float) -> None:
    """Stop a diagnostic that exceeds its declared wall-clock budget."""
    if perf_counter() > deadline:
        raise TimeoutError("D2 profiling exceeded its wall-clock cap")


def _measure(
    operation: Callable[[], None],
    *,
    warmups: int,
    repeats: int,
    deadline: float,
) -> dict[str, object]:
    """Measure one operation after untimed warmups."""
    for _ in range(warmups):
        _check_deadline(deadline)
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        _check_deadline(deadline)
        started = perf_counter()
        operation()
        samples.append(perf_counter() - started)
    return _timing_summary(samples)


def _timing_summary(samples: list[float]) -> dict[str, object]:
    """Summarize one nonempty timing sample without hiding its spread."""
    if not samples:
        raise ValueError("timing summary requires at least one sample")
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "relative_spread": (max(samples) - min(samples)) / median,
    }


def _kernel_once(batch: PreparedBatch) -> TensorKernelResult:
    """Run the exact rules kernel and consume a small result checksum."""
    result = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )
    checksum = int(result.state.round_num[0].item()) + len(result.events)
    if checksum < 0:  # pragma: no cover - timing guard
        raise RuntimeError("invalid D2 kernel checksum")
    return result


def _full_once(batch: PreparedBatch) -> None:
    """Run rules, observations, and legal masks as in the Chunk-6 benchmark."""
    result = _kernel_once(batch)
    observations = tensor_observations(result.state, result.next_decisions)
    masks = tensor_legal_action_masks(result.state, result.next_decisions)
    checksum = float(observations[0, 0].item()) + float(masks.sum().item())
    if not np.isfinite(checksum):  # pragma: no cover - timing guard
        raise RuntimeError("non-finite D2 profile checksum")


def _observation_mask_once(result: TensorKernelResult) -> None:
    """Measure the post-kernel observation and legal-mask work in isolation."""
    observations = tensor_observations(result.state, result.next_decisions)
    masks = tensor_legal_action_masks(result.state, result.next_decisions)
    if observations.shape[0] != masks.shape[0]:  # pragma: no cover
        raise RuntimeError("D2 observation and mask row counts differ")


def _clone_once(batch: PreparedBatch) -> None:
    """Measure the full fixed-state copy performed by every kernel call."""
    cloned = batch.tensor_state.clone()
    if cloned.batch_size != batch.tensor_state.batch_size:  # pragma: no cover
        raise RuntimeError("D2 state clone lost a lane")


def _validate_once(batch: PreparedBatch) -> None:
    """Measure full input-state validation in isolation."""
    batch.tensor_state.validate()


def _conversion_once(batch: PreparedBatch) -> None:
    """Measure the diagnostic legacy-to-tensor bridge separately."""
    converted = legacy_states_to_tensor(
        batch.source_states,
        game_ids=batch.game_ids,
    )
    if converted.batch_size != batch.tensor_state.batch_size:  # pragma: no cover
        raise RuntimeError("D2 conversion lost a lane")


def _relative_function_name(filename: str, line: int, name: str) -> str:
    """Format a cProfile function identity relative to the repository."""
    path = Path(filename)
    try:
        path_text = path.resolve().relative_to(_REPO_ROOT).as_posix()
    except (OSError, ValueError):
        path_text = filename
    return f"{path_text}:{line}:{name}"


def _python_profile(
    batch: PreparedBatch,
    *,
    repeats: int,
    top: int,
    deadline: float,
) -> dict[str, object]:
    """Capture Python call counts and cumulative time for exact full slices."""
    profiler = cProfile.Profile()
    _check_deadline(deadline)
    started = perf_counter()
    profiler.enable()
    for _ in range(repeats):
        _full_once(batch)
    profiler.disable()
    elapsed = perf_counter() - started
    stats = pstats.Stats(profiler)
    rows: list[dict[str, object]] = []
    for (filename, line, name), values in stats.stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = (
            values
        )
        rows.append(
            {
                "function": _relative_function_name(filename, line, name),
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": self_seconds,
                "cumulative_seconds": cumulative_seconds,
            }
        )
    rows.sort(key=lambda item: float(item["cumulative_seconds"]), reverse=True)
    return {
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "total_calls": stats.total_calls,
        "primitive_calls": stats.prim_calls,
        "top_by_cumulative_seconds": rows[:top],
    }


def _torch_profile(
    batch: PreparedBatch,
    *,
    repeats: int,
    top: int,
    deadline: float,
) -> dict[str, object]:
    """Capture eager PyTorch CPU operator counts and self time."""
    _check_deadline(deadline)
    started = perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as torch_profile:
        with record_function("direction_d2_full_slice"):
            for _ in range(repeats):
                _full_once(batch)
    elapsed = perf_counter() - started
    events = sorted(
        torch_profile.key_averages(),
        key=lambda event: event.self_cpu_time_total,
        reverse=True,
    )
    return {
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "top_by_self_cpu_time": [
            {
                "operator": event.key,
                "calls": event.count,
                "self_cpu_seconds": event.self_cpu_time_total / 1_000_000.0,
                "total_cpu_seconds": event.cpu_time_total / 1_000_000.0,
            }
            for event in events[:top]
        ],
    }


def _reference_arm(
    benchmark: dict[str, Any], batch_size: int
) -> dict[str, Any]:
    """Find the matching frozen Chunk-6 benchmark arm."""
    for arm in benchmark.get("arms", []):
        if int(arm["batch_size"]) == batch_size:
            return dict(arm)
    raise ValueError(f"Chunk-6 artifact has no batch {batch_size} arm")


def run_profile(args: argparse.Namespace) -> int:
    """Run the bounded component curve and deep profiles."""
    correctness = _read_json(args.correctness_artifact)
    if (
        correctness.get("stage") != "D2_chunk_5_random_differential"
        or correctness.get("status") != "pass"
        or int(correctness.get("completed_transitions", 0)) < 100_000
        or int(correctness.get("semantic_mismatches", -1)) != 0
    ):
        raise ValueError("D2 profiling requires a passed 100k correctness artifact")
    benchmark = _read_json(args.benchmark_artifact)
    if (
        benchmark.get("stage") != "D2_chunk_6_kernel_benchmark"
        or benchmark.get("status") != "pass"
    ):
        raise ValueError("D2 profiling requires a completed Chunk-6 benchmark")

    started = perf_counter()
    deadline = started + args.timeout_seconds
    curve: list[dict[str, object]] = []
    prepared: dict[int, PreparedBatch] = {}
    for arm_index, batch_size in enumerate(args.batch_sizes):
        _check_deadline(deadline)
        batch = prepare_batch(
            args.case_start + arm_index * 1_000,
            batch_size,
            args.seed,
        )
        prepared[batch_size] = batch
        result = _kernel_once(batch)
        components = {
            "scalar_reference": _timing_summary(
                _measure_scalar(
                    batch,
                    args.seed,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    deadline=deadline,
                )
            ),
            "full_slice": _measure(
                lambda: _full_once(batch),
                warmups=args.warmups,
                repeats=args.repeats,
                deadline=deadline,
            ),
            "kernel_only": _measure(
                lambda: _kernel_once(batch),
                warmups=args.warmups,
                repeats=args.repeats,
                deadline=deadline,
            ),
            "observation_and_mask": _measure(
                lambda: _observation_mask_once(result),
                warmups=args.warmups,
                repeats=args.repeats,
                deadline=deadline,
            ),
            "state_clone": _measure(
                lambda: _clone_once(batch),
                warmups=args.warmups,
                repeats=args.repeats,
                deadline=deadline,
            ),
            "state_validate": _measure(
                lambda: _validate_once(batch),
                warmups=args.warmups,
                repeats=args.repeats,
                deadline=deadline,
            ),
            "legacy_to_tensor_conversion": _measure(
                lambda: _conversion_once(batch),
                warmups=args.warmups,
                repeats=args.repeats,
                deadline=deadline,
            ),
        }
        reference = _reference_arm(benchmark, batch_size)
        scalar_seconds = float(
            dict(components["scalar_reference"])["median_seconds"]
        )
        full_seconds = float(dict(components["full_slice"])["median_seconds"])
        measured_speedup = scalar_seconds / full_seconds
        curve.append(
            {
                "batch_size": batch_size,
                "components": components,
                "reference_scalar_median_seconds": scalar_seconds,
                "reference_chunk6_kernel_speedup": reference["kernel_speedup"],
                "profile_curve_speedup": measured_speedup,
                "improvement_needed_for_1x": 1.0 / measured_speedup,
                "improvement_needed_for_3x": 3.0 / measured_speedup,
                "improvement_needed_for_5x": 5.0 / measured_speedup,
            }
        )
        print(
            f"D2 profile batch={batch_size}: full={full_seconds:.6f}s, "
            f"speed={measured_speedup:.3f}x, "
            f"need_for_3x={3.0 / measured_speedup:.1f}x",
            flush=True,
        )

    deep_profiles: dict[str, object] = {}
    for batch_size in args.deep_batch_sizes:
        batch = prepared.get(batch_size)
        if batch is None:
            raise ValueError(f"deep batch {batch_size} was not prepared")
        deep_profiles[str(batch_size)] = {
            "python": _python_profile(
                batch,
                repeats=args.profile_repeats,
                top=args.top,
                deadline=deadline,
            ),
            "torch": _torch_profile(
                batch,
                repeats=args.profile_repeats,
                top=args.top,
                deadline=deadline,
            ),
        }

    current_identity = _source_tree_identity()
    artifact = {
        "schema_version": 1,
        "stage": "D2_cpu_profile",
        "status": "pass",
        "hypothesis": (
            "Fine-grained eager PyTorch dispatch, tensor-to-Python boundaries, "
            "state copying and validation, and Python event materialization "
            "dominate the exact D2 CPU slice."
        ),
        "confidence_before_run": 0.85,
        "device": "cpu",
        "batch_sizes": args.batch_sizes,
        "deep_batch_sizes": args.deep_batch_sizes,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "profile_repeats": args.profile_repeats,
        "seed": args.seed,
        "case_start": args.case_start,
        "curve": curve,
        "deep_profiles": deep_profiles,
        "elapsed_seconds": perf_counter() - started,
        "timeout_seconds": args.timeout_seconds,
        "source_identity": current_identity,
        "correctness_artifact": str(args.correctness_artifact),
        "correctness_source_identity": correctness.get("source_identity"),
        "benchmark_artifact": str(args.benchmark_artifact),
        "benchmark_source_identity": benchmark.get("source_identity"),
        "provenance_note": (
            "The profiling harness changes the experiments-tree identity but "
            "does not change the D2 production kernel."
        ),
    }
    _write_json(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


def main() -> int:
    """Parse the bounded D2 profiling command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--deep-batch-sizes", type=int, nargs="+", default=[64, 256]
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7_007)
    parser.add_argument("--case-start", type=int, default=300_000)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--correctness-artifact",
        type=Path,
        default=_REPO_ROOT / "runs/direction_d/d2_chunk5_differential.json",
    )
    parser.add_argument(
        "--benchmark-artifact",
        type=Path,
        default=_REPO_ROOT / "runs/direction_d/d2_chunk6_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "runs/direction_d/d2_cpu_profile.json",
    )
    args = parser.parse_args()
    if not args.batch_sizes or any(size < 1 or size > 256 for size in args.batch_sizes):
        parser.error("batch sizes must all be in 1..256")
    if any(size not in args.batch_sizes for size in args.deep_batch_sizes):
        parser.error("deep batch sizes must be included in batch sizes")
    if args.warmups < 0 or args.repeats < 1 or args.profile_repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")
    if args.top < 1 or args.timeout_seconds <= 0:
        parser.error("top and timeout-seconds must be positive")
    return run_profile(args)


if __name__ == "__main__":
    sys.exit(main())
