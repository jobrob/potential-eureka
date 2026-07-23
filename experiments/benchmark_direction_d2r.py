"""Bounded correctness and CPU affordability gate for Direction D2-R1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from time import perf_counter
from typing import Any, Callable

import torch

from experiments.benchmark_direction_d2 import _measure_scalar, _tensor_once
from experiments.verify_direction_d2_differential import (
    PreparedBatch,
    _source_tree_identity,
    prepare_batch,
)
from heat.ml.vector_env.compiled import (
    apply_compiled_common_core,
    materialize_receipts,
    prepare_action_matrix,
    pure_cards_to_react_common,
)
from heat.ml.vector_env.kernels import apply_cards_to_react
from heat.ml.vector_env.observations import (
    tensor_legal_action_masks,
    tensor_observations,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _common_case_indices(start: int, count: int) -> list[int]:
    """Select unique normal/traffic cases while retaining varied seeds."""
    aligned = start - start % 5
    return [aligned + (offset // 2) * 5 + offset % 2 for offset in range(count)]


def prepare_common_batch(start: int, count: int, seed: int) -> PreparedBatch:
    """Build only the explicitly supported R1 normal and traffic cases."""
    return prepare_batch(
        start,
        count,
        seed,
        case_indices=_common_case_indices(start, count),
    )


def _ensure_msvc_environment() -> str | None:
    """Load Visual Studio's compiler environment when ``cl`` is not on PATH."""
    if os.name != "nt" or shutil.which("cl") is not None:
        return shutil.which("cl")
    vswhere = Path(
        os.environ.get(
            "ProgramFiles(x86)",
            r"C:\Program Files (x86)",
        )
    ) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        raise RuntimeError("PyTorch Inductor requires MSVC, but vswhere.exe is missing")
    install = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    vcvars = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        raise RuntimeError("Visual Studio is installed without vcvars64.bat")
    environment = subprocess.run(
        f'call "{vcvars}" >nul && set',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    for line in environment.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            os.environ["PATH" if name.casefold() == "path" else name] = value
    compiler = shutil.which("cl")
    if compiler is None:
        candidates = sorted(
            (Path(install) / "VC" / "Tools" / "MSVC").glob(
                "*/bin/Hostx64/x64/cl.exe"
            )
        )
        if not candidates:
            raise RuntimeError("vcvars64 completed but cl.exe is still unavailable")
        compiler = str(candidates[-1])
        os.environ["PATH"] = str(candidates[-1].parent) + os.pathsep + os.environ["PATH"]
    return compiler


def _assert_exact(
    batch: PreparedBatch,
    compiled_function: Callable[..., Any],
) -> None:
    """Compare one compiled batch against the complete eager D2 oracle."""
    eager = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )
    actual = apply_compiled_common_core(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
        compiled_function=compiled_function,
    )
    for (expected_name, expected), (actual_name, observed) in zip(
        eager.state.tensor_fields(), actual.state.tensor_fields(), strict=True
    ):
        if expected_name != actual_name or not torch.equal(expected, observed):
            raise AssertionError(f"compiled state mismatch in {expected_name}")
    if materialize_receipts(actual.receipts, actual.state) != eager.events:
        raise AssertionError("compiled numeric receipt mismatch")
    if any(
        not torch.equal(expected, observed)
        for expected, observed in (
            (eager.next_decisions.game_ids, actual.next_decisions.game_ids),
            (eager.next_decisions.player_ids, actual.next_decisions.player_ids),
            (eager.next_decisions.kinds, actual.next_decisions.kinds),
        )
    ):
        raise AssertionError("compiled next-decision mismatch")
    if not torch.equal(
        actual.observations, tensor_observations(eager.state, eager.next_decisions)
    ):
        raise AssertionError("compiled observation mismatch")
    if not torch.equal(
        actual.legal_masks,
        tensor_legal_action_masks(eager.state, eager.next_decisions),
    ):
        raise AssertionError("compiled legal-mask mismatch")


def _measure(
    operation: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
    deadline: float,
) -> list[float]:
    """Measure a warmed operation while enforcing the shared wall-clock cap."""
    for _ in range(warmups):
        if perf_counter() > deadline:
            raise TimeoutError("D2-R1 benchmark exceeded its wall-clock cap")
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        if perf_counter() > deadline:
            raise TimeoutError("D2-R1 benchmark exceeded its wall-clock cap")
        started = perf_counter()
        operation()
        samples.append(perf_counter() - started)
    return samples


def _summary(samples: list[float], batch_size: int) -> dict[str, object]:
    """Report every sample plus the median throughput and spread."""
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "median_games_per_second": batch_size / median,
        "relative_spread": (max(samples) - min(samples)) / median,
    }


def _compile_and_first_call(
    batch: PreparedBatch,
) -> tuple[Callable[..., Any], float, dict[str, int]]:
    """Compile one frozen shape and return first-call time plus graph counters."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(pure_cards_to_react_common, fullgraph=True)
    actions = prepare_action_matrix(batch.tensor_state, batch.decisions, batch.actions)
    started = perf_counter()
    compiled(
        batch.tensor_state,
        actions,
        batch.draw_inputs.card_ids,
        batch.draw_inputs.lengths,
    )
    compile_seconds = perf_counter() - started
    counters = {
        "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
    }
    return compiled, compile_seconds, counters


def run_benchmark(args: argparse.Namespace) -> int:
    """Run R1 correctness first, then its bounded 64/256 CPU speed gate."""
    compiler = _ensure_msvc_environment()
    started = perf_counter()
    deadline = started + args.timeout_seconds
    correctness_batches = [
        prepare_common_batch(args.case_start + index * 10_000, 256, args.seed)
        for index in range(2)
    ]
    compiled_256, compile_seconds, compile_counters = _compile_and_first_call(
        correctness_batches[0]
    )
    completed_transitions = 0
    stress_draws = 0
    for index, batch in enumerate(correctness_batches, start=1):
        if perf_counter() > deadline:
            raise TimeoutError("D2-R1 benchmark exceeded its wall-clock cap")
        _assert_exact(batch, compiled_256)
        completed_transitions += batch.transition_count
        stress_draws += batch.stress_draw_count
        print(
            f"D2-R1 correctness {index}/2: {completed_transitions} transitions, "
            "zero mismatches",
            flush=True,
        )
    if completed_transitions < 1_000:
        raise AssertionError("D2-R1 correctness campaign did not reach 1,000 transitions")
    final_counters = {
        "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
    }
    if final_counters != {"unique_graphs": 1, "graph_breaks": 0}:
        raise AssertionError(f"D2-R1 graph capture changed: {final_counters}")

    arms: list[dict[str, object]] = []
    for batch_size in (64, 256):
        batch = (
            correctness_batches[0]
            if batch_size == 256
            else prepare_common_batch(args.case_start + 30_000, batch_size, args.seed)
        )
        if batch_size == 256:
            compiled = compiled_256
            arm_compile_seconds = compile_seconds
            counters = compile_counters
        else:
            compiled, arm_compile_seconds, counters = _compile_and_first_call(batch)
        actions = prepare_action_matrix(batch.tensor_state, batch.decisions, batch.actions)

        def compiled_operation() -> object:
            """Run the warm compiled graph for this frozen arm."""
            return compiled(
                batch.tensor_state,
                actions,
                batch.draw_inputs.card_ids,
                batch.draw_inputs.lengths,
            )
        scalar_samples = _measure_scalar(
            batch,
            args.seed,
            warmups=args.warmups,
            repeats=args.repeats,
            deadline=deadline,
        )
        eager_samples = _measure(
            lambda: _tensor_once(batch),
            warmups=args.warmups,
            repeats=args.repeats,
            deadline=deadline,
        )
        compiled_samples = _measure(
            compiled_operation,
            warmups=args.warmups,
            repeats=args.repeats,
            deadline=deadline,
        )
        scalar = _summary(scalar_samples, batch_size)
        eager = _summary(eager_samples, batch_size)
        warm = _summary(compiled_samples, batch_size)
        scalar_seconds = float(scalar["median_seconds"])
        eager_seconds = float(eager["median_seconds"])
        compiled_seconds = float(warm["median_seconds"])
        arm = {
            "batch_size": batch_size,
            "compile_first_call_seconds": arm_compile_seconds,
            "compile_counters_after_first_call": counters,
            "scalar": scalar,
            "eager_d2": eager,
            "compiled": warm,
            "speedup_over_eager_d2": eager_seconds / compiled_seconds,
            "speedup_over_scalar": scalar_seconds / compiled_seconds,
        }
        arms.append(arm)
        print(
            f"D2-R1 batch={batch_size}: "
            f"eager_speedup={arm['speedup_over_eager_d2']:.3f}x, "
            f"scalar_speedup={arm['speedup_over_scalar']:.3f}x",
            flush=True,
        )

    batch_256 = next(arm for arm in arms if arm["batch_size"] == 256)
    compile_gate = final_counters == {"unique_graphs": 1, "graph_breaks": 0}
    correctness_gate = completed_transitions >= 1_000
    speed_gate = (
        float(batch_256["speedup_over_eager_d2"]) >= 10.0
        and float(batch_256["speedup_over_scalar"]) >= 1.0
    )
    decision = "continue" if compile_gate and correctness_gate and speed_gate else "stop"
    artifact = {
        "schema_version": 1,
        "stage": "D2_R1_compiled_common_core",
        "status": "pass" if compile_gate and correctness_gate else "fail",
        "decision": decision,
        "hypothesis": (
            "A pure full-graph cards-to-REACT common core runs at least 10x "
            "faster than eager D2 and at least as fast as scalar at batch 256."
        ),
        "confidence_before_run": 0.55,
        "scope": {
            "code": "R1 normal play, recorded stress, traffic, movement, receipts, observations, masks",
            "compute": "CPU only, batches 64 and 256, under 30 minutes",
            "excluded": ["reshuffle", "replenish continuation", "finish", "CUDA", "training"],
        },
        "compiler": compiler,
        "torch_version": torch.__version__,
        "source_identity": _source_tree_identity(),
        "completed_transitions": completed_transitions,
        "semantic_mismatches": 0,
        "stress_draws": stress_draws,
        "compile_seconds_batch_256": compile_seconds,
        "final_compile_counters_batch_256": final_counters,
        "arms": arms,
        "gates": {
            "fullgraph_zero_breaks_recompiles": compile_gate,
            "at_least_1000_zero_mismatch_transitions": correctness_gate,
            "batch_256_at_least_10x_eager_and_1x_scalar": speed_gate,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"D2-R1 decision: {decision.upper()} -> {args.output}", flush=True)
    return 0 if decision == "continue" else 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded D2-R1 gate command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7_007)
    parser.add_argument("--case-start", type=int, default=600_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "runs" / "direction_d" / "d2r_r1_gate.json",
    )
    args = parser.parse_args(argv)
    if args.warmups < 1 or args.repeats < 3:
        parser.error("R1 requires at least one warmup and three repeats")
    if args.timeout_seconds <= 0 or args.timeout_seconds > 1_800:
        parser.error("timeout must be in (0, 1800] seconds")
    return args


if __name__ == "__main__":
    raise SystemExit(run_benchmark(_parse_args()))
