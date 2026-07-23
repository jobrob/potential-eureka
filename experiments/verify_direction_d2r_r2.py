"""Bounded 10k differential and CPU speed gate for Direction D2-R2."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch

from experiments.benchmark_direction_d2 import _measure_scalar
from experiments.benchmark_direction_d2r import (
    _ensure_msvc_environment,
    _measure,
    _source_tree_identity,
    _summary,
)
from experiments.verify_direction_d2_differential import PreparedBatch, prepare_batch
from heat.ml.vector_env.compiled import (
    apply_compiled_r2,
    materialize_receipts,
    prepare_action_matrix,
    pure_cards_to_react_r2,
)
from heat.ml.vector_env.kernels import apply_cards_to_react
from heat.ml.vector_env.observations import (
    tensor_legal_action_masks,
    tensor_observations,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUPPORTED_MODES = (0, 1, 3, 4)


def _r2_case_indices(start: int, count: int) -> list[int]:
    """Select every non-reshuffle fixture mode with unique case identities."""
    aligned = start - start % 5
    return [
        aligned + (offset // len(_SUPPORTED_MODES)) * 5 + _SUPPORTED_MODES[
            offset % len(_SUPPORTED_MODES)
        ]
        for offset in range(count)
    ]


def prepare_r2_batch(start: int, count: int, seed: int) -> PreparedBatch:
    """Build a varied batch that excludes only the R3 reshuffle cases."""
    return prepare_batch(
        start,
        count,
        seed,
        case_indices=_r2_case_indices(start, count),
    )


def _assert_exact_r2(
    batch: PreparedBatch,
    compiled: Callable[..., Any],
) -> object:
    """Demand exact eager parity at every R2 public comparison point."""
    eager = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )
    actual = apply_compiled_r2(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
        compiled_function=compiled,
    )
    for (expected_name, expected), (actual_name, observed) in zip(
        eager.state.tensor_fields(), actual.state.tensor_fields(), strict=True
    ):
        if expected_name != actual_name or not torch.equal(expected, observed):
            raise AssertionError(f"R2 compiled state mismatch in {expected_name}")
    if materialize_receipts(actual.receipts, actual.state) != eager.events:
        raise AssertionError("R2 compiled receipt mismatch")
    for expected, observed in (
        (eager.next_decisions.game_ids, actual.next_decisions.game_ids),
        (eager.next_decisions.player_ids, actual.next_decisions.player_ids),
        (eager.next_decisions.kinds, actual.next_decisions.kinds),
    ):
        if not torch.equal(expected, observed):
            raise AssertionError("R2 compiled next-decision mismatch")
    if not torch.equal(
        actual.observations, tensor_observations(eager.state, eager.next_decisions)
    ):
        raise AssertionError("R2 compiled observation mismatch")
    if not torch.equal(
        actual.legal_masks,
        tensor_legal_action_masks(eager.state, eager.next_decisions),
    ):
        raise AssertionError("R2 compiled legal-mask mismatch")
    return actual


def _compiled_call(
    compiled: Callable[..., Any], batch: PreparedBatch, actions: torch.Tensor
) -> object:
    """Invoke the pure graph with the fixed R2 ABI."""
    return compiled(
        batch.tensor_state,
        actions,
        batch.draw_inputs.card_ids,
        batch.draw_inputs.lengths,
        batch.draw_inputs.replenish_card_ids,
        batch.draw_inputs.replenish_lengths,
    )


def run_gate(args: argparse.Namespace) -> int:
    """Run exactness/determinism first and time only after every check passes."""
    compiler_path = _ensure_msvc_environment()
    started = perf_counter()
    deadline = started + args.timeout_seconds
    first = prepare_r2_batch(args.case_start, args.batch_size, args.seed)
    actions = prepare_action_matrix(
        first.tensor_state,
        first.decisions,
        first.actions,
        allow_forced_heat=True,
    )
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(pure_cards_to_react_r2, fullgraph=True)
    compile_started = perf_counter()
    _compiled_call(compiled, first, actions)
    compile_seconds = perf_counter() - compile_started

    completed = 0
    stress_draws = 0
    replenish_draws = 0
    first_result: object | None = None
    batch_count = 0
    while completed < args.transitions:
        batch = prepare_r2_batch(
            args.case_start + batch_count * 100_000,
            args.batch_size,
            args.seed,
        )
        result = _assert_exact_r2(batch, compiled)
        if first_result is None:
            first_result = result
        completed += batch.transition_count
        stress_draws += batch.stress_draw_count
        replenish_draws += batch.replenish_draw_count
        batch_count += 1
        print(
            f"D2-R2: {completed}/{args.transitions} transitions, zero mismatches",
            flush=True,
        )
        if perf_counter() > deadline:
            raise TimeoutError("D2-R2 gate exceeded its wall-clock cap")

    repeat = _assert_exact_r2(first, compiled)
    buffer = BytesIO()
    torch.save(first.tensor_state, buffer)
    buffer.seek(0)
    restored_state = torch.load(buffer, weights_only=False)
    restored = PreparedBatch(
        first.source_states,
        first.scalar_states,
        first.scalar_choices,
        first.game_ids,
        first.modes,
        restored_state,
        first.decisions,
        first.actions,
        first.draw_inputs,
        first.expected_events,
        first.react_decisions,
        first.stress_draw_count,
        first.replenish_draw_count,
    )
    _assert_exact_r2(restored, compiled)
    if first_result is None or type(repeat) is not type(first_result):
        raise AssertionError("R2 repeat result changed type")

    scalar_samples = _measure_scalar(
        first,
        args.seed,
        warmups=args.warmups,
        repeats=args.repeats,
        deadline=deadline,
    )
    compiled_samples = _measure(
        lambda: _compiled_call(compiled, first, actions),
        warmups=args.warmups,
        repeats=args.repeats,
        deadline=deadline,
    )
    scalar = _summary(scalar_samples, args.batch_size)
    warm = _summary(compiled_samples, args.batch_size)
    speedup = float(scalar["median_seconds"]) / float(warm["median_seconds"])
    counters = {
        "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
    }
    graph_gate = counters == {"unique_graphs": 1, "graph_breaks": 0}
    speed_gate = speedup >= 1.0
    decision = "continue" if graph_gate and speed_gate else "stop"
    artifact = {
        "schema_version": 1,
        "stage": "D2_R2_nonreshuffle_differential",
        "status": "pass",
        "decision": decision,
        "hypothesis": (
            "All non-reshuffle cards-to-REACT paths remain exact in one graph "
            "and at least match scalar speed at batch 256."
        ),
        "confidence_before_run": 0.65,
        "scope": {
            "code": "all deterministic rules, continuation, compact events, observations, masks",
            "compute": "CPU only, at least 10k transitions, five timing repeats, under 30 minutes",
            "excluded": ["reshuffle", "compiled RNG", "CUDA", "training"],
        },
        "compiler": compiler_path,
        "torch_version": torch.__version__,
        "source_identity": _source_tree_identity(),
        "completed_transitions": completed,
        "semantic_mismatches": 0,
        "stress_draws": stress_draws,
        "replenish_draws": replenish_draws,
        "repeat_exact": True,
        "save_resume_exact": True,
        "compile_seconds": compile_seconds,
        "compile_counters": counters,
        "scalar": scalar,
        "compiled": warm,
        "speedup_over_scalar": speedup,
        "gates": {
            "at_least_10000_zero_mismatch_transitions": completed >= 10_000,
            "repeat_and_save_resume_exact": True,
            "one_full_graph": graph_gate,
            "at_least_scalar_speed": speed_gate,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"D2-R2: scalar_speedup={speedup:.3f}x, decision={decision.upper()} -> {args.output}",
        flush=True,
    )
    return 0 if decision == "continue" else 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded R2 command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7_007)
    parser.add_argument("--case-start", type=int, default=700_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "runs" / "direction_d" / "d2r_r2_gate.json",
    )
    args = parser.parse_args(argv)
    if args.transitions < 10_000 or args.batch_size != 256:
        parser.error("R2 requires at least 10k transitions at batch 256")
    if args.repeats < 3 or args.timeout_seconds <= 0 or args.timeout_seconds > 1_800:
        parser.error("R2 requires at least 3 repeats and a timeout in (0, 1800]")
    return args


if __name__ == "__main__":
    raise SystemExit(run_gate(_parse_args()))
