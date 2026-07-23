#!/usr/bin/env python
"""Five-repeat A2 native hard-slice speed gate after exactness passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Any, Callable

import numpy as np

if __package__:
    from experiments.verify_direction_d2_differential import (
        prepare_batch,
        rebuild_scalar_replays,
        run_scalar_to_next_react,
    )
    from experiments.verify_direction_d3_native import (
        _react_decision,
        native_source_tree_identity,
        run_native_tail,
        run_scalar_tail,
    )
else:
    from verify_direction_d2_differential import (
        prepare_batch,
        rebuild_scalar_replays,
        run_scalar_to_next_react,
    )
    from verify_direction_d3_native import (
        _react_decision,
        native_source_tree_identity,
        run_native_tail,
        run_scalar_tail,
    )
from heat.ml.native_env.bridge import NativeStateBridge, legacy_state_to_native


def _load_correctness(path: Path) -> dict[str, Any]:
    """Refuse timing unless the complete 100k A2 artifact is green."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("stage") != "D3_A2_native_hard_slice_differential"
        or artifact.get("status") != "pass"
        or int(artifact.get("completed_transitions", 0)) < 100_000
        or int(artifact.get("semantic_mismatches", -1)) != 0
    ):
        raise ValueError("A2 timing requires a passed 100k exactness artifact")
    return artifact


def _measure(
    prepare: Callable[[], Any],
    run: Callable[[Any], None],
    *,
    warmups: int,
    repeats: int,
    deadline: float,
) -> list[float]:
    """Prepare mutable copies outside timing, then measure only branch execution."""
    samples: list[float] = []
    for repeat in range(warmups + repeats):
        if perf_counter() > deadline:
            raise TimeoutError("A2 benchmark exceeded its timeout")
        value = prepare()
        started = perf_counter()
        run(value)
        elapsed = perf_counter() - started
        if repeat >= warmups:
            samples.append(elapsed)
    return samples


def _summary(samples: list[float], transitions: int) -> dict[str, object]:
    """Report raw samples, median, spread, and actual transition throughput."""
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "relative_spread": (max(samples) - min(samples)) / median,
        "median_transitions_per_second": transitions / median,
    }


def run_benchmark(args: argparse.Namespace) -> int:
    """Measure matched scalar/native hard slices and apply the A2 stop rule."""
    correctness = _load_correctness(args.correctness_artifact)
    started = perf_counter()
    deadline = started + args.timeout_seconds
    batch = prepare_batch(args.case_start, args.games, args.seed)
    transitions = batch.transition_count + len(batch.game_ids)

    base_native = [
        legacy_state_to_native(state, game_id=game_id)
        for state, game_id in zip(batch.source_states, batch.game_ids, strict=True)
    ]
    choice_arrays: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    for bridge, choices in zip(base_native, batch.scalar_choices, strict=True):
        ids = np.zeros((6, 4), dtype=np.int64)
        lengths = np.zeros((6,), dtype=np.int64)
        players = bridge._template.player_ids[0].tolist()
        vocabulary = {
            card_id: index + 1
            for index, card_id in enumerate(bridge._template.card_id_vocabulary)
        }
        for player_id, cards in choices.items():
            slot = players.index(player_id)
            lengths[slot] = len(cards)
            ids[slot, : len(cards)] = [vocabulary[card.id] for card in cards]
        choice_arrays.append((ids, lengths))
    tail_decisions = [
        _react_decision(react.legal, args.case_start + index)
        for index, react in enumerate(batch.react_decisions)
    ]

    def prepare_scalar() -> tuple[list[Any], list[dict[int, tuple[Any, ...]]]]:
        return rebuild_scalar_replays(batch, args.seed)

    def run_scalar(value: tuple[list[Any], list[dict[int, tuple[Any, ...]]]]) -> None:
        states, choices = value
        for index, (state, selected, react, decision) in enumerate(
            zip(states, choices, batch.react_decisions, tail_decisions, strict=True)
        ):
            _events, reached, _stress, _replenish = run_scalar_to_next_react(
                state, selected, record_draws=False
            )
            if reached.player_id != react.player_id:
                raise RuntimeError("scalar benchmark decision drift")
            run_scalar_tail(
                state,
                reached.player_id,
                decision,
                take_slipstream=(args.case_start + index) % 2 == 1,
            )

    def prepare_native() -> list[NativeStateBridge]:
        return [
            NativeStateBridge(bridge._native.clone(), bridge._template)
            for bridge in base_native
        ]

    def run_native(states: list[NativeStateBridge]) -> None:
        for index, (bridge, arrays, react, decision) in enumerate(
            zip(
                states,
                choice_arrays,
                batch.react_decisions,
                tail_decisions,
                strict=True,
            )
        ):
            first = bridge._native.apply_cards_to_react(*arrays)
            if first["player_id"] != react.player_id:
                raise RuntimeError("native benchmark decision drift")
            run_native_tail(
                bridge,
                react.player_id,
                decision,
                take_slipstream=(args.case_start + index) % 2 == 1,
            )

    scalar_samples = _measure(
        prepare_scalar,
        run_scalar,
        warmups=args.warmups,
        repeats=args.repeats,
        deadline=deadline,
    )
    native_samples = _measure(
        prepare_native,
        run_native,
        warmups=args.warmups,
        repeats=args.repeats,
        deadline=deadline,
    )
    scalar = _summary(scalar_samples, transitions)
    native = _summary(native_samples, transitions)
    speedup = float(scalar["median_seconds"]) / float(native["median_seconds"])
    decision = "continue" if speedup >= 3.0 else "profile_once" if speedup >= 2.0 else "stop"
    artifact = {
        "schema_version": 1,
        "stage": "D3_A2_native_hard_slice_benchmark",
        "status": "pass",
        "decision": decision,
        "hypothesis": "The exact native representative hard slice is at least 3x scalar speed.",
        "confidence_before_run": 0.75,
        "scope": {
            "games": args.games,
            "transitions_per_repeat": transitions,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "included": [
                "cards to first REACT",
                "stress and reshuffle",
                "traffic and finish",
                "react/boost movement",
                "slipstream, corner/spin, discard, replenish",
                "next-decision continuation",
            ],
            "excluded": ["fixture construction", "state cloning", "observations", "masks"],
        },
        "scalar": scalar,
        "native": native,
        "speedup_over_scalar": speedup,
        "minimum_3x_pass": speedup >= 3.0,
        "design_5x_goal_pass": speedup >= 5.0,
        "below_2x_stop": speedup < 2.0,
        "seed": args.seed,
        "case_start": args.case_start,
        "elapsed_seconds": perf_counter() - started,
        "timeout_seconds": args.timeout_seconds,
        "source_identity": native_source_tree_identity(),
        "correctness_artifact": str(args.correctness_artifact),
        "correctness_source_identity": correctness["source_identity"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if decision == "continue" else 2


def _parse_args() -> argparse.Namespace:
    """Parse the fixed five-repeat performance gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9_009)
    parser.add_argument("--case-start", type=int, default=900_000)
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument(
        "--correctness-artifact",
        type=Path,
        default=Path("runs/direction_d/d3_a2_native_differential.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a2_native_benchmark.json"),
    )
    args = parser.parse_args()
    if args.games < 1 or args.warmups < 0 or args.repeats < 5:
        parser.error("games must be positive, warmups non-negative, and repeats at least five")
    if not 0 < args.timeout_seconds <= 3_600:
        parser.error("timeout must be in (0, 3600]")
    return args


if __name__ == "__main__":
    raise SystemExit(run_benchmark(_parse_args()))
