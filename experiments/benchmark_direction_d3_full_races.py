#!/usr/bin/env python
"""Five-repeat A5 full-rule speed gate after exactness passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Callable

import numpy as np

if __package__:
    from experiments.verify_direction_d3_full_races import _source_state
    from experiments.verify_direction_d3_native import native_source_tree_identity
else:
    from verify_direction_d3_full_races import _source_state
    from verify_direction_d3_native import native_source_tree_identity

from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.ml.action_codec import decode_legal_action, legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.native_env.bridge import NativeStateBridge, legacy_state_to_native
from heat.models.game_state import GameState


_KIND_CODE = {
    DecisionKind.GEAR: 1,
    DecisionKind.CARDS: 2,
    DecisionKind.REACT: 3,
    DecisionKind.SLIPSTREAM: 4,
    DecisionKind.DISCARD: 5,
}
_KIND_NAME_CODE = {kind.value: code for kind, code in _KIND_CODE.items()}


def _load_correctness(path: Path) -> dict[str, object]:
    """Refuse timing unless the complete million-transition artifact is green."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("stage") != "D3_A5_native_full_race_differential"
        or artifact.get("status") != "pass"
        or int(artifact.get("completed_transitions", 0)) < 1_000_000
        or int(artifact.get("semantic_mismatches", -1)) != 0
    ):
        raise ValueError("A5 timing requires a passed 1M exactness artifact")
    return artifact


def _action_key(case: int, round_num: int, kind_code: int, player_id: int) -> int:
    """Return a stable policy-choice key independent of runner implementation."""
    return (
        case * 1_000_003
        + round_num * 10_007
        + kind_code * 101
        + player_id * 17
    )


def _choose(mask: np.ndarray, key: int) -> int:
    """Select one legal flat index using a deterministic corpus key."""
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        raise RuntimeError("benchmark decision has no encoded legal action")
    return int(legal[key % len(legal)])


def _scalar_action(state: GameState, decision: Decision, case: int) -> int:
    """Encode the scalar observation/mask and choose the keyed legal action."""
    encode_observation(state, decision.player_id, decision)
    mask = legal_action_mask(decision, state)
    return _choose(
        mask,
        _action_key(
            case, state.round_num, _KIND_CODE[decision.kind], decision.player_id
        ),
    )


def run_scalar(states: list[tuple[int, GameState]]) -> int:
    """Run matched scalar full episodes including observation and mask production."""
    transitions = 0
    for case, state in states:
        while not state.is_game_over and state.round_num <= MAX_ROUNDS:
            driver = run_round_driver(state)
            try:
                decision = next(driver)
            except StopIteration:
                continue
            while True:
                action = _scalar_action(state, decision, case)
                transitions += 1
                try:
                    decision = driver.send(
                        decode_legal_action(decision, state, action)
                    )
                except StopIteration:
                    break
    return transitions


def _native_row(
    bridge: NativeStateBridge,
    player_id: int,
    kind_code: int,
    *,
    case: int,
    round_num: int,
) -> int:
    """Produce the native observation/mask and select the same keyed action."""
    bridge._native.observation(player_id, kind_code)
    mask = np.asarray(
        bridge._native.legal_mask(player_id, kind_code), dtype=np.bool_
    )
    return _choose(mask, _action_key(case, round_num, kind_code, player_id))


def run_native(states: list[tuple[int, NativeStateBridge]]) -> int:
    """Run matched native full episodes through frozen flat-action submissions."""
    transitions = 0
    for case, bridge in states:
        while int(bridge.boundary_receipt()[3]) <= MAX_ROUNDS:
            boundary = bridge.start_round()
            if boundary["kind"] == "game_complete":
                break
            round_num = int(bridge.boundary_receipt()[3])
            gear_actions = {
                int(player_id): _native_row(
                    bridge,
                    int(player_id),
                    1,
                    case=case,
                    round_num=round_num,
                )
                for player_id in boundary["player_ids"]  # type: ignore[union-attr]
            }
            transitions += len(gear_actions)
            boundary = bridge.apply_gear_actions(gear_actions)

            card_actions = {
                int(player_id): _native_row(
                    bridge,
                    int(player_id),
                    2,
                    case=case,
                    round_num=round_num,
                )
                for player_id in boundary["player_ids"]  # type: ignore[union-attr]
            }
            transitions += len(card_actions)
            boundary = bridge.apply_card_actions(card_actions)

            while boundary["kind"] != "round_complete":
                kind_code = _KIND_NAME_CODE[str(boundary["kind"])]
                player_id = int(boundary["player_id"])
                action = _native_row(
                    bridge,
                    player_id,
                    kind_code,
                    case=case,
                    round_num=round_num,
                )
                transitions += 1
                boundary = bridge.apply_flat_action(player_id, action)
    return transitions


def _measure(
    prepare: Callable[[], object],
    run: Callable[[object], int],
    *,
    warmups: int,
    repeats: int,
    deadline: float,
) -> tuple[list[float], int]:
    """Prepare outside timing and return samples plus the matched row count."""
    samples: list[float] = []
    transitions = -1
    for repeat in range(warmups + repeats):
        if perf_counter() > deadline:
            raise TimeoutError("A5 benchmark exceeded its timeout")
        value = prepare()
        started = perf_counter()
        observed = run(value)
        elapsed = perf_counter() - started
        if transitions < 0:
            transitions = observed
        elif observed != transitions:
            raise RuntimeError("benchmark transition count drifted")
        if repeat >= warmups:
            samples.append(elapsed)
    return samples, transitions


def _summary(samples: list[float], transitions: int) -> dict[str, object]:
    """Summarize raw samples, median throughput, and repeat spread."""
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
    """Measure matched full-rule episodes and apply the frozen A5 stop rule."""
    correctness = _load_correctness(args.correctness_artifact)
    started = perf_counter()
    deadline = started + args.timeout_seconds
    cases = list(range(args.case_start, args.case_start + args.games))

    def prepare_scalar() -> list[tuple[int, GameState]]:
        return [
            (case, _source_state(args.seed + case, 2 + case % 5))
            for case in cases
        ]

    def prepare_native() -> list[tuple[int, NativeStateBridge]]:
        return [
            (
                case,
                legacy_state_to_native(
                    _source_state(args.seed + case, 2 + case % 5),
                    game_id=3_000_000 + case,
                ),
            )
            for case in cases
        ]

    scalar_samples, scalar_transitions = _measure(
        prepare_scalar,
        run_scalar,  # type: ignore[arg-type]
        warmups=args.warmups,
        repeats=args.repeats,
        deadline=deadline,
    )
    native_samples, native_transitions = _measure(
        prepare_native,
        run_native,  # type: ignore[arg-type]
        warmups=args.warmups,
        repeats=args.repeats,
        deadline=deadline,
    )
    if scalar_transitions != native_transitions:
        raise RuntimeError("scalar/native benchmark transition counts differ")
    scalar = _summary(scalar_samples, scalar_transitions)
    native = _summary(native_samples, native_transitions)
    speedup = float(scalar["median_seconds"]) / float(native["median_seconds"])
    decision = "continue" if speedup >= 3.0 else "profile_once" if speedup >= 2.0 else "stop"
    artifact = {
        "schema_version": 1,
        "stage": "D3_A5_native_full_race_benchmark",
        "status": "pass",
        "decision": decision,
        "hypothesis": "Exact native full rules remain at least 3x scalar speed.",
        "confidence_before_run": 0.70,
        "scope": {
            "games": args.games,
            "transitions_per_repeat": scalar_transitions,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "included": [
                "all complete/truncated race rounds",
                "all five decision kinds",
                "observations and legal masks",
                "flat action decoding and exact rules",
            ],
            "excluded": ["fixture construction", "policy inference", "PPO update"],
        },
        "scalar": scalar,
        "native": native,
        "speedup_over_scalar": speedup,
        "minimum_3x_pass": speedup >= 3.0,
        "below_2x_stop": speedup < 2.0,
        "seed": args.seed,
        "case_start": args.case_start,
        "elapsed_seconds": perf_counter() - started,
        "timeout_seconds": args.timeout_seconds,
        "source_identity": native_source_tree_identity(),
        "correctness_source_identity": correctness["source_identity"],
        "correctness_artifact": str(args.correctness_artifact),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if speedup >= 3.0 else 1


def _parse_args() -> argparse.Namespace:
    """Parse the bounded five-repeat A5 benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--case-start", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=9_005)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument(
        "--correctness-artifact",
        type=Path,
        default=Path("runs/direction_d/d3_a5_full_race_differential.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a5_full_race_benchmark.json"),
    )
    args = parser.parse_args()
    if args.games < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("games/repeats must be positive and warmups non-negative")
    if not 0 < args.timeout_seconds <= 3_600:
        parser.error("timeout must be in (0, 3600]")
    return args


if __name__ == "__main__":
    raise SystemExit(run_benchmark(_parse_args()))
