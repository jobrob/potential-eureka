#!/usr/bin/env python
"""Bounded A2 native hard-slice differential campaign.

The campaign compares the compact C++ state against the D0 semantic oracle at
the first REACT boundary and again after the movement/corner/discard tail. It
does not compare observations, masks, or production receipts; those are A4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

if __package__:
    from experiments.verify_direction_d2_differential import (
        prepare_batch,
    )
else:
    from verify_direction_d2_differential import (
        prepare_batch,
    )
from heat.engine import rules
from heat.engine.phases import (
    ReactDecision,
    step_adrenaline,
    step_check_corner,
    step_discard,
    step_react,
    step_replenish,
    step_reveal_and_move,
    step_slipstream,
)
from heat.ml.native_env.bridge import NativeStateBridge, legacy_state_to_native
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.models.game_state import GameState


_REPO_ROOT = Path(__file__).resolve().parents[1]


def native_source_tree_identity() -> str:
    """Hash every source family that can affect the native differential gate."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "native",
            "src",
            "experiments",
            "tests/native",
            "pyproject.toml",
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    digest = hashlib.sha256()
    for relative in sorted(line for line in result.stdout.splitlines() if line):
        path = _REPO_ROOT / relative
        if path.is_file() and path.suffix.lower() in {
            ".cpp",
            ".h",
            ".html",
            ".json",
            ".py",
            ".toml",
        }:
            digest.update(relative.replace("\\", "/").encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _react_decision(options: Any, case_index: int) -> ReactDecision:
    """Choose a deterministic legal tail action that exercises optional branches."""
    return ReactDecision(
        cooldown_count=min(1, int(options.max_cooldown)),
        use_boost=bool(options.can_boost and case_index % 2 == 0),
        use_adrenaline_speed=bool(options.has_adrenaline and case_index % 3 == 0),
        use_adrenaline_cooldown=bool(
            options.has_adrenaline and case_index % 5 == 0
        ),
    )


def _advance_scalar_to_next_react(
    state: GameState, completed_player: int
) -> int | None:
    """Advance already-chosen cards through skipped seats to the next REACT."""
    completed_index = state.turn_order.index(completed_player)
    for player_id in state.turn_order[completed_index + 1 :]:
        player = state.get_player(player_id)
        if player.finished:
            continue
        if player.cluttered:
            player.gear = 1
            step_replenish(state, player)
            continue
        step_reveal_and_move(state, player)
        if player.finished:
            step_replenish(state, player)
            continue
        step_adrenaline(state, player)
        return player_id
    for player in state.active_players:
        player.spun_out = False
    state.round_num += 1
    return None


def run_scalar_tail(
    state: GameState,
    player_id: int,
    decision: ReactDecision,
    *,
    take_slipstream: bool,
) -> tuple[str, int | None]:
    """Run the scalar movement tail to the next REACT or round boundary."""
    player = state.get_player(player_id)
    step_react(state, player, decision)
    if player.finished:
        step_replenish(state, player)
        next_player = _advance_scalar_to_next_react(state, player_id)
        return ("react", next_player) if next_player is not None else ("round_complete", None)
    if rules.legal_slipstream(player, list(state.active_players), state.track):
        step_slipstream(state, player, take_slipstream)
    step_check_corner(state, player)
    if rules.legal_discards(player):
        step_discard(state, player, [])
    step_replenish(state, player)
    next_player = _advance_scalar_to_next_react(state, player_id)
    return ("react", next_player) if next_player is not None else ("round_complete", None)


def run_native_tail(
    native: NativeStateBridge,
    player_id: int,
    decision: ReactDecision,
    *,
    take_slipstream: bool,
) -> dict[str, object]:
    """Run the same native tail, submitting each real decision boundary."""
    boundary = native.apply_react(player_id, decision)
    if boundary["kind"] == "slipstream":
        boundary = native.apply_slipstream(player_id, take_slipstream)
    if boundary["kind"] == "discard":
        boundary = native.apply_discard(player_id, [])
    return boundary


def _assert_snapshot(
    native: NativeStateBridge,
    scalar: GameState,
    *,
    game_id: int,
    stage: str,
) -> None:
    """Raise a replayable mismatch at one frozen semantic boundary."""
    actual = native.semantic_snapshot()
    expected = canonical_game_state(scalar)
    if actual != expected:
        raise AssertionError(f"game {game_id} {stage} semantic mismatch")


def run_campaign(args: argparse.Namespace) -> int:
    """Run exactness first and write a bounded pass/fail receipt."""
    started = perf_counter()
    deadline = started + args.timeout_seconds
    completed = 0
    games = 0
    tails = 0
    branch_counts: dict[str, int] = {}
    failure: dict[str, object] | None = None
    try:
        while completed < args.transitions:
            if perf_counter() > deadline:
                raise TimeoutError("A2 differential campaign exceeded its timeout")
            batch = prepare_batch(games, args.batch_size, args.seed)
            for offset, (source, scalar, choices, game_id, mode, react) in enumerate(
                zip(
                    batch.source_states,
                    batch.scalar_states,
                    batch.scalar_choices,
                    batch.game_ids,
                    batch.modes,
                    batch.react_decisions,
                    strict=True,
                )
            ):
                native = legacy_state_to_native(source, game_id=game_id)
                first = native.apply_cards_to_react(choices)
                if first["kind"] != "react" or first["player_id"] != react.player_id:
                    raise AssertionError(f"game {game_id} first decision mismatch")
                _assert_snapshot(native, scalar, game_id=game_id, stage="cards_to_react")

                case_index = games + offset
                decision = _react_decision(react.legal, case_index)
                take_slipstream = case_index % 2 == 1
                expected_kind, expected_player = run_scalar_tail(
                    scalar,
                    react.player_id,
                    decision,
                    take_slipstream=take_slipstream,
                )
                boundary = run_native_tail(
                    native,
                    react.player_id,
                    decision,
                    take_slipstream=take_slipstream,
                )
                if boundary["kind"] != expected_kind:
                    raise AssertionError(f"game {game_id} tail decision-kind mismatch")
                if expected_player is not None and boundary["player_id"] != expected_player:
                    raise AssertionError(f"game {game_id} tail player mismatch")
                _assert_snapshot(native, scalar, game_id=game_id, stage="movement_tail")

                branch_counts[mode] = branch_counts.get(mode, 0) + 1
                completed += len(choices) + 1
                tails += 1
            games += len(batch.game_ids)
            if completed // 10_000 != (completed - batch.transition_count - len(batch.game_ids)) // 10_000:
                print(
                    f"D3 A2: {completed}/{args.transitions} transitions, "
                    f"{games} games, zero mismatches, {perf_counter() - started:.1f}s",
                    flush=True,
                )
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "completed_transitions": completed,
            "games": games,
        }

    artifact = {
        "schema_version": 1,
        "stage": "D3_A2_native_hard_slice_differential",
        "status": "pass" if failure is None and completed >= args.transitions else "fail",
        "hypothesis": (
            "The compact C++ cards-to-REACT and movement-to-next-decision slices "
            "preserve exact scalar state, decision identity, and RNG semantics."
        ),
        "confidence_before_run": 0.75,
        "requested_transitions": args.transitions,
        "completed_transitions": completed,
        "games": games,
        "movement_tails": tails,
        "semantic_mismatches": 0 if failure is None else 1,
        "branch_counts": branch_counts,
        "compared": [
            "D0 semantic snapshot",
            "next decision kind/player",
            "card-zone order and identities",
            "CPython RNG state",
        ],
        "deferred_to_A4": ["native observations", "native masks", "numeric event receipts"],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "elapsed_seconds": perf_counter() - started,
        "timeout_seconds": args.timeout_seconds,
        "source_identity": native_source_tree_identity(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "pass" else 1


def _parse_args() -> argparse.Namespace:
    """Parse the frozen A2 correctness scope and explicit timeout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8_008)
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a2_native_differential.json"),
    )
    args = parser.parse_args()
    if args.transitions < 1 or not 1 <= args.batch_size <= 256:
        parser.error("transitions must be positive and batch size must be 1..256")
    if not 0 < args.timeout_seconds <= 3_600:
        parser.error("timeout must be in (0, 3600]")
    return args


if __name__ == "__main__":
    raise SystemExit(run_campaign(_parse_args()))
