#!/usr/bin/env python
"""Bounded A5 full-race differential campaign for the native engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np

if __package__:
    from experiments.verify_direction_d3_native import native_source_tree_identity
else:
    from verify_direction_d3_native import native_source_tree_identity

from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.ml.action_codec import decode_legal_action, legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.native_env.bridge import (
    NativeStateBridge,
    legacy_state_to_native,
    scalar_canonical_digest,
)
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.models.game_state import GameState
from heat.tracks.generator import generate_track


def _source_state(seed: int, seats: int) -> GameState:
    """Create one generated-track race with the training lap convention."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 1_000_000)
    for player in state.players:
        player.lap = 1
    return state


def _check_row(
    native: NativeStateBridge,
    state: GameState,
    decision: Decision,
) -> np.ndarray:
    """Compare native observation/mask bytes and return the legal indices."""
    native_observation = native.observation(decision.player_id, decision)
    scalar_observation = encode_observation(state, decision.player_id, decision)
    if not np.array_equal(native_observation, scalar_observation):
        raise AssertionError("native observation mismatch")
    native_mask = native.legal_mask(decision)
    scalar_mask = legal_action_mask(decision, state)
    if not np.array_equal(native_mask, scalar_mask):
        raise AssertionError("native legal-mask mismatch")
    legal = np.flatnonzero(scalar_mask)
    if len(legal) == 0:
        raise AssertionError("full-race decision has no encoded legal action")
    return legal


def _check_digest(
    native: NativeStateBridge,
    state: GameState,
    *,
    game_id: int,
) -> None:
    """Compare every logical compact field, card identity/order, and RNG word."""
    expected = scalar_canonical_digest(
        state,
        game_id=game_id,
        card_id_vocabulary=native._template.card_id_vocabulary,
    )
    if native.canonical_digest() != expected:
        raise AssertionError("native state/RNG digest mismatch")


def _random_action(
    native: NativeStateBridge,
    state: GameState,
    decision: Decision,
    rng: random.Random,
) -> int:
    """Choose one legal flat action from the independently seeded policy RNG."""
    legal = _check_row(native, state, decision)
    return int(legal[rng.randrange(len(legal))])


def run_race(case: int, base_seed: int) -> dict[str, int | bool]:
    """Run one complete-or-bounded scalar/native episode in exact lockstep."""
    seats = 2 + case % 5
    seed = base_seed + case
    game_id = 2_000_000 + case
    state = _source_state(seed, seats)
    native = legacy_state_to_native(state, game_id=game_id)
    policy_rng = random.Random(seed + 2_000_000)
    transitions = 0
    rounds = 0
    events = 0

    while not state.is_game_over and state.round_num <= MAX_ROUNDS:
        rounds += 1
        boundary = native.start_round()
        if boundary["kind"] != "gear":
            raise AssertionError("native race did not start at gear")
        driver = run_round_driver(state)
        decision = next(driver)

        gear_actions: dict[int, int] = {}
        while decision.kind is DecisionKind.GEAR:
            action = _random_action(native, state, decision, policy_rng)
            gear_actions[decision.player_id] = action
            transitions += 1
            decision = driver.send(decode_legal_action(decision, state, action))
        boundary = native.apply_gear_actions(gear_actions)
        _check_digest(native, state, game_id=game_id)

        card_actions: dict[int, int] = {}
        completed_during_cards = False
        while decision.kind is DecisionKind.CARDS:
            action = _random_action(native, state, decision, policy_rng)
            card_actions[decision.player_id] = action
            transitions += 1
            try:
                decision = driver.send(
                    decode_legal_action(decision, state, action)
                )
            except StopIteration as stopped:
                events += len(stopped.value)
                completed_during_cards = True
                break
        boundary = native.apply_card_actions(card_actions)
        _check_digest(native, state, game_id=game_id)
        if completed_during_cards:
            if boundary["kind"] != "round_complete":
                raise AssertionError("native card completion boundary mismatch")
            continue

        while True:
            if boundary["kind"] != decision.kind.value:
                raise AssertionError("native sequential decision kind mismatch")
            if boundary["player_id"] != decision.player_id:
                raise AssertionError("native sequential decision player mismatch")
            action = _random_action(native, state, decision, policy_rng)
            engine_action = decode_legal_action(decision, state, action)
            transitions += 1
            try:
                next_decision = driver.send(engine_action)
            except StopIteration as stopped:
                events += len(stopped.value)
                boundary = native.apply_flat_action(decision.player_id, action)
                _check_digest(native, state, game_id=game_id)
                if boundary["kind"] != "round_complete":
                    raise AssertionError("native round completion mismatch")
                break
            boundary = native.apply_flat_action(decision.player_id, action)
            _check_digest(native, state, game_id=game_id)
            decision = next_decision

    if native.semantic_snapshot() != canonical_game_state(state):
        raise AssertionError("native final semantic state mismatch")
    return {
        "transitions": transitions,
        "rounds": rounds,
        "events": events,
        "terminated": state.is_game_over,
        "truncated": not state.is_game_over,
        "seats": seats,
    }


def run_campaign(args: argparse.Namespace) -> int:
    """Run the frozen A5 corpus and emit a replayable JSON receipt."""
    started = perf_counter()
    deadline = started + args.timeout_seconds
    transitions = games = rounds = events = terminated = truncated = 0
    next_heartbeat = 10_000
    seat_counts = {str(seats): 0 for seats in range(2, 7)}
    failure: dict[str, object] | None = None
    try:
        while transitions < args.transitions:
            if perf_counter() > deadline:
                raise TimeoutError("A5 full-race campaign exceeded its timeout")
            result = run_race(games, args.seed)
            transitions += int(result["transitions"])
            rounds += int(result["rounds"])
            events += int(result["events"])
            terminated += int(bool(result["terminated"]))
            truncated += int(bool(result["truncated"]))
            seat_counts[str(result["seats"])] += 1
            games += 1
            if transitions >= next_heartbeat:
                print(
                    f"D3 A5: {transitions}/{args.transitions} transitions, "
                    f"{games} games, zero mismatches, {perf_counter() - started:.1f}s",
                    flush=True,
                )
                next_heartbeat = (transitions // 10_000 + 1) * 10_000
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "completed_transitions": transitions,
            "games": games,
        }

    artifact = {
        "schema_version": 1,
        "stage": "D3_A5_native_full_race_differential",
        "status": "pass" if failure is None and transitions >= args.transitions else "fail",
        "hypothesis": (
            "The complete single-thread native race engine preserves exact scalar "
            "state, RNG, observations, masks, decisions, and final results."
        ),
        "confidence_before_run": 0.70,
        "requested_transitions": args.transitions,
        "completed_transitions": transitions,
        "games": games,
        "rounds": rounds,
        "scalar_events_observed": events,
        "terminated_games": terminated,
        "truncated_games": truncated,
        "seat_counts": seat_counts,
        "semantic_mismatches": 0 if failure is None else 1,
        "compared_at_every_boundary": [
            "decision kind and player",
            "float32 observation bytes",
            "516-bit legal mask",
            "compact logical state and ordered card identities",
            "CPython MT19937 state",
        ],
        "compared_at_episode_end": ["full D0 semantic snapshot", "finish order"],
        "seed": args.seed,
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
    """Parse the bounded one-million-transition A5 gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=9_005)
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a5_full_race_differential.json"),
    )
    args = parser.parse_args()
    if args.transitions < 1:
        parser.error("transitions must be positive")
    if not 0 < args.timeout_seconds <= 3_600:
        parser.error("timeout must be in (0, 3600]")
    return args


if __name__ == "__main__":
    raise SystemExit(run_campaign(_parse_args()))
