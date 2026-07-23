"""Full-race and flat-action gates for Direction D3 chunk A5."""

from __future__ import annotations

import pytest

from heat.agents.heuristic_agent import HeuristicAgent
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.ml.action_codec import (
    decode_legal_action,
    encode_action_index,
    legal_action_mask,
)
from heat.ml.native_env.bridge import (
    NativeStateBridge,
    legacy_state_to_native,
    scalar_canonical_digest,
)
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.models.game_state import GameState
from heat.tracks.generator import generate_track


def _state(seed: int, seats: int) -> GameState:
    """Create a deterministic full-race source state."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 100_000)
    for player in state.players:
        player.lap = 1
    return state


def _racing_action(state: GameState, decision: Decision) -> int:
    """Choose the existing deterministic heuristic's legal flat action."""
    agent = HeuristicAgent()
    player_id = decision.player_id
    if decision.kind is DecisionKind.GEAR:
        action = agent.choose_gear(state, player_id, decision.legal)  # type: ignore[arg-type]
    elif decision.kind is DecisionKind.CARDS:
        action = agent.choose_cards(state, player_id, decision.legal)  # type: ignore[arg-type]
    elif decision.kind is DecisionKind.REACT:
        mask = legal_action_mask(decision, state)
        return next(index for index in (501, 500, 499, 498) if mask[index])
    elif decision.kind is DecisionKind.SLIPSTREAM:
        action = agent.choose_slipstream(state, player_id)
    else:
        action = agent.choose_discard(
            state, player_id, decision.legal  # type: ignore[arg-type]
        )
    index = encode_action_index(decision, action)
    assert legal_action_mask(decision, state)[index]
    return index


def _assert_state(native: NativeStateBridge, scalar: GameState) -> None:
    """Compare all mutable state and RNG fields at one full-race boundary."""
    assert native.canonical_digest() == scalar_canonical_digest(
        scalar,
        game_id=int(native.boundary_receipt()[1]),
        card_id_vocabulary=native._template.card_id_vocabulary,
    )
    assert native.semantic_snapshot() == canonical_game_state(scalar)


def _run_race(seed: int, seats: int, *, compare_every_boundary: bool) -> int:
    """Drive one scalar/native race from flat actions through completion."""
    scalar = _state(seed, seats)
    native = legacy_state_to_native(scalar, game_id=120_000 + seed)
    transitions = 0
    while not scalar.is_game_over and scalar.round_num <= MAX_ROUNDS:
        gear_boundary = native.start_round()
        assert gear_boundary["kind"] == "gear"
        driver = run_round_driver(scalar)
        decision = next(driver)

        gear_actions: dict[int, int] = {}
        while decision.kind is DecisionKind.GEAR:
            action = _racing_action(scalar, decision)
            gear_actions[decision.player_id] = action
            transitions += 1
            decision = driver.send(decode_legal_action(decision, scalar, action))
        native.apply_gear_actions(gear_actions)
        if compare_every_boundary:
            _assert_state(native, scalar)

        card_actions: dict[int, int] = {}
        completed_during_cards = False
        while decision.kind is DecisionKind.CARDS:
            action = _racing_action(scalar, decision)
            card_actions[decision.player_id] = action
            transitions += 1
            try:
                decision = driver.send(
                    decode_legal_action(decision, scalar, action)
                )
            except StopIteration:
                completed_during_cards = True
                break
        boundary = native.apply_card_actions(card_actions)
        if compare_every_boundary:
            _assert_state(native, scalar)
        if completed_during_cards:
            assert boundary["kind"] == "round_complete"
            continue

        while True:
            assert boundary["kind"] == decision.kind.value
            assert boundary["player_id"] == decision.player_id
            action = _racing_action(scalar, decision)
            engine_action = decode_legal_action(decision, scalar, action)
            transitions += 1
            try:
                next_decision = driver.send(engine_action)
            except StopIteration:
                boundary = native.apply_flat_action(decision.player_id, action)
                assert boundary["kind"] == "round_complete"
                if compare_every_boundary:
                    _assert_state(native, scalar)
                break
            boundary = native.apply_flat_action(decision.player_id, action)
            if compare_every_boundary:
                _assert_state(native, scalar)
            decision = next_decision

    assert scalar.is_game_over
    _assert_state(native, scalar)
    assert native.canonical_digest() == int(native.boundary_receipt()[-1])
    return transitions


@pytest.mark.parametrize("seats", range(2, 7))
def test_flat_action_full_races_match_every_boundary(seats: int) -> None:
    """Each supported field size completes with exact state at every pause."""
    assert _run_race(6_000 + seats, seats, compare_every_boundary=True) > 0


def test_illegal_flat_action_is_rejected_before_mutation() -> None:
    """The native codec revalidates submitted policy actions."""
    scalar = _state(6_100, 2)
    native = legacy_state_to_native(scalar, game_id=126_100)
    native.start_round()
    digest = native.canonical_digest()
    with pytest.raises(ValueError, match="not legal"):
        native.apply_gear_actions({0: 3, 1: 3})
    assert native.canonical_digest() == digest
