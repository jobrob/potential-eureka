"""Explicit single-game state-machine gates for Direction D3 chunk A3."""

from __future__ import annotations

import random

import pytest

from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.phases import ReactDecision
from heat.ml.native_env.bridge import NativeStateBridge, legacy_state_to_native
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.tracks.generator import generate_track


def _state(seed: int, seats: int) -> GameState:
    """Create one deterministic started full-rules state."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 50_000)
    for player in state.players:
        player.lap = 1
    return state


def _choose(decision: Decision, rng: random.Random) -> object:
    """Choose one deterministic legal action for every scalar decision kind."""
    if decision.kind in (DecisionKind.GEAR, DecisionKind.CARDS):
        legal = list(decision.legal)  # type: ignore[arg-type]
        return legal[rng.randrange(len(legal))]
    if decision.kind is DecisionKind.REACT:
        options = decision.legal
        use_adrenaline_cooldown = bool(
            options.has_adrenaline and rng.randrange(2)
        )
        return ReactDecision(
            cooldown_count=rng.randrange(
                options.max_cooldown + int(use_adrenaline_cooldown) + 1
            ),
            use_boost=bool(options.can_boost and rng.randrange(2)),
            use_adrenaline_speed=bool(
                options.has_adrenaline and rng.randrange(2)
            ),
            use_adrenaline_cooldown=use_adrenaline_cooldown,
        )
    if decision.kind is DecisionKind.SLIPSTREAM:
        return bool(rng.randrange(2))
    if decision.kind is DecisionKind.DISCARD:
        cards = list(decision.legal)  # type: ignore[arg-type]
        return cards[: rng.randrange(len(cards) + 1)]
    raise AssertionError(f"unsupported decision {decision.kind}")


def _submit_native(
    native: NativeStateBridge,
    decision: Decision,
    action: object,
) -> dict[str, object]:
    """Submit one decoded scalar action to the matching native boundary."""
    if decision.kind is DecisionKind.REACT:
        assert isinstance(action, ReactDecision)
        return native.apply_react(decision.player_id, action)
    if decision.kind is DecisionKind.SLIPSTREAM:
        assert isinstance(action, bool)
        return native.apply_slipstream(decision.player_id, action)
    if decision.kind is DecisionKind.DISCARD:
        assert isinstance(action, list)
        assert all(isinstance(card, Card) for card in action)
        return native.apply_discard(decision.player_id, action)
    raise AssertionError("simultaneous decisions use their group submission")


def _assert_boundary(
    native: NativeStateBridge,
    scalar: GameState,
    boundary: dict[str, object],
    decision: Decision,
) -> None:
    """Compare decision identity and every D0 semantic field at a pause point."""
    assert boundary["kind"] == decision.kind.value
    assert boundary["player_id"] == decision.player_id
    assert native.semantic_snapshot() == canonical_game_state(scalar)


def _run_round_lockstep(
    native: NativeStateBridge,
    scalar: GameState,
    rng: random.Random,
) -> set[DecisionKind]:
    """Drive one complete scalar/native round and compare every yield boundary."""
    kinds: set[DecisionKind] = set()
    gear_boundary = native.start_round()
    driver = run_round_driver(scalar)
    decision = next(driver)
    assert gear_boundary["kind"] == "gear"
    assert decision.kind is DecisionKind.GEAR

    gear_choices: dict[int, tuple[int, int]] = {}
    while decision.kind is DecisionKind.GEAR:
        kinds.add(decision.kind)
        action = _choose(decision, rng)
        assert isinstance(action, tuple)
        gear_choices[decision.player_id] = action
        decision = driver.send(action)
    assert gear_boundary["player_ids"] == list(gear_choices)
    cards_boundary = native.apply_gears(gear_choices)
    assert decision.kind is DecisionKind.CARDS
    assert cards_boundary["kind"] == "cards"
    assert native.semantic_snapshot() == canonical_game_state(scalar)

    card_choices: dict[int, tuple[Card, ...]] = {}
    while decision.kind is DecisionKind.CARDS:
        kinds.add(decision.kind)
        action = _choose(decision, rng)
        assert isinstance(action, tuple)
        card_choices[decision.player_id] = action
        decision = driver.send(action)
    boundary = native.apply_cards_to_react(card_choices)
    _assert_boundary(native, scalar, boundary, decision)

    while True:
        kinds.add(decision.kind)
        action = _choose(decision, rng)
        try:
            next_decision = driver.send(action)
        except StopIteration:
            boundary = _submit_native(native, decision, action)
            assert boundary["kind"] == "round_complete"
            assert native.semantic_snapshot() == canonical_game_state(scalar)
            return kinds
        boundary = _submit_native(native, decision, action)
        _assert_boundary(native, scalar, boundary, next_decision)
        decision = next_decision


@pytest.mark.parametrize("seats", range(2, 7))
def test_randomized_rounds_match_driver_at_every_decision(seats: int) -> None:
    """Three rounds preserve state and program position for every seat count."""
    scalar = _state(4_000 + seats, seats)
    native = legacy_state_to_native(scalar, game_id=60_000 + seats)
    rng = random.Random(70_000 + seats)
    observed: set[DecisionKind] = set()
    for _ in range(3):
        observed |= _run_round_lockstep(native, scalar, rng)
    assert {DecisionKind.GEAR, DecisionKind.CARDS, DecisionKind.REACT} <= observed


def test_spun_player_gear_is_forced_and_omitted_from_ready_group() -> None:
    """A spun-out seat is resolved natively while other gear rows stay simultaneous."""
    scalar = _state(4_100, 3)
    scalar.players[1].spun_out = True
    scalar.players[1].gear = 4
    native = legacy_state_to_native(scalar, game_id=61_000)
    boundary = native.start_round()
    assert boundary["player_ids"] == [0, 2]

    choices = {
        decision.player_id: decision.legal[0]  # type: ignore[index]
        for decision in _gear_decisions(scalar)
    }
    scalar.compute_turn_order()
    from heat.engine.phases import phase_shift_gears

    phase_shift_gears(scalar, {1: (1, 0), **choices})
    native.apply_gears(choices)
    assert native.semantic_snapshot() == canonical_game_state(scalar)


def test_randomized_lockstep_corpus_reaches_every_decision_kind() -> None:
    """A bounded generated-track corpus covers all five resumable boundaries."""
    observed: set[DecisionKind] = set()
    for case in range(40):
        seats = 2 + case % 5
        scalar = _state(4_500 + case, seats)
        native = legacy_state_to_native(scalar, game_id=65_000 + case)
        observed |= _run_round_lockstep(
            native, scalar, random.Random(75_000 + case)
        )
    assert observed == set(DecisionKind)


def _gear_decisions(state: GameState) -> list[Decision]:
    """Return scalar simultaneous gear rows without importing test internals."""
    from heat.engine.driver import simultaneous_decisions

    return simultaneous_decisions(state, DecisionKind.GEAR)


def test_program_counter_rejects_wrong_or_duplicate_submission() -> None:
    """A stale or wrong decision cannot mutate the native game."""
    scalar = _state(4_200, 2)
    native = legacy_state_to_native(scalar, game_id=62_000)
    native.start_round()
    with pytest.raises(ValueError, match="pending native decision"):
        native.apply_cards_to_react({})

    choices = {
        decision.player_id: decision.legal[0]  # type: ignore[index]
        for decision in _gear_decisions(scalar)
    }
    native.apply_gears(choices)
    with pytest.raises(ValueError, match="pending native decision"):
        native.apply_gears(choices)
